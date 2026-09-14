"""Generate the frozen F54 learner-occupancy classic policy pilot dataset."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_orbit_digest
from .classic_bridge import ClassicAIBridge
from .codec import (
    mirror_action_index,
    position_hash,
    state_from_serializable,
    state_to_serializable,
)
from .game import legal_action_indices
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f51_learner_teacher_probe import (
    SELECTION as F51_SELECTION,
    TEACHER_ROOT as F51_TEACHER_ROOT,
    _load_analysis_records,
    _open_store,
    summarize_targets,
    tie_preserving_rank_target,
)
from .phase4f52_teacher_budget_correction import (
    SEARCH_MANIFEST as F52_SEARCH_MANIFEST,
    TARGET_RESULT as F52_TARGET_RESULT,
    _analysis_with_bridge,
    _load_f52_records,
)
from .phase4f53_policy_head_preparation import (
    DEVELOPMENT_COUNTS,
    F54_CONTRACT,
    GATE as F53_GATE,
    REPORT as F53_REPORT,
    TRAIN_COUNTS,
    TRAIN_SELECTION,
    verify_phase4f53,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_CONTRACT_SHA256 = "0c4559a884bb8e7e6a24b02c8d6cb2997c22c83a514db5fbc3f109b0489ead17"
EXPECTED_F53_GATE_SHA256 = "71011667f8182897ba7b48091702d845e3742ce109014369f2e296db4cad5796"
EXPECTED_F53_REPORT_SHA256 = "904be49b985766f0bdef3b5d13842603a8c008d60e44ff89dd6656611afef3a8"
EXPECTED_TRAIN_SELECTION_SHA256 = "95da56ca2244096ab9ce2ecee53aa6397de918b68eaafbe3a38a7cbc26d516a4"
EXPECTED_F51_SELECTION_SHA256 = "57b3c0b41c766ca6db6a0e89300f5a29f6a7c0166b3db8177d7029f1b2ae8f0b"
EXPECTED_F52_SEARCH_SHA256 = "d175b540b9653dcd715014464b973a52d133e5697dead7f09a2fd9046b865b94"
EXPECTED_F52_TARGET_SHA256 = "a4693b86962e0b85c842cb898fd093be255d1232c458f4c69545168cc84f4c7a"

ROOT = Path("training/runs/phase4/phase4f/f54-learner-teacher-policy-dataset-v1")
SEARCH_ROOT = ROOT / "classic-root"
SEARCH_MANIFEST = SEARCH_ROOT / "manifest.json"
DATASET_ROOT = ROOT / "dataset"
DATASET_MANIFEST = ROOT / "dataset-manifest.json"
REPORT = ROOT / "generation-report.json"
GATE = ROOT / "dataset-gate.json"
RUN_ID_PREFIX = "phase4f54-learner-teacher-policy-dataset"

OUTPUT_LIMIT_BYTES = 768 * 1024**2
BUDGETS = {"early": 16_384, "middle": 16_384, "late": 4_096}
SEARCH_CHUNK_POSITIONS = 8
DATASET_CHUNK_POSITIONS = 64
POLICY_TEMPERATURE = 1.0
POLICY_TOP_K = 12
SEARCHED_MASS = 0.97


class Phase4F54Error(RuntimeError):
    """Raised when the frozen policy-only dataset cannot be generated safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f54-policy-dataset-progress",
                "schemaVersion": SCHEMA_VERSION,
                "stage": stage,
                "status": status,
                **fields,
            }
        ).decode("ascii"),
        end="",
        flush=True,
    )


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F54Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace, paths=("training/quoridor/phase4f54_policy_dataset.py",)
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    output_bytes = directory_size(workspace / ROOT) if (workspace / ROOT).exists() else 0
    free_bytes = shutil.disk_usage(workspace).free
    if output_bytes > OUTPUT_LIMIT_BYTES:
        raise Phase4F54Error("F54 output exceeded its frozen 768 MiB limit")
    remaining_output = OUTPUT_LIMIT_BYTES - output_bytes
    if artifact_bytes + remaining_output >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F54Error("F54 remaining write reserve exceeds the 16 GiB artifact cap")
    if free_bytes - remaining_output < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F54Error("F54 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "f54OutputBytes": output_bytes,
        "f54OutputLimitBytes": OUTPUT_LIMIT_BYTES,
        "remainingWriteReserveBytes": remaining_output,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f53(workspace)
    gate = _mapping(workspace / F53_GATE, "F53 gate")
    report = _mapping(workspace / F53_REPORT, "F53 report")
    contract = _mapping(workspace / F54_CONTRACT, "F54 contract")
    train_selection = _mapping(workspace / TRAIN_SELECTION, "F53 train selection")
    hashes = {
        "gate": _verify_self_hash(gate, "gateSha256", "F53 gate"),
        "report": _verify_self_hash(report, "reportSha256", "F53 report"),
        "contract": _verify_self_hash(contract, "contractSha256", "F54 contract"),
        "trainSelection": _verify_self_hash(
            train_selection, "selectionSha256", "F53 train selection"
        ),
    }
    if (
        verification.get("status") != "verified"
        or hashes
        != {
            "gate": EXPECTED_F53_GATE_SHA256,
            "report": EXPECTED_F53_REPORT_SHA256,
            "contract": EXPECTED_CONTRACT_SHA256,
            "trainSelection": EXPECTED_TRAIN_SELECTION_SHA256,
        }
        or gate.get("f54PilotDatasetGenerationAuthorized") is not True
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or contract.get("authorizations", {}).get("f54PilotDatasetGenerationAuthorized")
        is not True
        or contract.get("authorizations", {}).get("cudaInferenceAuthorized") is not False
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
        or contract.get("dataset", {}).get("valueTarget") is not None
    ):
        raise Phase4F54Error("F53 does not authorize the frozen F54 dataset")
    return {
        "gate": gate,
        "report": report,
        "contract": contract,
        "trainSelection": train_selection,
        "hashes": hashes,
    }


def _materialize_split(
    workspace: Path,
    split: str,
    rows: Sequence[Mapping[str, object]],
    expected_games: int,
) -> tuple[dict[int, object], dict[int, Mapping[str, object]]]:
    metadata = {int(row["ordinal"]): row for row in rows}
    requested: dict[str, dict[int, int]] = defaultdict(dict)
    for ordinal, row in metadata.items():
        requested[str(row["gameId"])][int(row["plyIndex"])] = ordinal
    store = _open_store(workspace, split, expected_games)
    states: dict[int, object] = {}
    for game_index, game_id in enumerate(store.game_ids, 1):
        _game, game_states = store.load_game_with_states(game_id)
        for ply_index, ordinal in requested.get(game_id, {}).items():
            state = game_states[ply_index]
            row = metadata[ordinal]
            if (
                position_hash(state) != row["positionHash"]
                or position_orbit_digest(state) != row["positionOrbit"]
                or int(state.move_number) != row["absoluteMoveNumber"]
                or int(state.current_player) != row["currentPlayer"]
            ):
                raise Phase4F54Error(f"materialized {split} state differs from selection")
            states[ordinal] = state
        if game_index % 16 == 0 or game_index == store.game_count:
            _progress(
                "state-materialization",
                "running",
                split=split,
                games=game_index,
                gamesTotal=store.game_count,
                positions=len(states),
                positionsTotal=len(rows),
            )
    if set(states) != set(metadata):
        raise Phase4F54Error(f"not all {split} states could be materialized")
    return states, metadata


def _search_chunk_path(band: str, index: int) -> Path:
    return SEARCH_ROOT / f"train-{band}" / f"chunk-{index:04d}.json"


def _run_search_band(
    workspace: Path,
    states: Mapping[int, object],
    metadata: Mapping[int, Mapping[str, object]],
    band: str,
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
    actors: int,
) -> list[dict[str, object]]:
    ordinals = sorted(ordinal for ordinal, row in metadata.items() if row["band"] == band)
    node_limit = BUDGETS[band]
    chunks = [
        ordinals[start : start + SEARCH_CHUNK_POSITIONS]
        for start in range(0, len(ordinals), SEARCH_CHUNK_POSITIONS)
    ]
    references: list[dict[str, object] | None] = [None] * len(chunks)

    def build(index: int, chunk: Sequence[int]) -> tuple[int, dict[str, object]]:
        relative = _search_chunk_path(band, index)
        path = workspace / relative
        if path.exists():
            payload = _mapping(path, "F54 search chunk")
            _verify_self_hash(payload, "chunkSha256", "F54 search chunk")
            if (
                payload.get("contractSha256") != contract_sha
                or payload.get("implementationSha256") != source_sha
                or payload.get("selectionSha256") != selection_sha
                or payload.get("band") != band
                or payload.get("nodeLimit") != node_limit
                or payload.get("ordinals") != list(chunk)
            ):
                raise Phase4F54Error("existing F54 search chunk identity differs")
            return index, payload
        records: list[dict[str, object]] = []
        with ClassicAIBridge(request_timeout=600.0) as bridge:
            for ordinal in chunk:
                records.append(
                    {
                        "ordinal": ordinal,
                        "band": band,
                        "positionHash": metadata[ordinal]["positionHash"],
                        "analysis": _analysis_with_bridge(
                            bridge, states[ordinal], node_limit
                        ),
                    }
                )
        payload = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-train-classic-root-chunk",
                "status": "completed",
                "contractSha256": contract_sha,
                "implementationSha256": source_sha,
                "selectionSha256": selection_sha,
                "band": band,
                "nodeLimit": node_limit,
                "chunkIndex": index,
                "ordinals": list(chunk),
                "records": records,
            },
            "chunkSha256",
        )
        _write_immutable(path, payload, "F54 search chunk")
        return index, payload

    with ThreadPoolExecutor(max_workers=actors) as pool:
        futures = {
            pool.submit(build, index, chunk): index
            for index, chunk in enumerate(chunks)
        }
        completed = 0
        for future in as_completed(futures):
            index, payload = future.result()
            relative = _search_chunk_path(band, index)
            references[index] = {
                "path": relative.as_posix(),
                "sha256": payload["chunkSha256"],
                "band": band,
                "nodeLimit": node_limit,
                "positions": len(payload["records"]),
                "ordinals": payload["ordinals"],
            }
            completed += len(payload["records"])
            _progress(
                "classic-root",
                "running",
                band=band,
                nodeLimit=node_limit,
                positions=completed,
                positionsTotal=len(ordinals),
            )
            _resources(workspace)
    return [dict(reference) for reference in references if reference is not None]


def _load_search_records(
    workspace: Path, references: Sequence[Mapping[str, object]]
) -> dict[int, Mapping[str, object]]:
    records: dict[int, Mapping[str, object]] = {}
    source_sha = _source(workspace)["sha256"]
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F54 search chunk")
        if (
            _verify_self_hash(chunk, "chunkSha256", "F54 search chunk")
            != reference.get("sha256")
            or chunk.get("contractSha256") != EXPECTED_CONTRACT_SHA256
            or chunk.get("implementationSha256") != source_sha
            or chunk.get("selectionSha256") != EXPECTED_TRAIN_SELECTION_SHA256
            or chunk.get("band") != reference.get("band")
            or chunk.get("nodeLimit") != reference.get("nodeLimit")
            or chunk.get("ordinals") != reference.get("ordinals")
        ):
            raise Phase4F54Error("F54 search chunk reference changed")
        for row in chunk["records"]:
            ordinal = int(row["ordinal"])
            if ordinal in records:
                raise Phase4F54Error("F54 search ordinal is duplicated")
            analysis = row.get("analysis")
            if (
                not isinstance(analysis, Mapping)
                or row.get("band") != reference.get("band")
                or analysis.get("budget", {}).get("limit")
                != reference.get("nodeLimit")
            ):
                raise Phase4F54Error("F54 search record differs from its chunk")
            records[ordinal] = row["analysis"]
    return records


def _load_corrected_development(
    workspace: Path,
    metadata: Mapping[int, Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    f51_manifest = _mapping(
        workspace / F51_TEACHER_ROOT / "manifest.json", "F51 teacher manifest"
    )
    f51_primary = _load_analysis_records(
        workspace, f51_manifest["primary"]["chunks"]
    )
    f51_high = _load_analysis_records(
        workspace, f51_manifest["stability"]["chunks"]
    )
    f52_manifest = _mapping(workspace / F52_SEARCH_MANIFEST, "F52 search manifest")
    if (
        _verify_self_hash(f52_manifest, "manifestSha256", "F52 search manifest")
        != EXPECTED_F52_SEARCH_SHA256
    ):
        raise Phase4F54Error("F52 search manifest changed")
    generated = _load_f52_records(
        workspace, f52_manifest["generatedPrimaryChunks"]
    )
    corrected: dict[int, Mapping[str, object]] = {}
    for ordinal, row in metadata.items():
        if row["band"] == "late":
            corrected[ordinal] = f51_primary[ordinal]
        elif ordinal in f51_high:
            corrected[ordinal] = f51_high[ordinal]
        else:
            corrected[ordinal] = generated[ordinal]
    if len(corrected) != sum(DEVELOPMENT_COUNTS.values()):
        raise Phase4F54Error("F52 development analysis inventory is incomplete")
    return corrected


def _analysis_digest(analysis: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(analysis))


def build_sample(
    state: object,
    metadata: Mapping[str, object],
    analysis: Mapping[str, object],
    contract_sha: str,
) -> dict[str, object]:
    policy = tie_preserving_rank_target(
        analysis,
        POLICY_TEMPERATURE,
        top_k=POLICY_TOP_K,
        searched_mass=SEARCHED_MASS,
    )
    sample = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f54-learner-classic-policy-sample",
            "contractSha256": contract_sha,
            "ordinal": int(metadata["ordinal"]),
            "split": metadata["split"],
            "band": metadata["band"],
            "source": {
                "gameId": metadata["gameId"],
                "plyIndex": int(metadata["plyIndex"]),
                "positionHash": metadata["positionHash"],
                "positionOrbit": metadata["positionOrbit"],
            },
            "state": state_to_serializable(state),
            "teacherAnalysisSha256": _analysis_digest(analysis),
            "teacherAnalysis": dict(analysis),
            "policy": [
                [action, probability] for action, probability in sorted(policy.items())
            ],
        },
        "sampleSha256",
    )
    validate_sample(sample, contract_sha)
    return sample


def validate_sample(sample: Mapping[str, object], contract_sha: str) -> None:
    _verify_self_hash(sample, "sampleSha256", "F54 sample")
    if sample.get("contractSha256") != contract_sha or "value" in sample:
        raise Phase4F54Error("F54 sample contract or value target is invalid")
    state = state_from_serializable(sample["state"])
    source = sample.get("source")
    analysis = sample.get("teacherAnalysis")
    if not isinstance(source, Mapping) or not isinstance(analysis, Mapping):
        raise Phase4F54Error("F54 sample provenance is malformed")
    if (
        source.get("positionHash") != position_hash(state)
        or source.get("positionOrbit") != position_orbit_digest(state)
        or sample.get("teacherAnalysisSha256") != _analysis_digest(analysis)
    ):
        raise Phase4F54Error("F54 sample state or teacher analysis changed")
    band = str(sample["band"])
    budget = analysis.get("budget")
    if (
        band not in BUDGETS
        or not isinstance(budget, Mapping)
        or budget.get("kind") != "nodes"
        or budget.get("limit") != BUDGETS[band]
        or budget.get("maxDepth") != 5
        or analysis.get("format") != "legal-action-root-scores-v1"
        or analysis.get("rootScoreKind") != "alpha-beta-return-v1"
    ):
        raise Phase4F54Error("F54 teacher budget or root format differs")
    legal = set(legal_action_indices(state))
    root_scores = analysis.get("rootScores")
    if not isinstance(root_scores, list):
        raise Phase4F54Error("F54 root score inventory is malformed")
    root_actions = [int(row["actionIndex"]) for row in root_scores]
    if (
        len(root_actions) != len(set(root_actions))
        or set(root_actions) != legal
        or analysis.get("legalActions") != len(legal)
        or analysis.get("actionIndex") not in legal
    ):
        raise Phase4F54Error("F54 root score legal action inventory differs")
    policy_rows = sample.get("policy")
    if not isinstance(policy_rows, list):
        raise Phase4F54Error("F54 policy is malformed")
    policy = {int(row[0]): float(row[1]) for row in policy_rows}
    if (
        len(policy) != len(policy_rows)
        or set(policy) != legal
        or any(not math.isfinite(value) or value <= 0 for value in policy.values())
        or abs(sum(policy.values()) - 1.0) > 1e-9
    ):
        raise Phase4F54Error("F54 policy is illegal or not normalized")
    expected = tie_preserving_rank_target(
        analysis,
        POLICY_TEMPERATURE,
        top_k=POLICY_TOP_K,
        searched_mass=SEARCHED_MASS,
    )
    if policy != expected:
        raise Phase4F54Error("F54 policy does not match its teacher root analysis")
    mirrored = {mirror_action_index(action): value for action, value in policy.items()}
    restored = {mirror_action_index(action): value for action, value in mirrored.items()}
    if max(abs(policy[action] - restored[action]) for action in policy) > 1e-12:
        raise Phase4F54Error("F54 policy mirror round trip changed")


def _dataset_chunk_path(split: str, index: int) -> Path:
    return DATASET_ROOT / split / f"chunk-{index:04d}.json"


def _write_dataset_chunks(
    workspace: Path,
    split: str,
    states: Mapping[int, object],
    metadata: Mapping[int, Mapping[str, object]],
    analyses: Mapping[int, Mapping[str, object]],
    contract_sha: str,
    source_sha: str,
) -> list[dict[str, object]]:
    ordinals = sorted(metadata)
    references: list[dict[str, object]] = []
    for index, start in enumerate(range(0, len(ordinals), DATASET_CHUNK_POSITIONS)):
        chunk_ordinals = ordinals[start : start + DATASET_CHUNK_POSITIONS]
        samples = [
            build_sample(states[ordinal], metadata[ordinal], analyses[ordinal], contract_sha)
            for ordinal in chunk_ordinals
        ]
        payload = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-policy-dataset-chunk",
                "status": "completed",
                "contractSha256": contract_sha,
                "implementationSha256": source_sha,
                "split": split,
                "chunkIndex": index,
                "positions": len(samples),
                "ordinals": chunk_ordinals,
                "samples": samples,
            },
            "chunkSha256",
        )
        relative = _dataset_chunk_path(split, index)
        _write_immutable(workspace / relative, payload, "F54 dataset chunk")
        references.append(
            {
                "path": relative.as_posix(),
                "sha256": payload["chunkSha256"],
                "split": split,
                "positions": len(samples),
                "ordinals": chunk_ordinals,
            }
        )
        _progress(
            "dataset",
            "running",
            split=split,
            positions=min(start + len(samples), len(ordinals)),
            positionsTotal=len(ordinals),
        )
        _resources(workspace)
    return references


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * quantile
    lower, upper = math.floor(location), math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _deep_verify_dataset(
    workspace: Path,
    references: Sequence[Mapping[str, object]],
    contract_sha: str,
    source_sha: str,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    bands: Counter[tuple[str, str]] = Counter()
    sides: Counter[tuple[str, int]] = Counter()
    games: dict[str, set[str]] = defaultdict(set)
    orbits: dict[str, set[str]] = defaultdict(set)
    sample_hashes: list[str] = []
    entropies: dict[str, list[float]] = defaultdict(list)
    top1: dict[str, list[float]] = defaultdict(list)
    dev_analyses: list[Mapping[str, object]] = []
    seen_ordinals: set[int] = set()
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F54 dataset chunk")
        if (
            _verify_self_hash(chunk, "chunkSha256", "F54 dataset chunk")
            != reference.get("sha256")
            or chunk.get("contractSha256") != contract_sha
            or chunk.get("implementationSha256") != source_sha
            or chunk.get("split") != reference.get("split")
            or chunk.get("ordinals") != reference.get("ordinals")
        ):
            raise Phase4F54Error("F54 dataset chunk reference changed")
        samples = chunk.get("samples")
        if not isinstance(samples, list) or len(samples) != reference.get("positions"):
            raise Phase4F54Error("F54 dataset chunk position count changed")
        for sample in samples:
            validate_sample(sample, contract_sha)
            ordinal = int(sample["ordinal"])
            if ordinal in seen_ordinals:
                raise Phase4F54Error("F54 dataset ordinal is duplicated")
            seen_ordinals.add(ordinal)
            split = str(sample["split"])
            band = str(sample["band"])
            source = sample["source"]
            policy = [float(row[1]) for row in sample["policy"]]
            counts[split] += 1
            bands[(split, band)] += 1
            sides[(split, int(sample["state"]["currentPlayer"]))] += 1
            games[split].add(str(source["gameId"]))
            orbit = str(source["positionOrbit"])
            if orbit in orbits[split]:
                raise Phase4F54Error("F54 dataset contains a duplicate split orbit")
            orbits[split].add(orbit)
            sample_hashes.append(str(sample["sampleSha256"]))
            entropies[split].append(-sum(value * math.log(value) for value in policy))
            top1[split].append(max(policy))
            if split == "development":
                dev_analyses.append(sample["teacherAnalysis"])
    expected_counts = {
        "train": sum(TRAIN_COUNTS.values()),
        "development": sum(DEVELOPMENT_COUNTS.values()),
    }
    if dict(counts) != expected_counts:
        raise Phase4F54Error("F54 dataset split counts differ")
    actual_bands = {
        split: {
            band: bands[(split, band)] for band in ("early", "middle", "late")
        }
        for split in ("train", "development")
    }
    if actual_bands != {
        "train": TRAIN_COUNTS,
        "development": DEVELOPMENT_COUNTS,
    }:
        raise Phase4F54Error("F54 dataset ply-band counts differ")
    if {str(side): sides[("train", side)] for side in (0, 1)} != {
        "0": sum(TRAIN_COUNTS.values()) // 2,
        "1": sum(TRAIN_COUNTS.values()) // 2,
    }:
        raise Phase4F54Error("F54 train side balance differs")
    if orbits["train"] & orbits["development"] or games["train"] & games["development"]:
        raise Phase4F54Error("F54 train/development isolation failed")
    calibration = summarize_targets(dev_analyses, POLICY_TEMPERATURE)
    f52_target = _mapping(workspace / F52_TARGET_RESULT, "F52 target result")
    if (
        _verify_self_hash(f52_target, "resultSha256", "F52 target result")
        != EXPECTED_F52_TARGET_SHA256
    ):
        raise Phase4F54Error("F52 target result changed")
    expected_calibration = next(
        candidate
        for candidate in f52_target["candidates"]
        if float(candidate["temperature"]) == POLICY_TEMPERATURE
    )
    for key, value in calibration.items():
        if key not in expected_calibration or expected_calibration[key] != value:
            raise Phase4F54Error("F54 development targets do not reproduce F52 calibration")
    fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "contractSha256": contract_sha,
                "chunkSha256s": [reference["sha256"] for reference in references],
                "sampleSha256s": sample_hashes,
            }
        )
    )
    return {
        "deepVerified": True,
        "datasetFingerprint": fingerprint,
        "positions": dict(counts),
        "uniqueMirrorOrbits": {split: len(values) for split, values in sorted(orbits.items())},
        "trainDevelopmentOrbitOverlap": 0,
        "trainDevelopmentGameOverlap": 0,
        "byPlyBand": actual_bands,
        "bySideToMove": {
            split: {str(side): sides[(split, side)] for side in (0, 1)}
            for split in ("train", "development")
        },
        "policy": {
            split: {
                "meanEntropy": mean(entropies[split]),
                "top1ProbabilityP95": _percentile(top1[split], 0.95),
            }
            for split in ("train", "development")
        },
        "developmentF52Calibration": calibration,
    }


def _load_manifest_search_records(
    workspace: Path, manifest: Mapping[str, object]
) -> dict[int, Mapping[str, object]]:
    references = manifest.get("chunks")
    if not isinstance(references, list):
        raise Phase4F54Error("F54 search manifest chunk inventory is malformed")
    return _load_search_records(workspace, references)


def run_phase4f54(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f54(workspace), "reused": True}
    evidence = _evidence(workspace)
    contract = evidence["contract"]
    source = _source(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if ledger.total_hours != float(contract["resources"]["baselineGlobalGpuHours"]):
        raise Phase4F54Error("GPU ledger changed before the CPU-only F54 run")
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        resources = _resources(workspace)
        _progress(
            "preflight",
            "verified",
            contractSha256=contract["contractSha256"],
            globalGpuHours=ledger.total_hours,
        )
        f51_selection = _mapping(workspace / F51_SELECTION, "F51 selection")
        if (
            _verify_self_hash(f51_selection, "selectionSha256", "F51 selection")
            != EXPECTED_F51_SELECTION_SHA256
        ):
            raise Phase4F54Error("F51 selection changed")
        train_rows = evidence["trainSelection"]["rows"]
        development_rows = [
            row
            for row in f51_selection["rows"]
            if row.get("split") == "development"
        ]
        train_states, train_metadata = _materialize_split(
            workspace, "train", train_rows, 256
        )
        dev_states, dev_metadata = _materialize_split(
            workspace, "development", development_rows, 64
        )
        actors = int(contract["generation"]["classicActors"])
        references: list[dict[str, object]] = []
        for band in ("early", "middle", "late"):
            references.extend(
                _run_search_band(
                    workspace,
                    train_states,
                    train_metadata,
                    band,
                    str(contract["contractSha256"]),
                    str(source["sha256"]),
                    str(evidence["hashes"]["trainSelection"]),
                    actors,
                )
            )
        train_analyses = _load_search_records(workspace, references)
        if set(train_analyses) != set(train_metadata):
            raise Phase4F54Error("F54 train search inventory is incomplete")
        search_manifest = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-train-classic-root-manifest",
                "status": "completed",
                "contractSha256": contract["contractSha256"],
                "implementationSha256": source["sha256"],
                "selectionSha256": evidence["hashes"]["trainSelection"],
                "positions": len(train_analyses),
                "byPlyBand": dict(TRAIN_COUNTS),
                "nodeBudgetByBand": dict(BUDGETS),
                "chunks": references,
            },
            "manifestSha256",
        )
        _write_immutable(workspace / SEARCH_MANIFEST, search_manifest, "F54 search manifest")
        dev_analyses = _load_corrected_development(workspace, dev_metadata)
        dataset_references = _write_dataset_chunks(
            workspace,
            "train",
            train_states,
            train_metadata,
            train_analyses,
            str(contract["contractSha256"]),
            str(source["sha256"]),
        )
        dataset_references.extend(
            _write_dataset_chunks(
                workspace,
                "development",
                dev_states,
                dev_metadata,
                dev_analyses,
                str(contract["contractSha256"]),
                str(source["sha256"]),
            )
        )
        verification = _deep_verify_dataset(
            workspace,
            dataset_references,
            str(contract["contractSha256"]),
            str(source["sha256"]),
        )
        manifest = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-learner-teacher-policy-dataset",
                "status": "completed",
                "format": contract["dataset"]["format"],
                "contractPath": F54_CONTRACT.as_posix(),
                "contractSha256": contract["contractSha256"],
                "implementationSha256": source["sha256"],
                "f53GateSha256": evidence["hashes"]["gate"],
                "trainSelectionSha256": evidence["hashes"]["trainSelection"],
                "f52SearchManifestSha256": EXPECTED_F52_SEARCH_SHA256,
                "searchManifestPath": SEARCH_MANIFEST.as_posix(),
                "searchManifestSha256": search_manifest["manifestSha256"],
                "datasetFingerprint": verification["datasetFingerprint"],
                "chunks": dataset_references,
                "verification": verification,
                "resources": resources,
                "gpuHoursUsed": 0.0,
                "trainingStarted": False,
                "optimizerSteps": 0,
            },
            "manifestSha256",
        )
        _write_immutable(workspace / DATASET_MANIFEST, manifest, "F54 dataset manifest")
        final_ledger = GpuBudgetLedger(
            workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
        )
        if final_ledger.total_hours != ledger.total_hours:
            raise Phase4F54Error("GPU ledger changed during CPU-only F54")
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-policy-dataset-generation-report",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "trainingStarted": False,
                "optimizerSteps": 0,
                "source": source,
                "contractSha256": contract["contractSha256"],
                "searchManifestSha256": search_manifest["manifestSha256"],
                "datasetManifestPath": DATASET_MANIFEST.as_posix(),
                "datasetManifestSha256": manifest["manifestSha256"],
                "datasetFingerprint": verification["datasetFingerprint"],
                "verification": verification,
                "resources": _resources(workspace),
                "globalGpuHours": final_ledger.total_hours,
                "authorizations": {
                    "f55PolicyHeadPilotPreparationAuthorized": True,
                    "cudaTrainingAuthorized": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "nextStage": {
                    "name": "Phase 4F55 isolated policy-head pilot preparation",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "reportSha256",
        )
        _write_immutable(workspace / REPORT, report, "F54 generation report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f54-policy-dataset-gate",
                "status": "completed",
                "datasetPassed": True,
                "outcomeBand": "isolated-policy-head-pilot-preparation-candidate",
                "manifestPath": DATASET_MANIFEST.as_posix(),
                "manifestSha256": manifest["manifestSha256"],
                "datasetFingerprint": verification["datasetFingerprint"],
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "positions": verification["positions"],
                "f55PolicyHeadPilotPreparationAuthorized": True,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "nextStage": report["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F54 dataset gate")
    return {**gate, "globalGpuHours": final_ledger.total_hours, "reused": False}


def verify_phase4f54(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    evidence = _evidence(workspace)
    source = _source(workspace)
    search = _mapping(workspace / SEARCH_MANIFEST, "F54 search manifest")
    manifest = _mapping(workspace / DATASET_MANIFEST, "F54 dataset manifest")
    report = _mapping(workspace / REPORT, "F54 report")
    gate = _mapping(workspace / GATE, "F54 gate")
    search_sha = _verify_self_hash(search, "manifestSha256", "F54 search manifest")
    manifest_sha = _verify_self_hash(manifest, "manifestSha256", "F54 dataset manifest")
    report_sha = _verify_self_hash(report, "reportSha256", "F54 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F54 gate")
    search_records = _load_manifest_search_records(workspace, search)
    if len(search_records) != sum(TRAIN_COUNTS.values()):
        raise Phase4F54Error("F54 search manifest position count differs")
    verification = _deep_verify_dataset(
        workspace,
        manifest.get("chunks", []),
        EXPECTED_CONTRACT_SHA256,
        str(source["sha256"]),
    )
    if (
        search.get("implementationSha256") != source["sha256"]
        or search.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or search.get("selectionSha256") != EXPECTED_TRAIN_SELECTION_SHA256
        or manifest.get("implementationSha256") != source["sha256"]
        or manifest.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or manifest.get("searchManifestSha256") != search_sha
        or manifest.get("datasetFingerprint") != verification["datasetFingerprint"]
        or report.get("source") != source
        or report.get("contractSha256") != evidence["hashes"]["contract"]
        or report.get("datasetManifestSha256") != manifest_sha
        or report.get("datasetFingerprint") != verification["datasetFingerprint"]
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or gate.get("manifestSha256") != manifest_sha
        or gate.get("reportSha256") != report_sha
        or gate.get("datasetFingerprint") != verification["datasetFingerprint"]
        or gate.get("datasetPassed") is not True
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F54Error("F54 immutable dataset evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f54-policy-dataset-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "manifestPath": DATASET_MANIFEST.as_posix(),
        "manifestSha256": manifest_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "datasetFingerprint": manifest["datasetFingerprint"],
        "positions": verification["positions"],
        "f55PolicyHeadPilotPreparationAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f54(args.workspace_root)
        if args.command == "run"
        else verify_phase4f54(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
