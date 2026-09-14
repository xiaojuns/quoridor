"""Phase 4F42 CPU-only corrected early-curriculum dataset generation.

The dataset stores raw states with sparse floating-point policy targets and
soft value targets.  It deliberately does not use ReplayStore because that
legacy path reconstructs value labels as terminal +/-1 outcomes.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .codec import position_hash, state_from_serializable, state_to_serializable
from .game import legal_action_indices
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f40_target_probe import (
    CHUNKS as F40_CHUNKS,
    RESULT as F40_RESULT,
    SELECTION as F40_SELECTION,
    _materialize_selection,
    _open_seed_store,
)
from .phase4f41_target_protocol import (
    CONTRACT as F41_CONTRACT,
    REPORT as F41_REPORT,
    corrected_target,
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
EXPECTED_F41_CONTRACT_SHA256 = (
    "eaad63d8015682131d2f9ba76f447af916375055ca666b3e15c128eca7e430c1"
)
EXPECTED_F41_REPORT_SHA256 = (
    "006f63be511e9db0030d0e5c55ee16008a52aaab306bc310d285a55afad4d463"
)
EXPECTED_F40_RESULT_SHA256 = (
    "51215cab27cf88c9f72d621d90794ddffe6c81ebc20b3d732f87cad71ab55e16"
)

ROOT = Path("training/runs/phase4/phase4f/f42-early-curriculum-dataset-v1")
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "dataset-manifest.json"
GATE = ROOT / "dataset-gate.json"
RUN_ID = "phase4f42-early-curriculum-dataset-v1"
CHUNK_POSITIONS = 64
TRAIN_POSITIONS = 4096
DEVELOPMENT_POSITIONS = 512
WRITE_RESERVE_BYTES = 256 * 1024**2


class Phase4F42Error(RuntimeError):
    """Raised when corrected curriculum data cannot be generated safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f42-curriculum-dataset-progress",
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
        raise Phase4F42Error("workspace does not contain training/quoridor")
    return workspace


def _implementation_sha256(workspace: Path) -> str:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f42_curriculum_dataset.py",
            "training/quoridor/phase4f41_target_protocol.py",
            "training/quoridor/phase4f40_target_probe.py",
            "training/quoridor/codec.py",
            "training/quoridor/game.py",
        ),
    ).sha256


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F42Error("16 GiB artifact cap cannot reserve F42 outputs")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F42Error("insufficient free disk for F42 outputs")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "reservedWriteBytes": WRITE_RESERVE_BYTES,
    }


def _load_evidence(
    workspace: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    contract = _mapping(workspace / F41_CONTRACT, "F41 contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F41 contract")
    report = _mapping(workspace / F41_REPORT, "F41 report")
    report_sha = _verify_self_hash(report, "reportSha256", "F41 report")
    result = _mapping(workspace / F40_RESULT, "F40 result")
    result_sha = _verify_self_hash(result, "resultSha256", "F40 result")
    selection = _mapping(workspace / F40_SELECTION, "F40 selection")
    selection_sha = _verify_self_hash(selection, "selectionSha256", "F40 selection")
    if (
        contract_sha != EXPECTED_F41_CONTRACT_SHA256
        or report_sha != EXPECTED_F41_REPORT_SHA256
        or result_sha != EXPECTED_F40_RESULT_SHA256
        or contract.get("status") != "frozen"
        or contract.get("correctionReportSha256") != report_sha
        or contract.get("f40ResultSha256") != result_sha
        or contract.get("authorizations", {}).get("datasetGenerationAuthorized") is not True
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
        or report.get("targetProtocolPassed") is not True
        or report.get("trainingStarted") is not False
        or result.get("selectionSha256") != selection_sha
    ):
        raise Phase4F42Error("F40/F41 evidence does not authorize F42")
    chunk_rows: list[dict[str, object]] = []
    references = result.get("chunks")
    if not isinstance(references, list) or len(references) != 72:
        raise Phase4F42Error("F40 result has an invalid chunk inventory")
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F40 chunk")
        chunk_sha = _verify_self_hash(chunk, "chunkSha256", "F40 chunk")
        if chunk_sha != reference.get("sha256"):
            raise Phase4F42Error("F40 chunk reference changed")
        chunk_rows.extend(dict(row) for row in chunk["rows"])
    if len(chunk_rows) != TRAIN_POSITIONS + DEVELOPMENT_POSITIONS:
        raise Phase4F42Error("F40 chunks do not contain the authorized positions")
    return contract, report, selection, chunk_rows


def _chunk_path(split: str, split_index: int) -> Path:
    return DATASET / split / f"chunk-{split_index:04d}.json"


def _sample_payload(
    position: object,
    row: Mapping[str, object],
    temperature: float,
    blocks: Sequence[Mapping[str, float]],
    contract_sha256: str,
) -> dict[str, object]:
    policy, value, components = corrected_target(row, temperature, blocks)
    state = position.state
    if int(row["ordinal"]) != position.ordinal or row.get("split") != position.split:
        raise Phase4F42Error("materialized position differs from F40 row")
    sample = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f42-curriculum-target-sample",
            "contractSha256": contract_sha256,
            "ordinal": position.ordinal,
            "split": position.split,
            "band": position.band,
            "source": {
                "gameId": position.game_id,
                "plyIndex": position.ply_index,
                "teacherActionIndex": position.teacher_action_index,
                "teacherOrbitSha256": position.orbit_sha256,
                "positionSha256": sha256_bytes(position_hash(state).encode("ascii")),
            },
            "state": state_to_serializable(state),
            "policy": [[action, probability] for action, probability in sorted(policy.items())],
            "value": value,
            "valueComponents": components,
        },
        "sampleSha256",
    )
    _validate_sample(sample, contract_sha256)
    return sample


def _validate_sample(sample: Mapping[str, object], contract_sha256: str) -> None:
    _verify_self_hash(sample, "sampleSha256", "F42 sample")
    if sample.get("contractSha256") != contract_sha256:
        raise Phase4F42Error("F42 sample contract differs")
    state = state_from_serializable(sample["state"])
    source = sample.get("source")
    if not isinstance(source, Mapping):
        raise Phase4F42Error("F42 sample source is malformed")
    digest = sha256_bytes(position_hash(state).encode("ascii"))
    if source.get("positionSha256") != digest:
        raise Phase4F42Error("F42 sample state digest differs")
    legal = set(legal_action_indices(state))
    policy = sample.get("policy")
    if not isinstance(policy, list) or not policy:
        raise Phase4F42Error("F42 sample policy is empty")
    actions: set[int] = set()
    probability_sum = 0.0
    for item in policy:
        if not isinstance(item, list) or len(item) != 2:
            raise Phase4F42Error("F42 sparse policy row is malformed")
        action, probability = item
        if type(action) is not int or action not in legal or action in actions:
            raise Phase4F42Error("F42 sparse policy contains an invalid action")
        if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)) or probability <= 0:
            raise Phase4F42Error("F42 sparse policy contains an invalid probability")
        actions.add(action)
        probability_sum += float(probability)
    if abs(probability_sum - 1.0) > 1e-12:
        raise Phase4F42Error("F42 sparse policy is not normalized")
    teacher = source.get("teacherActionIndex")
    if type(teacher) is not int or teacher not in legal:
        raise Phase4F42Error("F42 teacher action is illegal")
    value = sample.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -1 <= value <= 1:
        raise Phase4F42Error("F42 value target is invalid")


def _write_chunks(
    workspace: Path,
    positions: Sequence[object],
    rows: Sequence[Mapping[str, object]],
    report: Mapping[str, object],
    contract_sha256: str,
    implementation_sha256: str,
) -> list[dict[str, object]]:
    calibration = report["policyCalibration"]
    value_calibration = report["valueCalibration"]
    temperature = float(calibration["classicTemperature"])
    blocks = value_calibration["blocks"]
    references: list[dict[str, object]] = []
    split_indices = Counter()
    for chunk_index, start in enumerate(range(0, len(rows), CHUNK_POSITIONS)):
        chunk_rows = rows[start : start + CHUNK_POSITIONS]
        chunk_positions = positions[start : start + CHUNK_POSITIONS]
        split = str(chunk_rows[0]["split"])
        if any(row.get("split") != split for row in chunk_rows):
            raise Phase4F42Error("F42 chunk crosses a split boundary")
        split_index = split_indices[split]
        split_indices[split] += 1
        samples = [
            _sample_payload(position, row, temperature, blocks, contract_sha256)
            for position, row in zip(chunk_positions, chunk_rows, strict=True)
        ]
        payload = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f42-curriculum-target-chunk",
                "status": "completed",
                "contractSha256": contract_sha256,
                "implementationSha256": implementation_sha256,
                "chunkIndex": chunk_index,
                "split": split,
                "splitChunkIndex": split_index,
                "positions": len(samples),
                "firstOrdinal": samples[0]["ordinal"],
                "lastOrdinal": samples[-1]["ordinal"],
                "samples": samples,
            },
            "chunkSha256",
        )
        relative = _chunk_path(split, split_index)
        _write_immutable(workspace / relative, payload, "F42 dataset chunk")
        references.append(
            {
                "path": relative.as_posix(),
                "sha256": payload["chunkSha256"],
                "split": split,
                "positions": len(samples),
                "firstOrdinal": samples[0]["ordinal"],
                "lastOrdinal": samples[-1]["ordinal"],
            }
        )
        _progress(
            "dataset",
            "chunk-completed",
            chunk=chunk_index + 1,
            chunksTotal=72,
            positions=min(start + len(samples), len(rows)),
            positionsTotal=len(rows),
        )
    return references


def _deep_verify_chunks(
    workspace: Path, references: Sequence[Mapping[str, object]], contract_sha256: str
) -> dict[str, object]:
    counts = Counter()
    bands = Counter()
    sides = Counter()
    orbits: set[str] = set()
    sample_hashes: list[str] = []
    ordinals: list[int] = []
    policies: list[dict[int, float]] = []
    values: list[float] = []
    teacher_ranks: list[int] = []
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F42 dataset chunk")
        chunk_sha = _verify_self_hash(chunk, "chunkSha256", "F42 dataset chunk")
        if chunk_sha != reference.get("sha256"):
            raise Phase4F42Error("F42 dataset chunk reference differs")
        for sample in chunk["samples"]:
            _validate_sample(sample, contract_sha256)
            source = sample["source"]
            orbit = str(source["teacherOrbitSha256"])
            if orbit in orbits:
                raise Phase4F42Error("F42 dataset contains a duplicate mirror orbit")
            orbits.add(orbit)
            split = str(sample["split"])
            counts[split] += 1
            bands[(split, str(sample["band"]))] += 1
            sides[(split, int(sample["state"]["currentPlayer"]))] += 1
            ordinals.append(int(sample["ordinal"]))
            sample_hashes.append(str(sample["sampleSha256"]))
            policy = {int(action): float(probability) for action, probability in sample["policy"]}
            policies.append(policy)
            values.append(float(sample["value"]))
            ranked = sorted(policy, key=lambda action: (-policy[action], action))
            teacher_ranks.append(ranked.index(int(source["teacherActionIndex"])) + 1)
    if ordinals != list(range(TRAIN_POSITIONS + DEVELOPMENT_POSITIONS)):
        raise Phase4F42Error("F42 dataset ordinals are incomplete")
    if counts != Counter({"train": TRAIN_POSITIONS, "development": DEVELOPMENT_POSITIONS}):
        raise Phase4F42Error("F42 dataset split counts differ")
    fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "contractSha256": contract_sha256,
                "chunkSha256s": [reference["sha256"] for reference in references],
                "sampleSha256s": sample_hashes,
            }
        )
    )
    return {
        "deepVerified": True,
        "datasetFingerprint": fingerprint,
        "positions": dict(sorted(counts.items())),
        "uniqueMirrorOrbits": len(orbits),
        "byPlyBand": {
            split: {band: bands[(split, band)] for band in ("early", "middle", "late")}
            for split in ("train", "development")
        },
        "bySideToMove": {
            split: {str(side): sides[(split, side)] for side in (0, 1)}
            for split in ("train", "development")
        },
        "allTargets": {
            "policyMeanEntropy": mean(-sum(p * math.log(p) for p in policy.values()) for policy in policies),
            "policyTop1ProbabilityP95": _percentile([max(policy.values()) for policy in policies], 0.95),
            "classicActionTop5Coverage": mean(rank <= 5 for rank in teacher_ranks),
            "valueMeanAbsolute": mean(abs(value) for value in values),
            "valuePositiveRate": mean(value > 0 for value in values),
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def run_phase4f42(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f42(workspace), "reused": True}
    resources = _resources(workspace)
    contract, report, selection, rows = _load_evidence(workspace)
    implementation_sha256 = _implementation_sha256(workspace)
    _progress("preflight", "verified", contractSha256=contract["contractSha256"])
    store, _source_manifest = _open_seed_store(workspace)
    positions, materialized = _materialize_selection(store, selection["positions"])
    for expected, actual in zip(selection["positions"], materialized, strict=True):
        if expected.get("positionSha256") != actual.get("positionSha256"):
            raise Phase4F42Error("F40 selected source state changed")
    if [position.ordinal for position in positions] != list(range(len(rows))):
        raise Phase4F42Error("materialized F40 positions are reordered")
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=RUN_ID):
        references = _write_chunks(
            workspace,
            positions,
            rows,
            report,
            str(contract["contractSha256"]),
            implementation_sha256,
        )
        verification = _deep_verify_chunks(
            workspace, references, str(contract["contractSha256"])
        )
        manifest = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f42-corrected-early-curriculum-dataset",
                "status": "completed",
                "format": "raw-state-sparse-soft-policy-soft-value-v1",
                "contractPath": F41_CONTRACT.as_posix(),
                "contractSha256": contract["contractSha256"],
                "implementationSha256": implementation_sha256,
                "f40ResultSha256": EXPECTED_F40_RESULT_SHA256,
                "f41ReportSha256": EXPECTED_F41_REPORT_SHA256,
                "datasetFingerprint": verification["datasetFingerprint"],
                "chunks": references,
                "verification": verification,
                "resources": resources,
                "gpuHoursUsed": 0.0,
                "trainingStarted": False,
                "optimizerSteps": 0,
            },
            "manifestSha256",
        )
        _write_immutable(workspace / MANIFEST, manifest, "F42 dataset manifest")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f42-curriculum-dataset-gate",
                "status": "completed",
                "outcomeBand": "corrected-curriculum-dataset-ready",
                "manifestPath": MANIFEST.as_posix(),
                "manifestSha256": manifest["manifestSha256"],
                "datasetFingerprint": verification["datasetFingerprint"],
                "deepVerified": True,
                "positions": verification["positions"],
                "pilotPreparationAuthorized": True,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "nextStage": {
                    "name": "Phase 4F43 corrected early-curriculum training pilot preparation",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F42 dataset gate")
    return {**gate, "reused": False}


def verify_phase4f42(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    gate = _mapping(workspace / GATE, "F42 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F42 gate")
    manifest = _mapping(workspace / MANIFEST, "F42 manifest")
    manifest_sha = _verify_self_hash(manifest, "manifestSha256", "F42 manifest")
    if (
        gate.get("manifestSha256") != manifest_sha
        or gate.get("datasetFingerprint") != manifest.get("datasetFingerprint")
        or gate.get("deepVerified") is not True
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
        or manifest.get("trainingStarted") is not False
        or manifest.get("optimizerSteps") != 0
    ):
        raise Phase4F42Error("F42 manifest and gate are inconsistent")
    verification = _deep_verify_chunks(
        workspace, manifest["chunks"], str(manifest["contractSha256"])
    )
    if verification["datasetFingerprint"] != manifest.get("datasetFingerprint"):
        raise Phase4F42Error("F42 deep verification fingerprint differs")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f42-curriculum-dataset-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "manifestPath": MANIFEST.as_posix(),
        "manifestSha256": manifest_sha,
        "datasetFingerprint": manifest["datasetFingerprint"],
        "positions": verification["positions"],
        "pilotPreparationAuthorized": gate["pilotPreparationAuthorized"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_phase4f42(args.workspace_root) if args.command == "run" else verify_phase4f42(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
