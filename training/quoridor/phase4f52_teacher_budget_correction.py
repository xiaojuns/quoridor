"""Correct and verify the F51 classic teacher search budget by ply band."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_orbit_digest
from .classic_bridge import ClassicAIBridge, ClassicSearchBudget
from .codec import position_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f51_learner_teacher_probe import (
    ROOT as F51_ROOT,
    SELECTION as F51_SELECTION,
    TEACHER_ROOT as F51_TEACHER_ROOT,
    TARGET_RESULT as F51_TARGET_RESULT,
    _load_analysis_records,
    _open_store,
    _ranked_scores,
    select_stability_ordinals,
    summarize_targets,
    target_checks,
    verify_phase4f51,
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
EXPECTED_F51_GATE_SHA256 = "d0dcc5aae397d444a11459b04b6c5f585000bcfb2adf4efcef873312935be1d0"
EXPECTED_F51_REPORT_SHA256 = "5e0b39be88cf11f49c0ce40cf227dd6183dc55642dfc78e5f096343ed4d99164"
EXPECTED_F51_TARGET_SHA256 = "83774e4df3ab9ab9cd79c7fcd798f59ef6cd4e4871d868357ba11b0ccf4bf36f"
EXPECTED_F51_SELECTION_SHA256 = "57b3c0b41c766ca6db6a0e89300f5a29f6a7c0166b3db8177d7029f1b2ae8f0b"
EXPECTED_F51_TEACHER_SHA256 = "939049a52348a3c85f1de54ec6a1fc236d8ebf9ba078815bef30c976c9d60300"

CONTRACT = Path("training/configs/phase4f52-teacher-budget-correction-frozen-v1.json")
ROOT = Path("training/runs/phase4/phase4f/f52-teacher-budget-correction-v1")
PREPARATION = ROOT / "preparation-report.json"
SEARCH_ROOT = ROOT / "classic-root"
SEARCH_MANIFEST = SEARCH_ROOT / "manifest.json"
TARGET_RESULT = ROOT / "target-calibration.json"
REPORT = ROOT / "correction-report.json"
GATE = ROOT / "correction-gate.json"
RUN_ID_PREFIX = "phase4f52-teacher-budget-correction"

CLASSIC_ACTORS = min(8, max(1, (os.cpu_count() or 2) // 2))
PRIMARY_CHUNK = 8
REFERENCE_CHUNK = 4
OUTPUT_RESERVE_BYTES = 256 * 1024**2
STABILITY_MINIMUM = 0.90
PRIMARY_BUDGETS = {"early": 16_384, "middle": 16_384, "late": 4_096}
REFERENCE_BUDGETS = {"early": 65_536, "middle": 65_536, "late": 16_384}
F51_STABILITY = {"actionAgreement": 0.7890625, "meanTop12Jaccard": 0.8488728183581125}


class Phase4F52Error(RuntimeError):
    """Raised when F52 evidence or corrected root search is invalid."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f52-teacher-budget-correction-progress",
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
        raise Phase4F52Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=("training/quoridor/phase4f52_teacher_budget_correction.py",),
    ).to_payload()


def _f51_evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f51(workspace)
    gate = _mapping(workspace / F51_ROOT / "probe-gate.json", "F51 gate")
    report = _mapping(workspace / F51_ROOT / "probe-report.json", "F51 report")
    target = _mapping(workspace / F51_TARGET_RESULT, "F51 target result")
    selection = _mapping(workspace / F51_SELECTION, "F51 selection")
    teacher = _mapping(workspace / F51_TEACHER_ROOT / "manifest.json", "F51 teacher manifest")
    hashes = {
        "gate": _verify_self_hash(gate, "gateSha256", "F51 gate"),
        "report": _verify_self_hash(report, "reportSha256", "F51 report"),
        "target": _verify_self_hash(target, "resultSha256", "F51 target result"),
        "selection": _verify_self_hash(selection, "selectionSha256", "F51 selection"),
        "teacher": _verify_self_hash(teacher, "manifestSha256", "F51 teacher manifest"),
    }
    if (
        verification.get("status") != "verified"
        or hashes
        != {
            "gate": EXPECTED_F51_GATE_SHA256,
            "report": EXPECTED_F51_REPORT_SHA256,
            "target": EXPECTED_F51_TARGET_SHA256,
            "selection": EXPECTED_F51_SELECTION_SHA256,
            "teacher": EXPECTED_F51_TEACHER_SHA256,
        }
        or gate.get("outcomeBand") != "probe-correction-required"
        or gate.get("probePassed") is not False
        or gate.get("classicStabilityPassed") is not False
        or gate.get("targetCalibrationPassed") is not True
        or report.get("trainingStarted") is not False
        or report.get("largeDatasetGenerationStarted") is not False
    ):
        raise Phase4F52Error("F51 is not the accepted correction evidence")
    return {
        "gate": gate,
        "report": report,
        "target": target,
        "selection": selection,
        "teacher": teacher,
        "hashes": hashes,
    }


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    output_bytes = directory_size(workspace / ROOT) if (workspace / ROOT).exists() else 0
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + OUTPUT_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F52Error("16 GiB artifact cap cannot reserve F52 evidence")
    if free_bytes - OUTPUT_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F52Error("F52 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "f52OutputBytes": output_bytes,
        "reservedWriteBytes": OUTPUT_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def budget_for_band(band: str, *, reference: bool = False) -> int:
    budgets = REFERENCE_BUDGETS if reference else PRIMARY_BUDGETS
    try:
        return budgets[band]
    except KeyError as error:
        raise ValueError(f"unknown absolute ply band: {band}") from error


def _contract_payload(
    source: Mapping[str, object], evidence: Mapping[str, object], global_gpu_hours: float
) -> dict[str, object]:
    hashes = evidence["hashes"]
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f52-teacher-budget-correction-contract",
            "status": "frozen",
            "objective": (
                "replace unstable 4096-node early/middle classic labels with a "
                "banded budget proven against an independent higher-budget reference"
            ),
            "sourceSha256": source["sha256"],
            "inputs": {
                "f51GatePath": (F51_ROOT / "probe-gate.json").as_posix(),
                "f51GateSha256": hashes["gate"],
                "f51ReportPath": (F51_ROOT / "probe-report.json").as_posix(),
                "f51ReportSha256": hashes["report"],
                "f51TargetResultSha256": hashes["target"],
                "f51SelectionPath": F51_SELECTION.as_posix(),
                "f51SelectionSha256": hashes["selection"],
                "f51TeacherManifestPath": (F51_TEACHER_ROOT / "manifest.json").as_posix(),
                "f51TeacherManifestSha256": hashes["teacher"],
            },
            "dataset": {
                "source": "F51-development-selection-only",
                "positions": 512,
                "bands": {"early": 256, "middle": 192, "late": 64},
                "stabilityPositions": {"early": 64, "middle": 48, "late": 16},
                "newPrimarySearches": 336,
                "newReferenceSearches": 112,
                "screenTrajectoryUse": False,
                "valueTarget": None,
            },
            "teacherProtocol": {
                "engine": "classic-hard-negamax",
                "maxDepth": 5,
                "primaryNodeBudgetByBand": dict(PRIMARY_BUDGETS),
                "referenceNodeBudgetByBand": dict(REFERENCE_BUDGETS),
                "reuse": {
                    "primary16384": "F51 stability records for selected early/middle positions",
                    "primary4096": "F51 primary records for all late positions",
                    "reference16384": "F51 stability records for selected late positions",
                },
                "rootOutput": "legal-action-root-scores-v1",
                "scoreSemantics": "actual-alpha-beta-return-or-bound-v1",
                "topK": 12,
            },
            "stabilityGate": {
                "requiredScopes": ["all", "early", "middle", "late"],
                "minimumActionAgreement": STABILITY_MINIMUM,
                "minimumTop12Jaccard": STABILITY_MINIMUM,
                "thresholdRelaxationAllowed": False,
            },
            "policyTargetCalibration": {
                "algorithm": "tie-preserving-score-rank-softmax-v1",
                "rankTemperatureCandidates": [1.0, 1.5, 2.0, 3.0],
                "calibrationSplit": "same-frozen-F51-development-only",
                "searchedTopKMass": 0.97,
                "maximumExcludedLegalActionMass": 0.03,
                "acceptance": {
                    "meanEntropyMinimum": 0.80,
                    "meanEntropyMaximum": 1.80,
                    "top1ProbabilityP95Maximum": 0.75,
                    "classicMaximumScoreGroupMassMinimum": 0.35,
                    "classicTop5MassMinimum": 0.80,
                    "policyMassMaximumAbsoluteError": 1e-9,
                    "mirrorMaximumAbsoluteError": 1e-12,
                },
                "selection": "lowest-temperature-candidate-passing-all-development-gates",
            },
            "resources": {
                "baselineGlobalGpuHours": global_gpu_hours,
                "gpuHoursAuthorized": 0.0,
                "globalGpuHoursLimit": 120.0,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "maximumOutputBytes": OUTPUT_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
                "classicActors": CLASSIC_ACTORS,
            },
            "authorizations": {
                "correctedClassicRootAnalysisAuthorized": True,
                "cudaInferenceAuthorized": False,
                "largeDatasetGenerationAuthorized": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F52 corrected teacher search execution",
                "requiresSeparateUserConfirmation": False,
            },
        },
        "contractSha256",
    )


def prepare_phase4f52(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    evidence = _f51_evidence(workspace)
    source = _source(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    expected = _contract_payload(source, evidence, ledger.total_hours)
    run_id = f"{RUN_ID_PREFIX}-prepare-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        contract_path = workspace / CONTRACT
        if contract_path.exists():
            contract = _mapping(contract_path, "F52 contract")
            _verify_self_hash(contract, "contractSha256", "F52 contract")
            if contract != expected:
                raise Phase4F52Error("existing F52 contract differs from current frozen inputs")
        else:
            contract = expected
            _write_immutable(contract_path, contract, "F52 contract")
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f52-teacher-budget-correction-preparation",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "trainingStarted": False,
                "largeDatasetGenerationStarted": False,
                "source": source,
                "contractPath": CONTRACT.as_posix(),
                "contractSha256": contract["contractSha256"],
                "f51Failure": {
                    "actionAgreement": evidence["report"]["stabilityAudit"]["actionAgreement"],
                    "meanTop12Jaccard": evidence["report"]["stabilityAudit"]["meanTop12Jaccard"],
                    "targetCalibrationPassed": evidence["gate"]["targetCalibrationPassed"],
                },
                "correction": {
                    "primaryNodeBudgetByBand": dict(PRIMARY_BUDGETS),
                    "referenceNodeBudgetByBand": dict(REFERENCE_BUDGETS),
                    "newPrimarySearches": 336,
                    "newReferenceSearches": 112,
                },
                "resources": resources,
                "formalTrainingAuthorized": False,
                "nextStage": contract["nextStage"],
            },
            "reportSha256",
        )
        preparation_path = workspace / PREPARATION
        if preparation_path.exists():
            existing = _mapping(preparation_path, "F52 preparation")
            _verify_self_hash(existing, "reportSha256", "F52 preparation")
            if (
                existing.get("status") != "completed"
                or existing.get("source") != source
                or existing.get("contractSha256") != contract["contractSha256"]
                or existing.get("cpuOnly") is not True
                or existing.get("cudaAccessed") is not False
                or existing.get("trainingStarted") is not False
                or existing.get("largeDatasetGenerationStarted") is not False
            ):
                raise Phase4F52Error("existing F52 preparation report changed")
            report = existing
        else:
            _write_immutable(preparation_path, report, "F52 preparation report")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f52-preparation-result",
        "status": "completed",
        "contractPath": CONTRACT.as_posix(),
        "contractSha256": contract["contractSha256"],
        "reportPath": PREPARATION.as_posix(),
        "reportSha256": report["reportSha256"],
        "correctedSearchAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": contract["nextStage"],
    }


def _contract(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    prepared = prepare_phase4f52(workspace)
    contract = _mapping(workspace / CONTRACT, "F52 contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F52 contract")
    if (
        contract_sha != prepared["contractSha256"]
        or contract.get("sourceSha256") != _source(workspace)["sha256"]
        or contract.get("authorizations", {}).get("correctedClassicRootAnalysisAuthorized") is not True
        or contract.get("authorizations", {}).get("cudaInferenceAuthorized") is not False
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
    ):
        raise Phase4F52Error("F52 contract is not executable")
    return contract, _f51_evidence(workspace)


def _materialize_development_states(
    workspace: Path, selection: Mapping[str, object]
) -> tuple[dict[int, object], dict[int, Mapping[str, object]]]:
    rows = selection.get("rows")
    if not isinstance(rows, list) or len(rows) != 4608:
        raise Phase4F52Error("F51 selection row inventory changed")
    metadata = {
        int(row["ordinal"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("split") == "development"
    }
    if len(metadata) != 512:
        raise Phase4F52Error("F51 development selection must contain 512 positions")
    requested: dict[str, dict[int, int]] = defaultdict(dict)
    for ordinal, row in metadata.items():
        requested[str(row["gameId"])][int(row["plyIndex"])] = ordinal
    store = _open_store(workspace, "development", 64)
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
                raise Phase4F52Error("materialized development state differs from F51")
            states[ordinal] = state
        if game_index % 16 == 0 or game_index == store.game_count:
            _progress(
                "state-materialization",
                "running",
                games=game_index,
                gamesTotal=store.game_count,
                positions=len(states),
                positionsTotal=512,
            )
    if set(states) != set(metadata):
        raise Phase4F52Error("F51 development states could not all be materialized")
    return states, metadata


def _analysis_with_bridge(
    bridge: ClassicAIBridge, state: object, node_limit: int
) -> dict[str, object]:
    result = bridge.analyze(
        state,
        budget=ClassicSearchBudget.nodes(node_limit),
        max_depth=5,
        timeout=600.0,
    )
    scores = [
        {
            "actionIndex": item.action_index,
            "searched": item.searched,
            "score": item.score,
        }
        for item in result.root_scores
    ]
    return {
        "format": "legal-action-root-scores-v1",
        "budget": {"kind": "nodes", "limit": node_limit, "maxDepth": 5},
        "actionIndex": result.action_index,
        "score": result.score,
        "depth": result.depth,
        "nodes": result.nodes,
        "elapsedMs": result.elapsed_ms,
        "rootScoreKind": result.root_score_kind,
        "legalActions": len(scores),
        "searchedActions": sum(bool(item["searched"]) for item in scores),
        "rootScores": scores,
    }


def _chunk_path(kind: str, index: int) -> Path:
    return SEARCH_ROOT / kind / f"chunk-{index:04d}.json"


def _run_search_chunks(
    workspace: Path,
    states: Mapping[int, object],
    metadata: Mapping[int, Mapping[str, object]],
    ordinals: Sequence[int],
    *,
    kind: str,
    node_limit: int,
    chunk_size: int,
    contract_sha: str,
    source_sha: str,
) -> list[dict[str, object]]:
    chunks = [
        list(ordinals[start : start + chunk_size])
        for start in range(0, len(ordinals), chunk_size)
    ]
    references: list[dict[str, object] | None] = [None] * len(chunks)

    def build(index: int, chunk: Sequence[int]) -> tuple[int, dict[str, object]]:
        relative = _chunk_path(kind, index)
        path = workspace / relative
        if path.exists():
            payload = _mapping(path, "F52 classic chunk")
            _verify_self_hash(payload, "chunkSha256", "F52 classic chunk")
            if (
                payload.get("contractSha256") != contract_sha
                or payload.get("sourceSha256") != source_sha
                or payload.get("nodeLimit") != node_limit
                or payload.get("ordinals") != list(chunk)
            ):
                raise Phase4F52Error("F52 classic chunk identity differs")
            return index, payload
        records: list[dict[str, object]] = []
        with ClassicAIBridge(request_timeout=600.0) as bridge:
            for ordinal in chunk:
                records.append(
                    {
                        "ordinal": ordinal,
                        "band": metadata[ordinal]["band"],
                        "positionHash": metadata[ordinal]["positionHash"],
                        "analysis": _analysis_with_bridge(
                            bridge, states[ordinal], node_limit
                        ),
                    }
                )
        payload = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": f"phase4f52-{kind}-classic-root-chunk",
                "status": "completed",
                "contractSha256": contract_sha,
                "sourceSha256": source_sha,
                "nodeLimit": node_limit,
                "chunkIndex": index,
                "ordinals": list(chunk),
                "records": records,
            },
            "chunkSha256",
        )
        _write_immutable(path, payload, "F52 classic chunk")
        return index, payload

    with ThreadPoolExecutor(max_workers=CLASSIC_ACTORS) as pool:
        futures = {
            pool.submit(build, index, chunk): index
            for index, chunk in enumerate(chunks)
        }
        completed = 0
        for future in as_completed(futures):
            index, payload = future.result()
            relative = _chunk_path(kind, index)
            references[index] = {
                "path": relative.as_posix(),
                "sha256": payload["chunkSha256"],
                "positions": len(payload["records"]),
                "ordinals": payload["ordinals"],
            }
            completed += len(payload["records"])
            _progress(
                "classic-root",
                "running",
                analysis=kind,
                nodeLimit=node_limit,
                positions=completed,
                positionsTotal=len(ordinals),
            )
            _resources(workspace)
    return [dict(reference) for reference in references if reference is not None]


def _load_f52_records(
    workspace: Path, references: Sequence[Mapping[str, object]]
) -> dict[int, Mapping[str, object]]:
    records: dict[int, Mapping[str, object]] = {}
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F52 classic chunk")
        if (
            _verify_self_hash(chunk, "chunkSha256", "F52 classic chunk")
            != reference["sha256"]
        ):
            raise Phase4F52Error("F52 classic chunk reference changed")
        for row in chunk["records"]:
            ordinal = int(row["ordinal"])
            if ordinal in records:
                raise Phase4F52Error("F52 classic ordinal is duplicated")
            records[ordinal] = row["analysis"]
    return records


def _top12_jaccard(
    left: Mapping[str, object], right: Mapping[str, object]
) -> float:
    left_actions = {action for action, _score in _ranked_scores(left)[:12]}
    right_actions = {action for action, _score in _ranked_scores(right)[:12]}
    return len(left_actions & right_actions) / max(
        1, len(left_actions | right_actions)
    )


def summarize_banded_stability(
    primary: Mapping[int, Mapping[str, object]],
    reference: Mapping[int, Mapping[str, object]],
    metadata: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    if set(primary) != set(reference):
        raise ValueError("primary/reference stability ordinals differ")
    values: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {
        "all": []
    }
    for band in ("early", "middle", "late"):
        values[band] = []
    for ordinal in sorted(primary):
        band = str(metadata[ordinal]["band"])
        pair = (primary[ordinal], reference[ordinal])
        values["all"].append(pair)
        values[band].append(pair)
    result: dict[str, object] = {}
    for scope, pairs in values.items():
        agreements = [float(left["actionIndex"] == right["actionIndex"]) for left, right in pairs]
        jaccards = [_top12_jaccard(left, right) for left, right in pairs]
        result[scope] = {
            "positions": len(pairs),
            "actionAgreement": mean(agreements),
            "meanTop12Jaccard": mean(jaccards),
            "primaryCompletedDepths": dict(
                sorted(Counter(str(left["depth"]) for left, _right in pairs).items())
            ),
            "referenceCompletedDepths": dict(
                sorted(Counter(str(right["depth"]) for _left, right in pairs).items())
            ),
            "primaryMeanSearchedCoverage": mean(
                float(left["searchedActions"]) / float(left["legalActions"])
                for left, _right in pairs
            ),
        }
    return result


def stability_checks(summary: Mapping[str, object]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for scope in ("all", "early", "middle", "late"):
        item = summary[scope]
        checks[f"{scope}ActionAgreement"] = (
            float(item["actionAgreement"]) >= STABILITY_MINIMUM
        )
        checks[f"{scope}Top12Jaccard"] = (
            float(item["meanTop12Jaccard"]) >= STABILITY_MINIMUM
        )
    return checks


def summarize_protocol_shift(
    original: Mapping[int, Mapping[str, object]],
    corrected: Mapping[int, Mapping[str, object]],
    metadata: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for band in ("early", "middle", "late"):
        ordinals = sorted(
            ordinal for ordinal, row in metadata.items() if row["band"] == band
        )
        result[band] = {
            "positions": len(ordinals),
            "actionAgreementWithF51Primary": mean(
                float(original[ordinal]["actionIndex"] == corrected[ordinal]["actionIndex"])
                for ordinal in ordinals
            ),
            "meanTop12JaccardWithF51Primary": mean(
                _top12_jaccard(original[ordinal], corrected[ordinal])
                for ordinal in ordinals
            ),
        }
    return result


def _load_existing_search_manifest(
    workspace: Path, contract_sha: str, source_sha: str
) -> dict[str, object] | None:
    path = workspace / SEARCH_MANIFEST
    if not path.exists():
        return None
    manifest = _mapping(path, "F52 search manifest")
    _verify_self_hash(manifest, "manifestSha256", "F52 search manifest")
    if (
        manifest.get("contractSha256") != contract_sha
        or manifest.get("sourceSha256") != source_sha
    ):
        raise Phase4F52Error("F52 search manifest belongs to different inputs")
    return manifest


def run_phase4f52(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f52(workspace), "reused": True}
    contract, evidence = _contract(workspace)
    source = _source(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if ledger.total_hours != float(contract["resources"]["baselineGlobalGpuHours"]):
        raise Phase4F52Error("GPU ledger changed during CPU-only F52")
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        _progress("preflight", "verified", contractSha256=contract["contractSha256"], globalGpuHours=ledger.total_hours)
        states, metadata = _materialize_development_states(workspace, evidence["selection"])
        f51_primary = _load_analysis_records(
            workspace, evidence["teacher"]["primary"]["chunks"]
        )
        f51_high = _load_analysis_records(
            workspace, evidence["teacher"]["stability"]["chunks"]
        )
        stability_ordinals = select_stability_ordinals(evidence["selection"]["rows"])
        stability_set = set(stability_ordinals)

        new_primary = sorted(
            ordinal
            for ordinal, row in metadata.items()
            if row["band"] in {"early", "middle"} and ordinal not in stability_set
        )
        if len(new_primary) != 336:
            raise Phase4F52Error("F52 new primary search inventory differs")
        primary_refs = _run_search_chunks(
            workspace,
            states,
            metadata,
            new_primary,
            kind="primary-16384",
            node_limit=16_384,
            chunk_size=PRIMARY_CHUNK,
            contract_sha=str(contract["contractSha256"]),
            source_sha=str(source["sha256"]),
        )
        high_ordinals = sorted(
            ordinal
            for ordinal in stability_ordinals
            if metadata[ordinal]["band"] in {"early", "middle"}
        )
        if len(high_ordinals) != 112:
            raise Phase4F52Error("F52 high-budget reference inventory differs")
        reference_refs = _run_search_chunks(
            workspace,
            states,
            metadata,
            high_ordinals,
            kind="reference-65536",
            node_limit=65_536,
            chunk_size=REFERENCE_CHUNK,
            contract_sha=str(contract["contractSha256"]),
            source_sha=str(source["sha256"]),
        )
        generated_primary = _load_f52_records(workspace, primary_refs)
        generated_reference = _load_f52_records(workspace, reference_refs)

        corrected: dict[int, Mapping[str, object]] = {}
        origin_counts: Counter[str] = Counter()
        for ordinal, row in metadata.items():
            band = str(row["band"])
            if band == "late":
                corrected[ordinal] = f51_primary[ordinal]
                origin_counts["f51-primary-4096"] += 1
            elif ordinal in f51_high:
                corrected[ordinal] = f51_high[ordinal]
                origin_counts["f51-stability-16384"] += 1
            else:
                corrected[ordinal] = generated_primary[ordinal]
                origin_counts["f52-primary-16384"] += 1
        if len(corrected) != 512:
            raise Phase4F52Error("corrected teacher inventory is incomplete")

        stability_primary = {ordinal: corrected[ordinal] for ordinal in stability_ordinals}
        stability_reference: dict[int, Mapping[str, object]] = {}
        for ordinal in stability_ordinals:
            if metadata[ordinal]["band"] == "late":
                stability_reference[ordinal] = f51_high[ordinal]
            else:
                stability_reference[ordinal] = generated_reference[ordinal]
        stability = summarize_banded_stability(
            stability_primary, stability_reference, metadata
        )
        checks = stability_checks(stability)

        acceptance = contract["policyTargetCalibration"]["acceptance"]
        candidates: list[dict[str, object]] = []
        selected_temperature: float | None = None
        corrected_records = [corrected[ordinal] for ordinal in sorted(corrected)]
        for raw_temperature in contract["policyTargetCalibration"][
            "rankTemperatureCandidates"
        ]:
            candidate = summarize_targets(corrected_records, float(raw_temperature))
            candidate_checks = target_checks(candidate, acceptance)
            candidate = {
                **candidate,
                "checks": candidate_checks,
                "passed": all(candidate_checks.values()),
            }
            candidates.append(candidate)
            if selected_temperature is None and candidate["passed"]:
                selected_temperature = float(raw_temperature)
        target_passed = selected_temperature is not None
        protocol_shift = summarize_protocol_shift(
            {ordinal: f51_primary[ordinal] for ordinal in metadata}, corrected, metadata
        )

        search_manifest = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f52-corrected-classic-root-manifest",
                "status": "completed",
                "contractSha256": contract["contractSha256"],
                "sourceSha256": source["sha256"],
                "f51TeacherManifestSha256": EXPECTED_F51_TEACHER_SHA256,
                "developmentPositions": 512,
                "stabilityPositions": 128,
                "primaryNodeBudgetByBand": dict(PRIMARY_BUDGETS),
                "referenceNodeBudgetByBand": dict(REFERENCE_BUDGETS),
                "correctedPrimaryOrigins": dict(sorted(origin_counts.items())),
                "stabilityOrdinals": stability_ordinals,
                "generatedPrimaryChunks": primary_refs,
                "generatedReferenceChunks": reference_refs,
            },
            "manifestSha256",
        )
        existing_manifest = _load_existing_search_manifest(
            workspace, str(contract["contractSha256"]), str(source["sha256"])
        )
        if existing_manifest is not None and existing_manifest != search_manifest:
            raise Phase4F52Error("F52 immutable search manifest changed")
        if existing_manifest is None:
            _write_immutable(
                workspace / SEARCH_MANIFEST,
                search_manifest,
                "F52 search manifest",
            )

        target_result = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f52-corrected-teacher-target-calibration",
                "status": "completed",
                "contractSha256": contract["contractSha256"],
                "searchManifestSha256": search_manifest["manifestSha256"],
                "positions": 512,
                "candidates": candidates,
                "selectedRankTemperature": selected_temperature,
                "targetCalibrationPassed": target_passed,
                "valueTarget": None,
            },
            "resultSha256",
        )
        _write_immutable(
            workspace / TARGET_RESULT, target_result, "F52 target calibration"
        )
        stability_passed = all(checks.values())
        correction_passed = stability_passed and target_passed
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f52-teacher-budget-correction",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "trainingStarted": False,
                "optimizerSteps": 0,
                "largeDatasetGenerationStarted": False,
                "screenTrajectoriesUsed": False,
                "source": source,
                "contractPath": CONTRACT.as_posix(),
                "contractSha256": contract["contractSha256"],
                "f51Evidence": evidence["hashes"],
                "searchManifestPath": SEARCH_MANIFEST.as_posix(),
                "searchManifestSha256": search_manifest["manifestSha256"],
                "stability": {
                    "summary": stability,
                    "checks": checks,
                    "passed": stability_passed,
                    "f51Baseline": F51_STABILITY,
                },
                "protocolShift": protocol_shift,
                "targetCalibrationPath": TARGET_RESULT.as_posix(),
                "targetCalibrationSha256": target_result["resultSha256"],
                "targetCalibrationPassed": target_passed,
                "selectedRankTemperature": selected_temperature,
                "correctionPassed": correction_passed,
                "resources": resources,
                "globalGpuHours": ledger.total_hours,
                "authorizations": {
                    "f53PolicyHeadPilotPreparationAuthorized": correction_passed,
                    "largeDatasetGenerationAuthorized": False,
                    "cudaTrainingAuthorized": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "nextStage": {
                    "name": (
                        "Phase 4F53 policy-head-only pilot preparation"
                        if correction_passed
                        else "Phase 4F53 teacher search correction review"
                    ),
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "reportSha256",
        )
        _write_immutable(workspace / REPORT, report, "F52 correction report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f52-teacher-budget-correction-gate",
                "status": "completed",
                "correctionPassed": correction_passed,
                "classicStabilityPassed": stability_passed,
                "targetCalibrationPassed": target_passed,
                "outcomeBand": (
                    "policy-head-pilot-preparation-candidate"
                    if correction_passed
                    else "teacher-search-correction-insufficient"
                ),
                "selectedRankTemperature": selected_temperature,
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f53PolicyHeadPilotPreparationAuthorized": correction_passed,
                "largeDatasetGenerationAuthorized": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "nextStage": report["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F52 correction gate")
        return {**gate, "globalGpuHours": ledger.total_hours, "reused": False}


def verify_phase4f52(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F52 contract")
    report = _mapping(workspace / REPORT, "F52 report")
    gate = _mapping(workspace / GATE, "F52 gate")
    target = _mapping(workspace / TARGET_RESULT, "F52 target calibration")
    search = _mapping(workspace / SEARCH_MANIFEST, "F52 search manifest")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F52 contract")
    report_sha = _verify_self_hash(report, "reportSha256", "F52 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F52 gate")
    target_sha = _verify_self_hash(target, "resultSha256", "F52 target calibration")
    search_sha = _verify_self_hash(search, "manifestSha256", "F52 search manifest")
    primary_records = _load_f52_records(
        workspace, search.get("generatedPrimaryChunks", [])
    )
    reference_records = _load_f52_records(
        workspace, search.get("generatedReferenceChunks", [])
    )
    _f51_evidence(workspace)
    if (
        contract.get("sourceSha256") != _source(workspace)["sha256"]
        or report.get("source") != _source(workspace)
        or report.get("contractSha256") != contract_sha
        or report.get("searchManifestSha256") != search_sha
        or report.get("targetCalibrationSha256") != target_sha
        or gate.get("reportSha256") != report_sha
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or len(primary_records) != 336
        or len(reference_records) != 112
        or len(search.get("stabilityOrdinals", [])) != 128
    ):
        raise Phase4F52Error("F52 immutable evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f52-teacher-budget-correction-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "correctionPassed": gate["correctionPassed"],
        "outcomeBand": gate["outcomeBand"],
        "selectedRankTemperature": gate["selectedRankTemperature"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "verify"), nargs="?", default="run"
    )
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f52(args.workspace_root)
    elif args.command == "run":
        payload = run_phase4f52(args.workspace_root)
    else:
        payload = verify_phase4f52(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
