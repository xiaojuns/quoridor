"""Freeze the F54 learner-occupancy policy dataset generation protocol."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f45_curriculum_pilot_review import (
    EXPECTED_F21_CONFIG_FILE_SHA256,
    EXPECTED_F21_MODEL_SHA256,
    EXPECTED_F21_STATE_SHA256,
    F21_CONFIG,
    F21_MODEL,
)
from .phase4f51_learner_teacher_probe import SELECTION as F51_SELECTION
from .phase4f52_teacher_budget_correction import (
    GATE as F52_GATE,
    REPORT as F52_REPORT,
    SEARCH_MANIFEST as F52_SEARCH_MANIFEST,
    TARGET_RESULT as F52_TARGET_RESULT,
    verify_phase4f52,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_F52_GATE_SHA256 = "81f804e2e5c58446b117cb6fae10968d93288fd4c11f0819dc2fae1cec868ab8"
EXPECTED_F52_REPORT_SHA256 = "7ce644477c88fc252e603dce6b61a838f1dcd2085244eba07a6daf2a6c03f63f"
EXPECTED_F52_SEARCH_SHA256 = "d175b540b9653dcd715014464b973a52d133e5697dead7f09a2fd9046b865b94"
EXPECTED_F52_TARGET_SHA256 = "a4693b86962e0b85c842cb898fd093be255d1232c458f4c69545168cc84f4c7a"
EXPECTED_F51_SELECTION_SHA256 = "57b3c0b41c766ca6db6a0e89300f5a29f6a7c0166b3db8177d7029f1b2ae8f0b"

ROOT = Path("training/runs/phase4/phase4f/f53-policy-head-pilot-preparation-v1")
TRAIN_SELECTION = ROOT / "train-selection.json"
REPORT = ROOT / "preparation-report.json"
GATE = ROOT / "preparation-gate.json"
F54_CONTRACT = Path(
    "training/configs/phase4f54-learner-teacher-policy-dataset-frozen-v1.json"
)
RUN_ID_PREFIX = "phase4f53-policy-head-pilot-preparation"

TRAIN_COUNTS = {"early": 512, "middle": 384, "late": 128}
DEVELOPMENT_COUNTS = {"early": 256, "middle": 192, "late": 64}
PRIMARY_BUDGETS = {"early": 16_384, "middle": 16_384, "late": 4_096}
F54_OUTPUT_RESERVE_BYTES = 768 * 1024**2
F53_WRITE_RESERVE_BYTES = 16 * 1024**2
CLASSIC_ACTORS = min(8, max(1, (os.cpu_count() or 2) // 2))


class Phase4F53Error(RuntimeError):
    """Raised when F52 cannot safely authorize the isolated dataset step."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F53Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=("training/quoridor/phase4f53_policy_head_preparation.py",),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + F53_WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F53Error("16 GiB artifact cap cannot reserve F53 evidence")
    if artifact_bytes + F54_OUTPUT_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F53Error("16 GiB artifact cap cannot reserve the F54 pilot dataset")
    if free_bytes - F54_OUTPUT_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F53Error("F54 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "f53WriteReserveBytes": F53_WRITE_RESERVE_BYTES,
        "f54OutputReserveBytes": F54_OUTPUT_RESERVE_BYTES,
    }


def _f52_evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f52(workspace)
    gate = _mapping(workspace / F52_GATE, "F52 gate")
    report = _mapping(workspace / F52_REPORT, "F52 report")
    search = _mapping(workspace / F52_SEARCH_MANIFEST, "F52 search manifest")
    target = _mapping(workspace / F52_TARGET_RESULT, "F52 target result")
    hashes = {
        "gate": _verify_self_hash(gate, "gateSha256", "F52 gate"),
        "report": _verify_self_hash(report, "reportSha256", "F52 report"),
        "search": _verify_self_hash(search, "manifestSha256", "F52 search manifest"),
        "target": _verify_self_hash(target, "resultSha256", "F52 target result"),
    }
    if (
        verification.get("status") != "verified"
        or hashes
        != {
            "gate": EXPECTED_F52_GATE_SHA256,
            "report": EXPECTED_F52_REPORT_SHA256,
            "search": EXPECTED_F52_SEARCH_SHA256,
            "target": EXPECTED_F52_TARGET_SHA256,
        }
        or gate.get("correctionPassed") is not True
        or gate.get("classicStabilityPassed") is not True
        or gate.get("targetCalibrationPassed") is not True
        or gate.get("f53PolicyHeadPilotPreparationAuthorized") is not True
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or target.get("selectedRankTemperature") != 1.0
        or report.get("trainingStarted") is not False
        or report.get("largeDatasetGenerationStarted") is not False
    ):
        raise Phase4F53Error("F52 is not the accepted policy-target correction evidence")
    return {
        "gate": gate,
        "report": report,
        "search": search,
        "target": target,
        "hashes": hashes,
    }


def _initialization(workspace: Path) -> dict[str, object]:
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_FILE_SHA256:
        raise Phase4F53Error("F21 network config changed")
    if sha256_file(workspace / F21_MODEL) != EXPECTED_F21_MODEL_SHA256:
        raise Phase4F53Error("F21 initialization model changed")
    try:
        payload = torch.load(workspace / F21_MODEL, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F53Error("F21 initialization model is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != EXPECTED_F21_STATE_SHA256
        or payload.get("format") != "quoridor-phase4-candidate-model-only"
        or payload.get("generation") != 1
        or payload.get("step") != 192
    ):
        raise Phase4F53Error("F21 initialization identity changed")
    return {
        "modelPath": F21_MODEL.as_posix(),
        "modelFileSha256": EXPECTED_F21_MODEL_SHA256,
        "modelStateSha256": EXPECTED_F21_STATE_SHA256,
        "networkConfigPath": F21_CONFIG.as_posix(),
        "networkConfigFileSha256": EXPECTED_F21_CONFIG_FILE_SHA256,
        "resetOptimizer": True,
        "emptyReplay": True,
    }


def select_train_rows(
    rows: Sequence[Mapping[str, object]],
    counts: Mapping[str, int] = TRAIN_COUNTS,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for band in ("early", "middle", "late"):
        total = int(counts[band])
        if total <= 0 or total % 2:
            raise ValueError("train band counts must be positive and even")
        for side in (0, 1):
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row.get("split") == "train"
                    and row.get("band") == band
                    and row.get("currentPlayer") == side
                ),
                key=lambda row: (str(row["rank"]), str(row["positionOrbit"])),
            )
            required = total // 2
            if len(candidates) < required:
                raise Phase4F53Error(
                    f"F51 train selection cannot fill {band}/side-{side}"
                )
            selected.extend(dict(row) for row in candidates[:required])
    return sorted(selected, key=lambda row: int(row["ordinal"]))


def selection_audit(
    selected: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_orbits = [str(row["positionOrbit"]) for row in selected]
    development_orbits = {str(row["positionOrbit"]) for row in development}
    counts = Counter((str(row["band"]), int(row["currentPlayer"])) for row in selected)
    games = {str(row["gameId"]) for row in selected}
    development_games = {str(row["gameId"]) for row in development}
    return {
        "positions": len(selected),
        "uniqueOrbits": len(set(selected_orbits)),
        "developmentOrbitOverlap": len(set(selected_orbits) & development_orbits),
        "trainDevelopmentGameOverlap": len(games & development_games),
        "byBand": {
            band: sum(1 for row in selected if row["band"] == band)
            for band in ("early", "middle", "late")
        },
        "byBandAndSide": {
            f"{band}/side-{side}": counts[(band, side)]
            for band in ("early", "middle", "late")
            for side in (0, 1)
        },
    }


def selection_checks(audit: Mapping[str, object]) -> dict[str, bool]:
    expected_side = {
        f"{band}/side-{side}": total // 2
        for band, total in TRAIN_COUNTS.items()
        for side in (0, 1)
    }
    return {
        "positionCount": audit.get("positions") == sum(TRAIN_COUNTS.values()),
        "uniqueMirrorOrbits": audit.get("uniqueOrbits") == audit.get("positions"),
        "developmentOrbitDisjoint": audit.get("developmentOrbitOverlap") == 0,
        "wholeGameSplitDisjoint": audit.get("trainDevelopmentGameOverlap") == 0,
        "bandCounts": audit.get("byBand") == TRAIN_COUNTS,
        "bandSideBalance": audit.get("byBandAndSide") == expected_side,
    }


def _selection_payload(
    source_sha: str,
    f51_selection_sha: str,
    rows: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f53-policy-pilot-train-selection",
            "status": "completed",
            "sourceSha256": source_sha,
            "f51SelectionSha256": f51_selection_sha,
            "algorithm": "lowest-sha256-rank-per-absolute-ply-band-and-side-v1",
            "positions": len(rows),
            "audit": dict(audit),
            "rows": [dict(row) for row in rows],
        },
        "selectionSha256",
    )


def _f54_contract(
    source: Mapping[str, object],
    evidence: Mapping[str, object],
    selection: Mapping[str, object],
    initialization: Mapping[str, object],
    baseline_gpu_hours: float,
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f54-learner-teacher-policy-dataset-contract",
            "status": "frozen",
            "objective": (
                "build a leakage-free pilot-scale policy-only dataset from learner "
                "occupancy using the F52-stable classic search protocol"
            ),
            "f53SourceSha256": source["sha256"],
            "inputs": {
                "f52GatePath": F52_GATE.as_posix(),
                "f52GateSha256": evidence["hashes"]["gate"],
                "f52ReportPath": F52_REPORT.as_posix(),
                "f52ReportSha256": evidence["hashes"]["report"],
                "f52SearchManifestPath": F52_SEARCH_MANIFEST.as_posix(),
                "f52SearchManifestSha256": evidence["hashes"]["search"],
                "f52TargetPath": F52_TARGET_RESULT.as_posix(),
                "f52TargetSha256": evidence["hashes"]["target"],
                "f51SelectionPath": F51_SELECTION.as_posix(),
                "f51SelectionSha256": EXPECTED_F51_SELECTION_SHA256,
                "f53TrainSelectionPath": TRAIN_SELECTION.as_posix(),
                "f53TrainSelectionSha256": selection["selectionSha256"],
            },
            "dataset": {
                "format": "learner-occupancy-classic-policy-only-v1",
                "trainPositions": sum(TRAIN_COUNTS.values()),
                "developmentPositions": sum(DEVELOPMENT_COUNTS.values()),
                "trainByBand": dict(TRAIN_COUNTS),
                "developmentByBand": dict(DEVELOPMENT_COUNTS),
                "trainSelection": "balanced-by-absolute-ply-band-and-side-v1",
                "developmentSelection": "all-F52-corrected-development-positions",
                "uniqueMirrorOrbitRequired": True,
                "wholeGameSplitDisjointRequired": True,
                "screenTrajectoryUse": False,
                "valueTarget": None,
            },
            "teacher": {
                "engine": "classic-hard-negamax",
                "maxDepth": 5,
                "nodeBudgetByBand": dict(PRIMARY_BUDGETS),
                "rootOutput": "legal-action-root-scores-v1",
                "scoreSemantics": "actual-alpha-beta-return-or-bound-v1",
                "policyTarget": {
                    "algorithm": "tie-preserving-score-rank-softmax-v1",
                    "rankTemperature": 1.0,
                    "topK": 12,
                    "searchedTopKMass": 0.97,
                    "maximumExcludedLegalActionMass": 0.03,
                },
                "developmentRootReuseRequired": True,
            },
            "generation": {
                "device": "cpu",
                "classicActors": CLASSIC_ACTORS,
                "searchChunkPositions": 8,
                "datasetChunkPositions": 64,
                "resumableImmutableChunks": True,
                "trainingStarted": False,
                "optimizerSteps": 0,
            },
            "downstreamPilotIntent": {
                "initialization": dict(initialization),
                "selfPlayGames": 0,
                "replayStoreUsage": False,
                "trainableParameters": ["policy_head"],
                "frozenParameters": [
                    "input_conv",
                    "input_bn",
                    "trunk",
                    "value_features",
                    "value_head",
                ],
                "requiredInvariant": "frozen tensors and value outputs unchanged",
                "pilotTrainingRequiresLaterDatasetGateAndSeparateConfirmation": True,
            },
            "gates": {
                "legalPolicyActionsRequired": True,
                "policyMassMaximumAbsoluteError": 1e-9,
                "mirrorMaximumAbsoluteError": 1e-12,
                "developmentTargetFingerprintMustMatchF52": True,
                "trainDevelopmentOrbitOverlapMaximum": 0,
                "trainDevelopmentGameOverlapMaximum": 0,
            },
            "resources": {
                "baselineGlobalGpuHours": baseline_gpu_hours,
                "gpuHoursAuthorized": 0.0,
                "globalGpuHoursLimit": 120.0,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "maximumOutputBytes": F54_OUTPUT_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "authorizations": {
                "f54PilotDatasetGenerationAuthorized": True,
                "largeDatasetGenerationAuthorized": False,
                "cudaInferenceAuthorized": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F54 learner-occupancy policy pilot dataset generation",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "contractSha256",
    )


def prepare_phase4f53(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f53(workspace), "reused": True}
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        evidence = _f52_evidence(workspace)
        source = _source(workspace)
        resources = _resources(workspace)
        initialization = _initialization(workspace)
        ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
        if any(row.get("status") == "running" for row in ledger.runs.values()):
            raise Phase4F53Error("a running GPU ledger entry blocks F53 preparation")
        f51_selection = _mapping(workspace / F51_SELECTION, "F51 selection")
        f51_selection_sha = _verify_self_hash(
            f51_selection, "selectionSha256", "F51 selection"
        )
        rows = f51_selection.get("rows")
        if f51_selection_sha != EXPECTED_F51_SELECTION_SHA256 or not isinstance(rows, list):
            raise Phase4F53Error("F51 position selection changed")
        train_rows = select_train_rows(rows)
        development_rows = [
            row for row in rows if isinstance(row, Mapping) and row.get("split") == "development"
        ]
        audit = selection_audit(train_rows, development_rows)
        checks = selection_checks(audit)
        if not all(checks.values()) or len(development_rows) != sum(DEVELOPMENT_COUNTS.values()):
            raise Phase4F53Error("F53 train/development isolation gate failed")
        selection = _selection_payload(
            str(source["sha256"]), f51_selection_sha, train_rows, audit
        )
        contract = _f54_contract(
            source, evidence, selection, initialization, ledger.total_hours
        )
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f53-policy-head-pilot-preparation",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "trainingStarted": False,
                "optimizerSteps": 0,
                "datasetGenerationStarted": False,
                "source": source,
                "f52Evidence": evidence["hashes"],
                "trainSelectionPath": TRAIN_SELECTION.as_posix(),
                "trainSelectionSha256": selection["selectionSha256"],
                "selectionAudit": audit,
                "selectionChecks": checks,
                "initialization": initialization,
                "executionContractPath": F54_CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "resources": resources,
                "globalGpuHours": ledger.total_hours,
                "authorizations": contract["authorizations"],
                "nextStage": contract["nextStage"],
            },
            "reportSha256",
        )
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f53-policy-head-pilot-preparation-gate",
                "status": "completed",
                "preparationPassed": True,
                "outcomeBand": "learner-policy-pilot-dataset-generation-ready",
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "executionContractPath": F54_CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "trainSelectionSha256": selection["selectionSha256"],
                "f54PilotDatasetGenerationAuthorized": True,
                "largeDatasetGenerationAuthorized": False,
                "datasetGenerationStarted": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "nextStage": contract["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / TRAIN_SELECTION, selection, "F53 train selection")
        _write_immutable(workspace / F54_CONTRACT, contract, "F54 dataset contract")
        _write_immutable(workspace / REPORT, report, "F53 preparation report")
        _write_immutable(workspace / GATE, gate, "F53 preparation gate")
    return {**gate, "globalGpuHours": ledger.total_hours, "reused": False}


def verify_phase4f53(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    evidence = _f52_evidence(workspace)
    selection = _mapping(workspace / TRAIN_SELECTION, "F53 train selection")
    contract = _mapping(workspace / F54_CONTRACT, "F54 dataset contract")
    report = _mapping(workspace / REPORT, "F53 report")
    gate = _mapping(workspace / GATE, "F53 gate")
    selection_sha = _verify_self_hash(selection, "selectionSha256", "F53 train selection")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F54 dataset contract")
    report_sha = _verify_self_hash(report, "reportSha256", "F53 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F53 gate")
    rows = selection.get("rows")
    if not isinstance(rows, list):
        raise Phase4F53Error("F53 train selection rows are malformed")
    f51 = _mapping(workspace / F51_SELECTION, "F51 selection")
    f51_sha = _verify_self_hash(f51, "selectionSha256", "F51 selection")
    source_rows = f51.get("rows")
    development = (
        [row for row in source_rows if isinstance(row, Mapping) and row.get("split") == "development"]
        if isinstance(source_rows, list)
        else []
    )
    expected_rows = select_train_rows(source_rows) if isinstance(source_rows, list) else []
    audit = selection_audit(rows, development)
    checks = selection_checks(audit)
    if (
        f51_sha != EXPECTED_F51_SELECTION_SHA256
        or rows != expected_rows
        or report.get("source") != _source(workspace)
        or report.get("f52Evidence") != evidence["hashes"]
        or selection.get("sourceSha256") != _source(workspace)["sha256"]
        or selection.get("audit") != audit
        or not all(checks.values())
        or report.get("selectionChecks") != checks
        or report.get("trainSelectionSha256") != selection_sha
        or report.get("executionContractSha256") != contract_sha
        or gate.get("reportSha256") != report_sha
        or gate.get("executionContractSha256") != contract_sha
        or gate.get("trainSelectionSha256") != selection_sha
        or contract.get("inputs", {}).get("f52GateSha256") != EXPECTED_F52_GATE_SHA256
        or contract.get("inputs", {}).get("f53TrainSelectionSha256") != selection_sha
        or contract.get("f53SourceSha256") != _source(workspace)["sha256"]
        or contract.get("dataset", {}).get("trainPositions") != sum(TRAIN_COUNTS.values())
        or contract.get("dataset", {}).get("developmentPositions")
        != sum(DEVELOPMENT_COUNTS.values())
        or contract.get("dataset", {}).get("valueTarget") is not None
        or contract.get("authorizations", {}).get("f54PilotDatasetGenerationAuthorized") is not True
        or contract.get("authorizations", {}).get("cudaInferenceAuthorized") is not False
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
        or report.get("initialization") != _initialization(workspace)
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("datasetGenerationStarted") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
        or report.get("trainingStarted") is not False
        or report.get("datasetGenerationStarted") is not False
    ):
        raise Phase4F53Error("F53 immutable preparation evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f53-policy-head-pilot-preparation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "executionContractPath": F54_CONTRACT.as_posix(),
        "executionContractSha256": contract_sha,
        "trainSelectionSha256": selection_sha,
        "f54PilotDatasetGenerationAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        prepare_phase4f53(args.workspace_root)
        if args.command == "prepare"
        else verify_phase4f53(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
