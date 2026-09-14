"""Freeze the learner-occupancy and actual classic-root target probe."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .config import load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f45_curriculum_pilot_review import (
    EXPECTED_F21_MODEL_SHA256,
    EXPECTED_F21_STATE_SHA256,
    F21_CONFIG,
    F21_MODEL,
)
from .phase4f49_teacher_search_audit import (
    CLASSIC_RESULT as F49_CLASSIC_RESULT,
    GATE as F49_GATE,
    MCTS_RESULT as F49_MCTS_RESULT,
    REPORT as F49_REPORT,
    verify_phase4f49,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_F49_GATE_SHA256 = "0c1ef7cc8c0e511e3bb1e7751cd014021fa2d12f61d2d4aa1b2fde692134c5d9"
EXPECTED_F49_REPORT_SHA256 = "b779ea967b605e9ed75d03e998a696f7dd90e625556f682176d8bda1b5ffc9b8"
EXPECTED_F49_CLASSIC_SHA256 = "c67b88354dc1b2bc6ca56bcc29046613ce3800dc7851c720dfa5a9f266b067a0"
EXPECTED_F49_MCTS_SHA256 = "03f38ccf5f0fa647fe845043889edc3b1c22bc5549987036f0b7d70aba67d3b4"

ROOT = Path("training/runs/phase4/phase4f/f50-learner-teacher-root-preparation-v1")
REPORT = ROOT / "preparation-report.json"
GATE = ROOT / "preparation-gate.json"
F51_CONTRACT = Path(
    "training/configs/phase4f51-learner-occupancy-teacher-root-probe-frozen-v1.json"
)
RUN_ID = "phase4f50-learner-teacher-root-preparation-v1"
WRITE_RESERVE_BYTES = 32 * 1024**2
F51_OUTPUT_LIMIT_BYTES = 768 * 1024**2
F51_GPU_HOURS_LIMIT = 4.0
GLOBAL_GPU_HOURS_LIMIT = 120.0


class Phase4F50Error(RuntimeError):
    """Raised when F49 cannot safely authorize the F51 probe."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F50Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=("training/quoridor/phase4f50_learner_teacher_preparation.py",),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F50Error("16 GiB artifact cap cannot reserve F50 evidence")
    if artifact_bytes + F51_OUTPUT_LIMIT_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F50Error("16 GiB artifact cap cannot reserve the frozen F51 probe")
    if free_bytes - F51_OUTPUT_LIMIT_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F50Error("F51 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "f50WriteReserveBytes": WRITE_RESERVE_BYTES,
        "f51OutputLimitBytes": F51_OUTPUT_LIMIT_BYTES,
    }


def _f49_evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f49(workspace)
    report = _mapping(workspace / F49_REPORT, "F49 report")
    gate = _mapping(workspace / F49_GATE, "F49 gate")
    classic = _mapping(workspace / F49_CLASSIC_RESULT, "F49 classic result")
    mcts = _mapping(workspace / F49_MCTS_RESULT, "F49 MCTS result")
    hashes = {
        "gate": _verify_self_hash(gate, "gateSha256", "F49 gate"),
        "report": _verify_self_hash(report, "reportSha256", "F49 report"),
        "classic": _verify_self_hash(classic, "resultSha256", "F49 classic result"),
        "mcts": _verify_self_hash(mcts, "resultSha256", "F49 MCTS result"),
    }
    if (
        verification.get("status") != "verified"
        or hashes["gate"] != EXPECTED_F49_GATE_SHA256
        or hashes["report"] != EXPECTED_F49_REPORT_SHA256
        or hashes["classic"] != EXPECTED_F49_CLASSIC_SHA256
        or hashes["mcts"] != EXPECTED_F49_MCTS_SHA256
        or gate.get("auditPassed") is not True
        or gate.get("outcomeBand") != "search-budget-coupling"
        or gate.get("trainingStarted") is not False
        or gate.get("datasetGenerationStarted") is not False
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
    ):
        raise Phase4F50Error("F49 is not the accepted read-only audit evidence")
    return {"report": report, "gate": gate, "classic": classic, "mcts": mcts}


def preparation_checks(evidence: Mapping[str, object]) -> dict[str, bool]:
    report = evidence["report"]
    classic = evidence["classic"]["summary"]
    mcts = evidence["mcts"]["summary"]
    diagnosis = report["diagnosis"]
    return {
        "f49AuditPassed": evidence["gate"]["auditPassed"] is True,
        "actualClassicRootAvailable": classic["actualNegamaxRootScoresAvailable"] is True,
        "classicBudgetSensitivityObserved": float(classic["budgetActionAgreement"]) < 0.90,
        "f44SearchBudgetSensitivityObserved": (
            float(mcts["withinModelConvergence"]["F44"]["800"]["actionAgreementWith3200"])
            < 0.80
        ),
        "valueIsNotPrimaryRegressionSignal": (
            float(diagnosis["measurements"]["f44MinusF21OutcomeMseAt3200"]) < 0.0
        ),
        "screenTrajectoriesExcluded": report["screenTrajectoriesTrainingUseAllowed"] is False,
        "noTrainingOccurred": report["trainingStarted"] is False,
        "noDatasetGenerationOccurred": report["datasetGenerationStarted"] is False,
    }


def _f51_contract(
    workspace: Path,
    evidence: Mapping[str, object],
    source: Mapping[str, object],
    baseline_gpu_hours: float,
) -> dict[str, object]:
    config = load_experiment_config(workspace / F21_CONFIG)
    report = evidence["report"]
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f51-learner-occupancy-teacher-root-probe-contract",
            "status": "frozen",
            "objective": (
                "validate isolated learner-occupancy states and stable actual classic-root "
                "policy targets before any large dataset or policy training"
            ),
            "f50SourceSha256": source["sha256"],
            "inputs": {
                "f49GatePath": F49_GATE.as_posix(),
                "f49GateSha256": EXPECTED_F49_GATE_SHA256,
                "f49ReportPath": F49_REPORT.as_posix(),
                "f49ReportSha256": EXPECTED_F49_REPORT_SHA256,
                "f49ClassicResultSha256": EXPECTED_F49_CLASSIC_SHA256,
                "f49MctsResultSha256": EXPECTED_F49_MCTS_SHA256,
                "f21ModelPath": F21_MODEL.as_posix(),
                "f21ModelFileSha256": EXPECTED_F21_MODEL_SHA256,
                "f21ModelStateSha256": EXPECTED_F21_STATE_SHA256,
                "f21ConfigPath": F21_CONFIG.as_posix(),
                "f21ConfigFileSha256": sha256_file(workspace / F21_CONFIG),
                "f21SemanticConfigSha256": fingerprint_config(config).sha256,
                "nativeBackendSha256": loaded_native_backend_hash(),
                "classicImplementation": {
                    "engine": "classic-hard-negamax",
                    "aiSourcePath": "src/ai.js",
                    "aiSourceSha256": sha256_file(workspace / "src/ai.js"),
                    "bridgePath": "scripts/classic-ai-bridge.mjs",
                    "bridgeSha256": sha256_file(workspace / "scripts/classic-ai-bridge.mjs"),
                    "pythonAdapterPath": "training/quoridor/classic_bridge.py",
                    "pythonAdapterSha256": sha256_file(workspace / "training/quoridor/classic_bridge.py"),
                },
            },
            "rollout": {
                "source": "F21-learner-occupancy-self-play",
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "games": {"train": 256, "development": 64},
                "splitUnit": "whole-game-before-position-extraction",
                "seeds": {"train": 2026090201, "development": 2026090202},
                "mcts": {
                    "simulations": 800,
                    "cBase": config.mcts.c_base,
                    "cInit": config.mcts.c_init,
                    "rootNoise": True,
                    "temperatureSchedule": "f21-frozen-self-play-schedule",
                },
                "maximumPlies": 128,
                "positionOrbitDeduplication": "horizontal-mirror-orbit-v1",
                "forbiddenSources": [
                    "F46-screen-trajectories",
                    "all-historical-screen-and-confirmation-trajectories",
                    "F42-corrected-curriculum-dataset",
                ],
                "requiredExclusionAudit": "content-hashed-orbit-inventory-with-zero-overlap",
            },
            "probeDataset": {
                "format": "policy-only-actual-classic-root-v1",
                "positions": {"train": 4096, "development": 512, "final": 0},
                "absolutePlyBands": {
                    "definition": "absolute-state-move-number-v1",
                    "boundaries": {"earlyMaximumExclusive": 16, "lateMinimumInclusive": 40},
                    "train": {"early": 2048, "middle": 1536, "late": 512},
                    "development": {"early": 256, "middle": 192, "late": 64},
                },
                "uniqueOrbitRequired": True,
                "gameDisjointSplitsRequired": True,
                "screenOrbitOverlapMaximum": 0,
                "valueTarget": None,
            },
            "classicTeacher": {
                "engine": "classic-hard-negamax",
                "primaryBudget": {"kind": "nodes", "limit": 4096, "maxDepth": 5},
                "stabilityAudit": {
                    "positions": 128,
                    "split": "development",
                    "selection": "stratified-by-absolute-ply-band-and-side-v1",
                    "comparisonBudget": {"kind": "nodes", "limit": 16384, "maxDepth": 5},
                    "minimumActionAgreement": 0.90,
                    "minimumTop12Jaccard": 0.90,
                },
                "rootOutput": "legal-action-root-scores-v1",
                "scoreSemantics": "actual-alpha-beta-return-or-bound-v1",
                "topK": 12,
                "forbiddenTargetSources": [
                    "one-ply-static-heuristic",
                    "student-mcts-policy",
                    "terminal-outcome-value-mixture",
                ],
            },
            "policyTargetCalibration": {
                "algorithm": "tie-preserving-score-rank-softmax-v1",
                "rankTemperatureCandidates": [1.0, 1.5, 2.0, 3.0],
                "calibrationSplit": "development-only",
                "searchedTopKMass": 0.97,
                "maximumExcludedLegalActionMass": 0.03,
                "mirrorRequired": True,
                "acceptance": {
                    "meanEntropyMinimum": 0.80,
                    "meanEntropyMaximum": 1.80,
                    "top1ProbabilityP95Maximum": 0.75,
                    "classicMaximumScoreGroupMassMinimum": 0.35,
                    "classicTop5MassMinimum": 0.80,
                    "policyMassMaximumAbsoluteError": 1e-9,
                    "mirrorMaximumAbsoluteError": 1e-12,
                },
                "selection": (
                    "lowest-temperature-candidate-passing-all-development-gates; "
                    "no screen-derived hyperparameter selection"
                ),
            },
            "scaleGate": {
                "pilotMustPassBeforeExpansion": True,
                "minimumUniqueTrainOrbits": 50000,
                "targetUniqueTrainOrbits": 100000,
                "largeDatasetGenerationAuthorized": False,
                "policyTrainingAuthorized": False,
            },
            "resources": {
                "baselineGlobalGpuHours": baseline_gpu_hours,
                "probeGpuHoursLimit": F51_GPU_HOURS_LIMIT,
                "maximumGlobalGpuHours": baseline_gpu_hours + F51_GPU_HOURS_LIMIT,
                "globalGpuHoursLimit": GLOBAL_GPU_HOURS_LIMIT,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "maximumOutputBytes": F51_OUTPUT_LIMIT_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "authorizations": {
                "learnerOccupancyProbeGenerationAuthorized": True,
                "diagnosticCudaInferenceAuthorized": True,
                "classicRootAnalysisAuthorized": True,
                "largeDatasetGenerationAuthorized": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F51 learner-occupancy teacher-root target probe",
                "requiresSeparateUserConfirmation": True,
            },
            "f49Diagnosis": report["diagnosis"],
        },
        "contractSha256",
    )


def prepare_phase4f50(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f50(workspace), "reused": True}
    evidence = _f49_evidence(workspace)
    checks = preparation_checks(evidence)
    if not all(checks.values()):
        raise Phase4F50Error("F49 findings do not support the constrained F51 probe")
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline_gpu_hours = ledger.total_hours
    if baseline_gpu_hours + F51_GPU_HOURS_LIMIT > GLOBAL_GPU_HOURS_LIMIT:
        raise Phase4F50Error("global 120 GPU-hour budget cannot fit F51")
    source = _source(workspace)
    contract = _f51_contract(workspace, evidence, source, baseline_gpu_hours)
    classic_summary = evidence["classic"]["summary"]
    mcts_summary = evidence["mcts"]["summary"]
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f50-learner-occupancy-teacher-root-preparation",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingStarted": False,
            "datasetGenerationStarted": False,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "source": source,
            "inputs": {
                "f49GateSha256": EXPECTED_F49_GATE_SHA256,
                "f49ReportSha256": EXPECTED_F49_REPORT_SHA256,
                "f49ClassicResultSha256": EXPECTED_F49_CLASSIC_SHA256,
                "f49MctsResultSha256": EXPECTED_F49_MCTS_SHA256,
            },
            "checks": checks,
            "observations": {
                "classic587Vs4096ActionAgreement": classic_summary["budgetActionAgreement"],
                "classic587Vs4096Top12Jaccard": classic_summary["meanTop12Jaccard"],
                "f21ActionStability800To3200": mcts_summary["withinModelConvergence"]["F21"]["800"]["actionAgreementWith3200"],
                "f44ActionStability800To3200": mcts_summary["withinModelConvergence"]["F44"]["800"]["actionAgreementWith3200"],
                "f21F44ActionAgreementAt3200": mcts_summary["bySimulation"]["3200"]["actionAgreement"],
                "f44MinusF21OutcomeMseAt3200": evidence["report"]["diagnosis"]["measurements"]["f44MinusF21OutcomeMseAt3200"],
            },
            "decisions": {
                "rolloutSource": "F21 learner occupancy with split-specific seeds",
                "teacherSource": "actual classic-hard root search only",
                "primaryTeacherNodes": 4096,
                "stabilityComparisonNodes": 16384,
                "rawAlphaBetaScoreSoftmaxAllowed": False,
                "policyTarget": "tie-preserving rank-softened top-12 plus 0.03 legal smoothing",
                "valueTargetGenerated": False,
                "screenDataUsedForTrainingOrCalibration": False,
                "probeBeforeLargeScale": True,
            },
            "executionContractPath": F51_CONTRACT.as_posix(),
            "executionContractSha256": contract["contractSha256"],
            "resources": resources,
            "gpuLedger": {
                "baselineGlobalGpuHours": baseline_gpu_hours,
                "f51LimitHours": F51_GPU_HOURS_LIMIT,
                "maximumGlobalGpuHours": baseline_gpu_hours + F51_GPU_HOURS_LIMIT,
            },
            "nextStage": contract["nextStage"],
        },
        "reportSha256",
    )
    gate = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f50-learner-teacher-root-preparation-gate",
            "status": "completed",
            "preparationPassed": True,
            "f51ProbeAuthorized": True,
            "largeDatasetGenerationAuthorized": False,
            "cudaTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "reportPath": REPORT.as_posix(),
            "reportSha256": report["reportSha256"],
            "executionContractPath": F51_CONTRACT.as_posix(),
            "executionContractSha256": contract["contractSha256"],
            "nextStage": contract["nextStage"],
        },
        "gateSha256",
    )
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=RUN_ID):
        _write_immutable(workspace / F51_CONTRACT, contract, "F51 frozen contract")
        _write_immutable(workspace / REPORT, report, "F50 preparation report")
        _write_immutable(workspace / GATE, gate, "F50 preparation gate")
    return {**gate, "reused": False}


def verify_phase4f50(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    evidence = _f49_evidence(workspace)
    report = _mapping(workspace / REPORT, "F50 report")
    gate = _mapping(workspace / GATE, "F50 gate")
    contract = _mapping(workspace / F51_CONTRACT, "F51 contract")
    report_sha = _verify_self_hash(report, "reportSha256", "F50 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F50 gate")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F51 contract")
    if (
        report.get("source") != _source(workspace)
        or report.get("checks") != preparation_checks(evidence)
        or not all(report["checks"].values())
        or report.get("executionContractSha256") != contract_sha
        or gate.get("reportSha256") != report_sha
        or gate.get("executionContractSha256") != contract_sha
        or gate.get("f51ProbeAuthorized") is not True
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or contract.get("authorizations", {}).get("learnerOccupancyProbeGenerationAuthorized") is not True
        or contract.get("authorizations", {}).get("largeDatasetGenerationAuthorized") is not False
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
    ):
        raise Phase4F50Error("F50/F51 immutable evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f50-learner-teacher-root-preparation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "executionContractPath": F51_CONTRACT.as_posix(),
        "executionContractSha256": contract_sha,
        "f51ProbeAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        prepare_phase4f50(args.workspace_root)
        if args.command == "prepare"
        else verify_phase4f50(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
