"""Review F56 and freeze a hard-KL trust-region policy-head pilot.

F57 is CPU-only. It does not load training data into an optimizer, access CUDA,
or authorize formal training. Its only mutable output is review evidence and a
frozen F58 pilot contract.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f55_policy_head_pilot_preparation import ANCHOR_SET, F54_MANIFEST
from .phase4f56_policy_head_pilot import (
    EXPECTED_CONTRACT_SHA256 as EXPECTED_F56_CONTRACT_SHA256,
    GATE as F56_GATE,
    REPORT as F56_REPORT,
    verify_phase4f56,
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
EXPECTED_F56_GATE_SHA256 = "6ff9cea9a3ed2110f4cd544f051845979c2109e7f64cc689d40393ae0f55cea1"
EXPECTED_F56_REPORT_SHA256 = "94619667d53562d1ed3bab260f0799f819d8e4329401594c95013a9275d4c0a1"
EXPECTED_F56_CHECKPOINT_SHA256 = "10d6e86ceacb573ffa9626d4e8c312459276a9b25c8dddf4e91562eb61cf75af"
EXPECTED_F56_CHECKPOINT_STATE_SHA256 = "bafb7ec8d87a6a978e2c13e02a139639e6cd55fe11f0f558f52aa76dd2b33aa9"

ROOT = Path("training/runs/phase4/phase4f/f57-policy-head-pilot-review-v1")
REPORT = ROOT / "review-report.json"
GATE = ROOT / "review-gate.json"
F58_CONTRACT = Path("training/configs/phase4f58-kl-trust-region-policy-head-pilot-frozen-v1.json")
RUN_ID_PREFIX = "phase4f57-policy-head-pilot-review"

F57_WRITE_RESERVE_BYTES = 64 * 1024**2
F58_OUTPUT_RESERVE_BYTES = 384 * 1024**2
F58_GPU_HOURS = 1.0
GLOBAL_GPU_HOURS_LIMIT = 120.0


class Phase4F57Error(RuntimeError):
    """Raised when F56 cannot be reviewed or F58 cannot be frozen safely."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F57Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f57_policy_head_pilot_review.py",
            "training/quoridor/phase4f56_policy_head_pilot.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    reserve = F57_WRITE_RESERVE_BYTES + F58_OUTPUT_RESERVE_BYTES
    if artifact_bytes + reserve >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F57Error("16 GiB artifact cap cannot reserve F57 and F58 evidence")
    if free_bytes - reserve < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F57Error("F58 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "f57WriteReserveBytes": F57_WRITE_RESERVE_BYTES,
        "f58OutputReserveBytes": F58_OUTPUT_RESERVE_BYTES,
    }


def _ledger_snapshot(workspace: Path) -> dict[str, object]:
    relative = PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    ledger = GpuBudgetLedger(workspace / relative)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F57Error("a running GPU ledger entry blocks F57 freeze")
    if ledger.total_hours + F58_GPU_HOURS > GLOBAL_GPU_HOURS_LIMIT:
        raise Phase4F57Error("the global GPU budget cannot reserve F58")
    return {
        "path": relative.as_posix(),
        "fileSha256": sha256_file(workspace / relative),
        "totalGpuHours": ledger.total_hours,
        "maximumAfterF58": ledger.total_hours + F58_GPU_HOURS,
    }


def _f56_evidence(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    verification = verify_phase4f56(workspace)
    gate = _mapping(workspace / F56_GATE, "F56 gate")
    report = _mapping(workspace / F56_REPORT, "F56 report")
    checkpoint = report.get("selectedCheckpoint")
    if (
        verification.get("status") != "verified"
        or verification.get("gateSha256") != EXPECTED_F56_GATE_SHA256
        or verification.get("reportSha256") != EXPECTED_F56_REPORT_SHA256
        or _verify_self_hash(gate, "gateSha256", "F56 gate") != EXPECTED_F56_GATE_SHA256
        or _verify_self_hash(report, "reportSha256", "F56 report")
        != EXPECTED_F56_REPORT_SHA256
        or report.get("contractSha256") != EXPECTED_F56_CONTRACT_SHA256
        or report.get("pilotPassed") is not False
        or report.get("selectedEligible") is not False
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or gate.get("outcomeBand") != "isolated-policy-head-pilot-failed"
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("fileSha256") != EXPECTED_F56_CHECKPOINT_SHA256
        or checkpoint.get("modelStateSha256") != EXPECTED_F56_CHECKPOINT_STATE_SHA256
    ):
        raise Phase4F57Error("F56 is not the accepted failed pilot evidence")
    return gate, report


def _relative_improvement(baseline: float, current: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(current) or baseline <= 0.0:
        raise Phase4F57Error("F56 policy-loss metric is invalid")
    return (baseline - current) / baseline


def _review_content_from_report(report: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "reviewContentSha256",
        "f58ContractPath",
        "f58ContractSha256",
        "f58CudaPilotAuthorized",
        "nextStage",
        "reportSha256",
    }
    return _self_hashed(
        {key: value for key, value in report.items() if key not in excluded},
        "reviewContentSha256",
    )


def review_evaluations(
    report: Mapping[str, object], anchor_limit: float
) -> dict[str, object]:
    evaluations = report.get("evaluations")
    expected_steps = [0, 8, 16, 24, 32]
    if (
        not isinstance(evaluations, list)
        or [entry.get("step") for entry in evaluations] != expected_steps
    ):
        raise Phase4F57Error("F56 evaluation schedule changed")
    baseline_metrics = evaluations[0].get("metrics")
    if not isinstance(baseline_metrics, Mapping):
        raise Phase4F57Error("F56 baseline metrics are malformed")
    baseline_losses = baseline_metrics.get("policyLoss")
    if not isinstance(baseline_losses, Mapping):
        raise Phase4F57Error("F56 baseline policy losses are malformed")

    rows: list[dict[str, object]] = []
    policy_sequences = {band: [] for band in ("all", "early", "middle", "late")}
    kl_sequence: list[float] = []
    non_anchor_checks_pass = True
    exact_invariants = True
    for index, entry in enumerate(evaluations):
        metrics = entry.get("metrics")
        checks = entry.get("checks")
        if not isinstance(metrics, Mapping) or not isinstance(checks, Mapping):
            raise Phase4F57Error("F56 evaluation entry is malformed")
        losses = metrics.get("policyLoss")
        frozen = metrics.get("frozenTensors")
        if not isinstance(losses, Mapping) or not isinstance(frozen, Mapping):
            raise Phase4F57Error("F56 evaluation metrics are malformed")
        values = {band: float(losses[band]) for band in policy_sequences}
        kl = float(metrics["anchorForwardKl"])
        if not all(math.isfinite(value) for value in (*values.values(), kl)):
            raise Phase4F57Error("F56 evaluation contains a non-finite metric")
        for band, value in values.items():
            policy_sequences[band].append(value)
        kl_sequence.append(kl)
        if index > 0:
            non_anchor_checks_pass &= all(
                value is True for name, value in checks.items() if name != "anchorForwardKl"
            )
        exact_invariants &= (
            frozen.get("passed") is True
            and float(frozen.get("maximumAbsoluteError", math.inf)) == 0.0
            and not frozen.get("changedTensorNames")
            and float(metrics.get("valueOutputMaximumAbsoluteError", math.inf)) == 0.0
        )
        rows.append(
            {
                "step": entry["step"],
                "eligible": entry.get("eligible"),
                "anchorForwardKl": kl,
                "anchorLimitRatio": kl / anchor_limit,
                "policyLoss": values,
                "relativePolicyImprovement": {
                    band: _relative_improvement(float(baseline_losses[band]), value)
                    for band, value in values.items()
                },
                "failedChecks": [name for name, value in checks.items() if value is not True],
            }
        )

    post = rows[1:]
    checks = {
        "pilotFailedOnlyOnAnchorConstraint": non_anchor_checks_pass
        and all(row["failedChecks"] == ["anchorForwardKl"] for row in post),
        "noObservedEligibleUpdatedCheckpoint": all(row["eligible"] is False for row in post),
        "firstObservedUpdateAlreadyAboveAnchorLimit": float(post[0]["anchorForwardKl"])
        > anchor_limit,
        "policyLossStrictlyImprovesAtEveryObservation": all(
            all(before > after for before, after in zip(values, values[1:]))
            for values in policy_sequences.values()
        ),
        "anchorKlStrictlyIncreasesAtEveryObservation": all(
            before < after for before, after in zip(kl_sequence, kl_sequence[1:])
        ),
        "frozenAndValueInvariantsExact": exact_invariants,
        "trainingWasPolicyOnly": report.get("training", {}).get("selfPlayGames") == 0
        and report.get("training", {}).get("replayStoreUsage") is False
        and report.get("training", {}).get("valueTarget") is None,
    }
    final = rows[-1]
    return {
        "anchorForwardKlMaximum": anchor_limit,
        "observations": rows,
        "finalRelativePolicyImprovement": final["relativePolicyImprovement"],
        "firstObservedAnchorLimitMultiple": post[0]["anchorLimitRatio"],
        "finalAnchorLimitMultiple": final["anchorLimitRatio"],
        "diagnosis": {
            "primaryCause": "soft-anchor-penalty-did-not-enforce-the-preservation-boundary",
            "transferSignal": "positive-in-all-ply-bands",
            "observabilityGap": "no-evaluation-between-optimizer-steps-0-and-8",
            "selectedF56CheckpointUse": "diagnostic-only-not-promotion-eligible",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _f58_contract(
    *,
    review_content_sha: str,
    source_sha: str,
    f56_report: Mapping[str, object],
    ledger: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f58-kl-trust-region-policy-head-pilot-contract",
            "status": "frozen",
            "objective": "test learner-occupancy policy transfer while hard-limiting F21 anchor drift",
            "f57ReviewContentSha256": review_content_sha,
            "f57SourceSha256": source_sha,
            "inputs": {
                "datasetManifestPath": F54_MANIFEST.as_posix(),
                "datasetManifestSha256": "a87a7493e2434b944bb3adda944da32ef0e4f53f91dc8ff36a427a8b3f3e26da",
                "datasetFingerprint": f56_report["datasetFingerprint"],
                "trainPositions": 1024,
                "developmentPositions": 512,
                "anchorSetPath": ANCHOR_SET.as_posix(),
                "anchorSetSha256": "78e8cc8cd027dc2dfa591e781c501a0b7dd4d830cafd1c16237735e5ccab873b",
                "anchorPositions": 512,
            },
            "initialization": {
                "mode": "candidate-model",
                "modelPath": "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/checkpoints/candidate-model-checkpoint-g00000001-s000000000192.pt",
                "modelFileSha256": "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01",
                "modelStateSha256": "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527",
                "networkConfigPath": "training/configs/phase4f21-stage-a-block-01-v1.json",
                "networkConfigFileSha256": "bdaa011c52a3a3f0925dcf603d52bd19efe4b7cee6039d6e03140b14d007a14e",
                "resetOptimizer": True,
                "emptyReplay": True,
                "f56DiagnosticCheckpointInitializationAllowed": False,
            },
            "network": {
                "inputChannels": 16,
                "channels": 192,
                "residualBlocks": 16,
                "seReduction": 8,
                "policyChannels": 32,
                "valueChannels": 32,
                "valueHidden": 256,
            },
            "training": {
                "device": "cuda",
                "modelMode": "eval",
                "selfPlayGames": 0,
                "replayStoreUsage": False,
                "valueTarget": None,
                "optimizer": "AdamW",
                "batchSize": 256,
                "anchorPenaltyBatchSize": 256,
                "optimizerSteps": 64,
                "evaluationSteps": [0, 1, 2, 4, 8, 16, 24, 32, 48, 64],
                "learningRate": 1e-5,
                "minimumLearningRate": 2e-6,
                "schedule": "cosine-annealing",
                "weightDecay": 1e-4,
                "maximumGradientNorm": 5.0,
                "cudaAmp": True,
                "seed": 20260914,
                "batchMix": {"early": 0.5, "middle": 0.375, "late": 0.125},
                "mirrorProbability": 0.5,
                "loss": {
                    "teacherPolicy": "legal-masked-soft-cross-entropy",
                    "teacherPolicyWeight": 1.0,
                    "f21PolicyPreservation": "legal-masked-forward-kl",
                    "f21PolicyPreservationWeight": 2.0,
                    "value": None,
                },
                "trainableParameters": ["policy_head"],
                "frozenParameters": [
                    "input_conv",
                    "input_bn",
                    "trunk",
                    "value_features",
                    "value_head",
                ],
                "frozenBuffers": "all-model-buffers-v1",
                "checkpointSelection": "minimum-development-policy-loss-among-eligible-evaluations",
            },
            "trustRegion": {
                "metric": "full-anchor-identity-and-mirror-mean-forward-kl",
                "reference": "unmodified-F21-policy",
                "checkedAfterEveryProposedOptimizerStep": True,
                "hardMaximum": 0.02,
                "projectionTargetMaximum": 0.018,
                "projection": "pre-to-post-policy-head-parameter-segment-bisection-v1",
                "bisectionIterations": 12,
                "minimumAcceptedScale": 1.0 / 4096.0,
                "rejectUpdateWhenNoPositiveScaleIsFeasible": True,
                "maximumConsecutiveRejectedUpdates": 8,
                "optimizerMomentsFollowUnprojectedGradient": True,
                "violationIsFatal": True,
            },
            "evaluation": {
                "split": "development",
                "usesMirrorAverage": True,
                "compareAgainstUnmodifiedInitialization": True,
                "finalSplitUsage": False,
                "acceptance": {
                    "finiteTrainingAndEvaluation": True,
                    "overallPolicyLossRelativeImprovementMinimum": 0.005,
                    "earlyPolicyLossRelativeImprovementMinimum": 0.0,
                    "middlePolicyLossRelativeRegressionMaximum": 0.01,
                    "latePolicyLossRelativeRegressionMaximum": 0.02,
                    "anchorForwardKlMaximum": 0.02,
                    "frozenTensorMaximumAbsoluteError": 0.0,
                    "valueOutputMaximumAbsoluteError": 0.0,
                    "policyHeadMustChange": True,
                    "acceptedOptimizerUpdatesMinimum": 1,
                },
            },
            "preflight": {
                "currentSourceCudaAmpMemorySmokeRequired": True,
                "datasetDeepVerificationRequired": True,
                "policyHeadIsolationSmokeRequired": True,
                "trustRegionProjectionCpuUnitTestsRequired": True,
            },
            "resources": {
                "baselineGlobalGpuHours": ledger["totalGpuHours"],
                "gpuHoursLimit": F58_GPU_HOURS,
                "maximumGlobalGpuHours": ledger["maximumAfterF58"],
                "globalGpuHoursLimit": GLOBAL_GPU_HOURS_LIMIT,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "maximumOutputBytes": F58_OUTPUT_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "authorizations": {
                "cudaPilotTrainingAuthorized": True,
                "selfPlayAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F58 run KL trust-region policy-head pilot",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "contractSha256",
    )


def run_phase4f57(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f57(workspace), "reused": True}
    _gate, f56_report = _f56_evidence(workspace)
    review = review_evaluations(f56_report, anchor_limit=0.02)
    if not review["passed"]:
        raise Phase4F57Error("F56 evidence does not support the trust-region diagnosis")
    source = _source(workspace)
    resources = _resources(workspace)
    ledger = _ledger_snapshot(workspace)
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        report_core = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f57-isolated-policy-head-pilot-review",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingStarted": False,
            "optimizerSteps": 0,
            "f56GateSha256": EXPECTED_F56_GATE_SHA256,
            "f56ReportSha256": EXPECTED_F56_REPORT_SHA256,
            "f56ContractSha256": EXPECTED_F56_CONTRACT_SHA256,
            "f56SelectedCheckpointSha256": EXPECTED_F56_CHECKPOINT_SHA256,
            "review": review,
            "reviewPassed": True,
            "source": source,
            "gpuLedger": ledger,
            "resources": resources,
            "globalGpuHours": ledger["totalGpuHours"],
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
        }
        review_content = _self_hashed(report_core, "reviewContentSha256")
        contract = _f58_contract(
            review_content_sha=review_content["reviewContentSha256"],
            source_sha=str(source["sha256"]),
            f56_report=f56_report,
            ledger=ledger,
        )
        report = _self_hashed(
            {
                **report_core,
                "reviewContentSha256": review_content["reviewContentSha256"],
                "f58ContractPath": F58_CONTRACT.as_posix(),
                "f58ContractSha256": contract["contractSha256"],
                "f58CudaPilotAuthorized": True,
                "nextStage": contract["nextStage"],
            },
            "reportSha256",
        )
        _write_immutable(workspace / F58_CONTRACT, contract, "F58 pilot contract")
        _write_immutable(workspace / REPORT, report, "F57 review report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f57-policy-head-pilot-review-gate",
                "status": "completed",
                "outcomeBand": "kl-trust-region-pilot-ready",
                "reviewPassed": True,
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f58ContractPath": F58_CONTRACT.as_posix(),
                "f58ContractSha256": contract["contractSha256"],
                "f58CudaPilotAuthorized": True,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "globalGpuHours": ledger["totalGpuHours"],
                "nextStage": contract["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F57 review gate")
    return {**gate, "reused": False}


def verify_phase4f57(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    _gate, f56_report = _f56_evidence(workspace)
    expected_review = review_evaluations(f56_report, anchor_limit=0.02)
    report = _mapping(workspace / REPORT, "F57 report")
    report_sha = _verify_self_hash(report, "reportSha256", "F57 report")
    gate = _mapping(workspace / GATE, "F57 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F57 gate")
    contract = _mapping(workspace / F58_CONTRACT, "F58 contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F58 contract")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    budget = contract.get("resources")
    content = _review_content_from_report(report)
    expected_contract = None
    if isinstance(budget, Mapping):
        expected_contract = _f58_contract(
            review_content_sha=str(content["reviewContentSha256"]),
            source_sha=str(report.get("source", {}).get("sha256")),
            f56_report=f56_report,
            ledger={
                "totalGpuHours": float(budget["baselineGlobalGpuHours"]),
                "maximumAfterF58": float(budget["maximumGlobalGpuHours"]),
            },
        )
    if (
        report.get("source") != _source(workspace)
        or report.get("review") != expected_review
        or report.get("reviewPassed") is not True
        or report.get("cpuOnly") is not True
        or report.get("cudaAccessed") is not False
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or report.get("f58ContractSha256") != contract_sha
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or report.get("reviewContentSha256") != content["reviewContentSha256"]
        or gate.get("reportSha256") != report_sha
        or gate.get("f58ContractSha256") != contract_sha
        or gate.get("f58CudaPilotAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
        or contract.get("status") != "frozen"
        or contract.get("f57ReviewContentSha256") != report.get("reviewContentSha256")
        or contract.get("authorizations", {}).get("cudaPilotTrainingAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
        or not isinstance(budget, Mapping)
        or contract != expected_contract
        or ledger.total_hours < float(budget["baselineGlobalGpuHours"])
        or ledger.total_hours > float(budget["maximumGlobalGpuHours"])
    ):
        raise Phase4F57Error("F57/F58 frozen provenance changed")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f57-policy-head-pilot-review-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "f58ContractPath": F58_CONTRACT.as_posix(),
        "f58ContractSha256": contract_sha,
        "reviewPassed": True,
        "f58CudaPilotAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f57(args.workspace_root)
        if args.command == "run"
        else verify_phase4f57(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
