"""Review F21 value scaling and freeze the external-transfer experiment design.

F63 is deliberately CPU-only.  It reviews immutable F22/F61/F62 evidence and
freezes the design boundary for F64; it does not run an arena, load CUDA, train
a model, or authorize value-head optimization.
"""

from __future__ import annotations

import argparse
import shutil
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
from .phase4f61_value_scale_screen import (
    EXPECTED_F21_CONFIG_SHA256,
    EXPECTED_F21_MODEL_SHA256,
    EXPECTED_F21_STATE_SHA256,
    F21_CONFIG,
    F21_MODEL,
    _load_f21_state,
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
ROOT = Path("training/runs/phase4/phase4f/f63-value-calibration-review-v1")
REPORT = ROOT / "review-report.json"
GATE = ROOT / "review-gate.json"
F64_DESIGN = Path(
    "training/configs/phase4f64-scaled-f21-classic-confirmation-design-frozen-v1.json"
)

F22_GATE = Path("training/runs/phase4/phase4f/stage-a-screen-v2/screen-gate.json")
F22_RESULT = Path(
    "training/runs/phase4/phase4f/stage-a-screen-v2/screen-arena/result.json"
)
F61_GATE = Path(
    "training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-gate.json"
)
F61_RESULT = Path(
    "training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-arena/result.json"
)
F62_GATE = Path(
    "training/runs/phase4/phase4f/f62-f21-value-scale-confirmation-v1/"
    "confirmation-gate.json"
)
F62_RESULT = Path(
    "training/runs/phase4/phase4f/f62-f21-value-scale-confirmation-v1/"
    "confirmation-arena/result.json"
)

EXPECTED_F22_GATE_SHA256 = (
    "850b023a86d752d939a34ef0fb373e718d261f05ec0303883bd0852db3fad14b"
)
EXPECTED_F22_RESULT_SHA256 = (
    "4cf946cd840e094102ee56f096581e7ef5c315da62e542bde365bd8d1a4710f5"
)
EXPECTED_F61_GATE_SHA256 = (
    "35ab6ddba6a387ecca274b6592f39c5427b8957000737172345de429abf9891d"
)
EXPECTED_F61_RESULT_SHA256 = (
    "d59263c33e52e1b8d54f4f1f5c171ad56cb6dd2740a626ac2b243ac17faeb8fb"
)
EXPECTED_F62_GATE_SHA256 = (
    "eb773d1a5311ba0af9afa496934c24885e1f5cbed7c08061d615fabc3544fd66"
)
EXPECTED_F62_RESULT_SHA256 = (
    "60502a924dd43a5d6e7fe3cdc9097d9ae3f8cea5be52fbe681f72f4fc9fc50b9"
)

ROOT_CAUSE_REPORT = Path("docs/PHASE4_ROOT_CAUSE_ABLATION.md")
F60_POSTMORTEM = Path("docs/PHASE4_F60_POSTMORTEM.md")
EXPECTED_ROOT_CAUSE_SHA256 = (
    "65ba86701048d5ef23c45dde2832e810975821dfbfdcf33b778b61005c0d432b"
)
EXPECTED_F60_POSTMORTEM_SHA256 = (
    "c0c28468eb08ad3b7e3fe937b733b1eee786eb02fa12491b918b6ddc69b13bca"
)

F63_WRITE_RESERVE_BYTES = 32 * 1024**2
F64_ARENA_WRITE_RESERVE_BYTES = 128 * 1024**2
F64_GPU_HOURS = 8.0
GLOBAL_GPU_HOURS_LIMIT = 120.0
ARTIFACT_LIMIT_BYTES = EXTENDED_ARTIFACT_LIMIT_BYTES


class Phase4F63Error(RuntimeError):
    """Raised when the value-calibration review cannot be reproduced."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F63Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "docs/PHASE4_ROOT_CAUSE_ABLATION.md",
            "docs/PHASE4_F60_POSTMORTEM.md",
            "training/quoridor/network.py",
            "training/quoridor/trainer.py",
            "training/quoridor/phase4f61_value_scale_screen.py",
            "training/quoridor/phase4f62_value_scale_confirmation.py",
            "training/quoridor/phase4f63_value_calibration_review.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    reserve = F63_WRITE_RESERVE_BYTES + F64_ARENA_WRITE_RESERVE_BYTES
    if artifact_bytes + reserve >= ARTIFACT_LIMIT_BYTES:
        raise Phase4F63Error("16 GiB artifact cap cannot reserve F63 and F64 evidence")
    if free_bytes - reserve < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F63Error("F64 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "f63WriteReserveBytes": F63_WRITE_RESERVE_BYTES,
        "f64ArenaWriteReserveBytes": F64_ARENA_WRITE_RESERVE_BYTES,
    }


def _ledger(workspace: Path) -> tuple[GpuBudgetLedger, dict[str, object]]:
    relative = PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    ledger = GpuBudgetLedger(workspace / relative)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F63Error("a running GPU ledger entry blocks F63 review")
    if ledger.total_hours + F64_GPU_HOURS > GLOBAL_GPU_HOURS_LIMIT:
        raise Phase4F63Error("global GPU budget cannot reserve the F64 confirmation")
    return ledger, {
        "path": relative.as_posix(),
        "fileSha256": sha256_file(workspace / relative),
        "totalGpuHours": ledger.total_hours,
        "globalLimitHours": GLOBAL_GPU_HOURS_LIMIT,
        "f64LimitHours": F64_GPU_HOURS,
        "maximumAfterF64": ledger.total_hours + F64_GPU_HOURS,
    }


def _arena_evidence(
    workspace: Path,
    *,
    label: str,
    gate_path: Path,
    result_path: Path,
    gate_sha256: str,
    result_sha256: str,
    outcome_band: str,
) -> dict[str, object]:
    gate = _mapping(workspace / gate_path, f"{label} gate")
    stored_gate_sha = _verify_self_hash(gate, "gateSha256", f"{label} gate")
    result = _mapping(workspace / result_path, f"{label} result")
    stored_result_sha = _verify_self_hash(
        result, "resultSha256", f"{label} result"
    )
    if (
        stored_gate_sha != gate_sha256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != outcome_band
        or gate.get("resultSha256") != result_sha256
        or stored_result_sha != result_sha256
        or gate.get("statistics") != result.get("statistics")
    ):
        raise Phase4F63Error(f"{label} evidence changed")
    return {
        "gatePath": gate_path.as_posix(),
        "gateSha256": stored_gate_sha,
        "resultPath": result_path.as_posix(),
        "resultFileSha256": sha256_file(workspace / result_path),
        "resultSha256": result_sha256,
        "statistics": gate["statistics"],
        "outcomeBand": outcome_band,
    }


def _evidence(workspace: Path) -> dict[str, object]:
    if (
        sha256_file(workspace / ROOT_CAUSE_REPORT) != EXPECTED_ROOT_CAUSE_SHA256
        or sha256_file(workspace / F60_POSTMORTEM)
        != EXPECTED_F60_POSTMORTEM_SHA256
        or sha256_file(workspace / F21_MODEL) != EXPECTED_F21_MODEL_SHA256
        or sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256
    ):
        raise Phase4F63Error("F21 or diagnostic input changed")
    _load_f21_state(workspace)
    f22 = _arena_evidence(
        workspace,
        label="F22",
        gate_path=F22_GATE,
        result_path=F22_RESULT,
        gate_sha256=EXPECTED_F22_GATE_SHA256,
        result_sha256=EXPECTED_F22_RESULT_SHA256,
        outcome_band="stage-b-transition-candidate",
    )
    f61 = _arena_evidence(
        workspace,
        label="F61",
        gate_path=F61_GATE,
        result_path=F61_RESULT,
        gate_sha256=EXPECTED_F61_GATE_SHA256,
        result_sha256=EXPECTED_F61_RESULT_SHA256,
        outcome_band="value-scale-screen-positive",
    )
    f62 = _arena_evidence(
        workspace,
        label="F62",
        gate_path=F62_GATE,
        result_path=F62_RESULT,
        gate_sha256=EXPECTED_F62_GATE_SHA256,
        result_sha256=EXPECTED_F62_RESULT_SHA256,
        outcome_band="value-scale-confirmation-passed",
    )
    if (
        f22["statistics"]["games"] != 64
        or f22["statistics"]["candidateWins"] != 36
        or f61["statistics"]["games"] != 64
        or f61["statistics"]["candidateWins"] != 45
        or f62["statistics"]["games"] != 400
        or f62["statistics"]["candidateWins"] != 231
        or float(f62["statistics"]["winRateCI"]["lower"]) <= 0.5
    ):
        raise Phase4F63Error("accepted F22/F61/F62 statistics changed")
    return {
        "f21": {
            "modelPath": F21_MODEL.as_posix(),
            "modelSha256": EXPECTED_F21_MODEL_SHA256,
            "modelStateSha256": EXPECTED_F21_STATE_SHA256,
            "configPath": F21_CONFIG.as_posix(),
            "configFileSha256": EXPECTED_F21_CONFIG_SHA256,
        },
        "diagnostics": {
            "rootCauseReportPath": ROOT_CAUSE_REPORT.as_posix(),
            "rootCauseReportSha256": EXPECTED_ROOT_CAUSE_SHA256,
            "f60PostmortemPath": F60_POSTMORTEM.as_posix(),
            "f60PostmortemSha256": EXPECTED_F60_POSTMORTEM_SHA256,
        },
        "f22UnscaledVersusClassicHard": f22,
        "f61ScaledVersusUnscaledScreen": f61,
        "f62ScaledVersusUnscaledConfirmation": f62,
    }


def _findings(evidence: Mapping[str, object]) -> dict[str, object]:
    f22 = evidence["f22UnscaledVersusClassicHard"]["statistics"]
    f61 = evidence["f61ScaledVersusUnscaledScreen"]["statistics"]
    f62 = evidence["f62ScaledVersusUnscaledConfirmation"]["statistics"]
    return {
        "confirmed": {
            "f21HasUsefulPolicyAndNonGloballyInvertedValue": True,
            "unscaledF21PreviouslyClearedA64GameClassicHardScreen": True,
            "halvingOnlySearchValueImprovesF21AgainstUnscaledF21": True,
            "causalConfirmationLowerBoundAboveHalf": True,
        },
        "effect": {
            "f22UnscaledVersusClassicHardWinRate": f22["winRate"],
            "f61ScaleHalfVersusOneWinRate": f61["winRate"],
            "f62ScaleHalfVersusOneWinRate": f62["winRate"],
            "f62WinRateCILower": f62["winRateCI"]["lower"],
            "f62Elo": f62["elo"],
            "f62EloCILower": f62["eloCI"]["lower"],
        },
        "notEstablished": {
            "valueScaleHalfIsGloballyOptimal": True,
            "terminalOutcomeLabelsShouldBeMultipliedByHalf": True,
            "valueHeadWeightsShouldBeRetrained": True,
            "benefitTransfersToClassicHard": True,
            "benefitTransfersAcrossSimulationBudgets": True,
        },
        "identifiability": {
            "finding": "search-value-influence-is-too-large-at-the-deployed-runtime",
            "ambiguity": "scaling-Q-also-changes-the-PUCT-Q-to-exploration-balance",
            "consequence": "do-not-label-the-result-as-probability-calibration-or-retrain-the-value-head",
        },
    }


def _f64_design(
    *,
    source: Mapping[str, object],
    evidence: Mapping[str, object],
    ledger: Mapping[str, object],
    resources: Mapping[str, object],
    review_sha256: str,
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f64-scaled-f21-classic-hard-confirmation-design",
            "status": "frozen",
            "reviewReportSha256": review_sha256,
            "source": source,
            "evidence": {
                "f22GateSha256": evidence["f22UnscaledVersusClassicHard"]["gateSha256"],
                "f61GateSha256": evidence["f61ScaledVersusUnscaledScreen"]["gateSha256"],
                "f62GateSha256": evidence["f62ScaledVersusUnscaledConfirmation"]["gateSha256"],
            },
            "candidate": {
                **evidence["f21"],
                "engine": "native-mcts",
                "simulations": 800,
                "valueScale": 0.5,
                "policyAndWeightsIdenticalToF21": True,
            },
            "opponent": {
                "engine": "classic-negamax",
                "difficulty": "classic-hard",
                "budgetKind": "nodes",
                "budgetLimit": 587,
                "maximumDepth": 5,
            },
            "openingIsolation": {
                "suite": "fresh-development",
                "openingPairs": 200,
                "games": 400,
                "excludeAllTrainingReplayPositionOrbits": True,
                "excludeAllPriorArenaPositionOrbits": True,
                "excludeF22F61F62OpeningsAndTrajectories": True,
                "screenTrajectoriesTrainingUseAllowed": False,
            },
            "workload": {
                "actors": 32,
                "mctsSeed": 20260922,
                "bootstrapSeed": 20260922,
                "bootstrapSamples": 10_000,
                "confidence": 0.90,
                "rootNoise": False,
                "temperature": 0.0,
            },
            "decisionPolicy": {
                "primaryMetric": "paired-arena-candidate-win-rate",
                "minimumWinRate": 0.55,
                "minimumOneSided95Lower": 0.50,
                "drawsRequired": 0,
                "passOutcome": "scaled-f21-runtime-promotion-candidate",
                "nonPassOutcome": "value-scale-external-transfer-not-confirmed",
                "noAdditionalScreenBeforeDecision": True,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "budget": {
                "baselineGlobalHours": ledger["totalGpuHours"],
                "confirmationLimitHours": F64_GPU_HOURS,
                "globalLimitHours": GLOBAL_GPU_HOURS_LIMIT,
                "artifactLimitBytes": ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": F64_ARENA_WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "resourcesAtReview": resources,
            "formalTrainingAuthorized": False,
            "valueHeadTrainingAuthorized": False,
            "promotionAuthorized": False,
            "confirmationPreparationAuthorized": True,
        },
        "designSha256",
    )


def _report_core(
    *,
    source: Mapping[str, object],
    evidence: Mapping[str, object],
    findings: Mapping[str, object],
    ledger: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f63-value-head-calibration-design-review",
        "status": "completed",
        "cpuOnly": True,
        "cudaAccessed": False,
        "trainingStarted": False,
        "optimizerSteps": 0,
        "source": source,
        "evidence": evidence,
        "findings": findings,
        "designDecision": {
            "selectedIntervention": "explicit-search-runtime-value-scale-0.5",
            "valueHeadSemanticOutput": "retain-terminal-outcome-in-minus-one-to-one",
            "modelWeightChange": "none",
            "targetChange": "none",
            "reason": "F62-identifies-harmful-search-value-influence-not-a-training-target-transform",
            "externalTransferTest": "scaled-F21-versus-classic-hard-400-game-confirmation",
            "ifExternalTransferPasses": "consider-runtime-calibration-promotion-before-any-value-training",
            "ifExternalTransferFails": "stop-value-scale-lineage-and-audit-PUCT-exploration-balance",
        },
        "processCorrection": {
            "skipAnother64GameScreen": True,
            "reason": "F22-already-screened-F21-versus-classic-and-F62-powered-the-intervention",
            "singlePoweredExternalDecision": True,
        },
        "ledger": ledger,
        "resources": resources,
        "reviewPassed": True,
        "f64ConfirmationPreparationAuthorized": True,
        "formalTrainingAuthorized": False,
        "valueHeadTrainingAuthorized": False,
        "promotionAuthorized": False,
    }


def run_phase4f63(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    run_id = "phase4f63-value-calibration-review"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        if (workspace / REPORT).exists():
            return verify_phase4f63(workspace)
        resources = _resources(workspace)
        _, ledger = _ledger(workspace)
        evidence = _evidence(workspace)
        source = _source(workspace)
        findings = _findings(evidence)
        report = _self_hashed(
            _report_core(
                source=source,
                evidence=evidence,
                findings=findings,
                ledger=ledger,
                resources=resources,
            ),
            "reportSha256",
        )
        design = _f64_design(
            source=source,
            evidence=evidence,
            ledger=ledger,
            resources=resources,
            review_sha256=str(report["reportSha256"]),
        )
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f63-value-head-calibration-design-review-gate",
                "status": "completed",
                "outcomeBand": "external-transfer-confirmation-candidate",
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f64DesignPath": F64_DESIGN.as_posix(),
                "f64DesignSha256": design["designSha256"],
                "reviewPassed": True,
                "f64ConfirmationPreparationAuthorized": True,
                "formalTrainingAuthorized": False,
                "valueHeadTrainingAuthorized": False,
                "promotionAuthorized": False,
                "globalGpuHours": ledger["totalGpuHours"],
                "nextStage": {
                    "name": "Phase 4F64 scaled F21 versus classic-hard confirmation preparation",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "gateSha256",
        )
        _write_immutable(workspace / REPORT, report, "F63 review report")
        _write_immutable(workspace / F64_DESIGN, design, "F64 frozen design")
        _write_immutable(workspace / GATE, gate, "F63 review gate")
    return verify_phase4f63(workspace)


def verify_phase4f63(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F63 review report")
    gate = _mapping(workspace / GATE, "F63 review gate")
    design = _mapping(workspace / F64_DESIGN, "F64 frozen design")
    report_sha = _verify_self_hash(report, "reportSha256", "F63 review report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F63 review gate")
    design_sha = _verify_self_hash(design, "designSha256", "F64 frozen design")
    _, ledger = _ledger(workspace)
    evidence = _evidence(workspace)
    source = _source(workspace)
    findings = _findings(evidence)
    expected_core = _report_core(
        source=source,
        evidence=evidence,
        findings=findings,
        ledger=report["ledger"],
        resources=report["resources"],
    )
    expected_report = _self_hashed(expected_core, "reportSha256")
    expected_design = _f64_design(
        source=source,
        evidence=evidence,
        ledger=report["ledger"],
        resources=report["resources"],
        review_sha256=report_sha,
    )
    if (
        report != expected_report
        or design != expected_design
        or gate.get("reportSha256") != report_sha
        or gate.get("f64DesignSha256") != design_sha
        or gate.get("reviewPassed") is not True
        or gate.get("f64ConfirmationPreparationAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
        or gate.get("valueHeadTrainingAuthorized") is not False
        or gate.get("promotionAuthorized") is not False
        or float(ledger["totalGpuHours"]) < float(report["ledger"]["totalGpuHours"])
    ):
        raise Phase4F63Error("F63/F64 frozen provenance changed")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f63-value-head-calibration-design-review-verification",
        "status": "verified",
        "reviewPassed": True,
        "f64ConfirmationPreparationAuthorized": True,
        "formalTrainingAuthorized": False,
        "valueHeadTrainingAuthorized": False,
        "promotionAuthorized": False,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "f64DesignPath": F64_DESIGN.as_posix(),
        "f64DesignSha256": design_sha,
        "globalGpuHours": ledger["totalGpuHours"],
        "outcomeBand": gate["outcomeBand"],
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f63(args.workspace_root)
        if args.command == "run"
        else verify_phase4f63(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
