"""Review F44 evidence and freeze the F46 fresh-lineage screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .openings import OpeningSuite, build_opening_suite, load_opening_suite, opening_suite_to_payload
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f44_curriculum_pilot import verify_phase4f44
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
F44_GATE = Path("training/runs/phase4/phase4f/f44-early-curriculum-pilot-v1/pilot-gate.json")
F44_REPORT = Path("training/runs/phase4/phase4f/f44-early-curriculum-pilot-v1/pilot-report.json")
EXPECTED_F44_GATE_SHA256 = "5e2786dba442a24bfed8b8237b5856b24999695354af0e0f48e1bd3f0b858f9b"
EXPECTED_F44_REPORT_SHA256 = "e2bdc0551f8c08c1412fb6e29052aa0939eeeb9a9c8aafd458804a48dda4baa9"
EXPECTED_F44_MODEL_SHA256 = "120894608c468c0283a5e8dc4a3f35a4d12cf3e138077e0b742c730efae24087"
EXPECTED_F44_STATE_SHA256 = "fcfc513db0ad8381451b9749476bd7117d99c64e285767913f059d03c9b62de6"

F21_MODEL = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/checkpoints/"
    "candidate-model-checkpoint-g00000001-s000000000192.pt"
)
F21_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
EXPECTED_F21_MODEL_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
EXPECTED_F21_STATE_SHA256 = "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
EXPECTED_F21_CONFIG_FILE_SHA256 = "bdaa011c52a3a3f0925dcf603d52bd19efe4b7cee6039d6e03140b14d007a14e"

FRESH_SUITE = Path(
    "training/runs/phase4/phase4f/f31-fresh-development-screen-v1/recovery-fresh-dev-v2.json"
)
EXPECTED_FRESH_SUITE_FILE_SHA256 = "8ca0fc56b74c1c05e2352d9774e97dd3f9acae1b508d51998ffb2262fac58aa9"
EXPECTED_FRESH_SUITE_HASH = "sha256:e4d420136a2579ed0ec514ecccc503544d9da2244e8cc93aba985fc223c65c43"
F32_ISOLATION_AUDIT = Path(
    "training/runs/phase4/phase4f/f31-fresh-development-screen-v1/isolation-audit.json"
)
EXPECTED_F32_ISOLATION_FILE_SHA256 = "34e01fd3c49e71c5815d5fc6ab8f60389c3cdfa4945b72d49c58bc0ff26824e3"

ROOT = Path("training/runs/phase4/phase4f/f45-curriculum-pilot-review-v1")
DELTA_AUDIT = ROOT / "model-delta-audit.json"
SCREEN_SUITE = ROOT / "f46-fresh-lineage-screen-32.json"
REPORT = ROOT / "pilot-review.json"
GATE = ROOT / "review-gate.json"
F46_CONTRACT = Path("training/configs/phase4f46-curriculum-lineage-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260825
MCTS_SEED = 20260825
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
SCREEN_GPU_HOURS = 2.0
ARENA_WRITE_RESERVE_BYTES = 32 * 1024**2


class Phase4F45ReviewError(RuntimeError):
    """Raised when F44 review evidence cannot authorize F46."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F45ReviewError("workspace does not contain training/quoridor")
    return workspace


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F45ReviewError("artifact path escapes workspace") from error


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f45_curriculum_pilot_review.py",
            "training/quoridor/phase4f46_curriculum_lineage_screen.py",
            "training/quoridor/phase4_arena.py",
            "training/quoridor/evaluator.py",
            "training/quoridor/mcts.py",
            "training/quoridor/native_mcts.py",
            "training/quoridor/network.py",
            "training/quoridor/openings.py",
            "training/quoridor/paired_arena.py",
            "training/quoridor/selectors.py",
        ),
    ).to_payload()


def _load_state(path: Path, expected_file: str, expected_state: str) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_file:
        raise Phase4F45ReviewError(f"model file failed validation: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F45ReviewError(f"model payload is unreadable: {path}") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or not isinstance(state, Mapping):
        raise Phase4F45ReviewError(f"model payload has no modelState: {path}")
    if canonical_model_state_sha256(state) != expected_state:
        raise Phase4F45ReviewError(f"model state hash differs: {path}")
    return dict(payload), state


def _f44_evidence(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    verification = verify_phase4f44(workspace)
    if (
        verification.get("gateSha256") != EXPECTED_F44_GATE_SHA256
        or verification.get("reportSha256") != EXPECTED_F44_REPORT_SHA256
        or verification.get("pilotPassed") is not True
        or verification.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F45ReviewError("F44 is not the accepted pilot evidence")
    report = _mapping(workspace / F44_REPORT, "F44 report")
    if _verify_self_hash(report, "reportSha256", "F44 report") != EXPECTED_F44_REPORT_SHA256:
        raise Phase4F45ReviewError("F44 report hash differs")
    checkpoint = report.get("selectedCheckpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("step") != 32:
        raise Phase4F45ReviewError("F44 did not select step 32")
    payload, state = _load_state(
        workspace / str(checkpoint["path"]), EXPECTED_F44_MODEL_SHA256, EXPECTED_F44_STATE_SHA256
    )
    if (
        payload.get("format") != "quoridor-phase4f44-curriculum-pilot-model-only"
        or payload.get("step") != 32
        or payload.get("modelStateSha256") != EXPECTED_F44_STATE_SHA256
    ):
        raise Phase4F45ReviewError("F44 checkpoint contract differs")
    return report, state


def _f21_evidence(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_FILE_SHA256:
        raise Phase4F45ReviewError("F21 config file hash differs")
    payload, state = _load_state(
        workspace / F21_MODEL, EXPECTED_F21_MODEL_SHA256, EXPECTED_F21_STATE_SHA256
    )
    identity = payload.get("identity")
    if (
        payload.get("format") != "quoridor-phase4-candidate-model-only"
        or payload.get("generation") != 1
        or payload.get("step") != 192
        or not isinstance(identity, Mapping)
    ):
        raise Phase4F45ReviewError("F21 candidate contract differs")
    return dict(payload), state


def _is_frozen_key(name: str) -> bool:
    if name.startswith(("input_conv.", "input_bn.")):
        return True
    if not name.startswith("trunk."):
        return False
    try:
        return int(name.split(".", 2)[1]) < 8
    except (IndexError, ValueError):
        return False


def model_delta_audit(
    initial: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    if tuple(initial) != tuple(candidate):
        raise Phase4F45ReviewError("F44 and F21 model keys differ")
    frozen_changed: list[str] = []
    trainable_changed: list[str] = []
    unchanged_trainable: list[str] = []
    maximum = 0.0
    changed_elements = 0
    for name, before in initial.items():
        after = candidate[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise Phase4F45ReviewError(f"tensor contract differs for {name}")
        equal = torch.equal(before, after)
        if _is_frozen_key(name):
            if not equal:
                frozen_changed.append(name)
            continue
        if equal:
            unchanged_trainable.append(name)
            continue
        trainable_changed.append(name)
        changed_elements += int(torch.count_nonzero(before != after).item())
        if before.is_floating_point():
            maximum = max(maximum, float((before - after).abs().max().item()))
    checks = {
        "sameTensorContract": True,
        "frozenPrefixUnchanged": not frozen_changed,
        "trainableRegionChanged": bool(trainable_changed),
        "modelStateChanged": canonical_model_state_sha256(initial) != canonical_model_state_sha256(candidate),
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f45-model-delta-audit",
        "checks": checks,
        "frozenTensorCount": sum(_is_frozen_key(name) for name in initial),
        "frozenChanged": frozen_changed,
        "trainableTensorCount": sum(not _is_frozen_key(name) for name in initial),
        "trainableChangedCount": len(trainable_changed),
        "trainableChanged": trainable_changed,
        "unchangedTrainable": unchanged_trainable,
        "changedElements": changed_elements,
        "maximumAbsoluteDelta": maximum,
        "passed": all(checks.values()),
    }


def loss_review(report: Mapping[str, object]) -> dict[str, object]:
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, list) or [item.get("step") for item in evaluations] != [0, 8, 16, 24, 32]:
        raise Phase4F45ReviewError("F44 evaluation schedule differs")
    baseline = evaluations[0]["metrics"]
    selected = evaluations[-1]["metrics"]
    improvements: dict[str, object] = {}
    for band in ("all", "early", "middle", "late"):
        improvements[band] = {
            metric: (float(baseline[band][metric]) - float(selected[band][metric]))
            / float(baseline[band][metric])
            for metric in ("policyLoss", "valueLoss", "totalLoss")
        }
    totals = [float(item["metrics"]["all"]["totalLoss"]) for item in evaluations]
    early_totals = [float(item["metrics"]["early"]["totalLoss"]) for item in evaluations]
    checks = {
        "selectedStep32": report.get("selectedCheckpoint", {}).get("step") == 32,
        "allPostBaselineEligible": all(item.get("eligible") is True for item in evaluations[1:]),
        "overallTotalLossStrictlyDecreasing": all(a > b for a, b in zip(totals, totals[1:])),
        "earlyTotalLossStrictlyDecreasing": all(a > b for a, b in zip(early_totals, early_totals[1:])),
        "overallRelativeImprovementAtLeastTenPercent": improvements["all"]["totalLoss"] >= 0.10,
        "earlyRelativeImprovementAtLeastTenPercent": improvements["early"]["totalLoss"] >= 0.10,
        "noBandTotalLossRegression": all(improvements[band]["totalLoss"] > 0.0 for band in ("middle", "late")),
    }
    return {
        "baselineStep": 0,
        "selectedStep": 32,
        "relativeImprovements": improvements,
        "overallTotalLossSequence": totals,
        "earlyTotalLossSequence": early_totals,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _fresh_suite_evidence(workspace: Path) -> tuple[OpeningSuite, dict[str, object]]:
    full_path = workspace / FRESH_SUITE
    if sha256_file(full_path) != EXPECTED_FRESH_SUITE_FILE_SHA256:
        raise Phase4F45ReviewError("F32 full fresh suite file hash differs")
    full = load_opening_suite(full_path, expected_suite="recovery-fresh-dev-v1")
    if full.content_hash != EXPECTED_FRESH_SUITE_HASH or len(full.openings) != 200:
        raise Phase4F45ReviewError("F32 full fresh suite contract differs")
    audit_path = workspace / F32_ISOLATION_AUDIT
    if sha256_file(audit_path) != EXPECTED_F32_ISOLATION_FILE_SHA256:
        raise Phase4F45ReviewError("F32 isolation audit file hash differs")
    audit = _mapping(audit_path, "F32 isolation audit")
    audit_sha = _verify_self_hash(audit, "auditSha256", "F32 isolation audit")
    if audit.get("status") != "completed" or audit.get("freshOverlapCount") != 0:
        raise Phase4F45ReviewError("F32 fresh suite is not isolated")

    full_hashes = {opening.content_hash for opening in full.openings}
    excluded: set[str] = set()
    references: list[dict[str, object]] = []
    for path in sorted((workspace / "training/runs/phase4/phase4f").rglob("opening-suite.json")):
        try:
            suite = load_opening_suite(path)
        except Exception:
            continue
        overlap = full_hashes.intersection(opening.content_hash for opening in suite.openings)
        if not overlap:
            continue
        excluded.update(overlap)
        references.append(
            {
                "path": _relative(path, workspace),
                "fileSha256": sha256_file(path),
                "suiteHash": suite.content_hash,
                "overlapOpeningPairs": len(overlap),
            }
        )
    eligible = [opening for opening in full.openings if opening.content_hash not in excluded]
    if len(eligible) < OPENING_PAIRS:
        raise Phase4F45ReviewError("not enough unused F32 fresh openings")

    def score(opening: object) -> str:
        return hashlib.sha256(
            f"{SELECTION_SEED}\0{getattr(opening, 'content_hash')}".encode("ascii")
        ).hexdigest()

    selected = tuple(sorted(eligible, key=lambda opening: (score(opening), opening.id))[:OPENING_PAIRS])
    screen = build_opening_suite(selected, suite="recovery-fresh-dev-v1")
    return screen, {
        "fullPath": FRESH_SUITE.as_posix(),
        "fullFileSha256": EXPECTED_FRESH_SUITE_FILE_SHA256,
        "fullSuiteHash": full.content_hash,
        "fullOpeningPairs": len(full.openings),
        "inheritedIsolationAuditPath": F32_ISOLATION_AUDIT.as_posix(),
        "inheritedIsolationAuditFileSha256": EXPECTED_F32_ISOLATION_FILE_SHA256,
        "inheritedIsolationAuditSha256": audit_sha,
        "inheritedReplayOverlapCount": 0,
        "historicalArenaReferences": references,
        "excludedOpeningPairs": len(excluded),
        "eligibleOpeningPairs": len(eligible),
        "selectionAlgorithm": "sha256(seed-ascii+nul+opening-content-hash)-lowest-v1",
        "selectionSeed": SELECTION_SEED,
        "selectedOpeningIds": [opening.id for opening in screen.openings],
        "selectedSuiteHash": screen.content_hash,
        "selectedOpeningPairs": len(screen.openings),
    }


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F45ReviewError("16 GiB artifact cap cannot reserve F46 output")
    if free_bytes - ARENA_WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F45ReviewError("F46 would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def run_phase4f45(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f45(workspace), "reused": True}
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F45ReviewError("a running GPU ledger entry blocks F45 freeze")
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F45ReviewError("global GPU budget cannot reserve F46")
    f44_report, f44_state = _f44_evidence(workspace)
    f21_payload, f21_state = _f21_evidence(workspace)
    delta = _self_hashed(model_delta_audit(f21_state, f44_state), "auditSha256")
    losses = loss_review(f44_report)
    screen, suite_reference = _fresh_suite_evidence(workspace)
    resources = _resources(workspace)
    source = _source(workspace)
    identity = f21_payload["identity"]
    workload = _self_hashed(
        {
            "suite": "recovery-fresh-dev-v1",
            "suiteHash": screen.content_hash,
            "openingPairs": OPENING_PAIRS,
            "games": GAMES,
            "candidateSimulations": SIMULATIONS,
            "opponentSimulations": SIMULATIONS,
            "actors": ACTORS,
            "mctsSeed": MCTS_SEED,
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapSamples": BOOTSTRAP_SAMPLES,
            "confidence": CONFIDENCE,
        },
        "workloadSha256",
    )
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f46-corrected-curriculum-versus-f21-fresh-lineage-screen",
            "status": "frozen",
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "candidate": {
                "selectorId": "phase4f44-curriculum-step-00000032",
                "modelPath": f44_report["selectedCheckpoint"]["path"],
                "modelFileSha256": EXPECTED_F44_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F44_STATE_SHA256,
                "configPath": "training/configs/phase4f44-early-curriculum-pilot-frozen-v1.json",
                "configSha256": f44_report["contractSha256"],
                "sourceSha256": f44_report["runnerSha256"],
            },
            "opponent": {
                "selectorId": "phase4f21-stage-a-best-g00000001-s000000000192",
                "modelPath": F21_MODEL.as_posix(),
                "modelFileSha256": EXPECTED_F21_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "configPath": F21_CONFIG.as_posix(),
                "configFileSha256": EXPECTED_F21_CONFIG_FILE_SHA256,
                "configSha256": identity["configSha256"],
                "sourceSha256": identity["sourceSha256"],
            },
            "runtimeConfigPath": F21_CONFIG.as_posix(),
            "openingSuite": {
                **suite_reference,
                "selectedPath": SCREEN_SUITE.as_posix(),
                "selectedFileSha256": None,
            },
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger.total_hours,
                "screenLimitHours": SCREEN_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "regressionBelowWinRate": 0.45,
                "inconclusiveBelowWinRate": 0.50,
                "retainedAtOrAboveWinRate": 0.50,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "authorizations": {
                "f46LineageScreenAuthorized": True,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "screenTrajectoriesTrainingUseAllowed": False,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    run_id = "phase4f45-curriculum-pilot-review"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        _write_immutable(workspace / DELTA_AUDIT, delta, "F45 model delta audit")
        _write_immutable(workspace / SCREEN_SUITE, opening_suite_to_payload(screen), "F46 opening suite")
        contract["openingSuite"]["selectedFileSha256"] = sha256_file(workspace / SCREEN_SUITE)
        contract = _self_hashed({k: v for k, v in contract.items() if k != "contractSha256"}, "contractSha256")
        _write_immutable(workspace / F46_CONTRACT, contract, "F46 contract")
        review_passed = bool(delta["passed"] and losses["passed"])
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f45-corrected-curriculum-pilot-review",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "f44GateSha256": EXPECTED_F44_GATE_SHA256,
                "f44ReportSha256": EXPECTED_F44_REPORT_SHA256,
                "modelDeltaAuditPath": DELTA_AUDIT.as_posix(),
                "modelDeltaAuditSha256": delta["auditSha256"],
                "lossReview": losses,
                "freshOpeningSuite": contract["openingSuite"],
                "reviewPassed": review_passed,
                "f46LineageScreenAuthorized": review_passed,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "f46ContractPath": F46_CONTRACT.as_posix(),
                "f46ContractSha256": contract["contractSha256"],
                "source": source,
                "globalGpuHours": ledger.total_hours,
                "resources": resources,
            },
            "reportSha256",
        )
        _write_immutable(workspace / REPORT, report, "F45 review")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f45-curriculum-pilot-review-gate",
                "status": "completed",
                "reviewPassed": review_passed,
                "outcomeBand": "f46-fresh-lineage-screen-ready" if review_passed else "curriculum-pilot-review-failed",
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f46ContractPath": F46_CONTRACT.as_posix(),
                "f46ContractSha256": contract["contractSha256"],
                "f46LineageScreenAuthorized": review_passed,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "nextStage": {
                    "name": "Phase 4F46 fresh-lineage screen",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F45 gate")
    return {**gate, "globalGpuHours": ledger.total_hours, "reused": False}


def verify_phase4f45(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    _f44_report, f44_state = _f44_evidence(workspace)
    _f21_payload, f21_state = _f21_evidence(workspace)
    delta = _mapping(workspace / DELTA_AUDIT, "F45 delta audit")
    _verify_self_hash(delta, "auditSha256", "F45 delta audit")
    if delta != _self_hashed(model_delta_audit(f21_state, f44_state), "auditSha256"):
        raise Phase4F45ReviewError("F45 model delta audit is not reproducible")
    report = _mapping(workspace / REPORT, "F45 review")
    _verify_self_hash(report, "reportSha256", "F45 review")
    gate = _mapping(workspace / GATE, "F45 gate")
    _verify_self_hash(gate, "gateSha256", "F45 gate")
    contract = _mapping(workspace / F46_CONTRACT, "F46 contract")
    _verify_self_hash(contract, "contractSha256", "F46 contract")
    screen, reference = _fresh_suite_evidence(workspace)
    selected_path = workspace / SCREEN_SUITE
    expected_opening = {
        **reference,
        "selectedPath": SCREEN_SUITE.as_posix(),
        "selectedFileSha256": sha256_file(selected_path),
    }
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        contract.get("source") != _source(workspace)
        or contract.get("openingSuite") != expected_opening
        or load_opening_suite(selected_path, expected_suite="recovery-fresh-dev-v1") != screen
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("f46ContractSha256") != contract.get("contractSha256")
        or report.get("modelDeltaAuditSha256") != delta.get("auditSha256")
        or report.get("reviewPassed") is not True
        or gate.get("reportSha256") != report.get("reportSha256")
        or gate.get("f46ContractSha256") != contract.get("contractSha256")
        or gate.get("f46LineageScreenAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
        or not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F45ReviewError("F45/F46 frozen provenance changed")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f45-curriculum-pilot-review-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate["gateSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "f46ContractPath": F46_CONTRACT.as_posix(),
        "f46ContractSha256": contract["contractSha256"],
        "f46LineageScreenAuthorized": True,
        "formalTrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = run_phase4f45(args.workspace_root) if args.command == "run" else verify_phase4f45(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
