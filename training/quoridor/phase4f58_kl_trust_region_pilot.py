"""Run the frozen F58 hard-KL trust-region policy-head CUDA pilot."""

from __future__ import annotations

import argparse
import io
import math
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .network import PolicyValueNetwork
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f55_policy_head_pilot_preparation import (
    EXPECTED_F21_STATE_SHA256,
    EXPECTED_F54_DATASET_FINGERPRINT,
    _load_initial_model,
    freeze_policy_head,
    verify_phase4f55,
)
from .phase4f56_policy_head_pilot import (
    StratifiedSampler,
    TensorCache,
    _amp_context,
    _baseline_values,
    _cpu_state,
    _cuda_preflight,
    _forward_kl,
    _load_inputs,
    _memory_smoke as _f56_memory_smoke,
    _mixed_anchor_reference,
    _reference_anchor_policies,
    _select_batch,
    _teacher_loss,
    acceptance_checks,
    evaluate_model,
)
from .phase4f57_policy_head_pilot_review import F58_CONTRACT, verify_phase4f57
from .run_manifest import (
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_CONTRACT_SHA256 = "67bb8d52329934b5c9a3b66038a13318ae5e3df9b896a2d423dfa8c2408d20ea"
ROOT = Path("training/runs/phase4/phase4f/f58-kl-trust-region-policy-head-pilot-v1")
CHECKPOINT = ROOT / "selected-policy-head-model.pt"
REPORT = ROOT / "pilot-report.json"
GATE = ROOT / "pilot-gate.json"
RUN_ID_PREFIX = "phase4f58-kl-trust-region-policy-head-pilot"


class Phase4F58Error(RuntimeError):
    """Raised when the frozen F58 pilot cannot run or verify safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f58-kl-trust-region-pilot-progress",
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
        raise Phase4F58Error("workspace does not contain training/quoridor")
    return workspace


def _runner_source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f58_kl_trust_region_pilot.py",
            "training/quoridor/phase4f57_policy_head_pilot_review.py",
            "training/quoridor/phase4f56_policy_head_pilot.py",
            "training/quoridor/phase4f55_policy_head_pilot_preparation.py",
            "training/quoridor/codec.py",
            "training/quoridor/network.py",
        ),
    ).to_payload()


def _contract(workspace: Path) -> dict[str, object]:
    payload = _mapping(workspace / F58_CONTRACT, "F58 contract")
    if (
        _verify_self_hash(payload, "contractSha256", "F58 contract")
        != EXPECTED_CONTRACT_SHA256
        or payload.get("status") != "frozen"
        or payload.get("authorizations", {}).get("cudaPilotTrainingAuthorized")
        is not True
        or payload.get("authorizations", {}).get("formalTrainingAuthorized")
        is not False
        or payload.get("initialization", {}).get(
            "f56DiagnosticCheckpointInitializationAllowed"
        )
        is not False
    ):
        raise Phase4F58Error("F58 contract is not the accepted frozen contract")
    return payload


def _resource_preflight(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, int]:
    limits = contract["resources"]
    artifacts = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    output = directory_size(workspace / ROOT) if (workspace / ROOT).exists() else 0
    reserve = int(limits["maximumOutputBytes"]) - output
    free = shutil.disk_usage(workspace).free
    if reserve < 0:
        raise Phase4F58Error("F58 output already exceeds its frozen limit")
    if artifacts + reserve >= int(limits["artifactLimitBytes"]):
        raise Phase4F58Error("F58 write reserve exceeds the frozen artifact cap")
    if free - reserve < int(limits["minimumFreeDiskBytes"]):
        raise Phase4F58Error("F58 reserve violates the free-disk stop line")
    return {
        "artifactBytes": artifacts,
        "artifactLimitBytes": int(limits["artifactLimitBytes"]),
        "f58OutputBytes": output,
        "remainingWriteReserveBytes": reserve,
        "freeDiskBytes": free,
        "minimumFreeDiskBytes": int(limits["minimumFreeDiskBytes"]),
    }


def trainable_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    )
    if not state or any(not name.startswith("policy_head.") for name in state):
        raise Phase4F58Error("F58 trainable state is not isolated to policy_head")
    return state


def apply_trainable_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if tuple(sorted(parameters)) != tuple(sorted(state)):
        raise Phase4F58Error("policy-head projection tensor contract changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise Phase4F58Error("policy-head projection tensor shape changed")
            parameter.copy_(value.to(parameter.device))


def interpolate_trainable_state(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    scale: float,
) -> None:
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("projection scale must be finite and in [0, 1]")
    if tuple(before) != tuple(after):
        raise Phase4F58Error("projection endpoints have different tensor contracts")
    interpolated = OrderedDict(
        (name, before[name] + (after[name] - before[name]) * scale)
        for name in before
    )
    apply_trainable_state(model, interpolated)


def project_trainable_update(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    measure: Callable[[], float],
    *,
    before_metric: float,
    target: float,
    hard_maximum: float,
    bisection_iterations: int,
    minimum_scale: float,
) -> dict[str, object]:
    if (
        not math.isfinite(before_metric)
        or before_metric > target
        or not 0.0 < target <= hard_maximum
        or type(bisection_iterations) is not int
        or bisection_iterations < 1
        or not 0.0 < minimum_scale <= 1.0
    ):
        raise Phase4F58Error("invalid trust-region projection precondition")
    apply_trainable_state(model, after)
    proposed_metric = float(measure())
    if not math.isfinite(proposed_metric):
        raise Phase4F58Error("proposed trust-region metric is non-finite")
    if proposed_metric <= target:
        if proposed_metric > hard_maximum:
            raise Phase4F58Error("accepted update exceeds the hard KL maximum")
        return {
            "accepted": True,
            "projected": False,
            "scale": 1.0,
            "beforeMetric": before_metric,
            "proposedMetric": proposed_metric,
            "acceptedMetric": proposed_metric,
        }

    low = 0.0
    high = 1.0
    low_metric = before_metric
    for _ in range(bisection_iterations):
        middle = (low + high) / 2.0
        interpolate_trainable_state(model, before, after, middle)
        metric = float(measure())
        if math.isfinite(metric) and metric <= target:
            low = middle
            low_metric = metric
        else:
            high = middle
    if low < minimum_scale:
        apply_trainable_state(model, before)
        restored = float(measure())
        if not math.isfinite(restored) or restored > hard_maximum:
            raise Phase4F58Error("rejected update could not restore the trust region")
        return {
            "accepted": False,
            "projected": True,
            "scale": 0.0,
            "beforeMetric": before_metric,
            "proposedMetric": proposed_metric,
            "acceptedMetric": restored,
        }
    interpolate_trainable_state(model, before, after, low)
    accepted_metric = float(measure())
    if (
        not math.isfinite(accepted_metric)
        or accepted_metric > target
        or accepted_metric > hard_maximum
    ):
        raise Phase4F58Error("projected update violates the trust region")
    return {
        "accepted": True,
        "projected": True,
        "scale": low,
        "beforeMetric": before_metric,
        "proposedMetric": proposed_metric,
        "acceptedMetric": accepted_metric,
        "bisectionLowMetric": low_metric,
    }


def projection_cpu_smoke() -> dict[str, object]:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = nn.Linear(1, 1, bias=False)

    model = TinyModel()
    model.policy_head.weight.requires_grad_(True)
    with torch.no_grad():
        model.policy_head.weight.zero_()
    before = trainable_state(model)
    after = OrderedDict(
        (name, tensor + 1.0) for name, tensor in before.items()
    )

    def metric() -> float:
        return float(model.policy_head.weight.detach().abs().item())

    result = project_trainable_update(
        model,
        before,
        after,
        metric,
        before_metric=0.0,
        target=0.25,
        hard_maximum=0.30,
        bisection_iterations=12,
        minimum_scale=1.0 / 4096.0,
    )
    passed = bool(
        result["accepted"] is True
        and result["projected"] is True
        and 0.249 <= float(result["acceptedMetric"]) <= 0.25
        and 0.249 <= float(result["scale"]) <= 0.25
    )
    if not passed:
        raise Phase4F58Error("F58 trust-region projection CPU smoke failed")
    return {"passed": True, **result}


def full_anchor_forward_kl(
    model: PolicyValueNetwork,
    anchor: TensorCache,
    reference: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    weighted = 0.0
    positions = 0
    model.eval()
    with torch.no_grad():
        for states, masks, expected in (
            (anchor.identity_states, anchor.identity_masks, reference[0]),
            (anchor.mirrored_states, anchor.mirrored_masks, reference[1]),
        ):
            for start in range(0, anchor.positions, 128):
                stop = min(start + 128, anchor.positions)
                batch_masks = masks[start:stop].to(device)
                logits, _ = model(states[start:stop].to(device))
                count = stop - start
                value = _forward_kl(
                    expected[start:stop].to(device), logits.float(), batch_masks
                )
                weighted += float(value.item()) * count
                positions += count
    if positions != anchor.positions * 2:
        raise Phase4F58Error("full-anchor KL did not cover identity and mirror")
    result = weighted / positions
    if not math.isfinite(result):
        raise Phase4F58Error("full-anchor KL is non-finite")
    return result


def _memory_smoke(
    workspace: Path,
    contract: Mapping[str, object],
    teacher: TensorCache,
    anchor: TensorCache,
    device: torch.device,
) -> dict[str, object]:
    smoke_contract = dict(contract)
    settings = dict(contract["training"])
    settings["anchorBatchSize"] = settings["anchorPenaltyBatchSize"]
    smoke_contract["training"] = settings
    result = _f56_memory_smoke(
        workspace, smoke_contract, teacher, anchor, device
    )
    result["projectionCpuSmoke"] = projection_cpu_smoke()
    result["trustRegionFullAnchorPositions"] = anchor.positions * 2
    return result


def _acceptance_checks(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    limits: Mapping[str, object],
    accepted_updates: int,
) -> dict[str, bool]:
    checks = acceptance_checks(baseline, candidate, limits)
    checks["acceptedOptimizerUpdates"] = accepted_updates >= int(
        limits["acceptedOptimizerUpdatesMinimum"]
    )
    return checks


def _trust_region_converged(
    records: Sequence[Mapping[str, object]], trust: Mapping[str, object]
) -> bool:
    maximum_rejections = int(trust["maximumConsecutiveRejectedUpdates"])
    if maximum_rejections <= 0 or len(records) < maximum_rejections:
        return False
    trailing = records[-maximum_rejections:]
    return (
        all(record.get("accepted") is False for record in trailing)
        and float(trust["projectionTargetMaximum"])
        - float(records[-1]["acceptedMetric"])
        <= 1e-6
    )


def audit_training_evidence(
    report: Mapping[str, object], contract: Mapping[str, object]
) -> dict[str, object]:
    records = report.get("trustRegion", {}).get("records")
    evaluations = report.get("evaluations")
    expected_steps = [int(value) for value in contract["training"]["evaluationSteps"]]
    target = float(contract["trustRegion"]["projectionTargetMaximum"])
    hard = float(contract["trustRegion"]["hardMaximum"])
    minimum_scale = float(contract["trustRegion"]["minimumAcceptedScale"])
    optimizer_steps = int(contract["training"]["optimizerSteps"])
    training = report.get("training", {})
    executed_steps = training.get("optimizerStepsExecuted")
    stop_reason = training.get("stopReason")
    if not isinstance(executed_steps, int):
        executed_steps = len(records) if isinstance(records, list) else -1
    expected_evaluation_steps = [
        step for step in expected_steps if step <= executed_steps
    ]
    if (
        not isinstance(records, list)
        or not 0 < executed_steps <= optimizer_steps
        or len(records) != executed_steps
        or [record.get("step") for record in records]
        != list(range(1, executed_steps + 1))
        or not isinstance(evaluations, list)
        or [entry.get("step") for entry in evaluations]
        != expected_evaluation_steps
    ):
        raise Phase4F58Error("F58 training evidence schedule is malformed")

    if executed_steps == optimizer_steps:
        if stop_reason not in (None, "optimizer-steps-completed"):
            raise Phase4F58Error("F58 completed training has an invalid stop reason")
    else:
        if (
            stop_reason != "trust-region-converged"
            or not _trust_region_converged(records, contract["trustRegion"])
        ):
            raise Phase4F58Error(
                "F58 early-stop evidence does not prove trust-region convergence"
            )

    accepted = 0
    projected = 0
    rejected = 0
    previous_metric = float(evaluations[0]["metrics"]["anchorForwardKl"])
    for record in records:
        before_metric = float(record["beforeMetric"])
        proposed_metric = float(record["proposedMetric"])
        accepted_metric = float(record["acceptedMetric"])
        scale = float(record["scale"])
        is_accepted = record.get("accepted") is True
        is_projected = record.get("projected") is True
        if (
            not all(
                math.isfinite(value)
                for value in (
                    before_metric,
                    proposed_metric,
                    accepted_metric,
                    scale,
                    float(record["teacherPolicyLoss"]),
                    float(record["anchorPenaltyBatchForwardKl"]),
                    float(record["combinedLoss"]),
                    float(record["gradientNorm"]),
                    float(record["learningRate"]),
                )
            )
            or abs(before_metric - previous_metric) > 1e-7
            or accepted_metric > target
            or accepted_metric > hard
            or not 0.0 <= scale <= 1.0
            or (is_accepted and scale < minimum_scale)
            or (not is_accepted and (scale != 0.0 or not is_projected))
            or (not is_projected and scale != 1.0)
        ):
            raise Phase4F58Error("F58 trust-region record is inconsistent")
        accepted += is_accepted
        projected += is_projected
        rejected += not is_accepted
        previous_metric = accepted_metric

    baseline = evaluations[0]["metrics"]
    eligible: list[Mapping[str, object]] = []
    for entry in evaluations:
        step = int(entry["step"])
        metrics = entry["metrics"]
        accepted_at_step = sum(
            record.get("accepted") is True for record in records[:step]
        )
        if (
            entry.get("acceptedOptimizerUpdates") != accepted_at_step
            or float(metrics["anchorForwardKl"]) > hard
            or metrics["frozenTensors"]["passed"] is not True
            or float(metrics["frozenTensors"]["maximumAbsoluteError"]) != 0.0
            or metrics["frozenTensors"]["changedTensorNames"]
            or float(metrics["valueOutputMaximumAbsoluteError"]) != 0.0
        ):
            raise Phase4F58Error("F58 evaluation invariant is inconsistent")
        if step == 0:
            expected_checks = {"baseline": True}
            expected_eligible = False
        else:
            expected_checks = _acceptance_checks(
                baseline,
                metrics,
                contract["evaluation"]["acceptance"],
                accepted_at_step,
            )
            expected_eligible = all(expected_checks.values())
        if entry.get("checks") != expected_checks or entry.get("eligible") is not expected_eligible:
            raise Phase4F58Error("F58 evaluation eligibility is not reproducible")
        if expected_eligible:
            eligible.append(entry)

    observed = evaluations[1:]
    selected_pool = eligible if eligible else observed
    selected = min(
        selected_pool,
        key=lambda entry: (float(entry["metrics"]["policyLoss"]["all"]), int(entry["step"])),
    )
    passed = bool(eligible)
    trust = report.get("trustRegion", {})
    if (
        training.get("optimizerSteps") != optimizer_steps
        or training.get("optimizerStepsExecuted", executed_steps) != executed_steps
        or training.get("acceptedOptimizerUpdates") != accepted
        or training.get("projectedUpdates") != projected
        or training.get("rejectedUpdates") != rejected
        or report.get("selectedCheckpoint", {}).get("step") != selected["step"]
        or report.get("selectedEligible") is not passed
        or report.get("pilotPassed") is not passed
        or float(trust.get("maximumAcceptedMetric", math.inf))
        != max(float(record["acceptedMetric"]) for record in records)
        or float(trust.get("maximumProposedMetric", math.inf))
        != max(float(record["proposedMetric"]) for record in records)
    ):
        raise Phase4F58Error("F58 aggregate training evidence is inconsistent")
    return {
        "acceptedOptimizerUpdates": accepted,
        "projectedUpdates": projected,
        "rejectedUpdates": rejected,
        "optimizerStepsPlanned": optimizer_steps,
        "optimizerStepsExecuted": executed_steps,
        "stopReason": stop_reason or "optimizer-steps-completed",
        "eligibleEvaluationSteps": [int(entry["step"]) for entry in eligible],
        "selectedStep": int(selected["step"]),
        "pilotPassed": passed,
        "maximumAcceptedMetric": max(
            float(record["acceptedMetric"]) for record in records
        ),
    }


def _write_checkpoint(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    *,
    step: int,
    contract_sha: str,
    runner_sha: str,
) -> dict[str, object]:
    model_state_sha = canonical_model_state_sha256(state)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "format": "quoridor-phase4f58-kl-trust-region-policy-head-pilot-model-only",
        "step": step,
        "contractSha256": contract_sha,
        "runnerSha256": runner_sha,
        "initialModelStateSha256": EXPECTED_F21_STATE_SHA256,
        "modelStateSha256": model_state_sha,
        "modelState": OrderedDict(
            (name, tensor.clone()) for name, tensor in state.items()
        ),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    path = workspace / CHECKPOINT
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise Phase4F58Error("existing F58 checkpoint differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_create_bytes(path, encoded)
    return {
        "path": CHECKPOINT.as_posix(),
        "fileSha256": sha256_file(path),
        "modelStateSha256": model_state_sha,
        "step": step,
        "bytes": path.stat().st_size,
    }


def run_phase4f58(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f58(workspace), "reused": True}
    _progress("preflight", "started")
    f57 = verify_phase4f57(workspace)
    _progress("preflight", "f57-verified")
    _progress("dataset-deep-verification", "started")
    f55 = verify_phase4f55(workspace)
    _progress(
        "dataset-deep-verification",
        "completed",
        trainPositions=f55["positions"]["train"],
        developmentPositions=f55["positions"]["development"],
        anchorPositions=f55["positions"]["anchor"],
    )
    contract = _contract(workspace)
    if (
        f57.get("f58ContractSha256") != EXPECTED_CONTRACT_SHA256
        or f55.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
    ):
        raise Phase4F58Error("F55/F57 evidence does not authorize F58")
    device = torch.device(device_name)
    cuda = _cuda_preflight(device)
    resources = _resource_preflight(workspace, contract)
    source = _runner_source(workspace)
    train, development, anchor = _load_inputs(workspace, contract)
    settings = contract["training"]
    trust = contract["trustRegion"]
    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline_hours = float(contract["resources"]["baselineGlobalGpuHours"])
    maximum_hours = float(contract["resources"]["maximumGlobalGpuHours"])
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F58Error("a running GPU ledger entry blocks F58")
    if not baseline_hours <= ledger.total_hours <= maximum_hours:
        raise Phase4F58Error("F58 GPU budget envelope is unavailable")
    run_baseline_hours = ledger.total_hours
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    started = time.monotonic()
    status = "interrupted"
    try:
        with RunLock(
            workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id
        ):
            ledger.begin(run_id, external_hours=0.0)
            smoke = _memory_smoke(workspace, contract, train, anchor, device)
            _progress(
                "memory-smoke",
                "completed",
                peakAllocatedBytes=smoke["peakAllocatedBytes"],
                peakReservedBytes=smoke["peakReservedBytes"],
            )
            model = _load_initial_model(workspace).to(device)
            trainable = freeze_policy_head(model)
            initial_state = _cpu_state(model)
            anchor_reference = _reference_anchor_policies(model, anchor, device)
            baseline_values = _baseline_values(model, development, device)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(settings["learningRate"]),
                weight_decay=float(settings["weightDecay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(settings["optimizerSteps"]),
                eta_min=float(settings["minimumLearningRate"]),
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=bool(settings["cudaAmp"])
            )
            teacher_sampler = StratifiedSampler(
                train.bands,
                batch_size=int(settings["batchSize"]),
                mix=settings["batchMix"],
                seed=seed,
            )
            anchor_sampler = StratifiedSampler(
                anchor.bands,
                batch_size=int(settings["anchorPenaltyBatchSize"]),
                mix=settings["batchMix"],
                seed=seed + 1,
            )
            evaluation_steps = set(int(value) for value in settings["evaluationSteps"])
            evaluations: list[dict[str, object]] = []
            projection_records: list[dict[str, object]] = []
            baseline: dict[str, object] | None = None
            best: tuple[float, int, OrderedDict[str, torch.Tensor]] | None = None
            best_observed: tuple[
                float, int, OrderedDict[str, torch.Tensor]
            ] | None = None
            accepted_updates = 0
            projected_updates = 0
            rejected_updates = 0
            consecutive_rejections = 0
            stop_reason = "optimizer-steps-completed"
            timing: list[float] = []
            current_anchor_kl = full_anchor_forward_kl(
                model, anchor, anchor_reference, device
            )
            if current_anchor_kl > float(trust["projectionTargetMaximum"]):
                raise Phase4F58Error("F21 initialization starts outside trust region")
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(int(settings["optimizerSteps"]) + 1):
                if step in evaluation_steps:
                    metrics = evaluate_model(
                        model,
                        development,
                        anchor,
                        anchor_reference,
                        baseline_values,
                        initial_state,
                        device,
                    )
                    if abs(float(metrics["anchorForwardKl"]) - current_anchor_kl) > 1e-7:
                        raise Phase4F58Error("evaluation and trust-region KL disagree")
                    if baseline is None:
                        baseline = metrics
                        checks = {"baseline": True}
                        eligible = False
                    else:
                        checks = _acceptance_checks(
                            baseline,
                            metrics,
                            contract["evaluation"]["acceptance"],
                            accepted_updates,
                        )
                        eligible = all(checks.values())
                    evaluations.append(
                        {
                            "step": step,
                            "acceptedOptimizerUpdates": accepted_updates,
                            "metrics": metrics,
                            "checks": checks,
                            "eligible": eligible,
                        }
                    )
                    score = float(metrics["policyLoss"]["all"])
                    if step > 0 and (
                        best_observed is None or score < best_observed[0]
                    ):
                        best_observed = (score, step, _cpu_state(model))
                    if eligible and (best is None or score < best[0]):
                        best = (score, step, _cpu_state(model))
                    total = ledger.update(
                        run_id,
                        elapsed_seconds=time.monotonic() - started,
                        status="running",
                    )
                    _progress(
                        "evaluation",
                        "completed",
                        step=step,
                        eligible=eligible,
                        acceptedOptimizerUpdates=accepted_updates,
                        policyLoss=metrics["policyLoss"]["all"],
                        earlyPolicyLoss=metrics["policyLoss"]["early"],
                        anchorForwardKl=metrics["anchorForwardKl"],
                        globalGpuHours=total,
                    )
                if step == int(settings["optimizerSteps"]):
                    break

                step_started = time.monotonic()
                before = trainable_state(model)
                teacher_indices, teacher_mirror = teacher_sampler.batch()
                anchor_indices, anchor_mirror = anchor_sampler.batch()
                teacher_batch = _select_batch(
                    train, teacher_indices, teacher_mirror, device
                )
                anchor_batch = _select_batch(
                    anchor, anchor_indices, anchor_mirror, device
                )
                reference = _mixed_anchor_reference(
                    anchor_reference[0],
                    anchor_reference[1],
                    anchor_indices,
                    anchor_mirror,
                    device,
                )
                model.eval()
                with _amp_context(bool(settings["cudaAmp"])):
                    teacher_logits, _ = model(teacher_batch["states"])
                    anchor_logits, _ = model(anchor_batch["states"])
                    teacher_loss = _teacher_loss(teacher_logits.float(), teacher_batch)
                    anchor_loss = _forward_kl(
                        reference, anchor_logits.float(), anchor_batch["legalMasks"]
                    )
                    loss = (
                        float(settings["loss"]["teacherPolicyWeight"])
                        * teacher_loss
                        + float(settings["loss"]["f21PolicyPreservationWeight"])
                        * anchor_loss
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    float(settings["maximumGradientNorm"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                after = trainable_state(model)
                projection = project_trainable_update(
                    model,
                    before,
                    after,
                    lambda: full_anchor_forward_kl(
                        model, anchor, anchor_reference, device
                    ),
                    before_metric=current_anchor_kl,
                    target=float(trust["projectionTargetMaximum"]),
                    hard_maximum=float(trust["hardMaximum"]),
                    bisection_iterations=int(trust["bisectionIterations"]),
                    minimum_scale=float(trust["minimumAcceptedScale"]),
                )
                current_anchor_kl = float(projection["acceptedMetric"])
                if projection["accepted"]:
                    accepted_updates += 1
                    consecutive_rejections = 0
                else:
                    rejected_updates += 1
                    consecutive_rejections += 1
                if projection["projected"]:
                    projected_updates += 1
                record = {
                    "step": step + 1,
                    **projection,
                    "teacherPolicyLoss": float(teacher_loss.item()),
                    "anchorPenaltyBatchForwardKl": float(anchor_loss.item()),
                    "combinedLoss": float(loss.item()),
                    "gradientNorm": float(gradient_norm.item()),
                    "learningRate": float(optimizer.param_groups[0]["lr"]),
                }
                projection_records.append(record)
                timing.append(time.monotonic() - step_started)
                elapsed = time.monotonic() - started
                if run_baseline_hours + elapsed / 3600.0 > maximum_hours:
                    raise Phase4F58Error("F58 pilot GPU budget was exhausted")
                if (step + 1) % 4 == 0:
                    total = ledger.update(
                        run_id, elapsed_seconds=elapsed, status="running"
                    )
                    _progress(
                        "training",
                        "progress",
                        step=step + 1,
                        optimizerSteps=int(settings["optimizerSteps"]),
                        acceptedOptimizerUpdates=accepted_updates,
                        projectedUpdates=projected_updates,
                        rejectedUpdates=rejected_updates,
                        projectionScale=projection["scale"],
                        anchorForwardKl=current_anchor_kl,
                        teacherPolicyLoss=float(teacher_loss.item()),
                        globalGpuHours=total,
                    )
                if consecutive_rejections >= int(
                    trust["maximumConsecutiveRejectedUpdates"]
                ):
                    if not _trust_region_converged(projection_records, trust):
                        raise Phase4F58Error(
                            "F58 rejected updates before reaching the trust-region boundary"
                        )
                    stop_reason = "trust-region-converged"
                    total = ledger.update(
                        run_id, elapsed_seconds=elapsed, status="running"
                    )
                    _progress(
                        "training",
                        "converged",
                        step=step + 1,
                        optimizerSteps=int(settings["optimizerSteps"]),
                        acceptedOptimizerUpdates=accepted_updates,
                        projectedUpdates=projected_updates,
                        rejectedUpdates=rejected_updates,
                        consecutiveRejectedUpdates=consecutive_rejections,
                        anchorForwardKl=current_anchor_kl,
                        globalGpuHours=total,
                    )
                    break

            if baseline is None or best_observed is None:
                raise Phase4F58Error("F58 evaluation schedule did not complete")
            selected = best if best is not None else best_observed
            checkpoint = _write_checkpoint(
                workspace,
                selected[2],
                step=selected[1],
                contract_sha=EXPECTED_CONTRACT_SHA256,
                runner_sha=str(source["sha256"]),
            )
            passed = best is not None
            outcome = (
                "kl-trust-region-policy-head-pilot-passed"
                if passed
                else "kl-trust-region-policy-head-pilot-failed"
            )
            elapsed = time.monotonic() - started
            total = ledger.update(run_id, elapsed_seconds=elapsed, status="completed")
            status = "completed"
            report = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f58-kl-trust-region-policy-head-pilot",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": passed,
                    "contractPath": F58_CONTRACT.as_posix(),
                    "contractSha256": EXPECTED_CONTRACT_SHA256,
                    "runnerSource": source,
                    "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
                    "initialModelStateSha256": EXPECTED_F21_STATE_SHA256,
                    "memorySmoke": smoke,
                    "training": {
                        "optimizerSteps": int(settings["optimizerSteps"]),
                        "optimizerStepsExecuted": len(projection_records),
                        "stopReason": stop_reason,
                        "acceptedOptimizerUpdates": accepted_updates,
                        "projectedUpdates": projected_updates,
                        "rejectedUpdates": rejected_updates,
                        "trainableParameterNames": list(trainable),
                        "selfPlayGames": 0,
                        "replayStoreUsage": False,
                        "valueTarget": None,
                        "meanStepSeconds": mean(timing),
                        "maximumStepSeconds": max(timing),
                    },
                    "trustRegion": {
                        "metric": trust["metric"],
                        "projectionTargetMaximum": trust[
                            "projectionTargetMaximum"
                        ],
                        "hardMaximum": trust["hardMaximum"],
                        "maximumAcceptedMetric": max(
                            float(record["acceptedMetric"])
                            for record in projection_records
                        ),
                        "maximumProposedMetric": max(
                            float(record["proposedMetric"])
                            for record in projection_records
                        ),
                        "records": projection_records,
                    },
                    "evaluations": evaluations,
                    "selectedCheckpoint": checkpoint,
                    "selectedEligible": passed,
                    "cuda": {
                        **cuda,
                        "peakAllocatedBytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peakReservedBytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    },
                    "resources": resources,
                    "elapsedSeconds": elapsed,
                    "baselineGlobalGpuHours": baseline_hours,
                    "runBaselineGlobalGpuHours": run_baseline_hours,
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "reportSha256",
            )
            _write_immutable(workspace / REPORT, report, "F58 pilot report")
            gate = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f58-kl-trust-region-policy-head-pilot-gate",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": passed,
                    "reportPath": REPORT.as_posix(),
                    "reportSha256": report["reportSha256"],
                    "selectedCheckpoint": checkpoint,
                    "optimizerSteps": int(settings["optimizerSteps"]),
                    "optimizerStepsExecuted": len(projection_records),
                    "stopReason": stop_reason,
                    "acceptedOptimizerUpdates": accepted_updates,
                    "projectedUpdates": projected_updates,
                    "rejectedUpdates": rejected_updates,
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "nextStage": {
                        "name": "Phase 4F59 KL trust-region pilot review",
                        "requiresSeparateUserConfirmation": True,
                    },
                },
                "gateSha256",
            )
            _write_immutable(workspace / GATE, gate, "F58 pilot gate")
            return {**gate, "reused": False}
    except BaseException:
        try:
            ledger.update(
                run_id,
                elapsed_seconds=time.monotonic() - started,
                status=status,
            )
        except Exception:
            pass
        raise


def verify_phase4f58(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    authorization = verify_phase4f57(workspace)
    if authorization.get("f58ContractSha256") != EXPECTED_CONTRACT_SHA256:
        raise Phase4F58Error("F57 no longer authorizes the F58 contract")
    report = _mapping(workspace / REPORT, "F58 report")
    report_sha = _verify_self_hash(report, "reportSha256", "F58 report")
    gate = _mapping(workspace / GATE, "F58 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F58 gate")
    checkpoint = gate.get("selectedCheckpoint")
    if not isinstance(checkpoint, Mapping):
        raise Phase4F58Error("F58 selected checkpoint metadata is malformed")
    path = workspace / str(checkpoint["path"])
    if sha256_file(path) != checkpoint.get("fileSha256"):
        raise Phase4F58Error("F58 selected checkpoint bytes changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F58Error("F58 selected checkpoint is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    audit = audit_training_evidence(report, contract)
    expected_outcome = (
        "kl-trust-region-policy-head-pilot-passed"
        if audit["pilotPassed"]
        else "kl-trust-region-policy-head-pilot-failed"
    )
    if (
        not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != checkpoint.get("modelStateSha256")
        or payload.get("format")
        != "quoridor-phase4f58-kl-trust-region-policy-head-pilot-model-only"
        or payload.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or report.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or report.get("runnerSource") != _runner_source(workspace)
        or report.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
        or report.get("initialModelStateSha256") != EXPECTED_F21_STATE_SHA256
        or report.get("selectedCheckpoint") != checkpoint
        or report.get("memorySmoke", {}).get("passed") is not True
        or report.get("memorySmoke", {}).get("projectionCpuSmoke", {}).get("passed")
        is not True
        or report.get("training", {}).get("trainableParameterNames")
        != [
            "policy_head.0.weight",
            "policy_head.1.weight",
            "policy_head.1.bias",
            "policy_head.4.weight",
            "policy_head.4.bias",
        ]
        or report.get("training", {}).get("selfPlayGames") != 0
        or report.get("training", {}).get("replayStoreUsage") is not False
        or report.get("training", {}).get("valueTarget") is not None
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or report.get("outcomeBand") != expected_outcome
        or gate.get("reportSha256") != report_sha
        or gate.get("optimizerSteps") != 64
        or gate.get("optimizerStepsExecuted")
        != report.get("training", {}).get("optimizerStepsExecuted")
        or gate.get("stopReason") != report.get("training", {}).get("stopReason")
        or gate.get("pilotPassed") is not audit["pilotPassed"]
        or gate.get("outcomeBand") != expected_outcome
        or gate.get("acceptedOptimizerUpdates")
        != report.get("training", {}).get("acceptedOptimizerUpdates")
        or gate.get("projectedUpdates")
        != report.get("training", {}).get("projectedUpdates")
        or gate.get("rejectedUpdates")
        != report.get("training", {}).get("rejectedUpdates")
        or gate.get("formalTrainingAuthorized") is not False
        or gate.get("promotionAuthorized") is not False
    ):
        raise Phase4F58Error("F58 pilot artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f58-kl-trust-region-policy-head-pilot-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "selectedCheckpoint": checkpoint,
        "pilotPassed": gate["pilotPassed"],
        "outcomeBand": gate["outcomeBand"],
        "acceptedOptimizerUpdates": gate["acceptedOptimizerUpdates"],
        "optimizerStepsExecuted": gate["optimizerStepsExecuted"],
        "stopReason": gate["stopReason"],
        "projectedUpdates": gate["projectedUpdates"],
        "rejectedUpdates": gate["rejectedUpdates"],
        "trainingAudit": audit,
        "globalGpuHours": gate["globalGpuHours"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f58(args.workspace_root, args.device)
        if args.command == "run"
        else verify_phase4f58(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
