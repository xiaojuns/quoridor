from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch import nn

from quoridor.phase4f58_kl_trust_region_pilot import (
    Phase4F58Error,
    _acceptance_checks,
    _trust_region_converged,
    apply_trainable_state,
    audit_training_evidence,
    interpolate_trainable_state,
    project_trainable_update,
    projection_cpu_smoke,
    trainable_state,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(1, 1, bias=False)
        self.policy_head = nn.Linear(1, 1, bias=False)
        self.frozen.weight.requires_grad_(False)
        with torch.no_grad():
            self.frozen.weight.fill_(7.0)
            self.policy_head.weight.zero_()


def _endpoints(model: TinyModel) -> tuple[OrderedDict, OrderedDict]:
    before = trainable_state(model)
    after = OrderedDict((name, value + 1.0) for name, value in before.items())
    return before, after


def test_projection_cpu_smoke_passes() -> None:
    result = projection_cpu_smoke()
    assert result["passed"] is True
    assert result["acceptedMetric"] <= 0.25


def test_trust_region_rejection_plateau_is_convergence_only_at_boundary() -> None:
    trust = {
        "maximumConsecutiveRejectedUpdates": 8,
        "projectionTargetMaximum": 0.018,
    }
    records = [
        {"accepted": False, "acceptedMetric": 0.0179999} for _ in range(8)
    ]
    assert _trust_region_converged(records, trust) is True

    records[-1]["acceptedMetric"] = 0.01
    assert _trust_region_converged(records, trust) is False
    records[-1]["acceptedMetric"] = 0.0179999
    records[-1]["accepted"] = True
    assert _trust_region_converged(records, trust) is False


def test_projection_bisects_to_feasible_parameter_segment() -> None:
    model = TinyModel()
    before, after = _endpoints(model)

    def metric() -> float:
        return float(model.policy_head.weight.detach().abs().item())

    result = project_trainable_update(
        model,
        before,
        after,
        metric,
        before_metric=0.0,
        target=0.2,
        hard_maximum=0.25,
        bisection_iterations=12,
        minimum_scale=1.0 / 4096.0,
    )
    assert result["accepted"] is True
    assert result["projected"] is True
    assert 0.199 <= result["scale"] <= 0.2
    assert metric() <= 0.2
    assert model.frozen.weight.item() == pytest.approx(7.0)


def test_projection_accepts_proposal_inside_region() -> None:
    model = TinyModel()
    before, after = _endpoints(model)
    after["policy_head.weight"].fill_(0.1)

    def metric() -> float:
        return float(model.policy_head.weight.detach().abs().item())

    result = project_trainable_update(
        model,
        before,
        after,
        metric,
        before_metric=0.0,
        target=0.2,
        hard_maximum=0.25,
        bisection_iterations=12,
        minimum_scale=1.0 / 4096.0,
    )
    assert result["accepted"] is True
    assert result["projected"] is False
    assert result["scale"] == 1.0


def test_projection_rejects_when_minimum_scale_is_not_feasible() -> None:
    model = TinyModel()
    before, after = _endpoints(model)

    def metric() -> float:
        value = float(model.policy_head.weight.detach().abs().item())
        return 0.0 if value == 0.0 else 1.0

    result = project_trainable_update(
        model,
        before,
        after,
        metric,
        before_metric=0.0,
        target=0.2,
        hard_maximum=0.25,
        bisection_iterations=12,
        minimum_scale=1.0 / 4096.0,
    )
    assert result["accepted"] is False
    assert result["scale"] == 0.0
    assert model.policy_head.weight.item() == 0.0


def test_state_application_rejects_wrong_tensor_contract() -> None:
    model = TinyModel()
    with pytest.raises(Phase4F58Error, match="contract"):
        apply_trainable_state(model, {"wrong": torch.zeros(1)})
    with pytest.raises(ValueError, match="scale"):
        interpolate_trainable_state(model, *(_endpoints(model)), 1.1)


def test_acceptance_requires_an_accepted_optimizer_update() -> None:
    baseline = {
        "policyLoss": {"all": 2.0, "early": 2.0, "middle": 1.0, "late": 0.5}
    }
    candidate = {
        "policyLoss": {"all": 1.9, "early": 1.9, "middle": 1.0, "late": 0.5},
        "anchorForwardKl": 0.01,
        "valueOutputMaximumAbsoluteError": 0.0,
        "frozenTensors": {"passed": True, "maximumAbsoluteError": 0.0},
        "policyHeadChanged": True,
    }
    limits = {
        "overallPolicyLossRelativeImprovementMinimum": 0.005,
        "earlyPolicyLossRelativeImprovementMinimum": 0.0,
        "middlePolicyLossRelativeRegressionMaximum": 0.01,
        "latePolicyLossRelativeRegressionMaximum": 0.02,
        "anchorForwardKlMaximum": 0.02,
        "frozenTensorMaximumAbsoluteError": 0.0,
        "valueOutputMaximumAbsoluteError": 0.0,
        "acceptedOptimizerUpdatesMinimum": 1,
    }
    assert _acceptance_checks(baseline, candidate, limits, 0)[
        "acceptedOptimizerUpdates"
    ] is False
    assert all(_acceptance_checks(baseline, candidate, limits, 1).values())


def test_training_evidence_audit_recomputes_selection_and_counts() -> None:
    limits = {
        "overallPolicyLossRelativeImprovementMinimum": 0.005,
        "earlyPolicyLossRelativeImprovementMinimum": 0.0,
        "middlePolicyLossRelativeRegressionMaximum": 0.01,
        "latePolicyLossRelativeRegressionMaximum": 0.02,
        "anchorForwardKlMaximum": 0.02,
        "frozenTensorMaximumAbsoluteError": 0.0,
        "valueOutputMaximumAbsoluteError": 0.0,
        "acceptedOptimizerUpdatesMinimum": 1,
    }
    contract = {
        "training": {"optimizerSteps": 2, "evaluationSteps": [0, 1, 2]},
        "trustRegion": {
            "projectionTargetMaximum": 0.018,
            "hardMaximum": 0.02,
            "minimumAcceptedScale": 1.0 / 4096.0,
        },
        "evaluation": {"acceptance": limits},
    }

    def metrics(loss: float, kl: float, changed: bool) -> dict[str, object]:
        return {
            "policyLoss": {
                "all": loss,
                "early": loss,
                "middle": 1.0,
                "late": 0.5,
            },
            "anchorForwardKl": kl,
            "frozenTensors": {
                "passed": True,
                "maximumAbsoluteError": 0.0,
                "changedTensorNames": [],
            },
            "valueOutputMaximumAbsoluteError": 0.0,
            "policyHeadChanged": changed,
        }

    baseline = metrics(2.0, 0.0, False)
    step1 = metrics(1.98, 0.01, True)
    step2 = metrics(1.90, 0.018, True)
    evaluations = [
        {
            "step": 0,
            "acceptedOptimizerUpdates": 0,
            "metrics": baseline,
            "checks": {"baseline": True},
            "eligible": False,
        },
        {
            "step": 1,
            "acceptedOptimizerUpdates": 1,
            "metrics": step1,
            "checks": _acceptance_checks(baseline, step1, limits, 1),
            "eligible": True,
        },
        {
            "step": 2,
            "acceptedOptimizerUpdates": 2,
            "metrics": step2,
            "checks": _acceptance_checks(baseline, step2, limits, 2),
            "eligible": True,
        },
    ]
    common = {
        "teacherPolicyLoss": 1.0,
        "anchorPenaltyBatchForwardKl": 0.01,
        "combinedLoss": 1.02,
        "gradientNorm": 1.0,
        "learningRate": 1e-5,
    }
    records = [
        {
            "step": 1,
            "accepted": True,
            "projected": False,
            "scale": 1.0,
            "beforeMetric": 0.0,
            "proposedMetric": 0.01,
            "acceptedMetric": 0.01,
            **common,
        },
        {
            "step": 2,
            "accepted": True,
            "projected": True,
            "scale": 0.5,
            "beforeMetric": 0.01,
            "proposedMetric": 0.03,
            "acceptedMetric": 0.018,
            **common,
        },
    ]
    report = {
        "pilotPassed": True,
        "selectedEligible": True,
        "selectedCheckpoint": {"step": 2},
        "evaluations": evaluations,
        "training": {
            "optimizerSteps": 2,
            "acceptedOptimizerUpdates": 2,
            "projectedUpdates": 1,
            "rejectedUpdates": 0,
        },
        "trustRegion": {
            "records": records,
            "maximumAcceptedMetric": 0.018,
            "maximumProposedMetric": 0.03,
        },
    }
    audit = audit_training_evidence(report, contract)
    assert audit["pilotPassed"] is True
    assert audit["selectedStep"] == 2
    assert audit["projectedUpdates"] == 1

    early_contract = deepcopy(contract)
    early_contract["training"] = {
        "optimizerSteps": 10,
        "evaluationSteps": [0, 1, 2, 4, 8, 10],
    }
    early_contract["trustRegion"]["maximumConsecutiveRejectedUpdates"] = 1
    early_report = deepcopy(report)
    early_report["training"].update(
        {
            "optimizerSteps": 10,
            "optimizerStepsExecuted": 2,
            "stopReason": "trust-region-converged",
            "acceptedOptimizerUpdates": 1,
            "projectedUpdates": 1,
            "rejectedUpdates": 1,
        }
    )
    early_report["trustRegion"]["records"][0].update(
        {"proposedMetric": 0.0179999, "acceptedMetric": 0.0179999}
    )
    early_report["trustRegion"]["records"][1].update(
        {
            "accepted": False,
            "scale": 0.0,
            "beforeMetric": 0.0179999,
            "acceptedMetric": 0.0179999,
        }
    )
    early_report["trustRegion"]["maximumAcceptedMetric"] = 0.0179999
    early_report["evaluations"][1]["metrics"]["anchorForwardKl"] = 0.0179999
    early_report["evaluations"][2] = deepcopy(early_report["evaluations"][1])
    early_report["evaluations"][2]["step"] = 2
    early_report["selectedCheckpoint"]["step"] = 1
    early_audit = audit_training_evidence(early_report, early_contract)
    assert early_audit["optimizerStepsExecuted"] == 2
    assert early_audit["stopReason"] == "trust-region-converged"
    assert early_audit["selectedStep"] == 1

    report["selectedCheckpoint"]["step"] = 1
    with pytest.raises(Phase4F58Error, match="aggregate"):
        audit_training_evidence(report, contract)
