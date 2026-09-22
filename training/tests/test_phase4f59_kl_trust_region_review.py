from __future__ import annotations

from collections import OrderedDict

import torch

from quoridor.phase4f59_kl_trust_region_review import (
    policy_head_delta_audit,
    review_f58,
)


def test_policy_head_delta_audit_requires_exact_frozen_tensors() -> None:
    baseline = OrderedDict(
        [("trunk.weight", torch.zeros(2)), ("policy_head.weight", torch.zeros(2))]
    )
    candidate = OrderedDict(
        [("trunk.weight", torch.zeros(2)), ("policy_head.weight", torch.ones(2))]
    )
    assert policy_head_delta_audit(baseline, candidate)["passed"] is True
    candidate["trunk.weight"] = torch.ones(2)
    audit = policy_head_delta_audit(baseline, candidate)
    assert audit["passed"] is False
    assert audit["nonPolicyChangedTensorNames"] == ["trunk.weight"]


def _review_report(regress_early: bool = False) -> tuple[dict[str, object], dict[str, object]]:
    evaluations = []
    for index, step in enumerate((0, 1, 2, 4, 8, 16, 24, 32)):
        factor = 1.0 - 0.01 * index
        losses = {
            "all": 4.0 * factor,
            "early": 5.0 * factor,
            "middle": 3.0 * factor,
            "late": 2.0 * factor,
        }
        if regress_early and step == 32:
            losses["early"] = 5.1
        evaluations.append(
            {
                "step": step,
                "eligible": step >= 2,
                "metrics": {
                    "policyLoss": losses,
                    "frozenTensors": {
                        "passed": True,
                        "maximumAbsoluteError": 0.0,
                        "changedTensorNames": [],
                    },
                    "valueOutputMaximumAbsoluteError": 0.0,
                },
            }
        )
    report = {
        "pilotPassed": True,
        "selectedCheckpoint": {"step": 32},
        "evaluations": evaluations,
    }
    audit = {
        "stopReason": "trust-region-converged",
        "maximumAcceptedMetric": 0.017999,
        "eligibleEvaluationSteps": [2, 4, 8, 16, 24, 32],
    }
    return report, audit


def test_review_accepts_monotonic_policy_transfer() -> None:
    report, audit = _review_report()
    review = review_f58(report, audit)
    assert review["passed"] is True
    assert review["relativePolicyLossImprovements"]["early"] > 0.04


def test_review_rejects_early_policy_regression() -> None:
    report, audit = _review_report(regress_early=True)
    review = review_f58(report, audit)
    assert review["passed"] is False
    assert review["checks"]["allBandLossesStrictlyDecrease"] is False
