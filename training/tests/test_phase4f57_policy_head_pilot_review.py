from __future__ import annotations

import pytest

from quoridor.phase4f57_policy_head_pilot_review import review_evaluations


def _report() -> dict[str, object]:
    rows = []
    for index, step in enumerate((0, 8, 16, 24, 32)):
        factor = 1.0 - 0.04 * index
        metrics = {
            "policyLoss": {
                "all": 4.0 * factor,
                "early": 5.0 * factor,
                "middle": 3.0 * factor,
                "late": 2.0 * factor,
            },
            "anchorForwardKl": (1e-9, 0.06, 0.10, 0.14, 0.17)[index],
            "frozenTensors": {
                "passed": True,
                "maximumAbsoluteError": 0.0,
                "changedTensorNames": [],
            },
            "valueOutputMaximumAbsoluteError": 0.0,
            "policyHeadChanged": index > 0,
        }
        checks = (
            {"baseline": True}
            if index == 0
            else {
                "anchorForwardKl": False,
                "earlyPolicyLoss": True,
                "finiteTrainingAndEvaluation": True,
                "frozenTensors": True,
                "latePolicyLoss": True,
                "middlePolicyLoss": True,
                "overallPolicyLoss": True,
                "policyHeadChanged": True,
                "valueOutput": True,
            }
        )
        rows.append({"step": step, "metrics": metrics, "checks": checks, "eligible": False})
    return {
        "evaluations": rows,
        "training": {"selfPlayGames": 0, "replayStoreUsage": False, "valueTarget": None},
    }


def test_review_identifies_soft_constraint_as_only_failure() -> None:
    review = review_evaluations(_report(), anchor_limit=0.02)
    assert review["passed"] is True
    assert review["diagnosis"]["primaryCause"] == (
        "soft-anchor-penalty-did-not-enforce-the-preservation-boundary"
    )
    assert review["firstObservedAnchorLimitMultiple"] == pytest.approx(3.0)
    assert review["finalRelativePolicyImprovement"]["early"] == pytest.approx(0.16)


def test_review_rejects_non_anchor_failure() -> None:
    report = _report()
    report["evaluations"][1]["checks"]["valueOutput"] = False
    review = review_evaluations(report, anchor_limit=0.02)
    assert review["passed"] is False
    assert review["checks"]["pilotFailedOnlyOnAnchorConstraint"] is False


def test_review_requires_frozen_evaluation_schedule() -> None:
    report = _report()
    report["evaluations"].pop()
    with pytest.raises(Exception, match="schedule"):
        review_evaluations(report, anchor_limit=0.02)
