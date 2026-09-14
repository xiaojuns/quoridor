from __future__ import annotations

import torch

from quoridor.codec import state_to_serializable
from quoridor.game import create_initial_state
from quoridor.phase4f56_policy_head_pilot import (
    StratifiedSampler,
    acceptance_checks,
    tensorize_anchor_samples,
)


def test_stratified_sampler_uses_frozen_mix() -> None:
    bands = ("early",) * 16 + ("middle",) * 12 + ("late",) * 4
    sampler = StratifiedSampler(
        bands,
        batch_size=16,
        mix={"early": 0.5, "middle": 0.375, "late": 0.125},
        seed=9,
    )
    indices, mirrored = sampler.batch()
    selected = [bands[index] for index in indices]
    assert selected.count("early") == 8
    assert selected.count("middle") == 6
    assert selected.count("late") == 2
    assert len(mirrored) == 16


def test_anchor_tensor_adapter_has_legal_mask() -> None:
    state = create_initial_state()
    sample = {"state": state_to_serializable(state), "band": "early"}
    identity = tensorize_anchor_samples([sample])
    mirrored = tensorize_anchor_samples([sample], mirrored=True)
    assert identity["states"].shape == (1, 16, 9, 9)
    assert identity["legalMasks"].dtype == torch.bool
    assert identity["legalMasks"].sum() == mirrored["legalMasks"].sum()


def test_acceptance_checks_require_policy_gain_and_invariants() -> None:
    baseline = {
        "policyLoss": {"all": 2.0, "early": 2.0, "middle": 1.0, "late": 0.5}
    }
    candidate = {
        "policyLoss": {"all": 1.9, "early": 1.9, "middle": 1.005, "late": 0.505},
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
    }
    assert all(acceptance_checks(baseline, candidate, limits).values())
    candidate["valueOutputMaximumAbsoluteError"] = 1e-7
    checks = acceptance_checks(baseline, candidate, limits)
    assert checks["valueOutput"] is False
