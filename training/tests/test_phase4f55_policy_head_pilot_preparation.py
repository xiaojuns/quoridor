from __future__ import annotations

import torch
import pytest

from quoridor.game import create_initial_state, legal_action_indices
from quoridor.codec import state_to_serializable
from quoridor.network import NetworkConfig, PolicyValueNetwork
from quoridor.phase4f55_policy_head_pilot_preparation import (
    anchor_audit,
    freeze_policy_head,
    run_isolation_smoke,
    select_anchor_rows,
    tensorize_policy_samples,
)


def _rows(count: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for band in ("early", "middle", "late"):
        for side in (0, 1):
            for index in range(count):
                rows.append(
                    {
                        "ordinal": ordinal,
                        "split": "train",
                        "band": band,
                        "currentPlayer": side,
                        "rank": f"{index:064x}",
                        "positionOrbit": f"{band}-{side}-{index}",
                    }
                )
                ordinal += 1
    return rows


def test_anchor_selection_is_balanced_and_disjoint() -> None:
    counts = {"early": 4, "middle": 4, "late": 2}
    excluded = {"early-0-0", "middle-1-0"}
    selected = select_anchor_rows(_rows(), excluded, counts)
    audit = anchor_audit(selected, excluded, {"late-1-7"})
    assert audit["positions"] == 10
    assert audit["uniqueMirrorOrbits"] == 10
    assert audit["supervisedOrbitOverlap"] == 0
    assert audit["developmentOrbitOverlap"] == 0
    assert audit["byPlyBand"] == counts
    assert audit["byPlyBandAndSide"] == {
        "early/side-0": 2,
        "early/side-1": 2,
        "middle/side-0": 2,
        "middle/side-1": 2,
        "late/side-0": 1,
        "late/side-1": 1,
    }


def _policy_sample() -> dict[str, object]:
    state = create_initial_state()
    actions = legal_action_indices(state)
    probability = 1.0 / len(actions)
    return {
        "state": state_to_serializable(state),
        "policy": [[action, probability] for action in actions],
    }


def test_tensor_adapter_preserves_policy_mass_and_mirror() -> None:
    sample = _policy_sample()
    identity = tensorize_policy_samples([sample])
    mirrored = tensorize_policy_samples([sample], mirrored=True)
    assert identity["states"].shape == (1, 16, 9, 9)
    assert identity["policies"].sum().item() == pytest.approx(1.0)
    assert mirrored["policies"].sum().item() == pytest.approx(1.0)
    assert identity["legalMasks"].sum() == mirrored["legalMasks"].sum()


def test_policy_head_smoke_keeps_frozen_state_and_value_exact() -> None:
    torch.manual_seed(7)
    model = PolicyValueNetwork(
        NetworkConfig(input_channels=16, channels=16, blocks=1, policy_channels=4, value_channels=4, value_hidden=8)
    )
    batch = tensorize_policy_samples([_policy_sample(), _policy_sample()])
    result = run_isolation_smoke(model, batch)
    assert result["passed"] is True
    assert result["valueOutputMaximumAbsoluteDelta"] == 0.0
    assert result["frozenChangedTensorNames"] == []
    assert result["frozenGradientNames"] == []
    assert all(name.startswith("policy_head.") for name in result["changedTensorNames"])


def test_freeze_policy_head_exposes_only_policy_parameters() -> None:
    model = PolicyValueNetwork(NetworkConfig.smoke())
    names = freeze_policy_head(model)
    assert names
    assert all(name.startswith("policy_head.") for name in names)
    assert model.training is False
