from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from quoridor.phase4f61_value_scale_screen import (
    ValueScaledEvaluator,
    _identities,
    arena_contract_sha256,
    outcome_band,
)


class _FakeEvaluator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.encoder = None

    def enable_native_encoder(self, module: object) -> None:
        self.encoder = module

    def evaluate_batch(self, states: object) -> list[tuple[list[float], float]]:
        return [([float(index), 7.0], value) for index, value in enumerate(states)]


def _contract() -> dict[str, object]:
    return {
        "contractSha256": "a" * 64,
        "source": {"sha256": "b" * 64},
        "model": {"fileSha256": "c" * 64},
        "candidate": {
            "selectorId": "scaled",
            "runtime": {"runtimeSha256": "d" * 64},
        },
        "opponent": {
            "selectorId": "baseline",
            "runtime": {"runtimeSha256": "e" * 64},
        },
        "workload": {"workloadSha256": "f" * 64},
        "openingSuite": {"suiteHash": "sha256:" + "1" * 64},
    }


def test_value_scaled_evaluator_preserves_policy_and_scales_only_value() -> None:
    base = _FakeEvaluator()
    evaluator = ValueScaledEvaluator(base, 0.5)  # type: ignore[arg-type]
    first, second = evaluator.evaluate_batch((0.8, -0.4))
    assert first == ([0.0, 7.0], pytest.approx(0.4))
    assert second == ([1.0, 7.0], pytest.approx(-0.2))
    marker = SimpleNamespace()
    evaluator.enable_native_encoder(marker)
    assert base.encoder is marker
    assert evaluator.device == torch.device("cpu")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), True])
def test_value_scaled_evaluator_rejects_invalid_scale(value: object) -> None:
    with pytest.raises(ValueError):
        ValueScaledEvaluator(_FakeEvaluator(), value)  # type: ignore[arg-type]


def test_selector_identities_share_model_but_bind_distinct_runtimes() -> None:
    candidate, opponent = _identities(_contract())
    assert candidate.artifact_hash == opponent.artifact_hash == "c" * 64
    assert candidate.config_hash == "d" * 64
    assert opponent.config_hash == "e" * 64
    assert candidate.config_hash != opponent.config_hash
    assert candidate.source_hash == opponent.source_hash == "b" * 64


def test_arena_contract_binds_both_value_scaled_runtimes() -> None:
    contract = _contract()
    original = arena_contract_sha256(contract)
    contract["candidate"]["runtime"]["runtimeSha256"] = "2" * 64
    assert arena_contract_sha256(contract) != original


def test_outcome_requires_more_than_half_the_games() -> None:
    assert outcome_band(0.50) == "value-scale-not-supported"
    assert outcome_band(0.500001) == "value-scale-screen-positive"
