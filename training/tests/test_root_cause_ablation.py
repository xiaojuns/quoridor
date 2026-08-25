from __future__ import annotations

from quoridor.game import create_initial_state
from quoridor.root_cause_ablation import ComponentEvaluator, _band


class _Learned:
    def evaluate_batch(self, states):
        return [([float(index) for index in range(209)], 0.75) for _ in states]

    def enable_native_encoder(self, module):
        self.module = module


def test_component_evaluator_isolates_policy_and_value() -> None:
    state = create_initial_state()
    policy_only = ComponentEvaluator(
        _Learned(), learned_policy=True, learned_value=False
    )
    value_only = ComponentEvaluator(
        _Learned(), learned_policy=False, learned_value=True
    )
    zero = ComponentEvaluator(_Learned(), learned_policy=False, learned_value=False)

    policy, value = policy_only.evaluate(state)
    assert policy[208] == 208.0
    assert value == 0.0
    policy, value = value_only.evaluate(state)
    assert tuple(policy) == (0.0,) * 209
    assert value == 0.75
    policy, value = zero.evaluate(state)
    assert tuple(policy) == (0.0,) * 209
    assert value == 0.0
    assert zero.calls == 1
    assert zero.positions == 1


def test_ply_bands_match_fixed_health_thresholds() -> None:
    assert _band(0, 30) == "early"
    assert _band(15, 80) == "early"
    assert _band(16, 30) == "middle"
    assert _band(39, 80) == "middle"
    assert _band(40, 80) == "late"
