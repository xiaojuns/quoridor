"""Inference integration: exact coordinate/perspective and bounded evaluation."""

from __future__ import annotations

import math

import pytest
import torch

from quoridor import engine
from quoridor.encoder import encode_episode
from quoridor.episode import advance_episode, fail_episode, interrupt_episode, start_episode
from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
from quoridor.project_runtime import BudgetExpired, Deadline
from quoridor.puct import SearchConfig, run_search
from quoridor.search_evaluator import NeuralEvaluator


class ControlledModel(PolicyValueNetwork):
    def __init__(self, *, after_forward=None):
        super().__init__(NetworkConfig.smoke())
        self.register_buffer("controlled_logits", torch.arange(209, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("controlled_values", torch.tensor([0.375]))
        self.after_forward = after_forward
        self.calls = 0
        self.observations = []
        self.eval()

    def forward(self, inputs):
        self.calls += 1
        self.observations.append((inputs.clone(), torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        if self.after_forward is not None:
            self.after_forward()
        return self.controlled_logits, self.controlled_values


def asymmetric_episode(to_play=0):
    return start_episode(engine.BoardState(
        pawns=((6, 2), (2, 5)),
        horizontal_walls=frozenset({(3, 1), (5, 4)}),
        vertical_walls=frozenset({(1, 6)}),
        walls_remaining=(8, 9),
        to_play=to_play,
    ))


@pytest.mark.parametrize("to_play", (0, 1))
@pytest.mark.parametrize("mirror", (False, True))
def test_logits_inverse_transform_all_actions_and_values_keep_current_player_view(to_play, mirror):
    model = ControlledModel()
    episode = asymmetric_episode(to_play)
    result = NeuralEvaluator(model, mirror=mirror)(episode, deadline=Deadline(30))
    # Independent index arithmetic checks squares (extent 8) and anchors
    # (extent 7), not a second call to the adapter's permutation utility.
    for action in range(209):
        offset, width = (0, 9) if action < 81 else ((81, 8) if action < 145 else (145, 8))
        row, col = divmod(action - offset, width)
        if to_play == 1:
            row, col = width - 1 - row, width - 1 - col
        if mirror:
            col = width - 1 - col
        canonical = offset + width * row + col
        assert result.logits[action] == float(canonical)
    assert result.value == 0.375
    inputs, grad_enabled, inference_enabled = model.observations[0]
    torch.testing.assert_close(inputs[0], encode_episode(episode, mirror=mirror).features)
    assert grad_enabled is False and inference_enabled is True


def test_real_cpu_model_inference_preserves_weights_batchnorm_buffers_rng_and_gradients():
    model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    observed = []
    handle = model.register_forward_pre_hook(
        lambda module, args: observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
    )
    try:
        result = NeuralEvaluator(model)(asymmetric_episode(1), deadline=Deadline(30))
    finally:
        handle.remove()
    assert len(result.logits) == 209 and all(math.isfinite(x) for x in result.logits)
    assert -1 <= result.value <= 1
    assert observed == [(False, True)]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("to_play", (0, 1))
@pytest.mark.parametrize("mirror", (False, True))
def test_search_masks_illegal_logits_after_mapping_into_original_coordinates(to_play, mirror):
    episode = asymmetric_episode(to_play)
    legal = engine.legal_actions(episode.state)
    favored = legal[0]
    illegal = next(action for action in range(209) if action not in legal)
    transform = encode_episode(episode, mirror=mirror).transform
    model = ControlledModel()
    model.controlled_logits.zero_()
    model.controlled_logits[0, transform.action(favored)] = 25.0
    model.controlled_logits[0, transform.action(illegal)] = 1000.0
    result = run_search(
        episode, NeuralEvaluator(model, mirror=mirror), SearchConfig(simulations=1),
        deadline=Deadline(30),
    )
    assert result.status == "completed" and result.action == favored
    assert result.completed_simulations == sum(result.visits) == 1
    assert sum(result.policy) == pytest.approx(1.0)
    assert sum(result.priors) == pytest.approx(1.0)
    assert result.visits[favored] == 1 and result.policy[favored] == 1.0
    for action in range(209):
        if action not in legal:
            assert result.visits[action] == result.priors[action] == result.policy[action] == 0


def test_adapter_uses_explicit_model_dtype_and_device_without_fallback():
    model = ControlledModel().double()
    NeuralEvaluator(model)(start_episode(), deadline=Deadline(30))
    inputs = model.observations[0][0]
    assert inputs.dtype == torch.float64 and inputs.device == model.input_conv.weight.device


@pytest.mark.parametrize("mode_change", ("whole_model", "single_batchnorm"))
def test_caller_owns_eval_mode_and_mode_changes_are_rejected(mode_change):
    model = ControlledModel()
    evaluator = NeuralEvaluator(model)
    if mode_change == "whole_model":
        model.train()
    else:
        model.input_bn.train()
    with pytest.raises(ValueError, match="eval mode"):
        evaluator(start_episode(), deadline=Deadline(30))
    assert model.calls == 0
    assert model.input_bn.training
    with pytest.raises(ValueError, match="eval mode"):
        NeuralEvaluator(model)


def test_expired_deadline_avoids_encoding_and_forward(monkeypatch):
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    model = ControlledModel()
    now[0] = 1.0
    def forbidden(*args, **kwargs):
        pytest.fail("encoding must not run after deadline expiration")
    monkeypatch.setattr("quoridor.search_evaluator.encode_episode", forbidden)
    with pytest.raises(BudgetExpired):
        NeuralEvaluator(model)(start_episode(), deadline=deadline)
    assert model.calls == 0


def test_expiration_during_encoding_prevents_forward(monkeypatch):
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    model = ControlledModel()
    def slow_encoding(*args, **kwargs):
        result = encode_episode(*args, **kwargs)
        now[0] = 2.0
        return result
    monkeypatch.setattr("quoridor.search_evaluator.encode_episode", slow_encoding)
    with pytest.raises(BudgetExpired):
        NeuralEvaluator(model)(start_episode(), deadline=deadline)
    assert model.calls == 0


def test_expiration_during_forward_discards_completed_prediction():
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    model = ControlledModel(after_forward=lambda: now.__setitem__(0, 2.0))
    with pytest.raises(BudgetExpired):
        NeuralEvaluator(model)(start_episode(), deadline=deadline)
    assert model.calls == 1


@pytest.mark.parametrize("stop", ("goal", "draw", "ply_limit", "interrupted", "failed"))
def test_finished_episodes_never_enter_network(stop):
    if stop == "goal":
        episode = start_episode(engine.BoardState(pawns=((1, 2), (7, 6))))
        episode = advance_episode(episode, engine.encode_move(0, 2))
    elif stop == "draw":
        episode = start_episode()
        for point in [(8, 3), (0, 3), (8, 4), (0, 4)] * 2:
            episode = advance_episode(episode, engine.encode_move(*point))
    elif stop == "ply_limit":
        episode = advance_episode(start_episode(max_game_plies=1), engine.encode_move(7, 4))
    elif stop == "interrupted":
        episode = interrupt_episode(start_episode())
    else:
        episode = fail_episode(start_episode(), "infrastructure_error", detail="test failure")
    model = ControlledModel()
    with pytest.raises(ValueError, match="finished episodes"):
        NeuralEvaluator(model)(episode, deadline=Deadline(30))
    assert model.calls == 0


@pytest.mark.parametrize("kind", ("nan_legal", "inf_illegal", "nan_value", "high_value", "low_value", "logit_shape", "value_shape", "integer_logits"))
def test_invalid_predictions_fail_without_repair_or_clipping(kind):
    model = ControlledModel()
    if kind == "nan_legal":
        model.controlled_logits[0, engine.encode_move(7, 4)] = float("nan")
    elif kind == "inf_illegal":
        model.controlled_logits[0, engine.encode_move(0, 0)] = float("inf")
    elif kind == "nan_value":
        model.controlled_values[0] = float("nan")
    elif kind == "high_value":
        model.controlled_values[0] = 1.01
    elif kind == "low_value":
        model.controlled_values[0] = -1.01
    elif kind == "logit_shape":
        model.controlled_logits = torch.zeros(209)
    elif kind == "value_shape":
        model.controlled_values = torch.zeros(1, 1)
    else:
        model.controlled_logits = torch.zeros((1, 209), dtype=torch.int64)
    with pytest.raises(ValueError):
        NeuralEvaluator(model)(start_episode(), deadline=Deadline(30))


def test_invalid_adapter_inputs_are_explicit_errors():
    with pytest.raises(TypeError, match="PolicyValueNetwork"):
        NeuralEvaluator(torch.nn.Identity())
    with pytest.raises(TypeError, match="mirror"):
        NeuralEvaluator(ControlledModel(), mirror=1)
    evaluator = NeuralEvaluator(ControlledModel())
    with pytest.raises(TypeError, match="Deadline"):
        evaluator(start_episode(), deadline=None)
    with pytest.raises(TypeError, match="Episode"):
        evaluator(engine.initial_state(), deadline=Deadline(30))
