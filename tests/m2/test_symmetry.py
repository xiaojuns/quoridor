"""Exact spatial/history symmetry properties for the new rule contract."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import pytest
import torch

from quoridor import engine
from quoridor.episode import advance_episode, enforce_deadline, fail_episode, interrupt_episode, start_episode
from quoridor.symmetry import IDENTITY, MIRROR, ROTATE_SWAP, Symmetry, canonical_transform


SYMMETRIES = (IDENTITY, MIRROR, ROTATE_SWAP, Symmetry(mirror=True, rotate_swap=True))


def asymmetric_episode(to_play=0):
    return start_episode(engine.BoardState(
        pawns=((6, 2), (2, 5)),
        horizontal_walls=frozenset({(3, 1), (5, 4)}),
        vertical_walls=frozenset({(1, 6)}),
        walls_remaining=(8, 9),
        to_play=to_play,
    ))


def test_coordinate_transforms_use_different_square_and_wall_extents():
    assert ROTATE_SWAP.square((2, 3)) == (6, 5)
    assert ROTATE_SWAP.wall((2, 3)) == (5, 4)
    assert MIRROR.square((2, 3)) == (2, 5)
    assert MIRROR.wall((2, 3)) == (2, 4)
    both = Symmetry(mirror=True, rotate_swap=True)
    assert both.square((2, 3)) == (6, 3)
    assert both.wall((2, 3)) == (5, 3)


@pytest.mark.parametrize("transform", SYMMETRIES)
def test_all_actions_are_permuted_and_exactly_invertible(transform):
    mapped = [transform.action(action) for action in range(209)]
    assert sorted(mapped) == list(range(209))
    assert [transform.inverse().action(action) for action in mapped] == list(range(209))
    policy = tuple(index / 100 for index in range(209))
    transformed = transform.policy(policy)
    assert transform.inverse().policy(transformed) == policy
    for action in range(209):
        assert transformed[transform.action(action)] == policy[action]
    tensor = torch.arange(209, dtype=torch.float32)
    assert torch.equal(torch.stack(transform.inverse().policy(transform.policy(tensor))), tensor)


@pytest.mark.parametrize("transform", SYMMETRIES)
@pytest.mark.parametrize("to_play", (0, 1))
def test_asymmetric_board_legal_support_and_every_legal_transition_commute(transform, to_play):
    episode = asymmetric_episode(to_play)
    transformed = transform.episode(episode)
    assert transform.inverse().episode(transformed) == episode
    assert transformed.state.to_play == transform.player(to_play)
    assert transformed.state.walls_remaining[transform.player(0)] == 8
    original_mask = engine.legal_action_mask(episode.state)
    transformed_mask = engine.legal_action_mask(transformed.state)
    assert transform.policy(original_mask) == transformed_mask
    policy = tuple((index + 1) if legal else 0 for index, legal in enumerate(original_mask))
    assert all((value > 0) == legal for value, legal in zip(transform.policy(policy), transformed_mask))
    for action in engine.legal_actions(episode.state):
        after = advance_episode(episode, action)
        mapped_after = advance_episode(transformed, transform.action(action))
        assert transform.episode(after) == mapped_after


@settings(max_examples=25, derandomize=True, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=1000), max_size=12), st.booleans())
def test_reachable_histories_keep_exact_keys_legality_and_transitions(choices, second_player):
    episode = asymmetric_episode(int(second_player))
    for choice in choices:
        if episode.finished:
            break
        actions = engine.legal_actions(episode.state)
        action = actions[choice % len(actions)]
        previous = episode
        episode = advance_episode(episode, action)
        for transform in SYMMETRIES:
            transformed = transform.episode(episode)
            assert transform.inverse().episode(transformed) == episode
            assert transformed.history == tuple(
                engine.position_key(transform.state(engine.board_from_position_key(key)))
                for key in episode.history
            )
            assert transformed.repetition_count == episode.repetition_count
            assert transformed.plies == episode.plies
            assert transformed == advance_episode(transform.episode(previous), transform.action(action))
            assert transform.policy(engine.legal_action_mask(episode.state)) == engine.legal_action_mask(transformed.state)


@pytest.mark.parametrize("transform", SYMMETRIES)
@pytest.mark.parametrize("winner", (0, 1))
def test_goal_values_and_recorded_player_targets_keep_current_player_sign(transform, winner):
    before = start_episode(engine.BoardState(pawns=((1, 2), (7, 6)), to_play=winner))
    goal = engine.encode_move(0, 2) if winner == 0 else engine.encode_move(8, 6)
    episode = advance_episode(before, goal)
    transformed = transform.episode(episode)
    assert transformed.outcome.winner == transform.player(winner)
    assert transformed.outcome.value_for(transformed.state.to_play) == -1.0
    recorded_players = (0, 1, 1, 0)
    assert transformed.value_targets(map(transform.player, recorded_players)) == episode.value_targets(recorded_players)
    assert transform.inverse().episode(transformed) == episode


@pytest.mark.parametrize("transform", SYMMETRIES)
def test_repetition_draw_and_unlabelled_stops_survive_transformation(transform):
    episode = start_episode()
    for point in ((8, 3), (0, 3), (8, 4), (0, 4)) * 2:
        episode = advance_episode(episode, engine.encode_move(*point))
    truncated = advance_episode(start_episode(max_game_plies=1), engine.encode_move(7, 4))
    interrupted = interrupt_episode(asymmetric_episode())
    deadline = enforce_deadline(asymmetric_episode(), SimpleNamespace(seconds=1.0, elapsed_seconds=1.25))
    failed = fail_episode(asymmetric_episode(), reason="numerical_error", detail="nonfinite value")
    for original in (episode, truncated, interrupted, deadline, failed):
        transformed = transform.episode(original)
        assert transformed.outcome == original.outcome
        assert transform.inverse().episode(transformed) == original
    assert transform.episode(episode).value_targets([0, 1]) == (0.0, 0.0)
    for stopped in (truncated, interrupted, deadline, failed):
        assert transform.episode(stopped).value_targets([0, 1]) is None


@pytest.mark.parametrize("to_play", (0, 1))
def test_canonicalization_uses_one_transform_for_whole_history(to_play):
    episode = asymmetric_episode(to_play)
    for _ in range(3):
        episode = advance_episode(episode, engine.legal_actions(episode.state)[0])
    transform = canonical_transform(episode.state)
    transformed = transform.episode(episode)
    assert transformed.state.to_play == 0
    assert transformed.history == transform.history(episode.history)
    assert [engine.board_from_position_key(key).to_play for key in transformed.history] == [1, 0, 1, 0]
    assert transform.inverse().history(transformed.history) == episode.history


@pytest.mark.parametrize("values", ([0] * 208, [0] * 210, [float("nan")] * 209, [float("inf")] * 209, ["1"] * 209, [[0]] * 209))
def test_policy_rejects_malformed_inputs(values):
    with pytest.raises(ValueError):
        MIRROR.policy(values)


def test_policy_rejects_batched_tensor():
    with pytest.raises(ValueError, match="dimension"):
        MIRROR.policy(torch.zeros(209, 1))


@pytest.mark.parametrize("transform", SYMMETRIES)
def test_transform_rejects_bad_action_and_legacy_state(transform):
    from quoridor.game import create_initial_state

    with pytest.raises(TypeError, match="BoardState"):
        transform.state(create_initial_state())
    with pytest.raises((ValueError, TypeError)):
        transform.action(True)
    with pytest.raises(ValueError):
        transform.action(209)


def test_symmetry_has_no_ml_import_dependency():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise RuntimeError('ML dependency attempted: ' + fullname)
sys.meta_path.insert(0, BlockML())
from quoridor.symmetry import ROTATE_SWAP
from quoridor.episode import start_episode
assert ROTATE_SWAP.episode(start_episode()).state.to_play == 1
"""
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": str(root / "training")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
