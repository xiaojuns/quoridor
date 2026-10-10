"""Feature semantics, canonical perspective and explicit partial observability."""

from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import pytest
import torch

from quoridor import engine
from quoridor.encoder import encode_episode
from quoridor.episode import advance_episode, enforce_deadline, fail_episode, interrupt_episode, start_episode
from quoridor.feature_spec import ENCODER_VERSION, INPUT_CHANNELS, PLANE_NAMES, encoder_metadata
from quoridor.symmetry import MIRROR, ROTATE_SWAP


def asymmetric_episode(to_play=0):
    return start_episode(engine.BoardState(
        pawns=((6, 2), (2, 5)),
        horizontal_walls=frozenset({(3, 1), (5, 4)}),
        vertical_walls=frozenset({(1, 6)}),
        walls_remaining=(8, 9),
        to_play=to_play,
    ))


def advance_points(episode, points):
    for point in points:
        episode = advance_episode(episode, engine.encode_move(*point))
    return episode


def test_initial_features_match_versioned_plane_contract():
    episode = start_episode()
    encoded = encode_episode(episode)
    x = encoded.features
    assert ENCODER_VERSION == "az-encoder-v1-partial-16"
    assert INPUT_CHANNELS == len(PLANE_NAMES) == 16
    assert encoder_metadata()["history_observation"] == "partial-current-position-v1"
    assert x.shape == (16, 9, 9) and x.dtype == torch.float32
    assert encoded.legal_mask.shape == (209,) and encoded.legal_mask.dtype == torch.bool
    assert x.device.type == encoded.legal_mask.device.type == "cpu"
    assert torch.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
    assert x[0].sum() == 1 and x[0, 8, 4] == 1
    assert x[1].sum() == 1 and x[1, 0, 4] == 1
    assert x[2:4].sum() == 0 and torch.all(x[4:6] == 1)
    assert x[6].sum() == 3 and x[7:9].sum() == 128
    assert x[9, 0].sum() == 9 and x[9, 1:].sum() == 0
    assert x[10, 8].sum() == 9 and x[10, :8].sum() == 0
    assert torch.all(x[11] == 0)
    torch.testing.assert_close(x[12], torch.full((9, 9), 1 / 3))
    torch.testing.assert_close(x[13:15], torch.full((2, 9, 9), 8 / 80))
    assert torch.all(x[15] == 1)
    assert encoded.legal_mask.tolist() == list(engine.legal_action_mask(episode.state))


@pytest.mark.parametrize("to_play", (0, 1))
@pytest.mark.parametrize("mirror", (False, True))
def test_asymmetric_wall_anchor_planes_inventories_distances_and_legal_support(to_play, mirror):
    episode = asymmetric_episode(to_play)
    encoded = encode_episode(episode, mirror=mirror)
    state = encoded.transform.state(episode.state)
    x = encoded.features
    assert state.to_play == 0
    assert tuple(torch.nonzero(x[0], as_tuple=False)[0].tolist()) == state.pawns[0]
    assert tuple(torch.nonzero(x[1], as_tuple=False)[0].tolist()) == state.pawns[1]
    assert set(map(tuple, torch.nonzero(x[2], as_tuple=False).tolist())) == state.horizontal_walls
    assert set(map(tuple, torch.nonzero(x[3], as_tuple=False).tolist())) == state.vertical_walls
    assert torch.all(x[4] == state.walls_remaining[0] / 10)
    assert torch.all(x[5] == state.walls_remaining[1] / 10)
    assert torch.all(x[13] == engine.shortest_path_distance(state, 0) / 80)
    assert torch.all(x[14] == engine.shortest_path_distance(state, 1) / 80)
    assert torch.equal(x[6].flatten().bool(), encoded.legal_mask[:81])
    assert torch.equal(x[7, :8, :8].flatten().bool(), encoded.legal_mask[81:145])
    assert torch.equal(x[8, :8, :8].flatten().bool(), encoded.legal_mask[145:])
    for index in (2, 3, 7, 8):
        assert x[index, 8].sum() == 0 and x[index, :, 8].sum() == 0
    assert encoded.transform.inverse().policy(encoded.legal_mask.tolist()) == engine.legal_action_mask(episode.state)


@pytest.mark.parametrize("to_play", (0, 1))
def test_player_rotation_swap_has_identical_canonical_inputs_and_unchanged_current_value(to_play):
    episode = asymmetric_episode(to_play)
    encoded = encode_episode(episode)
    swapped = encode_episode(ROTATE_SWAP.episode(episode))
    assert torch.equal(encoded.features, swapped.features)
    assert torch.equal(encoded.legal_mask, swapped.legal_mask)
    goal_start = start_episode(engine.BoardState(pawns=((1, 2), (7, 6)), to_play=to_play))
    target = (0, 2) if to_play == 0 else (8, 6)
    goal = advance_points(goal_start, [target])
    canonical = encode_episode(goal).transform.episode(goal)
    assert canonical.state.to_play == 0
    assert canonical.outcome.value_for(0) == goal.outcome.value_for(goal.state.to_play) == -1.0


def test_mirror_feature_transform_uses_eight_column_wall_anchor_reflection():
    episode = asymmetric_episode()
    original = encode_episode(episode).features
    mirrored = encode_episode(episode, mirror=True).features
    for plane in range(16):
        if plane in (2, 3, 7, 8):
            assert torch.equal(mirrored[plane, :8, :8], original[plane, :8, :8].flip(-1))
            assert mirrored[plane, 8].sum() == 0 and mirrored[plane, :, 8].sum() == 0
        else:
            assert torch.equal(mirrored[plane], original[plane].flip(-1))
    assert torch.equal(mirrored, encode_episode(MIRROR.episode(episode)).features)


def test_history_observation_collision_has_distinct_future_draw_adjudication():
    # Same current board, side, count and ply; earlier intermediate positions
    # differ. Both tensors intentionally collide under the partial v1 input.
    left = advance_points(start_episode(), [(8, 3), (0, 3), (8, 4), (0, 4)])
    right = advance_points(start_episode(), [(8, 5), (0, 5), (8, 4), (0, 4)])
    assert left.state == right.state
    assert left.plies == right.plies == 4
    assert left.repetition_count == right.repetition_count == 2
    assert left.history != right.history and left.adjudication_key != right.adjudication_key
    a, b = encode_episode(left), encode_episode(right)
    assert torch.equal(a.features, b.features) and torch.equal(a.legal_mask, b.legal_mask)
    torch.testing.assert_close(a.features[11], torch.full((9, 9), 4 / 256))
    torch.testing.assert_close(a.features[12], torch.full((9, 9), 2 / 3))
    # Revisit an intermediate left position without returning to the starting
    # position. It is occurrence three only in the left history.
    continuation = [(8, 3), (0, 3), (8, 2), (0, 4), (8, 3)]
    left = advance_points(left, continuation)
    right = advance_points(right, continuation)
    assert left.state == right.state
    assert left.outcome.reason == "repetition" and left.outcome.terminated
    assert left.outcome.value_for(left.state.to_play) == 0
    assert right.outcome is None
    assert not encode_episode(left).legal_mask.any()
    assert encode_episode(right).legal_mask.any()


@pytest.mark.parametrize("kind", ("goal", "repetition", "ply_limit", "interrupted", "time_budget", "failed"))
def test_all_completed_and_unlabelled_stops_have_no_legal_policy_support(kind):
    if kind == "goal":
        episode = advance_points(start_episode(engine.BoardState(pawns=((1, 2), (7, 6)))), [(0, 2)])
    elif kind == "repetition":
        episode = advance_points(start_episode(), [(8, 3), (0, 3), (8, 4), (0, 4)] * 2)
    elif kind == "ply_limit":
        episode = advance_points(start_episode(max_game_plies=1), [(7, 4)])
    elif kind == "interrupted":
        episode = interrupt_episode(start_episode())
    elif kind == "time_budget":
        episode = enforce_deadline(start_episode(), SimpleNamespace(seconds=1.0, elapsed_seconds=1.25))
    else:
        episode = fail_episode(start_episode(), reason="numerical_error", detail="test failure")
    encoded = encode_episode(episode)
    assert episode.finished and not encoded.legal_mask.any()
    assert torch.all(encoded.features[6:9] == 0)
    assert torch.isfinite(encoded.features).all()


def test_encoder_calls_public_engine_apis_for_legality_and_paths(monkeypatch):
    episode = asymmetric_episode()
    calls = []

    def mask(state):
        calls.append(("mask", state))
        return tuple(index == 17 for index in range(209))

    def distance(state, player):
        calls.append(("distance", state, player))
        return 16 + player

    monkeypatch.setattr(engine, "legal_action_mask", mask)
    monkeypatch.setattr(engine, "shortest_path_distance", distance)
    encoded = encode_episode(episode)
    assert encoded.legal_mask.sum() == 1 and encoded.legal_mask[17]
    assert encoded.features[6].flatten()[17] == 1
    assert torch.all(encoded.features[13] == 16 / 80)
    assert torch.all(encoded.features[14] == 17 / 80)
    assert sum(call[0] == "mask" for call in calls) == 1
    assert {call[2] for call in calls if call[0] == "distance"} == {0, 1}


@settings(max_examples=20, derandomize=True, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=1000), max_size=8), st.booleans())
def test_reachable_inputs_are_finite_bounded_and_policy_maps_to_exact_engine_mask(choices, to_play):
    episode = asymmetric_episode(int(to_play))
    for choice in choices:
        if episode.finished:
            break
        legal = engine.legal_actions(episode.state)
        episode = advance_episode(episode, legal[choice % len(legal)])
    for mirror in (False, True):
        encoded = encode_episode(episode, mirror=mirror)
        assert torch.isfinite(encoded.features).all()
        assert ((encoded.features >= 0) & (encoded.features <= 1)).all()
        expected = (False,) * 209 if episode.finished else engine.legal_action_mask(episode.state)
        assert encoded.transform.inverse().policy(encoded.legal_mask.tolist()) == expected
        assert torch.equal(encoded.features, encode_episode(ROTATE_SWAP.episode(episode), mirror=mirror).features)


def test_each_encoding_owns_its_tensor_storage():
    episode = asymmetric_episode()
    first, second = encode_episode(episode), encode_episode(episode)
    first.features.zero_()
    first.legal_mask.zero_()
    assert second.features.sum() > 0 and second.legal_mask.any()
    assert episode.state.pawns == ((6, 2), (2, 5))


def test_encoder_uses_explicit_cpu_even_with_a_non_cpu_default_device():
    with torch.device("meta"):
        encoded = encode_episode(start_episode())
    assert encoded.features.device.type == encoded.legal_mask.device.type == "cpu"
    assert torch.isfinite(encoded.features).all() and encoded.legal_mask.any()


def test_encoder_rejects_legacy_state_or_a_board_without_exact_episode():
    from quoridor.game import create_initial_state

    for value in (create_initial_state(), engine.initial_state(), None):
        with pytest.raises(TypeError, match="Episode"):
            encode_episode(value)
