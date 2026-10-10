"""Tactical and reachable-state contracts for the new board-only engine."""

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

from hypothesis import given, settings, strategies as st
import pytest

from quoridor import engine


def board(*, horizontal=(), vertical=(), to_play=0, pawns=((8, 4), (0, 4))):
    placed = len(horizontal) + len(vertical)
    return engine.BoardState(
        pawns=pawns,
        horizontal_walls=frozenset(horizontal),
        vertical_walls=frozenset(vertical),
        walls_remaining=(10 - min(10, placed), 10 - max(0, placed - 10)),
        to_play=to_play,
    )


def oracle_has_path(state, player):
    """Independent graph flood-fill using the two explicit edges of each wall."""

    blocked = set()
    for row, col in state.horizontal_walls:
        for c in (col, col + 1):
            blocked.add(frozenset(((row, c), (row + 1, c))))
    for row, col in state.vertical_walls:
        for r in (row, row + 1):
            blocked.add(frozenset(((r, col), (r, col + 1))))
    visited = {state.pawns[player]}
    frontier = [state.pawns[player]]
    while frontier:
        row, col = frontier.pop()
        if row == (0 if player == 0 else 8):
            return True
        for destination in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not all(0 <= value < 9 for value in destination):
                continue
            if destination in visited or frozenset(((row, col), destination)) in blocked:
                continue
            visited.add(destination)
            frontier.append(destination)
    return False


def test_all_209_actions_roundtrip_and_have_exact_fixed_boundaries():
    assert engine.ACTION_SPACE_SIZE == 209
    for action in range(209):
        assert engine.encode_action(engine.decode_action(action)) == action
    assert [engine.encode_move(0, 0), engine.encode_move(8, 8)] == [0, 80]
    assert [engine.encode_wall("h", 0, 0), engine.encode_wall("h", 7, 7)] == [81, 144]
    assert [engine.encode_wall("v", 0, 0), engine.encode_wall("v", 7, 7)] == [145, 208]


@pytest.mark.parametrize("action", [-1, 209, True, False, 1.0, "1", None])
def test_action_indices_reject_invalid_types_and_bounds(action):
    with pytest.raises((TypeError, ValueError)):
        engine.decode_action(action)
    with pytest.raises((TypeError, ValueError)):
        engine.apply_action(engine.initial_state(), action)


@pytest.mark.parametrize("action", [
    {}, {"type": "move", "row": 9, "col": 0},
    {"type": "move", "row": True, "col": 0},
    {"type": "move", "row": 0, "col": 0, "extra": 1},
    {"type": "wall", "orientation": "x", "row": 0, "col": 0},
    {"type": "wall", "orientation": "h", "row": 0, "col": 8},
])
def test_action_mappings_reject_malformed_records(action):
    with pytest.raises((TypeError, ValueError)):
        engine.encode_action(action)


def test_initial_legal_actions_and_position_key():
    state = engine.initial_state()
    assert engine.legal_actions(state) == (67, 75, 77, *range(81, 209))
    assert len(engine.legal_action_mask(state)) == 209
    assert sum(engine.legal_action_mask(state)) == 131
    payload = json.loads(engine.position_key(state))
    assert payload == ["quoridor-v1", 0, [[8, 4], [0, 4]], [10, 10], [], []]
    assert engine.position_key(state) != engine.position_key(replace(state, to_play=1))
    assert not hasattr(state, "position_counts")
    assert not hasattr(state, "move_number")
    assert not hasattr(state, "winner")


def test_position_keys_roundtrip_without_silently_normalizing_records():
    for state in (engine.initial_state(), board(horizontal=((3, 3),), vertical=((5, 5),), to_play=1)):
        key = engine.position_key(state)
        assert engine.board_from_position_key(key) == state
        assert engine.position_key(engine.board_from_position_key(key)) == key
        with pytest.raises(ValueError, match="canonical"):
            engine.board_from_position_key(json.dumps(json.loads(key), indent=2))
    wall_key = engine.position_key(board(horizontal=((3, 3),)))
    repeated = json.loads(wall_key)
    repeated[4].append([3, 3])
    with pytest.raises(ValueError, match="canonical"):
        engine.board_from_position_key(json.dumps(repeated, separators=(",", ":")))


@pytest.mark.parametrize("key", [
    None, [], 12, "", "{}", "[1,0,8,4,0,4,10,10,[],[]]",
    '["quoridor-v1",0,[[8,4],[0,4]],[10,10],[],[],"extra"]',
    '["quoridor-v1",false,[[8,4],[0,4]],[10,10],[],[]]',
    '["quoridor-v1",0,[[8,4],[0,4]],[10,10],[[3,3]],[]]',
    '["quoridor-v1",0,[[8,4],[0,4]],[10,10],[{}],[]]',
])
def test_position_keys_reject_legacy_or_invalid_payloads(key):
    with pytest.raises((TypeError, ValueError)):
        engine.board_from_position_key(key)


@pytest.mark.parametrize("opponent,beyond,diagonals,horizontal,vertical", [
    ((3, 4), (2, 4), ((3, 3), (3, 5)), ((2, 4),), ()),
    ((5, 4), (6, 4), ((5, 3), (5, 5)), ((5, 4),), ()),
    ((4, 3), (4, 2), ((3, 3), (5, 3)), (), ((4, 2),)),
    ((4, 5), (4, 6), ((3, 5), (5, 5)), (), ((4, 5),)),
])
def test_straight_jumps_and_blocked_jump_diagonals_all_directions(opponent, beyond, diagonals, horizontal, vertical):
    clear = board(pawns=((4, 4), opponent))
    clear_moves = engine.legal_pawn_moves(clear)
    assert beyond in clear_moves
    assert opponent not in clear_moves
    assert not set(diagonals) & set(clear_moves)
    blocked = board(pawns=clear.pawns, horizontal=horizontal, vertical=vertical)
    blocked_moves = engine.legal_pawn_moves(blocked)
    assert beyond not in blocked_moves
    assert set(diagonals) <= set(blocked_moves)
    assert opponent not in blocked_moves


@pytest.mark.parametrize("pawns,to_play,diagonals", [
    (((1, 4), (0, 4)), 0, {(0, 3), (0, 5)}),
    (((8, 4), (7, 4)), 1, {(8, 3), (8, 5)}),
    (((4, 1), (4, 0)), 0, {(3, 0), (5, 0)}),
    (((4, 7), (4, 8)), 0, {(3, 8), (5, 8)}),
])
def test_board_edges_enable_diagonal_alternatives(pawns, to_play, diagonals):
    state = board(pawns=pawns, to_play=to_play)
    assert diagonals <= set(engine.legal_pawn_moves(state))


def test_wall_between_pawns_prevents_jumps_and_side_wall_blocks_one_diagonal():
    between = board(pawns=((4, 4), (3, 4)), horizontal=((3, 4),))
    assert not {(3, 4), (2, 4), (3, 3), (3, 5)} & set(engine.legal_pawn_moves(between))
    blocked_side = board(pawns=((4, 4), (3, 4)), horizontal=((2, 4),), vertical=((3, 3),))
    assert (3, 3) not in engine.legal_pawn_moves(blocked_side)
    assert (3, 5) in engine.legal_pawn_moves(blocked_side)


def test_walls_block_exactly_two_edges_in_both_directions():
    state = board(horizontal=((3, 3),), vertical=((5, 5),))
    for start, end in (((3, 3), (4, 3)), ((3, 4), (4, 4)), ((5, 5), (5, 6)), ((6, 5), (6, 6))):
        assert engine.is_passage_blocked(state, start, end)
        assert engine.is_passage_blocked(state, end, start)
    assert not engine.is_passage_blocked(state, (3, 5), (4, 5))
    assert not engine.is_passage_blocked(state, (7, 5), (7, 6))
    assert engine.is_passage_blocked(state, (8, 8), (9, 8))
    assert engine.is_passage_blocked(state, (4, 4), (5, 5))


@pytest.mark.parametrize("orientation,first,overlap,adjacent", [
    ("h", (3, 3), (3, 4), (3, 5)),
    ("v", (3, 3), (4, 3), (5, 3)),
])
def test_wall_overlap_crossing_and_end_to_end_adjacency(orientation, first, overlap, adjacent):
    state = board(**{"horizontal" if orientation == "h" else "vertical": (first,)})
    assert not engine.is_legal_wall(state, orientation, *first)
    assert not engine.is_legal_wall(state, orientation, *overlap)
    assert not engine.is_legal_wall(state, "v" if orientation == "h" else "h", *first)
    assert engine.is_legal_wall(state, orientation, *adjacent)
    for action in (engine.encode_wall(orientation, *first), engine.encode_wall(orientation, *overlap)):
        with pytest.raises(ValueError, match="illegal wall"):
            engine.apply_action(state, action)


@pytest.mark.parametrize("row", [0, 7])
def test_structurally_valid_wall_cannot_seal_either_players_path(row):
    state = board(horizontal=((row, 3),), vertical=((row, 2),))
    assert engine.is_wall_structurally_valid(state, "v", row, 4)
    assert not engine.is_legal_wall(state, "v", row, 4)
    assert engine.encode_wall("v", row, 4) not in engine.legal_actions(state)
    with pytest.raises(ValueError, match="illegal wall"):
        engine.apply_action(state, engine.encode_wall("v", row, 4))


def test_depleted_inventory_disables_walls_only():
    walls = ((0, 0), (0, 2), (0, 4), (0, 6), (2, 0), (2, 2), (2, 4), (2, 6), (4, 0), (4, 2))
    state = board(horizontal=walls)
    assert state.walls_remaining == (0, 10)
    assert engine.legal_actions(state)
    assert all(action < 81 for action in engine.legal_actions(state))


def test_path_existence_ignores_pawn_occupancy():
    state = board(pawns=((4, 4), (3, 4)))
    assert engine.shortest_path_distance(state, 0) == 4
    assert engine.shortest_path_distance(state, 1) == 5


@pytest.mark.parametrize("changes", [
    {"to_play": True}, {"to_play": 2},
    {"pawns": [[8, 4], [0, 4]]}, {"pawns": ((8, True), (0, 4))},
    {"pawns": ((4, 4), (4, 4))}, {"pawns": ((9, 4), (0, 4))},
    {"pawns": ((0, 3), (8, 4))},
    {"pawns": ((0, 3), (0, 4)), "to_play": 0},
    {"walls_remaining": [10, 10]}, {"walls_remaining": (True, 10)},
    {"walls_remaining": (11, 9)}, {"walls_remaining": (-1, 10)},
    {"walls_remaining": (9, 10)},
    {"horizontal_walls": {(3, 3)}},
    {"horizontal_walls": frozenset({(8, 0)}), "walls_remaining": (9, 10)},
    {"horizontal_walls": frozenset({(False, 0)}), "walls_remaining": (9, 10)},
    {"horizontal_walls": frozenset({(3, 3), (3, 4)}), "walls_remaining": (8, 10)},
    {"vertical_walls": frozenset({(3, 3), (4, 3)}), "walls_remaining": (8, 10)},
    {"horizontal_walls": frozenset({(3, 3)}), "vertical_walls": frozenset({(3, 3)}), "walls_remaining": (8, 10)},
    {"horizontal_walls": frozenset({(7, 3)}), "vertical_walls": frozenset({(7, 2), (7, 4)}), "walls_remaining": (7, 10)},
])
def test_constructor_rejects_invalid_or_mutable_state(changes):
    with pytest.raises((TypeError, ValueError)):
        engine.BoardState(**changes)


def test_input_is_immutable_and_every_initial_action_matches_legal_mask():
    state = engine.initial_state()
    before = engine.position_key(state)
    original_hash = hash(state)
    mask = engine.legal_action_mask(state)
    for action, legal in enumerate(mask):
        if legal:
            child = engine.apply_action(state, action)
            assert child.to_play == 1
            assert child != state
        else:
            with pytest.raises(ValueError):
                engine.apply_action(state, action)
        assert engine.position_key(state) == before
        assert hash(state) == original_hash
    with pytest.raises(FrozenInstanceError):
        state.to_play = 1
    with pytest.raises(TypeError):
        state.pawns[0] = (7, 4)
    with pytest.raises(AttributeError):
        state.horizontal_walls.add((3, 3))


def test_goal_switches_turn_and_disables_all_actions():
    state = board(pawns=((1, 3), (0, 4)))
    result = engine.apply_action(state, engine.encode_move(0, 3))
    assert engine.goal_winner(result) == 0
    assert result.to_play == 1
    assert not engine.legal_actions(result)
    assert not any(engine.legal_action_mask(result))
    with pytest.raises(ValueError, match="after a goal"):
        engine.apply_action(result, engine.encode_move(1, 4))


@given(st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=18))
@settings(max_examples=25, deadline=None, derandomize=True)
def test_reachable_state_invariants_and_transition_mask_agreement(choices):
    state = engine.initial_state()
    for choice in choices:
        legal = engine.legal_actions(state)
        if not legal:
            assert engine.goal_winner(state) is not None
            break
        old_key = engine.position_key(state)
        old_hash = hash(state)
        child = engine.apply_action(state, legal[choice % len(legal)])
        engine.validate_state(child)
        assert child.to_play == 1 - state.to_play
        assert child.pawns[0] != child.pawns[1]
        assert sum(child.walls_remaining) + len(child.horizontal_walls) + len(child.vertical_walls) == 20
        assert all(oracle_has_path(child, player) for player in (0, 1))
        assert engine.position_key(state) == old_key and hash(state) == old_hash
        assert engine.board_from_position_key(engine.position_key(child)) == child
        state = child
    mask = engine.legal_action_mask(state)
    assert tuple(action for action, allowed in enumerate(mask) if allowed) == engine.legal_actions(state)
    for action, allowed in enumerate(mask):
        if allowed:
            engine.validate_state(engine.apply_action(state, action))
        else:
            with pytest.raises(ValueError):
                engine.apply_action(state, action)


def test_engine_imports_and_plays_without_neural_or_clock_dependencies():
    root = Path(__file__).resolve().parents[2]
    script = """
import builtins
import sys
original_import = builtins.__import__
forbidden = {'torch', 'numpy', 'time', 'datetime', 'quoridor.network',
             'quoridor.project_runtime', 'quoridor.episode'}
def guarded_import(name, *args, **kwargs):
    if name in forbidden or any(name.startswith(prefix + '.') for prefix in forbidden):
        raise AssertionError('forbidden engine dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from quoridor import engine
state = engine.initial_state()
state = engine.apply_action(state, engine.encode_move(7, 4))
assert state.to_play == 1
assert engine.legal_actions(state)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "training")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
