"""Board-only reference rules for ``quoridor-v1``.

This module deliberately has no episode, neural, legacy-engine, or clock
dependency. A board ends only at a goal; repetition and operational limits
belong to :mod:`quoridor.episode`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from math import inf


RULES_VERSION = "quoridor-v1"
CODEC_VERSION = "az209-v1"
BOARD_SIZE = 9
WALL_GRID_SIZE = 8
STARTING_WALLS = 10
ACTION_SPACE_SIZE = 209
MOVE_ACTION_COUNT = 81
HORIZONTAL_WALL_ACTION_OFFSET = 81
VERTICAL_WALL_ACTION_OFFSET = 145
WALL_ACTION_COUNT = 64
Position = tuple[int, int]
Wall = tuple[int, int]
PositionKey = str
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _player(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("player must be integer 0 or 1")
    return value


def _coordinate(value: object, size: int, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value < size:
        raise ValueError(f"{label} must be in [0, {size - 1}]")
    return value


def _position(value: object, size: int, label: str) -> Position:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be a tuple of two integers")
    return (
        _coordinate(value[0], size, f"{label} row"),
        _coordinate(value[1], size, f"{label} col"),
    )


@dataclass(frozen=True, slots=True)
class BoardState:
    """A validated immutable board, with no history or outcome metadata.

    Containers must already be tuples/frozensets; mutable containers are rejected
    rather than retained through a frozen dataclass. Inventories plus placed
    walls must total twenty. Construction validates structure and goal paths;
    this is board validity, not a claim that an arbitrary board is reachable.
    """

    pawns: tuple[Position, Position] = ((8, 4), (0, 4))
    walls_remaining: tuple[int, int] = (STARTING_WALLS, STARTING_WALLS)
    horizontal_walls: frozenset[Wall] = frozenset()
    vertical_walls: frozenset[Wall] = frozenset()
    to_play: int = 0

    def __post_init__(self) -> None:
        validate_state(self)


def _require_board(state: object) -> BoardState:
    if type(state) is not BoardState:
        raise TypeError("state must be a quoridor-v1 BoardState")
    return state


def validate_state(state: BoardState) -> None:
    """Reject malformed or structurally impossible boards without mutation."""

    _require_board(state)
    _player(state.to_play)
    if type(state.pawns) is not tuple or len(state.pawns) != 2:
        raise TypeError("pawns must be a tuple of two positions")
    for pawn in state.pawns:
        _position(pawn, BOARD_SIZE, "pawn")
    if state.pawns[0] == state.pawns[1]:
        raise ValueError("pawns cannot occupy the same square")
    if type(state.walls_remaining) is not tuple or len(state.walls_remaining) != 2:
        raise TypeError("walls_remaining must be a tuple of two integers")
    for count in state.walls_remaining:
        _coordinate(count, STARTING_WALLS + 1, "wall inventory")
    for orientation, walls in (("h", state.horizontal_walls), ("v", state.vertical_walls)):
        if type(walls) is not frozenset:
            raise TypeError("wall collections must be frozensets")
        for wall in walls:
            row, col = _position(wall, WALL_GRID_SIZE, "wall")
            neighbor = (row, col + 1) if orientation == "h" else (row + 1, col)
            if neighbor in walls:
                raise ValueError("parallel walls cannot overlap")
    if state.horizontal_walls & state.vertical_walls:
        raise ValueError("walls cannot cross")
    placed = len(state.horizontal_walls) + len(state.vertical_walls)
    if sum(state.walls_remaining) + placed != 2 * STARTING_WALLS:
        raise ValueError("wall inventories and placed walls must total twenty")
    goals = tuple(player for player in (0, 1) if state.pawns[player][0] == (0 if player == 0 else 8))
    if len(goals) > 1:
        raise ValueError("both players cannot already have reached their goal")
    if goals and goals[0] == state.to_play:
        raise ValueError("side to play must switch after a goal move")
    if not all(has_path_to_goal(state, player) for player in (0, 1)):
        raise ValueError("each player must retain a wall-only path to its goal")


def initial_state(to_play: int = 0) -> BoardState:
    return BoardState(to_play=to_play)


def position_key(state: BoardState) -> PositionKey:
    """Stable full-position key; episode history and ply count are excluded."""

    _require_board(state)
    return json.dumps(
        [RULES_VERSION, state.to_play, state.pawns, state.walls_remaining,
         sorted(state.horizontal_walls), sorted(state.vertical_walls)],
        separators=(",", ":"),
    )


def board_from_position_key(key: PositionKey) -> BoardState:
    """Restore a canonical new-version board key, rejecting legacy/lossy keys.

    This restores one board only. Episode histories must additionally establish
    their legal transitions; a sequence of valid keys is not proof of gameplay.
    """

    if type(key) is not str:
        raise TypeError("position key must be a string")
    return _board_from_position_key(key)


@lru_cache(maxsize=4096)
def _board_from_position_key(key: PositionKey) -> BoardState:
    try:
        payload = json.loads(key)
    except (ValueError, RecursionError) as error:
        raise ValueError("position key must be valid JSON") from error
    if type(payload) is not list or len(payload) != 6 or payload[0] != RULES_VERSION:
        raise ValueError("position key must use the quoridor-v1 structure")
    _, to_play, pawns, remaining, horizontal, vertical = payload
    if (type(pawns) is not list or len(pawns) != 2
            or type(remaining) is not list or len(remaining) != 2
            or type(horizontal) is not list or type(vertical) is not list):
        raise ValueError("position key has invalid board arrays")
    if any(type(point) is not list or len(point) != 2 for point in (*pawns, *horizontal, *vertical)):
        raise ValueError("position key coordinates must be two-entry arrays")
    try:
        state = BoardState(
            pawns=tuple(tuple(pawn) for pawn in pawns),
            walls_remaining=tuple(remaining),
            horizontal_walls=frozenset(tuple(wall) for wall in horizontal),
            vertical_walls=frozenset(tuple(wall) for wall in vertical),
            to_play=to_play,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"position key contains an invalid board: {error}") from error
    if position_key(state) != key:
        raise ValueError("position key must use its exact canonical encoding")
    return state


def encode_move(row: int, col: int) -> int:
    return _coordinate(row, BOARD_SIZE, "move row") * BOARD_SIZE + _coordinate(col, BOARD_SIZE, "move col")


def encode_wall(orientation: str, row: int, col: int) -> int:
    if orientation not in ("h", "v"):
        raise ValueError("wall orientation must be h or v")
    offset = HORIZONTAL_WALL_ACTION_OFFSET if orientation == "h" else VERTICAL_WALL_ACTION_OFFSET
    return offset + _coordinate(row, WALL_GRID_SIZE, "wall row") * WALL_GRID_SIZE + _coordinate(col, WALL_GRID_SIZE, "wall col")


def decode_action(index: int) -> dict[str, object]:
    index = _coordinate(index, ACTION_SPACE_SIZE, "action index")
    if index < MOVE_ACTION_COUNT:
        return {"type": "move", "row": index // BOARD_SIZE, "col": index % BOARD_SIZE}
    orientation = "h" if index < VERTICAL_WALL_ACTION_OFFSET else "v"
    offset = HORIZONTAL_WALL_ACTION_OFFSET if orientation == "h" else VERTICAL_WALL_ACTION_OFFSET
    anchor = index - offset
    return {"type": "wall", "orientation": orientation, "row": anchor // WALL_GRID_SIZE, "col": anchor % WALL_GRID_SIZE}


def encode_action(action: Mapping[str, object]) -> int:
    if not isinstance(action, Mapping):
        raise TypeError("action must be a mapping")
    if action.get("type") == "move" and set(action) == {"type", "row", "col"}:
        return encode_move(action["row"], action["col"])
    if action.get("type") == "wall" and set(action) == {"type", "orientation", "row", "col"}:
        return encode_wall(action["orientation"], action["row"], action["col"])
    raise ValueError("action must contain exactly the move or wall fields")


def _blocked(horizontal: frozenset[Wall], vertical: frozenset[Wall], start: Position, end: Position) -> bool:
    if not all(0 <= x < BOARD_SIZE for x in (*start, *end)):
        return True
    if abs(start[0] - end[0]) + abs(start[1] - end[1]) != 1:
        return True
    if start[0] != end[0]:
        row = min(start[0], end[0])
        return (row, start[1]) in horizontal or (row, start[1] - 1) in horizontal
    col = min(start[1], end[1])
    return (start[0], col) in vertical or (start[0] - 1, col) in vertical


def is_passage_blocked(state: BoardState, start: Position, end: Position) -> bool:
    _require_board(state)
    # Outside-board steps are blocked; malformed coordinates are rejected.
    for point in (start, end):
        if type(point) is not tuple or len(point) != 2 or any(type(x) is not int for x in point):
            raise TypeError("passage endpoints must be integer coordinate tuples")
    return _blocked(state.horizontal_walls, state.vertical_walls, start, end)


def _distance(start: Position, goal: int, horizontal: frozenset[Wall], vertical: frozenset[Wall]) -> float:
    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        cell, distance = frontier.popleft()
        if cell[0] == goal:
            return float(distance)
        for dr, dc in _DIRECTIONS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor not in visited and not _blocked(horizontal, vertical, cell, neighbor):
                visited.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return inf


def shortest_path_distance(state: BoardState, player: int) -> float:
    """Distance using walls only; the other pawn cannot seal a path."""

    _require_board(state)
    _player(player)
    return _distance(state.pawns[player], 0 if player == 0 else 8, state.horizontal_walls, state.vertical_walls)


def has_path_to_goal(state: BoardState, player: int) -> bool:
    return shortest_path_distance(state, player) != inf


def goal_winner(state: BoardState) -> int | None:
    _require_board(state)
    for player in (0, 1):
        if state.pawns[player][0] == (0 if player == 0 else 8):
            return player
    return None


def legal_pawn_moves(state: BoardState, player: int | None = None) -> tuple[Position, ...]:
    _require_board(state)
    player = state.to_play if player is None else _player(player)
    if goal_winner(state) is not None:
        return ()
    pawn, opponent = state.pawns[player], state.pawns[1 - player]
    moves: set[Position] = set()
    for dr, dc in _DIRECTIONS:
        adjacent = (pawn[0] + dr, pawn[1] + dc)
        if _blocked(state.horizontal_walls, state.vertical_walls, pawn, adjacent):
            continue
        if adjacent != opponent:
            moves.add(adjacent)
            continue
        beyond = (opponent[0] + dr, opponent[1] + dc)
        if not _blocked(state.horizontal_walls, state.vertical_walls, opponent, beyond):
            moves.add(beyond)
        else:
            sides = ((0, -1), (0, 1)) if dr else ((-1, 0), (1, 0))
            for side_dr, side_dc in sides:
                destination = (opponent[0] + side_dr, opponent[1] + side_dc)
                if not _blocked(state.horizontal_walls, state.vertical_walls, opponent, destination):
                    moves.add(destination)
    return tuple(sorted(moves))


def is_wall_structurally_valid(state: BoardState, orientation: str, row: int, col: int) -> bool:
    _require_board(state)
    if orientation not in ("h", "v") or type(row) is not int or type(col) is not int:
        return False
    if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
        return False
    if orientation == "h":
        return not any((row, c) in state.horizontal_walls for c in (col - 1, col, col + 1)) and (row, col) not in state.vertical_walls
    return not any((r, col) in state.vertical_walls for r in (row - 1, row, row + 1)) and (row, col) not in state.horizontal_walls


def is_legal_wall(state: BoardState, orientation: str, row: int, col: int, player: int | None = None) -> bool:
    _require_board(state)
    player = state.to_play if player is None else _player(player)
    if goal_winner(state) is not None or state.walls_remaining[player] == 0:
        return False
    if not is_wall_structurally_valid(state, orientation, row, col):
        return False
    horizontal = state.horizontal_walls | {(row, col)} if orientation == "h" else state.horizontal_walls
    vertical = state.vertical_walls | {(row, col)} if orientation == "v" else state.vertical_walls
    return all(_distance(state.pawns[p], 0 if p == 0 else 8, horizontal, vertical) != inf for p in (0, 1))


def legal_actions(state: BoardState) -> tuple[int, ...]:
    """All legal actions in ascending action-ID order, without pruning."""

    _require_board(state)
    if goal_winner(state) is not None:
        return ()
    moves = [encode_move(*position) for position in legal_pawn_moves(state)]
    walls = [encode_wall(orientation, row, col)
             for orientation in ("h", "v")
             for row in range(WALL_GRID_SIZE) for col in range(WALL_GRID_SIZE)
             if is_legal_wall(state, orientation, row, col)]
    return tuple(moves + walls)


def legal_action_mask(state: BoardState) -> tuple[bool, ...]:
    legal = frozenset(legal_actions(state))
    return tuple(index in legal for index in range(ACTION_SPACE_SIZE))


def apply_action(state: BoardState, action_index: int) -> BoardState:
    """Apply a legal action and switch sides, including after a goal move."""

    _require_board(state)
    action = decode_action(action_index)
    if goal_winner(state) is not None:
        raise ValueError("cannot apply an action after a goal")
    player = state.to_play
    if action["type"] == "move":
        destination = (action["row"], action["col"])
        if destination not in legal_pawn_moves(state):
            raise ValueError(f"illegal pawn move: {destination}")
        pawns = tuple(destination if p == player else state.pawns[p] for p in (0, 1))
        return replace(state, pawns=pawns, to_play=1 - player)
    orientation, row, col = action["orientation"], action["row"], action["col"]
    if not is_legal_wall(state, orientation, row, col):
        raise ValueError(f"illegal wall placement: {orientation} {row},{col}")
    remaining = tuple(state.walls_remaining[p] - (p == player) for p in (0, 1))
    field = "horizontal_walls" if orientation == "h" else "vertical_walls"
    return replace(state, walls_remaining=remaining, to_play=1 - player,
                   **{field: getattr(state, field) | {(row, col)}})


# The design document's short transition name is also supported.
apply = apply_action
