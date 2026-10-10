"""Legacy pure-Python rules (repetition and ply-cap losses).

The implementation deliberately favors clarity and cross-language conformance
over search speed.  Native search code can replace it later while continuing to
use the same immutable state and 209-action contract. The new quoridor-v1
board/episode contract lives in ``engine`` / ``episode``; do not use this module
to adjudicate new-version replay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import inf
from typing import Mapping, Optional


BOARD_SIZE = 9
WALL_GRID_SIZE = BOARD_SIZE - 1
STARTING_WALLS = 10
REPETITION_LIMIT = 3
MAX_HALF_MOVES = 256

Position = tuple[int, int]
Wall = tuple[int, int]

_DIRECTIONS: tuple[Position, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True, slots=True)
class State:
    """A rule-engine state.

    Board collections are immutable.  ``position_counts`` is treated as an
    immutable mapping by this module: every transition copies it before adding
    the resulting position. Canonical and mirrored transforms preserve the full
    history by structurally transforming every versioned position hash.
    """

    pawns: tuple[Position, Position] = ((8, 4), (0, 4))
    walls_remaining: tuple[int, int] = (STARTING_WALLS, STARTING_WALLS)
    horizontal_walls: frozenset[Wall] = frozenset()
    vertical_walls: frozenset[Wall] = frozenset()
    current_player: int = 0
    winner: Optional[int] = None
    move_number: int = 0
    position_counts: Mapping[str, int] = field(default_factory=dict, repr=False)
    termination_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _assert_player(self.current_player)
        if self.winner is not None:
            _assert_player(self.winner)
        if len(self.pawns) != 2 or any(not _inside(*pawn) for pawn in self.pawns):
            raise ValueError("pawns must contain two in-board coordinates")
        if (
            len(self.walls_remaining) != 2
            or any(type(count) is not int or count < 0 for count in self.walls_remaining)
        ):
            raise ValueError("walls_remaining must contain two non-negative integers")
        if type(self.move_number) is not int or self.move_number < 0:
            raise ValueError("move_number must be a non-negative integer")


def _assert_player(player: int) -> None:
    if type(player) is not int or player not in (0, 1):
        raise ValueError(f"invalid player: {player!r}")


def _inside(row: int, col: int) -> bool:
    return (
        type(row) is int
        and type(col) is int
        and 0 <= row < BOARD_SIZE
        and 0 <= col < BOARD_SIZE
    )


def create_initial_state(starting_player: int = 0) -> State:
    """Return a fresh initial state and count its initial position once."""

    _assert_player(starting_player)
    state = State(current_player=starting_player)
    from .codec import position_hash

    return replace(state, position_counts={position_hash(state): 1})


def is_passage_blocked(
    state: State,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
) -> bool:
    """Return whether a wall or board edge blocks a cardinal one-cell step."""

    if (
        not _inside(from_row, from_col)
        or not _inside(to_row, to_col)
        or abs(to_row - from_row) + abs(to_col - from_col) != 1
    ):
        return True

    if to_row != from_row:
        boundary_row = min(from_row, to_row)
        return (
            (boundary_row, from_col) in state.horizontal_walls
            or (boundary_row, from_col - 1) in state.horizontal_walls
        )

    boundary_col = min(from_col, to_col)
    return (
        (from_row, boundary_col) in state.vertical_walls
        or (from_row - 1, boundary_col) in state.vertical_walls
    )


def legal_pawn_moves(state: State, player: Optional[int] = None) -> tuple[Position, ...]:
    """Return legal step, jump, and blocked-jump diagonal destinations."""

    player = state.current_player if player is None else player
    _assert_player(player)
    if get_winner(state) is not None:
        return ()

    pawn = state.pawns[player]
    opponent = state.pawns[1 - player]
    moves: list[Position] = []

    for row_delta, col_delta in _DIRECTIONS:
        adjacent = (pawn[0] + row_delta, pawn[1] + col_delta)
        if is_passage_blocked(state, *pawn, *adjacent):
            continue

        if adjacent != opponent:
            moves.append(adjacent)
            continue

        jump = (opponent[0] + row_delta, opponent[1] + col_delta)
        if not is_passage_blocked(state, *opponent, *jump):
            moves.append(jump)
            continue

        side_directions = (
            ((-1, 0), (1, 0))
            if row_delta == 0
            else ((0, -1), (0, 1))
        )
        for side_row_delta, side_col_delta in side_directions:
            diagonal = (
                opponent[0] + side_row_delta,
                opponent[1] + side_col_delta,
            )
            if not is_passage_blocked(state, *opponent, *diagonal):
                moves.append(diagonal)

    return tuple(moves)


def is_wall_structurally_valid(
    state: State,
    orientation: str,
    row: int,
    col: int,
) -> bool:
    """Check bounds, overlap, adjacency overlap, and crossing constraints."""

    if (
        orientation not in ("h", "v")
        or type(row) is not int
        or type(col) is not int
        or not 0 <= row < WALL_GRID_SIZE
        or not 0 <= col < WALL_GRID_SIZE
    ):
        return False

    if orientation == "h":
        if any((row, candidate) in state.horizontal_walls for candidate in (col - 1, col, col + 1)):
            return False
        return (row, col) not in state.vertical_walls

    if any((candidate, col) in state.vertical_walls for candidate in (row - 1, row, row + 1)):
        return False
    return (row, col) not in state.horizontal_walls


def shortest_path_distance(state: State, player: int) -> float:
    """Return wall-only shortest distance to a player's goal row."""

    _assert_player(player)
    start = state.pawns[player]
    goal_row = 0 if player == 0 else BOARD_SIZE - 1
    if start[0] == goal_row:
        return 0.0

    queue: deque[Position] = deque([start])
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for row_delta, col_delta in _DIRECTIONS:
            destination = (current[0] + row_delta, current[1] + col_delta)
            if (
                not _inside(*destination)
                or destination in distances
                or is_passage_blocked(state, *current, *destination)
            ):
                continue
            if destination[0] == goal_row:
                return float(next_distance)
            distances[destination] = next_distance
            queue.append(destination)

    return inf


def has_path_to_goal(state: State, player: int) -> bool:
    return shortest_path_distance(state, player) != inf


def is_legal_wall(
    state: State,
    orientation: str,
    row: int,
    col: int,
    player: Optional[int] = None,
) -> bool:
    """Return whether a player may place the requested wall."""

    player = state.current_player if player is None else player
    if player not in (0, 1):
        return False
    if (
        get_winner(state) is not None
        or state.walls_remaining[player] <= 0
        or not is_wall_structurally_valid(state, orientation, row, col)
    ):
        return False

    if orientation == "h":
        probe = replace(
            state,
            horizontal_walls=state.horizontal_walls | {(row, col)},
        )
    else:
        probe = replace(
            state,
            vertical_walls=state.vertical_walls | {(row, col)},
        )
    return has_path_to_goal(probe, 0) and has_path_to_goal(probe, 1)


def legal_wall_indices(state: State, player: Optional[int] = None) -> tuple[int, ...]:
    """Return legal wall indices in action-space order."""

    player = state.current_player if player is None else player
    _assert_player(player)
    if get_winner(state) is not None or state.walls_remaining[player] <= 0:
        return ()

    from .codec import encode_wall

    indices: list[int] = []
    for orientation in ("h", "v"):
        for row in range(WALL_GRID_SIZE):
            for col in range(WALL_GRID_SIZE):
                if is_legal_wall(state, orientation, row, col, player):
                    indices.append(encode_wall(orientation, row, col))
    return tuple(indices)


def legal_action_indices(state: State, player: Optional[int] = None) -> tuple[int, ...]:
    """Return all legal 0..208 action indices, sorted ascending."""

    player = state.current_player if player is None else player
    _assert_player(player)
    if get_winner(state) is not None:
        return ()

    from .codec import encode_move

    moves = (encode_move(row, col) for row, col in legal_pawn_moves(state, player))
    return tuple(sorted((*moves, *legal_wall_indices(state, player))))


def get_winner(state: State) -> Optional[int]:
    if state.winner is not None:
        return state.winner
    if state.pawns[0][0] == 0:
        return 0
    if state.pawns[1][0] == BOARD_SIZE - 1:
        return 1
    return None


def is_terminal(state: State) -> bool:
    return get_winner(state) is not None


def winner_value(state: State, perspective_player: Optional[int] = None) -> Optional[float]:
    """Return a terminal value from ``perspective_player``'s point of view.

    The default perspective is the side to move, which is the convention used
    during MCTS backup.  Non-terminal states return ``None``.
    """

    perspective_player = (
        state.current_player if perspective_player is None else perspective_player
    )
    _assert_player(perspective_player)
    winner = get_winner(state)
    if winner is None:
        return None
    return 1.0 if winner == perspective_player else -1.0


def apply_action_index(state: State, action_index: int) -> State:
    """Validate and apply one encoded action without mutating ``state``."""

    if is_terminal(state):
        raise ValueError("cannot apply an action after the game has ended")

    from .codec import decode_action, position_hash

    action = decode_action(action_index)
    player = state.current_player
    pawns = state.pawns
    walls_remaining = state.walls_remaining
    horizontal_walls = state.horizontal_walls
    vertical_walls = state.vertical_walls

    if action["type"] == "move":
        destination = (action["row"], action["col"])
        if destination not in legal_pawn_moves(state, player):
            raise ValueError(f"illegal pawn move to {destination[0]},{destination[1]}")
        mutable_pawns = list(pawns)
        mutable_pawns[player] = destination
        pawns = (mutable_pawns[0], mutable_pawns[1])
    else:
        orientation = action["orientation"]
        row = action["row"]
        col = action["col"]
        if not is_legal_wall(state, orientation, row, col, player):
            raise ValueError(f"illegal wall placement: {orientation} {row},{col}")
        if orientation == "h":
            horizontal_walls = horizontal_walls | {(row, col)}
        else:
            vertical_walls = vertical_walls | {(row, col)}
        mutable_remaining = list(walls_remaining)
        mutable_remaining[player] -= 1
        walls_remaining = (mutable_remaining[0], mutable_remaining[1])

    next_state = State(
        pawns=pawns,
        walls_remaining=walls_remaining,
        horizontal_walls=frozenset(horizontal_walls),
        vertical_walls=frozenset(vertical_walls),
        current_player=1 - player,
        winner=None,
        move_number=state.move_number + 1,
        position_counts=dict(state.position_counts),
        termination_reason=None,
    )

    goal_winner = get_winner(next_state)
    if goal_winner is not None:
        return replace(
            next_state,
            winner=goal_winner,
            termination_reason="goal",
        )

    next_hash = position_hash(next_state)
    position_counts = dict(next_state.position_counts)
    occurrence = position_counts.get(next_hash, 0) + 1
    position_counts[next_hash] = occurrence
    next_state = replace(next_state, position_counts=position_counts)

    if occurrence >= REPETITION_LIMIT:
        return replace(
            next_state,
            winner=1 - player,
            termination_reason="repetition",
        )
    if next_state.move_number >= MAX_HALF_MOVES:
        return replace(
            next_state,
            winner=1 - player,
            termination_reason="move-limit",
        )
    return next_state


__all__ = [
    "BOARD_SIZE",
    "WALL_GRID_SIZE",
    "STARTING_WALLS",
    "REPETITION_LIMIT",
    "MAX_HALF_MOVES",
    "State",
    "create_initial_state",
    "is_passage_blocked",
    "legal_pawn_moves",
    "is_wall_structurally_valid",
    "shortest_path_distance",
    "has_path_to_goal",
    "is_legal_wall",
    "legal_wall_indices",
    "legal_action_indices",
    "get_winner",
    "is_terminal",
    "winner_value",
    "apply_action_index",
]
