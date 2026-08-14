"""Versioned AlphaZero state and action codecs shared with the web game."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Callable, Mapping, Optional

from .game import BOARD_SIZE, WALL_GRID_SIZE, State


CODEC_VERSION = 1

MOVE_ACTION_OFFSET = 0
MOVE_ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
HORIZONTAL_WALL_ACTION_OFFSET = MOVE_ACTION_OFFSET + MOVE_ACTION_COUNT
WALL_ACTION_COUNT = WALL_GRID_SIZE * WALL_GRID_SIZE
VERTICAL_WALL_ACTION_OFFSET = HORIZONTAL_WALL_ACTION_OFFSET + WALL_ACTION_COUNT
ACTION_SPACE_SIZE = VERTICAL_WALL_ACTION_OFFSET + WALL_ACTION_COUNT


def _require_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _require_coordinate(value: object, label: str, size: int) -> int:
    coordinate = _require_integer(value, label)
    if not 0 <= coordinate < size:
        raise ValueError(f"{label} must be between 0 and {size - 1}")
    return coordinate


def _require_player(player: object) -> int:
    player = _require_integer(player, "player")
    if player not in (0, 1):
        raise ValueError(f"invalid player: {player}")
    return player


def _require_action_index(index: object) -> int:
    index = _require_integer(index, "action index")
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"action index must be between 0 and {ACTION_SPACE_SIZE - 1}")
    return index


def encode_move(row: int, col: int) -> int:
    row = _require_coordinate(row, "move row", BOARD_SIZE)
    col = _require_coordinate(col, "move col", BOARD_SIZE)
    return MOVE_ACTION_OFFSET + row * BOARD_SIZE + col


def encode_wall(orientation: str, row: int, col: int) -> int:
    if orientation not in ("h", "v"):
        raise ValueError('wall orientation must be "h" or "v"')
    row = _require_coordinate(row, "wall row", WALL_GRID_SIZE)
    col = _require_coordinate(col, "wall col", WALL_GRID_SIZE)
    offset = (
        HORIZONTAL_WALL_ACTION_OFFSET
        if orientation == "h"
        else VERTICAL_WALL_ACTION_OFFSET
    )
    return offset + row * WALL_GRID_SIZE + col


def encode_action(action: Mapping[str, object]) -> int:
    """Encode a JS-compatible action mapping into the fixed action space."""

    if not isinstance(action, Mapping):
        raise TypeError("action must be a mapping")
    action_type = action.get("type")
    if action_type == "move":
        return encode_move(action.get("row"), action.get("col"))
    if action_type == "wall":
        orientation = action.get("orientation")
        if not isinstance(orientation, str):
            raise TypeError("wall orientation must be a string")
        return encode_wall(orientation, action.get("row"), action.get("col"))
    raise ValueError('action type must be "move" or "wall"')


def decode_action(index: int) -> dict[str, object]:
    """Decode an action index to the DTO shape used by the browser engine."""

    index = _require_action_index(index)
    if index < HORIZONTAL_WALL_ACTION_OFFSET:
        return {
            "type": "move",
            "row": index // BOARD_SIZE,
            "col": index % BOARD_SIZE,
        }

    if index < VERTICAL_WALL_ACTION_OFFSET:
        orientation = "h"
        offset = HORIZONTAL_WALL_ACTION_OFFSET
    else:
        orientation = "v"
        offset = VERTICAL_WALL_ACTION_OFFSET
    wall_index = index - offset
    return {
        "type": "wall",
        "orientation": orientation,
        "row": wall_index // WALL_GRID_SIZE,
        "col": wall_index % WALL_GRID_SIZE,
    }


def _sorted_wall_keys(walls: frozenset[tuple[int, int]]) -> list[str]:
    return sorted(f"{row},{col}" for row, col in walls)


def position_hash(state: State) -> str:
    """Return the byte-compatible Phase 0 position hash.

    Move count, result metadata, and repetition history are excluded so the key
    identifies the playable position only.
    """

    payload = [
        CODEC_VERSION,
        state.current_player,
        state.pawns[0][0],
        state.pawns[0][1],
        state.pawns[1][0],
        state.pawns[1][1],
        state.walls_remaining[0],
        state.walls_remaining[1],
        _sorted_wall_keys(state.horizontal_walls),
        _sorted_wall_keys(state.vertical_walls),
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def current_repetition_count(state: State) -> int:
    """Return the current board position's positive occurrence count."""

    count = state.position_counts.get(position_hash(state))
    return count if type(count) is int and count > 0 else 1


def _transformed_state(
    source: State,
    *,
    pawns: tuple[tuple[int, int], tuple[int, int]],
    walls_remaining: tuple[int, int],
    horizontal_walls: frozenset[tuple[int, int]],
    vertical_walls: frozenset[tuple[int, int]],
    current_player: int,
    winner: Optional[int],
    history_transform: Callable[[str], str],
) -> State:
    transformed = State(
        pawns=pawns,
        walls_remaining=walls_remaining,
        horizontal_walls=horizontal_walls,
        vertical_walls=vertical_walls,
        current_player=current_player,
        winner=winner,
        move_number=source.move_number,
        position_counts={},
        termination_reason=source.termination_reason,
    )
    counts: dict[str, int] = {}
    for historical_hash, count in source.position_counts.items():
        if not isinstance(historical_hash, str) or not historical_hash:
            raise ValueError("position_counts keys must be non-empty strings")
        if type(count) is not int or count <= 0:
            raise ValueError("position_counts values must be positive integers")
        transformed_hash = history_transform(historical_hash)
        if transformed_hash in counts:
            raise ValueError("transformed repetition history contains a collision")
        counts[transformed_hash] = count
    current_hash = position_hash(transformed)
    if current_hash not in counts:
        counts[current_hash] = current_repetition_count(source)
    return replace(transformed, position_counts=counts)


def _transform_position_hash(
    value: str,
    *,
    rotate_and_swap: bool = False,
    mirror: bool = False,
) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("position_counts contains an invalid position hash") from error
    if not isinstance(payload, list) or len(payload) != 10:
        raise ValueError("position_counts contains an invalid position hash")
    (
        version,
        current_player,
        p0_row,
        p0_col,
        p1_row,
        p1_col,
        walls0,
        walls1,
        horizontal,
        vertical,
    ) = payload
    if version != CODEC_VERSION or current_player not in (0, 1):
        raise ValueError("position_counts contains an unsupported position hash")
    if not isinstance(horizontal, list) or not isinstance(vertical, list):
        raise ValueError("position_counts contains an invalid wall list")

    def wall_key(key: object) -> str:
        row, col = _parse_wall_key(key)
        if rotate_and_swap:
            row = WALL_GRID_SIZE - 1 - row
            col = WALL_GRID_SIZE - 1 - col
        elif mirror:
            col = WALL_GRID_SIZE - 1 - col
        return f"{row},{col}"

    if rotate_and_swap:
        p0_row, p0_col, p1_row, p1_col = (
            BOARD_SIZE - 1 - p1_row,
            BOARD_SIZE - 1 - p1_col,
            BOARD_SIZE - 1 - p0_row,
            BOARD_SIZE - 1 - p0_col,
        )
        current_player = 1 - current_player
        walls0, walls1 = walls1, walls0
    elif mirror:
        p0_col = BOARD_SIZE - 1 - p0_col
        p1_col = BOARD_SIZE - 1 - p1_col
    transformed = [
        version,
        current_player,
        p0_row,
        p0_col,
        p1_row,
        p1_col,
        walls0,
        walls1,
        sorted(wall_key(key) for key in horizontal),
        sorted(wall_key(key) for key in vertical),
    ]
    return json.dumps(transformed, separators=(",", ":"), ensure_ascii=True)


def rotate_and_swap_state(state: State) -> State:
    """Rotate 180 degrees and exchange player identities, including history."""

    rotate_pawn = lambda pawn: (BOARD_SIZE - 1 - pawn[0], BOARD_SIZE - 1 - pawn[1])
    rotate_wall = lambda wall: (
        WALL_GRID_SIZE - 1 - wall[0],
        WALL_GRID_SIZE - 1 - wall[1],
    )
    winner = None if state.winner is None else 1 - state.winner
    return _transformed_state(
        state,
        pawns=(rotate_pawn(state.pawns[1]), rotate_pawn(state.pawns[0])),
        walls_remaining=(state.walls_remaining[1], state.walls_remaining[0]),
        horizontal_walls=frozenset(rotate_wall(wall) for wall in state.horizontal_walls),
        vertical_walls=frozenset(rotate_wall(wall) for wall in state.vertical_walls),
        current_player=1 - state.current_player,
        winner=winner,
        history_transform=lambda value: _transform_position_hash(
            value,
            rotate_and_swap=True,
        ),
    )


def canonicalize_state(state: State) -> State:
    """Make the side to move player 0 at the bottom, moving toward row zero."""

    if state.current_player == 1:
        return rotate_and_swap_state(state)
    return _transformed_state(
        state,
        pawns=state.pawns,
        walls_remaining=state.walls_remaining,
        horizontal_walls=frozenset(state.horizontal_walls),
        vertical_walls=frozenset(state.vertical_walls),
        current_player=0,
        winner=state.winner,
        history_transform=lambda value: value,
    )


def _rotate_action(action: Mapping[str, object]) -> dict[str, object]:
    transformed = dict(action)
    if action.get("type") == "move":
        transformed["row"] = BOARD_SIZE - 1 - _require_coordinate(
            action.get("row"), "move row", BOARD_SIZE
        )
        transformed["col"] = BOARD_SIZE - 1 - _require_coordinate(
            action.get("col"), "move col", BOARD_SIZE
        )
    elif action.get("type") == "wall":
        transformed["row"] = WALL_GRID_SIZE - 1 - _require_coordinate(
            action.get("row"), "wall row", WALL_GRID_SIZE
        )
        transformed["col"] = WALL_GRID_SIZE - 1 - _require_coordinate(
            action.get("col"), "wall col", WALL_GRID_SIZE
        )
    else:
        raise ValueError(f"unknown action type: {action.get('type')!r}")
    return transformed


def to_canonical_action(
    action: Mapping[str, object], original_current_player: int
) -> dict[str, object]:
    player = _require_player(original_current_player)
    return dict(action) if player == 0 else _rotate_action(action)


def from_canonical_action(
    action: Mapping[str, object], original_current_player: int
) -> dict[str, object]:
    return to_canonical_action(action, original_current_player)


def to_canonical_action_index(index: int, original_current_player: int) -> int:
    return encode_action(
        to_canonical_action(decode_action(index), original_current_player)
    )


def from_canonical_action_index(index: int, original_current_player: int) -> int:
    return to_canonical_action_index(index, original_current_player)


def rotate_and_swap_action_index(index: int) -> int:
    """Rotate one action through the player-role symmetry."""

    return to_canonical_action_index(index, 1)


# A descriptive alias used by callers that do not need the inverse terminology.
canonicalize_action_index = to_canonical_action_index


def mirror_state(state: State) -> State:
    """Reflect a state and every entry in its repetition history left-to-right."""

    mirror_pawn = lambda pawn: (pawn[0], BOARD_SIZE - 1 - pawn[1])
    mirror_wall = lambda wall: (wall[0], WALL_GRID_SIZE - 1 - wall[1])
    return _transformed_state(
        state,
        pawns=(mirror_pawn(state.pawns[0]), mirror_pawn(state.pawns[1])),
        walls_remaining=state.walls_remaining,
        horizontal_walls=frozenset(mirror_wall(wall) for wall in state.horizontal_walls),
        vertical_walls=frozenset(mirror_wall(wall) for wall in state.vertical_walls),
        current_player=state.current_player,
        winner=state.winner,
        history_transform=lambda value: _transform_position_hash(
            value,
            mirror=True,
        ),
    )


def mirror_position_hash(value: str) -> str:
    """Reflect a validated playable-position hash left-to-right."""

    validate_position_hash(value)
    return _transform_position_hash(value, mirror=True)


def mirror_action(action: Mapping[str, object]) -> dict[str, object]:
    transformed = dict(action)
    if action.get("type") == "move":
        transformed["row"] = _require_coordinate(action.get("row"), "move row", BOARD_SIZE)
        transformed["col"] = BOARD_SIZE - 1 - _require_coordinate(
            action.get("col"), "move col", BOARD_SIZE
        )
    elif action.get("type") == "wall":
        transformed["row"] = _require_coordinate(action.get("row"), "wall row", WALL_GRID_SIZE)
        transformed["col"] = WALL_GRID_SIZE - 1 - _require_coordinate(
            action.get("col"), "wall col", WALL_GRID_SIZE
        )
    else:
        raise ValueError(f"unknown action type: {action.get('type')!r}")
    return transformed


def mirror_action_index(index: int) -> int:
    return encode_action(mirror_action(decode_action(index)))


# Short name requested by the training interface.
mirror_index = mirror_action_index


def state_to_serializable(state: State) -> dict[str, object]:
    """Create the deterministic JSON-safe DTO committed in Phase 0 fixtures."""

    return {
        "codecVersion": CODEC_VERSION,
        "pawns": [
            {"row": state.pawns[0][0], "col": state.pawns[0][1]},
            {"row": state.pawns[1][0], "col": state.pawns[1][1]},
        ],
        "wallsRemaining": list(state.walls_remaining),
        "horizontalWalls": _sorted_wall_keys(state.horizontal_walls),
        "verticalWalls": _sorted_wall_keys(state.vertical_walls),
        "currentPlayer": state.current_player,
        "winner": state.winner,
        "moveNumber": state.move_number,
        "terminationReason": state.termination_reason,
        "repetitionCount": current_repetition_count(state),
    }


def _parse_position(value: object, label: str, size: int) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return (
        _require_coordinate(value.get("row"), f"{label} row", size),
        _require_coordinate(value.get("col"), f"{label} col", size),
    )


def _parse_wall_key(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        raise TypeError("wall keys must be strings")
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"invalid wall key: {value!r}")
    try:
        row, col = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"invalid wall key: {value!r}") from error
    return (
        _require_coordinate(row, "wall row", WALL_GRID_SIZE),
        _require_coordinate(col, "wall col", WALL_GRID_SIZE),
    )


def validate_position_hash(value: object) -> str:
    """Validate and return one canonical, versioned repetition-history key."""

    if not isinstance(value, str) or not value:
        raise ValueError("position hash must be a non-empty string")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("position hash is not valid JSON") from error
    if not isinstance(payload, list) or len(payload) != 10:
        raise ValueError("position hash must contain the versioned 10-field payload")
    (
        version,
        current_player,
        p0_row,
        p0_col,
        p1_row,
        p1_col,
        walls0,
        walls1,
        raw_horizontal,
        raw_vertical,
    ) = payload
    if version != CODEC_VERSION:
        raise ValueError("position hash codec version is unsupported")
    if not isinstance(raw_horizontal, list) or not isinstance(raw_vertical, list):
        raise ValueError("position hash wall collections must be lists")
    horizontal = tuple(_parse_wall_key(key) for key in raw_horizontal)
    vertical = tuple(_parse_wall_key(key) for key in raw_vertical)
    if list(raw_horizontal) != sorted(raw_horizontal) or len(set(horizontal)) != len(
        horizontal
    ):
        raise ValueError("position hash horizontal walls must be sorted and unique")
    if list(raw_vertical) != sorted(raw_vertical) or len(set(vertical)) != len(
        vertical
    ):
        raise ValueError("position hash vertical walls must be sorted and unique")
    state = State(
        pawns=(
            (
                _require_coordinate(p0_row, "player 0 row", BOARD_SIZE),
                _require_coordinate(p0_col, "player 0 column", BOARD_SIZE),
            ),
            (
                _require_coordinate(p1_row, "player 1 row", BOARD_SIZE),
                _require_coordinate(p1_col, "player 1 column", BOARD_SIZE),
            ),
        ),
        walls_remaining=(
            _require_integer(walls0, "player 0 walls"),
            _require_integer(walls1, "player 1 walls"),
        ),
        horizontal_walls=frozenset(horizontal),
        vertical_walls=frozenset(vertical),
        current_player=_require_player(current_player),
    )
    if state.pawns[0] == state.pawns[1]:
        raise ValueError("position hash pawns must occupy distinct squares")
    if position_hash(state) != value:
        raise ValueError("position hash is not in canonical serialized form")
    return value


def state_from_serializable(dto: Mapping[str, object]) -> State:
    """Restore a DTO with current-position-only repetition metadata."""

    if not isinstance(dto, Mapping):
        raise TypeError("state DTO must be a mapping")
    version = dto.get("codecVersion", CODEC_VERSION)
    if version != CODEC_VERSION:
        raise ValueError(f"unsupported codec version: {version!r}")

    raw_pawns = dto.get("pawns")
    raw_remaining = dto.get("wallsRemaining")
    raw_horizontal = dto.get("horizontalWalls")
    raw_vertical = dto.get("verticalWalls")
    if not isinstance(raw_pawns, (list, tuple)) or len(raw_pawns) != 2:
        raise ValueError("pawns must contain two positions")
    if not isinstance(raw_remaining, (list, tuple)) or len(raw_remaining) != 2:
        raise ValueError("wallsRemaining must contain two counts")
    if not isinstance(raw_horizontal, (list, tuple)):
        raise TypeError("horizontalWalls must be a sequence")
    if not isinstance(raw_vertical, (list, tuple)):
        raise TypeError("verticalWalls must be a sequence")

    winner = dto.get("winner")
    if winner is not None:
        winner = _require_player(winner)
    termination_reason = dto.get("terminationReason")
    if termination_reason is not None and not isinstance(termination_reason, str):
        raise TypeError("terminationReason must be a string or null")

    state = State(
        pawns=(
            _parse_position(raw_pawns[0], "player 0 pawn", BOARD_SIZE),
            _parse_position(raw_pawns[1], "player 1 pawn", BOARD_SIZE),
        ),
        walls_remaining=(
            _require_integer(raw_remaining[0], "player 0 walls remaining"),
            _require_integer(raw_remaining[1], "player 1 walls remaining"),
        ),
        horizontal_walls=frozenset(_parse_wall_key(key) for key in raw_horizontal),
        vertical_walls=frozenset(_parse_wall_key(key) for key in raw_vertical),
        current_player=_require_player(dto.get("currentPlayer")),
        winner=winner,
        move_number=_require_integer(dto.get("moveNumber", 0), "moveNumber"),
        position_counts={},
        termination_reason=termination_reason,
    )
    repetition_count = dto.get("repetitionCount", 1)
    repetition_count = _require_integer(repetition_count, "repetitionCount")
    if repetition_count <= 0:
        raise ValueError("repetitionCount must be positive")
    return replace(
        state,
        position_counts={position_hash(state): repetition_count},
    )


__all__ = [
    "CODEC_VERSION",
    "MOVE_ACTION_OFFSET",
    "MOVE_ACTION_COUNT",
    "HORIZONTAL_WALL_ACTION_OFFSET",
    "WALL_ACTION_COUNT",
    "VERTICAL_WALL_ACTION_OFFSET",
    "ACTION_SPACE_SIZE",
    "encode_move",
    "encode_wall",
    "encode_action",
    "decode_action",
    "position_hash",
    "validate_position_hash",
    "current_repetition_count",
    "canonicalize_state",
    "rotate_and_swap_state",
    "to_canonical_action",
    "from_canonical_action",
    "to_canonical_action_index",
    "from_canonical_action_index",
    "rotate_and_swap_action_index",
    "canonicalize_action_index",
    "mirror_state",
    "mirror_position_hash",
    "mirror_action",
    "mirror_action_index",
    "mirror_index",
    "state_to_serializable",
    "state_from_serializable",
]
