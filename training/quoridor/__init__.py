"""Lazy legacy Phase 0-4 exports; new APIs use explicit submodule imports.

Importing a tensor-only model must not import rule engines as a package side
effect. Historical root names remain available on first access, with their
original legacy semantics.
"""

from importlib import import_module

_GAME_EXPORTS = frozenset({
    "BOARD_SIZE", "MAX_HALF_MOVES", "REPETITION_LIMIT", "STARTING_WALLS",
    "WALL_GRID_SIZE", "State", "apply_action_index", "create_initial_state",
    "get_winner", "has_path_to_goal", "is_legal_wall", "is_passage_blocked",
    "is_terminal", "is_wall_structurally_valid", "legal_action_indices",
    "legal_pawn_moves", "legal_wall_indices", "shortest_path_distance", "winner_value",
})

__all__ = [
    "ACTION_SPACE_SIZE",
    "BOARD_SIZE",
    "CODEC_VERSION",
    "HORIZONTAL_WALL_ACTION_OFFSET",
    "MAX_HALF_MOVES",
    "MOVE_ACTION_COUNT",
    "MOVE_ACTION_OFFSET",
    "REPETITION_LIMIT",
    "STARTING_WALLS",
    "VERTICAL_WALL_ACTION_OFFSET",
    "WALL_ACTION_COUNT",
    "WALL_GRID_SIZE",
    "State",
    "apply_action_index",
    "canonicalize_action_index",
    "canonicalize_state",
    "create_initial_state",
    "current_repetition_count",
    "decode_action",
    "encode_action",
    "encode_move",
    "encode_wall",
    "from_canonical_action",
    "from_canonical_action_index",
    "get_winner",
    "has_path_to_goal",
    "is_legal_wall",
    "is_passage_blocked",
    "is_terminal",
    "is_wall_structurally_valid",
    "legal_action_indices",
    "legal_pawn_moves",
    "legal_wall_indices",
    "mirror_action",
    "mirror_action_index",
    "mirror_index",
    "mirror_position_hash",
    "mirror_state",
    "position_hash",
    "rotate_and_swap_action_index",
    "rotate_and_swap_state",
    "shortest_path_distance",
    "state_from_serializable",
    "state_to_serializable",
    "to_canonical_action",
    "to_canonical_action_index",
    "winner_value",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = ".game" if name in _GAME_EXPORTS else ".codec"
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
