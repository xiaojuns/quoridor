"""Pure metadata for the first new-rule, partially observed network input.

No game objects, rule implementations or tensor libraries belong in this file.
Changing feature meaning requires a new encoder version, even at the same shape.
"""

ENCODER_VERSION = "az-encoder-v1-partial-16"
MODEL_VERSION = "az-resnet-v1"
INPUT_CHANNELS = 16
BOARD_SIZE = 9
ACTION_SIZE = 209
PLANE_NAMES = (
    "self_pawn",
    "opponent_pawn",
    "horizontal_walls",
    "vertical_walls",
    "self_walls_remaining",
    "opponent_walls_remaining",
    "legal_pawn_targets",
    "legal_horizontal_anchors",
    "legal_vertical_anchors",
    "self_goal_row",
    "opponent_goal_row",
    "ply_fraction",
    "current_repetition_fraction",
    "self_goal_distance",
    "opponent_goal_distance",
    "ones",
)


def encoder_metadata() -> dict:
    """Fresh JSON-safe identity; checkpoint loaders compare the complete layout."""
    return {
        "version": ENCODER_VERSION,
        "input_channels": INPUT_CHANNELS,
        "board_size": BOARD_SIZE,
        "action_size": ACTION_SIZE,
        "planes": list(PLANE_NAMES),
        "history_observation": "partial-current-position-v1",
        "canonicalization": "to-play-zero-rotate180-swap-v1",
        "normalization": {
            "wall_inventory_divisor": 10,
            "distance_divisor": 80,
            "repetition_divisor": 3,
            "ply_divisor": "episode.max_game_plies",
        },
    }
