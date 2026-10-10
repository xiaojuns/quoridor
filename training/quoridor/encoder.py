"""Engine-derived, partially observed policy/value inputs.

``az-encoder-v1-partial-16`` intentionally exposes only the current position's
repetition count, not all earlier position counts. Distinct exact episode
histories can therefore produce identical tensors but different future draws.
Search adjudication must retain ``Episode.history``; these features are not a
Markov state or an admissible key for exact outcome caching.

The network receives tensors only. All legality and goal distances here are
derived through the public engine API, with episode completion handled by the
episode layer. Wall feature anchors occupy the upper-left 8 by 8 subgrid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import engine
from .episode import Episode, REPETITION_LIMIT
from .feature_spec import ENCODER_VERSION, INPUT_CHANNELS, PLANE_NAMES
from .symmetry import Symmetry, canonical_transform


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    """An unbatched CPU tensor, its legal mask, and original-to-input transform.

    Apply ``transform.inverse().action(action)`` or ``.policy(policy)`` to map
    predicted canonical actions or probabilities back to the original board.
    Values already refer to the same current player and need no sign change.
    Tensor storage belongs to this result and never aliases the episode.
    """

    features: torch.Tensor
    legal_mask: torch.Tensor
    transform: Symmetry


def encode_episode(episode: Episode, *, mirror: bool = False) -> EncodedPosition:
    """Encode an exact episode as float32[16,9,9] and bool[209].

    The acting player becomes player zero with goal row zero. For every stopped
    episode (goal, repetition, truncation, failure), legal mask/feature planes
    are all false/zero. Such inputs can still be inspected but cannot propose
    another action. No value label is invented for a runtime stop.
    """

    if type(episode) is not Episode:
        raise TypeError("encoder requires a quoridor-v1 Episode")
    transform = canonical_transform(episode.state, mirror=mirror)
    state = transform.state(episode.state)
    features = torch.zeros(
        (INPUT_CHANNELS, engine.BOARD_SIZE, engine.BOARD_SIZE),
        dtype=torch.float32,
        device="cpu",
    )
    row, col = state.pawns[0]
    features[0, row, col] = 1.0
    row, col = state.pawns[1]
    features[1, row, col] = 1.0
    for row, col in state.horizontal_walls:
        features[2, row, col] = 1.0
    for row, col in state.vertical_walls:
        features[3, row, col] = 1.0
    features[4].fill_(state.walls_remaining[0] / engine.STARTING_WALLS)
    features[5].fill_(state.walls_remaining[1] / engine.STARTING_WALLS)

    legal_mask = torch.tensor(
        (False,) * engine.ACTION_SPACE_SIZE if episode.finished else engine.legal_action_mask(state),
        dtype=torch.bool,
        device="cpu",
    )
    features[6] = legal_mask[:81].reshape(9, 9)
    features[7, :8, :8] = legal_mask[81:145].reshape(8, 8)
    features[8, :8, :8] = legal_mask[145:].reshape(8, 8)
    features[9, 0, :] = 1.0
    features[10, 8, :] = 1.0
    # These scalars are invariant under one global history transformation; no
    # per-position canonicalization or approximation of the history is needed.
    features[11].fill_(episode.plies / episode.max_game_plies)
    features[12].fill_(episode.repetition_count / REPETITION_LIMIT)
    features[13].fill_(engine.shortest_path_distance(state, 0) / 80)
    features[14].fill_(engine.shortest_path_distance(state, 1) / 80)
    features[15].fill_(1.0)
    return EncodedPosition(features, legal_mask, transform)
