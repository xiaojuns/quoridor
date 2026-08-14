from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


BOARD_SIZE = 9
ACTION_SIZE = 209
LEGACY_INPUT_CHANNELS = 14
PHASE4_INPUT_CHANNELS = 16
INPUT_CHANNELS = LEGACY_INPUT_CHANNELS
SUPPORTED_INPUT_CHANNELS = frozenset({LEGACY_INPUT_CHANNELS, PHASE4_INPUT_CHANNELS})
MAX_WALLS = 10
MAX_HALF_MOVES = 256
REPETITION_LIMIT = 3
DISTANCE_SCALE = BOARD_SIZE * BOARD_SIZE - 1

PLANE_NAMES = (
    "self_pawn",
    "opponent_pawn",
    "horizontal_walls",
    "vertical_walls",
    "self_walls_remaining",
    "opponent_walls_remaining",
    "legal_pawn_targets",
    "self_goal_distance",
    "opponent_goal_distance",
    "self_goal_row",
    "opponent_goal_row",
    "move_progress",
    "repetition_count",
    "ones",
)
PHASE4_PLANE_NAMES = PLANE_NAMES + (
    "second_occurrence_pawn_targets",
    "repetition_loss_pawn_targets",
)


@dataclass(frozen=True)
class NetworkConfig:
    """Serializable architecture settings stored alongside each checkpoint."""

    input_channels: int = INPUT_CHANNELS
    board_size: int = BOARD_SIZE
    action_size: int = ACTION_SIZE
    channels: int = 192
    blocks: int = 16
    se_reduction: int = 8
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 256

    def __post_init__(self) -> None:
        if self.input_channels not in SUPPORTED_INPUT_CHANNELS:
            supported = ", ".join(str(value) for value in sorted(SUPPORTED_INPUT_CHANNELS))
            raise ValueError(f"input_channels must be one of: {supported}")
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"board_size must be {BOARD_SIZE}")
        if self.action_size != ACTION_SIZE:
            raise ValueError(f"action_size must be {ACTION_SIZE}")

        positive_fields = (
            "channels",
            "blocks",
            "se_reduction",
            "policy_channels",
            "value_channels",
            "value_hidden",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def smoke(cls) -> "NetworkConfig":
        return cls(
            channels=32,
            blocks=2,
            se_reduction=8,
            policy_channels=8,
            value_channels=8,
            value_hidden=64,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "NetworkConfig":
        return cls(**dict(values))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        scale = self.pool(inputs)
        scale = torch.relu(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return inputs * scale


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_reduction)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = torch.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = self.se(outputs)
        return torch.relu(outputs + residual)


class PolicyValueNetwork(nn.Module):
    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        config = self.config

        self.input_conv = nn.Conv2d(
            config.input_channels,
            config.channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm2d(config.channels)
        self.trunk = nn.Sequential(
            *(
                ResidualBlock(config.channels, config.se_reduction)
                for _ in range(config.blocks)
            )
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                config.channels, config.policy_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                config.policy_channels * config.board_size * config.board_size,
                config.action_size,
            ),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(
                config.channels, config.value_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(config.value_channels),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(
                config.value_channels * config.board_size * config.board_size,
                config.value_hidden,
            ),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

        # A zero residual branch makes early optimization behave like a shallow net.
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        expected = (
            self.config.input_channels,
            self.config.board_size,
            self.config.board_size,
        )
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                "inputs must have shape "
                f"[batch, {expected[0]}, {expected[1]}, {expected[2]}]"
            )

        features = torch.relu(self.input_bn(self.input_conv(inputs)))
        features = self.trunk(features)
        policy_logits = self.policy_head(features)
        value = self.value_head(self.value_features(features)).squeeze(-1)
        return policy_logits, value


def mask_policy_logits(
    policy_logits: Tensor,
    legal_mask: Tensor,
    invalid_value: float = -torch.inf,
) -> Tensor:
    """Masks illegal actions without changing any legal action logit."""

    if not policy_logits.is_floating_point():
        raise TypeError("policy_logits must use a floating-point dtype")
    if policy_logits.shape[-1] != ACTION_SIZE:
        raise ValueError(f"policy_logits last dimension must be {ACTION_SIZE}")

    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(mask, policy_logits.shape)
    except RuntimeError as error:
        raise ValueError("legal_mask is not broadcastable to policy_logits") from error
    return policy_logits.masked_fill(~mask, invalid_value)


def masked_policy(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Returns legal-action probabilities, or all zeros for a terminal row."""

    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(mask, policy_logits.shape)
    except RuntimeError as error:
        raise ValueError("legal_mask is not broadcastable to policy_logits") from error

    has_legal_action = mask.any(dim=-1, keepdim=True)
    safe_mask = mask | ~has_legal_action
    probabilities = torch.softmax(
        mask_policy_logits(policy_logits, safe_mask), dim=-1
    )
    probabilities = probabilities * mask.to(dtype=probabilities.dtype)
    normalizer = probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    return probabilities / normalizer


masked_softmax = masked_policy


def legal_action_mask(
    state: Any, *, device: torch.device | str | None = None
) -> Tensor:
    from .codec import canonicalize_state
    from .game import legal_action_indices

    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    indices = tuple(legal_action_indices(canonicalize_state(state)))
    if indices:
        mask[list(indices)] = True
    return mask


def _goal_distance_map(state: Any, goal_row: int) -> list[list[int]]:
    from .game import is_passage_blocked

    unreachable = BOARD_SIZE * BOARD_SIZE
    distances = [[unreachable] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    queue: deque[tuple[int, int]] = deque()

    for col in range(BOARD_SIZE):
        distances[goal_row][col] = 0
        queue.append((goal_row, col))

    while queue:
        row, col = queue.popleft()
        next_distance = distances[row][col] + 1
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + row_delta
            next_col = col + col_delta
            if not (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE):
                continue
            if distances[next_row][next_col] <= next_distance:
                continue
            if is_passage_blocked(state, row, col, next_row, next_col):
                continue
            distances[next_row][next_col] = next_distance
            queue.append((next_row, next_col))

    return distances


def _current_repetition_count(state: Any) -> int:
    from .codec import current_repetition_count

    return int(current_repetition_count(state))


def encode_state(
    state: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    input_channels: int = INPUT_CHANNELS,
) -> Tensor:
    """Encode a state using the legacy or Phase 4 canonical feature contract."""

    if not dtype.is_floating_point:
        raise TypeError("encode_state dtype must be floating point")
    if input_channels not in SUPPORTED_INPUT_CHANNELS:
        raise ValueError("unsupported input channel contract")

    from .codec import (
        canonicalize_state,
        current_repetition_count,
        to_canonical_action_index,
    )
    from .game import apply_action_index, legal_action_indices, legal_pawn_moves

    canonical = canonicalize_state(state)
    planes = torch.zeros(
        (input_channels, BOARD_SIZE, BOARD_SIZE), dtype=dtype, device=device
    )

    self_pawn, opponent_pawn = canonical.pawns
    planes[0, self_pawn[0], self_pawn[1]] = 1
    planes[1, opponent_pawn[0], opponent_pawn[1]] = 1

    for row, col in canonical.horizontal_walls:
        planes[2, row, col] = 1
    for row, col in canonical.vertical_walls:
        planes[3, row, col] = 1

    planes[4].fill_(
        min(max(canonical.walls_remaining[0], 0), MAX_WALLS) / MAX_WALLS
    )
    planes[5].fill_(
        min(max(canonical.walls_remaining[1], 0), MAX_WALLS) / MAX_WALLS
    )

    for row, col in legal_pawn_moves(canonical):
        planes[6, row, col] = 1

    for plane_index, goal_row in ((7, 0), (8, BOARD_SIZE - 1)):
        distances = _goal_distance_map(canonical, goal_row)
        values = torch.tensor(distances, dtype=dtype, device=device)
        planes[plane_index] = values.clamp_max(DISTANCE_SCALE) / DISTANCE_SCALE

    planes[9, 0, :] = 1
    planes[10, BOARD_SIZE - 1, :] = 1
    planes[11].fill_(
        min(max(canonical.move_number, 0), MAX_HALF_MOVES) / MAX_HALF_MOVES
    )
    repetition_count = _current_repetition_count(canonical)
    planes[12].fill_(
        min(max(repetition_count, 0), REPETITION_LIMIT) / REPETITION_LIMIT
    )
    planes[13].fill_(1)
    if input_channels == PHASE4_INPUT_CHANNELS:
        for action_index in legal_action_indices(state):
            if action_index >= BOARD_SIZE * BOARD_SIZE:
                continue
            child = apply_action_index(state, action_index)
            canonical_index = to_canonical_action_index(
                action_index,
                state.current_player,
            )
            row, col = divmod(canonical_index, BOARD_SIZE)
            if child.termination_reason == "repetition":
                planes[15, row, col] = 1
            elif current_repetition_count(child) == 2:
                planes[14, row, col] = 1
    return planes


__all__ = [
    "ACTION_SIZE",
    "INPUT_CHANNELS",
    "LEGACY_INPUT_CHANNELS",
    "PHASE4_INPUT_CHANNELS",
    "SUPPORTED_INPUT_CHANNELS",
    "PLANE_NAMES",
    "PHASE4_PLANE_NAMES",
    "NetworkConfig",
    "PolicyValueNetwork",
    "ResidualBlock",
    "SqueezeExcitation",
    "encode_state",
    "legal_action_mask",
    "mask_policy_logits",
    "masked_policy",
    "masked_softmax",
]
