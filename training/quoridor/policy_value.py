"""Tensor-only policy/value model for the versioned AlphaZero player.

The caller owns encoding, action transforms, legal policy masks and value labels.
This module intentionally has no game, engine, episode or encoder dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .feature_spec import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, MODEL_VERSION


@dataclass(frozen=True)
class NetworkConfig:
    """Complete, strictly typed architecture metadata stored in checkpoints."""

    model_version: str = MODEL_VERSION
    input_channels: int = INPUT_CHANNELS
    board_size: int = BOARD_SIZE
    action_size: int = ACTION_SIZE
    channels: int = 64
    blocks: int = 4
    policy_channels: int = 8
    value_channels: int = 8
    value_hidden: int = 64

    def __post_init__(self) -> None:
        if type(self.model_version) is not str or self.model_version != MODEL_VERSION:
            raise ValueError(f"model_version must be {MODEL_VERSION!r}")
        for field in fields(self):
            if field.name == "model_version":
                continue
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an integer, not a boolean or float")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive")
        for name, required in (
            ("input_channels", INPUT_CHANNELS),
            ("board_size", BOARD_SIZE),
            ("action_size", ACTION_SIZE),
        ):
            if getattr(self, name) != required:
                raise ValueError(f"{name} must be {required}")

    @classmethod
    def smoke(cls) -> NetworkConfig:
        """Small architecture for CPU engineering checks, not a tuned player."""
        return cls(channels=8, blocks=1, policy_channels=4, value_channels=4, value_hidden=16)

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> NetworkConfig:
        if not isinstance(values, Mapping):
            raise TypeError("network configuration must be a mapping")
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise ValueError("network configuration requires exactly all architecture fields")
        return cls(**dict(values))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = torch.relu(self.bn1(self.conv1(inputs)))
        residual = self.bn2(self.conv2(residual))
        return torch.relu(inputs + residual)


class PolicyValueNetwork(nn.Module):
    """Residual trunk with raw policy logits and current-player tanh values.

    Inputs must share the model's floating dtype and device. No conversion,
    canonicalization, legality mask or rule adjudication happens in this class.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = NetworkConfig()
        if not isinstance(config, NetworkConfig):
            raise TypeError("config must be a NetworkConfig")
        self.config = config
        self.input_conv = nn.Conv2d(
            config.input_channels, config.channels, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.channels)
        self.trunk = nn.Sequential(*(_ResidualBlock(config.channels) for _ in range(config.blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, config.policy_channels, 1, bias=False),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.policy_channels * config.board_size**2, config.action_size),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(config.channels, config.value_channels, 1, bias=False),
            nn.BatchNorm2d(config.value_channels),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.value_channels * config.board_size**2, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch Tensor")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        if inputs.layout != torch.strided:
            raise ValueError("inputs must be a dense strided tensor")
        expected = (self.config.input_channels, self.config.board_size, self.config.board_size)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected or inputs.shape[0] < 1:
            raise ValueError(f"inputs must have shape [batch >= 1, {expected[0]}, {expected[1]}, {expected[2]}]")
        if inputs.dtype != self.input_conv.weight.dtype:
            raise ValueError("inputs dtype must match the model dtype")
        if inputs.device != self.input_conv.weight.device:
            raise ValueError("inputs device must match the model device")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must contain only finite values")

        features = torch.relu(self.input_bn(self.input_conv(inputs)))
        features = self.trunk(features)
        logits = self.policy_head(features)
        values = self.value_head(self.value_features(features)).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError("network produced nonfinite policy logits or values")
        return logits, values


__all__ = ["NetworkConfig", "PolicyValueNetwork"]
