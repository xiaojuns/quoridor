from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .storage import atomic_torch_save


ACTION_SIZE = 209


@dataclass(frozen=True)
class TrainingSample:
    state: torch.Tensor
    policy: torch.Tensor
    value: float

    def normalized(self) -> "TrainingSample":
        if not isinstance(self.state, torch.Tensor):
            raise TypeError("sample state must be a tensor")
        if not isinstance(self.policy, torch.Tensor):
            raise TypeError("sample policy must be a tensor")
        state = (
            self.state.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        policy = (
            self.policy.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        if state.ndim != 3 or state.shape[-2:] != (9, 9):
            raise ValueError("sample state must have shape [channels, 9, 9]")
        if policy.shape != (ACTION_SIZE,):
            raise ValueError(f"sample policy must have shape [{ACTION_SIZE}]")
        if not bool(torch.isfinite(state).all().item()):
            raise ValueError("sample state must contain only finite values")
        if not bool(torch.isfinite(policy).all().item()):
            raise ValueError("sample policy must contain only finite values")
        if bool((policy < 0).any().item()):
            raise ValueError("sample policy must be non-negative")
        policy_sum = float(policy.sum().item())
        if policy_sum <= 0:
            raise ValueError("sample policy must contain positive probability mass")
        policy = policy / policy_sum
        value = float(self.value)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("sample value must be finite and in [-1, 1]")
        return TrainingSample(state=state, policy=policy, value=value)


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._samples: deque[TrainingSample] = deque(maxlen=self.capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: TrainingSample) -> None:
        self._samples.append(sample.normalized())

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._samples:
            raise ValueError("cannot sample an empty replay buffer")
        count = min(batch_size, len(self._samples))
        selected = self._random.sample(list(self._samples), count)
        return {
            "states": torch.stack([sample.state for sample in selected]).to(device),
            "policies": torch.stack([sample.policy for sample in selected]).to(device),
            "values": torch.tensor(
                [sample.value for sample in selected],
                dtype=torch.float32,
                device=device,
            ),
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "random_state": self._random.getstate(),
            "samples": [
                {
                    "state": sample.state,
                    "policy": sample.policy,
                    "value": sample.value,
                }
                for sample in self._samples
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        samples = state.get("samples", [])
        self._samples = deque(maxlen=self.capacity)
        for sample in samples[-self.capacity :]:
            self.add(
                TrainingSample(
                    state=sample["state"],
                    policy=sample["policy"],
                    value=sample["value"],
                ),
            )
        if "random_state" in state:
            self._random.setstate(state["random_state"])

    def save(self, path: str | Path, *, artifact_root: str | Path, limit_bytes: int) -> Path:
        return atomic_torch_save(
            self.state_dict(),
            path,
            artifact_root=artifact_root,
            limit_bytes=limit_bytes,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ReplayBuffer":
        state = torch.load(path, map_location=map_location, weights_only=False)
        replay = cls(int(state["capacity"]))
        replay.load_state_dict(state)
        return replay
