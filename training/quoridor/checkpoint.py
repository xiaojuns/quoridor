from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .storage import atomic_json_save, atomic_torch_save


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        artifact_limit_bytes: int,
        max_to_keep: int = 5,
    ) -> None:
        if artifact_limit_bytes <= 0:
            raise ValueError("artifact_limit_bytes must be positive")
        if max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive")
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.artifact_limit_bytes = int(artifact_limit_bytes)
        self.max_to_keep = int(max_to_keep)

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        replay_state: dict | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        if step < 0:
            raise ValueError("step must be non-negative")
        config_payload = asdict(config) if is_dataclass(config) else config
        payload = {
            "version": 1,
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_payload,
            "rng": capture_rng_state(),
            "replay": replay_state,
            "extra": extra or {},
        }
        path = self.checkpoint_dir / f"checkpoint-{step:08d}.pt"
        latest = self.latest()
        if latest is not None and latest != path:
            latest_step = int(latest.stem.rsplit("-", 1)[1])
            if step < latest_step:
                raise ValueError(
                    f"refusing to save stale step {step} over latest step {latest_step}"
                )
        existing = sorted(
            checkpoint
            for checkpoint in self.checkpoint_dir.glob("checkpoint-*.pt")
            if checkpoint != path
        )
        prune_count = max(0, len(existing) + 1 - self.max_to_keep)
        prunable_bytes = sum(
            checkpoint.stat().st_size for checkpoint in existing[:prune_count]
        )
        atomic_torch_save(
            payload,
            path,
            artifact_root=self.run_dir,
            # The final retained set must fit the cap. One replacement may
            # temporarily coexist with checkpoints that are pruned below.
            limit_bytes=self.artifact_limit_bytes + prunable_bytes,
        )
        self._prune()
        atomic_json_save(
            {"version": 1, "step": step, "path": str(path.name)},
            self.checkpoint_dir / "latest.json",
        )
        return path

    def latest(self) -> Path | None:
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.pt"))
        return checkpoints[-1] if checkpoints else None

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path) if path is not None else self.latest()
        if checkpoint_path is None:
            raise FileNotFoundError("no checkpoint is available")
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if restore_rng:
            restore_rng_state(payload["rng"])
        return payload

    def _prune(self) -> None:
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.pt"))
        for checkpoint in checkpoints[: -self.max_to_keep]:
            checkpoint.unlink()
