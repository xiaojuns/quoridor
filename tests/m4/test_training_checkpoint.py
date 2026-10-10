from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
from quoridor.replay import ReplayBuffer, ReplaySample
from quoridor.trainer import AlphaZeroTrainer
from quoridor.training_checkpoint import (
    IncompatibleTrainingCheckpointError,
    default_lineage,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _batch() -> dict[str, torch.Tensor]:
    states = torch.zeros(2, 16, 9, 9)
    policies = torch.zeros(2, 209)
    policies[:, 0] = 0.5
    policies[:, 1] = 0.5
    masks = torch.zeros(2, 209, dtype=torch.bool)
    masks[:, :2] = True
    return {
        "states": states,
        "policies": policies,
        "legal_masks": masks,
        "values": torch.tensor([1.0, -1.0]),
    }


class _Replay:
    def __init__(self) -> None:
        self.samples = [torch.tensor([1.0])]

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "samples": self.samples}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.samples = list(state["samples"])


def _trainer(seed: int = 11) -> AlphaZeroTrainer:
    torch.manual_seed(seed)
    return AlphaZeroTrainer(
        PolicyValueNetwork(NetworkConfig.smoke()),
        use_amp=False,
        require_legal_masks=True,
        schedule_steps=3,
    )


def test_checkpoint_roundtrip_restores_model_optimizer_runtime_and_replay(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.train_step(_batch())
    replay = _Replay()
    lineage = default_lineage(config_sha256="a" * 64)
    path = tmp_path / "resume.pt"
    artifact = save_training_checkpoint(path, trainer, replay=replay, lineage=lineage)

    restored = _trainer()
    restored_replay = _Replay()
    loaded = load_training_checkpoint(
        path,
        restored,
        replay=restored_replay,
        expected_lineage=lineage,
        expected_config_sha256="a" * 64,
    )
    assert artifact.sha256 == loaded.artifact.sha256
    assert restored.step == trainer.step
    assert restored_replay.samples[0].item() == 1.0
    for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    trainer = _trainer()
    path = tmp_path / "rng.pt"
    save_training_checkpoint(path, trainer, lineage=default_lineage(config_sha256="b" * 64))
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    # Disturb every stream, then load and compare the sequence that would have
    # followed immediately after the save point.
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored = _trainer()
    load_training_checkpoint(path, restored, expected_config_sha256="b" * 64)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == pytest.approx(expected)


def test_checkpoint_rejects_changed_lineage_and_corrupt_replay(tmp_path: Path) -> None:
    trainer = _trainer()
    path = tmp_path / "lineage.pt"
    lineage = default_lineage(config_sha256="c" * 64)
    save_training_checkpoint(path, trainer, lineage=lineage)
    restored = _trainer()
    with pytest.raises(IncompatibleTrainingCheckpointError):
        load_training_checkpoint(path, restored, expected_config_sha256="d" * 64)


def test_resume_reproduces_seeded_replay_sample_and_next_update(tmp_path: Path) -> None:
    def fresh() -> tuple[AlphaZeroTrainer, ReplayBuffer]:
        trainer = _trainer(17)
        replay = ReplayBuffer(capacity=4, seed=23)
        for index in range(4):
            policy = torch.zeros(209)
            policy[index] = 1.0
            replay.add(ReplaySample(
                features=torch.zeros(16, 9, 9),
                legal_mask=torch.ones(209, dtype=torch.bool),
                policy=policy,
                to_play=index % 2,
                value=float(index % 2),
                action=index,
                episode_plies=index,
                history=(str(index),),
            ))
        return trainer, replay

    trainer, replay = fresh()
    trainer.train_step(replay.sample(2))
    path = tmp_path / "resume-determinism.pt"
    save_training_checkpoint(
        path, trainer, replay=replay, lineage=default_lineage(config_sha256="e" * 64)
    )
    first_batch = replay.sample(2)
    first_metrics = trainer.train_step(first_batch)

    resumed, resumed_replay = fresh()
    load_training_checkpoint(
        path,
        resumed,
        replay=resumed_replay,
        expected_config_sha256="e" * 64,
    )
    second_batch = resumed_replay.sample(2)
    second_metrics = resumed.train_step(second_batch)
    assert torch.equal(first_batch["actions"], second_batch["actions"])
    assert first_metrics.loss == pytest.approx(second_metrics.loss, abs=0.0)
    assert all(torch.equal(left, right) for left, right in zip(
        trainer.model.parameters(), resumed.model.parameters()
    ))
