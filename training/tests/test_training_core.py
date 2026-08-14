from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from quoridor.checkpoint import CheckpointManager
from quoridor.cli import main as cli_main
from quoridor.config import ExperimentConfig, load_experiment_config
from quoridor.mcts import MCTSConfig
from quoridor.replay import ReplayBuffer, TrainingSample
from quoridor.storage import ArtifactBudgetError, atomic_torch_save
from quoridor.trainer import AlphaZeroTrainer


ROOT = Path(__file__).resolve().parents[2]


class TinyPolicyValueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(2 * 9 * 9, 32), nn.ReLU())
        self.policy = nn.Linear(32, 209)
        self.value = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        return self.policy(hidden), self.value(hidden).squeeze(1)


def make_sample(marker: float) -> TrainingSample:
    state = torch.full((2, 9, 9), marker)
    policy = torch.zeros(209)
    policy[int(marker) % 209] = 1
    return TrainingSample(state=state, policy=policy, value=marker / 10)


class TrainingCoreTests(unittest.TestCase):
    def test_smoke_config_and_replay_capacity_are_deterministic(self) -> None:
        config = load_experiment_config(ROOT / "training/configs/smoke.json")
        self.assertEqual(config.artifact_limit_bytes, 8 * 1024**3)
        self.assertEqual(config.network.channels, 32)
        self.assertEqual(config.self_play.actors, 1)
        self.assertEqual(config.pipeline, "reference")
        phase4 = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        self.assertEqual(phase4.pipeline, "phase4")
        with self.assertRaisesRegex(RuntimeError, "legacy in-memory"):
            cli_main(
                [
                    "train",
                    "--config",
                    str(ROOT / "training/configs/phase4-bootstrap.json"),
                    "--engine",
                    "native",
                ]
            )
        extended = ExperimentConfig.from_dict(
            {"artifactLimitBytes": 12 * 1024**3}
        )
        self.assertEqual(extended.artifact_limit_bytes, 12 * 1024**3)
        expanded = ExperimentConfig.from_dict(
            {"artifactLimitBytes": 16 * 1024**3}
        )
        self.assertEqual(expanded.artifact_limit_bytes, 16 * 1024**3)
        with self.assertRaisesRegex(ValueError, "16 GiB"):
            ExperimentConfig.from_dict({"artifactLimitBytes": 17 * 1024**3})

        replay = ReplayBuffer(3, seed=7)
        replay.extend(make_sample(float(index)) for index in range(5))
        self.assertEqual(len(replay), 3)
        batch = replay.sample(2)
        self.assertEqual(tuple(batch["states"].shape), (2, 2, 9, 9))
        self.assertEqual(tuple(batch["policies"].shape), (2, 209))

    def test_replay_round_trip_and_artifact_budget(self) -> None:
        replay = ReplayBuffer(4, seed=11)
        replay.extend([make_sample(1), make_sample(2)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = replay.save(
                root / "replay.pt",
                artifact_root=root,
                limit_bytes=1_000_000,
            )
            restored = ReplayBuffer.load(path)
            self.assertEqual(len(restored), 2)
            self.assertEqual(restored.capacity, 4)

            with self.assertRaises(ArtifactBudgetError):
                atomic_torch_save(
                    torch.ones(1024),
                    root / "too-large.pt",
                    artifact_root=root,
                    limit_bytes=32,
                )

    def test_train_step_updates_parameters(self) -> None:
        model = TinyPolicyValueNetwork()
        replay = ReplayBuffer(8)
        replay.extend(make_sample(float(index)) for index in range(1, 5))
        trainer = AlphaZeroTrainer(model)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = trainer.train_step(replay.sample(4))

        self.assertEqual(trainer.step, 1)
        self.assertGreater(metrics.loss, 0)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(before, model.parameters())
            ),
        )

    def test_checkpoint_restores_model_optimizer_and_replay(self) -> None:
        model = TinyPolicyValueNetwork()
        trainer = AlphaZeroTrainer(model)
        replay = ReplayBuffer(4)
        replay.add(make_sample(1))
        trainer.train_step(replay.sample(1))

        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(
                directory,
                artifact_limit_bytes=4_000_000,
                max_to_keep=2,
            )
            path = manager.save(
                step=trainer.step,
                model=model,
                optimizer=trainer.optimizer,
                config={"name": "test"},
                replay_state=replay.state_dict(),
            )
            expected = [parameter.detach().clone() for parameter in model.parameters()]
            for parameter in model.parameters():
                parameter.data.zero_()

            payload = manager.load(model, trainer.optimizer, path=path, restore_rng=False)
            self.assertEqual(payload["step"], 1)
            self.assertEqual(len(payload["replay"]["samples"]), 1)
            for restored, saved in zip(model.parameters(), expected):
                self.assertTrue(torch.equal(restored, saved))

    def test_checkpoint_pruning_allows_one_file_replacement_within_budget(self) -> None:
        model = TinyPolicyValueNetwork()
        trainer = AlphaZeroTrainer(model)
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(
                directory,
                artifact_limit_bytes=4_000_000,
                max_to_keep=1,
            )
            first = manager.save(
                step=2,
                model=model,
                optimizer=trainer.optimizer,
                config={"name": "test"},
            )
            manager.artifact_limit_bytes = first.stat().st_size + 1024
            second = manager.save(
                step=3,
                model=model,
                optimizer=trainer.optimizer,
                config={"name": "test"},
            )
            self.assertTrue(second.exists())
            self.assertFalse(first.exists())
            with self.assertRaisesRegex(ValueError, "stale step"):
                manager.save(
                    step=1,
                    model=model,
                    optimizer=trainer.optimizer,
                    config={"name": "test"},
                )

    def test_seed_contract_matches_native_uint64_boundary(self) -> None:
        for seed in (-1, 2**63, True):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    MCTSConfig(seed=seed)


if __name__ == "__main__":
    unittest.main()
