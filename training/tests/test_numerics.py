from __future__ import annotations

import math
import copy
import unittest

import torch
from torch import nn

from quoridor.evaluator import TorchEvaluator
from quoridor.game import create_initial_state
from quoridor.replay import TrainingSample
from quoridor.trainer import AlphaZeroTrainer, NonFiniteTrainingError


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(2 * 9 * 9, 16), nn.ReLU())
        self.policy = nn.Linear(16, 209)
        self.value = nn.Sequential(nn.Linear(16, 1), nn.Tanh())

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        return self.policy(hidden), self.value(hidden).squeeze(1)


class InvalidEvaluatorModel(nn.Module):
    def __init__(self, *, invalid_value: bool) -> None:
        super().__init__()
        self.invalid_value = invalid_value

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((states.shape[0], 209), device=states.device)
        values = torch.zeros(states.shape[0], device=states.device)
        if self.invalid_value:
            values[0] = float("nan")
        else:
            logits[0, 0] = float("inf")
        return logits, values


def valid_batch() -> dict[str, torch.Tensor]:
    policies = torch.zeros((4, 209))
    policies[:, 0] = 1
    return {
        "states": torch.zeros((4, 2, 9, 9)),
        "policies": policies,
        "values": torch.tensor([1.0, -1.0, 0.5, -0.5]),
    }


class NumericSafetyTests(unittest.TestCase):
    def test_replay_rejects_non_finite_or_negative_samples(self) -> None:
        state = torch.zeros((2, 9, 9))
        policy = torch.zeros(209)
        policy[0] = 1
        invalid_samples = []
        invalid_state = state.clone()
        invalid_state[0, 0, 0] = float("nan")
        invalid_samples.append(TrainingSample(invalid_state, policy, 1.0))
        invalid_policy = policy.clone()
        invalid_policy[0] = float("inf")
        invalid_samples.append(TrainingSample(state, invalid_policy, 1.0))
        negative_policy = policy.clone()
        negative_policy[1] = -0.25
        invalid_samples.append(TrainingSample(state, negative_policy, 1.0))
        invalid_samples.append(TrainingSample(state, policy, float("nan")))

        for sample in invalid_samples:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    sample.normalized()

    def test_evaluator_rejects_non_finite_outputs_before_clamping(self) -> None:
        for invalid_value in (False, True):
            with self.subTest(invalid_value=invalid_value):
                evaluator = TorchEvaluator(
                    InvalidEvaluatorModel(invalid_value=invalid_value),
                    "cpu",
                )
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    evaluator.evaluate(create_initial_state())

    def test_trainer_rejects_non_finite_batch_without_advancing(self) -> None:
        trainer = AlphaZeroTrainer(TinyModel())
        batch = valid_batch()
        batch["states"][0, 0, 0, 0] = float("nan")

        with self.assertRaises(NonFiniteTrainingError):
            trainer.train_step(batch)
        self.assertEqual(trainer.step, 0)

    def test_value_weights_exclude_masked_targets_from_training(self) -> None:
        torch.manual_seed(1234)
        first_model = TinyModel()
        second_model = copy.deepcopy(first_model)
        first = AlphaZeroTrainer(first_model, use_amp=False)
        second = AlphaZeroTrainer(second_model, use_amp=False)
        first_batch = valid_batch()
        second_batch = {key: value.clone() for key, value in first_batch.items()}
        second_batch["values"][1] = -second_batch["values"][1]
        second_batch["values"][3] = -second_batch["values"][3]
        weights = torch.tensor([1.0, 0.0, 1.0, 0.0])

        first_metric = first.train_step(first_batch, value_weights=weights)
        second_metric = second.train_step(second_batch, value_weights=weights)

        self.assertEqual(first_metric, second_metric)
        for first_parameter, second_parameter in zip(
            first_model.parameters(), second_model.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_value_weights_must_retain_a_valid_target(self) -> None:
        trainer = AlphaZeroTrainer(TinyModel(), use_amp=False)
        with self.assertRaisesRegex(ValueError, "retain at least one"):
            trainer.train_step(valid_batch(), value_weights=torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            trainer.train_step(
                valid_batch(),
                value_weights=torch.tensor([1.0, 1.0, 1.0, 2.0]),
            )

    def test_trainer_runtime_restores_schedule_and_step(self) -> None:
        trainer = AlphaZeroTrainer(
            TinyModel(),
            learning_rate=1e-3,
            schedule_steps=8,
            minimum_learning_rate=1e-5,
        )
        trainer.train_step(valid_batch())
        trainer.train_step(valid_batch())
        runtime = trainer.runtime_state_dict()
        optimizer = trainer.optimizer.state_dict()

        restored = AlphaZeroTrainer(
            TinyModel(),
            learning_rate=1e-3,
            schedule_steps=8,
            minimum_learning_rate=1e-5,
        )
        restored.optimizer.load_state_dict(optimizer)
        restored.load_runtime_state_dict(runtime)

        self.assertEqual(restored.step, 2)
        self.assertEqual(restored.scheduler.last_epoch, trainer.scheduler.last_epoch)
        self.assertTrue(
            math.isclose(
                restored.optimizer.param_groups[0]["lr"],
                trainer.optimizer.param_groups[0]["lr"],
            )
        )

    def test_cosine_schedule_stays_at_minimum_after_its_horizon(self) -> None:
        trainer = AlphaZeroTrainer(
            TinyModel(),
            learning_rate=1e-3,
            schedule_steps=4,
            minimum_learning_rate=1e-4,
        )
        learning_rates = []
        for _ in range(8):
            learning_rates.append(trainer.train_step(valid_batch()).learning_rate)
        self.assertTrue(all(rate == 1e-4 for rate in learning_rates[3:]))

        with self.assertRaisesRegex(ValueError, "between zero"):
            AlphaZeroTrainer(
                TinyModel(),
                learning_rate=1e-4,
                minimum_learning_rate=2e-4,
            )

    def test_amp_resume_mode_must_match_before_scaler_restore(self) -> None:
        trainer = AlphaZeroTrainer(TinyModel(), use_amp=False)
        runtime = trainer.runtime_state_dict()
        runtime["ampEnabled"] = True
        with self.assertRaisesRegex(ValueError, "AMP mode"):
            trainer.load_runtime_state_dict(runtime)


if __name__ == "__main__":
    unittest.main()
