from __future__ import annotations

import unittest

import torch

from quoridor.codec import state_to_serializable
from quoridor.game import create_initial_state, legal_action_indices
from quoridor.network import NetworkConfig, PolicyValueNetwork
from quoridor.phase4f43_curriculum_pilot import tensorize_samples
from quoridor.phase4f44_curriculum_pilot import (
    EncodedCurriculum,
    StratifiedBatchSampler,
    _training_batch,
    acceptance_checks,
    freeze_early_trunk,
    keep_frozen_batch_norm_eval,
)


class Phase4F44CurriculumPilotTests(unittest.TestCase):
    def _sample(self, ordinal: int, band: str) -> dict[str, object]:
        state = create_initial_state()
        actions = legal_action_indices(state)[:2]
        return {
            "ordinal": ordinal,
            "band": band,
            "state": state_to_serializable(state),
            "policy": [[actions[0], 0.6], [actions[1], 0.4]],
            "value": 0.2,
        }

    def _cache(self) -> EncodedCurriculum:
        samples = [self._sample(index, "early") for index in range(308)]
        samples += [self._sample(308 + index, "middle") for index in range(102)]
        samples += [self._sample(410 + index, "late") for index in range(102)]
        identity = tensorize_samples(samples)
        mirrored = tensorize_samples(samples, mirrored=[True] * len(samples))
        return EncodedCurriculum(
            identity["states"],
            identity["policies"],
            mirrored["states"],
            mirrored["policies"],
            identity["values"],
            tuple(sample["band"] for sample in samples),
            tuple(sample["ordinal"] for sample in samples),
        )

    def test_stratified_sampler_is_deterministic_and_exact(self) -> None:
        bands = ["early"] * 616 + ["middle"] * 204 + ["late"] * 204
        left = StratifiedBatchSampler(bands, seed=7)
        right = StratifiedBatchSampler(bands, seed=7)
        left_indices, left_mirror = left.batch(0)
        right_indices, right_mirror = right.batch(0)
        self.assertEqual((left_indices, left_mirror), (right_indices, right_mirror))
        counts = {band: sum(bands[index] == band for index in left_indices) for band in ("early", "middle", "late")}
        self.assertEqual(counts, {"early": 308, "middle": 102, "late": 102})
        self.assertEqual(len(left_mirror), 512)

    def test_training_batch_selects_mirrored_rows_without_changing_values(self) -> None:
        cache = self._cache()
        indices = list(range(512))
        flags = [index % 2 == 0 for index in indices]
        batch = _training_batch(cache, indices, flags, torch.device("cpu"))
        self.assertEqual(tuple(batch["states"].shape), (512, 16, 9, 9))
        self.assertTrue(torch.allclose(batch["policies"].sum(dim=1), torch.ones(512)))
        self.assertTrue(torch.equal(batch["values"], torch.full((512,), 0.2)))

    def test_freeze_boundary_excludes_first_half_and_keeps_frozen_bn_eval(self) -> None:
        model = PolicyValueNetwork(
            NetworkConfig(input_channels=16, channels=16, blocks=16, policy_channels=4, value_channels=4, value_hidden=16)
        )
        trainable = freeze_early_trunk(model)
        self.assertTrue(any(name.startswith("trunk.8.") for name in trainable))
        self.assertFalse(any(name.startswith("trunk.7.") for name in trainable))
        model.train()
        keep_frozen_batch_norm_eval(model)
        self.assertFalse(model.input_bn.training)
        self.assertFalse(model.trunk[7].bn1.training)
        self.assertTrue(model.trunk[8].bn1.training)

    def test_acceptance_requires_improvement_without_band_regression(self) -> None:
        baseline = {
            band: {"policyLoss": 1.0, "valueLoss": 1.0, "totalLoss": 2.0}
            for band in ("early", "middle", "late", "all")
        }
        candidate = {band: dict(values) for band, values in baseline.items()}
        candidate["early"] = {"policyLoss": 0.98, "valueLoss": 0.98, "totalLoss": 1.96}
        candidate["all"] = {"policyLoss": 0.99, "valueLoss": 0.99, "totalLoss": 1.98}
        acceptance = {
            "earlyTotalLossRelativeImprovementMinimum": 0.01,
            "overallTotalLossRelativeImprovementMinimum": 0.005,
            "earlyPolicyLossRelativeRegressionMaximum": 0.01,
            "earlyValueLossRelativeRegressionMaximum": 0.01,
            "middleTotalLossRelativeRegressionMaximum": 0.02,
            "lateTotalLossRelativeRegressionMaximum": 0.02,
        }
        self.assertTrue(all(acceptance_checks(baseline, candidate, acceptance).values()))
        candidate["late"]["totalLoss"] = 2.1
        self.assertFalse(acceptance_checks(baseline, candidate, acceptance)["lateTotalLoss"])


if __name__ == "__main__":
    unittest.main()
