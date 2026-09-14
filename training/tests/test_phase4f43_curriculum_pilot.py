from __future__ import annotations

import unittest

import torch

from quoridor.codec import mirror_action_index, state_to_serializable
from quoridor.game import apply_action_index, create_initial_state, legal_action_indices
from quoridor.phase4f43_curriculum_pilot import (
    _representative_samples,
    _restore_mirrored_policy,
    tensorize_samples,
)


class Phase4F43CurriculumPilotTests(unittest.TestCase):
    def _sample(self, *, player_one: bool = False, band: str = "early") -> dict[str, object]:
        state = create_initial_state()
        if player_one:
            asymmetric = next(
                action
                for action in legal_action_indices(state)
                if mirror_action_index(action) != action
            )
            state = apply_action_index(state, asymmetric)
        actions = legal_action_indices(state)[:2]
        return {
            "state": state_to_serializable(state),
            "policy": [[actions[0], 0.7], [actions[1], 0.3]],
            "value": 0.25,
            "band": band,
        }

    def test_tensor_adapter_preserves_soft_targets(self) -> None:
        batch = tensorize_samples([self._sample()])
        self.assertEqual(tuple(batch["states"].shape), (1, 16, 9, 9))
        self.assertEqual(tuple(batch["policies"].shape), (1, 209))
        self.assertTrue(torch.isclose(batch["policies"].sum(), torch.tensor(1.0)))
        self.assertEqual(int(torch.count_nonzero(batch["policies"])), 2)
        self.assertAlmostEqual(float(batch["values"][0]), 0.25)

    def test_mirror_transform_updates_state_and_policy_together(self) -> None:
        sample = self._sample(player_one=True)
        identity = tensorize_samples([sample])
        mirrored = tensorize_samples([sample], mirrored=[True])
        restored = _restore_mirrored_policy(mirrored["policies"][0])
        self.assertTrue(torch.equal(identity["policies"][0], restored))
        self.assertTrue(torch.equal(identity["values"], mirrored["values"]))
        self.assertFalse(torch.equal(identity["states"], mirrored["states"]))

    def test_restore_mirror_is_an_involution(self) -> None:
        policy = torch.arange(209, dtype=torch.float32)
        restored_twice = _restore_mirrored_policy(_restore_mirrored_policy(policy))
        self.assertTrue(torch.equal(policy, restored_twice))
        self.assertEqual(mirror_action_index(mirror_action_index(0)), 0)

    def test_representative_sample_selection_covers_bands_and_sides(self) -> None:
        samples = [
            self._sample(player_one=side == 1, band=band)
            for band in ("early", "middle", "late")
            for side in (0, 1)
        ]
        chosen = _representative_samples(samples)
        keys = {(sample["band"], sample["state"]["currentPlayer"]) for sample in chosen}
        self.assertEqual(len(keys), 6)


if __name__ == "__main__":
    unittest.main()
