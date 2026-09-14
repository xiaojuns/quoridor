from __future__ import annotations

import math
import unittest

from quoridor.codec import mirror_action_index, mirror_state
from quoridor.game import create_initial_state, legal_action_indices
from quoridor.phase4f40_target_probe import (
    BAND_COUNTS,
    _path_tempo_target,
    classic_candidates,
    fit_isotonic,
    isotonic_predict,
    target_for_row,
)


class Phase4F40TargetProbeTests(unittest.TestCase):
    def test_frozen_band_counts_are_exact(self) -> None:
        self.assertEqual(sum(BAND_COUNTS["train"].values()), 4096)
        self.assertEqual(sum(BAND_COUNTS["development"].values()), 512)

    def test_isotonic_fit_is_monotone(self) -> None:
        blocks = fit_isotonic(
            [-0.8, -0.2, -0.2, 0.1, 0.7],
            [-1.0, 1.0, -1.0, -1.0, 1.0],
        )
        predictions = [isotonic_predict(blocks, value) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        self.assertEqual(predictions, sorted(predictions))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in predictions))

    def test_classic_candidates_are_mirror_equivariant(self) -> None:
        state = create_initial_state()
        teacher = legal_action_indices(state)[0]
        original = classic_candidates(state, teacher)
        mirrored = classic_candidates(mirror_state(state), mirror_action_index(teacher))
        restored = sorted(
            (mirror_action_index(int(item["actionIndex"])), float(item["score"]))
            for item in mirrored
        )
        expected = sorted((int(item["actionIndex"]), float(item["score"])) for item in original)
        self.assertEqual(restored, expected)
        self.assertIn(len(original), (11, 12))
        self.assertEqual(
            len({int(item["actionIndex"]) for item in original}), len(original)
        )
        self.assertAlmostEqual(_path_tempo_target(state), _path_tempo_target(mirror_state(state)))

    def test_target_blend_is_normalized_and_bounded(self) -> None:
        row = {
            "classicCandidates": [
                {"actionIndex": 1, "score": 2.0},
                {"actionIndex": 2, "score": 1.0},
            ],
            "mcts": {"visitCounts": [[1, 3], [3, 1]], "rootValue": 0.4},
            "outcome": 1.0,
            "pathTempo": -0.2,
        }
        isotonic = fit_isotonic([-1.0, 1.0], [-1.0, 1.0])
        policy, value, components = target_for_row(row, 1.0, isotonic)
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertEqual(set(policy), {1, 2, 3})
        self.assertTrue(all(math.isfinite(item) and item >= 0 for item in policy.values()))
        self.assertTrue(-1.0 <= value <= 1.0)
        self.assertEqual(components["terminalOutcome"], 1.0)


if __name__ == "__main__":
    unittest.main()
