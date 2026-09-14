from __future__ import annotations

import math
import unittest

from quoridor.phase4f48_regression_safe_redesign import (
    _rank,
    absolute_ply_band,
    jensen_shannon,
    redesign_checks,
)


class Phase4F48RedesignTests(unittest.TestCase):
    def test_absolute_ply_band_uses_state_move_number(self) -> None:
        self.assertEqual(absolute_ply_band(15), "early")
        self.assertEqual(absolute_ply_band(16), "middle")
        self.assertEqual(absolute_ply_band(39), "middle")
        self.assertEqual(absolute_ply_band(40), "late")

    def test_rank_is_deterministic_on_ties(self) -> None:
        self.assertEqual(_rank([0.5, 0.5, 0.0], 0), 1)
        self.assertEqual(_rank([0.5, 0.5, 0.0], 1), 2)

    def test_jensen_shannon_is_symmetric_and_bounded(self) -> None:
        left = [0.9, 0.1]
        right = [0.1, 0.9]
        self.assertAlmostEqual(jensen_shannon(left, right), jensen_shannon(right, left))
        self.assertGreater(jensen_shannon(left, right), 0.0)
        self.assertLessEqual(jensen_shannon(left, right), math.log(2.0))

    def test_redesign_requires_all_observed_failures(self) -> None:
        policy = {
            "surrogatePolicy": {"teacherActionTop1Coverage": 0.75},
            "bandMismatchPositions": 10,
        }
        f47 = {
            "diagnosis": {"primaryCause": "off-distribution-policy-drift-amplified-by-mcts"},
            "screenTrajectoriesTrainingUseAllowed": False,
        }
        self.assertTrue(all(redesign_checks(policy, f47, 0.55).values()))


if __name__ == "__main__":
    unittest.main()
