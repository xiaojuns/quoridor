from __future__ import annotations

import math
import unittest

from quoridor.phase4f47_lineage_regression_diagnosis import (
    _softmax,
    jensen_shannon,
    ply_band,
)


class Phase4F47DiagnosisTests(unittest.TestCase):
    def test_ply_bands_use_absolute_move_number(self) -> None:
        self.assertEqual(ply_band(15), "early")
        self.assertEqual(ply_band(16), "middle")
        self.assertEqual(ply_band(39), "middle")
        self.assertEqual(ply_band(40), "late")

    def test_softmax_is_normalized_and_shift_invariant(self) -> None:
        left = _softmax([0.0, 1.0, 2.0])
        right = _softmax([10.0, 11.0, 12.0])
        self.assertAlmostEqual(sum(left), 1.0)
        self.assertEqual(left, right)

    def test_jensen_shannon_is_symmetric_and_bounded(self) -> None:
        left = [0.9, 0.1]
        right = [0.1, 0.9]
        self.assertAlmostEqual(jensen_shannon(left, right), jensen_shannon(right, left))
        self.assertGreater(jensen_shannon(left, right), 0.0)
        self.assertLessEqual(jensen_shannon(left, right), math.log(2.0))
        self.assertAlmostEqual(jensen_shannon(left, left), 0.0)


if __name__ == "__main__":
    unittest.main()
