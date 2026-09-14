from __future__ import annotations

import unittest

from quoridor.phase4f46_curriculum_lineage_screen import outcome_band


class Phase4F46ScreenTests(unittest.TestCase):
    def test_outcome_bands(self) -> None:
        self.assertEqual(outcome_band(0.449), "curriculum-lineage-regression")
        self.assertEqual(outcome_band(0.45), "curriculum-lineage-inconclusive")
        self.assertEqual(outcome_band(0.499), "curriculum-lineage-inconclusive")
        self.assertEqual(outcome_band(0.50), "curriculum-lineage-retained")


if __name__ == "__main__":
    unittest.main()
