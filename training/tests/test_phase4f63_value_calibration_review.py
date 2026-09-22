from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f63_value_calibration_review import (
    F64_DESIGN,
    GATE,
    REPORT,
    _evidence,
    _findings,
    verify_phase4f63,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F63ValueCalibrationReviewTests(unittest.TestCase):
    def test_evidence_separates_causal_result_from_external_transfer(self) -> None:
        evidence = _evidence(ROOT)
        findings = _findings(evidence)
        self.assertEqual(
            evidence["f62ScaledVersusUnscaledConfirmation"]["statistics"]["games"],
            400,
        )
        self.assertGreater(
            evidence["f62ScaledVersusUnscaledConfirmation"]["statistics"]
            ["winRateCI"]["lower"],
            0.5,
        )
        self.assertTrue(findings["notEstablished"]["benefitTransfersToClassicHard"])
        self.assertEqual(
            findings["identifiability"]["ambiguity"],
            "scaling-Q-also-changes-the-PUCT-Q-to-exploration-balance",
        )

    def test_frozen_design_uses_powered_external_confirmation(self) -> None:
        design = json.loads((ROOT / F64_DESIGN).read_bytes())
        self.assertEqual(design["candidate"]["valueScale"], 0.5)
        self.assertEqual(design["candidate"]["simulations"], 800)
        self.assertEqual(design["opponent"]["difficulty"], "classic-hard")
        self.assertEqual(design["openingIsolation"]["openingPairs"], 200)
        self.assertEqual(design["openingIsolation"]["games"], 400)
        self.assertTrue(design["decisionPolicy"]["noAdditionalScreenBeforeDecision"])
        self.assertFalse(design["formalTrainingAuthorized"])
        self.assertFalse(design["valueHeadTrainingAuthorized"])
        self.assertFalse(design["promotionAuthorized"])

    def test_completed_review_is_cpu_only_and_reproducible(self) -> None:
        result = verify_phase4f63(ROOT)
        report = json.loads((ROOT / REPORT).read_bytes())
        gate = json.loads((ROOT / GATE).read_bytes())
        self.assertEqual(result["status"], "verified")
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["cudaAccessed"])
        self.assertEqual(report["optimizerSteps"], 0)
        self.assertFalse(report["trainingStarted"])
        self.assertTrue(gate["f64ConfirmationPreparationAuthorized"])


if __name__ == "__main__":
    unittest.main()
