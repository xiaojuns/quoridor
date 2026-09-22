from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f64_scaled_classic_confirmation import (
    CLASSIC_NODES,
    CONTRACT,
    GAMES,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    SIMULATIONS,
    SUITE_PATH,
    VALUE_SCALE,
    _contract,
    _identities,
    _stored_suite,
    arena_contract_sha256,
    outcome_band,
    verify_preparation,
)
from quoridor.phase4f29_fresh_screen import _arena_references


ROOT = Path(__file__).resolve().parents[2]


class Phase4F64ScaledClassicConfirmationTests(unittest.TestCase):
    def test_historical_arena_index_includes_f62_confirmation(self) -> None:
        paths = {entry["path"] for entry in _arena_references(ROOT)}
        self.assertIn(
            "training/runs/phase4/phase4f/f62-f21-value-scale-confirmation-v1/"
            "confirmation-arena/result.json",
            paths,
        )

    def test_preparation_is_frozen_and_training_closed(self) -> None:
        result = verify_preparation(ROOT)
        contract = _contract(ROOT)
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(result["status"], "verified")
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["cudaAccessed"])
        self.assertEqual(report["optimizerSteps"], 0)
        self.assertFalse(contract["authorizations"]["trainingAuthorized"])
        self.assertFalse(contract["authorizations"]["valueHeadTrainingAuthorized"])
        self.assertFalse(contract["authorizations"]["promotionAuthorized"])

    def test_suite_is_unique_and_audited(self) -> None:
        suite, audit = _stored_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(audit["freshOverlapCount"], 0)
        self.assertTrue(audit["f22TrajectoriesExcluded"])
        self.assertTrue(audit["f61TrajectoriesExcluded"])
        self.assertTrue(audit["f62TrajectoriesExcluded"])
        self.assertTrue((ROOT / SUITE_PATH).is_file())

    def test_runtime_and_classic_identity_are_distinct(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        candidate, opponent = _identities(contract)
        self.assertEqual(contract["workload"]["games"], GAMES)
        self.assertEqual(contract["workload"]["candidateSimulations"], SIMULATIONS)
        self.assertEqual(contract["workload"]["candidateValueScale"], VALUE_SCALE)
        self.assertEqual(contract["workload"]["classicBudgetLimit"], CLASSIC_NODES)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, "classic-negamax")
        self.assertNotEqual(candidate.artifact_hash, opponent.artifact_hash)
        self.assertEqual(len(arena_contract_sha256(contract)), 64)

    def test_outcome_policy_requires_both_thresholds(self) -> None:
        self.assertEqual(
            outcome_band(0.55, 0.51), "scaled-f21-runtime-promotion-candidate"
        )
        self.assertEqual(
            outcome_band(0.549, 0.51), "value-scale-external-transfer-not-confirmed"
        )
        self.assertEqual(
            outcome_band(0.60, 0.50), "value-scale-external-transfer-not-confirmed"
        )


if __name__ == "__main__":
    unittest.main()
