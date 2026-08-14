from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.openings import load_opening_suite
from quoridor.phase4f25_screen import (
    CANDIDATE_SIMULATIONS,
    CLASSIC_NODES,
    CONTRACT,
    EXPECTED_F24_REVIEW_SHA256,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    PRIOR_SCREEN_SUITES,
    SCREEN_SUITE,
    _candidate_payload,
    _identities,
    _selected_suite,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F25ScreenTests(unittest.TestCase):
    def test_candidate_is_f24_latest_without_copy(self) -> None:
        candidate, state = _candidate_payload(ROOT)
        self.assertEqual(candidate["generation"], 1)
        self.assertEqual(candidate["step"], 192)
        self.assertEqual(
            candidate["candidateModelSha256"],
            "c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f",
        )
        self.assertEqual(candidate["f24ReviewSha256"], EXPECTED_F24_REVIEW_SHA256)
        self.assertGreater(len(state), 0)

    def test_selected_suite_excludes_every_prior_screen_opening(self) -> None:
        suite, reference = _selected_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["excludedOpeningIds"], 96)
        self.assertEqual(reference["eligibleOpeningPairs"], 104)
        prior_ids = set()
        for item, relative in zip(reference["priorScreens"], PRIOR_SCREEN_SUITES):
            self.assertEqual(item["openingPairs"], 32)
            prior_ids.update(
                opening.id
                for opening in load_opening_suite(
                    ROOT / relative, expected_suite="recovery-dev-v1"
                ).openings
            )
        selected_ids = {opening.id for opening in suite.openings}
        self.assertEqual(len(selected_ids), OPENING_PAIRS)
        self.assertFalse(selected_ids & prior_ids)

    def test_frozen_contract_keeps_training_and_promotion_closed(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        _, opponent = _identities(contract)
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertFalse(contract["additionalStageBTrainingAuthorized"])
        self.assertFalse(contract["confirmationArenaAuthorized"])
        self.assertFalse(contract["formalPromotion"])
        self.assertFalse(contract["screenTrajectoriesTrainingUseAllowed"])
        self.assertEqual(contract["workload"]["candidateSimulations"], CANDIDATE_SIMULATIONS)
        self.assertEqual(contract["workload"]["classicBudgetLimit"], CLASSIC_NODES)
        self.assertEqual(contract["budget"]["artifactLimitBytes"], 12 * 1024**3)
        self.assertTrue((ROOT / SCREEN_SUITE).is_file())

    def test_preparation_is_canonical_and_self_hashed(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))


if __name__ == "__main__":
    unittest.main()
