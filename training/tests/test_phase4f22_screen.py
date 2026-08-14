from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.openings import load_opening_suite
from quoridor.phase4f22_screen import (
    CANDIDATE_SIMULATIONS,
    CLASSIC_NODES,
    CONTRACT,
    OPENING_PAIRS,
    PRIOR_SCREEN_SUITES,
    SCREEN_SUITE,
    _candidate_payload,
    _identities,
    _selected_suite,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F22ScreenTests(unittest.TestCase):
    def test_candidate_is_f21_best_without_copy(self) -> None:
        candidate, state = _candidate_payload(ROOT)
        self.assertEqual(candidate["generation"], 1)
        self.assertEqual(candidate["step"], 192)
        self.assertEqual(
            candidate["candidateModelSha256"],
            "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01",
        )
        self.assertGreater(len(state), 0)

    def test_selected_suite_excludes_every_prior_screen_opening(self) -> None:
        suite, reference = _selected_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["excludedOpeningIds"], 64)
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

    def test_frozen_contract_keeps_stage_b_closed(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        _, opponent = _identities(contract)
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertFalse(contract["stageBTrainingAuthorized"])
        self.assertFalse(contract["additionalStageATrainingAuthorized"])
        self.assertFalse(contract["screenTrajectoriesTrainingUseAllowed"])
        self.assertEqual(contract["workload"]["candidateSimulations"], CANDIDATE_SIMULATIONS)
        self.assertEqual(contract["workload"]["classicBudgetLimit"], CLASSIC_NODES)
        self.assertEqual(contract["budget"]["artifactLimitBytes"], 12 * 1024**3)
        self.assertTrue((ROOT / CONTRACT).is_file())
        self.assertTrue((ROOT / SCREEN_SUITE).is_file())


if __name__ == "__main__":
    unittest.main()
