from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.openings import load_opening_suite
from quoridor.phase4f28_conservative_screen import (
    CONTRACT,
    EXPECTED_F27_REVIEW_SHA256,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    PRIOR_SCREEN_SUITES,
    SCREEN_SUITE,
    SIMULATIONS,
    _f24_parent,
    _f27_candidate,
    _f27_evidence,
    _identities,
    _selected_suite,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F28ConservativeScreenTests(unittest.TestCase):
    def test_models_are_exact_f27_candidate_and_f24_parent(self) -> None:
        candidate, candidate_state = _f27_candidate(ROOT)
        parent, parent_state = _f24_parent(ROOT)
        self.assertEqual(candidate["stage"], "stage-b-conservative")
        self.assertEqual(parent["stage"], "stage-b-parent")
        self.assertEqual(
            candidate["candidateModelSha256"],
            "6c7756474c05e5ebe2a6a8acfbac6a2b35f693a297bd112898589141f337a996",
        )
        self.assertEqual(
            parent["candidateModelSha256"],
            "c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f",
        )
        self.assertGreater(len(candidate_state), 0)
        self.assertEqual(candidate_state.keys(), parent_state.keys())

    def test_f27_review_explicitly_authorizes_f28(self) -> None:
        candidate, _ = _f27_candidate(ROOT)
        evidence = _f27_evidence(ROOT, candidate)
        self.assertEqual(evidence["reviewSha256"], EXPECTED_F27_REVIEW_SHA256)
        self.assertTrue(evidence["conservativeStageBPassed"])
        self.assertTrue(evidence["f28ScreenAuthorized"])

    def test_selected_suite_excludes_all_160_prior_openings(self) -> None:
        suite, reference = _selected_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["excludedOpeningIds"], 160)
        self.assertEqual(reference["eligibleOpeningPairs"], 40)
        prior_ids = set()
        for item, relative in zip(reference["priorScreens"], PRIOR_SCREEN_SUITES):
            self.assertEqual(item["openingPairs"], 32)
            prior_ids.update(
                opening.id
                for opening in load_opening_suite(
                    ROOT / relative, expected_suite="recovery-dev-v1"
                ).openings
            )
        self.assertFalse({opening.id for opening in suite.openings} & prior_ids)

    def test_contract_is_symmetric_and_keeps_decisions_closed(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        candidate, opponent = _identities(contract)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, "native-mcts")
        self.assertNotEqual(candidate.artifact_hash, opponent.artifact_hash)
        self.assertEqual(contract["workload"]["candidateSimulations"], SIMULATIONS)
        self.assertEqual(contract["workload"]["opponentSimulations"], SIMULATIONS)
        self.assertFalse(contract["additionalStageBTrainingAuthorized"])
        self.assertFalse(contract["parentRollbackAuthorized"])
        self.assertFalse(contract["formalPromotion"])
        self.assertFalse(contract["screenTrajectoriesTrainingUseAllowed"])
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
