from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.openings import load_opening_suite
from quoridor.phase4f26_lineage_screen import (
    CONTRACT,
    EXPECTED_F25_GATE_SHA256,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    PRIOR_SCREEN_SUITES,
    SCREEN_SUITE,
    SIMULATIONS,
    _f25_evidence,
    _identities,
    _selected_suite,
    _stage_a_candidate,
    _stage_b_candidate,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F26LineageScreenTests(unittest.TestCase):
    def test_lineage_models_are_exact_stage_b_and_stage_a_candidates(self) -> None:
        stage_b, stage_b_state = _stage_b_candidate(ROOT)
        stage_a, stage_a_state = _stage_a_candidate(ROOT)
        self.assertEqual(stage_b["stage"], "stage-b")
        self.assertEqual(stage_a["stage"], "stage-a")
        self.assertEqual(
            stage_b["candidateModelSha256"],
            "c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f",
        )
        self.assertEqual(
            stage_a["candidateModelSha256"],
            "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01",
        )
        self.assertGreater(len(stage_b_state), 0)
        self.assertEqual(stage_b_state.keys(), stage_a_state.keys())

    def test_f25_gate_is_below_confirmation_and_candidate_matches(self) -> None:
        stage_b, _ = _stage_b_candidate(ROOT)
        evidence = _f25_evidence(ROOT, stage_b)
        self.assertEqual(evidence["gateSha256"], EXPECTED_F25_GATE_SHA256)
        self.assertEqual(evidence["outcomeBand"], "stage-b-below-confirmation")
        self.assertEqual(evidence["candidateWins"], 28)
        self.assertEqual(evidence["candidateLosses"], 36)

    def test_selected_suite_excludes_all_128_prior_screen_openings(self) -> None:
        suite, reference = _selected_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["excludedOpeningIds"], 128)
        self.assertEqual(reference["eligibleOpeningPairs"], 72)
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

    def test_contract_is_symmetric_and_keeps_training_closed(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        candidate, opponent = _identities(contract)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, "native-mcts")
        self.assertNotEqual(candidate.artifact_hash, opponent.artifact_hash)
        self.assertEqual(contract["workload"]["candidateSimulations"], SIMULATIONS)
        self.assertEqual(contract["workload"]["opponentSimulations"], SIMULATIONS)
        self.assertFalse(contract["additionalStageBTrainingAuthorized"])
        self.assertFalse(contract["stageBRestartAuthorized"])
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
