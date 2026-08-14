from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f30_fresh_lineage_screen import (
    CONTRACT,
    EXPECTED_F29_GATE_SHA256,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    SCREEN_SUITE,
    SIMULATIONS,
    _f29_evidence,
    _identities,
    _stage_a,
    _stage_b,
    _suite_reference,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F30FreshLineageScreenTests(unittest.TestCase):
    def test_models_are_exact_f21_and_f27_candidates(self) -> None:
        stage_a, stage_a_state = _stage_a(ROOT)
        stage_b, stage_b_state = _stage_b(ROOT)
        self.assertEqual(stage_a["stage"], "stage-a")
        self.assertEqual(stage_b["stage"], "stage-b-conservative")
        self.assertEqual(
            stage_a["candidateModelSha256"],
            "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01",
        )
        self.assertEqual(
            stage_b["candidateModelSha256"],
            "6c7756474c05e5ebe2a6a8acfbac6a2b35f693a297bd112898589141f337a996",
        )
        self.assertGreater(len(stage_a_state), 0)
        self.assertEqual(stage_a_state.keys(), stage_b_state.keys())

    def test_f29_gate_and_fresh_suite_are_exactly_reused(self) -> None:
        stage_b, _ = _stage_b(ROOT)
        evidence = _f29_evidence(ROOT, stage_b)
        suite, reference = _suite_reference(ROOT)
        self.assertEqual(evidence["gateSha256"], EXPECTED_F29_GATE_SHA256)
        self.assertEqual(evidence["outcomeBand"], "fresh-development-insufficient")
        self.assertEqual(evidence["freshOverlapCount"], 0)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["path"], SCREEN_SUITE.as_posix())
        self.assertTrue(reference["reusedExactlyFromF29"])

    def test_contract_is_symmetric_and_keeps_decisions_closed(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        candidate, opponent = _identities(contract)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, "native-mcts")
        self.assertNotEqual(candidate.artifact_hash, opponent.artifact_hash)
        self.assertEqual(contract["workload"]["candidateSimulations"], SIMULATIONS)
        self.assertEqual(contract["workload"]["opponentSimulations"], SIMULATIONS)
        self.assertFalse(contract["trainingAuthorized"])
        self.assertFalse(contract["finalConfirmationAuthorized"])
        self.assertFalse(contract["formalPromotion"])
        self.assertFalse(contract["screenTrajectoriesTrainingUseAllowed"])
        self.assertEqual(contract["budget"]["artifactLimitBytes"], 12 * 1024**3)

    def test_preparation_is_canonical_and_self_hashed(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))


if __name__ == "__main__":
    unittest.main()
