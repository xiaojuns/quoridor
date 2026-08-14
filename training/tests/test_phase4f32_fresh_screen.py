from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.openings import load_opening_suite, opening_position_orbit_digest
from quoridor.phase4f32_fresh_screen import (
    CONTRACT,
    EXPECTED_CANDIDATE_SHA256,
    FROZEN_SUITES,
    FULL_SUITE,
    GENERATION,
    ISOLATION_AUDIT,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    SCREEN_SUITE,
    STEP,
    _candidate_payload,
    _identities,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F32FreshScreenTests(unittest.TestCase):
    def test_candidate_is_exact_f31_generation_five(self) -> None:
        candidate, state = _candidate_payload(ROOT)
        self.assertEqual(candidate["stage"], "stage-b-extended")
        self.assertEqual(candidate["generation"], GENERATION)
        self.assertEqual(candidate["step"], STEP)
        self.assertEqual(candidate["candidateModelSha256"], EXPECTED_CANDIDATE_SHA256)
        self.assertGreater(len(state), 0)

    def test_fresh_suite_is_disjoint_from_every_frozen_suite(self) -> None:
        fresh = load_opening_suite(
            ROOT / FULL_SUITE, expected_suite="recovery-fresh-dev-v1"
        )
        forbidden = set()
        for relative, name in FROZEN_SUITES:
            suite = load_opening_suite(ROOT / relative, expected_suite=name)
            forbidden.update(
                opening_position_orbit_digest(item) for item in suite.openings
            )
        self.assertEqual(len(fresh.openings), 200)
        self.assertFalse(
            {opening_position_orbit_digest(item) for item in fresh.openings}
            & forbidden
        )

    def test_audit_and_contract_close_training_and_promotion(self) -> None:
        audit = json.loads((ROOT / ISOLATION_AUDIT).read_bytes())
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        screen = load_opening_suite(
            ROOT / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1"
        )
        candidate, opponent = _identities(contract)
        self.assertEqual(audit["freshOverlapCount"], 0)
        self.assertEqual(len(screen.openings), OPENING_PAIRS)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertFalse(contract["confirmationArenaAuthorized"])
        self.assertFalse(contract["formalPromotion"])
        self.assertFalse(contract["additionalTrainingAuthorized"])
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
