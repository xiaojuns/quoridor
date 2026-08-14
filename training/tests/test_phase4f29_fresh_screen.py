from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.openings import load_opening_suite, opening_position_orbit_digest
from quoridor.openings import SUITE_NAMES
from quoridor.phase4f29_fresh_screen import (
    CONTRACT,
    FROZEN_SUITES,
    FULL_SUITE,
    ISOLATION_AUDIT,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    SCREEN_SUITE,
    _candidate_payload,
    _f28_evidence,
    _identities,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F29FreshScreenTests(unittest.TestCase):
    def test_fresh_suite_name_is_part_of_the_opening_schema(self) -> None:
        self.assertIn("recovery-fresh-dev-v1", SUITE_NAMES)

    def test_candidate_is_exact_f27_and_f28_retained_it(self) -> None:
        candidate, state = _candidate_payload(ROOT)
        evidence = _f28_evidence(ROOT, candidate)
        self.assertEqual(candidate["stage"], "stage-b-conservative")
        self.assertEqual(
            candidate["candidateModelSha256"],
            "6c7756474c05e5ebe2a6a8acfbac6a2b35f693a297bd112898589141f337a996",
        )
        self.assertGreater(len(state), 0)
        self.assertEqual(evidence["outcomeBand"], "conservative-stage-b-retained")

    def test_fresh_suite_is_disjoint_from_all_frozen_suites(self) -> None:
        fresh = load_opening_suite(ROOT / FULL_SUITE, expected_suite="recovery-fresh-dev-v1")
        prior_orbits = set()
        for relative, name in FROZEN_SUITES:
            suite = load_opening_suite(ROOT / relative, expected_suite=name)
            prior_orbits.update(opening_position_orbit_digest(item) for item in suite.openings)
        self.assertEqual(len(fresh.openings), 200)
        self.assertFalse(
            {opening_position_orbit_digest(item) for item in fresh.openings} & prior_orbits
        )

    def test_audit_and_contract_keep_decisions_closed(self) -> None:
        audit = json.loads((ROOT / ISOLATION_AUDIT).read_bytes())
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        screen = load_opening_suite(ROOT / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1")
        candidate, opponent = _identities(contract)
        self.assertEqual(audit["freshOverlapCount"], 0)
        self.assertEqual(len(screen.openings), OPENING_PAIRS)
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertFalse(contract["confirmationArenaAuthorized"])
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
