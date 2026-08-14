from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.openings import load_opening_suite
from quoridor.phase4f20_screen import (
    CANDIDATE_SIMULATIONS,
    CLASSIC_NODES,
    CONTRACT,
    OPENING_PAIRS,
    PRIOR_SCREEN_SUITES,
    SCREEN_SUITE,
    GATE,
    _candidate_payload,
    _identities,
    _selected_suite,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F20ScreenTests(unittest.TestCase):
    def test_candidate_is_f19_best_without_copy(self) -> None:
        candidate, state = _candidate_payload(ROOT)
        self.assertEqual(candidate["generation"], 1)
        self.assertEqual(candidate["step"], 192)
        self.assertEqual(
            candidate["candidateModelSha256"],
            "ea7df3db48a0110ce2dcd7dde7d603e455bf7fc578e95418096eb09f016a02d1",
        )
        self.assertGreater(len(state), 0)

    def test_selected_suite_excludes_all_prior_screen_openings(self) -> None:
        suite, reference = _selected_suite(ROOT)
        self.assertEqual(len(suite.openings), OPENING_PAIRS)
        self.assertEqual(reference["excludedOpeningIds"], 32)
        prior_ids = set()
        for item, relative in zip(reference["priorScreens"], PRIOR_SCREEN_SUITES):
            self.assertEqual(item["openingPairs"], 32)
            prior_ids.update(
                opening.id
                for opening in load_opening_suite(
                    ROOT / relative, expected_suite="recovery-dev-v1"
                ).openings
            )
        selected_ids = set(reference["selectedOpeningIds"])
        self.assertEqual(len(selected_ids), OPENING_PAIRS)
        self.assertFalse(selected_ids & prior_ids)

    def test_frozen_contract_keeps_stage_b_closed(self) -> None:
        raw = (ROOT / CONTRACT).read_bytes()
        contract = json.loads(raw)
        core = dict(contract)
        stored = core.pop("contractSha256")
        self.assertEqual(raw, canonical_json_bytes(contract))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        _, opponent = _identities(contract)
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertFalse(contract["stageBTrainingAuthorized"])
        self.assertFalse(contract["screenTrajectoriesTrainingUseAllowed"])
        self.assertEqual(contract["workload"]["candidateSimulations"], CANDIDATE_SIMULATIONS)
        self.assertEqual(contract["workload"]["classicBudgetLimit"], CLASSIC_NODES)
        self.assertTrue((ROOT / CONTRACT).is_file())
        self.assertTrue((ROOT / SCREEN_SUITE).is_file())

        gate_raw = (ROOT / GATE).read_bytes()
        gate = json.loads(gate_raw)
        gate_core = dict(gate)
        gate_hash = gate_core.pop("gateSha256")
        self.assertEqual(gate_raw, canonical_json_bytes(gate))
        self.assertEqual(gate_hash, sha256_bytes(canonical_json_bytes(gate_core)))
        self.assertEqual(gate["outcomeBand"], "continue-stage-a")
        self.assertFalse(gate["stageBTrainingAuthorized"])


if __name__ == "__main__":
    unittest.main()
