from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.openings import load_opening_suite, opening_position_orbit_digest
from quoridor.classic_bridge import CLASSIC_ENGINE
from quoridor.phase4f32_fresh_screen import FULL_SUITE as F32_FULL_SUITE
from quoridor.phase4f33_confirmation import (
    BUDGET_EXTENSION,
    BUDGET_EXTENSION_V2,
    CONFIRMATION_SUITE,
    CONTRACT,
    EXTENDED_CONFIRMATION_GPU_HOURS,
    FINAL_CONFIRMATION_GPU_HOURS,
    EXPECTED_F32_GATE_SHA256,
    GAMES,
    ISOLATION_AUDIT,
    MINIMUM_LOWER_BOUND,
    MINIMUM_WIN_RATE,
    OPENING_PAIRS,
    PREPARATION_REPORT,
    _f32_evidence,
    _identities,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F33ConfirmationTests(unittest.TestCase):
    def test_f32_gate_authorizes_confirmation(self) -> None:
        evidence = _f32_evidence(ROOT)
        self.assertEqual(evidence["gateSha256"], EXPECTED_F32_GATE_SHA256)
        self.assertEqual(
            evidence["outcomeBand"], "f31-fresh-confirmation-candidate"
        )
        self.assertEqual(evidence["statistics"]["candidateWins"], 34)

    def test_confirmation_suite_is_new_and_complete(self) -> None:
        confirmation = load_opening_suite(
            ROOT / CONFIRMATION_SUITE, expected_suite="recovery-fresh-dev-v1"
        )
        f32 = load_opening_suite(
            ROOT / F32_FULL_SUITE, expected_suite="recovery-fresh-dev-v1"
        )
        confirmation_orbits = {
            opening_position_orbit_digest(item) for item in confirmation.openings
        }
        f32_orbits = {opening_position_orbit_digest(item) for item in f32.openings}
        self.assertEqual(len(confirmation.openings), OPENING_PAIRS)
        self.assertEqual(len(confirmation_orbits), OPENING_PAIRS)
        self.assertFalse(confirmation_orbits & f32_orbits)

    def test_contract_freezes_formal_confirmation_policy(self) -> None:
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        candidate, opponent = _identities(contract)
        self.assertEqual(contract["workload"]["openingPairs"], OPENING_PAIRS)
        self.assertEqual(contract["workload"]["games"], GAMES)
        self.assertEqual(
            contract["decisionPolicy"]["minimumWinRate"], MINIMUM_WIN_RATE
        )
        self.assertEqual(
            contract["decisionPolicy"]["minimumOneSided95Lower"],
            MINIMUM_LOWER_BOUND,
        )
        self.assertEqual(candidate.engine, "native-mcts")
        self.assertEqual(opponent.engine, CLASSIC_ENGINE)
        self.assertTrue(contract["confirmationArenaAuthorized"])
        self.assertFalse(contract["formalPromotion"])
        self.assertFalse(contract["additionalTrainingAuthorized"])
        self.assertFalse(contract["arenaTrajectoriesTrainingUseAllowed"])
        self.assertEqual(contract["budget"]["artifactLimitBytes"], 12 * 1024**3)

    def test_audit_and_preparation_are_canonical_and_self_hashed(self) -> None:
        for relative, field in (
            (ISOLATION_AUDIT, "auditSha256"),
            (PREPARATION_REPORT, "reportSha256"),
        ):
            raw = (ROOT / relative).read_bytes()
            payload = json.loads(raw)
            core = dict(payload)
            stored = core.pop(field)
            self.assertEqual(raw, canonical_json_bytes(payload))
            self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        audit = json.loads((ROOT / ISOLATION_AUDIT).read_bytes())
        self.assertEqual(audit["freshOverlapCount"], 0)
        self.assertEqual(audit["confirmationOpeningPairs"], OPENING_PAIRS)
        self.assertTrue(audit["f32FullPoolExcluded"])

    def test_budget_extension_is_budget_only_and_preserves_arena(self) -> None:
        extension = json.loads((ROOT / BUDGET_EXTENSION).read_bytes())
        contract = json.loads((ROOT / CONTRACT).read_bytes())
        core = dict(extension)
        stored = core.pop("extensionSha256")
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        self.assertEqual(
            extension["originalContractSha256"], contract["contractSha256"]
        )
        self.assertEqual(
            extension["extendedLimitHours"], EXTENDED_CONFIRMATION_GPU_HOURS
        )
        self.assertEqual(extension["progressAtExtension"]["completedPairs"], 143)
        self.assertFalse(extension["semanticChangesAllowed"])

        final_extension = json.loads((ROOT / BUDGET_EXTENSION_V2).read_bytes())
        final_core = dict(final_extension)
        final_stored = final_core.pop("extensionSha256")
        self.assertEqual(
            final_stored, sha256_bytes(canonical_json_bytes(final_core))
        )
        self.assertEqual(
            final_extension["previousExtensionSha256"],
            extension["extensionSha256"],
        )
        self.assertEqual(
            final_extension["extendedLimitHours"], FINAL_CONFIRMATION_GPU_HOURS
        )
        self.assertEqual(final_extension["progressAtExtension"]["completedPairs"], 190)
        self.assertFalse(final_extension["semanticChangesAllowed"])


if __name__ == "__main__":
    unittest.main()
