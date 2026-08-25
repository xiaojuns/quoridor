from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f39_early_curriculum import (
    AUDIT,
    CONTRACT,
    REPORT,
    TARGET_MIX,
    TARGET_POSITIONS,
    legal_action_bucket,
    ply_band,
    split_bucket,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F39EarlyCurriculumTests(unittest.TestCase):
    def test_absolute_ply_bands_and_legal_buckets(self) -> None:
        self.assertEqual(ply_band(0), "early")
        self.assertEqual(ply_band(15), "early")
        self.assertEqual(ply_band(16), "middle")
        self.assertEqual(ply_band(39), "middle")
        self.assertEqual(ply_band(40), "late")
        self.assertEqual(legal_action_bucket(4), "01-04")
        self.assertEqual(legal_action_bucket(5), "05-32")
        self.assertEqual(legal_action_bucket(64), "33-64")
        self.assertEqual(legal_action_bucket(65), "65-plus")
        with self.assertRaises(ValueError):
            ply_band(-1)
        with self.assertRaises(ValueError):
            legal_action_bucket(0)

    def test_whole_game_split_is_deterministic(self) -> None:
        assignments = [split_bucket(f"game-{index}") for index in range(2000)]
        self.assertEqual(assignments, [split_bucket(f"game-{index}") for index in range(2000)])
        self.assertGreater(assignments.count("train"), assignments.count("development"))
        self.assertGreater(assignments.count("train"), assignments.count("final"))

    def test_frozen_outputs_are_self_hashed_and_training_is_blocked(self) -> None:
        for relative, field in (
            (AUDIT, "auditSha256"),
            (CONTRACT, "contractSha256"),
            (REPORT, "reportSha256"),
        ):
            payload = json.loads((ROOT / relative).read_bytes())
            self.assertEqual((ROOT / relative).read_bytes(), canonical_json_bytes(payload))
            core = dict(payload)
            stored = core.pop(field)
            self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))

        contract = json.loads((ROOT / CONTRACT).read_bytes())
        self.assertEqual(contract["dataset"]["minimumPositions"], TARGET_POSITIONS)
        self.assertEqual(contract["dataset"]["trainingBatchMix"], TARGET_MIX)
        self.assertTrue(contract["targetProbe"]["authorized"])
        self.assertEqual(contract["targetProbe"]["finalPositions"], 0)
        self.assertFalse(contract["authorizations"]["datasetGenerationAuthorized"])
        self.assertFalse(contract["authorizations"]["cudaTrainingAuthorized"])
        self.assertFalse(contract["authorizations"]["formalTrainingAuthorized"])
        self.assertFalse(contract["valueTarget"]["environmentRewardShaping"])

        report = json.loads((ROOT / REPORT).read_bytes())
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["cudaAccessed"])
        self.assertFalse(report["trainingStarted"])

    def test_seed_holdout_is_sealed_and_orbit_disjoint(self) -> None:
        audit = json.loads((ROOT / AUDIT).read_bytes())
        isolation = audit["isolation"]
        self.assertEqual(isolation["trainHoldoutOrbitOverlap"], 0)
        self.assertFalse(isolation["sealedHoldoutMayTrain"])
        self.assertFalse(isolation["rootCauseDiagnosticPositionsMayTrain"])
        self.assertFalse(isolation["arenaTrajectoriesMayTrain"])
        self.assertGreater(audit["capacityGap"]["minimumAdditionalTrainPositions"], 0)


if __name__ == "__main__":
    unittest.main()
