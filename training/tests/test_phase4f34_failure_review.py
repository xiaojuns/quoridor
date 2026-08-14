from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f34_failure_review import (
    ANALYSIS,
    EVALUATION_PROTOCOL,
    EXPECTED_F21_CANDIDATE_SHA256,
    ISOLATION_AUDIT,
    REPORT,
    TRAINING_PROTOCOL,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


def load_self_hashed(path: Path, field: str) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    core = dict(payload)
    stored = core.pop(field)
    if raw != canonical_json_bytes(payload):
        raise AssertionError(f"{path} is not canonical JSON")
    if stored != sha256_bytes(canonical_json_bytes(core)):
        raise AssertionError(f"{path} self hash mismatch")
    return payload


class Phase4F34Tests(unittest.TestCase):
    def test_f33_failure_analysis_identifies_early_policy_weakness(self) -> None:
        analysis = load_self_hashed(ROOT / ANALYSIS, "analysisSha256")
        self.assertEqual(analysis["overall"]["games"], 400)
        self.assertEqual(analysis["overall"]["candidateWins"], 181)
        self.assertLess(analysis["openingBands"]["early-08-12"]["winRate"], 0.4)
        self.assertGreater(
            analysis["candidateWallRate"]["losses"],
            analysis["candidateWallRate"]["wins"],
        )
        self.assertFalse(analysis["dataUse"]["exactF33PositionsTrainingUseAllowed"])

    def test_guidance_is_disjoint_and_rollback_is_f21(self) -> None:
        audit = load_self_hashed(ROOT / ISOLATION_AUDIT, "auditSha256")
        report = load_self_hashed(ROOT / REPORT, "reportSha256")
        self.assertEqual(audit["splits"]["train"]["f33PositionOrbitOverlap"], 0)
        self.assertEqual(audit["splits"]["holdout"]["f33PositionOrbitOverlap"], 0)
        self.assertEqual(
            report["rollback"]["candidateModelSha256"],
            EXPECTED_F21_CANDIDATE_SHA256,
        )

    def test_protocols_do_not_authorize_training_or_reuse_evaluation(self) -> None:
        training = load_self_hashed(ROOT / TRAINING_PROTOCOL, "protocolSha256")
        evaluation = load_self_hashed(ROOT / EVALUATION_PROTOCOL, "protocolSha256")
        self.assertFalse(training["trainingAuthorized"])
        self.assertFalse(training["pilotAuthorized"])
        self.assertFalse(training["rewardContract"]["classicHardRewardShaping"])
        self.assertEqual(training["classicGuidance"]["challengeReplayFraction"], 0.5)
        self.assertEqual(training["stageB"]["pureSelfPlayChallengeReplayFraction"], 0.0)
        self.assertFalse(training["dataPolicy"]["f32AndF33TrajectoriesTrainingUseAllowed"])
        self.assertFalse(evaluation["arenaAuthorized"])
        self.assertTrue(
            evaluation["isolation"]["excludeAllHistoricalArenaTrajectoriesIncludingF32AndF33"]
        )


if __name__ == "__main__":
    unittest.main()
