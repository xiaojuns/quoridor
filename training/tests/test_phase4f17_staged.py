from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.phase4f17_staged import (
    ARENA_RESULT_PATHS,
    CONTRACT_PATH,
    PROTOCOL_PATH,
    REPORT_PATH,
)
from quoridor.run_manifest import (
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


def load_self_hashed(path: Path, field: str) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    core = dict(payload)
    stored = core.pop(field)
    if raw != canonical_json_bytes(payload):
        raise AssertionError(f"{path} is not canonical JSON")
    if stored != sha256_bytes(canonical_json_bytes(core)):
        raise AssertionError(f"{path} has an invalid self hash")
    return payload


class Phase4F17StagedTests(unittest.TestCase):
    def test_dataset_contract_binds_all_prior_arena_results(self) -> None:
        contract = load_self_hashed(ROOT / CONTRACT_PATH, "contractSha256")
        evidence = contract["excludedTrajectoryEvidence"]
        self.assertEqual(len(evidence), len(ARENA_RESULT_PATHS))
        self.assertEqual(
            [item["path"] for item in evidence],
            [path.as_posix() for path in ARENA_RESULT_PATHS],
        )
        self.assertEqual(contract["exclusionSet"]["orbits"], 72_831)
        self.assertEqual(contract["datasetEvidence"]["holdoutPositions"], 2_356)
        self.assertTrue(contract["trainingUseAllowed"])

    def test_two_stage_protocol_removes_teacher_influence_in_stage_b(self) -> None:
        protocol = load_self_hashed(ROOT / PROTOCOL_PATH, "protocolSha256")
        self.assertFalse(protocol["rewardContract"]["classicHardRewardShaping"])
        self.assertEqual(protocol["stageA"]["challengeReplayFraction"], 0.4)
        self.assertEqual(protocol["stageB"]["challengeReplayFraction"], 0.0)
        self.assertEqual(protocol["stageB"]["classicHardOptimizationWeight"], 0.0)
        self.assertTrue(protocol["stageB"]["fixedClassicHoldoutDiagnosticOnly"])
        self.assertIn("latest-pure-self-play", protocol["stageB"]["selection"])
        self.assertTrue(protocol["transitionGate"]["resetsOptimizerAndReplay"])

    def test_preparation_report_remains_valid_historical_cpu_only_evidence(self) -> None:
        report = load_self_hashed(ROOT / REPORT_PATH, "reportSha256")
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["cudaAccessed"])
        self.assertFalse(report["trainingAuthorized"])
        self.assertEqual(len(report["source"]["sha256"]), 64)
        self.assertEqual(report["datasetAudit"]["evaluationOverlap"], 0)
        self.assertEqual(report["datasetAudit"]["trainHoldoutOverlap"], 0)
        self.assertEqual(
            report["datasetContract"]["fileSha256"],
            sha256_file(ROOT / CONTRACT_PATH),
        )


if __name__ == "__main__":
    unittest.main()
