from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f27_stage_b import (
    CONFIG,
    EXECUTION,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_F26_GATE_SHA256,
    LEARNING_RATE,
    MAXIMUM_GENERATIONS,
    PREPARATION_REPORT,
    SOURCE_ARCHIVE,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F27StageBTests(unittest.TestCase):
    def test_preparation_binds_f26_and_f24_candidate(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(
            report["transitionEvidence"]["gateSha256"],
            EXPECTED_F26_GATE_SHA256,
        )
        self.assertEqual(
            report["initialization"]["artifactSha256"],
            EXPECTED_CANDIDATE_SHA256,
        )
        self.assertTrue(report["conservativeStageBTrainingAuthorized"])
        self.assertFalse(report["additionalStageBTrainingAuthorized"])
        self.assertEqual(report["lineage"]["source"], "f24-latest-stage-b-candidate")
        self.assertFalse(report["lineage"]["f24ReplayInherited"])
        self.assertFalse(report["lineage"]["optimizerInherited"])
        self.assertFalse(report["lineage"]["screenTrajectoriesImported"])

    def test_config_is_low_rate_two_generation_pure_selfplay(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.training.learning_stage, "pure-self-play")
        self.assertIsNone(config.training.challenge_replay)
        self.assertIsNotNone(config.training.health_holdout)
        self.assertEqual(config.training.learning_rate, LEARNING_RATE)
        self.assertEqual(config.self_play.games, 128)
        self.assertEqual(config.training.training_steps, 96)
        self.assertEqual(config.initialization.artifact_sha256, EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(config.initialization.recovery_generation, 1)
        self.assertEqual(config.initialization.recovery_step, 192)
        self.assertTrue(config.initialization.reset_optimizer)
        self.assertEqual(execution.stage, "phase4f-stage-b-continuation")
        self.assertEqual(execution.maximum_generations, MAXIMUM_GENERATIONS)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)

    def test_archive_references_source_files_without_copying(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        archive = json.loads((ROOT / SOURCE_ARCHIVE).read_bytes())
        self.assertFalse(report["archive"]["modelFilesCopied"])
        self.assertEqual(archive["candidate"]["candidateModelSha256"], EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(set(archive["files"]), {"candidateModel", "checkpoint"})

    def test_classic_is_diagnostic_only(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(report["optimizerInput"]["learningRate"], LEARNING_RATE)
        self.assertFalse(report["optimizerInput"]["challengeReplayConfigured"])
        self.assertEqual(report["optimizerInput"]["classicHardOptimizationWeight"], 0.0)
        self.assertTrue(report["optimizerInput"]["fixedClassicHoldoutDiagnosticOnly"])

    def test_preparation_report_is_canonical_and_self_hashed(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))


if __name__ == "__main__":
    unittest.main()
