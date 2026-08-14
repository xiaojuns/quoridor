from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f24_stage_b import (
    CONFIG,
    EXECUTION,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_F23_REVIEW_SHA256,
    MAXIMUM_GENERATIONS,
    PREPARATION_REPORT,
)
from quoridor.run_manifest import (
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F24StageBTests(unittest.TestCase):
    def test_preparation_binds_pilot_but_does_not_inherit_it(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(
            report["pilotEvidence"]["reviewSha256"],
            EXPECTED_F23_REVIEW_SHA256,
        )
        self.assertTrue(report["formalStageBTrainingAuthorized"])
        self.assertFalse(report["additionalStageBTrainingAuthorized"])
        self.assertEqual(report["lineage"]["source"], "f21-best-stage-a-candidate")
        self.assertFalse(report["lineage"]["f23PilotModelInherited"])
        self.assertFalse(report["lineage"]["f23PilotReplayInherited"])
        self.assertFalse(report["lineage"]["optimizerInherited"])
        self.assertFalse(report["lineage"]["stageAReplayImported"])
        self.assertFalse(report["lineage"]["screenTrajectoriesImported"])

    def test_config_is_two_generation_pure_selfplay_block(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.artifact_limit_bytes, MAXIMUM_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(config.training.learning_stage, "pure-self-play")
        self.assertIsNone(config.training.challenge_replay)
        self.assertIsNotNone(config.training.health_holdout)
        self.assertEqual(config.self_play.games, 128)
        self.assertEqual(
            config.initialization.artifact_sha256,
            EXPECTED_CANDIDATE_SHA256,
        )
        self.assertEqual(config.initialization.recovery_generation, 1)
        self.assertEqual(config.initialization.recovery_step, 192)
        self.assertTrue(config.initialization.reset_optimizer)
        self.assertEqual(execution.stage, "phase4f-stage-b")
        self.assertEqual(execution.maximum_generations, MAXIMUM_GENERATIONS)
        self.assertEqual(
            execution.stage_budget.baseline_global_hours,
            report_baseline := json.loads((ROOT / PREPARATION_REPORT).read_bytes())[
                "pilotEvidence"
            ]["baselineGlobalHours"],
        )
        self.assertEqual(report_baseline, 59.08072287766703)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)

    def test_classic_hard_is_diagnostic_only(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
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
