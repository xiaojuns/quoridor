from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f23_stage_b import (
    CONFIG,
    EXECUTION,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_F22_GATE_SHA256,
    MAXIMUM_GENERATIONS,
    PREPARATION_REPORT,
)
from quoridor.run_manifest import (
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F23StageBTests(unittest.TestCase):
    def test_transition_is_bound_but_formal_stage_b_remains_closed(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(
            report["transitionEvidence"]["gateSha256"],
            EXPECTED_F22_GATE_SHA256,
        )
        self.assertEqual(
            report["transitionEvidence"]["outcomeBand"],
            "stage-b-transition-candidate",
        )
        self.assertTrue(report["stageBPilotAuthorized"])
        self.assertFalse(report["formalStageBTrainingAuthorized"])
        self.assertEqual(
            report["sourceCandidate"]["candidateModelSha256"],
            EXPECTED_CANDIDATE_SHA256,
        )

    def test_lineage_resets_optimizer_and_all_training_replay(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertFalse(report["lineage"]["optimizerInherited"])
        self.assertFalse(report["lineage"]["selfPlayReplayInherited"])
        self.assertFalse(report["lineage"]["stageAReplayImported"])
        self.assertFalse(report["lineage"]["screenTrajectoriesImported"])
        self.assertFalse(report["optimizerInput"]["challengeReplayConfigured"])
        self.assertEqual(report["optimizerInput"]["classicHardOptimizationWeight"], 0.0)
        self.assertTrue(report["optimizerInput"]["fixedClassicHoldoutDiagnosticOnly"])

    def test_config_is_one_generation_pure_selfplay_pilot(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.artifact_limit_bytes, MAXIMUM_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(config.training.learning_stage, "pure-self-play")
        self.assertIsNone(config.training.challenge_replay)
        self.assertIsNotNone(config.training.health_holdout)
        self.assertEqual(config.self_play.games, 128)
        self.assertEqual(config.initialization.recovery_generation, 1)
        self.assertEqual(config.initialization.recovery_step, 192)
        self.assertTrue(config.initialization.reset_optimizer)
        self.assertEqual(execution.stage, "phase4f-stage-b-pilot")
        self.assertEqual(execution.maximum_generations, MAXIMUM_GENERATIONS)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)

    def test_preparation_report_is_canonical_and_self_hashed(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))


if __name__ == "__main__":
    unittest.main()
