from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f19_stage_a import (
    CONFIG,
    EXECUTION,
    PREPARATION_REPORT,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F19StageATests(unittest.TestCase):
    def test_historical_preparation_authorized_only_stage_a(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        self.assertTrue(report["formalStageATrainingAuthorized"])
        self.assertFalse(report["stageBTrainingAuthorized"])
        self.assertTrue(report["independentLineage"]["resetOptimizer"])
        self.assertFalse(report["independentLineage"]["pilotReplayInherited"])
        self.assertEqual(len(report["source"]["sha256"]), 64)

        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))

    def test_execution_freezes_two_generations_and_two_gpu_hours(self) -> None:
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(execution.stage, "phase4f-stage-a")
        self.assertEqual(execution.status, "frozen")
        self.assertEqual(execution.maximum_generations, 2)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)
        self.assertFalse(execution.formal_promotion)

    def test_formal_config_is_guided_and_recovers_generation_11(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        self.assertEqual(config.training.learning_stage, "classic-hard-guided")
        self.assertEqual(config.training.challenge_replay.fraction, 0.4)
        self.assertEqual(config.training.training_steps, 96)
        self.assertEqual(config.self_play.games, 128)
        self.assertEqual(config.initialization.recovery_generation, 11)
        self.assertTrue(config.initialization.reset_optimizer)


if __name__ == "__main__":
    unittest.main()
