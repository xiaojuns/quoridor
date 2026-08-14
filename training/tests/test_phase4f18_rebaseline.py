from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f18_rebaseline import (
    PILOT_CONFIG,
    PILOT_EXECUTION,
    PREPARATION_REPORT,
    STAGE_A_CONFIG,
    STAGE_A_EXECUTION,
    _verified_preparation,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F18RebaselineTests(unittest.TestCase):
    def test_historical_preparation_keeps_formal_stage_closed(self) -> None:
        payload = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        report = payload
        self.assertTrue(report["cpuOnly"])
        self.assertTrue(report["cudaPilotAuthorized"])
        self.assertFalse(report["formalStageATrainingAuthorized"])
        self.assertEqual(report["globalGpuHourLimit"], 120.0)
        self.assertEqual(len(report["source"]["sha256"]), 64)

        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        payload = json.loads(raw)
        core = dict(payload)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(payload))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))

    def test_pilot_is_one_generation_but_formal_stage_a_is_two(self) -> None:
        pilot = load_phase4_execution_contract(ROOT / PILOT_EXECUTION)
        formal = load_phase4_execution_contract(ROOT / STAGE_A_EXECUTION)
        self.assertEqual(pilot.stage, "phase4f18-pilot")
        self.assertEqual(pilot.status, "frozen")
        self.assertEqual(pilot.maximum_generations, 1)
        self.assertEqual(pilot.stage_budget.limit_hours, 2.0)
        self.assertEqual(formal.stage, "phase4f-stage-a")
        self.assertEqual(formal.status, "prepared")
        self.assertEqual(formal.maximum_generations, 2)
        self.assertFalse(formal.budget_frozen)

    def test_configs_use_identical_guided_training_semantics(self) -> None:
        pilot = load_experiment_config(ROOT / PILOT_CONFIG)
        formal = load_experiment_config(ROOT / STAGE_A_CONFIG)
        for config in (pilot, formal):
            self.assertEqual(config.training.learning_stage, "classic-hard-guided")
            self.assertEqual(config.training.challenge_replay.fraction, 0.4)
            self.assertEqual(config.training.learning_rate, 0.0001)
            self.assertEqual(config.training.minimum_learning_rate, 0.00001)
            self.assertEqual(config.training.max_checkpoints, 3)
            self.assertIsNotNone(config.training.health_holdout)
            self.assertEqual(config.initialization.mode, "recovery")
            self.assertEqual(config.initialization.recovery_generation, 11)
            self.assertEqual(config.initialization.recovery_step, 1152)
            self.assertTrue(config.initialization.reset_optimizer)
        self.assertNotEqual(pilot.run_dir, formal.run_dir)


if __name__ == "__main__":
    unittest.main()
