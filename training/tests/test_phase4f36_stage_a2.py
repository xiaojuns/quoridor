from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_budget import PHASE4F36_PILOT_STAGE
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f36_stage_a2 import (
    CONFIG,
    EXECUTION,
    F35_SOURCE_GIT_COMMIT,
    REPORT,
    RUN_DIR,
)
from quoridor.run_manifest import MAXIMUM_ARTIFACT_LIMIT_BYTES, canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F36StageA2Tests(unittest.TestCase):
    def test_recovery_report_freezes_remaining_budget_and_ledger(self) -> None:
        report = json.loads((ROOT / REPORT).read_bytes())
        self.assertEqual((ROOT / REPORT).read_bytes(), canonical_json_bytes(report))
        self.assertEqual(
            report["reportSha256"],
            "49a7c9027756c80e5baec26e0e378c55c68cfe31e40ddda6d51a39e14de21bcd",
        )
        self.assertTrue(report["trainingAuthorized"])
        self.assertFalse(report["formalPromotionAuthorized"])
        self.assertEqual(report["pilot"]["remainingStageGpuHours"], 1.99923)
        self.assertEqual(report["historicalF35"]["maximumGlobalGpuHours"], 78.14802950652549)
        runs = report["gpuLedger"]["runs"]
        self.assertTrue(runs)
        self.assertFalse(any(record["status"] == "running" for record in runs.values()))
        self.assertIn(
            "phase4d-training-memory-smoke-20260814-122855-287076",
            runs,
        )

    def test_v2_config_and_execution_are_isolated(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.name, "phase4f36-stage-a2-pilot-v2")
        self.assertEqual(config.run_dir, RUN_DIR)
        self.assertEqual(config.artifact_limit_bytes, MAXIMUM_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(config.training.challenge_replay.fraction, 0.5)
        self.assertIsNotNone(config.training.challenge_replay.stratified_sampling)
        self.assertEqual(execution.stage, PHASE4F36_PILOT_STAGE)
        self.assertEqual(execution.maximum_generations, 1)
        self.assertEqual(execution.stage_budget.limit_hours, 1.99923)

    def test_launcher_passes_frozen_gate_inputs_and_is_resumable(self) -> None:
        launcher = (
            ROOT / "scripts/run-phase4f36-stage-a2-pilot-v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '--replay-input-gate --config "${CONFIG}"',
            launcher,
        )
        self.assertIn('--execution-contract "${EXECUTION}"', launcher)
        self.assertIn("RESUME_ARGS+=(--resume)", launcher)
        self.assertEqual(
            F35_SOURCE_GIT_COMMIT,
            "3d7fd8736b5075012e1e40cfa302024471558522",
        )


if __name__ == "__main__":
    unittest.main()
