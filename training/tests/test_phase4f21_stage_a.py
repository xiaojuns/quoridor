from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f21_stage_a import (
    CAP_EXTENSION,
    CONFIG,
    EXECUTION,
    EXPECTED_F20_GATE_SHA256,
    MAXIMUM_GENERATIONS,
    PREPARATION_REPORT,
    RUN_DIR,
)
from quoridor.run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F21StageATests(unittest.TestCase):
    def test_preparation_binds_f20_and_keeps_stage_b_closed(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        self.assertEqual(
            report["historicalEvidence"]["f20GateSha256"],
            EXPECTED_F20_GATE_SHA256,
        )
        self.assertFalse(report["stageBTrainingAuthorized"])
        self.assertFalse(report["lineage"]["optimizerInherited"])
        self.assertFalse(report["lineage"]["selfPlayReplayInherited"])
        self.assertFalse(report["lineage"]["screenTrajectoriesImported"])

    def test_cap_extension_is_scoped_and_preserves_history(self) -> None:
        raw = (ROOT / CAP_EXTENSION).read_bytes()
        contract = json.loads(raw)
        core = dict(contract)
        stored = core.pop("capContractSha256")
        self.assertEqual(raw, canonical_json_bytes(contract))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        self.assertEqual(
            contract["previousDefaultArtifactLimitBytes"],
            DEFAULT_ARTIFACT_LIMIT_BYTES,
        )
        self.assertEqual(
            contract["artifactLimitBytes"],
            MAXIMUM_ARTIFACT_LIMIT_BYTES,
        )
        self.assertTrue(contract["historicalEightGiBContractsUnchanged"])
        self.assertEqual(contract["scope"]["runDir"], RUN_DIR.as_posix())

    def test_config_and_execution_freeze_final_guided_microblock(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.artifact_limit_bytes, 12 * 1024**3)
        self.assertEqual(config.training.learning_stage, "classic-hard-guided")
        self.assertEqual(config.training.challenge_replay.fraction, 0.4)
        self.assertEqual(config.initialization.recovery_generation, 1)
        self.assertEqual(config.initialization.recovery_step, 192)
        self.assertTrue(config.initialization.reset_optimizer)
        self.assertEqual(execution.stage, "phase4f-stage-a-continuation")
        self.assertEqual(execution.maximum_generations, MAXIMUM_GENERATIONS)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)

    def test_preparation_report_is_self_hashed(self) -> None:
        raw = (ROOT / PREPARATION_REPORT).read_bytes()
        report = json.loads(raw)
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(raw, canonical_json_bytes(report))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))


if __name__ == "__main__":
    unittest.main()
