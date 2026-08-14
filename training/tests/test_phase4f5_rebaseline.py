from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.phase4_budget import PHASE4F5_STAGE
from quoridor.phase4f5_rebaseline import (
    CALIBRATION_GPU_HOURS,
    DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS,
    FRESH_FINAL_RESERVE_GPU_HOURS,
    MEMORY_ROOT,
    RECOVERY_TRAINING_GPU_HOURS,
    _calibration_execution_contract,
    _effective_config_sha256,
    _formal_payload,
    _pilot_payload,
    _self_hashed,
    _valid_memory_report,
)
from quoridor.run_manifest import canonical_json_bytes
from quoridor.config import load_experiment_config
from quoridor.run_manifest import fingerprint_config
from quoridor.config import seed_everything
from quoridor.network import PolicyValueNetwork
from quoridor.phase4_checkpoint import canonical_model_state_sha256
from quoridor.phase4_training import _network_config


ROOT = Path(__file__).resolve().parents[2]


class Phase4F5RebaselineTests(unittest.TestCase):
    def test_pilot_only_changes_identity_and_run_directory(self) -> None:
        formal = _formal_payload(ROOT)
        pilot = _pilot_payload(ROOT)
        self.assertEqual(pilot.pop("name"), "phase4f5-recovery-pilot-v3")
        self.assertEqual(
            pilot.pop("runDir"),
            "training/runs/phase4/phase4f5/rebaseline/pilot-v3",
        )
        formal.pop("name")
        formal.pop("runDir")
        self.assertEqual(pilot, formal)

    def test_calibration_execution_contract_is_bounded(self) -> None:
        contract = _calibration_execution_contract(43.0, "a" * 64)
        self.assertEqual(contract.stage, PHASE4F5_STAGE)
        self.assertFalse(contract.formal_promotion)
        self.assertEqual(contract.maximum_generations, 1)
        self.assertEqual(contract.stage_budget.baseline_global_hours, 43.0)
        self.assertEqual(contract.stage_budget.limit_hours, CALIBRATION_GPU_HOURS)

    def test_effective_config_fingerprint_normalizes_relative_run_dir(self) -> None:
        config = load_experiment_config(
            ROOT / "training/configs/phase4f-recovery-pilot-v1.json"
        )
        resolved = replace(config, run_dir=(ROOT / config.run_dir).resolve())
        self.assertEqual(
            _effective_config_sha256(config, ROOT),
            fingerprint_config(resolved).sha256,
        )

    def test_recovery_base_hash_matches_its_seeded_network(self) -> None:
        payload = _formal_payload(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_bytes(canonical_json_bytes(payload))
            config = load_experiment_config(path)
        seed_everything(config.seed)
        model = PolicyValueNetwork(_network_config(config))
        self.assertEqual(
            canonical_model_state_sha256(model),
            config.initialization.expected_base_model_sha256,
        )

    def test_budget_reserves_fit_the_global_limit_at_frozen_baseline(self) -> None:
        total = (
            43.14977626422679
            + CALIBRATION_GPU_HOURS
            + RECOVERY_TRAINING_GPU_HOURS
            + DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS
            + FRESH_FINAL_RESERVE_GPU_HOURS
        )
        self.assertLessEqual(total, 72.0)

    def test_memory_report_with_missing_gates_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            report_path = workspace / MEMORY_ROOT / "bad" / "report.json"
            report_path.parent.mkdir(parents=True)
            report = _self_hashed(
                {
                    "kind": "phase4f5-cuda-amp-memory-smoke",
                    "status": "completed",
                    "passed": True,
                    "preparationSha256": "p",
                    "sourceSha256": "s",
                    "nativeBackendSha256": "n",
                    "gates": {},
                },
                "reportSha256",
            )
            report_path.write_bytes(canonical_json_bytes(report))
            prepared = {
                "preparationSha256": "p",
                "source": {"sha256": "s"},
                "nativeBackendSha256": "n",
            }
            self.assertIsNone(_valid_memory_report(workspace, prepared))


if __name__ == "__main__":
    unittest.main()
