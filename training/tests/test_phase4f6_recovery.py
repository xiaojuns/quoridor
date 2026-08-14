from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quoridor.calibration import GpuBudgetLedger
from quoridor.config import load_experiment_config
from quoridor.phase4_budget import PHASE4F_RECOVERY_STAGE
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f6_recovery import (
    FORMAL_CONFIG,
    FORMAL_EXECUTION,
    FORMAL_MAXIMUM_GENERATIONS,
    FORMAL_PURPOSE,
    F5_RUN_ID,
    RECOVERY_GPU_HOURS,
    _load_canonical,
    _verify_ledger_append_only,
    verify_phase4f5_history,
)
from quoridor.run_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]


class Phase4F6RecoveryTests(unittest.TestCase):
    def test_completed_f5_history_authorizes_the_exact_formal_inputs(self) -> None:
        evidence = verify_phase4f5_history(ROOT)
        self.assertEqual(evidence["status"], "verified")
        self.assertEqual(evidence["runId"], F5_RUN_ID)
        self.assertEqual(evidence["limitHours"], RECOVERY_GPU_HOURS)
        self.assertEqual(evidence["formalConfigPath"], FORMAL_CONFIG.as_posix())
        self.assertEqual(
            evidence["executionContractPath"], FORMAL_EXECUTION.as_posix()
        )

    def test_formal_execution_policy_is_bounded_and_not_a_promotion(self) -> None:
        contract = load_phase4_execution_contract(ROOT / FORMAL_EXECUTION)
        self.assertEqual(contract.stage, PHASE4F_RECOVERY_STAGE)
        self.assertEqual(contract.purpose, FORMAL_PURPOSE)
        self.assertFalse(contract.formal_promotion)
        self.assertEqual(
            contract.maximum_generations, FORMAL_MAXIMUM_GENERATIONS
        )
        self.assertEqual(contract.stage_budget.limit_hours, RECOVERY_GPU_HOURS)

    def test_formal_config_uses_the_isolated_recovery_lineage(self) -> None:
        config = load_experiment_config(ROOT / FORMAL_CONFIG)
        self.assertEqual(config.name, "phase4f-recovery-v2")
        self.assertEqual(
            config.run_dir,
            Path("training/runs/phase4/phase4f/recovery-v2"),
        )
        self.assertEqual(config.self_play.games, 128)
        self.assertEqual(config.self_play.actors, 64)
        self.assertEqual(config.mcts.simulations, 512)
        self.assertEqual(config.training.training_steps, 96)
        self.assertEqual(config.training.batch_size, 512)
        self.assertEqual(config.training.challenge_replay.fraction, 0.25)

    def test_canonical_loader_rejects_noncanonical_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not canonical JSON"):
                _load_canonical(path, "fixture")

    def test_gpu_ledger_verifier_accepts_only_append_only_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            path = workspace / "training/runs/phase4/gpu-budget-ledger.json"
            ledger = GpuBudgetLedger(path)
            ledger.record_external_hours(4.0)
            snapshot = {
                "path": "training/runs/phase4/gpu-budget-ledger.json",
                "totalGpuHours": ledger.total_hours,
                "externalHoursFloor": ledger.external_hours_floor,
                "runs": {},
            }
            ledger.record_external_hours(5.0)
            self.assertEqual(
                _verify_ledger_append_only(
                    workspace,
                    snapshot,
                    baseline_hours=4.0,
                    maximum_hours=6.0,
                ),
                5.0,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["externalHoursFloor"] = 3.0
            core = dict(payload)
            core.pop("ledgerSha256")
            from quoridor.run_manifest import sha256_bytes

            payload["ledgerSha256"] = sha256_bytes(canonical_json_bytes(core))
            path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(RuntimeError, "floor regressed"):
                _verify_ledger_append_only(
                    workspace,
                    snapshot,
                    baseline_hours=4.0,
                    maximum_hours=6.0,
                )


if __name__ == "__main__":
    unittest.main()
