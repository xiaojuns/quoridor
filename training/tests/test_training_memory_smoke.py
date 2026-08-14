from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from quoridor.network import NetworkConfig, PolicyValueNetwork
from quoridor.phase4_budget import (
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_BASELINE_REPORT_SHA256,
    PHASE4D_BASELINE_RUN_ID,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
    Phase4StageBudgetError,
    Phase4StageBudgetLedger,
)
from quoridor.run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    ImmutableArtifactExistsError,
    canonical_json_bytes,
    sha256_bytes,
)
from quoridor.trainer import AlphaZeroTrainer
from quoridor.training_memory_smoke import (
    REPORT_RESERVE_BYTES,
    RUN_KIND,
    SCHEMA_VERSION,
    TrainingMemorySmokeContract,
    TrainingMemorySmokeInvariantError,
    TrainingMemorySmokePreflightError,
    assert_report_write_reserve,
    audit_run_directory,
    build_synthetic_batch,
    collect_preflight,
    load_immutable_report,
    run_training_steps,
    write_immutable_report,
)


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int):
        if index != 0:
            raise AssertionError("unexpected device")
        return SimpleNamespace(
            name="Fake GPU",
            total_memory=24 * 1024**3,
            major=7,
            minor=5,
        )

    def mem_get_info(self, index: int):
        if index != 0:
            raise AssertionError("unexpected device")
        return 20 * 1024**3, 24 * 1024**3


def _stage_ledger(root: Path) -> Phase4StageBudgetLedger:
    ledger = Phase4StageBudgetLedger(root / "stage-budget-ledger.json")
    ledger.ensure_stage(
        PHASE4D_STAGE,
        PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
        PHASE4D_GPU_HOUR_LIMIT,
    )
    return ledger


class Phase4StageBudgetTests(unittest.TestCase):
    def test_stage_contract_round_trips_with_calibration_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-budget-ledger.json"
            ledger = Phase4StageBudgetLedger(path)
            record = ledger.ensure_stage(
                PHASE4D_STAGE,
                PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
                PHASE4D_GPU_HOUR_LIMIT,
            )
            self.assertEqual(record["baselineRunId"], PHASE4D_BASELINE_RUN_ID)
            self.assertEqual(
                record["baselineReportSha256"],
                PHASE4D_BASELINE_REPORT_SHA256,
            )

            reopened = Phase4StageBudgetLedger(path)
            usage = reopened.assert_available(
                PHASE4D_STAGE,
                PHASE4D_BASELINE_GLOBAL_GPU_HOURS + 2.5,
            )
            self.assertEqual(usage["usedHours"], 2.5)
            self.assertEqual(usage["remainingHours"], 7.5)
            self.assertTrue(usage["available"])

    def test_stage_budget_exhaustion_and_checksum_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-budget-ledger.json"
            ledger = _stage_ledger(Path(directory))
            with self.assertRaisesRegex(Phase4StageBudgetError, "exhausted"):
                ledger.assert_available(
                    PHASE4D_STAGE,
                    PHASE4D_BASELINE_GLOBAL_GPU_HOURS
                    + PHASE4D_GPU_HOUR_LIMIT,
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["stages"][PHASE4D_STAGE]["limitHours"] = 20.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Phase4StageBudgetError, "checksum"):
                Phase4StageBudgetLedger(path)

    def test_phase4d_stage_rejects_non_frozen_budget_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Phase4StageBudgetLedger(
                Path(directory) / "stage-budget-ledger.json"
            )
            with self.assertRaisesRegex(Phase4StageBudgetError, "frozen contract"):
                ledger.ensure_stage(
                    PHASE4D_STAGE,
                    baseline_global_hours=0.0,
                    limit_hours=1_000.0,
                )

    def test_phase4d_load_rejects_rehashed_non_frozen_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-budget-ledger.json"
            _stage_ledger(Path(directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["stages"][PHASE4D_STAGE]["limitHours"] = 1_000.0
            core = dict(payload)
            core.pop("ledgerSha256")
            payload["ledgerSha256"] = sha256_bytes(canonical_json_bytes(core))
            path.write_text(
                canonical_json_bytes(payload).decode("utf-8"), encoding="utf-8"
            )

            with self.assertRaisesRegex(Phase4StageBudgetError, "frozen contract"):
                Phase4StageBudgetLedger(path)

    def test_generic_stage_provenance_is_paired_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-budget-ledger.json"
            ledger = Phase4StageBudgetLedger(path)
            record = ledger.ensure_stage(
                "phase4e",
                baseline_global_hours=7.5,
                limit_hours=20.0,
                baseline_run_id="phase4e-calibration-1",
                baseline_report_sha256="a" * 64,
            )
            self.assertEqual(record["baselineRunId"], "phase4e-calibration-1")
            self.assertEqual(record["baselineReportSha256"], "a" * 64)

            reopened = Phase4StageBudgetLedger(path)
            self.assertEqual(
                reopened.ensure_stage(
                    "phase4e",
                    baseline_global_hours=7.5,
                    limit_hours=20.0,
                    baseline_run_id="phase4e-calibration-1",
                    baseline_report_sha256="a" * 64,
                ),
                record,
            )
            with self.assertRaisesRegex(
                Phase4StageBudgetError,
                "frozen contract",
            ):
                reopened.ensure_stage(
                    "phase4e",
                    baseline_global_hours=7.5,
                    limit_hours=20.0,
                    baseline_run_id="phase4e-calibration-2",
                    baseline_report_sha256="b" * 64,
                )

    def test_generic_stage_rejects_partial_baseline_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Phase4StageBudgetLedger(
                Path(directory) / "stage-budget-ledger.json"
            )
            with self.assertRaisesRegex(
                Phase4StageBudgetError,
                "provenance must be complete",
            ):
                ledger.ensure_stage(
                    "phase4e",
                    baseline_global_hours=7.5,
                    limit_hours=20.0,
                    baseline_run_id="phase4e-calibration-1",
                )


class TrainingMemoryContractTests(unittest.TestCase):
    def test_write_reserve_is_checked_before_persisting_control_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                TrainingMemorySmokePreflightError,
                "20 GiB",
            ):
                assert_report_write_reserve(
                    root,
                    disk_usage=lambda _path: SimpleNamespace(
                        free=DEFAULT_MINIMUM_FREE_DISK_BYTES
                        + REPORT_RESERVE_BYTES
                        - 1
                    ),
                )

    def test_write_reserve_accepts_explicit_frozen_twelve_gib_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "existing.bin").open("wb") as artifact:
                artifact.truncate(DEFAULT_ARTIFACT_LIMIT_BYTES + 1)
            resources = assert_report_write_reserve(
                root,
                artifact_limit_bytes=MAXIMUM_ARTIFACT_LIMIT_BYTES,
                disk_usage=lambda _path: SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES
                    + REPORT_RESERVE_BYTES
                ),
            )
            self.assertEqual(
                resources["artifactLimitBytes"],
                MAXIMUM_ARTIFACT_LIMIT_BYTES,
            )

    def test_write_reserve_rejects_unknown_artifact_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "8 or 12 GiB"):
                assert_report_write_reserve(
                    directory,
                    artifact_limit_bytes=10 * 1024**3,
                )

    def test_contract_is_frozen_to_production_network_and_batch(self) -> None:
        contract = TrainingMemorySmokeContract()
        network = contract.network_config()
        self.assertEqual(network.input_channels, 16)
        self.assertEqual(network.channels, 192)
        self.assertEqual(network.blocks, 16)
        self.assertEqual(contract.batch_size, 512)
        self.assertEqual(contract.total_steps, 4)
        self.assertTrue(contract.to_payload()["precision"]["ampRequired"])
        with self.assertRaisesRegex(ValueError, "batch_size is frozen"):
            replace(contract, batch_size=256)

    def test_fake_cuda_preflight_records_fixed_resource_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_ledger = _stage_ledger(root)
            result = collect_preflight(
                root,
                device_index=0,
                global_gpu_hours=PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
                stage_ledger=stage_ledger,
                cuda_api=FakeCuda(),
                disk_usage=lambda _path: SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES
                    + REPORT_RESERVE_BYTES
                    + 1
                ),
                find_executable=lambda name: (
                    "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
                ),
                baseline_verifier=lambda _root: {
                    "runId": PHASE4D_BASELINE_RUN_ID,
                    "reportSha256": PHASE4D_BASELINE_REPORT_SHA256,
                    "status": "verified",
                },
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["cuda"]["name"], "Fake GPU")
        self.assertEqual(result["cuda"]["computeCapability"], [7, 5])
        self.assertEqual(result["gpuHours"]["stage"]["limitHours"], 10.0)
        self.assertEqual(
            result["gpuHours"]["stageBaselineEvidence"]["runId"],
            PHASE4D_BASELINE_RUN_ID,
        )

    def test_preflight_rejects_missing_cuda_before_training(self) -> None:
        class NoCuda(FakeCuda):
            def is_available(self) -> bool:
                return False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                TrainingMemorySmokePreflightError, "CUDA is not available"
            ):
                collect_preflight(
                    root,
                    device_index=0,
                    global_gpu_hours=PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
                    stage_ledger=_stage_ledger(root),
                    cuda_api=NoCuda(),
                    baseline_verifier=lambda _root: {"status": "verified"},
                )


class TrainingMemoryExecutionTests(unittest.TestCase):
    def test_cpu_executes_real_policy_value_backward_optimizer_and_scheduler(self) -> None:
        model = PolicyValueNetwork(
            NetworkConfig(
                input_channels=16,
                channels=8,
                blocks=1,
                se_reduction=4,
                policy_channels=4,
                value_channels=4,
                value_hidden=16,
            )
        )
        trainer = AlphaZeroTrainer(
            model,
            learning_rate=0.001,
            weight_decay=0.0001,
            max_grad_norm=5.0,
            use_amp=True,
            schedule_steps=8,
            minimum_learning_rate=0.0001,
        )
        batch = build_synthetic_batch(
            batch_size=4,
            input_channels=16,
            board_size=9,
            action_size=209,
            device="cpu",
            seed=17,
        )

        result = run_training_steps(
            trainer,
            batch,
            warmup_steps=1,
            measured_steps=2,
            require_cuda_amp=False,
        )

        self.assertEqual(result["optimizerSteps"], 3)
        self.assertEqual(result["forwardCalls"], 3)
        self.assertEqual(
            result["actualOutputShapes"],
            {"policy": [4, 209], "value": [4]},
        )
        self.assertEqual(result["optimizer"]["kind"], "AdamW")
        self.assertGreater(result["optimizer"]["stateEntries"], 0)
        self.assertEqual(result["scheduler"]["kind"], "CosineAnnealingLR")
        self.assertGreater(result["metrics"]["gradientNorm"]["maximum"], 0)
        self.assertFalse(result["ampEnabled"])
        self.assertFalse(result["amp"]["scalerEnabled"])
        self.assertFalse(result["amp"]["cudaAutocastObserved"])
        self.assertEqual(result["amp"]["policyOutputDtypes"], ["float32"])

    def test_cuda_amp_requirement_rejects_a_mutated_declaration(self) -> None:
        model = PolicyValueNetwork(
            NetworkConfig(
                input_channels=16,
                channels=8,
                blocks=1,
                se_reduction=4,
                policy_channels=4,
                value_channels=4,
                value_hidden=16,
            )
        )
        trainer = AlphaZeroTrainer(model, use_amp=True)
        trainer.amp_enabled = True
        batch = build_synthetic_batch(
            batch_size=2,
            input_channels=16,
            board_size=9,
            action_size=209,
            device="cpu",
            seed=19,
        )

        with self.assertRaisesRegex(
            TrainingMemorySmokeInvariantError, "GradScaler"
        ):
            run_training_steps(
                trainer,
                batch,
                warmup_steps=1,
                measured_steps=1,
                require_cuda_amp=True,
            )

    def test_run_directory_audit_rejects_checkpoint_or_replay_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run-manifest.json").write_text("{}\n", encoding="utf-8")
            audit = audit_run_directory(
                root,
                expected_files=frozenset({"run-manifest.json"}),
            )
            self.assertEqual(audit["checkpointWrites"], 0)
            (root / "checkpoint.pt").write_bytes(b"forbidden")
            with self.assertRaisesRegex(
                TrainingMemorySmokeInvariantError, "forbidden"
            ):
                audit_run_directory(
                    root,
                    expected_files=frozenset({"run-manifest.json"}),
                )

    def test_report_is_checksummed_and_immutable(self) -> None:
        report = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": RUN_KIND,
            "runId": "unit-test",
            "status": "interrupted",
            "passed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_immutable_report(path, report)
            stored = load_immutable_report(path)
            self.assertEqual(stored["status"], "interrupted")
            self.assertEqual(len(stored["reportSha256"]), 64)
            with self.assertRaises(ImmutableArtifactExistsError):
                write_immutable_report(path, report)


if __name__ == "__main__":
    unittest.main()
