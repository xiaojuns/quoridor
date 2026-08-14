from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quoridor.phase4_budget import (
    PHASE4E1_BASELINE_KIND,
    Phase4StageBudgetError,
    verify_phase4e_baseline_report,
)
from quoridor.phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_SCHEMA_VERSION,
    PHASE4_EXECUTION_STATUS_FROZEN,
    PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT,
)
from quoridor.phase4e1_contract import (
    FREEZE_MAXIMUM_GENERATIONS,
    FREEZE_RESERVE_GPU_HOURS,
)
from quoridor.phase4_training import (
    Phase4TrainingError,
    _verify_phase4e_training_inputs,
    run_phase4_training,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "phase4e1-calibration-test"
BASELINE_HOURS = 10.5
LIMIT_HOURS = 20.0
CONFIG_SHA256 = "1" * 64
SOURCE_SHA256 = "2" * 64
NATIVE_SHA256 = "3" * 64
BOOTSTRAP_BASELINE = {
    "manifestPath": "training/runs/phase4/baselines/bootstrap/manifest.json",
    "manifestSelfSha256": "4" * 64,
    "modelPath": "training/runs/phase4/baselines/bootstrap/model.pt",
    "modelFileSha256": "5" * 64,
}
TRAINING_LIMIT_HOURS = (
    PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT
    - BASELINE_HOURS
    - FREEZE_RESERVE_GPU_HOURS
)


def _report_core() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": PHASE4E1_BASELINE_KIND,
        "status": "completed",
        "decisionEligible": True,
        "formalPromotion": False,
        "trainingPerformed": False,
        "checkpointsWritten": 0,
        "trainingAuthorized": False,
        "runId": RUN_ID,
        "resourceUsage": {
            "globalGpuHours": BASELINE_HOURS,
        },
        "recommendedStageBudget": {
            "baselineGlobalHours": BASELINE_HOURS,
            "limitHours": LIMIT_HOURS,
        },
        "approvedWorkloads": [CONFIG_SHA256],
        "sourceSha256": SOURCE_SHA256,
        "nativeBackendSha256": NATIVE_SHA256,
        "bootstrapBaseline": BOOTSTRAP_BASELINE,
    }


def _self_hashed(core: dict[str, object]) -> dict[str, object]:
    return {
        **core,
        "reportSha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _write_report(
    control_root: Path,
    payload: dict[str, object],
    *,
    canonical: bool = True,
) -> Path:
    path = control_root / "phase4e1" / "calibration" / RUN_ID / "report.json"
    path.parent.mkdir(parents=True)
    encoded = (
        canonical_json_bytes(payload)
        if canonical
        else json.dumps(payload, indent=2).encode("utf-8")
    )
    path.write_bytes(encoded)
    return path


def _verify(control_root: Path, payload: dict[str, object]) -> dict[str, object]:
    return verify_phase4e_baseline_report(
        control_root,
        baseline_run_id=RUN_ID,
        baseline_report_sha256=payload["reportSha256"],
        baseline_global_hours=BASELINE_HOURS,
        limit_hours=LIMIT_HOURS,
    )


def _write_frozen_phase4e_contract(workspace: Path) -> Path:
    path = workspace / "phase4e-execution.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": PHASE4_EXECUTION_SCHEMA_VERSION,
                "stage": "phase4e",
                "status": PHASE4_EXECUTION_STATUS_FROZEN,
                "purpose": "phase4e-bootstrap-and-formal-promotion",
                "formalPromotion": True,
                "maximumGenerations": FREEZE_MAXIMUM_GENERATIONS,
                "stageBudget": {
                    "status": PHASE4_BUDGET_STATUS_FROZEN,
                    "baselineGlobalHours": BASELINE_HOURS,
                    "limitHours": TRAINING_LIMIT_HOURS,
                    "baselineRunId": RUN_ID,
                    "baselineReportSha256": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class Phase4EBaselineReportTests(unittest.TestCase):
    def test_accepts_canonical_self_hashed_decision_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            control_root = workspace / "training/runs/phase4"
            payload = _self_hashed(_report_core())
            path = _write_report(control_root, payload)

            with patch(
                "quoridor.phase4e1_calibration."
                "verify_phase4e1_decision_report",
                return_value=payload,
            ) as complete_verifier:
                evidence = _verify(control_root, payload)

            self.assertEqual(evidence["path"], str(path))
            self.assertEqual(evidence["runId"], RUN_ID)
            self.assertEqual(evidence["reportSha256"], payload["reportSha256"])
            self.assertEqual(evidence["globalGpuHours"], BASELINE_HOURS)
            self.assertEqual(evidence["limitHours"], LIMIT_HOURS)
            self.assertEqual(evidence["approvedWorkloads"], [CONFIG_SHA256])
            self.assertEqual(evidence["sourceSha256"], SOURCE_SHA256)
            self.assertEqual(evidence["nativeBackendSha256"], NATIVE_SHA256)
            self.assertEqual(
                evidence["bootstrapBaseline"],
                BOOTSTRAP_BASELINE,
            )
            self.assertIsNot(
                evidence["bootstrapBaseline"],
                payload["bootstrapBaseline"],
            )
            self.assertEqual(evidence["status"], "verified")
            complete_verifier.assert_called_once_with(
                workspace_root=workspace.resolve(),
                report_path=path,
                verify_ledger=True,
            )

    def test_minimal_self_hashed_report_cannot_bypass_full_e1_verifier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            control_root = workspace / "training/runs/phase4"
            payload = _self_hashed(_report_core())
            path = _write_report(control_root, payload)

            with (
                patch(
                    "quoridor.phase4e1_calibration."
                    "verify_phase4e1_decision_report",
                    side_effect=RuntimeError("incomplete E1 evidence tree"),
                ) as complete_verifier,
                self.assertRaisesRegex(
                    Phase4StageBudgetError,
                    "failed complete E1 verification",
                ),
            ):
                _verify(control_root, payload)

            complete_verifier.assert_called_once_with(
                workspace_root=workspace.resolve(),
                report_path=path,
                verify_ledger=True,
            )

    def test_complete_verifier_must_return_bootstrap_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            control_root = workspace / "training/runs/phase4"
            payload = _self_hashed(_report_core())
            _write_report(control_root, payload)
            incomplete = dict(payload)
            incomplete.pop("bootstrapBaseline")

            with (
                patch(
                    "quoridor.phase4e1_calibration."
                    "verify_phase4e1_decision_report",
                    return_value=incomplete,
                ),
                self.assertRaisesRegex(
                    Phase4StageBudgetError,
                    "lost its Bootstrap evidence",
                ),
            ):
                _verify(control_root, payload)

    def test_rejects_noncanonical_or_tampered_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            payload = _self_hashed(_report_core())
            _write_report(control_root, payload, canonical=False)
            with self.assertRaisesRegex(Phase4StageBudgetError, "canonical"):
                _verify(control_root, payload)

        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            payload = _self_hashed(_report_core())
            path = _write_report(control_root, payload)
            path.write_bytes(b'{"reportSha256":"' + b"f" * 64 + b'","x":NaN}\n')
            with self.assertRaisesRegex(Phase4StageBudgetError, "canonical"):
                _verify(control_root, payload)

        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            payload = _self_hashed(_report_core())
            payload["status"] = "failed"
            _write_report(control_root, payload)
            with self.assertRaisesRegex(Phase4StageBudgetError, "self hash"):
                _verify(control_root, payload)

        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            payload = _self_hashed(_report_core())
            _write_report(control_root, payload)
            with self.assertRaisesRegex(Phase4StageBudgetError, "contract"):
                verify_phase4e_baseline_report(
                    control_root,
                    baseline_run_id=RUN_ID,
                    baseline_report_sha256="f" * 64,
                    baseline_global_hours=BASELINE_HOURS,
                    limit_hours=LIMIT_HOURS,
                )

    def test_rejects_ineligible_or_gpu_work_report(self) -> None:
        invalid_values = {
            "schemaVersion": 2,
            "kind": "another-kind",
            "status": "running",
            "decisionEligible": False,
            "formalPromotion": True,
            "trainingPerformed": True,
            "checkpointsWritten": 1,
            "trainingAuthorized": True,
            "runId": "another-run",
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                control_root = Path(directory)
                core = _report_core()
                core[field] = value
                payload = _self_hashed(core)
                _write_report(control_root, payload)
                with self.assertRaises(Phase4StageBudgetError):
                    _verify(control_root, payload)

    def test_rejects_budget_or_identity_mismatch(self) -> None:
        mutations = (
            ("resourceUsage", {"globalGpuHours": BASELINE_HOURS + 1}),
            (
                "recommendedStageBudget",
                {
                    "baselineGlobalHours": BASELINE_HOURS + 1,
                    "limitHours": LIMIT_HOURS,
                },
            ),
            (
                "recommendedStageBudget",
                {
                    "baselineGlobalHours": BASELINE_HOURS,
                    "limitHours": LIMIT_HOURS + 1,
                },
            ),
            ("approvedWorkloads", []),
            ("approvedWorkloads", [CONFIG_SHA256, CONFIG_SHA256]),
            ("approvedWorkloads", ["not-a-sha"]),
            ("sourceSha256", "not-a-sha"),
            ("nativeBackendSha256", "not-a-sha"),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                control_root = Path(directory)
                core = _report_core()
                core[field] = value
                payload = _self_hashed(core)
                _write_report(control_root, payload)
                with self.assertRaises(Phase4StageBudgetError):
                    _verify(control_root, payload)

    def test_rejects_path_traversal_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            with self.assertRaisesRegex(Phase4StageBudgetError, "run id"):
                verify_phase4e_baseline_report(
                    control_root,
                    baseline_run_id="../escape",
                    baseline_report_sha256="f" * 64,
                    baseline_global_hours=BASELINE_HOURS,
                    limit_hours=LIMIT_HOURS,
                )

        with tempfile.TemporaryDirectory() as directory:
            control_root = Path(directory)
            payload = _self_hashed(_report_core())
            target = control_root / "outside.json"
            target.write_bytes(canonical_json_bytes(payload))
            path = (
                control_root
                / "phase4e1"
                / "calibration"
                / RUN_ID
                / "report.json"
            )
            path.parent.mkdir(parents=True)
            path.symlink_to(target)
            with self.assertRaisesRegex(Phase4StageBudgetError, "symlink"):
                _verify(control_root, payload)


class Phase4ETrainingProvenanceTests(unittest.TestCase):
    def test_training_inputs_must_match_every_approved_identity(self) -> None:
        evidence = {
            "approvedWorkloads": [CONFIG_SHA256],
            "sourceSha256": SOURCE_SHA256,
            "nativeBackendSha256": NATIVE_SHA256,
        }
        _verify_phase4e_training_inputs(
            evidence,
            config_sha256=CONFIG_SHA256,
            source_sha256=SOURCE_SHA256,
            native_backend_sha256=NATIVE_SHA256,
        )

        mismatches = (
            ("config_sha256", "4" * 64, "workload"),
            ("source_sha256", "4" * 64, "source"),
            ("native_backend_sha256", "4" * 64, "native"),
        )
        for argument, value, message in mismatches:
            with self.subTest(argument=argument):
                arguments = {
                    "config_sha256": CONFIG_SHA256,
                    "source_sha256": SOURCE_SHA256,
                    "native_backend_sha256": NATIVE_SHA256,
                }
                arguments[argument] = value
                with self.assertRaisesRegex(Phase4TrainingError, message):
                    _verify_phase4e_training_inputs(evidence, **arguments)

    def test_phase4e_training_requires_explicit_bootstrap_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            contract_path = _write_frozen_phase4e_contract(workspace)
            with (
                patch(
                    "quoridor.phase4_training."
                    "_verify_phase4e_bootstrap_baseline"
                ) as bootstrap_verifier,
                patch(
                    "quoridor.phase4_training."
                    "verify_phase4e_baseline_report"
                ) as baseline_verifier,
                self.assertRaisesRegex(
                    Phase4TrainingError,
                    "requires the E1-reviewed Bootstrap baseline",
                ),
            ):
                run_phase4_training(
                    config_path=ROOT
                    / "training/configs/phase4e-bootstrap-v1.json",
                    workspace_root=workspace,
                    generations=1,
                    resume=False,
                    device_override=None,
                    external_gpu_hours=0.0,
                    execution_contract_path=contract_path,
                )

            bootstrap_verifier.assert_not_called()
            baseline_verifier.assert_not_called()

    def test_phase4e_training_rejects_bootstrap_baseline_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            contract_path = _write_frozen_phase4e_contract(workspace)
            bootstrap_root = workspace / "bootstrap-baseline"
            local_evidence = dict(BOOTSTRAP_BASELINE)
            reviewed_evidence = {
                **BOOTSTRAP_BASELINE,
                "modelFileSha256": "6" * 64,
            }
            baseline_provenance = {
                "approvedWorkloads": [CONFIG_SHA256],
                "sourceSha256": SOURCE_SHA256,
                "nativeBackendSha256": NATIVE_SHA256,
                "bootstrapBaseline": reviewed_evidence,
            }
            with (
                patch(
                    "quoridor.phase4_training."
                    "_verify_phase4e_bootstrap_baseline",
                    return_value=local_evidence,
                ) as bootstrap_verifier,
                patch(
                    "quoridor.phase4_training."
                    "verify_phase4e_baseline_report",
                    return_value=baseline_provenance,
                ) as baseline_verifier,
                patch(
                    "quoridor.phase4_training.loaded_native_backend_hash"
                ) as native_hash,
                self.assertRaisesRegex(
                    Phase4TrainingError,
                    "Bootstrap baseline differs",
                ),
            ):
                run_phase4_training(
                    config_path=ROOT
                    / "training/configs/phase4e-bootstrap-v1.json",
                    workspace_root=workspace,
                    generations=1,
                    resume=False,
                    device_override=None,
                    external_gpu_hours=0.0,
                    execution_contract_path=contract_path,
                    bootstrap_baseline_root=bootstrap_root,
                )

            bootstrap_verifier.assert_called_once_with(
                workspace,
                bootstrap_root,
            )
            baseline_verifier.assert_called_once_with(
                workspace / "training/runs/phase4",
                baseline_run_id=RUN_ID,
                baseline_report_sha256="a" * 64,
                baseline_global_hours=BASELINE_HOURS,
                limit_hours=TRAINING_LIMIT_HOURS,
            )
            native_hash.assert_not_called()

    def test_unknown_execution_stage_fails_before_native_or_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config_path = workspace / "config.json"
            contract_path = workspace / "contract.json"
            config_path.write_text(
                json.dumps(
                    {
                        "pipeline": "phase4",
                        "name": "unknown-stage-test",
                        "seed": 20260714,
                        "device": "cuda",
                        "runDir": "training/runs/phase4/unknown-stage-test",
                        "artifactLimitBytes": 8 * 1024**3,
                        "initialization": {
                            "mode": "random",
                            "expectedBaseModelSha256": "5" * 64,
                            "resetOptimizer": True,
                        },
                        "network": {
                            "inputChannels": 16,
                            "channels": 192,
                            "residualBlocks": 16,
                        },
                        "mcts": {"simulations": 256},
                        "selfPlay": {"games": 1, "actors": 1},
                        "training": {
                            "replayCapacity": 1280,
                            "batchSize": 1,
                            "trainingSteps": 8,
                            "maxCheckpoints": 2,
                        },
                        "arena": {"games": 2, "simulations": 256},
                    }
                ),
                encoding="utf-8",
            )
            contract_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": PHASE4_EXECUTION_SCHEMA_VERSION,
                        "stage": "unreviewed-stage",
                        "status": PHASE4_EXECUTION_STATUS_FROZEN,
                        "purpose": "test",
                        "formalPromotion": False,
                        "maximumGenerations": 5,
                        "stageBudget": {
                            "status": PHASE4_BUDGET_STATUS_FROZEN,
                            "baselineGlobalHours": 1.0,
                            "limitHours": 1.0,
                            "baselineRunId": "test",
                            "baselineReportSha256": "6" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "quoridor.phase4_training.loaded_native_backend_hash"
            ) as native_hash:
                with self.assertRaisesRegex(
                    Phase4TrainingError,
                    "unsupported Phase 4 execution stage",
                ):
                    run_phase4_training(
                        config_path=config_path,
                        workspace_root=workspace,
                        generations=1,
                        resume=False,
                        device_override=None,
                        external_gpu_hours=0.0,
                        execution_contract_path=contract_path,
                    )
            native_hash.assert_not_called()


if __name__ == "__main__":
    unittest.main()
