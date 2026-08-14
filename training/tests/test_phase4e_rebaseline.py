from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quoridor.calibration import GpuBudgetLedger
from quoridor.phase4_budget import PHASE4E_STAGE, Phase4StageBudgetLedger
from quoridor.phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_SCHEMA_VERSION,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionBudget,
    Phase4ExecutionContract,
)
from quoridor.phase4e1_contract import load_phase4e1_contract
from quoridor.phase4e_rebaseline import (
    SUPERSESSION_ATTESTATION_NAME,
    Phase4ERebaselineError,
    freeze_and_rebase_phase4e,
    supersede_completed_e1_evidence,
)
from quoridor.run_manifest import (
    ResourceBudget,
    build_run_manifest,
    canonical_json_bytes,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_RELATIVE = Path(
    "training/configs/phase4e1-calibration-v1.json"
)
OLD_EXECUTION_RELATIVE = Path(
    "training/configs/phase4e-execution-frozen-v1.json"
)
CONTROL_RELATIVE = Path("training/runs/phase4")
OLD_SOURCE = "1" * 64
NEW_SOURCE = "2" * 64
NATIVE_SHA256 = "3" * 64


def _self_hashed(core: dict[str, object], field: str) -> dict[str, object]:
    return {
        **core,
        field: sha256_bytes(canonical_json_bytes(core)),
    }


class SupersedeFixture:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        calibration_path = workspace / CALIBRATION_RELATIVE
        calibration_path.parent.mkdir(parents=True)
        calibration_path.write_bytes(
            (ROOT / CALIBRATION_RELATIVE).read_bytes()
        )
        bootstrap_config = Path(
            "training/configs/phase4e-bootstrap-v1.json"
        )
        (workspace / bootstrap_config).write_bytes(
            (ROOT / bootstrap_config).read_bytes()
        )
        self.contract = load_phase4e1_contract(calibration_path)
        self.run_id = (
            f"phase4e1-calibration-{self.contract.sha256[:16]}"
        )
        self.run_root = (
            workspace
            / "training/runs/phase4/phase4e1/calibration"
            / self.run_id
        )
        self.run_root.mkdir(parents=True)

        report_core = {
            "schemaVersion": 1,
            "kind": "phase4e1-standard-calibration-and-arena-smoke",
            "status": "completed",
            "runId": self.run_id,
        }
        self.report = _self_hashed(report_core, "reportSha256")
        self.report_path = self.run_root / "report.json"
        self.report_path.write_bytes(canonical_json_bytes(self.report))

        source_path = workspace / "historical-source.txt"
        source_path.write_text("historical", encoding="utf-8")
        manifest = build_run_manifest(
            workspace_root=workspace,
            config={"fixture": "completed-e1"},
            budget=ResourceBudget(),
            run_id=self.run_id,
            metadata={"fixture": True},
            source_paths=[source_path],
        )
        manifest.payload["provenance"]["source"]["sha256"] = OLD_SOURCE
        manifest_core = dict(manifest.payload)
        manifest_core.pop("manifestSha256")
        manifest.payload["manifestSha256"] = sha256_bytes(
            canonical_json_bytes(manifest_core)
        )
        self.manifest_sha256 = manifest.payload["manifestSha256"]
        (self.run_root / "run-manifest.json").write_bytes(
            canonical_json_bytes(manifest.payload)
        )
        (self.run_root / "historical-evidence.bin").write_bytes(b"evidence")

        self.old_execution = Phase4ExecutionContract(
            schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
            stage=PHASE4E_STAGE,
            status=PHASE4_EXECUTION_STATUS_FROZEN,
            purpose="phase4e-bootstrap-and-formal-promotion",
            formal_promotion=True,
            maximum_generations=64,
            stage_budget=Phase4ExecutionBudget(
                status=PHASE4_BUDGET_STATUS_FROZEN,
                baseline_global_hours=10.0,
                limit_hours=60.0,
                baseline_run_id=self.run_id,
                baseline_report_sha256=self.report["reportSha256"],
            ),
        )
        old_execution_path = workspace / OLD_EXECUTION_RELATIVE
        old_execution_path.write_bytes(
            canonical_json_bytes(self.old_execution.to_payload())
        )
        bootstrap_incident_core = {
            "schemaVersion": 1,
            "kind": "phase4e-bootstrap-failed-run-quarantine",
            "oldExecutionContractSha256": self.old_execution.sha256,
            "originalSourceSha256": OLD_SOURCE,
            "replacementSourceSha256": NEW_SOURCE,
        }
        self.bootstrap_incident = _self_hashed(
            bootstrap_incident_core,
            "incidentSha256",
        )
        self.bootstrap_incident_path = (
            workspace
            / "training/runs/phase4/phase4e/quarantine/bootstrap/"
            "quarantine-incident.json"
        )
        self.bootstrap_incident_path.parent.mkdir(parents=True)
        self.bootstrap_incident_path.write_bytes(
            canonical_json_bytes(self.bootstrap_incident)
        )

        control = workspace / CONTROL_RELATIVE
        stage_ledger = Phase4StageBudgetLedger(
            control / "stage-budget-ledger.json"
        )
        self.old_stage_record = stage_ledger.ensure_stage(
            PHASE4E_STAGE,
            10.0,
            60.0,
            baseline_run_id=self.run_id,
            baseline_report_sha256=self.report["reportSha256"],
        )
        gpu_ledger_path = control / "gpu-budget-ledger.json"
        GpuBudgetLedger(gpu_ledger_path).record_external_hours(10.0)
        self.gpu_ledger_before = gpu_ledger_path.read_bytes()

    def historical_verifier(
        self,
        report_path: Path,
        workspace: Path,
        contract: object,
    ) -> dict[str, object]:
        if report_path != self.report_path:
            raise AssertionError("unexpected report")
        if workspace != self.workspace:
            raise AssertionError("unexpected workspace")
        if contract != self.contract:
            raise AssertionError("unexpected contract")
        return {
            "status": "completed",
            "decisionEligible": True,
            "runId": self.run_id,
            "reportSha256": self.report["reportSha256"],
            "sourceSha256": OLD_SOURCE,
            "nativeBackendSha256": NATIVE_SHA256,
            "runManifestSha256": self.manifest_sha256,
            "recommendedStageBudget": {
                "baselineGlobalHours": 10.0,
                "limitHours": 60.0,
            },
        }

    def call(self, *, execute: bool = True) -> dict[str, object]:
        return supersede_completed_e1_evidence(
            workspace_root=self.workspace,
            calibration_contract_path=CALIBRATION_RELATIVE,
            report_path=self.report_path,
            old_execution_contract_path=OLD_EXECUTION_RELATIVE,
            bootstrap_quarantine_incident_path=(
                self.bootstrap_incident_path
            ),
            expected_old_source_sha256=OLD_SOURCE,
            expected_replacement_source_sha256=NEW_SOURCE,
            execute=execute,
            historical_verifier=self.historical_verifier,
            bootstrap_incident_verifier=self.bootstrap_incident_verifier,
        )

    def bootstrap_incident_verifier(
        self,
        incident_path: Path,
        workspace: Path,
        old_execution_sha256: str,
        original_source_sha256: str,
        replacement_source_sha256: str,
    ) -> dict[str, object]:
        if (
            incident_path != self.bootstrap_incident_path
            or workspace != self.workspace
            or old_execution_sha256 != self.old_execution.sha256
            or original_source_sha256 != OLD_SOURCE
            or replacement_source_sha256 != NEW_SOURCE
        ):
            raise AssertionError("unexpected Bootstrap incident verification")
        return dict(self.bootstrap_incident)

    def prepare_replacement_report(
        self,
        quarantine_reference: dict[str, object],
    ) -> None:
        self.run_root.mkdir(parents=True)
        report_core = {
            "schemaVersion": 1,
            "kind": "phase4e1-standard-calibration-and-arena-smoke",
            "status": "completed",
            "runId": self.run_id,
            "sourceSha256": NEW_SOURCE,
        }
        self.replacement_report = _self_hashed(
            report_core,
            "reportSha256",
        )
        self.report_path.write_bytes(
            canonical_json_bytes(self.replacement_report)
        )
        source_path = self.workspace / "replacement-source.txt"
        source_path.write_text("replacement", encoding="utf-8")
        manifest = build_run_manifest(
            workspace_root=self.workspace,
            config={"fixture": "replacement-e1"},
            budget=ResourceBudget(),
            run_id=self.run_id,
            metadata={"quarantineIncident": quarantine_reference},
            source_paths=[source_path],
        )
        manifest.payload["provenance"]["source"]["sha256"] = NEW_SOURCE
        manifest_core = dict(manifest.payload)
        manifest_core.pop("manifestSha256")
        manifest.payload["manifestSha256"] = sha256_bytes(
            canonical_json_bytes(manifest_core)
        )
        self.replacement_manifest_sha256 = manifest.payload[
            "manifestSha256"
        ]
        (self.run_root / "run-manifest.json").write_bytes(
            canonical_json_bytes(manifest.payload)
        )
        gpu_path = self.workspace / CONTROL_RELATIVE / (
            "gpu-budget-ledger.json"
        )
        GpuBudgetLedger(gpu_path).record_external_hours(12.0)

    def replacement_decision_verifier(
        self,
        report_path: Path,
        workspace: Path,
        contract: object,
    ) -> dict[str, object]:
        if (
            report_path != self.report_path
            or workspace != self.workspace
            or contract != self.contract
        ):
            raise AssertionError("unexpected replacement report verification")
        return {
            "status": "completed",
            "decisionEligible": True,
            "runId": self.run_id,
            "reportSha256": self.replacement_report["reportSha256"],
            "sourceSha256": NEW_SOURCE,
            "nativeBackendSha256": NATIVE_SHA256,
            "runManifestSha256": self.replacement_manifest_sha256,
            "recommendedStageBudget": {
                "baselineGlobalHours": 12.0,
                "limitHours": 58.0,
            },
        }

    def rebase(self, *, execute: bool) -> dict[str, object]:
        return freeze_and_rebase_phase4e(
            workspace_root=self.workspace,
            calibration_contract_path=CALIBRATION_RELATIVE,
            report_path=self.report_path,
            old_execution_contract_path=OLD_EXECUTION_RELATIVE,
            new_execution_contract_path=(
                "training/configs/phase4e-execution-frozen-v2.json"
            ),
            bootstrap_quarantine_incident_path=(
                self.bootstrap_incident_path
            ),
            expected_old_source_sha256=OLD_SOURCE,
            expected_replacement_source_sha256=NEW_SOURCE,
            execute=execute,
            decision_verifier=self.replacement_decision_verifier,
            bootstrap_incident_verifier=self.bootstrap_incident_verifier,
        )


class Phase4ECompletedSupersessionTests(unittest.TestCase):
    def _patches(self):
        return (
            patch(
                "quoridor.phase4e_rebaseline.fingerprint_sources",
                return_value=SimpleNamespace(sha256=NEW_SOURCE),
            ),
            patch(
                "quoridor.phase4e1_calibration.fingerprint_sources",
                return_value=SimpleNamespace(sha256=NEW_SOURCE),
            ),
            patch(
                "quoridor.phase4e1_calibration._verify_e1_run_manifest"
            ),
            patch(
                "quoridor.phase4e_rebaseline.loaded_native_backend_hash",
                return_value=NATIVE_SHA256,
            ),
        )

    def test_completed_evidence_is_bound_to_existing_quarantine_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            gpu_path = fixture.workspace / CONTROL_RELATIVE / (
                "gpu-budget-ledger.json"
            )
            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = (
                self._patches()
            )
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ):
                result = fixture.call()

                self.assertEqual(result["status"], "superseded")
                self.assertFalse(fixture.run_root.exists())
                reference = result["quarantineIncident"]
                incident_path = fixture.workspace / reference["path"]
                self.assertTrue(incident_path.is_file())
                attestation_path = (
                    incident_path.parent / SUPERSESSION_ATTESTATION_NAME
                )
                attestation = json.loads(
                    attestation_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    attestation["oldExecutionContract"][
                        "executionContractSha256"
                    ],
                    fixture.old_execution.sha256,
                )
                self.assertEqual(
                    attestation["oldDecisionReport"]["reportSha256"],
                    fixture.report["reportSha256"],
                )
                self.assertEqual(
                    attestation["stageLedgerBefore"]["phase4eRecord"],
                    fixture.old_stage_record,
                )
                incident = json.loads(
                    incident_path.read_text(encoding="utf-8")
                )
                self.assertIn(
                    SUPERSESSION_ATTESTATION_NAME,
                    [item["path"] for item in incident["files"]],
                )
                self.assertEqual(
                    incident["replacementSourceSha256"],
                    NEW_SOURCE,
                )

                empty_replacement_root = fixture.run_root
                empty_replacement_root.mkdir(parents=True)
                recovered = (
                    __import__(
                        "quoridor.phase4e1_calibration",
                        fromlist=["_quarantine_source_mismatched_run"],
                    )._quarantine_source_mismatched_run(
                        workspace=fixture.workspace,
                        control_root=fixture.workspace / CONTROL_RELATIVE,
                        run_root=empty_replacement_root,
                        run_id=fixture.run_id,
                        contract=fixture.contract,
                        native_sha256=NATIVE_SHA256,
                    )
                )
                self.assertEqual(recovered, reference)
                self.assertEqual(gpu_path.read_bytes(), fixture.gpu_ledger_before)

    def test_rename_failure_resumes_from_sealed_incident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = (
                self._patches()
            )
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
                patch(
                    "quoridor.phase4e_rebaseline.os.replace",
                    side_effect=OSError("injected rename failure"),
                ),
                self.assertRaisesRegex(
                    Phase4ERebaselineError,
                    "atomically quarantine",
                ),
            ):
                fixture.call()
            self.assertTrue(fixture.run_root.is_dir())
            self.assertTrue(
                (
                    fixture.run_root / SUPERSESSION_ATTESTATION_NAME
                ).is_file()
            )
            self.assertTrue(
                (fixture.run_root / "quarantine-incident.json").is_file()
            )

            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = self._patches()
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ):
                dry_run = fixture.call(execute=False)
            self.assertEqual(
                dry_run["status"],
                "verified-dry-run-pending",
            )
            self.assertEqual(
                dry_run["pendingState"],
                "sealed-before-move",
            )

            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = (
                self._patches()
            )
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ):
                result = fixture.call()
            self.assertEqual(result["status"], "superseded")
            self.assertFalse(fixture.run_root.exists())

    def test_rename_failure_reuses_sealed_incident_after_ledger_append(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            source, calibration_source, manifest, native = self._patches()
            with (
                source,
                calibration_source,
                manifest,
                native,
                patch(
                    "quoridor.phase4e_rebaseline.os.replace",
                    side_effect=OSError("injected rename failure"),
                ),
                self.assertRaisesRegex(
                    Phase4ERebaselineError,
                    "atomically quarantine",
                ),
            ):
                fixture.call()
            sealed = (
                fixture.run_root / "quarantine-incident.json"
            ).read_bytes()
            gpu_path = (
                fixture.workspace
                / CONTROL_RELATIVE
                / "gpu-budget-ledger.json"
            )
            GpuBudgetLedger(gpu_path).record_external_hours(11.0)

            source, calibration_source, manifest, native = self._patches()
            with source, calibration_source, manifest, native:
                result = fixture.call()
            self.assertEqual(result["status"], "superseded")
            incident_path = (
                fixture.workspace / result["quarantineIncident"]["path"]
            )
            self.assertEqual(incident_path.read_bytes(), sealed)

    def test_quarantine_symlink_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            escape = fixture.workspace / "escape"
            escape.mkdir()
            quarantine = (
                fixture.workspace
                / "training/runs/phase4/phase4e1/quarantine"
            )
            quarantine.symlink_to(escape, target_is_directory=True)
            source, calibration_source, manifest, native = self._patches()
            with (
                source,
                calibration_source,
                manifest,
                native,
                self.assertRaisesRegex(
                    Phase4ERebaselineError,
                    "symlink",
                ),
            ):
                fixture.call()
            self.assertEqual(list(escape.iterdir()), [])

    def test_completed_rename_is_idempotently_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = (
                self._patches()
            )
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ):
                first = fixture.call()
                second = fixture.call()
            self.assertEqual(second["status"], "already-superseded")
            self.assertEqual(
                second["quarantineIncident"],
                first["quarantineIncident"],
            )

    def test_post_rename_fsync_failure_is_retried_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            source, calibration_source, manifest, native = self._patches()
            with (
                source,
                calibration_source,
                manifest,
                native,
                patch(
                    "quoridor.phase4e_rebaseline._fsync_directory",
                    side_effect=Phase4ERebaselineError(
                        "injected parent fsync failure"
                    ),
                ),
                self.assertRaisesRegex(
                    Phase4ERebaselineError,
                    "parent fsync",
                ),
            ):
                fixture.call()
            self.assertFalse(fixture.run_root.exists())

            synced: list[Path] = []

            def record_sync(path: Path) -> None:
                synced.append(path)

            source, calibration_source, manifest, native = self._patches()
            with (
                source,
                calibration_source,
                manifest,
                native,
                patch(
                    "quoridor.phase4e_rebaseline._fsync_directory",
                    side_effect=record_sync,
                ),
            ):
                result = fixture.call()
            self.assertEqual(result["status"], "already-superseded")
            incident = fixture.workspace / result["quarantineIncident"]["path"]
            self.assertEqual(
                synced,
                [fixture.run_root.parent, incident.parent.parent],
            )

    def test_historical_verification_failure_leaves_tree_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SupersedeFixture(Path(directory).resolve())
            before = sorted(
                path.relative_to(fixture.run_root).as_posix()
                for path in fixture.run_root.rglob("*")
            )

            def reject(*_args):
                raise Phase4ERebaselineError("historical deep verification failed")

            (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
            ) = (
                self._patches()
            )
            with (
                source_patch,
                calibration_source_patch,
                manifest_patch,
                native_patch,
                self.assertRaisesRegex(
                    Phase4ERebaselineError,
                    "historical deep verification",
                ),
            ):
                supersede_completed_e1_evidence(
                    workspace_root=fixture.workspace,
                    calibration_contract_path=CALIBRATION_RELATIVE,
                    report_path=fixture.report_path,
                    old_execution_contract_path=OLD_EXECUTION_RELATIVE,
                    bootstrap_quarantine_incident_path=(
                        fixture.bootstrap_incident_path
                    ),
                    expected_old_source_sha256=OLD_SOURCE,
                    expected_replacement_source_sha256=NEW_SOURCE,
                    execute=True,
                    historical_verifier=reject,
                    bootstrap_incident_verifier=(
                        fixture.bootstrap_incident_verifier
                    ),
                )
            after = sorted(
                path.relative_to(fixture.run_root).as_posix()
                for path in fixture.run_root.rglob("*")
            )
            self.assertEqual(after, before)


class Phase4EStageRebaseTests(unittest.TestCase):
    def _prepare(self, workspace: Path) -> SupersedeFixture:
        fixture = SupersedeFixture(workspace)
        patches = Phase4ECompletedSupersessionTests()._patches()
        source_patch, calibration_source_patch, manifest_patch, native_patch = (
            patches
        )
        with (
            source_patch,
            calibration_source_patch,
            manifest_patch,
            native_patch,
        ):
            superseded = fixture.call()
        fixture.prepare_replacement_report(
            superseded["quarantineIncident"]
        )
        return fixture

    def _rebase_patches(self):
        return (
            patch(
                "quoridor.phase4e_rebaseline.fingerprint_sources",
                return_value=SimpleNamespace(sha256=NEW_SOURCE),
            ),
            patch(
                "quoridor.phase4e1_calibration._verify_e1_run_manifest"
            ),
            patch(
                "quoridor.phase4e_rebaseline.loaded_native_backend_hash",
                return_value=NATIVE_SHA256,
            ),
        )

    def test_freeze_v2_and_stage_rebase_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._prepare(Path(directory).resolve())
            gpu_path = fixture.workspace / CONTROL_RELATIVE / (
                "gpu-budget-ledger.json"
            )
            gpu_before = gpu_path.read_bytes()
            source_patch, manifest_patch, native_patch = (
                self._rebase_patches()
            )
            with source_patch, manifest_patch, native_patch:
                first = fixture.rebase(execute=True)
                (
                    fixture.workspace
                    / "training/runs/phase4/phase4e/bootstrap-v1"
                ).mkdir(parents=True)
                second = fixture.rebase(execute=True)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "already-completed")
            self.assertEqual(
                first["newExecutionContractSha256"],
                second["newExecutionContractSha256"],
            )
            stage = Phase4StageBudgetLedger(
                fixture.workspace
                / CONTROL_RELATIVE
                / "stage-budget-ledger.json"
            )
            self.assertEqual(
                stage.stages[PHASE4E_STAGE]["baselineGlobalHours"],
                12.0,
            )
            self.assertEqual(
                stage.stages[PHASE4E_STAGE]["limitHours"],
                58.0,
            )
            self.assertEqual(gpu_path.read_bytes(), gpu_before)
            self.assertTrue(
                (
                    fixture.workspace
                    / "training/configs/"
                    "phase4e-execution-frozen-v2.json"
                ).is_file()
            )

    def test_rebase_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._prepare(Path(directory).resolve())
            stage_path = (
                fixture.workspace
                / CONTROL_RELATIVE
                / "stage-budget-ledger.json"
            )
            stage_before = stage_path.read_bytes()
            source_patch, manifest_patch, native_patch = (
                self._rebase_patches()
            )
            with source_patch, manifest_patch, native_patch:
                result = fixture.rebase(execute=False)
            self.assertEqual(result["status"], "verified-dry-run")
            self.assertEqual(stage_path.read_bytes(), stage_before)
            self.assertFalse(
                (
                    fixture.workspace
                    / "training/configs/"
                    "phase4e-execution-frozen-v2.json"
                ).exists()
            )
            self.assertFalse(
                (
                    fixture.workspace
                    / "training/runs/phase4/phase4e1/stage-rebases"
                ).exists()
            )

    def test_prepared_dry_run_is_read_only_then_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._prepare(Path(directory).resolve())
            stage_path = (
                fixture.workspace
                / CONTROL_RELATIVE
                / "stage-budget-ledger.json"
            )
            stage_before = stage_path.read_bytes()

            def reject_stage_write(path, payload):
                if Path(path) == stage_path:
                    raise OSError("injected stage ledger write failure")
                from quoridor.run_manifest import atomic_replace_bytes

                return atomic_replace_bytes(path, payload)

            source, manifest, native = self._rebase_patches()
            with (
                source,
                manifest,
                native,
                patch(
                    "quoridor.phase4e_rebaseline.atomic_replace_bytes",
                    side_effect=reject_stage_write,
                ),
                self.assertRaisesRegex(OSError, "injected stage ledger"),
            ):
                fixture.rebase(execute=True)
            prepared = list(
                (
                    fixture.workspace
                    / "training/runs/phase4/phase4e1/stage-rebases"
                ).glob("*/prepared.json")
            )
            self.assertEqual(len(prepared), 1)
            self.assertEqual(stage_path.read_bytes(), stage_before)

            source, manifest, native = self._rebase_patches()
            with source, manifest, native:
                dry_run = fixture.rebase(execute=False)
            self.assertEqual(
                dry_run["status"],
                "verified-dry-run-pending",
            )
            self.assertEqual(
                dry_run["pendingState"],
                "prepared-before-ledger",
            )
            self.assertEqual(stage_path.read_bytes(), stage_before)

            source, manifest, native = self._rebase_patches()
            with source, manifest, native:
                resumed = fixture.rebase(execute=True)
            self.assertEqual(resumed["status"], "completed")

    def test_ledger_write_crash_resumes_and_seals_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._prepare(Path(directory).resolve())
            from quoridor import phase4e1_calibration as calibration

            original_commit = calibration._commit_immutable_json

            def fail_completed(path, payload, *, label):
                if label == "completed Phase 4E stage rebase":
                    raise RuntimeError("injected completed receipt failure")
                return original_commit(path, payload, label=label)

            source, manifest, native = self._rebase_patches()
            with (
                source,
                manifest,
                native,
                patch.object(
                    calibration,
                    "_commit_immutable_json",
                    side_effect=fail_completed,
                ),
                self.assertRaisesRegex(RuntimeError, "completed receipt"),
            ):
                fixture.rebase(execute=True)
            stage = Phase4StageBudgetLedger(
                fixture.workspace
                / CONTROL_RELATIVE
                / "stage-budget-ledger.json"
            )
            self.assertEqual(
                stage.stages[PHASE4E_STAGE]["baselineGlobalHours"],
                12.0,
            )

            source, manifest, native = self._rebase_patches()
            with source, manifest, native:
                resumed = fixture.rebase(execute=True)
            self.assertEqual(resumed["status"], "completed")
            completed = list(
                (
                    fixture.workspace
                    / "training/runs/phase4/phase4e1/stage-rebases"
                ).glob("*/completed.json")
            )
            self.assertEqual(len(completed), 1)


if __name__ == "__main__":
    unittest.main()
