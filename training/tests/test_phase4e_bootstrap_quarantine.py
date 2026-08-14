from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quoridor import phase4e_bootstrap_quarantine as quarantine
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
OLD_CONTRACT_PATH = (
    ROOT / "training/configs/phase4e-execution-frozen-v1.json"
)
OLD_CONTRACT_SHA256 = (
    "0ae33064ebd4ded3d9f9f0b96e9aaa89a736dcf3b5b05bcaa44d418afc9743ad"
)
ORIGINAL_SOURCE_SHA256 = "1" * 64
REPLACEMENT_SOURCE_SHA256 = "2" * 64
LEDGER_RUN_ID = "phase4e-training-phase4e-bootstrap-v1-test"
ACCOUNTED_SECONDS = 1.3450553622096777


def _self_hashed(
    core: dict[str, object],
    field: str,
) -> dict[str, object]:
    return {
        **core,
        field: sha256_bytes(canonical_json_bytes(core)),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


class Phase4EBootstrapQuarantineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.control_root = self.workspace / "training/runs/phase4"
        self.run_dir = self.control_root / "phase4e/bootstrap-v1"
        self.quarantine_root = self.control_root / "phase4e/quarantine"
        self.ledger_path = self.control_root / "gpu-budget-ledger.json"
        self.log_path = (
            self.control_root / "logs/phase4e-training-test.log"
        )
        self.contract_path = (
            self.workspace
            / "training/configs/phase4e-execution-frozen-v1.json"
        )
        self.contract_path.parent.mkdir(parents=True, exist_ok=True)
        self.contract_path.write_bytes(OLD_CONTRACT_PATH.read_bytes())
        self._create_failed_run()
        self._write_ledger()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            "Traceback (most recent call last):\n"
            f"{quarantine.DEFAULT_FAILURE_MARKER}\n",
            encoding="utf-8",
        )

    def _create_failed_run(self) -> None:
        for relative in (
            "checkpoints",
            "reports",
            "replay",
            "replay-tensor-cache/chunks",
        ):
            (self.run_dir / relative).mkdir(parents=True, exist_ok=True)
        manifest_core = {
            "schemaVersion": 1,
            "runId": "phase4e-bootstrap-v1-20260714",
            "createdAt": "2026-07-28T10:13:34Z",
            "executionPolicy": {
                "scope": "local-machine-only",
                "localOnly": True,
                "networkRequired": False,
                "cloudServices": False,
            },
            "workspace": str(self.workspace),
            "provenance": {
                "gitMetadataUsed": False,
                "source": {
                    "algorithm": "sha256-path-content-v1",
                    "sha256": ORIGINAL_SOURCE_SHA256,
                    "fileCount": 1,
                    "bytes": 1,
                    "files": [
                        {
                            "path": "training/quoridor/old.py",
                            "bytes": 1,
                            "sha256": "3" * 64,
                        }
                    ],
                },
                "config": {
                    "algorithm": "sha256-canonical-json-v1",
                    "sha256": "4" * 64,
                    "payload": {
                        "name": quarantine.DEFAULT_EXPERIMENT,
                        "run_dir": str(self.run_dir),
                    },
                },
            },
            "environment": {"torch": {"cudaAvailable": True}},
            "resourceBudget": {
                "artifactLimitBytes": 8 * 1024**3,
                "minimumFreeDiskBytes": 20 * 1024**3,
                "gpuHours": 59.96104,
                "maximumGenerations": 64,
                "metadata": {
                    "executionContractSha256": OLD_CONTRACT_SHA256,
                },
            },
            "metadata": {
                "stage": "phase4e",
                "formalPromotion": True,
                "executionContractSha256": OLD_CONTRACT_SHA256,
            },
        }
        manifest = _self_hashed(manifest_core, "manifestSha256")
        self.manifest_sha256 = str(manifest["manifestSha256"])
        _write_json(self.run_dir / "run-manifest.json", manifest)

        replay = {
            "schemaVersion": 1,
            "codecVersion": 1,
            "shardSchemaVersion": 1,
            "actionSpaceSize": 209,
            "capacity": {
                "games": 100000,
                "plies": 100000,
                "bytes": 6 * 1024**3,
            },
            "nextSequence": 0,
            "shards": [],
            "totals": {"games": 0, "plies": 0, "shardBytes": 0},
        }
        replay_bytes = canonical_json_bytes(replay)
        (self.run_dir / "replay/manifest.json").write_bytes(replay_bytes)
        cache_core = {
            "schemaVersion": 1,
            "algorithmVersion": 1,
            "contract": {
                "sourceSha256": ORIGINAL_SOURCE_SHA256,
                "encoderSha256": "5" * 64,
            },
            "chunks": [],
            "entries": [],
            "totals": {
                "games": 0,
                "plies": 0,
                "chunks": 0,
                "cacheBytes": 0,
            },
            "rawManifestSha256": sha256_bytes(replay_bytes),
            "rawWindowSha256": "6" * 64,
            "samplingAlgorithm": "python-random-sample-v1",
            "mirrorRngAlgorithm": "shared-one-draw-per-sample-v1",
        }
        _write_json(
            self.run_dir / "replay-tensor-cache/manifest.json",
            _self_hashed(cache_core, "manifestSha256"),
        )

    def _write_ledger(
        self,
        *,
        failed_seconds: float = ACCOUNTED_SECONDS,
        extra_runs: dict[str, dict[str, object]] | None = None,
    ) -> None:
        runs: dict[str, dict[str, object]] = {
            "completed-before-bootstrap": {
                "startedAt": "2026-07-28T09:00:00Z",
                "updatedAt": "2026-07-28T09:00:01Z",
                "accountedSeconds": 3600.0,
                "status": "completed",
            },
            LEDGER_RUN_ID: {
                "startedAt": "2026-07-28T10:13:34Z",
                "updatedAt": "2026-07-28T10:13:35Z",
                "accountedSeconds": failed_seconds,
                "status": "failed",
            },
        }
        if extra_runs:
            runs.update(extra_runs)
        core = {
            "schemaVersion": 1,
            "limitHours": 72.0,
            "externalHoursFloor": 9.0,
            "runs": runs,
            "updatedAt": "2026-07-28T10:13:35Z",
        }
        _write_json(
            self.ledger_path,
            _self_hashed(core, "ledgerSha256"),
        )

    def _kwargs(self) -> dict[str, object]:
        return {
            "workspace_root": self.workspace,
            "original_run_dir": self.run_dir,
            "quarantine_root": self.quarantine_root,
            "gpu_ledger_path": self.ledger_path,
            "failure_log_path": self.log_path,
            "gpu_ledger_run_id": LEDGER_RUN_ID,
            "old_execution_contract_sha256": OLD_CONTRACT_SHA256,
            "run_manifest_sha256": self.manifest_sha256,
            "replacement_source_sha256": REPLACEMENT_SOURCE_SHA256,
            "old_execution_contract_path": self.contract_path,
        }

    @contextmanager
    def _guards(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    quarantine,
                    "fingerprint_sources",
                    return_value=SimpleNamespace(
                        sha256=REPLACEMENT_SOURCE_SHA256
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    quarantine,
                    "_directory_size",
                    return_value=1024,
                )
            )
            stack.enter_context(
                patch.object(
                    quarantine.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=30 * 1024**3),
                )
            )
            yield

    def test_dry_run_then_atomic_quarantine_is_idempotent_and_bound(self) -> None:
        ledger_before = self.ledger_path.read_bytes()
        log_before = self.log_path.read_bytes()
        with self._guards():
            plan = quarantine.inspect_failed_bootstrap_run(**self._kwargs())
            self.assertEqual(plan["status"], "ready")
            self.assertTrue(plan["dryRun"])
            self.assertTrue(self.run_dir.is_dir())
            self.assertFalse(self.quarantine_root.exists())

            reference = quarantine.quarantine_failed_bootstrap_run(
                **self._kwargs()
            )
            self.assertEqual(
                set(reference),
                {"path", "bytes", "sha256", "incidentSha256"},
            )
            self.assertFalse(self.run_dir.exists())
            payload = quarantine.verify_quarantine_incident_attestation(
                workspace_root=self.workspace,
                incident_path=self.workspace / str(reference["path"]),
                expected_old_execution_contract_sha256=OLD_CONTRACT_SHA256,
                expected_original_source_sha256=ORIGINAL_SOURCE_SHA256,
                expected_replacement_source_sha256=(
                    REPLACEMENT_SOURCE_SHA256
                ),
            )
            self.assertEqual(payload["gpuLedgerStatus"], "failed")
            self.assertEqual(
                payload["gpuLedgerAccountedSeconds"], ACCOUNTED_SECONDS
            )
            self.assertEqual(
                payload["oldExecutionContract"]["sha256"],
                OLD_CONTRACT_SHA256,
            )
            self.assertIn("checkpoints", payload["directories"])
            self.assertIn(
                "replay-tensor-cache/chunks", payload["directories"]
            )
            self.assertEqual(
                quarantine.quarantine_failed_bootstrap_run(**self._kwargs()),
                reference,
            )
            self.assertEqual(
                quarantine.verify_quarantine_reference(
                    reference, **self._kwargs()
                ),
                reference,
            )
        self.assertEqual(self.ledger_path.read_bytes(), ledger_before)
        self.assertEqual(self.log_path.read_bytes(), log_before)

    def test_sealed_but_unmoved_run_recovers_after_atomic_rename_failure(
        self,
    ) -> None:
        with self._guards():
            with patch.object(
                quarantine.os,
                "replace",
                side_effect=OSError("injected rename failure"),
            ):
                with self.assertRaisesRegex(
                    quarantine.Phase4EBootstrapQuarantineError,
                    "atomically quarantine",
                ):
                    quarantine.quarantine_failed_bootstrap_run(
                        **self._kwargs()
                    )
            self.assertTrue(
                (self.run_dir / quarantine.INCIDENT_NAME).is_file()
            )
            self.assertTrue(
                (self.run_dir / quarantine.PRESERVED_LOG_NAME).is_file()
            )
            self._write_ledger(
                extra_runs={
                    "post-crash-append": {
                        "startedAt": "2026-07-28T11:00:00Z",
                        "updatedAt": "2026-07-28T11:00:03Z",
                        "accountedSeconds": 3.0,
                        "status": "completed",
                    }
                }
            )
            reference = quarantine.quarantine_failed_bootstrap_run(
                **self._kwargs()
            )
            self.assertFalse(self.run_dir.exists())
            self.assertTrue((self.workspace / str(reference["path"])).is_file())

    def test_fsync_failure_after_rename_is_retried_only_by_execute(
        self,
    ) -> None:
        real_fsync_directory = quarantine._fsync_directory
        failed_sync_paths: list[Path] = []

        def fail_destination_parent(path: Path) -> None:
            failed_sync_paths.append(path)
            if len(failed_sync_paths) == 2:
                raise quarantine.Phase4EBootstrapQuarantineError(
                    "injected quarantine parent fsync failure"
                )
            real_fsync_directory(path)

        with self._guards():
            with (
                patch.object(
                    quarantine,
                    "_fsync_directory",
                    side_effect=fail_destination_parent,
                ),
                self.assertRaisesRegex(
                    quarantine.Phase4EBootstrapQuarantineError,
                    "injected quarantine parent fsync failure",
                ),
            ):
                quarantine.quarantine_failed_bootstrap_run(
                    **self._kwargs()
                )

            self.assertEqual(
                failed_sync_paths,
                [self.run_dir.parent, self.quarantine_root],
            )
            self.assertFalse(self.run_dir.exists())

            with patch.object(
                quarantine, "_fsync_directory"
            ) as dry_run_fsync:
                plan = quarantine.inspect_failed_bootstrap_run(
                    **self._kwargs()
                )
            dry_run_fsync.assert_not_called()
            self.assertEqual(plan["status"], "already-quarantined")

            destination = (
                self.workspace / str(plan["reference"]["path"])
            ).parent
            recovered_sync_paths: list[Path] = []

            def record_recovery_sync(path: Path) -> None:
                recovered_sync_paths.append(path)
                real_fsync_directory(path)

            with patch.object(
                quarantine,
                "_fsync_directory",
                side_effect=record_recovery_sync,
            ):
                reference = quarantine.quarantine_failed_bootstrap_run(
                    **self._kwargs()
                )

            self.assertEqual(reference, plan["reference"])
            self.assertEqual(
                recovered_sync_paths,
                [self.run_dir.parent, destination.parent],
            )

    def test_current_gpu_ledger_may_append_but_failed_entry_cannot_change(
        self,
    ) -> None:
        with self._guards():
            reference = quarantine.quarantine_failed_bootstrap_run(
                **self._kwargs()
            )
            incident_path = self.workspace / str(reference["path"])
            self._write_ledger(
                extra_runs={
                    "later-run": {
                        "startedAt": "2026-07-28T11:00:00Z",
                        "updatedAt": "2026-07-28T11:00:02Z",
                        "accountedSeconds": 2.0,
                        "status": "completed",
                    }
                }
            )
            quarantine.verify_quarantine_incident_attestation(
                workspace_root=self.workspace,
                incident_path=incident_path,
                expected_old_execution_contract_sha256=OLD_CONTRACT_SHA256,
                expected_original_source_sha256=ORIGINAL_SOURCE_SHA256,
                expected_replacement_source_sha256=(
                    REPLACEMENT_SOURCE_SHA256
                ),
            )
            self._write_ledger(failed_seconds=ACCOUNTED_SECONDS + 1.0)
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "append-only",
            ):
                quarantine.verify_quarantine_incident_attestation(
                    workspace_root=self.workspace,
                    incident_path=incident_path,
                    expected_old_execution_contract_sha256=(
                        OLD_CONTRACT_SHA256
                    ),
                    expected_original_source_sha256=(
                        ORIGINAL_SOURCE_SHA256
                    ),
                    expected_replacement_source_sha256=(
                        REPLACEMENT_SOURCE_SHA256
                    ),
                )

    def test_rejects_progress_symlink_and_untrusted_replacement_source(
        self,
    ) -> None:
        with self._guards():
            (self.run_dir / "checkpoints/untrusted.pt").write_bytes(b"x")
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "checkpoints contains training progress",
            ):
                quarantine.inspect_failed_bootstrap_run(**self._kwargs())
            (self.run_dir / "checkpoints/untrusted.pt").unlink()

        target = self.workspace / "real-log.txt"
        target.write_text(
            quarantine.DEFAULT_FAILURE_MARKER, encoding="utf-8"
        )
        self.log_path.unlink()
        self.log_path.symlink_to(target)
        with self._guards():
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "symlink",
            ):
                quarantine.inspect_failed_bootstrap_run(**self._kwargs())

        self.log_path.unlink()
        self.log_path.write_text(
            quarantine.DEFAULT_FAILURE_MARKER, encoding="utf-8"
        )
        with (
            patch.object(
                quarantine,
                "fingerprint_sources",
                return_value=SimpleNamespace(sha256="9" * 64),
            ),
            self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "current workspace",
            ),
        ):
            quarantine.inspect_failed_bootstrap_run(**self._kwargs())

    def test_preserved_log_tamper_and_destination_collision_are_rejected(
        self,
    ) -> None:
        with self._guards():
            reference = quarantine.quarantine_failed_bootstrap_run(
                **self._kwargs()
            )
            incident_path = self.workspace / str(reference["path"])
            preserved = incident_path.parent / quarantine.PRESERVED_LOG_NAME
            preserved.write_bytes(preserved.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "failure-log evidence changed",
            ):
                quarantine.verify_quarantine_incident_attestation(
                    workspace_root=self.workspace,
                    incident_path=incident_path,
                    expected_old_execution_contract_sha256=(
                        OLD_CONTRACT_SHA256
                    ),
                    expected_original_source_sha256=(
                        ORIGINAL_SOURCE_SHA256
                    ),
                    expected_replacement_source_sha256=(
                        REPLACEMENT_SOURCE_SHA256
                    ),
                )

    def test_matching_destination_symlink_is_a_collision(self) -> None:
        with self._guards():
            plan = quarantine.inspect_failed_bootstrap_run(**self._kwargs())
            destination = self.workspace / str(plan["quarantinePath"])
            destination.parent.mkdir(parents=True)
            target = self.workspace / "unrelated-quarantine"
            target.mkdir()
            destination.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "collision is invalid",
            ):
                quarantine.inspect_failed_bootstrap_run(**self._kwargs())

    def test_attestation_rejects_archive_outside_canonical_root(self) -> None:
        with self._guards():
            reference = quarantine.quarantine_failed_bootstrap_run(
                **self._kwargs()
            )
            incident_path = self.workspace / str(reference["path"])
            copied_root = self.workspace / "copied-archive"
            shutil.copytree(incident_path.parent, copied_root)
            with self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "outside the canonical root",
            ):
                quarantine.verify_quarantine_incident_attestation(
                    workspace_root=self.workspace,
                    incident_path=copied_root / quarantine.INCIDENT_NAME,
                    expected_old_execution_contract_sha256=(
                        OLD_CONTRACT_SHA256
                    ),
                    expected_original_source_sha256=(
                        ORIGINAL_SOURCE_SHA256
                    ),
                    expected_replacement_source_sha256=(
                        REPLACEMENT_SOURCE_SHA256
                    ),
                )

    def test_cli_is_dry_run_unless_execute_is_explicit(self) -> None:
        base = [
            "--workspace-root",
            str(self.workspace),
            "--run-dir",
            str(self.run_dir),
            "--quarantine-root",
            str(self.quarantine_root),
            "--gpu-ledger",
            str(self.ledger_path),
            "--failure-log",
            str(self.log_path),
            "--gpu-ledger-run-id",
            LEDGER_RUN_ID,
            "--old-execution-contract",
            str(self.contract_path),
            "--old-execution-contract-sha256",
            OLD_CONTRACT_SHA256,
            "--run-manifest-sha256",
            self.manifest_sha256,
            "--replacement-source-sha256",
            REPLACEMENT_SOURCE_SHA256,
        ]
        with self._guards():
            dry_stdout = SimpleNamespace(buffer=io.BytesIO())
            with patch.object(quarantine.sys, "stdout", dry_stdout):
                self.assertEqual(quarantine.main(base), 0)
            self.assertTrue(self.run_dir.is_dir())
            self.assertFalse(self.quarantine_root.exists())

            execute_stdout = SimpleNamespace(buffer=io.BytesIO())
            with patch.object(quarantine.sys, "stdout", execute_stdout):
                self.assertEqual(quarantine.main([*base, "--execute"]), 0)
            result = json.loads(execute_stdout.buffer.getvalue())
            self.assertEqual(
                set(result),
                {"path", "bytes", "sha256", "incidentSha256"},
            )
            self.assertFalse(self.run_dir.exists())

    def test_resource_preflight_counts_the_entire_write_reserve(self) -> None:
        reserve = 4096
        with (
            patch.object(
                quarantine,
                "_directory_size",
                return_value=(
                    quarantine._ARTIFACT_LIMIT_BYTES - reserve + 1
                ),
            ),
            patch.object(
                quarantine.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=30 * 1024**3),
            ),
            self.assertRaisesRegex(
                quarantine.Phase4EBootstrapQuarantineError,
                "8 GiB artifact cap",
            ),
        ):
            quarantine._resource_preflight(
                self.control_root, reserve_bytes=reserve
            )


if __name__ == "__main__":
    unittest.main()
