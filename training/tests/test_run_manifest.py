from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from quoridor.run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    GenerationReportStore,
    ImmutableArtifactExistsError,
    ManifestCorruptionError,
    ResourceBudget,
    RunAlreadyLockedError,
    RunLock,
    build_run_manifest,
    collect_runtime_environment,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    save_run_manifest,
)


FIXED_ENVIRONMENT = {
    "python": {"version": "3.10.12"},
    "torch": {"version": "2.7.1+cu126", "cudaAvailable": True},
    "platform": {"system": "Linux", "machine": "x86_64"},
}


class SourceAndConfigFingerprintTests(unittest.TestCase):
    def test_source_hash_is_path_order_and_mtime_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            first = root / "source/a.py"
            second = root / "source/b.py"
            first.write_text("VALUE = 1\n", encoding="utf-8")
            second.write_text("VALUE = 2\n", encoding="utf-8")

            forward = fingerprint_sources(root, [first, second])
            os.utime(first, (1, 1))
            reverse = fingerprint_sources(root, [second, first])
            self.assertEqual(forward, reverse)
            self.assertEqual(
                [item.path for item in forward.files],
                ["source/a.py", "source/b.py"],
            )

            second.write_text("VALUE = 3\n", encoding="utf-8")
            changed = fingerprint_sources(root, [root / "source"])
            self.assertNotEqual(changed.sha256, forward.sha256)

    def test_config_hash_uses_canonical_semantic_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text('{"network":{"blocks":16},"seed":7}\n')
            second.write_text(
                '{\n  "seed": 7,\n  "network": {"blocks": 16}\n}\n'
            )
            first_hash = fingerprint_config(first)
            second_hash = fingerprint_config(second)
            mapping_hash = fingerprint_config(
                {"seed": 7, "network": {"blocks": 16}}
            )
            self.assertEqual(first_hash.sha256, second_hash.sha256)
            self.assertEqual(first_hash, mapping_hash)

    def test_source_paths_must_be_inside_workspace_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as other:
            source = Path(workspace) / "model.py"
            source.write_text("pass\n")
            self.assertEqual(fingerprint_sources(workspace, [source]).files[0].path, "model.py")
            outside = Path(other) / "outside.py"
            outside.write_text("pass\n")
            with self.assertRaisesRegex(ValueError, "outside workspace"):
                fingerprint_sources(workspace, [outside])


class RunManifestTests(unittest.TestCase):
    def _record(self, root: Path):
        source = root / "trainer.py"
        source.write_text("STEP = 1\n", encoding="utf-8")
        return build_run_manifest(
            workspace_root=root,
            config={"seed": 17, "device": "cuda"},
            budget=ResourceBudget(gpu_hours=72),
            run_id="phase4-test",
            source_paths=[source],
            environment=FIXED_ENVIRONMENT,
            metadata={"purpose": "unit-test"},
            created_at="2026-07-14T00:00:00Z",
        )

    def test_manifest_records_local_policy_environment_and_stable_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root)
            payload = record.payload
            self.assertTrue(payload["executionPolicy"]["localOnly"])
            self.assertFalse(payload["executionPolicy"]["networkRequired"])
            self.assertFalse(payload["provenance"]["gitMetadataUsed"])
            self.assertEqual(
                payload["resourceBudget"]["artifactLimitBytes"],
                DEFAULT_ARTIFACT_LIMIT_BYTES,
            )
            self.assertEqual(payload["environment"], FIXED_ENVIRONMENT)
            self.assertEqual(record.sha256, payload["manifestSha256"])

            path = save_run_manifest(record, root / "run-manifest.json")
            self.assertEqual(load_run_manifest(path), record)
            with self.assertRaises(ImmutableArtifactExistsError):
                save_run_manifest(record, path)
            self.assertFalse(tuple(root.glob(".*.tmp")))

    def test_manifest_self_hash_detects_content_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_run_manifest(
                self._record(root), root / "run-manifest.json"
            )
            payload = json.loads(path.read_text())
            payload["runId"] = "mutated"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ManifestCorruptionError, "hash"):
                load_run_manifest(path)

    def test_budget_and_metadata_reject_invalid_values(self) -> None:
        extended = ResourceBudget(artifact_limit_bytes=MAXIMUM_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(
            extended.artifact_limit_bytes,
            MAXIMUM_ARTIFACT_LIMIT_BYTES,
        )
        expanded = ResourceBudget(artifact_limit_bytes=EXTENDED_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(expanded.artifact_limit_bytes, EXTENDED_ARTIFACT_LIMIT_BYTES)
        with self.assertRaisesRegex(ValueError, "16 GiB"):
            ResourceBudget(artifact_limit_bytes=EXTENDED_ARTIFACT_LIMIT_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "20 GiB"):
            ResourceBudget(
                minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES - 1
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ResourceBudget(metadata={"bad": float("nan")})

    def test_runtime_environment_has_required_python_torch_and_platform_fields(self) -> None:
        environment = collect_runtime_environment()
        self.assertIn("version", environment["python"])
        self.assertIn("cudaAvailable", environment["torch"])
        self.assertIn("machine", environment["platform"])


class GenerationReportAndLockTests(unittest.TestCase):
    def test_generation_reports_are_atomic_addressed_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GenerationReportStore(
                directory,
                run_manifest_sha256="a" * 64,
            )
            first = store.write(
                0,
                {"loss": 1.25, "games": 64},
                created_at="2026-07-14T01:00:00Z",
            )
            store.write(
                2,
                {"loss": 0.75, "games": 128},
                created_at="2026-07-14T02:00:00Z",
            )
            self.assertEqual(store.generations(), (0, 2))
            self.assertEqual(store.load(0), first)
            with self.assertRaises(ImmutableArtifactExistsError):
                store.write(0, {"loss": 0.0})
            self.assertFalse(tuple(Path(directory).glob(".*.tmp")))

    def test_generation_report_hash_and_run_identity_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GenerationReportStore(directory, run_manifest_sha256="b" * 64)
            record = store.write(3, {"accepted": True})
            payload = json.loads(record.path.read_text())
            payload["report"]["accepted"] = False
            record.path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ManifestCorruptionError, "hash"):
                store.load(3)

            other = GenerationReportStore(directory, run_manifest_sha256="c" * 64)
            with self.assertRaisesRegex(ManifestCorruptionError, "another run"):
                other.load(3)

    def test_fcntl_lock_allows_exactly_one_writer_and_can_be_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.lock"
            first = RunLock(path, run_id="phase4")
            second = RunLock(path, run_id="phase4")
            first.acquire()
            self.assertTrue(first.acquired)
            with self.assertRaises(RunAlreadyLockedError):
                second.acquire()
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["pid"], os.getpid())
            self.assertEqual(metadata["runId"], "phase4")
            first.release()
            self.assertFalse(first.acquired)

            with second:
                self.assertTrue(second.acquired)
            self.assertFalse(second.acquired)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
