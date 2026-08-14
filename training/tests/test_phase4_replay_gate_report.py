from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import quoridor.phase4_training as phase4_training
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes, sha256_file


SOURCE_SHA256 = "1" * 64
NATIVE_SHA256 = "2" * 64
PREFIX_SHA256 = "4" * 64


class ReplayInputGateReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name).resolve()
        self.control_root = (
            self.workspace / phase4_training.PHASE4_CONTROL_RELATIVE
        )
        self.fixture_root = (
            self.workspace
            / phase4_training.REPLAY_INPUT_GATE_FIXTURE_RELATIVE
        )
        self.fixture_root.mkdir(parents=True)
        runtime_payload = {
            "workspace": str(self.workspace),
            "hostname": "test-host",
            "machine": "test-machine",
            "processor": "test-processor",
            "platform": "test-platform",
            "cpuCount": 8,
            "python": "test-python",
            "torch": "test-torch",
        }
        self.runtime = {
            "sha256": sha256_bytes(canonical_json_bytes(runtime_payload)),
            "payload": runtime_payload,
        }

    def _valid_core(self) -> dict[str, object]:
        batch_seconds = [0.25] * phase4_training.REPLAY_INPUT_GATE_BATCHES
        refresh_seconds = 100.0
        fixture_plies = phase4_training.REPLAY_INPUT_GATE_FIXTURE_PLIES
        return {
            "schemaVersion": 1,
            "kind": "phase4-replay-input-gate",
            "status": "completed",
            "sourceSha256": SOURCE_SHA256,
            "nativeBackendSha256": NATIVE_SHA256,
            "runtime": copy.deepcopy(self.runtime),
            "fixturePath": str(
                phase4_training.REPLAY_INPUT_GATE_FIXTURE_RELATIVE
            ),
            "fixtureManifestSha256": (
                phase4_training.REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256
            ),
            "fixtureGames": phase4_training.REPLAY_INPUT_GATE_FIXTURE_GAMES,
            "fixturePlies": fixture_plies,
            "cache": {
                "algorithmVersion": (
                    phase4_training.REPLAY_CACHE_ALGORITHM_VERSION
                ),
                "manifestSha256": (
                    phase4_training.REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256
                ),
                "prefixSha256": PREFIX_SHA256,
                "games": phase4_training.REPLAY_INPUT_GATE_FIXTURE_GAMES,
                "plies": fixture_plies,
                "estimatedBytes": (
                    fixture_plies
                    * phase4_training.REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY
                ),
                "reusedGames": 0,
                "decodedGames": (
                    phase4_training.REPLAY_INPUT_GATE_FIXTURE_GAMES
                ),
                "decodedPlies": fixture_plies,
                "refreshSeconds": refresh_seconds,
                "cumulativeBuildSeconds": refresh_seconds,
                "decodedPositionsPerSecond": (
                    fixture_plies / refresh_seconds
                ),
            },
            "parity": {
                "sample": True,
                "rng": True,
                "tensors": True,
                "legacySeconds": 0.5,
                "cachedSeconds": 0.25,
            },
            "inputTiming": {
                "batches": phase4_training.REPLAY_INPUT_GATE_BATCHES,
                "secondsMean": sum(batch_seconds) / len(batch_seconds),
                "secondsMaximum": max(batch_seconds),
                "initialGateBatches": (
                    phase4_training.REPLAY_INPUT_GATE_BATCHES
                ),
                "maximumMeanSeconds": (
                    phase4_training.REPLAY_INPUT_GATE_MAX_MEAN_SECONDS
                ),
                "maximumInitialBatchSeconds": (
                    phase4_training.REPLAY_INPUT_GATE_MAX_BATCH_SECONDS
                ),
                "maximumLiveBatchSeconds": (
                    phase4_training.REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS
                ),
                "initialGateCompleted": True,
                "batchSeconds": batch_seconds,
            },
            "gates": {
                "fixture": True,
                "cacheBuildSeconds": True,
                "cacheBuildThroughput": True,
                "sampleParity": True,
                "rngParity": True,
                "tensorParity": True,
                "batchMean": True,
                "batchMaximum": True,
            },
        }

    def _report_path(self) -> Path:
        return phase4_training._replay_input_gate_report_path(
            self.control_root,
            SOURCE_SHA256,
            NATIVE_SHA256,
            str(self.runtime["sha256"]),
        )

    def _write_core(
        self,
        core: dict[str, object],
        *,
        trailing_whitespace: bool = False,
    ) -> Path:
        report = {
            **core,
            "reportCoreSha256": sha256_bytes(canonical_json_bytes(core)),
        }
        encoded = canonical_json_bytes(report)
        if trailing_whitespace:
            encoded += b" "
        path = self._report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return path

    def _load(self) -> dict[str, object]:
        with patch.object(
            phase4_training,
            "_replay_input_gate_runtime",
            return_value=copy.deepcopy(self.runtime),
        ), patch.object(
            phase4_training,
            "_verify_replay_input_gate_fixture",
        ):
            return phase4_training._load_replay_input_gate_report(
                workspace=self.workspace,
                control_root=self.control_root,
                source_sha256=SOURCE_SHA256,
                native_backend_sha256=NATIVE_SHA256,
            )

    def test_valid_canonical_report_returns_bound_evidence(self) -> None:
        report_path = self._write_core(self._valid_core())

        evidence = self._load()

        self.assertEqual(
            evidence["path"],
            str(report_path.relative_to(self.workspace)),
        )
        self.assertEqual(evidence["sha256"], sha256_file(report_path))
        self.assertEqual(evidence["sourceSha256"], SOURCE_SHA256)
        self.assertEqual(evidence["nativeBackendSha256"], NATIVE_SHA256)
        self.assertEqual(evidence["runtimeSha256"], self.runtime["sha256"])
        self.assertEqual(
            evidence["fixtureManifestSha256"],
            phase4_training.REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256,
        )

    def test_noncanonical_json_is_rejected_before_reuse(self) -> None:
        self._write_core(self._valid_core(), trailing_whitespace=True)

        with self.assertRaisesRegex(
            phase4_training.Phase4TrainingError,
            "not canonical JSON",
        ):
            self._load()

    def test_report_core_self_hash_rejects_canonical_tampering(self) -> None:
        path = self._write_core(self._valid_core())
        payload = phase4_training.json.loads(path.read_bytes())
        payload["parity"]["cachedSeconds"] = 0.75
        path.write_bytes(canonical_json_bytes(payload))

        with self.assertRaisesRegex(
            phase4_training.Phase4TrainingError,
            "core hash differs",
        ):
            self._load()

    def test_source_native_and_runtime_content_are_exactly_bound(self) -> None:
        mutations = (
            ("source", lambda core: core.__setitem__("sourceSha256", "a" * 64)),
            (
                "native",
                lambda core: core.__setitem__(
                    "nativeBackendSha256",
                    "b" * 64,
                ),
            ),
            (
                "runtime",
                lambda core: core.__setitem__(
                    "runtime",
                    {
                        "sha256": "c" * 64,
                        "payload": {
                            **self.runtime["payload"],
                            "hostname": "another-host",
                        },
                    },
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(binding=label):
                core = self._valid_core()
                mutate(core)
                self._write_core(core)
                with self.assertRaisesRegex(
                    phase4_training.Phase4TrainingError,
                    "provenance is stale",
                ):
                    self._load()

    def test_report_address_changes_with_each_provenance_hash(self) -> None:
        original = self._report_path()
        for label, source, native, runtime in (
            ("source", "a" * 64, NATIVE_SHA256, self.runtime["sha256"]),
            ("native", SOURCE_SHA256, "b" * 64, self.runtime["sha256"]),
            ("runtime", SOURCE_SHA256, NATIVE_SHA256, "c" * 64),
        ):
            with self.subTest(binding=label):
                changed = phase4_training._replay_input_gate_report_path(
                    self.control_root,
                    str(source),
                    str(native),
                    str(runtime),
                )
                self.assertNotEqual(changed, original)

    def test_exact_schema_workload_and_gate_contracts_are_enforced(self) -> None:
        cases = (
            (
                "top-level schema",
                lambda core: core.__setitem__("unexpected", True),
                "schema changed",
            ),
            (
                "cache schema",
                lambda core: core["cache"].__setitem__("unexpected", True),
                "workload is invalid",
            ),
            (
                "frozen game count",
                lambda core: core["cache"].__setitem__(
                    "games",
                    phase4_training.REPLAY_INPUT_GATE_FIXTURE_GAMES - 1,
                ),
                "workload is invalid",
            ),
            (
                "declared threshold",
                lambda core: core["inputTiming"].__setitem__(
                    "maximumMeanSeconds",
                    phase4_training.REPLAY_INPUT_GATE_MAX_MEAN_SECONDS + 0.01,
                ),
                "workload is invalid",
            ),
            (
                "failed gate",
                lambda core: core["gates"].__setitem__("tensorParity", False),
                "did not pass every gate",
            ),
            (
                "batch timing",
                lambda core: core["inputTiming"]["batchSeconds"].__setitem__(
                    0,
                    phase4_training.REPLAY_INPUT_GATE_MAX_BATCH_SECONDS + 0.01,
                ),
                "batch timings are invalid",
            ),
            (
                "build duration",
                lambda core: (
                    core["cache"].__setitem__(
                        "refreshSeconds",
                        phase4_training.REPLAY_INPUT_GATE_MAX_BUILD_SECONDS
                        + 0.01,
                    ),
                    core["cache"].__setitem__(
                        "cumulativeBuildSeconds",
                        phase4_training.REPLAY_INPUT_GATE_MAX_BUILD_SECONDS
                        + 0.01,
                    ),
                ),
                "cache build seconds is outside its threshold",
            ),
            (
                "mean duration",
                lambda core: core["inputTiming"].__setitem__(
                    "secondsMean",
                    phase4_training.REPLAY_INPUT_GATE_MAX_MEAN_SECONDS + 0.01,
                ),
                "batch mean seconds is outside its threshold",
            ),
            (
                "maximum duration",
                lambda core: core["inputTiming"].__setitem__(
                    "secondsMaximum",
                    phase4_training.REPLAY_INPUT_GATE_MAX_BATCH_SECONDS + 0.01,
                ),
                "batch maximum seconds is outside its threshold",
            ),
            (
                "timing summary",
                lambda core: core["inputTiming"].__setitem__(
                    "secondsMean",
                    0.26,
                ),
                "timing summary is inconsistent",
            ),
            (
                "build throughput",
                lambda core: core["cache"].__setitem__(
                    "decodedPositionsPerSecond",
                    (
                        phase4_training
                        .REPLAY_INPUT_GATE_MIN_BUILD_POSITIONS_PER_SECOND
                        - 0.01
                    ),
                ),
                "cache build rate is invalid",
            ),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(contract=label):
                core = self._valid_core()
                mutate(core)
                self._write_core(core)
                with self.assertRaisesRegex(
                    phase4_training.Phase4TrainingError,
                    expected_error,
                ):
                    self._load()

    def test_existing_report_is_reused_idempotently_without_reconstruction(
        self,
    ) -> None:
        report_path = self._write_core(self._valid_core())
        original_bytes = report_path.read_bytes()
        progress = Mock()
        source = SimpleNamespace(sha256=SOURCE_SHA256)

        with patch.object(
            phase4_training,
            "_verify_replay_input_gate_fixture",
        ), patch.object(
            phase4_training,
            "fingerprint_sources",
            return_value=source,
        ) as fingerprint_sources, patch.object(
            phase4_training,
            "loaded_native_backend_hash",
            return_value=NATIVE_SHA256,
        ) as native_hash, patch.object(
            phase4_training,
            "_replay_input_gate_runtime",
            return_value=copy.deepcopy(self.runtime),
        ) as runtime, patch.object(
            phase4_training.shutil,
            "copytree",
            side_effect=AssertionError("fixture copy must not run on reuse"),
        ) as copytree, patch.object(
            phase4_training,
            "_load_replay_store_from_manifest",
            side_effect=AssertionError("replay loading must not run on reuse"),
        ) as load_replay, patch.object(
            phase4_training,
            "ReplayReconstructionCache",
            side_effect=AssertionError("reconstruction must not run on reuse"),
        ) as reconstruct:
            first = phase4_training._run_phase4_replay_input_gate_locked(
                workspace_root=self.workspace,
                progress_callback=progress,
            )
            second = phase4_training._run_phase4_replay_input_gate_locked(
                workspace_root=self.workspace,
                progress_callback=progress,
            )

        self.assertTrue(first["reused"])
        self.assertEqual(first, second)
        self.assertEqual(report_path.read_bytes(), original_bytes)
        progress.assert_not_called()
        copytree.assert_not_called()
        load_replay.assert_not_called()
        reconstruct.assert_not_called()
        self.assertEqual(fingerprint_sources.call_count, 4)
        self.assertEqual(native_hash.call_count, 4)
        self.assertEqual(runtime.call_count, 6)

    def test_public_gate_holds_the_single_writer_lock_while_running(self) -> None:
        events: list[str] = []
        lock_identity: dict[str, object] = {}
        progress = Mock()
        expected = {"status": "completed"}

        class RecordingLock:
            def __init__(inner_self, path: Path, *, run_id: str) -> None:
                lock_identity["path"] = path
                lock_identity["runId"] = run_id

            def __enter__(inner_self) -> RecordingLock:
                events.append("enter")
                return inner_self

            def __exit__(
                inner_self,
                exc_type: object,
                exc_value: object,
                traceback: object,
            ) -> None:
                events.append("exit")

        def run_locked(
            *,
            workspace_root: str | Path,
            artifact_limit_bytes: int,
            progress_callback: object,
        ) -> dict[str, object]:
            self.assertEqual(events, ["enter"])
            self.assertEqual(Path(workspace_root), self.workspace)
            self.assertEqual(
                artifact_limit_bytes,
                phase4_training.EXTENDED_ARTIFACT_LIMIT_BYTES,
            )
            self.assertIs(progress_callback, progress)
            events.append("run")
            return expected

        with patch.object(
            phase4_training,
            "_static_write_preflight",
        ) as preflight, patch.object(
            phase4_training,
            "RunLock",
            RecordingLock,
        ), patch.object(
            phase4_training,
            "_run_phase4_replay_input_gate_locked",
            side_effect=run_locked,
        ):
            result = phase4_training.run_phase4_replay_input_gate(
                workspace_root=self.workspace,
                artifact_limit_bytes=phase4_training.EXTENDED_ARTIFACT_LIMIT_BYTES,
                progress_callback=progress,
            )

        self.assertIs(result, expected)
        self.assertEqual(events, ["enter", "run", "exit"])
        self.assertEqual(
            lock_identity["path"],
            self.control_root.resolve() / phase4_training.GPU_LOCK_NAME,
        )
        self.assertRegex(
            str(lock_identity["runId"]),
            r"^phase4-replay-input-gate-\d+-\d+$",
        )
        preflight.assert_called_once_with(
            artifact_root=self.control_root.resolve(),
            artifact_limit_bytes=phase4_training.EXTENDED_ARTIFACT_LIMIT_BYTES,
            reserve_bytes=phase4_training._REPORT_WRITE_RESERVE_BYTES,
        )

    def test_public_gate_rejects_an_unfrozen_artifact_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "8, 12, or 16 GiB"):
            phase4_training.run_phase4_replay_input_gate(
                workspace_root=self.workspace,
                artifact_limit_bytes=10 * 1024**3,
            )


if __name__ == "__main__":
    unittest.main()
