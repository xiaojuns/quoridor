from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import quoridor.phase4_training as phase4_training
import quoridor.run_manifest as run_manifest
from quoridor.config import load_experiment_config
from quoridor.phase4_training import (
    Phase4ResourceLimitReached,
    Phase4TrainingError,
    prepare_phase4_lineage,
    run_phase4_training,
)
from quoridor.run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ResourceBudget,
    SourceFingerprint,
    build_run_manifest,
    fingerprint_sources,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = "1" * 64
NATIVE_SHA256 = "2" * 64
FIXED_ENVIRONMENT = {
    "python": {"version": "test"},
    "torch": {"version": "test", "cudaAvailable": False},
    "platform": {"system": "test"},
}


class RecordingLedger:
    def __init__(self, events: list[str], *, total_hours: float = 5.0) -> None:
        self.events = events
        self.total_hours = total_hours
        self.begins: list[tuple[str, float]] = []
        self.updates: list[tuple[str, float, str]] = []

    def begin(self, run_id: str, *, external_hours: float) -> float:
        self.events.append("ledger.begin")
        self.begins.append((run_id, external_hours))
        self.total_hours = max(self.total_hours, external_hours)
        return self.total_hours

    def update(
        self,
        run_id: str,
        *,
        elapsed_seconds: float,
        status: str,
    ) -> float:
        self.events.append(f"ledger.update.{status}")
        self.updates.append((run_id, elapsed_seconds, status))
        return self.total_hours


class RecordingStageLedger:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def ensure_stage(self, *_args, **_kwargs):
        self.events.append("stage.ensure")
        return {}

    def assert_available(self, *_args, **_kwargs):
        self.events.append("stage.available")
        return {"remainingHours": 5.0}

    @staticmethod
    def usage(_stage: str, global_hours: float):
        return {"globalHours": global_hours}


class DeviceResolutionFailure(RuntimeError):
    pass


class CacheRefreshFailure(RuntimeError):
    pass


class StopAfterLineage(RuntimeError):
    pass


class Phase4TrainingLifecycleTests(unittest.TestCase):
    def _config(self, workspace: Path):
        config = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        return replace(
            config,
            run_dir=workspace / "training/runs/phase4/lane",
            device="cuda",
        )

    def _base_patches(
        self,
        stack: ExitStack,
        *,
        config,
        ledger: RecordingLedger,
        stage_ledger: RecordingStageLedger,
    ) -> None:
        source = SourceFingerprint(
            sha256=SOURCE_SHA256,
            files=(),
            byte_size=0,
        )
        stack.enter_context(
            patch.object(phase4_training, "load_experiment_config", return_value=config)
        )
        stack.enter_context(
            patch.object(phase4_training, "_static_write_preflight", return_value={})
        )
        stack.enter_context(
            patch.object(phase4_training, "RunLock", return_value=nullcontext())
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "verify_phase4d_baseline_report",
                return_value={},
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "GpuBudgetLedger",
                return_value=ledger,
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "Phase4StageBudgetLedger",
                return_value=stage_ledger,
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "loaded_native_backend_hash",
                return_value=NATIVE_SHA256,
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "fingerprint_sources",
                return_value=source,
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training,
                "_load_replay_input_gate_report",
                return_value={
                    "sha256": "3" * 64,
                    "sourceSha256": SOURCE_SHA256,
                    "nativeBackendSha256": NATIVE_SHA256,
                },
            )
        )
        stack.enter_context(
            patch.object(
                phase4_training.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES + 1024**3
                ),
            )
        )
        stack.enter_context(
            patch.object(phase4_training, "directory_size", return_value=0)
        )

    @staticmethod
    def _run(workspace: Path, *, resume: bool, external_gpu_hours: float = 0.0):
        return run_phase4_training(
            config_path=workspace / "config.json",
            workspace_root=workspace,
            generations=1,
            resume=resume,
            device_override=None,
            external_gpu_hours=external_gpu_hours,
        )

    def test_fresh_device_failure_is_terminally_accounted_before_run_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = self._config(workspace)
            events: list[str] = []
            ledger = RecordingLedger(events)
            stage = RecordingStageLedger(events)
            lineage = Mock(side_effect=AssertionError("fresh lineage ran too early"))
            replay_store = Mock(side_effect=AssertionError("fresh replay ran too early"))

            def fail_device(_device: str):
                events.append("resolve_device")
                raise DeviceResolutionFailure("device preflight failed")

            with ExitStack() as stack:
                self._base_patches(
                    stack,
                    config=config,
                    ledger=ledger,
                    stage_ledger=stage,
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "resolve_device",
                        side_effect=fail_device,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "prepare_phase4_lineage",
                        lineage,
                    )
                )
                stack.enter_context(
                    patch.object(phase4_training, "ReplayStore", replay_store)
                )
                with self.assertRaisesRegex(
                    DeviceResolutionFailure,
                    "device preflight failed",
                ):
                    self._run(workspace, resume=False)

            self.assertEqual(
                [status for _, _, status in ledger.updates],
                ["running", "failed"],
            )
            self.assertLess(events.index("ledger.begin"), events.index("resolve_device"))
            lineage.assert_not_called()
            replay_store.assert_not_called()
            self.assertFalse((config.run_dir / "run-manifest.json").exists())
            self.assertFalse((config.run_dir / "replay").exists())

    def test_resume_cache_failure_precedes_device_and_gpu_ledger_begin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = self._config(workspace)
            (config.run_dir / "replay").mkdir(parents=True)
            (config.run_dir / "run-manifest.json").write_text(
                "fixture\n",
                encoding="ascii",
            )
            (config.run_dir / "replay/manifest.json").write_text(
                "fixture\n",
                encoding="ascii",
            )
            events: list[str] = []
            ledger = RecordingLedger(events)
            stage = RecordingStageLedger(events)
            resolve_device = Mock(
                side_effect=AssertionError("device resolved before resume cache")
            )
            cache = Mock()
            cache.refresh.side_effect = CacheRefreshFailure("cache is corrupt")

            with ExitStack() as stack:
                self._base_patches(
                    stack,
                    config=config,
                    ledger=ledger,
                    stage_ledger=stage,
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "prepare_phase4_lineage",
                        return_value=SimpleNamespace(resumed=True),
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "ReplayStore",
                        return_value=SimpleNamespace(game_count=1, ply_count=1),
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "ReplayReconstructionCache",
                        return_value=cache,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "resolve_device",
                        resolve_device,
                    )
                )
                with self.assertRaisesRegex(CacheRefreshFailure, "cache is corrupt"):
                    self._run(workspace, resume=True)

            cache.refresh.assert_called_once_with()
            resolve_device.assert_not_called()
            self.assertEqual(ledger.begins, [])
            self.assertEqual(ledger.updates, [])

    def test_external_gpu_floor_stops_after_begin_before_device_or_fresh_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = self._config(workspace)
            events: list[str] = []
            ledger = RecordingLedger(events)
            stage = RecordingStageLedger(events)
            resolve_device = Mock(
                side_effect=AssertionError("device resolved after exhausted floor")
            )
            lineage = Mock(side_effect=AssertionError("lineage wrote after exhausted floor"))

            with ExitStack() as stack:
                self._base_patches(
                    stack,
                    config=config,
                    ledger=ledger,
                    stage_ledger=stage,
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "resolve_device",
                        resolve_device,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "prepare_phase4_lineage",
                        lineage,
                    )
                )
                with self.assertRaisesRegex(
                    Phase4ResourceLimitReached,
                    "120 GPU-hour",
                ):
                    self._run(
                        workspace,
                        resume=False,
                        external_gpu_hours=120.0,
                    )

            self.assertEqual(len(ledger.begins), 1)
            self.assertEqual(ledger.begins[0][1], 120.0)
            self.assertEqual(
                [status for _, _, status in ledger.updates],
                ["running", "failed"],
            )
            resolve_device.assert_not_called()
            lineage.assert_not_called()
            self.assertFalse((config.run_dir / "run-manifest.json").exists())
            self.assertFalse((config.run_dir / "replay").exists())

    def test_fresh_runtime_environment_and_manifest_write_follow_ledger_begin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = self._config(workspace)
            events: list[str] = []
            ledger = RecordingLedger(events)
            stage = RecordingStageLedger(events)
            source = SourceFingerprint(
                sha256=SOURCE_SHA256,
                files=(),
                byte_size=0,
            )
            real_save = phase4_training.save_run_manifest

            def resolve(_device: str):
                events.append("resolve_device")
                return torch.device("cpu")

            def collect_environment():
                events.append("collect_runtime_environment")
                return FIXED_ENVIRONMENT

            def save_manifest(record, path):
                events.append("save_run_manifest")
                return real_save(record, path)

            def stop_at_replay(*_args, **_kwargs):
                events.append("ReplayStore")
                raise StopAfterLineage("lineage boundary reached")

            with ExitStack() as stack:
                self._base_patches(
                    stack,
                    config=config,
                    ledger=ledger,
                    stage_ledger=stage,
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "resolve_device",
                        side_effect=resolve,
                    )
                )
                stack.enter_context(
                    patch.object(
                        run_manifest,
                        "fingerprint_sources",
                        return_value=source,
                    )
                )
                stack.enter_context(
                    patch.object(
                        run_manifest,
                        "collect_runtime_environment",
                        side_effect=collect_environment,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "save_run_manifest",
                        side_effect=save_manifest,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phase4_training,
                        "ReplayStore",
                        side_effect=stop_at_replay,
                    )
                )
                with self.assertRaisesRegex(
                    StopAfterLineage,
                    "lineage boundary reached",
                ):
                    self._run(workspace, resume=False)

            self.assertLess(
                events.index("ledger.begin"),
                events.index("collect_runtime_environment"),
            )
            self.assertLess(
                events.index("collect_runtime_environment"),
                events.index("save_run_manifest"),
            )
            self.assertLess(
                events.index("save_run_manifest"),
                events.index("ReplayStore"),
            )
            self.assertTrue((config.run_dir / "run-manifest.json").is_file())
            self.assertFalse((config.run_dir / "replay").exists())
            self.assertEqual(ledger.updates[-1][2], "failed")

    def test_expected_source_hash_blocks_manifest_commit_on_build_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="ascii")
            expected = fingerprint_sources(workspace, [source]).sha256
            config = replace(
                self._config(workspace),
                device="cpu",
            )
            preflight = Mock(
                side_effect=AssertionError("write preflight ran after source drift")
            )

            def build_after_drift(**kwargs):
                source.write_text("VERSION = 2\n", encoding="ascii")
                return build_run_manifest(**kwargs)

            with (
                patch.object(
                    phase4_training,
                    "build_run_manifest",
                    side_effect=build_after_drift,
                ),
                patch.object(
                    phase4_training,
                    "_static_write_preflight",
                    preflight,
                ),
            ):
                with self.assertRaisesRegex(
                    Phase4TrainingError,
                    "before run manifest commit",
                ):
                    prepare_phase4_lineage(
                        workspace_root=workspace,
                        run_dir=config.run_dir,
                        config=config,
                        budget=ResourceBudget(gpu_hours=10),
                        resume=False,
                        source_paths=[source],
                        environment=FIXED_ENVIRONMENT,
                        expected_source_sha256=expected,
                    )

            preflight.assert_not_called()
            self.assertFalse((config.run_dir / "run-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
