from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import quoridor.phase4_replay_migration as replay_migration
from quoridor.phase4_baseline import prepare_bootstrap_baseline
from quoridor.phase4_replay_migration import (
    MIGRATION_MANIFEST_NAME,
    Phase4ReplayMigrationConfigurationError,
    Phase4ReplayMigrationCorruptionError,
    Phase4ReplayMigrationLineageError,
    main,
    prepare_warm_replay_migration,
    verify_warm_replay_migration,
)
from quoridor.replay_store import ReplayStore
from quoridor.run_manifest import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from training.tests.test_phase4_baseline import (
    BASELINE_RELATIVE,
    PILOT_RELATIVE,
    _PilotFixture,
    _self_hashed,
    _write_json,
)
from training.tests.test_replay_store import _game, _three_ply_game


SOURCE_CAPACITY = {
    "games": 8,
    "plies": 20,
    "bytes": 1_000_000,
}
TARGET_CAPACITY = {
    "games": 16,
    "plies": 40,
    "bytes": 2_000_000,
}
TARGET_RELATIVE = Path("training/runs/phase4/phase4e/replay")


class _MigrationPilotFixture(_PilotFixture):
    def _build_lane(
        self,
        lane: str,
        relative_payload: dict[str, object],
        *,
        generation_offset: int,
    ) -> dict[str, object]:
        artifact = super()._build_lane(
            lane,
            relative_payload,
            generation_offset=generation_offset,
        )
        if lane != "warm":
            return artifact

        run_dir = artifact["runDir"]
        checkpoint_path = artifact["checkpointPath"]
        assert isinstance(run_dir, Path)
        assert isinstance(checkpoint_path, Path)
        replay_root = run_dir / "replay"
        store = ReplayStore(
            replay_root,
            max_games=SOURCE_CAPACITY["games"],
            max_plies=SOURCE_CAPACITY["plies"],
            max_bytes=SOURCE_CAPACITY["bytes"],
        )
        store.append(_game("fixture-zeta", generation=2))
        store.append(_three_ply_game("fixture-alpha"))
        store.append(_game("fixture-middle", generation=5))
        verification = store.verify(deep=True)
        manifest_path = replay_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prefix_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "schemaVersion": 1,
                    "shards": manifest["shards"],
                }
            )
        )
        snapshot = {
            "manifestSha256": sha256_file(manifest_path),
            "games": verification.game_count,
            "plies": verification.ply_count,
            "bytes": sum(
                int(entry["bytes"]) for entry in manifest["shards"]
            ),
            "manifestBytes": manifest_path.stat().st_size,
            "committedBytes": verification.committed_bytes,
            "deepVerified": True,
            "gameIds": list(store.game_ids),
            "prefixSha256": prefix_sha256,
        }

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint["replaySnapshot"] = snapshot
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_record = {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "step": 320,
        }

        report_path = run_dir / "reports/generation-00000004.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["report"]["checkpoint"] = checkpoint_record
        report_core = dict(report)
        report_core.pop("reportSha256")
        _write_json(
            report_path,
            _self_hashed(report_core, "reportSha256"),
        )
        metadata_path = checkpoint_path.with_suffix(".json")
        _write_json(
            metadata_path,
            {
                "schemaVersion": 1,
                "generation": 4,
                "step": 320,
                "file": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha256,
                "identity": checkpoint["identity"],
            },
        )
        _write_json(
            checkpoint_path.parent / "latest.json",
            {
                "schemaVersion": 1,
                "generation": 4,
                "step": 320,
                "metadata": metadata_path.name,
                "sha256": checkpoint_sha256,
            },
        )
        artifact["checkpointSha256"] = checkpoint_sha256
        self.replay_root = replay_root
        self.replay_snapshot = snapshot
        return artifact


class Phase4ReplayMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _MigrationPilotFixture(Path(self.temporary.name))
        prepare_bootstrap_baseline(
            workspace_root=self.fixture.root,
            pilot_root=PILOT_RELATIVE,
            output_root=BASELINE_RELATIVE,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def target_root(self) -> Path:
        return self.fixture.root / TARGET_RELATIVE

    def arguments(
        self,
        *,
        target_root: str | Path = TARGET_RELATIVE,
        capacity: dict[str, int] | None = None,
    ) -> dict[str, object]:
        selected = capacity or TARGET_CAPACITY
        return {
            "workspace_root": self.fixture.root,
            "baseline_root": BASELINE_RELATIVE,
            "source_root": (
                PILOT_RELATIVE / "lanes/warm/replay"
            ),
            "target_root": target_root,
            "target_max_games": selected["games"],
            "target_max_plies": selected["plies"],
            "target_max_bytes": selected["bytes"],
        }

    def pilot_files(self) -> dict[str, tuple[int, str]]:
        return {
            path.relative_to(self.fixture.pilot_root).as_posix(): (
                path.stat().st_size,
                sha256_file(path),
            )
            for path in self.fixture.pilot_root.rglob("*")
            if path.is_file()
        }

    def test_copy_migration_is_deep_verified_idempotent_and_ordered(self) -> None:
        pilot_before = self.pilot_files()
        first = prepare_warm_replay_migration(**self.arguments())
        second = prepare_warm_replay_migration(**self.arguments())
        verified = verify_warm_replay_migration(**self.arguments())

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["manifestSha256"], second["manifestSha256"])
        self.assertEqual(first["manifestSha256"], verified["manifestSha256"])
        self.assertEqual(first["games"], 3)
        self.assertEqual(first["plies"], 5)
        self.assertEqual(first["copied"], 3)
        self.assertEqual(first["hardLinked"], 0)
        self.assertTrue(first["lineageVerified"])
        self.assertFalse(first["formalPromotion"])
        self.assertEqual(self.pilot_files(), pilot_before)

        source_manifest = json.loads(
            (self.fixture.replay_root / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        target_manifest = json.loads(
            (self.target_root / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            target_manifest["shards"],
            source_manifest["shards"],
        )
        self.assertEqual(
            target_manifest["nextSequence"],
            source_manifest["nextSequence"],
        )
        self.assertEqual(
            target_manifest["capacity"],
            TARGET_CAPACITY,
        )
        self.assertEqual(
            first["prefixSha256"],
            self.fixture.replay_snapshot["prefixSha256"],
        )
        for entry in source_manifest["shards"]:
            source = self.fixture.replay_root / entry["file"]
            target = self.target_root / entry["file"]
            self.assertFalse(os.path.samefile(source, target))
            self.assertEqual(sha256_file(source), sha256_file(target))

    def test_hardlink_mode_is_verified(self) -> None:
        result = prepare_warm_replay_migration(
            **self.arguments(
                target_root="training/runs/phase4/phase4e/replay-linked"
            ),
            transfer_mode="hardlink",
        )
        linked_root = (
            self.fixture.root
            / "training/runs/phase4/phase4e/replay-linked"
        )
        source_manifest = json.loads(
            (self.fixture.replay_root / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["hardLinked"], 3)
        self.assertEqual(result["copied"], 0)
        for entry in source_manifest["shards"]:
            self.assertTrue(
                os.path.samefile(
                    self.fixture.replay_root / entry["file"],
                    linked_root / entry["file"],
                )
            )

    def test_insufficient_capacity_and_nonempty_target_are_rejected(self) -> None:
        insufficient = {
            **TARGET_CAPACITY,
            "games": SOURCE_CAPACITY["games"] - 1,
        }
        with self.assertRaisesRegex(
            Phase4ReplayMigrationConfigurationError,
            "at least the source",
        ):
            prepare_warm_replay_migration(
                **self.arguments(capacity=insufficient)
            )
        self.assertFalse(self.target_root.exists())

        self.target_root.mkdir(parents=True)
        (self.target_root / "unrelated.txt").write_text(
            "occupied\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase4ReplayMigrationConfigurationError,
            "must be empty",
        ):
            prepare_warm_replay_migration(**self.arguments())

    def test_partial_transfer_failure_rolls_back_only_new_target(self) -> None:
        pilot_before = self.pilot_files()
        original = replay_migration._transfer_shard
        calls = 0

        def fail_on_second(
            source: Path,
            destination: Path,
            mode: str,
        ) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise Phase4ReplayMigrationConfigurationError(
                    "injected transfer failure"
                )
            return original(source, destination, mode)

        with mock.patch.object(
            replay_migration,
            "_transfer_shard",
            side_effect=fail_on_second,
        ):
            with self.assertRaisesRegex(
                Phase4ReplayMigrationConfigurationError,
                "injected",
            ):
                prepare_warm_replay_migration(**self.arguments())
        self.assertFalse(self.target_root.exists())
        self.assertEqual(self.pilot_files(), pilot_before)

    def test_wrong_source_and_pilot_target_are_rejected(self) -> None:
        wrong_source = (
            self.fixture.root
            / "training/runs/phase4/unrelated-replay"
        )
        ReplayStore(
            wrong_source,
            max_games=SOURCE_CAPACITY["games"],
            max_plies=SOURCE_CAPACITY["plies"],
            max_bytes=SOURCE_CAPACITY["bytes"],
        )
        with self.assertRaisesRegex(
            Phase4ReplayMigrationLineageError,
            "not the verified",
        ):
            prepare_warm_replay_migration(
                **{
                    **self.arguments(),
                    "source_root": wrong_source,
                }
            )
        with self.assertRaisesRegex(
            Phase4ReplayMigrationConfigurationError,
            "verified pilot",
        ):
            prepare_warm_replay_migration(
                **self.arguments(
                    target_root=PILOT_RELATIVE / "phase4e-replay"
                )
            )

    def test_source_shard_tampering_is_rejected_without_cleanup(self) -> None:
        manifest = json.loads(
            (self.fixture.replay_root / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        shard = self.fixture.replay_root / manifest["shards"][0]["file"]
        shard.write_bytes(shard.read_bytes() + b"tampered")
        with self.assertRaisesRegex(
            Phase4ReplayMigrationCorruptionError,
            "hash/size mismatch",
        ):
            prepare_warm_replay_migration(**self.arguments())
        self.assertTrue(shard.is_file())
        self.assertFalse(self.target_root.exists())

    def test_target_shard_and_manifest_tampering_are_rejected(self) -> None:
        prepare_warm_replay_migration(**self.arguments())
        with self.assertRaisesRegex(
            Phase4ReplayMigrationConfigurationError,
            "differs from the requested",
        ):
            verify_warm_replay_migration(
                **self.arguments(
                    capacity={
                        **TARGET_CAPACITY,
                        "games": TARGET_CAPACITY["games"] + 1,
                    }
                )
            )
        manifest = json.loads(
            (self.target_root / "manifest.json").read_text(encoding="utf-8")
        )
        shard = self.target_root / manifest["shards"][0]["file"]
        shard.write_bytes(shard.read_bytes() + b"tampered")
        with self.assertRaisesRegex(
            Phase4ReplayMigrationCorruptionError,
            "hash/size mismatch",
        ):
            verify_warm_replay_migration(**self.arguments())

        other_target = Path("training/runs/phase4/phase4e/replay-other")
        prepare_warm_replay_migration(
            **self.arguments(target_root=other_target)
        )
        migration_path = (
            self.fixture.root / other_target / MIGRATION_MANIFEST_NAME
        )
        migration_path.write_bytes(migration_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            Phase4ReplayMigrationCorruptionError,
            "not canonical",
        ):
            verify_warm_replay_migration(
                **self.arguments(target_root=other_target)
            )

    def test_cli_prepare_and_verify(self) -> None:
        common = [
            "--workspace-root",
            str(self.fixture.root),
            "--baseline-root",
            str(BASELINE_RELATIVE),
            "--source-root",
            str(PILOT_RELATIVE / "lanes/warm/replay"),
            "--target-root",
            str(TARGET_RELATIVE),
            "--target-max-games",
            str(TARGET_CAPACITY["games"]),
            "--target-max-plies",
            str(TARGET_CAPACITY["plies"]),
            "--target-max-bytes",
            str(TARGET_CAPACITY["bytes"]),
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["prepare", *common]), 0)
        self.assertTrue(json.loads(output.getvalue())["created"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["verify", *common]), 0)
        self.assertTrue(json.loads(output.getvalue())["lineageVerified"])


if __name__ == "__main__":
    unittest.main()
