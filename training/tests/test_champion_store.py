from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from quoridor.champion_store import (
    CHAMPION_MODEL_SCHEMA_VERSION,
    CHAMPION_STORE_SCHEMA_VERSION,
    ArtifactReference,
    ChampionBudgetError,
    ChampionConfigurationError,
    ChampionCorruptionError,
    ChampionStore,
)
from quoridor.run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class TinyModel(nn.Module):
    def __init__(self, marker: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.fill_(marker)
            self.linear.bias.fill_(-marker)


def _arena_reference(root: Path, generation: int) -> ArtifactReference:
    path = root / f"arena-{generation:04d}.json"
    path.write_text(
        json.dumps({"generation": generation, "accepted": True}),
        encoding="utf-8",
    )
    return ArtifactReference.from_file(path, relative_to=root)


def _publish(
    store: ChampionStore,
    root: Path,
    generation: int,
    parent_id: str | None,
    *,
    milestone: bool = False,
    best: bool = False,
    product: bool = False,
):
    return store.publish(
        TinyModel(float(generation + 1)),
        generation=generation,
        parent_id=parent_id,
        config_sha256=_digest("config"),
        source_sha256=_digest("source"),
        opening_suite_sha256=_digest("openings"),
        arena_report=_arena_reference(root, generation),
        milestone=milestone,
        best=best,
        product=product,
        metadata={"elo": generation * 10.0},
        accepted_at=f"2026-07-14T{generation % 24:02d}:00:00Z",
    )


class ChampionStoreRoundTripTests(unittest.TestCase):
    def test_model_only_publish_manifest_hash_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champions = root / "champions"
            store = ChampionStore(champions)
            source_model = TinyModel(3.0)
            arena = _arena_reference(root, 0)
            record = store.publish(
                source_model,
                generation=0,
                parent_id=None,
                config_sha256=_digest("config"),
                source_sha256=_digest("source"),
                opening_suite_sha256=_digest("openings"),
                arena_report=arena,
                best=True,
                product=True,
                metadata={"winRate": 0.625},
                accepted_at="2026-07-14T00:00:00Z",
            )

            self.assertTrue(record.retained)
            self.assertEqual(store.pointers["latest"], record.champion_id)
            self.assertEqual(store.pointers["best"], record.champion_id)
            self.assertEqual(store.pointers["product"], record.champion_id)
            self.assertEqual(record.arena_report, arena)
            self.assertLessEqual(store.committed_bytes, DEFAULT_ARTIFACT_LIMIT_BYTES)

            loaded = store.load_state_dict(role="product")
            for key, value in source_model.state_dict().items():
                self.assertTrue(torch.equal(loaded[key], value))
            target = TinyModel(-10.0)
            self.assertEqual(store.load_into(target), record)
            for key, value in source_model.state_dict().items():
                self.assertTrue(torch.equal(target.state_dict()[key], value))

            model_path = champions / record.file_name
            payload = torch.load(model_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(payload),
                {"schemaVersion", "format", "generation", "modelState"},
            )
            self.assertEqual(payload["schemaVersion"], CHAMPION_MODEL_SCHEMA_VERSION)
            self.assertNotIn("optimizer", payload)
            self.assertNotIn("replay", payload)
            manifest = json.loads((champions / "manifest.json").read_text())
            self.assertEqual(manifest["schemaVersion"], CHAMPION_STORE_SCHEMA_VERSION)
            self.assertEqual(manifest["champions"][0]["parentId"], None)
            self.assertEqual(
                manifest["champions"][0]["openingSuiteSha256"],
                _digest("openings"),
            )
            store.verify(deep=True)
            self.assertFalse(tuple(champions.glob(".*.tmp")))

    def test_store_reopens_and_is_independent_from_checkpoint_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champions = root / "champions"
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "checkpoint-00000001.pt").write_bytes(b"training-only")
            first = ChampionStore(champions)
            record = _publish(first, root, 0, None, best=True)
            (checkpoints / "checkpoint-00000001.pt").unlink()

            resumed = ChampionStore(champions)
            self.assertEqual(resumed.records, first.records)
            self.assertTrue((champions / record.file_name).is_file())
            resumed.verify(deep=True)

    def test_stale_store_publish_reloads_and_rejects_lost_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champions = root / "champions"
            first_writer = ChampionStore(champions)
            stale_writer = ChampionStore(champions)

            first = _publish(first_writer, root, 0, None)
            with self.assertRaisesRegex(ValueError, "parent_id"):
                _publish(stale_writer, root, 1, None)

            reopened = ChampionStore(champions)
            self.assertEqual(
                [record.champion_id for record in reopened.records],
                [first.champion_id],
            )
            self.assertEqual(reopened.pointers["latest"], first.champion_id)
            self.assertEqual(stale_writer.records, reopened.records)
            self.assertEqual(len(tuple(champions.glob("champion-*.pt"))), 1)
            reopened.verify(deep=True)

    def test_lineage_and_reference_hashes_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions")
            first = _publish(store, root, 2, None)
            with self.assertRaisesRegex(ValueError, "parent_id"):
                _publish(store, root, 3, None)
            with self.assertRaisesRegex(ValueError, "advance"):
                _publish(store, root, 2, first.champion_id)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                store.publish(
                    TinyModel(),
                    generation=3,
                    parent_id=first.champion_id,
                    config_sha256="bad",
                    source_sha256=_digest("source"),
                    opening_suite_sha256=_digest("openings"),
                    arena_report=_arena_reference(root, 3),
                )


class ChampionRetentionTests(unittest.TestCase):
    def test_retains_recent_eight_latest_milestones_best_and_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions")
            parent = None
            published = []
            for generation in range(12):
                record = _publish(
                    store,
                    root,
                    generation,
                    parent,
                    milestone=2 <= generation <= 6,
                    best=generation == 1,
                    product=generation == 0,
                )
                published.append(record)
                parent = record.champion_id

            expected = {
                published[0].champion_id,
                published[1].champion_id,
                *[record.champion_id for record in published[3:]],
            }
            retained = {record.champion_id for record in store.retained_records}
            self.assertEqual(retained, expected)
            self.assertEqual(len(store.records), 12)
            pruned = store.records[2]
            self.assertFalse(pruned.retained)
            self.assertIsNone(pruned.file_name)
            self.assertEqual(pruned.byte_size, 0)
            self.assertEqual(store.records[3].parent_id, published[2].champion_id)
            with self.assertRaises(KeyError):
                store.load_state_dict(pruned.champion_id)
            store.verify(deep=True)

    def test_role_assignment_protects_retained_model_and_rejects_pruned_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(
                root / "champions",
                recent_to_keep=2,
                milestones_to_keep=1,
            )
            first = _publish(store, root, 0, None)
            second = _publish(store, root, 1, first.champion_id)
            store.set_role(first.champion_id, "best")
            third = _publish(store, root, 2, second.champion_id)
            fourth = _publish(store, root, 3, third.champion_id)
            self.assertTrue(store.resolve(first.champion_id).retained)
            self.assertEqual(store.resolve(role="best").champion_id, first.champion_id)
            self.assertFalse(store.resolve(second.champion_id).retained)
            with self.assertRaisesRegex(ValueError, "pruned"):
                store.set_role(second.champion_id, "product")
            marked = store.set_role(third.champion_id, "milestone")
            self.assertEqual(marked.champion_id, third.champion_id)
            self.assertEqual(store.resolve(role="latest").champion_id, fourth.champion_id)

    def test_archive_budget_is_strict_and_never_exceeds_eight_gib(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 GiB"):
            ChampionStore(
                "/tmp/not-created-by-invalid-budget",
                budget_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES + 1,
            )

        with tempfile.TemporaryDirectory() as measurement_directory:
            root = Path(measurement_directory)
            measurement = ChampionStore(root / "champions")
            _publish(measurement, root, 0, None)
            required = measurement.committed_bytes
        with tempfile.TemporaryDirectory() as limited_directory:
            root = Path(limited_directory)
            limited = ChampionStore(
                root / "champions",
                budget_bytes=required - 200,
            )
            with self.assertRaises(ChampionBudgetError):
                _publish(limited, root, 0, None)
            self.assertEqual(len(limited), 0)
            self.assertFalse(tuple((root / "champions").glob("champion-*.pt")))


class ChampionGlobalResourceTests(unittest.TestCase):
    def test_publish_enforces_global_artifact_and_free_disk_stop_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions", artifact_root=root)
            with patch(
                "quoridor.champion_store.directory_size",
                return_value=DEFAULT_ARTIFACT_LIMIT_BYTES,
            ):
                with self.assertRaisesRegex(
                    ChampionBudgetError,
                    "global Phase 4 artifact budget",
                ):
                    _publish(store, root, 0, None)
            self.assertEqual(len(store), 0)
            self.assertFalse(tuple((root / "champions").glob("champion-*.pt")))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions", artifact_root=root)
            with patch(
                "quoridor.champion_store.directory_size",
                side_effect=(0, 0, DEFAULT_ARTIFACT_LIMIT_BYTES + 1),
            ):
                with self.assertRaisesRegex(
                    ChampionBudgetError,
                    "global Phase 4 artifact budget",
                ):
                    _publish(store, root, 0, None)
            reopened = ChampionStore(root / "champions", artifact_root=root)
            self.assertEqual(len(reopened), 0)
            self.assertFalse(tuple((root / "champions").glob("champion-*.pt")))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions", artifact_root=root)
            with patch(
                "quoridor.champion_store.shutil.disk_usage",
                return_value=SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES
                ),
            ):
                with self.assertRaisesRegex(
                    ChampionBudgetError,
                    "free-disk stop line",
                ):
                    _publish(store, root, 0, None)
            self.assertEqual(len(store), 0)
            self.assertFalse(tuple((root / "champions").glob("champion-*.pt")))

    def test_champion_root_cannot_escape_or_traverse_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_root.mkdir()
            with self.assertRaisesRegex(ValueError, "inside"):
                ChampionStore(
                    root / "outside" / "champions",
                    artifact_root=artifact_root,
                )

            target = artifact_root / "real-champions"
            target.mkdir()
            linked = artifact_root / "champions"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                ChampionStore(linked, artifact_root=artifact_root)


class ChampionCorruptionTests(unittest.TestCase):
    def test_missing_or_tampered_arena_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champions = root / "champions"
            store = ChampionStore(champions)
            _publish(store, root, 0, None)
            report = root / "arena-0000.json"
            original = report.read_bytes()
            report.unlink()
            with self.assertRaisesRegex(ChampionCorruptionError, "arena report"):
                store.verify(deep=True)
            with self.assertRaisesRegex(ChampionCorruptionError, "arena report"):
                ChampionStore(champions)

            report.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ChampionCorruptionError, "checksum"):
                ChampionStore(champions)

    def test_nonfinite_model_is_rejected_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions")
            model = TinyModel()
            with torch.no_grad():
                model.linear.weight[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                store.publish(
                    model,
                    generation=0,
                    parent_id=None,
                    config_sha256=_digest("config"),
                    source_sha256=_digest("source"),
                    opening_suite_sha256=_digest("openings"),
                    arena_report=_arena_reference(root, 0),
                )
            self.assertEqual(len(store), 0)

    def test_model_checksum_corruption_fails_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions")
            record = _publish(store, root, 0, None)
            path = root / "champions" / record.file_name
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 0xFF
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ChampionCorruptionError, "checksum"):
                store.load_state_dict(record.champion_id)
            with self.assertRaises(ChampionCorruptionError):
                store.verify(deep=True)

    def test_manifest_mutation_missing_model_and_config_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champions = root / "champions"
            store = ChampionStore(champions)
            record = _publish(store, root, 0, None)
            manifest_path = champions / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            manifest = json.loads(manifest_path.read_text())
            manifest["totals"]["accepted"] = 99
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ChampionCorruptionError, "hash"):
                ChampionStore(champions)

            manifest_path.write_bytes(original_manifest)
            (champions / record.file_name).unlink()
            with self.assertRaisesRegex(ChampionCorruptionError, "missing"):
                ChampionStore(champions)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ChampionStore(root / "champions", recent_to_keep=2)
            _publish(store, root, 0, None)
            with self.assertRaises(ChampionConfigurationError):
                ChampionStore(root / "champions", recent_to_keep=3)


if __name__ == "__main__":
    unittest.main()
