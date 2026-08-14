from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from quoridor.phase4_checkpoint import (
    CheckpointIdentity,
    Phase4CheckpointCorruptionError,
    Phase4CheckpointStore,
)
from quoridor.trainer import AlphaZeroTrainer


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(inputs)


def identity(marker: str = "a") -> CheckpointIdentity:
    return CheckpointIdentity(
        run_manifest_sha256=marker * 64,
        config_sha256="b" * 64,
        source_sha256="c" * 64,
    )


class Phase4CheckpointTests(unittest.TestCase):
    def _store(self, root: Path, marker: str = "a") -> Phase4CheckpointStore:
        return Phase4CheckpointStore(
            root / "checkpoints",
            identity=identity(marker),
            artifact_root=root,
            artifact_limit_bytes=4_000_000,
            max_to_keep=2,
        )

    @staticmethod
    def _save(
        store: Phase4CheckpointStore,
        model: nn.Module,
        trainer: AlphaZeroTrainer,
        generation: int,
    ):
        return store.save(
            generation=generation,
            model=model,
            trainer=trainer,
            generation_data={"generation": generation},
            replay_snapshot={
                "manifestSha256": "e" * 64,
                "games": generation + 1,
                "plies": generation + 2,
                "bytes": 10,
                "manifestBytes": 100,
                "committedBytes": 110,
                "deepVerified": True,
                "gameIds": [f"game-{index}" for index in range(generation + 1)],
            },
        )

    def test_corrupt_newest_checkpoint_falls_back_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            first = self._save(store, model, trainer, 0)
            trainer.step = 1
            second = self._save(store, model, trainer, 1)
            second.data_path.write_bytes(b"not a torch checkpoint")

            original_load = torch.load
            loaded_paths: list[Path] = []

            def recording_load(path, *args, **kwargs):
                loaded_paths.append(Path(path))
                return original_load(path, *args, **kwargs)

            restored_model = TinyNetwork()
            restored_trainer = AlphaZeroTrainer(restored_model, use_amp=False)
            with mock.patch("quoridor.phase4_checkpoint.torch.load", recording_load):
                loaded = store.restore(
                    restored_model,
                    restored_trainer,
                    map_location="cpu",
                    restore_rng=False,
                )

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.record, first)
            self.assertEqual(loaded_paths, [first.data_path])
            self.assertEqual(restored_trainer.step, 0)
            self.assertEqual(len(loaded.rejected_newer), 1)
            self.assertRegex(loaded.rejected_newer[0], r"(byte size|SHA-256) mismatch")
            quarantined = store.quarantine_rejected(loaded.rejected_newer)
            self.assertEqual(len(quarantined), 3)
            self.assertFalse(second.data_path.exists())
            self.assertFalse(second.metadata_path.exists())
            rerun = self._save(store, model, trainer, 1)
            self.assertTrue(rerun.data_path.exists())

    def test_candidate_model_is_safe_and_corruption_rejects_newest_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            first = self._save(store, model, trainer, 0)
            self.assertIsNotNone(first.candidate_model_path)
            candidate = torch.load(
                first.candidate_model_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                candidate["checkpointSha256"],
                first.sha256,
            )
            self.assertEqual(
                candidate["modelStateSha256"],
                first.model_state_sha256,
            )

            trainer.step = 1
            second = self._save(store, model, trainer, 1)
            self.assertIsNotNone(second.candidate_model_path)
            second.candidate_model_path.write_bytes(b"damaged")
            loaded = store.latest_valid()

            self.assertEqual(loaded.record, first)
            self.assertEqual(len(loaded.rejected_newer), 1)
            self.assertRegex(
                loaded.rejected_newer[0],
                r"(byte size|SHA-256) mismatch",
            )

    def test_semantically_invalid_newest_payload_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            first = self._save(store, model, trainer, 0)
            trainer.step = 1
            second = self._save(store, model, trainer, 1)
            payload = torch.load(second.data_path, weights_only=False)
            payload["model"] = {"wrong": torch.zeros(1)}
            torch.save(payload, second.data_path)
            metadata = json.loads(second.metadata_path.read_text())
            from quoridor.run_manifest import sha256_file

            metadata["bytes"] = second.data_path.stat().st_size
            metadata["sha256"] = sha256_file(second.data_path)
            second.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            restored_model = TinyNetwork()
            restored_trainer = AlphaZeroTrainer(restored_model, use_amp=False)
            loaded = store.restore(
                restored_model,
                restored_trainer,
                map_location="cpu",
                restore_rng=False,
            )
            self.assertEqual(loaded.record, first)
            self.assertIn("state_dict", loaded.rejected_newer[0])

    def test_invalid_rng_in_newest_checkpoint_falls_back_and_restores_older_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            first = self._save(store, model, trainer, 0)
            first_payload = torch.load(first.data_path, weights_only=False)
            trainer.step = 1
            second = self._save(store, model, trainer, 1)
            payload = torch.load(second.data_path, weights_only=False)
            payload["rng"]["python"] = ("invalid",)
            torch.save(payload, second.data_path)
            metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
            from quoridor.run_manifest import sha256_file

            metadata["bytes"] = second.data_path.stat().st_size
            metadata["sha256"] = sha256_file(second.data_path)
            second.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            restored_model = TinyNetwork()
            restored_trainer = AlphaZeroTrainer(restored_model, use_amp=False)
            loaded = store.restore(
                restored_model,
                restored_trainer,
                map_location="cpu",
                restore_rng=True,
            )

            self.assertEqual(loaded.record, first)
            self.assertIn("state", loaded.rejected_newer[0])
            self.assertEqual(random.getstate(), first_payload["rng"]["python"])
            self.assertTrue(
                torch.equal(torch.get_rng_state(), first_payload["rng"]["torch"])
            )

    def test_orphan_data_is_removed_and_wrong_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            orphan = checkpoint_root / "checkpoint-g00000000-s000000000000.pt"
            orphan.write_bytes(b"orphan")
            self._store(root)
            self.assertFalse(orphan.exists())

            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            self._save(self._store(root), model, trainer, 0)
            foreign = self._store(root, marker="d")
            with self.assertRaisesRegex(
                Phase4CheckpointCorruptionError, "no valid"
            ):
                foreign.latest_valid()

    def test_initialization_checkpoint_is_explicit_and_sorts_before_generation_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            initialized = store.save(
                generation=-1,
                model=model,
                trainer=trainer,
                generation_data={
                    "schemaVersion": 1,
                    "kind": "initialization",
                    "nextGeneration": 0,
                    "mode": "random",
                    "seed": 7,
                    "baseModelSha256": "d" * 64,
                    "initializedModelSha256": "d" * 64,
                    "optimizerReset": True,
                    "artifact": None,
                },
                replay_snapshot={
                    "manifestSha256": "e" * 64,
                    "games": 0,
                    "plies": 0,
                    "bytes": 0,
                    "manifestBytes": 100,
                    "committedBytes": 100,
                    "deepVerified": True,
                    "gameIds": [],
                },
            )
            self.assertEqual(store.latest_valid().record, initialized)
            generation_zero = self._save(store, model, trainer, 0)
            self.assertEqual(store.latest_valid().record, generation_zero)

            bad_store = self._store(root / "bad")
            with self.assertRaisesRegex(
                Phase4CheckpointCorruptionError, "initialization"
            ):
                bad_store.save(
                    generation=-1,
                    model=model,
                    trainer=trainer,
                    generation_data={"generation": -1},
                    replay_snapshot={
                        "manifestSha256": "e" * 64,
                        "games": 0,
                        "plies": 0,
                        "bytes": 0,
                        "manifestBytes": 100,
                        "committedBytes": 100,
                        "deepVerified": True,
                        "gameIds": [],
                    },
                )

    def test_champion_initialization_checkpoint_binds_full_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            champion = {
                "kind": "published-champion",
                "championId": "champion-14",
                "generation": 14,
                "artifactPath": "champions/champion-14.pt",
                "artifactSha256": "1" * 64,
                "modelStateSha256": "2" * 64,
                "championManifestPath": "champions/manifest.json",
                "championManifestSha256": "3" * 64,
                "championManifestFileSha256": "4" * 64,
                "sourceSha256": "5" * 64,
                "configSha256": "6" * 64,
                "openingSuiteSha256": "7" * 64,
                "formalPromotion": True,
            }
            generation_data = {
                "schemaVersion": 1,
                "kind": "initialization",
                "nextGeneration": 0,
                "mode": "champion",
                "seed": 20260803,
                "baseModelSha256": "8" * 64,
                "initializedModelSha256": "2" * 64,
                "optimizerReset": True,
                "artifact": champion,
            }
            replay_snapshot = {
                "manifestSha256": "9" * 64,
                "games": 0,
                "plies": 0,
                "bytes": 0,
                "manifestBytes": 100,
                "committedBytes": 100,
                "deepVerified": True,
                "gameIds": [],
            }
            initialized = store.save(
                generation=-1,
                model=model,
                trainer=trainer,
                generation_data=generation_data,
                replay_snapshot=replay_snapshot,
            )
            self.assertEqual(store.latest_valid().record, initialized)

            unsafe_store = self._store(root / "unsafe")
            unsafe = dict(champion)
            unsafe["championManifestPath"] = "../manifest.json"
            with self.assertRaisesRegex(
                Phase4CheckpointCorruptionError,
                "path is unsafe",
            ):
                unsafe_store.save(
                    generation=-1,
                    model=model,
                    trainer=trainer,
                    generation_data={**generation_data, "artifact": unsafe},
                    replay_snapshot=replay_snapshot,
                )

    def test_recovery_initialization_checkpoint_binds_archive_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyNetwork()
            trainer = AlphaZeroTrainer(model, use_amp=False)
            store = self._store(root)
            recovery = {
                "kind": "recovery-candidate",
                "generation": 11,
                "step": 1152,
                "artifactPath": "archive/candidate-model.pt",
                "artifactSha256": "1" * 64,
                "modelStateSha256": "2" * 64,
                "recoveryManifestPath": "archive-manifest.json",
                "recoveryManifestSha256": "3" * 64,
                "recoveryManifestFileSha256": "4" * 64,
                "sourceSha256": "5" * 64,
                "configSha256": "6" * 64,
                "runManifestSha256": "7" * 64,
                "checkpointSha256": "8" * 64,
            }
            generation_data = {
                "schemaVersion": 1,
                "kind": "initialization",
                "nextGeneration": 0,
                "mode": "recovery",
                "seed": 20260808,
                "baseModelSha256": "9" * 64,
                "initializedModelSha256": "2" * 64,
                "optimizerReset": True,
                "artifact": recovery,
            }
            replay_snapshot = {
                "manifestSha256": "a" * 64,
                "games": 0,
                "plies": 0,
                "bytes": 0,
                "manifestBytes": 100,
                "committedBytes": 100,
                "deepVerified": True,
                "gameIds": [],
            }
            initialized = store.save(
                generation=-1,
                model=model,
                trainer=trainer,
                generation_data=generation_data,
                replay_snapshot=replay_snapshot,
            )
            self.assertEqual(store.latest_valid().record, initialized)

            wrong_mode = dict(generation_data)
            wrong_mode["mode"] = "model"
            with self.assertRaisesRegex(
                Phase4CheckpointCorruptionError,
                "mode and artifact provenance disagree",
            ):
                self._store(root / "wrong-mode").save(
                    generation=-1,
                    model=model,
                    trainer=trainer,
                    generation_data=wrong_mode,
                    replay_snapshot=replay_snapshot,
                )


if __name__ == "__main__":
    unittest.main()
