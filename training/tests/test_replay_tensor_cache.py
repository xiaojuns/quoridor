from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from quoridor.codec import ACTION_SPACE_SIZE, encode_move, position_hash
from quoridor.game import State
from quoridor.network import PHASE4_INPUT_CHANNELS
from quoridor.phase4_training import ReplayBatchEncoder
from quoridor.replay_store import ReplayGame, ReplayPly, ReplayStore
from quoridor.replay_tensor_cache import (
    REPLAY_MIRROR_RNG_ALGORITHM,
    REPLAY_SAMPLING_ALGORITHM,
    REPLAY_TENSOR_CACHE_ALGORITHM_VERSION,
    REPLAY_TENSOR_CACHE_BYTES_PER_POSITION,
    REPLAY_TENSOR_CACHE_MAX_BYTES,
    ReplayTensorCache,
    ReplayTensorCacheCapacityError,
    ReplayTensorCacheCorruptionError,
    ReplayTensorCacheStaleError,
)


SOURCE_SHA256 = "1" * 64
ENCODER_SHA256 = "2" * 64


def _game(game_id: str, *, column: int, generation: int = 0) -> ReplayGame:
    state = State(
        pawns=((1, column), (0, 8)),
        walls_remaining=(0, 0),
        current_player=0,
        move_number=0,
        position_counts={},
    )
    state = replace(state, position_counts={position_hash(state): 1})
    action = encode_move(0, column)
    counts = [0] * ACTION_SPACE_SIZE
    counts[action] = 17 + column
    return ReplayGame(
        game_id=game_id,
        initial_state=state,
        plies=(
            ReplayPly(
                action_index=action,
                visit_counts=tuple(counts),
                root_value=column / 10,
                model_generation=generation,
                model_id=f"model-{generation}",
                metadata={"column": column},
            ),
        ),
        metadata={"game": game_id, "generation": generation},
    )


def _store(
    root: Path,
    *,
    games: int,
    capacity: int = 8,
) -> ReplayStore:
    store = ReplayStore(
        root,
        max_games=capacity,
        max_plies=capacity,
        max_bytes=4_000_000,
    )
    for index in range(games):
        store.append(
            _game(
                f"game-{index}",
                column=index,
                generation=index,
            )
        )
    return store


def _cache(root: Path, store: ReplayStore, **kwargs) -> ReplayTensorCache:
    return ReplayTensorCache(
        root,
        store,
        source_sha256=SOURCE_SHA256,
        encoder_sha256=ENCODER_SHA256,
        **kwargs,
    )


class ReplayTensorCacheParityTests(unittest.TestCase):
    def test_layout_fits_both_100k_and_300k_active_windows(self) -> None:
        self.assertEqual(REPLAY_TENSOR_CACHE_BYTES_PER_POSITION, 12_044)
        self.assertLess(
            100_000 * REPLAY_TENSOR_CACHE_BYTES_PER_POSITION,
            REPLAY_TENSOR_CACHE_MAX_BYTES,
        )
        self.assertLess(
            300_000 * REPLAY_TENSOR_CACHE_BYTES_PER_POSITION,
            REPLAY_TENSOR_CACHE_MAX_BYTES,
        )

    def test_batches_and_shared_rng_are_bit_exact_to_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=4)
            cache = _cache(root / "cache", store, chunk_positions=2)
            refresh = cache.refresh()
            self.assertEqual(refresh.encoded_games, 4)
            self.assertEqual(refresh.encoded_plies, 4)
            self.assertEqual(refresh.chunks, 2)
            encoder = ReplayBatchEncoder(
                input_channels=PHASE4_INPUT_CHANNELS,
                device="cpu",
            )

            cases = (
                (3, 1, 0.0),
                (17, 3, 0.5),
                (91, 4, 1.0),
                (20260715, 20, 0.5),
            )
            for seed, count, probability in cases:
                with self.subTest(
                    seed=seed,
                    count=count,
                    probability=probability,
                ):
                    legacy_rng = random.Random(seed)
                    tensor_rng = random.Random(seed)
                    legacy_samples = store.sample_plies(count, rng=legacy_rng)
                    legacy_batch = encoder.encode(
                        legacy_samples,
                        rng=legacy_rng,
                        mirror_probability=probability,
                    )
                    tensor_batch = cache.load_batch(
                        count,
                        rng=tensor_rng,
                        mirror_probability=probability,
                    )
                    for label in ("states", "policies", "values"):
                        self.assertTrue(
                            torch.equal(tensor_batch[label], legacy_batch[label]),
                            f"{label} differs for seed {seed}",
                        )
                    self.assertEqual(tensor_rng.getstate(), legacy_rng.getstate())

    def test_plan_preserves_random_sample_order_and_always_draws_mirror_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=4)
            cache = _cache(root / "cache", store)
            cache.refresh()

            for probability in (0.0, 0.5, 1.0):
                with self.subTest(probability=probability):
                    expected_rng = random.Random(41)
                    expected_indices = tuple(expected_rng.sample(range(4), 4))
                    expected_mirrors = tuple(
                        expected_rng.random() < probability for _ in expected_indices
                    )
                    actual_rng = random.Random(41)
                    plan = cache.sample_plan(
                        10,
                        rng=actual_rng,
                        mirror_probability=probability,
                    )
                    self.assertEqual(plan.global_indices, expected_indices)
                    self.assertEqual(plan.mirrored, expected_mirrors)
                    self.assertEqual(actual_rng.getstate(), expected_rng.getstate())
                    self.assertEqual(
                        plan.sampling_algorithm,
                        REPLAY_SAMPLING_ALGORITHM,
                    )
                    self.assertEqual(
                        plan.mirror_rng_algorithm,
                        REPLAY_MIRROR_RNG_ALGORITHM,
                    )

    def test_load_uses_read_only_memory_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=2)
            cache = _cache(root / "cache", store)
            cache.refresh()

            with patch(
                "quoridor.replay_tensor_cache.np.load",
                wraps=np.load,
            ) as load:
                cache.load_batch(2, rng=random.Random(7))
            self.assertEqual(load.call_count, 3)
            for call in load.call_args_list:
                self.assertEqual(call.kwargs["mmap_mode"], "r")
                self.assertIs(call.kwargs["allow_pickle"], False)


class ReplayTensorCacheRefreshTests(unittest.TestCase):
    def test_refresh_deeply_replays_each_new_shard_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=3)
            cache = _cache(root / "cache", store)
            from quoridor import replay_store as replay_store_module

            with patch.object(
                replay_store_module,
                "reconstruct_game",
                wraps=replay_store_module.reconstruct_game,
            ) as reconstructed:
                cache.refresh()
            self.assertEqual(reconstructed.call_count, 3)

    def test_refresh_reuses_append_and_fifo_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=2, capacity=2)
            cache = _cache(root / "cache", store, chunk_positions=1)
            first = cache.refresh()
            self.assertEqual(first.encoded_games, 2)

            store.append(_game("game-2", column=2, generation=2))
            self.assertEqual(store.game_ids, ("game-1", "game-2"))
            second = cache.refresh()
            self.assertEqual(second.reused_games, 1)
            self.assertEqual(second.encoded_games, 1)
            self.assertEqual(second.games, 2)
            self.assertEqual(second.plies, 2)

            legacy_rng = random.Random(81)
            tensor_rng = random.Random(81)
            encoder = ReplayBatchEncoder(
                input_channels=PHASE4_INPUT_CHANNELS,
                device="cpu",
            )
            legacy = encoder.encode(
                store.sample_plies(10, rng=legacy_rng),
                rng=legacy_rng,
            )
            actual = cache.load_batch(10, rng=tensor_rng)
            for label in legacy:
                self.assertTrue(torch.equal(actual[label], legacy[label]))
            self.assertEqual(tensor_rng.getstate(), legacy_rng.getstate())

            reopened_store = ReplayStore(
                root / "raw",
                max_games=2,
                max_plies=2,
                max_bytes=4_000_000,
            )
            reopened = _cache(
                root / "cache",
                reopened_store,
                chunk_positions=1,
            )
            no_op = reopened.refresh()
            self.assertEqual(no_op.reused_games, 2)
            self.assertEqual(no_op.encoded_games, 0)

    def test_fifo_keeps_a_partially_live_chunk_then_reclaims_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=2, capacity=2)
            cache = _cache(root / "cache", store, chunk_positions=8)
            cache.refresh()
            initial_manifest = json.loads(cache.manifest_path.read_text())
            initial_directory = initial_manifest["chunks"][0]["directory"]

            store.append(_game("game-2", column=2, generation=2))
            second = cache.refresh()
            self.assertEqual(second.reused_games, 1)
            self.assertEqual(second.chunks, 2)
            self.assertTrue((cache.chunks_root / initial_directory).is_dir())

            store.append(_game("game-3", column=3, generation=3))
            third = cache.refresh()
            self.assertEqual(third.reused_games, 1)
            self.assertEqual(third.chunks, 2)
            self.assertFalse((cache.chunks_root / initial_directory).exists())
            self.assertEqual(
                {path.name for path in cache.chunks_root.iterdir()},
                {
                    chunk["directory"]
                    for chunk in json.loads(cache.manifest_path.read_text())["chunks"]
                },
            )

    def test_sample_plan_fails_after_raw_window_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1, capacity=2)
            cache = _cache(root / "cache", store)
            cache.refresh()
            plan = cache.sample_plan(1, rng=random.Random(1))

            store.append(_game("game-1", column=1, generation=1))
            with self.assertRaisesRegex(ReplayTensorCacheStaleError, "raw replay"):
                cache.load_plan(plan)

    def test_snapshot_and_verification_bind_raw_and_cache_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=3)
            cache = _cache(root / "cache", store, chunk_positions=2)
            refresh = cache.refresh()
            verification = cache.verify(deep=True)
            snapshot = cache.snapshot()

            self.assertTrue(verification.deep)
            self.assertTrue(verification.memory_mapped)
            self.assertEqual(verification.games, 3)
            self.assertEqual(verification.plies, 3)
            self.assertEqual(
                snapshot["algorithmVersion"],
                REPLAY_TENSOR_CACHE_ALGORITHM_VERSION,
            )
            self.assertEqual(
                snapshot["manifestSha256"],
                refresh.manifest_sha256,
            )
            self.assertEqual(
                snapshot["manifestFileSha256"],
                refresh.manifest_file_sha256,
            )
            self.assertEqual(
                snapshot["rawManifestSha256"],
                refresh.raw_manifest_sha256,
            )


class ReplayTensorCacheFailureTests(unittest.TestCase):
    def test_capacity_fails_before_publishing_a_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1)
            cache = _cache(
                root / "cache",
                store,
                maximum_cache_bytes=REPLAY_TENSOR_CACHE_BYTES_PER_POSITION - 1,
            )
            with self.assertRaisesRegex(
                ReplayTensorCacheCapacityError,
                "4 GiB",
            ):
                cache.refresh()
            self.assertFalse(cache.manifest_path.exists())
            self.assertFalse(tuple(cache.chunks_root.iterdir()))

    def test_chunk_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=2)
            cache = _cache(root / "cache", store)
            cache.refresh()
            manifest = json.loads(cache.manifest_path.read_text())
            chunk = manifest["chunks"][0]
            state_path = (
                cache.chunks_root
                / chunk["directory"]
                / chunk["files"]["states"]["file"]
            )
            damaged = bytearray(state_path.read_bytes())
            damaged[-1] ^= 1
            state_path.write_bytes(damaged)

            with self.assertRaisesRegex(
                ReplayTensorCacheCorruptionError,
                "checksum",
            ):
                cache.load_batch(1, rng=random.Random(3))
            self.assertFalse(cache.is_built)

    def test_raw_shard_tampering_fails_closed_before_tensor_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1)
            cache = _cache(root / "cache", store)
            cache.refresh()
            shard_path = store.root / store.shards[0].file_name
            damaged = bytearray(shard_path.read_bytes())
            damaged[-1] ^= 1
            shard_path.write_bytes(damaged)

            with self.assertRaisesRegex(
                ReplayTensorCacheCorruptionError,
                "raw replay shard changed",
            ):
                cache.load_batch(1, rng=random.Random(3))
            self.assertFalse(cache.is_built)

    def test_manifest_tampering_is_rejected_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1)
            cache = _cache(root / "cache", store)
            cache.refresh()
            cache.manifest_path.write_bytes(
                cache.manifest_path.read_bytes() + b" "
            )

            with self.assertRaisesRegex(
                ReplayTensorCacheCorruptionError,
                "canonical",
            ):
                _cache(root / "cache", store)

    def test_manifest_tampering_cannot_be_overwritten_by_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1, capacity=2)
            cache = _cache(root / "cache", store)
            cache.refresh()
            cache.manifest_path.write_bytes(
                cache.manifest_path.read_bytes() + b" "
            )
            store.append(_game("game-1", column=1, generation=1))

            with self.assertRaisesRegex(
                ReplayTensorCacheStaleError,
                "manifest changed",
            ):
                cache.refresh()

    def test_interrupted_manifest_publish_leaves_only_recoverable_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=2)
            cache = _cache(root / "cache", store)

            with patch(
                "quoridor.replay_tensor_cache.atomic_replace_bytes",
                side_effect=RuntimeError("injected publish failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    cache.refresh()
            self.assertFalse(cache.manifest_path.exists())
            self.assertTrue(tuple(cache.chunks_root.iterdir()))

            recovered = _cache(root / "cache", store)
            self.assertFalse(tuple(recovered.chunks_root.iterdir()))
            refresh = recovered.refresh()
            self.assertEqual(refresh.encoded_games, 2)
            self.assertTrue(recovered.verify(deep=True).deep)

    def test_raw_manifest_change_during_build_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "raw", games=1, capacity=3)
            cache = _cache(root / "cache", store)
            changed = False
            real_encode = cache._encode_game

            def encode_and_append(game, states):
                nonlocal changed
                if not changed:
                    changed = True
                    store.append(_game("game-1", column=1, generation=1))
                return real_encode(game, states)

            with patch.object(cache, "_encode_game", side_effect=encode_and_append):
                with self.assertRaisesRegex(
                    ReplayTensorCacheStaleError,
                    "changed during",
                ):
                    cache.refresh()
            self.assertFalse(cache.manifest_path.exists())
            self.assertFalse(cache.is_built)


if __name__ == "__main__":
    unittest.main()
