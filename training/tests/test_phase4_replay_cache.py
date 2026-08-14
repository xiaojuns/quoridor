from __future__ import annotations

import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

import quoridor.phase4_training as phase4_training
from quoridor.codec import ACTION_SPACE_SIZE, encode_move, position_hash
from quoridor.game import State
from quoridor.network import PHASE4_INPUT_CHANNELS
from quoridor.phase4_training import (
    REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY,
    REPLAY_INPUT_GATE_BATCHES,
    REPLAY_INPUT_GATE_MAX_BATCH_SECONDS,
    REPLAY_INPUT_GATE_MAX_MEAN_SECONDS,
    REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS,
    Phase4ResourceLimitReached,
    Phase4ResumeError,
    Phase4TrainingError,
    ReplayBatchEncoder,
    ReplayCacheRefresh,
    ReplayInputPerformanceGate,
    ReplayReconstructionCache,
)
from quoridor.replay_store import (
    ReplayGame,
    ReplayPly,
    ReplayStore,
    reconstruct_game,
)


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
    game_count: int,
    max_games: int = 8,
) -> ReplayStore:
    store = ReplayStore(
        root,
        max_games=max_games,
        max_plies=max_games,
        max_bytes=1_000_000,
    )
    for index in range(game_count):
        store.append(
            _game(
                f"game-{index}",
                column=index,
                generation=index,
            )
        )
    return store


class ReplayCacheExactnessTests(unittest.TestCase):
    def test_samples_batches_and_rng_are_bit_exact_to_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=4)
            cache = ReplayReconstructionCache(store)
            refresh = cache.refresh()
            self.assertEqual(refresh.decoded_games, 4)
            self.assertEqual(refresh.decoded_plies, 4)
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
            for seed, count, mirror_probability in cases:
                with self.subTest(
                    seed=seed,
                    count=count,
                    mirror_probability=mirror_probability,
                ):
                    legacy_rng = random.Random(seed)
                    cached_rng = random.Random(seed)
                    legacy_samples = store.sample_plies(count, rng=legacy_rng)
                    cached_samples = cache.sample_plies(count, rng=cached_rng)
                    self.assertEqual(cached_samples, legacy_samples)
                    self.assertEqual(cached_rng.getstate(), legacy_rng.getstate())

                    legacy_batch = encoder.encode(
                        legacy_samples,
                        rng=legacy_rng,
                        mirror_probability=mirror_probability,
                    )
                    cached_batch = encoder.encode(
                        cached_samples,
                        rng=cached_rng,
                        mirror_probability=mirror_probability,
                    )
                    for label in ("states", "policies", "values"):
                        self.assertTrue(
                            torch.equal(cached_batch[label], legacy_batch[label]),
                            f"{label} differs for seed {seed}",
                        )
                    self.assertEqual(cached_rng.getstate(), legacy_rng.getstate())

    def test_refresh_reconstructs_each_game_once_and_only_decodes_appends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=3)
            cache = ReplayReconstructionCache(store)
            with patch.object(
                phase4_training,
                "reconstruct_game",
                wraps=reconstruct_game,
            ) as reconstructed:
                first = cache.refresh()
                self.assertEqual(reconstructed.call_count, 3)
                self.assertEqual(first.reused_games, 0)
                self.assertEqual(first.decoded_games, 3)

                for seed in range(10):
                    cache.sample_plies(3, rng=random.Random(seed))
                cache.assert_current(verify_shards=True)
                unchanged = cache.refresh()
                self.assertEqual(reconstructed.call_count, 3)
                self.assertEqual(unchanged.reused_games, 3)
                self.assertEqual(unchanged.decoded_games, 0)

                appended = _game("game-3", column=3, generation=3)
                store.append(appended)
                extended = cache.refresh()
                self.assertEqual(reconstructed.call_count, 4)
                self.assertEqual(extended.reused_games, 3)
                self.assertEqual(extended.decoded_games, 1)
                self.assertEqual(cache.game("game-3"), appended)
                self.assertEqual(cache.verification().game_count, 4)
                self.assertEqual(cache.verification().ply_count, 4)

    def test_reopened_store_rebuilds_the_same_resume_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_store = _store(root, game_count=4)
            first_cache = ReplayReconstructionCache(first_store)
            first_refresh = first_cache.refresh()

            reopened_store = ReplayStore(
                root,
                max_games=8,
                max_plies=8,
                max_bytes=1_000_000,
            )
            resumed_cache = ReplayReconstructionCache(reopened_store)
            resumed_refresh = resumed_cache.refresh()
            self.assertEqual(
                resumed_refresh.manifest_sha256,
                first_refresh.manifest_sha256,
            )
            self.assertEqual(resumed_refresh.prefix_sha256, first_refresh.prefix_sha256)

            encoder = ReplayBatchEncoder(
                input_channels=PHASE4_INPUT_CHANNELS,
                device="cpu",
            )
            for seed in (7, 101, 20260715):
                original_rng = random.Random(seed)
                resumed_rng = random.Random(seed)
                original = encoder.encode(
                    first_cache.sample_plies(3, rng=original_rng),
                    rng=original_rng,
                )
                resumed = encoder.encode(
                    resumed_cache.sample_plies(3, rng=resumed_rng),
                    rng=resumed_rng,
                )
                for label in original:
                    self.assertTrue(torch.equal(original[label], resumed[label]))
                self.assertEqual(original_rng.getstate(), resumed_rng.getstate())

    def test_returned_games_and_samples_cannot_mutate_cached_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=2)
            cache = ReplayReconstructionCache(store)
            cache.refresh()

            returned_game = cache.game("game-0")
            returned_game.metadata["poison"] = True
            returned_game.plies[0].metadata["poison"] = True
            clean_game = cache.game("game-0")
            self.assertNotIn("poison", clean_game.metadata)
            self.assertNotIn("poison", clean_game.plies[0].metadata)

            returned_sample = cache.sample_plies(
                1,
                rng=random.Random(7),
            )[0]
            returned_sample.state.position_counts.clear()
            returned_sample.game_metadata["poison"] = True
            returned_sample.ply_metadata["poison"] = True
            clean_sample = cache.sample_plies(
                1,
                rng=random.Random(7),
            )[0]
            self.assertTrue(clean_sample.state.position_counts)
            self.assertNotIn("poison", clean_sample.game_metadata)
            self.assertNotIn("poison", clean_sample.ply_metadata)
            cache.assert_current(verify_shards=True)


class ReplayCacheMutationTests(unittest.TestCase):
    def test_append_only_extension_succeeds_but_fifo_eviction_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=2, max_games=2)
            cache = ReplayReconstructionCache(store)
            cache.refresh()

            store.append(_game("game-2", column=2, generation=2))
            self.assertEqual(store.game_ids, ("game-1", "game-2"))
            with self.assertRaisesRegex(Phase4ResumeError, "append-only"):
                cache.refresh()
            self.assertFalse(cache.is_built)
            with self.assertRaisesRegex(Phase4ResumeError, "stale"):
                cache.sample_plies(1, rng=random.Random(1))

    def test_manifest_and_shard_tampering_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=2)
            cache = ReplayReconstructionCache(store)
            cache.refresh()

            store.manifest_path.write_bytes(store.manifest_path.read_bytes() + b" ")
            with self.assertRaisesRegex(Phase4ResumeError, "manifest changed"):
                cache.assert_current()
            self.assertFalse(cache.is_built)

        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=2)
            cache = ReplayReconstructionCache(store)
            cache.refresh()
            shard_path = store.root / store.shards[0].file_name
            damaged = bytearray(shard_path.read_bytes())
            damaged[-1] ^= 1
            shard_path.write_bytes(damaged)

            cache.assert_current()
            with self.assertRaisesRegex(Phase4ResumeError, "shard changed"):
                cache.assert_current(verify_shards=True)
            self.assertFalse(cache.is_built)

    def test_mutation_during_refresh_does_not_publish_a_partial_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=2)
            cache = ReplayReconstructionCache(store)
            changed = False

            def reconstruct_and_mutate(game: ReplayGame):
                nonlocal changed
                states = reconstruct_game(game)
                if not changed:
                    store.manifest_path.write_bytes(
                        store.manifest_path.read_bytes() + b" "
                    )
                    changed = True
                return states

            with patch.object(
                phase4_training,
                "reconstruct_game",
                side_effect=reconstruct_and_mutate,
            ):
                with self.assertRaisesRegex(
                    Phase4ResumeError,
                    "manifest changed during",
                ):
                    cache.refresh()
            self.assertFalse(cache.is_built)
            with self.assertRaisesRegex(
                Phase4ResumeError,
                "stale|has not been built",
            ):
                cache.snapshot()

    def test_estimated_memory_limit_fails_before_cache_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), game_count=1)
            cache = ReplayReconstructionCache(
                store,
                maximum_estimated_bytes=REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY - 1,
            )
            with self.assertRaisesRegex(Phase4ResourceLimitReached, "4 GiB estimate"):
                cache.refresh()
            self.assertFalse(cache.is_built)


class ReplayCacheBuildEnvelopeTests(unittest.TestCase):
    @staticmethod
    def _refresh(
        *,
        decoded_games: int,
        decoded_plies: int,
        refresh_seconds: float,
        games: int | None = None,
        plies: int | None = None,
        reused_games: int = 0,
    ) -> ReplayCacheRefresh:
        total_games = decoded_games if games is None else games
        total_plies = decoded_plies if plies is None else plies
        return ReplayCacheRefresh(
            algorithm_version=phase4_training.REPLAY_CACHE_ALGORITHM_VERSION,
            manifest_sha256="1" * 64,
            prefix_sha256="2" * 64,
            games=total_games,
            plies=total_plies,
            estimated_bytes=(
                total_plies * phase4_training.REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY
            ),
            reused_games=reused_games,
            decoded_games=decoded_games,
            decoded_plies=decoded_plies,
            refresh_seconds=refresh_seconds,
            cumulative_build_seconds=refresh_seconds,
        )

    def _assert_boundary(
        self,
        *,
        decoded_games: int,
        decoded_plies: int,
        expected_seconds: float,
        games: int | None = None,
        plies: int | None = None,
        reused_games: int = 0,
    ) -> None:
        at_limit = self._refresh(
            decoded_games=decoded_games,
            decoded_plies=decoded_plies,
            refresh_seconds=expected_seconds,
            games=games,
            plies=plies,
            reused_games=reused_games,
        )
        self.assertAlmostEqual(
            phase4_training._replay_cache_build_envelope_seconds(at_limit),
            expected_seconds,
        )
        self.assertAlmostEqual(
            phase4_training._validate_replay_cache_refresh(at_limit),
            expected_seconds,
        )

        over_limit = replace(
            at_limit,
            refresh_seconds=expected_seconds + 1e-6,
            cumulative_build_seconds=expected_seconds + 1e-6,
        )
        with self.assertRaises(Phase4ResourceLimitReached):
            phase4_training._validate_replay_cache_refresh(over_limit)

    def test_empty_refresh_has_no_build_envelope(self) -> None:
        refresh = self._refresh(
            decoded_games=0,
            decoded_plies=0,
            refresh_seconds=10_000.0,
            games=256,
            plies=20_000,
            reused_games=256,
        )

        self.assertIsNone(
            phase4_training._replay_cache_build_envelope_seconds(refresh)
        )
        self.assertIsNone(phase4_training._validate_replay_cache_refresh(refresh))

    def test_small_refresh_scales_from_the_game_axis(self) -> None:
        expected_seconds = (
            phase4_training.REPLAY_INPUT_GATE_MAX_BUILD_SECONDS
            / phase4_training.REPLAY_INPUT_GATE_FIXTURE_GAMES
        )
        self._assert_boundary(
            decoded_games=1,
            decoded_plies=1,
            expected_seconds=expected_seconds,
        )

    def test_warm_and_cold_shapes_select_the_slower_axis(self) -> None:
        cases = (
            ("warm-game-axis", 64, 2_725, 120.0),
            (
                "cold-ply-axis",
                64,
                7_713,
                (
                    phase4_training.REPLAY_INPUT_GATE_MAX_BUILD_SECONDS
                    * 7_713
                    / phase4_training.REPLAY_INPUT_GATE_FIXTURE_PLIES
                ),
            ),
        )
        for label, decoded_games, decoded_plies, expected_seconds in cases:
            with self.subTest(workload=label):
                self._assert_boundary(
                    decoded_games=decoded_games,
                    decoded_plies=decoded_plies,
                    expected_seconds=expected_seconds,
                )

    def test_incremental_refresh_ignores_reused_replay_work(self) -> None:
        self._assert_boundary(
            decoded_games=64,
            decoded_plies=2_725,
            expected_seconds=120.0,
            games=320,
            plies=30_000,
            reused_games=256,
        )

    def test_resume_rebuild_scales_on_either_dominant_axis(self) -> None:
        cases = (
            ("game-axis", 320, 20_000, 600.0),
            ("ply-axis", 64, 35_905, 600.0),
        )
        for label, decoded_games, decoded_plies, expected_seconds in cases:
            with self.subTest(workload=label):
                self._assert_boundary(
                    decoded_games=decoded_games,
                    decoded_plies=decoded_plies,
                    expected_seconds=expected_seconds,
                )

    def test_formal_explicit_cap_preserves_seconds_and_throughput_gates(
        self,
    ) -> None:
        cold_shape = self._refresh(
            decoded_games=64,
            decoded_plies=7_713,
            refresh_seconds=120.01,
        )
        self.assertGreater(
            phase4_training._replay_cache_build_envelope_seconds(cold_shape),
            cold_shape.refresh_seconds,
        )
        with self.assertRaisesRegex(
            Phase4ResourceLimitReached,
            "exceeded 120.0s",
        ):
            phase4_training._validate_replay_cache_refresh(
                cold_shape,
                maximum_build_seconds=120.0,
            )

        fixture_shape = self._refresh(
            decoded_games=64,
            decoded_plies=phase4_training.REPLAY_INPUT_GATE_FIXTURE_PLIES,
            refresh_seconds=119.9,
        )
        with self.assertRaisesRegex(
            Phase4ResourceLimitReached,
            "throughput fell below",
        ):
            phase4_training._validate_replay_cache_refresh(
                fixture_shape,
                maximum_build_seconds=120.0,
            )

        accepted = replace(
            fixture_shape,
            refresh_seconds=100.0,
            cumulative_build_seconds=100.0,
        )
        self.assertEqual(
            phase4_training._validate_replay_cache_refresh(
                accepted,
                maximum_build_seconds=120.0,
            ),
            120.0,
        )

    def test_inconsistent_decoded_counts_fail_closed(self) -> None:
        valid = self._refresh(
            decoded_games=1,
            decoded_plies=1,
            refresh_seconds=0.1,
        )
        cases = (
            ("negative count", replace(valid, decoded_games=-1), "non-negative"),
            (
                "decoded games exceed total",
                replace(valid, games=1, reused_games=1),
                "inconsistent",
            ),
            (
                "decoded plies exceed total",
                replace(valid, plies=0),
                "inconsistent",
            ),
            (
                "games without plies",
                replace(valid, plies=0, decoded_plies=0),
                "both be non-zero",
            ),
            (
                "plies without games",
                replace(valid, games=0, decoded_games=0),
                "both be non-zero",
            ),
        )
        for label, refresh, expected_error in cases:
            with self.subTest(workload=label):
                with self.assertRaisesRegex(
                    Phase4TrainingError,
                    expected_error,
                ):
                    phase4_training._replay_cache_build_envelope_seconds(
                        refresh
                    )
                with self.assertRaisesRegex(
                    Phase4TrainingError,
                    expected_error,
                ):
                    phase4_training._validate_replay_cache_refresh(refresh)


class ReplayInputPerformanceGateTests(unittest.TestCase):
    def test_invalid_timings_are_rejected_without_updating_the_gate(self) -> None:
        for value in (True, -1.0, math.nan, math.inf, -math.inf, "slow"):
            with self.subTest(value=value):
                gate = ReplayInputPerformanceGate()
                with self.assertRaisesRegex(Phase4TrainingError, "finite"):
                    gate.observe(value)  # type: ignore[arg-type]
                self.assertEqual(gate.count, 0)

    def test_slow_initial_mean_fails_before_eighth_optimizer_update(self) -> None:
        gate = ReplayInputPerformanceGate()
        optimizer_updates = 0
        slow = REPLAY_INPUT_GATE_MAX_MEAN_SECONDS + 0.01
        self.assertLess(slow, REPLAY_INPUT_GATE_MAX_BATCH_SECONDS)
        with self.assertRaisesRegex(Phase4ResourceLimitReached, "mean exceeded"):
            for _ in range(REPLAY_INPUT_GATE_BATCHES):
                gate.observe(slow)
                optimizer_updates += 1
        self.assertEqual(optimizer_updates, REPLAY_INPUT_GATE_BATCHES - 1)
        self.assertEqual(gate.count, REPLAY_INPUT_GATE_BATCHES)
        self.assertFalse(gate.initial_completed)

    def test_fast_gate_completes_once_and_live_spike_fails(self) -> None:
        gate = ReplayInputPerformanceGate()
        completions = sum(
            gate.observe(0.01) for _ in range(REPLAY_INPUT_GATE_BATCHES + 4)
        )
        self.assertEqual(completions, 1)
        payload = gate.to_payload()
        self.assertTrue(payload["initialGateCompleted"])
        self.assertEqual(payload["batches"], REPLAY_INPUT_GATE_BATCHES + 4)

        with self.assertRaisesRegex(Phase4ResourceLimitReached, "batch exceeded"):
            gate.observe(REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS + 0.01)


if __name__ == "__main__":
    unittest.main()
