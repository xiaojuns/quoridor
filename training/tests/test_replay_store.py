from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.codec import encode_move, position_hash
from quoridor.game import State, apply_action_index, create_initial_state, legal_action_indices
from quoridor.replay_store import (
    REPLAY_SHARD_SCHEMA_VERSION,
    REPLAY_STORE_SCHEMA_VERSION,
    ReplayCapacityError,
    ReplayConfigurationError,
    ReplayCorruptionError,
    ReplayGame,
    ReplayPly,
    ReplayStore,
    reconstruct_game,
)


def _initial_state(*, row: int = 1, move_number: int = 2) -> State:
    state = State(
        pawns=((row, 3), (0, 8)),
        walls_remaining=(0, 0),
        current_player=0,
        move_number=move_number,
    )
    historical = create_initial_state(starting_player=1)
    return replace(
        state,
        position_counts={
            position_hash(historical): 2,
            position_hash(state): 1,
        },
    )


def _game(
    game_id: str,
    *,
    initial_state: State | None = None,
    actions: tuple[int, ...] | None = None,
    generation: int = 7,
) -> ReplayGame:
    state = initial_state or _initial_state()
    actions = actions or (encode_move(0, 3),)
    plies: list[ReplayPly] = []
    for index, action in enumerate(actions):
        legal = legal_action_indices(state)
        if action not in legal:
            raise AssertionError(f"test action {action} is illegal at ply {index}")
        counts = [0] * 209
        counts[action] = 32 + index
        alternative = next((candidate for candidate in legal if candidate != action), None)
        if alternative is not None:
            counts[alternative] = 3
        plies.append(
            ReplayPly(
                action_index=action,
                visit_counts=tuple(counts),
                root_value=0.25 - index * 0.1,
                model_generation=generation + index % 2,
                model_id=f"champion-{generation + index % 2}",
                metadata={"searchSeed": 100 + index},
            )
        )
        state = apply_action_index(state, action)
    return ReplayGame(
        game_id=game_id,
        initial_state=initial_state or _initial_state(),
        plies=tuple(plies),
        metadata={"actor": 2, "opening": ["phase4", generation]},
    )


def _three_ply_game(game_id: str) -> ReplayGame:
    initial = _initial_state(row=2)
    return _game(
        game_id,
        initial_state=initial,
        actions=(encode_move(1, 3), encode_move(0, 7), encode_move(0, 3)),
    )


class ReplayStoreRoundTripTests(unittest.TestCase):
    def test_replay_game_requires_complete_initial_history(self) -> None:
        initial = replace(_initial_state(), move_number=3)
        with self.assertRaisesRegex(ValueError, "move_number"):
            _game("incomplete-history", initial_state=initial)

    def test_game_shard_round_trip_preserves_full_history_and_search_metadata(self) -> None:
        game = _game("round-trip")
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=8,
                max_plies=100,
                max_bytes=1_000_000,
            )
            shard = store.append(game)

            self.assertEqual(store.game_count, 1)
            self.assertEqual(store.ply_count, 1)
            self.assertEqual(shard.minimum_generation, 7)
            self.assertEqual(shard.maximum_generation, 7)
            self.assertLessEqual(store.committed_bytes, store.max_bytes)

            restored = store.load_game(game.game_id)
            self.assertEqual(restored, game)
            self.assertEqual(
                dict(restored.initial_state.position_counts),
                dict(game.initial_state.position_counts),
            )
            states = reconstruct_game(restored)
            self.assertEqual(len(states), 2)
            self.assertEqual(states[-1].winner, 0)
            self.assertEqual(states[-1].termination_reason, "goal")

            verification = store.verify(deep=True)
            self.assertEqual(verification.game_count, 1)
            self.assertEqual(verification.ply_count, 1)
            self.assertTrue(verification.deep)

            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["schemaVersion"], REPLAY_STORE_SCHEMA_VERSION)
            self.assertEqual(
                manifest["shardSchemaVersion"], REPLAY_SHARD_SCHEMA_VERSION
            )
            self.assertEqual(manifest["totals"]["games"], 1)
            self.assertEqual(len(tuple(Path(directory).glob("game-*.qrp"))), 1)
            self.assertFalse(tuple(Path(directory).glob(".*.tmp")))

    def test_deep_verification_can_bind_decoded_game_metadata(self) -> None:
        game = _game("validated")
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            store.append(game)
            seen: list[str] = []

            verification = store.verify(
                deep=True,
                game_validator=lambda decoded, _states: seen.append(
                    decoded.game_id
                ),
            )

            self.assertTrue(verification.deep)
            self.assertEqual(seen, ["validated"])
            with self.assertRaisesRegex(ValueError, "requires deep"):
                store.verify(
                    deep=False,
                    game_validator=lambda _decoded, _states: None,
                )

    def test_sample_reconstructs_raw_ply_and_exposes_augmentation_hook(self) -> None:
        game = _three_ply_game("sampling")
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            store.append(game)

            def mark_for_future_mirroring(sample):
                return replace(sample, augmentation="mirror-extension")

            samples = store.sample_plies(
                20,
                rng=random.Random(17),
                transform=mark_for_future_mirroring,
            )
            self.assertEqual(len(samples), 3)
            self.assertEqual({sample.ply_index for sample in samples}, {0, 1, 2})
            self.assertTrue(
                all(sample.augmentation == "mirror-extension" for sample in samples)
            )
            states = reconstruct_game(game)
            for sample in samples:
                self.assertEqual(sample.state, states[sample.ply_index])
                self.assertEqual(
                    sample.visit_counts,
                    game.plies[sample.ply_index].visit_counts,
                )
                expected = 1.0 if sample.state.current_player == 0 else -1.0
                self.assertEqual(sample.outcome, expected)

    def test_store_reopens_without_loading_tensor_replay_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            first.append(_game("resume"))
            resumed = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            self.assertEqual(resumed.game_ids, ("resume",))
            self.assertEqual(resumed.load_game("resume"), _game("resume"))
            with self.assertRaises(ReplayConfigurationError):
                ReplayStore(
                    directory,
                    max_games=5,
                    max_plies=20,
                    max_bytes=1_000_000,
                )
            with self.assertRaises(ReplayConfigurationError):
                ReplayStore(
                    directory,
                    max_games=1,
                    max_plies=20,
                    max_bytes=1_000_000,
                    recover=True,
                )
            self.assertEqual(resumed.game_ids, ("resume",))


class ReplayStoreCapacityTests(unittest.TestCase):
    def test_fifo_capacity_evicts_whole_game_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=2,
                max_plies=2,
                max_bytes=1_000_000,
            )
            first = store.append(_game("game-1", generation=1))
            store.append(_game("game-2", generation=2))
            store.append(_game("game-3", generation=3))

            self.assertEqual(store.game_ids, ("game-2", "game-3"))
            self.assertFalse((Path(directory) / first.file_name).exists())
            with self.assertRaises(KeyError):
                store.load_game("game-1")
            store.verify(deep=True)

    def test_single_game_must_fit_ply_and_byte_budgets(self) -> None:
        large_game = _three_ply_game("too-many-plies")
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=2,
                max_bytes=1_000_000,
            )
            with self.assertRaises(ReplayCapacityError):
                store.append(large_game)
            self.assertEqual(store.game_count, 0)

        game = _game("byte-budget")
        with tempfile.TemporaryDirectory() as measurement_directory:
            measurement = ReplayStore(
                measurement_directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            measurement.append(game)
            required = measurement.committed_bytes
        with tempfile.TemporaryDirectory() as limited_directory:
            limited = ReplayStore(
                limited_directory,
                max_games=4,
                max_plies=20,
                max_bytes=required - 100,
            )
            with self.assertRaises(ReplayCapacityError):
                limited.append(game)
            self.assertEqual(limited.game_count, 0)

    def test_duplicate_game_id_is_rejected_without_changing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            store.append(_game("duplicate"))
            manifest_before = (Path(directory) / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                store.append(_game("duplicate"))
            self.assertEqual(
                (Path(directory) / "manifest.json").read_bytes(), manifest_before
            )


class ReplayStoreValidationTests(unittest.TestCase):
    def test_invalid_trajectory_and_nonfinite_search_data_are_rejected(self) -> None:
        state = _initial_state()
        valid_action = encode_move(0, 3)
        counts = [0] * 209
        counts[valid_action] = 1
        with self.assertRaisesRegex(ValueError, "finite"):
            ReplayPly(
                action_index=valid_action,
                visit_counts=tuple(counts),
                root_value=float("nan"),
                model_generation=0,
                model_id="bad",
            )

        illegal_search_counts = counts.copy()
        illegal_search_counts[encode_move(8, 8)] = 1
        invalid_game = ReplayGame(
            game_id="illegal-search-edge",
            initial_state=state,
            plies=(
                ReplayPly(
                    action_index=valid_action,
                    visit_counts=tuple(illegal_search_counts),
                    root_value=0.0,
                    model_generation=0,
                    model_id="model-0",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "illegal actions"):
            reconstruct_game(invalid_game)

        incomplete_state = create_initial_state()
        incomplete_action = legal_action_indices(incomplete_state)[0]
        incomplete_counts = [0] * 209
        incomplete_counts[incomplete_action] = 1
        incomplete = ReplayGame(
            game_id="incomplete",
            initial_state=incomplete_state,
            plies=(
                ReplayPly(
                    action_index=incomplete_action,
                    visit_counts=tuple(incomplete_counts),
                    root_value=0.0,
                    model_generation=0,
                    model_id="model-0",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "terminal"):
            reconstruct_game(incomplete)

        invalid_history = replace(state, position_counts={"bad": 3})
        with self.assertRaisesRegex(ValueError, "position hash"):
            ReplayGame(
                game_id="invalid-history-hash",
                initial_state=invalid_history,
                plies=_game("valid").plies,
            )

    def test_corrupted_shard_is_detected_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            entry = store.append(_game("corrupt-me"))
            path = Path(directory) / entry.file_name
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 0xFF
            path.write_bytes(damaged)

            store.verify(deep=False)
            with self.assertRaisesRegex(ReplayCorruptionError, "checksum"):
                store.verify(deep=True)
            with self.assertRaises(ReplayCorruptionError):
                store.load_game("corrupt-me")

    def test_manifest_loss_can_be_recovered_from_self_describing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            store.append(_game("recover-1", generation=1))
            store.append(_game("recover-2", generation=2))
            (Path(directory) / "manifest.json").write_text("not-json", encoding="utf-8")

            with self.assertRaises(ReplayCorruptionError):
                ReplayStore(
                    directory,
                    max_games=4,
                    max_plies=20,
                    max_bytes=1_000_000,
                )
            recovered = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
                recover=True,
            )
            self.assertEqual(recovered.game_ids, ("recover-1", "recover-2"))
            recovered.verify(deep=True)

    def test_recovery_never_prunes_shards_to_fit_a_new_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            store.append(_game("keep-1"))
            store.append(_game("keep-2"))
            (Path(directory) / "manifest.json").write_text(
                "not-json", encoding="utf-8"
            )
            with self.assertRaisesRegex(ReplayConfigurationError, "capacity"):
                ReplayStore(
                    directory,
                    max_games=1,
                    max_plies=20,
                    max_bytes=1_000_000,
                    recover=True,
                )
            self.assertEqual(len(tuple(Path(directory).glob("game-*.qrp"))), 2)
            recovered = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
                recover=True,
            )
            self.assertEqual(recovered.game_ids, ("keep-1", "keep-2"))

    def test_missing_committed_shard_and_unknown_manifest_schema_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            entry = store.append(_game("missing"))
            (Path(directory) / entry.file_name).unlink()
            with self.assertRaisesRegex(ReplayCorruptionError, "missing"):
                ReplayStore(
                    directory,
                    max_games=4,
                    max_plies=20,
                    max_bytes=1_000_000,
                )

        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(
                directory,
                max_games=4,
                max_plies=20,
                max_bytes=1_000_000,
            )
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schemaVersion"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ReplayCorruptionError, "schema"):
                ReplayStore(
                    directory,
                    max_games=4,
                    max_plies=20,
                    max_bytes=1_000_000,
                )


if __name__ == "__main__":
    unittest.main()
