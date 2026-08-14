from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.classic_bridge import ClassicBridgeResult
from quoridor.codec import (
    decode_action,
    encode_move,
    mirror_action_index,
    mirror_state,
    position_hash,
)
from quoridor.game import apply_action_index, create_initial_state, legal_action_indices
from quoridor.openings import state_to_opening_payload
from quoridor.replay_store import ReplayGame, ReplayPly, reconstruct_game
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes
from quoridor.teacher_data import (
    TEACHER_POLICY_FLOOR_UNITS,
    TEACHER_POLICY_SCALE,
    TeacherDataConfig,
    TeacherDataContractError,
    TeacherDataError,
    TeacherDatasetGenerator,
    canonical_state_orbit_digest,
    evaluation_position_orbit_digest,
    exclusion_set_sha256,
    split_for_game,
    teacher_policy_counts,
)


class FakeClassicBridge:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def choose(self, state, *, budget, max_depth, timeout):
        del max_depth, timeout
        self.calls += 1
        action = legal_action_indices(state)[0]
        return ClassicBridgeResult(
            action_index=action,
            action=decode_action(action),
            score=12.0,
            depth=1,
            nodes=budget.limit,
            elapsed_ms=0.0,
            position_hash=position_hash(state),
            history_entries=len(state.position_counts),
            history_occurrences=sum(state.position_counts.values()),
            budget=budget,
        )

    def close(self) -> None:
        self.closed = True


class TeacherEvaluationExclusionTests(unittest.TestCase):
    def test_generation_discards_an_excluded_playable_position_orbit(self) -> None:
        exclusions = tuple(
            sorted(
                {
                    evaluation_position_orbit_digest(create_initial_state(player))
                    for player in (0, 1)
                }
            )
        )
        config = TeacherDataConfig(
            target_positions=20,
            seed=19,
            node_budget=4,
            max_depth=1,
            exploration_probability=0.0,
            opening_depths=(0,),
            request_timeout_seconds=1.0,
            maximum_attempts=1,
            exclusion_set_sha256=exclusion_set_sha256(exclusions),
            exclusion_orbit_count=len(exclusions),
        )
        bridge = FakeClassicBridge()
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "phase4"
            generator = TeacherDatasetGenerator(
                config,
                output_root=artifact_root / "teacher",
                artifact_root=artifact_root,
                workspace_root=workspace,
                resource_checker=no_resource_limits,
                excluded_position_orbits=exclusions,
            )
            with self.assertRaisesRegex(TeacherDataError, "maximum teacher"):
                generator.generate(bridge_factory=lambda: bridge)
        self.assertEqual(bridge.calls, 0)
        self.assertTrue(bridge.closed)

    def test_exclusion_set_is_bound_to_teacher_config(self) -> None:
        digest = evaluation_position_orbit_digest(create_initial_state(0))
        config = TeacherDataConfig(
            target_positions=20,
            exclusion_set_sha256=exclusion_set_sha256((digest,)),
            exclusion_orbit_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "phase4"
            with self.assertRaisesRegex(ValueError, "exclusion set differs"):
                TeacherDatasetGenerator(
                    config,
                    output_root=artifact_root / "teacher",
                    artifact_root=artifact_root,
                    workspace_root=Path(__file__).resolve().parents[2],
                    resource_checker=no_resource_limits,
                )


def no_resource_limits(_path: Path) -> dict[str, int]:
    return {"artifactBytes": 0, "freeDiskBytes": 100 * 1024**3}


def raw_state_digest(state) -> str:
    return sha256_bytes(canonical_json_bytes(state_to_opening_payload(state)))


def mirrored_teacher_game(game: ReplayGame, *, game_index: int, split: str) -> ReplayGame:
    states = reconstruct_game(game)
    mirrored_plies: list[ReplayPly] = []
    for state, ply in zip(states, game.plies):
        mirrored_state = mirror_state(state)
        mirrored_counts = [0] * len(ply.visit_counts)
        for action, count in enumerate(ply.visit_counts):
            mirrored_counts[mirror_action_index(action)] = count
        metadata = dict(ply.metadata)
        metadata.update(
            {
                "teacherActionIndex": mirror_action_index(
                    metadata["teacherActionIndex"]
                ),
                "teacherStateSha256": raw_state_digest(mirrored_state),
                "teacherOrbitSha256": canonical_state_orbit_digest(mirrored_state),
                "positionHash": position_hash(mirrored_state),
            }
        )
        mirrored_plies.append(
            replace(
                ply,
                action_index=mirror_action_index(ply.action_index),
                visit_counts=tuple(mirrored_counts),
                metadata=metadata,
            )
        )
    game_metadata = dict(game.metadata)
    game_metadata.update(
        {
            "datasetSplit": split,
            "gameIndex": game_index,
            "openingActions": [
                mirror_action_index(action)
                for action in game_metadata["openingActions"]
            ],
        }
    )
    return ReplayGame(
        game_id=f"teacher-mirror-leak-{game_index:08d}",
        initial_state=mirror_state(game.initial_state),
        plies=tuple(mirrored_plies),
        metadata=game_metadata,
    )


class TeacherPolicyTests(unittest.TestCase):
    def test_policy_retains_exactly_all_legal_actions_with_fixed_floor(self) -> None:
        state = create_initial_state()
        legal = legal_action_indices(state)
        teacher_action = legal[-1]

        counts = teacher_policy_counts(legal, teacher_action)

        self.assertEqual(len(counts), 209)
        self.assertEqual(sum(counts), TEACHER_POLICY_SCALE)
        self.assertEqual(
            {index for index, count in enumerate(counts) if count > 0}, set(legal)
        )
        for action in legal:
            self.assertGreaterEqual(counts[action], TEACHER_POLICY_FLOOR_UNITS)
        self.assertGreater(counts[teacher_action], TEACHER_POLICY_FLOOR_UNITS)

    def test_config_freezes_ten_percent_holdout_and_validates_floor(self) -> None:
        config = TeacherDataConfig(target_positions=50_000)
        self.assertEqual(config.train_target, 45_000)
        self.assertEqual(config.holdout_target, 5_000)
        self.assertEqual(
            config.to_payload()["policy"]["legalSupport"], "complete"
        )
        with self.assertRaisesRegex(ValueError, "divisible by 10"):
            TeacherDataConfig(target_positions=51)
        with self.assertRaisesRegex(ValueError, "leaves no mass"):
            TeacherDataConfig(
                target_positions=10,
                policy_scale=209,
                policy_floor_units=1,
            )

    def test_split_is_stable_and_close_to_ten_percent_over_many_games(self) -> None:
        first = [split_for_game(1234, index) for index in range(1_000)]
        second = [split_for_game(1234, index) for index in range(1_000)]
        self.assertEqual(first, second)
        holdouts = first.count("holdout")
        self.assertGreaterEqual(holdouts, 70)
        self.assertLessEqual(holdouts, 130)

    def test_deduplication_digest_collapses_horizontal_mirror_orbit(self) -> None:
        state = apply_action_index(create_initial_state(), encode_move(8, 3))
        mirrored = mirror_state(state)

        self.assertNotEqual(position_hash(state), position_hash(mirrored))
        self.assertEqual(
            canonical_state_orbit_digest(state),
            canonical_state_orbit_digest(mirrored),
        )


class TeacherDatasetIntegrationTests(unittest.TestCase):
    def test_deep_resume_rejects_mirror_leak_between_splits(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        config = TeacherDataConfig(
            target_positions=20,
            seed=91,
            node_budget=8,
            max_depth=1,
            exploration_probability=0.0,
            opening_depths=(2,),
            request_timeout_seconds=1.0,
            maximum_attempts=200,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "phase4"
            output_root = artifact_root / "teacher"
            generator = TeacherDatasetGenerator(
                config,
                output_root=output_root,
                artifact_root=artifact_root,
                workspace_root=workspace,
                resource_checker=no_resource_limits,
            )
            generator.generate(
                bridge_factory=FakeClassicBridge,
                maximum_new_games=1,
            )
            source_split = (
                "train" if generator.stores["train"].game_count else "holdout"
            )
            target_split = "holdout" if source_split == "train" else "train"
            source_game = next(generator.stores[source_split].iter_games())
            game_index = source_game.metadata["gameIndex"] + 1
            while split_for_game(config.seed, game_index) != target_split:
                game_index += 1
            generator.stores[target_split].append(
                mirrored_teacher_game(
                    source_game,
                    game_index=game_index,
                    split=target_split,
                )
            )

            with self.assertRaisesRegex(
                TeacherDataContractError, "mirrored or duplicate"
            ):
                TeacherDatasetGenerator(
                    config,
                    output_root=output_root,
                    artifact_root=artifact_root,
                    workspace_root=workspace,
                    resource_checker=no_resource_limits,
                )

    def test_generation_resumes_and_keeps_whole_game_holdout_isolated(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        config = TeacherDataConfig(
            target_positions=20,
            seed=73,
            node_budget=16,
            max_depth=2,
            exploration_probability=0.0,
            opening_depths=(2,),
            request_timeout_seconds=1.0,
            maximum_attempts=200,
        )
        bridges: list[FakeClassicBridge] = []

        def bridge_factory():
            bridge = FakeClassicBridge()
            bridges.append(bridge)
            return bridge

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "phase4"
            output_root = artifact_root / "teacher"
            first = TeacherDatasetGenerator(
                config,
                output_root=output_root,
                artifact_root=artifact_root,
                workspace_root=workspace,
                resource_checker=no_resource_limits,
            )
            paused = first.generate(
                bridge_factory=bridge_factory,
                maximum_new_games=1,
            )
            self.assertEqual(paused.status, "paused")
            self.assertEqual(paused.train_games + paused.holdout_games, 1)
            self.assertTrue(bridges[-1].closed)

            incompatible = TeacherDataConfig(
                target_positions=20,
                seed=73,
                node_budget=17,
                max_depth=2,
                exploration_probability=0.0,
                opening_depths=(2,),
                request_timeout_seconds=1.0,
                maximum_attempts=200,
            )
            with self.assertRaisesRegex(
                TeacherDataContractError, "config differs"
            ):
                TeacherDatasetGenerator(
                    incompatible,
                    output_root=output_root,
                    artifact_root=artifact_root,
                    workspace_root=workspace,
                    resource_checker=no_resource_limits,
                )

            manifest_bytes = paused.manifest_path.read_bytes()
            tampered = json.loads(manifest_bytes)
            tampered["nextGameIndex"] += 1
            paused.manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                TeacherDataContractError, "checksum mismatch"
            ):
                TeacherDatasetGenerator(
                    config,
                    output_root=output_root,
                    artifact_root=artifact_root,
                    workspace_root=workspace,
                    resource_checker=no_resource_limits,
                )
            paused.manifest_path.write_bytes(manifest_bytes)

            resumed = TeacherDatasetGenerator(
                config,
                output_root=output_root,
                artifact_root=artifact_root,
                workspace_root=workspace,
                resource_checker=no_resource_limits,
            )
            completed = resumed.generate(bridge_factory=bridge_factory)

            self.assertEqual(completed.status, "completed")
            self.assertGreaterEqual(completed.train_positions, config.train_target)
            self.assertGreaterEqual(
                completed.holdout_positions, config.holdout_target
            )
            self.assertIsNotNone(completed.dataset_fingerprint)
            self.assertTrue(bridges[-1].closed)

            verified = TeacherDatasetGenerator(
                config,
                output_root=output_root,
                artifact_root=artifact_root,
                workspace_root=workspace,
                resource_checker=no_resource_limits,
            ).verify(require_complete=True)
            self.assertEqual(
                verified.dataset_fingerprint, completed.dataset_fingerprint
            )

            train_digests: set[str] = set()
            holdout_digests: set[str] = set()
            game_indices: set[int] = set()
            for split, destination in (
                ("train", train_digests),
                ("holdout", holdout_digests),
            ):
                for game in resumed.stores[split].iter_games():
                    game_index = game.metadata["gameIndex"]
                    self.assertNotIn(game_index, game_indices)
                    game_indices.add(game_index)
                    self.assertEqual(split_for_game(config.seed, game_index), split)
                    states = reconstruct_game(game)
                    for state, ply in zip(states, game.plies):
                        destination.add(canonical_state_orbit_digest(state))
                        legal = set(legal_action_indices(state))
                        support = {
                            index
                            for index, count in enumerate(ply.visit_counts)
                            if count > 0
                        }
                        self.assertEqual(support, legal)
            self.assertTrue(train_digests.isdisjoint(holdout_digests))
            self.assertEqual(
                len(train_digests) + len(holdout_digests),
                completed.total_positions,
            )

            manifest = json.loads(completed.manifest_path.read_text())
            stored_hash = manifest.pop("manifestSha256")
            self.assertEqual(stored_hash, sha256_bytes(canonical_json_bytes(manifest)))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["actual"]["uniqueRawStates"], completed.total_positions
            )
            for category in (
                "startingPlayer",
                "winner",
                "sideToMove",
                "wallCountBucket",
                "openingDepth",
                "terminationReason",
            ):
                self.assertEqual(
                    sum(manifest["strata"][category].values()),
                    completed.total_positions,
                )


if __name__ == "__main__":
    unittest.main()
