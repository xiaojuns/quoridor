from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from quoridor import (
    ACTION_SPACE_SIZE,
    MAX_HALF_MOVES,
    State,
    apply_action_index,
    canonicalize_state,
    create_initial_state,
    current_repetition_count,
    decode_action,
    encode_action,
    encode_move,
    encode_wall,
    from_canonical_action_index,
    get_winner,
    is_passage_blocked,
    is_terminal,
    is_wall_structurally_valid,
    legal_action_indices,
    legal_pawn_moves,
    mirror_action_index,
    mirror_position_hash,
    mirror_state,
    position_hash,
    rotate_and_swap_action_index,
    rotate_and_swap_state,
    shortest_path_distance,
    state_from_serializable,
    state_to_serializable,
    to_canonical_action_index,
    winner_value,
)
from quoridor.network import encode_state


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "alpha-zero-conformance-v1.json"
)


class ActionCodecTests(unittest.TestCase):
    def test_all_209_actions_round_trip_and_transforms_are_involutions(self) -> None:
        self.assertEqual(ACTION_SPACE_SIZE, 209)
        for index in range(ACTION_SPACE_SIZE):
            with self.subTest(index=index):
                self.assertEqual(encode_action(decode_action(index)), index)
                self.assertEqual(mirror_action_index(mirror_action_index(index)), index)
                canonical = to_canonical_action_index(index, 1)
                self.assertEqual(from_canonical_action_index(canonical, 1), index)
                self.assertEqual(to_canonical_action_index(index, 0), index)

    def test_action_boundaries_match_phase_zero_layout(self) -> None:
        self.assertEqual(encode_move(0, 0), 0)
        self.assertEqual(encode_move(8, 8), 80)
        self.assertEqual(encode_wall("h", 0, 0), 81)
        self.assertEqual(encode_wall("h", 7, 7), 144)
        self.assertEqual(encode_wall("v", 0, 0), 145)
        self.assertEqual(encode_wall("v", 7, 7), 208)

    def test_invalid_action_values_are_rejected(self) -> None:
        for invalid in (-1, 209, True, 1.5, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    decode_action(invalid)
        with self.assertRaises(ValueError):
            encode_wall("x", 0, 0)
        with self.assertRaises(ValueError):
            encode_move(9, 0)

    def test_position_hash_mirror_matches_state_mirror_and_is_an_involution(self) -> None:
        state = apply_action_index(create_initial_state(), encode_move(8, 3))
        mirrored_hash = mirror_position_hash(position_hash(state))
        self.assertEqual(mirrored_hash, position_hash(mirror_state(state)))
        self.assertEqual(mirror_position_hash(mirrored_hash), position_hash(state))


class RuleEngineTests(unittest.TestCase):
    def test_initial_state_and_legal_mask(self) -> None:
        state = create_initial_state()
        self.assertEqual(position_hash(state), "[1,0,8,4,0,4,10,10,[],[]]")
        self.assertEqual(current_repetition_count(state), 1)
        self.assertEqual(
            legal_action_indices(state)[:3],
            (encode_move(7, 4), encode_move(8, 3), encode_move(8, 5)),
        )
        self.assertEqual(len(legal_action_indices(state)), 131)
        self.assertFalse(is_terminal(state))
        self.assertIsNone(winner_value(state))

    def test_pawn_jumps_and_diagonals_when_jump_is_blocked(self) -> None:
        open_jump = State(pawns=((4, 4), (3, 4)))
        self.assertIn((2, 4), legal_pawn_moves(open_jump))
        self.assertNotIn((3, 4), legal_pawn_moves(open_jump))

        blocked_jump = replace(open_jump, horizontal_walls=frozenset({(2, 4)}))
        moves = legal_pawn_moves(blocked_jump)
        self.assertNotIn((2, 4), moves)
        self.assertIn((3, 3), moves)
        self.assertIn((3, 5), moves)

    def test_passages_and_wall_structure_match_browser_rules(self) -> None:
        state = State(horizontal_walls=frozenset({(3, 3)}))
        self.assertTrue(is_passage_blocked(state, 3, 3, 4, 3))
        self.assertTrue(is_passage_blocked(state, 3, 4, 4, 4))
        self.assertFalse(is_passage_blocked(state, 3, 5, 4, 5))
        self.assertFalse(is_wall_structurally_valid(state, "h", 3, 4))
        self.assertFalse(is_wall_structurally_valid(state, "v", 3, 3))
        self.assertTrue(is_wall_structurally_valid(state, "v", 3, 5))
        self.assertEqual(shortest_path_distance(create_initial_state(), 0), 8.0)
        self.assertEqual(shortest_path_distance(create_initial_state(), 1), 8.0)

    def test_apply_is_immutable_and_rejects_illegal_actions(self) -> None:
        state = create_initial_state()
        before = state_to_serializable(state)
        next_state = apply_action_index(state, encode_move(7, 4))
        self.assertEqual(state_to_serializable(state), before)
        self.assertEqual(next_state.pawns[0], (7, 4))
        self.assertEqual(next_state.current_player, 1)
        self.assertEqual(next_state.move_number, 1)
        with self.assertRaises(ValueError):
            apply_action_index(state, encode_move(0, 0))

    def test_third_repetition_is_a_loss_for_the_mover(self) -> None:
        state = create_initial_state()
        cycle = (
            encode_move(8, 3),
            encode_move(0, 3),
            encode_move(8, 4),
            encode_move(0, 4),
        )
        for action in cycle:
            state = apply_action_index(state, action)
        self.assertFalse(is_terminal(state))
        self.assertEqual(current_repetition_count(state), 2)

        for action in cycle:
            state = apply_action_index(state, action)
        self.assertEqual(state.move_number, 8)
        self.assertEqual(state.winner, 0)
        self.assertEqual(state.termination_reason, "repetition")
        self.assertEqual(winner_value(state), 1.0)
        self.assertEqual(winner_value(state, 1), -1.0)
        self.assertEqual(legal_action_indices(state), ())

    def test_256th_half_move_is_a_loss_for_the_mover(self) -> None:
        state = replace(create_initial_state(), move_number=MAX_HALF_MOVES - 1)
        result = apply_action_index(state, encode_move(8, 3))
        self.assertEqual(result.move_number, MAX_HALF_MOVES)
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.termination_reason, "move-limit")

    def test_terminal_priority_is_goal_then_repetition_then_move_limit(self) -> None:
        goal_state = State(
            pawns=((1, 3), (0, 4)),
            move_number=MAX_HALF_MOVES - 1,
        )
        goal_state = replace(
            goal_state,
            position_counts={position_hash(goal_state): 1},
        )
        goal_result = apply_action_index(goal_state, encode_move(0, 3))
        self.assertEqual(goal_result.winner, 0)
        self.assertEqual(goal_result.termination_reason, "goal")
        self.assertEqual(winner_value(goal_result), -1.0)

        base = create_initial_state()
        probe = apply_action_index(base, encode_move(8, 3))
        repetition_at_limit = replace(
            base,
            move_number=MAX_HALF_MOVES - 1,
            position_counts={
                **base.position_counts,
                position_hash(probe): 2,
            },
        )
        result = apply_action_index(repetition_at_limit, encode_move(8, 3))
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.termination_reason, "repetition")


class SerializationTests(unittest.TestCase):
    def test_dto_round_trip_preserves_current_position_count(self) -> None:
        state = create_initial_state(starting_player=1)
        dto = state_to_serializable(state)
        restored = state_from_serializable(dto)
        self.assertEqual(state_to_serializable(restored), dto)
        self.assertEqual(position_hash(restored), position_hash(state))

    def test_canonical_and_mirror_transforms_do_not_mutate_source(self) -> None:
        state = apply_action_index(create_initial_state(1), encode_move(1, 4))
        before = state_to_serializable(state)
        canonical = canonicalize_state(state)
        mirrored = mirror_state(state)
        self.assertEqual(state_to_serializable(state), before)
        self.assertEqual(canonical.current_player, 0)
        self.assertEqual(mirror_state(mirrored), state)
        self.assertEqual(
            sum(mirrored.position_counts.values()),
            sum(state.position_counts.values()),
        )

    def test_player_role_transform_is_exact_and_not_a_new_augmentation(self) -> None:
        state = create_initial_state()
        for action in (
            encode_wall("v", 4, 5),
            encode_wall("h", 2, 1),
            encode_move(7, 4),
        ):
            state = apply_action_index(state, action)
        transformed = rotate_and_swap_state(state)
        self.assertEqual(rotate_and_swap_state(transformed), state)
        self.assertEqual(
            canonicalize_state(transformed),
            canonicalize_state(state),
        )
        self.assertTrue(
            torch.equal(
                encode_state(transformed),
                encode_state(state),
            )
        )
        mapped_legal = {
            rotate_and_swap_action_index(action)
            for action in legal_action_indices(state)
        }
        self.assertEqual(mapped_legal, set(legal_action_indices(transformed)))
        action = min(legal_action_indices(state))
        self.assertEqual(
            rotate_and_swap_state(apply_action_index(state, action)),
            apply_action_index(
                transformed,
                rotate_and_swap_action_index(action),
            ),
        )

        terminal_source = State(
            pawns=((1, 3), (7, 8)),
            walls_remaining=(0, 0),
            current_player=0,
            position_counts={},
        )
        terminal_source = replace(
            terminal_source,
            position_counts={position_hash(terminal_source): 1},
        )
        terminal = apply_action_index(terminal_source, encode_move(0, 3))
        transformed_terminal = rotate_and_swap_state(terminal)
        self.assertEqual(transformed_terminal.winner, 1)
        self.assertEqual(winner_value(transformed_terminal), winner_value(terminal))


class PhaseZeroFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_all_phase_zero_fixture_positions(self) -> None:
        self.assertEqual(self.fixture["schemaVersion"], 1)
        self.assertEqual(self.fixture["actionSpaceSize"], ACTION_SPACE_SIZE)
        self.assertEqual(self.fixture["positionCount"], 48)
        self.assertEqual(
            self.fixture["endRules"]["priority"],
            ["goal", "repetition", "move-limit"],
        )

        for entry in self.fixture["positions"]:
            with self.subTest(position=entry["id"]):
                state = state_from_serializable(entry["state"])
                canonical = canonicalize_state(state)
                mirrored = mirror_state(state)

                self.assertEqual(state_to_serializable(state), entry["state"])
                self.assertEqual(position_hash(state), entry["positionHash"])
                self.assertEqual(
                    list(legal_action_indices(state)),
                    entry["legalActionIndices"],
                )

                self.assertEqual(
                    state_to_serializable(canonical),
                    entry["canonical"]["state"],
                )
                self.assertEqual(
                    position_hash(canonical),
                    entry["canonical"]["positionHash"],
                )
                self.assertEqual(
                    list(legal_action_indices(canonical)),
                    entry["canonical"]["legalActionIndices"],
                )

                self.assertEqual(
                    state_to_serializable(mirrored),
                    entry["mirrored"]["state"],
                )
                self.assertEqual(
                    position_hash(mirrored),
                    entry["mirrored"]["positionHash"],
                )
                self.assertEqual(
                    list(legal_action_indices(mirrored)),
                    entry["mirrored"]["legalActionIndices"],
                )

                sample = entry["sample"]
                self.assertIn(sample["actionIndex"], legal_action_indices(state))
                self.assertEqual(
                    to_canonical_action_index(
                        sample["actionIndex"], state.current_player
                    ),
                    sample["canonicalActionIndex"],
                )
                self.assertEqual(
                    mirror_action_index(sample["actionIndex"]),
                    sample["mirroredActionIndex"],
                )

                next_state = apply_action_index(state, sample["actionIndex"])
                self.assertEqual(
                    position_hash(next_state),
                    sample["nextPositionHash"],
                )
                self.assertEqual(
                    position_hash(canonicalize_state(next_state)),
                    sample["canonicalNextPositionHash"],
                )
                self.assertEqual(
                    position_hash(mirror_state(next_state)),
                    sample["mirroredNextPositionHash"],
                )

                canonical_next = apply_action_index(
                    canonical, sample["canonicalActionIndex"]
                )
                self.assertEqual(
                    position_hash(canonicalize_state(canonical_next)),
                    sample["canonicalNextPositionHash"],
                )
                mirrored_next = apply_action_index(
                    mirrored, sample["mirroredActionIndex"]
                )
                self.assertEqual(
                    position_hash(mirrored_next),
                    sample["mirroredNextPositionHash"],
                )


if __name__ == "__main__":
    unittest.main()
