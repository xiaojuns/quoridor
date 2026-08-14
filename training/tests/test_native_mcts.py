from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest

from quoridor.codec import (
    canonicalize_state,
    encode_move,
    encode_wall,
    mirror_state,
    position_hash,
    rotate_and_swap_action_index,
    rotate_and_swap_state,
    state_to_serializable,
)
from quoridor.game import (
    State,
    apply_action_index,
    create_initial_state,
    legal_action_indices,
)
from quoridor.mcts import MCTSConfig
from quoridor.native_mcts import NativeMCTS
from quoridor.selfplay import SelfPlayConfig, play_self_games_batched


NATIVE_AVAILABLE = importlib.util.find_spec("quoridor_native") is not None


class BatchEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate(self, state: object):
        return [0.0] * 209, 0.0

    def evaluate_batch(self, states: list[object]):
        self.batch_sizes.append(len(states))
        return [([0.0] * 209, 0.0) for _ in states]


class PreferenceEvaluator:
    def __init__(self, action_index: int) -> None:
        self.action_index = action_index
        self.move_numbers: list[int] = []

    def _result(self, state: State):
        self.move_numbers.append(state.move_number)
        logits = [-100.0] * 209
        logits[self.action_index] = 100.0
        return logits, 0.0

    def evaluate(self, state: State):
        return self._result(state)

    def evaluate_batch(self, states: list[State]):
        return [self._result(state) for state in states]


@unittest.skipUnless(NATIVE_AVAILABLE, "native extension has not been built")
class NativeMCTSTests(unittest.TestCase):
    def test_native_symmetries_transform_complete_python_history(self) -> None:
        import quoridor_native

        state = create_initial_state(starting_player=1)
        state = apply_action_index(state, encode_wall("v", 4, 5))
        state = apply_action_index(state, encode_wall("h", 2, 1))
        cycle = (
            encode_move(1, 4),
            encode_move(7, 4),
            encode_move(0, 4),
            encode_move(8, 4),
        )
        for action_index in cycle:
            state = apply_action_index(state, action_index)

        self.assertEqual(state.current_player, 1)
        self.assertGreater(len(state.position_counts), 1)
        native_state = quoridor_native.state_from_python(state)

        expected_canonical = canonicalize_state(state)
        actual_canonical = quoridor_native.state_to_python(
            quoridor_native.canonicalize_state(native_state)
        )
        self.assertEqual(actual_canonical, expected_canonical)
        self.assertEqual(
            dict(actual_canonical.position_counts),
            dict(expected_canonical.position_counts),
        )

        expected_mirror = mirror_state(state)
        native_mirror = quoridor_native.mirror_state(native_state)
        actual_mirror = quoridor_native.state_to_python(native_mirror)
        self.assertEqual(actual_mirror, expected_mirror)
        self.assertEqual(
            dict(actual_mirror.position_counts),
            dict(expected_mirror.position_counts),
        )
        restored = quoridor_native.state_to_python(
            quoridor_native.mirror_state(native_mirror)
        )
        self.assertEqual(restored, state)

        expected_role_swap = rotate_and_swap_state(state)
        native_role_swap = quoridor_native.rotate_and_swap_state(native_state)
        actual_role_swap = quoridor_native.state_to_python(native_role_swap)
        self.assertEqual(actual_role_swap, expected_role_swap)
        self.assertEqual(
            dict(actual_role_swap.position_counts),
            dict(expected_role_swap.position_counts),
        )
        self.assertEqual(
            quoridor_native.state_to_python(
                quoridor_native.rotate_and_swap_state(native_role_swap)
            ),
            state,
        )
        self.assertEqual(
            set(legal_action_indices(actual_role_swap)),
            {
                rotate_and_swap_action_index(action)
                for action in legal_action_indices(state)
            },
        )

    def test_malformed_dto_raises_instead_of_crashing(self) -> None:
        import quoridor_native

        dto = state_to_serializable(create_initial_state())
        dto["pawns"] = [[8, 4], [0, 4]]
        with self.assertRaises(TypeError):
            quoridor_native.state_from_dict(dto)

    def test_multi_root_search_uses_real_batches(self) -> None:
        evaluator = BatchEvaluator()
        search = NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=3,
                temperature=0.0,
                add_root_noise=False,
                seed=17,
            ),
        )
        states = (create_initial_state(), create_initial_state())
        results = search.run_batch(
            states,
            temperatures=(0.0, 0.0),
            add_root_noise=False,
        )

        self.assertEqual(len(results), 2)
        self.assertIn(2, evaluator.batch_sizes)
        for state, result in zip(states, results):
            self.assertIn(result.action_index, legal_action_indices(state))
            self.assertEqual(sum(result.visit_counts), 3)
            self.assertEqual(len(result.policy), 209)

    def test_one_search_instance_serializes_concurrent_calls(self) -> None:
        evaluator = BatchEvaluator()
        search = NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=2,
                temperature=0.0,
                add_root_noise=False,
                seed=23,
            ),
        )
        state = create_initial_state()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: search.run(state), range(2)))
        self.assertTrue(
            all(result.action_index in legal_action_indices(state) for result in results)
        )

    def test_persistent_session_promotes_child_and_reuses_visits(self) -> None:
        state = replace(create_initial_state(), walls_remaining=(0, 0))
        preferred_action = encode_move(7, 4)
        evaluator = PreferenceEvaluator(preferred_action)
        search = NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=4,
                temperature=0.0,
                add_root_noise=False,
                seed=31,
            ),
        )
        session_id = search.create_session(state)
        first = search.run_session(session_id, state)

        self.assertEqual(first.action_index, preferred_action)
        self.assertEqual(evaluator.move_numbers.count(1), 1)
        nodes_before = search.session_node_count(session_id)
        next_state = apply_action_index(state, preferred_action)
        self.assertTrue(
            search.advance_session(session_id, preferred_action, next_state)
        )
        self.assertLess(search.session_node_count(session_id), nodes_before)

        second = search.run_session(session_id, next_state)
        self.assertGreater(sum(second.visit_counts), 4)
        self.assertEqual(evaluator.move_numbers.count(1), 1)
        self.assertTrue(search.sync_session(session_id, next_state))
        self.assertEqual(search.session_count, 1)
        self.assertTrue(search.release_session(session_id))
        self.assertFalse(search.release_session(session_id))
        self.assertEqual(search.session_count, 0)

    def test_session_rebuilds_on_same_hash_with_different_history(self) -> None:
        state = replace(create_initial_state(), walls_remaining=(0, 0))
        evaluator = PreferenceEvaluator(encode_move(7, 4))
        search = NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=3,
                temperature=0.0,
                add_root_noise=False,
                seed=37,
            ),
        )
        session_id = search.create_session(state)
        result = search.run_session(session_id, state)
        self.assertGreater(search.session_node_count(session_id), 1)

        mismatched = replace(
            state,
            position_counts={**state.position_counts, "unrelated-history": 2},
        )
        self.assertEqual(position_hash(mismatched), position_hash(state))
        self.assertFalse(search.sync_session(session_id, mismatched))
        self.assertEqual(search.session_node_count(session_id), 1)
        self.assertTrue(search.sync_session(session_id, mismatched))

        advance_id = search.create_session(state)
        advance_result = search.run_session(advance_id, state)
        self.assertIsNotNone(advance_result.action_index)
        next_state = apply_action_index(state, advance_result.action_index)
        wrong_history = replace(
            next_state,
            position_counts={
                **next_state.position_counts,
                "different-next-history": 2,
            },
        )
        self.assertFalse(
            search.advance_session(
                advance_id,
                advance_result.action_index,
                wrong_history,
            )
        )
        self.assertEqual(search.session_node_count(advance_id), 1)
        search.clear_sessions()
        self.assertEqual(search.session_count, 0)
        self.assertIn(result.action_index, legal_action_indices(state))

    def test_session_root_noise_is_sampled_once_per_decision(self) -> None:
        state = replace(create_initial_state(), walls_remaining=(0, 0))
        search = NativeMCTS(
            BatchEvaluator(),
            MCTSConfig(
                simulations=12,
                temperature=1.0,
                add_root_noise=True,
                seed=41,
            ),
        )
        session_id = search.create_session(state)
        first = search.run_session(
            session_id,
            state,
            temperature=1.0,
            add_root_noise=True,
        )
        second = search.run_session(
            session_id,
            state,
            temperature=1.0,
            add_root_noise=True,
        )
        self.assertEqual(
            {action: edge.prior for action, edge in first.root_edges.items()},
            {action: edge.prior for action, edge in second.root_edges.items()},
        )
        self.assertEqual(sum(second.visit_counts), 24)
        with self.assertRaises(ValueError):
            search.run_session(
                session_id,
                state,
                temperature=1.0,
                add_root_noise=False,
            )

        self.assertIsNotNone(second.action_index)
        next_state = apply_action_index(state, second.action_index)
        self.assertTrue(
            search.advance_session(session_id, second.action_index, next_state)
        )
        next_first = search.run_session(
            session_id,
            next_state,
            temperature=1.0,
            add_root_noise=True,
        )
        next_second = search.run_session(
            session_id,
            next_state,
            temperature=1.0,
            add_root_noise=True,
        )
        self.assertEqual(
            {action: edge.prior for action, edge in next_first.root_edges.items()},
            {action: edge.prior for action, edge in next_second.root_edges.items()},
        )

    def test_multiple_persistent_sessions_keep_batching_and_release(self) -> None:
        evaluator = BatchEvaluator()
        search = NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=2,
                temperature=0.0,
                add_root_noise=False,
                seed=43,
            ),
        )
        states = (create_initial_state(), create_initial_state())
        session_ids = tuple(search.create_session(state) for state in states)
        results = search.run_sessions(
            session_ids,
            states,
            temperatures=(0.0, 0.0),
            add_root_noise=False,
        )
        self.assertEqual(len(results), 2)
        self.assertIn(2, evaluator.batch_sizes)
        self.assertEqual(search.session_count, 2)
        with self.assertRaises(ValueError):
            search.run_sessions(
                (session_ids[0], session_ids[0]),
                states,
                temperatures=(0.0, 0.0),
                add_root_noise=False,
            )
        search.clear_sessions()
        self.assertEqual(search.session_count, 0)

    def test_batched_self_play_finishes_without_illegal_actions(self) -> None:
        evaluator = BatchEvaluator()
        initial = State(
            pawns=((1, 0), (0, 8)),
            walls_remaining=(0, 0),
            current_player=0,
            move_number=32,
        )
        games = play_self_games_batched(
            evaluator,
            SelfPlayConfig(
                mcts=MCTSConfig(
                    simulations=2,
                    add_root_noise=False,
                    seed=9,
                ),
                mirror_augmentation=False,
            ),
            2,
            initial_state=initial,
            search_factory=NativeMCTS,
        )

        self.assertEqual(len(games), 2)
        self.assertEqual([game.termination_reason for game in games], ["goal", "goal"])
        self.assertEqual([game.move_count for game in games], [1, 1])
        self.assertTrue(all(len(game.samples) == 1 for game in games))


if __name__ == "__main__":
    unittest.main()
