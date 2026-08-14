from __future__ import annotations

import unittest

from quoridor.arena import ArenaConfig, evaluate_candidate
from quoridor.codec import ACTION_SPACE_SIZE, decode_action, encode_move
from quoridor.game import State, apply_action_index, legal_action_indices
from quoridor.mcts import MCTSConfig, SearchResult
from quoridor.selfplay import (
    SelfPlayConfig,
    play_self_game,
    play_self_games_scheduled,
    scheduled_search_seed,
)


class ForwardEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, state):
        self.calls += 1
        logits = [-20.0] * ACTION_SPACE_SIZE
        legal_moves = []
        for action_index in legal_action_indices(state):
            action = decode_action(action_index)
            if action["type"] == "move":
                legal_moves.append((action["row"], action["col"], action_index))
        _, _, preferred = min(legal_moves)
        logits[preferred] = 20.0
        return logits, 0.0


def _search_result(action_index: int) -> SearchResult:
    policy = [0.0] * ACTION_SPACE_SIZE
    visits = [0] * ACTION_SPACE_SIZE
    policy[action_index] = 1.0
    visits[action_index] = 1
    return SearchResult(
        action_index=action_index,
        policy=tuple(policy),
        visit_counts=tuple(visits),
        root_value=0.0,
        root_edges={},
    )


class SessionSchedulerSearch:
    def __init__(self, _evaluator, config: MCTSConfig) -> None:
        self.seed = config.seed
        self.next_session_id = 1
        self.states: dict[int, State] = {}
        self.calls: dict[int, int] = {}
        self.run_batches: list[tuple[int, ...]] = []
        self.advanced: list[int] = []
        self.released: list[int] = []
        self.max_open_sessions = 0

    def create_session(self, state: State) -> int:
        session_id = self.next_session_id
        self.next_session_id += 1
        self.states[session_id] = state
        self.calls[session_id] = 0
        self.max_open_sessions = max(self.max_open_sessions, len(self.states))
        return session_id

    def run_sessions(
        self,
        session_ids,
        states,
        *,
        temperatures,
        add_root_noise,
    ):
        del temperatures, add_root_noise
        self.run_batches.append(tuple(session_ids))
        results = []
        for session_id, state in zip(session_ids, states):
            if self.states[session_id] != state:
                raise AssertionError("scheduler passed a stale session state")
            call = self.calls[session_id]
            self.calls[session_id] = call + 1
            if session_id % 2 == 1:
                action_index = encode_move(0, 0)
            elif state.current_player == 0:
                action_index = encode_move(1, 1)
            else:
                action_index = encode_move(8, 8)
            results.append(_search_result(action_index))
        return tuple(results)

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: State,
    ) -> bool:
        expected = apply_action_index(self.states[session_id], action_index)
        if expected != next_state:
            raise AssertionError("scheduler advanced with the wrong state")
        self.states[session_id] = next_state
        self.advanced.append(session_id)
        return True

    def release_session(self, session_id: int) -> bool:
        self.released.append(session_id)
        return self.states.pop(session_id, None) is not None


class FailingSessionSchedulerSearch(SessionSchedulerSearch):
    def run_sessions(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected search failure")


class LegacyBatchSearch:
    def __init__(self, _evaluator, config: MCTSConfig) -> None:
        self.seed = config.seed
        self.batch_sizes: list[int] = []
        self.calls = 0

    def run_batch(
        self,
        states,
        *,
        temperatures,
        add_root_noise,
    ):
        del temperatures, add_root_noise
        self.batch_sizes.append(len(states))
        results = []
        for index, state in enumerate(states):
            if state.current_player == 1:
                action_index = encode_move(8, 8)
            elif self.calls == 0 and index == 1:
                action_index = encode_move(1, 1)
            else:
                action_index = encode_move(0, 0)
            results.append(_search_result(action_index))
        self.calls += 1
        return tuple(results)


class SelfPlayArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_config = MCTSConfig(
            simulations=2,
            add_root_noise=False,
            temperature=0.0,
            seed=4,
        )

    def test_self_play_finishes_and_attaches_zero_sum_targets(self) -> None:
        evaluator = ForwardEvaluator()
        game = play_self_game(
            evaluator,
            SelfPlayConfig(
                mcts=self.search_config,
                temperature_moves=0,
                mirror_augmentation=True,
                seed=9,
            ),
        )

        self.assertEqual(game.termination_reason, "goal")
        self.assertGreater(game.move_count, 0)
        self.assertEqual(len(game.samples), game.move_count * 2)
        self.assertEqual({sample.value for sample in game.samples}, {-1.0, 1.0})
        for sample in game.samples:
            self.assertEqual(tuple(sample.policy.shape), (ACTION_SPACE_SIZE,))
            self.assertAlmostEqual(float(sample.policy.sum()), 1.0, places=6)

    def test_arena_swaps_candidate_sides_and_has_no_draw_bucket(self) -> None:
        result = evaluate_candidate(
            ForwardEvaluator(),
            ForwardEvaluator(),
            mcts_config=self.search_config,
            arena_config=ArenaConfig(
                games=2,
                simulations=2,
                acceptance_rate=0.5,
                seed=3,
            ),
        )

        self.assertEqual(result.candidate_wins + result.champion_wins, 2)
        self.assertEqual(result.games, 2)
        self.assertTrue(0 <= result.candidate_win_rate <= 1)

    def test_dynamic_session_scheduler_refills_and_preserves_game_order(self) -> None:
        created: list[SessionSchedulerSearch] = []

        def factory(evaluator, config):
            search = SessionSchedulerSearch(evaluator, config)
            created.append(search)
            return search

        initial = State(
            pawns=((1, 0), (7, 8)),
            walls_remaining=(0, 0),
            current_player=0,
            move_number=32,
        )
        config = SelfPlayConfig(
            mcts=MCTSConfig(
                simulations=1,
                temperature=0.0,
                add_root_noise=False,
                seed=5,
            ),
            temperature_moves=0,
            mirror_augmentation=False,
            seed=101,
            collect_samples=False,
            collect_raw_records=True,
        )
        progress: list[tuple[int, int]] = []
        streamed: list[tuple[int, int]] = []
        games = play_self_games_scheduled(
            ForwardEvaluator(),
            config,
            4,
            actors=2,
            initial_state=initial,
            search_factory=factory,
            progress_callback=lambda moves, completed: progress.append(
                (moves, completed)
            ),
            completed_game_callback=lambda index, game: streamed.append(
                (index, game.move_count)
            ),
        )

        self.assertEqual([game.move_count for game in games], [1, 2, 1, 2])
        self.assertEqual([game.winner for game in games], [0, 1, 0, 1])
        self.assertTrue(all(not game.samples for game in games))
        self.assertTrue(
            all(len(game.raw_plies) == game.move_count for game in games)
        )
        self.assertEqual(streamed, [(0, 1), (1, 2), (2, 1), (3, 2)])
        self.assertEqual(progress[0], (0, 0))
        self.assertEqual(progress[-1], (6, 4))
        search = created[0]
        self.assertEqual(search.run_batches, [(1, 2), (2, 3), (4,), (4,)])
        self.assertEqual(search.max_open_sessions, 2)
        self.assertEqual(search.seed, scheduled_search_seed(config.seed))
        self.assertEqual(search.released, [1, 2, 3, 4])
        self.assertEqual(search.states, {})
        self.assertEqual(len(search.advanced), sum(game.move_count for game in games))

        repeated: list[SessionSchedulerSearch] = []

        def repeated_factory(evaluator, search_config):
            search = SessionSchedulerSearch(evaluator, search_config)
            repeated.append(search)
            return search

        repeated_games = play_self_games_scheduled(
            ForwardEvaluator(),
            config,
            4,
            actors=2,
            initial_state=initial,
            search_factory=repeated_factory,
        )
        self.assertEqual(
            [(game.winner, game.move_count) for game in repeated_games],
            [(game.winner, game.move_count) for game in games],
        )
        self.assertEqual(repeated[0].run_batches, search.run_batches)
        self.assertEqual(repeated[0].seed, search.seed)

    def test_dynamic_scheduler_releases_sessions_after_search_failure(self) -> None:
        created: list[FailingSessionSchedulerSearch] = []

        def factory(evaluator, config):
            search = FailingSessionSchedulerSearch(evaluator, config)
            created.append(search)
            return search

        initial = State(
            pawns=((1, 0), (7, 8)),
            walls_remaining=(0, 0),
            move_number=32,
        )
        with self.assertRaisesRegex(RuntimeError, "injected search failure"):
            play_self_games_scheduled(
                ForwardEvaluator(),
                SelfPlayConfig(
                    mcts=self.search_config,
                    mirror_augmentation=False,
                    seed=103,
                ),
                3,
                actors=2,
                initial_state=initial,
                search_factory=factory,
            )
        self.assertEqual(created[0].released, [1, 2])
        self.assertEqual(created[0].states, {})

    def test_dynamic_scheduler_keeps_legacy_run_batch_compatibility(self) -> None:
        created: list[LegacyBatchSearch] = []

        def factory(evaluator, config):
            search = LegacyBatchSearch(evaluator, config)
            created.append(search)
            return search

        initial = State(
            pawns=((1, 0), (7, 8)),
            walls_remaining=(0, 0),
            move_number=32,
        )
        games = play_self_games_scheduled(
            ForwardEvaluator(),
            SelfPlayConfig(
                mcts=self.search_config,
                temperature_moves=0,
                mirror_augmentation=False,
                seed=107,
            ),
            3,
            actors=2,
            initial_state=initial,
            search_factory=factory,
        )
        self.assertEqual([game.move_count for game in games], [1, 2, 1])
        self.assertEqual(created[0].batch_sizes, [2, 2])


if __name__ == "__main__":
    unittest.main()
