from __future__ import annotations

import unittest

from quoridor.codec import encode_move, to_canonical_action_index
from quoridor.game import (
    State,
    apply_action_index,
    create_initial_state,
    is_terminal,
    legal_action_indices,
)
from quoridor.mcts import ACTION_SIZE, MCTS, MCTSConfig, run


class MockEvaluator:
    def __init__(
        self,
        *,
        preferred_action: int | None = None,
        values: tuple[float, ...] = (0.0,),
    ) -> None:
        self.preferred_action = preferred_action
        self.values = values
        self.calls: list[State] = []

    def evaluate(self, state: State) -> tuple[list[float], float]:
        call_index = len(self.calls)
        self.calls.append(state)
        logits = [0.0] * ACTION_SIZE
        if self.preferred_action is not None:
            logits = [-100.0] * ACTION_SIZE
            logits[self.preferred_action] = 100.0
        value = self.values[min(call_index, len(self.values) - 1)]
        return logits, value


def config(
    simulations: int,
    *,
    seed: int = 7,
    noise: bool = False,
    temperature: float = 0.0,
) -> MCTSConfig:
    return MCTSConfig(
        simulations=simulations,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.12,
        temperature=temperature,
        add_root_noise=noise,
        seed=seed,
    )


class MCTSTests(unittest.TestCase):
    def test_expands_every_legal_action_and_masks_illegal_logits(self) -> None:
        state = create_initial_state()
        legal = set(legal_action_indices(state))
        self.assertNotIn(0, legal)
        evaluator = MockEvaluator(preferred_action=0)

        result = MCTS(evaluator, config(1)).run(state)

        self.assertEqual(set(result.root_edges), legal)
        self.assertEqual(len(result.root_edges), 131)
        self.assertEqual(sum(action >= 81 for action in result.root_edges), 128)
        self.assertIn(result.action_index, legal)
        self.assertEqual(result.policy[0], 0.0)
        self.assertEqual(sum(result.policy), 1.0)
        self.assertAlmostEqual(
            sum(edge.prior for edge in result.root_edges.values()),
            1.0,
        )

    def test_leaf_value_changes_sign_when_backed_to_parent(self) -> None:
        state = State(walls_remaining=(0, 0))
        action = encode_move(7, 4)
        evaluator = MockEvaluator(preferred_action=action, values=(0.0, 0.75))

        result = MCTS(evaluator, config(1)).run(state)

        edge = result.root_edges[action]
        self.assertEqual(edge.N, 1)
        self.assertAlmostEqual(edge.W, -0.75)
        self.assertAlmostEqual(edge.Q, -0.75)
        self.assertAlmostEqual(result.root_value, -0.75)
        self.assertEqual(len(evaluator.calls), 2)

    def test_player_one_canonical_policy_finds_immediate_win(self) -> None:
        state = State(
            pawns=((8, 0), (7, 4)),
            walls_remaining=(0, 0),
            current_player=1,
        )
        winning_action = encode_move(8, 4)
        canonical_action = to_canonical_action_index(winning_action, 1)
        evaluator = MockEvaluator(preferred_action=canonical_action)

        result = MCTS(evaluator, config(4)).run(state)

        self.assertEqual(result.action_index, winning_action)
        self.assertEqual(result.root_edges[winning_action].N, 4)
        self.assertEqual(result.root_edges[winning_action].Q, 1.0)
        self.assertEqual(result.root_value, 1.0)
        self.assertEqual(len(evaluator.calls), 1)

        ended = apply_action_index(state, winning_action)
        self.assertTrue(is_terminal(ended))

    def test_terminal_root_never_calls_evaluator(self) -> None:
        state = State(
            pawns=((1, 4), (0, 0)),
            walls_remaining=(0, 0),
            current_player=0,
        )
        ended = apply_action_index(state, encode_move(0, 4))
        evaluator = MockEvaluator()

        result = MCTS(evaluator, config(8, noise=True)).run(ended)

        self.assertIsNone(result.action_index)
        self.assertEqual(result.root_value, -1.0)
        self.assertEqual(sum(result.policy), 0.0)
        self.assertEqual(sum(result.visit_counts), 0)
        self.assertEqual(result.root_edges, {})
        self.assertEqual(evaluator.calls, [])

    def test_visit_policy_and_dirichlet_noise_are_seed_deterministic(self) -> None:
        state = State(walls_remaining=(0, 0))
        settings = config(24, seed=91, noise=True, temperature=1.0)

        first = MCTS(MockEvaluator(), settings).run(state)
        second = MCTS(MockEvaluator(), settings).run(state)

        self.assertEqual(first.visit_counts, second.visit_counts)
        self.assertEqual(first.policy, second.policy)
        self.assertEqual(first.action_index, second.action_index)
        self.assertEqual(
            tuple(edge.prior for edge in first.root_edges.values()),
            tuple(edge.prior for edge in second.root_edges.values()),
        )
        self.assertAlmostEqual(sum(first.policy), 1.0)
        total_visits = sum(first.visit_counts)
        self.assertEqual(total_visits, settings.simulations)
        for action, visits in enumerate(first.visit_counts):
            self.assertAlmostEqual(first.policy[action], visits / total_visits)

        different = MCTS(
            MockEvaluator(),
            config(24, seed=92, noise=True, temperature=1.0),
        ).run(state)
        self.assertNotEqual(
            tuple(edge.prior for edge in first.root_edges.values()),
            tuple(edge.prior for edge in different.root_edges.values()),
        )

    def test_module_run_returns_a_self_play_ready_209_policy(self) -> None:
        state = State(walls_remaining=(0, 0))
        result = run(
            state,
            MockEvaluator(),
            config(3, temperature=1.0),
            add_root_noise=False,
        )

        self.assertEqual(len(result.policy), ACTION_SIZE)
        self.assertEqual(len(result.visit_counts), ACTION_SIZE)
        self.assertIn(result.action_index, legal_action_indices(state))


if __name__ == "__main__":
    unittest.main()
