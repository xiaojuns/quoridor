"""Independent real-rules oracles for the versioned reference PUCT search."""

import math
import random

from hypothesis import given, settings, strategies as st
import pytest

from quoridor import engine
from quoridor.episode import advance_episode, fail_episode, interrupt_episode, start_episode
from quoridor.project_runtime import Deadline
from quoridor.puct import Evaluation, InvalidEvaluationError, SearchConfig, SearchError, run_search


def advance_points(episode, points):
    for point in points:
        episode = advance_episode(episode, engine.encode_move(*point))
    return episode


class Uniform:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = []

    def __call__(self, episode, *, deadline):
        assert not episode.finished, "completed/stopped episodes must not reach inference"
        assert isinstance(deadline, Deadline)
        self.calls.append(episode)
        return Evaluation((0.0,) * 209, self.value)


class Scripted(Uniform):
    """Independent trajectory oracle; no search statistics inform its answers."""

    def __init__(self, root, points, *, final_value=0.0):
        super().__init__()
        self.start_ply = root.plies
        self.actions = tuple(engine.encode_move(*point) for point in points)
        self.final_value = final_value

    def __call__(self, episode, *, deadline):
        super().__call__(episode, deadline=deadline)
        depth = episode.plies - self.start_ply
        logits = [-500.0] * 209
        if depth < len(self.actions):
            action = self.actions[depth]
            assert action in engine.legal_actions(episode.state)
            logits[action] = 500.0
        value = self.final_value if depth == len(self.actions) else 0.0
        return Evaluation(tuple(logits), value)


def search(episode, evaluator=None, *, simulations=1, **options):
    return run_search(
        episode, evaluator if evaluator is not None else Uniform(),
        SearchConfig(simulations=simulations, **options), deadline=Deadline(60),
    )


def node_for(result, episode):
    matches = [node for node in result.nodes if node.adjudication_key == episode.adjudication_key]
    assert len(matches) == 1
    return matches[0]


def assert_legal_distribution(result, episode, simulations):
    legal = set(engine.legal_actions(episode.state))
    assert result.status == "completed"
    assert result.search_version == "puct-v1"
    assert result.policy_target_version == "root-visits-v1"
    assert result.completed_simulations == simulations
    assert sum(result.visits) == simulations
    assert len(result.visits) == len(result.policy) == len(result.priors) == len(result.value_sums) == 209
    assert result.action in legal
    assert math.fsum(result.policy) == pytest.approx(1.0)
    assert math.fsum(result.priors) == pytest.approx(1.0)
    assert set(node_for(result, episode).legal_actions) == legal
    for action in range(209):
        assert result.policy[action] == pytest.approx(result.visits[action] / simulations)
        assert result.visits[action] >= 0
        if action not in legal:
            assert result.priors[action] == result.policy[action] == result.visits[action] == result.value_sums[action] == 0


def test_initial_expansion_includes_all_128_walls_and_masks_illegal_logits():
    episode = start_episode()
    legal = set(engine.legal_actions(episode.state))

    def evaluator(position, *, deadline):
        return Evaluation(tuple(0.0 if action in legal else 1000.0 for action in range(209)), 0.0)

    result = search(episode, evaluator)
    assert len(legal) == 131 and len([action for action in legal if action >= 81]) == 128
    assert_legal_distribution(result, episode, 1)
    for action in legal:
        assert result.priors[action] == pytest.approx(1 / 131)
    assert result.action == min(legal)
    assert result.evaluations == 2  # Root expansion is not a simulation.


@pytest.mark.parametrize("to_play,pawns,winning_point", [
    (0, ((1, 2), (4, 7)), (0, 2)),
    (1, ((4, 2), (7, 7)), (8, 7)),
])
def test_uniform_evaluator_discovers_immediate_win_from_either_side(to_play, pawns, winning_point):
    episode = start_episode(engine.BoardState(pawns=pawns, to_play=to_play))
    # This independent one-ply oracle inspects real rule transitions, never Q.
    wins = {
        action for action in engine.legal_actions(episode.state)
        if engine.goal_winner(engine.apply_action(episode.state, action)) == to_play
    }
    assert wins == {engine.encode_move(*winning_point)}
    evaluator = Uniform()
    result = search(episode, evaluator, simulations=32)
    assert result.action in wins
    assert result.value_sums[result.action] == result.visits[result.action]
    assert result.terminal_leaves > 0
    assert all(not position.finished for position in evaluator.calls)
    assert_legal_distribution(result, episode, 32)


@pytest.mark.parametrize("pawns,points,expected_sums", [
    (((4, 1), (7, 7)), ((3, 1), (8, 7)), (-1.0, 1.0)),
    (((2, 1), (6, 7)), ((1, 1), (7, 7), (0, 1)), (1.0, -1.0, 1.0)),
])
def test_real_goal_oracle_backs_up_odd_and_even_paths(pawns, points, expected_sums):
    episode = start_episode(engine.BoardState(pawns=pawns))
    evaluator = Scripted(episode, points)
    result = search(episode, evaluator, simulations=len(points))
    assert result.status == "completed" and result.terminal_leaves == 1
    assert result.evaluations == len(points)
    for depth, (point, expected) in enumerate(zip(points, expected_sums)):
        action = engine.encode_move(*point)
        node = node_for(result, episode)
        assert node.visits[action] == len(points) - depth
        assert node.value_sums[action] == expected
        episode = advance_episode(episode, action)
    assert episode.outcome.reason == "goal"
    assert episode.outcome.value_for(episode.state.to_play) == -1.0


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("leaf_value", [-0.75, 0.75])
def test_nonterminal_leaf_oracle_checks_each_edge_sign_at_all_depths(depth, leaf_value):
    root = start_episode()
    points = ((7, 4), (1, 4), (6, 4), (2, 4))[:depth]
    evaluator = Scripted(root, points, final_value=leaf_value)
    result = search(root, evaluator, simulations=depth)
    assert result.status == "completed"
    assert result.terminal_leaves == result.cycle_cutoffs == 0
    assert result.evaluations == depth + 1
    episode = root
    for edge_depth, point in enumerate(points):
        action = engine.encode_move(*point)
        node = node_for(result, episode)
        expected_sum = leaf_value if (depth - edge_depth) % 2 == 0 else -leaf_value
        assert node.visits[action] == depth - edge_depth
        assert node.value_sums[action] == pytest.approx(expected_sum)
        episode = advance_episode(episode, action)
    assert not episode.finished


def test_raw_visit_target_does_not_become_one_hot_at_zero_temperature():
    episode = start_episode()
    result = search(episode, simulations=4)
    first_four = sorted(engine.legal_actions(episode.state))[:4]
    assert result.action == first_four[0]
    assert [result.visits[action] for action in first_four] == [1] * 4
    assert [result.policy[action] for action in first_four] == [0.25] * 4
    assert sum(probability > 0 for probability in result.policy) == 4
    stochastic = search(episode, simulations=4, mode="selfplay", temperature=1.0)
    assert stochastic.visits == result.visits and stochastic.policy == result.policy


def test_first_selection_uses_prior_and_remaining_actions_remain_expandable():
    episode = start_episode()
    preferred = max(engine.legal_actions(episode.state))

    def evaluator(position, *, deadline):
        return Evaluation(tuple(2.0 if action == preferred else 0.0 for action in range(209)), 0.0)

    result = search(episode, evaluator)
    assert result.action == preferred
    assert result.visits[preferred] == 1
    assert all(result.priors[action] > 0 for action in engine.legal_actions(episode.state))


def test_second_selection_uses_specified_square_root_not_legacy_plus_one():
    episode = start_episode()
    first, second = engine.encode_move(7, 4), engine.encode_move(8, 3)

    def evaluator(position, *, deadline):
        logits = [-1000.0] * 209
        logits[first], logits[second] = math.log(0.6), math.log(0.4)
        return Evaluation(tuple(logits), -0.18 if position.plies else 0.0)

    result = search(episode, evaluator, simulations=2)
    # After simulation one, Q(first)=+.18 and N(first)=1. The specified
    # scores are .18+1.5*.6/2=.63 and 1.5*.4=.60: choose first again.
    # Legacy sqrt(N+1) instead yields .8164 versus .8485 and chooses second.
    assert result.visits[first] == 2 and result.visits[second] == 0


def test_path_cycle_is_a_search_approximation_not_a_completed_draw():
    episode = start_episode()
    points = ((8, 3), (0, 3), (8, 4), (0, 4))
    actual = advance_points(episode, points)
    assert actual.repetition_count == 2 and actual.outcome is None
    result = search(episode, Scripted(episode, points), simulations=4)
    assert result.status == "completed" and result.cycle_cutoffs == 1
    assert result.terminal_leaves == 0 and result.evaluations == 4
    assert result.value_sums[engine.encode_move(8, 3)] == 0
    assert episode.history == (engine.position_key(episode.state),)
    assert episode.outcome is None and episode.value_targets([0]) is None


def test_full_history_distinguishes_real_repetition_from_identical_board_local_cycle():
    left = advance_points(start_episode(), ((8, 3), (0, 3), (8, 4), (0, 4)))
    right = advance_points(start_episode(), ((8, 5), (0, 5), (8, 4), (0, 4)))
    assert left.state == right.state and left.repetition_count == right.repetition_count == 2
    assert left.adjudication_key != right.adjudication_key
    points = ((8, 3), (0, 3), (8, 2), (0, 4), (8, 3))
    left_end, right_end = advance_points(left, points), advance_points(right, points)
    assert left_end.state == right_end.state
    assert left_end.outcome.reason == "repetition" and right_end.outcome is None
    first = search(left, Scripted(left, points), simulations=5)
    second = search(right, Scripted(right, points), simulations=5)
    # The final board is also a search-path ancestor. Exact third repetition
    # wins precedence in one history; only the other receives approximation.
    assert first.terminal_leaves == 1 and first.cycle_cutoffs == 0
    assert second.terminal_leaves == 0 and second.cycle_cutoffs == 1
    assert first.evaluations == second.evaluations == 5
    assert first.completed_simulations == second.completed_simulations == 5
    assert node_for(first, left).adjudication_key != node_for(second, right).adjudication_key
    assert all(node.adjudication_key[0][:len(left.history)] == left.history for node in first.nodes)
    assert all(node.adjudication_key[0][:len(right.history)] == right.history for node in second.nodes)


@pytest.mark.parametrize("ply_cap", [8, 256])
def test_true_third_repetition_leaf_is_exact_without_evaluation(ply_cap):
    cycle = ((8, 3), (0, 3), (8, 4), (0, 4))
    episode = advance_points(start_episode(max_game_plies=ply_cap), (*cycle, *cycle[:3]))
    result = search(episode, Scripted(episode, [cycle[-1]]))
    assert result.terminal_leaves == 1 and result.cycle_cutoffs == 0
    assert result.evaluations == 1 and result.completed_simulations == 1
    assert result.value_sums[engine.encode_move(*cycle[-1])] == 0


@pytest.mark.parametrize("kind", ["goal", "repetition", "ply_limit", "interrupted", "failed"])
def test_finished_roots_never_infer_or_offer_an_action(kind):
    if kind == "goal":
        episode = advance_points(start_episode(engine.BoardState(pawns=((1, 2), (4, 7)))), [(0, 2)])
    elif kind == "repetition":
        episode = advance_points(start_episode(), [(8, 3), (0, 3), (8, 4), (0, 4)] * 2)
    elif kind == "ply_limit":
        episode = advance_points(start_episode(max_game_plies=1), [(7, 4)])
    elif kind == "interrupted":
        episode = interrupt_episode(start_episode())
    else:
        episode = fail_episode(start_episode(), "numerical_error", detail="oracle")
    evaluator = Uniform()
    result = search(episode, evaluator)
    assert result.status == ("terminal" if episode.outcome.terminated else "root_stopped")
    assert result.root_value == episode.outcome.value_for(episode.state.to_play)
    assert result.action is None and result.completed_simulations == 0
    assert sum(result.visits) == sum(result.policy) == 0
    assert result.evaluations == 0 and evaluator.calls == []


def test_ply_limit_leaf_aborts_without_fabricating_a_value_or_draw():
    episode = start_episode(max_game_plies=1)
    evaluator = Uniform(value=0.75)
    result = search(episode, evaluator, simulations=8)
    assert result.status == "simulation_truncated" and result.action is None
    assert result.completed_simulations == sum(result.visits) == 0
    assert sum(result.policy) == sum(result.value_sums) == 0
    assert result.truncated_leaves == 1 and result.terminal_leaves == result.cycle_cutoffs == 0
    assert result.evaluations == 1 and len(evaluator.calls) == 1
    assert episode.outcome is None


def test_later_truncated_simulation_preserves_only_previously_completed_visits():
    episode = start_episode(max_game_plies=2)
    points = ((7, 4), (1, 4))
    result = search(episode, Scripted(episode, points), simulations=4)
    assert result.status == "simulation_truncated" and result.action is None
    assert result.completed_simulations == sum(result.visits) == 1
    assert result.visits[engine.encode_move(7, 4)] == 1
    assert result.policy[engine.encode_move(7, 4)] == 1.0
    assert result.truncated_leaves == 1 and result.terminal_leaves == 0
    assert result.evaluations == 2
    child = advance_points(episode, points[:1])
    assert sum(node_for(result, child).visits) == 0


def test_goal_at_ply_cap_keeps_exact_terminal_value():
    episode = start_episode(engine.BoardState(pawns=((1, 2), (4, 7))), max_game_plies=1)
    result = search(episode, Scripted(episode, [(0, 2)]), simulations=3)
    action = engine.encode_move(0, 2)
    assert result.status == "completed" and result.action == action
    assert result.terminal_leaves == 3 and result.truncated_leaves == 0
    assert result.value_sums[action] == result.visits[action] == 3


def test_selfplay_noise_is_seeded_and_evaluation_is_seed_independent():
    episode = start_episode()
    options = dict(simulations=5, mode="selfplay", root_noise=True, temperature=1.0)
    a = search(episode, seed=17, **options)
    b = search(episode, seed=17, **options)
    c = search(episode, seed=18, **options)
    assert (a.priors, a.visits, a.policy, a.action) == (b.priors, b.visits, b.policy, b.action)
    assert a.priors != c.priors
    assert_legal_distribution(a, episode, 5)
    plain_a = search(episode, seed=1, simulations=5)
    plain_b = search(episode, seed=2, simulations=5)
    assert (plain_a.priors, plain_a.visits, plain_a.policy, plain_a.action) == (plain_b.priors, plain_b.visits, plain_b.policy, plain_b.action)


def test_noise_is_root_only_and_search_preserves_process_random_state():
    episode = start_episode()
    before = random.getstate()
    result = search(episode, simulations=4, mode="selfplay", root_noise=True, temperature=1.0)
    assert random.getstate() == before
    children = [node for node in result.nodes if node.adjudication_key != episode.adjudication_key]
    assert children
    for node in children:
        for action in node.legal_actions:
            assert node.priors[action] == pytest.approx(1 / len(node.legal_actions))


def test_extreme_finite_dirichlet_normalization_fails_explicitly(monkeypatch):
    original_random = random.Random.random
    draws = 0

    def bounded_random(rng):
        nonlocal draws
        draws += 1
        assert draws < 1000, "regression guard: gamma rejection sampling failed to stop"
        return original_random(rng)

    monkeypatch.setattr(random.Random, "random", bounded_random)
    with pytest.raises(SearchError, match="Dirichlet.*normalization"):
        search(start_episode(), mode="selfplay", root_noise=True, dirichlet_alpha=1e307)
    assert draws > 0


@pytest.mark.parametrize("options", [
    {"root_noise": True}, {"temperature": 1.0}, {"simulations": 0},
    {"simulations": True}, {"c_puct": -1.0}, {"c_puct": math.nan},
    {"temperature": math.inf}, {"temperature": -1.0}, {"mode": "arena-ish"},
    {"dirichlet_alpha": 0.0}, {"dirichlet_fraction": 1.01}, {"seed": True},
])
def test_invalid_or_stochastic_evaluation_configuration_fails_explicitly(options):
    arguments = {"simulations": 1, **options}
    with pytest.raises((TypeError, ValueError)):
        SearchConfig(**arguments)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("action", [0, 67])
def test_nonfinite_logits_fail_even_on_illegal_actions(bad, action):
    logits = [0.0] * 209
    logits[action] = bad

    def evaluator(episode, *, deadline):
        return Evaluation(tuple(logits), 0.0)

    with pytest.raises(InvalidEvaluationError):
        search(start_episode(), evaluator)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.01, 1.01])
def test_invalid_values_are_not_clipped_or_replaced(value):
    with pytest.raises(InvalidEvaluationError):
        search(start_episode(), Uniform(value=value))


@pytest.mark.parametrize("size", [0, 208, 210])
def test_wrong_policy_shape_fails(size):
    def evaluator(episode, *, deadline):
        return Evaluation((0.0,) * size, 0.0)

    with pytest.raises(InvalidEvaluationError):
        search(start_episode(), evaluator)


@pytest.mark.parametrize("response", [None, ((0.0,) * 209, 0.0), Evaluation([0.0] * 209, 0.0)])
def test_malformed_evaluator_responses_cannot_supply_fallback_priors(response):
    with pytest.raises(InvalidEvaluationError):
        search(start_episode(), lambda episode, *, deadline: response)


def test_evaluator_exception_is_an_explicit_search_failure():
    def evaluator(episode, *, deadline):
        raise RuntimeError("oracle inference failed")

    with pytest.raises(SearchError, match="oracle inference failed"):
        search(start_episode(), evaluator)


@settings(max_examples=12, derandomize=True, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=1000), max_size=4), st.integers(1, 4))
def test_reachable_state_legal_support_counts_and_input_isolation(choices, simulations):
    episode = start_episode()
    for choice in choices:
        actions = engine.legal_actions(episode.state)
        episode = advance_episode(episode, actions[choice % len(actions)])
    original = (episode.state, episode.history, episode.outcome)
    result = search(episode, simulations=simulations)
    assert_legal_distribution(result, episode, simulations)
    assert (episode.state, episode.history, episode.outcome) == original


def test_search_requires_new_episode_not_legacy_or_board_only():
    from quoridor.game import create_initial_state

    for invalid in (engine.initial_state(), create_initial_state(), None):
        with pytest.raises((TypeError, ValueError)):
            search(invalid)
