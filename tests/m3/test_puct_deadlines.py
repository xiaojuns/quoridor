"""Fake-clock evidence for operation boundaries and truthful partial results."""

import random

import pytest

from quoridor import engine, puct
from quoridor.episode import advance_episode, start_episode
from quoridor.project_runtime import Deadline
from quoridor.puct import Evaluation, SearchConfig, run_search


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def assert_aborted(result, *, completed, elapsed):
    assert result.status == "budget_exhausted"
    assert result.action is None
    assert result.completed_simulations == sum(result.visits) == completed
    assert result.elapsed_seconds == pytest.approx(elapsed)
    assert result.budget_seconds == 1.0
    assert result.overrun_seconds == pytest.approx(max(0, elapsed - 1.0))
    if not completed:
        assert sum(result.policy) == sum(result.value_sums) == 0


@pytest.mark.parametrize("elapsed", [1.0, 1.25])
def test_already_exhausted_root_does_no_inference_or_transition(monkeypatch, elapsed):
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    clock.now = elapsed

    def forbidden(*args, **kwargs):
        pytest.fail("no new operation may start after the deadline")

    monkeypatch.setattr(puct, "advance_episode", forbidden)
    monkeypatch.setattr(engine, "legal_actions", forbidden)
    result = run_search(start_episode(), forbidden, SearchConfig(simulations=5), deadline=deadline)
    assert_aborted(result, completed=0, elapsed=elapsed)
    assert result.evaluations == 0 and result.root_value is None


@pytest.mark.parametrize("kind", ["goal", "ply_limit"])
def test_completed_or_stopped_root_precedes_expired_deadline(kind):
    if kind == "goal":
        root = start_episode(engine.BoardState(pawns=((1, 2), (4, 7))))
        episode = advance_episode(root, engine.encode_move(0, 2))
    else:
        episode = advance_episode(start_episode(max_game_plies=1), engine.encode_move(7, 4))
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    clock.now = 1.25

    def forbidden(*args, **kwargs):
        pytest.fail("stopped roots must not infer")

    result = run_search(episode, forbidden, SearchConfig(simulations=2), deadline=deadline)
    assert result.status == ("terminal" if kind == "goal" else "root_stopped")
    assert result.root_value == (-1.0 if kind == "goal" else None)
    assert result.action is None and result.evaluations == result.completed_simulations == 0
    assert result.elapsed_seconds == 1.25 and result.overrun_seconds == 0.25


@pytest.mark.parametrize("expired_on_call", [1, 2, 3])
def test_evaluator_overrun_discards_only_unfinished_simulation(expired_on_call):
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    episode = start_episode()
    calls = []
    preferred = (engine.encode_move(7, 4), engine.encode_move(1, 4))

    def evaluator(position, *, deadline):
        assert deadline is expected_deadline
        assert clock.now < 1
        calls.append(position)
        logits = [-500.0] * 209
        logits[preferred[min(position.plies, 1)]] = 500.0
        if len(calls) == expired_on_call:
            clock.now = 1.25
        return Evaluation(tuple(logits), 0.25)

    expected_deadline = deadline
    result = run_search(episode, evaluator, SearchConfig(simulations=5), deadline=deadline)
    completed = max(0, expired_on_call - 2)
    assert_aborted(result, completed=completed, elapsed=1.25)
    assert result.evaluations == len(calls) == expired_on_call
    if completed:
        assert result.visits[preferred[0]] == 1
        assert result.value_sums[preferred[0]] == -0.25
        assert result.policy[preferred[0]] == 1.0
    assert not episode.finished and episode.plies == 0


@pytest.mark.parametrize("goal", [False, True])
def test_transition_overrun_starts_no_further_evaluation_or_backup(monkeypatch, goal):
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    episode = start_episode(engine.BoardState(pawns=((1, 2), (4, 7)))) if goal else start_episode()
    selected = engine.encode_move(0, 2) if goal else engine.encode_move(7, 4)
    calls = []

    def evaluator(position, *, deadline):
        calls.append(position)
        assert len(calls) == 1
        logits = [-500.0] * 209
        logits[selected] = 500.0
        return Evaluation(tuple(logits), 0.0)

    actual_transition = puct.advance_episode
    transitions = []

    def delayed_transition(*args, **kwargs):
        assert clock.now < 1
        result = actual_transition(*args, **kwargs)
        transitions.append(result)
        clock.now = 1.25
        return result

    monkeypatch.setattr(puct, "advance_episode", delayed_transition)
    result = run_search(episode, evaluator, SearchConfig(simulations=3), deadline=deadline)
    assert_aborted(result, completed=0, elapsed=1.25)
    assert len(calls) == result.evaluations == len(transitions) == 1
    if goal:
        assert transitions[0].outcome.reason == "goal"
        assert transitions[0].outcome.value_for(transitions[0].state.to_play) == -1
    assert episode.plies == 0 and episode.outcome is None


def test_exact_deadline_during_child_evaluation_is_exhausted():
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    calls = 0

    def evaluator(episode, *, deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            clock.now = 1.0
        return Evaluation((0.0,) * 209, 0.75)

    result = run_search(start_episode(), evaluator, SearchConfig(simulations=1), deadline=deadline)
    assert_aborted(result, completed=0, elapsed=1.0)
    assert calls == result.evaluations == 2


def test_deadline_is_checked_inside_gamma_rejection_sampling(monkeypatch):
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    draws = calls = 0
    original_random = random.Random.random

    def timed_random(rng):
        nonlocal draws
        draws += 1
        # This independent guard prevents the test itself from hanging if a
        # regression removes deadline checks from the rejection sampler.
        assert draws < 100, "gamma kept drawing after its deadline"
        clock.now = draws * 0.25
        return original_random(rng)

    def evaluator(episode, *, deadline):
        nonlocal calls
        calls += 1
        return Evaluation((0.0,) * 209, 0.0)

    monkeypatch.setattr(random.Random, "random", timed_random)
    result = run_search(
        start_episode(), evaluator,
        SearchConfig(simulations=2, mode="selfplay", root_noise=True, dirichlet_alpha=1e308),
        deadline=deadline,
    )
    assert_aborted(result, completed=0, elapsed=1.0)
    assert draws == 4 and calls == result.evaluations == 1
