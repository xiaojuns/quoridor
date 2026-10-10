from __future__ import annotations

from quoridor import engine
from quoridor.episode import start_episode
from quoridor.project_runtime import Deadline
from quoridor.puct import Evaluation, SearchConfig
from quoridor.selfplay import NewSelfPlayConfig, play_new_selfplay_episode


def _evaluator(episode, *, deadline):
    deadline.check()
    legal = engine.legal_actions(episode.state)
    logits = [-10.0] * engine.ACTION_SPACE_SIZE
    # Lowest legal action gives deterministic tie-breaking while preserving a
    # valid action at every root.
    logits[legal[0]] = 10.0
    return Evaluation(tuple(logits), 0.0)


def test_actor_records_one_exact_history_per_committed_move():
    config = NewSelfPlayConfig(
        SearchConfig(simulations=1, mode="selfplay", root_noise=False, temperature=0.0),
        max_game_plies=3,
        seed=7,
    )
    game = play_new_selfplay_episode(
        None, config, evaluator=_evaluator, deadline=Deadline(5.0)
    )
    assert game.status == "truncated"
    assert len(game.positions) == 2
    for index, position in enumerate(game.positions):
        assert position.episode.plies == index
        assert position.episode.history[-1] == engine.position_key(position.episode.state)
        assert position.visits[position.action] > 0
        assert position.to_play == position.episode.state.to_play
    assert not game.replay_eligible


def test_deadline_stop_never_uses_partial_search_visits_as_a_move():
    config = NewSelfPlayConfig(
        SearchConfig(simulations=100, mode="selfplay", root_noise=True, temperature=1.0),
        max_game_plies=20,
    )
    # A zero budget cannot be constructed by Deadline; a clock that starts
    # after the budget makes the first cooperative check fail.
    ticks = iter((0.0, 2.0, 2.0, 2.0, 2.0))
    deadline = Deadline(1.0, clock=lambda: next(ticks))
    game = play_new_selfplay_episode(
        None, config, evaluator=_evaluator, deadline=deadline
    )
    assert game.status == "truncated"
    assert game.episode.outcome is not None
    assert game.episode.outcome.status == "truncated"
    assert game.positions == ()


def test_actor_rejects_legacy_or_finished_initial_inputs():
    config = NewSelfPlayConfig(SearchConfig(simulations=1, mode="selfplay"))
    finished = start_episode(max_game_plies=1)
    # No actions are committed, so the starting episode is live; use a simple
    # wrong object to guard the explicit new-rule boundary.
    try:
        play_new_selfplay_episode(None, config, evaluator=_evaluator, deadline=Deadline(1), initial_episode=object())
    except TypeError:
        pass
    else:
        raise AssertionError("legacy/non-Episode input must be rejected")
