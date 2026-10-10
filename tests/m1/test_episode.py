from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

from quoridor import engine
from quoridor.episode import (
    Episode,
    EpisodeFinishedError,
    Outcome,
    advance_episode,
    enforce_deadline,
    fail_episode,
    interrupt_episode,
    start_episode,
)
from quoridor.project_config import PROJECT_ROOT, load_project_config
from quoridor.project_runtime import Deadline, RunSession


def move(row: int, col: int) -> int:
    return 9 * row + col


def cycle(episode: Episode) -> Episode:
    for action in (move(8, 3), move(0, 3), move(8, 4), move(0, 4)):
        episode = advance_episode(episode, action)
    return episode


def one_move_from_goal(*, max_game_plies: int = 256) -> Episode:
    return start_episode(
        engine.BoardState(pawns=((1, 4), (6, 4))), max_game_plies=max_game_plies,
    )


def test_initial_and_committed_positions_are_counted_once_and_immutable():
    initial = start_episode()
    after = advance_episode(initial, move(7, 4))
    assert initial.history == (engine.position_key(initial.state),)
    assert initial.plies == 0 and initial.repetition_count == 1
    assert after.history == (*initial.history, engine.position_key(after.state))
    assert after.plies == 1 and after.state.to_play == 1
    assert initial.state.to_play == 0 and initial.state.pawns[0] == (8, 4)
    assert all(count == 1 for count in after.position_counts.values())
    with pytest.raises(TypeError):
        after.position_counts[after.history[-1]] = 99
    with pytest.raises(FrozenInstanceError):
        after.history = initial.history


def test_full_history_distinguishes_same_board_and_prevents_board_only_cache_key():
    initial = start_episode()
    repeated_once = cycle(initial)
    assert initial.state == repeated_once.state
    assert initial.repetition_count == 1
    assert repeated_once.repetition_count == 2
    assert initial.adjudication_key != repeated_once.adjudication_key
    assert len({initial.adjudication_key, repeated_once.adjudication_key}) == 2
    assert not repeated_once.finished
    assert replace(initial.state, to_play=1) != initial.state
    assert engine.position_key(replace(initial.state, to_play=1)) != initial.history[0]


def test_real_threefold_is_a_draw_and_precedes_ply_cap():
    original = start_episode(max_game_plies=8)
    result = cycle(cycle(original))
    assert result.plies == 8 and result.repetition_count == 3
    assert result.outcome == Outcome("terminated", "repetition")
    assert result.outcome.terminated and not result.outcome.truncated
    assert result.outcome.winner is None and result.outcome.replay_eligible
    assert result.value_targets([1, 0, 0, 1]) == (0.0, 0.0, 0.0, 0.0)
    assert original.repetition_count == 1 and original.outcome is None
    with pytest.raises(EpisodeFinishedError):
        advance_episode(result, move(8, 3))


def test_goal_precedes_ply_cap_and_terminal_current_player_is_loser():
    episode = one_move_from_goal(max_game_plies=1)
    result = advance_episode(episode, move(0, 4))
    assert result.plies == 1
    assert result.state.to_play == 1
    assert result.outcome == Outcome("terminated", "goal", winner=0)
    assert result.outcome.value_for(result.state.to_play) == -1.0
    assert result.value_targets([1, 1, 0, 1, 0]) == (-1.0, -1.0, 1.0, -1.0, 1.0)


def test_second_player_goal_labels_are_not_tied_to_move_parity():
    episode = start_episode(engine.BoardState(pawns=((2, 4), (7, 4)), to_play=1))
    result = advance_episode(episode, move(8, 4))
    assert result.outcome == Outcome("terminated", "goal", winner=1)
    assert result.state.to_play == 0
    assert result.value_targets([1, 0, 1]) == (1.0, -1.0, 1.0)


def test_unfinished_ply_cap_excludes_all_samples_instead_of_creating_a_draw():
    episode = start_episode(max_game_plies=1)
    result = advance_episode(episode, move(7, 4))
    assert result.outcome == Outcome("truncated", "ply_limit")
    assert result.outcome.truncated and not result.outcome.terminated
    assert result.outcome.winner is None and not result.outcome.replay_eligible
    assert result.outcome.value_for(0) is None
    assert result.outcome.value_for(1) is None
    assert result.value_targets([0]) is None
    assert episode.value_targets([0]) is None


def rectangle_cycle(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    """Hamiltonian cycle on an even-height rectangle, starting at (0, 0)."""

    path = [(0, 0)]
    for row in range(rows):
        columns_in_order = range(1, columns) if row % 2 == 0 else range(columns - 1, 0, -1)
        path.extend((row, col) for col in columns_in_order)
    path.extend((row, 0) for row in range(rows - 1, 0, -1))
    return tuple(path)


@pytest.mark.parametrize("finish_with_goal", [False, True])
def test_actual_256th_ply_distinguishes_goal_from_operational_truncation(finish_with_goal):
    # Disjoint 36- and 24-square pawn cycles repeat the full position every 144
    # plies; no third occurrence or goal can precede the actual default cap.
    path0 = tuple((row + 1, col) for row, col in rectangle_cycle(4, 9))
    path1 = tuple((col + 5, row) for row, col in rectangle_cycle(8, 3))
    episode = start_episode(engine.BoardState(pawns=(path0[0], path1[0])))
    for index in range(255):
        assert not episode.finished
        path = path0 if index % 2 == 0 else path1
        destination = path[(index // 2 + 1) % len(path)]
        episode = advance_episode(episode, move(*destination))
    assert episode.plies == 255 and not episode.finished
    assert episode.state.pawns[1][0] == 7
    destination = (8, episode.state.pawns[1][1]) if finish_with_goal else path1[128 % len(path1)]
    result = advance_episode(episode, move(*destination))
    assert result.plies == 256
    expected = Outcome("terminated", "goal", winner=1) if finish_with_goal else Outcome("truncated", "ply_limit")
    assert result.outcome == expected


def test_exact_deadline_prevents_an_action_including_an_immediate_win(monkeypatch):
    now = [0.0]
    deadline = Deadline(2, clock=lambda: now[0])
    episode = one_move_from_goal()
    now[0] = 2.0

    def forbidden(*args, **kwargs):
        pytest.fail("an expired episode must not start a transition")

    monkeypatch.setattr(engine, "apply_action", forbidden)
    result = advance_episode(episode, move(0, 4), deadline=deadline)
    assert result.state is episode.state and result.history is episode.history
    assert result.plies == 0
    assert result.outcome == Outcome(
        "truncated", "time_budget", elapsed_seconds=2.0, budget_seconds=2.0,
    )
    assert result.value_targets([0]) is None


@pytest.mark.parametrize("goal", [False, True])
def test_deadline_overrun_records_observed_elapsed_but_preserves_completed_goal(monkeypatch, goal):
    now = [0.0]
    deadline = Deadline(2, clock=lambda: now[0])
    episode = one_move_from_goal() if goal else start_episode()
    apply_action = engine.apply_action

    def delayed_transition(state, action):
        next_state = apply_action(state, action)
        now[0] = 2.75
        return next_state

    monkeypatch.setattr(engine, "apply_action", delayed_transition)
    result = advance_episode(episode, move(0, 4) if goal else move(7, 4), deadline=deadline)
    assert result.plies == 1
    if goal:
        assert result.outcome == Outcome("terminated", "goal", winner=0)
    else:
        assert result.outcome == Outcome(
            "truncated", "time_budget", elapsed_seconds=2.75, budget_seconds=2.0,
        )
        assert result.value_targets([0]) is None


def test_completed_repetition_is_not_relabelled_by_deadline_or_interruption():
    result = cycle(cycle(start_episode()))
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    now[0] = 1.5
    assert enforce_deadline(result, deadline) is result
    assert interrupt_episode(result) is result
    assert fail_episode(result, "infrastructure_error", detail="late worker error") is result


def test_goal_overrun_keeps_game_result_and_runtime_logs_actual_budget_exhaustion(tmp_path, monkeypatch):
    now = [0.0]
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    run = RunSession(config, run_root=tmp_path, clock=lambda: now[0])
    apply_action = engine.apply_action

    def overrun_transition(state, action):
        next_state = apply_action(state, action)
        now[0] = run.deadline.seconds + 0.25
        return next_state

    monkeypatch.setattr(engine, "apply_action", overrun_transition)
    with run:
        episode = advance_episode(one_move_from_goal(), move(0, 4), deadline=run.deadline)
        run.log("episode_finished", status=episode.outcome.status, reason=episode.outcome.reason)
    assert episode.outcome == Outcome("terminated", "goal", winner=0)
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["status"] == "budget_exhausted"
    assert summary["elapsed_seconds"] == run.deadline.seconds + 0.25
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert events[-2]["status"] == "terminated" and events[-2]["reason"] == "goal"
    assert events[-1]["status"] == "budget_exhausted"
    assert events[-1]["elapsed_seconds"] == run.deadline.seconds + 0.25


@pytest.mark.parametrize("action", [-1, 209, True, None, 1.5, move(0, 0)])
def test_illegal_action_records_failure_without_mutating_position_or_history(action):
    episode = start_episode()
    result = advance_episode(episode, action)
    assert result.state is episode.state
    assert result.history is episode.history and result.plies == 0
    assert result.outcome.status == "failed"
    assert result.outcome.reason == "illegal_action"
    assert result.outcome.detail
    assert result.outcome.winner is None
    assert result.value_targets([0, 1]) is None
    assert episode.outcome is None


@pytest.mark.parametrize("reason", ["numerical_error", "infrastructure_error"])
def test_explicit_failure_is_unlabelled_and_cannot_continue(reason):
    episode = advance_episode(start_episode(), move(7, 4))
    result = fail_episode(episode, reason, detail="synthetic failure evidence")
    assert result.outcome == Outcome("failed", reason, detail="synthetic failure evidence")
    assert result.state is episode.state and result.history is episode.history
    assert result.value_targets([0, 1]) is None
    with pytest.raises(EpisodeFinishedError):
        advance_episode(result, move(1, 4))


def test_interruption_is_a_truncation_with_no_result_label():
    episode = advance_episode(start_episode(), move(7, 4))
    result = interrupt_episode(episode, detail="operator interrupted actor")
    assert result.outcome == Outcome("truncated", "interrupted", detail="operator interrupted actor")
    assert result.plies == 1 and result.history is episode.history
    assert result.value_targets([0, 1]) is None


@pytest.mark.parametrize("arguments", [
    ("terminated", "goal", None),
    ("terminated", "repetition", 0),
    ("truncated", "goal", 0),
    ("truncated", "ply_limit", 1),
    ("failed", "repetition", None),
    ("completed", "goal", 0),
    ([], "goal", 0),
    ("terminated", [], 0),
])
def test_outcome_rejects_ambiguous_status_reason_or_winner(arguments):
    with pytest.raises(ValueError):
        Outcome(*arguments)


def test_episode_rejects_legacy_state_and_inconsistent_history_or_labels():
    from quoridor.game import create_initial_state

    with pytest.raises(TypeError, match="BoardState"):
        start_episode(create_initial_state())
    episode = start_episode()
    with pytest.raises(ValueError, match="history"):
        replace(episode, history=list(episode.history))
    with pytest.raises(ValueError, match="current full position"):
        replace(episode, history=("different position",))
    with pytest.raises(ValueError, match="completed result"):
        replace(episode, outcome=Outcome("terminated", "goal", winner=0))
    with pytest.raises(ValueError, match="contradicts"):
        replace(cycle(cycle(episode)), outcome=None)


def test_history_rejects_fake_ply_cap_garbage_legacy_and_disconnected_keys():
    from quoridor.codec import position_hash
    from quoridor.game import create_initial_state

    episode = start_episode()
    with pytest.raises(ValueError, match="completed result"):
        replace(episode, outcome=Outcome("truncated", "ply_limit"))
    for bad_key in ("garbage", position_hash(create_initial_state())):
        with pytest.raises(ValueError, match="position key"):
            replace(episode, history=(bad_key, *episode.history))
    disconnected = engine.BoardState(pawns=((5, 4), (0, 4)), to_play=1)
    with pytest.raises(ValueError, match="one legal action"):
        Episode(disconnected, (*episode.history, engine.position_key(disconnected)))


@pytest.mark.parametrize("orientation", ["h", "v"])
def test_wall_history_proves_placement_and_inventory_belong_to_the_acting_player(orientation):
    episode = start_episode()
    result = advance_episode(episode, engine.encode_wall(orientation, 3, 3))
    assert not result.finished and result.plies == 1
    assert result.state.walls_remaining == (9, 10)
    wrong_inventory = replace(result.state, walls_remaining=(10, 9))
    with pytest.raises(ValueError, match="one legal action"):
        Episode(wrong_inventory, (*episode.history, engine.position_key(wrong_inventory)))


def test_history_rejects_continuing_after_a_real_draw_or_ply_limit():
    drawn = cycle(cycle(start_episode()))
    child = engine.apply_action(drawn.state, move(7, 4))
    with pytest.raises(ValueError, match="already finished"):
        Episode(child, (*drawn.history, engine.position_key(child)))
    truncated = advance_episode(start_episode(max_game_plies=1), move(7, 4))
    child = engine.apply_action(truncated.state, move(1, 4))
    with pytest.raises(ValueError, match="already finished"):
        Episode(child, (*truncated.history, engine.position_key(child)), max_game_plies=1)


@pytest.mark.parametrize("invalid", [True, [], float("inf"), float("nan"), 10**400])
def test_outcome_and_deadline_reject_invalid_numeric_evidence(invalid):
    with pytest.raises(ValueError):
        Outcome("truncated", "time_budget", elapsed_seconds=invalid, budget_seconds=1)
    with pytest.raises(ValueError):
        enforce_deadline(start_episode(), SimpleNamespace(elapsed_seconds=invalid, seconds=1))
    with pytest.raises(ValueError):
        enforce_deadline(start_episode(), SimpleNamespace(elapsed_seconds=2, seconds=invalid))


@pytest.mark.parametrize("max_game_plies", [0, -1, True, 1.5])
def test_episode_requires_positive_integer_ply_limit(max_game_plies):
    with pytest.raises(ValueError):
        start_episode(max_game_plies=max_game_plies)


@given(st.lists(st.integers(min_value=0, max_value=100), max_size=40))
@settings(max_examples=40, deadline=None, derandomize=True)
def test_reachable_episode_histories_and_current_player_targets(choices):
    episode = start_episode(max_game_plies=32)
    recorded_players = []
    for choice in choices:
        if episode.finished:
            break
        old_state, old_history = episode.state, episode.history
        actions = tuple(engine.encode_move(*position) for position in engine.legal_pawn_moves(old_state))
        recorded_players.append(old_state.to_play)
        episode = advance_episode(episode, actions[choice % len(actions)])
        assert episode.history[:-1] == old_history
        assert episode.history[-1] == engine.position_key(episode.state)
        assert episode.state.to_play == 1 - old_state.to_play
        assert sum(episode.position_counts.values()) == episode.plies + 1
        assert episode.repetition_count <= 3
        assert episode.plies <= 32
    labels = episode.value_targets(recorded_players)
    if episode.outcome is None or not episode.outcome.terminated:
        assert labels is None
    elif episode.outcome.reason == "repetition":
        assert labels == (0.0,) * len(recorded_players)
    else:
        assert labels == tuple(
            1.0 if player == engine.goal_winner(episode.state) else -1.0
            for player in recorded_players
        )
