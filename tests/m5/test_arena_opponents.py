from collections import Counter
from types import SimpleNamespace

import pytest

from quoridor import engine
from quoridor.arena_opponents import (
    ClassicOpponent, _classic_state, opponent_identity, random_legal, shortest_path,
)
from quoridor.episode import advance_episode, start_episode
from quoridor.project_runtime import BudgetExpired, Deadline


def test_path_baseline_plays_legal_complete_games_for_both_starters():
    for player in (0, 1):
        episode = start_episode(engine.initial_state(player))
        deadline = Deadline(15)
        while not episode.finished:
            choice = shortest_path(episode, deadline=deadline)
            assert choice.action < 81
            episode = advance_episode(episode, choice.action)
        assert episode.outcome.reason == "goal"
        assert episode.outcome.winner in (0, 1)


def test_random_baseline_seed_history_and_legal_support():
    episode = start_episode()
    deadline = Deadline(15)
    for _ in range(12):
        first = random_legal(episode, deadline=deadline)
        assert first == random_legal(episode, deadline=deadline)
        assert first.action in engine.legal_actions(episode.state)
        episode = advance_episode(episode, first.action)


@pytest.mark.parametrize("choose", [random_legal, shortest_path])
def test_baselines_respect_budget_and_terminal_state(choose):
    now = [0]
    deadline = Deadline(1, clock=lambda: now[0])
    now[0] = 1
    with pytest.raises(BudgetExpired):
        choose(start_episode(), deadline=deadline)
    terminal = start_episode(engine.BoardState(pawns=((0, 4), (4, 4)), to_play=1))
    with pytest.raises(ValueError, match="live"):
        choose(terminal, deadline=Deadline(1))


def test_classic_adapter_preserves_full_repetition_history():
    from quoridor.codec import position_hash
    episode = start_episode()
    for action in [75, 3, 76, 4]:
        episode = advance_episode(episode, action)
    translated = _classic_state(episode)
    assert translated.move_number == 4
    assert sum(translated.position_counts.values()) == 5
    assert translated.position_counts[position_hash(translated)] == 2
    assert sorted(translated.position_counts.values()) == sorted(Counter(episode.history).values())
    assert translated.winner is None and translated.termination_reason is None


def test_classic_adapter_fixed_budget_and_new_rule_action_validation(monkeypatch):
    import quoridor.classic_bridge as bridge

    class FakeBridge:
        action = 67

        def __init__(self, **kwargs):
            assert 0 < kwargs["startup_timeout"] <= 5

        def choose(self, state, *, budget, max_depth, timeout):
            assert budget.kind == "nodes" and budget.limit == 587
            assert max_depth == 5 and timeout > 0
            assert state.current_player == 0
            return SimpleNamespace(action_index=self.action, nodes=500, depth=3)

        def close(self):
            pass

    monkeypatch.setattr(bridge, "ClassicAIBridge", FakeBridge)
    opponent = ClassicOpponent(deadline=Deadline(10))
    opponent.identity["settings"]["node_limit"] = 1
    assert opponent.choose(start_episode(), deadline=Deadline(10)).action == 67
    FakeBridge.action = 0
    with pytest.raises(ValueError, match="illegal"):
        opponent.choose(start_episode(), deadline=Deadline(10))
    opponent.close()


def test_fixed_opponents_are_diagnostics_and_reject_changed_config(tmp_path, monkeypatch):
    import quoridor.arena_opponents as module
    for kind in ("random_legal", "shortest_path", "classic"):
        identity = opponent_identity(kind)
        assert identity["formal_promotion"] is False
        assert identity["source_files"]
        assert len(identity["config_sha256"]) == 64
    modified = tmp_path / "opponents.yaml"
    modified.write_bytes(module.CONFIG_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(module, "CONFIG_PATH", modified)
    with pytest.raises(ValueError, match="digest"):
        opponent_identity("classic")
