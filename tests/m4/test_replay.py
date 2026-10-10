from __future__ import annotations

import torch

from quoridor.episode import advance_episode, start_episode
from quoridor.puct import SearchConfig
from quoridor.project_runtime import Deadline
from quoridor.replay import (
    ACTION_SIZE,
    DEFAULT_REPLAY_CAPACITY,
    ReplayBuffer,
    ReplayGame,
    ReplayPosition,
    ReplaySample,
)
from quoridor.selfplay import NewSelfPlayConfig, play_new_selfplay_episode
from quoridor.puct import Evaluation
from quoridor import engine


def _forward_evaluator(episode, *, deadline):
    deadline.check()
    logits = [-100.0] * ACTION_SIZE
    row, col = episode.state.pawns[episode.state.to_play]
    target = (row - 1, col) if episode.state.to_play == 0 else (row + 1, col)
    preferred = engine.encode_move(*target)
    if preferred not in engine.legal_actions(episode.state):
        preferred = engine.legal_actions(episode.state)[0]
    logits[preferred] = 100.0
    return Evaluation(tuple(logits), 0.0)


def _complete_game(max_game_plies=40):
    config = NewSelfPlayConfig(
        SearchConfig(simulations=1, mode="selfplay", root_noise=False, temperature=0.0),
        max_game_plies=max_game_plies,
    )
    return play_new_selfplay_episode(
        None, config, evaluator=_forward_evaluator, deadline=Deadline(10.0)
    )


def test_replay_admits_terminal_game_with_current_player_labels():
    game = _complete_game()
    assert game.status == "completed"
    assert game.replay_eligible
    replay = ReplayBuffer(capacity=100)
    assert replay.add_game(game.replay_game())
    assert len(replay) == len(game.positions)
    batch = replay.sample(len(replay))
    assert batch["features"].shape[1:] == (16, 9, 9)
    assert batch["legal_masks"].shape[1:] == (ACTION_SIZE,)
    assert batch["policies"].shape[1:] == (ACTION_SIZE,)
    assert set(batch["values"].tolist()) <= {-1.0, 1.0}
    assert set(batch["to_play"].tolist()) == {0, 1}


def test_truncated_game_is_excluded_wholesale_but_counted():
    game = _complete_game(max_game_plies=1)
    assert game.status == "truncated"
    replay = ReplayBuffer()
    assert not replay.add_game(game.replay_game())
    assert len(replay) == 0
    assert replay.truncated_games == 1
    assert replay.rejected_games == 1
    assert replay.rejected_positions == len(game.positions)


def test_replay_capacity_evicts_oldest_deterministically():
    def sample(value: float, action: int) -> ReplaySample:
        policy = torch.zeros(ACTION_SIZE)
        policy[action] = 1
        return ReplaySample(
            features=torch.zeros(16, 9, 9),
            legal_mask=torch.ones(ACTION_SIZE, dtype=torch.bool),
            policy=policy,
            to_play=0,
            value=value,
            action=action,
            episode_plies=0,
            history=("key",),
        )

    replay = ReplayBuffer(capacity=2)
    replay.add(sample(-1, 0))
    replay.add(sample(0, 1))
    replay.add(sample(1, 2))
    assert len(replay) == 2
    assert sorted(replay.sample(2)["values"].tolist()) == [0.0, 1.0]


def test_replay_default_capacity_is_the_declared_versioned_limit():
    assert ReplayBuffer().capacity == DEFAULT_REPLAY_CAPACITY


def test_replay_clones_tensor_storage_at_admission():
    policy = torch.zeros(ACTION_SIZE)
    policy[0] = 1
    sample = ReplaySample(
        features=torch.zeros(16, 9, 9),
        legal_mask=torch.ones(ACTION_SIZE, dtype=torch.bool),
        policy=policy,
        to_play=0,
        value=0,
        action=0,
        episode_plies=0,
        history=("key",),
    )
    replay = ReplayBuffer(capacity=1)
    replay.add(sample)
    sample.features[0, 0, 0] = 9
    assert float(replay.samples[0].features[0, 0, 0]) == 0


def test_repetition_draw_labels_every_recorded_player_zero():
    episode = start_episode()
    positions = []
    actions = [
        engine.encode_move(8, 3), engine.encode_move(0, 3),
        engine.encode_move(8, 4), engine.encode_move(0, 4),
    ] * 2
    for action in actions:
        policy = torch.zeros(ACTION_SIZE)
        policy[action] = 1.0
        visits = [0] * ACTION_SIZE
        visits[action] = 1
        positions.append(ReplayPosition(episode, action, tuple(policy.tolist()), tuple(visits)))
        episode = advance_episode(episode, action)
    assert episode.outcome is not None and episode.outcome.reason == "repetition"
    replay = ReplayBuffer(capacity=100)
    assert replay.add_game(ReplayGame(episode, tuple(positions)))
    batch = replay.sample(len(replay))
    assert len(batch["values"]) == len(positions)
    assert torch.equal(batch["values"], torch.zeros(len(positions)))
