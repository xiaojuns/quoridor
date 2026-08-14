from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import torch

from .codec import (
    ACTION_SPACE_SIZE,
    mirror_action_index,
    mirror_state,
    to_canonical_action_index,
)
from .game import (
    State,
    apply_action_index,
    create_initial_state,
    get_winner,
    is_terminal,
)
from .mcts import Evaluator, MCTS, MCTSConfig
from .network import INPUT_CHANNELS, SUPPORTED_INPUT_CHANNELS, encode_state
from .replay import TrainingSample


def scheduled_search_seed(seed: int) -> int:
    """Return the exact MCTS seed used by one scheduled self-play invocation."""

    if type(seed) is not int:
        raise TypeError("scheduled self-play seed must be an integer")
    return random.Random(seed).randrange(2**63)


@dataclass(frozen=True)
class SelfPlayConfig:
    mcts: MCTSConfig
    temperature_moves: int = 16
    mirror_augmentation: bool = True
    seed: int = 0
    input_channels: int = INPUT_CHANNELS
    collect_samples: bool = True
    collect_raw_records: bool = False

    def __post_init__(self) -> None:
        if self.temperature_moves < 0:
            raise ValueError("temperature_moves must be non-negative")
        if self.input_channels not in SUPPORTED_INPUT_CHANNELS:
            raise ValueError("unsupported self-play input channel contract")
        if not isinstance(self.collect_samples, bool):
            raise ValueError("collect_samples must be boolean")
        if not isinstance(self.collect_raw_records, bool):
            raise ValueError("collect_raw_records must be boolean")


@dataclass(frozen=True)
class SelfPlayPly:
    """Unencoded search target in the raw 209-action coordinate system."""

    action_index: int
    visit_counts: tuple[int, ...]
    root_value: float

    def __post_init__(self) -> None:
        if (
            type(self.action_index) is not int
            or not 0 <= self.action_index < ACTION_SPACE_SIZE
        ):
            raise ValueError("self-play action_index is outside the action space")
        counts = tuple(self.visit_counts)
        if len(counts) != ACTION_SPACE_SIZE:
            raise ValueError("self-play visit_counts must contain 209 entries")
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("self-play visit counts must be non-negative integers")
        if counts[self.action_index] <= 0:
            raise ValueError("selected self-play action must have positive visits")
        value = float(self.root_value)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("self-play root value must be finite and in [-1, 1]")
        object.__setattr__(self, "visit_counts", counts)
        object.__setattr__(self, "root_value", value)


@dataclass(frozen=True)
class SelfPlayGame:
    samples: tuple[TrainingSample, ...]
    initial_state: State
    raw_plies: tuple[SelfPlayPly, ...]
    winner: int
    move_count: int
    termination_reason: str


@dataclass
class _ActiveGame:
    game_index: int
    initial_state: State
    state: State
    starting_move_number: int
    records: list[tuple[torch.Tensor, torch.Tensor, int]]
    raw_plies: list[SelfPlayPly]
    session_id: int | None = None


def _finish_game(
    initial_state: State,
    state: State,
    records: list[tuple[torch.Tensor, torch.Tensor, int]],
    raw_plies: list[SelfPlayPly],
    starting_move_number: int,
) -> SelfPlayGame:
    winner = get_winner(state)
    if winner is None or state.termination_reason is None:
        raise RuntimeError("self-play ended without a zero-sum terminal result")
    samples = tuple(
        TrainingSample(
            state=encoded_state,
            policy=policy,
            value=1.0 if player == winner else -1.0,
        )
        for encoded_state, policy, player in records
    )
    return SelfPlayGame(
        samples=samples,
        initial_state=initial_state,
        raw_plies=tuple(raw_plies),
        winner=winner,
        move_count=state.move_number - starting_move_number,
        termination_reason=state.termination_reason,
    )


def _canonical_policy(
    raw_policy: Iterable[float],
    original_current_player: int,
) -> torch.Tensor:
    canonical = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
    for raw_index, probability in enumerate(raw_policy):
        if probability <= 0:
            continue
        canonical_index = to_canonical_action_index(
            raw_index,
            original_current_player,
        )
        canonical[canonical_index] += float(probability)
    total = canonical.sum()
    if total <= 0:
        raise RuntimeError("MCTS returned a policy without probability mass")
    return canonical / total


def _mirrored_policy(policy: torch.Tensor) -> torch.Tensor:
    mirrored = torch.zeros_like(policy)
    for action_index in range(ACTION_SPACE_SIZE):
        mirrored[mirror_action_index(action_index)] = policy[action_index]
    return mirrored


def play_self_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    initial_state: State | None = None,
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS] = MCTS,
) -> SelfPlayGame:
    state = initial_state or create_initial_state()
    if is_terminal(state):
        raise ValueError("self-play initial state must be non-terminal")
    starting_move_number = state.move_number

    search = search_factory(
        evaluator,
        replace(config.mcts, seed=scheduled_search_seed(config.seed)),
    )
    records: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    raw_plies: list[SelfPlayPly] = []
    game_initial_state = state

    while not is_terminal(state):
        temperature = 1.0 if state.move_number < config.temperature_moves else 0.0
        result = search.run(
            state,
            temperature=temperature,
            add_root_noise=config.mcts.add_root_noise,
        )
        if result.action_index is None:
            raise RuntimeError("MCTS returned no action for a non-terminal state")

        if config.collect_raw_records:
            raw_plies.append(
                SelfPlayPly(
                    action_index=result.action_index,
                    visit_counts=tuple(result.visit_counts),
                    root_value=result.root_value,
                )
            )

        if config.collect_samples:
            canonical_policy = _canonical_policy(result.policy, state.current_player)
            records.append(
                (
                    encode_state(state, input_channels=config.input_channels),
                    canonical_policy,
                    state.current_player,
                )
            )
            if config.mirror_augmentation:
                records.append(
                    (
                        encode_state(
                            mirror_state(state),
                            input_channels=config.input_channels,
                        ),
                        _mirrored_policy(canonical_policy),
                        state.current_player,
                    )
                )
        state = apply_action_index(state, result.action_index)

    return _finish_game(
        game_initial_state,
        state,
        records,
        raw_plies,
        starting_move_number,
    )


def play_self_games_batched(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    count: int,
    *,
    initial_state: State | None = None,
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS],
) -> tuple[SelfPlayGame, ...]:
    """Play several games with one batched native MCTS scheduler."""

    return play_self_games_scheduled(
        evaluator,
        config,
        count,
        actors=count,
        initial_state=initial_state,
        search_factory=search_factory,
    )


def play_self_games_scheduled(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    count: int,
    *,
    actors: int,
    initial_state: State | None = None,
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS],
    progress_callback: Callable[[int, int], None] | None = None,
    completed_game_callback: Callable[[int, SelfPlayGame], None] | None = None,
) -> tuple[SelfPlayGame, ...]:
    """Play ``count`` games with at most ``actors`` active roots.

    Completed slots are refilled in ascending game-index order. A session-aware
    native search keeps each selected subtree; older batched engines continue to
    work through ``run_batch``.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if actors <= 0:
        raise ValueError("actors must be positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if completed_game_callback is not None and not callable(completed_game_callback):
        raise TypeError("completed_game_callback must be callable or None")
    actor_limit = min(actors, count)
    if initial_state is not None and is_terminal(initial_state):
        raise ValueError("self-play initial state must be non-terminal")

    search = search_factory(
        evaluator,
        replace(config.mcts, seed=scheduled_search_seed(config.seed)),
    )
    run_batch = getattr(search, "run_batch", None)
    create_session = getattr(search, "create_session", None)
    run_sessions = getattr(search, "run_sessions", None)
    advance_session = getattr(search, "advance_session", None)
    release_session = getattr(search, "release_session", None)
    session_methods = (
        create_session,
        run_sessions,
        advance_session,
        release_session,
    )
    session_capable = all(callable(method) for method in session_methods)
    if not session_capable and any(callable(method) for method in session_methods):
        raise TypeError("search engine exposes an incomplete persistent session API")
    if not session_capable and not callable(run_batch):
        raise TypeError("batched self-play requires a search engine with run_batch")

    completed: list[SelfPlayGame | None] = [None] * count
    active: list[_ActiveGame] = []
    open_sessions: set[int] = set()
    next_game_index = 0
    processed_moves = 0

    def fill_slots() -> None:
        nonlocal next_game_index
        while len(active) < actor_limit and next_game_index < count:
            state = initial_state or create_initial_state()
            if is_terminal(state):
                raise ValueError("self-play initial state must be non-terminal")
            session_id = None
            if session_capable:
                session_id = int(create_session(state))
                open_sessions.add(session_id)
            active.append(
                _ActiveGame(
                    game_index=next_game_index,
                    initial_state=state,
                    state=state,
                    starting_move_number=state.move_number,
                    records=[],
                    raw_plies=[],
                    session_id=session_id,
                )
            )
            next_game_index += 1

    try:
        fill_slots()
        while active:
            if progress_callback is not None:
                progress_callback(
                    processed_moves,
                    sum(game is not None for game in completed),
                )
            active_states = [game.state for game in active]
            temperatures = [
                1.0 if state.move_number < config.temperature_moves else 0.0
                for state in active_states
            ]
            if session_capable:
                session_ids = [
                    game.session_id
                    for game in active
                    if game.session_id is not None
                ]
                if len(session_ids) != len(active):
                    raise RuntimeError("active self-play game is missing its session")
                results = run_sessions(
                    session_ids,
                    active_states,
                    temperatures=temperatures,
                    add_root_noise=config.mcts.add_root_noise,
                )
            else:
                results = run_batch(
                    active_states,
                    temperatures=temperatures,
                    add_root_noise=config.mcts.add_root_noise,
                )
            if len(results) != len(active):
                raise RuntimeError("batched MCTS returned the wrong result count")

            finished_indices: list[int] = []
            for active_index, (game, state, result) in enumerate(
                zip(active, active_states, results)
            ):
                if result.action_index is None:
                    raise RuntimeError(
                        "MCTS returned no action for a non-terminal state"
                    )
                if config.collect_raw_records:
                    game.raw_plies.append(
                        SelfPlayPly(
                            action_index=result.action_index,
                            visit_counts=tuple(result.visit_counts),
                            root_value=result.root_value,
                        )
                    )
                if config.collect_samples:
                    canonical_policy = _canonical_policy(
                        result.policy,
                        state.current_player,
                    )
                    game.records.append(
                        (
                            encode_state(
                                state,
                                input_channels=config.input_channels,
                            ),
                            canonical_policy,
                            state.current_player,
                        )
                    )
                    if config.mirror_augmentation:
                        game.records.append(
                            (
                                encode_state(
                                    mirror_state(state),
                                    input_channels=config.input_channels,
                                ),
                                _mirrored_policy(canonical_policy),
                                state.current_player,
                            )
                        )
                next_state = apply_action_index(state, result.action_index)
                if session_capable:
                    if game.session_id is None:
                        raise RuntimeError(
                            "active self-play game is missing its session"
                        )
                    advance_session(
                        game.session_id,
                        result.action_index,
                        next_state,
                    )
                game.state = next_state
                processed_moves += 1
                if is_terminal(next_state):
                    finished_game = _finish_game(
                        game.initial_state,
                        next_state,
                        game.records,
                        game.raw_plies,
                        game.starting_move_number,
                    )
                    completed[game.game_index] = finished_game
                    if session_capable:
                        release_session(game.session_id)
                        open_sessions.discard(game.session_id)
                    if completed_game_callback is not None:
                        completed_game_callback(game.game_index, finished_game)
                    finished_indices.append(active_index)

            for active_index in reversed(finished_indices):
                del active[active_index]
            if progress_callback is not None:
                progress_callback(
                    processed_moves,
                    sum(game is not None for game in completed),
                )
            fill_slots()
    finally:
        if session_capable:
            for session_id in sorted(open_sessions):
                release_session(session_id)
                open_sessions.discard(session_id)

    if any(game is None for game in completed):
        raise RuntimeError("self-play scheduler did not finish every game")
    return tuple(game for game in completed if game is not None)


def play_self_games(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    count: int,
    *,
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS] = MCTS,
) -> tuple[SelfPlayGame, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    return tuple(
        play_self_game(
            evaluator,
            replace(config, seed=config.seed + game_index),
            search_factory=search_factory,
        )
        for game_index in range(count)
    )


__all__ = [
    "SelfPlayConfig",
    "SelfPlayPly",
    "SelfPlayGame",
    "scheduled_search_seed",
    "play_self_game",
    "play_self_games",
    "play_self_games_batched",
    "play_self_games_scheduled",
]
