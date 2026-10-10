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


# ---------------------------------------------------------------------------
# quoridor-v1 M4 actor
# ---------------------------------------------------------------------------
# The legacy actor above intentionally remains available for historical phase
# scripts.  These types use immutable ``Episode`` records and the new puct-v1
# contract; they are kept in this module so callers have one self-play entry
# point while the two rule versions cannot be mixed accidentally.

from dataclasses import dataclass as _dataclass
import math as _math

from .episode import Episode as _Episode
from .episode import advance_episode as _advance_episode
from .episode import enforce_deadline as _enforce_deadline
from .episode import fail_episode as _fail_episode
from .episode import interrupt_episode as _interrupt_episode
from .engine import ACTION_SPACE_SIZE as _NEW_ACTION_SIZE
from .puct import Evaluation as _Evaluation
from .puct import SearchConfig as _SearchConfig
from .puct import SearchError as _SearchError
from .puct import run_search as _run_search
from .project_runtime import BudgetExpired as _BudgetExpired
from .project_runtime import Deadline as _Deadline
from .replay import ReplayGame as _ReplayGame
from .replay import ReplayPosition as _ReplayPosition


@_dataclass(frozen=True, slots=True)
class NewSelfPlayConfig:
    """Bounded self-play settings for the new rules/model contract."""

    search: _SearchConfig
    max_game_plies: int = 256
    seed: int = 0
    model_hash: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.search, _SearchConfig) or self.search.mode != "selfplay":
            raise ValueError("M4 self-play requires SearchConfig(mode='selfplay')")
        if type(self.max_game_plies) is not int or self.max_game_plies <= 0:
            raise ValueError("max_game_plies must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("self-play seed must be a non-negative integer")
        for label, value in (("model_hash", self.model_hash), ("config_hash", self.config_hash)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label} must be a non-empty string")


@_dataclass(frozen=True, slots=True)
class NewSelfPlayPosition:
    """One committed root and the visit target used to select its action."""

    episode: _Episode
    action: int
    visits: tuple[int, ...]
    policy: tuple[float, ...]
    root_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.episode, _Episode) or self.episode.finished:
            raise ValueError("self-play position must contain a live Episode")
        if type(self.action) is not int or not 0 <= self.action < _NEW_ACTION_SIZE:
            raise ValueError("self-play action is outside the action space")
        from . import engine as _new_engine

        if self.action not in _new_engine.legal_actions(self.episode.state):
            raise ValueError("self-play action must be legal at its root")
        visits = tuple(self.visits)
        if len(visits) != _NEW_ACTION_SIZE or any(type(item) is not int or item < 0 for item in visits):
            raise ValueError("self-play visits must contain 209 non-negative counts")
        if visits[self.action] <= 0:
            raise ValueError("selected self-play action must have positive visits")
        policy = tuple(float(item) for item in self.policy)
        if len(policy) != _NEW_ACTION_SIZE or any(not _math.isfinite(item) or item < 0 for item in policy):
            raise ValueError("self-play policy must contain 209 finite non-negative values")
        if _math.fsum(policy) <= 0 or policy[self.action] <= 0:
            raise ValueError("selected self-play action must have positive policy mass")
        legal = set(_new_engine.legal_actions(self.episode.state))
        if any(probability > 0 and action not in legal for action, probability in enumerate(policy)):
            raise ValueError("self-play policy must have support only on legal actions")
        value = float(self.root_value)
        if not _math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("self-play root value must be finite and in [-1,1]")
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "policy", tuple(item / _math.fsum(policy) for item in policy))
        object.__setattr__(self, "root_value", value)

    @property
    def to_play(self) -> int:
        return self.episode.state.to_play

    def replay_position(self) -> _ReplayPosition:
        return _ReplayPosition(
            episode=self.episode,
            action=self.action,
            policy=self.policy,
            visits=self.visits,
            root_value=self.root_value,
        )


@_dataclass(frozen=True, slots=True)
class NewSelfPlayGame:
    """Truthful actor result; only ``completed`` games enter baseline replay."""

    episode: _Episode
    positions: tuple[NewSelfPlayPosition, ...]
    status: str
    reason: str
    seed: int
    model_hash: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode, _Episode):
            raise TypeError("self-play result requires an Episode")
        if self.status not in {"completed", "truncated", "failed"}:
            raise ValueError("invalid self-play result status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("self-play result reason is required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("self-play result seed must be non-negative")
        for label, value in (("model_hash", self.model_hash), ("config_hash", self.config_hash)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label} must be a non-empty string")
        positions = tuple(self.positions)
        if any(not isinstance(item, NewSelfPlayPosition) for item in positions):
            raise TypeError("self-play positions have the wrong type")
        outcome = self.episode.outcome
        if self.status == "completed" and (outcome is None or not outcome.terminated):
            raise ValueError("completed self-play result must have a terminal outcome")
        if self.status == "truncated" and (outcome is None or not outcome.truncated):
            raise ValueError("truncated self-play result must have a truncated outcome")
        if self.status == "failed" and (outcome is None or outcome.status != "failed"):
            raise ValueError("failed self-play result must have a failed outcome")
        object.__setattr__(self, "positions", positions)

    @property
    def replay_eligible(self) -> bool:
        return self.status == "completed" and self.episode.outcome is not None and self.episode.outcome.replay_eligible

    def replay_game(self) -> _ReplayGame:
        return _ReplayGame(
            episode=self.episode,
            positions=tuple(position.replay_position() for position in self.positions),
            seed=self.seed,
            model_hash=self.model_hash,
            config_hash=self.config_hash,
        )


def play_new_selfplay_episode(
    model,
    config: NewSelfPlayConfig,
    *,
    deadline: _Deadline,
    initial_episode: _Episode | None = None,
    evaluator=None,
) -> NewSelfPlayGame:
    """Run one bounded v1 self-play game using a shared deadline.

    A fresh :class:`NeuralEvaluator` is created for each call when no explicit
    evaluator is supplied.  Every selected move must come from a completed
    puct search; partial visits, deadline exhaustion and all actor errors end
    the game without inventing a replay label.
    """

    if not isinstance(config, NewSelfPlayConfig):
        raise TypeError("config must be NewSelfPlayConfig")
    if not isinstance(deadline, _Deadline):
        raise TypeError("self-play requires a shared project Deadline")
    if initial_episode is None:
        from .episode import start_episode

        episode = start_episode(max_game_plies=config.max_game_plies)
    elif isinstance(initial_episode, _Episode):
        episode = initial_episode
        if episode.finished:
            raise ValueError("self-play initial episode must be live")
    else:
        raise TypeError("initial_episode must be a quoridor-v1 Episode")
    if episode.max_game_plies != config.max_game_plies:
        raise ValueError("initial episode and self-play max_game_plies differ")
    if evaluator is None:
        from .search_evaluator import NeuralEvaluator

        evaluator = NeuralEvaluator(model)
    if not callable(evaluator):
        raise TypeError("self-play evaluator must be callable")
    positions: list[NewSelfPlayPosition] = []
    move_seed = config.seed
    while not episode.finished:
        try:
            deadline.check()
            search_config = _SearchConfig(
                simulations=config.search.simulations,
                c_puct=config.search.c_puct,
                mode="selfplay",
                root_noise=config.search.root_noise,
                temperature=config.search.temperature,
                seed=move_seed,
                dirichlet_alpha=config.search.dirichlet_alpha,
                dirichlet_fraction=config.search.dirichlet_fraction,
            )
            result = _run_search(episode, evaluator, search_config, deadline=deadline)
        except _BudgetExpired:
            stopped = _enforce_deadline(episode, deadline)
            if not stopped.finished:
                stopped = _interrupt_episode(episode, detail="self-play deadline exhausted")
            return NewSelfPlayGame(stopped, tuple(positions), "truncated", "time_budget", config.seed, config.model_hash, config.config_hash)
        except (_SearchError, ValueError, TypeError, RuntimeError) as error:
            failed = _fail_episode(episode, "numerical_error", detail=str(error) or error.__class__.__name__)
            return NewSelfPlayGame(failed, tuple(positions), "failed", "numerical_error", config.seed, config.model_hash, config.config_hash)
        if result.status != "completed" or result.action is None:
            reason = "time_budget" if result.status == "budget_exhausted" else "search_incomplete"
            stopped = _enforce_deadline(episode, deadline) if reason == "time_budget" else _interrupt_episode(episode, detail=f"search status={result.status}")
            return NewSelfPlayGame(stopped, tuple(positions), "truncated", reason, config.seed, config.model_hash, config.config_hash)
        action = result.action
        root_legal = result.nodes[0].legal_actions if result.nodes else ()
        if action not in root_legal:
            failed = _fail_episode(episode, "illegal_action", detail="search selected an illegal action")
            return NewSelfPlayGame(failed, tuple(positions), "failed", "illegal_action", config.seed, config.model_hash, config.config_hash)
        try:
            position = NewSelfPlayPosition(episode, action, result.visits, result.policy, result.root_value if result.root_value is not None else 0.0)
            next_episode = _advance_episode(episode, action, deadline=deadline)
        except (ValueError, TypeError, RuntimeError) as error:
            failed = _fail_episode(episode, "illegal_action", detail=str(error) or error.__class__.__name__)
            return NewSelfPlayGame(failed, tuple(positions), "failed", "illegal_action", config.seed, config.model_hash, config.config_hash)
        positions.append(position)
        episode = next_episode
        move_seed += 1
    outcome = episode.outcome
    if outcome is not None and outcome.status == "failed":
        return NewSelfPlayGame(episode, tuple(positions), "failed", outcome.reason, config.seed, config.model_hash, config.config_hash)
    if outcome is None or not outcome.terminated:
        reason = outcome.reason if outcome is not None and outcome.truncated else "ply_limit"
        return NewSelfPlayGame(episode, tuple(positions), "truncated", reason, config.seed, config.model_hash, config.config_hash)
    return NewSelfPlayGame(episode, tuple(positions), "completed", outcome.reason, config.seed, config.model_hash, config.config_hash)


# Short aliases used by the M4 trainer/CLI integration.
play_selfplay_episode = play_new_selfplay_episode
run_selfplay_episode = play_new_selfplay_episode
play_selfplay = play_new_selfplay_episode
SelfPlayV1Config = NewSelfPlayConfig
SelfPlayV1Position = NewSelfPlayPosition
SelfPlayV1Game = NewSelfPlayGame

__all__ += [
    "NewSelfPlayConfig", "NewSelfPlayPosition", "NewSelfPlayGame",
    "SelfPlayV1Config", "SelfPlayV1Position", "SelfPlayV1Game",
    "play_new_selfplay_episode", "play_selfplay_episode",
    "run_selfplay_episode", "play_selfplay",
]
