"""Exact history and outcomes for ``quoridor-v1`` episodes.

Construct an episode with :func:`start_episode` and replay opening actions with
:func:`advance_episode`. Legacy ``game.State`` histories and result labels are
not accepted. The board engine knows neither repetition nor runtime limits.
This module is also independent of neural libraries and reads a caller-owned
deadline only at explicit operation boundaries.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

from . import engine


REPETITION_LIMIT = 3
DEFAULT_MAX_GAME_PLIES = 256
VALUE_PERSPECTIVE = "current-player-v1"


class EpisodeFinishedError(RuntimeError):
    """An action was requested after the episode had already ended."""


class Deadline(Protocol):
    """The timing fields supplied by ``project_runtime.Deadline``."""

    seconds: float

    @property
    def elapsed_seconds(self) -> float: ...


def _player(player: int) -> int:
    if type(player) is not int or player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return player


def _seconds(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} seconds must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} seconds must be finite") from error
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{label} seconds must be finite and {'positive' if positive else 'nonnegative'}")
    return number


@dataclass(frozen=True, slots=True)
class Outcome:
    """A completed result or an explicitly unlabelled stop.

    Only ``terminated`` outcomes have value targets. A missing winner is a draw
    only for ``terminated/repetition``; it never makes a truncation a draw.
    """

    status: str
    reason: str
    winner: int | None = None
    detail: str | None = None
    elapsed_seconds: float | None = None
    budget_seconds: float | None = None

    def __post_init__(self) -> None:
        reasons = {
            "terminated": {"goal", "repetition"},
            "truncated": {"ply_limit", "time_budget", "interrupted"},
            "failed": {"illegal_action", "numerical_error", "infrastructure_error"},
        }
        if (
            not isinstance(self.status, str)
            or not isinstance(self.reason, str)
            or self.status not in reasons
            or self.reason not in reasons[self.status]
        ):
            raise ValueError("invalid episode status/reason combination")
        if self.reason == "goal":
            _player(self.winner)
        elif self.winner is not None:
            raise ValueError("only a goal outcome can have a winner")
        if self.detail is not None and (not isinstance(self.detail, str) or not self.detail):
            raise ValueError("outcome detail must be a nonempty string")
        if (self.elapsed_seconds is None) != (self.budget_seconds is None):
            raise ValueError("deadline evidence requires both elapsed and budget seconds")
        for label, value in (("elapsed", self.elapsed_seconds), ("budget", self.budget_seconds)):
            if value is not None:
                _seconds(value, label)
        if self.budget_seconds is not None:
            if self.reason != "time_budget" or self.budget_seconds <= 0:
                raise ValueError("deadline evidence belongs to a positive time budget")
            if self.elapsed_seconds < self.budget_seconds:
                raise ValueError("time budget has not expired")

    @property
    def terminated(self) -> bool:
        return self.status == "terminated"

    @property
    def truncated(self) -> bool:
        return self.status == "truncated"

    @property
    def replay_eligible(self) -> bool:
        return self.terminated

    def value_for(self, to_play: int) -> float | None:
        """Return a label for the recorded player, never inferred from parity."""

        _player(to_play)
        if not self.terminated:
            return None
        if self.winner is None:
            return 0.0
        return 1.0 if to_play == self.winner else -1.0


@dataclass(frozen=True, slots=True)
class Episode:
    """An immutable board, complete ordered position history, and adjudication.

    ``history`` includes the starting position and every committed position.
    Construction validates every key and transition and rejects histories that
    continued past an earlier result. Replaying explicit opening actions keeps
    the original action provenance as well.
    """

    state: engine.BoardState
    history: tuple[str, ...]
    max_game_plies: int = DEFAULT_MAX_GAME_PLIES
    outcome: Outcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, engine.BoardState):
            raise TypeError("episodes require a quoridor-v1 BoardState")
        if type(self.max_game_plies) is not int or self.max_game_plies <= 0:
            raise ValueError("max_game_plies must be a positive integer")
        if (
            type(self.history) is not tuple
            or not self.history
            or any(not isinstance(key, str) or not key for key in self.history)
        ):
            raise ValueError("history must be a nonempty immutable tuple of position keys")
        if self.history[-1] != engine.position_key(self.state):
            raise ValueError("history must end with the current full position")
        counts: Counter[str] = Counter()
        for index, key in enumerate(self.history):
            historical_state = engine.board_from_position_key(key)
            if index:
                _validate_transition(self.history[index - 1], key)
            counts[key] += 1
            if index != len(self.history) - 1 and (
                counts[key] >= REPETITION_LIMIT
                or engine.goal_winner(historical_state) is not None
                or index >= self.max_game_plies
            ):
                raise ValueError("history continues beyond an already finished episode")
        if self.outcome is not None and not isinstance(self.outcome, Outcome):
            raise TypeError("outcome must be an Outcome or None")
        expected = _adjudicate(self.state, self.history, self.max_game_plies)
        if expected is not None and self.outcome != expected:
            raise ValueError("outcome contradicts the board/history adjudication")
        if expected is None and self.outcome is not None and (
            self.outcome.terminated or self.outcome.reason == "ply_limit"
        ):
            raise ValueError("a live board/history cannot have a completed result")

    @property
    def plies(self) -> int:
        return len(self.history) - 1

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    @property
    def repetition_count(self) -> int:
        return self.history.count(self.history[-1])

    @property
    def position_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(Counter(self.history)))

    @property
    def adjudication_key(self) -> tuple[tuple[str, ...], int, Outcome | None]:
        """A hashable cache key that cannot alias different episode histories."""

        return self.history, self.max_game_plies, self.outcome

    def value_targets(self, recorded_players: Iterable[int]) -> tuple[float, ...] | None:
        """Label all samples together, or exclude the entire unfinished episode."""

        players = tuple(_player(player) for player in recorded_players)
        if self.outcome is None or not self.outcome.replay_eligible:
            return None
        return tuple(self.outcome.value_for(player) for player in players)


@lru_cache(maxsize=4096)
def _validate_transition(before_key: str, after_key: str) -> None:
    """Use the engine to prove an adjacent history pair is one legal action."""

    before = engine.board_from_position_key(before_key)
    after = engine.board_from_position_key(after_key)
    candidates = [engine.encode_move(*after.pawns[before.to_play])]
    for orientation, added in (
        ("h", after.horizontal_walls - before.horizontal_walls),
        ("v", after.vertical_walls - before.vertical_walls),
    ):
        candidates.extend(engine.encode_wall(orientation, *wall) for wall in added)
    for action in candidates:
        try:
            actual = engine.apply_action(before, action)
        except (TypeError, ValueError):
            continue
        if actual == after:
            return
    raise ValueError("history positions are not connected by one legal action")


def _adjudicate(
    state: engine.BoardState, history: tuple[str, ...], max_game_plies: int
) -> Outcome | None:
    winner = engine.goal_winner(state)
    if winner is not None:
        return Outcome("terminated", "goal", winner=winner)
    if history.count(history[-1]) >= REPETITION_LIMIT:
        return Outcome("terminated", "repetition")
    if len(history) - 1 >= max_game_plies:
        return Outcome("truncated", "ply_limit")
    return None


def start_episode(
    state: engine.BoardState | None = None,
    *,
    max_game_plies: int = DEFAULT_MAX_GAME_PLIES,
) -> Episode:
    """Start from an explicit board, counting it once, with zero committed plies.

    For an opening prefix, call this at its initial board then advance each
    opening action. Starting only at its final board would lose its history.
    """

    state = engine.initial_state() if state is None else state
    if not isinstance(state, engine.BoardState):
        raise TypeError("episodes require a quoridor-v1 BoardState")
    if type(max_game_plies) is not int or max_game_plies <= 0:
        raise ValueError("max_game_plies must be a positive integer")
    history = (engine.position_key(state),)
    return Episode(state, history, max_game_plies, _adjudicate(state, history, max_game_plies))


def enforce_deadline(episode: Episode, deadline: Deadline) -> Episode:
    """Stop an unfinished episode at an expired operation boundary.

    Read elapsed time once so the recorded evidence is the actual observation.
    A completed result is not relabelled by a later deadline observation.
    """

    if episode.finished:
        return episode
    elapsed = _seconds(deadline.elapsed_seconds, "elapsed")
    budget = _seconds(deadline.seconds, "budget", positive=True)
    if elapsed < budget:
        return episode
    return replace(episode, outcome=Outcome(
        "truncated", "time_budget", elapsed_seconds=elapsed, budget_seconds=budget,
    ))


def advance_episode(
    episode: Episode,
    action: int,
    *,
    deadline: Deadline | None = None,
) -> Episode:
    """Commit one legal action with goal > repetition > ply-limit precedence.

    Invalid actions produce ``failed/illegal_action`` with the board and history
    unchanged. Deadline checks bracket the transition. A transition completed
    after its deadline keeps a real game result; an unfinished episode becomes
    unlabelled truncation. Future search/actors must also check their own work.
    """

    if episode.finished:
        raise EpisodeFinishedError("cannot advance a finished episode")
    if deadline is not None:
        episode = enforce_deadline(episode, deadline)
        if episode.finished:
            return episode
    try:
        state = engine.apply_action(episode.state, action)
    except (TypeError, ValueError) as error:
        return fail_episode(episode, "illegal_action", detail=str(error))
    history = (*episode.history, engine.position_key(state))
    result = Episode(
        state, history, episode.max_game_plies,
        _adjudicate(state, history, episode.max_game_plies),
    )
    return enforce_deadline(result, deadline) if deadline is not None else result


def interrupt_episode(episode: Episode, *, detail: str | None = None) -> Episode:
    """Record a caller-observed interruption without replacing an existing end."""

    if episode.finished:
        return episode
    return replace(episode, outcome=Outcome("truncated", "interrupted", detail=detail))


def fail_episode(episode: Episode, reason: str, *, detail: str) -> Episode:
    """Record an explicit error without inventing a winner or value labels."""

    if episode.finished:
        return episode
    return replace(episode, outcome=Outcome("failed", reason, detail=detail))
