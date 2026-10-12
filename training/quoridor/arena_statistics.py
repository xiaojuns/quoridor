"""Frozen paired-arena scoring and confidence gates for Quoridor v1.

The legacy :mod:`quoridor.statistics` module describes the old decisive-only
arena.  M5 needs a separate contract: a completed repetition draw is a score
of one half, while a truncated or failed game has no score and makes its
opening pair incomplete.  This module contains only arithmetic and validation
so it can be used by an arena runner without importing the game engine.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


DEFAULT_BOOTSTRAP_SEED = 20_261_010
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
PROMOTION_SCORE = 0.55
PROMOTION_LOWER_BOUND = 0.50

_COMPLETED = "completed"
_OUTCOMES = frozenset({"win", "loss", "draw"})
_STATUSES = frozenset({"completed", "truncated", "failed", "illegal", "numerical"})


class ScoringContractError(ValueError):
    """Raised when result records cannot be interpreted under the M5 contract."""


@dataclass(frozen=True, slots=True)
class ArenaGameResult:
    """One scheduled game from a candidate's point of view.

    ``outcome`` is required only for a completed game.  Incomplete statuses
    intentionally carry no outcome: they are never converted into a loss or a
    win.  ``reason`` is retained for audit output, especially for truncation
    and numerical/illegal failures.
    """

    opening_id: str
    leg: int
    candidate_player: int
    outcome: str | None = None
    status: str = _COMPLETED
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.opening_id, str) or not self.opening_id:
            raise ScoringContractError("opening_id must be a non-empty string")
        if type(self.leg) is not int or self.leg not in (0, 1):
            raise ScoringContractError("leg must be 0 or 1")
        if type(self.candidate_player) is not int or self.candidate_player not in (0, 1):
            raise ScoringContractError("candidate_player must be 0 or 1")
        if self.status not in _STATUSES:
            raise ScoringContractError(f"unsupported game status: {self.status!r}")
        if self.status == _COMPLETED:
            if self.outcome not in _OUTCOMES:
                raise ScoringContractError(
                    "completed game must have outcome win, loss, or draw"
                )
        elif self.outcome is not None:
            raise ScoringContractError("incomplete game must not have an outcome")

    @property
    def score(self) -> float | None:
        if self.status != _COMPLETED:
            return None
        return {"win": 1.0, "loss": 0.0, "draw": 0.5}[self.outcome]  # type: ignore[index]

    @property
    def complete(self) -> bool:
        return self.status == _COMPLETED


# Short alias for callers that prefer the less verbose name.
GameResult = ArenaGameResult


@dataclass(frozen=True, slots=True)
class PairScore:
    """The mean score of two side-swapped games for one opening."""

    opening_id: str
    game_count: int
    completed_games: int
    score: float | None
    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.opening_id, str) or not self.opening_id:
            raise ScoringContractError("pair opening_id must be non-empty")
        for value, label in (
            (self.game_count, "game_count"),
            (self.completed_games, "completed_games"),
        ):
            if type(value) is not int or value < 0:
                raise ScoringContractError(f"{label} must be non-negative integer")
        if self.completed_games > self.game_count:
            raise ScoringContractError("completed_games exceeds game_count")
        if self.complete != (self.game_count == 2 and self.completed_games == 2):
            raise ScoringContractError("pair completeness does not match game counts")
        if self.score is not None and (
            not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0
        ):
            raise ScoringContractError("pair score must be finite and in [0, 1]")
        if self.complete and self.score is None:
            raise ScoringContractError("complete pair must have a score")


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile interval over whole opening-pair scores."""

    lower: float
    upper: float
    confidence: float
    seed: int
    samples: int
    method: str = "opening-pair-percentile-v1"

    def __post_init__(self) -> None:
        _validate_bootstrap_settings(self.seed, self.samples, self.confidence)
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ScoringContractError("bootstrap endpoints must be finite")
        if self.lower > self.upper:
            raise ScoringContractError("bootstrap lower endpoint exceeds upper endpoint")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Fixed promotion thresholds, reported separately from diagnostic scores."""

    decision_eligible: bool
    protocol_frozen: bool
    score: float
    lower_bound: float | None

    def __post_init__(self) -> None:
        if type(self.decision_eligible) is not bool or type(self.protocol_frozen) is not bool:
            raise ScoringContractError("gate eligibility flags must be booleans")
        _validate_threshold(self.score, "score")
        if self.lower_bound is not None:
            _validate_threshold(self.lower_bound, "lower_bound")

    @property
    def passed(self) -> bool:
        return (
            self.decision_eligible
            and self.protocol_frozen
            and self.score >= PROMOTION_SCORE
            and self.lower_bound is not None
            and self.lower_bound > PROMOTION_LOWER_BOUND
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "decisionEligible": self.decision_eligible,
            "protocolFrozen": self.protocol_frozen,
            "score": self.score,
            "scoreMinimum": PROMOTION_SCORE,
            "scorePassed": self.score >= PROMOTION_SCORE,
            "lowerBound": self.lower_bound,
            "lowerBoundExclusiveMinimum": PROMOTION_LOWER_BOUND,
            "confidencePassed": self.lower_bound is not None and self.lower_bound > PROMOTION_LOWER_BOUND,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ArenaStatistics:
    """Audited W/L/D counts, paired bootstrap, and promotion eligibility."""

    scheduled_games: int
    completed_games: int
    complete_pairs: int
    incomplete_pairs: int
    wins: int
    losses: int
    draws: int
    denominator: int
    score: float
    bootstrap: BootstrapInterval
    pair_scores: tuple[float, ...]
    truncated_games: int
    failed_games: int
    illegal_games: int
    numerical_games: int
    decision_eligible: bool
    promotion_eligible: bool
    expected_pairs: int | None
    gate: GateDecision
    bootstrap_complete: bool

    @property
    def lower_bound(self) -> float | None:
        return self.bootstrap.lower if self.bootstrap_complete else None

    @property
    def games(self) -> int:
        """Compatibility alias for completed game count in score reports."""

        return self.completed_games

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduledGames": self.scheduled_games,
            "completedGames": self.completed_games,
            "completePairs": self.complete_pairs,
            "incompletePairs": self.incomplete_pairs,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "denominator": self.denominator,
            "score": self.score,
            "pairScores": list(self.pair_scores),
            "truncatedGames": self.truncated_games,
            "failedGames": self.failed_games,
            "illegalGames": self.illegal_games,
            "numericalGames": self.numerical_games,
            "decisionEligible": self.decision_eligible,
            "promotionEligible": self.promotion_eligible,
            "promotionGate": self.gate.to_dict(),
            "expectedPairs": self.expected_pairs,
            "bootstrap": {
                "status": "completed" if self.bootstrap_complete else "not_computed",
                "completePairs": len(self.pair_scores),
                "lower": self.bootstrap.lower if self.bootstrap_complete else None,
                "upper": self.bootstrap.upper if self.bootstrap_complete else None,
                "confidence": self.bootstrap.confidence,
                "seed": self.bootstrap.seed,
                "samples": self.bootstrap.samples,
                "method": self.bootstrap.method,
            },
        }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ScoringContractError("cannot calculate quantile for empty values")
    if probability <= 0.0:
        return float(values[0])
    if probability >= 1.0:
        return float(values[-1])
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    fraction = position - low
    return float(values[low] * (1.0 - fraction) + values[high] * fraction)


def _validate_bootstrap_settings(seed: int, samples: int, confidence: float) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ScoringContractError("seed must be an integer in [0, 2^63)")
    if type(samples) is not int or samples <= 0:
        raise ScoringContractError("samples must be positive integer")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ScoringContractError("confidence must be numeric")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ScoringContractError("confidence must be in (0, 1)")


def _validate_threshold(value: float, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ScoringContractError(f"{label} must be finite numeric in [0, 1]")


def paired_bootstrap(
    pair_scores: Iterable[float],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    check: Callable[[], None] | None = None,
) -> BootstrapInterval:
    """Resample complete opening pairs and return a percentile interval.

    The frozen central 95% interval uses the 2.5th and 97.5th percentiles.
    Resampling pairs preserves the side-swapped pairing correlation.  Linear
    interpolation follows ``position=(samples-1)*probability``.
    """

    _validate_bootstrap_settings(seed, samples, confidence)
    if check is not None and not callable(check):
        raise ScoringContractError("check must be callable or None")
    confidence = float(confidence)
    try:
        normalized = tuple(float(score) for score in pair_scores)
    except (TypeError, ValueError) as error:
        raise ScoringContractError("pair scores must be numeric") from error
    if not normalized:
        raise ScoringContractError("at least one complete pair is required")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in normalized):
        raise ScoringContractError("pair scores must be finite and in [0, 1]")

    rng = random.Random(seed)
    count = len(normalized)
    replicates = []
    for _ in range(samples):
        if check is not None:
            check()
        replicates.append(sum(normalized[rng.randrange(count)] for _ in range(count)) / count)
    if check is not None:
        check()
    replicates.sort()
    if check is not None:
        check()
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        lower=_quantile(replicates, tail),
        upper=_quantile(replicates, 1.0 - tail),
        confidence=confidence,
        seed=seed,
        samples=samples,
    )


def _normalize_expected_pairs(expected_pairs: int | None) -> int | None:
    if expected_pairs is None:
        return None
    if type(expected_pairs) is not int or expected_pairs <= 0:
        raise ScoringContractError("expected_pairs must be a positive integer or None")
    return expected_pairs


def summarize_arena(
    results: Iterable[ArenaGameResult],
    *,
    expected_pairs: int | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    promotion_score: float = PROMOTION_SCORE,
    promotion_lower_bound: float = PROMOTION_LOWER_BOUND,
    check: Callable[[], None] | None = None,
    compute_bootstrap: bool = True,
) -> ArenaStatistics:
    """Aggregate an arena run and evaluate the frozen promotion gates.

    The denominator is exactly ``wins + losses + draws``.  Incomplete records
    stay visible in scheduled/incomplete/truncation counters but never enter
    that denominator. Completed legs from incomplete pairs remain in the
    diagnostic score; only complete pairs enter the bootstrap. Any incomplete
    pair makes ``decision_eligible`` false and prevents promotion.

    ``check`` may enforce a shared deadline or cancellation at each resample.
    After cancellation a caller may request ``compute_bootstrap=False`` without
    a check to preserve counts and diagnostic score. Such a summary has no
    confidence endpoints and cannot support a decision or promotion.
    """

    expected_pairs = _normalize_expected_pairs(expected_pairs)
    _validate_bootstrap_settings(bootstrap_seed, bootstrap_samples, bootstrap_confidence)
    _validate_threshold(promotion_score, "promotion_score")
    _validate_threshold(promotion_lower_bound, "promotion_lower_bound")
    if check is not None and not callable(check):
        raise ScoringContractError("check must be callable or None")
    if type(compute_bootstrap) is not bool:
        raise ScoringContractError("compute_bootstrap must be bool")
    if check is not None:
        check()
    try:
        normalized = tuple(results)
    except TypeError as error:
        raise ScoringContractError("results must be iterable") from error
    if any(not isinstance(result, ArenaGameResult) for result in normalized):
        raise ScoringContractError("every result must be an ArenaGameResult")
    groups: dict[str, list[ArenaGameResult]] = defaultdict(list)
    for result in normalized:
        groups[result.opening_id].append(result)
    if expected_pairs is not None and len(groups) > expected_pairs:
        raise ScoringContractError("observed more opening pairs than scheduled")

    pair_values: list[float] = []
    complete_pairs = 0
    incomplete_pairs = 0
    wins = losses = draws = 0
    completed_games = 0
    truncated_games = failed_games = illegal_games = numerical_games = 0

    for opening_id in sorted(groups):
        games = groups[opening_id]
        # A malformed pair is represented as incomplete when its records are
        # otherwise valid; duplicate legs/candidate sides are a contract error
        # because the schedule itself cannot be audited.
        if len(games) > 2:
            raise ScoringContractError(f"opening {opening_id!r} has more than two games")
        if len(games) == 2:
            if {game.leg for game in games} != {0, 1}:
                raise ScoringContractError(f"opening {opening_id!r} must contain legs 0 and 1")
            if {game.candidate_player for game in games} != {0, 1}:
                raise ScoringContractError(
                    f"opening {opening_id!r} must exchange candidate ownership"
                )
        complete = len(games) == 2 and all(game.complete for game in games)
        completed_games += sum(game.complete for game in games)
        truncated_games += sum(game.status == "truncated" for game in games)
        failed_games += sum(game.status == "failed" for game in games)
        illegal_games += sum(game.status == "illegal" for game in games)
        numerical_games += sum(game.status == "numerical" for game in games)
        # Completed legs remain useful as a diagnostic even if their mate is
        # incomplete.  They are counted in W/L/D and the partial denominator,
        # while only complete pairs enter the bootstrap and any eligible gate.
        for game in games:
            if not game.complete:
                continue
            if game.outcome == "win":
                wins += 1
            elif game.outcome == "loss":
                losses += 1
            else:
                draws += 1
        if complete:
            complete_pairs += 1
            pair_values.append(sum(game.score for game in games) / 2.0)  # type: ignore[arg-type]
        else:
            incomplete_pairs += 1

    if expected_pairs is not None:
        incomplete_pairs += expected_pairs - len(groups)
    scheduled_games = 2 * expected_pairs if expected_pairs is not None else len(normalized)
    denominator = wins + losses + draws
    score = (
        (wins + 0.5 * draws) / denominator if denominator else 0.0
    )
    interval = paired_bootstrap(
        pair_values,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        check=check,
    ) if pair_values and compute_bootstrap else BootstrapInterval(
        lower=0.0,
        upper=0.0,
        confidence=float(bootstrap_confidence),
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    if check is not None:
        check()
    bootstrap_complete = bool(pair_values and compute_bootstrap)
    schedule_complete = scheduled_games == len(normalized) == completed_games
    no_errors = not any((truncated_games, failed_games, illegal_games, numerical_games))
    decision_eligible = bool(
        bootstrap_complete and schedule_complete and no_errors and incomplete_pairs == 0
    )
    # Formal promotion is tied to the frozen 200-opening development suite;
    # callers that omit the expected schedule can obtain diagnostics only.
    expected_ok = (
        expected_pairs == 200
        and bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
        and bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        and bootstrap_confidence == DEFAULT_BOOTSTRAP_CONFIDENCE
        and promotion_score == PROMOTION_SCORE
        and promotion_lower_bound == PROMOTION_LOWER_BOUND
    )
    gate = GateDecision(decision_eligible, expected_ok, score, interval.lower if bootstrap_complete else None)
    promotion_eligible = gate.passed
    return ArenaStatistics(
        scheduled_games=scheduled_games,
        completed_games=completed_games,
        complete_pairs=complete_pairs,
        incomplete_pairs=incomplete_pairs,
        wins=wins,
        losses=losses,
        draws=draws,
        denominator=denominator,
        score=score,
        bootstrap=interval,
        pair_scores=tuple(pair_values),
        truncated_games=truncated_games,
        failed_games=failed_games,
        illegal_games=illegal_games,
        numerical_games=numerical_games,
        decision_eligible=decision_eligible,
        promotion_eligible=promotion_eligible,
        expected_pairs=expected_pairs,
        gate=gate,
        bootstrap_complete=bootstrap_complete,
    )


def promotion_gate(statistics: ArenaStatistics) -> bool:
    """Return the already-audited promotion decision."""

    if not isinstance(statistics, ArenaStatistics):
        raise TypeError("statistics must be ArenaStatistics")
    return statistics.promotion_eligible


__all__ = [
    "ArenaGameResult",
    "ArenaStatistics",
    "BootstrapInterval",
    "DEFAULT_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "GameResult",
    "GateDecision",
    "PairScore",
    "PROMOTION_LOWER_BOUND",
    "PROMOTION_SCORE",
    "ScoringContractError",
    "paired_bootstrap",
    "promotion_gate",
    "summarize_arena",
]
