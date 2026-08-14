"""Paired-opening strength statistics with reproducible cluster bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


DEFAULT_BOOTSTRAP_SEED = 20_260_714
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
ELO_SCALE = 400.0


class ResultContractError(ValueError):
    """Raised when match results violate the no-draw paired-opening contract."""


@dataclass(frozen=True, slots=True)
class PairedGameResult:
    """One decisive game from a two-leg opening pair."""

    opening_id: str
    leg: int
    candidate_player: int
    winner: int

    def __post_init__(self) -> None:
        if not isinstance(self.opening_id, str) or not self.opening_id:
            raise ResultContractError("opening_id must be a non-empty string")
        if type(self.leg) is not int or self.leg not in (0, 1):
            raise ResultContractError("leg must be 0 or 1")
        if type(self.candidate_player) is not int or self.candidate_player not in (0, 1):
            raise ResultContractError("candidate_player must be 0 or 1")
        if type(self.winner) is not int or self.winner not in (0, 1):
            raise ResultContractError(
                "winner must be player 0 or 1; draws and missing winners are invalid"
            )

    @property
    def candidate_won(self) -> bool:
        return self.winner == self.candidate_player


@dataclass(frozen=True, slots=True)
class OpeningPairScore:
    opening_id: str
    candidate_wins: int

    @property
    def games(self) -> int:
        return 2

    @property
    def win_rate(self) -> float:
        return self.candidate_wins / self.games


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("confidence interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("confidence interval lower endpoint exceeds upper endpoint")
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PairedEvaluationStatistics:
    opening_pairs: int
    games: int
    candidate_wins: int
    candidate_losses: int
    win_rate: float
    win_rate_ci: ConfidenceInterval
    elo: float
    elo_ci: ConfidenceInterval
    bootstrap_seed: int
    bootstrap_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "openingPairs": self.opening_pairs,
            "games": self.games,
            "candidateWins": self.candidate_wins,
            "candidateLosses": self.candidate_losses,
            "draws": 0,
            "winRate": self.win_rate,
            "winRateCI": {
                "lower": self.win_rate_ci.lower,
                "upper": self.win_rate_ci.upper,
                "confidence": self.win_rate_ci.confidence,
            },
            "elo": self.elo,
            "eloCI": {
                "lower": self.elo_ci.lower,
                "upper": self.elo_ci.upper,
                "confidence": self.elo_ci.confidence,
            },
            "bootstrap": {
                "method": "opening-pair-cluster-percentile",
                "seed": self.bootstrap_seed,
                "samples": self.bootstrap_samples,
            },
            "eloBoundaryCorrection": "jeffreys-half-win-at-zero-or-one",
        }


def _validate_seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("bootstrap seed must be an integer in [0, 2^63)")
    return seed


def _validate_bootstrap_settings(
    samples: object,
    confidence: object,
) -> tuple[int, float]:
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return samples, confidence


def aggregate_opening_pairs(
    results: Iterable[PairedGameResult],
) -> tuple[OpeningPairScore, ...]:
    """Validate two exchanged-player legs and aggregate one score per opening."""

    try:
        normalized = tuple(results)
    except TypeError as error:
        raise ResultContractError("results must be iterable") from error
    if not normalized:
        raise ResultContractError("at least one opening pair is required")
    if any(not isinstance(result, PairedGameResult) for result in normalized):
        raise ResultContractError("every result must be a PairedGameResult")

    grouped: dict[str, list[PairedGameResult]] = defaultdict(list)
    for result in normalized:
        grouped[result.opening_id].append(result)

    pair_scores: list[OpeningPairScore] = []
    for opening_id in sorted(grouped):
        games = grouped[opening_id]
        if len(games) != 2:
            raise ResultContractError(
                f"opening {opening_id!r} must contain exactly two games"
            )
        if {game.leg for game in games} != {0, 1}:
            raise ResultContractError(
                f"opening {opening_id!r} must contain legs 0 and 1"
            )
        if {game.candidate_player for game in games} != {0, 1}:
            raise ResultContractError(
                f"opening {opening_id!r} must exchange candidate player ownership"
            )
        pair_scores.append(
            OpeningPairScore(
                opening_id=opening_id,
                candidate_wins=sum(game.candidate_won for game in games),
            )
        )
    return tuple(pair_scores)


def win_rate(results: Iterable[PairedGameResult]) -> float:
    """Return candidate wins / decisive games after validating complete pairs."""

    pairs = aggregate_opening_pairs(results)
    return sum(pair.candidate_wins for pair in pairs) / (2 * len(pairs))


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def cluster_bootstrap_win_rate_interval(
    pair_win_rates: Iterable[float],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConfidenceInterval:
    """Resample whole opening pairs, preserving correlation between their legs."""

    seed = _validate_seed(seed)
    bootstrap_samples, confidence = _validate_bootstrap_settings(
        bootstrap_samples, confidence
    )
    try:
        pair_scores = tuple(sorted(float(score) for score in pair_win_rates))
    except (TypeError, ValueError) as error:
        raise ValueError("pair_win_rates must be numeric") from error
    if not pair_scores:
        raise ValueError("at least one opening-pair score is required")
    if any(score not in (0.0, 0.5, 1.0) for score in pair_scores):
        raise ValueError("each two-leg opening-pair score must be 0, 0.5, or 1")

    rng = random.Random(seed)
    pair_count = len(pair_scores)
    replicates = []
    for _ in range(bootstrap_samples):
        total = 0.0
        for _ in range(pair_count):
            total += pair_scores[rng.randrange(pair_count)]
        replicates.append(total / pair_count)
    replicates.sort()

    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        lower=_quantile(replicates, tail),
        upper=_quantile(replicates, 1.0 - tail),
        confidence=confidence,
    )


def _boundary_corrected_probability(win_rate_value: float, games: int) -> float:
    """Keep Elo finite at an observed 0% or 100% without hiding the boundary."""

    if win_rate_value == 0.0:
        return 0.5 / (games + 1.0)
    if win_rate_value == 1.0:
        return (games + 0.5) / (games + 1.0)
    return win_rate_value


def elo_from_win_rate(win_rate_value: float, *, games: int) -> float:
    """Convert decisive-game win rate to Elo with finite endpoint correction."""

    if not isinstance(win_rate_value, (int, float)) or isinstance(win_rate_value, bool):
        raise ValueError("win_rate must be numeric")
    probability = float(win_rate_value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("win_rate must be in [0, 1]")
    if type(games) is not int or games <= 0:
        raise ValueError("games must be a positive integer")
    probability = _boundary_corrected_probability(probability, games)
    return ELO_SCALE * math.log10(probability / (1.0 - probability))


def summarize_paired_results(
    results: Iterable[PairedGameResult],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedEvaluationStatistics:
    """Calculate audited W/L, paired-bootstrap CI, and candidate Elo delta."""

    seed = _validate_seed(seed)
    bootstrap_samples, confidence = _validate_bootstrap_settings(
        bootstrap_samples, confidence
    )
    pairs = aggregate_opening_pairs(results)
    pair_win_rates = tuple(pair.win_rate for pair in pairs)
    games = 2 * len(pairs)
    candidate_wins = sum(pair.candidate_wins for pair in pairs)
    candidate_losses = games - candidate_wins
    observed_win_rate = candidate_wins / games
    win_rate_ci = cluster_bootstrap_win_rate_interval(
        pair_win_rates,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    elo_ci = ConfidenceInterval(
        lower=elo_from_win_rate(win_rate_ci.lower, games=games),
        upper=elo_from_win_rate(win_rate_ci.upper, games=games),
        confidence=confidence,
    )
    return PairedEvaluationStatistics(
        opening_pairs=len(pairs),
        games=games,
        candidate_wins=candidate_wins,
        candidate_losses=candidate_losses,
        win_rate=observed_win_rate,
        win_rate_ci=win_rate_ci,
        elo=elo_from_win_rate(observed_win_rate, games=games),
        elo_ci=elo_ci,
        bootstrap_seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE",
    "ELO_SCALE",
    "ResultContractError",
    "PairedGameResult",
    "OpeningPairScore",
    "ConfidenceInterval",
    "PairedEvaluationStatistics",
    "aggregate_opening_pairs",
    "win_rate",
    "cluster_bootstrap_win_rate_interval",
    "elo_from_win_rate",
    "summarize_paired_results",
]
