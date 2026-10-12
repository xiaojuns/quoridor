from __future__ import annotations

import random

import numpy as np
import pytest

from quoridor.arena_statistics import (
    ArenaGameResult,
    ScoringContractError,
    paired_bootstrap,
    summarize_arena,
)
from quoridor.project_runtime import BudgetExpired


def game(
    opening: str,
    leg: int,
    candidate_player: int,
    outcome: str | None,
    *,
    status: str = "completed",
    reason: str | None = None,
) -> ArenaGameResult:
    return ArenaGameResult(
        opening_id=opening,
        leg=leg,
        candidate_player=candidate_player,
        outcome=outcome,
        status=status,
        reason=reason,
    )


def pair(opening: str, first: str, second: str) -> tuple[ArenaGameResult, ...]:
    return (
        game(opening, 0, 0, first),
        game(opening, 1, 1, second),
    )


def test_draws_score_half_and_stay_in_the_denominator() -> None:
    # One all-draw pair plus one exchanged win/loss pair.
    stats = summarize_arena(pair("draw", "draw", "draw") + pair("split", "win", "loss"))

    assert stats.scheduled_games == 4
    assert stats.completed_games == 4
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.draws == 2
    assert stats.denominator == 4
    assert stats.score == pytest.approx(0.5)
    assert stats.pair_scores == pytest.approx((0.5, 0.5))
    assert stats.decision_eligible


def test_completed_draw_is_distinct_from_missing_or_truncated_outcome() -> None:
    stats = summarize_arena(
        pair("mixed", "draw", "win")
        + (
            game("cut", 0, 0, "loss"),
            game("cut", 1, 1, None, status="truncated", reason="deadline"),
        )
    )

    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.draws == 1
    assert stats.denominator == 3
    assert stats.score == pytest.approx(0.5)
    assert stats.scheduled_games == 4
    assert stats.completed_games == 3
    assert stats.truncated_games == 1
    assert stats.incomplete_pairs == 1
    assert not stats.decision_eligible
    assert not stats.promotion_eligible


@pytest.mark.parametrize("status, field", [("truncated", "truncated_games"), ("failed", "failed_games"), ("illegal", "illegal_games"), ("numerical", "numerical_games")])
def test_any_incomplete_or_error_record_blocks_promotion(status: str, field: str) -> None:
    stats = summarize_arena(
        pair("good", "win", "win")
        + (
            game("bad", 0, 0, "loss"),
            game("bad", 1, 1, None, status=status, reason="fixture"),
        ),
        expected_pairs=2,
    )

    assert getattr(stats, field) == 1
    assert stats.scheduled_games == 4
    assert stats.completed_games == 3
    # The completed leg from an incomplete pair remains in the diagnostic
    # denominator, but the whole comparison remains ineligible.
    assert stats.denominator == 3
    assert stats.incomplete_pairs == 1
    assert not stats.decision_eligible
    assert not stats.promotion_eligible


def test_missing_opening_pair_is_counted_without_reducing_scheduled_denominator() -> None:
    stats = summarize_arena(pair("only", "win", "win"), expected_pairs=2)

    assert stats.scheduled_games == 4
    assert stats.completed_games == 2
    assert stats.complete_pairs == 1
    assert stats.incomplete_pairs == 1
    assert stats.denominator == 2
    assert stats.score == 1.0
    assert not stats.decision_eligible


def test_pair_shape_and_side_swap_are_frozen_contract_errors() -> None:
    with pytest.raises(ScoringContractError, match="legs 0 and 1"):
        summarize_arena((game("x", 0, 0, "win"), game("x", 0, 1, "loss")))
    with pytest.raises(ScoringContractError, match="exchange candidate ownership"):
        summarize_arena((game("x", 0, 0, "win"), game("x", 1, 0, "loss")))
    with pytest.raises(ScoringContractError, match="more than two"):
        summarize_arena(pair("x", "win", "loss") + (game("x", 0, 0, "draw"),))


def test_pair_bootstrap_is_seeded_and_resamples_pairs_not_games() -> None:
    scores = (0.0, 0.5, 1.0)
    first = paired_bootstrap(scores, seed=17, samples=200)
    second = paired_bootstrap(iter(scores), seed=17, samples=200)

    assert first == second
    # Reproduce the exact percentile independently, including whole-pair draws.
    rng = random.Random(17)
    replicates = sorted(
        sum(scores[rng.randrange(len(scores))] for _ in scores) / len(scores)
        for _ in range(200)
    )
    expected = np.quantile(replicates, [0.025, 0.975], method="linear")
    assert first.lower == pytest.approx(expected[0])
    assert first.upper == pytest.approx(expected[1])


def test_pair_correlation_changes_confidence_despite_equal_game_score() -> None:
    # Both runs have 50% game score. Correlated pairs retain uncertainty;
    # every split pair is exactly .5 and therefore has a point interval.
    correlated = paired_bootstrap((0.0,) * 100 + (1.0,) * 100)
    split = paired_bootstrap((0.5,) * 200)
    assert correlated.lower < 0.5 < correlated.upper
    assert split.lower == split.upper == 0.5


def test_promotion_requires_both_score_and_strict_lower_bound() -> None:
    # Constant 0.75 pair scores make the confidence bound deterministic and
    # provide an independent positive promotion fixture.
    passing = []
    for index in range(200):
        passing.extend(pair(f"w{index}", "win", "draw"))
    stats = summarize_arena(passing, expected_pairs=200)

    assert stats.wins == 200
    assert stats.losses == 0
    assert stats.draws == 200
    assert stats.denominator == 400
    assert stats.score == pytest.approx(0.75)
    assert stats.lower_bound == pytest.approx(0.75)
    assert stats.decision_eligible
    assert stats.promotion_eligible

    # 0.55 is the exact score threshold, but the sampled lower bound does not
    # clear 0.50; promotion must remain false.
    threshold = []
    for index in range(110):
        threshold.extend(pair(f"tw{index}", "win", "win"))
    for index in range(90):
        threshold.extend(pair(f"tl{index}", "loss", "loss"))
    stats = summarize_arena(threshold, expected_pairs=200)
    assert stats.score == pytest.approx(0.55)
    assert stats.score >= 0.55
    assert stats.lower_bound <= 0.50
    assert not stats.promotion_eligible


def test_promotion_requires_the_frozen_two_hundred_pair_schedule() -> None:
    pairs = []
    for index in range(150):
        pairs.extend(pair(f"w{index}", "win", "win"))
    for index in range(49):
        pairs.extend(pair(f"l{index}", "loss", "loss"))
    stats = summarize_arena(pairs, expected_pairs=199)

    assert stats.complete_pairs == 199
    assert stats.decision_eligible
    assert not stats.promotion_eligible

    # A caller that omits the schedule can inspect a complete diagnostic, but
    # cannot accidentally turn it into formal promotion evidence.
    unrestricted = summarize_arena(pair("diag", "win", "draw"))
    assert unrestricted.decision_eligible
    assert not unrestricted.promotion_eligible


def test_exact_lower_bound_of_half_rejects_even_when_score_threshold_passes() -> None:
    # Sorted opening IDs deliberately bind the bootstrap input order.
    records = []
    for index in range(200):
        outcomes = ("loss", "loss") if index < 40 else ("win", "loss") if index < 140 else ("win", "win")
        records.extend(pair(f"opening-{index:03}", *outcomes))
    stats = summarize_arena(records, expected_pairs=200)
    assert stats.score == 0.55
    assert stats.lower_bound == 0.50
    assert stats.decision_eligible
    assert not stats.promotion_eligible


def test_score_threshold_is_inclusive_when_confidence_is_above_half() -> None:
    records = []
    for index in range(200):
        outcomes = ("win", "draw") if index < 40 else ("win", "loss")
        records.extend(pair(f"opening-{index:03}", *outcomes))
    stats = summarize_arena(records, expected_pairs=200)
    assert stats.score == 0.55
    assert stats.lower_bound > 0.50
    assert stats.promotion_eligible


def test_confidence_alone_cannot_replace_the_mean_score_gate() -> None:
    records = []
    for index in range(200):
        outcomes = ("win", "draw") if index < 32 else ("win", "loss")
        records.extend(pair(f"opening-{index:03}", *outcomes))
    stats = summarize_arena(records, expected_pairs=200)
    assert stats.score == 0.54
    assert stats.lower_bound > 0.50
    assert not stats.promotion_eligible


@pytest.mark.parametrize(
    "override",
    [
        {"bootstrap_seed": 7},
        {"bootstrap_samples": 100},
        {"bootstrap_confidence": 0.90},
        {"promotion_score": 0.0},
        {"promotion_lower_bound": 0.0},
    ],
)
def test_non_frozen_settings_never_allow_promotion(override: dict[str, object]) -> None:
    records = [record for index in range(200) for record in pair(f"{index:03}", "win", "win")]
    stats = summarize_arena(records, expected_pairs=200, **override)
    assert stats.score == stats.lower_bound == 1.0
    assert not stats.promotion_eligible


@pytest.mark.parametrize(
    "override",
    [
        {"bootstrap_seed": True}, {"bootstrap_seed": -1},
        {"bootstrap_seed": 2**63}, {"bootstrap_samples": False},
        {"bootstrap_samples": 0}, {"bootstrap_confidence": True},
        {"bootstrap_confidence": float("nan")},
        {"bootstrap_confidence": 1},
        {"promotion_score": float("nan")}, {"promotion_score": -0.1},
        {"promotion_lower_bound": float("inf")}, {"promotion_lower_bound": True},
    ],
)
@pytest.mark.parametrize("with_results", [False, True])
def test_settings_validation_is_identical_for_empty_and_nonempty_results(
    override: dict[str, object], with_results: bool,
) -> None:
    records = pair("one", "win", "win") if with_results else ()
    with pytest.raises(ScoringContractError):
        summarize_arena(records, **override)


@pytest.mark.parametrize("failure", [BudgetExpired("synthetic deadline"), KeyboardInterrupt()])
def test_bootstrap_propagates_deadline_and_interruption_within_resampling(failure) -> None:
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 7:
            raise failure

    with pytest.raises(type(failure)):
        summarize_arena(pair("pair", "win", "win"), check=check)
    assert calls == 7


def test_cancelled_bootstrap_fallback_retains_counts_without_false_confidence() -> None:
    records = [record for index in range(200) for record in pair(f"{index:03}", "win", "win")]
    stats = summarize_arena(records, expected_pairs=200, compute_bootstrap=False)
    assert stats.completed_games == stats.denominator == stats.wins == 400
    assert stats.complete_pairs == 200
    assert stats.score == 1.0
    assert stats.lower_bound is None
    assert not stats.decision_eligible
    assert not stats.promotion_eligible
    payload = stats.to_dict()
    assert payload["bootstrap"]["status"] == "not_computed"
    assert payload["bootstrap"]["lower"] is payload["bootstrap"]["upper"] is None
    assert payload["promotionGate"]["lowerBound"] is None
    assert not payload["promotionGate"]["confidencePassed"]
