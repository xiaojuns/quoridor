from __future__ import annotations

import math
import unittest

from quoridor.statistics import (
    PairedGameResult,
    ResultContractError,
    aggregate_opening_pairs,
    cluster_bootstrap_win_rate_interval,
    elo_from_win_rate,
    summarize_paired_results,
    win_rate,
)


def opening_pair(opening_id: str, candidate_wins: int) -> tuple[PairedGameResult, ...]:
    if candidate_wins == 2:
        winners = (0, 1)
    elif candidate_wins == 1:
        winners = (0, 0)
    elif candidate_wins == 0:
        winners = (1, 0)
    else:
        raise ValueError("candidate_wins must be 0, 1, or 2")
    return (
        PairedGameResult(opening_id, leg=0, candidate_player=0, winner=winners[0]),
        PairedGameResult(opening_id, leg=1, candidate_player=1, winner=winners[1]),
    )


class PairedStatisticsTests(unittest.TestCase):
    def test_summary_is_reproducible_and_input_order_independent(self) -> None:
        results = tuple(
            result
            for index, wins in enumerate((2, 1, 1, 0))
            for result in opening_pair(f"opening-{index}", wins)
        )
        first = summarize_paired_results(
            results,
            seed=1234,
            bootstrap_samples=2_000,
        )
        reordered = summarize_paired_results(
            reversed(results),
            seed=1234,
            bootstrap_samples=2_000,
        )

        self.assertEqual(first, reordered)
        self.assertEqual(first.opening_pairs, 4)
        self.assertEqual(first.games, 8)
        self.assertEqual(first.candidate_wins, 4)
        self.assertEqual(first.candidate_losses, 4)
        self.assertEqual(first.win_rate, 0.5)
        self.assertAlmostEqual(first.elo, 0.0)
        self.assertEqual(first.to_dict()["draws"], 0)
        self.assertEqual(
            first.to_dict()["bootstrap"]["method"],
            "opening-pair-cluster-percentile",
        )
        self.assertEqual(win_rate(results), 0.5)

    def test_bootstrap_resamples_whole_two_game_opening_clusters(self) -> None:
        interval = cluster_bootstrap_win_rate_interval(
            (0.0, 1.0),
            seed=77,
            bootstrap_samples=2_000,
            confidence=0.95,
        )
        self.assertEqual(interval.lower, 0.0)
        self.assertEqual(interval.upper, 1.0)

    def test_no_draw_and_complete_swapped_pair_contract_is_strict(self) -> None:
        with self.assertRaisesRegex(ResultContractError, "draws"):
            PairedGameResult("opening", leg=0, candidate_player=0, winner=None)  # type: ignore[arg-type]
        with self.assertRaises(ResultContractError):
            PairedGameResult("opening", leg=0, candidate_player=0, winner=True)
        with self.assertRaisesRegex(ResultContractError, "exactly two"):
            aggregate_opening_pairs(opening_pair("opening", 2)[:1])
        with self.assertRaisesRegex(ResultContractError, "legs 0 and 1"):
            aggregate_opening_pairs(
                (
                    PairedGameResult("opening", 0, 0, 0),
                    PairedGameResult("opening", 0, 1, 1),
                )
            )
        with self.assertRaisesRegex(ResultContractError, "exchange"):
            aggregate_opening_pairs(
                (
                    PairedGameResult("opening", 0, 0, 0),
                    PairedGameResult("opening", 1, 0, 1),
                )
            )

    def test_all_win_and_all_loss_elo_boundaries_are_finite(self) -> None:
        all_wins = tuple(
            result
            for index in range(3)
            for result in opening_pair(f"win-{index}", 2)
        )
        all_losses = tuple(
            result
            for index in range(3)
            for result in opening_pair(f"loss-{index}", 0)
        )
        winning = summarize_paired_results(
            all_wins,
            seed=9,
            bootstrap_samples=100,
        )
        losing = summarize_paired_results(
            all_losses,
            seed=9,
            bootstrap_samples=100,
        )

        self.assertEqual(winning.win_rate_ci.lower, 1.0)
        self.assertEqual(winning.win_rate_ci.upper, 1.0)
        self.assertEqual(losing.win_rate_ci.lower, 0.0)
        self.assertEqual(losing.win_rate_ci.upper, 0.0)
        self.assertTrue(math.isfinite(winning.elo))
        self.assertTrue(math.isfinite(winning.elo_ci.upper))
        self.assertTrue(math.isfinite(losing.elo))
        self.assertGreater(winning.elo, 0)
        self.assertLess(losing.elo, 0)
        self.assertAlmostEqual(winning.elo, -losing.elo)
        self.assertAlmostEqual(
            elo_from_win_rate(1.0, games=6),
            -elo_from_win_rate(0.0, games=6),
        )

    def test_invalid_bootstrap_and_elo_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cluster_bootstrap_win_rate_interval(())
        with self.assertRaises(ValueError):
            cluster_bootstrap_win_rate_interval((0.25, 0.75))
        with self.assertRaises(ValueError):
            cluster_bootstrap_win_rate_interval((0.5,), seed=-1)
        with self.assertRaises(ValueError):
            elo_from_win_rate(1.1, games=10)
        with self.assertRaises(ValueError):
            elo_from_win_rate(0.5, games=0)


if __name__ == "__main__":
    unittest.main()
