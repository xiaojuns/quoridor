from __future__ import annotations

import unittest

from quoridor.phase4f41_target_protocol import (
    VALUE_NEUTRAL_WEIGHT,
    VALUE_OUTCOME_WEIGHT,
    VALUE_PATH_TEMPO_WEIGHT,
    VALUE_SEARCH_WEIGHT,
    corrected_target,
    fit_policy_temperature,
    fit_regularized_isotonic,
    isotonic_predict,
)


class Phase4F41TargetProtocolTests(unittest.TestCase):
    def test_value_weights_include_explicit_neutral_shrinkage(self) -> None:
        self.assertAlmostEqual(
            VALUE_OUTCOME_WEIGHT
            + VALUE_SEARCH_WEIGHT
            + VALUE_PATH_TEMPO_WEIGHT
            + VALUE_NEUTRAL_WEIGHT,
            1.0,
        )

    def test_regularized_isotonic_is_monotone_and_shrunk(self) -> None:
        rows = []
        for index in range(128):
            rows.append(
                {
                    "split": "train",
                    "mcts": {"rootValue": -1.0 + 2.0 * index / 127.0},
                    "outcome": -1.0 if index < 64 else 1.0,
                }
            )
        blocks = fit_regularized_isotonic(rows)
        predictions = [isotonic_predict(blocks, -1.0 + index / 10.0) for index in range(21)]
        self.assertEqual(predictions, sorted(predictions))
        self.assertLess(max(abs(value) for value in predictions), 1.0)

    def test_policy_temperature_fits_final_blend(self) -> None:
        rows = [
            {
                "split": "train",
                "classicCandidates": [
                    {"actionIndex": 1, "score": 10.0},
                    {"actionIndex": 2, "score": 0.0},
                ],
                "mcts": {"visitCounts": [[1, 800]], "rootValue": 0.0},
                "teacherActionIndex": 1,
            }
            for _ in range(20)
        ]
        temperature, metrics = fit_policy_temperature(rows)
        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(metrics["top1ProbabilityP95"], 0.875)

    def test_corrected_target_is_bounded_and_normalized(self) -> None:
        row = {
            "split": "train",
            "classicCandidates": [
                {"actionIndex": 1, "score": 2.0},
                {"actionIndex": 2, "score": 1.0},
            ],
            "mcts": {"visitCounts": [[1, 3], [3, 1]], "rootValue": 0.5},
            "outcome": 1.0,
            "pathTempo": -0.5,
            "teacherActionIndex": 1,
        }
        blocks = [
            {
                "minimum": -1.0,
                "maximum": 1.0,
                "weight": 1.0,
                "value": 0.4,
            }
        ]
        policy, value, components = corrected_target(row, 10.0, blocks)
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertEqual(set(policy), {1, 2, 3})
        self.assertAlmostEqual(value, 0.31)
        self.assertEqual(components["regularizedSearch"], 0.4)


if __name__ == "__main__":
    unittest.main()
