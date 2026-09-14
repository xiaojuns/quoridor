from __future__ import annotations

import unittest
import shutil

from quoridor.classic_bridge import ClassicAIBridge, ClassicSearchBudget
from quoridor.game import create_initial_state, legal_action_indices
from quoridor.phase4f49_teacher_search_audit import diagnose, summarize_classic, summarize_mcts


def _mcts_row(action: int, value: float, visits: tuple[tuple[int, int], ...]) -> dict[str, object]:
    return {"actionIndex": action, "rootValue": value, "visits": [list(item) for item in visits]}


class Phase4F49AuditTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for bridge integration")
    def test_python_bridge_validates_complete_root_vector(self) -> None:
        state = create_initial_state()
        with ClassicAIBridge(request_timeout=30.0) as bridge:
            result = bridge.analyze(
                state, budget=ClassicSearchBudget.nodes(587), timeout=30.0
            )
        self.assertEqual(result.nodes, 587)
        self.assertEqual(
            {item.action_index for item in result.root_scores},
            set(legal_action_indices(state)),
        )
        self.assertTrue(any(item.searched for item in result.root_scores))
        self.assertTrue(any(not item.searched for item in result.root_scores))

    def test_classic_summary_requires_actual_root_format(self) -> None:
        def budget(action: int) -> dict[str, object]:
            return {
                "format": "legal-action-root-scores-v1",
                "rootScoreKind": "alpha-beta-return-v1",
                "actionIndex": action,
                "depth": 2,
                "legalActions": 3,
                "searchedActions": 2,
                "rootScores": [
                    {"actionIndex": 1, "searched": True, "score": 3.0},
                    {"actionIndex": 2, "searched": True, "score": 1.0},
                    {"actionIndex": 3, "searched": False, "score": None},
                ],
            }

        summary = summarize_classic([{"budgets": {"587": budget(1), "4096": budget(1)}}])
        self.assertTrue(summary["actualNegamaxRootScoresAvailable"])
        self.assertEqual(summary["budgetActionAgreement"], 1.0)
        self.assertAlmostEqual(summary["meanSearchedLegalActionCoverage"]["587"], 2 / 3)

    def test_mcts_summary_pairs_models_and_simulation_budgets(self) -> None:
        rows = {}
        for simulations in (100, 400, 800, 3200):
            rows[("F21", simulations)] = [_mcts_row(1, 0.25, ((1, simulations),))]
            rows[("F44", simulations)] = [_mcts_row(2, -0.25, ((2, simulations),))]
        summary = summarize_mcts(rows, [{"outcome": 1.0}])
        self.assertEqual(summary["bySimulation"]["3200"]["actionAgreement"], 0.0)
        self.assertEqual(summary["withinModelConvergence"]["F21"]["800"]["actionAgreementWith3200"], 1.0)
        diagnosis = diagnose(
            {"budgetActionAgreement": 0.75}, summary
        )
        self.assertEqual(diagnosis["primaryCause"], "persistent-network-policy-shift")


if __name__ == "__main__":
    unittest.main()
