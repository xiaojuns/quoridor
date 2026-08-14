from __future__ import annotations

import unittest

from quoridor.phase4f_failure_analysis import analyze_games


def game(*, won: bool, player: int, to_move: bool, value: float, action: int) -> dict[str, object]:
    return {
        "candidateWon": won,
        "candidatePlayer": player,
        "candidateToMove": to_move,
        "openingMoveNumber": 8,
        "initialPositionHash": '[1,0,8,4,0,4,9,9,[],[]]',
        "finalMoveNumber": 18,
        "terminationReason": "goal",
        "plies": [
            {
                "selectorRole": "candidate",
                "actionIndex": action,
                "selection": {"value": value},
            }
        ],
    }


class Phase4FFailureAnalysisTests(unittest.TestCase):
    def test_analysis_separates_outcomes_actions_and_pair_sweeps(self) -> None:
        first = game(won=False, player=0, to_move=True, value=0.75, action=81)
        second = game(won=False, player=1, to_move=False, value=-0.25, action=1)
        report = analyze_games([first, second], pairs=[[first, second]])

        self.assertEqual(report["overall"]["winRate"], 0.0)
        self.assertEqual(report["openingPairs"]["zeroCandidateWins"], 1)
        self.assertEqual(report["candidateDecisions"]["all"]["wallRate"], 0.5)
        self.assertEqual(
            report["candidateDecisions"]["losses"]["losingValueAtLeastHalf"],
            1,
        )
        self.assertEqual(report["firstCandidateValue"]["signAccuracy"], 0.5)

    def test_analysis_rejects_incomplete_pairs(self) -> None:
        sample = game(won=True, player=0, to_move=True, value=0.5, action=1)
        with self.assertRaisesRegex(Exception, "complete games and opening pairs"):
            analyze_games([sample], pairs=[[sample]])


if __name__ == "__main__":
    unittest.main()
