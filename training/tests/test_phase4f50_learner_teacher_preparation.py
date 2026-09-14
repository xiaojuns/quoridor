from __future__ import annotations

import unittest

from quoridor.phase4f50_learner_teacher_preparation import preparation_checks


class Phase4F50PreparationTests(unittest.TestCase):
    def test_checks_require_budget_coupling_and_exclusion(self) -> None:
        evidence = {
            "gate": {"auditPassed": True},
            "classic": {
                "summary": {
                    "actualNegamaxRootScoresAvailable": True,
                    "budgetActionAgreement": 0.84,
                }
            },
            "mcts": {
                "summary": {
                    "withinModelConvergence": {
                        "F44": {"800": {"actionAgreementWith3200": 0.79}}
                    }
                }
            },
            "report": {
                "diagnosis": {
                    "measurements": {"f44MinusF21OutcomeMseAt3200": -0.03}
                },
                "screenTrajectoriesTrainingUseAllowed": False,
                "trainingStarted": False,
                "datasetGenerationStarted": False,
            },
        }
        self.assertTrue(all(preparation_checks(evidence).values()))

    def test_checks_reject_value_as_primary_failure(self) -> None:
        evidence = {
            "gate": {"auditPassed": True},
            "classic": {
                "summary": {
                    "actualNegamaxRootScoresAvailable": True,
                    "budgetActionAgreement": 0.84,
                }
            },
            "mcts": {
                "summary": {
                    "withinModelConvergence": {
                        "F44": {"800": {"actionAgreementWith3200": 0.79}}
                    }
                }
            },
            "report": {
                "diagnosis": {
                    "measurements": {"f44MinusF21OutcomeMseAt3200": 0.03}
                },
                "screenTrajectoriesTrainingUseAllowed": False,
                "trainingStarted": False,
                "datasetGenerationStarted": False,
            },
        }
        self.assertFalse(preparation_checks(evidence)["valueIsNotPrimaryRegressionSignal"])


if __name__ == "__main__":
    unittest.main()
