from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from quoridor.phase4f45_curriculum_pilot_review import _is_frozen_key, loss_review, model_delta_audit


class Phase4F45ReviewTests(unittest.TestCase):
    def test_frozen_boundary_includes_input_and_first_eight_blocks(self) -> None:
        self.assertTrue(_is_frozen_key("input_conv.weight"))
        self.assertTrue(_is_frozen_key("input_bn.running_mean"))
        self.assertTrue(_is_frozen_key("trunk.7.bn2.num_batches_tracked"))
        self.assertFalse(_is_frozen_key("trunk.8.conv1.weight"))
        self.assertFalse(_is_frozen_key("policy_head.0.weight"))

    def test_delta_audit_rejects_frozen_change(self) -> None:
        initial = OrderedDict(
            [("input_conv.weight", torch.zeros(2)), ("trunk.8.conv1.weight", torch.zeros(2))]
        )
        candidate = OrderedDict(
            [("input_conv.weight", torch.ones(2)), ("trunk.8.conv1.weight", torch.ones(2))]
        )
        audit = model_delta_audit(initial, candidate)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["frozenChanged"], ["input_conv.weight"])

    def test_loss_review_requires_monotonic_early_and_overall_improvement(self) -> None:
        evaluations = []
        for index, step in enumerate((0, 8, 16, 24, 32)):
            factor = 1.0 - index * 0.04
            metrics = {
                band: {
                    "policyLoss": 1.0 * factor,
                    "valueLoss": 1.0 * factor,
                    "totalLoss": 2.0 * factor,
                }
                for band in ("all", "early", "middle", "late")
            }
            evaluations.append(
                {"step": step, "metrics": metrics, "eligible": False if step == 0 else True}
            )
        report = {"evaluations": evaluations, "selectedCheckpoint": {"step": 32}}
        self.assertTrue(loss_review(report)["passed"])


if __name__ == "__main__":
    unittest.main()
