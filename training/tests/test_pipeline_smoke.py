from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from quoridor.cli import run_smoke
from quoridor.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[2]


class PipelineSmokeTests(unittest.TestCase):
    def test_cpu_pipeline_runs_selfplay_update_resume_and_arena(self) -> None:
        config = load_experiment_config(ROOT / "training/configs/smoke.json")
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                config,
                run_dir=Path(directory),
                arena=replace(config.arena, games=2, simulations=2),
                mcts=replace(config.mcts, simulations=4),
            )
            summary = run_smoke(config, device_override="cpu")

            self.assertEqual(summary["device"], "cpu")
            self.assertEqual(summary["generation"], 0)
            self.assertEqual(summary["engine"], "python")
            self.assertEqual(
                summary["arena"]["championUpdated"],
                summary["arena"]["accepted"],
            )
            steps_per_run = config.training.training_steps
            self.assertEqual(summary["training"]["steps"], steps_per_run)
            self.assertEqual(len(summary["training"]["metrics"]), steps_per_run)
            self.assertGreater(summary["training"]["replaySize"], 0)
            self.assertEqual(summary["checkpoint"]["resumedStep"], steps_per_run)
            self.assertEqual(summary["selfPlay"]["illegalActions"], 0)
            self.assertEqual(
                summary["selfPlay"]["samples"],
                summary["selfPlay"]["moves"] * 2,
            )
            self.assertEqual(
                summary["arena"]["candidateWins"]
                + summary["arena"]["championWins"],
                2,
            )
            self.assertLess(
                summary["artifacts"]["bytes"],
                summary["artifacts"]["limitBytes"],
            )
            self.assertTrue((Path(directory) / "smoke-summary.json").exists())
            checkpoint_payload = torch.load(
                summary["checkpoint"]["path"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint_payload["extra"]["generation"], 0)
            self.assertEqual(
                checkpoint_payload["extra"]["championAccepted"],
                summary["arena"]["accepted"],
            )
            self.assertEqual(
                checkpoint_payload["extra"]["championModel"] is None,
                summary["arena"]["accepted"],
            )

            resumed = run_smoke(config, device_override="cpu", resume=True)
            self.assertEqual(resumed["generation"], 1)
            self.assertNotEqual(
                resumed["generationSeed"],
                summary["generationSeed"],
            )
            self.assertEqual(resumed["training"]["steps"], steps_per_run * 2)
            self.assertEqual(
                resumed["checkpoint"]["resumedStep"],
                steps_per_run * 2,
            )
            self.assertIsNotNone(resumed["checkpoint"]["resumedFrom"])
            self.assertGreater(
                resumed["training"]["replaySize"],
                summary["training"]["replaySize"],
            )


if __name__ == "__main__":
    unittest.main()
