from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from quoridor.config import DEFAULT_ARTIFACT_LIMIT_BYTES, load_experiment_config
from quoridor.phase4_execution import (
    PHASE4_BUDGET_STATUS_PENDING,
    PHASE4_EXECUTION_STATUS_PREPARED,
    Phase4ExecutionContractError,
    load_phase4_execution_contract,
)
from quoridor.phase4_training import validate_phase4_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "training/configs"
SCRIPTS_ROOT = ROOT / "scripts"
EXECUTION_CONTRACT = (
    CONFIG_ROOT / "phase4e-execution-prepared-v1.json"
)
BOOTSTRAP_CONFIG = CONFIG_ROOT / "phase4e-bootstrap-v1.json"
STANDARD_CONFIG = CONFIG_ROOT / "phase4e-standard-v1.json"


class Phase4EOpsTests(unittest.TestCase):
    def test_prepared_contract_is_versioned_and_cannot_authorize_training(
        self,
    ) -> None:
        contract = load_phase4_execution_contract(EXECUTION_CONTRACT)

        self.assertEqual(contract.stage, "phase4e")
        self.assertEqual(contract.status, PHASE4_EXECUTION_STATUS_PREPARED)
        self.assertEqual(
            contract.stage_budget.status,
            PHASE4_BUDGET_STATUS_PENDING,
        )
        self.assertEqual(contract.maximum_generations, 64)
        self.assertTrue(contract.formal_promotion)
        self.assertFalse(contract.budget_frozen)
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "pending",
        ):
            contract.require_frozen_budget()

    def test_bootstrap_and_standard_profiles_match_the_preparation_contract(
        self,
    ) -> None:
        contract = load_phase4_execution_contract(EXECUTION_CONTRACT)
        bootstrap = load_experiment_config(BOOTSTRAP_CONFIG)
        standard = load_experiment_config(STANDARD_CONFIG)
        validate_phase4_config(bootstrap, contract)
        validate_phase4_config(standard, contract)

        for config in (bootstrap, standard):
            self.assertEqual(config.pipeline, "phase4")
            self.assertEqual(config.device, "cuda")
            self.assertEqual(
                config.artifact_limit_bytes,
                DEFAULT_ARTIFACT_LIMIT_BYTES,
            )
            self.assertEqual(config.network.input_channels, 16)
            self.assertEqual(config.network.channels, 192)
            self.assertEqual(config.network.residual_blocks, 16)
            self.assertEqual(config.training.batch_size, 512)
            self.assertEqual(config.initialization.mode, "random")
            self.assertTrue(config.initialization.reset_optimizer)

        self.assertEqual(bootstrap.name, "phase4e-bootstrap-v1")
        self.assertEqual(bootstrap.self_play.games, 64)
        self.assertEqual(bootstrap.self_play.actors, 32)
        self.assertEqual(bootstrap.mcts.simulations, 256)
        self.assertEqual(bootstrap.training.replay_capacity, 100_000)
        self.assertEqual(bootstrap.training.training_steps, 64)

        self.assertEqual(standard.name, "phase4e-standard-v1")
        self.assertEqual(standard.self_play.games, 128)
        self.assertEqual(standard.self_play.actors, 64)
        self.assertEqual(standard.mcts.simulations, 512)
        self.assertEqual(standard.training.replay_capacity, 300_000)
        self.assertEqual(standard.training.training_steps, 96)
        self.assertEqual(standard.arena.games, 400)
        self.assertEqual(standard.arena.simulations, 800)

    def test_frozen_phase4d_bootstrap_config_was_not_changed(self) -> None:
        digest = hashlib.sha256(
            (CONFIG_ROOT / "phase4-bootstrap.json").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "08d884616da9b8d6ee19c3d6cf29c15100aad61aa624499e2ec8a955eb38a887",
        )

    def test_launchers_have_valid_shell_syntax_and_help(self) -> None:
        for name in ("prepare-phase4e.sh", "train-phase4e.sh"):
            path = SCRIPTS_ROOT / name
            self.assertTrue(os.access(path, os.X_OK))
            syntax = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            help_result = subprocess.run(
                [str(path), "--help"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("Usage:", help_result.stdout)

    def test_default_training_launcher_fails_closed_before_native_build(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_marker = Path(directory) / "native-build"
            environment = dict(os.environ)
            environment["QUORIDOR_NATIVE_BUILD_DIR"] = str(build_marker)
            result = subprocess.run(
                [str(SCRIPTS_ROOT / "train-phase4e.sh")],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not authorize training", result.stderr)
            self.assertIn("pending", result.stderr)
            self.assertFalse(build_marker.exists())
            self.assertNotIn("Configuring done", result.stdout)
            self.assertNotIn("[phase4e-training] command:", result.stdout)


if __name__ == "__main__":
    unittest.main()
