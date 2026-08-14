from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from quoridor.config import load_experiment_config
from quoridor.phase4_budget import PHASE4F_STAGE_B_EXTENDED
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f31_stage_b import (
    CONFIG,
    EXECUTION,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_F30_GATE_SHA256,
    EXPECTED_REPLAY_MANIFEST_SHA256,
    MAXIMUM_GENERATIONS,
    PREPARATION_REPORT,
    SOURCE_REPLAY_GAMES,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes
from quoridor.training_memory_smoke import verify_staged_preparation


ROOT = Path(__file__).resolve().parents[2]


class Phase4F31StageBTests(unittest.TestCase):
    def test_preparation_binds_f30_and_exact_f27_state(self) -> None:
        report = json.loads((ROOT / PREPARATION_REPORT).read_bytes())
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        self.assertEqual(
            report["transitionEvidence"]["gateSha256"],
            EXPECTED_F30_GATE_SHA256,
        )
        continuation = report["continuation"]
        self.assertEqual(
            continuation["candidateModelSha256"], EXPECTED_CANDIDATE_SHA256
        )
        self.assertEqual(
            continuation["replayManifestSha256"],
            EXPECTED_REPLAY_MANIFEST_SHA256,
        )
        self.assertEqual(continuation["replayGames"], SOURCE_REPLAY_GAMES)
        self.assertEqual(continuation["nextGeneration"], 2)
        for field in (
            "optimizerInherited",
            "schedulerInherited",
            "rngInherited",
            "trainerStepInherited",
            "replayInherited",
        ):
            self.assertTrue(continuation[field])
        self.assertFalse(continuation["screenTrajectoriesImported"])

    def test_config_and_execution_freeze_four_additional_generations(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(execution.stage, PHASE4F_STAGE_B_EXTENDED)
        self.assertEqual(execution.maximum_generations, MAXIMUM_GENERATIONS)
        self.assertEqual(config.training.learning_stage, "pure-self-play")
        self.assertIsNone(config.training.challenge_replay)
        self.assertIsNotNone(config.training.health_holdout)
        self.assertFalse(config.initialization.reset_optimizer)
        self.assertEqual(config.initialization.recovery_generation, 1)
        self.assertEqual(config.initialization.recovery_step, 192)

    def test_memory_smoke_accepts_frozen_extended_stage(self) -> None:
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        frozen_preparation = {
            "execution": {"path": EXECUTION.as_posix()},
            "formalExtendedStageBTrainingAuthorized": True,
        }
        with patch(
            "quoridor.phase4f31_stage_b._verified_preparation",
            return_value=frozen_preparation,
        ):
            prepared, evidence_kind, evidence_path = verify_staged_preparation(
                ROOT,
                execution,
            )
        self.assertEqual(
            evidence_kind,
            "phase4f31-stage-b-extended-preparation",
        )
        self.assertEqual(evidence_path, str(prepared["execution"]["path"]))
        self.assertTrue(prepared["formalExtendedStageBTrainingAuthorized"])


if __name__ == "__main__":
    unittest.main()
