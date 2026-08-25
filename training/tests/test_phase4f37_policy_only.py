from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_budget import PHASE4F38_PILOT_STAGE
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f37_policy_only import (
    CONFIG,
    EXECUTION,
    F21_CANDIDATE_SHA256,
    F36_SOURCE_GIT_COMMIT,
    PILOT_SEED,
    REPORT,
    RUN_DIR,
    _base_model_sha256,
    _generation_data,
)
from quoridor.run_manifest import (
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    fingerprint_config,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4F37PolicyOnlyTests(unittest.TestCase):
    def test_generation_loader_preserves_signed_checkpoint_wrapper(self) -> None:
        _, generation = _generation_data(
            ROOT,
            Path("training/runs/phase4/phase4f/stage-a2-pilot-v2"),
            "945c2344b7679bbd758e03c4e7f20b2bf04e5ac344983bfed992e1e9eb403697",
            0,
            "953bd7b21363a2d6d2789516790958b91c22048f4c47554d9c44754ee8823753",
        )
        self.assertEqual(
            generation["checkpoint"]["candidateModel"]["sha256"],
            "94cc1c3655c62ce1e4f2e2bb99567b07ed569f5a2a6ba58d4df4f701f6cc2014",
        )

    def test_report_rejects_f36_on_frozen_holdout_gate(self) -> None:
        payload = json.loads((ROOT / REPORT).read_bytes())
        self.assertEqual((ROOT / REPORT).read_bytes(), canonical_json_bytes(payload))
        core = dict(payload)
        stored = core.pop("reportSha256")
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        review = payload["f36Review"]
        self.assertFalse(review["candidateEligible"])
        self.assertFalse(review["screenAuthorized"])
        self.assertGreater(review["comparison"]["overall"]["totalLoss"], 0.0)
        self.assertLess(
            review["comparison"]["byPlyBand"]["early"]["totalLoss"], 0.0
        )
        self.assertGreater(
            review["comparison"]["byPlyBand"]["late"]["valueLoss"], 0.0
        )

    def test_policy_only_pilot_is_isolated_and_uses_extended_cap(self) -> None:
        config = load_experiment_config(ROOT / CONFIG)
        execution = load_phase4_execution_contract(ROOT / EXECUTION)
        self.assertEqual(config.name, "phase4f38-stage-a2-policy-only-pilot-v2")
        self.assertEqual(config.run_dir, RUN_DIR)
        self.assertEqual(config.seed, PILOT_SEED)
        self.assertEqual(
            config.initialization.expected_base_model_sha256,
            _base_model_sha256(PILOT_SEED),
        )
        self.assertNotEqual(
            config.initialization.expected_base_model_sha256,
            _base_model_sha256(20260820),
        )
        self.assertEqual(config.artifact_limit_bytes, EXTENDED_ARTIFACT_LIMIT_BYTES)
        self.assertEqual(config.initialization.artifact_sha256, F21_CANDIDATE_SHA256)
        self.assertTrue(config.initialization.reset_optimizer)
        self.assertEqual(
            config.training.challenge_replay.value_target_mode,
            "self-play-only",
        )
        self.assertEqual(execution.stage, PHASE4F38_PILOT_STAGE)
        self.assertEqual(execution.maximum_generations, 1)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)

        report = json.loads((ROOT / REPORT).read_bytes())
        effective = replace(config, run_dir=(ROOT / config.run_dir).resolve())
        self.assertEqual(
            fingerprint_config(effective).sha256,
            report["pilot"]["semanticConfigSha256"],
        )

    def test_f38_launcher_binds_all_frozen_inputs_and_can_resume(self) -> None:
        launcher = (
            ROOT / "scripts/run-phase4f38-stage-a2-policy-only-pilot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('16 * 1024 * 1024 * 1024', launcher)
        self.assertIn('--replay-input-gate --config "${CONFIG}"', launcher)
        self.assertIn('--execution-contract "${EXECUTION}"', launcher)
        self.assertIn("RESUME_ARGS+=(--resume)", launcher)
        self.assertEqual(
            F36_SOURCE_GIT_COMMIT,
            "8cd9129a8946c0758edd6afd5d146f88bf1d0169",
        )


if __name__ == "__main__":
    unittest.main()
