from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_training import validate_phase4_config
from quoridor.phase4f_recovery_freeze import (
    FREEZE_REPORT,
    FROZEN_CHALLENGE_CONTRACT,
    REBASELINE_PREPARATION,
    RECOVERY_CONFIG,
)
from quoridor.phase4f5_rebaseline import _verify_phase4f4_history
from quoridor.run_manifest import (
    canonical_json_bytes,
    fingerprint_config,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


def _canonical_self_hashed(path: Path, field: str) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    core = dict(payload)
    stored = core.pop(field)
    if raw != canonical_json_bytes(payload):
        raise AssertionError(f"{path} is not canonical JSON")
    if stored != sha256_bytes(canonical_json_bytes(core)):
        raise AssertionError(f"{path} self hash mismatch")
    return payload


class Phase4FRecoveryFreezeTests(unittest.TestCase):
    def test_completed_contract_binds_the_immutable_dataset(self) -> None:
        contract = _canonical_self_hashed(
            ROOT / FROZEN_CHALLENGE_CONTRACT, "contractSha256"
        )
        self.assertEqual(contract["status"], "completed")
        self.assertTrue(contract["trainingUseAllowed"])
        root = ROOT / contract["datasetRoot"]
        self.assertEqual(
            contract["datasetManifestSha256"],
            sha256_file(root / "teacher-manifest.json"),
        )
        self.assertEqual(
            contract["trainManifestSha256"],
            sha256_file(root / "train" / "manifest.json"),
        )
        self.assertEqual(
            contract["datasetFingerprint"],
            "2a4864ad464e0448b405e4e8879b40c06249dc2bcea420fcabd204ebbd8b9e95",
        )

    def test_recovery_config_is_parseable_but_not_yet_authorized(self) -> None:
        config = load_experiment_config(ROOT / RECOVERY_CONFIG)
        validate_phase4_config(config)
        self.assertEqual(config.initialization.mode, "champion")
        self.assertEqual(config.mcts.simulations, 512)
        self.assertEqual(config.training.batch_size, 512)
        self.assertEqual(config.training.challenge_replay.fraction, 0.25)
        report = _canonical_self_hashed(ROOT / FREEZE_REPORT, "reportSha256")
        self.assertFalse(report["trainingAuthorized"])
        self.assertEqual(
            report["workload"]["semanticConfigSha256"],
            fingerprint_config(config).sha256,
        )

    def test_rebaseline_preparation_is_cpu_only_and_non_executable(self) -> None:
        payload = _canonical_self_hashed(
            ROOT / REBASELINE_PREPARATION, "rebaselineSha256"
        )
        self.assertEqual(payload["status"], "prepared")
        self.assertTrue(payload["cpuOnly"])
        self.assertFalse(payload["cudaAccessAllowed"])
        self.assertFalse(payload["trainingAuthorized"])

    def test_phase4f4_outputs_remain_valid_historical_evidence(self) -> None:
        evidence = _verify_phase4f4_history(ROOT)
        self.assertEqual(evidence["path"], FREEZE_REPORT.as_posix())
        self.assertEqual(
            evidence["recoveryConfigSemanticSha256"],
            fingerprint_config(load_experiment_config(ROOT / RECOVERY_CONFIG)).sha256,
        )


if __name__ == "__main__":
    unittest.main()
