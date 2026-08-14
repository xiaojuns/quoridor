from __future__ import annotations

import json
import unittest
from pathlib import Path

from quoridor.config import load_experiment_config
from quoridor.phase4_budget import PHASE4F35_PILOT_STAGE
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4f35_stage_a2 import CONFIG, EXECUTION, INDEX, REPORT
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


WORKSPACE = Path(__file__).resolve().parents[2]


def _payload(path: Path) -> dict[str, object]:
    raw = (WORKSPACE / path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise AssertionError(f"not canonical: {path}")
    return value


def _self_hash(value: dict[str, object], field: str) -> str:
    core = dict(value)
    stored = core.pop(field)
    if stored != sha256_bytes(canonical_json_bytes(core)):
        raise AssertionError(f"bad self hash: {field}")
    return str(stored)


class Phase4F35StageA2Tests(unittest.TestCase):
    def test_preparation_is_cpu_only_and_does_not_authorize_pilot(self) -> None:
        report = _payload(REPORT)
        self.assertEqual(
            _self_hash(report, "reportSha256"),
            "21988e68132d6ca303f05b7d77ab2b3fa5b376bb5bbfdfcf5fff8c2922a6795c",
        )
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["cudaAccessed"])
        self.assertFalse(report["cudaPilotAuthorized"])
        self.assertFalse(report["formalStageA2TrainingAuthorized"])
        self.assertEqual(report["globalGpuHours"], 76.14802950652549)

    def test_index_is_complete_disjoint_and_has_real_capacity(self) -> None:
        index = _payload(INDEX)
        self.assertEqual(
            _self_hash(index, "indexSha256"),
            "f21368dd714cfdfd2861f566c78a3df5d286620c3296048347c6ceaf0d984c3d",
        )
        pools = index["pools"]
        flattened = [position for values in pools.values() for position in values]
        self.assertEqual(len(flattened), 27025)
        self.assertEqual(len(set(flattened)), 27025)
        self.assertEqual(set(flattened), set(range(27025)))
        self.assertGreaterEqual(min(len(values) for values in pools.values()), 1318)

    def test_pilot_freezes_stage_a2_sampler_and_one_generation(self) -> None:
        config = load_experiment_config(WORKSPACE / CONFIG)
        sampling = config.training.challenge_replay.stratified_sampling
        self.assertEqual(config.training.challenge_replay.fraction, 0.5)
        self.assertEqual(sampling.early_opening_depths, (8, 12))
        self.assertEqual(sampling.minimum_early_fraction, 0.6)
        self.assertEqual(sampling.balance_action_kind, ("move", "wall"))
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.training.minimum_learning_rate, 5e-6)
        execution = load_phase4_execution_contract(WORKSPACE / EXECUTION)
        self.assertEqual(execution.stage, PHASE4F35_PILOT_STAGE)
        self.assertEqual(execution.maximum_generations, 1)
        self.assertFalse(execution.formal_promotion)
        self.assertEqual(execution.stage_budget.baseline_global_hours, 76.14802950652549)
        self.assertEqual(execution.stage_budget.limit_hours, 2.0)


if __name__ == "__main__":
    unittest.main()
