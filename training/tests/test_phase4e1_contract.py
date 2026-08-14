from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quoridor.phase4e1_contract import (
    ARENA_ACTORS,
    ARENA_GAMES,
    ARENA_OPENING_PREFIX_COUNT,
    ARENA_SIMULATIONS,
    ARTIFACT_LIMIT_BYTES,
    CALIBRATION_GPU_HOUR_LIMIT,
    FREEZE_MAXIMUM_GENERATIONS,
    FREEZE_MINIMUM_STAGE_GPU_HOURS,
    FREEZE_RESERVE_GPU_HOURS,
    GLOBAL_GPU_HOUR_LIMIT,
    INITIAL_GLOBAL_GPU_HOURS,
    MINIMUM_FREE_DISK_BYTES,
    PHASE4E1_CONTRACT_KIND,
    PHASE4E1_STAGE,
    PHASE4E1_STATUS,
    SELFPLAY_ACTORS,
    SELFPLAY_GAMES,
    SELFPLAY_SIMULATIONS,
    Phase4E1CalibrationContract,
    Phase4E1ContractError,
    load_phase4e1_contract,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "training/configs/phase4e1-calibration-v1.json"


def _payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class Phase4E1CalibrationContractTests(unittest.TestCase):
    def test_checked_in_contract_round_trips_and_is_non_authorizing(self) -> None:
        contract = load_phase4e1_contract(CONTRACT_PATH)

        self.assertEqual(contract.to_payload(), _payload())
        self.assertEqual(contract.kind, PHASE4E1_CONTRACT_KIND)
        self.assertEqual(contract.stage, PHASE4E1_STAGE)
        self.assertEqual(contract.status, PHASE4E1_STATUS)
        self.assertFalse(contract.formal_promotion)
        self.assertFalse(contract.training_allowed)
        self.assertEqual(
            contract.sha256,
            sha256_bytes(canonical_json_bytes(contract.to_payload())),
        )
        self.assertEqual(contract.canonical_sha256, contract.sha256)

    def test_resources_and_freeze_bounds_are_frozen(self) -> None:
        contract = load_phase4e1_contract(CONTRACT_PATH)

        self.assertEqual(
            contract.resources.global_gpu_hour_limit,
            GLOBAL_GPU_HOUR_LIMIT,
        )
        self.assertEqual(
            contract.resources.calibration_gpu_hour_limit,
            CALIBRATION_GPU_HOUR_LIMIT,
        )
        self.assertEqual(
            contract.resources.initial_ledger.total_gpu_hours,
            INITIAL_GLOBAL_GPU_HOURS,
        )
        self.assertEqual(
            contract.resources.artifact_limit_bytes,
            ARTIFACT_LIMIT_BYTES,
        )
        self.assertEqual(
            contract.resources.minimum_free_disk_bytes,
            MINIMUM_FREE_DISK_BYTES,
        )
        self.assertEqual(
            contract.freeze.reserve_gpu_hours,
            FREEZE_RESERVE_GPU_HOURS,
        )
        self.assertEqual(
            contract.freeze.minimum_stage_gpu_hours,
            FREEZE_MINIMUM_STAGE_GPU_HOURS,
        )
        self.assertEqual(
            contract.freeze.maximum_generations,
            FREEZE_MAXIMUM_GENERATIONS,
        )
        committed = (
            contract.resources.initial_ledger.total_gpu_hours
            + contract.resources.calibration_gpu_hour_limit
            + contract.freeze.reserve_gpu_hours
            + contract.freeze.minimum_stage_gpu_hours
        )
        self.assertLessEqual(committed, contract.resources.global_gpu_hour_limit)

    def test_standard_and_arena_workloads_are_exact(self) -> None:
        contract = load_phase4e1_contract(CONTRACT_PATH)

        self.assertEqual(contract.self_play.games, SELFPLAY_GAMES)
        self.assertEqual(contract.self_play.actors, SELFPLAY_ACTORS)
        self.assertEqual(contract.self_play.simulations, SELFPLAY_SIMULATIONS)
        self.assertEqual(contract.self_play.temperature_moves, 24)
        self.assertEqual(contract.self_play.channels, 192)
        self.assertEqual(contract.self_play.residual_blocks, 16)

        self.assertEqual(
            contract.arena.opening_prefix_count,
            ARENA_OPENING_PREFIX_COUNT,
        )
        self.assertEqual(contract.arena.games, ARENA_GAMES)
        self.assertEqual(contract.arena.actors, ARENA_ACTORS)
        self.assertEqual(contract.arena.simulations, ARENA_SIMULATIONS)
        self.assertEqual(
            contract.arena.games,
            contract.arena.opening_prefix_count * 2,
        )
        self.assertEqual(
            contract.arena.candidate_model,
            contract.arena.opponent_model,
        )
        self.assertTrue(contract.arena.require_actor_refill)
        self.assertGreater(contract.arena.games, contract.arena.actors)

    def test_exact_field_schema_rejects_top_level_and_nested_tamper(self) -> None:
        payload = _payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(Phase4E1ContractError, "fields are invalid"):
            Phase4E1CalibrationContract.from_dict(payload)

        payload = _payload()
        payload["workloads"]["arena"]["unexpected"] = True
        with self.assertRaisesRegex(Phase4E1ContractError, "fields are invalid"):
            Phase4E1CalibrationContract.from_dict(payload)

        payload = _payload()
        payload["provenance"]["bootstrapConfig"].pop("semanticSha256")
        with self.assertRaisesRegex(Phase4E1ContractError, "fields are invalid"):
            Phase4E1CalibrationContract.from_dict(payload)

    def test_provenance_and_identity_tamper_fail_closed(self) -> None:
        for path, replacement in (
            (
                ("resources", "initialLedger", "ledgerSha256"),
                "0" * 64,
            ),
            (
                ("provenance", "preparedExecutionContract", "canonicalSha256"),
                "1" * 64,
            ),
            (
                ("provenance", "bootstrapBaseline", "modelFileSha256"),
                "2" * 64,
            ),
            (
                ("workloads", "arena", "suiteHash"),
                f"sha256:{'3' * 64}",
            ),
        ):
            payload = _payload()
            target = payload
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaisesRegex(
                Phase4E1ContractError,
                "frozen contract",
            ):
                Phase4E1CalibrationContract.from_dict(payload)

    def test_paths_must_be_canonical_workspace_relative(self) -> None:
        for invalid in (
            "/tmp/phase4e.json",
            "../phase4e.json",
            "training/../phase4e.json",
            r"training\phase4e.json",
        ):
            payload = _payload()
            payload["provenance"]["preparedExecutionContract"]["path"] = invalid
            with self.subTest(path=invalid), self.assertRaisesRegex(
                Phase4E1ContractError,
                "workspace-relative",
            ):
                Phase4E1CalibrationContract.from_dict(payload)

    def test_invalid_numeric_bounds_fail_before_contract_freeze(self) -> None:
        mutations = (
            (
                ("resources", "calibrationGpuHourLimit"),
                0,
                "positive",
            ),
            (
                ("freeze", "maximumGenerations"),
                0,
                "positive integer",
            ),
            (
                ("workloads", "arena", "openingPrefixCount"),
                41,
                "two legs",
            ),
            (
                ("workloads", "arena", "actors"),
                80,
                "more games than actors",
            ),
        )
        for path, replacement, message in mutations:
            payload = _payload()
            target = payload
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaisesRegex(
                Phase4E1ContractError,
                message,
            ):
                Phase4E1CalibrationContract.from_dict(payload)

    def test_loader_rejects_non_object_and_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(Phase4E1ContractError, "must be an object"):
                load_phase4e1_contract(path)

            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(Phase4E1ContractError, "cannot load"):
                load_phase4e1_contract(path)


if __name__ == "__main__":
    unittest.main()
