from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from quoridor import phase4e1_calibration as calibration
from quoridor.codec import ACTION_SPACE_SIZE
from quoridor.phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    load_phase4_execution_contract,
)
from quoridor.phase4e1_contract import (
    Phase4E1CalibrationContract,
    load_phase4e1_contract,
)
from quoridor.replay_store import ReplayGame, ReplayPly, ReplayVerification
from quoridor.run_manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "training/configs/phase4e1-calibration-v1.json"
SOURCE_SHA256 = "1" * 64
NATIVE_SHA256 = "2" * 64
BOOTSTRAP_WORKLOAD_SHA256 = "3" * 64
STANDARD_WORKLOAD_SHA256 = "4" * 64
RUN_MANIFEST_SHA256 = "5" * 64
BASELINE_EVIDENCE = {"status": "verified", "modelSha256": "6" * 64}


def _run_id(contract: Phase4E1CalibrationContract) -> str:
    return f"phase4e1-calibration-{contract.sha256[:16]}"


def _write_component(
    workspace: Path,
    relative_path: str,
    *,
    kind: str,
) -> Path:
    path = workspace / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration._self_hashed(
        {
            "schemaVersion": calibration.COMPONENT_SCHEMA_VERSION,
            "kind": kind,
            "status": "completed",
        },
        "reportSha256",
    )
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _write_final_report(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    *,
    global_gpu_hours: float = 9.25,
    approved_workloads: list[str] | None = None,
) -> tuple[Path, dict[str, object]]:
    run_id = _run_id(contract)
    run_relative = f"{calibration.DEFAULT_OUTPUT_ROOT.as_posix()}/{run_id}"
    selfplay_path = _write_component(
        workspace,
        f"{run_relative}/{calibration.SELFPLAY_REPORT_NAME}",
        kind=calibration.SELFPLAY_KIND,
    )
    arena_path = _write_component(
        workspace,
        f"{run_relative}/{calibration.ARENA_REPORT_NAME}",
        kind=calibration.ARENA_KIND,
    )
    ledger_core = {
        "schemaVersion": calibration.LEDGER_SCHEMA_VERSION,
        "limitHours": contract.resources.global_gpu_hour_limit,
        "externalHoursFloor": global_gpu_hours,
        "runs": {},
        "updatedAt": "2026-07-26T00:00:00Z",
    }
    ledger_snapshot = {
        **ledger_core,
        "ledgerSha256": calibration.sha256_bytes(
            canonical_json_bytes(ledger_core)
        ),
    }
    core = {
        "schemaVersion": calibration.REPORT_SCHEMA_VERSION,
        "kind": calibration.REPORT_KIND,
        "status": "completed",
        "createdAt": "2026-07-26T00:00:00Z",
        "runId": run_id,
        "decisionEligible": True,
        "formalPromotion": False,
        "trainingPerformed": False,
        "checkpointsWritten": 0,
        "trainingAuthorized": False,
        "calibrationContractSha256": contract.sha256,
        "preparedExecutionContractSha256": (
            contract.provenance.prepared_execution_contract_sha256
        ),
        "runManifestSha256": RUN_MANIFEST_SHA256,
        "sourceSha256": SOURCE_SHA256,
        "nativeBackendSha256": NATIVE_SHA256,
        "approvedWorkloads": (
            approved_workloads
            if approved_workloads is not None
            else [
                BOOTSTRAP_WORKLOAD_SHA256,
                STANDARD_WORKLOAD_SHA256,
            ]
        ),
        "bootstrapBaseline": BASELINE_EVIDENCE,
        "selfPlay": calibration._component_reference(
            selfplay_path, workspace
        ),
        "arena": calibration._component_reference(arena_path, workspace),
        "resourceUsage": {
            "initialGlobalGpuHours": (
                contract.resources.initial_ledger.total_gpu_hours
            ),
            "globalGpuHours": global_gpu_hours,
            "phase4e1GpuHours": (
                global_gpu_hours
                - contract.resources.initial_ledger.total_gpu_hours
            ),
            "calibrationLimitHours": (
                contract.resources.calibration_gpu_hour_limit
            ),
            "artifactBytes": 1024,
            "freeDiskBytes": (
                contract.resources.minimum_free_disk_bytes + 1024
            ),
            "ledgerEvidence": {
                "path": contract.resources.initial_ledger.path,
                "snapshot": ledger_snapshot,
            },
        },
        "recommendedStageBudget": calibration._recommend_budget(
            contract,
            global_gpu_hours=global_gpu_hours,
        ),
    }
    payload = calibration._self_hashed(core, "reportSha256")
    path = workspace / run_relative / calibration.REPORT_NAME
    path.write_bytes(canonical_json_bytes(payload))
    return path, payload


def _selfplay_scenario(
    contract: Phase4E1CalibrationContract,
) -> calibration.CalibrationScenario:
    return calibration.CalibrationScenario(
        network_name="192x16",
        channels=contract.self_play.channels,
        blocks=contract.self_play.residual_blocks,
        actors=contract.self_play.actors,
        simulations=contract.self_play.simulations,
        games=contract.self_play.games,
        seed=contract.self_play.seed,
    )


def _write_selfplay_report(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    *,
    result_mutation: dict[str, object] | None = None,
    replay_relative: str | None = None,
) -> tuple[Path, dict[str, object], Path]:
    scenario = _selfplay_scenario(contract)
    run_relative = (
        f"{calibration.DEFAULT_OUTPUT_ROOT.as_posix()}/{_run_id(contract)}"
    )
    if replay_relative is None:
        replay_relative = (
            f"{run_relative}/selfplay-attempts/"
            "phase4e1-calibration-fixture/replay"
        )
    replay_root = workspace / replay_relative
    replay_root.mkdir(parents=True, exist_ok=True)
    games = contract.self_play.games
    moves = games
    elapsed = float(games)
    committed_bytes = 4096
    result: dict[str, object] = {
        "status": "completed",
        "scenarioId": scenario.scenario_id,
        "network": {
            "name": scenario.network_name,
            "inputChannels": calibration.PHASE4_INPUT_CHANNELS,
            "channels": scenario.channels,
            "residualBlocks": scenario.blocks,
        },
        "games": games,
        "actors": scenario.actors,
        "simulations": scenario.simulations,
        "seed": scenario.seed,
        "temperatureMoves": contract.self_play.temperature_moves,
        "elapsedSeconds": elapsed,
        "gamesPerHour": games * 3600.0 / elapsed,
        "moves": moves,
        "movesPerSecond": moves / elapsed,
        "scheduledSimulations": moves * scenario.simulations,
        "scheduledSimulationsPerSecond": (
            moves * scenario.simulations / elapsed
        ),
        "evaluation": {
            "calls": 1,
            "positions": 1,
            "batchSize": {"p50": 1.0, "p95": 1.0, "maximum": 1.0},
        },
        "cudaMemory": {
            "peakAllocatedBytes": 1,
            "peakReservedBytes": 1,
        },
        "gpuUtilization": {
            "available": True,
            "sampleCount": 2,
            "error": None,
            "gpuPercent": {"p50": 50.0, "p95": 50.0},
        },
        "replay": {
            "enabled": True,
            "deepVerified": True,
            "gameCount": games,
            "plyCount": moves,
            "committedBytes": committed_bytes,
            "shards": games,
        },
        "rawReplayPliesCollected": moves,
        "trainingSamplesCollected": 0,
        "checkpointsWritten": 0,
        "winners": {"player0": games, "player1": 0},
        "terminationReasons": {"goal": games},
        "gameLength": {"minimum": 1, "maximum": 1, "mean": 1.0},
    }
    if result_mutation is not None:
        result.update(result_mutation)
    core = {
        "schemaVersion": calibration.COMPONENT_SCHEMA_VERSION,
        "kind": calibration.SELFPLAY_KIND,
        "status": "completed",
        "createdAt": "2026-07-26T00:00:00Z",
        "calibrationOnly": True,
        "trainingPerformed": False,
        "checkpointsWritten": 0,
        "calibrationContractSha256": contract.sha256,
        "sourceSha256": SOURCE_SHA256,
        "nativeBackendSha256": NATIVE_SHA256,
        "baselineModelSha256": contract.provenance.baseline_model_sha256,
        "replayRoot": replay_relative,
        "result": result,
    }
    payload = calibration._self_hashed(core, "reportSha256")
    path = workspace / run_relative / calibration.SELFPLAY_REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return path, payload, replay_root


def _mock_replay_store(
    contract: Phase4E1CalibrationContract,
    *,
    defect: str | None = None,
) -> Mock:
    scenario = _selfplay_scenario(contract)
    initial = calibration.create_initial_state()
    final = replace(initial, winner=0, termination_reason="goal")
    store = Mock()

    def verify(*, deep: bool, game_validator: object) -> ReplayVerification:
        if deep is not True or not callable(game_validator):
            raise AssertionError("deep replay verification callback is required")
        for index in range(contract.self_play.games):
            game_metadata = {
                "calibrationOnly": True,
                "scenarioId": scenario.scenario_id,
                "seed": scenario.seed,
            }
            ply_metadata = {
                "calibrationOnly": True,
                "requestedSimulations": scenario.simulations,
            }
            visit_mass = scenario.simulations
            if defect == "game-metadata":
                game_metadata["seed"] = scenario.seed + 1
            elif defect == "ply-metadata":
                ply_metadata["requestedSimulations"] = (
                    scenario.simulations - 1
                )
            elif defect == "visit-mass":
                visit_mass = scenario.simulations - 1
            visits = [0] * ACTION_SPACE_SIZE
            visits[0] = visit_mass
            game = ReplayGame(
                game_id=f"{scenario.scenario_id}-{index:06d}",
                initial_state=initial,
                plies=(
                    ReplayPly(
                        action_index=0,
                        visit_counts=tuple(visits),
                        root_value=0.0,
                        model_generation=0,
                        model_id=(
                            "phase4e1-bootstrap-"
                            f"{contract.provenance.baseline_model_sha256}"
                        ),
                        metadata=ply_metadata,
                    ),
                ),
                metadata=game_metadata,
            )
            game_validator(game, (initial, final))
        return ReplayVerification(
            game_count=contract.self_play.games,
            ply_count=contract.self_play.games,
            committed_bytes=4096,
            deep=True,
        )

    store.verify.side_effect = verify
    return store


def _create_default_source_tree(workspace: Path) -> str:
    files = {
        "training/quoridor/source.py": b"current source\n",
        "training/pyproject.toml": b"[project]\nname='fixture'\n",
        "native/include/quoridor.hpp": b"// fixture\n",
        "native/src/quoridor.cpp": b"// fixture\n",
        "native/CMakeLists.txt": b"cmake_minimum_required(VERSION 3.16)\n",
        "src/game.js": b"export {};\n",
        "src/ai.js": b"export {};\n",
        "src/alphazero/index.js": b"export {};\n",
        "scripts/fixture.sh": b"#!/bin/sh\n",
    }
    for relative, content in files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return calibration.fingerprint_sources(workspace).sha256


def _create_source_mismatch_fixture(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    *,
    running_e1_ledger: bool = False,
) -> tuple[Path, Path, str, str]:
    current_source_sha256 = _create_default_source_tree(workspace)
    run_id = _run_id(contract)
    control_root = workspace / "training/runs/phase4"
    run_root = workspace / calibration.DEFAULT_OUTPUT_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    legacy_source = workspace / "legacy-source.py"
    legacy_source.write_text("legacy source\n", encoding="utf-8")
    budget = calibration.ResourceBudget(
        artifact_limit_bytes=contract.resources.artifact_limit_bytes,
        minimum_free_disk_bytes=contract.resources.minimum_free_disk_bytes,
        gpu_hours=contract.resources.calibration_gpu_hour_limit,
        metadata={
            "phase": "4E1",
            "trainingAllowed": False,
            "checkpointWritesAllowed": False,
            "formalPromotion": False,
        },
    )
    record = calibration.build_run_manifest(
        workspace_root=workspace,
        config=contract.to_payload(),
        budget=budget,
        run_id=run_id,
        source_paths=(legacy_source,),
        environment={},
        metadata={
            "phase": "4E1",
            "kind": calibration.REPORT_KIND,
            "calibrationContractSha256": contract.sha256,
            "nativeBackendSha256": NATIVE_SHA256,
        },
    )
    calibration.save_run_manifest(
        record,
        run_root / calibration.RUN_MANIFEST_NAME,
    )
    (run_root / "partial-evidence.bin").write_bytes(b"partial GPU evidence")
    ledger_path = control_root / calibration.GPU_LEDGER_NAME
    ledger = calibration.GpuBudgetLedger(ledger_path)
    ledger.record_external_hours(
        contract.resources.initial_ledger.total_gpu_hours
    )
    if running_e1_ledger:
        ledger.begin(
            "phase4e1-calibration-active-fixture",
            external_hours=0.0,
        )
    return (
        run_root,
        ledger_path,
        record.payload["provenance"]["source"]["sha256"],
        current_source_sha256,
    )


class Phase4E1CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_phase4e1_contract(CONTRACT_PATH)

    def test_budget_recommendation_floors_and_enforces_minimum_boundary(
        self,
    ) -> None:
        recommendation = calibration._recommend_budget(
            self.contract,
            global_gpu_hours=(
                self.contract.resources.initial_ledger.total_gpu_hours
            ),
        )
        raw = (
            self.contract.resources.global_gpu_hour_limit
            - self.contract.resources.initial_ledger.total_gpu_hours
            - self.contract.freeze.reserve_gpu_hours
        )

        self.assertEqual(
            recommendation["limitHours"],
            math.floor(raw * 1_000_000.0) / 1_000_000.0,
        )
        self.assertEqual(
            recommendation["maximumGenerations"],
            self.contract.freeze.maximum_generations,
        )

        boundary = (
            self.contract.resources.global_gpu_hour_limit
            - self.contract.freeze.reserve_gpu_hours
            - self.contract.freeze.minimum_stage_gpu_hours
        )
        exact = calibration._recommend_budget(
            self.contract,
            global_gpu_hours=boundary,
        )
        self.assertEqual(
            exact["limitHours"],
            self.contract.freeze.minimum_stage_gpu_hours,
        )
        with self.assertRaisesRegex(
            calibration.Phase4E1CalibrationError,
            "minimum 48 GPU-hours",
        ):
            calibration._recommend_budget(
                self.contract,
                global_gpu_hours=boundary + 0.000001,
            )

    def test_ledger_snapshot_binds_the_reported_gpu_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            _, report = _write_final_report(
                workspace,
                self.contract,
                global_gpu_hours=9.25,
            )
            evidence = report["resourceUsage"]["ledgerEvidence"]

            calibration._verify_ledger_snapshot_evidence(
                evidence,
                workspace=workspace,
                contract=self.contract,
                reported_global_hours=9.25,
                verify_current=False,
            )
            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "differs from its ledger snapshot",
            ):
                calibration._verify_ledger_snapshot_evidence(
                    evidence,
                    workspace=workspace,
                    contract=self.contract,
                    reported_global_hours=9.0,
                    verify_current=False,
                )

    def test_canonical_and_self_hash_tampering_fail_closed(self) -> None:
        core = {"schemaVersion": 1, "status": "completed"}
        payload = calibration._self_hashed(core, "reportSha256")
        expected_digest = payload["reportSha256"]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(canonical_json_bytes(payload))
            loaded = calibration._load_canonical(path, "fixture report")
            self.assertEqual(
                calibration._verify_self_hash(
                    loaded,
                    field="reportSha256",
                    label="fixture report",
                ),
                expected_digest,
            )

            tampered = dict(payload)
            tampered["status"] = "failed"
            path.write_bytes(canonical_json_bytes(tampered))
            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "self hash mismatch",
            ):
                calibration._verify_self_hash(
                    calibration._load_canonical(path, "fixture report"),
                    field="reportSha256",
                    label="fixture report",
                )

            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "canonical JSON",
            ):
                calibration._load_canonical(path, "fixture report")

    def test_component_candidate_recovery_is_atomic_and_interruptible(self) -> None:
        payload = {"schemaVersion": 1, "status": "completed"}
        encoded = canonical_json_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "component.json"
            candidate = report.with_name(f".{report.name}.candidate")
            candidate.write_bytes(encoded)

            recovered = calibration._recover_component_candidate(
                report,
                label="fixture component",
                verifier=lambda path: json.loads(path.read_bytes()),
            )

            self.assertEqual(recovered, payload)
            self.assertEqual(report.read_bytes(), encoded)
            self.assertFalse(candidate.exists())

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "component.json"
            candidate = report.with_name(f".{report.name}.candidate")
            candidate.write_bytes(encoded)

            def interrupt(_path: Path) -> dict[str, object]:
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                calibration._recover_component_candidate(
                    report,
                    label="fixture component",
                    verifier=interrupt,
                )
            self.assertTrue(candidate.exists())
            self.assertFalse(report.exists())

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "component.json"
            candidate = report.with_name(f".{report.name}.candidate")
            candidate.write_bytes(encoded)

            def reject(_path: Path) -> dict[str, object]:
                raise calibration.Phase4E1CalibrationError("invalid candidate")

            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "invalid candidate",
            ):
                calibration._recover_component_candidate(
                    report,
                    label="fixture component",
                    verifier=reject,
                )
            self.assertFalse(candidate.exists())
            self.assertFalse(report.exists())

    def test_selfplay_report_minimal_deep_replay_fixture_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            report_path, expected, replay_root = _write_selfplay_report(
                workspace,
                self.contract,
            )
            store = _mock_replay_store(self.contract)
            with patch.object(
                calibration,
                "ReplayStore",
                return_value=store,
            ) as replay_store:
                verified = calibration._verify_selfplay_report(
                    report_path,
                    workspace=workspace,
                    contract=self.contract,
                    source_sha256=SOURCE_SHA256,
                    native_sha256=NATIVE_SHA256,
                )

            self.assertEqual(verified, expected)
            replay_store.assert_called_once_with(
                replay_root,
                max_games=self.contract.self_play.games,
                max_plies=self.contract.self_play.games * 256,
                max_bytes=512 * 1024**2,
            )
            verify_call = store.verify.call_args
            self.assertTrue(verify_call.kwargs["deep"])
            self.assertTrue(callable(verify_call.kwargs["game_validator"]))

    def test_selfplay_report_rejects_wrong_scenario_network_or_seed(
        self,
    ) -> None:
        scenario = _selfplay_scenario(self.contract)
        mutations = {
            "scenario": {"scenarioId": "forged-scenario"},
            "network": {
                "network": {
                    "name": scenario.network_name,
                    "inputChannels": calibration.PHASE4_INPUT_CHANNELS,
                    "channels": scenario.channels - 1,
                    "residualBlocks": scenario.blocks,
                }
            },
            "seed": {"seed": scenario.seed + 1},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory).resolve()
                report_path, _, _ = _write_selfplay_report(
                    workspace,
                    self.contract,
                    result_mutation=mutation,
                )
                with (
                    patch.object(calibration, "ReplayStore") as replay_store,
                    self.assertRaisesRegex(
                        calibration.Phase4E1CalibrationError,
                        "self-play report contract mismatch",
                    ),
                ):
                    calibration._verify_selfplay_report(
                        report_path,
                        workspace=workspace,
                        contract=self.contract,
                        source_sha256=SOURCE_SHA256,
                        native_sha256=NATIVE_SHA256,
                    )
                replay_store.assert_not_called()

    def test_selfplay_report_rejects_nondeterministic_replay_root(self) -> None:
        run_relative = (
            f"{calibration.DEFAULT_OUTPUT_ROOT.as_posix()}/"
            f"{_run_id(self.contract)}"
        )
        invalid_roots = {
            "outside-attempts": "training/runs/phase4/forged/replay",
            "invalid-attempt-id": (
                f"{run_relative}/selfplay-attempts/not-an-attempt/replay"
            ),
        }
        for label, replay_relative in invalid_roots.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory).resolve()
                report_path, _, _ = _write_selfplay_report(
                    workspace,
                    self.contract,
                    replay_relative=replay_relative,
                )
                with (
                    patch.object(calibration, "ReplayStore") as replay_store,
                    self.assertRaisesRegex(
                        calibration.Phase4E1CalibrationError,
                        "deterministic run directory|not a calibration attempt",
                    ),
                ):
                    calibration._verify_selfplay_report(
                        report_path,
                        workspace=workspace,
                        contract=self.contract,
                        source_sha256=SOURCE_SHA256,
                        native_sha256=NATIVE_SHA256,
                    )
                replay_store.assert_not_called()

    def test_selfplay_report_rejects_replay_workload_forgery(self) -> None:
        for defect in ("game-metadata", "ply-metadata", "visit-mass"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory).resolve()
                report_path, _, _ = _write_selfplay_report(
                    workspace,
                    self.contract,
                )
                store = _mock_replay_store(self.contract, defect=defect)
                with (
                    patch.object(
                        calibration,
                        "ReplayStore",
                        return_value=store,
                    ),
                    self.assertRaisesRegex(
                        calibration.Phase4E1CalibrationError,
                        "replay workload metadata does not verify",
                    ),
                ):
                    calibration._verify_selfplay_report(
                        report_path,
                        workspace=workspace,
                        contract=self.contract,
                        source_sha256=SOURCE_SHA256,
                        native_sha256=NATIVE_SHA256,
                    )

    def test_selfplay_report_rejects_forged_derived_summaries(self) -> None:
        games = self.contract.self_play.games
        mutations = {
            "games-throughput": {"gamesPerHour": 3601.0},
            "moves-throughput": {"movesPerSecond": 2.0},
            "simulation-throughput": {
                "scheduledSimulationsPerSecond": (
                    self.contract.self_play.simulations + 1.0
                )
            },
            "winner": {
                "winners": {"player0": games - 1, "player1": 1}
            },
            "termination": {
                "terminationReasons": {
                    "goal": games - 1,
                    "repetition": 1,
                }
            },
            "length": {
                "gameLength": {
                    "minimum": 1,
                    "maximum": 2,
                    "mean": 1.0,
                }
            },
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory).resolve()
                report_path, _, _ = _write_selfplay_report(
                    workspace,
                    self.contract,
                    result_mutation=mutation,
                )
                store = _mock_replay_store(self.contract)
                with (
                    patch.object(
                        calibration,
                        "ReplayStore",
                        return_value=store,
                    ),
                    self.assertRaisesRegex(
                        calibration.Phase4E1CalibrationError,
                        "self-play replay does not verify",
                    ),
                ):
                    calibration._verify_selfplay_report(
                        report_path,
                        workspace=workspace,
                        contract=self.contract,
                        source_sha256=SOURCE_SHA256,
                        native_sha256=NATIVE_SHA256,
                    )

    def test_final_report_uses_workspace_paths_and_exact_approved_workloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            report_path, expected = _write_final_report(
                workspace, self.contract
            )
            with (
                patch.object(
                    calibration,
                    "_verify_e1_run_manifest",
                ) as manifest_verifier,
                patch.object(
                    calibration,
                    "_verify_selfplay_report",
                    return_value={"status": "completed"},
                ),
                patch.object(
                    calibration,
                    "_verify_arena_report",
                    return_value={"status": "completed"},
                ),
                patch.object(
                    calibration,
                    "_approved_workloads",
                    return_value=[
                        BOOTSTRAP_WORKLOAD_SHA256,
                        STANDARD_WORKLOAD_SHA256,
                    ],
                ) as workloads_verifier,
                patch.object(
                    calibration,
                    "verify_bootstrap_baseline",
                    return_value=BASELINE_EVIDENCE,
                ),
            ):
                verified = calibration._verify_final_report(
                    report_path,
                    workspace=workspace,
                    contract=self.contract,
                    verify_ledger=False,
                )

            expected_root = (
                f"{calibration.DEFAULT_OUTPUT_ROOT.as_posix()}/"
                f"{_run_id(self.contract)}"
            )
            self.assertEqual(verified, expected)
            manifest_verifier.assert_called_once_with(
                report_path.parent / calibration.RUN_MANIFEST_NAME,
                workspace=workspace,
                contract=self.contract,
                run_id=_run_id(self.contract),
                expected_sha256=RUN_MANIFEST_SHA256,
                source_sha256=SOURCE_SHA256,
                native_sha256=NATIVE_SHA256,
            )
            workloads_verifier.assert_called_once_with(
                workspace,
                self.contract,
            )
            self.assertEqual(
                verified["selfPlay"]["path"],
                f"{expected_root}/{calibration.SELFPLAY_REPORT_NAME}",
            )
            self.assertEqual(
                verified["arena"]["path"],
                f"{expected_root}/{calibration.ARENA_REPORT_NAME}",
            )
            self.assertEqual(
                verified["approvedWorkloads"],
                [
                    BOOTSTRAP_WORKLOAD_SHA256,
                    STANDARD_WORKLOAD_SHA256,
                ],
            )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            report_path, _ = _write_final_report(
                workspace,
                self.contract,
                approved_workloads=[
                    BOOTSTRAP_WORKLOAD_SHA256,
                    BOOTSTRAP_WORKLOAD_SHA256,
                ],
            )
            with (
                patch.object(calibration, "_verify_e1_run_manifest"),
                patch.object(
                    calibration,
                    "_verify_selfplay_report",
                    return_value={"status": "completed"},
                ),
                patch.object(
                    calibration,
                    "_verify_arena_report",
                    return_value={"status": "completed"},
                ),
                patch.object(
                    calibration,
                    "_approved_workloads",
                    return_value=[
                        BOOTSTRAP_WORKLOAD_SHA256,
                        STANDARD_WORKLOAD_SHA256,
                    ],
                ),
                patch.object(
                    calibration,
                    "verify_bootstrap_baseline",
                    return_value=BASELINE_EVIDENCE,
                ),
                self.assertRaisesRegex(
                    calibration.Phase4E1CalibrationError,
                    "approved workload list",
                ),
            ):
                calibration._verify_final_report(
                    report_path,
                    workspace=workspace,
                    contract=self.contract,
                    verify_ledger=False,
                )

    def test_freeze_writes_verified_immutable_execution_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            global_gpu_hours = 9.25
            report_path, report = _write_final_report(
                workspace,
                self.contract,
                global_gpu_hours=global_gpu_hours,
            )
            output_relative = (
                "training/configs/phase4e-execution-frozen-test.json"
            )
            with patch.object(
                calibration,
                "verify_phase4e1_decision_report",
                return_value=report,
            ) as decision_verifier:
                result = calibration.freeze_phase4e_execution_contract(
                    workspace_root=workspace,
                    calibration_contract_path=CONTRACT_PATH,
                    report_path=report_path,
                    output_path=output_relative,
                )

            decision_verifier.assert_called_once_with(
                workspace_root=workspace,
                report_path=report_path,
                contract_path=CONTRACT_PATH,
                verify_ledger=True,
            )
            output_path = workspace / output_relative
            frozen = load_phase4_execution_contract(output_path)
            recommendation = report["recommendedStageBudget"]
            self.assertEqual(result["status"], "frozen")
            self.assertEqual(result["path"], output_relative)
            self.assertEqual(
                result["executionContractSha256"], frozen.sha256
            )
            self.assertEqual(frozen.status, PHASE4_EXECUTION_STATUS_FROZEN)
            self.assertTrue(frozen.formal_promotion)
            self.assertEqual(
                frozen.stage_budget.status,
                PHASE4_BUDGET_STATUS_FROZEN,
            )
            self.assertEqual(
                frozen.stage_budget.baseline_report_sha256,
                report["reportSha256"],
            )
            self.assertEqual(
                frozen.stage_budget.baseline_global_hours,
                recommendation["baselineGlobalHours"],
            )
            self.assertEqual(
                frozen.stage_budget.limit_hours,
                recommendation["limitHours"],
            )
            self.assertEqual(
                output_path.read_bytes(),
                canonical_json_bytes(frozen.to_payload()),
            )

    def test_arena_identity_binds_the_exact_search_configuration(self) -> None:
        search = calibration._arena_search_payload(ROOT, self.contract)
        search_sha256 = calibration.sha256_bytes(
            canonical_json_bytes(search)
        )
        candidate, opponent = calibration._arena_identities(
            ROOT,
            self.contract,
            source_sha256=SOURCE_SHA256,
        )

        self.assertEqual(search["simulations"], 800)
        self.assertEqual(search["seed"], self.contract.arena.seed)
        self.assertEqual(search["temperature"], 0.0)
        self.assertFalse(search["addRootNoise"])
        self.assertEqual(search["dirichletEpsilon"], 0.0)
        self.assertEqual(candidate.config_hash, search_sha256)
        self.assertEqual(opponent.config_hash, search_sha256)

    def test_source_mismatch_is_atomically_quarantined_and_bound_to_new_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            (
                run_root,
                ledger_path,
                original_source_sha256,
                current_source_sha256,
            ) = _create_source_mismatch_fixture(workspace, self.contract)
            ledger_before = ledger_path.read_bytes()
            control_root = workspace / "training/runs/phase4"
            run_id = _run_id(self.contract)

            with patch.object(
                calibration.torch.cuda,
                "is_available",
                side_effect=AssertionError(
                    "quarantine must happen before a CUDA query"
                ),
            ):
                reference = calibration._quarantine_source_mismatched_run(
                    workspace=workspace,
                    control_root=control_root,
                    run_root=run_root,
                    run_id=run_id,
                    contract=self.contract,
                    native_sha256=NATIVE_SHA256,
                )

            self.assertIsNotNone(reference)
            assert reference is not None
            self.assertFalse(run_root.exists())
            self.assertEqual(ledger_path.read_bytes(), ledger_before)
            incident_path = workspace / reference["path"]
            incident = calibration._verify_quarantine_incident(
                incident_path,
                workspace=workspace,
                contract=self.contract,
                run_id=run_id,
                source_sha256=current_source_sha256,
                native_sha256=NATIVE_SHA256,
            )
            self.assertEqual(
                incident["originalSourceSha256"],
                original_source_sha256,
            )
            self.assertEqual(
                incident["replacementSourceSha256"],
                current_source_sha256,
            )
            self.assertIn(
                "partial-evidence.bin",
                [item["path"] for item in incident["files"]],
            )

            run_root.mkdir(parents=True)
            recovered = calibration._quarantine_source_mismatched_run(
                workspace=workspace,
                control_root=control_root,
                run_root=run_root,
                run_id=run_id,
                contract=self.contract,
                native_sha256=NATIVE_SHA256,
            )
            self.assertEqual(recovered, reference)
            with patch.object(
                calibration.torch.cuda,
                "is_available",
                return_value=False,
            ):
                manifest_sha256, source_sha256 = (
                    calibration._prepare_run_manifest(
                        workspace=workspace,
                        run_root=run_root,
                        run_id=run_id,
                        contract=self.contract,
                        native_sha256=NATIVE_SHA256,
                        quarantine_incident=recovered,
                    )
                )
            self.assertEqual(source_sha256, current_source_sha256)
            manifest = calibration.load_run_manifest(
                run_root / calibration.RUN_MANIFEST_NAME
            )
            self.assertEqual(manifest.sha256, manifest_sha256)
            self.assertEqual(
                manifest.payload["metadata"]["quarantineIncident"],
                reference,
            )
            calibration._verify_e1_run_manifest(
                run_root / calibration.RUN_MANIFEST_NAME,
                workspace=workspace,
                contract=self.contract,
                run_id=run_id,
                expected_sha256=manifest_sha256,
                source_sha256=current_source_sha256,
                native_sha256=NATIVE_SHA256,
            )
            with patch.object(
                calibration.torch.cuda,
                "is_available",
                return_value=False,
            ):
                resumed_manifest, resumed_source = (
                    calibration._prepare_run_manifest(
                        workspace=workspace,
                        run_root=run_root,
                        run_id=run_id,
                        contract=self.contract,
                        native_sha256=NATIVE_SHA256,
                    )
                )
            self.assertEqual(resumed_manifest, manifest_sha256)
            self.assertEqual(resumed_source, current_source_sha256)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)

    def test_source_mismatch_quarantine_rejects_running_e1_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            run_root, ledger_path, _, _ = _create_source_mismatch_fixture(
                workspace,
                self.contract,
                running_e1_ledger=True,
            )
            ledger_before = ledger_path.read_bytes()
            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "running Phase 4E1 GPU ledger",
            ):
                calibration._quarantine_source_mismatched_run(
                    workspace=workspace,
                    control_root=workspace / "training/runs/phase4",
                    run_root=run_root,
                    run_id=_run_id(self.contract),
                    contract=self.contract,
                    native_sha256=NATIVE_SHA256,
                )
            self.assertTrue(run_root.is_dir())
            self.assertEqual(ledger_path.read_bytes(), ledger_before)
            self.assertFalse(calibration._quarantine_root(workspace).exists())

    def test_source_mismatch_quarantine_rejects_corrupt_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            run_root = (
                workspace
                / calibration.DEFAULT_OUTPUT_ROOT
                / _run_id(self.contract)
            )
            run_root.mkdir(parents=True)
            (run_root / calibration.RUN_MANIFEST_NAME).write_bytes(b"{}\n")
            with self.assertRaisesRegex(
                calibration.Phase4E1CalibrationError,
                "manifest does not verify",
            ):
                calibration._quarantine_source_mismatched_run(
                    workspace=workspace,
                    control_root=workspace / "training/runs/phase4",
                    run_root=run_root,
                    run_id=_run_id(self.contract),
                    contract=self.contract,
                    native_sha256=NATIVE_SHA256,
                )
            self.assertTrue(run_root.is_dir())
            self.assertFalse(calibration._quarantine_root(workspace).exists())

    def test_cleanup_failure_does_not_mask_primary_arena_error(self) -> None:
        primary = RuntimeError("primary arena failure")
        cleanup = ValueError("reconcile cleanup failure")
        with self.assertRaisesRegex(
            RuntimeError,
            "primary arena failure",
        ) as raised:
            calibration._raise_primary_or_cleanup(
                primary,
                [("reconcile arena store", cleanup)],
            )
        self.assertIs(raised.exception, primary)
        self.assertIs(raised.exception.__cause__, cleanup)

    def test_partial_arena_corruption_fails_before_cuda_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            cuda_available = Mock(
                side_effect=AssertionError("CUDA must not be queried")
            )
            with (
                patch.object(calibration, "_preflight", return_value={}),
                patch.object(
                    calibration,
                    "loaded_native_backend_hash",
                    return_value=NATIVE_SHA256,
                ),
                patch.object(
                    calibration,
                    "_prepare_run_manifest",
                    return_value=(RUN_MANIFEST_SHA256, SOURCE_SHA256),
                ),
                patch.object(
                    calibration,
                    "_verify_partial_arena_store",
                    side_effect=calibration.Phase4E1CalibrationError(
                        "partial pair corruption"
                    ),
                ) as partial_verifier,
                patch.object(
                    calibration.torch.cuda,
                    "is_available",
                    cuda_available,
                ),
                self.assertRaisesRegex(
                    calibration.Phase4E1CalibrationError,
                    "partial pair corruption",
                ),
            ):
                calibration.run_phase4e1_calibration(
                    workspace_root=workspace,
                    contract_path=CONTRACT_PATH,
                )

            partial_verifier.assert_called_once()
            cuda_available.assert_not_called()
            self.assertFalse(
                (
                    workspace
                    / "training/runs/phase4/gpu-budget-ledger.json"
                ).exists()
            )

    def test_existing_report_is_verified_at_contract_address_without_cuda(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            expected_path = (
                workspace
                / calibration.DEFAULT_OUTPUT_ROOT
                / f"phase4e1-calibration-{self.contract.sha256[:16]}"
                / calibration.REPORT_NAME
            )
            expected_path.parent.mkdir(parents=True)
            expected_path.touch()
            cached = {
                "status": "completed",
                "approvedWorkloads": [
                    BOOTSTRAP_WORKLOAD_SHA256,
                    STANDARD_WORKLOAD_SHA256,
                ],
            }
            with (
                patch.object(calibration, "_preflight", return_value={}),
                patch.object(
                    calibration,
                    "verify_phase4e1_decision_report",
                    return_value=cached,
                ) as verify,
                patch.object(
                    calibration.torch.cuda,
                    "is_available",
                    side_effect=AssertionError("CUDA must not be queried"),
                ),
            ):
                result = calibration.run_phase4e1_calibration(
                    workspace_root=workspace,
                    contract_path=CONTRACT_PATH,
                )

            self.assertEqual(result, cached)
            verify.assert_called_once_with(
                workspace_root=workspace,
                report_path=expected_path,
                contract_path=CONTRACT_PATH,
                verify_ledger=True,
            )

    def test_cli_help_lists_all_commands_without_cuda(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                calibration.torch.cuda,
                "is_available",
                side_effect=AssertionError("CUDA must not be queried"),
            ),
            contextlib.redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            calibration.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("{run,verify,freeze}", help_text)
        self.assertIn("Standard-shaped CUDA calibration", help_text)


if __name__ == "__main__":
    unittest.main()
