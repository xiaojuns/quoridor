from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from quoridor.config import ExperimentConfig, load_experiment_config, seed_everything
from quoridor.network import PolicyValueNetwork
from quoridor.phase4_checkpoint import canonical_model_state_sha256
from quoridor.phase4_pilot import (
    PHASE4D_ACTORS,
    PHASE4D_ARENA_GAMES,
    PHASE4D_BATCH_SIZE,
    PHASE4D_GAMES_PER_GENERATION,
    PHASE4D_REPLAY_CAPACITY,
    PHASE4D_SEED,
    PHASE4D_TRAINING_STEPS,
    PHASE4D_BASE_CONFIG_SHA256,
    LaneState,
    Phase4PilotContractError,
    Phase4ResourceLimitReached,
    _contains_teacher,
    _checkpoint_payload_from_report,
    _create_or_verify,
    _DurableJsonlSink,
    _assert_lane_runtime_binding,
    _assert_write_capacity,
    _lane_has_unreported_work,
    _load_canonical_base_config,
    _prepare_arena_commit,
    _result_summary_from_manifest,
    _schedule_prefix,
    _self_hashed_payload,
    _validate_pretraining_inputs,
    build_effective_config_payloads,
    decide_warm_cold,
    load_pilot_contract,
    next_step,
    prepare_pilot,
)
from quoridor.phase4_training import _network_config
from quoridor.run_manifest import (
    GenerationReportRecord,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    fingerprint_config,
)
from quoridor.statistics import PairedGameResult


ROOT = Path(__file__).resolve().parents[2]


def paired_results(candidate_wins: int) -> tuple[PairedGameResult, ...]:
    if not 0 <= candidate_wins <= PHASE4D_ARENA_GAMES:
        raise ValueError(candidate_wins)
    results = []
    remaining = candidate_wins
    for opening in range(PHASE4D_ARENA_GAMES // 2):
        for leg, candidate_player in ((0, 0), (1, 1)):
            candidate_won = remaining > 0
            if candidate_won:
                remaining -= 1
            results.append(
                PairedGameResult(
                    opening_id=f"opening-{opening:02d}",
                    leg=leg,
                    candidate_player=candidate_player,
                    winner=(
                        candidate_player if candidate_won else 1 - candidate_player
                    ),
                )
            )
    return tuple(results)


class EffectiveConfigTests(unittest.TestCase):
    def test_configs_freeze_equal_base_and_isolate_warm_provenance(self) -> None:
        base_path = ROOT / "training/configs/phase4-bootstrap.json"
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        pretraining = {
            "artifactPath": "training/runs/phase4/pretrain/warm-model.pt",
            "artifactSha256": "1" * 64,
            "modelStateSha256": "2" * 64,
            "baseModelStateSha256": base_payload["initialization"][
                "expectedBaseModelSha256"
            ],
            "teacherDatasetSha256": "3" * 64,
            "reportSha256": "4" * 64,
        }
        payloads = build_effective_config_payloads(
            base_payload,
            pretraining,
            pilot_root_relative="training/runs/phase4/phase4d-pilot",
        )
        cold = ExperimentConfig.from_dict(payloads["cold"])
        warm = ExperimentConfig.from_dict(payloads["warm"])

        self.assertEqual(cold.seed, warm.seed)
        self.assertEqual(cold.seed, PHASE4D_SEED)
        self.assertEqual(
            cold.initialization.expected_base_model_sha256,
            warm.initialization.expected_base_model_sha256,
        )
        self.assertEqual(cold.initialization.mode, "random")
        self.assertEqual(warm.initialization.mode, "model")
        self.assertTrue(cold.initialization.reset_optimizer)
        self.assertTrue(warm.initialization.reset_optimizer)
        self.assertNotEqual(cold.run_dir, warm.run_dir)
        self.assertEqual(cold.self_play.games, PHASE4D_GAMES_PER_GENERATION)
        self.assertEqual(cold.self_play.actors, PHASE4D_ACTORS)
        self.assertEqual(cold.training.batch_size, PHASE4D_BATCH_SIZE)
        self.assertEqual(cold.training.training_steps, PHASE4D_TRAINING_STEPS)
        self.assertEqual(cold.training.replay_capacity, PHASE4D_REPLAY_CAPACITY)
        self.assertEqual(
            warm.initialization.pretraining_report_sha256,
            pretraining["reportSha256"],
        )
        self.assertEqual(
            warm.initialization.teacher_dataset_sha256,
            pretraining["teacherDatasetSha256"],
        )

    def test_immutable_writer_never_overwrites_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            _create_or_verify(path, b"one\n", "test")
            _create_or_verify(path, b"one\n", "test")
            with self.assertRaisesRegex(Phase4PilotContractError, "does not match"):
                _create_or_verify(path, b"two\n", "test")
            self.assertEqual(path.read_bytes(), b"one\n")

    def test_prepare_is_idempotent_and_contract_binds_effective_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config_path = workspace / "training/configs/phase4-bootstrap.json"
            dev_path = workspace / "training/fixtures/openings/dev.json"
            summary_path = workspace / "training/runs/phase4/pretrain/summary.json"
            config_path.parent.mkdir(parents=True)
            dev_path.parent.mkdir(parents=True)
            summary_path.parent.mkdir(parents=True)
            shutil.copy2(
                ROOT / "training/configs/phase4-bootstrap.json", config_path
            )
            shutil.copy2(ROOT / "training/fixtures/openings/dev.json", dev_path)
            summary_path.write_text("{}\n", encoding="utf-8")
            base = load_experiment_config(config_path)
            pretraining = {
                "summaryPath": summary_path.relative_to(workspace).as_posix(),
                "summarySha256": "0" * 64,
                "artifactPath": "training/runs/phase4/pretrain/warm-model.pt",
                "artifactSha256": "1" * 64,
                "modelStateSha256": "2" * 64,
                "baseModelStateSha256": (
                    base.initialization.expected_base_model_sha256
                ),
                "teacherDatasetSha256": "3" * 64,
                "reportPath": "training/runs/phase4/pretrain/report.json",
                "reportSha256": "4" * 64,
            }
            output = workspace / "training/runs/phase4/phase4d-pilot"
            runtime = {
                "sourceSha256": "5" * 64,
                "nativeBackendSha256": "6" * 64,
            }
            with patch(
                "quoridor.phase4_pilot._validate_pretraining_inputs",
                return_value=pretraining,
            ), patch(
                "quoridor.phase4_pilot._current_runtime_provenance",
                return_value=runtime,
            ):
                first = prepare_pilot(
                    workspace_root=workspace,
                    pretraining_summary_path=summary_path,
                    base_config_path=config_path,
                    output_root=output,
                    verify_resources=False,
                )
                second = prepare_pilot(
                    workspace_root=workspace,
                    pretraining_summary_path=summary_path,
                    base_config_path=config_path,
                    output_root=output,
                    verify_resources=False,
                )
            self.assertEqual(first["contractSha256"], second["contractSha256"])
            contract_path = workspace / first["contractPath"]
            contract = load_pilot_contract(contract_path, workspace_root=workspace)
            self.assertEqual(contract["contractSha256"], first["contractSha256"])
            self.assertEqual(
                contract["workload"]["schedule"][0],
                {"ordinal": 0, "lane": "cold", "generation": 0},
            )
            self.assertEqual(
                contract["workload"]["schedule"][1],
                {"ordinal": 1, "lane": "warm", "generation": 0},
            )
            with patch(
                "quoridor.phase4_pilot._current_runtime_provenance",
                return_value=runtime,
            ):
                self.assertEqual(
                    next_step(workspace_root=workspace, contract_path=contract_path),
                    ("cold", 1),
                )

    def test_mutated_base_protocol_is_rejected(self) -> None:
        source = ROOT / "training/configs/phase4-bootstrap.json"
        mutations = (
            ("training", "learningRate", 0.123),
            ("mcts", "cBase", 0),
            ("selfPlay", "temperatureMoves", 0),
            ("training", "maxCheckpoints", 99),
        )
        for section, field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(source.read_text(encoding="utf-8"))
                payload[section][field] = value
                path = Path(directory) / "phase4-bootstrap.json"
                path.write_bytes(canonical_json_bytes(payload))
                self.assertNotEqual(sha256_file(path), PHASE4D_BASE_CONFIG_SHA256)
                with self.assertRaisesRegex(
                    Phase4PilotContractError, "canonical phase4-bootstrap"
                ):
                    _load_canonical_base_config(path)


class PretrainingCompatibilityTests(unittest.TestCase):
    def test_shallow_self_hashed_pretraining_report_is_rejected(self) -> None:
        base = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        seed_everything(base.seed)
        model = PolicyValueNetwork(_network_config(base))
        model_sha256 = canonical_model_state_sha256(model)
        teacher_sha256 = "3" * 64
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "training/runs/phase4/pretrain"
            run_root.mkdir(parents=True)
            report_path = run_root / "report.json"
            artifact_path = run_root / "warm-model.pt"
            summary_path = run_root / "summary.json"
            report_core = {
                "schemaVersion": 1,
                "kind": "phase4d-teacher-pretraining",
                "status": "completed",
                "warmArtifactPath": artifact_path.name,
                "baseModelStateSha256": model_sha256,
                "modelStateSha256": model_sha256,
                "teacherDatasetSha256": teacher_sha256,
                "optimizerResetRequiredForSelfPlay": True,
                "teacherReplayAllowedInSelfPlay": False,
            }
            report = {
                **report_core,
                "reportCoreSha256": sha256_bytes(
                    canonical_json_bytes(report_core)
                ),
            }
            report_path.write_bytes(canonical_json_bytes(report))
            report_sha256 = sha256_file(report_path)
            torch.save(
                {
                    "schemaVersion": 1,
                    "format": "quoridor-phase4-model-only",
                    "network": _network_config(base).to_dict(),
                    "baseModelStateSha256": model_sha256,
                    "modelStateSha256": model_sha256,
                    "provenance": {
                        "teacherDatasetSha256": teacher_sha256,
                        "pretrainingReportSha256": report_sha256,
                    },
                    "modelState": model.state_dict(),
                },
                artifact_path,
            )
            summary = {
                "schemaVersion": 1,
                "status": "completed",
                "artifactPath": artifact_path.name,
                "artifactSha256": sha256_file(artifact_path),
                "modelStateSha256": model_sha256,
                "baseModelStateSha256": model_sha256,
                "teacherDatasetSha256": teacher_sha256,
                "reportPath": report_path.name,
                "reportSha256": report_sha256,
            }
            summary_path.write_bytes(canonical_json_bytes(summary))
            with self.assertRaisesRegex(
                Phase4PilotContractError, "completion bundle does not verify"
            ):
                _validate_pretraining_inputs(
                    workspace=workspace,
                    summary_path=summary_path,
                    base_config=base,
                )


class ResourceBoundaryTests(unittest.TestCase):
    class _Guard:
        def __init__(self, limit: int | None = None) -> None:
            self.limit = limit
            self.calls: list[int] = []

        def check(
            self, *, force: bool = False, reserve_bytes: int = 0
        ) -> dict[str, object]:
            self.calls.append(reserve_bytes)
            if self.limit is not None and reserve_bytes > self.limit:
                raise Phase4ResourceLimitReached("write reserve crossed boundary")
            return {
                "freeDiskBytes": 20 * 1024**3 + reserve_bytes,
                "artifactBytes": 0,
                "reservedWriteBytes": reserve_bytes,
                "forced": force,
            }

    def test_write_reserve_accepts_exact_boundaries(self) -> None:
        reserve = 4096
        with patch(
            "quoridor.phase4_pilot.directory_size",
            return_value=8 * 1024**3 - reserve,
        ), patch(
            "quoridor.phase4_pilot.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3 + reserve),
        ):
            result = _assert_write_capacity(Path("/unused"), reserve_bytes=reserve)
        self.assertEqual(result["reservedWriteBytes"], reserve)

    def test_write_reserve_rejects_either_boundary_by_one_byte(self) -> None:
        reserve = 4096
        with patch(
            "quoridor.phase4_pilot.directory_size",
            return_value=8 * 1024**3 - reserve + 1,
        ), patch(
            "quoridor.phase4_pilot.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3 + reserve),
        ):
            with self.assertRaisesRegex(Phase4ResourceLimitReached, "8 GiB"):
                _assert_write_capacity(Path("/unused"), reserve_bytes=reserve)
        with patch(
            "quoridor.phase4_pilot.directory_size",
            return_value=0,
        ), patch(
            "quoridor.phase4_pilot.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3 + reserve - 1),
        ):
            with self.assertRaisesRegex(Phase4ResourceLimitReached, "20 GiB"):
                _assert_write_capacity(Path("/unused"), reserve_bytes=reserve)

    def test_arena_commit_reserves_final_serialized_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "partial.jsonl"
            partial.write_bytes(b"game\n" * 32)
            guard = self._Guard()
            report, encoded, required, resources = _prepare_arena_commit(
                {"schemaVersion": 1, "status": "completed"},
                partial,
                guard,  # type: ignore[arg-type]
            )
            self.assertEqual(guard.calls[-1], required)
            self.assertEqual(report["resourcesBeforeCommit"]["reservedWriteBytes"], guard.calls[0])
            self.assertEqual(resources["reservedWriteBytes"], required)
            self.assertEqual(encoded, canonical_json_bytes(report))
            self.assertGreater(required, guard.calls[0])

            boundary_guard = self._Guard(limit=guard.calls[0])
            with self.assertRaisesRegex(
                Phase4ResourceLimitReached, "crossed boundary"
            ):
                _prepare_arena_commit(
                    {"schemaVersion": 1, "status": "completed"},
                    partial,
                    boundary_guard,  # type: ignore[arg-type]
                )

    def test_streaming_jsonl_reserves_each_encoded_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            guard = self._Guard()
            sink = _DurableJsonlSink(
                Path(directory) / "partial.jsonl",
                guard,  # type: ignore[arg-type]
            )
            try:
                value = '{"gameId":"one"}\n'
                self.assertEqual(sink.write(value), len(value))
                sink.flush()
            finally:
                sink.close()
            self.assertEqual(guard.calls, [len(value.encode("utf-8")) + 4096])


class PilotDecisionTests(unittest.TestCase):
    def test_warm_requires_both_raw_and_one_sided_checks(self) -> None:
        strong = decide_warm_cold(paired_results(64))
        self.assertTrue(strong.warm_selected)
        self.assertEqual(strong.selected_lane, "warm")
        self.assertGreater(strong.one_sided_95_lower, 0.5)

        below_raw = decide_warm_cold(paired_results(35))
        self.assertFalse(below_raw.raw_threshold_passed)
        self.assertFalse(below_raw.warm_selected)
        self.assertEqual(below_raw.selected_lane, "cold")

    def test_teacher_marker_detection_is_recursive(self) -> None:
        self.assertFalse(_contains_teacher({"pipeline": "phase4", "model": "sha256:x"}))
        self.assertTrue(_contains_teacher({"nested": [{"teacherModelId": "x"}]}))
        self.assertTrue(_contains_teacher("phase4d-classic-hard-teacher"))

    def test_partial_arena_is_never_a_result_and_manifest_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "result-manifest.json"
            (root / "partial-1.games.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertIsNone(
                _result_summary_from_manifest(
                    manifest_path, expected_contract_sha256="a" * 64
                )
            )

            report = root / "report.json"
            games = root / "games.jsonl"
            report.write_bytes(b"report\n")
            games.write_text(
                "".join(
                    json.dumps({"gameId": f"game-{index}"}) + "\n"
                    for index in range(64)
                ),
                encoding="utf-8",
            )
            core = {
                "schemaVersion": 1,
                "status": "completed",
                "createdAt": "2026-01-01T00:00:00Z",
                "contractSha256": "a" * 64,
                "report": {
                    "file": report.name,
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "bytes": report.stat().st_size,
                },
                "gamesJsonl": {
                    "file": games.name,
                    "sha256": hashlib.sha256(games.read_bytes()).hexdigest(),
                    "bytes": games.stat().st_size,
                    "completedGames": 64,
                },
                "decision": {
                    "selectedLane": "cold",
                    "screenOnly": True,
                    "formalChampionDeclaration": False,
                },
            }
            manifest_path.write_bytes(
                canonical_json_bytes(
                    _self_hashed_payload(core, "resultManifestSha256")
                )
            )
            loaded = _result_summary_from_manifest(
                manifest_path, expected_contract_sha256="a" * 64
            )
            self.assertEqual(loaded["decision"]["selectedLane"], "cold")

            games.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(Phase4PilotContractError, "does not verify"):
                _result_summary_from_manifest(
                    manifest_path, expected_contract_sha256="a" * 64
                )


class AlternatingScheduleTests(unittest.TestCase):
    @staticmethod
    def _record(generation: int, created_at: str) -> GenerationReportRecord:
        payload = {
            "createdAt": created_at,
            "report": {},
        }
        return GenerationReportRecord(
            generation=generation,
            path=Path(f"generation-{generation}.json"),
            sha256="a" * 64,
            payload=payload,
        )

    def setUp(self) -> None:
        self.config = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )

    def _state(self, lane: str, times: tuple[str, ...]) -> LaneState:
        return LaneState(
            lane=lane,
            config=self.config,
            manifest_sha256="b" * 64,
            source_sha256="c" * 64,
            reports=tuple(
                self._record(generation, created)
                for generation, created in enumerate(times)
            ),
        )

    def test_accepts_only_cold_then_warm_prefix(self) -> None:
        cold = self._state("cold", ("2026-01-01T00:00:01Z", "2026-01-01T00:00:03Z"))
        warm = self._state("warm", ("2026-01-01T00:00:02Z",))
        prefix = _schedule_prefix(cold, warm)
        self.assertEqual([(lane, generation) for lane, generation, _ in prefix], [
            ("cold", 0),
            ("warm", 0),
            ("cold", 1),
        ])

        ahead = self._state(
            "warm", ("2026-01-01T00:00:02Z", "2026-01-01T00:00:04Z")
        )
        with self.assertRaisesRegex(Phase4PilotContractError, "alternation"):
            _schedule_prefix(self._state("cold", ("2026-01-01T00:00:01Z",)), ahead)

    def test_rejects_timestamp_order_that_cannot_prove_schedule(self) -> None:
        cold = self._state("cold", ("2026-01-01T00:00:03Z",))
        warm = self._state("warm", ("2026-01-01T00:00:02Z",))
        with self.assertRaisesRegex(Phase4PilotContractError, "timestamps"):
            _schedule_prefix(cold, warm)

    def test_started_lane_must_match_contract_runtime_before_other_lane(self) -> None:
        state = self._state("cold", ("2026-01-01T00:00:01Z",))
        state = replace(state, native_backend_sha256="d" * 64)
        contract = {
            "runtimeProvenance": {
                "sourceSha256": "e" * 64,
                "nativeBackendSha256": "d" * 64,
            }
        }
        with self.assertRaisesRegex(
            Phase4PilotContractError, "differs from the pilot contract"
        ):
            _assert_lane_runtime_binding(state, contract)

    def test_unreported_replay_is_left_for_idempotent_target_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "lane"
            replay_manifest = run_dir / "replay/manifest.json"
            replay_manifest.parent.mkdir(parents=True)
            replay_manifest.write_text(
                json.dumps({"totals": {"games": 65}}) + "\n",
                encoding="utf-8",
            )
            state = LaneState(
                lane="cold",
                config=replace(self.config, run_dir=run_dir),
                manifest_sha256="b" * 64,
                source_sha256="c" * 64,
                reports=(self._record(0, "2026-01-01T00:00:01Z"),),
            )
            self.assertTrue(_lane_has_unreported_work(state))

    def test_reported_checkpoint_uses_exact_three_key_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "lane"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            config = replace(self.config, run_dir=run_dir)
            identity = {
                "runManifestSha256": "b" * 64,
                "configSha256": fingerprint_config(config).sha256,
                "sourceSha256": "c" * 64,
            }
            generation_data = {"generation": 0}
            checkpoint_path = checkpoint_dir / "checkpoint-g00000000-s000000000064.pt"
            torch.save(
                {
                    "schemaVersion": 1,
                    "generation": 0,
                    "step": 64,
                    "identity": identity,
                    "generationData": generation_data,
                },
                checkpoint_path,
            )
            digest = sha256_file(checkpoint_path)
            record = GenerationReportRecord(
                generation=0,
                path=run_dir / "reports/generation-00000000.json",
                sha256="d" * 64,
                payload={
                    "report": {
                        "checkpoint": {
                            "path": checkpoint_path.name,
                            "bytes": checkpoint_path.stat().st_size,
                            "sha256": digest,
                            "step": 64,
                        },
                        "generationData": generation_data,
                    }
                },
            )
            state = LaneState(
                lane="cold",
                config=config,
                manifest_sha256=identity["runManifestSha256"],
                source_sha256=identity["sourceSha256"],
                reports=(record,),
                native_backend_sha256="e" * 64,
            )
            path, payload, loaded_digest = _checkpoint_payload_from_report(
                state, record
            )
            self.assertEqual(path, checkpoint_path)
            self.assertEqual(payload["identity"], identity)
            self.assertEqual(loaded_digest, digest)


if __name__ == "__main__":
    unittest.main()
