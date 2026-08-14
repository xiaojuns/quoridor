from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from quoridor.config import ExperimentConfig
from quoridor.phase4_baseline import (
    BASELINE_KIND,
    BASELINE_MANIFEST_NAME,
    BASELINE_MODEL_FORMAT,
    BASELINE_MODEL_NAME,
    Phase4BaselineCorruptionError,
    Phase4BaselineLineageError,
    load_bootstrap_baseline_state,
    main,
    prepare_bootstrap_baseline,
    verify_bootstrap_baseline,
)
from quoridor.phase4_checkpoint import canonical_model_state_sha256
from quoridor.phase4_pilot import (
    PHASE4D_BASE_CONFIG_SHA256,
    PHASE4D_PILOT_GENERATIONS,
    build_effective_config_payloads,
)
from quoridor.run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    GenerationReportStore,
    ResourceBudget,
    build_run_manifest,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_RELATIVE = Path("training/runs/phase4/phase4d-pilot-v2")
BASELINE_RELATIVE = Path(
    "training/runs/phase4/baselines/bootstrap-warm-v2"
)


def _self_hashed(
    core: dict[str, object],
    field: str,
) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


class _PilotFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pilot_root = root / PILOT_RELATIVE
        self.baseline_root = root / BASELINE_RELATIVE
        self.contract_path = self.pilot_root / "pilot-contract.json"
        self.result_path = self.pilot_root / "arena/result-manifest.json"
        self.native_sha256 = "b" * 64
        self.source_path = root / "source.txt"
        self.source_path.write_text("frozen-pilot-source\n", encoding="utf-8")
        self.source_sha256 = fingerprint_sources(
            root, paths=["source.txt"]
        ).sha256
        self.base_path = root / "training/configs/phase4-bootstrap.json"
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            PROJECT_ROOT / "training/configs/phase4-bootstrap.json",
            self.base_path,
        )
        assert sha256_file(self.base_path) == PHASE4D_BASE_CONFIG_SHA256
        self.lanes: dict[str, dict[str, object]] = {}
        self._build()

    def _build(self) -> None:
        base_payload = json.loads(self.base_path.read_text(encoding="utf-8"))
        pretraining = {
            "summaryPath": "training/fake-pretraining/summary.json",
            "summarySha256": "1" * 64,
            "artifactPath": "training/fake-pretraining/warm-model.pt",
            "artifactSha256": "2" * 64,
            "modelStateSha256": "3" * 64,
            "baseModelStateSha256": base_payload["initialization"][
                "expectedBaseModelSha256"
            ],
            "teacherDatasetSha256": "4" * 64,
            "reportPath": "training/fake-pretraining/report.json",
            "reportSha256": "5" * 64,
        }
        effective = build_effective_config_payloads(
            base_payload,
            pretraining,
            pilot_root_relative=PILOT_RELATIVE.as_posix(),
        )
        config_records: dict[str, dict[str, object]] = {}
        relative_configs: dict[str, ExperimentConfig] = {}
        for lane in ("cold", "warm"):
            path = self.pilot_root / f"effective-configs/{lane}.json"
            _write_json(path, effective[lane])
            config = ExperimentConfig.from_dict(effective[lane])
            relative_configs[lane] = config
            config_records[lane] = {
                "path": path.relative_to(self.root).as_posix(),
                "fileSha256": sha256_file(path),
                "configSha256": fingerprint_config(config).sha256,
                "runDir": config.run_dir.as_posix(),
            }

        schedule = [
            {
                "ordinal": 2 * generation + (1 if lane == "warm" else 0),
                "lane": lane,
                "generation": generation,
            }
            for generation in range(PHASE4D_PILOT_GENERATIONS)
            for lane in ("cold", "warm")
        ]
        opening_ids = [f"opening-{index:024x}" for index in range(32)]
        contract_core = {
            "schemaVersion": 1,
            "kind": "phase4d-warm-cold-pilot",
            "createdAt": "2026-07-25T00:00:00Z",
            "localOnly": True,
            "formalChampionDeclaration": False,
            "runtimeProvenance": {
                "sourceSha256": self.source_sha256,
                "nativeBackendSha256": self.native_sha256,
            },
            "baseConfig": {
                "path": self.base_path.relative_to(self.root).as_posix(),
                "fileSha256": PHASE4D_BASE_CONFIG_SHA256,
            },
            "pretraining": pretraining,
            "configs": config_records,
            "workload": {
                "seed": 20_260_714,
                "network": "192x16",
                "generationsPerLane": 5,
                "gamesPerGeneration": 64,
                "actors": 32,
                "selfPlaySimulations": 256,
                "batchSize": 512,
                "trainingStepsPerGeneration": 64,
                "schedule": schedule,
            },
            "arena": {
                "sourceSuitePath": "training/fixtures/openings/dev.json",
                "sourceSuiteFileSha256": "6" * 64,
                "sourceSuiteHash": "sha256:" + "7" * 64,
                "selection": "first-32-in-frozen-suite-order",
                "openingIds": opening_ids,
                "screenSuiteHash": "sha256:" + "8" * 64,
                "games": 64,
                "simulationsPerSide": 256,
                "bootstrapSeed": 20_260_714,
                "bootstrapSamples": 10_000,
                "warmThresholds": {
                    "rawWinRateAtLeast": 0.55,
                    "oneSided95LowerGreaterThan": 0.5,
                },
            },
            "resourceLimits": {
                "globalGpuHours": 72.0,
                "phase4dGpuHours": 10.0,
                "artifactBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
        }
        contract = _self_hashed(contract_core, "contractSha256")
        _write_json(self.contract_path, contract)

        for lane in ("cold", "warm"):
            self.lanes[lane] = self._build_lane(
                lane,
                effective[lane],
                generation_offset=0 if lane == "cold" else 1,
            )
        self._write_arena(contract, opening_ids)

    def _build_lane(
        self,
        lane: str,
        relative_payload: dict[str, object],
        *,
        generation_offset: int,
    ) -> dict[str, object]:
        run_dir = self.pilot_root / f"lanes/{lane}"
        absolute_config = ExperimentConfig.from_dict(
            {**relative_payload, "runDir": str(run_dir)}
        )
        runtime_config_sha256 = fingerprint_config(absolute_config).sha256
        budget = ResourceBudget(
            gpu_hours=10.0,
            maximum_generations=5,
            metadata={"nativeBackendSha256": self.native_sha256},
        )
        manifest = build_run_manifest(
            workspace_root=self.root,
            config=absolute_config,
            budget=budget,
            run_id=f"fixture-{lane}",
            source_paths=["source.txt"],
            environment={},
            metadata={
                "formalPromotion": False,
                "pipeline": "phase4",
                "stage": "phase4d",
            },
            created_at="2026-07-25T00:00:00Z",
        )
        save_run_manifest(manifest, run_dir / "run-manifest.json")

        reports = GenerationReportStore(
            run_dir / "reports",
            run_manifest_sha256=manifest.sha256,
        )
        for generation in range(4):
            reports.write(
                generation,
                {
                    "fixture": generation,
                    "generationData": {
                        "schemaVersion": 1,
                        "generation": generation,
                    },
                },
                created_at=(
                    f"2026-07-25T00:00:{2 * generation + generation_offset:02d}Z"
                ),
            )

        model = OrderedDict(
            [
                (
                    "linear.weight",
                    torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]]
                        if lane == "warm"
                        else [[-1.0, -2.0], [-3.0, -4.0]],
                        dtype=torch.float32,
                    ),
                ),
                ("counter", torch.tensor([4], dtype=torch.int64)),
            ]
        )
        generation_data = {
            "schemaVersion": 1,
            "generation": 4,
            "promotion": {
                "status": "not-run",
                "reason": "fixture pilot is screen-only",
            },
            "training": {
                "trainerStep": 320,
                "batchSize": 512,
            },
        }
        identity = {
            "runManifestSha256": manifest.sha256,
            "configSha256": runtime_config_sha256,
            "sourceSha256": self.source_sha256,
        }
        checkpoint = {
            "schemaVersion": 1,
            "generation": 4,
            "step": 320,
            "identity": identity,
            "model": model,
            "optimizer": {"state": {}, "param_groups": []},
            "trainerRuntime": {},
            "rng": {},
            "generationData": generation_data,
            "replaySnapshot": {},
        }
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            checkpoint_dir / "checkpoint-g00000004-s000000000320.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_record = {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "step": 320,
        }
        reports.write(
            4,
            {
                "checkpoint": checkpoint_record,
                "generationData": generation_data,
            },
            created_at=f"2026-07-25T00:00:{8 + generation_offset:02d}Z",
        )
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = {
            "schemaVersion": 1,
            "generation": 4,
            "step": 320,
            "file": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "identity": identity,
        }
        _write_json(metadata_path, metadata)
        _write_json(
            checkpoint_dir / "latest.json",
            {
                "schemaVersion": 1,
                "generation": 4,
                "step": 320,
                "metadata": metadata_path.name,
                "sha256": checkpoint_sha256,
            },
        )
        return {
            "runDir": run_dir,
            "manifestSha256": manifest.sha256,
            "runtimeConfigSha256": runtime_config_sha256,
            "checkpointPath": checkpoint_path,
            "checkpointSha256": checkpoint_sha256,
            "modelStateSha256": canonical_model_state_sha256(model),
        }

    def _decision(self) -> dict[str, object]:
        return {
            "candidateLane": "warm",
            "opponentLane": "cold",
            "selectedLane": "warm",
            "warmSelected": True,
            "rawWinRate": 1.0,
            "oneSided95Lower": 0.9,
            "checks": {
                "rawWinRate": {
                    "passed": True,
                    "requirement": ">= 0.55",
                },
                "oneSided95Lower": {
                    "passed": True,
                    "requirement": "> 0.50",
                },
            },
            "interval": {
                "method": "opening-pair-cluster-percentile",
                "bootstrapSeed": 20_260_714,
                "bootstrapSamples": 10_000,
            },
            "screenOnly": True,
            "formalChampionDeclaration": False,
            "reason": "warm passed fixture screen",
        }

    def _identity(self, lane: str) -> dict[str, object]:
        artifact = self.lanes[lane]
        return {
            "selectorId": f"phase4d-{lane}-g4",
            "engine": "native-mcts",
            "artifactHash": artifact["checkpointSha256"],
            "configHash": artifact["runtimeConfigSha256"],
            "sourceHash": self.source_sha256,
            "verified": True,
        }

    def _lane_report(self, lane: str) -> dict[str, object]:
        artifact = self.lanes[lane]
        checkpoint = artifact["checkpointPath"]
        assert isinstance(checkpoint, Path)
        return {
            "lane": lane,
            "status": "completed",
            "nextGeneration": 5,
            "manifestSha256": artifact["manifestSha256"],
            "sourceSha256": self.source_sha256,
            "latestCheckpoint": {
                "path": checkpoint.relative_to(self.root).as_posix(),
                "sha256": artifact["checkpointSha256"],
                "modelStateSha256": artifact["modelStateSha256"],
                "generation": 4,
                "step": 320,
            },
        }

    def _write_arena(
        self,
        contract: dict[str, object],
        opening_ids: list[str],
    ) -> None:
        warm_identity = self._identity("warm")
        cold_identity = self._identity("cold")
        runtime = {
            "kind": "native-mcts",
            "trusted": True,
            "identityBound": True,
            "configuredSimulations": 256,
            "backendHash": self.native_sha256,
        }
        games = []
        for opening_id in opening_ids:
            for leg in (0, 1):
                games.append(
                    {
                        "gameId": f"{opening_id}:leg-{leg}",
                        "openingId": opening_id,
                        "leg": leg,
                        "candidateWon": True,
                        "candidateIdentity": warm_identity,
                        "opponentIdentity": cold_identity,
                    }
                )
        summary = {
            "openingPairs": 32,
            "games": 64,
            "candidateWins": 64,
            "candidateLosses": 0,
            "draws": 0,
            "winRate": 1.0,
            "winRateCI": {
                "confidence": 0.95,
                "lower": 0.9,
                "upper": 1.0,
            },
        }
        schedule = [
            {
                "lane": lane,
                "generation": generation,
                "createdAt": (
                    f"2026-07-25T00:00:"
                    f"{2 * generation + (1 if lane == 'warm' else 0):02d}Z"
                ),
            }
            for generation in range(5)
            for lane in ("cold", "warm")
        ]
        decision = self._decision()
        report = {
            "schemaVersion": 1,
            "recordType": "phase4d-warm-cold-screen",
            "status": "completed",
            "createdAt": "2026-07-25T00:01:00Z",
            "contractPath": self.contract_path.relative_to(self.root).as_posix(),
            "contractSha256": contract["contractSha256"],
            "nativeBackendSha256": self.native_sha256,
            "schedule": schedule,
            "lanes": {
                "cold": self._lane_report("cold"),
                "warm": self._lane_report("warm"),
            },
            "artifacts": {
                lane: {
                    "checkpointPath": str(
                        self.lanes[lane]["checkpointPath"]
                    ),
                    "checkpointSha256": self.lanes[lane][
                        "checkpointSha256"
                    ],
                    "modelStateSha256": self.lanes[lane][
                        "modelStateSha256"
                    ],
                }
                for lane in ("cold", "warm")
            },
            "arena": {
                "schemaVersion": 1,
                "recordType": "paired-arena-result",
                "suite": "dev",
                "suiteHash": "sha256:" + "8" * 64,
                "candidateIdentity": warm_identity,
                "opponentIdentity": cold_identity,
                "candidateRuntime": runtime,
                "opponentRuntime": runtime,
                "summary": summary,
                "games": games,
            },
            "decision": decision,
        }
        self._commit_result(report, games, decision, contract["contractSha256"])

    def _commit_result(
        self,
        report: dict[str, object],
        games: list[dict[str, object]],
        decision: dict[str, object],
        contract_sha256: object,
    ) -> None:
        arena_root = self.pilot_root / "arena"
        arena_root.mkdir(parents=True, exist_ok=True)
        report_bytes = canonical_json_bytes(report)
        report_sha256 = sha256_bytes(report_bytes)
        report_path = arena_root / f"report-{report_sha256}.json"
        report_path.write_bytes(report_bytes)
        games_bytes = b"".join(canonical_json_bytes(game) for game in games)
        games_sha256 = sha256_bytes(games_bytes)
        games_path = arena_root / f"games-{games_sha256}.jsonl"
        games_path.write_bytes(games_bytes)
        result_core = {
            "schemaVersion": 1,
            "status": "completed",
            "createdAt": "2026-07-25T00:01:01Z",
            "contractSha256": contract_sha256,
            "report": {
                "file": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": report_sha256,
            },
            "gamesJsonl": {
                "file": games_path.name,
                "bytes": games_path.stat().st_size,
                "sha256": games_sha256,
                "completedGames": 64,
            },
            "decision": decision,
        }
        _write_json(
            self.result_path,
            _self_hashed(result_core, "resultManifestSha256"),
        )

    def rewrite_wrong_winner(self) -> None:
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        old_report = self.result_path.parent / result["report"]["file"]
        report = json.loads(old_report.read_text(encoding="utf-8"))
        decision = dict(report["decision"])
        decision["selectedLane"] = "cold"
        decision["warmSelected"] = False
        decision["reason"] = "tampered winner"
        report["decision"] = decision
        games = report["arena"]["games"]
        self._commit_result(
            report,
            games,
            decision,
            result["contractSha256"],
        )


class Phase4BootstrapBaselineTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], _PilotFixture]:
        temporary = tempfile.TemporaryDirectory()
        return temporary, _PilotFixture(Path(temporary.name))

    def test_prepare_is_model_only_idempotent_and_verifiable(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            first = prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            second = prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            verified = verify_bootstrap_baseline(
                workspace_root=fixture.root,
                baseline_root=BASELINE_RELATIVE,
            )

            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["manifestSha256"], second["manifestSha256"])
            self.assertEqual(first["manifestSha256"], verified["manifestSha256"])
            self.assertEqual(first["generation"], 4)
            self.assertEqual(first["step"], 320)
            self.assertTrue(verified["lineageVerified"])

            model_path = fixture.baseline_root / BASELINE_MODEL_NAME
            model = torch.load(model_path, map_location="cpu", weights_only=True)
            self.assertEqual(set(model), {
                "schemaVersion",
                "kind",
                "format",
                "generation",
                "step",
                "formalPromotion",
                "provenance",
                "modelStateSha256",
                "modelState",
            })
            self.assertEqual(model["kind"], BASELINE_KIND)
            self.assertEqual(model["format"], BASELINE_MODEL_FORMAT)
            self.assertFalse(model["formalPromotion"])
            for forbidden in ("optimizer", "replay", "rng", "trainerRuntime"):
                self.assertNotIn(forbidden, model)

            state = load_bootstrap_baseline_state(
                workspace_root=fixture.root,
                baseline_root=BASELINE_RELATIVE,
            )
            self.assertEqual(
                canonical_model_state_sha256(state),
                fixture.lanes["warm"]["modelStateSha256"],
            )
            manifest = json.loads(
                (fixture.baseline_root / BASELINE_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(manifest["formalPromotion"])
            self.assertFalse(manifest["championStorePublished"])

    def test_cli_prepare_and_verify(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "prepare",
                            "--workspace-root",
                            str(fixture.root),
                            "--pilot-root",
                            str(PILOT_RELATIVE),
                            "--output-root",
                            str(BASELINE_RELATIVE),
                        ]
                    ),
                    0,
                )
            self.assertTrue(json.loads(output.getvalue())["created"])

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "verify",
                            "--workspace-root",
                            str(fixture.root),
                            "--baseline-root",
                            str(BASELINE_RELATIVE),
                        ]
                    ),
                    0,
                )
            self.assertTrue(json.loads(output.getvalue())["lineageVerified"])

    def test_internally_rehashed_wrong_winner_is_rejected(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            fixture.rewrite_wrong_winner()
            with self.assertRaisesRegex(
                Phase4BaselineLineageError,
                "did not select warm",
            ):
                prepare_bootstrap_baseline(
                    workspace_root=fixture.root,
                    pilot_root=PILOT_RELATIVE,
                    output_root=BASELINE_RELATIVE,
                )

    def test_contract_tampering_and_incomplete_checkpoint_are_rejected(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            contract = json.loads(
                fixture.contract_path.read_text(encoding="utf-8")
            )
            contract["formalChampionDeclaration"] = True
            _write_json(fixture.contract_path, contract)
            with self.assertRaisesRegex(
                Phase4BaselineLineageError,
                "contract does not verify",
            ):
                prepare_bootstrap_baseline(
                    workspace_root=fixture.root,
                    pilot_root=PILOT_RELATIVE,
                    output_root=BASELINE_RELATIVE,
                )

        temporary, fixture = self.fixture()
        with temporary:
            checkpoint = fixture.lanes["warm"]["checkpointPath"]
            assert isinstance(checkpoint, Path)
            checkpoint.with_suffix(".json").unlink()
            with self.assertRaisesRegex(
                Phase4BaselineLineageError,
                "metadata",
            ):
                prepare_bootstrap_baseline(
                    workspace_root=fixture.root,
                    pilot_root=PILOT_RELATIVE,
                    output_root=BASELINE_RELATIVE,
                )

    def test_verify_rejects_manifest_and_model_tampering(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            manifest_path = fixture.baseline_root / BASELINE_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["formalPromotion"] = True
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                Phase4BaselineCorruptionError,
                "self hash",
            ):
                verify_bootstrap_baseline(
                    workspace_root=fixture.root,
                    baseline_root=BASELINE_RELATIVE,
                )

        temporary, fixture = self.fixture()
        with temporary:
            prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            (fixture.baseline_root / BASELINE_MODEL_NAME).write_bytes(
                b"tampered-model\n"
            )
            with self.assertRaisesRegex(
                Phase4BaselineCorruptionError,
                "model file does not verify",
            ):
                verify_bootstrap_baseline(
                    workspace_root=fixture.root,
                    baseline_root=BASELINE_RELATIVE,
                )

    def test_verify_rejects_incomplete_archive_and_prepare_recovers_orphan(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            manifest_path = fixture.baseline_root / BASELINE_MANIFEST_NAME
            manifest_path.unlink()
            with self.assertRaisesRegex(
                Phase4BaselineCorruptionError,
                "manifest",
            ):
                verify_bootstrap_baseline(
                    workspace_root=fixture.root,
                    baseline_root=BASELINE_RELATIVE,
                )
            recovered = prepare_bootstrap_baseline(
                workspace_root=fixture.root,
                pilot_root=PILOT_RELATIVE,
                output_root=BASELINE_RELATIVE,
            )
            self.assertTrue(recovered["created"])
            self.assertTrue(manifest_path.is_file())

    def test_prepare_rejects_conflicting_orphan_model(self) -> None:
        temporary, fixture = self.fixture()
        with temporary:
            fixture.baseline_root.mkdir(parents=True)
            torch.save(
                {
                    "schemaVersion": 1,
                    "kind": BASELINE_KIND,
                    "format": BASELINE_MODEL_FORMAT,
                    "generation": 4,
                    "step": 320,
                    "formalPromotion": False,
                    "provenance": {
                        "contractSha256": "0" * 64,
                        "resultManifestSha256": "0" * 64,
                        "checkpointSha256": "0" * 64,
                        "configSha256": "0" * 64,
                        "sourceSha256": "0" * 64,
                        "nativeBackendSha256": "0" * 64,
                    },
                    "modelStateSha256": canonical_model_state_sha256(
                        {"weight": torch.ones(1)}
                    ),
                    "modelState": {"weight": torch.ones(1)},
                },
                fixture.baseline_root / BASELINE_MODEL_NAME,
            )
            with self.assertRaisesRegex(
                Phase4BaselineCorruptionError,
                "orphan baseline model differs",
            ):
                prepare_bootstrap_baseline(
                    workspace_root=fixture.root,
                    pilot_root=PILOT_RELATIVE,
                    output_root=BASELINE_RELATIVE,
                )


if __name__ == "__main__":
    unittest.main()
