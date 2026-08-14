from __future__ import annotations

import json
import random
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import quoridor.phase4_training as phase4_training

from quoridor.codec import (
    encode_move,
    mirror_action_index,
    position_hash,
    to_canonical_action_index,
)
from quoridor.config import (
    ChallengeReplaySettings,
    HealthHoldoutSettings,
    InitializationSettings,
    TrainingSettings,
    load_experiment_config,
    seed_everything,
)
from quoridor.champion_store import ArtifactReference, ChampionStore
from quoridor.challenge_replay import merge_tensor_batches, mixed_batch_counts
from quoridor.game import State, apply_action_index
from quoridor.network import PHASE4_INPUT_CHANNELS
from quoridor.network import PolicyValueNetwork
from quoridor.phase4_training import (
    Phase4TrainingDriver,
    Phase4TrainingError,
    Phase4ResumeError,
    Phase4ResourceLimitReached,
    ReplayBatchEncoder,
    _RuntimeBudgetGuard,
    _network_config,
    _replay_prefix_sha256,
    _selfplay_invocation,
    evaluate_fixed_health_holdout,
    fixed_holdout_selection,
    load_phase4_warm_artifact,
    load_phase4_champion_artifact,
    load_phase4_recovery_artifact,
    prepare_phase4_lineage,
    pure_selfplay_selection,
    reconcile_best_fixed_holdout_candidate,
    replay_game_from_selfplay,
    validate_phase4_config,
)
from quoridor.paired_arena import loaded_native_backend_hash
from quoridor.phase4_checkpoint import (
    Phase4CheckpointLoad,
    Phase4CheckpointRecord,
    canonical_model_state_sha256,
)
from quoridor.replay_store import (
    ReplayGame,
    ReplayPly,
    ReplaySample,
    ReplayStore,
    reconstruct_game,
)
from quoridor.run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ResourceBudget,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from quoridor.selfplay import SelfPlayGame, SelfPlayPly, scheduled_search_seed
from quoridor.trainer import TrainMetrics


ROOT = Path(__file__).resolve().parents[2]
FIXED_ENVIRONMENT = {
    "python": {"version": "test"},
    "torch": {"version": "test", "cudaAvailable": False},
    "platform": {"system": "test"},
}


def nonterminal_player_one_state() -> State:
    initial = State(
        pawns=((8, 4), (0, 4)),
        walls_remaining=(10, 10),
        current_player=0,
        move_number=0,
        position_counts={},
    )
    initial = replace(initial, position_counts={position_hash(initial): 1})
    return apply_action_index(initial, encode_move(7, 4))


class ReplayBatchEncoderTests(unittest.TestCase):
    def test_visit_counts_follow_canonical_and_mirror_transforms(self) -> None:
        state = nonterminal_player_one_state()
        action = encode_move(1, 4)
        counts = [0] * 209
        counts[action] = 17
        sample = ReplaySample(
            game_id="game",
            ply_index=0,
            state=state,
            action_index=action,
            visit_counts=tuple(counts),
            root_value=0.0,
            outcome=-1.0,
            model_generation=0,
            model_id="model",
            game_metadata={},
            ply_metadata={},
        )
        encoder = ReplayBatchEncoder(
            input_channels=PHASE4_INPUT_CHANNELS,
            device="cpu",
        )
        identity = encoder.encode(
            [sample], rng=random.Random(1), mirror_probability=0.0
        )
        expected = to_canonical_action_index(action, state.current_player)
        self.assertEqual(identity["policies"][0, expected], 1)
        self.assertEqual(tuple(identity["states"].shape), (1, 16, 9, 9))
        self.assertEqual(identity["values"].tolist(), [-1.0])

        mirrored = encoder.encode(
            [sample], rng=random.Random(1), mirror_probability=1.0
        )
        expected_mirror = to_canonical_action_index(
            mirror_action_index(action), state.current_player
        )
        self.assertEqual(mirrored["policies"][0, expected_mirror], 1)
        self.assertTrue(torch.allclose(mirrored["policies"].sum(1), torch.ones(1)))


class ChallengeReplayMixTests(unittest.TestCase):
    def test_mixed_batch_count_and_tensor_concatenation(self) -> None:
        self.assertEqual(mixed_batch_counts(512, 0.25), (384, 128))
        selfplay = {
            "states": torch.zeros((3, 2)),
            "policies": torch.zeros((3, 4)),
            "values": torch.zeros((3,)),
        }
        challenge = {
            "states": torch.ones((1, 2)),
            "policies": torch.ones((1, 4)),
            "values": torch.ones((1,)),
        }
        merged = merge_tensor_batches(selfplay, challenge)
        self.assertEqual(tuple(merged["states"].shape), (4, 2))
        self.assertTrue(torch.equal(merged["states"][-1], torch.ones(2)))

    def test_learning_stages_enforce_teacher_isolation(self) -> None:
        challenge = ChallengeReplaySettings(
            root=Path("challenge"),
            contract_path=Path("contract.json"),
            contract_sha256="1" * 64,
            dataset_fingerprint="2" * 64,
            dataset_manifest_sha256="3" * 64,
            train_manifest_sha256="4" * 64,
            fraction=0.4,
        )
        holdout = HealthHoldoutSettings(
            root=challenge.root,
            contract_path=challenge.contract_path,
            contract_sha256=challenge.contract_sha256,
            dataset_fingerprint=challenge.dataset_fingerprint,
            dataset_manifest_sha256=challenge.dataset_manifest_sha256,
            train_manifest_sha256=challenge.train_manifest_sha256,
            holdout_manifest_sha256="5" * 64,
        )
        guided = TrainingSettings(
            challenge_replay=challenge,
            health_holdout=holdout,
            learning_stage="classic-hard-guided",
        )
        self.assertEqual(guided.challenge_replay.fraction, 0.4)
        pure = TrainingSettings(
            health_holdout=holdout,
            learning_stage="pure-self-play",
        )
        self.assertIsNone(pure.challenge_replay)
        with self.assertRaisesRegex(ValueError, "requires challengeReplay"):
            TrainingSettings(
                health_holdout=holdout,
                learning_stage="classic-hard-guided",
            )
        with self.assertRaisesRegex(ValueError, "must not declare challengeReplay"):
            TrainingSettings(
                challenge_replay=challenge,
                health_holdout=holdout,
                learning_stage="pure-self-play",
            )


class FixedHealthHoldoutTests(unittest.TestCase):
    def test_full_holdout_evaluation_is_stratified_and_selects_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = State(
                pawns=((1, 4), (0, 8)),
                walls_remaining=(0, 0),
                current_player=0,
                move_number=0,
                position_counts={},
            )
            initial = replace(
                initial,
                position_counts={position_hash(initial): 1},
            )
            action = encode_move(0, 4)
            counts = [0] * 209
            counts[action] = 8
            replay = ReplayStore(
                root / "holdout",
                max_games=2,
                max_plies=2,
                max_bytes=1_000_000,
            )
            replay.append(
                ReplayGame(
                    game_id="holdout-0",
                    initial_state=initial,
                    plies=(
                        ReplayPly(
                            action_index=action,
                            visit_counts=tuple(counts),
                            root_value=1.0,
                            model_generation=0,
                            model_id="classic-hard",
                        ),
                    ),
                    metadata={"teacherEngine": "classic-negamax"},
                )
            )
            cache = phase4_training.ReplayTensorCache(
                root / "cache",
                replay,
                source_sha256="1" * 64,
                encoder_sha256="2" * 64,
            )
            cache.refresh()
            base = load_experiment_config(
                ROOT / "training/configs/phase4-bootstrap.json"
            )
            network = replace(
                base.network,
                channels=8,
                residual_blocks=1,
                policy_channels=2,
                value_channels=2,
                value_hidden=8,
            )
            model = PolicyValueNetwork(
                _network_config(replace(base, network=network))
            )
            metrics = evaluate_fixed_health_holdout(
                model,
                cache,
                batch_size=1,
                device=torch.device("cpu"),
                seed=7,
                use_amp=False,
            )
            self.assertEqual(metrics["positions"], 1)
            self.assertEqual(metrics["bySideToMove"]["0"]["positions"], 1)
            self.assertEqual(metrics["byPlyBand"]["early"]["positions"], 1)

            reports = phase4_training.GenerationReportStore(
                root / "reports",
                run_manifest_sha256="a" * 64,
            )
            first = {**metrics, "totalLoss": metrics["totalLoss"] + 1.0}
            reports.write(
                0,
                {"generationData": {"fixedHealthHoldout": first}},
                created_at="2026-08-08T00:00:00Z",
            )
            selection = fixed_holdout_selection(
                reports,
                generation=1,
                current=metrics,
            )
            self.assertTrue(selection["currentIsBest"])
            self.assertEqual(selection["bestGeneration"], 1)

    def test_best_candidate_index_is_rebuilt_from_immutable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            reports = phase4_training.GenerationReportStore(
                root / "reports",
                run_manifest_sha256="a" * 64,
            )
            losses = (0.75, 0.90)
            for generation, loss in enumerate(losses):
                checkpoint = checkpoints / (
                    f"checkpoint-g{generation:08d}-s{generation + 1:012d}.pt"
                )
                candidate = checkpoints / f"candidate-model-{checkpoint.name}"
                checkpoint.write_bytes(f"checkpoint-{generation}".encode("ascii"))
                candidate.write_bytes(f"candidate-{generation}".encode("ascii"))
                reports.write(
                    generation,
                    {
                        "checkpoint": {
                            "path": checkpoint.name,
                            "sha256": sha256_file(checkpoint),
                            "step": generation + 1,
                            "candidateModel": {
                                "path": candidate.name,
                                "sha256": sha256_file(candidate),
                                "modelStateSha256": str(generation + 1) * 64,
                            },
                        },
                        "generationData": {
                            "fixedHealthHoldout": {"totalLoss": loss},
                        },
                    },
                    created_at=f"2026-08-08T00:00:0{generation}Z",
                )
            (root / "best-candidate.json").write_text("corrupt", encoding="ascii")

            marker = reconcile_best_fixed_holdout_candidate(root, reports)

            self.assertIsNotNone(marker)
            self.assertEqual(marker["generation"], 0)
            self.assertFalse(marker["selection"]["currentIsBest"])
            self.assertEqual(
                json.loads((root / "best-candidate.json").read_bytes()),
                marker,
            )

    def test_pure_selfplay_selection_uses_latest_not_classic_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports = phase4_training.GenerationReportStore(
                Path(directory) / "reports",
                run_manifest_sha256="a" * 64,
            )
            reports.write(
                0,
                {
                    "generationData": {
                        "training": {"learningStage": "pure-self-play"},
                        "fixedHealthHoldout": {"totalLoss": 0.5},
                    }
                },
                created_at="2026-08-08T00:00:00Z",
            )

            selection = pure_selfplay_selection(reports, generation=1)

            self.assertEqual(selection["bestGeneration"], 1)
            self.assertTrue(selection["classicHoldoutDiagnosticOnly"])
            self.assertNotIn("bestTotalLoss", selection)


class ChampionInitializationTests(unittest.TestCase):
    def test_published_champion_load_is_manifest_and_tensor_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            control = workspace / "training/runs/phase4"
            control.mkdir(parents=True)
            arena_path = control / "arena.json"
            arena_path.write_text("{}", encoding="ascii")
            base = load_experiment_config(ROOT / "training/configs/phase4e-bootstrap-v1.json")
            seed_everything(base.seed)
            model = PolicyValueNetwork(_network_config(base))
            with torch.no_grad():
                next(model.parameters()).add_(0.25)
            store = ChampionStore(control / "champions", artifact_root=control)
            record = store.publish(
                model,
                generation=14,
                parent_id=None,
                config_sha256="1" * 64,
                source_sha256="2" * 64,
                opening_suite_sha256="3" * 64,
                arena_report=ArtifactReference.from_file(
                    arena_path, relative_to=control
                ),
                accepted_at="2026-08-02T00:00:00Z",
            )
            manifest_path = store.manifest_path
            manifest = json.loads(manifest_path.read_bytes())
            artifact_path = store.root / str(record.file_name)
            expected_state_sha256 = canonical_model_state_sha256(model)
            config = replace(
                base,
                run_dir=workspace / "run",
                initialization=InitializationSettings(
                    mode="champion",
                    expected_base_model_sha256=base.initialization.expected_base_model_sha256,
                    artifact_path=artifact_path,
                    artifact_sha256=record.model_sha256,
                    champion_id=record.champion_id,
                    champion_manifest_path=manifest_path,
                    champion_manifest_sha256=manifest["manifestSha256"],
                    champion_manifest_file_sha256=sha256_file(manifest_path),
                    model_state_sha256=expected_state_sha256,
                    reset_optimizer=True,
                ),
            )
            state, provenance = load_phase4_champion_artifact(
                workspace_root=workspace,
                config=config,
            )
            self.assertEqual(canonical_model_state_sha256(state), expected_state_sha256)
            self.assertEqual(provenance["championId"], record.champion_id)

            tampered = replace(
                config,
                initialization=replace(
                    config.initialization,
                    artifact_sha256="f" * 64,
                ),
            )
            with self.assertRaisesRegex(Phase4ResumeError, "artifact SHA-256"):
                load_phase4_champion_artifact(
                    workspace_root=workspace,
                    config=tampered,
                )


class RecoveryInitializationTests(unittest.TestCase):
    def _fixture(self, workspace: Path):
        base = load_experiment_config(ROOT / "training/configs/phase4-bootstrap.json")
        network = replace(
            base.network,
            channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=2,
            value_hidden=8,
        )
        provisional = replace(
            base,
            name="recovery-initialization-test",
            device="cpu",
            run_dir=workspace / "run",
            network=network,
            self_play=replace(base.self_play, games=1, actors=1),
            training=replace(
                base.training,
                replay_capacity=1280,
                batch_size=2,
                training_steps=phase4_training.REPLAY_INPUT_GATE_BATCHES,
                max_checkpoints=2,
            ),
        )
        seed_everything(provisional.seed)
        cold = PolicyValueNetwork(_network_config(provisional))
        base_sha256 = canonical_model_state_sha256(cold)
        with torch.no_grad():
            next(cold.parameters()).add_(0.125)
        state_sha256 = canonical_model_state_sha256(cold)
        archive = workspace / "archive"
        archive.mkdir()
        artifact = archive / "candidate-model.pt"
        source_sha256 = "1" * 64
        config_sha256 = "2" * 64
        run_manifest_sha256 = "3" * 64
        checkpoint_sha256 = "4" * 64
        torch.save(
            {
                "schemaVersion": 1,
                "format": "quoridor-phase4-candidate-model-only",
                "generation": 7,
                "step": 640,
                "identity": {
                    "runManifestSha256": run_manifest_sha256,
                    "configSha256": config_sha256,
                    "sourceSha256": source_sha256,
                },
                "checkpointSha256": checkpoint_sha256,
                "modelStateSha256": state_sha256,
                "modelState": cold.state_dict(),
            },
            artifact,
        )
        artifact_sha256 = sha256_file(artifact)
        core = {
            "schemaVersion": 1,
            "kind": "phase4-recovery-candidate-archive",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "candidate": {
                "candidateModelSha256": artifact_sha256,
                "checkpointSha256": checkpoint_sha256,
                "configSha256": config_sha256,
                "generation": 7,
                "modelStateSha256": state_sha256,
                "runManifestSha256": run_manifest_sha256,
                "sourceSha256": source_sha256,
                "step": 640,
            },
            "files": {
                "candidateModel": {
                    "path": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": artifact_sha256,
                }
            },
            "bytes": artifact.stat().st_size,
        }
        archive_sha256 = sha256_bytes(canonical_json_bytes(core))
        manifest = workspace / "archive-manifest.json"
        manifest.write_bytes(
            canonical_json_bytes({**core, "archiveSha256": archive_sha256})
        )
        config = replace(
            provisional,
            initialization=InitializationSettings(
                mode="recovery",
                expected_base_model_sha256=base_sha256,
                artifact_path=artifact,
                artifact_sha256=artifact_sha256,
                recovery_manifest_path=manifest,
                recovery_manifest_sha256=archive_sha256,
                recovery_manifest_file_sha256=sha256_file(manifest),
                recovery_source_sha256=source_sha256,
                recovery_config_sha256=config_sha256,
                recovery_run_manifest_sha256=run_manifest_sha256,
                recovery_checkpoint_sha256=checkpoint_sha256,
                recovery_generation=7,
                recovery_step=640,
                model_state_sha256=state_sha256,
                reset_optimizer=True,
            ),
        )
        return config, state_sha256

    def test_recovery_artifact_is_archive_bound_and_initializes_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="ascii")
            config, state_sha256 = self._fixture(workspace)
            state, provenance = load_phase4_recovery_artifact(
                workspace_root=workspace,
                config=config,
            )
            self.assertEqual(canonical_model_state_sha256(state), state_sha256)
            self.assertEqual(provenance["kind"], "recovery-candidate")

            lineage = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=ResourceBudget(
                    gpu_hours=10,
                    metadata={"nativeBackendSha256": loaded_native_backend_hash()},
                ),
                resume=False,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )

            class Guard:
                @staticmethod
                def check(*, force=False, reserve_bytes=0):
                    return {"force": force, "reserveBytes": reserve_bytes}

            driver = Phase4TrainingDriver(
                workspace_root=workspace,
                control_root=workspace / "training/runs/phase4",
                config=config,
                device=torch.device("cpu"),
                lineage=lineage,
                budget_guard=Guard(),
            )
            driver.initialize_new_run()
            self.assertEqual(driver.loaded.record.generation, -1)
            self.assertEqual(driver.loaded.payload["generationData"]["mode"], "recovery")
            self.assertEqual(canonical_model_state_sha256(driver.model), state_sha256)
            self.assertFalse(driver.trainer.optimizer.state)
            self.assertEqual(driver.replay.game_count, 0)

    def test_recovery_artifact_rejects_hash_and_lineage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config, _ = self._fixture(workspace)
            bad_artifact = replace(
                config,
                initialization=replace(
                    config.initialization,
                    artifact_sha256="f" * 64,
                ),
            )
            with self.assertRaisesRegex(Phase4ResumeError, "artifact SHA-256"):
                load_phase4_recovery_artifact(
                    workspace_root=workspace,
                    config=bad_artifact,
                )
            bad_lineage = replace(
                config,
                initialization=replace(
                    config.initialization,
                    recovery_config_sha256="e" * 64,
                ),
            )
            with self.assertRaisesRegex(Phase4ResumeError, "provenance mismatch"):
                load_phase4_recovery_artifact(
                    workspace_root=workspace,
                    config=bad_lineage,
                )


class ReplayCacheSelectionTests(unittest.TestCase):
    def test_phase4d_keeps_legacy_cache_and_phase4e_uses_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = ReplayStore(
                root / "replay",
                max_games=10,
                max_plies=10,
                max_bytes=1_000_000,
            )
            legacy = phase4_training._create_phase4_replay_cache(
                replay=replay,
                run_dir=root / "phase4d",
                execution_contract=phase4_training.PHASE4D_EXECUTION_CONTRACT,
                source_sha256="1" * 64,
                encoder_sha256="2" * 64,
            )
            self.assertIsInstance(
                legacy,
                phase4_training.ReplayReconstructionCache,
            )

            prepared = phase4_training.Phase4ExecutionContract.from_dict(
                {
                    "schemaVersion": 1,
                    "stage": "phase4e",
                    "status": "prepared",
                    "purpose": "cpu-test",
                    "formalPromotion": True,
                    "maximumGenerations": 64,
                    "stageBudget": {
                        "status": "pending",
                        "baselineGlobalHours": None,
                        "limitHours": None,
                        "baselineRunId": None,
                        "baselineReportSha256": None,
                    },
                }
            )
            with patch.object(
                phase4_training,
                "load_native_module",
                return_value=SimpleNamespace(
                    encode_states=lambda *_args, **_kwargs: None
                ),
            ):
                tensor = phase4_training._create_phase4_replay_cache(
                    replay=replay,
                    run_dir=root / "phase4e",
                    execution_contract=prepared,
                    source_sha256="1" * 64,
                    encoder_sha256="2" * 64,
                )
            self.assertIsInstance(tensor, phase4_training.ReplayTensorCache)
            self.assertEqual(tensor.root, root / "phase4e/replay-tensor-cache")


class ReplayConversionTests(unittest.TestCase):
    def test_selfplay_conversion_records_honest_scheduler_metadata(self) -> None:
        initial = State(
            pawns=((1, 4), (0, 8)),
            walls_remaining=(0, 0),
            current_player=0,
            move_number=0,
            position_counts={},
        )
        history = {position_hash(initial): 1}
        initial = replace(initial, position_counts=history)
        action = encode_move(0, 4)
        counts = [0] * 209
        counts[action] = 4
        game = SelfPlayGame(
            samples=(),
            initial_state=initial,
            raw_plies=(SelfPlayPly(action, tuple(counts), 0.25),),
            winner=0,
            move_count=1,
            termination_reason="goal",
        )
        replay = replay_game_from_selfplay(
            game,
            game_id="phase4-g00000000-game000003",
            generation=0,
            model_id="initial:model",
            scheduler_seed=17,
            game_index=3,
            resumed_subset=True,
            invocation=_selfplay_invocation(
                generation=0,
                model_id="initial:model",
                scheduler_seed=17,
                scheduled_game_indices=[3],
                configured_actors=32,
                simulations=256,
                temperature_moves=24,
            ),
        )
        self.assertEqual(reconstruct_game(replay)[-1].winner, 0)
        self.assertEqual(replay.metadata["schedulerSeed"], 17)
        self.assertEqual(replay.metadata["gameIndex"], 3)
        self.assertTrue(replay.metadata["resumedSubset"])
        self.assertEqual(
            replay.metadata["invocation"]["searchSeed"],
            scheduled_search_seed(17),
        )
        config = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
        driver.config = config
        driver._validate_game_invocation(
            replay,
            generation=0,
            model_id="initial:model",
            generation_seed=17,
            game_index=3,
        )
        bad_invocation = dict(replay.metadata["invocation"])
        bad_invocation["actualActors"] = 31
        tampered = replace(
            replay,
            metadata={**replay.metadata, "invocation": bad_invocation},
        )
        with self.assertRaisesRegex(Phase4ResumeError, "hash or workload"):
            driver._validate_game_invocation(
                tampered,
                generation=0,
                model_id="initial:model",
                generation_seed=17,
                game_index=3,
            )


class ReplayResumeBindingTests(unittest.TestCase):
    @staticmethod
    def _game(game_id: str, generation: int, model_id: str) -> ReplayGame:
        initial = State(
            pawns=((1, 4), (0, 8)),
            walls_remaining=(0, 0),
            current_player=0,
            move_number=0,
            position_counts={},
        )
        initial = replace(initial, position_counts={position_hash(initial): 1})
        action = encode_move(0, 4)
        counts = [0] * 209
        counts[action] = 2
        return ReplayGame(
            game_id=game_id,
            initial_state=initial,
            plies=(
                ReplayPly(
                    action_index=action,
                    visit_counts=tuple(counts),
                    root_value=0.0,
                    model_generation=generation,
                    model_id=model_id,
                ),
            ),
            metadata={"generation": generation, "modelId": model_id},
        )

    def test_checkpoint_allows_only_valid_next_generation_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = ReplayStore(
                root,
                max_games=10,
                max_plies=10,
                max_bytes=1_000_000,
            )
            replay.append(
                self._game("phase4-g00000000-game000000", 0, "initial:model")
            )
            verified = replay.verify(deep=True)
            shard_bytes = sum(shard.byte_size for shard in replay.shards)
            snapshot = {
                "manifestSha256": sha256_file(replay.manifest_path),
                "games": verified.game_count,
                "plies": verified.ply_count,
                "bytes": shard_bytes,
                "manifestBytes": verified.committed_bytes - shard_bytes,
                "committedBytes": verified.committed_bytes,
                "deepVerified": True,
                "gameIds": list(replay.game_ids),
                "prefixSha256": _replay_prefix_sha256(replay),
            }
            record = Phase4CheckpointRecord(
                generation=0,
                step=1,
                data_path=root / "checkpoint.pt",
                metadata_path=root / "checkpoint.json",
                byte_size=1,
                sha256="d" * 64,
            )
            loaded = Phase4CheckpointLoad(
                record=record,
                payload={"replaySnapshot": snapshot},
                rejected_newer=(),
            )
            config = load_experiment_config(
                ROOT / "training/configs/phase4-bootstrap.json"
            )
            driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
            driver.replay = replay
            driver.config = replace(
                config, self_play=replace(config.self_play, games=1)
            )
            driver._validate_replay_extension(replay.verify(deep=True), loaded)

            bad_prefix = dict(snapshot)
            bad_prefix["prefixSha256"] = "f" * 64
            with self.assertRaisesRegex(Phase4ResumeError, "shard prefix"):
                driver._validate_replay_extension(
                    replay.verify(deep=True),
                    replace(loaded, payload={"replaySnapshot": bad_prefix}),
                )

            next_model = f"sha256:{record.sha256}"
            generation_seed = (config.seed + 1_000_003) % (2**63)
            invocation = _selfplay_invocation(
                generation=1,
                model_id=next_model,
                scheduler_seed=generation_seed,
                scheduled_game_indices=[0],
                configured_actors=config.self_play.actors,
                simulations=config.mcts.simulations,
                temperature_moves=config.self_play.temperature_moves,
            )
            game = self._game("phase4-g00000001-game000000", 1, next_model)
            replay.append(
                replace(
                    game,
                    metadata={
                        **game.metadata,
                        "schedulerSeed": generation_seed,
                        "gameIndex": 0,
                        "resumedSubset": False,
                        "invocationGameIndex": 0,
                        "invocation": invocation,
                    },
                )
            )
            driver._validate_replay_extension(replay.verify(deep=True), loaded)

            replay.append(
                self._game("phase4-g00000002-game000000", 2, "foreign")
            )
            with self.assertRaisesRegex(Phase4ResumeError, "beyond"):
                driver._validate_replay_extension(replay.verify(deep=True), loaded)

    def test_fifo_eviction_can_resume_only_as_the_valid_next_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = ReplayStore(
                root / "replay",
                max_games=1,
                max_plies=1,
                max_bytes=1_000_000,
            )
            replay.append(
                self._game("phase4-g00000000-game000000", 0, "initial:model")
            )
            cache = phase4_training.ReplayTensorCache(
                root / "cache",
                replay,
                source_sha256="1" * 64,
                encoder_sha256="2" * 64,
            )
            cache.refresh()
            snapshot = phase4_training._replay_snapshot(
                replay,
                phase4_training._replay_cache_verification(cache, deep=True),
            )
            record = Phase4CheckpointRecord(
                generation=0,
                step=1,
                data_path=root / "checkpoint.pt",
                metadata_path=root / "checkpoint.json",
                byte_size=1,
                sha256="d" * 64,
            )
            loaded = Phase4CheckpointLoad(
                record=record,
                payload={"replaySnapshot": snapshot},
                rejected_newer=(),
            )
            config = load_experiment_config(
                ROOT / "training/configs/phase4-bootstrap.json"
            )
            config = replace(
                config,
                self_play=replace(config.self_play, games=1),
            )
            next_model = f"sha256:{record.sha256}"
            generation_seed = (config.seed + 1_000_003) % (2**63)
            invocation = _selfplay_invocation(
                generation=1,
                model_id=next_model,
                scheduler_seed=generation_seed,
                scheduled_game_indices=[0],
                configured_actors=config.self_play.actors,
                simulations=config.mcts.simulations,
                temperature_moves=config.self_play.temperature_moves,
            )
            next_game = self._game(
                "phase4-g00000001-game000000",
                1,
                next_model,
            )
            replay.append(
                replace(
                    next_game,
                    metadata={
                        **next_game.metadata,
                        "schedulerSeed": generation_seed,
                        "gameIndex": 0,
                        "resumedSubset": False,
                        "invocationGameIndex": 0,
                        "invocation": invocation,
                    },
                )
            )
            self.assertEqual(
                replay.game_ids,
                ("phase4-g00000001-game000000",),
            )
            cache.refresh()

            driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
            driver.replay = replay
            driver.replay_cache = cache
            driver.config = config
            driver._validate_replay_extension(
                phase4_training._replay_cache_verification(cache, deep=True),
                loaded,
            )

            tampered = copy.deepcopy(snapshot)
            tampered["shards"][0]["sequence"] = "0"
            with self.assertRaisesRegex(
                Phase4ResumeError,
                "shard values",
            ):
                driver._validate_replay_extension(
                    phase4_training._replay_cache_verification(cache),
                    replace(loaded, payload={"replaySnapshot": tampered}),
                )


class ConsecutiveGenerationTests(unittest.TestCase):
    def test_same_process_second_generation_uses_committed_replay_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = ReplayStore(
                root / "replay",
                max_games=10,
                max_plies=10,
                max_bytes=1_000_000,
            )
            cache = phase4_training.ReplayReconstructionCache(replay)
            cache.refresh()
            initial_snapshot = phase4_training._replay_snapshot(
                replay,
                cache.verification(),
            )
            config = load_experiment_config(
                ROOT / "training/configs/phase4-bootstrap.json"
            )
            config = replace(
                config,
                self_play=replace(config.self_play, games=1, actors=1),
                training=replace(
                    config.training,
                    batch_size=1,
                    training_steps=phase4_training.REPLAY_INPUT_GATE_BATCHES,
                ),
            )

            class Guard:
                @staticmethod
                def check(*, force=False, reserve_bytes=0):
                    return {"force": force, "reserveBytes": reserve_bytes}

            class Trainer:
                def __init__(self):
                    self.step = 0

                def train_step(self, _batch):
                    self.step += 1
                    return TrainMetrics(
                        loss=1.0,
                        policy_loss=0.5,
                        value_loss=0.5,
                        grad_norm=1.0,
                        learning_rate=0.001,
                    )

            class Checkpoints:
                def __init__(self, checkpoint_root: Path, trainer: Trainer):
                    self.root = checkpoint_root
                    self.root.mkdir()
                    self.trainer = trainer

                def save(self, *, generation, **_kwargs):
                    token = chr(ord("b") + generation)
                    candidate_token = chr(ord("d") + generation)
                    state_token = "f" if generation == 0 else "0"
                    return Phase4CheckpointRecord(
                        generation=generation,
                        step=self.trainer.step,
                        data_path=self.root / f"checkpoint-{generation}.pt",
                        metadata_path=self.root / f"checkpoint-{generation}.json",
                        byte_size=1,
                        sha256=token * 64,
                        candidate_model_path=(
                            self.root / f"candidate-model-{generation}.pt"
                        ),
                        candidate_model_byte_size=2,
                        candidate_model_sha256=candidate_token * 64,
                        model_state_sha256=state_token * 64,
                    )

            class Reports:
                def __init__(self, report_root: Path):
                    self.root = report_root
                    self.root.mkdir()
                    self.payloads = {}

                def write(self, generation, payload):
                    self.payloads[generation] = payload
                    return SimpleNamespace(
                        path=self.root / f"generation-{generation}.json",
                        sha256=str(generation) * 64,
                    )

            trainer = Trainer()
            driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
            driver.config = config
            driver.device = torch.device("cpu")
            driver.model = object()
            driver.trainer = trainer
            driver.replay = replay
            driver.replay_cache = cache
            driver.batch_encoder = SimpleNamespace(encode=lambda *_args, **_kwargs: {})
            driver.native_module = object()
            driver.budget_guard = Guard()
            driver.checkpoints = Checkpoints(root / "checkpoints", trainer)
            driver.reports = Reports(root / "reports")
            driver.progress_callback = None
            driver.fallback_rejections = ()
            driver.loaded = Phase4CheckpointLoad(
                record=Phase4CheckpointRecord(
                    generation=-1,
                    step=0,
                    data_path=root / "initial.pt",
                    metadata_path=root / "initial.json",
                    byte_size=1,
                    sha256="a" * 64,
                ),
                payload={"replaySnapshot": initial_snapshot},
                rejected_newer=(),
            )

            def converted_game(
                _selfplay_game,
                *,
                game_id,
                generation,
                model_id,
                **_kwargs,
            ):
                column = 4 + generation
                initial = State(
                    pawns=((1, column), (0, 8)),
                    walls_remaining=(0, 0),
                    current_player=0,
                    move_number=0,
                    position_counts={},
                )
                initial = replace(
                    initial,
                    position_counts={position_hash(initial): 1},
                )
                action = encode_move(0, column)
                counts = [0] * 209
                counts[action] = 2
                return ReplayGame(
                    game_id=game_id,
                    initial_state=initial,
                    plies=(
                        ReplayPly(
                            action_index=action,
                            visit_counts=tuple(counts),
                            root_value=0.0,
                            model_generation=generation,
                            model_id=model_id,
                        ),
                    ),
                    metadata={
                        "generation": generation,
                        "modelId": model_id,
                        "winner": 0,
                        "terminationReason": "goal",
                    },
                )

            driver._validate_game_invocation = Mock(
                side_effect=lambda _game, *, generation, **_kwargs: {
                    "invocationId": f"generation-{generation}",
                    "actualActors": 1,
                }
            )
            evaluator = Mock()

            def complete_one_game(
                _evaluator,
                _config,
                game_count,
                *,
                completed_game_callback,
                **_kwargs,
            ):
                self.assertEqual(game_count, 1)
                completed_game_callback(0, object())

            with (
                patch.object(
                    phase4_training,
                    "TorchEvaluator",
                    return_value=evaluator,
                ),
                patch.object(
                    phase4_training,
                    "play_self_games_scheduled",
                    side_effect=complete_one_game,
                ),
                patch.object(
                    phase4_training,
                    "replay_game_from_selfplay",
                    side_effect=converted_game,
                ),
                patch.object(
                    phase4_training,
                    "_checkpoint_write_reserve_bytes",
                    return_value=1,
                ),
            ):
                summaries = driver.run(2)

            self.assertEqual([item["generation"] for item in summaries], [0, 1])
            self.assertEqual(driver.loaded.record.generation, 1)
            self.assertEqual(
                driver.loaded.payload["replaySnapshot"]["gameIds"],
                [
                    "phase4-g00000000-game000000",
                    "phase4-g00000001-game000000",
                ],
            )
            self.assertEqual(replay.game_count, 2)
            self.assertEqual(cache.verification().game_count, 2)
            latest = driver.loaded.record
            self.assertEqual(
                driver.reports.payloads[1]["checkpoint"]["candidateModel"],
                {
                    "path": latest.candidate_model_path.name,
                    "bytes": latest.candidate_model_byte_size,
                    "sha256": latest.candidate_model_sha256,
                    "modelStateSha256": latest.model_state_sha256,
                },
            )


class ReportReconciliationTests(unittest.TestCase):
    def test_report_beyond_latest_valid_checkpoint_is_always_rejected(self) -> None:
        class Reports:
            @staticmethod
            def generations():
                return (0, 1)

            @staticmethod
            def load(generation):
                return generation

        driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
        driver.reports = Reports()
        driver.loaded = Phase4CheckpointLoad(
            record=Phase4CheckpointRecord(
                generation=0,
                step=1,
                data_path=Path("checkpoint.pt"),
                metadata_path=Path("checkpoint.json"),
                byte_size=1,
                sha256="d" * 64,
            ),
            payload={"generationData": {}},
            rejected_newer=(),
        )

        with self.assertRaisesRegex(
            Phase4ResumeError,
            "report exists beyond the latest valid checkpoint",
        ):
            driver._reconcile_reports()

    def test_reconciled_checkpoint_at_target_does_not_advance(self) -> None:
        class Reports:
            def __init__(self):
                self.writes = []

            @staticmethod
            def generations():
                return ()

            def write(self, generation, report):
                self.writes.append((generation, report))
                return None

        class Guard:
            def __init__(self):
                self.calls = []

            def check(self, *, force=False, reserve_bytes=0):
                self.calls.append((force, reserve_bytes))
                return {}

        driver = Phase4TrainingDriver.__new__(Phase4TrainingDriver)
        driver.reports = Reports()
        driver.budget_guard = Guard()
        driver.loaded = Phase4CheckpointLoad(
            record=Phase4CheckpointRecord(
                generation=0,
                step=64,
                data_path=Path("checkpoint.pt"),
                metadata_path=Path("checkpoint.json"),
                byte_size=1,
                sha256="d" * 64,
            ),
            payload={"generationData": {"generation": 0}},
            rejected_newer=(),
        )
        driver.run_generation = Mock(side_effect=AssertionError("must not advance"))

        driver._reconcile_reports()
        self.assertEqual(driver.run_to_target(1), [])
        self.assertEqual(len(driver.reports.writes), 1)
        self.assertTrue(driver.budget_guard.calls[0][0])
        self.assertGreater(driver.budget_guard.calls[0][1], 0)
        driver.run_generation.assert_not_called()


class RuntimeBudgetGuardTests(unittest.TestCase):
    class Ledger:
        total_hours = 4.0

        @staticmethod
        def update(_run_id, *, elapsed_seconds, status):
            del elapsed_seconds, status
            return 4.0

    class StageLedger:
        @staticmethod
        def assert_available(_stage, _global_hours):
            return {"remainingHours": 5.0}

    def _guard(self, root: Path, artifact_limit: int) -> _RuntimeBudgetGuard:
        return _RuntimeBudgetGuard(
            control_root=root,
            artifact_limit_bytes=artifact_limit,
            ledger=self.Ledger(),
            stage_ledger=self.StageLedger(),
            ledger_run_id="test",
            started=0.0,
            clock=lambda: 1.0,
        )

    def test_write_reserve_protects_disk_and_artifact_stop_lines(self) -> None:
        reserve = 4096
        artifact_limit = 100_000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = self._guard(root, artifact_limit)
            with (
                patch.object(
                    phase4_training.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(
                        free=DEFAULT_MINIMUM_FREE_DISK_BYTES + reserve - 1
                    ),
                ),
                patch.object(phase4_training, "directory_size", return_value=0),
            ):
                with self.assertRaisesRegex(
                    Phase4ResourceLimitReached, "20 GiB"
                ):
                    guard.check(reserve_bytes=reserve)

            guard = self._guard(root, artifact_limit)
            with (
                patch.object(
                    phase4_training.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(
                        free=DEFAULT_MINIMUM_FREE_DISK_BYTES + reserve
                    ),
                ) as disk_usage,
                patch.object(
                    phase4_training,
                    "directory_size",
                    return_value=artifact_limit - reserve,
                ),
            ):
                result = guard.check(reserve_bytes=reserve)
                self.assertEqual(result["reservedWriteBytes"], reserve)
                guard.check(reserve_bytes=reserve)
                self.assertEqual(disk_usage.call_count, 2)

            guard = self._guard(root, artifact_limit)
            with (
                patch.object(
                    phase4_training.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(
                        free=DEFAULT_MINIMUM_FREE_DISK_BYTES + reserve
                    ),
                ),
                patch.object(
                    phase4_training,
                    "directory_size",
                    return_value=artifact_limit - reserve + 1,
                ),
            ):
                with self.assertRaisesRegex(
                    Phase4ResourceLimitReached, "configured artifact cap"
                ):
                    guard.check(reserve_bytes=reserve)


class Phase4LineageTests(unittest.TestCase):
    def _config(self, run_dir: Path):
        config = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        return replace(config, run_dir=run_dir, device="cpu")

    def test_lineage_requires_exact_config_and_source_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            config = self._config(workspace / "run")
            budget = ResourceBudget(gpu_hours=10)
            created = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=budget,
                resume=False,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )
            self.assertFalse(created.resumed)
            resumed = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=budget,
                resume=True,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.identity, created.identity)

            with self.assertRaisesRegex(Phase4ResumeError, "config changed"):
                prepare_phase4_lineage(
                    workspace_root=workspace,
                    run_dir=config.run_dir,
                    config=replace(config, seed=config.seed + 1),
                    budget=budget,
                    resume=True,
                    source_paths=[source],
                    environment=FIXED_ENVIRONMENT,
                )
            source.write_text("VERSION = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(Phase4ResumeError, "source changed"):
                prepare_phase4_lineage(
                    workspace_root=workspace,
                    run_dir=config.run_dir,
                    config=config,
                    budget=budget,
                    resume=True,
                    source_paths=[source],
                    environment=FIXED_ENVIRONMENT,
                )

    def test_new_lineage_preflights_before_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            config = self._config(workspace / "run")
            with patch.object(
                phase4_training.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES
                ),
            ):
                with self.assertRaisesRegex(
                    Phase4ResourceLimitReached, "20 GiB"
                ):
                    prepare_phase4_lineage(
                        workspace_root=workspace,
                        run_dir=config.run_dir,
                        config=config,
                        budget=ResourceBudget(gpu_hours=10),
                        resume=False,
                        source_paths=[source],
                        environment=FIXED_ENVIRONMENT,
                    )
            self.assertFalse((config.run_dir / "run-manifest.json").exists())

    def test_phase4_config_rejects_legacy_and_short_replay(self) -> None:
        reference = load_experiment_config(ROOT / "training/configs/reference.json")
        self.assertEqual(reference.initialization.mode, "random")
        self.assertIsNone(reference.initialization.expected_base_model_sha256)
        with self.assertRaisesRegex(ValueError, "pipeline=phase4"):
            validate_phase4_config(reference)
        phase4 = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        self.assertEqual(phase4.initialization.mode, "random")
        self.assertEqual(
            len(phase4.initialization.expected_base_model_sha256), 64
        )
        short = replace(
            phase4,
            training=replace(phase4.training, replay_capacity=100),
        )
        with self.assertRaisesRegex(ValueError, "five-generation"):
            validate_phase4_config(short)
        too_few_steps = replace(
            phase4,
            training=replace(
                phase4.training,
                training_steps=phase4_training.REPLAY_INPUT_GATE_BATCHES - 1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "at least eight"):
            validate_phase4_config(too_few_steps)


class InitializationContractTests(unittest.TestCase):
    @staticmethod
    def _warm_config(workspace: Path):
        base = load_experiment_config(
            ROOT / "training/configs/phase4-bootstrap.json"
        )
        network = replace(
            base.network,
            channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=2,
            value_hidden=8,
        )
        provisional = replace(
            base,
            name="phase4-warm-test",
            device="cpu",
            run_dir=workspace / "training/runs/phase4/warm-test",
            network=network,
            self_play=replace(base.self_play, games=1, actors=1),
            training=replace(
                base.training,
                replay_capacity=1280,
                batch_size=2,
                training_steps=phase4_training.REPLAY_INPUT_GATE_BATCHES,
            ),
        )
        seed_everything(provisional.seed)
        cold = PolicyValueNetwork(_network_config(provisional))
        base_sha256 = canonical_model_state_sha256(cold)
        warm = copy.deepcopy(cold)
        with torch.no_grad():
            next(warm.parameters()).add_(0.125)
        warm_sha256 = canonical_model_state_sha256(warm)
        teacher_sha256 = "1" * 64
        report_sha256 = "2" * 64
        artifact_path = workspace / "warm-model.pt"
        torch.save(
            {
                "schemaVersion": 1,
                "format": "quoridor-phase4-model-only",
                "network": _network_config(provisional).to_dict(),
                "baseModelStateSha256": base_sha256,
                "modelStateSha256": warm_sha256,
                "provenance": {
                    "teacherDatasetSha256": teacher_sha256,
                    "pretrainingReportSha256": report_sha256,
                },
                "modelState": warm.state_dict(),
            },
            artifact_path,
        )
        artifact_sha256 = sha256_file(artifact_path)
        config = replace(
            provisional,
            initialization=InitializationSettings(
                mode="model",
                expected_base_model_sha256=base_sha256,
                artifact_path=artifact_path,
                artifact_sha256=artifact_sha256,
                teacher_dataset_sha256=teacher_sha256,
                pretraining_report_sha256=report_sha256,
                reset_optimizer=True,
            ),
        )
        return config, warm_sha256

    def test_bootstrap_baseline_commits_model_only_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            warm_config, warm_sha256 = self._warm_config(workspace)
            baseline_state, _ = load_phase4_warm_artifact(
                workspace_root=workspace,
                config=warm_config,
            )
            base_sha256 = warm_config.initialization.expected_base_model_sha256
            config = replace(
                warm_config,
                name="phase4-bootstrap-baseline-test",
                run_dir=workspace / "training/runs/phase4/bootstrap-test",
                initialization=InitializationSettings(
                    mode="random",
                    expected_base_model_sha256=base_sha256,
                    reset_optimizer=True,
                ),
            )
            baseline_root = workspace / "bootstrap-baseline"
            baseline_root.mkdir()
            lineage = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=ResourceBudget(
                    gpu_hours=10,
                    metadata={
                        "nativeBackendSha256": loaded_native_backend_hash(),
                    },
                ),
                resume=False,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )

            class Guard:
                @staticmethod
                def check(*, force=False, reserve_bytes=0):
                    return {"force": force, "reserveBytes": reserve_bytes}

            verification = {
                "schemaVersion": 1,
                "manifestSha256": "3" * 64,
                "modelSha256": "4" * 64,
                "modelStateSha256": warm_sha256,
            }
            driver = Phase4TrainingDriver(
                workspace_root=workspace,
                control_root=workspace / "training/runs/phase4",
                config=config,
                device=torch.device("cpu"),
                lineage=lineage,
                budget_guard=Guard(),
                bootstrap_baseline_root=baseline_root,
            )
            with (
                patch.object(
                    phase4_training,
                    "_verify_phase4e_bootstrap_baseline",
                    return_value=verification,
                ) as baseline_verifier,
                patch(
                    "quoridor.phase4_baseline.load_bootstrap_baseline_state",
                    return_value=baseline_state,
                ) as baseline_loader,
            ):
                driver.initialize_new_run()

            baseline_verifier.assert_called_once_with(
                workspace.resolve(),
                baseline_root.resolve(),
            )
            baseline_loader.assert_called_once_with(
                workspace_root=workspace.resolve(),
                baseline_root=baseline_root.resolve(),
                verify_lineage=False,
            )
            self.assertIsNotNone(driver.loaded)
            assert driver.loaded is not None
            self.assertEqual(driver.loaded.record.generation, -1)
            self.assertEqual(driver.next_generation, 0)
            self.assertEqual(
                canonical_model_state_sha256(driver.model),
                warm_sha256,
            )
            self.assertFalse(driver.trainer.optimizer.state)
            self.assertEqual(driver.reports.generations(), ())
            self.assertEqual(driver.replay.game_count, 0)
            generation_data = driver.loaded.payload["generationData"]
            self.assertEqual(generation_data["mode"], "model")
            self.assertEqual(
                generation_data["artifact"],
                {
                    "kind": "phase4e-bootstrap-baseline",
                    "schemaVersion": 1,
                    "baselineRoot": "bootstrap-baseline",
                    "manifestSha256": "3" * 64,
                    "modelSha256": "4" * 64,
                    "modelStateSha256": warm_sha256,
                    "formalPromotion": False,
                },
            )

    def test_warm_artifact_is_strict_and_initial_checkpoint_resets_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "driver.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            config, warm_sha256 = self._warm_config(workspace)
            state, provenance = load_phase4_warm_artifact(
                workspace_root=workspace,
                config=config,
            )
            self.assertEqual(canonical_model_state_sha256(state), warm_sha256)
            self.assertEqual(
                provenance["artifactSha256"], config.initialization.artifact_sha256
            )
            unbound_config = replace(config, run_dir=workspace / "unbound-run")
            unbound_lineage = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=unbound_config.run_dir,
                config=unbound_config,
                budget=ResourceBudget(gpu_hours=10),
                resume=False,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )

            class Guard:
                @staticmethod
                def check(*, force=False, reserve_bytes=0):
                    return {"force": force, "reserveBytes": reserve_bytes}

            with self.assertRaisesRegex(
                Phase4TrainingError,
                "native backend hash",
            ):
                Phase4TrainingDriver(
                    workspace_root=workspace,
                    control_root=workspace / "training/runs/phase4",
                    config=unbound_config,
                    device=torch.device("cpu"),
                    lineage=unbound_lineage,
                    budget_guard=Guard(),
                )
            lineage = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=ResourceBudget(
                    gpu_hours=10,
                    metadata={
                        "nativeBackendSha256": loaded_native_backend_hash(),
                    },
                ),
                resume=False,
                source_paths=[source],
                environment=FIXED_ENVIRONMENT,
            )

            driver = Phase4TrainingDriver(
                workspace_root=workspace,
                control_root=workspace / "training/runs/phase4",
                config=config,
                device=torch.device("cpu"),
                lineage=lineage,
                budget_guard=Guard(),
            )
            driver.initialize_new_run()
            self.assertEqual(driver.loaded.record.generation, -1)
            self.assertEqual(driver.next_generation, 0)
            self.assertEqual(canonical_model_state_sha256(driver.model), warm_sha256)
            self.assertFalse(driver.trainer.optimizer.state)
            self.assertEqual(
                driver.current_model_id,
                f"sha256:{driver.loaded.record.sha256}",
            )
            self.assertEqual(driver.reports.generations(), ())
            self.assertEqual(driver.replay.game_count, 0)

            tampered = replace(
                config,
                initialization=replace(
                    config.initialization, artifact_sha256="f" * 64
                ),
            )
            with self.assertRaisesRegex(Phase4ResumeError, "SHA-256"):
                load_phase4_warm_artifact(
                    workspace_root=workspace,
                    config=tampered,
                )

            partial_game = ReplayResumeBindingTests._game(
                "phase4-g00000000-game000000",
                0,
                driver.current_model_id,
            )
            invocation = _selfplay_invocation(
                generation=0,
                model_id=driver.current_model_id,
                scheduler_seed=config.seed,
                scheduled_game_indices=[0],
                configured_actors=config.self_play.actors,
                simulations=config.mcts.simulations,
                temperature_moves=config.self_play.temperature_moves,
            )
            driver.replay.append(
                replace(
                    partial_game,
                    metadata={
                        **partial_game.metadata,
                        "schedulerSeed": config.seed,
                        "gameIndex": 0,
                        "resumedSubset": False,
                        "invocationGameIndex": 0,
                        "invocation": invocation,
                    },
                )
            )
            config.initialization.artifact_path.unlink()
            resumed_driver = Phase4TrainingDriver(
                workspace_root=workspace,
                control_root=workspace / "training/runs/phase4",
                config=config,
                device=torch.device("cpu"),
                lineage=replace(lineage, resumed=True),
                budget_guard=Guard(),
            )
            resumed_driver.restore()
            self.assertEqual(resumed_driver.loaded.record.generation, -1)
            self.assertEqual(resumed_driver.next_generation, 0)
            self.assertEqual(resumed_driver.replay.game_count, 1)
            self.assertEqual(
                canonical_model_state_sha256(resumed_driver.model), warm_sha256
            )


if __name__ == "__main__":
    unittest.main()
