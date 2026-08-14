from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from quoridor.calibration import GPU_HOUR_BUDGET
from quoridor.codec import ACTION_SPACE_SIZE
from quoridor.game import (
    apply_action_index,
    create_initial_state,
    is_terminal,
    legal_action_indices,
)
from quoridor.network import NetworkConfig, PolicyValueNetwork, encode_state
from quoridor.phase4_checkpoint import (
    CheckpointIdentity,
    Phase4CheckpointStore,
    canonical_model_state_sha256,
)
from quoridor.phase4_gates import (
    Phase4GateError,
    find_valid_memory_smoke,
    verify_memory_smoke_evidence,
)
from quoridor.phase4_budget import (
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
)
from quoridor.phase4_training import ReplayBatchEncoder, load_phase4_warm_artifact
from quoridor.replay_store import ReplayGame, ReplayPly, ReplayStore
from quoridor.run_manifest import (
    ResourceBudget,
    build_run_manifest,
    canonical_json_bytes,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
)
from quoridor.teacher_pretraining import (
    CheckpointModels,
    DEFAULT_RUN_ROOT,
    INPUT_PERFORMANCE_GATE_STEPS,
    PREENCODED_CACHE_ALGORITHM_VERSION,
    PREENCODED_CACHE_MAX_BYTES,
    RUN_KIND,
    InputPerformanceGate,
    MetricAccumulator,
    PreencodedReplaySplit,
    TeacherDataset,
    TeacherPreencodedCache,
    TeacherPretrainingContract,
    TeacherPretrainingContractError,
    TeacherPretrainingResourceError,
    _save_checkpoint,
    _summary,
    _checkpoint_generation_data,
    _checkpoint_replay_snapshot,
    _validate_restored,
    _warm_artifact_payload,
    build_teacher_preencoded_cache,
    cached_training_batch,
    evaluate_cached_holdout,
    evaluate_holdout,
    load_trusted_native_encoder,
    load_completed_pretraining_bundle,
    load_warm_artifact,
    restore_pretraining_checkpoint,
    run_teacher_pretraining,
    training_batch,
    main as teacher_pretraining_main,
)
from quoridor.trainer import AlphaZeroTrainer
from quoridor.training_memory_smoke import (
    RUN_KIND as MEMORY_SMOKE_KIND,
    TrainingMemorySmokeContract,
    write_immutable_report,
)


def _smoke_contract() -> TeacherPretrainingContract:
    return TeacherPretrainingContract(
        seed=123,
        channels=8,
        residual_blocks=1,
        se_reduction=2,
        policy_channels=2,
        value_channels=2,
        value_hidden=8,
        batch_size=4,
        optimizer_steps=2,
        evaluation_interval=1,
        mirror_probability=0.5,
        learning_rate=0.001,
        minimum_learning_rate=0.0001,
        use_amp=False,
    )


def _forward_game(game_id: str, *, lateral_opening: bool = False) -> ReplayGame:
    state = create_initial_state()
    if lateral_opening:
        lateral = [
            action
            for action in legal_action_indices(state)
            if action < 81 and action // 9 == 8 and action % 9 != 4
        ]
        state = apply_action_index(state, min(lateral))
    initial = state
    plies: list[ReplayPly] = []
    while not is_terminal(state):
        pawn_actions = [action for action in legal_action_indices(state) if action < 81]
        player = state.current_player
        action = min(
            pawn_actions,
            key=lambda candidate: (
                candidate // 9 if player == 0 else -(candidate // 9),
                candidate,
            ),
        )
        counts = [0] * ACTION_SPACE_SIZE
        counts[action] = 1
        plies.append(
            ReplayPly(
                action_index=action,
                visit_counts=tuple(counts),
                root_value=0.0,
                model_generation=0,
                model_id="pseudo-teacher",
            )
        )
        state = apply_action_index(state, action)
    return ReplayGame(game_id=game_id, initial_state=initial, plies=tuple(plies))


def _store(root: Path, name: str) -> ReplayStore:
    store = ReplayStore(root / name, max_games=4, max_plies=256, max_bytes=8 * 1024**2)
    store.append(_forward_game(f"{name}-game-0"))
    store.append(_forward_game(f"{name}-game-1", lateral_opening=True))
    return store


def _dataset(root: Path) -> TeacherDataset:
    train = _store(root, "train")
    holdout = _store(root, "holdout")
    teacher_manifest_path = root / "teacher-manifest.json"
    run_manifest_path = root / "run-manifest.json"
    source_path = root / "fixture-source.txt"
    teacher_manifest_path.write_bytes(b"teacher-manifest-fixture\n")
    source_path.write_bytes(b"teacher-pretraining-fixture\n")
    run_manifest = build_run_manifest(
        workspace_root=root,
        config={"kind": "teacher-data-fixture"},
        budget=ResourceBudget(),
        run_id="teacher-data-fixture",
        source_paths=(source_path,),
        environment={},
        created_at="2026-07-25T00:00:00Z",
    )
    save_run_manifest(run_manifest, run_manifest_path)
    return TeacherDataset(
        root=root,
        train=train,
        holdout=holdout,
        dataset_fingerprint="a" * 64,
        teacher_manifest_sha256=sha256_file(teacher_manifest_path),
        run_manifest_sha256=run_manifest.sha256,
        config_sha256="d" * 64,
        train_manifest_sha256=sha256_file(train.manifest_path),
        holdout_manifest_sha256=sha256_file(holdout.manifest_path),
        train_positions=train.ply_count,
        holdout_positions=holdout.ply_count,
    )


def _native_python_encoder(states: object, *, input_channels: int) -> object:
    return torch.stack(
        [encode_state(state, input_channels=input_channels) for state in states]
    ).numpy()


class PretrainingCoreTests(unittest.TestCase):
    def test_noncanonical_teacher_root_is_rejected_before_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            control = workspace / "training/runs/phase4"
            with self.assertRaisesRegex(
                TeacherPretrainingContractError, "canonical.*teacher-data"
            ):
                run_teacher_pretraining(
                    workspace_root=workspace,
                    teacher_root=control / "alternate-teacher-data",
                    run_root=control / "teacher-pretraining",
                    resume=False,
                    device_index=0,
                    external_gpu_hours=0.0,
                )

    def test_production_contract_is_frozen(self) -> None:
        contract = TeacherPretrainingContract.production()
        contract.assert_production()
        self.assertEqual(contract.network_config().channels, 192)
        self.assertEqual(contract.network_config().blocks, 16)
        self.assertEqual(contract.batch_size, 512)
        self.assertEqual(contract.optimizer_steps, 512)
        self.assertEqual(contract.evaluation_interval, 64)
        with self.assertRaisesRegex(Exception, "frozen"):
            _smoke_contract().assert_production()

    def test_small_network_trains_on_train_and_evaluates_fixed_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            contract = _smoke_contract()
            torch.manual_seed(contract.seed)
            model = PolicyValueNetwork(contract.network_config())
            trainer = AlphaZeroTrainer(
                model,
                learning_rate=contract.learning_rate,
                weight_decay=contract.weight_decay,
                max_grad_norm=contract.max_grad_norm,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            encoder = ReplayBatchEncoder(input_channels=16, device="cpu")

            first = training_batch(dataset.train, encoder, contract, step=1)
            repeated = training_batch(dataset.train, encoder, contract, step=1)
            for key in first:
                self.assertTrue(torch.equal(first[key], repeated[key]))
            metric = trainer.train_step(first)
            self.assertTrue(torch.isfinite(torch.tensor(metric.loss)).item())

            before = evaluate_holdout(model, dataset.holdout, encoder, contract)
            after = evaluate_holdout(model, dataset.holdout, encoder, contract)
            self.assertEqual(before, after)
            self.assertEqual(before["positions"], dataset.holdout.ply_count)
            self.assertGreater(before["totalLoss"], 0.0)

    def test_checkpoint_restores_live_and_best_models_with_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root / "data")
            contract = _smoke_contract()
            torch.manual_seed(contract.seed)
            current = PolicyValueNetwork(contract.network_config())
            bundle = CheckpointModels(current)
            trainer = AlphaZeroTrainer(
                current,
                learning_rate=contract.learning_rate,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            encoder = ReplayBatchEncoder(input_channels=16, device="cpu")
            metric = trainer.train_step(
                training_batch(dataset.train, encoder, contract, step=1)
            )
            metrics = MetricAccumulator()
            metrics.add(metric)
            bundle.update_best()
            identity = CheckpointIdentity("1" * 64, "2" * 64, "3" * 64)
            store = Phase4CheckpointStore(
                root / "checkpoints",
                identity=identity,
                artifact_root=root,
                artifact_limit_bytes=64 * 1024**2,
                max_to_keep=2,
            )
            base_hash = canonical_model_state_sha256(
                PolicyValueNetwork(contract.network_config())
            )
            contract_hash = sha256_bytes(canonical_json_bytes(contract.to_payload()))
            evaluation = {
                "step": 1,
                "positions": dataset.holdout.ply_count,
                "policyLoss": 1.0,
                "valueLoss": 0.5,
                "totalLoss": 1.5,
            }
            _save_checkpoint(
                store,
                next_generation=0,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=bundle,
                trainer=trainer,
                base_model_sha256=base_hash,
                last_evaluation_step=1,
                best_step=1,
                best_metrics=evaluation,
                evaluations=[evaluation],
                metrics=metrics,
                reason="evaluation",
            )

            torch.manual_seed(contract.seed)
            restored_current = PolicyValueNetwork(contract.network_config())
            restored_bundle = CheckpointModels(restored_current)
            restored_trainer = AlphaZeroTrainer(
                restored_current,
                learning_rate=contract.learning_rate,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            loaded = store.restore(
                restored_bundle, restored_trainer, map_location="cpu"
            )
            self.assertIsNotNone(loaded)
            cursor = _validate_restored(
                loaded,
                contract=contract,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=restored_bundle,
                trainer=restored_trainer,
                base_model_sha256=base_hash,
            )
            self.assertEqual(cursor[0], 1)
            self.assertEqual(cursor[1], 1)
            self.assertEqual(cursor[4].count, 1)
            self.assertEqual(
                canonical_model_state_sha256(restored_bundle.best),
                canonical_model_state_sha256(bundle.best),
            )

    def test_semantically_invalid_or_bad_rng_newest_checkpoint_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root / "data")
            contract = _smoke_contract()
            torch.manual_seed(contract.seed)
            current = PolicyValueNetwork(contract.network_config())
            bundle = CheckpointModels(current)
            trainer = AlphaZeroTrainer(
                current,
                learning_rate=contract.learning_rate,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            encoder = ReplayBatchEncoder(input_channels=16, device="cpu")
            identity = CheckpointIdentity("4" * 64, "5" * 64, "6" * 64)
            store = Phase4CheckpointStore(
                root / "checkpoints",
                identity=identity,
                artifact_root=root,
                artifact_limit_bytes=64 * 1024**2,
                max_to_keep=2,
            )
            torch.manual_seed(contract.seed)
            base_hash = canonical_model_state_sha256(
                PolicyValueNetwork(contract.network_config())
            )
            contract_hash = sha256_bytes(canonical_json_bytes(contract.to_payload()))
            metrics = MetricAccumulator()
            first_metric = trainer.train_step(
                training_batch(dataset.train, encoder, contract, step=1)
            )
            metrics.add(first_metric)
            bundle.update_best()
            first_eval = {
                "step": 1,
                "positions": dataset.holdout.ply_count,
                "policyLoss": 1.0,
                "valueLoss": 0.5,
                "totalLoss": 1.5,
            }
            _save_checkpoint(
                store,
                next_generation=0,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=bundle,
                trainer=trainer,
                base_model_sha256=base_hash,
                last_evaluation_step=1,
                best_step=1,
                best_metrics=first_eval,
                evaluations=[first_eval],
                metrics=metrics,
                reason="evaluation",
            )

            second_metric = trainer.train_step(
                training_batch(dataset.train, encoder, contract, step=2)
            )
            metrics.add(second_metric)
            second_eval = {
                "step": 2,
                "positions": dataset.holdout.ply_count,
                "policyLoss": 2.0,
                "valueLoss": 0.5,
                "totalLoss": 2.5,
            }
            invalid = _checkpoint_generation_data(
                generation=1,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=bundle,
                trainer=trainer,
                base_model_sha256=base_hash,
                last_evaluation_step=2,
                best_step=1,
                best_metrics=first_eval,
                evaluations=[first_eval, second_eval],
                metrics=metrics,
                reason="evaluation",
            )
            invalid["teacherDatasetSha256"] = "f" * 64
            store.save(
                generation=1,
                model=bundle,
                trainer=trainer,
                generation_data=invalid,
                replay_snapshot=_checkpoint_replay_snapshot(dataset),
            )

            restored_current = PolicyValueNetwork(contract.network_config())
            restored_bundle = CheckpointModels(restored_current)
            restored_trainer = AlphaZeroTrainer(
                restored_current,
                learning_rate=contract.learning_rate,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            restored = restore_pretraining_checkpoint(
                store,
                contract=contract,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=restored_bundle,
                trainer=restored_trainer,
                base_model_sha256=base_hash,
                map_location="cpu",
            )
            self.assertIsNotNone(restored)
            loaded, cursor = restored
            self.assertEqual(loaded.record.generation, 0)
            self.assertEqual(restored_trainer.step, 1)
            self.assertEqual(cursor[0], 1)
            self.assertTrue((root / "checkpoints/rejected").is_dir())

            valid_with_bad_rng = _checkpoint_generation_data(
                generation=2,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=bundle,
                trainer=trainer,
                base_model_sha256=base_hash,
                last_evaluation_step=2,
                best_step=1,
                best_metrics=first_eval,
                evaluations=[first_eval, second_eval],
                metrics=metrics,
                reason="evaluation",
            )
            bad_rng_record = store.save(
                generation=2,
                model=bundle,
                trainer=trainer,
                generation_data=valid_with_bad_rng,
                replay_snapshot=_checkpoint_replay_snapshot(dataset),
            )
            bad_payload = torch.load(
                bad_rng_record.data_path, map_location="cpu", weights_only=False
            )
            bad_payload["rng"] = {"python": "invalid"}
            torch.save(bad_payload, bad_rng_record.data_path)
            metadata = json.loads(
                bad_rng_record.metadata_path.read_text(encoding="utf-8")
            )
            metadata["bytes"] = bad_rng_record.data_path.stat().st_size
            metadata["sha256"] = sha256_file(bad_rng_record.data_path)
            bad_rng_record.metadata_path.write_bytes(canonical_json_bytes(metadata))

            retry_current = PolicyValueNetwork(contract.network_config())
            retry_bundle = CheckpointModels(retry_current)
            retry_trainer = AlphaZeroTrainer(
                retry_current,
                learning_rate=contract.learning_rate,
                use_amp=False,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            retried = restore_pretraining_checkpoint(
                store,
                contract=contract,
                contract_sha256=contract_hash,
                dataset=dataset,
                bundle=retry_bundle,
                trainer=retry_trainer,
                base_model_sha256=base_hash,
                map_location="cpu",
            )
            self.assertIsNotNone(retried)
            self.assertEqual(retried[0].record.generation, 0)
            self.assertEqual(retry_trainer.step, 1)

    def test_native_encoder_binding_rechecks_binary_identity(self) -> None:
        module = SimpleNamespace(encode_states=lambda *args, **kwargs: None)
        with patch(
            "quoridor.teacher_pretraining.loaded_native_backend_hash",
            side_effect=["a" * 64, "a" * 64],
        ), patch(
            "quoridor.teacher_pretraining.load_native_module", return_value=module
        ):
            loaded, digest = load_trusted_native_encoder()
        self.assertIs(loaded, module)
        self.assertEqual(digest, "a" * 64)

        with patch(
            "quoridor.teacher_pretraining.loaded_native_backend_hash",
            side_effect=["a" * 64, "b" * 64],
        ), patch(
            "quoridor.teacher_pretraining.load_native_module", return_value=module
        ):
            with self.assertRaisesRegex(Exception, "identity changed"):
                load_trusted_native_encoder()


class PreencodedCacheTests(unittest.TestCase):
    def _cache(
        self,
        dataset: TeacherDataset,
        contract: TeacherPretrainingContract,
        *,
        progress: list[dict[str, object]] | None = None,
    ) -> TeacherPreencodedCache:
        return build_teacher_preencoded_cache(
            dataset,
            native_encoder=_native_python_encoder,
            contract=contract,
            progress_callback=(
                None if progress is None else lambda payload: progress.append(dict(payload))
            ),
        )

    def test_cached_batches_are_bit_exact_and_cannot_mutate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            contract = _smoke_contract()
            progress: list[dict[str, object]] = []
            cache = self._cache(dataset, contract, progress=progress)
            legacy_encoder = ReplayBatchEncoder(input_channels=16, device="cpu")

            for step in range(1, 9):
                legacy = training_batch(
                    dataset.train,
                    legacy_encoder,
                    contract,
                    step=step,
                )
                cached = cached_training_batch(
                    cache.train,
                    contract,
                    step=step,
                    device="cpu",
                )
                for key in legacy:
                    self.assertTrue(
                        torch.equal(legacy[key], cached[key]),
                        f"{key} differs at step {step}",
                    )

            expected_bytes = (
                dataset.train_positions + dataset.holdout_positions
            ) * (2 * 16 * 9 * 9 * 4 + 2 * ACTION_SPACE_SIZE * 4 + 4)
            self.assertEqual(cache.byte_size, expected_bytes)
            self.assertLessEqual(cache.byte_size, PREENCODED_CACHE_MAX_BYTES)
            self.assertEqual(
                cache.algorithm_version,
                PREENCODED_CACHE_ALGORITHM_VERSION,
            )
            self.assertTrue(cache.train.states.is_contiguous())
            self.assertEqual(cache.train.states.device.type, "cpu")

            before = cache.train.states.clone()
            batch = cached_training_batch(
                cache.train,
                contract,
                step=1,
                device="cpu",
            )
            batch["states"].zero_()
            self.assertTrue(torch.equal(cache.train.states, before))
            completed = [
                item
                for item in progress
                if item["stage"] == "cache-build"
                and item["status"] == "completed"
            ]
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["bytes"], expected_bytes)
            self.assertGreaterEqual(completed[0]["buildSeconds"], 0.0)

    def test_cache_accepts_run_manifest_self_hash_distinct_from_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            run_manifest_path = dataset.root / "run-manifest.json"
            self.assertNotEqual(
                dataset.run_manifest_sha256,
                sha256_file(run_manifest_path),
            )

            cache = self._cache(dataset, _smoke_contract())

            self.assertEqual(cache.dataset_fingerprint, dataset.dataset_fingerprint)

    def test_cache_rejects_corrupt_run_manifest_before_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            run_manifest_path = dataset.root / "run-manifest.json"
            payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            payload["metadata"]["tampered"] = True
            run_manifest_path.write_bytes(canonical_json_bytes(payload))

            with self.assertRaisesRegex(
                TeacherPretrainingContractError,
                "run-manifest is invalid before cache construction",
            ):
                self._cache(dataset, _smoke_contract())

    def test_cache_rejects_valid_run_manifest_replacement_during_construction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            replacement_path = dataset.root / "replacement-run-manifest.json"
            replacement = build_run_manifest(
                workspace_root=dataset.root,
                config={"kind": "teacher-data-fixture"},
                budget=ResourceBudget(),
                run_id="replacement-teacher-data-fixture",
                source_paths=(dataset.root / "fixture-source.txt",),
                environment={},
                created_at="2026-07-25T00:00:00Z",
            )
            save_run_manifest(replacement, replacement_path)
            replaced = False

            def replacing_encoder(states: object, *, input_channels: int) -> object:
                nonlocal replaced
                if not replaced:
                    replacement_path.replace(dataset.root / "run-manifest.json")
                    replaced = True
                return _native_python_encoder(states, input_channels=input_channels)

            with self.assertRaisesRegex(
                TeacherPretrainingContractError,
                "dataset manifests changed during cache construction",
            ):
                build_teacher_preencoded_cache(
                    dataset,
                    native_encoder=replacing_encoder,
                    contract=_smoke_contract(),
                )

    def test_cached_holdout_is_bit_exact_to_legacy_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            contract = _smoke_contract()
            cache = self._cache(dataset, contract)
            torch.manual_seed(contract.seed)
            model = PolicyValueNetwork(contract.network_config())
            legacy = evaluate_holdout(
                model,
                dataset.holdout,
                ReplayBatchEncoder(input_channels=16, device="cpu"),
                contract,
            )
            cached = evaluate_cached_holdout(model, cache.holdout, contract)
            self.assertEqual(cached, legacy)

    def test_cached_training_and_holdout_perform_no_replay_io_or_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            contract = _smoke_contract()
            encoder_calls = 0

            def counting_encoder(states: object, *, input_channels: int) -> object:
                nonlocal encoder_calls
                encoder_calls += 1
                return _native_python_encoder(states, input_channels=input_channels)

            cache = build_teacher_preencoded_cache(
                dataset,
                native_encoder=counting_encoder,
                contract=contract,
            )
            calls_after_build = encoder_calls
            torch.manual_seed(contract.seed)
            model = PolicyValueNetwork(contract.network_config())
            with patch.object(
                dataset.train,
                "sample_plies",
                side_effect=AssertionError("cached training touched ReplayStore"),
            ), patch.object(
                dataset.holdout,
                "sample_plies",
                side_effect=AssertionError("cached holdout touched ReplayStore"),
            ):
                for step in range(1, 65):
                    cached_training_batch(
                        cache.train,
                        contract,
                        step=step,
                        device="cpu",
                    )
                evaluate_cached_holdout(model, cache.holdout, contract)
            self.assertEqual(encoder_calls, calls_after_build)

    def test_cache_rechecks_native_identity_after_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            with patch(
                "quoridor.teacher_pretraining.loaded_native_backend_hash",
                side_effect=["a" * 64, "b" * 64],
            ):
                with self.assertRaisesRegex(
                    TeacherPretrainingContractError,
                    "during cache construction",
                ):
                    build_teacher_preencoded_cache(
                        dataset,
                        native_encoder=_native_python_encoder,
                        contract=_smoke_contract(),
                        native_backend_sha256="a" * 64,
                    )

    def test_cache_rechecks_top_level_dataset_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            changed = False

            def mutating_encoder(states: object, *, input_channels: int) -> object:
                nonlocal changed
                if not changed:
                    (dataset.root / "teacher-manifest.json").write_bytes(
                        b"changed-during-cache\n"
                    )
                    changed = True
                return _native_python_encoder(states, input_channels=input_channels)

            with self.assertRaisesRegex(
                TeacherPretrainingContractError,
                "dataset manifests changed during cache construction",
            ):
                build_teacher_preencoded_cache(
                    dataset,
                    native_encoder=mutating_encoder,
                    contract=_smoke_contract(),
                )

    def test_slow_input_gate_stops_before_eighth_optimizer_update(self) -> None:
        gate = InputPerformanceGate()
        optimizer_updates = 0
        with self.assertRaisesRegex(
            TeacherPretrainingResourceError,
            "first 8 new optimizer steps",
        ):
            for _ in range(INPUT_PERFORMANCE_GATE_STEPS):
                gate.observe(1.01)
                optimizer_updates += 1
        self.assertEqual(gate.count, INPUT_PERFORMANCE_GATE_STEPS)
        self.assertEqual(optimizer_updates, INPUT_PERFORMANCE_GATE_STEPS - 1)
        fast_gate = InputPerformanceGate()
        completed_events = sum(
            fast_gate.observe(0.01)
            for _ in range(INPUT_PERFORMANCE_GATE_STEPS + 20)
        )
        self.assertEqual(completed_events, 1)

    def test_cache_precedes_cuda_queries_lineage_and_ledger_begin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            control = workspace / "training/runs/phase4"
            teacher_root = control / "teacher-data"
            dataset = _dataset(teacher_root)
            events: list[str] = []
            fake_cache = SimpleNamespace(byte_size=1)
            fake_ledger = SimpleNamespace()
            fake_stage_ledger = SimpleNamespace()

            def cache_first(*args: object, **kwargs: object) -> object:
                events.append("cache")
                return fake_cache

            def cuda_after_cache() -> bool:
                events.append("cuda")
                return False

            with patch(
                "quoridor.teacher_pretraining.RunLock",
                return_value=nullcontext(),
            ), patch(
                "quoridor.teacher_pretraining.verify_phase4d_baseline_report",
                return_value={},
            ), patch(
                "quoridor.teacher_pretraining.load_teacher_dataset",
                return_value=dataset,
            ), patch(
                "quoridor.teacher_pretraining.find_valid_memory_smoke",
                return_value={"fileSha256": "1" * 64},
            ), patch(
                "quoridor.teacher_pretraining.load_trusted_native_encoder",
                return_value=(
                    SimpleNamespace(encode_states=_native_python_encoder),
                    "2" * 64,
                ),
            ), patch(
                "quoridor.teacher_pretraining.GpuBudgetLedger",
                return_value=fake_ledger,
            ), patch(
                "quoridor.teacher_pretraining.Phase4StageBudgetLedger",
                return_value=fake_stage_ledger,
            ), patch(
                "quoridor.teacher_pretraining._assert_finalization_resources",
                side_effect=lambda *args, **kwargs: events.append("disk"),
            ), patch(
                "quoridor.teacher_pretraining._assert_gpu_budget_preflight",
                side_effect=lambda *args, **kwargs: events.append("budget"),
            ), patch(
                "quoridor.teacher_pretraining.build_teacher_preencoded_cache",
                side_effect=cache_first,
            ), patch(
                "quoridor.teacher_pretraining.torch.cuda.is_available",
                side_effect=cuda_after_cache,
            ), patch(
                "quoridor.teacher_pretraining.torch.cuda.device_count",
                side_effect=AssertionError("device_count ran before availability gate"),
            ), patch(
                "quoridor.teacher_pretraining._lineage",
                side_effect=AssertionError("lineage ran before cache/CUDA preflight"),
            ):
                with self.assertRaisesRegex(
                    TeacherPretrainingResourceError,
                    "CUDA is not available",
                ):
                    run_teacher_pretraining(
                        workspace_root=workspace,
                        teacher_root=teacher_root,
                        run_root=control / "teacher-pretraining-v2",
                        resume=False,
                        device_index=0,
                        external_gpu_hours=0.0,
                    )
            self.assertEqual(events, ["disk", "budget", "cache", "cuda"])

    def test_existing_lineage_is_strictly_checked_before_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            control = workspace / "training/runs/phase4"
            teacher_root = control / "teacher-data"
            dataset = _dataset(teacher_root)
            run_root = control / "teacher-pretraining-v2"
            run_root.mkdir(parents=True)
            (run_root / "run-manifest.json").write_text("existing", encoding="ascii")
            events: list[str] = []

            def existing_lineage(*args: object, **kwargs: object) -> object:
                events.append("lineage")
                return (
                    SimpleNamespace(sha256="3" * 64),
                    SimpleNamespace(),
                    True,
                )

            with patch(
                "quoridor.teacher_pretraining.RunLock",
                return_value=nullcontext(),
            ), patch(
                "quoridor.teacher_pretraining.verify_phase4d_baseline_report",
                return_value={},
            ), patch(
                "quoridor.teacher_pretraining.load_teacher_dataset",
                return_value=dataset,
            ), patch(
                "quoridor.teacher_pretraining.find_valid_memory_smoke",
                return_value={"fileSha256": "1" * 64},
            ), patch(
                "quoridor.teacher_pretraining.load_trusted_native_encoder",
                return_value=(
                    SimpleNamespace(encode_states=_native_python_encoder),
                    "2" * 64,
                ),
            ), patch(
                "quoridor.teacher_pretraining.GpuBudgetLedger",
                return_value=SimpleNamespace(),
            ), patch(
                "quoridor.teacher_pretraining.Phase4StageBudgetLedger",
                return_value=SimpleNamespace(),
            ), patch(
                "quoridor.teacher_pretraining._lineage",
                side_effect=existing_lineage,
            ), patch(
                "quoridor.teacher_pretraining._assert_finalization_resources",
                side_effect=lambda *args, **kwargs: events.append("disk"),
            ), patch(
                "quoridor.teacher_pretraining._assert_gpu_budget_preflight",
                side_effect=lambda *args, **kwargs: events.append("budget"),
            ), patch(
                "quoridor.teacher_pretraining.build_teacher_preencoded_cache",
                side_effect=lambda *args, **kwargs: (
                    events.append("cache") or SimpleNamespace(byte_size=1)
                ),
            ), patch(
                "quoridor.teacher_pretraining.torch.cuda.is_available",
                side_effect=lambda: events.append("cuda") or False,
            ):
                with self.assertRaisesRegex(
                    TeacherPretrainingResourceError,
                    "CUDA is not available",
                ):
                    run_teacher_pretraining(
                        workspace_root=workspace,
                        teacher_root=teacher_root,
                        run_root=run_root,
                        resume=True,
                        device_index=0,
                        external_gpu_hours=0.0,
                    )
            self.assertEqual(
                events,
                ["lineage", "disk", "budget", "cache", "cuda"],
            )

    def test_interrupt_message_does_not_claim_a_cache_checkpoint(self) -> None:
        printed: list[tuple[object, ...]] = []
        with patch(
            "quoridor.teacher_pretraining.run_teacher_pretraining",
            side_effect=KeyboardInterrupt,
        ), patch(
            "builtins.print",
            side_effect=lambda *args, **kwargs: printed.append(args),
        ):
            status = teacher_pretraining_main([])
        self.assertEqual(status, 130)
        payload = json.loads(str(printed[-1][0]))
        self.assertEqual(payload["status"], "interrupted")
        self.assertNotIn("checkpoint committed", payload["message"])

    def test_launcher_uses_v2_unbuffered_python_and_interrupt_safe_tee(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        launcher = (
            workspace / "scripts/pretrain-phase4d-teacher.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('RUN_ROOT="${CONTROL_ROOT}/teacher-pretraining-v2"', launcher)
        self.assertIn("python3 -u -m quoridor.teacher_pretraining", launcher)
        self.assertIn('tee -i -a "${LOG_PATH}"', launcher)
        self.assertEqual(
            DEFAULT_RUN_ROOT,
            Path("training/runs/phase4/teacher-pretraining-v2"),
        )


class WarmArtifactTests(unittest.TestCase):
    def test_exact_artifact_is_accepted_by_phase4_initialization_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "training/runs/phase4/teacher-pretraining"
            run_root.mkdir(parents=True)
            contract = _smoke_contract()
            torch.manual_seed(contract.seed)
            bundle = CheckpointModels(PolicyValueNetwork(contract.network_config()))
            base_hash = canonical_model_state_sha256(bundle.current)
            model_hash = canonical_model_state_sha256(bundle.best)
            report_core = {
                "schemaVersion": 1,
                "kind": "phase4d-teacher-pretraining",
                "status": "completed",
                "baseModelStateSha256": base_hash,
                "modelStateSha256": model_hash,
                "teacherDatasetSha256": "a" * 64,
            }
            report = {
                **report_core,
                "reportCoreSha256": sha256_bytes(canonical_json_bytes(report_core)),
            }
            report_path = run_root / "report.json"
            report_path.write_bytes(canonical_json_bytes(report))
            report_sha = sha256_file(report_path)
            artifact_path = run_root / "warm-model.pt"
            torch.save(
                _warm_artifact_payload(
                    contract=contract,
                    bundle=bundle,
                    base_model_sha256=base_hash,
                    dataset_sha256="a" * 64,
                    report_file_sha256=report_sha,
                ),
                artifact_path,
            )
            loaded = load_warm_artifact(artifact_path)
            self.assertEqual(
                set(loaded),
                {
                    "schemaVersion",
                    "format",
                    "network",
                    "baseModelStateSha256",
                    "modelStateSha256",
                    "provenance",
                    "modelState",
                },
            )
            initialization = SimpleNamespace(
                mode="model",
                artifact_path=artifact_path.relative_to(workspace),
                artifact_sha256=sha256_file(artifact_path),
                teacher_dataset_sha256="a" * 64,
                pretraining_report_sha256=report_sha,
                expected_base_model_sha256=base_hash,
            )
            network = contract.network_config()
            config = SimpleNamespace(
                initialization=initialization,
                network=SimpleNamespace(
                    input_channels=network.input_channels,
                    channels=network.channels,
                    residual_blocks=network.blocks,
                    se_reduction=network.se_reduction,
                    policy_channels=network.policy_channels,
                    value_channels=network.value_channels,
                    value_hidden=network.value_hidden,
                ),
            )
            state, provenance = load_phase4_warm_artifact(
                workspace_root=workspace, config=config
            )
            self.assertEqual(canonical_model_state_sha256(state), model_hash)
            self.assertEqual(provenance["pretrainingReportSha256"], report_sha)

            summary = _summary(
                workspace=workspace,
                run_root=run_root,
                dataset_sha256="a" * 64,
            )
            self.assertEqual(
                set(summary),
                {
                    "schemaVersion",
                    "status",
                    "artifactPath",
                    "artifactSha256",
                    "modelStateSha256",
                    "baseModelStateSha256",
                    "teacherDatasetSha256",
                    "reportPath",
                    "reportSha256",
                },
            )
            self.assertEqual(summary["reportSha256"], report_sha)

    def test_strict_completion_loader_rejects_rehashed_fake_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            control = workspace / "training/runs/phase4"
            run_root = control / "teacher-pretraining"
            run_root.mkdir(parents=True)
            (workspace / "source.txt").write_text("source", encoding="ascii")
            dataset = _dataset(control / "teacher-data")
            contract = TeacherPretrainingContract.production()
            memory = {"path": "memory.json", "fileSha256": "7" * 64}
            native_hash = "8" * 64
            baseline = {"status": "verified", "reportSha256": "9" * 64}
            config = {
                "kind": RUN_KIND,
                "contract": contract.to_payload(),
                "teacherDataset": dataset.snapshot(),
                "memorySmoke": memory,
                "nativeEncoderBackendSha256": native_hash,
            }
            budget = ResourceBudget(
                gpu_hours=PHASE4D_GPU_HOUR_LIMIT,
                metadata={
                    "globalGpuHourLimit": GPU_HOUR_BUDGET,
                    "stage": PHASE4D_STAGE,
                    "stageBaselineGlobalGpuHours": PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
                    "baselineCalibration": baseline,
                },
            )
            manifest = build_run_manifest(
                workspace_root=workspace,
                config=config,
                budget=budget,
                run_id="fake-pretraining",
                source_paths=("source.txt",),
                environment={},
                metadata={
                    "phase": "4D",
                    "kind": RUN_KIND,
                    "purpose": "hard-teacher-warm-model",
                    "teacherReplayUsedAfterPretraining": False,
                },
            )
            save_run_manifest(manifest, run_root / "run-manifest.json")

            smoke_contract = _smoke_contract()
            torch.manual_seed(smoke_contract.seed)
            bundle = CheckpointModels(
                PolicyValueNetwork(smoke_contract.network_config())
            )
            fake_base = "f" * 64
            model_hash = canonical_model_state_sha256(bundle.best)
            evaluations = [
                {
                    "step": step,
                    "positions": dataset.holdout_positions,
                    "policyLoss": 2.0 - step / 1024.0,
                    "valueLoss": 0.5,
                    "totalLoss": 2.5 - step / 1024.0,
                }
                for step in range(64, 513, 64)
            ]
            report_core = {
                "schemaVersion": 1,
                "kind": RUN_KIND,
                "status": "completed",
                "createdAt": manifest.payload["createdAt"],
                "completedAt": "2026-07-22T00:00:00Z",
                "runManifestSha256": manifest.sha256,
                "teacherDatasetSha256": dataset.dataset_fingerprint,
                "baseModelStateSha256": fake_base,
                "modelStateSha256": model_hash,
                "warmArtifactPath": "training/runs/phase4/teacher-pretraining/warm-model.pt",
                "contract": contract.to_payload(),
                "dataset": dataset.snapshot(),
                "memorySmoke": memory,
                "nativeEncoderBackendSha256": native_hash,
                "baselineCalibration": baseline,
                "optimizerResetRequiredForSelfPlay": True,
                "teacherReplayAllowedInSelfPlay": False,
                "best": evaluations[-1],
                "evaluations": evaluations,
                "training": {"steps": 512},
                "resourcesAtReport": {},
                "finalCheckpoint": {
                    "generation": 7,
                    "step": 512,
                    "path": "training/runs/phase4/teacher-pretraining/checkpoints/fake.pt",
                    "sha256": "1" * 64,
                },
            }
            report = {
                **report_core,
                "reportCoreSha256": sha256_bytes(canonical_json_bytes(report_core)),
            }
            report_path = run_root / "report.json"
            report_path.write_bytes(canonical_json_bytes(report))
            report_sha = sha256_file(report_path)
            artifact_path = run_root / "warm-model.pt"
            torch.save(
                _warm_artifact_payload(
                    contract=smoke_contract,
                    bundle=bundle,
                    base_model_sha256=fake_base,
                    dataset_sha256=dataset.dataset_fingerprint,
                    report_file_sha256=report_sha,
                ),
                artifact_path,
            )
            summary = {
                "schemaVersion": 1,
                "status": "completed",
                "artifactPath": artifact_path.relative_to(workspace).as_posix(),
                "artifactSha256": sha256_file(artifact_path),
                "modelStateSha256": model_hash,
                "baseModelStateSha256": fake_base,
                "teacherDatasetSha256": dataset.dataset_fingerprint,
                "reportPath": report_path.relative_to(workspace).as_posix(),
                "reportSha256": report_sha,
            }
            summary_path = run_root / "summary.json"
            summary_path.write_bytes(canonical_json_bytes(summary))
            source_sha = manifest.payload["provenance"]["source"]["sha256"]
            with patch(
                "quoridor.teacher_pretraining.load_teacher_dataset",
                return_value=dataset,
            ), patch(
                "quoridor.teacher_pretraining.verify_memory_smoke_evidence",
                return_value=memory,
            ), patch(
                "quoridor.teacher_pretraining.load_trusted_native_encoder",
                return_value=(SimpleNamespace(), native_hash),
            ), patch(
                "quoridor.teacher_pretraining.verify_phase4d_baseline_report",
                return_value=baseline,
            ), patch(
                "quoridor.teacher_pretraining.fingerprint_sources",
                return_value=SimpleNamespace(sha256=source_sha),
            ):
                with self.assertRaisesRegex(Exception, "report lineage"):
                    load_completed_pretraining_bundle(
                        summary_path,
                        workspace_root=workspace,
                        teacher_root=control / "teacher-data",
                    )


class MemorySmokeGateTests(unittest.TestCase):
    def _write_candidate(self, workspace: Path, *, passing: bool) -> tuple[Path, str]:
        run_dir = workspace / "training/runs/phase4/training-memory-smoke/smoke-1"
        run_dir.mkdir(parents=True)
        manifest = build_run_manifest(
            workspace_root=workspace,
            config={"kind": MEMORY_SMOKE_KIND},
            budget=ResourceBudget(),
            run_id="smoke-1",
            source_paths=("source.txt",),
            environment={},
        )
        save_run_manifest(manifest, run_dir / "run-manifest.json")
        contract = TrainingMemorySmokeContract().to_payload()
        report = {
            "schemaVersion": 1,
            "kind": MEMORY_SMOKE_KIND,
            "runId": "smoke-1",
            "completedAt": "2026-07-22T00:00:00Z",
            "status": "completed",
            "passed": passing,
            "runManifestSha256": manifest.sha256,
            "contract": contract,
            "gates": {"one": True, "two": passing},
        }
        report_path = run_dir / "report.json"
        write_immutable_report(report_path, report)
        source_sha = manifest.payload["provenance"]["source"]["sha256"]
        return report_path, source_sha

    def test_gate_binds_actual_file_hash_and_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "source.txt").write_text("source", encoding="ascii")
            report_path, source_sha = self._write_candidate(workspace, passing=True)
            with patch(
                "quoridor.phase4_gates.fingerprint_sources",
                return_value=SimpleNamespace(sha256=source_sha),
            ):
                evidence = find_valid_memory_smoke(
                    workspace / "training/runs/phase4", workspace
                )
            self.assertEqual(evidence["fileSha256"], sha256_file(report_path))
            self.assertTrue(all(evidence["gates"].values()))
            with patch(
                "quoridor.phase4_gates.fingerprint_sources",
                return_value=SimpleNamespace(sha256=source_sha),
            ):
                self.assertEqual(
                    verify_memory_smoke_evidence(
                        evidence,
                        workspace / "training/runs/phase4",
                        workspace,
                    ),
                    evidence,
                )
                tampered = {**evidence, "fileSha256": "0" * 64}
                with self.assertRaises(Phase4GateError):
                    verify_memory_smoke_evidence(
                        tampered,
                        workspace / "training/runs/phase4",
                        workspace,
                    )

    def test_gate_rejects_a_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "source.txt").write_text("source", encoding="ascii")
            _report_path, source_sha = self._write_candidate(workspace, passing=False)
            with patch(
                "quoridor.phase4_gates.fingerprint_sources",
                return_value=SimpleNamespace(sha256=source_sha),
            ):
                with self.assertRaises(Phase4GateError):
                    find_valid_memory_smoke(
                        workspace / "training/runs/phase4", workspace
                    )


if __name__ == "__main__":
    unittest.main()
