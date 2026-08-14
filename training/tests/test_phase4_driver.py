from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch

from quoridor.checkpoint import capture_rng_state
from quoridor.champion_store import ArtifactReference, ChampionStore
from quoridor.codec import encode_move
from quoridor.game import State, create_initial_state, legal_pawn_moves
from quoridor.openings import (
    build_opening_suite,
    generate_opening_suite,
    load_opening_suite,
)
from quoridor.paired_arena import (
    SelectorIdentity,
    SelectorRuntime,
    _play_leg,
    _snapshot_opening,
    _validated_legs,
)
from quoridor.phase4_arena import OpeningPairStore
from quoridor.phase4_checkpoint import (
    PHASE4_CANDIDATE_MODEL_FORMAT,
    PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
    PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION,
    canonical_model_state_sha256,
)
from quoridor.phase4_driver import (
    PHASE4_CONFIRMATION_SUITE_HASH,
    PHASE4_SCREEN_SUITE_HASH,
    Phase4PromotionController,
    Phase4PromotionError,
    bootstrap_opponent_descriptor,
    champion_opponent_descriptor,
    phase4_arena_contract_sha256,
    select_history_opponents,
)
from quoridor.run_manifest import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from quoridor.selectors import MCTS_ROOT_VISIT_SEMANTICS, SelectionResult


ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _goal_action(state: State) -> int:
    goal_row = 0 if state.current_player == 0 else 8
    row, col = min(
        legal_pawn_moves(state),
        key=lambda position: (
            abs(position[0] - goal_row),
            abs(position[1] - 4),
        ),
    )
    return encode_move(row, col)


class _GoalSelector:
    def __init__(self, simulations: int) -> None:
        self.simulations = simulations

    def choose(self, state: State) -> SelectionResult:
        return SelectionResult(
            action_index=_goal_action(state),
            engine="native-mcts",
            value=0.0,
            score=None,
            simulations=self.simulations,
            nodes=self.simulations,
            elapsed_ms=0.1,
            depth=0,
            metadata={
                "configuredSimulations": self.simulations,
                "batchSize": 1,
                "rootVisits": self.simulations,
                "retainedRootVisits": 0,
                "visitCountSemantics": MCTS_ROOT_VISIT_SEMANTICS,
            },
        )

    def close(self) -> None:
        pass


class Phase4FrozenPromotionSuiteTests(unittest.TestCase):
    def test_promotion_uses_dev_and_never_the_final_holdout(self) -> None:
        dev = load_opening_suite(
            ROOT / "training/fixtures/openings/dev.json",
            expected_suite="dev",
        )
        final = load_opening_suite(
            ROOT / "training/fixtures/openings/final.json",
            expected_suite="final",
        )
        screen = build_opening_suite(dev.openings[:32], suite="dev")

        self.assertEqual(PHASE4_SCREEN_SUITE_HASH, screen.content_hash)
        self.assertEqual(PHASE4_CONFIRMATION_SUITE_HASH, dev.content_hash)
        self.assertNotEqual(PHASE4_SCREEN_SUITE_HASH, final.content_hash)
        self.assertNotEqual(PHASE4_CONFIRMATION_SUITE_HASH, final.content_hash)


class Phase4PromotionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        screen_suite_hash = self._suite(2, "dev").content_hash
        confirmation_suite_hash = self._suite(3, "dev").content_hash
        self.constants = patch.multiple(
            "quoridor.phase4_driver",
            PHASE4_SCREEN_OPENING_PAIRS=2,
            PHASE4_SCREEN_GAMES=4,
            PHASE4_SCREEN_SIMULATIONS=2,
            PHASE4_SCREEN_MINIMUM_WIN_RATE=0.50,
            PHASE4_CONFIRMATION_OPENING_PAIRS=3,
            PHASE4_CONFIRMATION_GAMES=6,
            PHASE4_CONFIRMATION_SIMULATIONS=2,
            PHASE4_CONFIRMATION_MINIMUM_WIN_RATE=0.50,
            PHASE4_CONFIRMATION_MINIMUM_LOWER=0.49,
            PHASE4_HISTORY_MINIMUM_WIN_RATE=0.50,
            PHASE4_HISTORY_MINIMUM_LOWER=0.49,
            PHASE4_SCREEN_SUITE_HASH=screen_suite_hash,
            PHASE4_CONFIRMATION_SUITE_HASH=confirmation_suite_hash,
        )
        self.constants.start()
        self.addCleanup(self.constants.stop)

    def _checkpoint(
        self,
        root: Path,
        *,
        generation: int = 4,
        run_name: str = "training-run",
    ) -> tuple[Path, Path, str, str]:
        state = OrderedDict(
            {
                "linear.bias": torch.tensor([0.25], dtype=torch.float32),
                "linear.weight": torch.tensor([[1.5]], dtype=torch.float32),
            }
        )
        run_root = root / run_name
        checkpoint_root = run_root / "checkpoints"
        report_root = run_root / "reports"
        checkpoint_root.mkdir(parents=True)
        report_root.mkdir()
        config_sha256 = _digest(f"{run_name}-config")
        source_sha256 = _digest(f"{run_name}-source")
        manifest_core = {
            "schemaVersion": 1,
            "runId": run_name,
            "createdAt": "2026-07-26T00:00:00Z",
            "executionPolicy": {
                "scope": "local-machine-only",
                "localOnly": True,
                "networkRequired": False,
                "cloudServices": False,
            },
            "workspace": str(root),
            "provenance": {
                "gitMetadataUsed": False,
                "source": {"sha256": source_sha256},
                "config": {"sha256": config_sha256},
            },
            "environment": {},
            "resourceBudget": {},
            "metadata": {
                "pipeline": "phase4",
                "stage": "phase4e",
                "purpose": "formal-training",
                "formalPromotion": True,
                "executionContractSha256": _digest("phase4e-contract"),
            },
        }
        manifest_sha256 = sha256_bytes(canonical_json_bytes(manifest_core))
        manifest_path = run_root / "run-manifest.json"
        manifest_path.write_bytes(
            canonical_json_bytes(
                {**manifest_core, "manifestSha256": manifest_sha256}
            )
        )
        step = (generation + 1) * 64
        stem = f"checkpoint-g{generation:08d}-s{step:012d}"
        path = checkpoint_root / f"{stem}.pt"
        identity = {
            "runManifestSha256": manifest_sha256,
            "configSha256": config_sha256,
            "sourceSha256": source_sha256,
        }
        replay_snapshot = {
            "gameIds": [],
            "games": 0,
            "plies": 0,
            "bytes": 0,
            "manifestBytes": 1,
            "committedBytes": 1,
            "deepVerified": True,
            "manifestSha256": _digest(f"{run_name}-replay"),
        }
        generation_data = {
            "generation": generation,
            "replay": replay_snapshot,
        }
        torch.save(
            {
                "schemaVersion": 1,
                "generation": generation,
                "step": step,
                "identity": identity,
                "model": state,
                "optimizer": {},
                "trainerRuntime": {"step": step},
                "rng": capture_rng_state(),
                "generationData": generation_data,
                "replaySnapshot": replay_snapshot,
            },
            path,
        )
        checkpoint_sha256 = sha256_file(path)
        model_state_sha256 = canonical_model_state_sha256(state)
        candidate_model_path = checkpoint_root / f"candidate-model-{stem}.pt"
        torch.save(
            {
                "schemaVersion": PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
                "format": PHASE4_CANDIDATE_MODEL_FORMAT,
                "generation": generation,
                "step": step,
                "identity": identity,
                "checkpointSha256": checkpoint_sha256,
                "modelStateSha256": model_state_sha256,
                "modelState": state,
            },
            candidate_model_path,
        )
        candidate_model_sha256 = sha256_file(candidate_model_path)
        candidate_model_metadata = {
            "schemaVersion": PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
            "format": PHASE4_CANDIDATE_MODEL_FORMAT,
            "file": candidate_model_path.name,
            "bytes": candidate_model_path.stat().st_size,
            "sha256": candidate_model_sha256,
            "modelStateSha256": model_state_sha256,
        }
        path.with_suffix(".json").write_bytes(
            canonical_json_bytes(
                {
                    "schemaVersion": PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION,
                    "generation": generation,
                    "step": step,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint_sha256,
                    "identity": identity,
                    "candidateModel": candidate_model_metadata,
                }
            )
        )
        report_core = {
            "schemaVersion": 1,
            "generation": generation,
            "createdAt": "2026-07-26T00:00:01Z",
            "runManifestSha256": manifest_sha256,
            "report": {
                "checkpoint": {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint_sha256,
                    "step": step,
                    "candidateModel": {
                        "path": candidate_model_path.name,
                        "bytes": candidate_model_path.stat().st_size,
                        "sha256": candidate_model_sha256,
                        "modelStateSha256": model_state_sha256,
                    },
                },
                "generationData": generation_data,
            },
        }
        (report_root / f"generation-{generation:08d}.json").write_bytes(
            canonical_json_bytes(
                {
                    **report_core,
                    "reportSha256": sha256_bytes(
                        canonical_json_bytes(report_core)
                    ),
                }
            )
        )
        return (
            path,
            manifest_path,
            candidate_model_sha256,
            model_state_sha256,
        )

    @staticmethod
    def _suite(pairs: int, name: str):
        initial = create_initial_state()
        first_actions = tuple(
            encode_move(row, col)
            for row, col in legal_pawn_moves(initial)
        )
        action_sequences = [()]
        action_sequences.extend((action,) for action in first_actions)
        return generate_opening_suite(
            action_sequences[:pairs],
            suite=name,
        )

    def _bootstrap(self, root: Path) -> dict[str, object]:
        state = OrderedDict(
            {"weight": torch.tensor([0.125], dtype=torch.float32)}
        )
        model_state_sha256 = canonical_model_state_sha256(state)
        path = root / "baselines" / "bootstrap-warm-v2" / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schemaVersion": 1,
                "kind": "phase4e-bootstrap-baseline",
                "format": "quoridor-phase4e-bootstrap-model-only",
                "formalPromotion": False,
                "modelStateSha256": model_state_sha256,
                "modelState": state,
            },
            path,
        )
        artifact_sha256 = sha256_file(path)
        constants = patch.multiple(
            "quoridor.phase4_driver",
            PHASE4_BOOTSTRAP_MODEL_SHA256=artifact_sha256,
            PHASE4_BOOTSTRAP_MODEL_STATE_SHA256=model_state_sha256,
        )
        constants.start()
        self.addCleanup(constants.stop)
        return bootstrap_opponent_descriptor(
            baseline_id="bootstrap-warm-v2",
            artifact_sha256=artifact_sha256,
            model_state_sha256=model_state_sha256,
        )

    def _result(
        self,
        root: Path,
        *,
        name: str,
        pairs: int,
        simulations: int,
        suite_hash: str,
        checkpoint_sha256: str,
        opponent_sha256: str,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        suite = self._suite(pairs, "dev")
        self.assertEqual(suite.content_hash, suite_hash)
        identity = SelectorIdentity(
            selector_id="candidate",
            artifact_hash=checkpoint_sha256,
            config_hash=_digest("training-run-config"),
            source_hash=_digest("training-run-source"),
            engine="native-mcts",
        )
        opponent_identity = SelectorIdentity(
            selector_id="opponent",
            artifact_hash=opponent_sha256,
            config_hash=_digest("opponent-config"),
            source_hash=_digest("opponent-source"),
            engine="native-mcts",
        )
        runtime = SelectorRuntime(
            implementation="quoridor.native_mcts.NativeMCTS",
            kind="native-mcts",
            configured_simulations=simulations,
            budget_kind=None,
            budget_limit=None,
            max_depth=None,
            identity_bound=True,
            backend_hash=_digest("native"),
        )
        contract_sha256 = phase4_arena_contract_sha256(
            gate=name,
            execution_contract_sha256=_digest("phase4e-contract"),
            candidate_artifact_sha256=checkpoint_sha256,
            opponent_artifact_sha256=opponent_sha256,
            suite_hash=suite_hash,
        )
        store = OpeningPairStore(
            root,
            contract_sha256=contract_sha256,
            suite=suite,
            candidate_identity=identity,
            opponent_identity=opponent_identity,
            candidate_runtime=runtime,
            opponent_runtime=runtime,
        )
        candidate = _GoalSelector(simulations)
        opponent = _GoalSelector(simulations)
        for ordinal, original in enumerate(suite.openings):
            opening = _snapshot_opening(original)
            games = tuple(
                _play_leg(
                    suite,
                    opening,
                    leg,
                    candidate,
                    opponent,
                    identity,
                    opponent_identity,
                    runtime,
                    runtime,
                )
                for leg in _validated_legs(opening)
            )
            store.commit_pair(ordinal, games)
        store.finalize(
            bootstrap_seed=20260714,
            bootstrap_samples=10000,
            confidence=0.9,
        )
        return root / "result.json"

    def _controller(
        self,
        root: Path,
    ) -> tuple[Phase4PromotionController, Path, str, dict[str, object], str, str]:
        checkpoint, run_manifest, checkpoint_sha256, _ = self._checkpoint(root)
        store = ChampionStore(
            root / "champions",
            artifact_root=root,
        )
        controller = Phase4PromotionController(
            root / "candidate",
            artifact_root=root,
            execution_contract_sha256=_digest("phase4e-contract"),
            champion_store=store,
        )
        opponent = self._bootstrap(root)
        screen_suite = self._suite(2, "dev").content_hash
        confirmation_suite = self._suite(3, "dev").content_hash
        controller.prepare_candidate(
            checkpoint_path=checkpoint,
            run_manifest_path=run_manifest,
            primary_opponent=opponent,
        )
        return (
            controller,
            checkpoint,
            checkpoint_sha256,
            opponent,
            screen_suite,
            confirmation_suite,
        )

    def _seed_champions(
        self,
        store: ChampionStore,
        root: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        arena_path = root / "prior-arena.json"
        arena_path.write_bytes(canonical_json_bytes({"status": "accepted"}))
        reference = ArtifactReference.from_file(arena_path, relative_to=root)
        first_state = OrderedDict(
            {"weight": torch.tensor([0.5], dtype=torch.float32)}
        )
        first_state_sha256 = canonical_model_state_sha256(first_state)
        first = store.publish(
            first_state,
            generation=0,
            parent_id=None,
            config_sha256=_digest("prior-config-0"),
            source_sha256=_digest("prior-source-0"),
            opening_suite_sha256=_digest("prior-suite-0"),
            arena_report=reference,
            metadata={
                "candidateModelStateSha256": first_state_sha256,
            },
        )
        second_state = OrderedDict(
            {"weight": torch.tensor([0.75], dtype=torch.float32)}
        )
        second_state_sha256 = canonical_model_state_sha256(second_state)
        second = store.publish(
            second_state,
            generation=1,
            parent_id=first.champion_id,
            config_sha256=_digest("prior-config-1"),
            source_sha256=_digest("prior-source-1"),
            opening_suite_sha256=_digest("prior-suite-1"),
            arena_report=reference,
            metadata={
                "candidateModelStateSha256": second_state_sha256,
            },
        )
        return (
            champion_opponent_descriptor(first),
            champion_opponent_descriptor(second),
        )

    def _seed_champion_series(
        self,
        store: ChampionStore,
        root: Path,
        count: int,
    ) -> list[dict[str, object]]:
        arena_path = root / "history-arena.json"
        arena_path.write_bytes(canonical_json_bytes({"status": "accepted"}))
        reference = ArtifactReference.from_file(arena_path, relative_to=root)
        parent_id = store.pointers["latest"]
        records = []
        for generation in range(count):
            state = OrderedDict(
                {
                    "weight": torch.tensor(
                        [10.0 + generation], dtype=torch.float32
                    )
                }
            )
            record = store.publish(
                state,
                generation=generation,
                parent_id=parent_id,
                config_sha256=_digest(f"history-config-{generation}"),
                source_sha256=_digest(f"history-source-{generation}"),
                opening_suite_sha256=_digest(f"history-suite-{generation}"),
                arena_report=reference,
                metadata={
                    "candidateModelStateSha256": (
                        canonical_model_state_sha256(state)
                    ),
                },
            )
            records.append(champion_opponent_descriptor(record))
            parent_id = record.champion_id
        return records

    def test_full_gate_bundle_publishes_first_formal_champion_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                controller,
                _,
                checkpoint_sha256,
                opponent,
                screen_suite,
                confirmation_suite,
            ) = self._controller(root)
            self.assertEqual(controller.status().next_gate, "health")
            controller.record_health()
            screen = self._result(
                root / "screen-arena",
                name="screen",
                pairs=2,
                simulations=2,
                suite_hash=screen_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            controller.record_arena_gate("screen", screen)
            confirmation = self._result(
                root / "confirmation-arena",
                name="confirmation",
                pairs=3,
                simulations=2,
                suite_hash=confirmation_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            controller.record_arena_gate("confirmation", confirmation)
            self.assertEqual(controller.status().next_gate, "promotion-bundle")
            bundle = controller.build_promotion_bundle()
            self.assertTrue(bundle["formalPromotion"])
            first = controller.publish()
            second = controller.publish()
            self.assertEqual(first.champion_id, second.champion_id)
            self.assertEqual(controller.champion_store.pointers["latest"], first.champion_id)
            self.assertIsNone(controller.champion_store.pointers["best"])
            self.assertIsNone(controller.champion_store.pointers["product"])
            self.assertTrue(first.metadata["formalPromotion"])

    def test_failed_screen_stops_later_gates_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                controller,
                _,
                checkpoint_sha256,
                opponent,
                screen_suite,
                confirmation_suite,
            ) = self._controller(root)
            controller.record_health()
            screen = self._result(
                root / "screen-arena",
                name="screen",
                pairs=2,
                simulations=2,
                suite_hash=screen_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            confirmation = self._result(
                root / "confirmation-arena",
                name="confirmation",
                pairs=3,
                simulations=2,
                suite_hash=confirmation_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            with patch(
                "quoridor.phase4_driver.PHASE4_SCREEN_MINIMUM_WIN_RATE",
                0.51,
            ):
                evidence = controller.record_arena_gate("screen", screen)
                self.assertFalse(evidence["passed"])
                self.assertEqual(controller.status().state, "rejected")
                with self.assertRaises(Phase4PromotionError):
                    controller.record_arena_gate(
                        "confirmation", confirmation
                    )
                with self.assertRaises(Phase4PromotionError):
                    controller.publish()
            self.assertEqual(len(controller.champion_store), 0)

    def test_rehashed_pair_result_cannot_forge_the_summary_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                controller,
                _,
                checkpoint_sha256,
                opponent,
                screen_suite,
                _,
            ) = self._controller(root)
            controller.record_health()
            result_path = self._result(
                root / "screen-arena",
                name="screen",
                pairs=2,
                simulations=2,
                suite_hash=screen_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            artifact = result["pairArtifacts"][0]
            old_pair_path = result_path.parent / artifact["file"]
            pair = json.loads(old_pair_path.read_text(encoding="utf-8"))
            game = pair["games"][0]
            game["winner"] = 1 - game["winner"]
            game["candidateWon"] = not game["candidateWon"]
            pair_core = dict(pair)
            pair_core.pop("pairSha256")
            pair_sha256 = sha256_bytes(canonical_json_bytes(pair_core))
            new_pair_path = (
                result_path.parent
                / (
                    f"pair-{artifact['ordinal']:06d}-{artifact['openingId']}-"
                    f"{pair_sha256[:12]}.json"
                )
            )
            new_pair_path.write_bytes(
                canonical_json_bytes(
                    {**pair_core, "pairSha256": pair_sha256}
                )
            )
            old_pair_path.unlink()
            artifact.update(
                {
                    "file": new_pair_path.name,
                    "sha256": sha256_file(new_pair_path),
                    "bytes": new_pair_path.stat().st_size,
                }
            )
            result_core = dict(result)
            result_core.pop("resultSha256")
            result_path.write_bytes(
                canonical_json_bytes(
                    {
                        **result_core,
                        "resultSha256": sha256_bytes(
                            canonical_json_bytes(result_core)
                        ),
                    }
                )
            )

            with self.assertRaises(Phase4PromotionError):
                controller.record_arena_gate("screen", result_path)

    def test_rehashed_result_cannot_forge_confidence_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                controller,
                _,
                checkpoint_sha256,
                opponent,
                screen_suite,
                _,
            ) = self._controller(root)
            controller.record_health()
            result_path = self._result(
                root / "screen-arena",
                name="screen",
                pairs=2,
                simulations=2,
                suite_hash=screen_suite,
                checkpoint_sha256=checkpoint_sha256,
                opponent_sha256=str(opponent["artifactSha256"]),
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["statistics"]["winRateCI"]["lower"] = 0.59
            result_core = dict(result)
            result_core.pop("resultSha256")
            result_path.write_bytes(
                canonical_json_bytes(
                    {
                        **result_core,
                        "resultSha256": sha256_bytes(
                            canonical_json_bytes(result_core)
                        ),
                    }
                )
            )

            with self.assertRaises(Phase4PromotionError):
                controller.record_arena_gate("screen", result_path)

    def test_candidate_cadence_and_history_dedup_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, run_manifest, _, _ = self._checkpoint(root)
            store = ChampionStore(root / "champions", artifact_root=root)
            controller = Phase4PromotionController(
                root / "candidate",
                artifact_root=root,
                execution_contract_sha256=_digest("phase4e-contract"),
                champion_store=store,
            )
            primary = self._bootstrap(root)
            bad_checkpoint, bad_manifest, _, _ = self._checkpoint(
                root,
                generation=5,
                run_name="bad-cadence-run",
            )
            with self.assertRaises(Phase4PromotionError):
                controller.prepare_candidate(
                    checkpoint_path=bad_checkpoint,
                    run_manifest_path=bad_manifest,
                    primary_opponent=primary,
                )
            candidate = controller.prepare_candidate(
                checkpoint_path=checkpoint,
                run_manifest_path=run_manifest,
                primary_opponent=primary,
            )
            self.assertEqual(candidate["historyOpponents"], [])

    def test_real_rng_checkpoint_is_admitted_only_through_safe_companion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, run_manifest, candidate_model_sha256, _ = (
                self._checkpoint(root)
            )
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
            store = ChampionStore(root / "champions", artifact_root=root)
            controller = Phase4PromotionController(
                root / "candidate",
                artifact_root=root,
                execution_contract_sha256=_digest("phase4e-contract"),
                champion_store=store,
            )
            candidate = controller.prepare_candidate(
                checkpoint_path=checkpoint,
                run_manifest_path=run_manifest,
                primary_opponent=self._bootstrap(root),
            )
            self.assertEqual(
                candidate["candidateModel"]["sha256"],
                candidate_model_sha256,
            )

    def test_candidate_history_is_derived_from_champion_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, run_manifest, _, _ = self._checkpoint(root)
            store = ChampionStore(root / "champions", artifact_root=root)
            first, latest = self._seed_champions(store, root)
            controller = Phase4PromotionController(
                root / "candidate",
                artifact_root=root,
                execution_contract_sha256=_digest("phase4e-contract"),
                champion_store=store,
            )
            candidate = controller.prepare_candidate(
                checkpoint_path=checkpoint,
                run_manifest_path=run_manifest,
                primary_opponent=latest,
            )
            self.assertEqual(candidate["historyOpponents"], [first])

            forged = dict(candidate)
            forged["historyOpponents"] = []
            forged.pop("candidateSha256")
            forged["candidateSha256"] = sha256_bytes(
                canonical_json_bytes(forged)
            )
            controller.manifest_path.write_bytes(canonical_json_bytes(forged))
            with self.assertRaises(Phase4PromotionError):
                controller.load_candidate()

    def test_history_scans_past_excluded_primary_before_taking_recent_five(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, run_manifest, _, _ = self._checkpoint(root)
            store = ChampionStore(root / "champions", artifact_root=root)
            champions = self._seed_champion_series(store, root, 7)
            controller = Phase4PromotionController(
                root / "candidate",
                artifact_root=root,
                execution_contract_sha256=_digest("phase4e-contract"),
                champion_store=store,
            )

            candidate = controller.prepare_candidate(
                checkpoint_path=checkpoint,
                run_manifest_path=run_manifest,
                primary_opponent=champions[-1],
            )

            self.assertEqual(
                candidate["historyOpponents"],
                list(reversed(champions[1:6])),
            )

    def test_low_retention_and_pruned_history_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_retention = ChampionStore(
                root / "low-retention",
                artifact_root=root,
                recent_to_keep=5,
            )
            with self.assertRaisesRegex(
                Phase4PromotionError,
                "frozen champion resource contract",
            ):
                Phase4PromotionController(
                    root / "candidate",
                    artifact_root=root,
                    execution_contract_sha256=_digest("phase4e-contract"),
                    champion_store=low_retention,
                )

            custom_budget = ChampionStore(
                root / "custom-budget",
                artifact_root=root,
                global_artifact_limit_bytes=8 * 1024 * 1024 * 1024 - 1,
            )
            with self.assertRaisesRegex(
                Phase4PromotionError,
                "frozen champion resource contract",
            ):
                Phase4PromotionController(
                    root / "custom-candidate",
                    artifact_root=root,
                    execution_contract_sha256=_digest("phase4e-contract"),
                    champion_store=custom_budget,
                )

            pruned_store = ChampionStore(
                root / "pruned",
                artifact_root=root,
                recent_to_keep=1,
            )
            champions = self._seed_champion_series(pruned_store, root, 2)
            with self.assertRaisesRegex(
                Phase4PromotionError,
                "required history champion model was pruned",
            ):
                select_history_opponents(
                    pruned_store,
                    exclude_model_sha256=(
                        str(champions[-1]["modelStateSha256"]),
                    ),
                )

    def test_self_hashed_gate_without_source_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                controller,
                _,
                _,
                opponent,
                _,
                _,
            ) = self._controller(root)
            controller.record_health()
            core = {
                "schemaVersion": 1,
                "kind": "phase4e-gate-evidence",
                "gate": "screen",
                "opponent": opponent,
                "candidateSha256": controller.load_candidate()[
                    "candidateSha256"
                ],
                "passed": True,
                "checks": {"forged": True},
            }
            path = controller._gate_path("screen", str(opponent["id"]))
            path.write_bytes(
                canonical_json_bytes(
                    {
                        **core,
                        "gateSha256": sha256_bytes(
                            canonical_json_bytes(core)
                        ),
                    }
                )
            )
            with self.assertRaises(Phase4PromotionError):
                controller.status()

    def test_arbitrary_model_file_cannot_become_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arbitrary = root / "candidate.pt"
            torch.save(
                {
                    "schemaVersion": 1,
                    "generation": 4,
                    "step": 320,
                    "model": OrderedDict(
                        {"weight": torch.tensor([1.0])}
                    ),
                },
                arbitrary,
            )
            store = ChampionStore(root / "champions", artifact_root=root)
            controller = Phase4PromotionController(
                root / "candidate",
                artifact_root=root,
                execution_contract_sha256=_digest("phase4e-contract"),
                champion_store=store,
            )
            with self.assertRaises(Phase4PromotionError):
                controller.prepare_candidate(
                    checkpoint_path=arbitrary,
                    run_manifest_path=root / "missing-run-manifest.json",
                    primary_opponent=bootstrap_opponent_descriptor(
                        baseline_id="bootstrap",
                        artifact_sha256=_digest("bootstrap-artifact"),
                        model_state_sha256=_digest("bootstrap-state"),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
