"""Durable Phase 4 self-play and training driver.

Complete raw games in :class:`ReplayStore` remain the source of truth. Phase 4D
keeps its frozen in-memory reconstruction path; later stages use a bounded,
reconstructable, disk-backed tensor cache for optimizer batches.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch.nn import functional as F

from .checkpoint import restore_rng_state
from .calibration import (
    GPU_HOUR_BUDGET,
    GPU_LEDGER_NAME,
    GPU_LOCK_NAME,
    GpuBudgetLedger,
)
from .challenge_replay import (
    merge_tensor_batches,
    mixed_batch_counts,
    open_bound_challenge_replay,
    open_bound_health_holdout,
)
from .challenge_sampler import load_stratified_challenge_sampler
from .champion_store import ChampionStore
from .codec import (
    ACTION_SPACE_SIZE,
    CODEC_VERSION,
    mirror_action_index,
    mirror_state,
    to_canonical_action_index,
)
from .config import (
    ExperimentConfig,
    load_experiment_config,
    resolve_device,
    seed_everything,
)
from .evaluator import TorchEvaluator
from .game import State, get_winner
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import (
    ACTION_SIZE,
    PHASE4_INPUT_CHANNELS,
    NetworkConfig,
    PolicyValueNetwork,
    encode_state,
)
from .phase4_budget import (
    PHASE4_CONTROL_RELATIVE,
    PHASE4_STAGE_LEDGER_NAME,
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
    PHASE4E_STAGE,
    PHASE4F5_STAGE,
    PHASE4F18_PILOT_STAGE,
    PHASE4F35_PILOT_STAGE,
    PHASE4F_RECOVERY_STAGE,
    PHASE4F_STAGE_A,
    PHASE4F_STAGE_A_CONTINUATION,
    PHASE4F_STAGE_B_PILOT,
    PHASE4F_STAGE_B,
    PHASE4F_STAGE_B_CONTINUATION,
    PHASE4F_STAGE_B_EXTENDED,
    Phase4StageBudgetLedger,
    verify_phase4d_baseline_report,
    verify_phase4e_baseline_report,
)
from .phase4_execution import (
    PHASE4D_EXECUTION_CONTRACT,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_checkpoint import (
    PHASE4_CANDIDATE_MODEL_FORMAT,
    PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
    CheckpointIdentity,
    Phase4CheckpointLoad,
    Phase4CheckpointRecord,
    Phase4CheckpointStore,
    canonical_model_state_sha256,
)
from .paired_arena import loaded_native_backend_hash
from .replay_store import (
    REPLAY_SHARD_SCHEMA_VERSION,
    REPLAY_STORE_SCHEMA_VERSION,
    ReplayGame,
    ReplayPly,
    ReplaySample,
    ReplayShardInfo,
    ReplayStore,
    ReplayVerification,
    reconstruct_game,
)
from .replay_tensor_cache import (
    REPLAY_MIRROR_RNG_ALGORITHM,
    REPLAY_SAMPLING_ALGORITHM,
    ReplayTensorSamplePlan,
    ReplayTensorCache,
)
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    GenerationReportStore,
    ManifestCorruptionError,
    ResourceBudget,
    RunLock,
    RunManifestRecord,
    atomic_create_bytes,
    atomic_replace_bytes,
    build_run_manifest,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .selfplay import (
    SelfPlayConfig,
    SelfPlayGame,
    play_self_games_scheduled,
    scheduled_search_seed,
)
from .storage import directory_size
from .trainer import AlphaZeroTrainer, TrainMetrics


PHASE4_TRAINING_SCHEMA_VERSION = 1
PHASE4_WARM_ARTIFACT_SCHEMA_VERSION = 1
PHASE4D_PILOT_GENERATIONS = 5
_BUDGET_CHECK_INTERVAL_SECONDS = 30.0
_LEDGER_HEARTBEAT_SECONDS = 5.0
_MIB = 1024 * 1024
_REPLAY_MANIFEST_WRITE_RESERVE_BYTES = 1 * _MIB
_REPLAY_GAME_WRITE_RESERVE_BYTES = 4 * _MIB
_REPORT_WRITE_RESERVE_BYTES = 1 * _MIB
_CHECKPOINT_FIXED_WRITE_RESERVE_BYTES = 64 * _MIB
_CONTROL_WRITE_RESERVE_BYTES = 4 * _MIB
_GIB = 1024 * _MIB
REPLAY_CACHE_ALGORITHM_VERSION = 1
REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY = 32 * 1024
REPLAY_CACHE_MAX_ESTIMATED_BYTES = 4 * _GIB
REPLAY_INPUT_GATE_BATCH_SIZE = 512
REPLAY_INPUT_GATE_BATCHES = 8
REPLAY_INPUT_GATE_MAX_BUILD_SECONDS = 120.0
REPLAY_INPUT_GATE_MIN_BUILD_POSITIONS_PER_SECOND = 60.0
REPLAY_INPUT_GATE_MAX_MEAN_SECONDS = 2.0
REPLAY_INPUT_GATE_MAX_BATCH_SECONDS = 3.0
REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS = 5.0
REPLAY_INPUT_GATE_SEED = 20260715
REPLAY_INPUT_GATE_FIXTURE_GAMES = 64
REPLAY_INPUT_GATE_FIXTURE_PLIES = 7181
REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256 = (
    "f3e872e9ed058935a44d60e93e49b88e729ac1f8d7eae1f5bf5e116133351762"
)
REPLAY_INPUT_GATE_FIXTURE_RELATIVE = Path(
    "training/runs/phase4/calibration/"
    "phase4-calibration-20260721-171946-972095/replay/"
    "192x16-a32-s256-g64"
)
REPLAY_INPUT_GATE_REPORT_RELATIVE = Path("replay-input-gate")


class Phase4TrainingError(RuntimeError):
    """Base class for formal Phase 4 training failures."""


class Phase4ResumeError(Phase4TrainingError):
    """Raised when persisted state cannot be resumed without ambiguity."""


class Phase4ResourceLimitReached(Phase4TrainingError):
    """Raised at a Phase 4 global or stage stop line."""


def _publish_progress(
    callback: Callable[[Mapping[str, object]], None] | None,
    *,
    stage: str,
    status: str,
    **details: object,
) -> None:
    if callback is None:
        return
    callback(
        {
            "schemaVersion": 1,
            "kind": "phase4-training-progress",
            "stage": stage,
            "status": status,
            **details,
        }
    )


def _existing_disk_anchor(path: Path) -> Path:
    anchor = path.resolve()
    while not anchor.exists():
        if anchor.parent == anchor:
            raise FileNotFoundError(f"no existing parent for resource check: {path}")
        anchor = anchor.parent
    return anchor


def _static_write_preflight(
    *,
    artifact_root: Path,
    artifact_limit_bytes: int,
    reserve_bytes: int,
) -> dict[str, int]:
    if type(artifact_limit_bytes) is not int or artifact_limit_bytes <= 0:
        raise ValueError("artifact_limit_bytes must be a positive integer")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    root = artifact_root.resolve()
    free_bytes = int(shutil.disk_usage(_existing_disk_anchor(root)).free)
    artifact_bytes = directory_size(root) if root.exists() else 0
    if free_bytes - reserve_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4ResourceLimitReached(
            "write reserve would cross the 20 GiB free-disk stop line"
        )
    if artifact_bytes + reserve_bytes > artifact_limit_bytes:
        raise Phase4ResourceLimitReached(
            "write reserve would exceed the configured artifact cap"
        )
    return {
        "freeDiskBytes": free_bytes,
        "artifactBytes": artifact_bytes,
        "reservedWriteBytes": reserve_bytes,
    }


@dataclass(frozen=True, slots=True)
class Phase4Lineage:
    manifest: RunManifestRecord
    identity: CheckpointIdentity
    resumed: bool


def _network_config(config: ExperimentConfig) -> NetworkConfig:
    return NetworkConfig(
        input_channels=config.network.input_channels,
        channels=config.network.channels,
        blocks=config.network.residual_blocks,
        se_reduction=config.network.se_reduction,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden=config.network.value_hidden,
    )


def _trainer(config: ExperimentConfig, model: PolicyValueNetwork) -> AlphaZeroTrainer:
    return AlphaZeroTrainer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        use_amp=config.training.use_amp,
        schedule_steps=config.training.schedule_steps,
        minimum_learning_rate=config.training.minimum_learning_rate,
    )


def validate_phase4_config(
    config: ExperimentConfig,
    execution_contract: Phase4ExecutionContract = PHASE4D_EXECUTION_CONTRACT,
) -> None:
    if not isinstance(execution_contract, Phase4ExecutionContract):
        raise TypeError("execution_contract must be a Phase4ExecutionContract")
    if config.pipeline != "phase4":
        raise ValueError("the Phase 4 driver requires pipeline=phase4")
    if config.network.input_channels != PHASE4_INPUT_CHANNELS:
        raise ValueError("Phase 4 training requires the 16-plane feature contract")
    if config.training.max_checkpoints < 2:
        raise ValueError("Phase 4 requires at least two checkpoints for fallback")
    if config.initialization.expected_base_model_sha256 is None:
        raise ValueError(
            "Phase 4 requires initialization.expectedBaseModelSha256"
        )
    continuation = (
        execution_contract.stage == PHASE4F_STAGE_B_EXTENDED
        and config.initialization.mode == "recovery"
        and not config.initialization.reset_optimizer
    )
    if not config.initialization.reset_optimizer and not continuation:
        raise ValueError(
            "only the frozen Stage B extended contract may retain optimizer state"
        )
    if config.training.training_steps < REPLAY_INPUT_GATE_BATCHES:
        raise ValueError(
            "Phase 4 requires at least eight training steps for the input gate"
        )
    if config.training.learning_stage is not None:
        if execution_contract.maximum_generations > 2 and not continuation:
            raise ValueError(
                "staged learning requires a two-generation execution ceiling"
            )
        if config.training.max_checkpoints < 3:
            raise ValueError(
                "staged learning requires three checkpoints to retain its best candidate"
            )
    minimum_retained_plies = (
        config.self_play.games
        * 256
        * min(execution_contract.maximum_generations, PHASE4D_PILOT_GENERATIONS)
    )
    if config.training.replay_capacity < minimum_retained_plies:
        raise ValueError(
            "training.replayCapacity must retain at least the initial "
            "five-generation training window"
        )


def _verify_phase4e_training_inputs(
    baseline_evidence: Mapping[str, object],
    *,
    config_sha256: str,
    source_sha256: str,
    native_backend_sha256: str,
) -> None:
    """Bind an eligible Phase 4E decision to the exact training runtime."""

    try:
        config_sha256 = validate_sha256(
            config_sha256,
            "effective Phase 4E workload sha256",
        )
        source_sha256 = validate_sha256(
            source_sha256,
            "current Phase 4E source sha256",
        )
        native_backend_sha256 = validate_sha256(
            native_backend_sha256,
            "current Phase 4E native backend sha256",
        )
    except ValueError as error:
        raise Phase4TrainingError(
            "current Phase 4E training provenance is invalid"
        ) from error
    approved = baseline_evidence.get("approvedWorkloads")
    if (
        not isinstance(approved, list)
        or not approved
        or config_sha256 not in approved
    ):
        raise Phase4TrainingError(
            "effective Phase 4E workload was not approved by the baseline report"
        )
    if baseline_evidence.get("sourceSha256") != source_sha256:
        raise Phase4TrainingError(
            "current Phase 4E source differs from the baseline report"
        )
    if baseline_evidence.get("nativeBackendSha256") != native_backend_sha256:
        raise Phase4TrainingError(
            "current Phase 4E native backend differs from the baseline report"
        )


def prepare_phase4_lineage(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    config: ExperimentConfig,
    budget: ResourceBudget,
    resume: bool,
    source_paths: Sequence[str | Path] | None = None,
    environment: Mapping[str, object] | None = None,
    expected_source_sha256: str | None = None,
    execution_contract: Phase4ExecutionContract = PHASE4D_EXECUTION_CONTRACT,
) -> Phase4Lineage:
    """Create a lineage once or strictly verify config and source on resume."""

    validate_phase4_config(config, execution_contract)
    workspace = Path(workspace_root).resolve()
    root = Path(run_dir).resolve()
    try:
        root.relative_to(workspace)
    except ValueError as error:
        raise ValueError("Phase 4 run_dir must be inside the workspace") from error
    manifest_path = root / "run-manifest.json"
    config_fingerprint = fingerprint_config(config)
    source_fingerprint = fingerprint_sources(workspace, source_paths)
    if expected_source_sha256 is not None:
        expected_source_sha256 = validate_sha256(
            expected_source_sha256,
            "expected source sha256",
        )
        if source_fingerprint.sha256 != expected_source_sha256:
            raise Phase4TrainingError(
                "training source changed before lineage preparation"
            )
    if manifest_path.exists():
        if not resume:
            raise Phase4ResumeError(
                "run manifest already exists; resume must be explicitly enabled"
            )
        manifest = load_run_manifest(manifest_path)
        provenance = manifest.payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ManifestCorruptionError("run manifest provenance is missing")
        stored_config = provenance.get("config")
        stored_source = provenance.get("source")
        if not isinstance(stored_config, Mapping) or not isinstance(
            stored_source, Mapping
        ):
            raise ManifestCorruptionError(
                "run manifest config/source provenance is invalid"
            )
        if stored_config.get("sha256") != config_fingerprint.sha256:
            raise Phase4ResumeError("experiment config changed since run creation")
        if stored_source.get("sha256") != source_fingerprint.sha256:
            raise Phase4ResumeError("training source changed since run creation")
        if manifest.payload.get("resourceBudget") != budget.to_payload():
            raise Phase4ResumeError("resource budget changed since run creation")
        return Phase4Lineage(
            manifest=manifest,
            identity=CheckpointIdentity(
                run_manifest_sha256=manifest.sha256,
                config_sha256=config_fingerprint.sha256,
                source_sha256=source_fingerprint.sha256,
            ),
            resumed=True,
        )
    if resume:
        raise Phase4ResumeError("resume requested but run-manifest.json is missing")

    protected_roots = (
        root / "checkpoints",
        root / "replay",
        root / "replay-tensor-cache",
        root / "reports",
    )
    if any(path.exists() for path in protected_roots):
        raise Phase4ResumeError(
            "run data exists without an immutable run manifest"
        )
    manifest = build_run_manifest(
        workspace_root=workspace,
        config=config,
        budget=budget,
        run_id=f"{config.name}-{config.seed}",
        source_paths=source_paths,
        environment=environment,
        metadata={
            "pipeline": "phase4",
            "stage": execution_contract.stage,
            "purpose": execution_contract.purpose,
            "formalPromotion": execution_contract.formal_promotion,
            "executionContract": execution_contract.to_payload(),
            "executionContractSha256": execution_contract.sha256,
        },
    )
    manifest_provenance = manifest.payload.get("provenance")
    manifest_source = (
        manifest_provenance.get("source")
        if isinstance(manifest_provenance, Mapping)
        else None
    )
    if (
        expected_source_sha256 is not None
        and (
            not isinstance(manifest_source, Mapping)
            or manifest_source.get("sha256") != expected_source_sha256
        )
    ):
        raise Phase4TrainingError(
            "training source changed before run manifest commit"
        )
    standard_control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    try:
        root.relative_to(standard_control_root)
    except ValueError:
        artifact_root = root
    else:
        artifact_root = standard_control_root
    _static_write_preflight(
        artifact_root=artifact_root,
        artifact_limit_bytes=budget.artifact_limit_bytes,
        reserve_bytes=max(
            _REPLAY_MANIFEST_WRITE_RESERVE_BYTES,
            len(canonical_json_bytes(manifest.payload)) + 64 * 1024,
        ),
    )
    save_run_manifest(manifest, manifest_path)
    return Phase4Lineage(
        manifest=manifest,
        identity=CheckpointIdentity(
            run_manifest_sha256=manifest.sha256,
            config_sha256=config_fingerprint.sha256,
            source_sha256=source_fingerprint.sha256,
        ),
        resumed=False,
    )


class ReplayBatchEncoder:
    """Reconstruct one ephemeral tensor batch from raw replay plies."""

    def __init__(
        self,
        *,
        input_channels: int,
        device: str | torch.device,
        native_encoder: Callable[..., object] | None = None,
    ) -> None:
        if input_channels != PHASE4_INPUT_CHANNELS:
            raise ValueError("Phase 4 replay batches require 16 input channels")
        if native_encoder is not None and not callable(native_encoder):
            raise TypeError("native_encoder must be callable or None")
        self.input_channels = input_channels
        self.device = torch.device(device)
        self.native_encoder = native_encoder

    def encode(
        self,
        samples: Sequence[ReplaySample],
        *,
        rng: random.Random,
        mirror_probability: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("cannot encode an empty replay batch")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        if not math.isfinite(mirror_probability) or not 0 <= mirror_probability <= 1:
            raise ValueError("mirror_probability must be in [0, 1]")

        states = []
        policies = []
        values = []
        for sample in samples:
            if not isinstance(sample, ReplaySample):
                raise TypeError("samples must contain ReplaySample values")
            mirrored = rng.random() < mirror_probability
            state = mirror_state(sample.state) if mirrored else sample.state
            policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
            for raw_index, count in enumerate(sample.visit_counts):
                if count <= 0:
                    continue
                transformed_index = (
                    mirror_action_index(raw_index) if mirrored else raw_index
                )
                canonical_index = to_canonical_action_index(
                    transformed_index,
                    state.current_player,
                )
                policy[canonical_index] += float(count)
            total = policy.sum()
            if not bool(torch.isfinite(total).item()) or total <= 0:
                raise ValueError("replay visit counts have no finite positive mass")
            states.append(state)
            policies.append(policy / total)
            values.append(float(sample.outcome))

        if self.native_encoder is None:
            encoded_states = torch.stack(
                [
                    encode_state(state, input_channels=self.input_channels)
                    for state in states
                ]
            )
        else:
            encoded = self.native_encoder(
                states,
                input_channels=self.input_channels,
            )
            encoded_states = torch.from_numpy(encoded)
            expected = (len(states), self.input_channels, 9, 9)
            if tuple(encoded_states.shape) != expected:
                raise ValueError(f"native encoder must return shape {expected}")
            if encoded_states.dtype != torch.float32:
                raise ValueError("native encoder must return float32 states")
        return {
            "states": encoded_states.to(self.device),
            "policies": torch.stack(policies).to(self.device),
            "values": torch.tensor(values, dtype=torch.float32, device=self.device),
        }


def _replay_shard_signature(shard: ReplayShardInfo) -> tuple[object, ...]:
    return (
        shard.sequence,
        shard.game_id,
        shard.file_name,
        shard.byte_size,
        shard.sha256,
        shard.ply_count,
        shard.minimum_generation,
        shard.maximum_generation,
    )


@dataclass(frozen=True, slots=True)
class _CachedReplayGame:
    shard: ReplayShardInfo
    game: ReplayGame
    states: tuple[State, ...]
    winner: int

    def __post_init__(self) -> None:
        if len(self.states) != len(self.game.plies):
            raise ValueError("cached replay state count must equal its ply count")
        if self.shard.game_id != self.game.game_id:
            raise ValueError("cached replay shard/game identity differs")
        if self.shard.ply_count != len(self.states):
            raise ValueError("cached replay shard ply count differs")
        if self.winner not in (0, 1):
            raise ValueError("cached replay winner must be player zero or one")


@dataclass(frozen=True, slots=True)
class ReplayCacheRefresh:
    algorithm_version: int
    manifest_sha256: str
    prefix_sha256: str
    games: int
    plies: int
    estimated_bytes: int
    reused_games: int
    decoded_games: int
    decoded_plies: int
    refresh_seconds: float
    cumulative_build_seconds: float

    @property
    def decoded_positions_per_second(self) -> float | None:
        if self.decoded_plies <= 0:
            return None
        return self.decoded_plies / max(self.refresh_seconds, 1e-12)

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithmVersion": self.algorithm_version,
            "manifestSha256": self.manifest_sha256,
            "prefixSha256": self.prefix_sha256,
            "games": self.games,
            "plies": self.plies,
            "estimatedBytes": self.estimated_bytes,
            "reusedGames": self.reused_games,
            "decodedGames": self.decoded_games,
            "decodedPlies": self.decoded_plies,
            "refreshSeconds": self.refresh_seconds,
            "cumulativeBuildSeconds": self.cumulative_build_seconds,
            "decodedPositionsPerSecond": self.decoded_positions_per_second,
        }


class ReplayReconstructionCache:
    """Ephemeral exact-state cache over one append-only replay store."""

    def __init__(
        self,
        replay: ReplayStore,
        *,
        maximum_estimated_bytes: int = REPLAY_CACHE_MAX_ESTIMATED_BYTES,
    ) -> None:
        if not isinstance(replay, ReplayStore):
            raise TypeError("replay must be a ReplayStore")
        if (
            type(maximum_estimated_bytes) is not int
            or maximum_estimated_bytes <= 0
        ):
            raise ValueError("maximum_estimated_bytes must be a positive integer")
        self.replay = replay
        self.maximum_estimated_bytes = maximum_estimated_bytes
        self._root = replay.root.resolve()
        self._capacity = (replay.max_games, replay.max_plies, replay.max_bytes)
        self._blocks: tuple[_CachedReplayGame, ...] = ()
        self._signatures: tuple[tuple[object, ...], ...] = ()
        self._manifest_sha256: str | None = None
        self._cumulative_build_seconds = 0.0
        self._stale = False
        self._last_refresh: ReplayCacheRefresh | None = None

    @property
    def is_built(self) -> bool:
        return self._last_refresh is not None and not self._stale

    def _assert_store_identity(self) -> None:
        if self.replay.root.resolve() != self._root or (
            self.replay.max_games,
            self.replay.max_plies,
            self.replay.max_bytes,
        ) != self._capacity:
            self._stale = True
            raise Phase4ResumeError("replay cache was rebound to another store")

    def _current_signatures(self) -> tuple[tuple[object, ...], ...]:
        return tuple(_replay_shard_signature(shard) for shard in self.replay.shards)

    def _validated_manifest_sha256(self) -> str:
        try:
            raw = self.replay.manifest_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase4ResumeError("replay manifest cannot be decoded") from error
        expected_keys = {
            "schemaVersion",
            "shardSchemaVersion",
            "codecVersion",
            "actionSpaceSize",
            "capacity",
            "nextSequence",
            "totals",
            "shards",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected_keys
            or canonical_json_bytes(payload) != raw
            or payload.get("schemaVersion") != REPLAY_STORE_SCHEMA_VERSION
            or payload.get("shardSchemaVersion") != REPLAY_SHARD_SCHEMA_VERSION
            or payload.get("codecVersion") != CODEC_VERSION
            or payload.get("actionSpaceSize") != ACTION_SPACE_SIZE
        ):
            raise Phase4ResumeError("replay manifest contract changed")
        capacity = payload.get("capacity")
        expected_capacity = {
            "games": self.replay.max_games,
            "plies": self.replay.max_plies,
            "bytes": self.replay.max_bytes,
        }
        current_shards = self.replay.shards
        totals = payload.get("totals")
        expected_totals = {
            "games": len(current_shards),
            "plies": sum(shard.ply_count for shard in current_shards),
            "shardBytes": sum(shard.byte_size for shard in current_shards),
        }
        raw_shards = payload.get("shards")
        if (
            not isinstance(capacity, Mapping)
            or set(capacity) != {"games", "plies", "bytes"}
            or any(
                type(capacity.get(key)) is not int or capacity.get(key) <= 0
                for key in ("games", "plies", "bytes")
            )
            or not isinstance(totals, Mapping)
            or set(totals) != {"games", "plies", "shardBytes"}
            or any(
                type(totals.get(key)) is not int or totals.get(key) < 0
                for key in ("games", "plies", "shardBytes")
            )
            or capacity != expected_capacity
            or totals != expected_totals
            or not isinstance(raw_shards, list)
            or len(raw_shards) != len(current_shards)
        ):
            raise Phase4ResumeError("replay manifest differs from the open store")
        manifest_signatures = []
        expected_shard_keys = {
            "sequence",
            "gameId",
            "file",
            "bytes",
            "sha256",
            "plies",
            "minimumGeneration",
            "maximumGeneration",
        }
        for raw_shard in raw_shards:
            if not isinstance(raw_shard, Mapping) or set(raw_shard) != expected_shard_keys:
                raise Phase4ResumeError("replay manifest shard schema changed")
            if (
                type(raw_shard.get("sequence")) is not int
                or raw_shard.get("sequence") < 0
                or not isinstance(raw_shard.get("gameId"), str)
                or not isinstance(raw_shard.get("file"), str)
                or Path(str(raw_shard.get("file"))).name != raw_shard.get("file")
                or type(raw_shard.get("bytes")) is not int
                or raw_shard.get("bytes") <= 0
                or type(raw_shard.get("plies")) is not int
                or raw_shard.get("plies") <= 0
                or type(raw_shard.get("minimumGeneration")) is not int
                or raw_shard.get("minimumGeneration") < 0
                or type(raw_shard.get("maximumGeneration")) is not int
                or raw_shard.get("maximumGeneration")
                < raw_shard.get("minimumGeneration")
            ):
                raise Phase4ResumeError(
                    "replay manifest shard values are invalid"
                )
            try:
                digest = validate_sha256(
                    raw_shard.get("sha256"),
                    "replay manifest shard sha256",
                )
            except ValueError as error:
                raise Phase4ResumeError(
                    "replay manifest shard hash is invalid"
                ) from error
            manifest_signatures.append(
                (
                    raw_shard.get("sequence"),
                    raw_shard.get("gameId"),
                    raw_shard.get("file"),
                    raw_shard.get("bytes"),
                    digest,
                    raw_shard.get("plies"),
                    raw_shard.get("minimumGeneration"),
                    raw_shard.get("maximumGeneration"),
                )
            )
        if tuple(manifest_signatures) != self._current_signatures():
            raise Phase4ResumeError("replay manifest shard list differs from memory")
        next_sequence = payload.get("nextSequence")
        minimum_next = current_shards[-1].sequence + 1 if current_shards else 0
        if type(next_sequence) is not int or next_sequence < minimum_next:
            raise Phase4ResumeError("replay manifest next sequence is invalid")
        return sha256_bytes(raw)

    def _verify_shard_files(self, shards: Sequence[ReplayShardInfo]) -> None:
        for shard in shards:
            path = self.replay.root / shard.file_name
            try:
                actual = sha256_file(path)
            except (OSError, ValueError) as error:
                self._stale = True
                raise Phase4ResumeError(
                    f"cached replay shard cannot be verified: {shard.file_name}"
                ) from error
            if actual != shard.sha256:
                self._stale = True
                raise Phase4ResumeError(
                    f"cached replay shard changed: {shard.file_name}"
                )

    def assert_current(self, *, verify_shards: bool = False) -> None:
        self._assert_store_identity()
        if self._stale or self._manifest_sha256 is None:
            raise Phase4ResumeError("replay cache is stale or has not been built")
        if self._current_signatures() != self._signatures:
            self._stale = True
            raise Phase4ResumeError("replay changed after cache construction")
        try:
            manifest_sha256 = self._validated_manifest_sha256()
        except (OSError, ValueError, Phase4ResumeError) as error:
            self._stale = True
            raise Phase4ResumeError(
                "replay manifest changed or cannot be verified"
            ) from error
        if manifest_sha256 != self._manifest_sha256:
            self._stale = True
            raise Phase4ResumeError("replay manifest changed after cache construction")
        if verify_shards:
            self._verify_shard_files(self.replay.shards)

    def refresh(self) -> ReplayCacheRefresh:
        """Atomically append newly committed shards to the in-memory cache."""

        self._assert_store_identity()
        if self._stale:
            raise Phase4ResumeError("stale replay cache cannot be refreshed")
        started = time.monotonic()
        current_shards = self.replay.shards
        current_signatures = tuple(
            _replay_shard_signature(shard) for shard in current_shards
        )
        previous_game_count = len(self._blocks)
        if self._signatures and (
            len(current_signatures) < len(self._signatures)
            or current_signatures[: len(self._signatures)] != self._signatures
        ):
            self._stale = True
            raise Phase4ResumeError(
                "replay cache requires an append-only shard extension"
            )
        estimated_bytes = (
            self.replay.ply_count * REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY
        )
        if estimated_bytes > self.maximum_estimated_bytes:
            self._stale = True
            raise Phase4ResourceLimitReached(
                "replay reconstruction cache exceeds its 4 GiB estimate"
            )
        try:
            manifest_before = self._validated_manifest_sha256()
            self._verify_shard_files(current_shards[: len(self._signatures)])
            blocks = list(self._blocks)
            decoded_plies = 0
            for shard in current_shards[len(self._signatures) :]:
                game = self.replay.load_game(shard.game_id)
                states = reconstruct_game(game)
                winner = get_winner(states[-1])
                if winner not in (0, 1):
                    raise Phase4ResumeError(
                        f"replay cache game has no winner: {game.game_id}"
                    )
                positions = tuple(states[:-1])
                blocks.append(
                    _CachedReplayGame(
                        shard=shard,
                        game=game,
                        states=positions,
                        winner=winner,
                    )
                )
                decoded_plies += len(positions)
            if self.replay.shards != current_shards:
                raise Phase4ResumeError(
                    "replay shard order changed during cache construction"
                )
            try:
                manifest_after = self._validated_manifest_sha256()
            except Phase4ResumeError as error:
                raise Phase4ResumeError(
                    "replay manifest changed during cache construction"
                ) from error
            if manifest_after != manifest_before:
                raise Phase4ResumeError(
                    "replay manifest changed during cache construction"
                )
            if sum(len(block.states) for block in blocks) != self.replay.ply_count:
                raise Phase4ResumeError("replay cache coverage is incomplete")
        except Exception:
            self._stale = True
            raise

        elapsed = max(0.0, time.monotonic() - started)
        cumulative_build_seconds = self._cumulative_build_seconds + elapsed
        refresh = ReplayCacheRefresh(
            algorithm_version=REPLAY_CACHE_ALGORITHM_VERSION,
            manifest_sha256=manifest_after,
            prefix_sha256=_replay_prefix_sha256(self.replay),
            games=len(blocks),
            plies=self.replay.ply_count,
            estimated_bytes=estimated_bytes,
            reused_games=previous_game_count,
            decoded_games=len(current_shards) - previous_game_count,
            decoded_plies=decoded_plies,
            refresh_seconds=elapsed,
            cumulative_build_seconds=cumulative_build_seconds,
        )
        # A signal between publication assignments must leave the cache unusable.
        self._stale = True
        self._blocks = tuple(blocks)
        self._signatures = current_signatures
        self._manifest_sha256 = manifest_after
        self._cumulative_build_seconds = cumulative_build_seconds
        self._last_refresh = refresh
        self._stale = False
        return refresh

    def verification(self) -> ReplayVerification:
        self.assert_current()
        return ReplayVerification(
            game_count=len(self._blocks),
            ply_count=sum(len(block.states) for block in self._blocks),
            committed_bytes=self.replay.committed_bytes,
            deep=True,
        )

    def game(self, game_id: str) -> ReplayGame:
        self.assert_current()
        for block in self._blocks:
            if block.game.game_id == game_id:
                return copy.deepcopy(block.game)
        raise KeyError(game_id)

    def sample_plies(
        self,
        count: int,
        *,
        rng: random.Random | None = None,
    ) -> tuple[ReplaySample, ...]:
        self.assert_current()
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        total = sum(len(block.states) for block in self._blocks)
        if total == 0:
            raise ValueError("cannot sample an empty replay cache")
        selected_count = min(count, total)
        generator = rng or random.SystemRandom()
        selected = generator.sample(range(total), selected_count)
        cumulative: list[int] = []
        running = 0
        for block in self._blocks:
            running += len(block.states)
            cumulative.append(running)
        samples = []
        for global_index in selected:
            block_index = bisect.bisect_right(cumulative, global_index)
            previous = cumulative[block_index - 1] if block_index else 0
            ply_index = global_index - previous
            block = self._blocks[block_index]
            cached_state = block.states[ply_index]
            state = replace(
                cached_state,
                position_counts=dict(cached_state.position_counts),
            )
            ply = block.game.plies[ply_index]
            samples.append(
                ReplaySample(
                    game_id=block.game.game_id,
                    ply_index=ply_index,
                    state=state,
                    action_index=ply.action_index,
                    visit_counts=ply.visit_counts,
                    root_value=ply.root_value,
                    outcome=1.0 if state.current_player == block.winner else -1.0,
                    model_generation=ply.model_generation,
                    model_id=ply.model_id,
                    game_metadata=copy.deepcopy(block.game.metadata),
                    ply_metadata=copy.deepcopy(ply.metadata),
                )
            )
        return tuple(samples)

    def snapshot(self) -> dict[str, object]:
        if self._last_refresh is None:
            raise Phase4ResumeError("replay cache has not been built")
        self.assert_current()
        return self._last_refresh.to_payload()


Phase4ReplayCache = ReplayReconstructionCache | ReplayTensorCache


def _refresh_replay_cache(cache: Phase4ReplayCache) -> object:
    refresh = cache.refresh()
    if isinstance(cache, ReplayReconstructionCache):
        _validate_replay_cache_refresh(refresh)
    return refresh


def _replay_cache_verification(
    cache: Phase4ReplayCache,
    *,
    deep: bool = False,
) -> ReplayVerification:
    if isinstance(cache, ReplayReconstructionCache):
        verification = cache.verification()
        if deep:
            cache.assert_current(verify_shards=True)
        return verification
    verification = cache.verify(deep=deep)
    return ReplayVerification(
        game_count=verification.games,
        ply_count=verification.plies,
        committed_bytes=cache.replay.committed_bytes,
        deep=True,
    )


def _replay_cache_game(cache: Phase4ReplayCache, game_id: str) -> ReplayGame:
    if isinstance(cache, ReplayReconstructionCache):
        return cache.game(game_id)
    return cache.replay.load_game(game_id)


def _assert_replay_cache_current(
    cache: Phase4ReplayCache,
    *,
    deep: bool = False,
) -> None:
    if isinstance(cache, ReplayReconstructionCache):
        cache.assert_current(verify_shards=deep)
    else:
        cache.verify(deep=deep)


def _replay_cache_snapshot(
    cache: Phase4ReplayCache,
    *,
    deep: bool = False,
) -> dict[str, object]:
    if isinstance(cache, ReplayReconstructionCache):
        if deep:
            cache.assert_current(verify_shards=True)
        return cache.snapshot()
    return cache.snapshot(deep=deep)


def _create_phase4_replay_cache(
    *,
    replay: ReplayStore,
    run_dir: Path,
    execution_contract: Phase4ExecutionContract,
    source_sha256: str,
    encoder_sha256: str,
) -> Phase4ReplayCache:
    if execution_contract.stage == PHASE4D_STAGE:
        return ReplayReconstructionCache(replay)
    native_module = load_native_module()
    return ReplayTensorCache(
        run_dir / "replay-tensor-cache",
        replay,
        source_sha256=source_sha256,
        encoder_sha256=encoder_sha256,
        native_encoder=native_module.encode_states,
    )


def replay_health_diagnostics(
    games: Sequence[ReplayGame],
) -> dict[str, object]:
    """Summarize player orientation and value calibration for self-play."""

    wins = Counter({0: 0, 1: 0})
    decisions = Counter({0: 0, 1: 0})
    sign_correct = 0
    high_confidence_wrong = 0
    squared_error = 0.0
    total = 0
    for game in games:
        states = reconstruct_game(game)
        winner = get_winner(states[-1])
        if winner not in (0, 1):
            raise Phase4TrainingError("health diagnostic game has no winner")
        wins[winner] += 1
        for state, ply in zip(states[:-1], game.plies, strict=True):
            target = 1.0 if state.current_player == winner else -1.0
            value = float(ply.root_value)
            if not math.isfinite(value):
                raise Phase4TrainingError("health diagnostic value is not finite")
            decisions[state.current_player] += 1
            sign_correct += (value >= 0.0) == (target > 0.0)
            high_confidence_wrong += (target < 0.0 and value >= 0.5) or (
                target > 0.0 and value <= -0.5
            )
            squared_error += (value - target) ** 2
            total += 1
    if not games or total == 0:
        raise Phase4TrainingError("health diagnostics require non-empty self-play")
    player_rates = {str(player): wins[player] / len(games) for player in (0, 1)}
    return {
        "schemaVersion": 1,
        "games": len(games),
        "decisions": total,
        "winsByOriginalPlayer": {str(player): wins[player] for player in (0, 1)},
        "winRateByOriginalPlayer": player_rates,
        "startingPlayerWinRateGap": abs(player_rates["0"] - player_rates["1"]),
        "decisionsByOriginalPlayer": {
            str(player): decisions[player] for player in (0, 1)
        },
        "valueSignAccuracy": sign_correct / total,
        "valueMeanSquaredError": squared_error / total,
        "highConfidenceWrong": high_confidence_wrong,
        "highConfidenceWrongRate": high_confidence_wrong / total,
    }


def evaluate_fixed_health_holdout(
    model: PolicyValueNetwork,
    cache: ReplayTensorCache,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    use_amp: bool,
) -> dict[str, object]:
    """Evaluate every fixed holdout position with one deterministic view."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("health holdout batch_size must be positive")
    verification = cache.verify(deep=True)
    total = verification.plies
    if total <= 0:
        raise Phase4TrainingError("fixed health holdout is empty")
    plan = cache.sample_plan(
        total,
        rng=random.Random(seed),
        mirror_probability=0.5,
    )
    strata: list[tuple[str, str]] = []
    for game_id in cache.replay.game_ids:
        _, states = cache.replay.load_game_with_states(game_id)
        final_ply = len(states) - 1
        for ply_index, state in enumerate(states[:-1]):
            band = "early" if ply_index < 16 else ("middle" if ply_index < 40 else "late")
            strata.append((str(state.current_player), band))
        if final_ply <= 0:
            raise Phase4TrainingError("fixed health holdout contains an empty game")
    if len(strata) != total:
        raise Phase4TrainingError("fixed health holdout strata coverage is incomplete")

    accumulators: dict[str, dict[str, float]] = {
        "all": {"positions": 0.0, "policy": 0.0, "value": 0.0},
    }
    for side in ("0", "1"):
        accumulators[f"side:{side}"] = {
            "positions": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
    for band in ("early", "middle", "late"):
        accumulators[f"band:{band}"] = {
            "positions": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }

    previous_training = model.training
    model.eval()
    amp_enabled = device.type == "cuda" and use_amp
    try:
        with torch.no_grad():
            for offset in range(0, total, batch_size):
                indices = plan.global_indices[offset : offset + batch_size]
                mirrored = plan.mirrored[offset : offset + batch_size]
                batch_plan = ReplayTensorSamplePlan(
                    manifest_sha256=plan.manifest_sha256,
                    raw_window_sha256=plan.raw_window_sha256,
                    total_positions=plan.total_positions,
                    global_indices=indices,
                    mirrored=mirrored,
                )
                batch = cache.load_plan(batch_plan, device=device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits, predicted = model(batch["states"])
                    policy_losses = -(
                        batch["policies"] * F.log_softmax(logits, dim=1)
                    ).sum(dim=1)
                    value_losses = F.mse_loss(
                        predicted,
                        batch["values"].reshape_as(predicted),
                        reduction="none",
                    ).reshape(-1)
                if not bool(torch.isfinite(policy_losses).all().item()) or not bool(
                    torch.isfinite(value_losses).all().item()
                ):
                    raise Phase4TrainingError("fixed health holdout loss is non-finite")
                for local, global_index in enumerate(indices):
                    side, band = strata[global_index]
                    policy = float(policy_losses[local].float().item())
                    value = float(value_losses[local].float().item())
                    for key in ("all", f"side:{side}", f"band:{band}"):
                        accumulators[key]["positions"] += 1
                        accumulators[key]["policy"] += policy
                        accumulators[key]["value"] += value
    finally:
        model.train(previous_training)

    def result(key: str) -> dict[str, float | int]:
        item = accumulators[key]
        positions = int(item["positions"])
        if positions <= 0:
            return {"positions": 0, "policyLoss": 0.0, "valueLoss": 0.0, "totalLoss": 0.0}
        policy = item["policy"] / positions
        value = item["value"] / positions
        return {
            "positions": positions,
            "policyLoss": policy,
            "valueLoss": value,
            "totalLoss": policy + value,
        }

    overall = result("all")
    return {
        "schemaVersion": 1,
        "seed": seed,
        **overall,
        "bySideToMove": {side: result(f"side:{side}") for side in ("0", "1")},
        "byPlyBand": {
            band: result(f"band:{band}") for band in ("early", "middle", "late")
        },
        "cache": cache.snapshot(deep=True),
    }


def fixed_holdout_selection(
    reports: GenerationReportStore,
    *,
    generation: int,
    current: Mapping[str, object],
) -> dict[str, object]:
    """Select the strict minimum fixed-holdout loss through this generation."""

    candidates: list[tuple[float, int]] = []
    for previous in reports.generations():
        record = reports.load(previous)
        report = record.payload.get("report")
        generation_data = report.get("generationData") if isinstance(report, Mapping) else None
        holdout = (
            generation_data.get("fixedHealthHoldout")
            if isinstance(generation_data, Mapping)
            else None
        )
        loss = holdout.get("totalLoss") if isinstance(holdout, Mapping) else None
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise Phase4ResumeError("prior fixed health holdout report is invalid")
        candidates.append((float(loss), previous))
    current_loss = current.get("totalLoss")
    if not isinstance(current_loss, (int, float)) or not math.isfinite(
        float(current_loss)
    ):
        raise Phase4TrainingError("current fixed health holdout loss is invalid")
    candidates.append((float(current_loss), generation))
    best_loss, best_generation = min(candidates)
    return {
        "schemaVersion": 1,
        "criterion": "strict-minimum-fixed-health-holdout-total-loss",
        "eligible": True,
        "evaluatedGenerations": [item[1] for item in sorted(candidates, key=lambda item: item[1])],
        "bestGeneration": best_generation,
        "bestTotalLoss": best_loss,
        "currentIsBest": best_generation == generation,
    }


def pure_selfplay_selection(
    reports: GenerationReportStore,
    *,
    generation: int,
) -> dict[str, object]:
    """Advance the latest pure-self-play generation pending an external screen."""

    prior = reports.generations()
    if prior != tuple(range(generation)):
        raise Phase4ResumeError("pure self-play report history contains a gap")
    return {
        "schemaVersion": 1,
        "criterion": "latest-pure-self-play-generation-pending-external-screen",
        "eligible": True,
        "evaluatedGenerations": [*prior, generation],
        "bestGeneration": generation,
        "currentIsBest": True,
        "classicHoldoutDiagnosticOnly": True,
    }


def reconcile_best_fixed_holdout_candidate(
    run_dir: Path,
    reports: GenerationReportStore,
) -> dict[str, object] | None:
    """Validate and deterministically rebuild the mutable best-candidate index."""

    candidates: list[tuple[float, int, Mapping[str, object]]] = []
    stages: set[str] = set()
    generations = reports.generations()
    for generation in generations:
        stored = reports.load(generation).payload.get("report")
        generation_data = (
            stored.get("generationData") if isinstance(stored, Mapping) else None
        )
        holdout = (
            generation_data.get("fixedHealthHoldout")
            if isinstance(generation_data, Mapping)
            else None
        )
        training = (
            generation_data.get("training")
            if isinstance(generation_data, Mapping)
            else None
        )
        stage = training.get("learningStage") if isinstance(training, Mapping) else None
        if isinstance(stage, str):
            stages.add(stage)
        if holdout is None:
            if candidates:
                raise Phase4ResumeError(
                    "fixed health holdout report history is incomplete"
                )
            continue
        loss = holdout.get("totalLoss") if isinstance(holdout, Mapping) else None
        checkpoint = stored.get("checkpoint") if isinstance(stored, Mapping) else None
        if (
            not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or not isinstance(checkpoint, Mapping)
        ):
            raise Phase4ResumeError("fixed health holdout report is invalid")
        candidates.append((float(loss), generation, checkpoint))
    if not candidates:
        return None
    if len(candidates) != len(generations):
        raise Phase4ResumeError(
            "fixed health holdout cannot begin after untracked generations"
        )

    if len(stages) > 1:
        raise Phase4ResumeError("one run cannot mix staged-learning modes")
    pure_selfplay = stages == {"pure-self-play"}
    if pure_selfplay:
        best_loss, best_generation, checkpoint = candidates[-1]
    else:
        best_loss, best_generation, checkpoint = min(candidates)
    candidate = checkpoint.get("candidateModel")
    checkpoint_name = checkpoint.get("path")
    candidate_name = candidate.get("path") if isinstance(candidate, Mapping) else None
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
        or not isinstance(candidate_name, str)
        or Path(candidate_name).name != candidate_name
        or not isinstance(candidate, Mapping)
    ):
        raise Phase4ResumeError("best fixed holdout checkpoint paths are invalid")
    try:
        checkpoint_sha256 = validate_sha256(
            checkpoint.get("sha256"), "best checkpoint sha256"
        )
        candidate_sha256 = validate_sha256(
            candidate.get("sha256"), "best candidate model sha256"
        )
        model_state_sha256 = validate_sha256(
            candidate.get("modelStateSha256"), "best model state sha256"
        )
    except ValueError as error:
        raise Phase4ResumeError("best fixed holdout checkpoint hashes are invalid") from error
    step = checkpoint.get("step")
    if type(step) is not int or step < 0:
        raise Phase4ResumeError("best fixed holdout checkpoint step is invalid")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    candidate_path = run_dir / "checkpoints" / candidate_name
    for path, expected_sha256 in (
        (checkpoint_path, checkpoint_sha256),
        (candidate_path, candidate_sha256),
    ):
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
            raise Phase4ResumeError(
                f"best fixed holdout artifact is missing or corrupt: {path.name}"
            )

    selection = (
        {
            "schemaVersion": 1,
            "criterion": "latest-pure-self-play-generation-pending-external-screen",
            "eligible": True,
            "evaluatedGenerations": [item[1] for item in candidates],
            "bestGeneration": best_generation,
            "currentIsBest": True,
            "classicHoldoutDiagnosticOnly": True,
        }
        if pure_selfplay
        else {
            "schemaVersion": 1,
            "criterion": "strict-minimum-fixed-health-holdout-total-loss",
            "eligible": True,
            "evaluatedGenerations": [item[1] for item in candidates],
            "bestGeneration": best_generation,
            "bestTotalLoss": best_loss,
            "currentIsBest": best_generation == generations[-1],
        }
    )
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "phase4-best-fixed-health-candidate",
        "generation": best_generation,
        "step": step,
        "checkpointPath": checkpoint_name,
        "checkpointSha256": checkpoint_sha256,
        "candidateModelPath": candidate_name,
        "candidateModelSha256": candidate_sha256,
        "modelStateSha256": model_state_sha256,
        "selection": selection,
    }
    expected_bytes = canonical_json_bytes(payload)
    marker_path = run_dir / "best-candidate.json"
    try:
        current_bytes = marker_path.read_bytes()
    except OSError:
        current_bytes = None
    if current_bytes != expected_bytes:
        atomic_replace_bytes(marker_path, expected_bytes)
    return payload


@dataclass(slots=True)
class ReplayInputPerformanceGate:
    """Fail before an optimizer update when cached input becomes too slow."""

    count: int = 0
    seconds_sum: float = 0.0
    maximum_seconds: float = 0.0
    initial_completed: bool = False

    @property
    def mean_seconds(self) -> float:
        return self.seconds_sum / self.count if self.count else 0.0

    def observe(self, seconds: float) -> bool:
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0
        ):
            raise Phase4TrainingError("replay input timing must be finite")
        value = float(seconds)
        self.count += 1
        self.seconds_sum += value
        self.maximum_seconds = max(self.maximum_seconds, value)
        initial = self.count <= REPLAY_INPUT_GATE_BATCHES
        maximum = (
            REPLAY_INPUT_GATE_MAX_BATCH_SECONDS
            if initial
            else REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS
        )
        if value > maximum:
            raise Phase4ResourceLimitReached(
                f"cached replay input batch exceeded {maximum:.1f}s"
            )
        if (
            self.count >= REPLAY_INPUT_GATE_BATCHES
            and self.mean_seconds > REPLAY_INPUT_GATE_MAX_MEAN_SECONDS
        ):
            raise Phase4ResourceLimitReached(
                "cached replay input mean exceeded 2.0s"
            )
        completed_now = (
            self.count == REPLAY_INPUT_GATE_BATCHES and not self.initial_completed
        )
        if completed_now:
            self.initial_completed = True
        return completed_now

    def to_payload(self) -> dict[str, object]:
        return {
            "batches": self.count,
            "secondsMean": self.mean_seconds,
            "secondsMaximum": self.maximum_seconds,
            "initialGateBatches": REPLAY_INPUT_GATE_BATCHES,
            "maximumMeanSeconds": REPLAY_INPUT_GATE_MAX_MEAN_SECONDS,
            "maximumInitialBatchSeconds": REPLAY_INPUT_GATE_MAX_BATCH_SECONDS,
            "maximumLiveBatchSeconds": REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS,
            "initialGateCompleted": self.initial_completed,
        }


def _validate_replay_cache_refresh(
    refresh: ReplayCacheRefresh,
    *,
    maximum_build_seconds: float | None = None,
) -> float | None:
    """Apply the fixed-fixture gate or the shape-aware live-workload envelope."""

    if (
        isinstance(refresh.refresh_seconds, bool)
        or not isinstance(refresh.refresh_seconds, (int, float))
        or not math.isfinite(float(refresh.refresh_seconds))
        or refresh.refresh_seconds < 0
    ):
        raise Phase4TrainingError(
            "replay cache refresh seconds must be finite and non-negative"
        )
    if maximum_build_seconds is not None:
        if (
            isinstance(maximum_build_seconds, bool)
            or not isinstance(maximum_build_seconds, (int, float))
            or not math.isfinite(float(maximum_build_seconds))
            or maximum_build_seconds <= 0
        ):
            raise ValueError("maximum_build_seconds must be finite and positive")
        if refresh.refresh_seconds > maximum_build_seconds:
            raise Phase4ResourceLimitReached(
                f"replay cache build exceeded {maximum_build_seconds:.1f}s"
            )
        rate = refresh.decoded_positions_per_second
        if (
            refresh.decoded_plies >= 1024
            and rate is not None
            and rate < REPLAY_INPUT_GATE_MIN_BUILD_POSITIONS_PER_SECOND
        ):
            raise Phase4ResourceLimitReached(
                "replay cache build throughput fell below 60 positions/s"
            )
        return float(maximum_build_seconds)

    allowed_seconds = _replay_cache_build_envelope_seconds(refresh)
    if (
        allowed_seconds is not None
        and refresh.refresh_seconds > allowed_seconds
    ):
        raise Phase4ResourceLimitReached(
            "replay cache build exceeded the gate-v2 workload envelope: "
            f"{refresh.refresh_seconds:.3f}s > {allowed_seconds:.3f}s for "
            f"{refresh.decoded_games} decoded games and "
            f"{refresh.decoded_plies} decoded plies"
        )
    return allowed_seconds


def _replay_cache_build_envelope_seconds(
    refresh: ReplayCacheRefresh,
) -> float | None:
    """Scale the 64-game fixture budget on both decoded workload axes."""

    counts = (
        (refresh.games, "games"),
        (refresh.plies, "plies"),
        (refresh.reused_games, "reused_games"),
        (refresh.decoded_games, "decoded_games"),
        (refresh.decoded_plies, "decoded_plies"),
    )
    for value, label in counts:
        if type(value) is not int or value < 0:
            raise Phase4TrainingError(
                f"replay cache {label} must be a non-negative integer"
            )
    if (
        refresh.reused_games + refresh.decoded_games != refresh.games
        or refresh.decoded_plies > refresh.plies
    ):
        raise Phase4TrainingError(
            "replay cache decoded workload is inconsistent with replay totals"
        )
    if refresh.decoded_games == 0 and refresh.decoded_plies == 0:
        return None
    if refresh.decoded_games == 0 or refresh.decoded_plies == 0:
        raise Phase4TrainingError(
            "replay cache decoded games and plies must both be non-zero"
        )
    return REPLAY_INPUT_GATE_MAX_BUILD_SECONDS * max(
        refresh.decoded_games / REPLAY_INPUT_GATE_FIXTURE_GAMES,
        refresh.decoded_plies / REPLAY_INPUT_GATE_FIXTURE_PLIES,
    )


def _load_replay_store_from_manifest(root: Path) -> ReplayStore:
    manifest_path = root / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4TrainingError("replay input gate manifest is unreadable") from error
    if not isinstance(payload, Mapping):
        raise Phase4TrainingError("replay input gate manifest must be an object")
    capacity = payload.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4TrainingError("replay input gate capacity is missing")
    values = []
    for key in ("games", "plies", "bytes"):
        value = capacity.get(key)
        if type(value) is not int or value <= 0:
            raise Phase4TrainingError(
                f"replay input gate capacity.{key} is invalid"
            )
        values.append(value)
    return ReplayStore(
        root,
        max_games=values[0],
        max_plies=values[1],
        max_bytes=values[2],
    )


def _verify_replay_input_gate_fixture(root: Path) -> None:
    manifest_path = root / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4TrainingError(
            "frozen replay input gate fixture is unreadable"
        ) from error
    if (
        sha256_bytes(raw) != REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256
        or not isinstance(payload, Mapping)
        or canonical_json_bytes(payload) != raw
    ):
        raise Phase4TrainingError(
            "frozen replay input gate fixture manifest changed"
        )
    shards = payload.get("shards")
    totals = payload.get("totals")
    if (
        not isinstance(shards, list)
        or len(shards) != REPLAY_INPUT_GATE_FIXTURE_GAMES
        or not isinstance(totals, Mapping)
        or totals.get("games") != REPLAY_INPUT_GATE_FIXTURE_GAMES
        or totals.get("plies") != REPLAY_INPUT_GATE_FIXTURE_PLIES
    ):
        raise Phase4TrainingError(
            "frozen replay input gate fixture workload changed"
        )
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise Phase4TrainingError(
                "frozen replay input gate shard metadata is invalid"
            )
        file_name = shard.get("file")
        byte_size = shard.get("bytes")
        try:
            expected_sha256 = validate_sha256(
                shard.get("sha256"),
                "frozen replay input gate shard sha256",
            )
        except ValueError as error:
            raise Phase4TrainingError(
                "frozen replay input gate shard hash is invalid"
            ) from error
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or type(byte_size) is not int
            or byte_size <= 0
        ):
            raise Phase4TrainingError(
                "frozen replay input gate shard identity is invalid"
            )
        path = root / file_name
        try:
            actual_size = path.stat().st_size
            actual_sha256 = sha256_file(path)
        except OSError as error:
            raise Phase4TrainingError(
                f"frozen replay input gate shard is missing: {file_name}"
            ) from error
        if actual_size != byte_size or actual_sha256 != expected_sha256:
            raise Phase4TrainingError(
                f"frozen replay input gate shard changed: {file_name}"
            )


def _replay_input_gate_runtime(workspace: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "workspace": str(workspace.resolve()),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "python": sys.version,
        "torch": torch.__version__,
    }
    return {
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
        "payload": payload,
    }


def _replay_input_gate_report_path(
    control_root: Path,
    source_sha256: str,
    native_backend_sha256: str,
    runtime_sha256: str,
) -> Path:
    validate_sha256(source_sha256, "replay input gate source sha256")
    validate_sha256(native_backend_sha256, "replay input gate native sha256")
    validate_sha256(runtime_sha256, "replay input gate runtime sha256")
    return (
        control_root
        / REPLAY_INPUT_GATE_REPORT_RELATIVE
        / (
            f"report-{source_sha256}-{native_backend_sha256}-"
            f"{runtime_sha256}.json"
        )
    )


def _load_replay_input_gate_report(
    *,
    workspace: Path,
    control_root: Path,
    source_sha256: str,
    native_backend_sha256: str,
) -> dict[str, object]:
    runtime = _replay_input_gate_runtime(workspace)
    path = _replay_input_gate_report_path(
        control_root,
        source_sha256,
        native_backend_sha256,
        str(runtime["sha256"]),
    )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4TrainingError(
            "current-source replay input gate report is missing or unreadable"
        ) from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise Phase4TrainingError(
            "replay input gate report is not canonical JSON"
        )
    expected_top_keys = {
        "schemaVersion",
        "kind",
        "status",
        "sourceSha256",
        "nativeBackendSha256",
        "runtime",
        "fixturePath",
        "fixtureManifestSha256",
        "fixtureGames",
        "fixturePlies",
        "cache",
        "parity",
        "inputTiming",
        "gates",
        "reportCoreSha256",
    }
    if set(payload) != expected_top_keys:
        raise Phase4TrainingError(
            "replay input gate report schema changed"
        )
    try:
        stored_core_sha256 = validate_sha256(
            payload.get("reportCoreSha256"),
            "replay input gate report core sha256",
        )
    except ValueError as error:
        raise Phase4TrainingError(
            "replay input gate report core hash is invalid"
        ) from error
    core = {
        key: value
        for key, value in payload.items()
        if key != "reportCoreSha256"
    }
    if sha256_bytes(canonical_json_bytes(core)) != stored_core_sha256:
        raise Phase4TrainingError(
            "replay input gate report core hash differs"
        )
    fixture_path = workspace / REPLAY_INPUT_GATE_FIXTURE_RELATIVE
    _verify_replay_input_gate_fixture(fixture_path)
    if (
        payload.get("schemaVersion") != 1
        or payload.get("kind") != "phase4-replay-input-gate"
        or payload.get("status") != "completed"
        or payload.get("sourceSha256") != source_sha256
        or payload.get("nativeBackendSha256") != native_backend_sha256
        or payload.get("runtime") != runtime
        or payload.get("fixturePath")
        != str(REPLAY_INPUT_GATE_FIXTURE_RELATIVE)
        or payload.get("fixtureManifestSha256")
        != REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256
    ):
        raise Phase4TrainingError(
            "replay input gate report provenance is stale"
        )
    gates = payload.get("gates")
    expected_gates = {
        "fixture",
        "cacheBuildSeconds",
        "cacheBuildThroughput",
        "sampleParity",
        "rngParity",
        "tensorParity",
        "batchMean",
        "batchMaximum",
    }
    if (
        not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or any(value is not True for value in gates.values())
    ):
        raise Phase4TrainingError("replay input gate report did not pass every gate")
    cache = payload.get("cache")
    timing = payload.get("inputTiming")
    parity = payload.get("parity")
    expected_cache_keys = {
        "algorithmVersion",
        "manifestSha256",
        "prefixSha256",
        "games",
        "plies",
        "estimatedBytes",
        "reusedGames",
        "decodedGames",
        "decodedPlies",
        "refreshSeconds",
        "cumulativeBuildSeconds",
        "decodedPositionsPerSecond",
    }
    expected_timing_keys = {
        "batches",
        "secondsMean",
        "secondsMaximum",
        "initialGateBatches",
        "maximumMeanSeconds",
        "maximumInitialBatchSeconds",
        "maximumLiveBatchSeconds",
        "initialGateCompleted",
        "batchSeconds",
    }
    expected_parity_keys = {
        "sample",
        "rng",
        "tensors",
        "legacySeconds",
        "cachedSeconds",
    }
    if (
        payload.get("fixtureGames") != REPLAY_INPUT_GATE_FIXTURE_GAMES
        or payload.get("fixturePlies") != REPLAY_INPUT_GATE_FIXTURE_PLIES
        or not isinstance(cache, Mapping)
        or set(cache) != expected_cache_keys
        or cache.get("algorithmVersion") != REPLAY_CACHE_ALGORITHM_VERSION
        or cache.get("manifestSha256")
        != REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256
        or cache.get("games") != REPLAY_INPUT_GATE_FIXTURE_GAMES
        or cache.get("plies") != REPLAY_INPUT_GATE_FIXTURE_PLIES
        or cache.get("estimatedBytes")
        != (
            REPLAY_INPUT_GATE_FIXTURE_PLIES
            * REPLAY_CACHE_ESTIMATED_BYTES_PER_PLY
        )
        or cache.get("reusedGames") != 0
        or cache.get("decodedGames") != REPLAY_INPUT_GATE_FIXTURE_GAMES
        or cache.get("decodedPlies") != REPLAY_INPUT_GATE_FIXTURE_PLIES
        or not isinstance(timing, Mapping)
        or set(timing) != expected_timing_keys
        or timing.get("batches") != REPLAY_INPUT_GATE_BATCHES
        or timing.get("initialGateBatches") != REPLAY_INPUT_GATE_BATCHES
        or timing.get("maximumMeanSeconds")
        != REPLAY_INPUT_GATE_MAX_MEAN_SECONDS
        or timing.get("maximumInitialBatchSeconds")
        != REPLAY_INPUT_GATE_MAX_BATCH_SECONDS
        or timing.get("maximumLiveBatchSeconds")
        != REPLAY_INPUT_LIVE_MAX_BATCH_SECONDS
        or timing.get("initialGateCompleted") is not True
        or not isinstance(parity, Mapping)
        or set(parity) != expected_parity_keys
        or parity.get("sample") is not True
        or parity.get("rng") is not True
        or parity.get("tensors") is not True
    ):
        raise Phase4TrainingError("replay input gate report workload is invalid")
    try:
        validate_sha256(
            cache.get("prefixSha256"),
            "replay input gate cache prefix sha256",
        )
    except ValueError as error:
        raise Phase4TrainingError(
            "replay input gate cache prefix hash is invalid"
        ) from error
    batch_seconds = timing.get("batchSeconds")
    if (
        not isinstance(batch_seconds, list)
        or len(batch_seconds) != REPLAY_INPUT_GATE_BATCHES
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= REPLAY_INPUT_GATE_MAX_BATCH_SECONDS
            for value in batch_seconds
        )
    ):
        raise Phase4TrainingError(
            "replay input gate batch timings are invalid"
        )
    numeric_limits = (
        (
            cache.get("refreshSeconds"),
            REPLAY_INPUT_GATE_MAX_BUILD_SECONDS,
            "cache build seconds",
        ),
        (
            cache.get("cumulativeBuildSeconds"),
            REPLAY_INPUT_GATE_MAX_BUILD_SECONDS,
            "cumulative cache build seconds",
        ),
        (
            timing.get("secondsMean"),
            REPLAY_INPUT_GATE_MAX_MEAN_SECONDS,
            "batch mean seconds",
        ),
        (
            timing.get("secondsMaximum"),
            REPLAY_INPUT_GATE_MAX_BATCH_SECONDS,
            "batch maximum seconds",
        ),
    )
    for value, maximum, label in numeric_limits:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= maximum
        ):
            raise Phase4TrainingError(
                f"replay input gate {label} is outside its threshold"
            )
    if (
        not math.isclose(
            float(timing["secondsMean"]),
            sum(float(value) for value in batch_seconds) / len(batch_seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(timing["secondsMaximum"]),
            max(float(value) for value in batch_seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(cache["refreshSeconds"]),
            float(cache["cumulativeBuildSeconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise Phase4TrainingError(
            "replay input gate timing summary is inconsistent"
        )
    for key in ("legacySeconds", "cachedSeconds"):
        value = parity.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise Phase4TrainingError(
                "replay input gate parity timing is invalid"
            )
    build_rate = cache.get("decodedPositionsPerSecond")
    refresh_seconds = float(cache["refreshSeconds"])
    expected_build_rate = (
        REPLAY_INPUT_GATE_FIXTURE_PLIES / max(refresh_seconds, 1e-12)
    )
    if (
        isinstance(build_rate, bool)
        or not isinstance(build_rate, (int, float))
        or not math.isfinite(float(build_rate))
        or float(build_rate)
        < REPLAY_INPUT_GATE_MIN_BUILD_POSITIONS_PER_SECOND
        or not math.isclose(
            float(build_rate),
            expected_build_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise Phase4TrainingError(
            "replay input gate cache build rate is invalid"
        )
    return {
        "path": str(path.relative_to(workspace)),
        "sha256": sha256_file(path),
        "reportCoreSha256": stored_core_sha256,
        "sourceSha256": source_sha256,
        "nativeBackendSha256": native_backend_sha256,
        "runtimeSha256": runtime["sha256"],
        "fixtureManifestSha256": REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256,
    }


def _run_phase4_replay_input_gate_locked(
    *,
    workspace_root: str | Path,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Build and persist current-source CPU replay-input evidence."""

    workspace = Path(workspace_root).resolve()
    if artifact_limit_bytes not in {
        DEFAULT_ARTIFACT_LIMIT_BYTES,
        MAXIMUM_ARTIFACT_LIMIT_BYTES,
    }:
        raise ValueError(
            "replay input gate artifact limit must be the frozen 8 or 12 GiB cap"
        )
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    fixture_root = (workspace / REPLAY_INPUT_GATE_FIXTURE_RELATIVE).resolve()
    if not fixture_root.is_dir():
        raise Phase4TrainingError(
            "frozen replay input gate fixture is missing"
        )
    _verify_replay_input_gate_fixture(fixture_root)
    source = fingerprint_sources(workspace)
    native_backend_sha256 = loaded_native_backend_hash()
    runtime = _replay_input_gate_runtime(workspace)
    report_path = _replay_input_gate_report_path(
        control_root,
        source.sha256,
        native_backend_sha256,
        str(runtime["sha256"]),
    )
    if report_path.exists():
        evidence = _load_replay_input_gate_report(
            workspace=workspace,
            control_root=control_root,
            source_sha256=source.sha256,
            native_backend_sha256=native_backend_sha256,
        )
        if (
            fingerprint_sources(workspace).sha256 != source.sha256
            or loaded_native_backend_hash() != native_backend_sha256
            or _replay_input_gate_runtime(workspace) != runtime
        ):
            raise Phase4TrainingError(
                "replay input gate provenance changed while reusing evidence"
            )
        return {
            "schemaVersion": 1,
            "status": "completed",
            "reused": True,
            "evidence": evidence,
        }

    _publish_progress(
        progress_callback,
        stage="replay-input-gate",
        status="started",
        fixture=str(REPLAY_INPUT_GATE_FIXTURE_RELATIVE),
    )
    with tempfile.TemporaryDirectory(prefix="quoridor-phase4-input-gate-") as directory:
        copied_root = Path(directory) / "replay"
        shutil.copytree(fixture_root, copied_root)
        replay = _load_replay_store_from_manifest(copied_root)
        if (
            replay.game_count != REPLAY_INPUT_GATE_FIXTURE_GAMES
            or replay.ply_count != REPLAY_INPUT_GATE_FIXTURE_PLIES
        ):
            raise Phase4TrainingError(
                "replay input gate fixture workload differs"
            )
        cache = ReplayReconstructionCache(replay)
        refresh = cache.refresh()
        _validate_replay_cache_refresh(
            refresh,
            maximum_build_seconds=REPLAY_INPUT_GATE_MAX_BUILD_SECONDS,
        )
        _publish_progress(
            progress_callback,
            stage="replay-input-gate",
            status="cache-built",
            **refresh.to_payload(),
        )

        native_module = load_native_module()
        encoder = ReplayBatchEncoder(
            input_channels=PHASE4_INPUT_CHANNELS,
            device="cpu",
            native_encoder=native_module.encode_states,
        )
        legacy_rng = random.Random(REPLAY_INPUT_GATE_SEED)
        cached_rng = random.Random(REPLAY_INPUT_GATE_SEED)
        legacy_started = time.monotonic()
        legacy_samples = replay.sample_plies(
            REPLAY_INPUT_GATE_BATCH_SIZE,
            rng=legacy_rng,
        )
        legacy_batch = encoder.encode(legacy_samples, rng=legacy_rng)
        legacy_seconds = max(0.0, time.monotonic() - legacy_started)
        cached_started = time.monotonic()
        cached_samples = cache.sample_plies(
            REPLAY_INPUT_GATE_BATCH_SIZE,
            rng=cached_rng,
        )
        cached_batch = encoder.encode(cached_samples, rng=cached_rng)
        cached_parity_seconds = max(0.0, time.monotonic() - cached_started)
        sample_parity = cached_samples == legacy_samples
        rng_parity = cached_rng.getstate() == legacy_rng.getstate()
        tensor_parity = all(
            torch.equal(cached_batch[key], legacy_batch[key])
            for key in ("states", "policies", "values")
        )
        if not sample_parity or not rng_parity or not tensor_parity:
            raise Phase4TrainingError(
                "cached replay input differs from the legacy production path"
            )
        _publish_progress(
            progress_callback,
            stage="replay-input-gate",
            status="parity-verified",
            legacySeconds=legacy_seconds,
            cachedSeconds=cached_parity_seconds,
        )

        gate = ReplayInputPerformanceGate()
        batch_seconds = []
        for local_step in range(REPLAY_INPUT_GATE_BATCHES):
            sample_seed = (
                20260714 + 10_000_019 + local_step * 97_409
            ) % (2**63)
            sample_rng = random.Random(sample_seed)
            batch_started = time.monotonic()
            samples = cache.sample_plies(
                REPLAY_INPUT_GATE_BATCH_SIZE,
                rng=sample_rng,
            )
            encoder.encode(samples, rng=sample_rng)
            elapsed = max(0.0, time.monotonic() - batch_started)
            batch_seconds.append(elapsed)
            gate.observe(elapsed)
        if not gate.initial_completed:
            raise Phase4TrainingError(
                "replay input performance gate did not complete"
            )

    _verify_replay_input_gate_fixture(fixture_root)
    if (
        fingerprint_sources(workspace).sha256 != source.sha256
        or loaded_native_backend_hash() != native_backend_sha256
        or _replay_input_gate_runtime(workspace) != runtime
    ):
        raise Phase4TrainingError(
            "replay input gate provenance changed during validation"
        )
    core: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "phase4-replay-input-gate",
        "status": "completed",
        "sourceSha256": source.sha256,
        "nativeBackendSha256": native_backend_sha256,
        "runtime": runtime,
        "fixturePath": str(REPLAY_INPUT_GATE_FIXTURE_RELATIVE),
        "fixtureManifestSha256": REPLAY_INPUT_GATE_FIXTURE_MANIFEST_SHA256,
        "fixtureGames": REPLAY_INPUT_GATE_FIXTURE_GAMES,
        "fixturePlies": REPLAY_INPUT_GATE_FIXTURE_PLIES,
        "cache": refresh.to_payload(),
        "parity": {
            "sample": sample_parity,
            "rng": rng_parity,
            "tensors": tensor_parity,
            "legacySeconds": legacy_seconds,
            "cachedSeconds": cached_parity_seconds,
        },
        "inputTiming": {
            **gate.to_payload(),
            "batchSeconds": batch_seconds,
        },
        "gates": {
            "fixture": True,
            "cacheBuildSeconds": (
                refresh.refresh_seconds
                <= REPLAY_INPUT_GATE_MAX_BUILD_SECONDS
            ),
            "cacheBuildThroughput": (
                refresh.decoded_positions_per_second is not None
                and refresh.decoded_positions_per_second
                >= REPLAY_INPUT_GATE_MIN_BUILD_POSITIONS_PER_SECOND
            ),
            "sampleParity": sample_parity,
            "rngParity": rng_parity,
            "tensorParity": tensor_parity,
            "batchMean": (
                gate.mean_seconds <= REPLAY_INPUT_GATE_MAX_MEAN_SECONDS
            ),
            "batchMaximum": (
                gate.maximum_seconds <= REPLAY_INPUT_GATE_MAX_BATCH_SECONDS
            ),
        },
    }
    if any(value is not True for value in core["gates"].values()):
        raise Phase4ResourceLimitReached(
            "replay input gate did not pass every frozen threshold"
        )
    report = {
        **core,
        "reportCoreSha256": sha256_bytes(canonical_json_bytes(core)),
    }
    encoded = canonical_json_bytes(report)
    _static_write_preflight(
        artifact_root=control_root,
        artifact_limit_bytes=artifact_limit_bytes,
        reserve_bytes=len(encoded) + _REPORT_WRITE_RESERVE_BYTES,
    )
    atomic_create_bytes(report_path, encoded)
    evidence = _load_replay_input_gate_report(
        workspace=workspace,
        control_root=control_root,
        source_sha256=source.sha256,
        native_backend_sha256=native_backend_sha256,
    )
    _publish_progress(
        progress_callback,
        stage="replay-input-gate",
        status="completed",
        reportPath=evidence["path"],
        reportSha256=evidence["sha256"],
        dataBatchSecondsMean=gate.mean_seconds,
        dataBatchSecondsMaximum=gate.maximum_seconds,
    )
    return {
        "schemaVersion": 1,
        "status": "completed",
        "reused": False,
        "evidence": evidence,
        "cache": refresh.to_payload(),
        "inputTiming": gate.to_payload(),
    }


def run_phase4_replay_input_gate(
    *,
    workspace_root: str | Path,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run the CPU-only gate under the shared Phase 4 single-writer lock."""

    workspace = Path(workspace_root).resolve()
    if artifact_limit_bytes not in {
        DEFAULT_ARTIFACT_LIMIT_BYTES,
        MAXIMUM_ARTIFACT_LIMIT_BYTES,
    }:
        raise ValueError(
            "replay input gate artifact limit must be the frozen 8 or 12 GiB cap"
        )
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    _static_write_preflight(
        artifact_root=control_root,
        artifact_limit_bytes=artifact_limit_bytes,
        reserve_bytes=_REPORT_WRITE_RESERVE_BYTES,
    )
    control_root.mkdir(parents=True, exist_ok=True)
    run_id = f"phase4-replay-input-gate-{os.getpid()}-{time.time_ns()}"
    with RunLock(control_root / GPU_LOCK_NAME, run_id=run_id):
        return _run_phase4_replay_input_gate_locked(
            workspace_root=workspace,
            artifact_limit_bytes=artifact_limit_bytes,
            progress_callback=progress_callback,
        )


def _selfplay_invocation(
    *,
    generation: int,
    model_id: str,
    scheduler_seed: int,
    scheduled_game_indices: Sequence[int],
    configured_actors: int,
    simulations: int,
    temperature_moves: int,
) -> dict[str, object]:
    indices = tuple(scheduled_game_indices)
    if (
        not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or tuple(sorted(set(indices))) != indices
    ):
        raise ValueError("scheduled game indices must be sorted unique integers")
    if type(configured_actors) is not int or configured_actors <= 0:
        raise ValueError("configured actors must be a positive integer")
    core: dict[str, object] = {
        "schemaVersion": 1,
        "generation": generation,
        "producerModelId": model_id,
        "schedulerSeed": scheduler_seed,
        "searchSeed": scheduled_search_seed(scheduler_seed),
        "scheduledGameIndices": list(indices),
        "configuredActors": configured_actors,
        "actualActors": min(configured_actors, len(indices)),
        "simulations": simulations,
        "temperatureMoves": temperature_moves,
    }
    return {
        **core,
        "invocationId": sha256_bytes(canonical_json_bytes(core)),
    }


def _validate_selfplay_invocation(
    value: object,
    *,
    generation: int,
    model_id: str,
    game_index: int,
    configured_actors: int,
    simulations: int,
    temperature_moves: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4ResumeError("self-play invocation metadata is missing")
    required = {
        "schemaVersion",
        "generation",
        "producerModelId",
        "schedulerSeed",
        "searchSeed",
        "scheduledGameIndices",
        "configuredActors",
        "actualActors",
        "simulations",
        "temperatureMoves",
        "invocationId",
    }
    if set(value) != required:
        raise Phase4ResumeError("self-play invocation metadata fields are invalid")
    indices = value.get("scheduledGameIndices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or sorted(set(indices)) != indices
        or game_index not in indices
    ):
        raise Phase4ResumeError("self-play invocation game schedule is invalid")
    scheduler_seed = value.get("schedulerSeed")
    if type(scheduler_seed) is not int:
        raise Phase4ResumeError("self-play invocation scheduler seed is invalid")
    expected = _selfplay_invocation(
        generation=generation,
        model_id=model_id,
        scheduler_seed=scheduler_seed,
        scheduled_game_indices=indices,
        configured_actors=configured_actors,
        simulations=simulations,
        temperature_moves=temperature_moves,
    )
    if dict(value) != expected:
        raise Phase4ResumeError("self-play invocation metadata hash or workload changed")
    return expected


def validate_selfplay_invocation(
    value: object,
    *,
    generation: int,
    model_id: str,
    game_index: int,
    configured_actors: int,
    simulations: int,
    temperature_moves: int,
) -> dict[str, object]:
    """Validate durable invocation metadata against the training protocol."""

    return _validate_selfplay_invocation(
        value,
        generation=generation,
        model_id=model_id,
        game_index=game_index,
        configured_actors=configured_actors,
        simulations=simulations,
        temperature_moves=temperature_moves,
    )


def replay_game_from_selfplay(
    game: SelfPlayGame,
    *,
    game_id: str,
    generation: int,
    model_id: str,
    scheduler_seed: int,
    game_index: int,
    resumed_subset: bool,
    invocation: Mapping[str, object],
) -> ReplayGame:
    if len(game.raw_plies) != game.move_count:
        raise ValueError("self-play game does not contain a complete raw trajectory")
    if invocation.get("schedulerSeed") != scheduler_seed:
        raise ValueError("self-play invocation scheduler seed does not match")
    scheduled_indices = invocation.get("scheduledGameIndices")
    if not isinstance(scheduled_indices, list) or game_index not in scheduled_indices:
        raise ValueError("self-play game is absent from its invocation schedule")
    return ReplayGame(
        game_id=game_id,
        initial_state=game.initial_state,
        plies=tuple(
            ReplayPly(
                action_index=ply.action_index,
                visit_counts=ply.visit_counts,
                root_value=ply.root_value,
                model_generation=generation,
                model_id=model_id,
                metadata={"requestedSimulations": None},
            )
            for ply in game.raw_plies
        ),
        metadata={
            "pipeline": "phase4",
            "generation": generation,
            "modelId": model_id,
            "schedulerSeed": scheduler_seed,
            "gameIndex": game_index,
            "resumedSubset": resumed_subset,
            "invocationGameIndex": scheduled_indices.index(game_index),
            "invocation": dict(invocation),
            "winner": game.winner,
            "terminationReason": game.termination_reason,
            "moves": game.move_count,
        },
    )


def load_phase4_warm_artifact(
    *,
    workspace_root: str | Path,
    config: ExperimentConfig,
) -> tuple[Mapping[str, torch.Tensor], dict[str, object]]:
    """Validate a model-only warm artifact before any pickle deserialization."""

    initialization = config.initialization
    if initialization.mode != "model" or initialization.artifact_path is None:
        raise ValueError("warm artifact loading requires initialization.mode=model")
    artifact_path = initialization.artifact_path
    if not artifact_path.is_absolute():
        artifact_path = Path(workspace_root) / artifact_path
    if artifact_path.is_symlink():
        raise FileNotFoundError(f"warm artifact must not be a symlink: {artifact_path}")
    artifact_path = artifact_path.resolve()
    try:
        artifact_path.relative_to(Path(workspace_root).resolve())
    except ValueError as error:
        raise ValueError("warm artifact must be inside the workspace") from error
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise FileNotFoundError(
            f"warm artifact is missing or is a symlink: {artifact_path}"
        )
    artifact_sha256 = sha256_file(artifact_path)
    if artifact_sha256 != initialization.artifact_sha256:
        raise Phase4ResumeError("warm artifact SHA-256 does not match config")
    try:
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise Phase4ResumeError("warm artifact cannot be deserialized safely") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schemaVersion",
        "format",
        "network",
        "baseModelStateSha256",
        "modelStateSha256",
        "provenance",
        "modelState",
    }:
        raise Phase4ResumeError("warm artifact schema fields are invalid")
    if (
        payload.get("schemaVersion") != PHASE4_WARM_ARTIFACT_SCHEMA_VERSION
        or payload.get("format") != "quoridor-phase4-model-only"
        or payload.get("network") != _network_config(config).to_dict()
        or payload.get("baseModelStateSha256")
        != initialization.expected_base_model_sha256
    ):
        raise Phase4ResumeError("warm artifact schema/network does not match config")
    provenance = payload.get("provenance")
    expected_provenance = {
        "teacherDatasetSha256": initialization.teacher_dataset_sha256,
        "pretrainingReportSha256": initialization.pretraining_report_sha256,
    }
    if not isinstance(provenance, Mapping) or dict(provenance) != expected_provenance:
        raise Phase4ResumeError("warm artifact teacher/report provenance mismatch")
    state = payload.get("modelState")
    if not isinstance(state, Mapping):
        raise Phase4ResumeError("warm artifact modelState must be a mapping")
    try:
        model_state_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4ResumeError("warm artifact tensor state is invalid") from error
    if payload.get("modelStateSha256") != model_state_sha256:
        raise Phase4ResumeError("warm artifact tensor-state hash mismatch")
    return state, {
        "artifactPath": artifact_path.relative_to(
            Path(workspace_root).resolve()
        ).as_posix(),
        "artifactSha256": artifact_sha256,
        "modelStateSha256": model_state_sha256,
        **expected_provenance,
    }


def load_phase4_champion_artifact(
    *,
    workspace_root: str | Path,
    config: ExperimentConfig,
) -> tuple[Mapping[str, torch.Tensor], dict[str, object]]:
    """Deeply verify and load one published model-only champion."""

    initialization = config.initialization
    if (
        initialization.mode != "champion"
        or initialization.artifact_path is None
        or initialization.champion_manifest_path is None
        or initialization.champion_id is None
    ):
        raise ValueError(
            "champion artifact loading requires initialization.mode=champion"
        )
    workspace = Path(workspace_root).resolve()

    def bound_file(path: Path, label: str) -> Path:
        candidate = path if path.is_absolute() else workspace / path
        if candidate.is_symlink():
            raise FileNotFoundError(f"{label} must not be a symlink: {candidate}")
        candidate = candidate.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as error:
            raise ValueError(f"{label} must be inside the workspace") from error
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"{label} is missing: {candidate}")
        return candidate

    artifact_path = bound_file(initialization.artifact_path, "champion artifact")
    manifest_path = bound_file(
        initialization.champion_manifest_path,
        "champion manifest",
    )
    if artifact_path.parent != manifest_path.parent or manifest_path.name != "manifest.json":
        raise Phase4ResumeError(
            "champion artifact and manifest must name one champion store"
        )
    if sha256_file(artifact_path) != initialization.artifact_sha256:
        raise Phase4ResumeError("champion artifact SHA-256 does not match config")
    if sha256_file(manifest_path) != initialization.champion_manifest_file_sha256:
        raise Phase4ResumeError("champion manifest file SHA-256 does not match config")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4ResumeError("champion manifest is unreadable") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("manifestSha256")
        != initialization.champion_manifest_sha256
        or canonical_json_bytes(manifest) != manifest_path.read_bytes()
    ):
        raise Phase4ResumeError("champion manifest binding does not match config")
    store = ChampionStore(
        artifact_path.parent,
        artifact_root=artifact_path.parent.parent,
    )
    store.verify(deep=True)
    try:
        record = store.resolve(initialization.champion_id)
    except KeyError as error:
        raise Phase4ResumeError("configured champion is absent from its manifest") from error
    if (
        not record.retained
        or record.file_name != artifact_path.name
        or record.model_sha256 != initialization.artifact_sha256
    ):
        raise Phase4ResumeError("champion record does not bind the configured artifact")
    state = store.load_state_dict(record.champion_id, map_location="cpu")
    try:
        state_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4ResumeError("champion tensor state is invalid") from error
    if state_sha256 != initialization.model_state_sha256:
        raise Phase4ResumeError("champion tensor-state SHA-256 does not match config")
    return state, {
        "kind": "published-champion",
        "championId": record.champion_id,
        "generation": record.generation,
        "artifactPath": artifact_path.relative_to(workspace).as_posix(),
        "artifactSha256": record.model_sha256,
        "modelStateSha256": state_sha256,
        "championManifestPath": manifest_path.relative_to(workspace).as_posix(),
        "championManifestSha256": initialization.champion_manifest_sha256,
        "championManifestFileSha256": initialization.champion_manifest_file_sha256,
        "sourceSha256": record.source_sha256,
        "configSha256": record.config_sha256,
        "openingSuiteSha256": record.opening_suite_sha256,
        "formalPromotion": bool(record.metadata.get("formalPromotion", False)),
    }


def load_phase4_recovery_artifact(
    *,
    workspace_root: str | Path,
    config: ExperimentConfig,
) -> tuple[Mapping[str, torch.Tensor], dict[str, object]]:
    """Load a model-only recovery candidate bound to its immutable archive."""

    initialization = config.initialization
    if (
        initialization.mode != "recovery"
        or initialization.artifact_path is None
        or initialization.recovery_manifest_path is None
    ):
        raise ValueError(
            "recovery artifact loading requires initialization.mode=recovery"
        )
    workspace = Path(workspace_root).resolve()

    def bound_file(path: Path, label: str) -> Path:
        candidate = path if path.is_absolute() else workspace / path
        if candidate.is_symlink():
            raise FileNotFoundError(f"{label} must not be a symlink: {candidate}")
        candidate = candidate.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as error:
            raise ValueError(f"{label} must be inside the workspace") from error
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"{label} is missing: {candidate}")
        return candidate

    artifact_path = bound_file(initialization.artifact_path, "recovery artifact")
    manifest_path = bound_file(
        initialization.recovery_manifest_path,
        "recovery archive manifest",
    )
    if sha256_file(artifact_path) != initialization.artifact_sha256:
        raise Phase4ResumeError("recovery artifact SHA-256 does not match config")
    if sha256_file(manifest_path) != initialization.recovery_manifest_file_sha256:
        raise Phase4ResumeError(
            "recovery archive manifest file SHA-256 does not match config"
        )
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4ResumeError("recovery archive manifest is unreadable") from error
    if not isinstance(manifest, Mapping) or canonical_json_bytes(manifest) != raw_manifest:
        raise Phase4ResumeError("recovery archive manifest is not canonical JSON")
    manifest_core = dict(manifest)
    archive_sha256 = manifest_core.pop("archiveSha256", None)
    if (
        archive_sha256 != initialization.recovery_manifest_sha256
        or archive_sha256 != sha256_bytes(canonical_json_bytes(manifest_core))
        or manifest.get("schemaVersion") != 1
        or manifest.get("status") != "completed"
        or manifest.get("cpuOnly") is not True
        or manifest.get("cudaAccessed") is not False
        or manifest.get("kind")
        not in {
            "phase4f11-generation-11-candidate-archive",
            "phase4-recovery-candidate-archive",
        }
    ):
        raise Phase4ResumeError("recovery archive manifest binding is invalid")

    candidate = manifest.get("candidate")
    files = manifest.get("files")
    if not isinstance(candidate, Mapping) or not isinstance(files, Mapping):
        raise Phase4ResumeError("recovery archive candidate metadata is invalid")
    expected_candidate = {
        "candidateModelSha256": initialization.artifact_sha256,
        "checkpointSha256": initialization.recovery_checkpoint_sha256,
        "configSha256": initialization.recovery_config_sha256,
        "generation": initialization.recovery_generation,
        "modelStateSha256": initialization.model_state_sha256,
        "runManifestSha256": initialization.recovery_run_manifest_sha256,
        "sourceSha256": initialization.recovery_source_sha256,
        "step": initialization.recovery_step,
    }
    if any(candidate.get(key) != value for key, value in expected_candidate.items()):
        raise Phase4ResumeError("recovery archive candidate provenance mismatch")

    candidate_file = files.get("candidateModel")
    if (
        not isinstance(candidate_file, Mapping)
        or candidate_file.get("path") != artifact_path.name
        or candidate_file.get("sha256") != initialization.artifact_sha256
        or candidate_file.get("bytes") != artifact_path.stat().st_size
    ):
        raise Phase4ResumeError("recovery archive candidate file binding is invalid")
    for label, reference in files.items():
        if (
            not isinstance(label, str)
            or not isinstance(reference, Mapping)
            or not isinstance(reference.get("path"), str)
            or Path(str(reference["path"])).name != reference["path"]
            or type(reference.get("bytes")) is not int
            or reference["bytes"] <= 0
            or not isinstance(reference.get("sha256"), str)
        ):
            raise Phase4ResumeError("recovery archive file reference is invalid")
        archived = artifact_path.parent / str(reference["path"])
        if (
            archived.is_symlink()
            or not archived.is_file()
            or archived.stat().st_size != reference["bytes"]
            or sha256_file(archived) != reference["sha256"]
        ):
            raise Phase4ResumeError(f"recovery archive file differs: {label}")
    if manifest.get("bytes") != sum(
        int(reference["bytes"]) for reference in files.values()
    ):
        raise Phase4ResumeError("recovery archive byte count mismatch")

    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Phase4ResumeError(
            "recovery artifact cannot be deserialized safely"
        ) from error
    required_fields = {
        "schemaVersion",
        "format",
        "generation",
        "step",
        "identity",
        "checkpointSha256",
        "modelStateSha256",
        "modelState",
    }
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required_fields
        or payload.get("schemaVersion") != PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION
        or payload.get("format") != PHASE4_CANDIDATE_MODEL_FORMAT
        or payload.get("generation") != initialization.recovery_generation
        or payload.get("step") != initialization.recovery_step
        or payload.get("checkpointSha256")
        != initialization.recovery_checkpoint_sha256
        or not isinstance(identity, Mapping)
        or dict(identity)
        != {
            "runManifestSha256": initialization.recovery_run_manifest_sha256,
            "configSha256": initialization.recovery_config_sha256,
            "sourceSha256": initialization.recovery_source_sha256,
        }
    ):
        raise Phase4ResumeError("recovery artifact schema/provenance is invalid")
    state = payload.get("modelState")
    if not isinstance(state, Mapping):
        raise Phase4ResumeError("recovery artifact modelState must be a mapping")
    try:
        state_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4ResumeError("recovery artifact tensor state is invalid") from error
    if (
        payload.get("modelStateSha256") != state_sha256
        or state_sha256 != initialization.model_state_sha256
    ):
        raise Phase4ResumeError("recovery artifact tensor-state hash mismatch")
    return state, {
        "kind": "recovery-candidate",
        "generation": initialization.recovery_generation,
        "step": initialization.recovery_step,
        "artifactPath": artifact_path.relative_to(workspace).as_posix(),
        "artifactSha256": initialization.artifact_sha256,
        "modelStateSha256": state_sha256,
        "recoveryManifestPath": manifest_path.relative_to(workspace).as_posix(),
        "recoveryManifestSha256": archive_sha256,
        "recoveryManifestFileSha256": initialization.recovery_manifest_file_sha256,
        "sourceSha256": initialization.recovery_source_sha256,
        "configSha256": initialization.recovery_config_sha256,
        "runManifestSha256": initialization.recovery_run_manifest_sha256,
        "checkpointSha256": initialization.recovery_checkpoint_sha256,
    }


class _RuntimeBudgetGuard:
    def __init__(
        self,
        *,
        control_root: Path,
        artifact_limit_bytes: int,
        ledger: GpuBudgetLedger,
        stage_ledger: Phase4StageBudgetLedger,
        ledger_run_id: str,
        started: float,
        stage: str = PHASE4D_STAGE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.control_root = control_root
        self.artifact_limit_bytes = artifact_limit_bytes
        self.ledger = ledger
        self.stage_ledger = stage_ledger
        self.ledger_run_id = ledger_run_id
        self.started = started
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        self.stage = stage
        self.clock = clock
        self.last_ledger_update = float("-inf")
        self.last_resource_check = float("-inf")

    def check(
        self,
        *,
        force: bool = False,
        reserve_bytes: int = 0,
    ) -> dict[str, object]:
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a non-negative integer")
        now = self.clock()
        elapsed = now - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("Phase 4 runtime clock regressed")
        if force or now - self.last_ledger_update >= _LEDGER_HEARTBEAT_SECONDS:
            global_hours = self.ledger.update(
                self.ledger_run_id,
                elapsed_seconds=elapsed,
                status="running",
            )
            self.last_ledger_update = now
        else:
            global_hours = self.ledger.total_hours
        if global_hours >= GPU_HOUR_BUDGET:
            raise Phase4ResourceLimitReached(
                f"{GPU_HOUR_BUDGET:g} GPU-hour global budget reached"
            )
        stage = self.stage_ledger.assert_available(self.stage, global_hours)

        resource_payload: dict[str, object] = {"stageGpu": stage}
        if (
            force
            or reserve_bytes
            or now - self.last_resource_check >= _BUDGET_CHECK_INTERVAL_SECONDS
        ):
            free_bytes = int(shutil.disk_usage(self.control_root).free)
            artifact_bytes = directory_size(self.control_root)
            if free_bytes - reserve_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
                raise Phase4ResourceLimitReached(
                    "write reserve would cross the 20 GiB free-disk stop line"
                )
            if artifact_bytes + reserve_bytes > self.artifact_limit_bytes:
                raise Phase4ResourceLimitReached(
                    "write reserve would exceed the configured artifact cap"
                )
            resource_payload.update(
                {
                    "freeDiskBytes": free_bytes,
                    "artifactBytes": artifact_bytes,
                    "reservedWriteBytes": reserve_bytes,
                }
            )
            self.last_resource_check = now
        return resource_payload


def _metrics_payload(metrics: Sequence[TrainMetrics]) -> dict[str, object]:
    if not metrics:
        raise ValueError("one generation must contain at least one training step")

    def summary(field: str) -> dict[str, float]:
        values = [float(getattr(metric, field)) for metric in metrics]
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "final": values[-1],
        }

    return {
        "steps": len(metrics),
        "loss": summary("loss"),
        "policyLoss": summary("policy_loss"),
        "valueLoss": summary("value_loss"),
        "gradNorm": summary("grad_norm"),
        "learningRate": summary("learning_rate"),
    }


def _replay_prefix_sha256(replay: ReplayStore, count: int | None = None) -> str:
    shards = replay.shards
    if count is None:
        count = len(shards)
    if type(count) is not int or not 0 <= count <= len(shards):
        raise ValueError("replay prefix count is outside the committed shard range")
    payload = {
        "schemaVersion": 1,
        "shards": [
            {
                "sequence": shard.sequence,
                "gameId": shard.game_id,
                "file": shard.file_name,
                "bytes": shard.byte_size,
                "sha256": shard.sha256,
                "plies": shard.ply_count,
                "minimumGeneration": shard.minimum_generation,
                "maximumGeneration": shard.maximum_generation,
            }
            for shard in shards[:count]
        ],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _replay_window_records(replay: ReplayStore) -> list[dict[str, object]]:
    return [
        {
            "sequence": shard.sequence,
            "gameId": shard.game_id,
            "file": shard.file_name,
            "bytes": shard.byte_size,
            "sha256": shard.sha256,
            "plies": shard.ply_count,
            "minimumGeneration": shard.minimum_generation,
            "maximumGeneration": shard.maximum_generation,
        }
        for shard in replay.shards
    ]


def _replay_window_sha256(replay: ReplayStore) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "shards": _replay_window_records(replay),
            }
        )
    )


def _replay_shard_from_record(value: object) -> ReplayShardInfo:
    if not isinstance(value, Mapping) or set(value) != {
        "sequence",
        "gameId",
        "file",
        "bytes",
        "sha256",
        "plies",
        "minimumGeneration",
        "maximumGeneration",
    }:
        raise Phase4ResumeError("checkpoint replay shard record is invalid")
    integer_fields = (
        "sequence",
        "bytes",
        "plies",
        "minimumGeneration",
        "maximumGeneration",
    )
    if (
        any(type(value.get(field)) is not int for field in integer_fields)
        or not isinstance(value.get("gameId"), str)
        or not isinstance(value.get("file"), str)
    ):
        raise Phase4ResumeError("checkpoint replay shard values are invalid")
    try:
        digest = validate_sha256(value.get("sha256"), "replay shard sha256")
        shard = ReplayShardInfo(
            sequence=value["sequence"],
            game_id=value["gameId"],
            file_name=value["file"],
            byte_size=value["bytes"],
            sha256=digest,
            ply_count=value["plies"],
            minimum_generation=value["minimumGeneration"],
            maximum_generation=value["maximumGeneration"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4ResumeError(
            "checkpoint replay shard values are invalid"
        ) from error
    if (
        shard.sequence < 0
        or not shard.game_id
        or Path(shard.file_name).name != shard.file_name
        or shard.byte_size <= 0
        or shard.ply_count <= 0
        or shard.minimum_generation < 0
        or shard.maximum_generation < shard.minimum_generation
    ):
        raise Phase4ResumeError("checkpoint replay shard values are invalid")
    return shard


def _tensor_payload_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def _checkpoint_write_reserve_bytes(
    model: PolicyValueNetwork,
    trainer: AlphaZeroTrainer,
    checkpoint_root: Path,
) -> int:
    tensor_bytes = _tensor_payload_bytes(model.state_dict()) + _tensor_payload_bytes(
        trainer.optimizer.state_dict()
    )
    observed_bytes = max(
        (path.stat().st_size for path in checkpoint_root.glob("checkpoint-*.pt")),
        default=0,
    )
    return max(
        _CHECKPOINT_FIXED_WRITE_RESERVE_BYTES,
        tensor_bytes * 2 + _CHECKPOINT_FIXED_WRITE_RESERVE_BYTES,
        observed_bytes * 2 + _CHECKPOINT_FIXED_WRITE_RESERVE_BYTES,
    )


def _replay_snapshot(replay: ReplayStore, verification: object) -> dict[str, object]:
    shard_bytes = sum(shard.byte_size for shard in replay.shards)
    committed_bytes = getattr(verification, "committed_bytes")
    return {
        "manifestSha256": sha256_file(replay.manifest_path),
        "games": getattr(verification, "game_count"),
        "plies": getattr(verification, "ply_count"),
        "bytes": shard_bytes,
        "manifestBytes": committed_bytes - shard_bytes,
        "committedBytes": committed_bytes,
        "deepVerified": getattr(verification, "deep"),
        "gameIds": list(replay.game_ids),
        "prefixSha256": _replay_prefix_sha256(replay),
        "windowContract": "ordered-fifo-shard-window-v2",
        "windowSha256": _replay_window_sha256(replay),
        "shards": _replay_window_records(replay),
        "samplingAlgorithm": REPLAY_SAMPLING_ALGORITHM,
        "mirrorRngAlgorithm": REPLAY_MIRROR_RNG_ALGORITHM,
    }


class Phase4TrainingDriver:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        control_root: str | Path,
        config: ExperimentConfig,
        device: torch.device,
        lineage: Phase4Lineage,
        budget_guard: _RuntimeBudgetGuard,
        replay: ReplayStore | None = None,
        replay_cache: Phase4ReplayCache | None = None,
        challenge_cache: ReplayTensorCache | None = None,
        progress_callback: Callable[[Mapping[str, object]], None] | None = None,
        execution_contract: Phase4ExecutionContract = PHASE4D_EXECUTION_CONTRACT,
        bootstrap_baseline_root: str | Path | None = None,
    ) -> None:
        validate_phase4_config(config, execution_contract)
        if not isinstance(execution_contract, Phase4ExecutionContract):
            raise TypeError("execution_contract must be a Phase4ExecutionContract")
        self.workspace_root = Path(workspace_root).resolve()
        self.control_root = Path(control_root).resolve()
        self.config = config
        self.device = device
        self.lineage = lineage
        self.budget_guard = budget_guard
        self.run_dir = config.run_dir.resolve()
        self.progress_callback = progress_callback
        self.execution_contract = execution_contract
        self.bootstrap_baseline_root = (
            None
            if bootstrap_baseline_root is None
            else Path(bootstrap_baseline_root).resolve()
        )
        if self.bootstrap_baseline_root is not None:
            try:
                self.bootstrap_baseline_root.relative_to(self.workspace_root)
            except ValueError as error:
                raise ValueError(
                    "bootstrap baseline must be inside the workspace"
                ) from error
        resource_budget = lineage.manifest.payload.get("resourceBudget")
        if not isinstance(resource_budget, Mapping):
            raise Phase4TrainingError("run manifest resource budget is missing")
        maximum_generations = resource_budget.get("maximumGenerations")
        if maximum_generations is not None and (
            maximum_generations != execution_contract.maximum_generations
        ):
            raise Phase4TrainingError(
                "run manifest generation ceiling differs from the execution contract"
            )
        self.maximum_generations = (
            execution_contract.maximum_generations
            if maximum_generations is None
            else int(maximum_generations)
        )

        seed_everything(config.seed)
        self.model = PolicyValueNetwork(_network_config(config)).to(device)
        self.base_model_sha256 = canonical_model_state_sha256(self.model)
        if (
            self.base_model_sha256
            != config.initialization.expected_base_model_sha256
        ):
            raise Phase4TrainingError(
                "seeded random base model does not match "
                "initialization.expectedBaseModelSha256"
            )
        self.trainer = _trainer(config, self.model)
        replay_bytes = max(1, config.artifact_limit_bytes * 3 // 4)
        replay_root = self.run_dir / "replay"
        if not (replay_root / "manifest.json").exists():
            self.budget_guard.check(
                force=True,
                reserve_bytes=_REPLAY_MANIFEST_WRITE_RESERVE_BYTES,
            )
        if replay is None:
            replay = ReplayStore(
                replay_root,
                max_games=config.training.replay_capacity,
                max_plies=config.training.replay_capacity,
                max_bytes=replay_bytes,
            )
        elif (
            replay.root.resolve() != replay_root.resolve()
            or replay.max_games != config.training.replay_capacity
            or replay.max_plies != config.training.replay_capacity
            or replay.max_bytes != replay_bytes
        ):
            raise Phase4TrainingError(
                "prebuilt replay store differs from the experiment config"
        )
        self.replay = replay
        if replay_cache is None:
            replay_cache = _create_phase4_replay_cache(
                replay=self.replay,
                run_dir=self.run_dir,
                execution_contract=execution_contract,
                source_sha256=lineage.identity.source_sha256,
                encoder_sha256=loaded_native_backend_hash(),
            )
        elif replay_cache.replay is not self.replay:
            raise Phase4TrainingError(
                "prebuilt replay cache belongs to another replay store"
            )
        self.replay_cache = replay_cache
        def dataset_progress(payload: Mapping[str, object]) -> None:
            details = dict(payload)
            status = str(details.pop("status"))
            _publish_progress(
                self.progress_callback,
                stage="teacher-dataset-validation",
                status=status,
                **details,
            )

        challenge_settings = config.training.challenge_replay
        bound_challenge = None
        if challenge_settings is None:
            if challenge_cache is not None:
                raise Phase4TrainingError(
                    "challenge cache was supplied without a challengeReplay config"
                )
            self.challenge_cache = None
            self.challenge_contract: Mapping[str, object] | None = None
        else:
            bound_challenge = open_bound_challenge_replay(
                workspace_root=self.workspace_root,
                settings=challenge_settings,
                progress_callback=dataset_progress,
            )
            if challenge_cache is None:
                challenge_cache = ReplayTensorCache(
                    self.run_dir / "challenge-replay-tensor-cache",
                    bound_challenge.store,
                    source_sha256=lineage.identity.source_sha256,
                    encoder_sha256=loaded_native_backend_hash(),
                    native_encoder=load_native_module().encode_states,
                )
            elif challenge_cache.replay.root.resolve() != bound_challenge.store.root.resolve():
                raise Phase4TrainingError(
                    "prebuilt challenge cache belongs to another replay store"
                )
            if not challenge_cache.is_built:
                challenge_cache.refresh()
            challenge_cache.verify(deep=True)
            self.challenge_cache = challenge_cache
            self.challenge_contract = bound_challenge.contract
        self.challenge_sampler = (
            None
            if self.challenge_cache is None
            or challenge_settings is None
            or challenge_settings.stratified_sampling is None
            else load_stratified_challenge_sampler(
                self.workspace_root,
                self.challenge_cache.replay,
                self.challenge_cache,
                challenge_settings.stratified_sampling,
            )
        )
        holdout_settings = config.training.health_holdout
        if holdout_settings is None:
            self.health_holdout_cache: ReplayTensorCache | None = None
            self.health_holdout_contract: Mapping[str, object] | None = None
            self.health_holdout_provenance: Mapping[str, object] | None = None
        else:
            bound_holdout = open_bound_health_holdout(
                workspace_root=self.workspace_root,
                settings=holdout_settings,
                bound_train=bound_challenge,
                progress_callback=dataset_progress,
            )
            health_cache = ReplayTensorCache(
                self.run_dir / "health-holdout-tensor-cache",
                bound_holdout.store,
                source_sha256=lineage.identity.source_sha256,
                encoder_sha256=loaded_native_backend_hash(),
                native_encoder=load_native_module().encode_states,
            )
            if not health_cache.is_built:
                health_cache.refresh()
            health_cache.verify(deep=True)
            self.health_holdout_cache = health_cache
            self.health_holdout_contract = bound_holdout.contract
            self.health_holdout_provenance = {
                "contractSha256": holdout_settings.contract_sha256,
                "datasetFingerprint": holdout_settings.dataset_fingerprint,
                "holdoutManifestSha256": holdout_settings.holdout_manifest_sha256,
                "excludedOrbits": bound_holdout.excluded_orbits,
                "trainOrbits": bound_holdout.train_orbits,
                "holdoutOrbits": bound_holdout.holdout_orbits,
            }
        self.checkpoints = Phase4CheckpointStore(
            self.run_dir / "checkpoints",
            identity=lineage.identity,
            artifact_root=self.control_root,
            artifact_limit_bytes=config.artifact_limit_bytes,
            max_to_keep=config.training.max_checkpoints,
        )
        self.reports = GenerationReportStore(
            self.run_dir / "reports",
            run_manifest_sha256=lineage.manifest.sha256,
        )
        metadata = resource_budget.get("metadata")
        if not isinstance(metadata, Mapping):
            raise Phase4TrainingError("run manifest resource budget metadata is missing")
        try:
            expected_native_sha256 = validate_sha256(
                metadata.get("nativeBackendSha256"),
                "nativeBackendSha256",
            )
        except ValueError as error:
            raise Phase4TrainingError(
                "run manifest does not bind a valid native backend hash"
            ) from error
        actual_native_sha256 = loaded_native_backend_hash()
        if actual_native_sha256 != expected_native_sha256:
            raise Phase4TrainingError(
                "loaded native backend differs from the immutable run manifest"
            )
        self.native_module = load_native_module()
        self.batch_encoder = ReplayBatchEncoder(
            input_channels=config.network.input_channels,
            device=device,
            native_encoder=self.native_module.encode_states,
        )
        self.loaded: Phase4CheckpointLoad | None = None
        self.fallback_rejections: tuple[str, ...] = ()

    def initialize_new_run(self) -> None:
        """Commit the cold/warm starting model before any replay can be written."""

        if not self.config.initialization.reset_optimizer:
            self._initialize_checkpoint_continuation()
            return

        if (
            self.checkpoints.candidates()
            or self.replay.game_count
            or self.reports.generations()
        ):
            raise Phase4ResumeError(
                "initial model import is allowed only before checkpoints/replay/reports"
            )
        initialization = self.config.initialization
        artifact: dict[str, object] | None = None
        initialization_mode = initialization.mode
        if self.bootstrap_baseline_root is not None:
            if initialization.mode != "random":
                raise Phase4TrainingError(
                    "bootstrap baseline and legacy model initialization "
                    "cannot be combined"
                )
            # Keep this import lazy because phase4_baseline audits pilot code
            # that imports the training driver.
            from .phase4_baseline import (
                BASELINE_KIND,
                load_bootstrap_baseline_state,
            )

            verification = _verify_phase4e_bootstrap_baseline(
                self.workspace_root,
                self.bootstrap_baseline_root,
            )
            state = load_bootstrap_baseline_state(
                workspace_root=self.workspace_root,
                baseline_root=self.bootstrap_baseline_root,
                verify_lineage=False,
            )
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise Phase4TrainingError(
                    "bootstrap baseline tensors do not match the configured network"
                ) from error
            initialization_mode = "model"
            artifact = {
                "kind": BASELINE_KIND,
                "schemaVersion": verification["schemaVersion"],
                "baselineRoot": self.bootstrap_baseline_root.relative_to(
                    self.workspace_root
                ).as_posix(),
                "manifestSha256": verification["manifestSha256"],
                "modelSha256": verification["modelSha256"],
                "modelStateSha256": verification["modelStateSha256"],
                "formalPromotion": False,
            }
        elif initialization.mode == "model":
            state, artifact = load_phase4_warm_artifact(
                workspace_root=self.workspace_root,
                config=self.config,
            )
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise Phase4TrainingError(
                    "warm artifact model tensors do not match the configured network"
                ) from error
        elif initialization.mode == "champion":
            state, artifact = load_phase4_champion_artifact(
                workspace_root=self.workspace_root,
                config=self.config,
            )
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise Phase4TrainingError(
                    "champion model tensors do not match the configured network"
                ) from error
        elif initialization.mode == "recovery":
            state, artifact = load_phase4_recovery_artifact(
                workspace_root=self.workspace_root,
                config=self.config,
            )
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise Phase4TrainingError(
                    "recovery model tensors do not match the configured network"
                ) from error
        if self.trainer.optimizer.state:
            raise Phase4TrainingError(
                "initialization must not import optimizer state"
            )
        initialized_sha256 = canonical_model_state_sha256(self.model)
        if artifact is not None and artifact["modelStateSha256"] != initialized_sha256:
            raise Phase4TrainingError(
                "loaded warm model does not match its canonical tensor-state hash"
            )
        if not self.replay_cache.is_built:
            _refresh_replay_cache(self.replay_cache)
        verification = _replay_cache_verification(self.replay_cache)
        replay_snapshot = _replay_snapshot(self.replay, verification)
        generation_data = {
            "schemaVersion": PHASE4_TRAINING_SCHEMA_VERSION,
            "kind": "initialization",
            "nextGeneration": 0,
            "mode": initialization_mode,
            "seed": self.config.seed,
            "baseModelSha256": self.base_model_sha256,
            "initializedModelSha256": initialized_sha256,
            "optimizerReset": True,
            "artifact": artifact,
        }
        self.budget_guard.check(
            force=True,
            reserve_bytes=_checkpoint_write_reserve_bytes(
                self.model,
                self.trainer,
                self.checkpoints.root,
            ),
        )
        record = self.checkpoints.save(
            generation=-1,
            model=self.model,
            trainer=self.trainer,
            generation_data=generation_data,
            replay_snapshot=replay_snapshot,
        )
        self.loaded = Phase4CheckpointLoad(
            record=record,
            payload={
                "generationData": generation_data,
                "replaySnapshot": replay_snapshot,
            },
            rejected_newer=(),
        )

    def _initialize_checkpoint_continuation(self) -> None:
        """Re-sign a verified source history without changing training state."""

        initialization = self.config.initialization
        if (
            self.execution_contract.stage != PHASE4F_STAGE_B_EXTENDED
            or initialization.mode != "recovery"
            or initialization.artifact_path is None
            or initialization.recovery_generation is None
            or initialization.recovery_step is None
            or initialization.recovery_run_manifest_sha256 is None
            or initialization.recovery_config_sha256 is None
            or initialization.recovery_source_sha256 is None
        ):
            raise Phase4TrainingError("checkpoint continuation provenance is incomplete")
        if self.checkpoints.candidates() or self.reports.generations():
            raise Phase4ResumeError(
                "checkpoint continuation is allowed only before checkpoints/reports"
            )
        source_artifact = (
            initialization.artifact_path
            if initialization.artifact_path.is_absolute()
            else self.workspace_root / initialization.artifact_path
        ).resolve()
        source_root = source_artifact.parent.parent
        source_replay = ReplayStore(
            source_root / "replay",
            max_games=self.config.training.replay_capacity,
            max_plies=self.config.training.replay_capacity,
            max_bytes=max(1, self.config.artifact_limit_bytes * 3 // 4),
        )
        # ReplayTensorCache preflight performs the full decode before the
        # driver starts. Here we compare the already bound immutable shard
        # manifest without decoding the same history a second time.
        source_verification = source_replay.verify(deep=False)
        current_verification = self.replay.verify(deep=False)
        if (
            self.replay.game_ids != source_replay.game_ids
            or current_verification != source_verification
            or sha256_file(self.replay.manifest_path)
            != sha256_file(source_replay.manifest_path)
        ):
            raise Phase4ResumeError("imported replay differs from the source lineage")
        source_store = Phase4CheckpointStore(
            source_root / "checkpoints",
            identity=CheckpointIdentity(
                run_manifest_sha256=initialization.recovery_run_manifest_sha256,
                config_sha256=initialization.recovery_config_sha256,
                source_sha256=initialization.recovery_source_sha256,
            ),
            artifact_root=self.control_root,
            artifact_limit_bytes=self.config.artifact_limit_bytes,
            max_to_keep=self.config.training.max_checkpoints,
        )
        imported: Phase4CheckpointLoad | None = None
        try:
            for generation in range(initialization.recovery_generation + 1):
                source = source_store.load_generation(
                    generation,
                    map_location=self.device,
                )
                if generation == initialization.recovery_generation and (
                    source.record.sha256 != initialization.recovery_checkpoint_sha256
                    or source.record.step != initialization.recovery_step
                ):
                    raise Phase4ResumeError("source continuation checkpoint changed")
                self.model.load_state_dict(source.payload["model"], strict=True)
                self.trainer.optimizer.load_state_dict(source.payload["optimizer"])
                self.trainer.load_runtime_state_dict(source.payload["trainerRuntime"])
                if generation == initialization.recovery_generation:
                    restore_rng_state(source.payload["rng"])
                record = self.checkpoints.save(
                    generation=generation,
                    model=self.model,
                    trainer=self.trainer,
                    generation_data=source.payload["generationData"],
                    replay_snapshot=source.payload["replaySnapshot"],
                )
                self.reports.write(
                    generation,
                    self._report_payload(record, source.payload["generationData"]),
                )
                imported = Phase4CheckpointLoad(
                    record=record,
                    payload={
                        "generationData": source.payload["generationData"],
                        "replaySnapshot": source.payload["replaySnapshot"],
                    },
                    rejected_newer=(),
                )
        except BaseException:
            # Migration is a short, deterministic transaction. Removing only
            # destination-lineage artifacts makes a Ctrl+C retry unambiguous.
            for path in self.checkpoints.root.glob("*"):
                if path.is_file():
                    path.unlink()
            for path in self.reports.root.glob("generation-*.json"):
                path.unlink()
            raise
        if imported is None:
            raise Phase4TrainingError("checkpoint continuation imported no generations")
        if canonical_model_state_sha256(self.model) != initialization.model_state_sha256:
            raise Phase4ResumeError("continued model tensor hash changed")
        self.loaded = imported
        self._validate_replay_extension(current_verification, imported)
        if self.config.training.health_holdout is not None:
            reconcile_best_fixed_holdout_candidate(self.run_dir, self.reports)

    def restore(self) -> None:
        if not self.replay_cache.is_built:
            _refresh_replay_cache(self.replay_cache)
        verification = _replay_cache_verification(self.replay_cache)
        self.loaded = self.checkpoints.restore(
            self.model,
            self.trainer,
            map_location=self.device,
        )
        if self.loaded is None:
            if (
                self.execution_contract.stage == PHASE4F_STAGE_B_EXTENDED
                and not self.config.initialization.reset_optimizer
                and not self.reports.generations()
            ):
                self._initialize_checkpoint_continuation()
                return
            if self.replay.game_count or self.reports.generations():
                raise Phase4ResumeError(
                    "replay/report exists without the initialization checkpoint"
                )
            self.initialize_new_run()
            return
        self._validate_replay_extension(verification, self.loaded)
        self._reconcile_reports()
        if self.config.training.health_holdout is not None:
            reconcile_best_fixed_holdout_candidate(self.run_dir, self.reports)
        if self.loaded is not None and self.loaded.rejected_newer:
            self.fallback_rejections = self.loaded.rejected_newer
            self.checkpoints.quarantine_rejected(self.loaded.rejected_newer)

    def _validate_replay_extension(
        self,
        verification: object,
        loaded: Phase4CheckpointLoad,
    ) -> None:
        snapshot = loaded.payload.get("replaySnapshot")
        if not isinstance(snapshot, Mapping):
            raise Phase4ResumeError("checkpoint replay snapshot is invalid")
        saved_ids = snapshot.get("gameIds")
        if (
            not isinstance(saved_ids, list)
            or any(not isinstance(game_id, str) for game_id in saved_ids)
            or snapshot.get("games") != len(saved_ids)
        ):
            raise Phase4ResumeError("checkpoint replay game IDs are invalid")
        if snapshot.get("windowContract") == "ordered-fifo-shard-window-v2":
            self._validate_fifo_replay_extension(
                snapshot,
                verification,
                loaded,
            )
            return
        current_ids = self.replay.game_ids
        saved_tuple = tuple(saved_ids)
        if current_ids[: len(saved_tuple)] != saved_tuple:
            raise Phase4ResumeError(
                "raw replay is not an append-only extension of the checkpoint"
            )
        try:
            saved_prefix_sha256 = validate_sha256(
                snapshot.get("prefixSha256"),
                "checkpoint replay prefix sha256",
            )
        except ValueError as error:
            raise Phase4ResumeError(
                "checkpoint replay prefix hash is invalid"
            ) from error
        if _replay_prefix_sha256(self.replay, len(saved_tuple)) != saved_prefix_sha256:
            raise Phase4ResumeError(
                "raw replay shard prefix differs from the checkpoint"
            )
        saved_plies = snapshot.get("plies")
        saved_bytes = snapshot.get("bytes")
        saved_manifest_bytes = snapshot.get("manifestBytes")
        saved_committed_bytes = snapshot.get("committedBytes")
        if (
            type(saved_plies) is not int
            or saved_plies < 0
            or type(saved_bytes) is not int
            or saved_bytes < 0
            or type(saved_manifest_bytes) is not int
            or saved_manifest_bytes <= 0
            or type(saved_committed_bytes) is not int
            or saved_committed_bytes != saved_bytes + saved_manifest_bytes
        ):
            raise Phase4ResumeError("checkpoint replay totals are invalid")
        current_bytes = sum(shard.byte_size for shard in self.replay.shards)
        current_committed_bytes = getattr(verification, "committed_bytes", -1)
        if (
            getattr(verification, "game_count", None) != len(current_ids)
            or getattr(verification, "ply_count", -1) < saved_plies
            or current_bytes < saved_bytes
            or current_committed_bytes < saved_committed_bytes
        ):
            raise Phase4ResumeError("raw replay totals regressed from the checkpoint")

        extra_ids = current_ids[len(saved_tuple) :]
        if not extra_ids:
            if getattr(verification, "ply_count", None) != saved_plies:
                raise Phase4ResumeError("checkpoint replay ply total changed")
            if current_bytes != saved_bytes:
                raise Phase4ResumeError("checkpoint replay shard-byte total changed")
            if current_committed_bytes != saved_committed_bytes:
                raise Phase4ResumeError("checkpoint replay committed bytes changed")
            if sha256_file(self.replay.manifest_path) != snapshot.get(
                "manifestSha256"
            ):
                raise Phase4ResumeError("checkpoint replay manifest hash changed")
            return

        next_generation = loaded.record.generation + 1
        expected_ids = {
            f"phase4-g{next_generation:08d}-game{index:06d}"
            for index in range(self.config.self_play.games)
        }
        expected_model_id = f"sha256:{loaded.record.sha256}"
        generation_seed = (self.config.seed + next_generation * 1_000_003) % (
            2**63
        )
        if len(extra_ids) > self.config.self_play.games or any(
            game_id not in expected_ids for game_id in extra_ids
        ):
            raise Phase4ResumeError(
                "raw replay extension contains data beyond the next generation"
            )
        extra_plies = 0
        for game_id in extra_ids:
            replay_cache = getattr(self, "replay_cache", None)
            game = (
                _replay_cache_game(replay_cache, game_id)
                if isinstance(
                    replay_cache,
                    (ReplayReconstructionCache, ReplayTensorCache),
                )
                and replay_cache.is_built
                else self.replay.load_game(game_id)
            )
            extra_plies += len(game.plies)
            game_index = int(game_id.removeprefix(
                f"phase4-g{next_generation:08d}-game"
            ))
            if (
                game.metadata.get("generation") != next_generation
                or game.metadata.get("modelId") != expected_model_id
                or any(ply.model_id != expected_model_id for ply in game.plies)
            ):
                raise Phase4ResumeError(
                    f"raw replay extension identity mismatch for {game_id}"
                )
            self._validate_game_invocation(
                game,
                generation=next_generation,
                model_id=expected_model_id,
                generation_seed=generation_seed,
                game_index=game_index,
            )
        if getattr(verification, "ply_count", None) != saved_plies + extra_plies:
            raise Phase4ResumeError("raw replay extension ply total is inconsistent")

    def _validate_fifo_replay_extension(
        self,
        snapshot: Mapping[str, object],
        verification: object,
        loaded: Phase4CheckpointLoad,
    ) -> None:
        raw_saved = snapshot.get("shards")
        if not isinstance(raw_saved, list):
            raise Phase4ResumeError("checkpoint replay window is missing shards")
        saved = tuple(_replay_shard_from_record(value) for value in raw_saved)
        if (
            tuple(shard.game_id for shard in saved)
            != tuple(snapshot.get("gameIds", ()))
            or any(
                right.sequence != left.sequence + 1
                for left, right in zip(saved, saved[1:])
            )
        ):
            raise Phase4ResumeError("checkpoint replay shard window is not ordered")
        expected_window_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "schemaVersion": 1,
                    "shards": [dict(value) for value in raw_saved],
                }
            )
        )
        try:
            stored_window_sha256 = validate_sha256(
                snapshot.get("windowSha256"),
                "checkpoint replay window sha256",
            )
        except ValueError as error:
            raise Phase4ResumeError(
                "checkpoint replay window hash is invalid"
            ) from error
        if stored_window_sha256 != expected_window_sha256:
            raise Phase4ResumeError("checkpoint replay window hash mismatch")
        current = self.replay.shards
        if any(
            right.sequence != left.sequence + 1
            for left, right in zip(current, current[1:])
        ):
            raise Phase4ResumeError("current replay shard window is not ordered")

        saved_by_sequence = {shard.sequence: shard for shard in saved}
        saved_last = saved[-1].sequence if saved else -1
        retained = tuple(shard for shard in current if shard.sequence <= saved_last)
        extras = tuple(shard for shard in current if shard.sequence > saved_last)
        if any(
            shard.sequence not in saved_by_sequence
            or shard != saved_by_sequence[shard.sequence]
            for shard in retained
        ):
            raise Phase4ResumeError("raw replay retained window differs from checkpoint")
        expected_retained = (
            saved[len(saved) - len(retained) :] if retained else ()
        )
        if retained != expected_retained:
            raise Phase4ResumeError(
                "raw replay did not evict an exact oldest shard prefix"
            )
        first_extra_sequence = saved_last + 1
        if extras and (
            extras[0].sequence != first_extra_sequence
            or any(
                right.sequence != left.sequence + 1
                for left, right in zip(extras, extras[1:])
            )
        ):
            raise Phase4ResumeError("raw replay extension skipped a shard sequence")
        if tuple(current[: len(retained)]) != retained:
            raise Phase4ResumeError("raw replay interleaves old and new shards")

        saved_plies = snapshot.get("plies")
        saved_bytes = snapshot.get("bytes")
        if (
            type(saved_plies) is not int
            or saved_plies < 0
            or type(saved_bytes) is not int
            or saved_bytes < 0
            or saved_plies != sum(shard.ply_count for shard in saved)
            or saved_bytes != sum(shard.byte_size for shard in saved)
        ):
            raise Phase4ResumeError("checkpoint replay window totals are invalid")
        evicted = saved[: len(saved) - len(retained)]
        expected_plies = (
            saved_plies
            - sum(shard.ply_count for shard in evicted)
            + sum(shard.ply_count for shard in extras)
        )
        expected_bytes = (
            saved_bytes
            - sum(shard.byte_size for shard in evicted)
            + sum(shard.byte_size for shard in extras)
        )
        current_bytes = sum(shard.byte_size for shard in current)
        if (
            getattr(verification, "game_count", None) != len(current)
            or getattr(verification, "ply_count", None) != expected_plies
            or current_bytes != expected_bytes
        ):
            raise Phase4ResumeError("raw replay FIFO window totals are inconsistent")

        if not extras:
            if evicted or current != saved:
                raise Phase4ResumeError(
                    "raw replay cannot evict shards without a new generation"
                )
            if (
                sha256_file(self.replay.manifest_path)
                != snapshot.get("manifestSha256")
            ):
                raise Phase4ResumeError("checkpoint replay manifest hash changed")
            return

        if evicted:
            newest_evicted = evicted[-1]
            candidate = (newest_evicted, *current)
            next_sequence = current[-1].sequence + 1 if current else 0
            candidate_manifest_bytes = len(
                self.replay._manifest_bytes(candidate, next_sequence)
            )
            candidate_committed_bytes = candidate_manifest_bytes + sum(
                shard.byte_size for shard in candidate
            )
            if (
                len(candidate) <= self.replay.max_games
                and sum(shard.ply_count for shard in candidate)
                <= self.replay.max_plies
                and candidate_committed_bytes <= self.replay.max_bytes
            ):
                raise Phase4ResumeError(
                    "raw replay evicted a shard before reaching capacity"
                )

        next_generation = loaded.record.generation + 1
        expected_ids = {
            f"phase4-g{next_generation:08d}-game{index:06d}"
            for index in range(self.config.self_play.games)
        }
        extra_ids = tuple(shard.game_id for shard in extras)
        if len(extra_ids) > self.config.self_play.games or any(
            game_id not in expected_ids for game_id in extra_ids
        ):
            raise Phase4ResumeError(
                "raw replay extension contains data beyond the next generation"
            )
        expected_model_id = f"sha256:{loaded.record.sha256}"
        generation_seed = (self.config.seed + next_generation * 1_000_003) % (
            2**63
        )
        for game_id in extra_ids:
            game = _replay_cache_game(self.replay_cache, game_id)
            game_index = int(
                game_id.removeprefix(
                    f"phase4-g{next_generation:08d}-game"
                )
            )
            if (
                game.metadata.get("generation") != next_generation
                or game.metadata.get("modelId") != expected_model_id
                or any(ply.model_id != expected_model_id for ply in game.plies)
            ):
                raise Phase4ResumeError(
                    f"raw replay extension identity mismatch for {game_id}"
                )
            self._validate_game_invocation(
                game,
                generation=next_generation,
                model_id=expected_model_id,
                generation_seed=generation_seed,
                game_index=game_index,
            )

    def _reconcile_reports(self) -> None:
        generations = self.reports.generations()
        for generation in generations:
            self.reports.load(generation)
        if self.loaded is None:
            if generations:
                raise Phase4ResumeError(
                    "generation reports exist without a valid checkpoint"
                )
            return
        checkpoint_generation = self.loaded.record.generation
        if checkpoint_generation == -1:
            if generations:
                raise Phase4ResumeError(
                    "generation reports exist before generation zero"
                )
            initialization = self.loaded.payload.get("generationData")
            if (
                not isinstance(initialization, Mapping)
                or initialization.get("kind") != "initialization"
                or initialization.get("nextGeneration") != 0
            ):
                raise Phase4ResumeError(
                    "generation -1 checkpoint is not an initialization checkpoint"
                )
            return
        if any(generation > checkpoint_generation for generation in generations):
            raise Phase4ResumeError(
                "generation report exists beyond the latest valid checkpoint"
            )
        expected_prior = tuple(range(checkpoint_generation))
        actual_prior = tuple(
            generation for generation in generations if generation < checkpoint_generation
        )
        if actual_prior != expected_prior:
            raise Phase4ResumeError("generation report history contains a gap")
        generation_data = self.loaded.payload["generationData"]
        expected = self._report_payload(self.loaded.record, generation_data)
        if checkpoint_generation in generations:
            stored = self.reports.load(checkpoint_generation).payload.get("report")
            if stored != expected:
                raise Phase4ResumeError(
                    "latest generation report does not match its checkpoint"
                )
        else:
            self.budget_guard.check(
                force=True,
                reserve_bytes=_REPORT_WRITE_RESERVE_BYTES,
            )
            self.reports.write(checkpoint_generation, expected)

    @staticmethod
    def _report_payload(
        record: Phase4CheckpointRecord,
        generation_data: Mapping[str, object],
    ) -> dict[str, object]:
        checkpoint: dict[str, object] = {
            "path": record.data_path.name,
            "bytes": record.byte_size,
            "sha256": record.sha256,
            "step": record.step,
        }
        if (
            record.candidate_model_path is not None
            and record.candidate_model_sha256 is not None
            and record.model_state_sha256 is not None
        ):
            checkpoint["candidateModel"] = {
                "path": record.candidate_model_path.name,
                "bytes": record.candidate_model_byte_size,
                "sha256": record.candidate_model_sha256,
                "modelStateSha256": record.model_state_sha256,
            }
        return {
            "checkpoint": checkpoint,
            "generationData": dict(generation_data),
        }

    @property
    def next_generation(self) -> int:
        return 0 if self.loaded is None else self.loaded.record.generation + 1

    @property
    def current_model_id(self) -> str:
        if self.loaded is not None:
            return f"sha256:{self.loaded.record.sha256}"
        payload = {
            "identity": self.lineage.identity.to_payload(),
            "seed": self.config.seed,
            "kind": "deterministic-random-initialization",
        }
        return f"initial:{sha256_bytes(canonical_json_bytes(payload))}"

    def _validate_game_invocation(
        self,
        game: ReplayGame,
        *,
        generation: int,
        model_id: str,
        generation_seed: int,
        game_index: int,
    ) -> dict[str, object]:
        metadata = game.metadata
        if (
            metadata.get("schedulerSeed") != generation_seed
            or metadata.get("gameIndex") != game_index
            or type(metadata.get("resumedSubset")) is not bool
        ):
            raise Phase4ResumeError(
                f"self-play scheduler identity mismatch for {game.game_id}"
            )
        invocation = _validate_selfplay_invocation(
            metadata.get("invocation"),
            generation=generation,
            model_id=model_id,
            game_index=game_index,
            configured_actors=self.config.self_play.actors,
            simulations=self.config.mcts.simulations,
            temperature_moves=self.config.self_play.temperature_moves,
        )
        scheduled = invocation["scheduledGameIndices"]
        if (
            invocation.get("schedulerSeed") != generation_seed
            or any(index >= self.config.self_play.games for index in scheduled)
            or metadata.get("invocationGameIndex") != scheduled.index(game_index)
            or metadata.get("resumedSubset")
            != (len(scheduled) < self.config.self_play.games)
        ):
            raise Phase4ResumeError(
                f"self-play invocation schedule mismatch for {game.game_id}"
            )
        return invocation

    def run(self, generations: int) -> list[dict[str, object]]:
        if type(generations) is not int or generations <= 0:
            raise ValueError("generations must be a positive integer")
        maximum_generations = getattr(
            self, "maximum_generations", PHASE4D_PILOT_GENERATIONS
        )
        execution_contract = getattr(
            self, "execution_contract", PHASE4D_EXECUTION_CONTRACT
        )
        if self.next_generation + generations > maximum_generations:
            raise Phase4TrainingError(
                f"{execution_contract.stage} is capped at "
                f"{maximum_generations} committed generations"
            )
        summaries = []
        for _ in range(generations):
            summaries.append(self.run_generation(self.next_generation))
        return summaries

    def run_to_target(self, target_generations: int) -> list[dict[str, object]]:
        """Commit through an absolute generation count without overshooting."""

        maximum_generations = getattr(
            self, "maximum_generations", PHASE4D_PILOT_GENERATIONS
        )
        if (
            type(target_generations) is not int
            or not 0 <= target_generations <= maximum_generations
        ):
            raise ValueError(
                "target generations must be an integer in "
                f"[0, {maximum_generations}]"
            )
        if self.next_generation > target_generations:
            raise Phase4ResumeError(
                "run already contains more generations than the requested target"
            )
        remaining = target_generations - self.next_generation
        return [] if remaining == 0 else self.run(remaining)

    def run_generation(self, generation: int) -> dict[str, object]:
        if generation != self.next_generation:
            raise ValueError("generation is not the next uncommitted generation")
        if self.loaded is None:
            raise Phase4ResumeError(
                "generation training requires a committed starting checkpoint"
            )
        started = time.monotonic()
        self.budget_guard.check(force=True)
        generation_seed = (self.config.seed + generation * 1_000_003) % (2**63)
        model_id = self.current_model_id
        game_prefix = f"phase4-g{generation:08d}-game"
        expected_ids = tuple(
            f"{game_prefix}{index:06d}"
            for index in range(self.config.self_play.games)
        )
        existing_ids = set(self.replay.game_ids).intersection(expected_ids)
        for game_id in sorted(existing_ids):
            game = _replay_cache_game(self.replay_cache, game_id)
            game_index = expected_ids.index(game_id)
            if (
                game.metadata.get("generation") != generation
                or game.metadata.get("modelId") != model_id
                or any(ply.model_id != model_id for ply in game.plies)
            ):
                raise Phase4ResumeError(
                    f"partial replay identity mismatch for {game_id}"
                )
            self._validate_game_invocation(
                game,
                generation=generation,
                model_id=model_id,
                generation_seed=generation_seed,
                game_index=game_index,
            )
        missing_indices = [
            index for index, game_id in enumerate(expected_ids) if game_id not in existing_ids
        ]

        evaluator = TorchEvaluator(self.model, self.device, use_amp=True)
        evaluator.enable_native_encoder(self.native_module)
        search_config = MCTSConfig(
            simulations=self.config.mcts.simulations,
            c_base=self.config.mcts.c_base,
            c_init=self.config.mcts.c_init,
            dirichlet_epsilon=self.config.mcts.dirichlet_epsilon,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            temperature=1.0,
            add_root_noise=True,
            seed=generation_seed,
        )
        if missing_indices:
            invocation = _selfplay_invocation(
                generation=generation,
                model_id=model_id,
                scheduler_seed=generation_seed,
                scheduled_game_indices=missing_indices,
                configured_actors=self.config.self_play.actors,
                simulations=self.config.mcts.simulations,
                temperature_moves=self.config.self_play.temperature_moves,
            )
            selfplay_config = SelfPlayConfig(
                mcts=search_config,
                temperature_moves=self.config.self_play.temperature_moves,
                mirror_augmentation=False,
                seed=generation_seed,
                input_channels=self.config.network.input_channels,
                collect_samples=False,
                collect_raw_records=True,
            )

            def persist(local_index: int, game: SelfPlayGame) -> None:
                actual_index = missing_indices[local_index]
                game_id = expected_ids[actual_index]
                replay_game = replay_game_from_selfplay(
                    game,
                    game_id=game_id,
                    generation=generation,
                    model_id=model_id,
                    scheduler_seed=generation_seed,
                    game_index=actual_index,
                    resumed_subset=bool(existing_ids),
                    invocation=invocation,
                )
                replay_game = replace(
                    replay_game,
                    plies=tuple(
                        replace(
                            ply,
                            metadata={
                                "requestedSimulations": self.config.mcts.simulations
                            },
                        )
                        for ply in replay_game.plies
                    ),
                )
                self.budget_guard.check(
                    force=True,
                    reserve_bytes=_REPLAY_GAME_WRITE_RESERVE_BYTES,
                )
                self.replay.append(replay_game)

            play_self_games_scheduled(
                evaluator,
                selfplay_config,
                len(missing_indices),
                actors=min(self.config.self_play.actors, len(missing_indices)),
                search_factory=NativeMCTS,
                progress_callback=lambda _moves, _games: self.budget_guard.check(),
                completed_game_callback=persist,
            )

        if not all(game_id in self.replay.game_ids for game_id in expected_ids):
            raise Phase4TrainingError("self-play generation did not commit every game")
        _publish_progress(
            self.progress_callback,
            stage="replay-cache",
            status="refresh-started",
            generation=generation,
            games=self.replay.game_count,
            plies=self.replay.ply_count,
        )
        cache_refresh = _refresh_replay_cache(self.replay_cache)
        verification = _replay_cache_verification(self.replay_cache)
        if self.loaded is None:
            raise Phase4ResumeError(
                "generation training requires a committed starting checkpoint"
            )
        self._validate_replay_extension(verification, self.loaded)
        _publish_progress(
            self.progress_callback,
            stage="replay-cache",
            status="refresh-completed",
            generation=generation,
            **cache_refresh.to_payload(),
        )
        generation_games = [
            _replay_cache_game(self.replay_cache, game_id)
            for game_id in expected_ids
        ]
        invocations: dict[str, dict[str, object]] = {}
        for game_id in expected_ids:
            game_index = expected_ids.index(game_id)
            game = _replay_cache_game(self.replay_cache, game_id)
            invocation = self._validate_game_invocation(
                game,
                generation=generation,
                model_id=model_id,
                generation_seed=generation_seed,
                game_index=game_index,
            )
            invocation_id = str(invocation["invocationId"])
            summary = invocations.setdefault(
                invocation_id,
                {**invocation, "completedGameIndices": []},
            )
            completed_indices = summary["completedGameIndices"]
            if not isinstance(completed_indices, list):
                raise RuntimeError("self-play invocation summary is invalid")
            completed_indices.append(game_index)
        invocation_summaries = []
        for summary in invocations.values():
            completed_indices = summary["completedGameIndices"]
            if not isinstance(completed_indices, list):
                raise RuntimeError("self-play invocation summary is invalid")
            summary["completedGameIndices"] = sorted(completed_indices)
            invocation_summaries.append(summary)
        maximum_actual_actors = max(
            int(summary["actualActors"]) for summary in invocation_summaries
        )
        selfplay_moves = sum(len(game.plies) for game in generation_games)
        winners = Counter(game.metadata.get("winner") for game in generation_games)
        terminations = Counter(
            game.metadata.get("terminationReason") for game in generation_games
        )

        metrics = []
        challenge_sampling_summaries: list[Mapping[str, object]] = []
        input_gate = ReplayInputPerformanceGate()
        for local_step in range(self.config.training.training_steps):
            sample_seed = (
                generation_seed + 10_000_019 + local_step * 97_409
            ) % (2**63)
            sample_rng = random.Random(sample_seed)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            batch_started = time.monotonic()
            challenge_counts: tuple[int, int] | None = None
            requested_selfplay = self.config.training.batch_size
            challenge_cache = getattr(self, "challenge_cache", None)
            if challenge_cache is not None:
                challenge_settings = self.config.training.challenge_replay
                if challenge_settings is None:
                    raise Phase4TrainingError("challenge replay settings disappeared")
                challenge_counts = mixed_batch_counts(
                    self.config.training.batch_size,
                    challenge_settings.fraction,
                )
                requested_selfplay = challenge_counts[0]
            if isinstance(self.replay_cache, ReplayTensorCache):
                batch = self.replay_cache.load_batch(
                    requested_selfplay,
                    rng=sample_rng,
                    mirror_probability=0.5,
                    device=self.device,
                )
            else:
                samples = self.replay_cache.sample_plies(
                    requested_selfplay,
                    rng=sample_rng,
                )
                batch = self.batch_encoder.encode(samples, rng=sample_rng)
            if challenge_cache is not None and challenge_counts is not None:
                if self.challenge_sampler is None:
                    challenge_batch = challenge_cache.load_batch(
                        challenge_counts[1],
                        rng=sample_rng,
                        mirror_probability=0.5,
                        device=self.device,
                    )
                else:
                    challenge_batch, sampling_summary = self.challenge_sampler.load_batch(
                        challenge_counts[1],
                        rng=sample_rng,
                        mirror_probability=0.5,
                        device=self.device,
                    )
                    challenge_sampling_summaries.append(sampling_summary)
                batch = merge_tensor_batches(batch, challenge_batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            input_seconds = max(0.0, time.monotonic() - batch_started)
            gate_completed = input_gate.observe(input_seconds)
            if gate_completed:
                _publish_progress(
                    self.progress_callback,
                    stage="replay-input-gate",
                    status="completed",
                    generation=generation,
                    **input_gate.to_payload(),
                )
            metrics.append(self.trainer.train_step(batch))
            self.budget_guard.check()

        if not input_gate.initial_completed:
            raise Phase4TrainingError(
                "replay input performance gate did not complete"
            )
        _assert_replay_cache_current(self.replay_cache, deep=True)
        challenge_snapshot = None
        challenge_cache = getattr(self, "challenge_cache", None)
        if challenge_cache is not None:
            challenge_snapshot = challenge_cache.snapshot(deep=True)
        fixed_holdout = None
        model_selection = None
        health_holdout_cache = getattr(self, "health_holdout_cache", None)
        if health_holdout_cache is not None:
            holdout_settings = self.config.training.health_holdout
            if holdout_settings is None:
                raise Phase4TrainingError("health holdout settings disappeared")
            fixed_holdout = {
                **evaluate_fixed_health_holdout(
                    self.model,
                    health_holdout_cache,
                    batch_size=self.config.training.batch_size,
                    device=self.device,
                    seed=int(holdout_settings.dataset_fingerprint[:16], 16),
                    use_amp=self.config.training.use_amp,
                ),
                "provenance": dict(self.health_holdout_provenance or {}),
            }
            model_selection = (
                pure_selfplay_selection(
                    self.reports,
                    generation=generation,
                )
                if self.config.training.learning_stage == "pure-self-play"
                else fixed_holdout_selection(
                    self.reports,
                    generation=generation,
                    current=fixed_holdout,
                )
            )
        verification = _replay_cache_verification(self.replay_cache, deep=True)
        replay_snapshot = _replay_snapshot(self.replay, verification)
        cache_snapshot = _replay_cache_snapshot(self.replay_cache, deep=True)
        cache_matches_replay = (
            (
                cache_snapshot.get("manifestSha256")
                == replay_snapshot["manifestSha256"]
                and cache_snapshot.get("prefixSha256")
                == replay_snapshot["prefixSha256"]
            )
            if isinstance(self.replay_cache, ReplayReconstructionCache)
            else (
                cache_snapshot.get("rawManifestSha256")
                == replay_snapshot["manifestSha256"]
                and cache_snapshot.get("rawWindowSha256")
                == replay_snapshot["windowSha256"]
            )
        )
        if (
            not cache_matches_replay
            or cache_snapshot.get("games") != replay_snapshot["games"]
            or cache_snapshot.get("plies") != replay_snapshot["plies"]
        ):
            raise Phase4ResumeError(
                "replay cache snapshot differs from durable replay"
            )
        generation_data = {
            "schemaVersion": PHASE4_TRAINING_SCHEMA_VERSION,
            "generation": generation,
            "generationSeed": generation_seed,
            "producerModelId": model_id,
            "selfPlay": {
                "games": len(generation_games),
                "moves": selfplay_moves,
                "actors": self.config.self_play.actors,
                "maximumActualActors": maximum_actual_actors,
                "invocations": invocation_summaries,
                "simulations": self.config.mcts.simulations,
                "reusedCompletedGames": len(existing_ids),
                "winners": {str(key): value for key, value in sorted(winners.items())},
                "terminationReasons": {
                    str(key): value for key, value in sorted(terminations.items())
                },
            },
            "training": {
                "trainerStep": self.trainer.step,
                "batchSize": self.config.training.batch_size,
                "learningStage": self.config.training.learning_stage,
                "metrics": _metrics_payload(metrics),
                "inputTiming": input_gate.to_payload(),
                "challengeReplay": (
                    None
                    if challenge_snapshot is None
                    else {
                        "fraction": self.config.training.challenge_replay.fraction,
                        "contractSha256": self.config.training.challenge_replay.contract_sha256,
                        "cache": challenge_snapshot,
                        "sampling": (
                            None
                            if self.challenge_sampler is None
                            else {
                                "algorithm": self.challenge_sampler.settings.algorithm,
                                "indexSha256": self.challenge_sampler.index_sha256,
                                "inventory": self.challenge_sampler.inventory,
                                "batches": challenge_sampling_summaries,
                            }
                        ),
                    }
                ),
            },
            "fixedHealthHoldout": fixed_holdout,
            "modelSelection": model_selection,
            "healthDiagnostics": replay_health_diagnostics(
                [_replay_cache_game(self.replay_cache, game_id) for game_id in expected_ids]
            ),
            "replay": replay_snapshot,
            "replayCache": cache_snapshot,
            "promotion": {
                "status": "not-run",
                "reason": (
                    "promotion gates are orchestrated outside generation training"
                    if getattr(
                        self,
                        "execution_contract",
                        PHASE4D_EXECUTION_CONTRACT,
                    ).formal_promotion
                    else (
                        f"{getattr(self, 'execution_contract', PHASE4D_EXECUTION_CONTRACT).stage} "
                        "is not a formal "
                        "promotion stage"
                    )
                ),
            },
            "resume": {
                "checkpointFallback": list(self.fallback_rejections),
            },
            "elapsedSeconds": time.monotonic() - started,
        }
        self.budget_guard.check(
            force=True,
            reserve_bytes=_checkpoint_write_reserve_bytes(
                self.model,
                self.trainer,
                self.checkpoints.root,
            ),
        )
        record = self.checkpoints.save(
            generation=generation,
            model=self.model,
            trainer=self.trainer,
            generation_data=generation_data,
            replay_snapshot=replay_snapshot,
        )
        self.budget_guard.check(
            force=True,
            reserve_bytes=_REPORT_WRITE_RESERVE_BYTES,
        )
        report = self.reports.write(
            generation,
            self._report_payload(record, generation_data),
        )
        if model_selection is not None:
            reconcile_best_fixed_holdout_candidate(self.run_dir, self.reports)
        self.loaded = Phase4CheckpointLoad(
            record=record,
            payload={
                "generationData": generation_data,
                "replaySnapshot": replay_snapshot,
            },
            rejected_newer=(),
        )
        self.fallback_rejections = ()
        resources = self.budget_guard.check(force=True)
        return {
            "generation": generation,
            "step": self.trainer.step,
            "checkpointPath": str(record.data_path),
            "checkpointSha256": record.sha256,
            "candidateModelPath": str(record.candidate_model_path),
            "candidateModelSha256": record.candidate_model_sha256,
            "modelStateSha256": record.model_state_sha256,
            "reportPath": str(report.path),
            "reportSha256": report.sha256,
            "selfPlayGames": len(generation_games),
            "selfPlayMoves": selfplay_moves,
            "replayGames": verification.game_count,
            "replayPlies": verification.ply_count,
            "replayCache": cache_snapshot,
            "inputTiming": input_gate.to_payload(),
            "fixedHealthHoldout": fixed_holdout,
            "modelSelection": model_selection,
            "resources": resources,
        }


def _effective_config(
    config: ExperimentConfig, device_override: str | None
) -> ExperimentConfig:
    return config if device_override is None else replace(config, device=device_override)


def _validate_phase4_run_intent(run_dir: Path, *, resume: bool) -> None:
    """Reject an ambiguous fresh/resume request without writing run state."""

    manifest_path = run_dir / "run-manifest.json"
    if resume:
        if not manifest_path.exists():
            raise Phase4ResumeError(
                "resume requested but run-manifest.json is missing"
            )
        replay_manifest_path = run_dir / "replay" / "manifest.json"
        if not replay_manifest_path.exists():
            raise Phase4ResumeError(
                "resume requested but replay/manifest.json is missing"
            )
        return
    if manifest_path.exists():
        raise Phase4ResumeError(
            "run manifest already exists; resume must be explicitly enabled"
        )
    protected_roots = (
        run_dir / "checkpoints",
        run_dir / "replay",
        run_dir / "replay-tensor-cache",
        run_dir / "reports",
    )
    if any(path.exists() for path in protected_roots):
        raise Phase4ResumeError(
            "run data exists without an immutable run manifest"
        )


def _import_continuation_replay(
    *,
    workspace: Path,
    config: ExperimentConfig,
    destination: Path,
) -> None:
    initialization = config.initialization
    if initialization.reset_optimizer:
        return
    if initialization.artifact_path is None:
        raise Phase4TrainingError("continuation source artifact is missing")
    artifact = (
        initialization.artifact_path
        if initialization.artifact_path.is_absolute()
        else workspace / initialization.artifact_path
    ).resolve()
    source = artifact.parent.parent / "replay"
    if source.is_symlink() or not (source / "manifest.json").is_file():
        raise Phase4ResumeError("continuation source replay is missing")
    if destination.exists():
        raise Phase4ResumeError("continuation replay destination already exists")
    temporary = destination.with_name(f".{destination.name}.import-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary, symlinks=False)
        if any(path.is_symlink() for path in temporary.rglob("*")):
            raise Phase4ResumeError("continuation replay contains a symlink")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _verify_phase4e_bootstrap_baseline(
    workspace: Path,
    baseline_root: Path,
) -> dict[str, object]:
    # Imported lazily because the baseline verifier audits phase4_pilot, which
    # uses this training module for its frozen Phase 4D compatibility path.
    from .phase4_baseline import verify_bootstrap_baseline

    return verify_bootstrap_baseline(
        workspace_root=workspace,
        baseline_root=baseline_root,
    )


def run_phase4_training(
    *,
    config_path: str | Path,
    workspace_root: str | Path,
    generations: int | None,
    resume: bool,
    device_override: str | None,
    external_gpu_hours: float,
    target_generations: int | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    execution_contract_path: str | Path | None = None,
    bootstrap_baseline_root: str | Path | None = None,
) -> dict[str, object]:
    if (generations is None) == (target_generations is None):
        raise ValueError(
            "exactly one of generations or target_generations must be provided"
        )
    if generations is not None and (
        type(generations) is not int or generations <= 0
    ):
        raise ValueError("generations must be a positive integer")
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    execution_contract = (
        PHASE4D_EXECUTION_CONTRACT
        if execution_contract_path is None
        else load_phase4_execution_contract(
            (
                Path(execution_contract_path)
                if Path(execution_contract_path).is_absolute()
                else workspace / execution_contract_path
            )
        )
    )
    resolved_bootstrap_root = (
        None
        if bootstrap_baseline_root is None
        else (
            Path(bootstrap_baseline_root).resolve()
            if Path(bootstrap_baseline_root).is_absolute()
            else (workspace / bootstrap_baseline_root).resolve()
        )
    )
    bootstrap_evidence = (
        None
        if resolved_bootstrap_root is None
        else _verify_phase4e_bootstrap_baseline(
            workspace,
            resolved_bootstrap_root,
        )
    )
    if execution_contract.stage == PHASE4E_STAGE and bootstrap_evidence is None:
        raise Phase4TrainingError(
            "Phase 4E training requires the E1-reviewed Bootstrap baseline"
        )
    stage_budget = execution_contract.require_frozen_budget()
    if target_generations is not None:
        execution_contract.validate_target_generations(target_generations)
    config = _effective_config(
        load_experiment_config(config_path), device_override
    )
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    validate_phase4_config(config, execution_contract)
    try:
        config.run_dir.resolve().relative_to(control_root)
    except ValueError as error:
        raise ValueError(
            "Phase 4 runDir must be inside training/runs/phase4"
        ) from error
    _static_write_preflight(
        artifact_root=control_root,
        artifact_limit_bytes=config.artifact_limit_bytes,
        reserve_bytes=_CONTROL_WRITE_RESERVE_BYTES,
    )
    control_root.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{execution_contract.stage}-training-"
        f"{config.name}-{os.getpid()}-{time.time_ns()}"
    )
    with RunLock(control_root / GPU_LOCK_NAME, run_id=run_id):
        if execution_contract.stage == PHASE4D_STAGE:
            baseline_provenance = verify_phase4d_baseline_report(control_root)
        elif execution_contract.stage == PHASE4E_STAGE:
            baseline_provenance = verify_phase4e_baseline_report(
                control_root,
                baseline_run_id=stage_budget.baseline_run_id,
                baseline_report_sha256=stage_budget.baseline_report_sha256,
                baseline_global_hours=stage_budget.baseline_global_hours,
                limit_hours=stage_budget.limit_hours,
            )
            if (
                bootstrap_evidence is None
                or baseline_provenance.get("bootstrapBaseline")
                != bootstrap_evidence
            ):
                raise Phase4TrainingError(
                    "Bootstrap baseline differs from the Phase 4E1 decision report"
                )
        elif execution_contract.stage == PHASE4F5_STAGE:
            from .phase4f5_rebaseline import verify_phase4f5_training_baseline

            baseline_provenance = verify_phase4f5_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_RECOVERY_STAGE:
            from .phase4f6_recovery import verify_phase4f6_training_baseline

            baseline_provenance = verify_phase4f6_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F18_PILOT_STAGE:
            from .phase4f18_rebaseline import verify_phase4f18_training_baseline

            baseline_provenance = verify_phase4f18_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F35_PILOT_STAGE:
            from .phase4f35_stage_a2 import verify_phase4f35_training_baseline

            baseline_provenance = verify_phase4f35_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_A:
            from .phase4f19_stage_a import verify_phase4f19_training_baseline

            baseline_provenance = verify_phase4f19_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_A_CONTINUATION:
            from .phase4f21_stage_a import verify_phase4f21_training_baseline

            baseline_provenance = verify_phase4f21_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_B_PILOT:
            from .phase4f23_stage_b import verify_phase4f23_training_baseline

            baseline_provenance = verify_phase4f23_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_B:
            from .phase4f24_stage_b import verify_phase4f24_training_baseline

            baseline_provenance = verify_phase4f24_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_B_CONTINUATION:
            from .phase4f27_stage_b import verify_phase4f27_training_baseline

            baseline_provenance = verify_phase4f27_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        elif execution_contract.stage == PHASE4F_STAGE_B_EXTENDED:
            from .phase4f31_stage_b import verify_phase4f31_training_baseline

            baseline_provenance = verify_phase4f31_training_baseline(
                workspace_root=workspace,
                execution_contract=execution_contract,
                config=config,
            )
        else:
            raise Phase4TrainingError(
                f"unsupported Phase 4 execution stage: {execution_contract.stage}"
            )
        ledger = GpuBudgetLedger(control_root / GPU_LEDGER_NAME)
        stage_ledger = Phase4StageBudgetLedger(
            control_root / PHASE4_STAGE_LEDGER_NAME
        )
        native_backend_sha256 = loaded_native_backend_hash()
        source_fingerprint = fingerprint_sources(workspace)
        if execution_contract.stage == PHASE4E_STAGE:
            _verify_phase4e_training_inputs(
                baseline_provenance,
                config_sha256=fingerprint_config(config).sha256,
                source_sha256=source_fingerprint.sha256,
                native_backend_sha256=native_backend_sha256,
            )
        elif execution_contract.stage in (
            PHASE4F5_STAGE,
            PHASE4F_RECOVERY_STAGE,
            PHASE4F18_PILOT_STAGE,
            PHASE4F35_PILOT_STAGE,
            PHASE4F_STAGE_A,
            PHASE4F_STAGE_A_CONTINUATION,
            PHASE4F_STAGE_B_PILOT,
            PHASE4F_STAGE_B,
            PHASE4F_STAGE_B_CONTINUATION,
            PHASE4F_STAGE_B_EXTENDED,
        ):
            if (
                baseline_provenance.get("sourceSha256")
                != source_fingerprint.sha256
                or baseline_provenance.get("nativeBackendSha256")
                != native_backend_sha256
                or baseline_provenance.get("semanticConfigSha256")
                != fingerprint_config(config).sha256
            ):
                raise Phase4TrainingError(
                    f"{execution_contract.stage} workload differs from its "
                    "CPU preparation"
                )
        stage_ledger.ensure_stage(
            execution_contract.stage,
            float(stage_budget.baseline_global_hours),
            float(stage_budget.limit_hours),
            baseline_run_id=stage_budget.baseline_run_id,
            baseline_report_sha256=stage_budget.baseline_report_sha256,
        )
        stage_ledger.assert_available(execution_contract.stage, ledger.total_hours)
        if ledger.total_hours >= GPU_HOUR_BUDGET:
            raise Phase4ResourceLimitReached(
                f"{GPU_HOUR_BUDGET:g} GPU-hour global budget is exhausted"
            )
        input_gate_evidence: dict[str, object] | None = None
        if config.device in ("auto", "cuda"):
            input_gate_evidence = _load_replay_input_gate_report(
                workspace=workspace,
                control_root=control_root,
                source_sha256=source_fingerprint.sha256,
                native_backend_sha256=native_backend_sha256,
            )
        resource_metadata: dict[str, object] = {
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "stage": execution_contract.stage,
            "stageBaselineGlobalGpuHours": stage_budget.baseline_global_hours,
            "baselineCalibration": baseline_provenance,
            "nativeBackendSha256": native_backend_sha256,
            "executionContract": execution_contract.to_payload(),
            "executionContractSha256": execution_contract.sha256,
        }
        if input_gate_evidence is not None:
            resource_metadata["replayInputGate"] = input_gate_evidence
        if bootstrap_evidence is not None:
            resource_metadata["bootstrapBaseline"] = bootstrap_evidence
        budget = ResourceBudget(
            artifact_limit_bytes=config.artifact_limit_bytes,
            minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
            gpu_hours=stage_budget.limit_hours,
            maximum_generations=execution_contract.maximum_generations,
            metadata=resource_metadata,
        )
        replay_bytes = max(1, config.artifact_limit_bytes * 3 // 4)
        replay_root = config.run_dir / "replay"
        _validate_phase4_run_intent(config.run_dir, resume=resume)

        lineage: Phase4Lineage | None = None
        replay: ReplayStore | None = None
        replay_cache: Phase4ReplayCache | None = None
        if resume:
            lineage = prepare_phase4_lineage(
                workspace_root=workspace,
                run_dir=config.run_dir,
                config=config,
                budget=budget,
                resume=True,
                expected_source_sha256=source_fingerprint.sha256,
                execution_contract=execution_contract,
            )
            replay = ReplayStore(
                replay_root,
                max_games=config.training.replay_capacity,
                max_plies=config.training.replay_capacity,
                max_bytes=replay_bytes,
            )
            replay_cache = _create_phase4_replay_cache(
                replay=replay,
                run_dir=config.run_dir,
                execution_contract=execution_contract,
                source_sha256=source_fingerprint.sha256,
                encoder_sha256=native_backend_sha256,
            )
            _publish_progress(
                progress_callback,
                stage="replay-cache-preflight",
                status="started",
                games=replay.game_count,
                plies=replay.ply_count,
            )
            preflight_refresh = _refresh_replay_cache(replay_cache)
            _publish_progress(
                progress_callback,
                stage="replay-cache-preflight",
                status="completed",
                **preflight_refresh.to_payload(),
            )
        if (
            fingerprint_sources(workspace).sha256 != source_fingerprint.sha256
            or loaded_native_backend_hash() != native_backend_sha256
        ):
            raise Phase4TrainingError(
                "training source or native backend changed during replay preflight"
            )
        if input_gate_evidence is not None:
            verified_evidence = _load_replay_input_gate_report(
                workspace=workspace,
                control_root=control_root,
                source_sha256=source_fingerprint.sha256,
                native_backend_sha256=native_backend_sha256,
            )
            if verified_evidence != input_gate_evidence:
                raise Phase4TrainingError(
                    "replay input gate evidence changed during replay preflight"
                )
        status = "failed"
        summary: dict[str, object] | None = None
        started = time.monotonic()
        ledger.begin(run_id, external_hours=external_gpu_hours)
        try:
            guard = _RuntimeBudgetGuard(
                control_root=control_root,
                artifact_limit_bytes=config.artifact_limit_bytes,
                ledger=ledger,
                stage_ledger=stage_ledger,
                ledger_run_id=run_id,
                started=started,
                stage=execution_contract.stage,
            )
            guard.check(force=True)
            device = resolve_device(config.device)
            if not resume:
                lineage = prepare_phase4_lineage(
                    workspace_root=workspace,
                    run_dir=config.run_dir,
                    config=config,
                    budget=budget,
                    resume=False,
                    expected_source_sha256=source_fingerprint.sha256,
                    execution_contract=execution_contract,
                )
                _static_write_preflight(
                    artifact_root=control_root,
                    artifact_limit_bytes=config.artifact_limit_bytes,
                    reserve_bytes=_REPLAY_MANIFEST_WRITE_RESERVE_BYTES,
                )
                try:
                    _import_continuation_replay(
                        workspace=workspace,
                        config=config,
                        destination=replay_root,
                    )
                except BaseException:
                    if (
                        not config.initialization.reset_optimizer
                        and not replay_root.exists()
                        and not (config.run_dir / "checkpoints").exists()
                        and not (config.run_dir / "reports").exists()
                    ):
                        (config.run_dir / "run-manifest.json").unlink(missing_ok=True)
                    raise
                replay = ReplayStore(
                    replay_root,
                    max_games=config.training.replay_capacity,
                    max_plies=config.training.replay_capacity,
                    max_bytes=replay_bytes,
                )
                replay_cache = _create_phase4_replay_cache(
                    replay=replay,
                    run_dir=config.run_dir,
                    execution_contract=execution_contract,
                    source_sha256=source_fingerprint.sha256,
                    encoder_sha256=native_backend_sha256,
                )
                _publish_progress(
                    progress_callback,
                    stage="replay-cache-preflight",
                    status="started",
                    games=replay.game_count,
                    plies=replay.ply_count,
                )
                preflight_refresh = _refresh_replay_cache(replay_cache)
                _publish_progress(
                    progress_callback,
                    stage="replay-cache-preflight",
                    status="completed",
                    **preflight_refresh.to_payload(),
                )
            if lineage is None or replay is None or replay_cache is None:
                raise RuntimeError("Phase 4 training preflight is incomplete")
            if (
                fingerprint_sources(workspace).sha256
                != source_fingerprint.sha256
                or loaded_native_backend_hash() != native_backend_sha256
            ):
                raise Phase4TrainingError(
                    "training source or native backend changed before driver start"
                )
            if input_gate_evidence is not None:
                verified_evidence = _load_replay_input_gate_report(
                    workspace=workspace,
                    control_root=control_root,
                    source_sha256=source_fingerprint.sha256,
                    native_backend_sha256=native_backend_sha256,
                )
                if verified_evidence != input_gate_evidence:
                    raise Phase4TrainingError(
                        "replay input gate evidence changed before driver start"
                    )
            driver = Phase4TrainingDriver(
                workspace_root=workspace,
                control_root=control_root,
                config=config,
                device=device,
                lineage=lineage,
                budget_guard=guard,
                replay=replay,
                replay_cache=replay_cache,
                progress_callback=progress_callback,
                execution_contract=execution_contract,
                bootstrap_baseline_root=resolved_bootstrap_root,
            )
            if resume:
                driver.restore()
            else:
                driver.initialize_new_run()
            if target_generations is not None:
                summaries = driver.run_to_target(target_generations)
            else:
                if generations is None:
                    raise RuntimeError("incremental generation count is missing")
                summaries = driver.run(generations)
            status = "completed"
            summary = {
                "schemaVersion": PHASE4_TRAINING_SCHEMA_VERSION,
                "status": status,
                "experiment": config.name,
                "device": str(device),
                "resumed": resume,
                "targetGenerations": target_generations,
                "generations": summaries,
                "manifestPath": str(config.run_dir / "run-manifest.json"),
                "baselineCalibration": baseline_provenance,
                "executionContractSha256": execution_contract.sha256,
                "stage": execution_contract.stage,
            }
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            final_global_hours = ledger.update(
                run_id, elapsed_seconds=elapsed, status=status
            )
        if summary is None:
            raise RuntimeError("Phase 4 training completed without a summary")
        summary["globalGpuHours"] = final_global_hours
        summary["stageGpu"] = stage_ledger.usage(
            execution_contract.stage, final_global_hours
        )
        return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run durable local Phase 4 Quoridor training"
    )
    parser.add_argument(
        "--config",
        default="training/configs/phase4-bootstrap.json",
        help="Phase 4 experiment JSON path",
    )
    parser.add_argument(
        "--execution-contract",
        help=(
            "versioned Phase 4 execution contract; omitted only for the "
            "frozen Phase 4D compatibility path"
        ),
    )
    parser.add_argument(
        "--bootstrap-baseline",
        help="verified Phase 4E bootstrap-baseline directory for a fresh lineage",
    )
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--generations",
        type=int,
        help="additional generations to commit (default: 1)",
    )
    workload.add_argument(
        "--target-generations",
        type=int,
        help="absolute committed generation count; equal target reconciles only",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="strictly resume an existing run lineage",
    )
    parser.add_argument(
        "--gpu-hours-used",
        type=float,
        default=0.0,
        help="monotonic Phase 4 GPU usage outside the shared ledger",
    )
    parser.add_argument(
        "--replay-input-gate",
        action="store_true",
        help="run the frozen CPU-only production replay-input gate",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help=argparse.SUPPRESS,
    )
    return parser


def _raise_interrupt(_signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _print_progress(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.replay_input_gate and (
        args.generations is not None
        or args.target_generations is not None
        or args.resume
        or args.device is not None
        or args.gpu_hours_used != 0.0
        or args.bootstrap_baseline is not None
    ):
        raise ValueError(
            "--replay-input-gate cannot be combined with training options"
        )
    generations = args.generations
    if (
        not args.replay_input_gate
        and generations is None
        and args.target_generations is None
    ):
        generations = 1
    if generations is not None and generations <= 0:
        raise ValueError("--generations must be positive")
    previous_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        if args.replay_input_gate:
            artifact_limit_bytes = DEFAULT_ARTIFACT_LIMIT_BYTES
            if args.execution_contract is not None:
                workspace = Path(args.workspace_root).resolve()
                execution_path = Path(args.execution_contract)
                config_path = Path(args.config)
                execution_contract = load_phase4_execution_contract(
                    execution_path
                    if execution_path.is_absolute()
                    else workspace / execution_path
                )
                config = load_experiment_config(
                    config_path if config_path.is_absolute() else workspace / config_path
                )
                validate_phase4_config(config, execution_contract)
                artifact_limit_bytes = config.artifact_limit_bytes
            summary = run_phase4_replay_input_gate(
                workspace_root=args.workspace_root,
                artifact_limit_bytes=artifact_limit_bytes,
                progress_callback=_print_progress,
            )
        else:
            summary = run_phase4_training(
                config_path=args.config,
                workspace_root=args.workspace_root,
                generations=generations,
                resume=args.resume,
                device_override=args.device,
                external_gpu_hours=args.gpu_hours_used,
                target_generations=args.target_generations,
                progress_callback=_print_progress,
                execution_contract_path=args.execution_contract,
                bootstrap_baseline_root=args.bootstrap_baseline,
            )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "message": (
                        "CPU replay-input evidence was not committed"
                        if args.replay_input_gate
                        else (
                            "completed replay games are durable; uncommitted optimizer "
                            "updates will restart from the last valid checkpoint"
                        )
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PHASE4_TRAINING_SCHEMA_VERSION",
    "PHASE4_WARM_ARTIFACT_SCHEMA_VERSION",
    "PHASE4D_PILOT_GENERATIONS",
    "REPLAY_CACHE_ALGORITHM_VERSION",
    "Phase4TrainingError",
    "Phase4ResumeError",
    "Phase4ResourceLimitReached",
    "Phase4Lineage",
    "ReplayBatchEncoder",
    "ReplayCacheRefresh",
    "ReplayReconstructionCache",
    "ReplayInputPerformanceGate",
    "replay_game_from_selfplay",
    "validate_selfplay_invocation",
    "load_phase4_warm_artifact",
    "load_phase4_champion_artifact",
    "validate_phase4_config",
    "prepare_phase4_lineage",
    "Phase4TrainingDriver",
    "run_phase4_replay_input_gate",
    "run_phase4_training",
    "create_parser",
    "main",
]
