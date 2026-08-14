"""Strict, resumable Phase 4D pretraining on isolated teacher replay.

The production entry point is deliberately frozen.  Tests may exercise the
same training/evaluation primitives with a smaller contract, but the CLI only
accepts the 192x16, batch-512, 512-step CUDA AMP workload.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .calibration import (
    GPU_HOUR_BUDGET,
    GPU_LEDGER_NAME,
    GPU_LOCK_NAME,
    GpuBudgetLedger,
)
from .checkpoint import capture_rng_state, restore_rng_state
from .config import seed_everything
from .game import get_winner
from .native_mcts import load_native_module
from .network import (
    ACTION_SIZE,
    PHASE4_INPUT_CHANNELS,
    NetworkConfig,
    PolicyValueNetwork,
)
from .phase4_budget import (
    PHASE4_CONTROL_RELATIVE,
    PHASE4_STAGE_LEDGER_NAME,
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
    Phase4StageBudgetLedger,
    verify_phase4d_baseline_report,
)
from .phase4_checkpoint import (
    CheckpointIdentity,
    Phase4CheckpointError,
    Phase4CheckpointLoad,
    Phase4CheckpointStore,
    canonical_model_state_sha256,
)
from .phase4_gates import find_valid_memory_smoke, verify_memory_smoke_evidence
from .paired_arena import loaded_native_backend_hash
from .phase4_training import ReplayBatchEncoder, _replay_prefix_sha256
from .replay_store import ReplaySample, ReplayStore, reconstruct_game
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ImmutableArtifactExistsError,
    ResourceBudget,
    RunLock,
    RunManifestError,
    RunManifestRecord,
    atomic_create_bytes,
    build_run_manifest,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)
from .storage import ArtifactBudgetError, directory_size
from .teacher_data import (
    TEACHER_DATASET_SCHEMA_VERSION,
    TeacherDataConfig,
    TeacherDataContractError,
    _merge_histograms,
    _verify_teacher_game,
)
from .trainer import AlphaZeroTrainer, TrainMetrics


SCHEMA_VERSION = 1
WARM_ARTIFACT_SCHEMA_VERSION = 1
RUN_KIND = "phase4d-teacher-pretraining"
DEFAULT_TEACHER_ROOT = Path("training/runs/phase4/teacher-data")
DEFAULT_RUN_ROOT = Path("training/runs/phase4/teacher-pretraining-v2")
WARM_ARTIFACT_NAME = "warm-model.pt"
REPORT_NAME = "report.json"
COMPLETION_MANIFEST_NAME = "completion-manifest.json"
SUMMARY_NAME = "summary.json"
CHECKPOINT_KEEP_COUNT = 2
RESOURCE_CHECK_INTERVAL_SECONDS = 30.0
LEDGER_HEARTBEAT_SECONDS = 5.0
FINALIZATION_RESERVE_BYTES = 512 * 1024**2
FINAL_METADATA_RESERVE_BYTES = 4 * 1024**2
CHECKPOINT_FIXED_RESERVE_BYTES = 8 * 1024**2
HOLDOUT_AUGMENTATION_SEED = 0x4D484F4C444F5554
PREENCODED_CACHE_SCHEMA_VERSION = 1
PREENCODED_CACHE_ALGORITHM_VERSION = 1
PREENCODED_CACHE_MAX_BYTES = 1024**3
PREENCODED_CACHE_CHUNK_SIZE = 512
PROGRESS_GAME_INTERVAL = 100
PROGRESS_POSITION_INTERVAL = 5_000
TRAINING_PROGRESS_INTERVAL = 16
INPUT_PERFORMANCE_GATE_STEPS = 8
INPUT_PERFORMANCE_MAX_MEAN_SECONDS = 1.0
PRODUCTION_BASE_MODEL_SHA256 = (
    "51e500bbc612222bdc4902438af9dbfcf367c95cc523d5384309403e8b3ba010"
)

_SOURCE_PATHS = (
    "training/quoridor/teacher_pretraining.py",
    "training/quoridor/teacher_data.py",
    "training/quoridor/replay_store.py",
    "training/quoridor/phase4_training.py",
    "training/quoridor/phase4_checkpoint.py",
    "training/quoridor/phase4_gates.py",
    "training/quoridor/paired_arena.py",
    "training/quoridor/phase4_budget.py",
    "training/quoridor/training_memory_smoke.py",
    "training/quoridor/trainer.py",
    "training/quoridor/network.py",
    "training/quoridor/native_mcts.py",
    "training/quoridor/codec.py",
    "training/quoridor/game.py",
    "training/quoridor/run_manifest.py",
    "training/quoridor/calibration.py",
    "native/include",
    "native/src",
    "native/CMakeLists.txt",
    "scripts/pretrain-phase4d-teacher.sh",
)


class TeacherPretrainingError(RuntimeError):
    """Base class for Phase 4D teacher pretraining failures."""


class TeacherPretrainingContractError(TeacherPretrainingError):
    """Raised when persisted input or output violates a frozen contract."""


class TeacherPretrainingResourceError(TeacherPretrainingError):
    """Raised when a global/stage/disk/artifact stop line is reached."""


ProgressCallback = Callable[[Mapping[str, object]], None]


def _publish_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    status: str,
    **fields: object,
) -> None:
    if callback is None:
        return
    callback(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4d-teacher-pretraining-progress",
            "stage": stage,
            "status": status,
            **fields,
        }
    )


@dataclass(frozen=True, slots=True)
class TeacherPretrainingContract:
    seed: int = 20260714
    input_channels: int = PHASE4_INPUT_CHANNELS
    channels: int = 192
    residual_blocks: int = 16
    se_reduction: int = 8
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 256
    batch_size: int = 512
    optimizer_steps: int = 512
    evaluation_interval: int = 64
    mirror_probability: float = 0.5
    learning_rate: float = 0.0003
    minimum_learning_rate: float = 0.00003
    weight_decay: float = 0.0001
    max_grad_norm: float = 5.0
    use_amp: bool = True

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "input_channels",
            "channels",
            "residual_blocks",
            "se_reduction",
            "policy_channels",
            "value_channels",
            "value_hidden",
            "batch_size",
            "optimizer_steps",
            "evaluation_interval",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.optimizer_steps % self.evaluation_interval:
            raise ValueError("optimizer_steps must be divisible by evaluation_interval")
        for name in (
            "mirror_probability",
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0, 1]")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and initial positive")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate must not exceed learning_rate")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("weight_decay/max_grad_norm are invalid")
        if type(self.use_amp) is not bool:
            raise TypeError("use_amp must be boolean")

    @classmethod
    def production(cls) -> "TeacherPretrainingContract":
        return cls()

    def assert_production(self) -> None:
        if self != type(self).production():
            raise TeacherPretrainingContractError(
                "the Phase 4D CLI contract is frozen at 192x16, batch 512, "
                "512 optimizer steps, and 64-step holdout intervals"
            )

    def network_config(self) -> NetworkConfig:
        return NetworkConfig(
            input_channels=self.input_channels,
            channels=self.channels,
            blocks=self.residual_blocks,
            se_reduction=self.se_reduction,
            policy_channels=self.policy_channels,
            value_channels=self.value_channels,
            value_hidden=self.value_hidden,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "network": self.network_config().to_dict(),
            "batchSize": self.batch_size,
            "optimizerSteps": self.optimizer_steps,
            "evaluationInterval": self.evaluation_interval,
            "holdoutEvaluations": self.optimizer_steps // self.evaluation_interval,
            "mirrorProbability": self.mirror_probability,
            "optimizer": {
                "kind": "AdamW",
                "learningRate": self.learning_rate,
                "minimumLearningRate": self.minimum_learning_rate,
                "schedule": "CosineAnnealingLR",
                "scheduleSteps": self.optimizer_steps,
                "weightDecay": self.weight_decay,
                "maxGradNorm": self.max_grad_norm,
            },
            "precision": {
                "ampRequired": self.use_amp,
                "autocastDtype": "float16" if self.use_amp else None,
                "parameterDtype": "float32",
            },
            "loss": {
                "policy": "cross-entropy",
                "value": "mean-squared-error",
                "total": "policy+value",
                "selection": "strict-minimum-fixed-holdout-total-loss",
            },
            "preencodedCache": {
                "schemaVersion": PREENCODED_CACHE_SCHEMA_VERSION,
                "algorithmVersion": PREENCODED_CACHE_ALGORITHM_VERSION,
                "layout": "cpu-float32-dual-horizontal-orientation",
                "indexOrder": "replay-global-ply-order",
                "maximumBytes": PREENCODED_CACHE_MAX_BYTES,
                "trainingSampling": "derived-seed-random-sample-without-replacement",
                "mirrorSampling": "derived-seed-bernoulli-per-selected-ply",
                "holdoutOrderSeed": HOLDOUT_AUGMENTATION_SEED,
            },
            "checkpointing": {
                "retained": CHECKPOINT_KEEP_COUNT,
                "contains": "current+best+adam+trainer+rng",
                "writeReserve": "2x-max(tensor-bytes,observed-checkpoint)+8MiB",
                "interruptBoundary": "completed-optimizer-step",
            },
        }


@dataclass(frozen=True, slots=True)
class TeacherDataset:
    root: Path
    train: ReplayStore
    holdout: ReplayStore
    dataset_fingerprint: str
    teacher_manifest_sha256: str
    run_manifest_sha256: str
    config_sha256: str
    train_manifest_sha256: str
    holdout_manifest_sha256: str
    train_positions: int
    holdout_positions: int

    def snapshot(self) -> dict[str, object]:
        return {
            "teacherDatasetSha256": self.dataset_fingerprint,
            "teacherManifestSha256": self.teacher_manifest_sha256,
            "teacherRunManifestSha256": self.run_manifest_sha256,
            "teacherConfigSha256": self.config_sha256,
            "trainManifestSha256": self.train_manifest_sha256,
            "holdoutManifestSha256": self.holdout_manifest_sha256,
            "trainPositions": self.train_positions,
            "holdoutPositions": self.holdout_positions,
        }


@dataclass(frozen=True, slots=True)
class PreencodedReplaySplit:
    """CPU float32 tensors in raw ReplayStore global-ply order."""

    split: str
    states: Tensor
    policies: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        if self.split not in {"train", "holdout"}:
            raise ValueError("preencoded split must be train or holdout")
        if (
            self.states.device.type != "cpu"
            or self.policies.device.type != "cpu"
            or self.values.device.type != "cpu"
        ):
            raise ValueError("preencoded cache tensors must remain on CPU")
        if (
            self.states.dtype != torch.float32
            or self.policies.dtype != torch.float32
            or self.values.dtype != torch.float32
        ):
            raise ValueError("preencoded cache tensors must use float32")
        if self.states.ndim != 5 or self.states.shape[1:] != (
            2,
            PHASE4_INPUT_CHANNELS,
            9,
            9,
        ):
            raise ValueError("preencoded state cache shape is invalid")
        if self.policies.ndim != 3 or self.policies.shape[1:] != (2, ACTION_SIZE):
            raise ValueError("preencoded policy cache shape is invalid")
        if self.values.ndim != 1:
            raise ValueError("preencoded value cache shape is invalid")
        if not (
            self.states.shape[0] == self.policies.shape[0] == self.values.shape[0]
        ):
            raise ValueError("preencoded cache tensor lengths differ")
        if self.values.numel() == 0:
            raise ValueError("preencoded cache cannot be empty")
        if not self.states.is_contiguous() or not self.policies.is_contiguous():
            raise ValueError("preencoded state/policy caches must be contiguous")

    @property
    def position_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.states, self.policies, self.values)
        )

    def batch(
        self,
        indices: Sequence[int],
        mirrored: Sequence[bool],
        *,
        device: str | torch.device,
    ) -> dict[str, Tensor]:
        selected = tuple(indices)
        orientations = tuple(mirrored)
        if not selected or len(selected) != len(orientations):
            raise ValueError("cache batch indices/orientations are invalid")
        if any(
            type(index) is not int or not 0 <= index < self.position_count
            for index in selected
        ):
            raise IndexError("cache batch index is outside the split")
        if any(type(value) is not bool for value in orientations):
            raise TypeError("cache mirror decisions must be boolean")
        index_tensor = torch.tensor(selected, dtype=torch.long)
        orientation_tensor = torch.tensor(orientations, dtype=torch.long)
        target = torch.device(device)
        return {
            "states": self.states[index_tensor, orientation_tensor].clone().to(target),
            "policies": self.policies[index_tensor, orientation_tensor].clone().to(target),
            "values": self.values[index_tensor].clone().to(target),
        }


@dataclass(frozen=True, slots=True)
class TeacherPreencodedCache:
    algorithm_version: int
    dataset_fingerprint: str
    train: PreencodedReplaySplit
    holdout: PreencodedReplaySplit
    build_seconds: float

    def __post_init__(self) -> None:
        if self.algorithm_version != PREENCODED_CACHE_ALGORITHM_VERSION:
            raise ValueError("unsupported teacher preencoded-cache algorithm")
        if self.train.split != "train" or self.holdout.split != "holdout":
            raise ValueError("teacher preencoded-cache splits are reversed")
        if self.byte_size > PREENCODED_CACHE_MAX_BYTES:
            raise TeacherPretrainingResourceError(
                "teacher preencoded cache exceeds the 1 GiB CPU limit"
            )
        if not math.isfinite(self.build_seconds) or self.build_seconds < 0:
            raise ValueError("cache build_seconds must be finite and non-negative")

    @property
    def byte_size(self) -> int:
        return self.train.byte_size + self.holdout.byte_size

    def snapshot(self) -> dict[str, object]:
        return {
            "schemaVersion": PREENCODED_CACHE_SCHEMA_VERSION,
            "algorithmVersion": self.algorithm_version,
            "datasetSha256": self.dataset_fingerprint,
            "bytes": self.byte_size,
            "maximumBytes": PREENCODED_CACHE_MAX_BYTES,
            "trainPositions": self.train.position_count,
            "holdoutPositions": self.holdout.position_count,
            "buildSeconds": self.build_seconds,
        }


@dataclass(frozen=True, slots=True)
class CompletedPretrainingBundle:
    summary: Mapping[str, object]
    report: Mapping[str, object]
    artifact_metadata: Mapping[str, object]
    run_manifest: RunManifestRecord
    completion_manifest: Mapping[str, object]
    teacher_dataset: TeacherDataset


@dataclass(slots=True)
class MetricAccumulator:
    count: int = 0
    loss_sum: float = 0.0
    policy_sum: float = 0.0
    value_sum: float = 0.0
    grad_sum: float = 0.0
    data_batch_seconds_sum: float = 0.0
    data_batch_seconds_maximum: float = 0.0
    optimizer_seconds_sum: float = 0.0
    optimizer_seconds_maximum: float = 0.0
    first_loss: float | None = None
    last_loss: float | None = None
    minimum_loss: float | None = None
    maximum_loss: float | None = None

    def add(
        self,
        metric: TrainMetrics,
        *,
        data_batch_seconds: float = 0.0,
        optimizer_seconds: float = 0.0,
    ) -> None:
        values = (
            metric.loss,
            metric.policy_loss,
            metric.value_loss,
            metric.grad_norm,
            data_batch_seconds,
            optimizer_seconds,
        )
        if not all(math.isfinite(float(value)) for value in values) or (
            data_batch_seconds < 0 or optimizer_seconds < 0
        ):
            raise TeacherPretrainingContractError("training metrics are non-finite")
        next_loss_sum = self.loss_sum + metric.loss
        next_policy_sum = self.policy_sum + metric.policy_loss
        next_value_sum = self.value_sum + metric.value_loss
        next_grad_sum = self.grad_sum + metric.grad_norm
        next_data_seconds = self.data_batch_seconds_sum + data_batch_seconds
        next_optimizer_seconds = self.optimizer_seconds_sum + optimizer_seconds
        next_first = metric.loss if self.first_loss is None else self.first_loss
        next_minimum = (
            metric.loss if self.minimum_loss is None else min(self.minimum_loss, metric.loss)
        )
        next_maximum = (
            metric.loss if self.maximum_loss is None else max(self.maximum_loss, metric.loss)
        )
        self.loss_sum = next_loss_sum
        self.policy_sum = next_policy_sum
        self.value_sum = next_value_sum
        self.grad_sum = next_grad_sum
        self.data_batch_seconds_sum = next_data_seconds
        self.data_batch_seconds_maximum = max(
            self.data_batch_seconds_maximum,
            data_batch_seconds,
        )
        self.optimizer_seconds_sum = next_optimizer_seconds
        self.optimizer_seconds_maximum = max(
            self.optimizer_seconds_maximum,
            optimizer_seconds,
        )
        self.first_loss = next_first
        self.last_loss = metric.loss
        self.minimum_loss = next_minimum
        self.maximum_loss = next_maximum
        # The cursor is committed last so an asynchronous signal cannot make a
        # partially updated accumulator look checkpoint-consistent.
        self.count += 1

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "lossSum": self.loss_sum,
            "policyLossSum": self.policy_sum,
            "valueLossSum": self.value_sum,
            "gradientNormSum": self.grad_sum,
            "dataBatchSecondsSum": self.data_batch_seconds_sum,
            "dataBatchSecondsMaximum": self.data_batch_seconds_maximum,
            "optimizerSecondsSum": self.optimizer_seconds_sum,
            "optimizerSecondsMaximum": self.optimizer_seconds_maximum,
            "firstLoss": self.first_loss,
            "lastLoss": self.last_loss,
            "minimumLoss": self.minimum_loss,
            "maximumLoss": self.maximum_loss,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "MetricAccumulator":
        if not isinstance(payload, Mapping):
            raise TeacherPretrainingContractError("checkpoint metrics are missing")
        expected = {
            "count",
            "lossSum",
            "policyLossSum",
            "valueLossSum",
            "gradientNormSum",
            "dataBatchSecondsSum",
            "dataBatchSecondsMaximum",
            "optimizerSecondsSum",
            "optimizerSecondsMaximum",
            "firstLoss",
            "lastLoss",
            "minimumLoss",
            "maximumLoss",
        }
        if set(payload) != expected:
            raise TeacherPretrainingContractError("checkpoint metrics schema is invalid")
        count = payload.get("count")
        if type(count) is not int or count < 0:
            raise TeacherPretrainingContractError("checkpoint metric count is invalid")
        result = cls(count=count)
        for field, key in (
            ("loss_sum", "lossSum"),
            ("policy_sum", "policyLossSum"),
            ("value_sum", "valueLossSum"),
            ("grad_sum", "gradientNormSum"),
            ("data_batch_seconds_sum", "dataBatchSecondsSum"),
            ("data_batch_seconds_maximum", "dataBatchSecondsMaximum"),
            ("optimizer_seconds_sum", "optimizerSecondsSum"),
            ("optimizer_seconds_maximum", "optimizerSecondsMaximum"),
            ("first_loss", "firstLoss"),
            ("last_loss", "lastLoss"),
            ("minimum_loss", "minimumLoss"),
            ("maximum_loss", "maximumLoss"),
        ):
            value = payload.get(key)
            if field in {
                "data_batch_seconds_sum",
                "data_batch_seconds_maximum",
                "optimizer_seconds_sum",
                "optimizer_seconds_maximum",
            } and value is None:
                raise TeacherPretrainingContractError(
                    f"checkpoint metric {key} is missing"
                )
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TeacherPretrainingContractError(
                    f"checkpoint metric {key} is invalid"
                )
            setattr(result, field, None if value is None else float(value))
        if (
            result.data_batch_seconds_sum < 0
            or result.data_batch_seconds_maximum < 0
            or result.optimizer_seconds_sum < 0
            or result.optimizer_seconds_maximum < 0
            or result.data_batch_seconds_maximum > result.data_batch_seconds_sum
            or result.optimizer_seconds_maximum > result.optimizer_seconds_sum
            or (
                count == 0
                and (
                    result.data_batch_seconds_sum != 0
                    or result.data_batch_seconds_maximum != 0
                    or result.optimizer_seconds_sum != 0
                    or result.optimizer_seconds_maximum != 0
                )
            )
        ):
            raise TeacherPretrainingContractError(
                "checkpoint timing metrics are invalid"
            )
        if count == 0 and any(
            value is not None
            for value in (
                result.first_loss,
                result.last_loss,
                result.minimum_loss,
                result.maximum_loss,
            )
        ):
            raise TeacherPretrainingContractError("empty checkpoint metrics are inconsistent")
        return result

    def summary(self) -> dict[str, object]:
        if self.count <= 0:
            raise TeacherPretrainingContractError("no optimizer metrics were recorded")
        return {
            "steps": self.count,
            "loss": {
                "first": self.first_loss,
                "last": self.last_loss,
                "minimum": self.minimum_loss,
                "maximum": self.maximum_loss,
                "mean": self.loss_sum / self.count,
            },
            "policyLossMean": self.policy_sum / self.count,
            "valueLossMean": self.value_sum / self.count,
            "gradientNormMean": self.grad_sum / self.count,
            "inputTiming": {
                "dataBatchSecondsMean": self.data_batch_seconds_sum / self.count,
                "dataBatchSecondsMaximum": self.data_batch_seconds_maximum,
                "optimizerSecondsMean": self.optimizer_seconds_sum / self.count,
                "optimizerSecondsMaximum": self.optimizer_seconds_maximum,
            },
        }


@dataclass(slots=True)
class InputPerformanceGate:
    """Bound new-step cache/H2D latency before substantial GPU budget is spent."""

    count: int = 0
    seconds_sum: float = 0.0

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
            raise TeacherPretrainingContractError(
                "training input timing is non-finite"
            )
        if self.count >= INPUT_PERFORMANCE_GATE_STEPS:
            return False
        self.seconds_sum += float(seconds)
        self.count += 1
        if (
            self.count == INPUT_PERFORMANCE_GATE_STEPS
            and self.mean_seconds > INPUT_PERFORMANCE_MAX_MEAN_SECONDS
        ):
            raise TeacherPretrainingResourceError(
                "cached training input/H2D mean exceeded 1.0s across the "
                "first 8 new optimizer steps"
            )
        return self.count == INPUT_PERFORMANCE_GATE_STEPS


class CheckpointModels(nn.Module):
    """Persist the live optimizer model and the best holdout model together."""

    def __init__(self, current: PolicyValueNetwork) -> None:
        super().__init__()
        self.current = current
        self.best = copy.deepcopy(current).cpu()
        self.best.requires_grad_(False)

    def update_best(self) -> None:
        state = {
            key: tensor.detach().cpu()
            for key, tensor in self.current.state_dict().items()
        }
        self.best.load_state_dict(state, strict=True)


def _load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise TeacherPretrainingContractError(f"{label} must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TeacherPretrainingContractError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise TeacherPretrainingContractError(f"{label} must be a JSON object")
    return value


def _canonical_teacher_root(
    workspace: Path,
    value: str | Path,
) -> Path:
    requested = Path(value)
    requested = requested if requested.is_absolute() else workspace / requested
    canonical = (workspace / DEFAULT_TEACHER_ROOT).resolve()
    if requested.is_symlink() or requested.resolve() != canonical:
        raise TeacherPretrainingContractError(
            "teacher root must be the canonical "
            "training/runs/phase4/teacher-data directory"
        )
    return canonical


def _verify_core_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    try:
        stored = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise TeacherPretrainingContractError(str(error)) from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != stored:
        raise TeacherPretrainingContractError(f"{label} self hash mismatch")
    return stored


def _teacher_config(payload: object) -> TeacherDataConfig:
    if not isinstance(payload, Mapping):
        raise TeacherPretrainingContractError("teacher config payload is invalid")
    search = payload.get("classicSearch")
    policy = payload.get("policy")
    if not isinstance(search, Mapping) or not isinstance(policy, Mapping):
        raise TeacherPretrainingContractError("teacher search/policy config is invalid")
    try:
        return TeacherDataConfig(
            target_positions=payload.get("targetPositions"),
            seed=payload.get("seed"),
            node_budget=search.get("nodeBudget"),
            max_depth=search.get("maxDepth"),
            exploration_probability=payload.get("explorationProbability"),
            opening_depths=tuple(payload.get("openingDepths", ())),
            request_timeout_seconds=payload.get("requestTimeoutSeconds"),
            maximum_attempts=payload.get("maximumAttempts"),
            policy_scale=policy.get("scale"),
            policy_floor_units=policy.get("floorUnits"),
        )
    except (TypeError, ValueError) as error:
        raise TeacherPretrainingContractError("teacher config cannot be reconstructed") from error


def load_teacher_dataset(
    root: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> TeacherDataset:
    """Require a completed dataset and deeply verify both isolated stores."""

    dataset_root = Path(root).resolve()
    _publish_progress(
        progress_callback,
        stage="dataset-validation",
        status="started",
        path=str(dataset_root),
    )
    teacher_manifest_path = dataset_root / "teacher-manifest.json"
    teacher_manifest = _load_json(teacher_manifest_path, "teacher manifest")
    _verify_core_hash(teacher_manifest, "manifestSha256", "teacher manifest")
    if (
        teacher_manifest.get("schemaVersion") != TEACHER_DATASET_SCHEMA_VERSION
        or teacher_manifest.get("status") != "completed"
    ):
        raise TeacherPretrainingContractError("teacher dataset is not completed")
    try:
        dataset_fingerprint = validate_sha256(
            teacher_manifest.get("datasetFingerprint"), "teacher dataset fingerprint"
        )
    except ValueError as error:
        raise TeacherPretrainingContractError(str(error)) from error
    if teacher_manifest.get("contentSha256") != dataset_fingerprint:
        raise TeacherPretrainingContractError("teacher content/dataset fingerprints differ")

    run_manifest = load_run_manifest(dataset_root / "run-manifest.json")
    if teacher_manifest.get("runManifestSha256") != run_manifest.sha256:
        raise TeacherPretrainingContractError("teacher run-manifest lineage mismatch")
    provenance = run_manifest.payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TeacherPretrainingContractError("teacher provenance is missing")
    config_record = provenance.get("config")
    if not isinstance(config_record, Mapping):
        raise TeacherPretrainingContractError("teacher config provenance is missing")
    config = _teacher_config(config_record.get("payload"))
    config_sha256 = fingerprint_config(config.to_payload()).sha256
    if (
        config_sha256 != config_record.get("sha256")
        or config_sha256 != teacher_manifest.get("configSha256")
    ):
        raise TeacherPretrainingContractError("teacher config fingerprint mismatch")

    stores_record = teacher_manifest.get("stores")
    targets = teacher_manifest.get("targets")
    if not isinstance(stores_record, Mapping) or not isinstance(targets, Mapping):
        raise TeacherPretrainingContractError("teacher store/target metadata is invalid")
    stores: dict[str, ReplayStore] = {}
    seen_orbits: set[str] = set()
    combined_histograms: dict[str, dict[str, int]] | None = None
    for split, target in (
        ("train", config.train_target),
        ("holdout", config.holdout_target),
    ):
        recorded = stores_record.get(split)
        if not isinstance(recorded, Mapping) or recorded.get("path") != split:
            raise TeacherPretrainingContractError(f"teacher {split} metadata is invalid")
        store = ReplayStore(
            dataset_root / split,
            max_games=target,
            max_plies=target + 256,
            max_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
        )
        # Opening the store already validates manifest/file metadata. Each
        # shard is then decoded and fully replayed exactly once below, which
        # provides the same deep coverage without a duplicate disk pass.
        verification = store.verify(deep=False)
        if verification.ply_count < target:
            raise TeacherPretrainingContractError(f"teacher {split} target is incomplete")
        manifest_sha256 = sha256_file(store.manifest_path)
        if (
            recorded.get("manifestSha256") != manifest_sha256
            or recorded.get("games") != verification.game_count
            or recorded.get("positions") != verification.ply_count
            or recorded.get("committedBytes") != verification.committed_bytes
        ):
            raise TeacherPretrainingContractError(
                f"teacher {split} manifest/count metadata mismatch"
            )
        _publish_progress(
            progress_callback,
            stage="dataset-validation",
            status="split-started",
            split=split,
            games=0,
            gamesTotal=verification.game_count,
            positions=0,
            positionsTotal=verification.ply_count,
        )
        verified_games = 0
        verified_positions = 0
        next_game_progress = PROGRESS_GAME_INTERVAL
        next_position_progress = PROGRESS_POSITION_INTERVAL
        for game in store.iter_games():
            try:
                digests, histograms, _game_index = _verify_teacher_game(
                    game, split=split, config=config
                )
            except TeacherDataContractError as error:
                raise TeacherPretrainingContractError(str(error)) from error
            if seen_orbits.intersection(digests):
                raise TeacherPretrainingContractError(
                    "teacher train/holdout contain a duplicate augmentation orbit"
                )
            seen_orbits.update(digests)
            if combined_histograms is None:
                combined_histograms = {
                    category: dict(values) for category, values in histograms.items()
                }
            else:
                _merge_histograms(combined_histograms, histograms)
            verified_games += 1
            verified_positions += len(game.plies)
            if (
                verified_games >= next_game_progress
                or verified_positions >= next_position_progress
            ):
                _publish_progress(
                    progress_callback,
                    stage="dataset-validation",
                    status="progress",
                    split=split,
                    games=verified_games,
                    gamesTotal=verification.game_count,
                    positions=verified_positions,
                    positionsTotal=verification.ply_count,
                )
                while next_game_progress <= verified_games:
                    next_game_progress += PROGRESS_GAME_INTERVAL
                while next_position_progress <= verified_positions:
                    next_position_progress += PROGRESS_POSITION_INTERVAL
        if (
            verified_games != verification.game_count
            or verified_positions != verification.ply_count
        ):
            raise TeacherPretrainingContractError(
                f"teacher {split} deep-verification coverage is incomplete"
            )
        _publish_progress(
            progress_callback,
            stage="dataset-validation",
            status="split-completed",
            split=split,
            games=verified_games,
            gamesTotal=verification.game_count,
            positions=verified_positions,
            positionsTotal=verification.ply_count,
        )
        stores[split] = store

    train = stores["train"]
    holdout = stores["holdout"]
    content_core = {
        "schemaVersion": TEACHER_DATASET_SCHEMA_VERSION,
        "configSha256": config_sha256,
        "stores": {
            "holdout": {
                "manifestSha256": sha256_file(holdout.manifest_path),
                "games": holdout.game_count,
                "positions": holdout.ply_count,
            },
            "train": {
                "manifestSha256": sha256_file(train.manifest_path),
                "games": train.game_count,
                "positions": train.ply_count,
            },
        },
    }
    if sha256_bytes(canonical_json_bytes(content_core)) != dataset_fingerprint:
        raise TeacherPretrainingContractError("teacher dataset content hash mismatch")
    actual = teacher_manifest.get("actual")
    if not isinstance(actual, Mapping) or (
        actual.get("trainPositions") != train.ply_count
        or actual.get("holdoutPositions") != holdout.ply_count
        or actual.get("uniqueAugmentationOrbits") != len(seen_orbits)
        or actual.get("uniqueRawStates") != len(seen_orbits)
    ):
        raise TeacherPretrainingContractError("teacher actual-count metadata mismatch")
    if combined_histograms != teacher_manifest.get("strata"):
        raise TeacherPretrainingContractError("teacher stratum histograms mismatch")
    dataset = TeacherDataset(
        root=dataset_root,
        train=train,
        holdout=holdout,
        dataset_fingerprint=dataset_fingerprint,
        teacher_manifest_sha256=sha256_file(teacher_manifest_path),
        run_manifest_sha256=run_manifest.sha256,
        config_sha256=config_sha256,
        train_manifest_sha256=sha256_file(train.manifest_path),
        holdout_manifest_sha256=sha256_file(holdout.manifest_path),
        train_positions=train.ply_count,
        holdout_positions=holdout.ply_count,
    )
    _publish_progress(
        progress_callback,
        stage="dataset-validation",
        status="completed",
        trainGames=train.game_count,
        holdoutGames=holdout.game_count,
        trainPositions=train.ply_count,
        holdoutPositions=holdout.ply_count,
        teacherDatasetSha256=dataset.dataset_fingerprint,
    )
    return dataset


def _preencoded_cache_bytes(position_count: int, input_channels: int) -> int:
    if type(position_count) is not int or position_count <= 0:
        raise ValueError("cache position_count must be a positive integer")
    if type(input_channels) is not int or input_channels <= 0:
        raise ValueError("cache input_channels must be a positive integer")
    float_bytes = torch.tensor([], dtype=torch.float32).element_size()
    return position_count * (
        2 * input_channels * 9 * 9 * float_bytes
        + 2 * ACTION_SIZE * float_bytes
        + float_bytes
    )


def _require_finite_cache_tensor(tensor: Tensor, label: str) -> None:
    flattened = tensor.reshape(-1)
    check_size = 1024 * 1024
    for offset in range(0, flattened.numel(), check_size):
        if not bool(torch.isfinite(flattened[offset : offset + check_size]).all().item()):
            raise TeacherPretrainingContractError(
                f"preencoded {label} cache contains non-finite values"
            )


def _validate_preencoded_split(split: PreencodedReplaySplit) -> None:
    _require_finite_cache_tensor(split.states, f"{split.split} state")
    _require_finite_cache_tensor(split.policies, f"{split.split} policy")
    _require_finite_cache_tensor(split.values, f"{split.split} value")
    if bool((split.policies < 0).any().item()):
        raise TeacherPretrainingContractError(
            f"preencoded {split.split} policies contain negative mass"
        )
    for offset in range(0, split.position_count, PREENCODED_CACHE_CHUNK_SIZE):
        policy_sums = split.policies[
            offset : offset + PREENCODED_CACHE_CHUNK_SIZE
        ].sum(dim=2)
        if not bool(
            torch.allclose(
                policy_sums,
                torch.ones_like(policy_sums),
                rtol=1e-4,
                atol=1e-5,
            )
        ):
            raise TeacherPretrainingContractError(
                f"preencoded {split.split} policy does not sum to one"
            )
    if bool(((split.values < -1) | (split.values > 1)).any().item()):
        raise TeacherPretrainingContractError(
            f"preencoded {split.split} values are outside [-1, 1]"
        )


def _build_preencoded_split(
    store: ReplayStore,
    *,
    split: str,
    encoder: ReplayBatchEncoder,
    contract: TeacherPretrainingContract,
    progress_callback: ProgressCallback | None,
) -> PreencodedReplaySplit:
    position_count = store.ply_count
    states = torch.empty(
        (position_count, 2, contract.input_channels, 9, 9),
        dtype=torch.float32,
        device="cpu",
    )
    policies = torch.empty(
        (position_count, 2, ACTION_SIZE),
        dtype=torch.float32,
        device="cpu",
    )
    values = torch.empty(position_count, dtype=torch.float32, device="cpu")
    samples: list[ReplaySample] = []
    cursor = 0
    processed_games = 0
    processed_positions = 0
    next_game_progress = PROGRESS_GAME_INTERVAL
    next_position_progress = PROGRESS_POSITION_INTERVAL

    def encode_pending() -> None:
        nonlocal cursor
        if not samples:
            return
        identity = encoder.encode(
            samples,
            rng=random.Random(0),
            mirror_probability=0.0,
        )
        mirrored = encoder.encode(
            samples,
            rng=random.Random(0),
            mirror_probability=1.0,
        )
        size = len(samples)
        end = cursor + size
        if end > position_count:
            raise TeacherPretrainingContractError(
                f"preencoded {split} cursor exceeds the replay manifest"
            )
        if not torch.equal(identity["values"], mirrored["values"]):
            raise TeacherPretrainingContractError(
                f"preencoded {split} mirror changed value labels"
            )
        for label, batch in (("identity", identity), ("mirror", mirrored)):
            if (
                batch["states"].device.type != "cpu"
                or batch["policies"].device.type != "cpu"
                or batch["values"].device.type != "cpu"
                or batch["states"].dtype != torch.float32
                or batch["policies"].dtype != torch.float32
                or batch["values"].dtype != torch.float32
            ):
                raise TeacherPretrainingContractError(
                    f"preencoded {split} {label} encoder output is not CPU float32"
                )
            _require_finite_cache_tensor(batch["states"], f"{split} {label} state")
            _require_finite_cache_tensor(batch["policies"], f"{split} {label} policy")
            _require_finite_cache_tensor(batch["values"], f"{split} {label} value")
        states[cursor:end, 0].copy_(identity["states"])
        states[cursor:end, 1].copy_(mirrored["states"])
        policies[cursor:end, 0].copy_(identity["policies"])
        policies[cursor:end, 1].copy_(mirrored["policies"])
        values[cursor:end].copy_(identity["values"])
        cursor = end
        samples.clear()

    _publish_progress(
        progress_callback,
        stage="cache-build",
        status="split-started",
        split=split,
        games=0,
        gamesTotal=store.game_count,
        positions=0,
        positionsTotal=position_count,
    )
    for game in store.iter_games():
        reconstructed = reconstruct_game(game)
        winner = get_winner(reconstructed[-1])
        if winner is None:
            raise TeacherPretrainingContractError(
                f"preencoded {split} replay has no winner"
            )
        for ply_index, (state, ply) in enumerate(
            zip(reconstructed, game.plies)
        ):
            samples.append(
                ReplaySample(
                    game_id=game.game_id,
                    ply_index=ply_index,
                    state=state,
                    action_index=ply.action_index,
                    visit_counts=ply.visit_counts,
                    root_value=ply.root_value,
                    outcome=1.0 if state.current_player == winner else -1.0,
                    model_generation=ply.model_generation,
                    model_id=ply.model_id,
                    game_metadata=game.metadata,
                    ply_metadata=ply.metadata,
                )
            )
            if len(samples) >= PREENCODED_CACHE_CHUNK_SIZE:
                encode_pending()
        processed_games += 1
        processed_positions += len(game.plies)
        if (
            processed_games >= next_game_progress
            or processed_positions >= next_position_progress
        ):
            _publish_progress(
                progress_callback,
                stage="cache-build",
                status="progress",
                split=split,
                games=processed_games,
                gamesTotal=store.game_count,
                positions=processed_positions,
                positionsTotal=position_count,
            )
            while next_game_progress <= processed_games:
                next_game_progress += PROGRESS_GAME_INTERVAL
            while next_position_progress <= processed_positions:
                next_position_progress += PROGRESS_POSITION_INTERVAL
    encode_pending()
    if (
        cursor != position_count
        or processed_games != store.game_count
        or processed_positions != position_count
    ):
        raise TeacherPretrainingContractError(
            f"preencoded {split} cache coverage is incomplete"
        )
    result = PreencodedReplaySplit(
        split=split,
        states=states,
        policies=policies,
        values=values,
    )
    _validate_preencoded_split(result)
    _publish_progress(
        progress_callback,
        stage="cache-build",
        status="split-completed",
        split=split,
        games=processed_games,
        gamesTotal=store.game_count,
        positions=processed_positions,
        positionsTotal=position_count,
        bytes=result.byte_size,
    )
    return result


def build_teacher_preencoded_cache(
    dataset: TeacherDataset,
    *,
    native_encoder: Callable[..., object],
    contract: TeacherPretrainingContract,
    native_backend_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TeacherPreencodedCache:
    """Decode each teacher shard once and retain both mirror views on CPU."""

    if not callable(native_encoder):
        raise TypeError("native_encoder must be callable")
    expected_bytes = _preencoded_cache_bytes(
        dataset.train_positions + dataset.holdout_positions,
        contract.input_channels,
    )
    if expected_bytes > PREENCODED_CACHE_MAX_BYTES:
        raise TeacherPretrainingResourceError(
            "teacher preencoded cache would exceed the 1 GiB CPU limit"
        )
    input_paths = {
        "teacher": dataset.root / "teacher-manifest.json",
        "run": dataset.root / "run-manifest.json",
        "train": dataset.train.manifest_path,
        "holdout": dataset.holdout.manifest_path,
    }
    before_manifest_files = {
        label: sha256_file(path) for label, path in input_paths.items()
    }
    expected_manifest_files = {
        "teacher": dataset.teacher_manifest_sha256,
        "train": dataset.train_manifest_sha256,
        "holdout": dataset.holdout_manifest_sha256,
    }
    if any(
        before_manifest_files[label] != expected
        for label, expected in expected_manifest_files.items()
    ):
        raise TeacherPretrainingContractError(
            "teacher dataset manifests changed before cache construction"
        )
    try:
        run_manifest = load_run_manifest(input_paths["run"])
    except RunManifestError as error:
        raise TeacherPretrainingContractError(
            "teacher run-manifest is invalid before cache construction"
        ) from error
    if run_manifest.sha256 != dataset.run_manifest_sha256:
        raise TeacherPretrainingContractError(
            "teacher run-manifest changed before cache construction"
        )
    if (
        native_backend_sha256 is not None
        and loaded_native_backend_hash() != native_backend_sha256
    ):
        raise TeacherPretrainingContractError(
            "native encoder binary changed before cache construction"
        )

    started = time.monotonic()
    _publish_progress(
        progress_callback,
        stage="cache-build",
        status="started",
        algorithmVersion=PREENCODED_CACHE_ALGORITHM_VERSION,
        expectedBytes=expected_bytes,
        maximumBytes=PREENCODED_CACHE_MAX_BYTES,
        trainPositions=dataset.train_positions,
        holdoutPositions=dataset.holdout_positions,
    )
    encoder = ReplayBatchEncoder(
        input_channels=contract.input_channels,
        device="cpu",
        native_encoder=native_encoder,
    )
    train = _build_preencoded_split(
        dataset.train,
        split="train",
        encoder=encoder,
        contract=contract,
        progress_callback=progress_callback,
    )
    holdout = _build_preencoded_split(
        dataset.holdout,
        split="holdout",
        encoder=encoder,
        contract=contract,
        progress_callback=progress_callback,
    )
    after_manifest_files = {
        label: sha256_file(path) for label, path in input_paths.items()
    }
    if after_manifest_files != before_manifest_files:
        raise TeacherPretrainingContractError(
            "teacher dataset manifests changed during cache construction"
        )
    try:
        run_manifest = load_run_manifest(input_paths["run"])
    except RunManifestError as error:
        raise TeacherPretrainingContractError(
            "teacher run-manifest is invalid after cache construction"
        ) from error
    if run_manifest.sha256 != dataset.run_manifest_sha256:
        raise TeacherPretrainingContractError(
            "teacher run-manifest changed during cache construction"
        )
    if (
        native_backend_sha256 is not None
        and loaded_native_backend_hash() != native_backend_sha256
    ):
        raise TeacherPretrainingContractError(
            "native encoder binary changed during cache construction"
        )
    elapsed = max(0.0, time.monotonic() - started)
    cache = TeacherPreencodedCache(
        algorithm_version=PREENCODED_CACHE_ALGORITHM_VERSION,
        dataset_fingerprint=dataset.dataset_fingerprint,
        train=train,
        holdout=holdout,
        build_seconds=elapsed,
    )
    if cache.byte_size != expected_bytes:
        raise TeacherPretrainingContractError(
            "teacher preencoded cache byte accounting is inconsistent"
        )
    _publish_progress(
        progress_callback,
        stage="cache-build",
        status="completed",
        **cache.snapshot(),
    )
    return cache


def _derived_seed(seed: int, namespace: str, step: int) -> int:
    return int.from_bytes(
        bytes.fromhex(
            sha256_bytes(f"phase4d-pretrain-v1:{seed}:{namespace}:{step}".encode())
        )[:8],
        "big",
    )


def load_trusted_native_encoder() -> tuple[object, str]:
    """Return the native encoder only when the loaded extension identity is stable."""

    first_hash = loaded_native_backend_hash()
    module = load_native_module()
    second_hash = loaded_native_backend_hash()
    if first_hash != second_hash or not callable(getattr(module, "encode_states", None)):
        raise TeacherPretrainingContractError(
            "native encoder extension identity changed or encode_states is unavailable"
        )
    return module, first_hash


@contextmanager
def _optimizer_step_boundary() -> object:
    """Delay SIGINT/SIGTERM until model, trainer cursor, and metrics agree."""

    signals = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def training_batch(
    store: ReplayStore,
    encoder: ReplayBatchEncoder,
    contract: TeacherPretrainingContract,
    *,
    step: int,
) -> dict[str, Tensor]:
    samples = store.sample_plies(
        contract.batch_size,
        rng=random.Random(_derived_seed(contract.seed, "sample", step)),
    )
    return encoder.encode(
        samples,
        rng=random.Random(_derived_seed(contract.seed, "mirror", step)),
        mirror_probability=contract.mirror_probability,
    )


def cached_training_batch(
    cache: PreencodedReplaySplit,
    contract: TeacherPretrainingContract,
    *,
    step: int,
    device: str | torch.device,
) -> dict[str, Tensor]:
    """Reproduce ReplayStore.sample_plies plus sample-time mirror decisions."""

    if cache.split != "train":
        raise ValueError("training batches require the train cache split")
    if type(step) is not int or step <= 0:
        raise ValueError("training step must be a positive integer")
    selected_count = min(contract.batch_size, cache.position_count)
    indices = random.Random(
        _derived_seed(contract.seed, "sample", step)
    ).sample(range(cache.position_count), selected_count)
    mirror_rng = random.Random(_derived_seed(contract.seed, "mirror", step))
    mirrored = [
        mirror_rng.random() < contract.mirror_probability for _ in indices
    ]
    return cache.batch(indices, mirrored, device=device)


def evaluate_holdout(
    model: PolicyValueNetwork,
    store: ReplayStore,
    encoder: ReplayBatchEncoder,
    contract: TeacherPretrainingContract,
) -> dict[str, float | int]:
    """Evaluate every isolated holdout sample with a fixed mirrored view."""

    samples = store.sample_plies(
        store.ply_count,
        rng=random.Random(HOLDOUT_AUGMENTATION_SEED),
    )
    mirror_rng = random.Random(HOLDOUT_AUGMENTATION_SEED ^ contract.seed)
    previous_training = model.training
    model.eval()
    policy_sum = 0.0
    value_sum = 0.0
    count = 0
    amp_enabled = contract.use_amp and next(model.parameters()).device.type == "cuda"
    try:
        with torch.no_grad():
            for offset in range(0, len(samples), contract.batch_size):
                chunk = samples[offset : offset + contract.batch_size]
                batch = encoder.encode(
                    chunk,
                    rng=mirror_rng,
                    mirror_probability=contract.mirror_probability,
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=amp_enabled
                ):
                    logits, predicted = model(batch["states"])
                    policy_loss = -(
                        batch["policies"] * F.log_softmax(logits, dim=1)
                    ).sum(dim=1)
                    value_loss = F.mse_loss(
                        predicted, batch["values"].reshape_as(predicted), reduction="none"
                    )
                if not bool(torch.isfinite(policy_loss).all().item()) or not bool(
                    torch.isfinite(value_loss).all().item()
                ):
                    raise TeacherPretrainingContractError(
                        "holdout evaluation produced a non-finite loss"
                    )
                policy_sum += float(policy_loss.float().sum().item())
                value_sum += float(value_loss.float().sum().item())
                count += len(chunk)
    finally:
        model.train(previous_training)
    if count != store.ply_count or count <= 0:
        raise TeacherPretrainingContractError("holdout evaluation coverage is incomplete")
    policy_mean = policy_sum / count
    value_mean = value_sum / count
    return {
        "positions": count,
        "policyLoss": policy_mean,
        "valueLoss": value_mean,
        "totalLoss": policy_mean + value_mean,
    }


def evaluate_cached_holdout(
    model: PolicyValueNetwork,
    cache: PreencodedReplaySplit,
    contract: TeacherPretrainingContract,
) -> dict[str, float | int]:
    """Evaluate the exact legacy holdout permutation and mirror RNG sequence."""

    if cache.split != "holdout":
        raise ValueError("holdout evaluation requires the holdout cache split")
    device = next(model.parameters()).device
    indices = random.Random(HOLDOUT_AUGMENTATION_SEED).sample(
        range(cache.position_count),
        cache.position_count,
    )
    mirror_rng = random.Random(HOLDOUT_AUGMENTATION_SEED ^ contract.seed)
    mirrored = [
        mirror_rng.random() < contract.mirror_probability for _ in indices
    ]
    previous_training = model.training
    model.eval()
    policy_sum = 0.0
    value_sum = 0.0
    count = 0
    amp_enabled = contract.use_amp and device.type == "cuda"
    try:
        with torch.no_grad():
            for offset in range(0, len(indices), contract.batch_size):
                batch_indices = indices[offset : offset + contract.batch_size]
                batch_mirrored = mirrored[
                    offset : offset + contract.batch_size
                ]
                batch = cache.batch(
                    batch_indices,
                    batch_mirrored,
                    device=device,
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=amp_enabled
                ):
                    logits, predicted = model(batch["states"])
                    policy_loss = -(
                        batch["policies"] * F.log_softmax(logits, dim=1)
                    ).sum(dim=1)
                    value_loss = F.mse_loss(
                        predicted,
                        batch["values"].reshape_as(predicted),
                        reduction="none",
                    )
                if not bool(torch.isfinite(policy_loss).all().item()) or not bool(
                    torch.isfinite(value_loss).all().item()
                ):
                    raise TeacherPretrainingContractError(
                        "cached holdout evaluation produced a non-finite loss"
                    )
                policy_sum += float(policy_loss.float().sum().item())
                value_sum += float(value_loss.float().sum().item())
                count += len(batch_indices)
    finally:
        model.train(previous_training)
    if count != cache.position_count or count <= 0:
        raise TeacherPretrainingContractError(
            "cached holdout evaluation coverage is incomplete"
        )
    policy_mean = policy_sum / count
    value_mean = value_sum / count
    return {
        "positions": count,
        "policyLoss": policy_mean,
        "valueLoss": value_mean,
        "totalLoss": policy_mean + value_mean,
    }


class _ResourceGuard:
    def __init__(
        self,
        *,
        control_root: Path,
        ledger: GpuBudgetLedger,
        stage_ledger: Phase4StageBudgetLedger,
        ledger_run_id: str,
        started: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.control_root = control_root
        self.ledger = ledger
        self.stage_ledger = stage_ledger
        self.ledger_run_id = ledger_run_id
        self.started = started
        self.clock = clock
        self.last_checked = float("-inf")
        self.last_heartbeat = float("-inf")

    def check(self, *, force: bool = False) -> dict[str, object]:
        now = self.clock()
        elapsed = max(0.0, now - self.started)
        estimated_global = self.ledger.total_hours
        run = self.ledger.runs.get(self.ledger_run_id)
        accounted = float(run["accountedSeconds"]) if isinstance(run, Mapping) else 0.0
        estimated_global += max(0.0, elapsed - accounted) / 3600.0
        if force or now - self.last_heartbeat >= LEDGER_HEARTBEAT_SECONDS:
            estimated_global = self.ledger.update(
                self.ledger_run_id, elapsed_seconds=elapsed, status="running"
            )
            self.last_heartbeat = now
        stage = self.stage_ledger.assert_available(PHASE4D_STAGE, estimated_global)
        if estimated_global >= GPU_HOUR_BUDGET:
            raise TeacherPretrainingResourceError("72 GPU-hour global budget is exhausted")
        if force or now - self.last_checked >= RESOURCE_CHECK_INTERVAL_SECONDS:
            free = int(shutil.disk_usage(self.control_root).free)
            artifacts = directory_size(self.control_root)
            if free < DEFAULT_MINIMUM_FREE_DISK_BYTES + FINALIZATION_RESERVE_BYTES:
                raise TeacherPretrainingResourceError(
                    "free disk reached the 20 GiB stop line plus finalization reserve"
                )
            if artifacts + FINALIZATION_RESERVE_BYTES > DEFAULT_ARTIFACT_LIMIT_BYTES:
                raise TeacherPretrainingResourceError(
                    "Phase 4 artifacts cannot reserve finalization space below 8 GiB"
                )
            self.last_checked = now
        return {
            "globalGpuHours": estimated_global,
            "stageGpuHours": stage,
            "freeDiskBytes": int(shutil.disk_usage(self.control_root).free),
            "artifactBytes": directory_size(self.control_root),
        }


def _assert_finalization_resources(control_root: Path) -> None:
    free = int(shutil.disk_usage(control_root).free)
    artifacts = directory_size(control_root)
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES + FINALIZATION_RESERVE_BYTES:
        raise TeacherPretrainingResourceError(
            "cannot reserve 512 MiB above the 20 GiB free-disk stop line"
        )
    if artifacts + FINALIZATION_RESERVE_BYTES > DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise TeacherPretrainingResourceError(
            "cannot reserve 512 MiB below the Phase 4 8 GiB artifact cap"
        )


def _assert_metadata_resources(control_root: Path) -> None:
    free = int(shutil.disk_usage(control_root).free)
    artifacts = directory_size(control_root)
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES + FINAL_METADATA_RESERVE_BYTES:
        raise TeacherPretrainingResourceError(
            "cannot reserve report metadata above the 20 GiB stop line"
        )
    if artifacts + FINAL_METADATA_RESERVE_BYTES > DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise TeacherPretrainingResourceError(
            "cannot reserve report metadata below the 8 GiB artifact cap"
        )


def _assert_gpu_budget_preflight(
    ledger: GpuBudgetLedger,
    stage_ledger: Phase4StageBudgetLedger,
    external_gpu_hours: float,
) -> float:
    prospective = ledger.total_hours + max(
        0.0, float(external_gpu_hours) - ledger.external_hours_floor
    )
    if prospective >= GPU_HOUR_BUDGET:
        raise TeacherPretrainingResourceError("72 GPU-hour global budget is exhausted")
    try:
        stage_ledger.assert_available(PHASE4D_STAGE, prospective)
    except RuntimeError as error:
        raise TeacherPretrainingResourceError(str(error)) from error
    return prospective


def _checkpoint_replay_snapshot(dataset: TeacherDataset) -> dict[str, object]:
    train = dataset.train
    if (
        train.ply_count != dataset.train_positions
        or sha256_file(train.manifest_path) != dataset.train_manifest_sha256
    ):
        raise TeacherPretrainingContractError(
            "teacher train ReplayStore changed after deep preflight verification"
        )
    shard_bytes = sum(shard.byte_size for shard in train.shards)
    manifest_bytes = train.committed_bytes - shard_bytes
    return {
        "manifestSha256": dataset.train_manifest_sha256,
        "games": train.game_count,
        "plies": train.ply_count,
        "bytes": shard_bytes,
        "manifestBytes": manifest_bytes,
        "committedBytes": train.committed_bytes,
        "deepVerified": True,
        "gameIds": list(train.game_ids),
        "prefixSha256": _replay_prefix_sha256(train),
    }


def _tensor_payload_bytes(value: object) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def checkpoint_write_reserve_bytes(
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
    checkpoint_root: Path,
) -> int:
    """Conservatively cover current+best+Adam tensors and temp coexistence."""

    tensor_bytes = _tensor_payload_bytes(bundle.state_dict()) + _tensor_payload_bytes(
        trainer.optimizer.state_dict()
    )
    observed = max(
        (path.stat().st_size for path in checkpoint_root.glob("checkpoint-*.pt")),
        default=0,
    )
    return max(
        CHECKPOINT_FIXED_RESERVE_BYTES,
        tensor_bytes * 2 + CHECKPOINT_FIXED_RESERVE_BYTES,
        observed * 2 + CHECKPOINT_FIXED_RESERVE_BYTES,
    )


def _assert_checkpoint_resources(
    store: Phase4CheckpointStore,
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
) -> None:
    reserve = checkpoint_write_reserve_bytes(bundle, trainer, store.root)
    free = int(shutil.disk_usage(store.artifact_root).free)
    artifacts = directory_size(store.artifact_root)
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES + reserve:
        raise TeacherPretrainingResourceError(
            "checkpoint cannot preserve the 20 GiB stop line during temp coexistence"
        )
    if artifacts + reserve > store.artifact_limit_bytes:
        raise TeacherPretrainingResourceError(
            "checkpoint cannot fit below the 8 GiB cap during temp coexistence"
        )


def _lineage(
    *,
    workspace: Path,
    run_root: Path,
    contract: TeacherPretrainingContract,
    dataset: TeacherDataset,
    memory_smoke: Mapping[str, object],
    native_backend_sha256: str,
    baseline: Mapping[str, object],
    resume: bool,
) -> tuple[RunManifestRecord, CheckpointIdentity, bool]:
    config_payload = {
        "kind": RUN_KIND,
        "contract": contract.to_payload(),
        "teacherDataset": dataset.snapshot(),
        "memorySmoke": dict(memory_smoke),
        "nativeEncoderBackendSha256": native_backend_sha256,
    }
    budget = ResourceBudget(
        artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
        minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
        gpu_hours=PHASE4D_GPU_HOUR_LIMIT,
        metadata={
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "stage": PHASE4D_STAGE,
            "stageBaselineGlobalGpuHours": PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
            "baselineCalibration": dict(baseline),
        },
    )
    config_fingerprint = fingerprint_config(config_payload)
    source_fingerprint = fingerprint_sources(workspace, _SOURCE_PATHS)
    path = run_root / "run-manifest.json"
    if path.exists():
        if not resume:
            raise TeacherPretrainingContractError(
                "pretraining run exists; --resume is required"
            )
        manifest = load_run_manifest(path)
        provenance = manifest.payload.get("provenance")
        stored_config = provenance.get("config") if isinstance(provenance, Mapping) else None
        stored_source = provenance.get("source") if isinstance(provenance, Mapping) else None
        if (
            not isinstance(stored_config, Mapping)
            or stored_config.get("sha256") != config_fingerprint.sha256
            or not isinstance(stored_source, Mapping)
            or stored_source.get("sha256") != source_fingerprint.sha256
            or manifest.payload.get("resourceBudget") != budget.to_payload()
        ):
            raise TeacherPretrainingContractError(
                "pretraining config/source/budget changed; strict resume refused"
            )
        resumed = True
    else:
        if resume:
            raise TeacherPretrainingContractError(
                "--resume requested but pretraining run manifest is missing"
            )
        protected = (
            run_root / "checkpoints",
            run_root / REPORT_NAME,
            run_root / WARM_ARTIFACT_NAME,
            run_root / COMPLETION_MANIFEST_NAME,
            run_root / SUMMARY_NAME,
        )
        if any(item.exists() for item in protected):
            raise TeacherPretrainingContractError(
                "pretraining artifacts exist without an immutable run manifest"
            )
        manifest = build_run_manifest(
            workspace_root=workspace,
            config=config_payload,
            budget=budget,
            run_id="phase4d-teacher-pretraining-v2-20260714",
            source_paths=_SOURCE_PATHS,
            metadata={
                "phase": "4D",
                "kind": RUN_KIND,
                "purpose": "hard-teacher-warm-model",
                "teacherReplayUsedAfterPretraining": False,
            },
        )
        run_root.mkdir(parents=True, exist_ok=True)
        save_run_manifest(manifest, path)
        resumed = False
    return (
        manifest,
        CheckpointIdentity(
            run_manifest_sha256=manifest.sha256,
            config_sha256=config_fingerprint.sha256,
            source_sha256=source_fingerprint.sha256,
        ),
        resumed,
    )


def _checkpoint_generation_data(
    *,
    generation: int,
    contract_sha256: str,
    dataset: TeacherDataset,
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
    base_model_sha256: str,
    last_evaluation_step: int,
    best_step: int | None,
    best_metrics: Mapping[str, object] | None,
    evaluations: Sequence[Mapping[str, object]],
    metrics: MetricAccumulator,
    reason: str,
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "generation": generation,
        "reason": reason,
        "step": trainer.step,
        "lastEvaluationStep": last_evaluation_step,
        "bestStep": best_step,
        "bestMetrics": dict(best_metrics) if best_metrics is not None else None,
        "evaluations": [dict(item) for item in evaluations],
        "trainingMetrics": metrics.to_payload(),
        "contractSha256": contract_sha256,
        "teacherDatasetSha256": dataset.dataset_fingerprint,
        "teacherDataset": dataset.snapshot(),
        "baseModelStateSha256": base_model_sha256,
        "currentModelStateSha256": canonical_model_state_sha256(bundle.current),
        "bestModelStateSha256": canonical_model_state_sha256(bundle.best),
    }


def _validate_restored(
    loaded: Phase4CheckpointLoad,
    *,
    contract: TeacherPretrainingContract,
    contract_sha256: str,
    dataset: TeacherDataset,
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
    base_model_sha256: str,
) -> tuple[int, int | None, dict[str, object] | None, list[dict[str, object]], MetricAccumulator]:
    data = loaded.payload.get("generationData")
    replay = loaded.payload.get("replaySnapshot")
    if not isinstance(data, Mapping) or not isinstance(replay, Mapping):
        raise TeacherPretrainingContractError("pretraining checkpoint metadata is missing")
    if (
        data.get("schemaVersion") != SCHEMA_VERSION
        or data.get("kind") != RUN_KIND
        or data.get("generation") != loaded.record.generation
        or data.get("step") != trainer.step
        or data.get("contractSha256") != contract_sha256
        or data.get("teacherDatasetSha256") != dataset.dataset_fingerprint
        or data.get("teacherDataset") != dataset.snapshot()
        or data.get("baseModelStateSha256") != base_model_sha256
        or dict(replay) != _checkpoint_replay_snapshot(dataset)
    ):
        raise TeacherPretrainingContractError("pretraining checkpoint lineage mismatch")
    if trainer.step > contract.optimizer_steps:
        raise TeacherPretrainingContractError("checkpoint step exceeds frozen workload")
    if data.get("currentModelStateSha256") != canonical_model_state_sha256(bundle.current):
        raise TeacherPretrainingContractError("checkpoint current-model hash mismatch")
    if data.get("bestModelStateSha256") != canonical_model_state_sha256(bundle.best):
        raise TeacherPretrainingContractError("checkpoint best-model hash mismatch")
    last_eval = data.get("lastEvaluationStep")
    best_step = data.get("bestStep")
    if type(last_eval) is not int or last_eval < 0 or last_eval > trainer.step:
        raise TeacherPretrainingContractError("checkpoint evaluation cursor is invalid")
    if best_step is not None and (
        type(best_step) is not int
        or best_step <= 0
        or best_step > last_eval
        or best_step % contract.evaluation_interval
    ):
        raise TeacherPretrainingContractError("checkpoint best step is invalid")
    raw_evaluations = data.get("evaluations")
    if not isinstance(raw_evaluations, list) or any(
        not isinstance(item, Mapping) for item in raw_evaluations
    ):
        raise TeacherPretrainingContractError("checkpoint evaluations are invalid")
    evaluations = [dict(item) for item in raw_evaluations]
    expected_steps = list(
        range(contract.evaluation_interval, last_eval + 1, contract.evaluation_interval)
    )
    if [item.get("step") for item in evaluations] != expected_steps:
        raise TeacherPretrainingContractError("checkpoint evaluation history is incomplete")
    best_metrics = data.get("bestMetrics")
    if (best_step is None) != (best_metrics is None):
        raise TeacherPretrainingContractError("checkpoint best metrics are inconsistent")
    if best_metrics is not None and (
        not isinstance(best_metrics, Mapping) or best_metrics.get("step") != best_step
    ):
        raise TeacherPretrainingContractError("checkpoint best metrics are invalid")
    metrics = MetricAccumulator.from_payload(data.get("trainingMetrics"))
    if metrics.count != trainer.step:
        raise TeacherPretrainingContractError("checkpoint metric/optimizer steps differ")
    return (
        last_eval,
        best_step,
        None if best_metrics is None else dict(best_metrics),
        evaluations,
        metrics,
    )


def restore_pretraining_checkpoint(
    store: Phase4CheckpointStore,
    *,
    contract: TeacherPretrainingContract,
    contract_sha256: str,
    dataset: TeacherDataset,
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
    base_model_sha256: str,
    map_location: str | torch.device,
) -> tuple[
    Phase4CheckpointLoad,
    tuple[int, int | None, dict[str, object] | None, list[dict[str, object]], MetricAccumulator],
] | None:
    """Restore with fallback across both structural and pretrain-semantic faults."""

    while store.candidates():
        rng_before_candidate = capture_rng_state()
        try:
            loaded = store.restore(
                bundle, trainer, map_location=map_location, restore_rng=False
            )
        except Phase4CheckpointError as error:
            raise TeacherPretrainingContractError(
                "no structurally valid pretraining checkpoint remains"
            ) from error
        if loaded is None:
            return None
        try:
            cursor = _validate_restored(
                loaded,
                contract=contract,
                contract_sha256=contract_sha256,
                dataset=dataset,
                bundle=bundle,
                trainer=trainer,
                base_model_sha256=base_model_sha256,
            )
            restore_rng_state(loaded.payload["rng"])
        except (
            KeyError,
            RuntimeError,
            TeacherPretrainingContractError,
            TypeError,
            ValueError,
        ) as error:
            try:
                restore_rng_state(rng_before_candidate)
            except (KeyError, RuntimeError, TypeError, ValueError) as rollback_error:
                raise TeacherPretrainingContractError(
                    "cannot roll back RNG after a rejected pretraining checkpoint"
                ) from rollback_error
            rejection = (
                f"{loaded.record.metadata_path.name}: semantic: "
                f"{type(error).__name__}: {error}"
            )
            store.quarantine_rejected((*loaded.rejected_newer, rejection))
            continue
        if loaded.rejected_newer:
            store.quarantine_rejected(loaded.rejected_newer)
        return loaded, cursor
    return None


def _save_checkpoint(
    store: Phase4CheckpointStore,
    *,
    next_generation: int,
    contract_sha256: str,
    dataset: TeacherDataset,
    bundle: CheckpointModels,
    trainer: AlphaZeroTrainer,
    base_model_sha256: str,
    last_evaluation_step: int,
    best_step: int | None,
    best_metrics: Mapping[str, object] | None,
    evaluations: Sequence[Mapping[str, object]],
    metrics: MetricAccumulator,
    reason: str,
) -> None:
    _assert_checkpoint_resources(store, bundle, trainer)
    store.save(
        generation=next_generation,
        model=bundle,
        trainer=trainer,
        generation_data=_checkpoint_generation_data(
            generation=next_generation,
            contract_sha256=contract_sha256,
            dataset=dataset,
            bundle=bundle,
            trainer=trainer,
            base_model_sha256=base_model_sha256,
            last_evaluation_step=last_evaluation_step,
            best_step=best_step,
            best_metrics=best_metrics,
            evaluations=evaluations,
            metrics=metrics,
            reason=reason,
        ),
        replay_snapshot=_checkpoint_replay_snapshot(dataset),
    )


def _runtime_checkpoint_consistent(
    *,
    contract: TeacherPretrainingContract,
    trainer: AlphaZeroTrainer,
    metrics: MetricAccumulator,
    last_evaluation_step: int,
    best_step: int | None,
    best_metrics: Mapping[str, object] | None,
    evaluations: Sequence[Mapping[str, object]],
) -> bool:
    expected_steps = list(
        range(
            contract.evaluation_interval,
            last_evaluation_step + 1,
            contract.evaluation_interval,
        )
    )
    return bool(
        metrics.count == trainer.step
        and 0 <= last_evaluation_step <= trainer.step
        and [item.get("step") for item in evaluations] == expected_steps
        and ((best_step is None) == (best_metrics is None))
        and (
            best_step is None
            or (
                best_step in expected_steps
                and isinstance(best_metrics, Mapping)
                and best_metrics.get("step") == best_step
            )
        )
    )


def _atomic_create_torch_artifact(
    payload: Mapping[str, object], destination: Path, control_root: Path
) -> None:
    _assert_finalization_resources(control_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        projected = directory_size(control_root)
        if projected + FINAL_METADATA_RESERVE_BYTES > DEFAULT_ARTIFACT_LIMIT_BYTES:
            raise ArtifactBudgetError("warm artifact would exceed the 8 GiB Phase 4 cap")
        if shutil.disk_usage(control_root).free < (
            DEFAULT_MINIMUM_FREE_DISK_BYTES + FINAL_METADATA_RESERVE_BYTES
        ):
            raise TeacherPretrainingResourceError(
                "warm artifact would cross the 20 GiB free-disk stop line"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ImmutableArtifactExistsError(
                f"immutable warm artifact already exists: {destination}"
            ) from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise TeacherPretrainingContractError("artifact path is outside the workspace") from error


def _report_payload(
    *,
    manifest: RunManifestRecord,
    workspace: Path,
    run_root: Path,
    contract: TeacherPretrainingContract,
    dataset: TeacherDataset,
    memory_smoke: Mapping[str, object],
    native_backend_sha256: str,
    baseline: Mapping[str, object],
    base_model_sha256: str,
    best_model_sha256: str,
    best_step: int,
    best_metrics: Mapping[str, object],
    evaluations: Sequence[Mapping[str, object]],
    metrics: MetricAccumulator,
    cache: TeacherPreencodedCache,
    resource_snapshot: Mapping[str, object],
    checkpoint: Phase4CheckpointLoad,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "status": "completed",
        "createdAt": manifest.payload["createdAt"],
        "completedAt": utc_timestamp(),
        "runManifestSha256": manifest.sha256,
        "teacherDatasetSha256": dataset.dataset_fingerprint,
        "baseModelStateSha256": base_model_sha256,
        "modelStateSha256": best_model_sha256,
        "warmArtifactPath": _relative(run_root / WARM_ARTIFACT_NAME, workspace),
        "contract": contract.to_payload(),
        "dataset": dataset.snapshot(),
        "memorySmoke": dict(memory_smoke),
        "nativeEncoderBackendSha256": native_backend_sha256,
        "baselineCalibration": dict(baseline),
        "optimizerResetRequiredForSelfPlay": True,
        "teacherReplayAllowedInSelfPlay": False,
        "best": {"step": best_step, **dict(best_metrics)},
        "evaluations": [dict(item) for item in evaluations],
        "training": {
            **metrics.summary(),
            "preencodedCache": cache.snapshot(),
        },
        "resourcesAtReport": dict(resource_snapshot),
        "finalCheckpoint": {
            "generation": checkpoint.record.generation,
            "step": checkpoint.record.step,
            "path": _relative(checkpoint.record.data_path, workspace),
            "sha256": checkpoint.record.sha256,
        },
    }
    return {**core, "reportCoreSha256": sha256_bytes(canonical_json_bytes(core))}


def load_pretraining_report(path: str | Path) -> dict[str, object]:
    payload = _load_json(Path(path), "pretraining report")
    _verify_core_hash(payload, "reportCoreSha256", "pretraining report")
    if (
        payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind") != RUN_KIND
        or payload.get("status") != "completed"
    ):
        raise TeacherPretrainingContractError("pretraining report schema/status is invalid")
    return payload


def _warm_artifact_payload(
    *,
    contract: TeacherPretrainingContract,
    bundle: CheckpointModels,
    base_model_sha256: str,
    dataset_sha256: str,
    report_file_sha256: str,
) -> dict[str, object]:
    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in bundle.best.state_dict().items()
    }
    model_sha256 = canonical_model_state_sha256(state)
    return {
        "schemaVersion": WARM_ARTIFACT_SCHEMA_VERSION,
        "format": "quoridor-phase4-model-only",
        "network": contract.network_config().to_dict(),
        "baseModelStateSha256": base_model_sha256,
        "modelStateSha256": model_sha256,
        "provenance": {
            "teacherDatasetSha256": dataset_sha256,
            "pretrainingReportSha256": report_file_sha256,
        },
        "modelState": state,
    }


def load_warm_artifact(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TeacherPretrainingContractError("warm artifact is missing or a symlink")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TeacherPretrainingContractError("warm artifact cannot be loaded safely") from error
    expected = {
        "schemaVersion",
        "format",
        "network",
        "baseModelStateSha256",
        "modelStateSha256",
        "provenance",
        "modelState",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise TeacherPretrainingContractError("warm artifact schema fields are invalid")
    state = payload.get("modelState")
    if not isinstance(state, Mapping):
        raise TeacherPretrainingContractError("warm artifact model state is invalid")
    if payload.get("modelStateSha256") != canonical_model_state_sha256(state):
        raise TeacherPretrainingContractError("warm artifact model hash mismatch")
    return dict(payload)


def _summary(
    *, workspace: Path, run_root: Path, dataset_sha256: str
) -> dict[str, object]:
    report_path = run_root / REPORT_NAME
    artifact_path = run_root / WARM_ARTIFACT_NAME
    report = load_pretraining_report(report_path)
    artifact = load_warm_artifact(artifact_path)
    report_file_sha256 = sha256_file(report_path)
    artifact_file_sha256 = sha256_file(artifact_path)
    provenance = artifact.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or artifact.get("baseModelStateSha256") != report.get("baseModelStateSha256")
        or artifact.get("modelStateSha256") != report.get("modelStateSha256")
        or report.get("teacherDatasetSha256") != dataset_sha256
        or provenance.get("teacherDatasetSha256") != dataset_sha256
        or provenance.get("pretrainingReportSha256") != report_file_sha256
    ):
        raise TeacherPretrainingContractError("report/warm artifact provenance mismatch")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "completed",
        "artifactPath": _relative(artifact_path, workspace),
        "artifactSha256": artifact_file_sha256,
        "modelStateSha256": artifact["modelStateSha256"],
        "baseModelStateSha256": artifact["baseModelStateSha256"],
        "teacherDatasetSha256": dataset_sha256,
        "reportPath": _relative(report_path, workspace),
        "reportSha256": report_file_sha256,
    }


def _write_completion_manifest(
    *, workspace: Path, run_root: Path, manifest: RunManifestRecord, summary: Mapping[str, object]
) -> None:
    path = run_root / COMPLETION_MANIFEST_NAME
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "status": "completed",
        "createdAt": utc_timestamp(),
        "runManifestSha256": manifest.sha256,
        "summary": dict(summary),
    }
    payload = {
        **core,
        "completionManifestSha256": sha256_bytes(canonical_json_bytes(core)),
    }
    atomic_create_bytes(path, canonical_json_bytes(payload))


def _write_or_validate_summary(run_root: Path, summary: Mapping[str, object]) -> None:
    path = run_root / SUMMARY_NAME
    expected = canonical_json_bytes(summary)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise TeacherPretrainingContractError(
                "existing immutable pretraining summary does not match artifacts"
            )
        return
    atomic_create_bytes(path, expected)


def _validate_completion(
    *, workspace: Path, run_root: Path, manifest: RunManifestRecord, dataset_sha256: str
) -> dict[str, object] | None:
    path = run_root / COMPLETION_MANIFEST_NAME
    if not path.exists():
        return None
    payload = _load_json(path, "pretraining completion manifest")
    _verify_core_hash(
        payload, "completionManifestSha256", "pretraining completion manifest"
    )
    summary = _summary(
        workspace=workspace, run_root=run_root, dataset_sha256=dataset_sha256
    )
    summary_path = run_root / SUMMARY_NAME
    if (
        summary_path.is_symlink()
        or not summary_path.is_file()
        or summary_path.read_bytes() != canonical_json_bytes(summary)
    ):
        raise TeacherPretrainingContractError(
            "completion manifest is missing its exact immutable summary"
        )
    if (
        payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind") != RUN_KIND
        or payload.get("status") != "completed"
        or payload.get("runManifestSha256") != manifest.sha256
        or payload.get("summary") != summary
    ):
        raise TeacherPretrainingContractError("completion manifest contract mismatch")
    return summary


def _resolved_workspace_file(
    workspace: Path, relative: object, *, label: str, parent: Path | None = None
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise TeacherPretrainingContractError(f"{label} path is invalid")
    unresolved = workspace / relative
    if unresolved.is_symlink():
        raise TeacherPretrainingContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(parent.resolve() if parent is not None else workspace)
    except ValueError as error:
        raise TeacherPretrainingContractError(f"{label} path escapes its root") from error
    if path.is_symlink() or not path.is_file():
        raise TeacherPretrainingContractError(f"{label} is missing or a symlink")
    return path


def _validate_evaluations(
    report: Mapping[str, object],
    *,
    contract: TeacherPretrainingContract,
    holdout_positions: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw = report.get("evaluations")
    expected_steps = list(
        range(
            contract.evaluation_interval,
            contract.optimizer_steps + 1,
            contract.evaluation_interval,
        )
    )
    if not isinstance(raw, list) or len(raw) != len(expected_steps):
        raise TeacherPretrainingContractError(
            "pretraining report must contain all eight fixed holdout evaluations"
        )
    evaluations: list[dict[str, object]] = []
    for expected_step, item in zip(expected_steps, raw):
        if not isinstance(item, Mapping) or set(item) != {
            "step",
            "positions",
            "policyLoss",
            "valueLoss",
            "totalLoss",
        }:
            raise TeacherPretrainingContractError("holdout evaluation schema is invalid")
        if item.get("step") != expected_step or item.get("positions") != holdout_positions:
            raise TeacherPretrainingContractError(
                "holdout evaluation cursor/coverage is invalid"
            )
        numbers: dict[str, float] = {}
        for field in ("policyLoss", "valueLoss", "totalLoss"):
            value = item.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise TeacherPretrainingContractError(
                    f"holdout evaluation {field} is invalid"
                )
            numbers[field] = float(value)
        if not math.isclose(
            numbers["totalLoss"],
            numbers["policyLoss"] + numbers["valueLoss"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise TeacherPretrainingContractError(
                "holdout total loss is not policy+value"
            )
        evaluations.append(dict(item))
    best = min(evaluations, key=lambda item: float(item["totalLoss"]))
    reported_best = report.get("best")
    if not isinstance(reported_best, Mapping) or dict(reported_best) != best:
        raise TeacherPretrainingContractError(
            "reported best model is not the strict minimum fixed-holdout loss"
        )
    return evaluations, best


def _validate_training_report(
    training: object,
    *,
    contract: TeacherPretrainingContract,
    dataset: TeacherDataset,
) -> None:
    if not isinstance(training, Mapping) or set(training) != {
        "steps",
        "loss",
        "policyLossMean",
        "valueLossMean",
        "gradientNormMean",
        "inputTiming",
        "preencodedCache",
    }:
        raise TeacherPretrainingContractError(
            "pretraining optimizer/timing report schema is invalid"
        )
    if training.get("steps") != contract.optimizer_steps:
        raise TeacherPretrainingContractError("pretraining optimizer-step report is invalid")
    timing = training.get("inputTiming")
    if not isinstance(timing, Mapping) or set(timing) != {
        "dataBatchSecondsMean",
        "dataBatchSecondsMaximum",
        "optimizerSecondsMean",
        "optimizerSecondsMaximum",
    }:
        raise TeacherPretrainingContractError("pretraining input timing is invalid")
    for value in timing.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise TeacherPretrainingContractError("pretraining timing is non-finite")
    cache = training.get("preencodedCache")
    expected_bytes = _preencoded_cache_bytes(
        dataset.train_positions + dataset.holdout_positions,
        contract.input_channels,
    )
    if not isinstance(cache, Mapping) or set(cache) != {
        "schemaVersion",
        "algorithmVersion",
        "datasetSha256",
        "bytes",
        "maximumBytes",
        "trainPositions",
        "holdoutPositions",
        "buildSeconds",
    }:
        raise TeacherPretrainingContractError(
            "pretraining cache report schema is invalid"
        )
    build_seconds = cache.get("buildSeconds")
    if (
        cache.get("schemaVersion") != PREENCODED_CACHE_SCHEMA_VERSION
        or cache.get("algorithmVersion") != PREENCODED_CACHE_ALGORITHM_VERSION
        or cache.get("datasetSha256") != dataset.dataset_fingerprint
        or cache.get("bytes") != expected_bytes
        or cache.get("maximumBytes") != PREENCODED_CACHE_MAX_BYTES
        or cache.get("trainPositions") != dataset.train_positions
        or cache.get("holdoutPositions") != dataset.holdout_positions
        or isinstance(build_seconds, bool)
        or not isinstance(build_seconds, (int, float))
        or not math.isfinite(float(build_seconds))
        or build_seconds < 0
    ):
        raise TeacherPretrainingContractError(
            "pretraining cache report does not match the frozen dataset/cache"
        )


def load_completed_pretraining_bundle(
    summary_path: str | Path,
    *,
    workspace_root: str | Path,
    teacher_root: str | Path = DEFAULT_TEACHER_ROOT,
) -> CompletedPretrainingBundle:
    """Deeply verify one completed warm model before any pilot consumes it."""

    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    source = Path(summary_path)
    source = source if source.is_absolute() else workspace / source
    if source.is_symlink():
        raise TeacherPretrainingContractError(
            "pretraining summary must not be a symlink"
        )
    source = source.resolve()
    try:
        source.relative_to(control_root)
    except ValueError as error:
        raise TeacherPretrainingContractError(
            "pretraining summary must be inside the Phase 4 control root"
        ) from error
    if source.name != SUMMARY_NAME or source.is_symlink() or not source.is_file():
        raise TeacherPretrainingContractError(
            "pretraining summary is missing, renamed, or a symlink"
        )
    summary = _load_json(source, "pretraining summary")
    expected_summary_fields = {
        "schemaVersion",
        "status",
        "artifactPath",
        "artifactSha256",
        "modelStateSha256",
        "baseModelStateSha256",
        "teacherDatasetSha256",
        "reportPath",
        "reportSha256",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("schemaVersion") != SCHEMA_VERSION
        or summary.get("status") != "completed"
        or source.read_bytes() != canonical_json_bytes(summary)
    ):
        raise TeacherPretrainingContractError(
            "pretraining summary schema or canonical bytes are invalid"
        )
    for field in (
        "artifactSha256",
        "modelStateSha256",
        "baseModelStateSha256",
        "teacherDatasetSha256",
        "reportSha256",
    ):
        try:
            validate_sha256(summary.get(field), field)
        except ValueError as error:
            raise TeacherPretrainingContractError(str(error)) from error

    run_root = source.parent
    report_path = _resolved_workspace_file(
        workspace, summary.get("reportPath"), label="pretraining report", parent=run_root
    )
    artifact_path = _resolved_workspace_file(
        workspace,
        summary.get("artifactPath"),
        label="warm artifact",
        parent=run_root,
    )
    if report_path != run_root / REPORT_NAME or artifact_path != run_root / WARM_ARTIFACT_NAME:
        raise TeacherPretrainingContractError("pretraining output names are not canonical")
    if sha256_file(report_path) != summary.get("reportSha256"):
        raise TeacherPretrainingContractError("pretraining report file hash mismatch")
    if sha256_file(artifact_path) != summary.get("artifactSha256"):
        raise TeacherPretrainingContractError("warm artifact file hash mismatch")
    report = load_pretraining_report(report_path)
    artifact = load_warm_artifact(artifact_path)

    teacher_path = _canonical_teacher_root(workspace, teacher_root)
    dataset = load_teacher_dataset(teacher_path)
    if summary.get("teacherDatasetSha256") != dataset.dataset_fingerprint:
        raise TeacherPretrainingContractError("summary teacher dataset is stale")

    manifest = load_run_manifest(run_root / "run-manifest.json")
    provenance = manifest.payload.get("provenance")
    config_record = provenance.get("config") if isinstance(provenance, Mapping) else None
    source_record = provenance.get("source") if isinstance(provenance, Mapping) else None
    if not isinstance(config_record, Mapping) or not isinstance(source_record, Mapping):
        raise TeacherPretrainingContractError("pretraining run provenance is invalid")
    current_source = fingerprint_sources(workspace, _SOURCE_PATHS)
    if source_record.get("sha256") != current_source.sha256:
        raise TeacherPretrainingContractError("pretraining source fingerprint is stale")

    contract = TeacherPretrainingContract.production()
    contract_payload = contract.to_payload()
    memory_smoke = report.get("memorySmoke")
    if not isinstance(memory_smoke, Mapping):
        raise TeacherPretrainingContractError("pretraining memory-smoke evidence is missing")
    verified_memory = verify_memory_smoke_evidence(
        memory_smoke, control_root, workspace
    )
    _native, native_backend_sha256 = load_trusted_native_encoder()
    baseline = verify_phase4d_baseline_report(control_root)
    expected_config = {
        "kind": RUN_KIND,
        "contract": contract_payload,
        "teacherDataset": dataset.snapshot(),
        "memorySmoke": verified_memory,
        "nativeEncoderBackendSha256": native_backend_sha256,
    }
    expected_config_fingerprint = fingerprint_config(expected_config)
    if (
        config_record.get("payload") != expected_config
        or config_record.get("sha256") != expected_config_fingerprint.sha256
    ):
        raise TeacherPretrainingContractError(
            "pretraining run config does not bind current verified inputs"
        )
    expected_budget = ResourceBudget(
        artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
        minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
        gpu_hours=PHASE4D_GPU_HOUR_LIMIT,
        metadata={
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "stage": PHASE4D_STAGE,
            "stageBaselineGlobalGpuHours": PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
            "baselineCalibration": baseline,
        },
    ).to_payload()
    if manifest.payload.get("resourceBudget") != expected_budget:
        raise TeacherPretrainingContractError("pretraining resource budget changed")
    metadata = manifest.payload.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("phase") != "4D"
        or metadata.get("kind") != RUN_KIND
        or metadata.get("teacherReplayUsedAfterPretraining") is not False
    ):
        raise TeacherPretrainingContractError("pretraining run metadata is invalid")

    expected_report_fields = {
        "schemaVersion",
        "kind",
        "status",
        "createdAt",
        "completedAt",
        "runManifestSha256",
        "teacherDatasetSha256",
        "baseModelStateSha256",
        "modelStateSha256",
        "warmArtifactPath",
        "contract",
        "dataset",
        "memorySmoke",
        "nativeEncoderBackendSha256",
        "baselineCalibration",
        "optimizerResetRequiredForSelfPlay",
        "teacherReplayAllowedInSelfPlay",
        "best",
        "evaluations",
        "training",
        "resourcesAtReport",
        "finalCheckpoint",
        "reportCoreSha256",
    }
    if set(report) != expected_report_fields or (
        report.get("createdAt") != manifest.payload.get("createdAt")
        or report.get("runManifestSha256") != manifest.sha256
        or report.get("teacherDatasetSha256") != dataset.dataset_fingerprint
        or report.get("baseModelStateSha256") != PRODUCTION_BASE_MODEL_SHA256
        or report.get("modelStateSha256") != summary.get("modelStateSha256")
        or report.get("warmArtifactPath") != summary.get("artifactPath")
        or report.get("contract") != contract_payload
        or report.get("dataset") != dataset.snapshot()
        or report.get("memorySmoke") != verified_memory
        or report.get("nativeEncoderBackendSha256") != native_backend_sha256
        or report.get("baselineCalibration") != baseline
        or report.get("optimizerResetRequiredForSelfPlay") is not True
        or report.get("teacherReplayAllowedInSelfPlay") is not False
    ):
        raise TeacherPretrainingContractError("pretraining report lineage is invalid")
    evaluations, best = _validate_evaluations(
        report, contract=contract, holdout_positions=dataset.holdout_positions
    )
    _validate_training_report(
        report.get("training"),
        contract=contract,
        dataset=dataset,
    )

    if (
        artifact.get("schemaVersion") != WARM_ARTIFACT_SCHEMA_VERSION
        or artifact.get("format") != "quoridor-phase4-model-only"
        or artifact.get("network") != contract.network_config().to_dict()
        or artifact.get("baseModelStateSha256") != PRODUCTION_BASE_MODEL_SHA256
        or artifact.get("modelStateSha256") != report.get("modelStateSha256")
        or artifact.get("provenance")
        != {
            "teacherDatasetSha256": dataset.dataset_fingerprint,
            "pretrainingReportSha256": summary["reportSha256"],
        }
    ):
        raise TeacherPretrainingContractError("warm artifact production lineage is invalid")

    identity = CheckpointIdentity(
        run_manifest_sha256=manifest.sha256,
        config_sha256=expected_config_fingerprint.sha256,
        source_sha256=current_source.sha256,
    )
    checkpoint_store = Phase4CheckpointStore(
        run_root / "checkpoints",
        identity=identity,
        artifact_root=control_root,
        artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
        max_to_keep=CHECKPOINT_KEEP_COUNT,
    )
    checkpoint = checkpoint_store.latest_valid(map_location="cpu")
    final_record = report.get("finalCheckpoint")
    if (
        checkpoint is None
        or checkpoint.rejected_newer
        or not isinstance(final_record, Mapping)
        or set(final_record) != {"generation", "step", "path", "sha256"}
        or final_record.get("generation") != checkpoint.record.generation
        or final_record.get("step") != contract.optimizer_steps
        or checkpoint.record.step != contract.optimizer_steps
        or final_record.get("sha256") != checkpoint.record.sha256
        or final_record.get("path") != _relative(checkpoint.record.data_path, workspace)
    ):
        raise TeacherPretrainingContractError("final pretraining checkpoint is invalid")
    generation_data = checkpoint.payload.get("generationData")
    replay_snapshot = checkpoint.payload.get("replaySnapshot")
    trainer_runtime = checkpoint.payload.get("trainerRuntime")
    bundled_state = checkpoint.payload.get("model")
    if (
        not isinstance(generation_data, Mapping)
        or set(generation_data)
        != {
            "schemaVersion",
            "kind",
            "generation",
            "reason",
            "step",
            "lastEvaluationStep",
            "bestStep",
            "bestMetrics",
            "evaluations",
            "trainingMetrics",
            "contractSha256",
            "teacherDatasetSha256",
            "teacherDataset",
            "baseModelStateSha256",
            "currentModelStateSha256",
            "bestModelStateSha256",
        }
        or generation_data.get("schemaVersion") != SCHEMA_VERSION
        or generation_data.get("kind") != RUN_KIND
        or generation_data.get("generation") != checkpoint.record.generation
        or generation_data.get("reason") != "evaluation"
        or generation_data.get("step") != contract.optimizer_steps
        or generation_data.get("lastEvaluationStep") != contract.optimizer_steps
        or generation_data.get("teacherDataset") != dataset.snapshot()
        or generation_data.get("teacherDatasetSha256") != dataset.dataset_fingerprint
        or generation_data.get("contractSha256")
        != fingerprint_config(contract_payload).sha256
        or generation_data.get("baseModelStateSha256") != PRODUCTION_BASE_MODEL_SHA256
        or generation_data.get("bestStep") != best["step"]
        or generation_data.get("bestMetrics") != best
        or generation_data.get("evaluations") != evaluations
        or not isinstance(generation_data.get("trainingMetrics"), Mapping)
        or generation_data["trainingMetrics"].get("count")
        != contract.optimizer_steps
        or replay_snapshot != _checkpoint_replay_snapshot(dataset)
        or not isinstance(trainer_runtime, Mapping)
        or trainer_runtime.get("step") != contract.optimizer_steps
        or trainer_runtime.get("ampEnabled") is not True
        or not isinstance(bundled_state, Mapping)
    ):
        raise TeacherPretrainingContractError(
            "final checkpoint does not contain the completed pretraining state"
        )
    best_state = {
        key.removeprefix("best."): tensor
        for key, tensor in bundled_state.items()
        if isinstance(key, str) and key.startswith("best.")
    }
    current_state = {
        key.removeprefix("current."): tensor
        for key, tensor in bundled_state.items()
        if isinstance(key, str) and key.startswith("current.")
    }
    if (
        not best_state
        or not current_state
        or canonical_model_state_sha256(best_state) != summary.get("modelStateSha256")
        or canonical_model_state_sha256(current_state)
        != generation_data.get("currentModelStateSha256")
        or generation_data.get("bestModelStateSha256")
        != summary.get("modelStateSha256")
    ):
        raise TeacherPretrainingContractError("checkpoint model tensor hashes are invalid")

    completion_path = run_root / COMPLETION_MANIFEST_NAME
    completion = _load_json(completion_path, "pretraining completion manifest")
    validated_summary = _validate_completion(
        workspace=workspace,
        run_root=run_root,
        manifest=manifest,
        dataset_sha256=dataset.dataset_fingerprint,
    )
    if validated_summary != summary:
        raise TeacherPretrainingContractError("completion/summary binding is invalid")
    artifact_metadata = {
        key: value for key, value in artifact.items() if key != "modelState"
    }
    return CompletedPretrainingBundle(
        summary=dict(summary),
        report=dict(report),
        artifact_metadata=artifact_metadata,
        run_manifest=manifest,
        completion_manifest=completion,
        teacher_dataset=dataset,
    )


def run_teacher_pretraining(
    *,
    workspace_root: str | Path,
    teacher_root: str | Path,
    run_root: str | Path,
    resume: bool,
    device_index: int,
    external_gpu_hours: float,
    contract: TeacherPretrainingContract | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    contract = contract or TeacherPretrainingContract.production()
    contract.assert_production()
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    teacher_path = _canonical_teacher_root(workspace, teacher_root)
    output_path = Path(run_root)
    output_path = output_path if output_path.is_absolute() else workspace / output_path
    output_path = output_path.resolve()
    try:
        output_path.relative_to(control_root)
    except ValueError as error:
        raise ValueError("run root must be inside training/runs/phase4") from error
    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    if (
        isinstance(external_gpu_hours, bool)
        or not isinstance(external_gpu_hours, (int, float))
        or not math.isfinite(float(external_gpu_hours))
        or external_gpu_hours < 0
    ):
        raise ValueError("external_gpu_hours must be finite and non-negative")

    invocation_id = f"phase4d-teacher-pretrain-{os.getpid()}-{time.time_ns()}"
    _publish_progress(
        progress_callback,
        stage="preflight",
        status="started",
        runRoot=str(output_path),
        resume=resume,
    )
    with RunLock(control_root / GPU_LOCK_NAME, run_id=invocation_id):
        baseline = verify_phase4d_baseline_report(control_root)
        _publish_progress(
            progress_callback,
            stage="preflight",
            status="baseline-verified",
        )
        dataset = load_teacher_dataset(
            teacher_path,
            progress_callback=progress_callback,
        )
        memory_smoke = find_valid_memory_smoke(control_root, workspace)
        _publish_progress(
            progress_callback,
            stage="preflight",
            status="memory-smoke-verified",
            reportSha256=memory_smoke["fileSha256"],
        )
        native, native_backend_sha256 = load_trusted_native_encoder()
        _publish_progress(
            progress_callback,
            stage="preflight",
            status="native-encoder-verified",
            nativeEncoderBackendSha256=native_backend_sha256,
        )
        ledger = GpuBudgetLedger(control_root / GPU_LEDGER_NAME)
        stage_ledger = Phase4StageBudgetLedger(control_root / PHASE4_STAGE_LEDGER_NAME)

        manifest_exists = (output_path / "run-manifest.json").exists()
        completion_exists = (output_path / COMPLETION_MANIFEST_NAME).exists()
        existing_lineage: (
            tuple[RunManifestRecord, CheckpointIdentity, bool] | None
        ) = None
        if manifest_exists:
            if not resume:
                raise TeacherPretrainingContractError(
                    "pretraining run exists; --resume is required"
                )
            existing_lineage = _lineage(
                workspace=workspace,
                run_root=output_path,
                contract=contract,
                dataset=dataset,
                memory_smoke=memory_smoke,
                native_backend_sha256=native_backend_sha256,
                baseline=baseline,
                resume=resume,
            )
        if existing_lineage is not None and completion_exists:
            manifest, _identity, _resumed = existing_lineage
            completed = _validate_completion(
                workspace=workspace,
                run_root=output_path,
                manifest=manifest,
                dataset_sha256=dataset.dataset_fingerprint,
            )
            if completed is None:
                raise TeacherPretrainingContractError(
                    "pretraining completion manifest disappeared during validation"
                )
            _publish_progress(
                progress_callback,
                stage="preflight",
                status="already-completed",
                runRoot=str(output_path),
            )
            return completed
        if not manifest_exists and resume:
            raise TeacherPretrainingContractError(
                "--resume requested but pretraining run manifest is missing"
            )

        _assert_finalization_resources(control_root)
        _assert_gpu_budget_preflight(
            ledger,
            stage_ledger,
            float(external_gpu_hours),
        )
        # Cache construction is deliberately before every CUDA query, runtime
        # environment collection, immutable lineage write, and GPU ledger
        # mutation. An interrupted cache build therefore consumes no GPU
        # budget and leaves no new lineage behind.
        cache = build_teacher_preencoded_cache(
            dataset,
            native_encoder=native.encode_states,
            contract=contract,
            native_backend_sha256=native_backend_sha256,
            progress_callback=progress_callback,
        )
        if not torch.cuda.is_available():
            raise TeacherPretrainingResourceError("PyTorch CUDA is not available")
        if device_index >= torch.cuda.device_count():
            raise TeacherPretrainingResourceError("CUDA device index is unavailable")
        if shutil.which("nvidia-smi") is None:
            raise TeacherPretrainingResourceError("nvidia-smi is required")
        _assert_finalization_resources(control_root)
        _assert_gpu_budget_preflight(
            ledger, stage_ledger, float(external_gpu_hours)
        )
        if existing_lineage is None:
            manifest, identity, resumed = _lineage(
                workspace=workspace,
                run_root=output_path,
                contract=contract,
                dataset=dataset,
                memory_smoke=memory_smoke,
                native_backend_sha256=native_backend_sha256,
                baseline=baseline,
                resume=resume,
            )
        else:
            manifest, identity, resumed = existing_lineage
        ledger.record_external_hours(float(external_gpu_hours))
        stage_ledger.ensure_stage(
            PHASE4D_STAGE,
            PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
            PHASE4D_GPU_HOUR_LIMIT,
        )
        stage_ledger.assert_available(PHASE4D_STAGE, ledger.total_hours)
        if ledger.total_hours >= GPU_HOUR_BUDGET:
            raise TeacherPretrainingResourceError("72 GPU-hour global budget is exhausted")

        ledger.begin(invocation_id, external_hours=float(external_gpu_hours))
        started = time.monotonic()
        accounting_status = "failed"
        try:
            seed_everything(contract.seed)
            current = PolicyValueNetwork(contract.network_config())
            base_model_sha256 = canonical_model_state_sha256(current)
            if base_model_sha256 != PRODUCTION_BASE_MODEL_SHA256:
                raise TeacherPretrainingContractError(
                    "seeded 192x16 base model hash differs from the frozen cold-start base"
                )
            bundle = CheckpointModels(current)
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device_index)
            current.to(device)
            trainer = AlphaZeroTrainer(
                current,
                learning_rate=contract.learning_rate,
                weight_decay=contract.weight_decay,
                max_grad_norm=contract.max_grad_norm,
                use_amp=contract.use_amp,
                schedule_steps=contract.optimizer_steps,
                minimum_learning_rate=contract.minimum_learning_rate,
            )
            if not trainer.amp_enabled or not trainer.scaler.is_enabled():
                raise TeacherPretrainingContractError("CUDA AMP/scaler is not enabled")
            if {str(parameter.dtype) for parameter in current.parameters()} != {"torch.float32"}:
                raise TeacherPretrainingContractError("model parameters must remain float32")
            if loaded_native_backend_hash() != native_backend_sha256:
                raise TeacherPretrainingContractError(
                    "native encoder binary changed after immutable lineage creation"
                )
            contract_sha256 = fingerprint_config(contract.to_payload()).sha256
            checkpoints = Phase4CheckpointStore(
                output_path / "checkpoints",
                identity=identity,
                artifact_root=control_root,
                artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
                max_to_keep=CHECKPOINT_KEEP_COUNT,
            )
            loaded: Phase4CheckpointLoad | None = None
            last_evaluation_step = 0
            best_step: int | None = None
            best_metrics: dict[str, object] | None = None
            evaluations: list[dict[str, object]] = []
            metrics = MetricAccumulator()
            next_generation = 0
            if resumed:
                restored = restore_pretraining_checkpoint(
                    checkpoints,
                    contract=contract,
                    contract_sha256=contract_sha256,
                    dataset=dataset,
                    bundle=bundle,
                    trainer=trainer,
                    base_model_sha256=base_model_sha256,
                    map_location=device,
                )
                if restored is not None:
                    loaded, cursor = restored
                    (
                        last_evaluation_step,
                        best_step,
                        best_metrics,
                        evaluations,
                        metrics,
                    ) = cursor
                    next_generation = loaded.record.generation + 1

            _publish_progress(
                progress_callback,
                stage="training",
                status="started",
                step=trainer.step,
                optimizerSteps=contract.optimizer_steps,
                resumed=loaded is not None,
                cacheBytes=cache.byte_size,
            )
            input_performance_gate = InputPerformanceGate()
            guard = _ResourceGuard(
                control_root=control_root,
                ledger=ledger,
                stage_ledger=stage_ledger,
                ledger_run_id=invocation_id,
                started=started,
            )
            status = "failed"
            interrupted = False
            try:
                guard.check(force=True)
                while True:
                    if (
                        trainer.step > 0
                        and trainer.step % contract.evaluation_interval == 0
                        and last_evaluation_step < trainer.step
                    ):
                        _publish_progress(
                            progress_callback,
                            stage="holdout",
                            status="started",
                            step=trainer.step,
                            positions=cache.holdout.position_count,
                        )
                        evaluation = {
                            "step": trainer.step,
                            **evaluate_cached_holdout(
                                current,
                                cache.holdout,
                                contract,
                            ),
                        }
                        _publish_progress(
                            progress_callback,
                            stage="holdout",
                            status="completed",
                            **evaluation,
                        )
                        evaluations.append(evaluation)
                        last_evaluation_step = trainer.step
                        if best_metrics is None or float(evaluation["totalLoss"]) < float(
                            best_metrics["totalLoss"]
                        ):
                            bundle.update_best()
                            best_step = trainer.step
                            best_metrics = dict(evaluation)
                        checkpoint_generation = next_generation
                        next_generation += 1
                        _save_checkpoint(
                            checkpoints,
                            next_generation=checkpoint_generation,
                            contract_sha256=contract_sha256,
                            dataset=dataset,
                            bundle=bundle,
                            trainer=trainer,
                            base_model_sha256=base_model_sha256,
                            last_evaluation_step=last_evaluation_step,
                            best_step=best_step,
                            best_metrics=best_metrics,
                            evaluations=evaluations,
                            metrics=metrics,
                            reason="evaluation",
                        )
                        _publish_progress(
                            progress_callback,
                            stage="checkpoint",
                            status="completed",
                            generation=checkpoint_generation,
                            step=trainer.step,
                            reason="evaluation",
                        )
                        guard.check(force=True)
                        continue
                    if trainer.step >= contract.optimizer_steps:
                        break
                    next_step = trainer.step + 1
                    batch_started = time.monotonic()
                    batch = cached_training_batch(
                        cache.train,
                        contract,
                        step=next_step,
                        device=device,
                    )
                    torch.cuda.synchronize(device)
                    data_batch_seconds = max(
                        0.0,
                        time.monotonic() - batch_started,
                    )
                    try:
                        input_gate_completed = input_performance_gate.observe(
                            data_batch_seconds
                        )
                    except TeacherPretrainingResourceError:
                        _publish_progress(
                            progress_callback,
                            stage="input-performance-gate",
                            status="failed",
                            steps=input_performance_gate.count,
                            dataBatchSecondsMean=(
                                input_performance_gate.mean_seconds
                            ),
                            maximumMeanSeconds=(
                                INPUT_PERFORMANCE_MAX_MEAN_SECONDS
                            ),
                        )
                        raise
                    if input_gate_completed:
                        _publish_progress(
                            progress_callback,
                            stage="input-performance-gate",
                            status="completed",
                            steps=input_performance_gate.count,
                            dataBatchSecondsMean=(
                                input_performance_gate.mean_seconds
                            ),
                            maximumMeanSeconds=(
                                INPUT_PERFORMANCE_MAX_MEAN_SECONDS
                            ),
                        )
                    with _optimizer_step_boundary():
                        optimizer_started = time.monotonic()
                        metric = trainer.train_step(batch)
                        torch.cuda.synchronize(device)
                        optimizer_seconds = max(
                            0.0,
                            time.monotonic() - optimizer_started,
                        )
                        metrics.add(
                            metric,
                            data_batch_seconds=data_batch_seconds,
                            optimizer_seconds=optimizer_seconds,
                        )
                    if (
                        trainer.step == 1
                        or
                        trainer.step % TRAINING_PROGRESS_INTERVAL == 0
                        or trainer.step == contract.optimizer_steps
                    ):
                        _publish_progress(
                            progress_callback,
                            stage="training",
                            status="progress",
                            step=trainer.step,
                            optimizerSteps=contract.optimizer_steps,
                            loss=metric.loss,
                            policyLoss=metric.policy_loss,
                            valueLoss=metric.value_loss,
                            gradientNorm=metric.grad_norm,
                            learningRate=metric.learning_rate,
                            dataBatchSeconds=data_batch_seconds,
                            dataBatchSecondsMean=(
                                metrics.data_batch_seconds_sum / metrics.count
                            ),
                            optimizerSeconds=optimizer_seconds,
                            optimizerSecondsMean=(
                                metrics.optimizer_seconds_sum / metrics.count
                            ),
                        )
                    guard.check()
                if (
                    trainer.step != contract.optimizer_steps
                    or last_evaluation_step != contract.optimizer_steps
                    or best_step is None
                    or best_metrics is None
                    or len(evaluations)
                    != contract.optimizer_steps // contract.evaluation_interval
                ):
                    raise TeacherPretrainingContractError(
                        "pretraining did not complete the frozen optimizer/evaluation workload"
                    )
                status = "running"
            except KeyboardInterrupt:
                interrupted = True
                status = "interrupted"
                try:
                    if _runtime_checkpoint_consistent(
                        contract=contract,
                        trainer=trainer,
                        metrics=metrics,
                        last_evaluation_step=last_evaluation_step,
                        best_step=best_step,
                        best_metrics=best_metrics,
                        evaluations=evaluations,
                    ):
                        _save_checkpoint(
                            checkpoints,
                            next_generation=next_generation,
                            contract_sha256=contract_sha256,
                            dataset=dataset,
                            bundle=bundle,
                            trainer=trainer,
                            base_model_sha256=base_model_sha256,
                            last_evaluation_step=last_evaluation_step,
                            best_step=best_step,
                            best_metrics=best_metrics,
                            evaluations=evaluations,
                            metrics=metrics,
                            reason="interrupt",
                        )
                        _publish_progress(
                            progress_callback,
                            stage="checkpoint",
                            status="completed",
                            generation=next_generation,
                            step=trainer.step,
                            reason="interrupt",
                        )
                except (
                    ArtifactBudgetError,
                    ImmutableArtifactExistsError,
                    Phase4CheckpointError,
                    TeacherPretrainingResourceError,
                ):
                    pass
            except TeacherPretrainingResourceError:
                status = "interrupted"
                try:
                    if _runtime_checkpoint_consistent(
                        contract=contract,
                        trainer=trainer,
                        metrics=metrics,
                        last_evaluation_step=last_evaluation_step,
                        best_step=best_step,
                        best_metrics=best_metrics,
                        evaluations=evaluations,
                    ):
                        _save_checkpoint(
                            checkpoints,
                            next_generation=next_generation,
                            contract_sha256=contract_sha256,
                            dataset=dataset,
                            bundle=bundle,
                            trainer=trainer,
                            base_model_sha256=base_model_sha256,
                            last_evaluation_step=last_evaluation_step,
                            best_step=best_step,
                            best_metrics=best_metrics,
                            evaluations=evaluations,
                            metrics=metrics,
                            reason="resource-stop",
                        )
                        _publish_progress(
                            progress_callback,
                            stage="checkpoint",
                            status="completed",
                            generation=next_generation,
                            step=trainer.step,
                            reason="resource-stop",
                        )
                except (
                    ArtifactBudgetError,
                    ImmutableArtifactExistsError,
                    Phase4CheckpointError,
                    TeacherPretrainingResourceError,
                ):
                    pass
                raise
            if interrupted:
                raise KeyboardInterrupt

            final_loaded = checkpoints.latest_valid(map_location="cpu")
            if final_loaded is None or final_loaded.record.step != contract.optimizer_steps:
                raise TeacherPretrainingContractError("final checkpoint is missing")
            final_global_hours = ledger.update(
                invocation_id,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                status="running",
            )
            resources = {
                "globalGpuHours": final_global_hours,
                "stageGpuHours": stage_ledger.usage(PHASE4D_STAGE, final_global_hours),
                "artifactBytes": directory_size(control_root),
                "freeDiskBytes": int(shutil.disk_usage(control_root).free),
            }
            best_model_sha256 = canonical_model_state_sha256(bundle.best)
            report_path = output_path / REPORT_NAME
            if report_path.exists():
                existing_report = load_pretraining_report(report_path)
                if (
                    existing_report.get("runManifestSha256") != manifest.sha256
                    or existing_report.get("teacherDatasetSha256")
                    != dataset.dataset_fingerprint
                    or existing_report.get("baseModelStateSha256") != base_model_sha256
                    or existing_report.get("modelStateSha256") != best_model_sha256
                ):
                    raise TeacherPretrainingContractError(
                        "existing immutable pretraining report does not match recovered state"
                    )
            else:
                report = _report_payload(
                    manifest=manifest,
                    workspace=workspace,
                    run_root=output_path,
                    contract=contract,
                    dataset=dataset,
                    memory_smoke=memory_smoke,
                    native_backend_sha256=native_backend_sha256,
                    baseline=baseline,
                    base_model_sha256=base_model_sha256,
                    best_model_sha256=best_model_sha256,
                    best_step=best_step,
                    best_metrics=best_metrics,
                    evaluations=evaluations,
                    metrics=metrics,
                    cache=cache,
                    resource_snapshot=resources,
                    checkpoint=final_loaded,
                )
                _assert_finalization_resources(control_root)
                atomic_create_bytes(report_path, canonical_json_bytes(report))
            report_file_sha256 = sha256_file(report_path)
            artifact_path = output_path / WARM_ARTIFACT_NAME
            if artifact_path.exists():
                existing_artifact = load_warm_artifact(artifact_path)
                if (
                    existing_artifact.get("modelStateSha256") != best_model_sha256
                    or existing_artifact.get("baseModelStateSha256") != base_model_sha256
                    or existing_artifact.get("provenance")
                    != {
                        "teacherDatasetSha256": dataset.dataset_fingerprint,
                        "pretrainingReportSha256": report_file_sha256,
                    }
                ):
                    raise TeacherPretrainingContractError(
                        "existing immutable warm artifact does not match report/model"
                    )
            else:
                _assert_finalization_resources(control_root)
                _atomic_create_torch_artifact(
                    _warm_artifact_payload(
                        contract=contract,
                        bundle=bundle,
                        base_model_sha256=base_model_sha256,
                        dataset_sha256=dataset.dataset_fingerprint,
                        report_file_sha256=report_file_sha256,
                    ),
                    artifact_path,
                    control_root,
                )
            summary = _summary(
                workspace=workspace,
                run_root=output_path,
                dataset_sha256=dataset.dataset_fingerprint,
            )
            _assert_metadata_resources(control_root)
            _write_or_validate_summary(output_path, summary)
            completion_path = output_path / COMPLETION_MANIFEST_NAME
            if completion_path.exists():
                validated = _validate_completion(
                    workspace=workspace,
                    run_root=output_path,
                    manifest=manifest,
                    dataset_sha256=dataset.dataset_fingerprint,
                )
                if validated != summary:
                    raise TeacherPretrainingContractError("completion summary mismatch")
            else:
                _assert_metadata_resources(control_root)
                _write_completion_manifest(
                    workspace=workspace,
                    run_root=output_path,
                    manifest=manifest,
                    summary=summary,
                )
            accounting_status = "completed"
            _publish_progress(
                progress_callback,
                stage="training",
                status="completed",
                step=trainer.step,
                optimizerSteps=contract.optimizer_steps,
            )
            return summary

        except KeyboardInterrupt:
            accounting_status = "interrupted"
            raise
        except TeacherPretrainingResourceError:
            accounting_status = "interrupted"
            raise
        finally:
            ledger.update(
                invocation_id,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                status=accounting_status,
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4D fixed CUDA AMP hard-teacher pretraining"
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gpu-hours-used", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )
    return parser


def _raise_interrupt(_signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _print_progress(payload: Mapping[str, object]) -> None:
    print(
        json.dumps(dict(payload), separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    previous_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        summary = run_teacher_pretraining(
            workspace_root=args.workspace_root,
            teacher_root=DEFAULT_TEACHER_ROOT,
            run_root=args.run_root,
            resume=args.resume,
            device_index=args.device_index,
            external_gpu_hours=args.gpu_hours_used,
            progress_callback=_print_progress,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "status": "interrupted",
                    "message": (
                        "interrupted; rerun the launcher or use resume mode "
                        "matching any existing lineage; the launcher auto-resumes "
                        "an existing lineage, and only completed training "
                        "boundaries may have checkpoints"
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(
        json.dumps(summary, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "WARM_ARTIFACT_SCHEMA_VERSION",
    "PRODUCTION_BASE_MODEL_SHA256",
    "PREENCODED_CACHE_SCHEMA_VERSION",
    "PREENCODED_CACHE_ALGORITHM_VERSION",
    "PREENCODED_CACHE_MAX_BYTES",
    "RUN_KIND",
    "DEFAULT_TEACHER_ROOT",
    "DEFAULT_RUN_ROOT",
    "TeacherPretrainingError",
    "TeacherPretrainingContractError",
    "TeacherPretrainingResourceError",
    "TeacherPretrainingContract",
    "TeacherDataset",
    "PreencodedReplaySplit",
    "TeacherPreencodedCache",
    "CompletedPretrainingBundle",
    "MetricAccumulator",
    "InputPerformanceGate",
    "CheckpointModels",
    "load_teacher_dataset",
    "build_teacher_preencoded_cache",
    "find_valid_memory_smoke",
    "training_batch",
    "cached_training_batch",
    "evaluate_holdout",
    "evaluate_cached_holdout",
    "load_pretraining_report",
    "load_warm_artifact",
    "load_completed_pretraining_bundle",
    "run_teacher_pretraining",
    "create_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
