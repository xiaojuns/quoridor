"""Phase 4D CUDA AMP training-memory smoke with fixed production dimensions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .calibration import (
    GPU_HOUR_BUDGET,
    GPU_LEDGER_NAME,
    GPU_LOCK_NAME,
    GpuBudgetLedger,
    NvidiaSmiProfiler,
)
from .config import seed_everything
from .network import (
    ACTION_SIZE,
    BOARD_SIZE,
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
from .phase4_execution import load_phase4_execution_contract
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    ImmutableArtifactExistsError,
    ResourceBudget,
    RunAlreadyLockedError,
    RunLock,
    atomic_create_bytes,
    build_run_manifest,
    canonical_json_bytes,
    save_run_manifest,
    sha256_bytes,
    utc_timestamp,
)
from .trainer import AlphaZeroTrainer, TrainMetrics


SCHEMA_VERSION = 1
RUN_KIND = "phase4d-cuda-amp-training-memory-smoke"
RUN_ID_PREFIX = "phase4d-training-memory-smoke"
REPORT_RESERVE_BYTES = 4 * 1024**2
DEFAULT_SEED = 20260714
LEDGER_HEARTBEAT_SECONDS = 5.0


class TrainingMemorySmokeError(RuntimeError):
    """Base class for Phase 4D smoke failures."""


class TrainingMemorySmokePreflightError(TrainingMemorySmokeError):
    """Raised before any model or synthetic training batch reaches CUDA."""


class TrainingMemorySmokeBudgetExceeded(TrainingMemorySmokeError):
    """Raised when a Phase 4 resource stop line is reached."""


class TrainingMemorySmokeInvariantError(TrainingMemorySmokeError):
    """Raised when the smoke no longer exercises the frozen training path."""


@dataclass(frozen=True, slots=True)
class TrainingMemorySmokeContract:
    """Frozen production-shaped workload; CLI options cannot resize it."""

    input_channels: int = PHASE4_INPUT_CHANNELS
    board_size: int = BOARD_SIZE
    action_size: int = ACTION_SIZE
    channels: int = 192
    residual_blocks: int = 16
    se_reduction: int = 8
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 256
    batch_size: int = 512
    warmup_steps: int = 1
    measured_steps: int = 3
    learning_rate: float = 0.0003
    minimum_learning_rate: float = 0.00003
    schedule_steps: int = 20_000
    weight_decay: float = 0.0001
    max_grad_norm: float = 5.0

    def __post_init__(self) -> None:
        expected: dict[str, int | float] = {
            "input_channels": PHASE4_INPUT_CHANNELS,
            "board_size": BOARD_SIZE,
            "action_size": ACTION_SIZE,
            "channels": 192,
            "residual_blocks": 16,
            "se_reduction": 8,
            "policy_channels": 32,
            "value_channels": 32,
            "value_hidden": 256,
            "batch_size": 512,
            "warmup_steps": 1,
            "measured_steps": 3,
            "learning_rate": 0.0003,
            "minimum_learning_rate": 0.00003,
            "schedule_steps": 20_000,
            "weight_decay": 0.0001,
            "max_grad_norm": 5.0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen at {value!r} for Phase 4D")

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.measured_steps

    def network_config(self) -> NetworkConfig:
        return NetworkConfig(
            input_channels=self.input_channels,
            board_size=self.board_size,
            action_size=self.action_size,
            channels=self.channels,
            blocks=self.residual_blocks,
            se_reduction=self.se_reduction,
            policy_channels=self.policy_channels,
            value_channels=self.value_channels,
            value_hidden=self.value_hidden,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "network": {
                "inputChannels": self.input_channels,
                "boardSize": self.board_size,
                "actionSize": self.action_size,
                "channels": self.channels,
                "residualBlocks": self.residual_blocks,
                "seReduction": self.se_reduction,
                "policyChannels": self.policy_channels,
                "valueChannels": self.value_channels,
                "valueHidden": self.value_hidden,
            },
            "batchSize": self.batch_size,
            "warmupSteps": self.warmup_steps,
            "measuredSteps": self.measured_steps,
            "totalOptimizerSteps": self.total_steps,
            "optimizer": {
                "kind": "AdamW",
                "learningRate": self.learning_rate,
                "weightDecay": self.weight_decay,
                "maxGradNorm": self.max_grad_norm,
            },
            "scheduler": {
                "kind": "CosineAnnealingLR",
                "scheduleSteps": self.schedule_steps,
                "minimumLearningRate": self.minimum_learning_rate,
            },
            "precision": {
                "ampRequired": True,
                "autocastDtype": "float16",
                "parameterDtype": "float32",
            },
            "data": {
                "source": "deterministic-synthetic-memory-smoke",
                "retained": False,
            },
            "forbiddenWork": {
                "selfPlay": True,
                "replayWrites": True,
                "checkpointWrites": True,
                "modelArtifactWrites": True,
            },
        }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise TrainingMemorySmokePreflightError(
                f"artifact tree must not contain symlinks: {item}"
            )
        if item.is_file():
            total += item.stat().st_size
    return total


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise TrainingMemorySmokePreflightError(
                f"cannot find an existing filesystem ancestor for {path}"
            )
        candidate = candidate.parent
    return candidate


def assert_report_write_reserve(
    output_root: str | Path,
    *,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
) -> dict[str, int]:
    """Fail before any ledger/manifest write could consume the final reserve."""

    root = Path(output_root).resolve()
    free_bytes = int(getattr(disk_usage(_existing_ancestor(root)), "free"))
    artifact_bytes = _directory_size(root)
    if free_bytes - REPORT_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise TrainingMemorySmokePreflightError(
            "the smoke report reserve would cross the 20 GiB free-disk stop line"
        )
    if artifact_limit_bytes not in {
        DEFAULT_ARTIFACT_LIMIT_BYTES,
        MAXIMUM_ARTIFACT_LIMIT_BYTES,
    }:
        raise ValueError("artifact_limit_bytes must be the frozen 8 or 12 GiB cap")
    if artifact_bytes + REPORT_RESERVE_BYTES > artifact_limit_bytes:
        raise TrainingMemorySmokePreflightError(
            "the smoke report reserve would exceed the frozen artifact cap"
        )
    return {
        "freeDiskBytes": free_bytes,
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": artifact_limit_bytes,
    }


def collect_preflight(
    output_root: str | Path,
    *,
    device_index: int,
    global_gpu_hours: float,
    stage_ledger: Phase4StageBudgetLedger,
    cuda_api: object = torch.cuda,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    find_executable: Callable[[str], str | None] = shutil.which,
    baseline_verifier: Callable[[str | Path], dict[str, object]] = (
        verify_phase4d_baseline_report
    ),
    stage: str = PHASE4D_STAGE,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
) -> dict[str, object]:
    """Validate every fixed local resource requirement before CUDA allocation."""

    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    if (
        isinstance(global_gpu_hours, bool)
        or not isinstance(global_gpu_hours, (int, float))
        or not math.isfinite(float(global_gpu_hours))
        or float(global_gpu_hours) < 0
    ):
        raise ValueError("global_gpu_hours must be finite and non-negative")
    global_gpu_hours = float(global_gpu_hours)
    if artifact_limit_bytes not in {
        DEFAULT_ARTIFACT_LIMIT_BYTES,
        MAXIMUM_ARTIFACT_LIMIT_BYTES,
    }:
        raise ValueError("artifact_limit_bytes must be the frozen 8 or 12 GiB cap")
    if global_gpu_hours >= GPU_HOUR_BUDGET:
        raise TrainingMemorySmokePreflightError(
            f"the Phase 4 {GPU_HOUR_BUDGET:g} GPU-hour budget is exhausted"
        )
    root = Path(output_root).resolve()
    try:
        baseline_evidence = baseline_verifier(root)
        stage_usage = stage_ledger.assert_available(stage, global_gpu_hours)
    except RuntimeError as error:
        raise TrainingMemorySmokePreflightError(str(error)) from error

    if not bool(cuda_api.is_available()):
        raise TrainingMemorySmokePreflightError("PyTorch CUDA is not available")
    device_count = int(cuda_api.device_count())
    if device_index >= device_count:
        raise TrainingMemorySmokePreflightError(
            f"CUDA device index {device_index} is outside {device_count} visible devices"
        )
    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        raise TrainingMemorySmokePreflightError("nvidia-smi is not available")

    filesystem_path = _existing_ancestor(root)
    usage = disk_usage(filesystem_path)
    free_bytes = int(getattr(usage, "free"))
    artifact_bytes = _directory_size(root)
    if free_bytes - REPORT_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise TrainingMemorySmokePreflightError(
            "less than the Phase 4 20 GiB free-disk stop line plus report reserve "
            "is available"
        )
    if artifact_bytes + REPORT_RESERVE_BYTES > artifact_limit_bytes:
        raise TrainingMemorySmokePreflightError(
            "the Phase 4 artifact tree cannot reserve a report below the frozen cap"
        )

    properties = cuda_api.get_device_properties(device_index)
    gpu: dict[str, object] = {
        "index": device_index,
        "name": str(properties.name),
        "totalMemoryBytes": int(properties.total_memory),
    }
    for name in ("major", "minor"):
        value = getattr(properties, name, None)
        if type(value) is int:
            gpu.setdefault("computeCapability", []).append(value)
    try:
        free_memory, visible_memory = cuda_api.mem_get_info(device_index)
        gpu["freeMemoryBytesAtStart"] = int(free_memory)
        gpu["visibleMemoryBytes"] = int(visible_memory)
    except (AttributeError, RuntimeError, TypeError):
        pass

    return {
        "passed": True,
        "localOnly": True,
        "cuda": gpu,
        "nvidiaSmi": {
            "path": str(Path(nvidia_smi).resolve()),
            "required": True,
        },
        "disk": {
            "filesystemPath": str(filesystem_path),
            "freeBytesAtStart": free_bytes,
            "minimumFreeBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        },
        "artifacts": {
            "root": str(root),
            "bytesAtStart": artifact_bytes,
            "limitBytes": artifact_limit_bytes,
            "reportReserveBytes": REPORT_RESERVE_BYTES,
        },
        "gpuHours": {
            "globalUsedAtStart": global_gpu_hours,
            "globalLimit": GPU_HOUR_BUDGET,
            "stage": stage_usage,
            "stageBaselineEvidence": baseline_evidence,
        },
    }


def resource_snapshot(
    output_root: str | Path,
    *,
    global_gpu_hours: float,
    stage_ledger: Phase4StageBudgetLedger,
    stage: str = PHASE4D_STAGE,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
) -> dict[str, object]:
    root = Path(output_root).resolve()
    free_bytes = int(getattr(disk_usage(_existing_ancestor(root)), "free"))
    artifact_bytes = _directory_size(root)
    stage_usage = stage_ledger.usage(stage, global_gpu_hours)
    within_limits = (
        global_gpu_hours < GPU_HOUR_BUDGET
        and bool(stage_usage["available"])
        and free_bytes - REPORT_RESERVE_BYTES >= DEFAULT_MINIMUM_FREE_DISK_BYTES
        and artifact_bytes + REPORT_RESERVE_BYTES <= artifact_limit_bytes
    )
    return {
        "globalGpuHours": global_gpu_hours,
        "globalGpuHourLimit": GPU_HOUR_BUDGET,
        "stageGpuHours": stage_usage,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "artifactBytesBeforeFinalReport": artifact_bytes,
        "artifactLimitBytes": artifact_limit_bytes,
        "reportReserveBytes": REPORT_RESERVE_BYTES,
        "withinLimits": within_limits,
    }


def assert_runtime_resources(
    output_root: str | Path,
    *,
    global_gpu_hours: float,
    stage_ledger: Phase4StageBudgetLedger,
    stage: str = PHASE4D_STAGE,
    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
) -> dict[str, object]:
    snapshot = resource_snapshot(
        output_root,
        global_gpu_hours=global_gpu_hours,
        stage_ledger=stage_ledger,
        stage=stage,
        artifact_limit_bytes=artifact_limit_bytes,
    )
    if not snapshot["withinLimits"]:
        raise TrainingMemorySmokeBudgetExceeded(
            "a Phase 4D GPU, disk, or artifact stop line was reached"
        )
    return snapshot


def build_synthetic_batch(
    *,
    batch_size: int,
    input_channels: int,
    board_size: int,
    action_size: int,
    device: torch.device | str,
    seed: int,
) -> dict[str, Tensor]:
    """Create deterministic targets without reading or retaining replay data."""

    for name, value in (
        ("batch_size", batch_size),
        ("input_channels", input_channels),
        ("board_size", board_size),
        ("action_size", action_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2^63)")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)
    states = torch.randn(
        batch_size,
        input_channels,
        board_size,
        board_size,
        device=resolved_device,
        dtype=torch.float32,
        generator=generator,
    )
    policies = torch.rand(
        batch_size,
        action_size,
        device=resolved_device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min_(torch.finfo(torch.float32).eps)
    policies.div_(policies.sum(dim=1, keepdim=True))
    values = torch.rand(
        batch_size,
        device=resolved_device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(2.0).sub_(1.0)
    return {"states": states, "policies": policies, "values": values}


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _series_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise TrainingMemorySmokeInvariantError(
            "training metric series must be non-empty and finite"
        )
    normalized = [float(value) for value in values]
    return {
        "first": normalized[0],
        "last": normalized[-1],
        "minimum": min(normalized),
        "maximum": max(normalized),
        "mean": sum(normalized) / len(normalized),
        "p50": _percentile(normalized, 0.50),
        "p95": _percentile(normalized, 0.95),
    }


def run_training_steps(
    trainer: AlphaZeroTrainer,
    batch: dict[str, Tensor],
    *,
    warmup_steps: int,
    measured_steps: int,
    require_cuda_amp: bool,
    synchronize: Callable[[], None] = lambda: None,
    progress_callback: Callable[[int, int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Exercise actual model, loss, backward, optimizer, scaler, and scheduler paths."""

    if type(trainer) is not AlphaZeroTrainer:
        raise TypeError("trainer must be the production AlphaZeroTrainer")
    for name, value in (
        ("warmup_steps", warmup_steps),
        ("measured_steps", measured_steps),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(require_cuda_amp) is not bool:
        raise TypeError("require_cuda_amp must be boolean")
    if require_cuda_amp and not trainer.amp_enabled:
        raise TrainingMemorySmokeInvariantError("CUDA AMP is not enabled")
    if require_cuda_amp and not trainer.scaler.is_enabled():
        raise TrainingMemorySmokeInvariantError("CUDA AMP GradScaler is not enabled")
    if trainer.step != 0 or trainer.optimizer.state:
        raise TrainingMemorySmokeInvariantError(
            "the memory smoke must start from a fresh optimizer without checkpoint state"
        )
    if not isinstance(trainer.optimizer, torch.optim.AdamW):
        raise TrainingMemorySmokeInvariantError("the optimizer must be AdamW")
    if not isinstance(
        trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
    ):
        raise TrainingMemorySmokeInvariantError(
            "the scheduler must be CosineAnnealingLR"
        )

    states = batch.get("states")
    policies = batch.get("policies")
    values = batch.get("values")
    if not all(isinstance(item, Tensor) for item in (states, policies, values)):
        raise TypeError("batch must contain tensor states, policies, and values")
    assert isinstance(states, Tensor)
    assert isinstance(policies, Tensor)
    assert isinstance(values, Tensor)
    expected_policy_shape = tuple(policies.shape)
    expected_value_shape = (states.shape[0],)
    forward_observations: list[dict[str, object]] = []

    def record_outputs(_module: object, _inputs: object, outputs: object) -> None:
        if (
            not isinstance(outputs, tuple)
            or len(outputs) != 2
            or not all(isinstance(output, Tensor) for output in outputs)
        ):
            raise TrainingMemorySmokeInvariantError(
                "model must return policy logits and value tensors"
            )
        policy_logits, predicted_values = outputs
        if tuple(policy_logits.shape) != expected_policy_shape:
            raise TrainingMemorySmokeInvariantError(
                "actual policy output does not match the target batch"
            )
        if tuple(predicted_values.shape) != expected_value_shape:
            raise TrainingMemorySmokeInvariantError(
                "actual value output does not contain one value per position"
            )
        forward_observations.append(
            {
                "shapes": {
                    "policy": list(policy_logits.shape),
                    "value": list(predicted_values.shape),
                },
                "cudaAutocastEnabled": torch.is_autocast_enabled("cuda"),
                "policyDtype": str(policy_logits.dtype).removeprefix("torch."),
                "valueDtype": str(predicted_values.dtype).removeprefix("torch."),
            }
        )

    total_steps = warmup_steps + measured_steps
    metrics: list[TrainMetrics] = []
    durations: list[float] = []
    handle = trainer.model.register_forward_hook(record_outputs)
    try:
        if progress_callback is not None:
            progress_callback(0, total_steps)
        for completed in range(1, total_steps + 1):
            synchronize()
            started = clock()
            metric = trainer.train_step(batch)
            synchronize()
            elapsed = clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise TrainingMemorySmokeInvariantError(
                    "training step clock returned an invalid duration"
                )
            metrics.append(metric)
            durations.append(elapsed)
            if progress_callback is not None:
                progress_callback(completed, total_steps)
    finally:
        handle.remove()

    if trainer.step != total_steps or len(forward_observations) != total_steps:
        raise TrainingMemorySmokeInvariantError(
            "training did not execute exactly one real forward pass per optimizer step"
        )
    if require_cuda_amp and not all(
        observation["cudaAutocastEnabled"] is True
        and observation["policyDtype"] == "float16"
        and observation["valueDtype"] == "float16"
        for observation in forward_observations
    ):
        raise TrainingMemorySmokeInvariantError(
            "the actual CUDA forward path did not use float16 autocast"
        )
    if not trainer.optimizer.state:
        raise TrainingMemorySmokeInvariantError(
            "AdamW optimizer state was not materialized by the real updates"
        )

    measured_durations = durations[warmup_steps:]
    measured_seconds = sum(measured_durations)
    return {
        "warmupSteps": warmup_steps,
        "measuredSteps": measured_steps,
        "optimizerSteps": trainer.step,
        "forwardCalls": len(forward_observations),
        "actualOutputShapes": forward_observations[-1]["shapes"],
        "ampEnabled": trainer.amp_enabled,
        "amp": {
            "declaredEnabled": trainer.amp_enabled,
            "scalerEnabled": trainer.scaler.is_enabled(),
            "cudaAutocastObserved": all(
                observation["cudaAutocastEnabled"] is True
                for observation in forward_observations
            ),
            "policyOutputDtypes": sorted(
                {str(observation["policyDtype"]) for observation in forward_observations}
            ),
            "valueOutputDtypes": sorted(
                {str(observation["valueDtype"]) for observation in forward_observations}
            ),
        },
        "optimizer": {
            "kind": type(trainer.optimizer).__name__,
            "stateEntries": len(trainer.optimizer.state),
        },
        "scheduler": {
            "kind": type(trainer.scheduler).__name__,
            "lastEpoch": trainer.scheduler.last_epoch,
            "configuredSteps": trainer.schedule_steps,
        },
        "metrics": {
            "loss": _series_summary([metric.loss for metric in metrics]),
            "policyLoss": _series_summary(
                [metric.policy_loss for metric in metrics]
            ),
            "valueLoss": _series_summary([metric.value_loss for metric in metrics]),
            "gradientNorm": _series_summary(
                [metric.grad_norm for metric in metrics]
            ),
            "learningRate": _series_summary(
                [metric.learning_rate for metric in metrics]
            ),
        },
        "timing": {
            "allStepSeconds": _series_summary(durations),
            "measuredStepSeconds": _series_summary(measured_durations),
            "measuredTotalSeconds": measured_seconds,
            "measuredSamplesPerSecond": (
                states.shape[0] * measured_steps / measured_seconds
                if measured_seconds > 0
                else None
            ),
        },
    }


def torch_memory_snapshot(device: torch.device, cuda_api: object = torch.cuda) -> dict[str, int]:
    free_bytes, total_bytes = cuda_api.mem_get_info(device)
    return {
        "allocatedBytes": int(cuda_api.memory_allocated(device)),
        "reservedBytes": int(cuda_api.memory_reserved(device)),
        "peakAllocatedBytes": int(cuda_api.max_memory_allocated(device)),
        "peakReservedBytes": int(cuda_api.max_memory_reserved(device)),
        "driverFreeBytes": int(free_bytes),
        "driverTotalBytes": int(total_bytes),
    }


def audit_run_directory(
    run_dir: str | Path,
    *,
    expected_files: frozenset[str],
) -> dict[str, object]:
    root = Path(run_dir)
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TrainingMemorySmokeInvariantError(
                f"smoke run directory contains a symlink: {path}"
            )
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    actual = frozenset(files)
    if actual != expected_files:
        raise TrainingMemorySmokeInvariantError(
            "smoke run directory contains forbidden replay, checkpoint, model, or "
            f"other artifacts: expected {sorted(expected_files)}, got {sorted(actual)}"
        )
    return {
        "passed": True,
        "files": sorted(actual),
        "selfPlayGames": 0,
        "replayWrites": 0,
        "checkpointWrites": 0,
        "modelArtifactWrites": 0,
    }


def write_immutable_report(path: str | Path, report: Mapping[str, object]) -> Path:
    core = dict(report)
    core.pop("reportSha256", None)
    payload = {
        **core,
        "reportSha256": sha256_bytes(canonical_json_bytes(core)),
    }
    encoded = canonical_json_bytes(payload)
    if len(encoded) > REPORT_RESERVE_BYTES:
        raise TrainingMemorySmokeInvariantError(
            "training-memory smoke report exceeds its 4 MiB artifact reserve"
        )
    return atomic_create_bytes(path, encoded)


def load_immutable_report(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingMemorySmokeInvariantError(
            f"cannot load training-memory smoke report {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TrainingMemorySmokeInvariantError("smoke report must be a mapping")
    stored_hash = payload.get("reportSha256")
    core = dict(payload)
    core.pop("reportSha256", None)
    if (
        payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind") != RUN_KIND
        or not isinstance(stored_hash, str)
        or sha256_bytes(canonical_json_bytes(core)) != stored_hash
    ):
        raise TrainingMemorySmokeInvariantError(
            "smoke report schema or checksum is invalid"
        )
    return dict(payload)


def _build_training_components(
    contract: TrainingMemorySmokeContract,
    *,
    device: torch.device,
    seed: int,
) -> tuple[PolicyValueNetwork, AlphaZeroTrainer, dict[str, Tensor]]:
    seed_everything(seed)
    model = PolicyValueNetwork(contract.network_config()).to(device)
    trainer = AlphaZeroTrainer(
        model,
        learning_rate=contract.learning_rate,
        weight_decay=contract.weight_decay,
        max_grad_norm=contract.max_grad_norm,
        use_amp=True,
        schedule_steps=contract.schedule_steps,
        minimum_learning_rate=contract.minimum_learning_rate,
    )
    if not trainer.amp_enabled:
        raise TrainingMemorySmokeInvariantError(
            "the production AlphaZeroTrainer did not enable CUDA AMP"
        )
    batch = build_synthetic_batch(
        batch_size=contract.batch_size,
        input_channels=contract.input_channels,
        board_size=contract.board_size,
        action_size=contract.action_size,
        device=device,
        seed=seed,
    )
    expected_shapes = {
        "states": (
            contract.batch_size,
            contract.input_channels,
            contract.board_size,
            contract.board_size,
        ),
        "policies": (contract.batch_size, contract.action_size),
        "values": (contract.batch_size,),
    }
    if any(tuple(batch[name].shape) != shape for name, shape in expected_shapes.items()):
        raise TrainingMemorySmokeInvariantError(
            "synthetic training batch does not match the frozen dimensions"
        )
    return model, trainer, batch


def _error_payload(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def verify_staged_preparation(
    workspace: Path,
    execution: object,
) -> tuple[Mapping[str, object], str, str]:
    """Verify the preparation bound to a staged Phase 4F smoke contract."""

    from .phase4_budget import (
        PHASE4F18_PILOT_STAGE,
        PHASE4F35_PILOT_STAGE,
        PHASE4F36_PILOT_STAGE,
        PHASE4F_STAGE_A,
        PHASE4F_STAGE_A_CONTINUATION,
        PHASE4F_STAGE_B_EXTENDED,
        PHASE4F_STAGE_B_PILOT,
        PHASE4F_STAGE_B,
        PHASE4F_STAGE_B_CONTINUATION,
    )

    stage = getattr(execution, "stage", None)
    if stage == PHASE4F18_PILOT_STAGE:
        from .phase4f18_rebaseline import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f18-stage-a-pilot-preparation"
        evidence_path = str(prepared["pilot"]["executionPath"])
    elif stage == PHASE4F35_PILOT_STAGE:
        from .phase4f35_stage_a2 import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f35-stage-a2-pilot-preparation"
        evidence_path = str(prepared["pilot"]["executionPath"])
    elif stage == PHASE4F36_PILOT_STAGE:
        from .phase4f36_stage_a2 import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f36-stage-a2-pilot-v2-preparation"
        evidence_path = str(prepared["pilot"]["executionPath"])
    elif stage == PHASE4F_STAGE_A:
        from .phase4f19_stage_a import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f19-stage-a-formal-preparation"
        evidence_path = str(prepared["execution"]["path"])
    elif stage == PHASE4F_STAGE_A_CONTINUATION:
        from .phase4f21_stage_a import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f21-final-stage-a-preparation"
        evidence_path = str(prepared["execution"]["path"])
    elif stage == PHASE4F_STAGE_B_PILOT:
        from .phase4f23_stage_b import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f23-stage-b-pilot-preparation"
        evidence_path = str(prepared["execution"]["path"])
    elif stage == PHASE4F_STAGE_B:
        from .phase4f24_stage_b import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f24-stage-b-formal-preparation"
        evidence_path = str(prepared["execution"]["path"])
    elif stage == PHASE4F_STAGE_B_CONTINUATION:
        from .phase4f27_stage_b import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f27-stage-b-continuation-preparation"
        evidence_path = str(prepared["execution"]["path"])
    elif stage == PHASE4F_STAGE_B_EXTENDED:
        from .phase4f31_stage_b import _verified_preparation

        prepared = _verified_preparation(workspace)
        evidence_kind = "phase4f31-stage-b-extended-preparation"
        evidence_path = str(prepared["execution"]["path"])
    else:
        raise TrainingMemorySmokePreflightError(
            "only frozen Phase 4F staged-training contracts may override "
            "the memory-smoke stage"
        )
    return prepared, evidence_kind, evidence_path


def _run_locked(
    args: argparse.Namespace,
    *,
    workspace: Path,
    output_root: Path,
    run_id: str,
    global_ledger: GpuBudgetLedger,
    stage_ledger: Phase4StageBudgetLedger,
) -> int:
    contract = TrainingMemorySmokeContract()
    artifact_limit_bytes = DEFAULT_ARTIFACT_LIMIT_BYTES
    if args.execution_contract is None:
        stage = PHASE4D_STAGE
        stage_baseline = PHASE4D_BASELINE_GLOBAL_GPU_HOURS
        stage_limit = PHASE4D_GPU_HOUR_LIMIT
        baseline_run_id = None
        baseline_report_sha256 = None
        try:
            baseline_evidence = verify_phase4d_baseline_report(output_root)
        except RuntimeError as error:
            raise TrainingMemorySmokePreflightError(str(error)) from error
    else:
        execution = load_phase4_execution_contract(args.execution_contract)
        budget = execution.require_frozen_budget()
        prepared, evidence_kind, evidence_path = verify_staged_preparation(
            workspace,
            execution,
        )
        stage = execution.stage
        stage_baseline = float(budget.baseline_global_hours)
        stage_limit = float(budget.limit_hours)
        baseline_run_id = budget.baseline_run_id
        baseline_report_sha256 = budget.baseline_report_sha256
        baseline_evidence = {
            "status": "verified",
            "kind": evidence_kind,
            "reportPath": evidence_path,
            "reportSha256": prepared["reportSha256"],
            "sourceSha256": prepared["source"]["sha256"],
        }
        resources = prepared.get("resources")
        if not isinstance(resources, Mapping):
            raise TrainingMemorySmokePreflightError(
                "staged preparation is missing its resource contract"
            )
        prepared_limit = resources.get("artifactLimitBytes")
        if prepared_limit not in {
            DEFAULT_ARTIFACT_LIMIT_BYTES,
            MAXIMUM_ARTIFACT_LIMIT_BYTES,
        }:
            raise TrainingMemorySmokePreflightError(
                "staged preparation has an invalid artifact cap"
            )
        artifact_limit_bytes = int(prepared_limit)
    assert_report_write_reserve(
        output_root,
        artifact_limit_bytes=artifact_limit_bytes,
    )
    global_before = global_ledger.record_external_hours(args.gpu_hours_used)
    stage_ledger.ensure_stage(
        stage,
        stage_baseline,
        stage_limit,
        baseline_run_id=baseline_run_id,
        baseline_report_sha256=baseline_report_sha256,
    )
    preflight = collect_preflight(
        output_root,
        device_index=args.device_index,
        global_gpu_hours=global_before,
        stage_ledger=stage_ledger,
        baseline_verifier=lambda _root: baseline_evidence,
        stage=stage,
        artifact_limit_bytes=artifact_limit_bytes,
    )

    run_dir = output_root / "training-memory-smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run-manifest.json"
    budget = ResourceBudget(
        artifact_limit_bytes=artifact_limit_bytes,
        minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
        gpu_hours=stage_limit,
        metadata={
            "phase": "4D",
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "stage": stage,
            "stageBaselineGlobalGpuHours": stage_baseline,
            "selfPlayAllowed": False,
            "replayWritesAllowed": False,
            "checkpointWritesAllowed": False,
            "modelArtifactWritesAllowed": False,
        },
    )
    manifest_config = {
        "kind": RUN_KIND,
        "seed": args.seed,
        "deviceIndex": args.device_index,
        "externalGpuHoursFloor": global_ledger.external_hours_floor,
        "globalGpuHoursBeforeRun": global_before,
        "stageBaselineEvidence": baseline_evidence,
        "contract": contract.to_payload(),
    }
    manifest = build_run_manifest(
        workspace_root=workspace,
        config=manifest_config,
        budget=budget,
        run_id=run_id,
        metadata={
            "phase": "4D",
            "kind": RUN_KIND,
            "purpose": "cuda-amp-training-memory-smoke",
        },
    )
    save_run_manifest(manifest, manifest_path)
    audit_run_directory(run_dir, expected_files=frozenset({"run-manifest.json"}))

    ledger_at_begin = global_ledger.begin(
        run_id,
        external_hours=args.gpu_hours_used,
    )
    if ledger_at_begin != global_before:
        raise TrainingMemorySmokeInvariantError(
            "global GPU ledger changed while the Phase 4 lock was held"
        )

    started = time.perf_counter()
    last_heartbeat = started
    status = "running"
    error_payload: dict[str, str] | None = None
    workload: dict[str, object] | None = None
    profiler: NvidiaSmiProfiler | None = None
    telemetry: dict[str, object] = {
        "available": False,
        "backend": "nvidia-smi",
        "sampleCount": 0,
        "error": "profiler did not start",
    }
    memory_before: dict[str, int] | None = None
    memory_after: dict[str, int] | None = None
    memory_after_cleanup: dict[str, int] | None = None
    memory_error: dict[str, str] | None = None
    artifact_audit: dict[str, object] | None = None
    model: PolicyValueNetwork | None = None
    trainer: AlphaZeroTrainer | None = None
    batch: dict[str, Tensor] | None = None
    parameter_count: int | None = None
    parameter_dtype: str | None = None
    device = torch.device(f"cuda:{args.device_index}")

    def progress_guard(completed: int, total: int) -> None:
        nonlocal last_heartbeat
        now = time.perf_counter()
        elapsed = now - started
        estimated_global = global_before + elapsed / 3600.0
        assert_runtime_resources(
            output_root,
            global_gpu_hours=estimated_global,
            stage_ledger=stage_ledger,
            stage=stage,
            artifact_limit_bytes=artifact_limit_bytes,
        )
        if now - last_heartbeat >= LEDGER_HEARTBEAT_SECONDS or completed == total:
            persisted_global = global_ledger.update(
                run_id,
                elapsed_seconds=elapsed,
                status="running",
            )
            if abs(persisted_global - estimated_global) > 1e-9:
                raise TrainingMemorySmokeInvariantError(
                    "global GPU ledger accounting diverged from elapsed runtime"
                )
            stage_ledger.assert_available(stage, persisted_global)
            last_heartbeat = now

    try:
        torch.cuda.set_device(args.device_index)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        profiler = NvidiaSmiProfiler(args.device_index)
        profiler.start()
        memory_before = torch_memory_snapshot(device)
        model, trainer, batch = _build_training_components(
            contract,
            device=device,
            seed=args.seed,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
        if parameter_dtypes != {"torch.float32"}:
            raise TrainingMemorySmokeInvariantError(
                "the frozen model parameters must use float32"
            )
        parameter_dtype = "float32"
        workload = run_training_steps(
            trainer,
            batch,
            warmup_steps=contract.warmup_steps,
            measured_steps=contract.measured_steps,
            require_cuda_amp=True,
            synchronize=lambda: torch.cuda.synchronize(device),
            progress_callback=progress_guard,
        )
        torch.cuda.synchronize(device)
        status = "completed"
    except KeyboardInterrupt as error:
        status = "interrupted"
        error_payload = {
            "type": "KeyboardInterrupt",
            "message": "Phase 4D training-memory smoke interrupted by user",
        }
    except Exception as error:
        status = "failed"
        error_payload = _error_payload(error)
    finally:
        if profiler is not None:
            telemetry = profiler.stop()
        try:
            memory_after = torch_memory_snapshot(device)
        except Exception as error:
            memory_error = _error_payload(error)
        batch = None
        trainer = None
        model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
            memory_after_cleanup = torch_memory_snapshot(device)
        except Exception as error:
            if memory_error is None:
                memory_error = _error_payload(error)

    elapsed_seconds = time.perf_counter() - started
    final_global_hours = global_before + elapsed_seconds / 3600.0
    resources = resource_snapshot(
        output_root,
        global_gpu_hours=final_global_hours,
        stage_ledger=stage_ledger,
        stage=stage,
        artifact_limit_bytes=artifact_limit_bytes,
    )
    try:
        artifact_audit = audit_run_directory(
            run_dir,
            expected_files=frozenset({"run-manifest.json"}),
        )
    except TrainingMemorySmokeInvariantError as error:
        if error_payload is None:
            error_payload = _error_payload(error)
        status = "failed"

    gates = {
        "workloadCompleted": status == "completed" and workload is not None,
        "frozenNetwork192x16": (
            contract.channels == 192 and contract.residual_blocks == 16
        ),
        "frozenBatch512": contract.batch_size == 512,
        "cudaAmpEnabled": bool(
            workload
            and workload.get("ampEnabled") is True
            and isinstance(workload.get("amp"), Mapping)
            and workload["amp"].get("declaredEnabled") is True
            and workload["amp"].get("scalerEnabled") is True
            and workload["amp"].get("cudaAutocastObserved") is True
            and workload["amp"].get("policyOutputDtypes") == ["float16"]
            and workload["amp"].get("valueOutputDtypes") == ["float16"]
        ),
        "policyAndValueForward": bool(
            workload
            and workload.get("forwardCalls") == contract.total_steps
            and workload.get("actualOutputShapes")
            == {
                "policy": [contract.batch_size, contract.action_size],
                "value": [contract.batch_size],
            }
        ),
        "backwardOptimizerScheduler": bool(
            workload
            and workload.get("optimizerSteps") == contract.total_steps
            and isinstance(workload.get("optimizer"), Mapping)
            and workload["optimizer"].get("kind") == "AdamW"
            and workload["optimizer"].get("stateEntries", 0) > 0
            and isinstance(workload.get("scheduler"), Mapping)
            and workload["scheduler"].get("kind") == "CosineAnnealingLR"
        ),
        "torchMemoryCaptured": bool(
            memory_after
            and memory_after.get("peakAllocatedBytes", 0) > 0
            and memory_after.get("peakReservedBytes", 0) > 0
            and memory_error is None
        ),
        "nvidiaSmiCaptured": bool(
            telemetry.get("available") is True
            and telemetry.get("sampleCount", 0) >= 2
            and telemetry.get("error") is None
        ),
        "float32ModelParameters": parameter_dtype == "float32",
        "noSelfPlayReplayCheckpointOrModelArtifacts": artifact_audit is not None,
        "resourceLimits": bool(resources["withinLimits"]),
    }
    passed = status == "completed" and all(gates.values())
    if status == "completed" and not passed:
        status = "failed"
        failed_gates = sorted(name for name, passed_gate in gates.items() if not passed_gate)
        error_payload = {
            "type": "TrainingMemorySmokeInvariantError",
            "message": f"Phase 4D smoke gates failed: {', '.join(failed_gates)}",
        }

    persisted_global_hours = global_ledger.update(
        run_id,
        elapsed_seconds=elapsed_seconds,
        status=status,
    )
    if abs(persisted_global_hours - final_global_hours) > 1e-9:
        raise TrainingMemorySmokeInvariantError(
            "final global GPU ledger accounting diverged from elapsed runtime"
        )

    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "runId": run_id,
        "createdAt": manifest.payload["createdAt"],
        "completedAt": utc_timestamp(),
        "status": status,
        "passed": passed,
        "error": error_payload,
        "runManifestSha256": manifest.sha256,
        "contract": contract.to_payload(),
        "preflight": preflight,
        "workload": workload,
        "model": {
            "parameterCount": parameter_count,
            "parameterDtype": parameter_dtype,
            "saved": False,
        },
        "memory": {
            "torch": {
                "beforeModelAndBatch": memory_before,
                "afterWorkload": memory_after,
                "afterCleanup": memory_after_cleanup,
                "error": memory_error,
            },
            "nvidiaSmi": telemetry,
        },
        "forbiddenWorkAudit": artifact_audit,
        "activity": {
            "selfPlayGames": 0,
            "replayWrites": 0,
            "checkpointWrites": 0,
            "modelArtifactWrites": 0,
            "syntheticBatchRetained": False,
        },
        "resourceUsage": resources,
        "elapsedSeconds": elapsed_seconds,
        "gates": gates,
    }
    write_immutable_report(report_path, report)
    stored_report = load_immutable_report(report_path)
    summary = {
        "status": status,
        "passed": passed,
        "runId": run_id,
        "reportPath": str(report_path),
        "manifestPath": str(manifest_path),
        "reportSha256": stored_report["reportSha256"],
        "optimizerSteps": workload.get("optimizerSteps") if workload else None,
        "peakAllocatedBytes": (
            memory_after.get("peakAllocatedBytes") if memory_after else None
        ),
        "peakReservedBytes": (
            memory_after.get("peakReservedBytes") if memory_after else None
        ),
        "nvidiaSmiSamples": telemetry.get("sampleCount"),
        "globalGpuHours": persisted_global_hours,
        "stage": stage,
        "stageGpuHours": resources["stageGpuHours"]["usedHours"],
        "phase4dGpuHours": (
            resources["stageGpuHours"]["usedHours"]
            if stage == PHASE4D_STAGE
            else None
        ),
    }
    print(canonical_json_bytes(summary).decode("utf-8"), end="")
    return 0 if passed else (130 if status == "interrupted" else 1)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4D fixed 192x16, batch-512 CUDA AMP training-memory smoke"
        ),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--gpu-hours-used",
        type=float,
        default=0.0,
        help="monotonic Phase 4 GPU hours used outside the persisted global ledger",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=None,
        help="bind the smoke to an approved non-Phase-4D stage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise ValueError("seed must be in [0, 2^63)")
    workspace = Path(__file__).resolve().parents[2]
    output_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    try:
        output_root.relative_to(workspace)
    except ValueError as error:
        raise TrainingMemorySmokePreflightError(
            "Phase 4 control root must remain inside the local workspace"
        ) from error
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_id = f"{RUN_ID_PREFIX}-{timestamp}-{os.getpid()}"

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_for_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_for_sigterm)
    try:
        with RunLock(output_root / GPU_LOCK_NAME, run_id=run_id):
            global_ledger = GpuBudgetLedger(output_root / GPU_LEDGER_NAME)
            stage_ledger = Phase4StageBudgetLedger(
                output_root / PHASE4_STAGE_LEDGER_NAME
            )
            return _run_locked(
                args,
                workspace=workspace,
                output_root=output_root,
                run_id=run_id,
                global_ledger=global_ledger,
                stage_ledger=stage_ledger,
            )
    except (
        TrainingMemorySmokePreflightError,
        RunAlreadyLockedError,
        ImmutableArtifactExistsError,
    ) as error:
        print(
            canonical_json_bytes(
                {
                    "status": "preflight-failed",
                    "error": _error_payload(error),
                }
            ).decode("utf-8"),
            end="",
            file=sys.stderr,
        )
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "RUN_KIND",
    "TrainingMemorySmokeError",
    "TrainingMemorySmokePreflightError",
    "TrainingMemorySmokeBudgetExceeded",
    "TrainingMemorySmokeInvariantError",
    "TrainingMemorySmokeContract",
    "collect_preflight",
    "resource_snapshot",
    "assert_runtime_resources",
    "build_synthetic_batch",
    "run_training_steps",
    "torch_memory_snapshot",
    "audit_run_directory",
    "write_immutable_report",
    "load_immutable_report",
    "create_parser",
    "main",
]
