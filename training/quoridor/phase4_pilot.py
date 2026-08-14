"""Phase 4D warm/cold pilot orchestration and paired screen.

The shell entry point deliberately invokes the durable Phase 4 trainer one
generation at a time.  This module owns the immutable experiment contract,
lane validation, and the final 64-game paired screen; it never performs
self-play training itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

import torch

from .calibration import (
    GPU_HOUR_BUDGET,
    LEGACY_GPU_HOUR_BUDGET,
    GPU_LEDGER_NAME,
    GPU_LOCK_NAME,
    GpuBudgetLedger,
)
from .config import ExperimentConfig, load_experiment_config, resolve_device, seed_everything
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import PolicyValueNetwork
from .openings import OpeningSuite, build_opening_suite, load_opening_suite
from .paired_arena import (
    GATE_BOOTSTRAP_SAMPLES,
    GATE_BOOTSTRAP_SEED,
    BoundMoveSelector,
    PairedArenaResult,
    SelectorIdentity,
    loaded_native_backend_hash,
    run_paired_arena,
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
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_training import (
    PHASE4D_PILOT_GENERATIONS,
    PHASE4_WARM_ARTIFACT_SCHEMA_VERSION,
    Phase4ResumeError,
    Phase4ResourceLimitReached,
    _RuntimeBudgetGuard,
    _network_config,
    validate_selfplay_invocation,
    validate_phase4_config,
)
from .replay_store import ReplayStore
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    GenerationReportRecord,
    GenerationReportStore,
    ImmutableArtifactExistsError,
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    normalize_json_mapping,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)
from .selectors import MCTSSelector
from .statistics import PairedGameResult, summarize_paired_results
from .storage import directory_size
from .teacher_pretraining import load_completed_pretraining_bundle


PHASE4D_PILOT_SCHEMA_VERSION = 1
PHASE4D_PRETRAINING_SUMMARY_SCHEMA_VERSION = 1
PHASE4D_ARENA_SIMULATIONS = 256
PHASE4D_ARENA_OPENINGS = 32
PHASE4D_ARENA_GAMES = 64
PHASE4D_RAW_WIN_RATE_THRESHOLD = 0.55
PHASE4D_ONE_SIDED_LOWER_THRESHOLD = 0.50
PHASE4D_SEED = 20_260_714
PHASE4D_GAMES_PER_GENERATION = 64
PHASE4D_ACTORS = 32
PHASE4D_TRAINING_STEPS = 64
PHASE4D_BATCH_SIZE = 512
PHASE4D_REPLAY_CAPACITY = 100_000
PHASE4D_CONFIG_RELATIVE = Path("effective-configs")
PHASE4D_CONTRACT_NAME = "pilot-contract.json"
PHASE4D_RESULT_MANIFEST_NAME = "result-manifest.json"
PHASE4D_DEFAULT_ROOT = Path("training/runs/phase4/phase4d-pilot")
PHASE4D_DEV_OPENINGS = Path("training/fixtures/openings/dev.json")
PHASE4D_BASE_CONFIG_RELATIVE = Path("training/configs/phase4-bootstrap.json")
PHASE4D_BASE_CONFIG_SHA256 = (
    "08d884616da9b8d6ee19c3d6cf29c15100aad61aa624499e2ec8a955eb38a887"
)
_METADATA_WRITE_OVERHEAD_BYTES = 64 * 1024
_JSONL_RECORD_WRITE_OVERHEAD_BYTES = 4 * 1024
_ARENA_RESULT_WRITE_OVERHEAD_BYTES = 16 * 1024

_PRETRAINING_SUMMARY_KEYS = frozenset(
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
    }
)
_WARM_ARTIFACT_KEYS = frozenset(
    {
        "schemaVersion",
        "format",
        "network",
        "baseModelStateSha256",
        "modelStateSha256",
        "provenance",
        "modelState",
    }
)


class Phase4PilotError(RuntimeError):
    """Base class for an invalid or incomplete Phase 4D pilot."""


class Phase4PilotContractError(Phase4PilotError):
    """Raised when immutable pilot provenance does not verify."""


class Phase4PilotIncomplete(Phase4PilotError):
    """Raised when the paired screen is requested before both lanes finish."""


@dataclass(frozen=True, slots=True)
class LaneState:
    lane: str
    config: ExperimentConfig
    manifest_sha256: str | None
    source_sha256: str | None
    reports: tuple[GenerationReportRecord, ...]
    native_backend_sha256: str | None = None

    @property
    def next_generation(self) -> int:
        return len(self.reports)

    @property
    def complete(self) -> bool:
        return self.next_generation == PHASE4D_PILOT_GENERATIONS


@dataclass(frozen=True, slots=True)
class PilotDecision:
    selected_lane: str
    warm_selected: bool
    raw_win_rate: float
    one_sided_95_lower: float
    raw_threshold_passed: bool
    lower_threshold_passed: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "candidateLane": "warm",
            "opponentLane": "cold",
            "selectedLane": self.selected_lane,
            "warmSelected": self.warm_selected,
            "rawWinRate": self.raw_win_rate,
            "oneSided95Lower": self.one_sided_95_lower,
            "checks": {
                "rawWinRate": {
                    "passed": self.raw_threshold_passed,
                    "requirement": ">= 0.55",
                },
                "oneSided95Lower": {
                    "passed": self.lower_threshold_passed,
                    "requirement": "> 0.50",
                },
            },
            "interval": {
                "method": "opening-pair-cluster-percentile",
                "interpretation": (
                    "lower 5% tail of the central 90% interval equals a "
                    "one-sided 95% lower bound"
                ),
                "bootstrapSeed": GATE_BOOTSTRAP_SEED,
                "bootstrapSamples": GATE_BOOTSTRAP_SAMPLES,
            },
            "screenOnly": True,
            "formalChampionDeclaration": False,
            "reason": (
                "warm passed both frozen Phase 4D pilot checks"
                if self.warm_selected
                else "warm did not pass both frozen Phase 4D pilot checks"
            ),
        }


def _load_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4PilotContractError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise Phase4PilotContractError(f"{label} must contain a JSON object")
    try:
        return normalize_json_mapping(value, label)
    except (TypeError, ValueError) as error:
        raise Phase4PilotContractError(f"{label} is not canonical JSON data") from error


def _resolve_workspace_file(
    workspace: Path,
    value: object,
    label: str,
    *,
    require_file: bool = True,
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise Phase4PilotContractError(f"{label} must be a non-empty path")
    raw = Path(value)
    path = raw if raw.is_absolute() else workspace / raw
    if path.is_symlink():
        raise Phase4PilotContractError(f"{label} must not be a symlink")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4PilotContractError(f"{label} must be inside the workspace") from error
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise Phase4PilotContractError(f"{label} does not name a regular file")
    return resolved, relative


def _checked_digest(value: object, label: str) -> str:
    try:
        return validate_sha256(value, label)
    except ValueError as error:
        raise Phase4PilotContractError(str(error)) from error


def _create_or_verify(path: Path, payload: bytes, label: str) -> Path:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise Phase4PilotContractError(
                f"existing immutable {label} does not match: {path}"
            )
        return path
    try:
        return atomic_create_bytes(path, payload)
    except ImmutableArtifactExistsError:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return path
        raise


def _self_hashed_payload(core: Mapping[str, object], field: str) -> dict[str, object]:
    normalized = normalize_json_mapping(core, "self-hashed payload")
    return {**normalized, field: sha256_bytes(canonical_json_bytes(normalized))}


def _verify_self_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    digest = _checked_digest(payload.get(field), f"{label}.{field}")
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4PilotContractError(f"{label} checksum mismatch")
    return digest


def _validate_frozen_workload(config: ExperimentConfig) -> None:
    validate_phase4_config(config)
    if (
        config.seed != PHASE4D_SEED
        or config.network.input_channels != 16
        or config.network.channels != 192
        or config.network.residual_blocks != 16
        or config.mcts.simulations != 256
        or config.self_play.games != PHASE4D_GAMES_PER_GENERATION
        or config.self_play.actors != PHASE4D_ACTORS
        or config.training.replay_capacity != PHASE4D_REPLAY_CAPACITY
        or config.training.batch_size != PHASE4D_BATCH_SIZE
        or config.training.training_steps != PHASE4D_TRAINING_STEPS
        or config.arena.games != PHASE4D_ARENA_GAMES
        or config.arena.simulations != PHASE4D_ARENA_SIMULATIONS
        or not config.training.use_amp
    ):
        raise Phase4PilotContractError(
            "config does not match the frozen Phase 4D workload"
        )


def _validate_frozen_base_config(config: ExperimentConfig) -> None:
    _validate_frozen_workload(config)
    if config.initialization.mode != "random":
        raise Phase4PilotContractError("base config must use random initialization")


def _load_canonical_base_config(
    path: Path,
) -> tuple[dict[str, object], ExperimentConfig]:
    if sha256_file(path) != PHASE4D_BASE_CONFIG_SHA256:
        raise Phase4PilotContractError(
            "base config does not match the canonical phase4-bootstrap.json bytes"
        )
    payload = _load_json_mapping(path, "base config")
    config = load_experiment_config(path)
    _validate_frozen_base_config(config)
    if payload != _config_payload(config):
        raise Phase4PilotContractError(
            "base config contains unknown, omitted, or noncanonical fields"
        )
    return payload, config


def _base_model_sha256(config: ExperimentConfig) -> str:
    seed_everything(config.seed)
    model = PolicyValueNetwork(_network_config(config))
    return canonical_model_state_sha256(model)


def _validate_pretraining_inputs(
    *,
    workspace: Path,
    summary_path: Path,
    base_config: ExperimentConfig,
) -> dict[str, object]:
    try:
        bundle = load_completed_pretraining_bundle(
            summary_path,
            workspace_root=workspace,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4PilotContractError(
            f"pretraining completion bundle does not verify: {error}"
        ) from error
    summary = dict(bundle.summary)
    if set(summary) != _PRETRAINING_SUMMARY_KEYS:
        raise Phase4PilotContractError(
            "pretraining summary fields do not match schema v1"
        )
    if (
        summary.get("schemaVersion") != PHASE4D_PRETRAINING_SUMMARY_SCHEMA_VERSION
        or summary.get("status") != "completed"
    ):
        raise Phase4PilotContractError("pretraining summary is not completed schema v1")
    digests = {
        key: _checked_digest(summary.get(key), f"pretraining summary {key}")
        for key in (
            "artifactSha256",
            "modelStateSha256",
            "baseModelStateSha256",
            "teacherDatasetSha256",
            "reportSha256",
        )
    }
    artifact_path, artifact_relative = _resolve_workspace_file(
        workspace, summary.get("artifactPath"), "pretraining artifact"
    )
    report_path, report_relative = _resolve_workspace_file(
        workspace, summary.get("reportPath"), "pretraining report"
    )
    if sha256_file(artifact_path) != digests["artifactSha256"]:
        raise Phase4PilotContractError("pretraining artifact SHA-256 mismatch")
    if sha256_file(report_path) != digests["reportSha256"]:
        raise Phase4PilotContractError("pretraining report SHA-256 mismatch")

    report = _load_json_mapping(report_path, "pretraining report")
    if (
        report.get("schemaVersion") != 1
        or report.get("kind") != "phase4d-teacher-pretraining"
        or report.get("status") != "completed"
        or report.get("warmArtifactPath") != artifact_relative
        or report.get("optimizerResetRequiredForSelfPlay") is not True
        or report.get("teacherReplayAllowedInSelfPlay") is not False
    ):
        raise Phase4PilotContractError(
            "pretraining report is not a completed self-play-safe v1 report"
        )
    _verify_self_hash(report, "reportCoreSha256", "pretraining report")
    for key in (
        "baseModelStateSha256",
        "modelStateSha256",
        "teacherDatasetSha256",
    ):
        if report.get(key) != digests[key]:
            raise Phase4PilotContractError(
                f"pretraining report {key} does not match its summary"
            )

    expected_base = base_config.initialization.expected_base_model_sha256
    if expected_base is None or digests["baseModelStateSha256"] != expected_base:
        raise Phase4PilotContractError(
            "pretraining base model does not match the cold initialization"
        )
    if _base_model_sha256(base_config) != expected_base:
        raise Phase4PilotContractError(
            "frozen seed/network no longer produces expectedBaseModelSha256"
        )

    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Phase4PilotContractError(
            "pretraining artifact cannot be deserialized with weights_only=True"
        ) from error
    if not isinstance(artifact, Mapping) or set(artifact) != _WARM_ARTIFACT_KEYS:
        raise Phase4PilotContractError("pretraining artifact fields do not match v1")
    expected_provenance = {
        "teacherDatasetSha256": digests["teacherDatasetSha256"],
        "pretrainingReportSha256": digests["reportSha256"],
    }
    if (
        artifact.get("schemaVersion") != PHASE4_WARM_ARTIFACT_SCHEMA_VERSION
        or artifact.get("format") != "quoridor-phase4-model-only"
        or artifact.get("network") != _network_config(base_config).to_dict()
        or artifact.get("baseModelStateSha256") != expected_base
        or artifact.get("modelStateSha256") != digests["modelStateSha256"]
        or artifact.get("provenance") != expected_provenance
    ):
        raise Phase4PilotContractError(
            "pretraining artifact schema, model, or provenance mismatch"
        )
    state = artifact.get("modelState")
    if not isinstance(state, Mapping):
        raise Phase4PilotContractError("pretraining artifact modelState is invalid")
    try:
        actual_model_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4PilotContractError(
            "pretraining artifact contains invalid model tensors"
        ) from error
    if actual_model_sha256 != digests["modelStateSha256"]:
        raise Phase4PilotContractError("pretraining tensor-state SHA-256 mismatch")

    return {
        "summaryPath": summary_path.relative_to(workspace).as_posix(),
        "summarySha256": sha256_file(summary_path),
        "artifactPath": artifact_relative,
        "artifactSha256": digests["artifactSha256"],
        "modelStateSha256": digests["modelStateSha256"],
        "baseModelStateSha256": digests["baseModelStateSha256"],
        "teacherDatasetSha256": digests["teacherDatasetSha256"],
        "reportPath": report_relative,
        "reportSha256": digests["reportSha256"],
    }


def _config_payload(config: ExperimentConfig) -> dict[str, object]:
    initialization: dict[str, object] = {
        "mode": config.initialization.mode,
        "expectedBaseModelSha256": config.initialization.expected_base_model_sha256,
        "resetOptimizer": config.initialization.reset_optimizer,
    }
    if config.initialization.mode == "model":
        initialization.update(
            {
                "artifactPath": config.initialization.artifact_path.as_posix(),
                "artifactSha256": config.initialization.artifact_sha256,
                "teacherDatasetSha256": config.initialization.teacher_dataset_sha256,
                "pretrainingReportSha256": config.initialization.pretraining_report_sha256,
            }
        )
    return {
        "pipeline": config.pipeline,
        "name": config.name,
        "seed": config.seed,
        "device": config.device,
        "runDir": config.run_dir.as_posix(),
        "artifactLimitBytes": config.artifact_limit_bytes,
        "initialization": initialization,
        "network": {
            "inputChannels": config.network.input_channels,
            "channels": config.network.channels,
            "residualBlocks": config.network.residual_blocks,
            "seReduction": config.network.se_reduction,
            "policyChannels": config.network.policy_channels,
            "valueChannels": config.network.value_channels,
            "valueHidden": config.network.value_hidden,
        },
        "mcts": {
            "simulations": config.mcts.simulations,
            "cBase": config.mcts.c_base,
            "cInit": config.mcts.c_init,
            "dirichletEpsilon": config.mcts.dirichlet_epsilon,
            "dirichletAlpha": config.mcts.dirichlet_alpha,
        },
        "selfPlay": {
            "games": config.self_play.games,
            "temperatureMoves": config.self_play.temperature_moves,
            "actors": config.self_play.actors,
        },
        "training": {
            "replayCapacity": config.training.replay_capacity,
            "batchSize": config.training.batch_size,
            "learningRate": config.training.learning_rate,
            "minimumLearningRate": config.training.minimum_learning_rate,
            "scheduleSteps": config.training.schedule_steps,
            "weightDecay": config.training.weight_decay,
            "maxGradNorm": config.training.max_grad_norm,
            "useAmp": config.training.use_amp,
            "trainingSteps": config.training.training_steps,
            "maxCheckpoints": config.training.max_checkpoints,
        },
        "arena": {
            "games": config.arena.games,
            "simulations": config.arena.simulations,
        },
    }


def build_effective_config_payloads(
    base_payload: Mapping[str, object],
    pretraining: Mapping[str, object],
    *,
    pilot_root_relative: str,
) -> dict[str, dict[str, object]]:
    """Return the two frozen configs without writing them."""

    try:
        cold = json.loads(json.dumps(base_payload))
    except (TypeError, ValueError) as error:
        raise Phase4PilotContractError("base config is not JSON-compatible") from error
    if not isinstance(cold, dict):
        raise Phase4PilotContractError("base config must be a JSON object")
    expected_base = pretraining.get("baseModelStateSha256")
    _checked_digest(expected_base, "pretraining baseModelStateSha256")
    cold["pipeline"] = "phase4"
    cold["name"] = "phase4d-cold-pilot"
    cold["seed"] = PHASE4D_SEED
    cold["device"] = "cuda"
    cold["runDir"] = f"{pilot_root_relative}/lanes/cold"
    cold["artifactLimitBytes"] = DEFAULT_ARTIFACT_LIMIT_BYTES
    cold["initialization"] = {
        "mode": "random",
        "expectedBaseModelSha256": expected_base,
        "resetOptimizer": True,
    }

    warm = json.loads(json.dumps(cold))
    warm["name"] = "phase4d-warm-pilot"
    warm["runDir"] = f"{pilot_root_relative}/lanes/warm"
    warm["initialization"] = {
        "mode": "model",
        "expectedBaseModelSha256": expected_base,
        "artifactPath": pretraining.get("artifactPath"),
        "artifactSha256": pretraining.get("artifactSha256"),
        "teacherDatasetSha256": pretraining.get("teacherDatasetSha256"),
        "pretrainingReportSha256": pretraining.get("reportSha256"),
        "resetOptimizer": True,
    }
    return {"cold": cold, "warm": warm}


def _assert_write_capacity(
    control_root: Path,
    *,
    reserve_bytes: int,
) -> dict[str, int]:
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    free_bytes = int(shutil.disk_usage(control_root).free)
    artifact_bytes = directory_size(control_root)
    if free_bytes - reserve_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4ResourceLimitReached(
            "write reserve would cross the 20 GiB free-disk stop line"
        )
    if artifact_bytes + reserve_bytes > DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4ResourceLimitReached(
            "write reserve would exceed the 8 GiB artifact cap"
        )
    return {
        "freeDiskBytes": free_bytes,
        "artifactBytes": artifact_bytes,
        "reservedWriteBytes": reserve_bytes,
    }


def _assert_shared_preflight(
    control_root: Path,
    *,
    reserve_bytes: int = 0,
) -> dict[str, object]:
    storage = _assert_write_capacity(
        control_root,
        reserve_bytes=reserve_bytes,
    )
    baseline = verify_phase4d_baseline_report(control_root)
    ledger = GpuBudgetLedger(control_root / GPU_LEDGER_NAME)
    stage_ledger = Phase4StageBudgetLedger(control_root / PHASE4_STAGE_LEDGER_NAME)
    stage_ledger.ensure_stage(
        PHASE4D_STAGE,
        PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
        PHASE4D_GPU_HOUR_LIMIT,
    )
    stage = stage_ledger.assert_available(PHASE4D_STAGE, ledger.total_hours)
    if ledger.total_hours >= GPU_HOUR_BUDGET:
        raise Phase4ResourceLimitReached("72 GPU-hour global budget is exhausted")
    return {
        "baselineCalibration": baseline,
        "globalGpuHours": ledger.total_hours,
        "stageGpu": stage,
        **storage,
    }


def _current_runtime_provenance(workspace: Path) -> dict[str, str]:
    try:
        source_sha256 = fingerprint_sources(workspace).sha256
        native_backend_sha256 = loaded_native_backend_hash()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4PilotContractError(
            "cannot establish current training source/native provenance"
        ) from error
    return {
        "sourceSha256": source_sha256,
        "nativeBackendSha256": native_backend_sha256,
    }


def _assert_current_runtime(
    workspace: Path,
    contract: Mapping[str, object],
) -> dict[str, str]:
    expected = contract.get("runtimeProvenance")
    if not isinstance(expected, Mapping):
        raise Phase4PilotContractError("pilot runtime provenance is missing")
    current = _current_runtime_provenance(workspace)
    if dict(expected) != current:
        raise Phase4PilotContractError(
            "current training source/native backend differs from the pilot contract"
        )
    return current


def prepare_pilot(
    *,
    workspace_root: str | Path,
    pretraining_summary_path: str | Path,
    base_config_path: str | Path,
    output_root: str | Path,
    verify_resources: bool = True,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    output = Path(output_root)
    output = output if output.is_absolute() else workspace / output
    output = output.resolve()
    try:
        output_relative = output.relative_to(control_root).as_posix()
    except ValueError as error:
        raise Phase4PilotContractError(
            "pilot output root must be inside training/runs/phase4"
        ) from error
    pilot_relative = f"{PHASE4_CONTROL_RELATIVE}/{output_relative}"

    summary_path, _ = _resolve_workspace_file(
        workspace, str(pretraining_summary_path), "pretraining summary"
    )
    base_path, base_relative = _resolve_workspace_file(
        workspace, str(base_config_path), "base config"
    )
    base_payload, base_config = _load_canonical_base_config(base_path)
    pretraining = _validate_pretraining_inputs(
        workspace=workspace,
        summary_path=summary_path,
        base_config=base_config,
    )

    dev_path = workspace / PHASE4D_DEV_OPENINGS
    dev = load_opening_suite(dev_path, expected_suite="dev")
    if len(dev.openings) < PHASE4D_ARENA_OPENINGS:
        raise Phase4PilotContractError("dev opening suite contains fewer than 32 openings")
    screen_suite = build_opening_suite(
        dev.openings[:PHASE4D_ARENA_OPENINGS], suite="dev"
    )
    runtime_provenance = _current_runtime_provenance(workspace)

    payloads = build_effective_config_payloads(
        base_payload,
        pretraining,
        pilot_root_relative=pilot_relative,
    )
    config_records: dict[str, dict[str, object]] = {}
    config_writes: list[tuple[Path, bytes, str]] = []
    effective_configs: dict[str, ExperimentConfig] = {}
    for lane in ("cold", "warm"):
        config_path = output / PHASE4D_CONFIG_RELATIVE / f"{lane}.json"
        config_bytes = canonical_json_bytes(payloads[lane])
        config = ExperimentConfig.from_dict(payloads[lane])
        _validate_frozen_workload(config)
        if config.initialization.mode != ("random" if lane == "cold" else "model"):
            raise Phase4PilotContractError(f"{lane} initialization mode is invalid")
        if _config_payload(config) != payloads[lane]:
            raise Phase4PilotContractError(
                f"{lane} effective config is not canonical"
            )
        effective_configs[lane] = config
        config_writes.append(
            (config_path, config_bytes, f"{lane} effective config")
        )
        config_records[lane] = {
            "path": config_path.relative_to(workspace).as_posix(),
            "fileSha256": sha256_bytes(config_bytes),
            "configSha256": fingerprint_config(config).sha256,
            "runDir": config.run_dir.as_posix(),
        }

    if (
        effective_configs["cold"].initialization.expected_base_model_sha256
        != effective_configs["warm"].initialization.expected_base_model_sha256
    ):
        raise Phase4PilotContractError("cold and warm base model hashes differ")

    schedule = [
        {"ordinal": 2 * generation, "lane": "cold", "generation": generation}
        for generation in range(PHASE4D_PILOT_GENERATIONS)
    ] + [
        {"ordinal": 2 * generation + 1, "lane": "warm", "generation": generation}
        for generation in range(PHASE4D_PILOT_GENERATIONS)
    ]
    schedule.sort(key=lambda item: int(item["ordinal"]))
    core = {
        "schemaVersion": PHASE4D_PILOT_SCHEMA_VERSION,
        "kind": "phase4d-warm-cold-pilot",
        "createdAt": utc_timestamp(),
        "localOnly": True,
        "formalChampionDeclaration": False,
        "runtimeProvenance": runtime_provenance,
        "baseConfig": {
            "path": base_relative,
            "fileSha256": PHASE4D_BASE_CONFIG_SHA256,
        },
        "pretraining": pretraining,
        "configs": config_records,
        "workload": {
            "seed": PHASE4D_SEED,
            "network": "192x16",
            "generationsPerLane": PHASE4D_PILOT_GENERATIONS,
            "gamesPerGeneration": PHASE4D_GAMES_PER_GENERATION,
            "actors": PHASE4D_ACTORS,
            "selfPlaySimulations": 256,
            "batchSize": PHASE4D_BATCH_SIZE,
            "trainingStepsPerGeneration": PHASE4D_TRAINING_STEPS,
            "schedule": schedule,
        },
        "arena": {
            "sourceSuitePath": PHASE4D_DEV_OPENINGS.as_posix(),
            "sourceSuiteFileSha256": sha256_file(dev_path),
            "sourceSuiteHash": dev.content_hash,
            "selection": "first-32-in-frozen-suite-order",
            "openingIds": [opening.id for opening in screen_suite.openings],
            "screenSuiteHash": screen_suite.content_hash,
            "games": PHASE4D_ARENA_GAMES,
            "simulationsPerSide": PHASE4D_ARENA_SIMULATIONS,
            "bootstrapSeed": GATE_BOOTSTRAP_SEED,
            "bootstrapSamples": GATE_BOOTSTRAP_SAMPLES,
            "warmThresholds": {
                "rawWinRateAtLeast": PHASE4D_RAW_WIN_RATE_THRESHOLD,
                "oneSided95LowerGreaterThan": PHASE4D_ONE_SIDED_LOWER_THRESHOLD,
            },
        },
        "resourceLimits": {
            "globalGpuHours": LEGACY_GPU_HOUR_BUDGET,
            "phase4dGpuHours": PHASE4D_GPU_HOUR_LIMIT,
            "artifactBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
            "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        },
    }
    contract_path = output / PHASE4D_CONTRACT_NAME
    run_id = f"phase4d-pilot-prepare-{os.getpid()}-{time.time_ns()}"
    with RunLock(control_root / GPU_LOCK_NAME, run_id=run_id):
        if _current_runtime_provenance(workspace) != runtime_provenance:
            raise Phase4PilotContractError(
                "training source/native backend changed during pilot preparation"
            )
        if contract_path.exists():
            contract = load_pilot_contract(contract_path, workspace_root=workspace)
            comparable = dict(contract)
            comparable.pop("contractSha256", None)
            comparable.pop("createdAt", None)
            expected = dict(core)
            expected.pop("createdAt", None)
            if comparable != expected:
                raise Phase4PilotContractError(
                    "existing pilot contract does not match requested preparation"
                )
        else:
            contract = _self_hashed_payload(core, "contractSha256")
        contract_bytes = canonical_json_bytes(contract)
        writes = [
            *config_writes,
            (contract_path, contract_bytes, "pilot contract"),
        ]
        for path, payload, label in writes:
            if path.exists() and (
                not path.is_file()
                or path.is_symlink()
                or path.read_bytes() != payload
            ):
                raise Phase4PilotContractError(
                    f"existing immutable {label} does not match: {path}"
                )
        missing_writes = [
            (path, payload)
            for path, payload, _label in writes
            if not path.exists()
        ]
        reserve_bytes = sum(
            len(payload) + _METADATA_WRITE_OVERHEAD_BYTES
            for _path, payload in missing_writes
        )
        if missing_writes or not (
            control_root / PHASE4_STAGE_LEDGER_NAME
        ).exists():
            reserve_bytes += _METADATA_WRITE_OVERHEAD_BYTES
        if verify_resources:
            resources = _assert_shared_preflight(
                control_root,
                reserve_bytes=reserve_bytes,
            )
        else:
            resources = {
                "verified": False,
                "reservedWriteBytes": reserve_bytes,
            }
        for path, payload, label in writes:
            _create_or_verify(path, payload, label)
    return {
        "status": "prepared",
        "contractPath": contract_path.relative_to(workspace).as_posix(),
        "contractSha256": contract["contractSha256"],
        "configs": config_records,
        "resources": resources,
    }


def load_pilot_contract(
    path: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    source, _ = _resolve_workspace_file(workspace, str(path), "pilot contract")
    payload = _load_json_mapping(source, "pilot contract")
    if (
        payload.get("schemaVersion") != PHASE4D_PILOT_SCHEMA_VERSION
        or payload.get("kind") != "phase4d-warm-cold-pilot"
        or payload.get("localOnly") is not True
        or payload.get("formalChampionDeclaration") is not False
    ):
        raise Phase4PilotContractError("unsupported pilot contract")
    _verify_self_hash(payload, "contractSha256", "pilot contract")
    runtime = payload.get("runtimeProvenance")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "sourceSha256",
        "nativeBackendSha256",
    }:
        raise Phase4PilotContractError("pilot runtime provenance is invalid")
    _checked_digest(runtime.get("sourceSha256"), "pilot source sha256")
    _checked_digest(
        runtime.get("nativeBackendSha256"), "pilot native backend sha256"
    )
    base_record = payload.get("baseConfig")
    if (
        not isinstance(base_record, Mapping)
        or set(base_record) != {"path", "fileSha256"}
        or base_record.get("fileSha256") != PHASE4D_BASE_CONFIG_SHA256
    ):
        raise Phase4PilotContractError("pilot base config provenance is invalid")
    base_path, _ = _resolve_workspace_file(
        workspace, base_record.get("path"), "pilot base config"
    )
    _load_canonical_base_config(base_path)
    workload = payload.get("workload")
    expected_schedule = [
        {
            "ordinal": ordinal,
            "lane": "cold" if ordinal % 2 == 0 else "warm",
            "generation": ordinal // 2,
        }
        for ordinal in range(2 * PHASE4D_PILOT_GENERATIONS)
    ]
    if (
        not isinstance(workload, Mapping)
        or workload.get("seed") != PHASE4D_SEED
        or workload.get("network") != "192x16"
        or workload.get("generationsPerLane") != PHASE4D_PILOT_GENERATIONS
        or workload.get("gamesPerGeneration") != PHASE4D_GAMES_PER_GENERATION
        or workload.get("actors") != PHASE4D_ACTORS
        or workload.get("selfPlaySimulations") != 256
        or workload.get("batchSize") != PHASE4D_BATCH_SIZE
        or workload.get("trainingStepsPerGeneration")
        != PHASE4D_TRAINING_STEPS
        or workload.get("schedule") != expected_schedule
    ):
        raise Phase4PilotContractError("pilot workload contract changed")
    arena = payload.get("arena")
    thresholds = arena.get("warmThresholds") if isinstance(arena, Mapping) else None
    opening_ids = arena.get("openingIds") if isinstance(arena, Mapping) else None
    if (
        not isinstance(arena, Mapping)
        or arena.get("sourceSuitePath") != PHASE4D_DEV_OPENINGS.as_posix()
        or arena.get("selection") != "first-32-in-frozen-suite-order"
        or not isinstance(opening_ids, list)
        or len(opening_ids) != PHASE4D_ARENA_OPENINGS
        or any(not isinstance(opening_id, str) for opening_id in opening_ids)
        or len(set(opening_ids)) != PHASE4D_ARENA_OPENINGS
        or arena.get("games") != PHASE4D_ARENA_GAMES
        or arena.get("simulationsPerSide") != PHASE4D_ARENA_SIMULATIONS
        or arena.get("bootstrapSeed") != GATE_BOOTSTRAP_SEED
        or arena.get("bootstrapSamples") != GATE_BOOTSTRAP_SAMPLES
        or not isinstance(thresholds, Mapping)
        or thresholds.get("rawWinRateAtLeast")
        != PHASE4D_RAW_WIN_RATE_THRESHOLD
        or thresholds.get("oneSided95LowerGreaterThan")
        != PHASE4D_ONE_SIDED_LOWER_THRESHOLD
    ):
        raise Phase4PilotContractError("pilot arena contract changed")
    limits = payload.get("resourceLimits")
    if not isinstance(limits, Mapping) or dict(limits) != {
        "globalGpuHours": LEGACY_GPU_HOUR_BUDGET,
        "phase4dGpuHours": PHASE4D_GPU_HOUR_LIMIT,
        "artifactBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }:
        raise Phase4PilotContractError("pilot resource limits changed")
    configs = payload.get("configs")
    if not isinstance(configs, Mapping) or set(configs) != {"cold", "warm"}:
        raise Phase4PilotContractError("pilot contract configs are invalid")
    for lane in ("cold", "warm"):
        record = configs[lane]
        if not isinstance(record, Mapping):
            raise Phase4PilotContractError(f"{lane} config record is invalid")
        config_path, _ = _resolve_workspace_file(
            workspace, record.get("path"), f"{lane} config"
        )
        if sha256_file(config_path) != record.get("fileSha256"):
            raise Phase4PilotContractError(f"{lane} config file SHA-256 mismatch")
        config = load_experiment_config(config_path)
        _validate_frozen_workload(config)
        if config.initialization.mode != ("random" if lane == "cold" else "model"):
            raise Phase4PilotContractError(f"{lane} initialization mode changed")
        if (
            config.name != f"phase4d-{lane}-pilot"
            or config.device != "cuda"
            or config.run_dir.as_posix() != record.get("runDir")
        ):
            raise Phase4PilotContractError(f"{lane} lane identity changed")
        if fingerprint_config(config).sha256 != record.get("configSha256"):
            raise Phase4PilotContractError(f"{lane} config fingerprint mismatch")
        if config.seed != PHASE4D_SEED:
            raise Phase4PilotContractError(f"{lane} seed changed")
    cold_config = load_experiment_config(workspace / configs["cold"]["path"])
    warm_config = load_experiment_config(workspace / configs["warm"]["path"])
    if (
        cold_config.initialization.expected_base_model_sha256
        != warm_config.initialization.expected_base_model_sha256
    ):
        raise Phase4PilotContractError("lane base-model identities differ")
    return payload


def _lane_config(
    contract: Mapping[str, object], lane: str, workspace: Path
) -> ExperimentConfig:
    if lane not in {"cold", "warm"}:
        raise ValueError("lane must be cold or warm")
    configs = contract["configs"]
    assert isinstance(configs, Mapping)
    record = configs[lane]
    assert isinstance(record, Mapping)
    config = load_experiment_config(workspace / str(record["path"]))
    if not config.run_dir.is_absolute():
        config = ExperimentConfig.from_dict(
            {**_config_payload(config), "runDir": str(workspace / config.run_dir)}
        )
    return config


def load_lane_state(
    *,
    workspace_root: str | Path,
    contract: Mapping[str, object],
    lane: str,
) -> LaneState:
    workspace = Path(workspace_root).resolve()
    config = _lane_config(contract, lane, workspace)
    run_dir = config.run_dir.resolve()
    manifest_path = run_dir / "run-manifest.json"
    protected = (run_dir / "checkpoints", run_dir / "replay", run_dir / "reports")
    if not manifest_path.exists():
        if any(path.exists() for path in protected):
            raise Phase4PilotContractError(
                f"{lane} has run data without a manifest"
            )
        return LaneState(lane, config, None, None, (), None)

    manifest = load_run_manifest(manifest_path)
    provenance = manifest.payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Phase4PilotContractError(f"{lane} manifest provenance is invalid")
    config_provenance = provenance.get("config")
    source_provenance = provenance.get("source")
    if not isinstance(config_provenance, Mapping) or not isinstance(
        source_provenance, Mapping
    ):
        raise Phase4PilotContractError(f"{lane} manifest provenance is incomplete")
    if config_provenance.get("sha256") != fingerprint_config(config).sha256:
        raise Phase4PilotContractError(f"{lane} manifest config identity mismatch")
    source_sha256 = _checked_digest(
        source_provenance.get("sha256"), f"{lane} source sha256"
    )
    resource_budget = manifest.payload.get("resourceBudget")
    budget_metadata = (
        resource_budget.get("metadata")
        if isinstance(resource_budget, Mapping)
        else None
    )
    if not isinstance(budget_metadata, Mapping):
        raise Phase4PilotContractError(f"{lane} resource provenance is invalid")
    native_backend_sha256 = _checked_digest(
        budget_metadata.get("nativeBackendSha256"),
        f"{lane} training native backend sha256",
    )
    reports = GenerationReportStore(
        run_dir / "reports", run_manifest_sha256=manifest.sha256
    )
    generations = reports.generations()
    if generations != tuple(range(len(generations))):
        raise Phase4PilotContractError(f"{lane} generation reports contain a gap")
    if len(generations) > PHASE4D_PILOT_GENERATIONS:
        raise Phase4PilotContractError(f"{lane} exceeds the five-generation pilot")
    records = tuple(reports.load(generation) for generation in generations)
    return LaneState(
        lane,
        config,
        manifest.sha256,
        source_sha256,
        records,
        native_backend_sha256,
    )


def _assert_lane_runtime_binding(
    state: LaneState,
    contract: Mapping[str, object],
) -> None:
    if state.manifest_sha256 is None:
        return
    runtime = contract.get("runtimeProvenance")
    if not isinstance(runtime, Mapping) or (
        state.source_sha256 != runtime.get("sourceSha256")
        or state.native_backend_sha256 != runtime.get("nativeBackendSha256")
    ):
        raise Phase4PilotContractError(
            f"{state.lane} source/native provenance differs from the pilot contract"
        )


def _schedule_prefix(
    cold: LaneState, warm: LaneState
) -> tuple[tuple[str, int, str], ...]:
    if cold.next_generation not in {
        warm.next_generation,
        warm.next_generation + 1,
    }:
        raise Phase4PilotContractError(
            "lane progress violates cold-gN then warm-gN alternation"
        )
    actual: list[tuple[str, int, str]] = []
    for generation in range(max(cold.next_generation, warm.next_generation)):
        if generation < cold.next_generation:
            created = cold.reports[generation].payload.get("createdAt")
            if not isinstance(created, str) or not created:
                raise Phase4PilotContractError("cold report timestamp is invalid")
            actual.append(("cold", generation, created))
        if generation < warm.next_generation:
            created = warm.reports[generation].payload.get("createdAt")
            if not isinstance(created, str) or not created:
                raise Phase4PilotContractError("warm report timestamp is invalid")
            actual.append(("warm", generation, created))
    for earlier, later in zip(actual, actual[1:]):
        if earlier[2] >= later[2]:
            raise Phase4PilotContractError(
                "generation report timestamps do not prove the frozen alternation"
            )
    return tuple(actual)


def _lane_has_unreported_work(state: LaneState) -> bool:
    """Detect durable work that the target-mode driver must reconcile first."""

    if state.manifest_sha256 is None:
        return False
    replay_manifest = state.config.run_dir / "replay" / "manifest.json"
    replay_ahead = False
    if replay_manifest.exists():
        replay_payload = _load_json_mapping(
            replay_manifest, f"{state.lane} replay manifest"
        )
        totals = replay_payload.get("totals")
        if isinstance(totals, Mapping):
            games = totals.get("games")
            if type(games) is int and games > (
                state.next_generation * PHASE4D_GAMES_PER_GENERATION
            ):
                replay_ahead = True
    for metadata_path in (state.config.run_dir / "checkpoints").glob(
        "checkpoint-g*-s*.json"
    ):
        try:
            metadata = _load_json_mapping(metadata_path, "checkpoint metadata")
        except Phase4PilotContractError:
            if replay_ahead:
                return True
            raise
        generation = metadata.get("generation")
        if type(generation) is int and generation >= state.next_generation:
            return True
    return replay_ahead


def next_step(
    *, workspace_root: str | Path, contract_path: str | Path
) -> tuple[str, int]:
    workspace = Path(workspace_root).resolve()
    contract = load_pilot_contract(contract_path, workspace_root=workspace)
    _assert_current_runtime(workspace, contract)
    cold = load_lane_state(workspace_root=workspace, contract=contract, lane="cold")
    warm = load_lane_state(workspace_root=workspace, contract=contract, lane="warm")
    _assert_lane_runtime_binding(cold, contract)
    _assert_lane_runtime_binding(warm, contract)
    if (
        cold.source_sha256 is not None
        and warm.source_sha256 is not None
        and (
            cold.source_sha256 != warm.source_sha256
            or cold.native_backend_sha256 != warm.native_backend_sha256
        )
    ):
        raise Phase4PilotContractError(
            "started lanes do not share source/native-backend provenance"
        )
    pending = {
        state.lane: _lane_has_unreported_work(state) for state in (cold, warm)
    }
    # A process may stop after the trainer commits a report but before the shell
    # runs its post-step validation.  Attest and deep-verify every committed
    # prefix here before allowing the other lane (or next round) to advance.
    # Partial replay or a checkpoint without its report goes through the
    # idempotent target-mode driver first and is validated immediately after.
    for state in (cold, warm):
        if state.reports and not pending[state.lane]:
            validate_lane(
                workspace_root=workspace,
                contract_path=contract_path,
                lane=state.lane,
            )
    _schedule_prefix(cold, warm)
    if cold.complete and warm.complete:
        if any(pending.values()):
            raise Phase4PilotContractError(
                "completed pilot contains unreported work beyond generation 4"
            )
        return "complete", PHASE4D_PILOT_GENERATIONS
    if cold.next_generation == warm.next_generation:
        lane, target = "cold", cold.next_generation + 1
    else:
        lane, target = "warm", warm.next_generation + 1
    pending_lanes = {name for name, has_work in pending.items() if has_work}
    if pending_lanes and pending_lanes != {lane}:
        raise Phase4PilotContractError(
            "unreported work exists outside the next frozen schedule lane"
        )
    return lane, target


def next_lane(
    *, workspace_root: str | Path, contract_path: str | Path
) -> str:
    return next_step(
        workspace_root=workspace_root, contract_path=contract_path
    )[0]


def _checkpoint_payload_from_report(
    state: LaneState,
    record: GenerationReportRecord,
) -> tuple[Path, Mapping[str, object], str]:
    report = record.payload.get("report")
    if not isinstance(report, Mapping):
        raise Phase4PilotContractError(f"{state.lane} generation report body is invalid")
    checkpoint = report.get("checkpoint")
    generation_data = report.get("generationData")
    if not isinstance(checkpoint, Mapping):
        raise Phase4PilotContractError(f"{state.lane} checkpoint record is invalid")
    file_name = checkpoint.get("path")
    digest = _checked_digest(
        checkpoint.get("sha256"), f"{state.lane} checkpoint sha256"
    )
    if (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not file_name.endswith(".pt")
    ):
        raise Phase4PilotContractError(f"{state.lane} checkpoint path is unsafe")
    path = state.config.run_dir / "checkpoints" / file_name
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != digest
        or checkpoint.get("bytes") != path.stat().st_size
    ):
        raise Phase4PilotContractError(f"{state.lane} checkpoint file does not verify")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise Phase4PilotContractError(
            f"{state.lane} checkpoint cannot be deserialized"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase4PilotContractError(f"{state.lane} checkpoint payload is invalid")
    expected_identity = {
        "runManifestSha256": state.manifest_sha256,
        "configSha256": fingerprint_config(state.config).sha256,
        "sourceSha256": state.source_sha256,
    }
    if (
        payload.get("schemaVersion") != 1
        or payload.get("generation") != record.generation
        or payload.get("step") != checkpoint.get("step")
        or payload.get("identity") != expected_identity
        or not isinstance(generation_data, Mapping)
        or payload.get("generationData") != generation_data
    ):
        raise Phase4PilotContractError(
            f"{state.lane} checkpoint lineage/report binding is invalid"
        )
    return path, payload, digest


def _initialization_attestation_path(state: LaneState) -> Path:
    return state.config.run_dir / "initialization-validation.json"


def _load_initialization_attestation(
    state: LaneState, contract_sha256: str
) -> dict[str, object] | None:
    path = _initialization_attestation_path(state)
    if not path.exists():
        return None
    payload = _load_json_mapping(path, f"{state.lane} initialization attestation")
    _verify_self_hash(payload, "attestationSha256", "initialization attestation")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("lane") != state.lane
        or payload.get("contractSha256") != contract_sha256
        or payload.get("manifestSha256") != state.manifest_sha256
        or payload.get("optimizerStateEmpty") is not True
        or payload.get("replayEmpty") is not True
    ):
        raise Phase4PilotContractError(
            f"{state.lane} initialization attestation does not match"
        )
    return payload


def _attest_initialization(
    state: LaneState,
    contract_sha256: str,
    *,
    control_root: Path,
) -> dict[str, object] | None:
    existing = _load_initialization_attestation(state, contract_sha256)
    if existing is not None:
        return existing
    if not state.reports:
        return None
    metadata_candidates = []
    for path in (state.config.run_dir / "checkpoints").glob("checkpoint-g*-s*.json"):
        metadata = _load_json_mapping(path, "checkpoint metadata")
        if metadata.get("generation") == -1:
            metadata_candidates.append((path, metadata))
    if len(metadata_candidates) != 1:
        raise Phase4PilotContractError(
            f"{state.lane} initialization checkpoint is unavailable before attestation"
        )
    metadata_path, metadata = metadata_candidates[0]
    data_name = metadata.get("file")
    digest = _checked_digest(metadata.get("sha256"), "initialization checkpoint sha256")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise Phase4PilotContractError("initialization checkpoint path is unsafe")
    data_path = metadata_path.parent / data_name
    expected_identity = {
        "runManifestSha256": state.manifest_sha256,
        "configSha256": fingerprint_config(state.config).sha256,
        "sourceSha256": state.source_sha256,
    }
    if (
        metadata.get("schemaVersion") != 1
        or metadata.get("generation") != -1
        or metadata.get("step") != 0
        or metadata.get("identity") != expected_identity
        or not data_path.is_file()
        or data_path.is_symlink()
        or metadata.get("bytes") != data_path.stat().st_size
        or sha256_file(data_path) != digest
    ):
        raise Phase4PilotContractError("initialization checkpoint does not verify")
    try:
        checkpoint = torch.load(data_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise Phase4PilotContractError(
            "initialization checkpoint cannot be deserialized"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise Phase4PilotContractError("initialization checkpoint is invalid")
    optimizer = checkpoint.get("optimizer")
    generation_data = checkpoint.get("generationData")
    replay = checkpoint.get("replaySnapshot")
    if (
        checkpoint.get("generation") != -1
        or checkpoint.get("step") != 0
        or checkpoint.get("schemaVersion") != 1
        or checkpoint.get("identity") != expected_identity
        or not isinstance(optimizer, Mapping)
        or optimizer.get("state") != {}
        or not isinstance(generation_data, Mapping)
        or generation_data.get("kind") != "initialization"
        or generation_data.get("mode")
        != ("random" if state.lane == "cold" else "model")
        or generation_data.get("optimizerReset") is not True
        or not isinstance(replay, Mapping)
        or replay.get("games") != 0
        or replay.get("plies") != 0
        or replay.get("gameIds") != []
    ):
        raise Phase4PilotContractError(
            f"{state.lane} initialization did not start with empty optimizer/replay"
        )
    model_state = checkpoint.get("model")
    trainer_runtime = checkpoint.get("trainerRuntime")
    if (
        not isinstance(model_state, Mapping)
        or not isinstance(trainer_runtime, Mapping)
        or trainer_runtime.get("step") != 0
        or canonical_model_state_sha256(model_state)
        != generation_data.get("initializedModelSha256")
    ):
        raise Phase4PilotContractError(
            f"{state.lane} initialization model/runtime is invalid"
        )
    expected_base = state.config.initialization.expected_base_model_sha256
    if generation_data.get("baseModelSha256") != expected_base:
        raise Phase4PilotContractError(f"{state.lane} initialization base hash mismatch")
    if state.lane == "cold":
        if (
            generation_data.get("initializedModelSha256") != expected_base
            or generation_data.get("artifact") is not None
        ):
            raise Phase4PilotContractError("cold initialization imported warm state")
    else:
        artifact = generation_data.get("artifact")
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "artifactPath",
            "artifactSha256",
            "modelStateSha256",
            "teacherDatasetSha256",
            "pretrainingReportSha256",
        }:
            raise Phase4PilotContractError("warm initialization provenance is invalid")
        initialization = state.config.initialization
        expected = {
            "artifactPath": initialization.artifact_path.as_posix(),
            "artifactSha256": initialization.artifact_sha256,
            "modelStateSha256": generation_data.get("initializedModelSha256"),
            "teacherDatasetSha256": initialization.teacher_dataset_sha256,
            "pretrainingReportSha256": initialization.pretraining_report_sha256,
        }
        if dict(artifact) != expected:
            raise Phase4PilotContractError("warm initialization provenance mismatch")

    core = {
        "schemaVersion": 1,
        "createdAt": utc_timestamp(),
        "lane": state.lane,
        "contractSha256": contract_sha256,
        "manifestSha256": state.manifest_sha256,
        "checkpointFile": data_name,
        "checkpointSha256": digest,
        "mode": generation_data.get("mode"),
        "baseModelSha256": expected_base,
        "initializedModelSha256": generation_data.get("initializedModelSha256"),
        "optimizerStateEmpty": True,
        "replayEmpty": True,
    }
    payload = _self_hashed_payload(core, "attestationSha256")
    encoded = canonical_json_bytes(payload)
    _assert_write_capacity(
        control_root,
        reserve_bytes=len(encoded) + _METADATA_WRITE_OVERHEAD_BYTES,
    )
    _create_or_verify(
        _initialization_attestation_path(state),
        encoded,
        f"{state.lane} initialization attestation",
    )
    return payload


def _contains_teacher(value: object) -> bool:
    if isinstance(value, str):
        return "teacher" in value.lower()
    if isinstance(value, Mapping):
        return any(
            _contains_teacher(key) or _contains_teacher(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_teacher(item) for item in value)
    return False


def _validate_replay(
    state: LaneState,
    *,
    expected_games_per_generation: int = PHASE4D_GAMES_PER_GENERATION,
) -> dict[str, object]:
    completed = state.next_generation
    if completed == 0:
        return {"games": 0, "plies": 0, "teacherReferences": 0}
    replay = ReplayStore(
        state.config.run_dir / "replay",
        max_games=state.config.training.replay_capacity,
        max_plies=state.config.training.replay_capacity,
        max_bytes=state.config.artifact_limit_bytes * 3 // 4,
    )
    verification = replay.verify(deep=True)
    expected_ids = {
        f"phase4-g{generation:08d}-game{index:06d}"
        for generation in range(completed)
        for index in range(expected_games_per_generation)
    }
    if set(replay.game_ids) != expected_ids:
        raise Phase4PilotContractError(
            f"{state.lane} replay is not exactly the pilot self-play set"
        )

    producer_ids: dict[int, str] = {}
    initialization_path = _initialization_attestation_path(state)
    initialization = _load_json_mapping(
        initialization_path, f"{state.lane} initialization attestation"
    )
    _verify_self_hash(initialization, "attestationSha256", "initialization attestation")
    producer_ids[0] = f"sha256:{initialization['checkpointSha256']}"
    for generation in range(1, completed):
        previous = state.reports[generation - 1].payload["report"]
        assert isinstance(previous, Mapping)
        checkpoint = previous.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise Phase4PilotContractError("previous checkpoint report is invalid")
        producer_ids[generation] = f"sha256:{checkpoint.get('sha256')}"

    games_by_generation = {generation: 0 for generation in range(completed)}
    invocations_by_generation: dict[int, dict[str, dict[str, object]]] = {
        generation: {} for generation in range(completed)
    }
    for game in replay.iter_games():
        metadata = game.metadata
        generation = metadata.get("generation")
        if type(generation) is not int or generation not in games_by_generation:
            raise Phase4PilotContractError(f"{state.lane} replay generation is invalid")
        expected_model_id = producer_ids[generation]
        game_index = metadata.get("gameIndex")
        expected_game_id = (
            f"phase4-g{generation:08d}-game{game_index:06d}"
            if type(game_index) is int
            else ""
        )
        generation_seed = (PHASE4D_SEED + generation * 1_000_003) % (2**63)
        if (
            game.game_id != expected_game_id
            or type(game_index) is not int
            or not 0 <= game_index < expected_games_per_generation
            or metadata.get("pipeline") != "phase4"
            or metadata.get("modelId") != expected_model_id
            or metadata.get("schedulerSeed") != generation_seed
            or _contains_teacher(metadata)
        ):
            raise Phase4PilotContractError(
                f"{state.lane} replay contains foreign or teacher game data"
            )
        try:
            invocation = validate_selfplay_invocation(
                metadata.get("invocation"),
                generation=generation,
                model_id=expected_model_id,
                game_index=game_index,
                configured_actors=state.config.self_play.actors,
                simulations=state.config.mcts.simulations,
                temperature_moves=state.config.self_play.temperature_moves,
            )
        except Phase4ResumeError as error:
            raise Phase4PilotContractError(
                f"{state.lane} replay invocation does not verify"
            ) from error
        scheduled = invocation["scheduledGameIndices"]
        assert isinstance(scheduled, list)
        if (
            any(index >= expected_games_per_generation for index in scheduled)
            or metadata.get("invocationGameIndex") != scheduled.index(game_index)
            or metadata.get("resumedSubset")
            != (len(scheduled) < expected_games_per_generation)
        ):
            raise Phase4PilotContractError(
                f"{state.lane} replay invocation schedule changed"
            )
        invocation_id = str(invocation["invocationId"])
        summary = invocations_by_generation[generation].setdefault(
            invocation_id,
            {**invocation, "completedGameIndices": []},
        )
        completed_indices = summary["completedGameIndices"]
        if not isinstance(completed_indices, list):
            raise Phase4PilotContractError("replay invocation summary is invalid")
        completed_indices.append(game_index)
        for ply in game.plies:
            if (
                ply.model_generation != generation
                or ply.model_id != expected_model_id
                or ply.metadata != {"requestedSimulations": 256}
                or sum(ply.visit_counts) < state.config.mcts.simulations
                or _contains_teacher(ply.metadata)
            ):
                raise Phase4PilotContractError(
                    f"{state.lane} replay contains foreign or teacher ply data"
                )
        games_by_generation[generation] += 1
    if any(count != expected_games_per_generation for count in games_by_generation.values()):
        raise Phase4PilotContractError(f"{state.lane} replay generation size mismatch")
    finalized_invocations: dict[int, list[dict[str, object]]] = {}
    for generation, summaries in invocations_by_generation.items():
        finalized: list[dict[str, object]] = []
        for invocation_id in sorted(summaries):
            summary = summaries[invocation_id]
            completed_indices = summary["completedGameIndices"]
            assert isinstance(completed_indices, list)
            summary["completedGameIndices"] = sorted(completed_indices)
            finalized.append(summary)
        finalized_invocations[generation] = finalized
    return {
        "games": verification.game_count,
        "plies": verification.ply_count,
        "bytes": verification.committed_bytes,
        "teacherReferences": 0,
        "deepVerified": verification.deep,
        "_invocationsByGeneration": finalized_invocations,
    }


def _validate_lane_locked(
    *,
    workspace_root: str | Path,
    contract_path: str | Path,
    lane: str,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    contract = load_pilot_contract(contract_path, workspace_root=workspace)
    _assert_current_runtime(workspace, contract)
    contract_sha256 = str(contract["contractSha256"])
    state = load_lane_state(workspace_root=workspace, contract=contract, lane=lane)
    _assert_lane_runtime_binding(state, contract)
    if state.manifest_sha256 is None:
        return {"lane": lane, "nextGeneration": 0, "status": "not-started"}
    attestation = _attest_initialization(
        state,
        contract_sha256,
        control_root=control_root,
    )
    if state.reports and attestation is None:
        raise Phase4PilotContractError(f"{lane} initialization was not attested")

    expected_step = 0
    if state.reports:
        if attestation is None:
            raise Phase4PilotContractError(f"{lane} initialization was not attested")
        expected_producer = f"sha256:{attestation['checkpointSha256']}"
    else:
        expected_producer = ""
    for generation, record in enumerate(state.reports):
        report = record.payload.get("report")
        if not isinstance(report, Mapping):
            raise Phase4PilotContractError(f"{lane} generation report is invalid")
        generation_data = report.get("generationData")
        checkpoint = report.get("checkpoint")
        expected_step += PHASE4D_TRAINING_STEPS
        if (
            not isinstance(generation_data, Mapping)
            or generation_data.get("generation") != generation
            or generation_data.get("generationSeed")
            != (PHASE4D_SEED + generation * 1_000_003) % (2**63)
            or generation_data.get("producerModelId") != expected_producer
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("step") != expected_step
        ):
            raise Phase4PilotContractError(f"{lane} generation workload is invalid")
        self_play = generation_data.get("selfPlay")
        training = generation_data.get("training")
        replay_record = generation_data.get("replay")
        promotion = generation_data.get("promotion")
        if (
            not isinstance(self_play, Mapping)
            or self_play.get("games") != PHASE4D_GAMES_PER_GENERATION
            or self_play.get("actors") != PHASE4D_ACTORS
            or self_play.get("simulations") != 256
            or not isinstance(training, Mapping)
            or training.get("trainerStep") != expected_step
            or training.get("batchSize") != PHASE4D_BATCH_SIZE
            or not isinstance(training.get("metrics"), Mapping)
            or training["metrics"].get("steps") != PHASE4D_TRAINING_STEPS
            or not isinstance(replay_record, Mapping)
            or replay_record.get("games")
            != (generation + 1) * PHASE4D_GAMES_PER_GENERATION
            or not isinstance(promotion, Mapping)
            or promotion.get("status") != "not-run"
        ):
            raise Phase4PilotContractError(f"{lane} generation settings changed")
        expected_producer = f"sha256:{checkpoint.get('sha256')}"

    replay = _validate_replay(state)
    invocations_by_generation = replay.pop("_invocationsByGeneration", {})
    if not isinstance(invocations_by_generation, Mapping):
        raise Phase4PilotContractError(f"{lane} replay invocation index is invalid")
    for generation, record in enumerate(state.reports):
        report = record.payload.get("report")
        generation_data = report.get("generationData") if isinstance(report, Mapping) else None
        self_play = (
            generation_data.get("selfPlay")
            if isinstance(generation_data, Mapping)
            else None
        )
        reported_invocations = (
            self_play.get("invocations") if isinstance(self_play, Mapping) else None
        )
        expected_invocations = invocations_by_generation.get(generation)
        if (
            not isinstance(reported_invocations, list)
            or not isinstance(expected_invocations, list)
            or sorted(
                reported_invocations,
                key=lambda item: str(item.get("invocationId"))
                if isinstance(item, Mapping)
                else "",
            )
            != expected_invocations
            or self_play.get("maximumActualActors")
            != max(
                int(item["actualActors"])
                for item in expected_invocations
            )
        ):
            raise Phase4PilotContractError(
                f"{lane} generation invocation report differs from durable replay"
            )
    latest_checkpoint: dict[str, object] | None = None
    if state.reports:
        path, checkpoint_payload, digest = _checkpoint_payload_from_report(
            state, state.reports[-1]
        )
        if (
            checkpoint_payload.get("generation") != state.next_generation - 1
            or checkpoint_payload.get("step") != expected_step
            or not isinstance(checkpoint_payload.get("model"), Mapping)
        ):
            raise Phase4PilotContractError(f"{lane} latest checkpoint is invalid")
        model_sha256 = canonical_model_state_sha256(checkpoint_payload["model"])
        latest_checkpoint = {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": digest,
            "modelStateSha256": model_sha256,
            "generation": state.next_generation - 1,
            "step": expected_step,
        }
    return {
        "lane": lane,
        "status": "completed" if state.complete else "in-progress",
        "nextGeneration": state.next_generation,
        "manifestSha256": state.manifest_sha256,
        "sourceSha256": state.source_sha256,
        "initialization": attestation,
        "replay": replay,
        "latestCheckpoint": latest_checkpoint,
    }


def validate_lane(
    *,
    workspace_root: str | Path,
    contract_path: str | Path,
    lane: str,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    run_id = f"phase4d-pilot-validate-{lane}-{os.getpid()}-{time.time_ns()}"
    with RunLock(control_root / GPU_LOCK_NAME, run_id=run_id):
        return _validate_lane_locked(
            workspace_root=workspace,
            contract_path=contract_path,
            lane=lane,
        )


def decide_warm_cold(games: Sequence[PairedGameResult]) -> PilotDecision:
    statistics = summarize_paired_results(
        games,
        seed=GATE_BOOTSTRAP_SEED,
        bootstrap_samples=GATE_BOOTSTRAP_SAMPLES,
        confidence=0.90,
    )
    raw_passed = statistics.win_rate >= PHASE4D_RAW_WIN_RATE_THRESHOLD
    lower_passed = (
        statistics.win_rate_ci.lower > PHASE4D_ONE_SIDED_LOWER_THRESHOLD
    )
    warm_selected = raw_passed and lower_passed
    return PilotDecision(
        selected_lane="warm" if warm_selected else "cold",
        warm_selected=warm_selected,
        raw_win_rate=statistics.win_rate,
        one_sided_95_lower=statistics.win_rate_ci.lower,
        raw_threshold_passed=raw_passed,
        lower_threshold_passed=lower_passed,
    )


class _BudgetedEvaluator:
    def __init__(self, evaluator: TorchEvaluator, guard: _RuntimeBudgetGuard) -> None:
        self.evaluator = evaluator
        self.guard = guard

    def enable_native_encoder(self, module: object) -> None:
        self.evaluator.enable_native_encoder(module)

    def evaluate(self, state: object) -> tuple[Sequence[float], float]:
        self.guard.check()
        return self.evaluator.evaluate(state)

    def evaluate_batch(
        self, states: Sequence[object]
    ) -> list[tuple[Sequence[float], float]]:
        self.guard.check()
        return self.evaluator.evaluate_batch(states)


class _DurableJsonlSink:
    def __init__(self, path: Path, guard: _RuntimeBudgetGuard) -> None:
        self.path = path
        self.guard = guard
        self._stream: TextIO = path.open("x", encoding="utf-8")

    def write(self, value: str) -> int:
        self.guard.check(
            reserve_bytes=(
                len(value.encode("utf-8")) + _JSONL_RECORD_WRITE_OVERHEAD_BYTES
            )
        )
        return self._stream.write(value)

    def flush(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self.flush()
        self._stream.close()


def _prepare_arena_commit(
    report_payload: Mapping[str, object],
    partial_path: Path,
    guard: _RuntimeBudgetGuard,
) -> tuple[dict[str, object], bytes, int, dict[str, object]]:
    provisional = dict(report_payload)
    provisional["resourcesBeforeCommit"] = None
    provisional_bytes = canonical_json_bytes(provisional)
    provisional_reserve = (
        len(provisional_bytes)
        + partial_path.stat().st_size
        + _ARENA_RESULT_WRITE_OVERHEAD_BYTES
    )
    reserved_resources = guard.check(
        force=True,
        reserve_bytes=provisional_reserve,
    )
    final = dict(report_payload)
    final["resourcesBeforeCommit"] = reserved_resources
    final_bytes = canonical_json_bytes(final)
    final_reserve = (
        len(final_bytes)
        + partial_path.stat().st_size
        + _ARENA_RESULT_WRITE_OVERHEAD_BYTES
    )
    resources_at_commit = guard.check(
        force=True,
        reserve_bytes=final_reserve,
    )
    return final, final_bytes, final_reserve, resources_at_commit


def _screen_suite(workspace: Path, contract: Mapping[str, object]) -> OpeningSuite:
    arena = contract.get("arena")
    if not isinstance(arena, Mapping):
        raise Phase4PilotContractError("pilot arena contract is invalid")
    source_path, _ = _resolve_workspace_file(
        workspace, arena.get("sourceSuitePath"), "dev opening suite"
    )
    if sha256_file(source_path) != arena.get("sourceSuiteFileSha256"):
        raise Phase4PilotContractError("dev opening suite file changed")
    full = load_opening_suite(source_path, expected_suite="dev")
    if full.content_hash != arena.get("sourceSuiteHash"):
        raise Phase4PilotContractError("dev opening suite identity changed")
    suite = build_opening_suite(
        full.openings[:PHASE4D_ARENA_OPENINGS], suite="dev"
    )
    if (
        suite.content_hash != arena.get("screenSuiteHash")
        or [opening.id for opening in suite.openings] != arena.get("openingIds")
    ):
        raise Phase4PilotContractError("frozen first-32 opening screen changed")
    return suite


def _load_final_model(
    state: LaneState,
    device: torch.device,
) -> tuple[PolicyValueNetwork, dict[str, object]]:
    if not state.complete:
        raise Phase4PilotIncomplete(f"{state.lane} lane has not completed five generations")
    path, checkpoint, digest = _checkpoint_payload_from_report(state, state.reports[-1])
    if checkpoint.get("generation") != PHASE4D_PILOT_GENERATIONS - 1:
        raise Phase4PilotContractError(f"{state.lane} final generation is not 4")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise Phase4PilotContractError(f"{state.lane} final model state is invalid")
    model = PolicyValueNetwork(_network_config(state.config))
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise Phase4PilotContractError(
            f"{state.lane} final model does not match its config"
        ) from error
    model_sha256 = canonical_model_state_sha256(model)
    model.to(device)
    model.eval()
    return model, {
        "checkpointPath": path,
        "checkpointSha256": digest,
        "modelStateSha256": model_sha256,
    }


def _result_summary_from_manifest(
    manifest_path: Path,
    *,
    expected_contract_sha256: str,
) -> dict[str, object] | None:
    if not manifest_path.exists():
        return None
    manifest = _load_json_mapping(manifest_path, "pilot result manifest")
    _verify_self_hash(manifest, "resultManifestSha256", "pilot result manifest")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("contractSha256") != expected_contract_sha256
        or manifest.get("status") != "completed"
    ):
        raise Phase4PilotContractError("pilot result manifest does not match contract")
    for key in ("report", "gamesJsonl"):
        record = manifest.get(key)
        if not isinstance(record, Mapping):
            raise Phase4PilotContractError(f"pilot result {key} record is invalid")
        path = manifest_path.parent / str(record.get("file"))
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
            or (
                key == "gamesJsonl"
                and record.get("completedGames") != PHASE4D_ARENA_GAMES
            )
        ):
            raise Phase4PilotContractError(f"pilot result {key} does not verify")
        if key == "gamesJsonl":
            try:
                game_records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise Phase4PilotContractError(
                    "pilot games JSONL is unreadable"
                ) from error
            if (
                len(game_records) != PHASE4D_ARENA_GAMES
                or any(not isinstance(game, Mapping) for game in game_records)
                or len({game.get("gameId") for game in game_records})
                != PHASE4D_ARENA_GAMES
            ):
                raise Phase4PilotContractError(
                    "pilot games JSONL does not contain 64 unique complete games"
                )
    decision = manifest.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("selectedLane") not in {"cold", "warm"}
        or decision.get("screenOnly") is not True
        or decision.get("formalChampionDeclaration") is not False
    ):
        raise Phase4PilotContractError("pilot result decision is invalid")
    return manifest


def _link_or_verify(source: Path, destination: Path) -> None:
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or sha256_file(source) != sha256_file(destination)
        ):
            raise Phase4PilotContractError(
                f"immutable arena artifact collision: {destination}"
            )
        return
    try:
        os.link(source, destination)
    except FileExistsError as error:
        if not destination.is_file() or sha256_file(source) != sha256_file(destination):
            raise Phase4PilotContractError(
                f"immutable arena artifact collision: {destination}"
            ) from error


def run_pilot_arena(
    *,
    workspace_root: str | Path,
    contract_path: str | Path,
    external_gpu_hours: float = 0.0,
    arena_runner: Callable[..., PairedArenaResult] = run_paired_arena,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = (workspace / PHASE4_CONTROL_RELATIVE).resolve()
    contract = load_pilot_contract(contract_path, workspace_root=workspace)
    contract_sha256 = str(contract["contractSha256"])
    contract_source = Path(contract_path)
    contract_source = (
        contract_source if contract_source.is_absolute() else workspace / contract_source
    ).resolve()
    result_root = contract_source.parent / "arena"
    result_root.mkdir(parents=True, exist_ok=True)
    result_manifest_path = result_root / PHASE4D_RESULT_MANIFEST_NAME
    existing = _result_summary_from_manifest(
        result_manifest_path, expected_contract_sha256=contract_sha256
    )
    if existing is not None:
        return {
            "status": "completed",
            "resumedResult": True,
            "resultManifestPath": result_manifest_path.relative_to(workspace).as_posix(),
            "resultManifestSha256": existing["resultManifestSha256"],
            "selectedLane": existing["decision"]["selectedLane"],
        }

    cold_validation = validate_lane(
        workspace_root=workspace, contract_path=contract_source, lane="cold"
    )
    warm_validation = validate_lane(
        workspace_root=workspace, contract_path=contract_source, lane="warm"
    )
    cold = load_lane_state(workspace_root=workspace, contract=contract, lane="cold")
    warm = load_lane_state(workspace_root=workspace, contract=contract, lane="warm")
    schedule = _schedule_prefix(cold, warm)
    if not cold.complete or not warm.complete or len(schedule) != 10:
        raise Phase4PilotIncomplete("both interleaved five-generation lanes are required")
    if cold.source_sha256 != warm.source_sha256:
        raise Phase4PilotContractError(
            "cold and warm lanes were not trained from the same source fingerprint"
        )
    if cold.native_backend_sha256 != warm.native_backend_sha256:
        raise Phase4PilotContractError(
            "cold and warm lanes were not trained with the same native backend"
        )
    suite = _screen_suite(workspace, contract)

    run_id = f"phase4d-pilot-arena-{os.getpid()}-{time.time_ns()}"
    with RunLock(control_root / GPU_LOCK_NAME, run_id=run_id):
        baseline = verify_phase4d_baseline_report(control_root)
        ledger = GpuBudgetLedger(control_root / GPU_LEDGER_NAME)
        stage_ledger = Phase4StageBudgetLedger(
            control_root / PHASE4_STAGE_LEDGER_NAME
        )
        stage_ledger.ensure_stage(
            PHASE4D_STAGE,
            PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
            PHASE4D_GPU_HOUR_LIMIT,
        )
        stage_ledger.assert_available(PHASE4D_STAGE, ledger.total_hours)
        if ledger.total_hours >= GPU_HOUR_BUDGET:
            raise Phase4ResourceLimitReached("72 GPU-hour global budget is exhausted")
        started = time.monotonic()
        ledger.begin(run_id, external_hours=external_gpu_hours)
        guard = _RuntimeBudgetGuard(
            control_root=control_root,
            artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
            ledger=ledger,
            stage_ledger=stage_ledger,
            ledger_run_id=run_id,
            started=started,
        )
        status = "failed"
        sink: _DurableJsonlSink | None = None
        candidate: BoundMoveSelector | None = None
        opponent: BoundMoveSelector | None = None
        partial_path = result_root / (
            f"partial-{os.getpid()}-{time.time_ns()}.games.jsonl"
        )
        try:
            guard.check(force=True)
            device = resolve_device("cuda")
            native_module = load_native_module()
            backend_sha256 = loaded_native_backend_hash()
            if backend_sha256 != cold.native_backend_sha256:
                raise Phase4PilotContractError(
                    "arena native backend differs from the training backend"
                )
            warm_model, warm_artifact = _load_final_model(warm, device)
            cold_model, cold_artifact = _load_final_model(cold, device)
            warm_evaluator = _BudgetedEvaluator(
                TorchEvaluator(warm_model, device, use_amp=True), guard
            )
            cold_evaluator = _BudgetedEvaluator(
                TorchEvaluator(cold_model, device, use_amp=True), guard
            )
            warm_evaluator.enable_native_encoder(native_module)
            cold_evaluator.enable_native_encoder(native_module)
            search_config = MCTSConfig(
                simulations=PHASE4D_ARENA_SIMULATIONS,
                c_base=warm.config.mcts.c_base,
                c_init=warm.config.mcts.c_init,
                dirichlet_epsilon=0.0,
                dirichlet_alpha=warm.config.mcts.dirichlet_alpha,
                temperature=0.0,
                add_root_noise=False,
                seed=PHASE4D_SEED,
            )
            warm_identity = SelectorIdentity(
                selector_id="phase4d-warm-g4",
                artifact_hash=str(warm_artifact["checkpointSha256"]),
                config_hash=fingerprint_config(warm.config).sha256,
                source_hash=warm.source_sha256,
                engine="native-mcts",
            )
            cold_identity = SelectorIdentity(
                selector_id="phase4d-cold-g4",
                artifact_hash=str(cold_artifact["checkpointSha256"]),
                config_hash=fingerprint_config(cold.config).sha256,
                source_hash=cold.source_sha256,
                engine="native-mcts",
            )
            candidate = BoundMoveSelector(
                MCTSSelector(
                    warm_evaluator,
                    search_config,
                    search_factory=NativeMCTS,
                    engine="native-mcts",
                ),
                warm_identity,
            )
            opponent = BoundMoveSelector(
                MCTSSelector(
                    cold_evaluator,
                    search_config,
                    search_factory=NativeMCTS,
                    engine="native-mcts",
                ),
                cold_identity,
            )
            sink = _DurableJsonlSink(partial_path, guard)
            arena = arena_runner(
                suite,
                candidate,
                opponent,
                jsonl_sink=sink,
                bootstrap_seed=GATE_BOOTSTRAP_SEED,
                bootstrap_samples=GATE_BOOTSTRAP_SAMPLES,
                confidence=0.95,
            )
            sink.close()
            sink = None
            if len(arena.games) != PHASE4D_ARENA_GAMES:
                raise Phase4PilotContractError("paired arena did not complete 64 games")
            decision = decide_warm_cold(
                tuple(game.to_paired_result() for game in arena.games)
            )
            report_payload = {
                "schemaVersion": 1,
                "recordType": "phase4d-warm-cold-screen",
                "status": "completed",
                "createdAt": utc_timestamp(),
                "contractPath": contract_source.relative_to(workspace).as_posix(),
                "contractSha256": contract_sha256,
                "baselineCalibration": baseline,
                "nativeBackendSha256": backend_sha256,
                "schedule": [
                    {"lane": lane, "generation": generation, "createdAt": created}
                    for lane, generation, created in schedule
                ],
                "lanes": {"cold": cold_validation, "warm": warm_validation},
                "artifacts": {"cold": cold_artifact, "warm": warm_artifact},
                "arena": arena.to_dict(),
                "arenaSearchConfig": {
                    "simulations": PHASE4D_ARENA_SIMULATIONS,
                    "cBase": search_config.c_base,
                    "cInit": search_config.c_init,
                    "dirichletEpsilon": search_config.dirichlet_epsilon,
                    "dirichletAlpha": search_config.dirichlet_alpha,
                    "temperature": search_config.temperature,
                    "addRootNoise": search_config.add_root_noise,
                    "seed": search_config.seed,
                },
                "decision": decision.to_payload(),
            }
            (
                report_payload,
                report_bytes,
                required,
                resources_at_commit,
            ) = _prepare_arena_commit(report_payload, partial_path, guard)
            report_sha256 = sha256_bytes(report_bytes)
            report_path = result_root / f"report-{report_sha256}.json"
            jsonl_sha256 = sha256_file(partial_path)
            jsonl_path = result_root / f"games-{jsonl_sha256}.jsonl"
            _link_or_verify(partial_path, jsonl_path)
            partial_path.unlink()
            _create_or_verify(report_path, report_bytes, "arena report")
            result_core = {
                "schemaVersion": 1,
                "status": "completed",
                "createdAt": utc_timestamp(),
                "contractSha256": contract_sha256,
                "report": {
                    "file": report_path.name,
                    "sha256": report_sha256,
                    "bytes": report_path.stat().st_size,
                },
                "gamesJsonl": {
                    "file": jsonl_path.name,
                    "sha256": jsonl_sha256,
                    "bytes": jsonl_path.stat().st_size,
                    "completedGames": PHASE4D_ARENA_GAMES,
                },
                "decision": decision.to_payload(),
                "resourcesAtCommit": resources_at_commit,
                "reservedWriteBytes": required,
            }
            result_manifest = _self_hashed_payload(
                result_core, "resultManifestSha256"
            )
            _create_or_verify(
                result_manifest_path,
                canonical_json_bytes(result_manifest),
                "arena result manifest",
            )
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            active_error = sys.exc_info()[1]
            cleanup_error: BaseException | None = None
            for resource in (sink, candidate, opponent):
                if resource is None:
                    continue
                try:
                    resource.close()
                except BaseException as error:  # Preserve accounting on cleanup failure.
                    if cleanup_error is None:
                        cleanup_error = error
            elapsed = max(0.0, time.monotonic() - started)
            try:
                final_global_hours = ledger.update(
                    run_id, elapsed_seconds=elapsed, status=status
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if cleanup_error is not None and active_error is None:
                raise cleanup_error
        if status != "completed":
            raise RuntimeError("arena exited without a completed result")
        return {
            "status": "completed",
            "resumedResult": False,
            "resultManifestPath": result_manifest_path.relative_to(workspace).as_posix(),
            "resultManifestSha256": result_manifest["resultManifestSha256"],
            "reportPath": report_path.relative_to(workspace).as_posix(),
            "reportSha256": report_sha256,
            "gamesJsonlPath": jsonl_path.relative_to(workspace).as_posix(),
            "gamesJsonlSha256": jsonl_sha256,
            "selectedLane": decision.selected_lane,
            "screenOnly": True,
            "globalGpuHours": final_global_hours,
            "stageGpu": stage_ledger.usage(PHASE4D_STAGE, final_global_hours),
        }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4D warm/cold pilot")
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create immutable pilot configs")
    prepare.add_argument("--pretraining-summary", required=True)
    prepare.add_argument(
        "--base-config", default="training/configs/phase4-bootstrap.json"
    )
    prepare.add_argument("--output-root", default=str(PHASE4D_DEFAULT_ROOT))

    next_parser = subparsers.add_parser(
        "next-lane", help="print cold, warm, or complete"
    )
    next_parser.add_argument("--contract", required=True)

    step_parser = subparsers.add_parser(
        "next-step", help="print the next lane and absolute generation target"
    )
    step_parser.add_argument("--contract", required=True)

    validate_parser = subparsers.add_parser(
        "validate-lane", help="deeply validate one completed lane prefix"
    )
    validate_parser.add_argument("--contract", required=True)
    validate_parser.add_argument("--lane", choices=("cold", "warm"), required=True)

    arena = subparsers.add_parser("arena", help="run the final 64-game pilot screen")
    arena.add_argument("--contract", required=True)
    arena.add_argument("--gpu-hours-used", type=float, default=0.0)
    return parser


def _raise_interrupt(_signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    workspace = Path(args.workspace_root).resolve()
    previous_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        if args.command == "prepare":
            result: object = prepare_pilot(
                workspace_root=workspace,
                pretraining_summary_path=args.pretraining_summary,
                base_config_path=args.base_config,
                output_root=args.output_root,
            )
        elif args.command == "next-lane":
            print(next_lane(workspace_root=workspace, contract_path=args.contract))
            return 0
        elif args.command == "next-step":
            lane, target = next_step(
                workspace_root=workspace, contract_path=args.contract
            )
            print(f"{lane} {target}")
            return 0
        elif args.command == "validate-lane":
            result = validate_lane(
                workspace_root=workspace,
                contract_path=args.contract,
                lane=args.lane,
            )
        elif args.command == "arena":
            result = run_pilot_arena(
                workspace_root=workspace,
                contract_path=args.contract,
                external_gpu_hours=args.gpu_hours_used,
            )
        else:  # pragma: no cover - argparse guarantees the command.
            raise AssertionError(args.command)
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "message": (
                        "training lanes resume from checksummed checkpoints; "
                        "a partial arena JSONL is not a result"
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PHASE4D_PILOT_SCHEMA_VERSION",
    "PHASE4D_ARENA_OPENINGS",
    "PHASE4D_ARENA_GAMES",
    "Phase4PilotError",
    "Phase4PilotContractError",
    "Phase4PilotIncomplete",
    "LaneState",
    "PilotDecision",
    "build_effective_config_payloads",
    "prepare_pilot",
    "load_pilot_contract",
    "load_lane_state",
    "next_step",
    "next_lane",
    "validate_lane",
    "decide_warm_cold",
    "run_pilot_arena",
    "create_parser",
    "main",
]
