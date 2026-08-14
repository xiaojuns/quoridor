"""Phase 4 native self-play throughput calibration.

This module exercises the production self-play path through compact raw replay
storage. It never creates an optimizer, encoded training sample, or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shutil
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch

from .config import seed_everything
from .evaluator import TorchEvaluator
from .game import create_initial_state
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PHASE4_INPUT_CHANNELS, PolicyValueNetwork
from .replay_store import ReplayGame, ReplayPly, ReplayStore
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ResourceBudget,
    RunLock,
    atomic_replace_bytes,
    build_run_manifest,
    canonical_json_bytes,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
)
from .selfplay import SelfPlayConfig, SelfPlayGame, play_self_games_scheduled


SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
LEGACY_GPU_HOUR_BUDGET = 72.0
GPU_HOUR_BUDGET = 120.0
GPU_LEDGER_NAME = "gpu-budget-ledger.json"
GPU_LOCK_NAME = ".phase4.lock"
LEDGER_HEARTBEAT_SECONDS = 5.0
DISK_CHECK_INTERVAL_SECONDS = 30.0
MAX_SCENARIO_REPLAY_BYTES = 512 * 1024**2
ARTIFACT_WRITE_RESERVE_BYTES = 4 * 1024**2
FULL_NETWORKS = (
    ("128x10", 128, 10),
    ("192x16", 192, 16),
)
FULL_ACTOR_COUNTS = (8, 16, 32, 64)
FULL_SIMULATION_COUNTS = (128, 256)


class CalibrationPreflightError(RuntimeError):
    """Raised before calibration when a fixed Phase 4 guard is not satisfied."""


class CalibrationBudgetExceeded(RuntimeError):
    """Raised as soon as a Phase 4 resource stop line is reached."""


class GpuBudgetLedger:
    """Self-hashed cumulative GPU accounting shared by every Phase 4 script."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.external_hours_floor = 0.0
        self.runs: dict[str, dict[str, object]] = {}
        if self.path.exists():
            self._load()

    @property
    def total_hours(self) -> float:
        run_seconds = sum(
            float(run["accountedSeconds"])
            for run in self.runs.values()
        )
        return self.external_hours_floor + run_seconds / 3600.0

    def record_external_hours(self, external_hours: float) -> float:
        """Persist the monotonic floor for Phase 4 work outside this ledger."""

        if (
            isinstance(external_hours, bool)
            or not isinstance(external_hours, (int, float))
            or not math.isfinite(float(external_hours))
            or external_hours < 0
        ):
            raise ValueError("external_hours must be finite and non-negative")
        self.external_hours_floor = max(
            self.external_hours_floor,
            float(external_hours),
        )
        self._save()
        return self.total_hours

    def begin(self, run_id: str, *, external_hours: float) -> float:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if run_id in self.runs:
            raise ValueError(f"GPU ledger already contains run {run_id}")
        if (
            isinstance(external_hours, bool)
            or not isinstance(external_hours, (int, float))
            or not math.isfinite(float(external_hours))
            or external_hours < 0
        ):
            raise ValueError("external_hours must be finite and non-negative")
        for run in self.runs.values():
            if run["status"] == "running":
                run["status"] = "abandoned"
        self.external_hours_floor = max(
            self.external_hours_floor,
            float(external_hours),
        )
        self.runs[run_id] = {
            "startedAt": utc_timestamp(),
            "updatedAt": utc_timestamp(),
            "accountedSeconds": 0.0,
            "status": "running",
        }
        self._save()
        return self.total_hours

    def update(self, run_id: str, *, elapsed_seconds: float, status: str) -> float:
        if status not in {"running", "completed", "interrupted", "failed"}:
            raise ValueError("unsupported GPU ledger run status")
        if not math.isfinite(float(elapsed_seconds)) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        try:
            run = self.runs[run_id]
        except KeyError as error:
            raise KeyError(f"GPU ledger does not contain run {run_id}") from error
        previous = float(run["accountedSeconds"])
        if elapsed_seconds < previous:
            raise ValueError("GPU ledger elapsed time must be monotonic")
        run["accountedSeconds"] = float(elapsed_seconds)
        run["updatedAt"] = utc_timestamp()
        run["status"] = status
        self._save()
        return self.total_hours

    def _core(self) -> dict[str, object]:
        return {
            "schemaVersion": LEDGER_SCHEMA_VERSION,
            "limitHours": GPU_HOUR_BUDGET,
            "externalHoursFloor": self.external_hours_floor,
            "runs": self.runs,
            "updatedAt": utc_timestamp(),
        }

    def _save(self) -> None:
        core = self._core()
        payload = {
            **core,
            "ledgerSha256": sha256_bytes(canonical_json_bytes(core)),
        }
        atomic_replace_bytes(self.path, canonical_json_bytes(payload))

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CalibrationPreflightError("GPU budget ledger is unreadable") from error
        if not isinstance(payload, Mapping):
            raise CalibrationPreflightError("GPU budget ledger must be a mapping")
        stored_hash = payload.get("ledgerSha256")
        core = dict(payload)
        core.pop("ledgerSha256", None)
        if (
            not isinstance(stored_hash, str)
            or sha256_bytes(canonical_json_bytes(core)) != stored_hash
        ):
            raise CalibrationPreflightError("GPU budget ledger checksum mismatch")
        if (
            core.get("schemaVersion") != LEDGER_SCHEMA_VERSION
            or core.get("limitHours")
            not in {LEGACY_GPU_HOUR_BUDGET, GPU_HOUR_BUDGET}
        ):
            raise CalibrationPreflightError("GPU budget ledger contract mismatch")
        external = core.get("externalHoursFloor")
        runs = core.get("runs")
        if (
            isinstance(external, bool)
            or not isinstance(external, (int, float))
            or not math.isfinite(float(external))
            or external < 0
            or not isinstance(runs, Mapping)
        ):
            raise CalibrationPreflightError("GPU budget ledger payload is invalid")
        normalized: dict[str, dict[str, object]] = {}
        for run_id, value in runs.items():
            if not isinstance(run_id, str) or not isinstance(value, Mapping):
                raise CalibrationPreflightError("GPU budget ledger run is invalid")
            seconds = value.get("accountedSeconds")
            status = value.get("status")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(float(seconds))
                or seconds < 0
                or status not in {
                    "running",
                    "completed",
                    "interrupted",
                    "failed",
                    "abandoned",
                }
            ):
                raise CalibrationPreflightError("GPU budget ledger run is invalid")
            normalized[run_id] = dict(value)
        self.external_hours_floor = float(external)
        self.runs = normalized


class NvidiaSmiProfiler:
    """Sample local GPU utilization without adding a Python package dependency."""

    def __init__(self, device_index: int, *, interval_seconds: float = 1.0) -> None:
        if type(device_index) is not int or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float, float, float | None]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU utilization profiler is already running")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_seconds + 2.0))
            if self._thread.is_alive():
                self.error = "nvidia-smi sampler did not stop"
        gpu = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        used_mib = [sample[2] for sample in self.samples]
        power = [sample[3] for sample in self.samples if sample[3] is not None]
        return {
            "available": bool(gpu),
            "backend": "nvidia-smi",
            "sampleCount": len(gpu),
            "intervalSeconds": self.interval_seconds,
            "error": self.error,
            "gpuPercent": self._metric(gpu),
            "memoryControllerPercent": self._metric(memory),
            "memoryUsedMiB": self._metric(used_mib),
            "powerWatts": self._metric(power),
        }

    @staticmethod
    def _metric(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "maximum": max(values, default=None),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                command = [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ]
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                fields = [item.strip() for item in result.stdout.strip().split(",")]
                if len(fields) != 4:
                    raise ValueError("nvidia-smi returned the wrong field count")
                parsed = [
                    float(value) if value not in {"N/A", "[N/A]"} else None
                    for value in fields
                ]
                if any(parsed[index] is None for index in range(3)):
                    raise ValueError("nvidia-smi omitted a required utilization field")
                required = [float(parsed[index]) for index in range(3)]
                if (
                    not all(math.isfinite(value) for value in required)
                    or not 0 <= required[0] <= 100
                    or not 0 <= required[1] <= 100
                    or required[2] < 0
                    or (
                        parsed[3] is not None
                        and (
                            not math.isfinite(float(parsed[3]))
                            or float(parsed[3]) < 0
                        )
                    )
                ):
                    raise ValueError("nvidia-smi returned invalid telemetry values")
                self.samples.append(
                    (required[0], required[1], required[2], parsed[3])
                )
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.error = f"{type(error).__name__}: {error}"
                return
            self._stop.wait(self.interval_seconds)


@dataclass(frozen=True, slots=True)
class CalibrationScenario:
    network_name: str
    channels: int
    blocks: int
    actors: int
    simulations: int
    games: int
    seed: int

    def __post_init__(self) -> None:
        if not self.network_name:
            raise ValueError("network_name must not be empty")
        for name in ("channels", "blocks", "actors", "simulations", "games"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.games > 1_000_000:
            raise ValueError("games is unreasonably large")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2^63)")

    @property
    def scenario_id(self) -> str:
        return (
            f"{self.network_name}-a{self.actors}-s{self.simulations}"
            f"-g{self.games}"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "scenarioId": self.scenario_id,
            "network": {
                "name": self.network_name,
                "inputChannels": PHASE4_INPUT_CHANNELS,
                "channels": self.channels,
                "residualBlocks": self.blocks,
            },
            "actors": self.actors,
            "simulations": self.simulations,
            "games": self.games,
            "seed": self.seed,
        }


def build_scenarios(
    profile: str,
    *,
    seed: int,
    full_games_per_actor: int = 2,
) -> tuple[CalibrationScenario, ...]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2^63)")
    if type(full_games_per_actor) is not int or full_games_per_actor <= 0:
        raise ValueError("full_games_per_actor must be a positive integer")

    if profile == "quick":
        return (
            CalibrationScenario(
                network_name="128x10",
                channels=128,
                blocks=10,
                actors=2,
                simulations=16,
                games=2,
                seed=seed,
            ),
        )

    scenarios: list[CalibrationScenario] = []
    combination_index = 0
    for simulations in FULL_SIMULATION_COUNTS:
        for actors in FULL_ACTOR_COUNTS:
            networks = (
                FULL_NETWORKS
                if combination_index % 2 == 0
                else tuple(reversed(FULL_NETWORKS))
            )
            common_seed = (seed + combination_index) % (2**63)
            for network_name, channels, blocks in networks:
                scenarios.append(
                    CalibrationScenario(
                        network_name=network_name,
                        channels=channels,
                        blocks=blocks,
                        actors=actors,
                        simulations=simulations,
                        games=actors * full_games_per_actor,
                        seed=common_seed,
                    )
                )
            combination_index += 1
    return tuple(scenarios)


def _percentile(values: Sequence[float | int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class EvaluationProfiler:
    """Measure the actual evaluator calls made by the native batch scheduler."""

    def __init__(
        self,
        evaluator: TorchEvaluator,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.evaluator = evaluator
        self.clock = clock
        self.batch_sizes: list[int] = []
        self.elapsed_seconds: list[float] = []

    @property
    def device(self) -> torch.device:
        return self.evaluator.device

    def enable_native_encoder(self, module: object) -> None:
        self.evaluator.enable_native_encoder(module)

    def evaluate(self, state: object):
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(self, states: Iterable[object]):
        state_list = list(states)
        started = self.clock()
        result = self.evaluator.evaluate_batch(state_list)
        elapsed = self.clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("evaluation clock returned an invalid duration")
        self.batch_sizes.append(len(state_list))
        self.elapsed_seconds.append(elapsed)
        return result

    def summary(self) -> dict[str, object]:
        return {
            "calls": len(self.batch_sizes),
            "positions": sum(self.batch_sizes),
            "batchSize": {
                "p50": _percentile(self.batch_sizes, 0.50),
                "p95": _percentile(self.batch_sizes, 0.95),
                "maximum": max(self.batch_sizes, default=None),
            },
            "latencyMilliseconds": {
                "p50": (
                    None
                    if not self.elapsed_seconds
                    else _percentile(self.elapsed_seconds, 0.50) * 1000.0
                ),
                "p95": (
                    None
                    if not self.elapsed_seconds
                    else _percentile(self.elapsed_seconds, 0.95) * 1000.0
                ),
                "total": sum(self.elapsed_seconds) * 1000.0,
            },
        }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise CalibrationPreflightError(
                f"artifact tree must not contain symlinks: {item}"
            )
        if item.is_file():
            total += item.stat().st_size
    return total


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise CalibrationPreflightError(
                f"cannot find an existing filesystem ancestor for {path}"
            )
        candidate = parent
    return candidate


def collect_preflight(
    output_root: str | Path,
    *,
    device_index: int,
    gpu_hours_used: float,
    cuda_api: object = torch.cuda,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
) -> dict[str, object]:
    """Validate the fixed local Phase 4 resource contract before any GPU work."""

    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    if (
        isinstance(gpu_hours_used, bool)
        or not isinstance(gpu_hours_used, (int, float))
        or not math.isfinite(float(gpu_hours_used))
        or float(gpu_hours_used) < 0
    ):
        raise ValueError("gpu_hours_used must be finite and non-negative")
    gpu_hours_used = float(gpu_hours_used)
    if gpu_hours_used >= GPU_HOUR_BUDGET:
        raise CalibrationPreflightError("the 72 GPU-hour budget is exhausted")

    if not bool(cuda_api.is_available()):
        raise CalibrationPreflightError("PyTorch CUDA is not available")
    device_count = int(cuda_api.device_count())
    if device_index >= device_count:
        raise CalibrationPreflightError(
            f"CUDA device index {device_index} is outside {device_count} visible devices"
        )
    properties = cuda_api.get_device_properties(device_index)

    root = Path(output_root).resolve()
    filesystem_path = _existing_ancestor(root)
    usage = disk_usage(filesystem_path)
    free_bytes = int(getattr(usage, "free"))
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise CalibrationPreflightError(
            "less than the Phase 4 20 GiB free-disk stop line is available"
        )
    artifact_bytes = _directory_size(root)
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise CalibrationPreflightError(
            "the Phase 4 calibration artifact tree has reached the 8 GiB cap"
        )

    gpu: dict[str, object] = {
        "index": device_index,
        "name": str(properties.name),
        "totalMemoryBytes": int(properties.total_memory),
    }
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
        "disk": {
            "filesystemPath": str(filesystem_path),
            "freeBytesAtStart": free_bytes,
            "minimumFreeBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        },
        "artifacts": {
            "root": str(root),
            "bytesAtStart": artifact_bytes,
            "limitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        },
        "gpuHours": {
            "usedBeforeCalibration": gpu_hours_used,
            "limit": GPU_HOUR_BUDGET,
            "remainingAtStart": GPU_HOUR_BUDGET - gpu_hours_used,
        },
    }


def assert_runtime_budget(
    output_root: str | Path,
    *,
    gpu_hours_used: float,
    calibration_started: float,
    clock: Callable[[], float] = time.perf_counter,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
) -> dict[str, object]:
    """Enforce stop lines between scenarios, never in the middle of a game."""

    root = Path(output_root).resolve()
    usage = disk_usage(_existing_ancestor(root))
    free_bytes = int(getattr(usage, "free"))
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise CalibrationBudgetExceeded("20 GiB free-disk stop line reached")
    artifact_bytes = _directory_size(root)
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise CalibrationBudgetExceeded("8 GiB artifact cap reached")
    elapsed_seconds = clock() - calibration_started
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise RuntimeError("calibration clock returned an invalid duration")
    calibration_gpu_hours = elapsed_seconds / 3600.0
    total_gpu_hours = gpu_hours_used + calibration_gpu_hours
    if total_gpu_hours >= GPU_HOUR_BUDGET:
        raise CalibrationBudgetExceeded("72 GPU-hour budget reached")
    return {
        "freeDiskBytes": free_bytes,
        "artifactBytes": artifact_bytes,
        "calibrationGpuHours": calibration_gpu_hours,
        "totalGpuHours": total_gpu_hours,
        "remainingGpuHours": GPU_HOUR_BUDGET - total_gpu_hours,
    }


def scenario_replay_budget(output_root: str | Path) -> int:
    """Reserve replay capacity without allowing a scenario to cross a stop line."""

    root = Path(output_root).resolve()
    artifact_remaining = (
        DEFAULT_ARTIFACT_LIMIT_BYTES
        - _directory_size(root)
        - ARTIFACT_WRITE_RESERVE_BYTES
    )
    free_bytes = int(shutil.disk_usage(_existing_ancestor(root)).free)
    disk_remaining = (
        free_bytes
        - DEFAULT_MINIMUM_FREE_DISK_BYTES
        - ARTIFACT_WRITE_RESERVE_BYTES
    )
    available = min(
        MAX_SCENARIO_REPLAY_BYTES,
        artifact_remaining,
        disk_remaining,
    )
    if available <= 0:
        raise CalibrationBudgetExceeded(
            "no replay capacity remains before a Phase 4 resource stop line"
        )
    return int(available)


def _cuda_memory_payload(device: torch.device, cuda_api: object) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "peakAllocatedBytes": None,
            "peakReservedBytes": None,
        }
    return {
        "peakAllocatedBytes": int(cuda_api.max_memory_allocated(device)),
        "peakReservedBytes": int(cuda_api.max_memory_reserved(device)),
    }


def run_scenario(
    scenario: CalibrationScenario,
    evaluator: TorchEvaluator,
    *,
    search_factory: Callable[..., object] = NativeMCTS,
    clock: Callable[[], float] = time.perf_counter,
    cuda_api: object = torch.cuda,
    warmup_batches: int = 2,
    replay_root: str | Path | None = None,
    replay_max_bytes: int = MAX_SCENARIO_REPLAY_BYTES,
    model_id: str = "phase4-calibration-random-model",
    temperature_moves: int = 16,
    progress_callback: Callable[[int, int], None] | None = None,
    telemetry_factory: Callable[[int], object] = NvidiaSmiProfiler,
) -> dict[str, object]:
    """Run one production-shaped self-play scenario without training."""

    if type(warmup_batches) is not int or warmup_batches < 0:
        raise ValueError("warmup_batches must be a non-negative integer")
    if type(replay_max_bytes) is not int or replay_max_bytes <= 0:
        raise ValueError("replay_max_bytes must be a positive integer")
    if type(temperature_moves) is not int or temperature_moves < 0:
        raise ValueError("temperature_moves must be a non-negative integer")
    device = torch.device(evaluator.device)
    warmup_states = [create_initial_state()] * scenario.actors
    for _ in range(warmup_batches):
        evaluator.evaluate_batch(warmup_states)
    if device.type == "cuda":
        cuda_api.synchronize(device)
        cuda_api.reset_peak_memory_stats(device)

    profiler = EvaluationProfiler(evaluator, clock=clock)
    replay: ReplayStore | None = None
    persisted_games = 0
    if replay_root is not None:
        replay = ReplayStore(
            replay_root,
            max_games=scenario.games,
            max_plies=scenario.games * 256,
            max_bytes=replay_max_bytes,
        )

    def persist_game(game_index: int, game: SelfPlayGame) -> None:
        nonlocal persisted_games
        if replay is None:
            return
        if len(game.raw_plies) != game.move_count:
            raise RuntimeError("raw replay ply count does not match the completed game")
        replay.append(
            ReplayGame(
                game_id=f"{scenario.scenario_id}-{game_index:06d}",
                initial_state=game.initial_state,
                plies=tuple(
                    ReplayPly(
                        action_index=ply.action_index,
                        visit_counts=ply.visit_counts,
                        root_value=ply.root_value,
                        model_generation=0,
                        model_id=model_id,
                        metadata={
                            "calibrationOnly": True,
                            "requestedSimulations": scenario.simulations,
                        },
                    )
                    for ply in game.raw_plies
                ),
                metadata={
                    "calibrationOnly": True,
                    "scenarioId": scenario.scenario_id,
                    "seed": scenario.seed,
                },
            )
        )
        persisted_games += 1
        if replay.game_count != persisted_games:
            raise CalibrationBudgetExceeded(
                "scenario replay capacity would evict calibration data"
            )

    config = SelfPlayConfig(
        mcts=MCTSConfig(
            simulations=scenario.simulations,
            temperature=1.0,
            add_root_noise=True,
            seed=scenario.seed,
        ),
        temperature_moves=temperature_moves,
        mirror_augmentation=False,
        seed=scenario.seed,
        input_channels=PHASE4_INPUT_CHANNELS,
        collect_samples=False,
        collect_raw_records=True,
    )

    telemetry: object | None = None
    telemetry_summary: dict[str, object] = {
        "available": False,
        "backend": None,
        "sampleCount": 0,
        "error": "CUDA telemetry is unavailable on a CPU scenario",
    }
    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = int(cuda_api.current_device())
        telemetry = telemetry_factory(device_index)
        telemetry.start()
    started = clock()
    try:
        games = play_self_games_scheduled(
            profiler,
            config,
            scenario.games,
            actors=scenario.actors,
            search_factory=search_factory,
            progress_callback=progress_callback,
            completed_game_callback=persist_game,
        )
        if device.type == "cuda":
            cuda_api.synchronize(device)
    finally:
        if telemetry is not None:
            telemetry_summary = telemetry.stop()
    elapsed_seconds = clock() - started
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise RuntimeError("scenario clock returned a non-positive duration")
    if len(games) != scenario.games:
        raise RuntimeError("self-play scheduler returned the wrong game count")
    if any(game.samples for game in games):
        raise RuntimeError("calibration must not collect training samples")
    if any(len(game.raw_plies) != game.move_count for game in games):
        raise RuntimeError("calibration raw replay is incomplete")

    move_count = sum(game.move_count for game in games)
    scheduled_simulations = move_count * scenario.simulations
    winners = Counter(game.winner for game in games)
    if set(winners) - {0, 1}:
        raise RuntimeError("calibration observed a non-zero-sum winner")
    terminations = Counter(game.termination_reason for game in games)
    game_lengths = [game.move_count for game in games]
    replay_started = clock()
    replay_payload: dict[str, object]
    if replay is None:
        replay_payload = {
            "enabled": False,
            "deepVerified": False,
        }
    else:
        verification = replay.verify(deep=True)
        if verification.game_count != scenario.games:
            raise RuntimeError("calibration replay did not retain every game")
        replay_payload = {
            "enabled": True,
            "deepVerified": verification.deep,
            "gameCount": verification.game_count,
            "plyCount": verification.ply_count,
            "committedBytes": verification.committed_bytes,
            "shards": len(replay.shards),
            "auditSeconds": clock() - replay_started,
        }

    return {
        **scenario.to_payload(),
        "temperatureMoves": temperature_moves,
        "status": "completed",
        "elapsedSeconds": elapsed_seconds,
        "gamesPerHour": scenario.games * 3600.0 / elapsed_seconds,
        "moves": move_count,
        "movesPerSecond": move_count / elapsed_seconds,
        "scheduledSimulations": scheduled_simulations,
        "scheduledSimulationsPerSecond": scheduled_simulations / elapsed_seconds,
        "evaluation": profiler.summary(),
        "gpuUtilization": telemetry_summary,
        "winners": {
            "player0": winners.get(0, 0),
            "player1": winners.get(1, 0),
        },
        "terminationReasons": dict(sorted(terminations.items())),
        "gameLength": {
            "minimum": min(game_lengths),
            "maximum": max(game_lengths),
            "mean": move_count / len(game_lengths),
        },
        "cudaMemory": _cuda_memory_payload(device, cuda_api),
        "replay": replay_payload,
        "trainingSamplesCollected": 0,
        "rawReplayPliesCollected": move_count,
        "checkpointsWritten": 0,
    }


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _finite_percentage(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0 <= float(value) <= 100
    )


def _has_decision_metrics(result: Mapping[str, object]) -> bool:
    evaluation = result.get("evaluation")
    batch_size = evaluation.get("batchSize") if isinstance(evaluation, Mapping) else None
    cuda_memory = result.get("cudaMemory")
    utilization = result.get("gpuUtilization")
    gpu_percent = (
        utilization.get("gpuPercent") if isinstance(utilization, Mapping) else None
    )
    replay = result.get("replay")
    winners = result.get("winners")
    terminations = result.get("terminationReasons")
    games = result.get("games")
    moves = result.get("moves")
    if type(games) is not int or type(moves) is not int or games <= 0 or moves <= 0:
        return False
    if not all(
        _positive_finite(result.get(field))
        for field in (
            "elapsedSeconds",
            "gamesPerHour",
            "movesPerSecond",
            "scheduledSimulationsPerSecond",
        )
    ):
        return False
    if (
        type(result.get("scheduledSimulations")) is not int
        or result.get("scheduledSimulations") != moves * result.get("simulations")
        or not isinstance(evaluation, Mapping)
        or type(evaluation.get("calls")) is not int
        or evaluation.get("calls") <= 0
        or type(evaluation.get("positions")) is not int
        or evaluation.get("positions") <= 0
        or not isinstance(batch_size, Mapping)
        or not _positive_finite(batch_size.get("p50"))
        or not _positive_finite(batch_size.get("p95"))
        or not _positive_finite(batch_size.get("maximum"))
    ):
        return False
    if (
        not isinstance(cuda_memory, Mapping)
        or not _positive_finite(cuda_memory.get("peakAllocatedBytes"))
        or not _positive_finite(cuda_memory.get("peakReservedBytes"))
        or not isinstance(utilization, Mapping)
        or utilization.get("available") is not True
        or utilization.get("error") is not None
        or type(utilization.get("sampleCount")) is not int
        or utilization.get("sampleCount") < 2
        or not isinstance(gpu_percent, Mapping)
        or not _finite_percentage(gpu_percent.get("p50"))
        or not _finite_percentage(gpu_percent.get("p95"))
    ):
        return False
    if (
        not isinstance(replay, Mapping)
        or replay.get("enabled") is not True
        or replay.get("deepVerified") is not True
        or replay.get("gameCount") != games
        or replay.get("plyCount") != moves
        or not _positive_finite(replay.get("committedBytes"))
        or result.get("rawReplayPliesCollected") != moves
        or result.get("trainingSamplesCollected") != 0
        or result.get("checkpointsWritten") != 0
    ):
        return False
    if not isinstance(winners, Mapping) or not isinstance(terminations, Mapping):
        return False
    if any(type(value) is not int or value < 0 for value in winners.values()):
        return False
    if any(type(value) is not int or value < 0 for value in terminations.values()):
        return False
    if (
        sum(winners.values()) != games
        or sum(terminations.values()) != games
        or set(terminations) - {"goal", "repetition", "move-limit"}
    ):
        return False
    return True


def decision_eligibility(
    profile: str,
    scenarios: Sequence[CalibrationScenario],
    results: Sequence[Mapping[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    reasons: list[str] = []
    if profile != "full":
        reasons.append("quick profile is smoke-only")
    if status != "completed":
        reasons.append("calibration has not completed")

    required = {
        (network_name, actors, simulations)
        for network_name, _, _ in FULL_NETWORKS
        for actors in FULL_ACTOR_COUNTS
        for simulations in FULL_SIMULATION_COUNTS
    }
    configured = {
        (scenario.network_name, scenario.actors, scenario.simulations)
        for scenario in scenarios
        if scenario.games >= scenario.actors * 2
    }
    missing_configuration = sorted(required - configured)
    if missing_configuration:
        reasons.append("full matrix configuration is incomplete")

    results_by_id: dict[str, list[Mapping[str, object]]] = {}
    for result in results:
        results_by_id.setdefault(str(result.get("scenarioId")), []).append(result)
    missing_results: list[str] = []
    invalid_results: list[str] = []
    for scenario in scenarios:
        matching = results_by_id.get(scenario.scenario_id, [])
        if not matching:
            missing_results.append(scenario.scenario_id)
            continue
        if len(matching) != 1:
            invalid_results.append(scenario.scenario_id)
            continue
        result = matching[0]
        network = result.get("network")
        result_matches = (
            result.get("status") == "completed"
            and isinstance(network, Mapping)
            and network.get("name") == scenario.network_name
            and network.get("inputChannels") == PHASE4_INPUT_CHANNELS
            and network.get("channels") == scenario.channels
            and network.get("residualBlocks") == scenario.blocks
            and result.get("actors") == scenario.actors
            and result.get("simulations") == scenario.simulations
            and result.get("games") == scenario.games
            and result.get("seed") == scenario.seed
            and _has_decision_metrics(result)
        )
        if not result_matches:
            invalid_results.append(scenario.scenario_id)
    if missing_results:
        reasons.append("one or more configured scenarios have no completed result")
    if invalid_results:
        reasons.append("one or more scenario results do not match their manifest")

    unexpected_results = sorted(
        scenario_id
        for scenario_id in results_by_id
        if scenario_id not in {scenario.scenario_id for scenario in scenarios}
    )
    if unexpected_results:
        reasons.append("report contains scenarios not present in its manifest")

    eligible = profile == "full" and not reasons
    return {
        "eligible": eligible,
        "reasons": reasons,
        "requiredMatrix": {
            "networks": [item[0] for item in FULL_NETWORKS],
            "actors": list(FULL_ACTOR_COUNTS),
            "simulations": list(FULL_SIMULATION_COUNTS),
            "minimumGamesPerScenario": "2 * actors",
            "scenarioCount": len(required),
        },
        "missingConfiguration": [list(item) for item in missing_configuration],
        "missingResults": missing_results,
        "invalidResults": invalid_results,
        "unexpectedResults": unexpected_results,
    }


def write_report(path: str | Path, report: Mapping[str, object]) -> Path:
    core = dict(report)
    core.pop("reportSha256", None)
    payload = {
        **core,
        "reportSha256": sha256_bytes(canonical_json_bytes(core)),
    }
    return atomic_replace_bytes(path, canonical_json_bytes(payload))


def _network_config(scenario: CalibrationScenario) -> NetworkConfig:
    return NetworkConfig(
        input_channels=PHASE4_INPUT_CHANNELS,
        channels=scenario.channels,
        blocks=scenario.blocks,
        se_reduction=8,
        policy_channels=32,
        value_channels=32,
        value_hidden=256,
    )


def build_calibration_models(
    scenarios: Sequence[CalibrationScenario],
    *,
    seed: int,
) -> dict[str, PolicyValueNetwork]:
    """Build exactly the unique networks declared by this calibration manifest."""

    models: dict[str, PolicyValueNetwork] = {}
    specifications: dict[str, tuple[int, int]] = {}
    for scenario in scenarios:
        specification = (scenario.channels, scenario.blocks)
        existing = specifications.get(scenario.network_name)
        if existing is not None:
            if existing != specification:
                raise ValueError(
                    f"network {scenario.network_name} has inconsistent dimensions"
                )
            continue
        seed_everything(seed)
        models[scenario.network_name] = PolicyValueNetwork(
            _network_config(scenario)
        ).eval()
        specifications[scenario.network_name] = specification
    if not models:
        raise ValueError("calibration manifest must declare at least one network")
    return models


def _report_summary(report: Mapping[str, object], report_path: Path) -> dict[str, object]:
    usage = report.get("resourceUsage", {})
    scenario_summaries: list[dict[str, object]] = []
    raw_scenarios = report.get("scenarios", [])
    if isinstance(raw_scenarios, list):
        for scenario in raw_scenarios:
            if not isinstance(scenario, Mapping):
                continue
            evaluation = scenario.get("evaluation", {})
            batch = (
                evaluation.get("batchSize")
                if isinstance(evaluation, Mapping)
                else {}
            )
            utilization = scenario.get("gpuUtilization", {})
            gpu_percent = (
                utilization.get("gpuPercent")
                if isinstance(utilization, Mapping)
                else {}
            )
            memory = scenario.get("cudaMemory", {})
            scenario_summaries.append(
                {
                    "scenarioId": scenario.get("scenarioId"),
                    "gamesPerHour": scenario.get("gamesPerHour"),
                    "scheduledSimulationsPerSecond": scenario.get(
                        "scheduledSimulationsPerSecond"
                    ),
                    "batchP50": batch.get("p50") if isinstance(batch, Mapping) else None,
                    "batchP95": batch.get("p95") if isinstance(batch, Mapping) else None,
                    "gpuUtilizationP50": (
                        gpu_percent.get("p50")
                        if isinstance(gpu_percent, Mapping)
                        else None
                    ),
                    "gpuTelemetryError": (
                        utilization.get("error")
                        if isinstance(utilization, Mapping)
                        else "missing"
                    ),
                    "peakReservedBytes": (
                        memory.get("peakReservedBytes")
                        if isinstance(memory, Mapping)
                        else None
                    ),
                }
            )
    return {
        "status": report.get("status"),
        "profile": report.get("profile"),
        "decisionEligible": report.get("decisionEligible"),
        "completedScenarios": len(report.get("scenarios", [])),
        "calibrationGpuHours": (
            usage.get("calibrationGpuHours")
            if isinstance(usage, Mapping)
            else None
        ),
        "decisionReasons": (
            report.get("decisionEligibility", {}).get("reasons")
            if isinstance(report.get("decisionEligibility"), Mapping)
            else None
        ),
        "scenarios": scenario_summaries,
        "reportPath": str(report_path),
        "manifestPath": str(report_path.parent / "run-manifest.json"),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4 native-MCTS CUDA throughput calibration (no training)",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="quick is smoke-only; full runs the decision matrix",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--gpu-hours-used",
        type=float,
        default=0.0,
        help="GPU hours used outside the persisted Phase 4 calibration ledger",
    )
    parser.add_argument(
        "--full-games-per-actor",
        type=int,
        default=2,
        help="full reports require at least 2 so actor refill is exercised",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def _run_locked(
    args: argparse.Namespace,
    *,
    workspace: Path,
    output_root: Path,
    run_id: str,
    ledger: GpuBudgetLedger,
) -> int:
    scenarios = build_scenarios(
        args.profile,
        seed=args.seed,
        full_games_per_actor=args.full_games_per_actor,
    )
    prior_gpu_hours = ledger.record_external_hours(args.gpu_hours_used)
    preflight = collect_preflight(
        output_root,
        device_index=args.device_index,
        gpu_hours_used=prior_gpu_hours,
    )
    native_module = load_native_module()
    if not callable(getattr(native_module, "encode_states", None)):
        raise CalibrationPreflightError(
            "quoridor_native does not expose the Phase 4 batch encoder"
        )

    models = build_calibration_models(scenarios, seed=args.seed)

    run_dir = output_root / "calibration" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    budget = ResourceBudget(
        artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
        minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
        gpu_hours=GPU_HOUR_BUDGET,
        metadata={
            "phase": 4,
            "purpose": "throughput-calibration",
            "trainingAllowed": False,
            "checkpointWritesAllowed": False,
        },
    )
    manifest_config = {
        "profile": args.profile,
        "seed": args.seed,
        "deviceIndex": args.device_index,
        "externalGpuHoursFloor": ledger.external_hours_floor,
        "cumulativeGpuHoursBeforeCalibration": prior_gpu_hours,
        "fullGamesPerActor": args.full_games_per_actor,
        "scenarios": [scenario.to_payload() for scenario in scenarios],
    }
    manifest = build_run_manifest(
        workspace_root=workspace,
        config=manifest_config,
        budget=budget,
        run_id=run_id,
        metadata={
            "phase": 4,
            "kind": "native-self-play-throughput-calibration",
            "profileCanBecomeDecisionEligible": args.profile == "full",
        },
    )
    save_run_manifest(manifest, run_dir / "run-manifest.json")
    extension_path = Path(str(native_module.__file__)).resolve()
    preflight["nativeExtension"] = {
        "path": str(extension_path),
        "sha256": sha256_file(extension_path),
    }

    started_gpu_hours = ledger.begin(
        run_id,
        external_hours=args.gpu_hours_used,
    )
    if started_gpu_hours != prior_gpu_hours:
        raise RuntimeError("GPU ledger changed while the Phase 4 lock was held")
    calibration_started = time.perf_counter()
    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4-native-self-play-throughput-calibration",
        "runId": run_id,
        "createdAt": utc_timestamp(),
        "status": "running",
        "profile": args.profile,
        "decisionEligible": False,
        "decisionEligibility": decision_eligibility(
            args.profile, scenarios, (), status="running"
        ),
        "trainingPerformed": False,
        "checkpointsWritten": 0,
        "runManifestSha256": manifest.sha256,
        "resourceBudget": budget.to_payload(),
        "preflight": preflight,
        "configuration": manifest_config,
        "scenarios": [],
        "activeScenario": None,
        "resourceUsage": {
            "calibrationGpuHours": 0.0,
            "totalGpuHours": prior_gpu_hours,
        },
    }
    write_report(report_path, report)

    exit_code = 0
    last_ledger_update = calibration_started
    last_disk_check = calibration_started

    def progress_guard(_moves: int, _games: int) -> None:
        nonlocal last_ledger_update, last_disk_check
        now = time.perf_counter()
        elapsed = now - calibration_started
        total_hours = prior_gpu_hours + elapsed / 3600.0
        if total_hours >= GPU_HOUR_BUDGET:
            raise CalibrationBudgetExceeded("72 GPU-hour budget reached")
        active = report.get("activeScenario")
        if isinstance(active, dict):
            active["movesCompleted"] = _moves
            active["gamesCompleted"] = _games
            active["updatedAt"] = utc_timestamp()
        if now - last_ledger_update >= LEDGER_HEARTBEAT_SECONDS:
            ledger.update(run_id, elapsed_seconds=elapsed, status="running")
            write_report(report_path, report)
            last_ledger_update = now
        if now - last_disk_check >= DISK_CHECK_INTERVAL_SECONDS:
            assert_runtime_budget(
                output_root,
                gpu_hours_used=prior_gpu_hours,
                calibration_started=calibration_started,
            )
            last_disk_check = now

    try:
        device = torch.device(f"cuda:{args.device_index}")
        for scenario in scenarios:
            report["activeScenario"] = {
                "scenarioId": scenario.scenario_id,
                "movesCompleted": 0,
                "gamesCompleted": 0,
                "startedAt": utc_timestamp(),
                "updatedAt": utc_timestamp(),
            }
            write_report(report_path, report)
            report["resourceUsage"] = assert_runtime_budget(
                output_root,
                gpu_hours_used=prior_gpu_hours,
                calibration_started=calibration_started,
            )
            model = models[scenario.network_name]
            evaluator = TorchEvaluator(model, device=device, use_amp=True)
            evaluator.enable_native_encoder(native_module)
            try:
                result = run_scenario(
                    scenario,
                    evaluator,
                    search_factory=NativeMCTS,
                    replay_root=run_dir / "replay" / scenario.scenario_id,
                    replay_max_bytes=scenario_replay_budget(output_root),
                    model_id=f"{run_id}:{scenario.network_name}",
                    progress_callback=progress_guard,
                )
                report["scenarios"].append(result)
                report["activeScenario"] = None
            finally:
                del evaluator
                model.to("cpu")
                torch.cuda.empty_cache()
            report["resourceUsage"] = assert_runtime_budget(
                output_root,
                gpu_hours_used=prior_gpu_hours,
                calibration_started=calibration_started,
            )
            elapsed = time.perf_counter() - calibration_started
            ledger.update(run_id, elapsed_seconds=elapsed, status="running")
            last_ledger_update = time.perf_counter()
            report["decisionEligibility"] = decision_eligibility(
                args.profile,
                scenarios,
                report["scenarios"],
                status="running",
            )
            write_report(report_path, report)

        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = {
            "type": "KeyboardInterrupt",
            "message": "calibration interrupted by user",
        }
        exit_code = 130
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        exit_code = 1
    finally:
        models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_seconds = time.perf_counter() - calibration_started
    report["completedAt"] = utc_timestamp()
    report["resourceUsage"] = {
        "calibrationGpuHours": elapsed_seconds / 3600.0,
        "totalGpuHours": prior_gpu_hours + elapsed_seconds / 3600.0,
        "remainingGpuHours": max(
            0.0,
            GPU_HOUR_BUDGET - prior_gpu_hours - elapsed_seconds / 3600.0,
        ),
        "artifactBytesBeforeFinalReport": _directory_size(output_root),
        "freeDiskBytes": int(shutil.disk_usage(_existing_ancestor(output_root)).free),
    }
    final_usage = report["resourceUsage"]
    if (
        report["status"] == "completed"
        and (
            final_usage["totalGpuHours"] >= GPU_HOUR_BUDGET
            or final_usage["artifactBytesBeforeFinalReport"]
            >= DEFAULT_ARTIFACT_LIMIT_BYTES
            or final_usage["freeDiskBytes"] < DEFAULT_MINIMUM_FREE_DISK_BYTES
        )
    ):
        report["status"] = "failed"
        report["error"] = {
            "type": "CalibrationBudgetExceeded",
            "message": "a Phase 4 resource stop line was reached",
        }
        exit_code = 1
    ledger.update(
        run_id,
        elapsed_seconds=elapsed_seconds,
        status=str(report["status"]),
    )
    eligibility = decision_eligibility(
        args.profile,
        scenarios,
        report["scenarios"],
        status=str(report["status"]),
    )
    report["decisionEligibility"] = eligibility
    report["decisionEligible"] = eligibility["eligible"]
    write_report(report_path, report)
    print(canonical_json_bytes(_report_summary(report, report_path)).decode(), end="")
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    workspace = Path(__file__).resolve().parents[2]
    output_root = (workspace / "training" / "runs" / "phase4").resolve()
    try:
        output_root.relative_to(workspace)
    except ValueError as error:
        raise CalibrationPreflightError(
            "output_root must remain inside the local workspace"
        ) from error
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_id = f"phase4-calibration-{timestamp}-{os.getpid()}"

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_for_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_for_sigterm)
    try:
        with RunLock(output_root / GPU_LOCK_NAME, run_id=run_id):
            ledger = GpuBudgetLedger(output_root / GPU_LEDGER_NAME)
            return _run_locked(
                args,
                workspace=workspace,
                output_root=output_root,
                run_id=run_id,
                ledger=ledger,
            )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "GPU_HOUR_BUDGET",
    "FULL_NETWORKS",
    "FULL_ACTOR_COUNTS",
    "FULL_SIMULATION_COUNTS",
    "CalibrationPreflightError",
    "CalibrationBudgetExceeded",
    "GpuBudgetLedger",
    "NvidiaSmiProfiler",
    "CalibrationScenario",
    "EvaluationProfiler",
    "build_scenarios",
    "build_calibration_models",
    "collect_preflight",
    "assert_runtime_budget",
    "scenario_replay_budget",
    "run_scenario",
    "decision_eligibility",
    "write_report",
    "create_parser",
    "main",
]
