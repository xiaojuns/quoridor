"""Resumable CUDA rebaseline for the frozen Phase 4F recovery workload."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import (
    GPU_HOUR_BUDGET,
    GPU_LEDGER_NAME,
    GPU_LOCK_NAME,
    GpuBudgetLedger,
    NvidiaSmiProfiler,
)
from .config import ExperimentConfig, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import (
    PHASE4_CONTROL_RELATIVE,
    PHASE4_STAGE_LEDGER_NAME,
    PHASE4F5_STAGE,
    Phase4StageBudgetLedger,
)
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_SCHEMA_VERSION,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionBudget,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4f_recovery_freeze import (
    FREEZE_REPORT,
    FROZEN_CHALLENGE_CONTRACT,
    REBASELINE_PREPARATION,
    RECOVERY_CONFIG as PHASE4F4_RECOVERY_CONFIG,
)
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    GenerationReportStore,
    ResourceBudget,
    RunLock,
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
)
from .storage import directory_size
from .training_memory_smoke import (
    TrainingMemorySmokeContract,
    _build_training_components,
    run_training_steps,
    torch_memory_snapshot,
)


SCHEMA_VERSION = 1
CALIBRATION_GPU_HOURS = 2.0
RECOVERY_TRAINING_GPU_HOURS = 14.0
DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS = 3.0
FRESH_FINAL_RESERVE_GPU_HOURS = 9.5
CALIBRATION_RUN_ID = "phase4f5-rebaseline-v4"
CALIBRATION_EXECUTION_CONTRACT = Path(
    "training/configs/phase4f5-calibration-execution-frozen-v1.json"
)
FORMAL_RECOVERY_CONFIG = Path("training/configs/phase4f-recovery-v2.json")
PILOT_CONFIG = Path("training/configs/phase4f-recovery-pilot-v3.json")
PREVIOUS_PREPARATION_CONTRACT = Path(
    "training/configs/phase4f5-rebaseline-prepared-v3.json"
)
PREPARATION_CONTRACT = Path(
    "training/configs/phase4f5-rebaseline-prepared-v4.json"
)
FORMAL_EXECUTION_CONTRACT = Path(
    "training/configs/phase4f-recovery-execution-frozen-v1.json"
)
RUN_ROOT = Path("training/runs/phase4/phase4f5/rebaseline")
PILOT_RUN_ROOT = RUN_ROOT / "pilot-v3"
MEMORY_ROOT = RUN_ROOT / "memory-smoke"
FINAL_REPORT = RUN_ROOT / "report.json"
FREEZE_RECEIPT = RUN_ROOT / "freeze-receipt.json"


class Phase4F5RebaselineError(RuntimeError):
    """Raised when the recovery rebaseline is stale, incomplete, or unsafe."""


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F5RebaselineError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F5RebaselineError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4F5RebaselineError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    stored = payload.get(field)
    core = dict(payload)
    core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(
        canonical_json_bytes(core)
    ):
        raise Phase4F5RebaselineError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F5RebaselineError(f"immutable artifact differs: {path}")
        return
    atomic_create_bytes(path, encoded)


def _resource_preflight(workspace: Path) -> dict[str, int]:
    control = workspace / PHASE4_CONTROL_RELATIVE
    artifact_bytes = directory_size(control)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4F5RebaselineError("Phase 4 artifacts exceed the 8 GiB cap")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F5RebaselineError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _ledger_snapshot(path: Path) -> dict[str, object]:
    ledger = _load_canonical(path, "GPU budget ledger")
    ledger_sha256 = _verify_self_hash(ledger, "ledgerSha256", "GPU budget ledger")
    runtime = GpuBudgetLedger(path)
    if any(value.get("status") == "running" for value in runtime.runs.values()):
        raise Phase4F5RebaselineError("a running GPU ledger entry blocks preparation")
    return {
        "path": path.as_posix(),
        "fileSha256": sha256_file(path),
        "ledgerSha256": ledger_sha256,
        "totalGpuHours": runtime.total_hours,
        "externalHoursFloor": runtime.external_hours_floor,
        "runs": {key: dict(value) for key, value in sorted(runtime.runs.items())},
    }


def _verify_ledger_append_only(
    workspace: Path, snapshot: Mapping[str, object], *, maximum_hours: float
) -> float:
    path_value = snapshot.get("path")
    if not isinstance(path_value, str):
        raise Phase4F5RebaselineError("prepared GPU ledger path is invalid")
    path = Path(path_value)
    if not path.is_absolute():
        path = workspace / path
    current = GpuBudgetLedger(path)
    if any(value.get("status") == "running" for value in current.runs.values()):
        raise Phase4F5RebaselineError("a running GPU ledger entry blocks verification")
    stored_runs = snapshot.get("runs")
    baseline = snapshot.get("totalGpuHours")
    if (
        not isinstance(stored_runs, Mapping)
        or isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
    ):
        raise Phase4F5RebaselineError("prepared GPU ledger snapshot is invalid")
    if any(current.runs.get(key) != value for key, value in stored_runs.items()):
        raise Phase4F5RebaselineError("GPU ledger is not append-only from preparation")
    if current.total_hours < float(baseline) or current.total_hours > maximum_hours:
        raise Phase4F5RebaselineError("Phase 4F5 GPU-hour envelope was exceeded")
    return current.total_hours


def _verify_phase4f4_history(workspace: Path) -> dict[str, object]:
    report_path = workspace / FREEZE_REPORT
    report = _load_canonical(report_path, "Phase 4F4 freeze report")
    report_sha256 = _verify_self_hash(
        report, "reportSha256", "Phase 4F4 freeze report"
    )
    challenge = report.get("challengeContract")
    dataset = report.get("dataset")
    workload = report.get("workload")
    if (
        report.get("status") != "completed"
        or report.get("cpuOnly") is not True
        or report.get("gpuAccessed") is not False
        or report.get("trainingAuthorized") is not False
        or not isinstance(challenge, Mapping)
        or challenge.get("path") != FROZEN_CHALLENGE_CONTRACT.as_posix()
        or challenge.get("fileSha256")
        != sha256_file(workspace / FROZEN_CHALLENGE_CONTRACT)
        or not isinstance(dataset, Mapping)
        or dataset.get("datasetManifestSha256")
        != sha256_file(workspace / str(dataset.get("datasetManifestPath")))
        or not isinstance(workload, Mapping)
        or workload.get("path") != PHASE4F4_RECOVERY_CONFIG.as_posix()
        or workload.get("fileSha256")
        != sha256_file(workspace / PHASE4F4_RECOVERY_CONFIG)
    ):
        raise Phase4F5RebaselineError("Phase 4F4 historical evidence changed")
    return {
        "path": FREEZE_REPORT.as_posix(),
        "fileSha256": sha256_file(report_path),
        "reportSha256": report_sha256,
        "challengeContractSha256": challenge["contractSha256"],
        "datasetFingerprint": dataset["datasetFingerprint"],
        "recoveryConfigSemanticSha256": workload["semanticConfigSha256"],
    }


def _formal_payload(workspace: Path) -> dict[str, object]:
    historical_path = workspace / PHASE4F4_RECOVERY_CONFIG
    formal = _load_canonical(historical_path, "historical recovery workload")
    formal["name"] = "phase4f-recovery-v2"
    formal["runDir"] = "training/runs/phase4/phase4f/recovery-v2"
    initialization = formal.get("initialization")
    if not isinstance(initialization, Mapping):
        raise Phase4F5RebaselineError("recovery initialization is invalid")
    formal["initialization"] = {
        **initialization,
        "expectedBaseModelSha256": (
            "87e87ec6f4f77b6156d56e095b1c23685d10d33cf14038ca4623b8a89d0b0120"
        ),
    }
    return formal


def _pilot_payload(workspace: Path) -> dict[str, object]:
    formal = _formal_payload(workspace)
    pilot = dict(formal)
    pilot["name"] = "phase4f5-recovery-pilot-v3"
    pilot["runDir"] = PILOT_RUN_ROOT.as_posix()
    return pilot


def _effective_config_sha256(config: ExperimentConfig, workspace: Path) -> str:
    effective = config
    if not effective.run_dir.is_absolute():
        effective = replace(
            effective,
            run_dir=(workspace / effective.run_dir).resolve(),
        )
    return fingerprint_config(effective).sha256


def _calibration_execution_contract(
    baseline_hours: float, freeze_report_sha256: str
) -> Phase4ExecutionContract:
    return Phase4ExecutionContract(
        schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
        stage=PHASE4F5_STAGE,
        status=PHASE4_EXECUTION_STATUS_FROZEN,
        purpose="phase4f-recovery-cuda-rebaseline",
        formal_promotion=False,
        maximum_generations=1,
        stage_budget=Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=baseline_hours,
            limit_hours=CALIBRATION_GPU_HOURS,
            baseline_run_id="phase4f4-recovery-freeze",
            baseline_report_sha256=freeze_report_sha256,
        ),
    )


def prepare_phase4f5_rebaseline(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    _resource_preflight(workspace)
    history = _verify_phase4f4_history(workspace)
    ledger_path = workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    ledger = _ledger_snapshot(ledger_path)
    current_hours = float(ledger["totalGpuHours"])
    execution_path = workspace / CALIBRATION_EXECUTION_CONTRACT
    previous_path = workspace / PREVIOUS_PREPARATION_CONTRACT
    previous: dict[str, object] | None = None
    if previous_path.exists():
        previous = _load_canonical(previous_path, "Phase 4F5 v3 preparation")
        previous_sha256 = _verify_self_hash(
            previous, "preparationSha256", "Phase 4F5 v3 preparation"
        )
        execution = load_phase4_execution_contract(execution_path)
        if (
            execution.stage != PHASE4F5_STAGE
            or execution.maximum_generations != 1
            or execution.formal_promotion is not False
            or execution.stage_budget.limit_hours != CALIBRATION_GPU_HOURS
            or execution.stage_budget.baseline_report_sha256
            != history["reportSha256"]
        ):
            raise Phase4F5RebaselineError(
                "existing Phase 4F5 calibration contract is invalid"
            )
    else:
        previous_sha256 = None
        execution = _calibration_execution_contract(
            current_hours, str(history["reportSha256"])
        )
        _write_immutable(execution_path, execution.to_payload())
    baseline = float(execution.stage_budget.baseline_global_hours)
    maximum = baseline + CALIBRATION_GPU_HOURS
    if current_hours > maximum:
        raise Phase4F5RebaselineError("Phase 4F5 calibration budget is exhausted")
    if maximum + (
        RECOVERY_TRAINING_GPU_HOURS
        + DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS
        + FRESH_FINAL_RESERVE_GPU_HOURS
    ) > GPU_HOUR_BUDGET:
        raise Phase4F5RebaselineError(
            "insufficient global budget for calibration and frozen recovery reserves"
        )
    formal_payload = _formal_payload(workspace)
    formal_path = workspace / FORMAL_RECOVERY_CONFIG
    _write_immutable(formal_path, formal_payload)
    pilot_payload = _pilot_payload(workspace)
    pilot_path = workspace / PILOT_CONFIG
    _write_immutable(pilot_path, pilot_payload)
    pilot_config = load_experiment_config(pilot_path)
    pilot_semantic = fingerprint_config(pilot_config).sha256
    pilot_effective_semantic = _effective_config_sha256(pilot_config, workspace)
    source = fingerprint_sources(workspace).to_payload()
    native_sha256 = loaded_native_backend_hash()
    old_rebaseline = _load_canonical(
        workspace / REBASELINE_PREPARATION, "Phase 4F4 rebaseline preparation"
    )
    old_rebaseline_sha256 = _verify_self_hash(
        old_rebaseline,
        "rebaselineSha256",
        "Phase 4F4 rebaseline preparation",
    )
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f5-recovery-rebaseline-preparation",
        "status": "prepared",
        "trainingAllowed": False,
        "formalRecoveryTrainingAllowed": False,
        "source": source,
        "nativeBackendSha256": native_sha256,
        "phase4f4": history,
        "supersedesRebaselinePreparationSha256": old_rebaseline_sha256,
        "supersedesPhase4f5Preparation": (
            {
                "path": PREVIOUS_PREPARATION_CONTRACT.as_posix(),
                "fileSha256": sha256_file(previous_path),
                "preparationSha256": previous_sha256,
                "reason": (
                    "checkpoint schema omitted champion initialization provenance"
                ),
            }
            if previous is not None
            else None
        ),
        "gpuLedgerSnapshot": ledger,
        "calibrationBudget": {
            "baselineGlobalHours": baseline,
            "limitHours": CALIBRATION_GPU_HOURS,
            "maximumGlobalHours": maximum,
        },
        "calibrationExecutionContract": {
            "path": CALIBRATION_EXECUTION_CONTRACT.as_posix(),
            "fileSha256": sha256_file(execution_path),
            "executionContractSha256": execution.sha256,
        },
        "pilotWorkload": {
            "path": PILOT_CONFIG.as_posix(),
            "fileSha256": sha256_file(pilot_path),
            "semanticConfigSha256": pilot_semantic,
            "effectiveSemanticConfigSha256": pilot_effective_semantic,
            "targetGenerations": 1,
        },
        "formalWorkload": {
            "path": FORMAL_RECOVERY_CONFIG.as_posix(),
            "fileSha256": sha256_file(formal_path),
            "semanticConfigSha256": fingerprint_config(
                load_experiment_config(formal_path)
            ).sha256,
        },
        "requiredGates": [
            "current-source replay input gate",
            "192x16 batch-512 CUDA AMP memory smoke",
            "one complete resumable 128-game 512-simulation mixed-replay generation",
            "96 optimizer steps with the live mixed-input timing gate",
            "zero corrupt or illegal replay games",
            "at least 26.5 GPU-hours remain after calibration",
        ],
    }
    prepared = _self_hashed(core, "preparationSha256")
    _write_immutable(workspace / PREPARATION_CONTRACT, prepared)
    return prepared


def _load_preparation(workspace: Path) -> dict[str, object]:
    path = workspace / PREPARATION_CONTRACT
    payload = _load_canonical(path, "Phase 4F5 preparation")
    _verify_self_hash(payload, "preparationSha256", "Phase 4F5 preparation")
    source = fingerprint_sources(workspace).to_payload()
    native_sha256 = loaded_native_backend_hash()
    pilot = payload.get("pilotWorkload")
    formal = payload.get("formalWorkload")
    execution = payload.get("calibrationExecutionContract")
    if (
        payload.get("status") != "prepared"
        or payload.get("trainingAllowed") is not False
        or payload.get("formalRecoveryTrainingAllowed") is not False
        or payload.get("source") != source
        or payload.get("nativeBackendSha256") != native_sha256
        or not isinstance(pilot, Mapping)
        or pilot.get("fileSha256") != sha256_file(workspace / PILOT_CONFIG)
        or pilot.get("semanticConfigSha256")
        != fingerprint_config(load_experiment_config(workspace / PILOT_CONFIG)).sha256
        or pilot.get("effectiveSemanticConfigSha256")
        != _effective_config_sha256(
            load_experiment_config(workspace / PILOT_CONFIG), workspace
        )
        or not isinstance(formal, Mapping)
        or formal.get("fileSha256")
        != sha256_file(workspace / FORMAL_RECOVERY_CONFIG)
        or formal.get("semanticConfigSha256")
        != fingerprint_config(
            load_experiment_config(workspace / FORMAL_RECOVERY_CONFIG)
        ).sha256
        or not isinstance(execution, Mapping)
        or execution.get("fileSha256")
        != sha256_file(workspace / CALIBRATION_EXECUTION_CONTRACT)
    ):
        raise Phase4F5RebaselineError("Phase 4F5 preparation is stale")
    return payload


def verify_phase4f5_preparation(
    workspace_root: str | Path,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    prepared = _load_preparation(workspace)
    maximum = float(prepared["calibrationBudget"]["maximumGlobalHours"])
    global_hours = _verify_ledger_append_only(
        workspace,
        prepared["gpuLedgerSnapshot"],
        maximum_hours=maximum,
    )
    return {
        "status": "verified",
        "preparationPath": PREPARATION_CONTRACT.as_posix(),
        "preparationSha256": prepared["preparationSha256"],
        "baselineGlobalHours": prepared["calibrationBudget"][
            "baselineGlobalHours"
        ],
        "globalGpuHours": global_hours,
        "calibrationLimitHours": CALIBRATION_GPU_HOURS,
        "remainingCalibrationHours": maximum - global_hours,
    }


def _validate_execution(
    workspace: Path,
    prepared: Mapping[str, object],
    execution_contract: Phase4ExecutionContract,
) -> None:
    reference = prepared.get("calibrationExecutionContract")
    budget = prepared.get("calibrationBudget")
    if (
        not isinstance(reference, Mapping)
        or not isinstance(budget, Mapping)
        or execution_contract.stage != PHASE4F5_STAGE
        or execution_contract.sha256 != reference.get("executionContractSha256")
        or execution_contract.maximum_generations != 1
        or execution_contract.formal_promotion is not False
        or execution_contract.stage_budget.baseline_global_hours
        != budget.get("baselineGlobalHours")
        or execution_contract.stage_budget.limit_hours != budget.get("limitHours")
    ):
        raise Phase4F5RebaselineError("Phase 4F5 execution contract changed")


def _valid_memory_report(
    workspace: Path, prepared: Mapping[str, object]
) -> dict[str, object] | None:
    root = workspace / MEMORY_ROOT
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/report.json"), reverse=True)
    for path in candidates:
        try:
            report = _load_canonical(path, "Phase 4F5 memory report")
            _verify_self_hash(report, "reportSha256", "Phase 4F5 memory report")
            gates = report.get("gates")
            required_gates = {
                "workloadCompleted",
                "cudaAmp",
                "memoryCaptured",
                "telemetryCaptured",
                "budgetAvailable",
                "resourceLimits",
            }
            if (
                report.get("kind") == "phase4f5-cuda-amp-memory-smoke"
                and report.get("status") == "completed"
                and report.get("passed") is True
                and report.get("preparationSha256")
                == prepared.get("preparationSha256")
                and report.get("sourceSha256") == prepared["source"]["sha256"]
                and report.get("nativeBackendSha256")
                == prepared.get("nativeBackendSha256")
                and isinstance(gates, Mapping)
                and set(gates) == required_gates
                and all(value is True for value in gates.values())
            ):
                return {
                    "path": path.relative_to(workspace).as_posix(),
                    "fileSha256": sha256_file(path),
                    "reportSha256": report["reportSha256"],
                    "runId": report["runId"],
                    "peakAllocatedBytes": report["memory"]["peakAllocatedBytes"],
                    "peakReservedBytes": report["memory"]["peakReservedBytes"],
                }
        except (OSError, KeyError, TypeError, ValueError, Phase4F5RebaselineError):
            continue
    return None


def run_memory_smoke(
    workspace_root: str | Path,
    *,
    device_index: int,
    external_gpu_hours: float,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    prepared = _load_preparation(workspace)
    existing = _valid_memory_report(workspace, prepared)
    if existing is not None:
        return {"status": "completed", "passed": True, "reused": True, **existing}
    if not torch.cuda.is_available():
        raise Phase4F5RebaselineError("PyTorch CUDA is unavailable")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise Phase4F5RebaselineError("CUDA device index is invalid")
    execution = load_phase4_execution_contract(
        workspace / CALIBRATION_EXECUTION_CONTRACT
    )
    _validate_execution(workspace, prepared, execution)
    control = workspace / PHASE4_CONTROL_RELATIVE
    run_id = f"phase4f5-memory-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_dir = workspace / MEMORY_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with RunLock(control / GPU_LOCK_NAME, run_id=run_id):
        ledger = GpuBudgetLedger(control / GPU_LEDGER_NAME)
        baseline = float(prepared["calibrationBudget"]["baselineGlobalHours"])
        maximum = float(prepared["calibrationBudget"]["maximumGlobalHours"])
        ledger.record_external_hours(external_gpu_hours)
        if ledger.total_hours > maximum:
            raise Phase4F5RebaselineError("Phase 4F5 calibration budget is exhausted")
        stage_ledger = Phase4StageBudgetLedger(control / PHASE4_STAGE_LEDGER_NAME)
        stage_ledger.ensure_stage(
            PHASE4F5_STAGE,
            baseline,
            CALIBRATION_GPU_HOURS,
            baseline_run_id=execution.stage_budget.baseline_run_id,
            baseline_report_sha256=execution.stage_budget.baseline_report_sha256,
        )
        stage_ledger.assert_available(PHASE4F5_STAGE, ledger.total_hours)
        contract = TrainingMemorySmokeContract()
        manifest = build_run_manifest(
            workspace_root=workspace,
            config={
                "kind": "phase4f5-cuda-amp-memory-smoke",
                "deviceIndex": device_index,
                "preparationSha256": prepared["preparationSha256"],
                "contract": contract.to_payload(),
            },
            budget=ResourceBudget(
                artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
                minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
                gpu_hours=CALIBRATION_GPU_HOURS,
                maximum_generations=1,
                metadata={"stage": PHASE4F5_STAGE, "selfPlayAllowed": False},
            ),
            run_id=run_id,
            metadata={"kind": "phase4f5-cuda-amp-memory-smoke"},
        )
        save_run_manifest(manifest, run_dir / "run-manifest.json")
        before = ledger.begin(run_id, external_hours=external_gpu_hours)
        started = time.perf_counter()
        status = "failed"
        error: dict[str, str] | None = None
        workload: Mapping[str, object] | None = None
        memory: dict[str, int] = {}
        telemetry: Mapping[str, object] = {}
        model = trainer = batch = None
        profiler: NvidiaSmiProfiler | None = None
        try:
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device_index)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            profiler = NvidiaSmiProfiler(device_index)
            profiler.start()
            model, trainer, batch = _build_training_components(
                contract, device=device, seed=20260803
            )
            workload = run_training_steps(
                trainer,
                batch,
                warmup_steps=contract.warmup_steps,
                measured_steps=contract.measured_steps,
                require_cuda_amp=True,
                synchronize=lambda: torch.cuda.synchronize(device),
            )
            torch.cuda.synchronize(device)
            memory = torch_memory_snapshot(device)
            status = "completed"
        except BaseException as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
            if isinstance(failure, KeyboardInterrupt):
                status = "interrupted"
        finally:
            if profiler is not None:
                telemetry = profiler.stop()
            batch = trainer = model = None
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        final_hours = ledger.update(run_id, elapsed_seconds=elapsed, status=status)
        stage_ledger.usage(PHASE4F5_STAGE, final_hours)
        gates = {
            "workloadCompleted": status == "completed" and workload is not None,
            "cudaAmp": bool(
                workload
                and workload.get("ampEnabled") is True
                and workload.get("optimizerSteps") == contract.total_steps
            ),
            "memoryCaptured": memory.get("peakAllocatedBytes", 0) > 0
            and memory.get("peakReservedBytes", 0) > 0,
            "telemetryCaptured": telemetry.get("available") is True
            and telemetry.get("sampleCount", 0) >= 2,
            "budgetAvailable": final_hours <= maximum,
            "resourceLimits": _resource_preflight(workspace)["freeDiskBytes"]
            >= DEFAULT_MINIMUM_FREE_DISK_BYTES,
        }
        passed = status == "completed" and all(gates.values())
        if not passed and status == "completed":
            status = "failed"
            error = {
                "type": "Phase4F5MemoryGateError",
                "message": "one or more memory smoke gates failed",
            }
        report_core = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f5-cuda-amp-memory-smoke",
            "status": status,
            "passed": passed,
            "runId": run_id,
            "runManifestSha256": manifest.sha256,
            "preparationSha256": prepared["preparationSha256"],
            "sourceSha256": prepared["source"]["sha256"],
            "nativeBackendSha256": prepared["nativeBackendSha256"],
            "deviceIndex": device_index,
            "workload": workload,
            "memory": memory,
            "nvidiaSmi": telemetry,
            "elapsedSeconds": elapsed,
            "globalGpuHoursBefore": before,
            "globalGpuHoursAfter": final_hours,
            "gates": gates,
            "error": error,
        }
        report = _self_hashed(report_core, "reportSha256")
        _write_immutable(run_dir / "report.json", report)
        if not passed:
            raise Phase4F5RebaselineError(str(error))
        return {
            "status": status,
            "passed": passed,
            "reused": False,
            "path": (run_dir / "report.json").relative_to(workspace).as_posix(),
            "fileSha256": sha256_file(run_dir / "report.json"),
            "reportSha256": report["reportSha256"],
            "runId": run_id,
            "peakAllocatedBytes": memory["peakAllocatedBytes"],
            "peakReservedBytes": memory["peakReservedBytes"],
        }


def verify_phase4f5_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    prepared = _load_preparation(workspace)
    _validate_execution(workspace, prepared, execution_contract)
    pilot = prepared["pilotWorkload"]
    semantic = fingerprint_config(config).sha256
    if semantic != pilot["effectiveSemanticConfigSha256"]:
        raise Phase4F5RebaselineError("pilot workload is not approved")
    baseline = float(prepared["calibrationBudget"]["baselineGlobalHours"])
    maximum = float(prepared["calibrationBudget"]["maximumGlobalHours"])
    hours = _verify_ledger_append_only(
        workspace, prepared["gpuLedgerSnapshot"], maximum_hours=maximum
    )
    memory = _valid_memory_report(workspace, prepared)
    if memory is None:
        raise Phase4F5RebaselineError("current-source CUDA memory smoke is missing")
    return {
        "status": "verified",
        "runId": CALIBRATION_RUN_ID,
        "reportSha256": prepared["preparationSha256"],
        "sourceSha256": prepared["source"]["sha256"],
        "nativeBackendSha256": prepared["nativeBackendSha256"],
        "semanticConfigSha256": semantic,
        "baselineGlobalHours": baseline,
        "limitHours": CALIBRATION_GPU_HOURS,
        "globalGpuHours": hours,
        "memorySmoke": memory,
    }


def _pilot_evidence(
    workspace: Path, prepared: Mapping[str, object]
) -> dict[str, object]:
    run_root = workspace / PILOT_RUN_ROOT
    manifest = load_run_manifest(run_root / "run-manifest.json")
    store = GenerationReportStore(
        run_root / "reports", run_manifest_sha256=manifest.sha256
    )
    record = store.load(0)
    report = record.payload.get("report")
    generation_data = report.get("generationData") if isinstance(report, Mapping) else None
    checkpoint = report.get("checkpoint") if isinstance(report, Mapping) else None
    selfplay = generation_data.get("selfPlay") if isinstance(generation_data, Mapping) else None
    training = generation_data.get("training") if isinstance(generation_data, Mapping) else None
    timing = training.get("inputTiming") if isinstance(training, Mapping) else None
    challenge = training.get("challengeReplay") if isinstance(training, Mapping) else None
    metrics = training.get("metrics") if isinstance(training, Mapping) else None
    candidate_model = (
        checkpoint.get("candidateModel") if isinstance(checkpoint, Mapping) else None
    )
    timing_mean = timing.get("secondsMean") if isinstance(timing, Mapping) else None
    if (
        not isinstance(selfplay, Mapping)
        or selfplay.get("games") != 128
        or selfplay.get("simulations") != 512
        or not isinstance(training, Mapping)
        or training.get("batchSize") != 512
        or not isinstance(metrics, Mapping)
        or metrics.get("steps") != 96
        or not isinstance(timing, Mapping)
        or timing.get("initialGateCompleted") is not True
        or isinstance(timing_mean, bool)
        or not isinstance(timing_mean, (int, float))
        or not math.isfinite(float(timing_mean))
        or float(timing_mean) > 2.0
        or not isinstance(challenge, Mapping)
        or challenge.get("fraction") != 0.25
        or challenge.get("contractSha256")
        != prepared["phase4f4"]["challengeContractSha256"]
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("step") != 96
        or not isinstance(checkpoint.get("sha256"), str)
        or not isinstance(candidate_model, Mapping)
        or not isinstance(candidate_model.get("sha256"), str)
    ):
        raise Phase4F5RebaselineError("recovery pilot report failed its gates")
    return {
        "runManifestPath": (run_root / "run-manifest.json").relative_to(workspace).as_posix(),
        "runManifestSha256": manifest.sha256,
        "reportPath": record.path.relative_to(workspace).as_posix(),
        "reportSha256": record.sha256,
        "checkpointSha256": checkpoint["sha256"],
        "candidateModelSha256": candidate_model["sha256"],
        "selfPlayGames": selfplay["games"],
        "selfPlayMoves": selfplay["moves"],
        "optimizerSteps": metrics["steps"],
        "inputTiming": dict(timing),
        "challengeReplay": dict(challenge),
    }


def _load_completed_freeze(
    workspace: Path, prepared: Mapping[str, object]
) -> dict[str, object] | None:
    paths = (
        workspace / FINAL_REPORT,
        workspace / FORMAL_EXECUTION_CONTRACT,
        workspace / FREEZE_RECEIPT,
    )
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise Phase4F5RebaselineError("Phase 4F5 freeze artifacts are incomplete")
    report = _load_canonical(paths[0], "Phase 4F5 final report")
    report_sha256 = _verify_self_hash(
        report, "reportSha256", "Phase 4F5 final report"
    )
    execution = load_phase4_execution_contract(paths[1])
    receipt = _load_canonical(paths[2], "Phase 4F5 freeze receipt")
    receipt_sha256 = _verify_self_hash(
        receipt, "receiptSha256", "Phase 4F5 freeze receipt"
    )
    usage = report.get("resourceUsage")
    if (
        report.get("kind") != "phase4f5-recovery-rebaseline-report"
        or report.get("status") != "completed"
        or report.get("decisionEligible") is not True
        or report.get("formalRecoveryTrainingAuthorized") is not True
        or report.get("trainingStarted") is not False
        or report.get("preparationSha256") != prepared.get("preparationSha256")
        or report.get("source") != prepared.get("source")
        or report.get("nativeBackendSha256")
        != prepared.get("nativeBackendSha256")
        or not isinstance(usage, Mapping)
        or execution.stage != "phase4f-recovery"
        or execution.formal_promotion is not False
        or execution.maximum_generations != 16
        or execution.stage_budget.limit_hours != RECOVERY_TRAINING_GPU_HOURS
        or execution.stage_budget.baseline_global_hours
        != usage.get("globalGpuHours")
        or execution.stage_budget.baseline_report_sha256 != report_sha256
        or receipt.get("status") != "completed"
        or receipt.get("trainingStarted") is not False
        or receipt.get("reportPath") != FINAL_REPORT.as_posix()
        or receipt.get("reportFileSha256") != sha256_file(paths[0])
        or receipt.get("reportSha256") != report_sha256
        or receipt.get("executionContractPath")
        != FORMAL_EXECUTION_CONTRACT.as_posix()
        or receipt.get("executionContractFileSha256") != sha256_file(paths[1])
        or receipt.get("executionContractSha256") != execution.sha256
    ):
        raise Phase4F5RebaselineError("Phase 4F5 completed freeze is invalid")
    return {
        "status": "completed",
        "decisionEligible": True,
        "reportPath": FINAL_REPORT.as_posix(),
        "reportSha256": report_sha256,
        "executionContractPath": FORMAL_EXECUTION_CONTRACT.as_posix(),
        "executionContractSha256": execution.sha256,
        "receiptPath": FREEZE_RECEIPT.as_posix(),
        "receiptSha256": receipt_sha256,
        "globalGpuHours": usage["globalGpuHours"],
        "phase4f5GpuHours": usage["phase4f5GpuHours"],
        "formalRecoveryTrainingAuthorized": True,
        "trainingStarted": False,
        "reused": True,
    }


def finalize_phase4f5_rebaseline(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    prepared = _load_preparation(workspace)
    completed = _load_completed_freeze(workspace, prepared)
    if completed is not None:
        return completed
    execution = load_phase4_execution_contract(
        workspace / CALIBRATION_EXECUTION_CONTRACT
    )
    _validate_execution(workspace, prepared, execution)
    memory = _valid_memory_report(workspace, prepared)
    if memory is None:
        raise Phase4F5RebaselineError("valid memory smoke is missing")
    pilot = _pilot_evidence(workspace, prepared)
    maximum = float(prepared["calibrationBudget"]["maximumGlobalHours"])
    global_hours = _verify_ledger_append_only(
        workspace, prepared["gpuLedgerSnapshot"], maximum_hours=maximum
    )
    required_reserve = (
        RECOVERY_TRAINING_GPU_HOURS
        + DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS
        + FRESH_FINAL_RESERVE_GPU_HOURS
    )
    if GPU_HOUR_BUDGET - global_hours < required_reserve:
        raise Phase4F5RebaselineError("less than 26.5 GPU-hours remain after calibration")
    resources = _resource_preflight(workspace)
    report_core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f5-recovery-rebaseline-report",
        "status": "completed",
        "decisionEligible": True,
        "formalRecoveryTrainingAuthorized": True,
        "trainingStarted": False,
        "preparationSha256": prepared["preparationSha256"],
        "source": prepared["source"],
        "nativeBackendSha256": prepared["nativeBackendSha256"],
        "approvedFormalWorkload": prepared["formalWorkload"],
        "memorySmoke": memory,
        "pilot": pilot,
        "resourceUsage": {
            **resources,
            "baselineGlobalGpuHours": prepared["calibrationBudget"]["baselineGlobalHours"],
            "globalGpuHours": global_hours,
            "phase4f5GpuHours": global_hours
            - float(prepared["calibrationBudget"]["baselineGlobalHours"]),
            "calibrationLimitHours": CALIBRATION_GPU_HOURS,
        },
        "frozenBudgetRecommendation": {
            "recoveryTrainingGpuHours": RECOVERY_TRAINING_GPU_HOURS,
            "developmentScreenReserveGpuHours": DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS,
            "freshFinalReserveGpuHours": FRESH_FINAL_RESERVE_GPU_HOURS,
            "globalGpuHoursRemaining": GPU_HOUR_BUDGET - global_hours,
        },
    }
    report = _self_hashed(report_core, "reportSha256")
    report_path = workspace / FINAL_REPORT
    _write_immutable(report_path, report)
    formal_execution = Phase4ExecutionContract(
        schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
        stage="phase4f-recovery",
        status=PHASE4_EXECUTION_STATUS_FROZEN,
        purpose="phase4f-strength-recovery-training",
        formal_promotion=False,
        maximum_generations=16,
        stage_budget=Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=global_hours,
            limit_hours=RECOVERY_TRAINING_GPU_HOURS,
            baseline_run_id=CALIBRATION_RUN_ID,
            baseline_report_sha256=report["reportSha256"],
        ),
    )
    execution_path = workspace / FORMAL_EXECUTION_CONTRACT
    _write_immutable(execution_path, formal_execution.to_payload())
    receipt_core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f5-recovery-execution-freeze-receipt",
        "status": "completed",
        "trainingStarted": False,
        "reportPath": FINAL_REPORT.as_posix(),
        "reportFileSha256": sha256_file(report_path),
        "reportSha256": report["reportSha256"],
        "executionContractPath": FORMAL_EXECUTION_CONTRACT.as_posix(),
        "executionContractFileSha256": sha256_file(execution_path),
        "executionContractSha256": formal_execution.sha256,
    }
    receipt = _self_hashed(receipt_core, "receiptSha256")
    _write_immutable(workspace / FREEZE_RECEIPT, receipt)
    return {
        "status": "completed",
        "decisionEligible": True,
        "reportPath": FINAL_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "executionContractPath": FORMAL_EXECUTION_CONTRACT.as_posix(),
        "executionContractSha256": formal_execution.sha256,
        "receiptPath": FREEZE_RECEIPT.as_posix(),
        "receiptSha256": receipt["receiptSha256"],
        "globalGpuHours": global_hours,
        "phase4f5GpuHours": report["resourceUsage"]["phase4f5GpuHours"],
        "formalRecoveryTrainingAuthorized": True,
        "trainingStarted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify-preparation", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--workspace-root", type=Path, default=Path.cwd())
    memory = subparsers.add_parser("memory-smoke")
    memory.add_argument("--workspace-root", type=Path, default=Path.cwd())
    memory.add_argument("--device-index", type=int, default=0)
    memory.add_argument("--gpu-hours-used", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f5_rebaseline(args.workspace_root)
        summary = {
            "status": payload["status"],
            "preparationPath": PREPARATION_CONTRACT.as_posix(),
            "preparationSha256": payload["preparationSha256"],
            "baselineGlobalHours": payload["calibrationBudget"]["baselineGlobalHours"],
            "calibrationLimitHours": payload["calibrationBudget"]["limitHours"],
        }
    elif args.command == "verify-preparation":
        summary = verify_phase4f5_preparation(args.workspace_root)
    elif args.command == "memory-smoke":
        summary = run_memory_smoke(
            args.workspace_root,
            device_index=args.device_index,
            external_gpu_hours=args.gpu_hours_used,
        )
    else:
        summary = finalize_phase4f5_rebaseline(args.workspace_root)
    print(canonical_json_bytes(summary).decode("ascii"), flush=True)
    return 0


__all__ = [
    "Phase4F5RebaselineError",
    "prepare_phase4f5_rebaseline",
    "verify_phase4f5_preparation",
    "run_memory_smoke",
    "verify_phase4f5_training_baseline",
    "finalize_phase4f5_rebaseline",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
