"""CPU-only launch preparation and provenance gates for Phase 4F recovery."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4F_RECOVERY_STAGE
from .phase4_execution import (
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_training import validate_phase4_config
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ImmutableArtifactExistsError,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .storage import directory_size


SCHEMA_VERSION = 1
CONTROL_ROOT = Path("training/runs/phase4")
F5_REPORT = CONTROL_ROOT / "phase4f5/rebaseline/report.json"
F5_RECEIPT = CONTROL_ROOT / "phase4f5/rebaseline/freeze-receipt.json"
FORMAL_CONFIG = Path("training/configs/phase4f-recovery-v2.json")
FORMAL_EXECUTION = Path(
    "training/configs/phase4f-recovery-execution-frozen-v1.json"
)
PREPARATION = Path("training/configs/phase4f6-recovery-prepared-v1.json")
FORMAL_RUN_DIR = Path("training/runs/phase4/phase4f/recovery-v2")

F5_RUN_ID = "phase4f5-rebaseline-v4"
FORMAL_PURPOSE = "phase4f-strength-recovery-training"
FORMAL_MAXIMUM_GENERATIONS = 16
RECOVERY_GPU_HOURS = 14.0
DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS = 3.0
FRESH_FINAL_RESERVE_GPU_HOURS = 9.5


class Phase4F6RecoveryError(RuntimeError):
    """Raised when formal recovery launch evidence is not trustworthy."""


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F6RecoveryError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F6RecoveryError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4F6RecoveryError(f"{label} is not canonical JSON")
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
        raise Phase4F6RecoveryError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F6RecoveryError(
                f"immutable Phase 4F6 artifact already differs: {path}"
            )
        return
    try:
        atomic_create_bytes(path, encoded)
    except ImmutableArtifactExistsError as error:
        raise Phase4F6RecoveryError(
            f"immutable Phase 4F6 artifact was created concurrently: {path}"
        ) from error


def _effective_config(
    workspace: Path, path: Path
) -> tuple[ExperimentConfig, str]:
    config = load_experiment_config(workspace / path)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return config, fingerprint_config(config).sha256


def _verify_referenced_self_hashed_file(
    workspace: Path,
    reference: Mapping[str, object],
    *,
    path_field: str,
    file_hash_field: str | None,
    self_hash_field: str,
    label: str,
    payload_self_hash_field: str | None = None,
) -> None:
    path_value = reference.get(path_field)
    expected_self_hash = reference.get(self_hash_field)
    if not isinstance(path_value, str) or not isinstance(expected_self_hash, str):
        raise Phase4F6RecoveryError(f"{label} reference is invalid")
    path = (workspace / path_value).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise Phase4F6RecoveryError(f"{label} escapes the workspace") from error
    payload = _load_canonical(path, label)
    if _verify_self_hash(
        payload, payload_self_hash_field or self_hash_field, label
    ) != expected_self_hash:
        raise Phase4F6RecoveryError(f"{label} self hash differs from F5")
    if file_hash_field is not None:
        expected_file_hash = reference.get(file_hash_field)
        if not isinstance(expected_file_hash, str) or sha256_file(path) != expected_file_hash:
            raise Phase4F6RecoveryError(f"{label} file hash differs from F5")


def verify_phase4f5_history(workspace_root: str | Path) -> dict[str, object]:
    """Verify immutable F5 completion without requiring its old source tree."""

    workspace = Path(workspace_root).resolve()
    report_path = workspace / F5_REPORT
    receipt_path = workspace / F5_RECEIPT
    execution_path = workspace / FORMAL_EXECUTION
    config_path = workspace / FORMAL_CONFIG

    report = _load_canonical(report_path, "Phase 4F5 final report")
    report_sha256 = _verify_self_hash(
        report, "reportSha256", "Phase 4F5 final report"
    )
    receipt = _load_canonical(receipt_path, "Phase 4F5 freeze receipt")
    receipt_sha256 = _verify_self_hash(
        receipt, "receiptSha256", "Phase 4F5 freeze receipt"
    )
    execution = load_phase4_execution_contract(execution_path)
    relative_config = load_experiment_config(config_path)
    approved_semantic_config_sha256 = fingerprint_config(relative_config).sha256
    config, effective_semantic_config_sha256 = _effective_config(
        workspace, FORMAL_CONFIG
    )
    validate_phase4_config(config, execution)

    approved = report.get("approvedFormalWorkload")
    usage = report.get("resourceUsage")
    recommendation = report.get("frozenBudgetRecommendation")
    source = report.get("source")
    if (
        report.get("schemaVersion") != SCHEMA_VERSION
        or report.get("kind") != "phase4f5-recovery-rebaseline-report"
        or report.get("status") != "completed"
        or report.get("decisionEligible") is not True
        or report.get("formalRecoveryTrainingAuthorized") is not True
        or report.get("trainingStarted") is not False
        or not isinstance(approved, Mapping)
        or not isinstance(usage, Mapping)
        or not isinstance(recommendation, Mapping)
        or not isinstance(source, Mapping)
        or source.get("sha256") is None
        or approved.get("path") != FORMAL_CONFIG.as_posix()
        or approved.get("fileSha256") != sha256_file(config_path)
        or approved.get("semanticConfigSha256")
        != approved_semantic_config_sha256
        or recommendation.get("recoveryTrainingGpuHours")
        != RECOVERY_GPU_HOURS
        or recommendation.get("developmentScreenReserveGpuHours")
        != DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS
        or recommendation.get("freshFinalReserveGpuHours")
        != FRESH_FINAL_RESERVE_GPU_HOURS
    ):
        raise Phase4F6RecoveryError("Phase 4F5 final report is not eligible")
    try:
        validate_sha256(source.get("sha256"), "Phase 4F5 source sha256")
        validate_sha256(
            report.get("nativeBackendSha256"), "Phase 4F5 native backend sha256"
        )
    except ValueError as error:
        raise Phase4F6RecoveryError("Phase 4F5 provenance is invalid") from error

    baseline_hours = usage.get("globalGpuHours")
    if (
        isinstance(baseline_hours, bool)
        or not isinstance(baseline_hours, (int, float))
        or float(baseline_hours) < 0
        or execution.stage != PHASE4F_RECOVERY_STAGE
        or execution.status != PHASE4_EXECUTION_STATUS_FROZEN
        or execution.purpose != FORMAL_PURPOSE
        or execution.formal_promotion is not False
        or execution.maximum_generations != FORMAL_MAXIMUM_GENERATIONS
        or execution.stage_budget.baseline_global_hours != float(baseline_hours)
        or execution.stage_budget.limit_hours != RECOVERY_GPU_HOURS
        or execution.stage_budget.baseline_run_id != F5_RUN_ID
        or execution.stage_budget.baseline_report_sha256 != report_sha256
    ):
        raise Phase4F6RecoveryError("formal recovery execution contract is invalid")

    if (
        receipt.get("schemaVersion") != SCHEMA_VERSION
        or receipt.get("kind")
        != "phase4f5-recovery-execution-freeze-receipt"
        or receipt.get("status") != "completed"
        or receipt.get("trainingStarted") is not False
        or receipt.get("reportPath") != F5_REPORT.as_posix()
        or receipt.get("reportFileSha256") != sha256_file(report_path)
        or receipt.get("reportSha256") != report_sha256
        or receipt.get("executionContractPath") != FORMAL_EXECUTION.as_posix()
        or receipt.get("executionContractFileSha256") != sha256_file(execution_path)
        or receipt.get("executionContractSha256") != execution.sha256
    ):
        raise Phase4F6RecoveryError("Phase 4F5 freeze receipt is invalid")

    memory = report.get("memorySmoke")
    pilot = report.get("pilot")
    if not isinstance(memory, Mapping) or not isinstance(pilot, Mapping):
        raise Phase4F6RecoveryError("Phase 4F5 runtime evidence is missing")
    _verify_referenced_self_hashed_file(
        workspace,
        memory,
        path_field="path",
        file_hash_field="fileSha256",
        self_hash_field="reportSha256",
        label="Phase 4F5 memory report",
    )
    _verify_referenced_self_hashed_file(
        workspace,
        pilot,
        path_field="reportPath",
        file_hash_field=None,
        self_hash_field="reportSha256",
        label="Phase 4F5 pilot report",
    )
    _verify_referenced_self_hashed_file(
        workspace,
        pilot,
        path_field="runManifestPath",
        file_hash_field=None,
        self_hash_field="runManifestSha256",
        payload_self_hash_field="manifestSha256",
        label="Phase 4F5 pilot run manifest",
    )

    return {
        "status": "verified",
        "runId": F5_RUN_ID,
        "reportPath": F5_REPORT.as_posix(),
        "reportSha256": report_sha256,
        "reportFileSha256": sha256_file(report_path),
        "receiptPath": F5_RECEIPT.as_posix(),
        "receiptSha256": receipt_sha256,
        "executionContractPath": FORMAL_EXECUTION.as_posix(),
        "executionContractSha256": execution.sha256,
        "formalConfigPath": FORMAL_CONFIG.as_posix(),
        "formalConfigFileSha256": sha256_file(config_path),
        "approvedSemanticConfigSha256": approved_semantic_config_sha256,
        "effectiveSemanticConfigSha256": effective_semantic_config_sha256,
        "historicalSourceSha256": source["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "baselineGlobalHours": float(baseline_hours),
        "limitHours": RECOVERY_GPU_HOURS,
    }


def _ledger_snapshot(path: Path) -> dict[str, object]:
    ledger = _load_canonical(path, "GPU budget ledger")
    ledger_sha256 = _verify_self_hash(ledger, "ledgerSha256", "GPU budget ledger")
    runtime = GpuBudgetLedger(path)
    if any(record.get("status") == "running" for record in runtime.runs.values()):
        raise Phase4F6RecoveryError("a running GPU ledger entry blocks preparation")
    return {
        "path": path.relative_to(path.parents[3]).as_posix(),
        "fileSha256": sha256_file(path),
        "ledgerSha256": ledger_sha256,
        "totalGpuHours": runtime.total_hours,
        "externalHoursFloor": runtime.external_hours_floor,
        "runs": {key: dict(value) for key, value in sorted(runtime.runs.items())},
    }


def _verify_ledger_append_only(
    workspace: Path,
    snapshot: Mapping[str, object],
    *,
    baseline_hours: float,
    maximum_hours: float,
) -> float:
    path_value = snapshot.get("path")
    stored_runs = snapshot.get("runs")
    prepared_total = snapshot.get("totalGpuHours")
    prepared_floor = snapshot.get("externalHoursFloor")
    if (
        not isinstance(path_value, str)
        or not isinstance(stored_runs, Mapping)
        or isinstance(prepared_total, bool)
        or not isinstance(prepared_total, (int, float))
        or isinstance(prepared_floor, bool)
        or not isinstance(prepared_floor, (int, float))
    ):
        raise Phase4F6RecoveryError("prepared GPU ledger snapshot is invalid")
    path = workspace / path_value
    current = GpuBudgetLedger(path)
    if any(record.get("status") == "running" for record in current.runs.values()):
        raise Phase4F6RecoveryError("a running GPU ledger entry blocks verification")
    if any(current.runs.get(key) != value for key, value in stored_runs.items()):
        raise Phase4F6RecoveryError("GPU ledger is not append-only from F6 preparation")
    if current.external_hours_floor < float(prepared_floor):
        raise Phase4F6RecoveryError("GPU ledger external-hours floor regressed")
    if not (
        baseline_hours
        <= float(prepared_total)
        <= current.total_hours
        <= maximum_hours
    ):
        raise Phase4F6RecoveryError("formal recovery GPU-hour envelope was exceeded")
    return current.total_hours


def _resource_preflight(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / CONTROL_ROOT)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4F6RecoveryError("Phase 4 artifacts reached the 8 GiB cap")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F6RecoveryError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def prepare_phase4f6_recovery(workspace_root: str | Path) -> dict[str, object]:
    """Freeze current CPU-side launch inputs without starting formal training."""

    workspace = Path(workspace_root).resolve()
    preparation_path = workspace / PREPARATION
    if preparation_path.exists():
        return verify_phase4f6_preparation(workspace)
    if (workspace / FORMAL_RUN_DIR).exists():
        raise Phase4F6RecoveryError(
            "formal recovery lineage exists before Phase 4F6 preparation"
        )

    history = verify_phase4f5_history(workspace)
    resources = _resource_preflight(workspace)
    ledger = _ledger_snapshot(workspace / CONTROL_ROOT / GPU_LEDGER_NAME)
    baseline = float(history["baselineGlobalHours"])
    total = float(ledger["totalGpuHours"])
    if not baseline <= total <= baseline + RECOVERY_GPU_HOURS:
        raise Phase4F6RecoveryError("formal recovery budget is already unavailable")
    if (
        total
        + RECOVERY_GPU_HOURS
        + DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS
        + FRESH_FINAL_RESERVE_GPU_HOURS
        > GPU_HOUR_BUDGET
    ):
        raise Phase4F6RecoveryError("the reviewed post-recovery reserves no longer fit")

    source = fingerprint_sources(workspace).to_payload()
    native_sha256 = loaded_native_backend_hash()
    if native_sha256 != history["nativeBackendSha256"]:
        raise Phase4F6RecoveryError(
            "native backend differs from the successful F5 memory/pilot evidence"
        )
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f6-formal-recovery-launch-preparation",
        "status": "prepared",
        "cpuOnly": True,
        "cudaAccessed": False,
        "launcherPrepared": True,
        "trainingStarted": False,
        "formalPromotion": False,
        "source": source,
        "nativeBackendSha256": native_sha256,
        "f5History": history,
        "formalWorkload": {
            "path": history["formalConfigPath"],
            "fileSha256": history["formalConfigFileSha256"],
            "approvedSemanticConfigSha256": history[
                "approvedSemanticConfigSha256"
            ],
            "effectiveSemanticConfigSha256": history[
                "effectiveSemanticConfigSha256"
            ],
            "runDir": FORMAL_RUN_DIR.as_posix(),
        },
        "executionContract": {
            "path": history["executionContractPath"],
            "fileSha256": sha256_file(workspace / FORMAL_EXECUTION),
            "contractSha256": history["executionContractSha256"],
            "maximumGenerations": FORMAL_MAXIMUM_GENERATIONS,
        },
        "gpuLedgerSnapshot": ledger,
        "resources": resources,
        "budget": {
            "globalLimitHours": GPU_HOUR_BUDGET,
            "baselineGlobalHours": baseline,
            "currentGlobalHours": total,
            "recoveryTrainingLimitHours": RECOVERY_GPU_HOURS,
            "developmentScreenReserveHours": DEVELOPMENT_SCREEN_RESERVE_GPU_HOURS,
            "freshFinalReserveHours": FRESH_FINAL_RESERVE_GPU_HOURS,
        },
        "runtimePolicy": {
            "currentSourceReplayInputGateRequired": True,
            "resumeFromCommittedCheckpoint": True,
            "defaultFirstTargetGenerations": 4,
            "maximumTargetGenerations": FORMAL_MAXIMUM_GENERATIONS,
        },
    }
    prepared = _self_hashed(core, "preparationSha256")
    _write_immutable(preparation_path, prepared)
    return prepared


def _load_preparation(workspace: Path) -> dict[str, object]:
    path = workspace / PREPARATION
    payload = _load_canonical(path, "Phase 4F6 preparation")
    _verify_self_hash(payload, "preparationSha256", "Phase 4F6 preparation")
    if (
        payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind")
        != "phase4f6-formal-recovery-launch-preparation"
        or payload.get("status") != "prepared"
        or payload.get("cpuOnly") is not True
        or payload.get("cudaAccessed") is not False
        or payload.get("launcherPrepared") is not True
        or payload.get("trainingStarted") is not False
        or payload.get("formalPromotion") is not False
    ):
        raise Phase4F6RecoveryError("Phase 4F6 preparation policy is invalid")
    return payload


def verify_phase4f6_preparation(
    workspace_root: str | Path,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    prepared = _load_preparation(workspace)
    history = verify_phase4f5_history(workspace)
    source = fingerprint_sources(workspace).to_payload()
    native_sha256 = loaded_native_backend_hash()
    formal = prepared.get("formalWorkload")
    execution_reference = prepared.get("executionContract")
    budget = prepared.get("budget")
    if (
        prepared.get("source") != source
        or prepared.get("nativeBackendSha256") != native_sha256
        or prepared.get("f5History") != history
        or not isinstance(formal, Mapping)
        or formal.get("path") != history["formalConfigPath"]
        or formal.get("fileSha256") != history["formalConfigFileSha256"]
        or formal.get("approvedSemanticConfigSha256")
        != history["approvedSemanticConfigSha256"]
        or formal.get("effectiveSemanticConfigSha256")
        != history["effectiveSemanticConfigSha256"]
        or formal.get("runDir") != FORMAL_RUN_DIR.as_posix()
        or not isinstance(execution_reference, Mapping)
        or execution_reference.get("path") != history["executionContractPath"]
        or execution_reference.get("fileSha256")
        != sha256_file(workspace / FORMAL_EXECUTION)
        or execution_reference.get("contractSha256")
        != history["executionContractSha256"]
        or not isinstance(budget, Mapping)
    ):
        raise Phase4F6RecoveryError("Phase 4F6 preparation inputs changed")
    hours = _verify_ledger_append_only(
        workspace,
        prepared.get("gpuLedgerSnapshot", {}),
        baseline_hours=float(history["baselineGlobalHours"]),
        maximum_hours=float(history["baselineGlobalHours"]) + RECOVERY_GPU_HOURS,
    )
    resources = _resource_preflight(workspace)
    return {
        "status": "verified",
        "preparationPath": PREPARATION.as_posix(),
        "preparationSha256": prepared["preparationSha256"],
        "sourceSha256": source["sha256"],
        "nativeBackendSha256": native_sha256,
        "semanticConfigSha256": history["effectiveSemanticConfigSha256"],
        "executionContractSha256": history["executionContractSha256"],
        "reportSha256": history["reportSha256"],
        "runId": F5_RUN_ID,
        "baselineGlobalHours": history["baselineGlobalHours"],
        "limitHours": RECOVERY_GPU_HOURS,
        "globalGpuHours": hours,
        "resources": resources,
        "trainingStarted": (workspace / FORMAL_RUN_DIR / "run-manifest.json").exists(),
    }


def verify_phase4f6_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    evidence = verify_phase4f6_preparation(workspace)
    expected_execution = load_phase4_execution_contract(workspace / FORMAL_EXECUTION)
    expected_config, semantic = _effective_config(workspace, FORMAL_CONFIG)
    if (
        execution_contract.to_payload() != expected_execution.to_payload()
        or execution_contract.sha256 != evidence["executionContractSha256"]
        or fingerprint_config(config).sha256 != semantic
        or fingerprint_config(config).sha256 != fingerprint_config(expected_config).sha256
    ):
        raise Phase4F6RecoveryError(
            "formal recovery execution or workload differs from F6 preparation"
        )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_phase4f6_recovery(args.workspace_root)
    else:
        result = verify_phase4f6_preparation(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


__all__ = [
    "FORMAL_CONFIG",
    "FORMAL_EXECUTION",
    "FORMAL_RUN_DIR",
    "PREPARATION",
    "Phase4F6RecoveryError",
    "prepare_phase4f6_recovery",
    "verify_phase4f5_history",
    "verify_phase4f6_preparation",
    "verify_phase4f6_training_baseline",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
