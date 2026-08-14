"""Freeze and verify the recoverable Phase 4F36 Stage A2 pilot."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F36_PILOT_STAGE
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_training import validate_phase4_config
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
F35_REPORT = Path(
    "training/runs/phase4/phase4f/f35-stage-a2-preparation-v1/preparation-report.json"
)
F35_REPORT_SHA256 = "21988e68132d6ca303f05b7d77ab2b3fa5b376bb5bbfdfcf5fff8c2922a6795c"
F35_SOURCE_GIT_COMMIT = "3d7fd8736b5075012e1e40cfa302024471558522"
F35_CONFIG = Path("training/configs/phase4f35-stage-a2-pilot-v1.json")
F35_EXECUTION = Path(
    "training/configs/phase4f35-stage-a2-pilot-execution-frozen-v1.json"
)
FAILED_SMOKE = Path(
    "training/runs/phase4/training-memory-smoke/"
    "phase4d-training-memory-smoke-20260814-122855-287076/report.json"
)
FAILED_SMOKE_REPORT_SHA256 = (
    "5e6390de3f936eeca2fda86541fda607c11aded15c93797f8b7fc894e8c8c061"
)
ROOT = Path("training/runs/phase4/phase4f/f36-stage-a2-recovery-v1")
REPORT = ROOT / "preparation-report.json"
CONFIG = Path("training/configs/phase4f36-stage-a2-pilot-v2.json")
EXECUTION = Path(
    "training/configs/phase4f36-stage-a2-pilot-execution-frozen-v2.json"
)
RUN_DIR = Path("training/runs/phase4/phase4f/stage-a2-pilot-v2")
ARTIFACT_RESERVE_BYTES = 1250 * 1024**2


class Phase4F36Error(RuntimeError):
    """Raised when the F36 recovery contract is stale or inconsistent."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F36Error(f"{label} is unreadable") from error
    if path.is_symlink() or not isinstance(payload, Mapping):
        raise Phase4F36Error(f"{label} is invalid")
    if raw != canonical_json_bytes(payload):
        raise Phase4F36Error(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(
        canonical_json_bytes(core)
    ):
        raise Phase4F36Error(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(
    path: Path, payload: Mapping[str, object], label: str
) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise Phase4F36Error(f"existing {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, encoded)


def _verify_f35_history(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / F35_REPORT, "F35 report")
    if (
        _verify_self_hash(report, "reportSha256", "F35 report")
        != F35_REPORT_SHA256
        or report.get("status") != "prepared"
        or report.get("cudaPilotAuthorized") is not False
    ):
        raise Phase4F36Error("F35 report is not the frozen preparation")
    source = report.get("source")
    files = source.get("files") if isinstance(source, Mapping) else None
    if not isinstance(files, list) or source.get("fileCount") != len(files):
        raise Phase4F36Error("F35 source inventory is invalid")
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping):
            raise Phase4F36Error("F35 source record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise Phase4F36Error("F35 source path is invalid")
        path = workspace / relative
        current_matches = (
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == record.get("bytes")
            and sha256_file(path) == record.get("sha256")
        )
        if current_matches:
            file_bytes = path.stat().st_size
        else:
            try:
                archived = subprocess.run(
                    ["git", "show", f"{F35_SOURCE_GIT_COMMIT}:{relative}"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise Phase4F36Error(
                    f"historical F35 source is unavailable: {relative}"
                ) from error
            if (
                len(archived) != record.get("bytes")
                or sha256_bytes(archived) != record.get("sha256")
            ):
                raise Phase4F36Error(
                    f"historical F35 Git source differs: {relative}"
                )
            file_bytes = len(archived)
        total_bytes += file_bytes
    if total_bytes != source.get("bytes"):
        raise Phase4F36Error("F35 source byte count changed")
    for path, key in (
        (F35_CONFIG, "configFileSha256"),
        (F35_EXECUTION, "executionFileSha256"),
    ):
        if sha256_file(workspace / path) != report["pilot"][key]:
            raise Phase4F36Error("F35 frozen pilot artifact changed")
    return report


def _verify_failed_smoke(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / FAILED_SMOKE, "F36 initial memory smoke")
    if (
        report.get("status") != "completed"
        or report.get("passed") is not True
        or report.get("reportSha256") != FAILED_SMOKE_REPORT_SHA256
    ):
        raise Phase4F36Error("F36 initial memory smoke evidence changed")
    preflight = report.get("preflight")
    gpu = preflight.get("gpuHours") if isinstance(preflight, Mapping) else None
    stage = gpu.get("stage") if isinstance(gpu, Mapping) else None
    if not isinstance(stage, Mapping) or stage.get("stage") != "phase4f35-stage-a2-pilot":
        raise Phase4F36Error("F36 initial memory smoke used the wrong stage")
    return report


def _ledger_snapshot(workspace: Path) -> dict[str, object]:
    relative = PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    path = workspace / relative
    ledger = GpuBudgetLedger(path)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F36Error("a running GPU ledger entry blocks F36 recovery")
    return {
        "path": relative.as_posix(),
        "fileSha256": sha256_file(path),
        "totalGpuHours": ledger.total_hours,
        "externalHoursFloor": ledger.external_hours_floor,
        "runs": {key: dict(value) for key, value in sorted(ledger.runs.items())},
    }


def _verify_ledger_append_only(
    workspace: Path, snapshot: Mapping[str, object], maximum_hours: float
) -> float:
    path = snapshot.get("path")
    runs = snapshot.get("runs")
    total = snapshot.get("totalGpuHours")
    floor = snapshot.get("externalHoursFloor")
    if (
        not isinstance(path, str)
        or not isinstance(runs, Mapping)
        or isinstance(total, bool)
        or not isinstance(total, (int, float))
        or isinstance(floor, bool)
        or not isinstance(floor, (int, float))
    ):
        raise Phase4F36Error("F36 GPU ledger snapshot is invalid")
    current = GpuBudgetLedger(workspace / path)
    if any(record.get("status") == "running" for record in current.runs.values()):
        raise Phase4F36Error("a running GPU ledger entry blocks F36 verification")
    if any(current.runs.get(key) != value for key, value in runs.items()):
        raise Phase4F36Error("GPU ledger is not append-only from F36 recovery")
    if current.external_hours_floor < float(floor):
        raise Phase4F36Error("GPU ledger external-hours floor regressed")
    if not float(total) <= current.total_hours <= maximum_hours:
        raise Phase4F36Error("F36 GPU-hour envelope was exceeded")
    return current.total_hours


def _resources(workspace: Path) -> dict[str, int]:
    artifacts = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free = shutil.disk_usage(workspace).free
    if artifacts + ARTIFACT_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F36Error("12 GiB cap cannot preserve the F36 pilot reserve")
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F36Error("less than 20 GiB free disk remains")
    return {
        "artifactBytes": artifacts,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_RESERVE_BYTES,
        "freeDiskBytes": free,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    return replace(config, run_dir=(workspace / config.run_dir).resolve())


def prepare_phase4f36(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    f35 = _verify_f35_history(workspace)
    smoke = _verify_failed_smoke(workspace)
    snapshot = _ledger_snapshot(workspace)
    original_baseline = float(f35["globalGpuHours"])
    maximum_hours = original_baseline + float(f35["pilot"]["stageGpuHours"])
    remaining = math.floor(
        (maximum_hours - float(snapshot["totalGpuHours"])) * 1_000_000
    ) / 1_000_000
    if remaining <= 0:
        raise Phase4F36Error("no F35 pilot GPU budget remains")

    config_payload = _mapping(workspace / F35_CONFIG, "F35 config")
    config_payload["name"] = "phase4f36-stage-a2-pilot-v2"
    config_payload["runDir"] = RUN_DIR.as_posix()
    execution_payload = {
        "schemaVersion": 1,
        "stage": PHASE4F36_PILOT_STAGE,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f36-recoverable-stage-a2-stratified-guidance-pilot",
        "formalPromotion": False,
        "maximumGenerations": 1,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": snapshot["totalGpuHours"],
            "limitHours": remaining,
            "baselineRunId": "phase4f35-stage-a2-preparation-v1",
            "baselineReportSha256": F35_REPORT_SHA256,
        },
    }
    _write_immutable(workspace / CONFIG, config_payload, "F36 pilot config")
    _write_immutable(workspace / EXECUTION, execution_payload, "F36 execution contract")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f36-stage-a2-pilot-v2-recovery-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingAuthorized": True,
            "formalPromotionAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "historicalF35": {
                "path": F35_REPORT.as_posix(),
                "reportSha256": F35_REPORT_SHA256,
                "maximumGlobalGpuHours": maximum_hours,
            },
            "failedAttempt": {
                "memorySmokePath": FAILED_SMOKE.as_posix(),
                "memorySmokeReportSha256": smoke["reportSha256"],
                "failure": "replay-input-gate-used-default-8-gib-cap",
            },
            "gpuLedger": snapshot,
            "pilot": {
                "configPath": CONFIG.as_posix(),
                "configFileSha256": sha256_file(workspace / CONFIG),
                "semanticConfigSha256": fingerprint_config(config).sha256,
                "executionPath": EXECUTION.as_posix(),
                "executionFileSha256": sha256_file(workspace / EXECUTION),
                "executionSha256": execution.sha256,
                "maximumGenerations": 1,
                "remainingStageGpuHours": remaining,
            },
            "resources": _resources(workspace),
        },
        "reportSha256",
    )
    _write_immutable(workspace / REPORT, report, "F36 recovery report")
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / REPORT, "F36 recovery report")
    _verify_self_hash(report, "reportSha256", "F36 recovery report")
    if (
        report.get("status") != "prepared"
        or report.get("trainingAuthorized") is not True
        or report.get("formalPromotionAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
    ):
        raise Phase4F36Error("F36 recovery preparation is stale")
    maximum = float(report["historicalF35"]["maximumGlobalGpuHours"])
    _verify_ledger_append_only(workspace, report["gpuLedger"], maximum)
    for path, key in (
        (CONFIG, "configFileSha256"),
        (EXECUTION, "executionFileSha256"),
    ):
        if sha256_file(workspace / path) != report["pilot"][key]:
            raise Phase4F36Error("F36 frozen pilot artifact changed")
    return report


def verify_phase4f36_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract != expected
        or fingerprint_config(config).sha256
        != report["pilot"]["semanticConfigSha256"]
    ):
        raise Phase4F36Error("F36 Stage A2 workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["pilot"]["semanticConfigSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "formalStageA2TrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-preparation"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    workspace = args.workspace_root.resolve()
    report = (
        prepare_phase4f36(workspace)
        if args.command == "prepare"
        else _verified_preparation(workspace)
    )
    print(
        canonical_json_bytes(
            {
                "status": report["status"],
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "remainingStageGpuHours": report["pilot"][
                    "remainingStageGpuHours"
                ],
                "nextAction": "run-phase4f36-stage-a2-pilot-v2",
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
