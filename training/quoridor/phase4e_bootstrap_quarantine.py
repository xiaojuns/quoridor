"""CPU-only quarantine for a failed Phase 4E bootstrap initialization.

The quarantine is intentionally narrower than a general training-run recovery
tool.  It accepts only a lineage that failed before its initialization
checkpoint: replay and tensor-cache manifests must be empty, and checkpoint and
report directories must contain no files.  The original run directory is
sealed with an incident and a copy of the failure log before one atomic rename.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from .run_manifest import (
    RunLock,
    RunManifestError,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    load_run_manifest,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .phase4_execution import (
    Phase4ExecutionContractError,
    load_phase4_execution_contract,
)


SCHEMA_VERSION = 1
INCIDENT_KIND = "phase4e-bootstrap-failed-run-quarantine"
INCIDENT_NAME = "quarantine-incident.json"
PRESERVED_LOG_NAME = "quarantine-training.log"
GPU_LOCK_NAME = ".phase4.lock"
DEFAULT_CONTROL_ROOT = Path("training/runs/phase4")
DEFAULT_QUARANTINE_ROOT = DEFAULT_CONTROL_ROOT / "phase4e/quarantine"
DEFAULT_GPU_LEDGER = DEFAULT_CONTROL_ROOT / "gpu-budget-ledger.json"
DEFAULT_OLD_EXECUTION_CONTRACT = Path(
    "training/configs/phase4e-execution-frozen-v1.json"
)
DEFAULT_EXPERIMENT = "phase4e-bootstrap-v1"
FAILURE_REASON = "bootstrap-baseline-import-name-error"
DEFAULT_FAILURE_MARKER = (
    "NameError: name 'verify_bootstrap_baseline' is not defined"
)
_LEDGER_STATUSES = {
    "running",
    "completed",
    "interrupted",
    "failed",
    "abandoned",
}
_ORIGINAL_RUN_FILES = {
    "run-manifest.json",
    "replay/manifest.json",
    "replay-tensor-cache/manifest.json",
}
_GIB = 1024**3
_ARTIFACT_LIMIT_BYTES = 8 * _GIB
_MINIMUM_FREE_DISK_BYTES = 20 * _GIB
_QUARANTINE_WRITE_RESERVE_BYTES = 1024**2


class Phase4EBootstrapQuarantineError(RuntimeError):
    """Raised when failed-run evidence cannot be safely quarantined."""


def _sha256(value: object, label: str) -> str:
    try:
        return validate_sha256(value, label)
    except ValueError as error:
        raise Phase4EBootstrapQuarantineError(
            f"{label} is not a valid SHA-256 digest"
        ) from error


def _workspace_path(
    workspace: Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as error:
        raise Phase4EBootstrapQuarantineError(
            f"{label} must be inside the workspace"
        ) from error
    cursor = workspace
    for part in relative.parts:
        cursor = cursor / part
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise Phase4EBootstrapQuarantineError(
                f"{label} must not traverse a symlink"
            )
    return absolute


def _relative(path: Path, workspace: Path, label: str) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4EBootstrapQuarantineError(
            f"{label} is outside the workspace"
        ) from error


def _load_canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise Phase4EBootstrapQuarantineError(
            f"{label} is not a regular file"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4EBootstrapQuarantineError(
            f"{label} is unreadable"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase4EBootstrapQuarantineError(f"{label} must be a mapping")
    normalized = dict(payload)
    if raw != canonical_json_bytes(normalized):
        raise Phase4EBootstrapQuarantineError(
            f"{label} is not canonical JSON"
        )
    return normalized, raw


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    digest = _sha256(payload.get(field), f"{label} {field}")
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4EBootstrapQuarantineError(
            f"{label} self hash does not verify"
        )
    return digest


def _self_hashed(
    core: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    normalized = dict(core)
    return {
        **normalized,
        field: sha256_bytes(canonical_json_bytes(normalized)),
    }


def _tree_inventory(
    root: Path,
    *,
    exclude_evidence: bool,
) -> dict[str, object]:
    excluded = (
        {INCIDENT_NAME, PRESERVED_LOG_NAME} if exclude_evidence else set()
    )
    files: list[dict[str, object]] = []
    directories: list[str] = []
    for item in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise Phase4EBootstrapQuarantineError(
                "bootstrap run contains a symlink"
            )
        if item.is_dir():
            directories.append(relative)
            continue
        if not item.is_file():
            raise Phase4EBootstrapQuarantineError(
                "bootstrap run contains a non-regular artifact"
            )
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {"directories": directories, "files": files}


def _validate_inventory(
    directories_value: object,
    files_value: object,
) -> dict[str, object]:
    if (
        not isinstance(directories_value, list)
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in directories_value
        )
        or directories_value != sorted(set(directories_value))
    ):
        raise Phase4EBootstrapQuarantineError(
            "quarantine directory inventory is invalid"
        )
    if not isinstance(files_value, list):
        raise Phase4EBootstrapQuarantineError(
            "quarantine file inventory must be a list"
        )
    normalized: list[dict[str, object]] = []
    previous = ""
    for item in files_value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise Phase4EBootstrapQuarantineError(
                "quarantine inventory entry is invalid"
            )
        path = item.get("path")
        byte_size = item.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path <= previous
            or type(byte_size) is not int
            or byte_size < 0
        ):
            raise Phase4EBootstrapQuarantineError(
                "quarantine inventory entry is unsafe"
            )
        digest = _sha256(item.get("sha256"), "inventory file sha256")
        normalized.append(
            {"path": path, "bytes": byte_size, "sha256": digest}
        )
        previous = path
    return {
        "directories": list(directories_value),
        "files": normalized,
    }


def _execution_contract_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    payload, raw = _load_canonical(
        path, "frozen Phase 4E execution contract"
    )
    try:
        contract = load_phase4_execution_contract(path)
        contract.require_frozen_budget()
    except (OSError, ValueError, Phase4ExecutionContractError) as error:
        raise Phase4EBootstrapQuarantineError(
            "frozen Phase 4E execution contract does not verify"
        ) from error
    if (
        contract.stage != "phase4e"
        or contract.formal_promotion is not True
        or contract.sha256 != expected_sha256
        or sha256_bytes(raw) != expected_sha256
        or payload != contract.to_payload()
    ):
        raise Phase4EBootstrapQuarantineError(
            "frozen Phase 4E execution contract hash or policy changed"
        )
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": expected_sha256,
    }


def _manifest_evidence(
    manifest_path: Path,
    *,
    run_dir: Path,
    expected_manifest_sha256: str,
    expected_contract_sha256: str,
    expected_experiment: str,
) -> tuple[dict[str, object], str]:
    try:
        record = load_run_manifest(manifest_path)
    except (OSError, ValueError, RunManifestError) as error:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run manifest does not verify"
        ) from error
    if manifest_path.read_bytes() != canonical_json_bytes(record.payload):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run manifest is not canonical JSON"
        )
    if record.sha256 != expected_manifest_sha256:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run manifest differs from the approved failed lineage"
        )
    payload = record.payload
    metadata = payload.get("metadata")
    provenance = payload.get("provenance")
    resource_budget = payload.get("resourceBudget")
    config = (
        provenance.get("config") if isinstance(provenance, Mapping) else None
    )
    source = (
        provenance.get("source") if isinstance(provenance, Mapping) else None
    )
    config_payload = (
        config.get("payload") if isinstance(config, Mapping) else None
    )
    budget_metadata = (
        resource_budget.get("metadata")
        if isinstance(resource_budget, Mapping)
        else None
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("stage") != "phase4e"
        or metadata.get("formalPromotion") is not True
        or metadata.get("executionContractSha256")
        != expected_contract_sha256
        or not isinstance(config_payload, Mapping)
        or config_payload.get("name") != expected_experiment
        or Path(str(config_payload.get("run_dir"))).resolve(strict=False)
        != run_dir
        or not isinstance(source, Mapping)
        or not isinstance(budget_metadata, Mapping)
        or budget_metadata.get("executionContractSha256")
        != expected_contract_sha256
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run manifest contract is invalid"
        )
    original_source_sha256 = _sha256(
        source.get("sha256"), "original source sha256"
    )
    return (
        {
            "file": "run-manifest.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "manifestSha256": record.sha256,
        },
        original_source_sha256,
    )


def _validate_empty_replay(run_dir: Path) -> None:
    replay_root = run_dir / "replay"
    payload, replay_bytes = _load_canonical(
        replay_root / "manifest.json",
        "bootstrap replay manifest",
    )
    totals = payload.get("totals")
    if (
        set(payload)
        != {
            "actionSpaceSize",
            "capacity",
            "codecVersion",
            "nextSequence",
            "schemaVersion",
            "shardSchemaVersion",
            "shards",
            "totals",
        }
        or payload.get("schemaVersion") != 1
        or payload.get("nextSequence") != 0
        or payload.get("shards") != []
        or not isinstance(totals, Mapping)
        or dict(totals) != {"games": 0, "plies": 0, "shardBytes": 0}
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap replay contains training progress"
        )
    replay_files = sorted(
        item.relative_to(replay_root).as_posix()
        for item in replay_root.rglob("*")
        if item.is_file() or item.is_symlink()
    )
    if replay_files != ["manifest.json"]:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap replay contains unexpected artifacts"
        )

    cache_root = run_dir / "replay-tensor-cache"
    cache, _ = _load_canonical(
        cache_root / "manifest.json",
        "bootstrap replay tensor-cache manifest",
    )
    _verify_self_hash(
        cache,
        field="manifestSha256",
        label="bootstrap replay tensor-cache manifest",
    )
    cache_totals = cache.get("totals")
    contract = cache.get("contract")
    if (
        set(cache)
        != {
            "algorithmVersion",
            "chunks",
            "contract",
            "entries",
            "manifestSha256",
            "mirrorRngAlgorithm",
            "rawManifestSha256",
            "rawWindowSha256",
            "samplingAlgorithm",
            "schemaVersion",
            "totals",
        }
        or cache.get("schemaVersion") != 1
        or cache.get("chunks") != []
        or cache.get("entries") != []
        or not isinstance(cache_totals, Mapping)
        or dict(cache_totals)
        != {"cacheBytes": 0, "chunks": 0, "games": 0, "plies": 0}
        or cache.get("rawManifestSha256") != sha256_bytes(replay_bytes)
        or not isinstance(contract, Mapping)
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap replay tensor cache contains training progress"
        )
    cache_files = sorted(
        item.relative_to(cache_root).as_posix()
        for item in cache_root.rglob("*")
        if item.is_file() or item.is_symlink()
    )
    if cache_files != ["manifest.json"]:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap replay tensor cache contains unexpected artifacts"
        )


def _validate_no_model_progress(run_dir: Path) -> dict[str, object]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run root is not a regular directory"
        )
    for directory_name in ("checkpoints", "reports"):
        directory = run_dir / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise Phase4EBootstrapQuarantineError(
                f"bootstrap {directory_name} root is invalid"
            )
        if any(directory.iterdir()):
            raise Phase4EBootstrapQuarantineError(
                f"bootstrap {directory_name} contains training progress"
            )
    _validate_empty_replay(run_dir)
    inventory = _tree_inventory(run_dir, exclude_evidence=True)
    if {
        item["path"] for item in inventory["files"]
    } != _ORIGINAL_RUN_FILES:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap run contains artifacts outside the failed pre-checkpoint state"
        )
    return inventory


def _validate_cache_source_binding(
    run_dir: Path,
    *,
    expected_source_sha256: str,
) -> None:
    cache, _ = _load_canonical(
        run_dir / "replay-tensor-cache/manifest.json",
        "bootstrap replay tensor-cache manifest",
    )
    contract = cache.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("sourceSha256") != expected_source_sha256
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap replay tensor cache source differs from the run manifest"
        )


def _validate_ledger_payload(
    payload: Mapping[str, object],
    *,
    gpu_ledger_run_id: str,
    label: str,
) -> tuple[str, float, dict[str, object]]:
    ledger_sha256 = _verify_self_hash(
        payload,
        field="ledgerSha256",
        label=label,
    )
    runs = payload.get("runs")
    external = payload.get("externalHoursFloor")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("limitHours") != 72.0
        or not isinstance(runs, Mapping)
        or isinstance(external, bool)
        or not isinstance(external, (int, float))
        or not math.isfinite(float(external))
        or float(external) < 0
    ):
        raise Phase4EBootstrapQuarantineError(
            f"{label} contract is invalid"
        )
    normalized_runs: dict[str, dict[str, object]] = {}
    for run_id, raw_entry in runs.items():
        if not isinstance(run_id, str) or not isinstance(raw_entry, Mapping):
            raise Phase4EBootstrapQuarantineError(
                f"{label} run entry is invalid"
            )
        entry = dict(raw_entry)
        seconds = entry.get("accountedSeconds")
        if (
            entry.get("status") not in _LEDGER_STATUSES
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) < 0
        ):
            raise Phase4EBootstrapQuarantineError(
                f"{label} run entry is invalid"
            )
        normalized_runs[run_id] = entry
    if any(entry.get("status") == "running" for entry in normalized_runs.values()):
        raise Phase4EBootstrapQuarantineError(
            "a running Phase 4 GPU ledger entry blocks bootstrap quarantine"
        )
    failed = normalized_runs.get(gpu_ledger_run_id)
    if failed is None or failed.get("status") != "failed":
        raise Phase4EBootstrapQuarantineError(
            "bootstrap GPU ledger invocation is not terminally failed"
        )
    total_hours = float(external) + sum(
        float(entry["accountedSeconds"])
        for entry in normalized_runs.values()
    ) / 3600.0
    return ledger_sha256, total_hours, failed


def _ledger_evidence(
    ledger_path: Path,
    *,
    gpu_ledger_run_id: str,
) -> tuple[dict[str, object], bytes]:
    payload, raw = _load_canonical(ledger_path, "Phase 4 GPU ledger")
    ledger_sha256, total_hours, failed = _validate_ledger_payload(
        payload,
        gpu_ledger_run_id=gpu_ledger_run_id,
        label="Phase 4 GPU ledger",
    )
    return (
        {
            "path": str(ledger_path),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "ledgerSha256": ledger_sha256,
            "globalGpuHours": total_hours,
            "failedRun": {
                "runId": gpu_ledger_run_id,
                "entry": failed,
            },
            "payload": payload,
        },
        raw,
    )


def _verify_captured_ledger(
    ledger: object,
    *,
    ledger_path: Path,
    workspace: Path,
    gpu_ledger_run_id: str,
    accounted_seconds: object,
) -> None:
    expected_fields = {
        "path",
        "bytes",
        "sha256",
        "ledgerSha256",
        "globalGpuHours",
        "failedRun",
        "payload",
    }
    if not isinstance(ledger, Mapping) or set(ledger) != expected_fields:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine GPU ledger evidence is invalid"
        )
    failed_run = ledger.get("failedRun")
    entry = (
        failed_run.get("entry") if isinstance(failed_run, Mapping) else None
    )
    global_hours = ledger.get("globalGpuHours")
    snapshot_payload = ledger.get("payload")
    if (
        ledger.get("path")
        != _relative(ledger_path, workspace, "Phase 4 GPU ledger")
        or type(ledger.get("bytes")) is not int
        or ledger.get("bytes") <= 0
        or not isinstance(global_hours, (int, float))
        or isinstance(global_hours, bool)
        or not math.isfinite(float(global_hours))
        or float(global_hours) < 0
        or not isinstance(failed_run, Mapping)
        or set(failed_run) != {"runId", "entry"}
        or failed_run.get("runId") != gpu_ledger_run_id
        or not isinstance(entry, Mapping)
        or entry.get("status") != "failed"
        or entry.get("accountedSeconds") != accounted_seconds
        or not isinstance(snapshot_payload, Mapping)
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine GPU ledger evidence is invalid"
        )
    snapshot_ledger_sha256, snapshot_global_hours, snapshot_failed = (
        _validate_ledger_payload(
            snapshot_payload,
            gpu_ledger_run_id=gpu_ledger_run_id,
            label="quarantine GPU ledger snapshot",
        )
    )
    snapshot_bytes = canonical_json_bytes(snapshot_payload)
    if (
        len(snapshot_bytes) != ledger.get("bytes")
        or sha256_bytes(snapshot_bytes)
        != _sha256(ledger.get("sha256"), "GPU ledger file sha256")
        or snapshot_ledger_sha256
        != _sha256(
            ledger.get("ledgerSha256"), "GPU ledger self sha256"
        )
        or snapshot_global_hours != global_hours
        or snapshot_failed != dict(entry)
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine GPU ledger snapshot changed"
        )
    current_ledger, _ = _ledger_evidence(
        ledger_path, gpu_ledger_run_id=gpu_ledger_run_id
    )
    current_payload = current_ledger["payload"]
    snapshot_runs = snapshot_payload.get("runs")
    current_runs = current_payload.get("runs")
    if (
        not isinstance(snapshot_runs, Mapping)
        or not isinstance(current_runs, Mapping)
        or any(
            current_runs.get(key) != value
            for key, value in snapshot_runs.items()
        )
        or float(current_payload.get("externalHoursFloor", -1))
        < float(snapshot_payload.get("externalHoursFloor", 0))
        or current_ledger["globalGpuHours"] < snapshot_global_hours
        or current_ledger["failedRun"]["entry"] != dict(entry)
    ):
        raise Phase4EBootstrapQuarantineError(
            "current GPU ledger is not an append-only extension of the incident"
        )


def _log_evidence(
    log_path: Path,
    *,
    workspace: Path,
    failure_marker: str,
) -> tuple[dict[str, object], bytes]:
    if log_path.is_symlink() or not log_path.is_file():
        raise Phase4EBootstrapQuarantineError(
            "bootstrap failure log is not a regular file"
        )
    raw = log_path.read_bytes()
    if not raw or failure_marker.encode("utf-8") not in raw:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap failure log does not contain the approved failure"
        )
    return (
        {
            "originalPath": _relative(
                log_path, workspace, "bootstrap failure log"
            ),
            "preservedFile": PRESERVED_LOG_NAME,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "failureMarker": failure_marker,
        },
        raw,
    )


def _directory_size(root: Path) -> int:
    total = 0
    for item in root.rglob("*"):
        if item.is_symlink():
            raise Phase4EBootstrapQuarantineError(
                "Phase 4 artifact root contains a symlink"
            )
        if item.is_file():
            total += item.stat().st_size
    return total


def _resource_preflight(
    control_root: Path,
    *,
    reserve_bytes: int,
) -> None:
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    if (
        _directory_size(control_root) + reserve_bytes
        > _ARTIFACT_LIMIT_BYTES
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine would exceed the 8 GiB artifact cap"
        )
    if int(shutil.disk_usage(control_root).free) - reserve_bytes < (
        _MINIMUM_FREE_DISK_BYTES
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine would cross the 20 GiB free-disk stop line"
        )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError as error:
        raise Phase4EBootstrapQuarantineError(
            "could not open quarantine parent for fsync"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise Phase4EBootstrapQuarantineError(
            "could not fsync quarantine parent"
        ) from error
    finally:
        os.close(descriptor)


def _fsync_quarantine_rename_parents(
    *,
    original_run_dir: Path,
    quarantine_destination: Path,
) -> None:
    _fsync_directory(original_run_dir.parent)
    _fsync_directory(quarantine_destination.parent)


def _destination_name(
    run_dir: Path,
    *,
    manifest_sha256: str,
    inventory_sha256: str,
    gpu_ledger_run_id: str,
) -> str:
    run_digest = sha256_bytes(gpu_ledger_run_id.encode("utf-8"))[:16]
    return (
        f"{run_dir.name}-manifest-{manifest_sha256}"
        f"-inventory-{inventory_sha256}-failure-{run_digest}"
    )


def _reference(
    incident_path: Path,
    *,
    workspace: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "path": _relative(
            incident_path, workspace, "bootstrap quarantine incident"
        ),
        "bytes": incident_path.stat().st_size,
        "sha256": sha256_file(incident_path),
        "incidentSha256": payload["incidentSha256"],
    }


def verify_quarantine_incident(
    incident_path: str | Path,
    *,
    workspace_root: str | Path,
    original_run_dir: str | Path,
    quarantine_root: str | Path,
    gpu_ledger_path: str | Path,
    failure_log_path: str | Path,
    gpu_ledger_run_id: str,
    old_execution_contract_sha256: str,
    run_manifest_sha256: str,
    replacement_source_sha256: str,
    old_execution_contract_path: str | Path = (
        DEFAULT_OLD_EXECUTION_CONTRACT
    ),
    expected_experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, object]:
    """Deeply verify a sealed failed-bootstrap quarantine incident."""

    workspace = Path(workspace_root).resolve()
    run_dir = _workspace_path(
        workspace, original_run_dir, label="original bootstrap run"
    )
    quarantine = _workspace_path(
        workspace, quarantine_root, label="bootstrap quarantine root"
    )
    ledger_path = _workspace_path(
        workspace, gpu_ledger_path, label="Phase 4 GPU ledger"
    )
    log_path = _workspace_path(
        workspace, failure_log_path, label="bootstrap failure log"
    )
    contract_path = _workspace_path(
        workspace,
        old_execution_contract_path,
        label="frozen Phase 4E execution contract",
    )
    path = _workspace_path(
        workspace, incident_path, label="bootstrap quarantine incident"
    )
    old_contract = _sha256(
        old_execution_contract_sha256, "old execution contract sha256"
    )
    expected_manifest = _sha256(
        run_manifest_sha256, "run manifest sha256"
    )
    replacement_source = _sha256(
        replacement_source_sha256, "replacement source sha256"
    )
    if fingerprint_sources(workspace).sha256 != replacement_source:
        raise Phase4EBootstrapQuarantineError(
            "replacement source does not match the current workspace"
        )
    contract_evidence = _execution_contract_evidence(
        contract_path, expected_sha256=old_contract
    )
    try:
        path.parent.relative_to(quarantine)
    except ValueError as error:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine incident is outside the quarantine root"
        ) from error
    payload, _ = _load_canonical(path, "bootstrap quarantine incident")
    incident_sha256 = _verify_self_hash(
        payload,
        field="incidentSha256",
        label="bootstrap quarantine incident",
    )
    expected_fields = {
        "schemaVersion",
        "kind",
        "status",
        "reason",
        "originalRunPath",
        "quarantinePath",
        "oldExecutionContractSha256",
        "oldExecutionContract",
        "runManifestSha256",
        "originalSourceSha256",
        "replacementSourceSha256",
        "gpuLedgerRunId",
        "gpuLedgerStatus",
        "gpuLedgerAccountedSeconds",
        "originalRunManifest",
        "gpuLedgerSnapshot",
        "failureLog",
        "directories",
        "files",
        "inventorySha256",
        "incidentSha256",
    }
    manifest = payload.get("originalRunManifest")
    ledger = payload.get("gpuLedgerSnapshot")
    failure_log = payload.get("failureLog")
    inventory = _validate_inventory(
        payload.get("directories"),
        payload.get("files"),
    )
    inventory_sha256 = _sha256(
        payload.get("inventorySha256"), "inventory sha256"
    )
    accounted = payload.get("gpuLedgerAccountedSeconds")
    if (
        set(payload) != expected_fields
        or payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind") != INCIDENT_KIND
        or payload.get("status") != "sealed"
        or payload.get("reason") != FAILURE_REASON
        or payload.get("originalRunPath")
        != _relative(run_dir, workspace, "original bootstrap run")
        or payload.get("quarantinePath")
        != _relative(path.parent, workspace, "bootstrap quarantine path")
        or payload.get("oldExecutionContractSha256") != old_contract
        or payload.get("oldExecutionContract") != {
            **contract_evidence,
            "path": _relative(
                contract_path,
                workspace,
                "frozen Phase 4E execution contract",
            ),
        }
        or payload.get("runManifestSha256") != expected_manifest
        or payload.get("replacementSourceSha256") != replacement_source
        or payload.get("gpuLedgerRunId") != gpu_ledger_run_id
        or payload.get("gpuLedgerStatus") != "failed"
        or isinstance(accounted, bool)
        or not isinstance(accounted, (int, float))
        or not math.isfinite(float(accounted))
        or float(accounted) < 0
        or not isinstance(manifest, Mapping)
        or not isinstance(ledger, Mapping)
        or not isinstance(failure_log, Mapping)
        or sha256_bytes(canonical_json_bytes(inventory))
        != inventory_sha256
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine incident contract mismatch"
        )
    expected_destination = _destination_name(
        run_dir,
        manifest_sha256=expected_manifest,
        inventory_sha256=inventory_sha256,
        gpu_ledger_run_id=gpu_ledger_run_id,
    )
    if path.parent.name != expected_destination:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine directory is not content-addressed"
        )

    manifest_path = path.parent / "run-manifest.json"
    manifest_evidence, original_source = _manifest_evidence(
        manifest_path,
        run_dir=run_dir,
        expected_manifest_sha256=expected_manifest,
        expected_contract_sha256=old_contract,
        expected_experiment=expected_experiment,
    )
    if (
        dict(manifest) != manifest_evidence
        or payload.get("originalSourceSha256") != original_source
        or original_source == replacement_source
    ):
        raise Phase4EBootstrapQuarantineError(
            "quarantined run manifest evidence changed"
        )
    _validate_cache_source_binding(
        path.parent, expected_source_sha256=original_source
    )

    if _tree_inventory(path.parent, exclude_evidence=True) != inventory:
        raise Phase4EBootstrapQuarantineError(
            "quarantined bootstrap artifact inventory changed"
        )
    _validate_no_model_progress(path.parent)

    _verify_captured_ledger(
        ledger,
        ledger_path=ledger_path,
        workspace=workspace,
        gpu_ledger_run_id=gpu_ledger_run_id,
        accounted_seconds=accounted,
    )

    expected_log_fields = {
        "originalPath",
        "preservedFile",
        "bytes",
        "sha256",
        "failureMarker",
    }
    preserved_log = path.parent / PRESERVED_LOG_NAME
    if (
        set(failure_log) != expected_log_fields
        or failure_log.get("originalPath")
        != _relative(log_path, workspace, "bootstrap failure log")
        or failure_log.get("preservedFile") != PRESERVED_LOG_NAME
        or type(failure_log.get("bytes")) is not int
        or failure_log.get("bytes") <= 0
        or preserved_log.is_symlink()
        or not preserved_log.is_file()
        or preserved_log.stat().st_size != failure_log.get("bytes")
        or sha256_file(preserved_log) != failure_log.get("sha256")
        or failure_log.get("failureMarker") != DEFAULT_FAILURE_MARKER
        or failure_log["failureMarker"].encode("utf-8")
        not in preserved_log.read_bytes()
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine failure-log evidence changed"
        )
    _sha256(failure_log.get("sha256"), "failure log sha256")
    if payload.get("incidentSha256") != incident_sha256:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine incident digest changed"
        )
    return payload


def verify_quarantine_reference(
    reference: object,
    *,
    workspace_root: str | Path,
    original_run_dir: str | Path,
    quarantine_root: str | Path,
    gpu_ledger_path: str | Path,
    failure_log_path: str | Path,
    gpu_ledger_run_id: str,
    old_execution_contract_sha256: str,
    run_manifest_sha256: str,
    replacement_source_sha256: str,
    old_execution_contract_path: str | Path = (
        DEFAULT_OLD_EXECUTION_CONTRACT
    ),
    expected_experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, object]:
    """Verify a stable incident reference and return a normalized copy."""

    if not isinstance(reference, Mapping) or set(reference) != {
        "path",
        "bytes",
        "sha256",
        "incidentSha256",
    }:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine reference is invalid"
        )
    workspace = Path(workspace_root).resolve()
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine reference path is invalid"
        )
    path = _workspace_path(
        workspace, path_value, label="bootstrap quarantine reference"
    )
    payload = verify_quarantine_incident(
        path,
        workspace_root=workspace,
        original_run_dir=original_run_dir,
        quarantine_root=quarantine_root,
        gpu_ledger_path=gpu_ledger_path,
        failure_log_path=failure_log_path,
        gpu_ledger_run_id=gpu_ledger_run_id,
        old_execution_contract_sha256=old_execution_contract_sha256,
        run_manifest_sha256=run_manifest_sha256,
        replacement_source_sha256=replacement_source_sha256,
        old_execution_contract_path=old_execution_contract_path,
        expected_experiment=expected_experiment,
    )
    if (
        type(reference.get("bytes")) is not int
        or reference.get("bytes") != path.stat().st_size
        or sha256_file(path)
        != _sha256(reference.get("sha256"), "incident file sha256")
        or payload.get("incidentSha256")
        != _sha256(
            reference.get("incidentSha256"), "incident self sha256"
        )
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine reference changed"
        )
    return dict(reference)


def verify_quarantine_incident_attestation(
    *,
    workspace_root: str | Path,
    incident_path: str | Path,
    expected_old_execution_contract_sha256: str,
    expected_original_source_sha256: str,
    expected_replacement_source_sha256: str,
) -> dict[str, object]:
    """Verify an incident using only the provenance expected by a rebaseline."""

    workspace = Path(workspace_root).resolve()
    canonical_quarantine_root = _workspace_path(
        workspace,
        DEFAULT_QUARANTINE_ROOT,
        label="canonical bootstrap quarantine root",
    )
    path = _workspace_path(
        workspace, incident_path, label="bootstrap quarantine incident"
    )
    try:
        relative_incident = path.relative_to(canonical_quarantine_root)
    except ValueError as error:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine attestation is outside the canonical root"
        ) from error
    if (
        len(relative_incident.parts) != 2
        or relative_incident.name != INCIDENT_NAME
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine attestation path is not canonical"
        )
    payload, _ = _load_canonical(path, "bootstrap quarantine incident")
    _verify_self_hash(
        payload,
        field="incidentSha256",
        label="bootstrap quarantine incident",
    )
    old_contract = _sha256(
        expected_old_execution_contract_sha256,
        "expected old execution contract sha256",
    )
    original_source = _sha256(
        expected_original_source_sha256,
        "expected original source sha256",
    )
    replacement_source = _sha256(
        expected_replacement_source_sha256,
        "expected replacement source sha256",
    )
    ledger = payload.get("gpuLedgerSnapshot")
    failure_log = payload.get("failureLog")
    old_contract_reference = payload.get("oldExecutionContract")
    required_strings = (
        payload.get("originalRunPath"),
        payload.get("quarantinePath"),
        payload.get("gpuLedgerRunId"),
        payload.get("runManifestSha256"),
    )
    if (
        payload.get("oldExecutionContractSha256") != old_contract
        or payload.get("originalSourceSha256") != original_source
        or payload.get("replacementSourceSha256") != replacement_source
        or not isinstance(ledger, Mapping)
        or not isinstance(ledger.get("path"), str)
        or not isinstance(failure_log, Mapping)
        or not isinstance(failure_log.get("originalPath"), str)
        or not isinstance(old_contract_reference, Mapping)
        or not isinstance(old_contract_reference.get("path"), str)
        or any(not isinstance(value, str) or not value for value in required_strings)
        or path.parent
        != _workspace_path(
            workspace,
            str(payload.get("quarantinePath")),
            label="bootstrap quarantine path",
        )
    ):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine attestation provenance is invalid"
        )
    verified = verify_quarantine_incident(
        path,
        workspace_root=workspace,
        original_run_dir=str(payload["originalRunPath"]),
        quarantine_root=canonical_quarantine_root,
        gpu_ledger_path=str(ledger["path"]),
        failure_log_path=str(failure_log["originalPath"]),
        gpu_ledger_run_id=str(payload["gpuLedgerRunId"]),
        old_execution_contract_sha256=old_contract,
        run_manifest_sha256=str(payload["runManifestSha256"]),
        replacement_source_sha256=replacement_source,
        old_execution_contract_path=str(old_contract_reference["path"]),
    )
    if verified.get("originalSourceSha256") != original_source:
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine original source changed"
        )
    return verified


def _verify_presealed_incident(
    *,
    workspace: Path,
    run_dir: Path,
    quarantine_root: Path,
    ledger_path: Path,
    log_path: Path,
    contract_path: Path,
    gpu_ledger_run_id: str,
    old_contract: str,
    manifest_sha256: str,
    replacement_source: str,
    expected_experiment: str,
) -> dict[str, object]:
    incident_path = run_dir / INCIDENT_NAME
    preserved_log = run_dir / PRESERVED_LOG_NAME
    if (
        not os.path.lexists(incident_path)
        or not os.path.lexists(preserved_log)
    ):
        raise Phase4EBootstrapQuarantineError(
            "partially sealed bootstrap quarantine evidence is incomplete"
        )
    payload, _ = _load_canonical(
        incident_path, "presealed bootstrap quarantine incident"
    )
    _verify_self_hash(
        payload,
        field="incidentSha256",
        label="presealed bootstrap quarantine incident",
    )
    expected_fields = {
        "schemaVersion",
        "kind",
        "status",
        "reason",
        "originalRunPath",
        "quarantinePath",
        "oldExecutionContractSha256",
        "oldExecutionContract",
        "runManifestSha256",
        "originalSourceSha256",
        "replacementSourceSha256",
        "gpuLedgerRunId",
        "gpuLedgerStatus",
        "gpuLedgerAccountedSeconds",
        "originalRunManifest",
        "gpuLedgerSnapshot",
        "failureLog",
        "directories",
        "files",
        "inventorySha256",
        "incidentSha256",
    }
    contract_evidence = _execution_contract_evidence(
        contract_path, expected_sha256=old_contract
    )
    inventory = _validate_inventory(
        payload.get("directories"), payload.get("files")
    )
    inventory_sha256 = _sha256(
        payload.get("inventorySha256"), "inventory sha256"
    )
    destination_value = payload.get("quarantinePath")
    destination = (
        _workspace_path(
            workspace,
            destination_value,
            label="bootstrap quarantine destination",
        )
        if isinstance(destination_value, str)
        else None
    )
    accounted = payload.get("gpuLedgerAccountedSeconds")
    if (
        set(payload) != expected_fields
        or payload.get("schemaVersion") != SCHEMA_VERSION
        or payload.get("kind") != INCIDENT_KIND
        or payload.get("status") != "sealed"
        or payload.get("reason") != FAILURE_REASON
        or payload.get("originalRunPath")
        != _relative(run_dir, workspace, "original bootstrap run")
        or destination is None
        or destination.parent != quarantine_root
        or payload.get("oldExecutionContractSha256") != old_contract
        or payload.get("oldExecutionContract")
        != {
            **contract_evidence,
            "path": _relative(
                contract_path,
                workspace,
                "frozen Phase 4E execution contract",
            ),
        }
        or payload.get("runManifestSha256") != manifest_sha256
        or payload.get("replacementSourceSha256") != replacement_source
        or fingerprint_sources(workspace).sha256 != replacement_source
        or payload.get("gpuLedgerRunId") != gpu_ledger_run_id
        or payload.get("gpuLedgerStatus") != "failed"
        or isinstance(accounted, bool)
        or not isinstance(accounted, (int, float))
        or not math.isfinite(float(accounted))
        or float(accounted) < 0
        or sha256_bytes(canonical_json_bytes(inventory))
        != inventory_sha256
    ):
        raise Phase4EBootstrapQuarantineError(
            "presealed bootstrap quarantine incident contract mismatch"
        )
    if destination.name != _destination_name(
        run_dir,
        manifest_sha256=manifest_sha256,
        inventory_sha256=inventory_sha256,
        gpu_ledger_run_id=gpu_ledger_run_id,
    ):
        raise Phase4EBootstrapQuarantineError(
            "presealed bootstrap destination is not content-addressed"
        )
    manifest_evidence, original_source = _manifest_evidence(
        run_dir / "run-manifest.json",
        run_dir=run_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_contract_sha256=old_contract,
        expected_experiment=expected_experiment,
    )
    if (
        payload.get("originalRunManifest") != manifest_evidence
        or payload.get("originalSourceSha256") != original_source
        or original_source == replacement_source
    ):
        raise Phase4EBootstrapQuarantineError(
            "presealed bootstrap manifest evidence changed"
        )
    _validate_cache_source_binding(
        run_dir, expected_source_sha256=original_source
    )
    if (
        _validate_no_model_progress(run_dir) != inventory
        or _tree_inventory(run_dir, exclude_evidence=True) != inventory
    ):
        raise Phase4EBootstrapQuarantineError(
            "presealed bootstrap inventory changed"
        )
    _verify_captured_ledger(
        payload.get("gpuLedgerSnapshot"),
        ledger_path=ledger_path,
        workspace=workspace,
        gpu_ledger_run_id=gpu_ledger_run_id,
        accounted_seconds=accounted,
    )
    failure_log = payload.get("failureLog")
    if (
        not isinstance(failure_log, Mapping)
        or set(failure_log)
        != {
            "originalPath",
            "preservedFile",
            "bytes",
            "sha256",
            "failureMarker",
        }
        or failure_log.get("originalPath")
        != _relative(log_path, workspace, "bootstrap failure log")
        or failure_log.get("preservedFile") != PRESERVED_LOG_NAME
        or failure_log.get("failureMarker") != DEFAULT_FAILURE_MARKER
        or type(failure_log.get("bytes")) is not int
        or failure_log.get("bytes") <= 0
        or preserved_log.is_symlink()
        or not preserved_log.is_file()
        or preserved_log.stat().st_size != failure_log.get("bytes")
        or sha256_file(preserved_log) != failure_log.get("sha256")
        or DEFAULT_FAILURE_MARKER.encode("utf-8")
        not in preserved_log.read_bytes()
    ):
        raise Phase4EBootstrapQuarantineError(
            "presealed bootstrap failure-log evidence changed"
        )
    _sha256(failure_log.get("sha256"), "failure log sha256")
    return payload


def _existing_incident(
    *,
    workspace: Path,
    run_dir: Path,
    quarantine_root: Path,
    ledger_path: Path,
    log_path: Path,
    gpu_ledger_run_id: str,
    old_contract: str,
    old_contract_path: Path,
    manifest_sha256: str,
    replacement_source: str,
    expected_experiment: str,
) -> dict[str, object] | None:
    run_digest = sha256_bytes(gpu_ledger_run_id.encode("utf-8"))[:16]
    pattern = (
        f"{run_dir.name}-manifest-{manifest_sha256}"
        f"-inventory-*-failure-{run_digest}"
    )
    if not quarantine_root.exists():
        return None
    if quarantine_root.is_symlink() or not quarantine_root.is_dir():
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine root is invalid"
        )
    matches = list(sorted(quarantine_root.glob(pattern)))
    if any(path.is_symlink() or not path.is_dir() for path in matches):
        raise Phase4EBootstrapQuarantineError(
            "bootstrap quarantine destination collision is invalid"
        )
    if len(matches) > 1:
        raise Phase4EBootstrapQuarantineError(
            "multiple bootstrap quarantine incidents match the failed lineage"
        )
    if not matches:
        return None
    incident_path = matches[0] / INCIDENT_NAME
    payload = verify_quarantine_incident(
        incident_path,
        workspace_root=workspace,
        original_run_dir=run_dir,
        quarantine_root=quarantine_root,
        gpu_ledger_path=ledger_path,
        failure_log_path=log_path,
        gpu_ledger_run_id=gpu_ledger_run_id,
        old_execution_contract_sha256=old_contract,
        old_execution_contract_path=old_contract_path,
        run_manifest_sha256=manifest_sha256,
        replacement_source_sha256=replacement_source,
        expected_experiment=expected_experiment,
    )
    return _reference(incident_path, workspace=workspace, payload=payload)


def inspect_failed_bootstrap_run(
    *,
    workspace_root: str | Path,
    original_run_dir: str | Path,
    quarantine_root: str | Path,
    gpu_ledger_path: str | Path,
    failure_log_path: str | Path,
    gpu_ledger_run_id: str,
    old_execution_contract_sha256: str,
    run_manifest_sha256: str,
    replacement_source_sha256: str,
    old_execution_contract_path: str | Path = (
        DEFAULT_OLD_EXECUTION_CONTRACT
    ),
    expected_experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, object]:
    """Read-only validation and deterministic quarantine plan."""

    if not isinstance(gpu_ledger_run_id, str) or not gpu_ledger_run_id:
        raise Phase4EBootstrapQuarantineError(
            "gpu_ledger_run_id must be a non-empty string"
        )
    if not isinstance(expected_experiment, str) or not expected_experiment:
        raise Phase4EBootstrapQuarantineError(
            "expected_experiment must be a non-empty string"
        )
    workspace = Path(workspace_root).resolve()
    run_dir = _workspace_path(
        workspace, original_run_dir, label="original bootstrap run"
    )
    quarantine = _workspace_path(
        workspace, quarantine_root, label="bootstrap quarantine root"
    )
    ledger_path = _workspace_path(
        workspace, gpu_ledger_path, label="Phase 4 GPU ledger"
    )
    log_path = _workspace_path(
        workspace, failure_log_path, label="bootstrap failure log"
    )
    contract_path = _workspace_path(
        workspace,
        old_execution_contract_path,
        label="frozen Phase 4E execution contract",
    )
    old_contract = _sha256(
        old_execution_contract_sha256, "old execution contract sha256"
    )
    manifest_sha256 = _sha256(
        run_manifest_sha256, "run manifest sha256"
    )
    replacement_source = _sha256(
        replacement_source_sha256, "replacement source sha256"
    )
    if fingerprint_sources(workspace).sha256 != replacement_source:
        raise Phase4EBootstrapQuarantineError(
            "replacement source does not match the current workspace"
        )
    _execution_contract_evidence(
        contract_path, expected_sha256=old_contract
    )
    if run_dir.exists():
        existing = _existing_incident(
            workspace=workspace,
            run_dir=run_dir,
            quarantine_root=quarantine,
            ledger_path=ledger_path,
            log_path=log_path,
            gpu_ledger_run_id=gpu_ledger_run_id,
            old_contract=old_contract,
            old_contract_path=contract_path,
            manifest_sha256=manifest_sha256,
            replacement_source=replacement_source,
            expected_experiment=expected_experiment,
        )
        if existing is not None:
            raise Phase4EBootstrapQuarantineError(
                "original bootstrap run and quarantine destination both exist"
            )
        inventory = _validate_no_model_progress(run_dir)
        manifest, original_source = _manifest_evidence(
            run_dir / "run-manifest.json",
            run_dir=run_dir,
            expected_manifest_sha256=manifest_sha256,
            expected_contract_sha256=old_contract,
            expected_experiment=expected_experiment,
        )
        if original_source == replacement_source:
            raise Phase4EBootstrapQuarantineError(
                "replacement source must differ from the failed lineage source"
            )
        _validate_cache_source_binding(
            run_dir, expected_source_sha256=original_source
        )
        ledger, _ = _ledger_evidence(
            ledger_path, gpu_ledger_run_id=gpu_ledger_run_id
        )
        failure_log, failure_log_bytes = _log_evidence(
            log_path,
            workspace=workspace,
            failure_marker=DEFAULT_FAILURE_MARKER,
        )
        _resource_preflight(
            ledger_path.parent,
            reserve_bytes=(
                2
                * (
                    len(failure_log_bytes)
                    + int(ledger["bytes"])
                    + len(canonical_json_bytes(inventory))
                )
                + _QUARANTINE_WRITE_RESERVE_BYTES
            ),
        )
        inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
        destination = quarantine / _destination_name(
            run_dir,
            manifest_sha256=manifest_sha256,
            inventory_sha256=inventory_sha256,
            gpu_ledger_run_id=gpu_ledger_run_id,
        )
        if os.path.lexists(destination):
            raise Phase4EBootstrapQuarantineError(
                "bootstrap quarantine destination already exists"
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "status": "ready",
            "dryRun": True,
            "originalRunPath": _relative(
                run_dir, workspace, "original bootstrap run"
            ),
            "quarantinePath": _relative(
                destination, workspace, "bootstrap quarantine destination"
            ),
            "oldExecutionContractSha256": old_contract,
            "runManifestSha256": manifest["manifestSha256"],
            "originalSourceSha256": original_source,
            "replacementSourceSha256": replacement_source,
            "gpuLedgerRunId": gpu_ledger_run_id,
            "gpuLedgerStatus": "failed",
            "gpuLedgerAccountedSeconds": ledger["failedRun"]["entry"][
                "accountedSeconds"
            ],
            "failureLogSha256": failure_log["sha256"],
            "inventorySha256": inventory_sha256,
            "directories": inventory["directories"],
            "files": inventory["files"],
        }
    existing = _existing_incident(
        workspace=workspace,
        run_dir=run_dir,
        quarantine_root=quarantine,
        ledger_path=ledger_path,
        log_path=log_path,
        gpu_ledger_run_id=gpu_ledger_run_id,
        old_contract=old_contract,
        old_contract_path=contract_path,
        manifest_sha256=manifest_sha256,
        replacement_source=replacement_source,
        expected_experiment=expected_experiment,
    )
    if existing is None:
        raise Phase4EBootstrapQuarantineError(
            "original bootstrap run and matching quarantine are both missing"
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "already-quarantined",
        "dryRun": True,
        "reference": existing,
    }


def _commit_exact(path: Path, payload: bytes, label: str) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise Phase4EBootstrapQuarantineError(
                f"existing {label} differs from the sealed evidence"
            )
        return
    try:
        atomic_create_bytes(path, payload)
    except (OSError, RunManifestError) as error:
        raise Phase4EBootstrapQuarantineError(
            f"could not seal {label}"
        ) from error


def quarantine_failed_bootstrap_run(
    *,
    workspace_root: str | Path,
    original_run_dir: str | Path,
    quarantine_root: str | Path,
    gpu_ledger_path: str | Path,
    failure_log_path: str | Path,
    gpu_ledger_run_id: str,
    old_execution_contract_sha256: str,
    run_manifest_sha256: str,
    replacement_source_sha256: str,
    old_execution_contract_path: str | Path = (
        DEFAULT_OLD_EXECUTION_CONTRACT
    ),
    expected_experiment: str = DEFAULT_EXPERIMENT,
    lock_path: str | Path | None = None,
) -> dict[str, object]:
    """Seal and atomically quarantine an empty, terminally failed lineage."""

    workspace = Path(workspace_root).resolve()
    run_dir = _workspace_path(
        workspace, original_run_dir, label="original bootstrap run"
    )
    quarantine = _workspace_path(
        workspace, quarantine_root, label="bootstrap quarantine root"
    )
    ledger_path = _workspace_path(
        workspace, gpu_ledger_path, label="Phase 4 GPU ledger"
    )
    log_path = _workspace_path(
        workspace, failure_log_path, label="bootstrap failure log"
    )
    contract_path = _workspace_path(
        workspace,
        old_execution_contract_path,
        label="frozen Phase 4E execution contract",
    )
    control_root = ledger_path.parent
    lock = _workspace_path(
        workspace,
        lock_path if lock_path is not None else control_root / GPU_LOCK_NAME,
        label="Phase 4 writer lock",
    )
    if os.path.lexists(lock) and (
        lock.is_symlink() or not lock.is_file()
    ):
        raise Phase4EBootstrapQuarantineError(
            "Phase 4 writer lock is not a regular file"
        )
    quarantine_lock_id = (
        f"phase4e-bootstrap-quarantine-{os.getpid()}-{time.time_ns()}"
    )
    with RunLock(lock, run_id=quarantine_lock_id):
        incident_exists = os.path.lexists(run_dir / INCIDENT_NAME)
        preserved_log_exists = os.path.lexists(
            run_dir / PRESERVED_LOG_NAME
        )
        if incident_exists or preserved_log_exists:
            if not run_dir.is_dir() or (
                incident_exists != preserved_log_exists
            ):
                raise Phase4EBootstrapQuarantineError(
                    "partially sealed bootstrap quarantine evidence is incomplete"
                )
            old_contract = _sha256(
                old_execution_contract_sha256,
                "old execution contract sha256",
            )
            expected_manifest = _sha256(
                run_manifest_sha256, "run manifest sha256"
            )
            replacement_source = _sha256(
                replacement_source_sha256, "replacement source sha256"
            )
            presealed = _verify_presealed_incident(
                workspace=workspace,
                run_dir=run_dir,
                quarantine_root=quarantine,
                ledger_path=ledger_path,
                log_path=log_path,
                contract_path=contract_path,
                gpu_ledger_run_id=gpu_ledger_run_id,
                old_contract=old_contract,
                manifest_sha256=expected_manifest,
                replacement_source=replacement_source,
                expected_experiment=expected_experiment,
            )
            destination = workspace / str(presealed["quarantinePath"])
            if os.path.lexists(destination):
                raise Phase4EBootstrapQuarantineError(
                    "bootstrap quarantine destination already exists"
                )
            quarantine.mkdir(parents=True, exist_ok=True)
            if run_dir.stat().st_dev != quarantine.stat().st_dev:
                raise Phase4EBootstrapQuarantineError(
                    "bootstrap run and quarantine root are on different filesystems"
                )
            try:
                os.replace(run_dir, destination)
            except OSError as error:
                raise Phase4EBootstrapQuarantineError(
                    "could not atomically quarantine failed bootstrap run"
                ) from error
            _fsync_quarantine_rename_parents(
                original_run_dir=run_dir,
                quarantine_destination=destination,
            )
            verified = verify_quarantine_incident(
                destination / INCIDENT_NAME,
                workspace_root=workspace,
                original_run_dir=run_dir,
                quarantine_root=quarantine,
                gpu_ledger_path=ledger_path,
                failure_log_path=log_path,
                gpu_ledger_run_id=gpu_ledger_run_id,
                old_execution_contract_sha256=old_contract,
                run_manifest_sha256=expected_manifest,
                replacement_source_sha256=replacement_source,
                old_execution_contract_path=contract_path,
                expected_experiment=expected_experiment,
            )
            return _reference(
                destination / INCIDENT_NAME,
                workspace=workspace,
                payload=verified,
            )
        plan = inspect_failed_bootstrap_run(
            workspace_root=workspace,
            original_run_dir=run_dir,
            quarantine_root=quarantine,
            gpu_ledger_path=ledger_path,
            failure_log_path=log_path,
            gpu_ledger_run_id=gpu_ledger_run_id,
            old_execution_contract_sha256=old_execution_contract_sha256,
            run_manifest_sha256=run_manifest_sha256,
            replacement_source_sha256=replacement_source_sha256,
            old_execution_contract_path=contract_path,
            expected_experiment=expected_experiment,
        )
        if plan["status"] == "already-quarantined":
            incident_path = workspace / str(plan["reference"]["path"])
            _fsync_quarantine_rename_parents(
                original_run_dir=run_dir,
                quarantine_destination=incident_path.parent,
            )
            return dict(plan["reference"])

        inventory = {
            "directories": list(plan["directories"]),
            "files": list(plan["files"]),
        }
        manifest, original_source = _manifest_evidence(
            run_dir / "run-manifest.json",
            run_dir=run_dir,
            expected_manifest_sha256=plan["runManifestSha256"],
            expected_contract_sha256=plan["oldExecutionContractSha256"],
            expected_experiment=expected_experiment,
        )
        ledger, ledger_bytes = _ledger_evidence(
            ledger_path, gpu_ledger_run_id=gpu_ledger_run_id
        )
        failure_log, log_bytes = _log_evidence(
            log_path,
            workspace=workspace,
            failure_marker=DEFAULT_FAILURE_MARKER,
        )
        destination = workspace / str(plan["quarantinePath"])
        quarantine.mkdir(parents=True, exist_ok=True)
        if quarantine.is_symlink() or not quarantine.is_dir():
            raise Phase4EBootstrapQuarantineError(
                "bootstrap quarantine root is invalid"
            )
        if run_dir.stat().st_dev != quarantine.stat().st_dev:
            raise Phase4EBootstrapQuarantineError(
                "bootstrap run and quarantine root are on different filesystems"
            )
        contract_evidence = _execution_contract_evidence(
            contract_path,
            expected_sha256=plan["oldExecutionContractSha256"],
        )
        incident_core = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": INCIDENT_KIND,
            "status": "sealed",
            "reason": FAILURE_REASON,
            "originalRunPath": plan["originalRunPath"],
            "quarantinePath": plan["quarantinePath"],
            "oldExecutionContractSha256": plan[
                "oldExecutionContractSha256"
            ],
            "oldExecutionContract": {
                **contract_evidence,
                "path": _relative(
                    contract_path,
                    workspace,
                    "frozen Phase 4E execution contract",
                ),
            },
            "runManifestSha256": plan["runManifestSha256"],
            "originalSourceSha256": original_source,
            "replacementSourceSha256": plan[
                "replacementSourceSha256"
            ],
            "gpuLedgerRunId": gpu_ledger_run_id,
            "gpuLedgerStatus": "failed",
            "gpuLedgerAccountedSeconds": ledger["failedRun"]["entry"][
                "accountedSeconds"
            ],
            "originalRunManifest": manifest,
            "gpuLedgerSnapshot": {
                **ledger,
                "path": _relative(
                    ledger_path, workspace, "Phase 4 GPU ledger"
                ),
            },
            "failureLog": failure_log,
            "directories": inventory["directories"],
            "files": inventory["files"],
            "inventorySha256": plan["inventorySha256"],
        }
        incident = _self_hashed(
            incident_core, field="incidentSha256"
        )
        incident_bytes = canonical_json_bytes(incident)
        _resource_preflight(
            control_root,
            reserve_bytes=(
                2 * (len(log_bytes) + len(incident_bytes))
                + _QUARANTINE_WRITE_RESERVE_BYTES
            ),
        )
        _commit_exact(
            run_dir / PRESERVED_LOG_NAME,
            log_bytes,
            "bootstrap failure-log snapshot",
        )
        _commit_exact(
            run_dir / INCIDENT_NAME,
            incident_bytes,
            "bootstrap quarantine incident",
        )
        if (
            ledger_path.read_bytes() != ledger_bytes
            or log_path.read_bytes() != log_bytes
            or fingerprint_sources(workspace).sha256
            != plan["replacementSourceSha256"]
        ):
            raise Phase4EBootstrapQuarantineError(
                "source, GPU ledger, or failure log changed while sealing quarantine"
            )
        try:
            os.replace(run_dir, destination)
        except OSError as error:
            raise Phase4EBootstrapQuarantineError(
                "could not atomically quarantine failed bootstrap run"
            ) from error
        _fsync_quarantine_rename_parents(
            original_run_dir=run_dir,
            quarantine_destination=destination,
        )
        incident_path = destination / INCIDENT_NAME
        verified = verify_quarantine_incident(
            incident_path,
            workspace_root=workspace,
            original_run_dir=run_dir,
            quarantine_root=quarantine,
            gpu_ledger_path=ledger_path,
            failure_log_path=log_path,
            gpu_ledger_run_id=gpu_ledger_run_id,
            old_execution_contract_sha256=plan[
                "oldExecutionContractSha256"
            ],
            old_execution_contract_path=contract_path,
            run_manifest_sha256=plan["runManifestSha256"],
            replacement_source_sha256=plan[
                "replacementSourceSha256"
            ],
            expected_experiment=expected_experiment,
        )
        if (
            ledger_path.read_bytes() != ledger_bytes
            or log_path.read_bytes() != log_bytes
            or fingerprint_sources(workspace).sha256
            != plan["replacementSourceSha256"]
        ):
            raise Phase4EBootstrapQuarantineError(
                "source, GPU ledger, or failure log changed during atomic quarantine"
            )
        return _reference(
            incident_path, workspace=workspace, payload=verified
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or execute the CPU-only quarantine of a failed "
            "Phase 4E bootstrap lineage"
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--quarantine-root", default=str(DEFAULT_QUARANTINE_ROOT)
    )
    parser.add_argument("--gpu-ledger", default=str(DEFAULT_GPU_LEDGER))
    parser.add_argument("--failure-log", required=True)
    parser.add_argument("--gpu-ledger-run-id", required=True)
    parser.add_argument("--old-execution-contract-sha256", required=True)
    parser.add_argument(
        "--old-execution-contract",
        default=str(DEFAULT_OLD_EXECUTION_CONTRACT),
    )
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--replacement-source-sha256", required=True)
    parser.add_argument("--expected-experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Seal and atomically move the run; omission is read-only dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    kwargs = {
        "workspace_root": args.workspace_root,
        "original_run_dir": args.run_dir,
        "quarantine_root": args.quarantine_root,
        "gpu_ledger_path": args.gpu_ledger,
        "failure_log_path": args.failure_log,
        "gpu_ledger_run_id": args.gpu_ledger_run_id,
        "old_execution_contract_sha256": (
            args.old_execution_contract_sha256
        ),
        "old_execution_contract_path": args.old_execution_contract,
        "run_manifest_sha256": args.run_manifest_sha256,
        "replacement_source_sha256": args.replacement_source_sha256,
        "expected_experiment": args.expected_experiment,
    }
    payload = (
        quarantine_failed_bootstrap_run(**kwargs)
        if args.execute
        else inspect_failed_bootstrap_run(**kwargs)
    )
    sys.stdout.buffer.write(canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FAILURE_MARKER",
    "DEFAULT_QUARANTINE_ROOT",
    "INCIDENT_KIND",
    "INCIDENT_NAME",
    "PRESERVED_LOG_NAME",
    "Phase4EBootstrapQuarantineError",
    "inspect_failed_bootstrap_run",
    "quarantine_failed_bootstrap_run",
    "verify_quarantine_incident",
    "verify_quarantine_incident_attestation",
    "verify_quarantine_reference",
]
