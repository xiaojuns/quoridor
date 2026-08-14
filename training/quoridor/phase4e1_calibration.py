"""Phase 4E1 Standard-shaped self-play and batched-arena calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .calibration import (
    CalibrationBudgetExceeded,
    CalibrationScenario,
    EvaluationProfiler,
    GpuBudgetLedger,
    LEDGER_SCHEMA_VERSION,
    NvidiaSmiProfiler,
    PHASE4_INPUT_CHANNELS,
    _has_decision_metrics,
    run_scenario,
)
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .game import State, create_initial_state
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import build_opening_suite, load_opening_suite
from .paired_arena import (
    BoundMoveSelector,
    SelectorIdentity,
    SelectorRuntime,
    loaded_native_backend_hash,
)
from .phase4_arena import (
    OpeningPairStore,
    run_resumable_paired_arena,
    verify_completed_arena_result,
)
from .phase4_baseline import (
    load_bootstrap_baseline_state,
    verify_bootstrap_baseline,
)
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_SCHEMA_VERSION,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionBudget,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4e1_contract import (
    Phase4E1CalibrationContract,
    load_phase4e1_contract,
)
from .replay_store import ReplayGame, ReplayStore
from .run_manifest import (
    ImmutableArtifactExistsError,
    ResourceBudget,
    RunLock,
    RunManifestError,
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
    utc_timestamp,
    validate_sha256,
)
from .selectors import MCTSSelector


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "phase4e1-standard-calibration-and-arena-smoke"
COMPONENT_SCHEMA_VERSION = 1
SELFPLAY_KIND = "phase4e1-standard-self-play-calibration"
ARENA_KIND = "phase4e1-batched-formal-arena-throughput-smoke"
PROGRESS_SCHEMA_VERSION = 1
GPU_LEDGER_NAME = "gpu-budget-ledger.json"
GPU_LOCK_NAME = ".phase4.lock"
DEFAULT_CONTRACT = Path("training/configs/phase4e1-calibration-v1.json")
DEFAULT_OUTPUT_ROOT = Path("training/runs/phase4/phase4e1/calibration")
REPORT_NAME = "report.json"
RUN_MANIFEST_NAME = "run-manifest.json"
SELFPLAY_REPORT_NAME = "selfplay-report.json"
ARENA_REPORT_NAME = "arena-report.json"
ARENA_PROGRESS_NAME = "arena-progress.json"
QUARANTINE_INCIDENT_NAME = "quarantine-incident.json"
QUARANTINE_INCIDENT_SCHEMA_VERSION = 1
QUARANTINE_INCIDENT_KIND = "phase4e1-source-mismatch-quarantine"
FROZEN_EXECUTION_DEFAULT = Path(
    "training/configs/phase4e-execution-frozen-v1.json"
)
HEARTBEAT_SECONDS = 5.0
REPORT_WRITE_RESERVE_BYTES = 8 * 1024**2


class Phase4E1CalibrationError(RuntimeError):
    """Raised when E1 evidence or a resource contract does not verify."""


def _emit_progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "schemaVersion": PROGRESS_SCHEMA_VERSION,
                "kind": "phase4e1-calibration-progress",
                "stage": stage,
                "status": status,
                **fields,
            }
        ).decode(),
        end="",
        flush=True,
    )


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(core)
    payload[field] = sha256_bytes(canonical_json_bytes(core))
    return payload


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    try:
        digest = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise Phase4E1CalibrationError(f"{label} self hash is invalid") from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4E1CalibrationError(f"{label} self hash mismatch")
    return digest


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4E1CalibrationError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4E1CalibrationError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4E1CalibrationError(f"{label} must be canonical JSON")
    return dict(payload)


def _commit_immutable_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    label: str,
) -> None:
    encoded = canonical_json_bytes(payload)
    try:
        atomic_create_bytes(path, encoded)
    except ImmutableArtifactExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4E1CalibrationError(
                f"immutable {label} already exists with different content"
            )


def _workspace_path(
    workspace: Path,
    relative: str,
    *,
    label: str,
    file: bool = True,
) -> Path:
    source = Path(relative)
    if (
        source.is_absolute()
        or not relative
        or any(part in {".", ".."} for part in source.parts)
    ):
        raise Phase4E1CalibrationError(
            f"{label} must be a canonical workspace-relative path"
        )
    path = Path(os.path.abspath(workspace / source))
    try:
        parts = path.relative_to(workspace).parts
    except ValueError as error:
        raise Phase4E1CalibrationError(f"{label} escapes the workspace") from error
    cursor = workspace
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise Phase4E1CalibrationError(f"{label} path contains a symlink")
    if (file and not path.is_file()) or (
        not file and not path.is_dir()
    ):
        kind = "file" if file else "directory"
        raise Phase4E1CalibrationError(f"{label} is not a regular {kind}")
    return path


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4E1CalibrationError("artifact path escapes the workspace") from error


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise Phase4E1CalibrationError(
                f"Phase 4 artifact tree contains a symlink: {item}"
            )
        if item.is_file():
            total += item.stat().st_size
    return total


def _percentile(values: Sequence[float | int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _effective_config_sha256(path: Path, workspace: Path) -> str:
    config = load_experiment_config(path)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return fingerprint_config(config).sha256


def _network_config(config_path: Path) -> NetworkConfig:
    settings = load_experiment_config(config_path).network
    return NetworkConfig(
        input_channels=settings.input_channels,
        channels=settings.channels,
        blocks=settings.residual_blocks,
        se_reduction=settings.se_reduction,
        policy_channels=settings.policy_channels,
        value_channels=settings.value_channels,
        value_hidden=settings.value_hidden,
    )


def _model(
    config_path: Path,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> PolicyValueNetwork:
    model = PolicyValueNetwork(_network_config(config_path))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4E1CalibrationError(
            "bootstrap baseline does not match the Standard network"
        ) from error
    return model.to(device).eval()


def _component_reference(path: Path, workspace: Path) -> dict[str, object]:
    payload = _load_canonical(path, path.name)
    digest = _verify_self_hash(
        payload,
        field="reportSha256",
        label=path.name,
    )
    return {
        "path": _relative(path, workspace),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "reportSha256": digest,
    }


def _validate_reference(
    reference: object,
    *,
    workspace: Path,
    label: str,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(reference, Mapping):
        raise Phase4E1CalibrationError(f"{label} reference is invalid")
    if set(reference) != {"path", "bytes", "sha256", "reportSha256"}:
        raise Phase4E1CalibrationError(f"{label} reference fields are invalid")
    path = _workspace_path(
        workspace,
        str(reference.get("path")),
        label=label,
    )
    try:
        digest = validate_sha256(reference.get("sha256"), f"{label} sha256")
        report_digest = validate_sha256(
            reference.get("reportSha256"),
            f"{label} reportSha256",
        )
    except ValueError as error:
        raise Phase4E1CalibrationError(f"{label} hashes are invalid") from error
    if (
        type(reference.get("bytes")) is not int
        or reference.get("bytes") != path.stat().st_size
        or sha256_file(path) != digest
    ):
        raise Phase4E1CalibrationError(f"{label} artifact does not verify")
    payload = _load_canonical(path, label)
    if _verify_self_hash(
        payload, field="reportSha256", label=label
    ) != report_digest:
        raise Phase4E1CalibrationError(f"{label} self hash differs from reference")
    return path, payload


class _ResourceGuard:
    def __init__(
        self,
        *,
        control_root: Path,
        contract: Phase4E1CalibrationContract,
        ledger: GpuBudgetLedger,
        ledger_run_id: str,
        started: float,
        global_at_start: float,
    ) -> None:
        self.control_root = control_root
        self.contract = contract
        self.ledger = ledger
        self.ledger_run_id = ledger_run_id
        self.started = started
        self.global_at_start = global_at_start
        self.last_heartbeat = started

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def estimated_global_hours(self) -> float:
        return self.global_at_start + self.elapsed_seconds / 3600.0

    def check(self, *, force: bool = False) -> dict[str, object]:
        now = time.monotonic()
        global_hours = self.estimated_global_hours
        resources = self.contract.resources
        used = global_hours - resources.initial_ledger.total_gpu_hours
        if global_hours >= resources.global_gpu_hour_limit:
            raise CalibrationBudgetExceeded("72 GPU-hour global budget reached")
        if used >= resources.calibration_gpu_hour_limit:
            raise CalibrationBudgetExceeded("Phase 4E1 6 GPU-hour limit reached")
        artifact_bytes = _directory_size(self.control_root)
        free_bytes = int(shutil.disk_usage(self.control_root).free)
        if artifact_bytes + REPORT_WRITE_RESERVE_BYTES >= resources.artifact_limit_bytes:
            raise CalibrationBudgetExceeded("Phase 4 artifact limit reached")
        if (
            free_bytes - REPORT_WRITE_RESERVE_BYTES
            < resources.minimum_free_disk_bytes
        ):
            raise CalibrationBudgetExceeded("20 GiB free-disk stop line reached")
        if force or now - self.last_heartbeat >= HEARTBEAT_SECONDS:
            self.ledger.update(
                self.ledger_run_id,
                elapsed_seconds=self.elapsed_seconds,
                status="running",
            )
            self.last_heartbeat = now
        return {
            "globalGpuHours": global_hours,
            "phase4e1GpuHours": used,
            "artifactBytes": artifact_bytes,
            "freeDiskBytes": free_bytes,
        }


def _recorded_e1_work_seconds(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> float:
    run_id = f"phase4e1-calibration-{contract.sha256[:16]}"
    run_root = workspace / DEFAULT_OUTPUT_ROOT / run_id
    seconds = 0.0
    selfplay_path = run_root / SELFPLAY_REPORT_NAME
    if selfplay_path.exists():
        payload = _load_canonical(selfplay_path, "E1 self-play ledger evidence")
        _verify_self_hash(
            payload,
            field="reportSha256",
            label="E1 self-play ledger evidence",
        )
        result = payload.get("result")
        elapsed = (
            result.get("elapsedSeconds")
            if isinstance(result, Mapping)
            else None
        )
        if (
            payload.get("calibrationContractSha256") != contract.sha256
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
        ):
            raise Phase4E1CalibrationError(
                "E1 self-play ledger evidence is invalid"
            )
        seconds += float(elapsed)
    progress_path = run_root / ARENA_PROGRESS_NAME
    if progress_path.exists():
        progress = _load_canonical(
            progress_path,
            "E1 arena ledger evidence",
        )
        _verify_self_hash(
            progress,
            field="progressSha256",
            label="E1 arena ledger evidence",
        )
        elapsed = progress.get("cumulativeSeconds")
        if (
            progress.get("schemaVersion") != PROGRESS_SCHEMA_VERSION
            or progress.get("calibrationContractSha256") != contract.sha256
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise Phase4E1CalibrationError(
                "E1 arena ledger evidence is invalid"
            )
        seconds += float(elapsed)
    return seconds


def _ledger_snapshot_evidence(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> dict[str, object]:
    ledger_path = _workspace_path(
        workspace,
        contract.resources.initial_ledger.path,
        label="GPU ledger snapshot",
    )
    GpuBudgetLedger(ledger_path)
    return {
        "path": _relative(ledger_path, workspace),
        "snapshot": _load_canonical(ledger_path, "GPU ledger snapshot"),
    }


def _verify_ledger_snapshot_evidence(
    value: object,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    reported_global_hours: float,
    verify_current: bool,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "snapshot"}:
        raise Phase4E1CalibrationError("E1 GPU ledger evidence is invalid")
    if value.get("path") != contract.resources.initial_ledger.path:
        raise Phase4E1CalibrationError("E1 GPU ledger path changed")
    snapshot = value.get("snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
        "schemaVersion",
        "limitHours",
        "externalHoursFloor",
        "runs",
        "updatedAt",
        "ledgerSha256",
    }:
        raise Phase4E1CalibrationError("E1 GPU ledger snapshot is invalid")
    snapshot_core = dict(snapshot)
    snapshot_sha256 = snapshot_core.pop("ledgerSha256", None)
    try:
        snapshot_sha256 = validate_sha256(
            snapshot_sha256,
            "E1 GPU ledger snapshot sha256",
        )
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "E1 GPU ledger snapshot hash is invalid"
        ) from error
    external = snapshot.get("externalHoursFloor")
    runs = snapshot.get("runs")
    if (
        sha256_bytes(canonical_json_bytes(snapshot_core)) != snapshot_sha256
        or snapshot.get("schemaVersion") != LEDGER_SCHEMA_VERSION
        or snapshot.get("limitHours")
        != contract.resources.global_gpu_hour_limit
        or not isinstance(external, (int, float))
        or isinstance(external, bool)
        or not math.isfinite(float(external))
        or float(external) < 0
        or not isinstance(runs, Mapping)
        or not isinstance(snapshot.get("updatedAt"), str)
        or not snapshot.get("updatedAt")
    ):
        raise Phase4E1CalibrationError("E1 GPU ledger snapshot does not verify")
    snapshot_seconds = 0.0
    normalized_runs: dict[str, Mapping[str, object]] = {}
    for run_id, run in runs.items():
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(run, Mapping)
            or set(run)
            != {"startedAt", "updatedAt", "accountedSeconds", "status"}
            or not isinstance(run.get("startedAt"), str)
            or not run.get("startedAt")
            or not isinstance(run.get("updatedAt"), str)
            or not run.get("updatedAt")
            or not isinstance(run.get("accountedSeconds"), (int, float))
            or isinstance(run.get("accountedSeconds"), bool)
            or not math.isfinite(float(run.get("accountedSeconds")))
            or float(run.get("accountedSeconds")) < 0
            or run.get("status")
            not in {
                "running",
                "completed",
                "interrupted",
                "failed",
                "abandoned",
            }
        ):
            raise Phase4E1CalibrationError(
                "E1 GPU ledger snapshot run is invalid"
            )
        snapshot_seconds += float(run["accountedSeconds"])
        normalized_runs[run_id] = run
    snapshot_total = float(external) + snapshot_seconds / 3600.0
    if snapshot_total != reported_global_hours:
        raise Phase4E1CalibrationError(
            "E1 report GPU total differs from its ledger snapshot"
        )
    if not verify_current:
        return
    current_path = _workspace_path(
        workspace,
        str(value["path"]),
        label="current GPU ledger",
    )
    current = GpuBudgetLedger(current_path)
    if (
        current.external_hours_floor < float(external)
        or current.total_hours < snapshot_total
    ):
        raise Phase4E1CalibrationError(
            "current GPU ledger regressed below the E1 snapshot"
        )
    for run_id, snapshot_run in normalized_runs.items():
        current_run = current.runs.get(run_id)
        if (
            not isinstance(current_run, Mapping)
            or current_run.get("startedAt") != snapshot_run.get("startedAt")
            or float(current_run.get("accountedSeconds", -1))
            < float(snapshot_run["accountedSeconds"])
            or (
                run_id.startswith("phase4e1-calibration-")
                and float(current_run.get("accountedSeconds", -1))
                != float(snapshot_run["accountedSeconds"])
            )
        ):
            raise Phase4E1CalibrationError(
                "current GPU ledger changed an E1 snapshot run"
            )


def _preflight(
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    contract_path: Path,
) -> dict[str, object]:
    resources = contract.resources
    provenance = contract.provenance
    control_root = _workspace_path(
        workspace,
        "training/runs/phase4",
        label="Phase 4 control root",
        file=False,
    )
    prepared_path = _workspace_path(
        workspace,
        provenance.prepared_execution_contract_path,
        label="prepared execution contract",
    )
    prepared = load_phase4_execution_contract(prepared_path)
    if (
        prepared.sha256 != provenance.prepared_execution_contract_sha256
        or prepared.status != "prepared"
        or prepared.budget_frozen
    ):
        raise Phase4E1CalibrationError("prepared execution contract changed")

    bootstrap_config_path = _workspace_path(
        workspace,
        provenance.bootstrap_config_path,
        label="Bootstrap config",
    )
    standard_config_path = _workspace_path(
        workspace,
        provenance.standard_config_path,
        label="Standard config",
    )
    if (
        fingerprint_config(bootstrap_config_path).sha256
        != provenance.bootstrap_config_sha256
        or fingerprint_config(standard_config_path).sha256
        != provenance.standard_config_sha256
    ):
        raise Phase4E1CalibrationError("versioned Phase 4E workload changed")

    baseline_manifest_path = _workspace_path(
        workspace,
        provenance.baseline_manifest_path,
        label="bootstrap baseline manifest",
    )
    baseline_model_path = _workspace_path(
        workspace,
        provenance.baseline_model_path,
        label="bootstrap baseline model",
    )
    baseline_manifest = _load_canonical(
        baseline_manifest_path, "bootstrap baseline manifest"
    )
    if (
        baseline_manifest.get("manifestSha256")
        != provenance.baseline_manifest_sha256
        or sha256_file(baseline_model_path) != provenance.baseline_model_sha256
    ):
        raise Phase4E1CalibrationError("bootstrap baseline hashes changed")
    baseline_evidence = verify_bootstrap_baseline(
        workspace_root=workspace,
        baseline_root=baseline_manifest_path.parent,
    )

    dev_path = _workspace_path(
        workspace,
        provenance.dev_suite_path,
        label="dev opening suite",
    )
    dev = load_opening_suite(dev_path, expected_suite="dev")
    if dev.content_hash != provenance.dev_suite_hash:
        raise Phase4E1CalibrationError("dev opening suite changed")
    arena_suite = build_opening_suite(
        dev.openings[: contract.arena.opening_prefix_count],
        suite="dev",
    )
    if (
        arena_suite.content_hash != contract.arena.suite_hash
        or len(arena_suite.openings) * 2 != contract.arena.games
    ):
        raise Phase4E1CalibrationError("arena smoke suite differs from contract")

    ledger_path = _workspace_path(
        workspace,
        resources.initial_ledger.path,
        label="GPU ledger",
    )
    ledger = GpuBudgetLedger(ledger_path)
    ledger_payload = _load_canonical(ledger_path, "GPU ledger")
    if ledger.total_hours < resources.initial_ledger.total_gpu_hours:
        raise Phase4E1CalibrationError("GPU ledger regressed below E1 baseline")
    e1_runs = {
        run_id: value
        for run_id, value in ledger.runs.items()
        if run_id.startswith("phase4e1-calibration-")
    }
    e1_accounted_seconds = sum(
        float(value["accountedSeconds"]) for value in e1_runs.values()
    )
    recorded_work_seconds = _recorded_e1_work_seconds(workspace, contract)
    if e1_accounted_seconds + 1e-6 < recorded_work_seconds:
        raise Phase4E1CalibrationError(
            "GPU ledger regressed behind durable Phase 4E1 work"
        )
    if (
        ledger.total_hours == resources.initial_ledger.total_gpu_hours
        and ledger_payload.get("ledgerSha256")
        != resources.initial_ledger.ledger_sha256
        and (
            not e1_runs
            or any(
                float(value["accountedSeconds"]) != 0.0
                for value in e1_runs.values()
            )
        )
    ):
        raise Phase4E1CalibrationError("initial GPU ledger snapshot changed")
    if _directory_size(control_root) >= resources.artifact_limit_bytes:
        raise Phase4E1CalibrationError("Phase 4 artifact limit is exhausted")
    if int(shutil.disk_usage(control_root).free) < resources.minimum_free_disk_bytes:
        raise Phase4E1CalibrationError("less than 20 GiB is free")

    return {
        "contractPath": _relative(contract_path, workspace),
        "contractSha256": contract.sha256,
        "preparedExecutionContractSha256": prepared.sha256,
        "bootstrapConfig": {
            "path": _relative(bootstrap_config_path, workspace),
            "semanticSha256": provenance.bootstrap_config_sha256,
            "effectiveSha256": _effective_config_sha256(
                bootstrap_config_path, workspace
            ),
        },
        "standardConfig": {
            "path": _relative(standard_config_path, workspace),
            "semanticSha256": provenance.standard_config_sha256,
            "effectiveSha256": _effective_config_sha256(
                standard_config_path, workspace
            ),
        },
        "bootstrapBaseline": baseline_evidence,
        "arenaSuiteHash": arena_suite.content_hash,
        "arenaSuiteOpenings": len(arena_suite.openings),
        "globalGpuHours": ledger.total_hours,
    }


def _quarantine_root(workspace: Path) -> Path:
    return workspace / DEFAULT_OUTPUT_ROOT.parent / "quarantine"


def _run_tree_inventory(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for item in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if item == root / QUARANTINE_INCIDENT_NAME:
            continue
        if item.is_symlink():
            raise Phase4E1CalibrationError(
                "source-mismatched E1 run contains a symlink"
            )
        if item.is_dir():
            continue
        if not item.is_file():
            raise Phase4E1CalibrationError(
                "source-mismatched E1 run contains a non-regular artifact"
            )
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return files


def _quarantine_manifest_reference(
    path: Path,
    record_sha256: str,
) -> dict[str, object]:
    return {
        "file": RUN_MANIFEST_NAME,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "manifestSha256": record_sha256,
    }


def _verify_quarantine_incident(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    payload = _load_canonical(path, "Phase 4E1 quarantine incident")
    digest = _verify_self_hash(
        payload,
        field="incidentSha256",
        label="Phase 4E1 quarantine incident",
    )
    expected_original = (
        workspace / DEFAULT_OUTPUT_ROOT / run_id
    ).relative_to(workspace).as_posix()
    quarantine_root = _quarantine_root(workspace)
    try:
        path.parent.relative_to(quarantine_root)
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident is outside the quarantine root"
        ) from error
    manifest = payload.get("originalRunManifest")
    ledger = payload.get("ledgerSnapshot")
    files = payload.get("files")
    if (
        set(payload)
        != {
            "schemaVersion",
            "kind",
            "status",
            "reason",
            "runId",
            "calibrationContractSha256",
            "nativeBackendSha256",
            "originalSourceSha256",
            "replacementSourceSha256",
            "originalRunPath",
            "quarantinePath",
            "originalRunManifest",
            "ledgerSnapshot",
            "files",
            "incidentSha256",
        }
        or payload.get("schemaVersion") != QUARANTINE_INCIDENT_SCHEMA_VERSION
        or payload.get("kind") != QUARANTINE_INCIDENT_KIND
        or payload.get("status") != "sealed"
        or payload.get("reason") != "run-manifest-source-mismatch"
        or payload.get("runId") != run_id
        or payload.get("calibrationContractSha256") != contract.sha256
        or payload.get("nativeBackendSha256") != native_sha256
        or payload.get("replacementSourceSha256") != source_sha256
        or payload.get("originalRunPath") != expected_original
        or payload.get("quarantinePath")
        != path.parent.relative_to(workspace).as_posix()
        or not isinstance(manifest, Mapping)
        or set(manifest)
        != {"file", "bytes", "sha256", "manifestSha256"}
        or not isinstance(ledger, Mapping)
        or set(ledger)
        != {
            "path",
            "bytes",
            "sha256",
            "ledgerSha256",
            "e1Runs",
        }
        or not isinstance(files, list)
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident contract mismatch"
        )
    try:
        original_source = validate_sha256(
            payload.get("originalSourceSha256"),
            "quarantine original source sha256",
        )
        manifest_file_sha256 = validate_sha256(
            manifest.get("sha256"),
            "quarantine manifest file sha256",
        )
        manifest_self_sha256 = validate_sha256(
            manifest.get("manifestSha256"),
            "quarantine manifest self sha256",
        )
        validate_sha256(
            ledger.get("sha256"),
            "quarantine ledger file sha256",
        )
        validate_sha256(
            ledger.get("ledgerSha256"),
            "quarantine ledger self sha256",
        )
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident hashes are invalid"
        ) from error
    expected_destination_name = (
        f"{run_id}-source-{original_source}-manifest-{manifest_self_sha256}"
    )
    if path.parent.name != expected_destination_name:
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine directory is not content-addressed"
        )
    manifest_path = path.parent / RUN_MANIFEST_NAME
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest.get("file") != RUN_MANIFEST_NAME
        or type(manifest.get("bytes")) is not int
        or manifest.get("bytes") != manifest_path.stat().st_size
        or sha256_file(manifest_path) != manifest_file_sha256
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantined run manifest reference is invalid"
        )
    _verify_e1_run_manifest(
        manifest_path,
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        expected_sha256=manifest_self_sha256,
        source_sha256=original_source,
        native_sha256=native_sha256,
    )
    ledger_runs = ledger.get("e1Runs")
    if (
        ledger.get("path")
        != (workspace / contract.resources.initial_ledger.path)
        .relative_to(workspace)
        .as_posix()
        or type(ledger.get("bytes")) is not int
        or ledger.get("bytes") < 0
        or not isinstance(ledger_runs, Mapping)
        or any(
            not isinstance(key, str)
            or not isinstance(value, Mapping)
            or value.get("status") == "running"
            for key, value in ledger_runs.items()
        )
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine ledger snapshot is invalid"
        )
    if files != _run_tree_inventory(path.parent):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantined artifact inventory changed"
        )
    if digest != payload.get("incidentSha256"):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident digest changed"
        )
    return payload


def _quarantine_incident_reference(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    payload = _verify_quarantine_incident(
        path,
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    return {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "incidentSha256": payload["incidentSha256"],
    }


def _verify_quarantine_incident_reference(
    reference: object,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    if (
        not isinstance(reference, Mapping)
        or set(reference) != {"path", "bytes", "sha256", "incidentSha256"}
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident reference is invalid"
        )
    path = _workspace_path(
        workspace,
        str(reference.get("path")),
        label="Phase 4E1 quarantine incident",
    )
    try:
        file_sha256 = validate_sha256(
            reference.get("sha256"),
            "Phase 4E1 quarantine incident file sha256",
        )
        incident_sha256 = validate_sha256(
            reference.get("incidentSha256"),
            "Phase 4E1 quarantine incident self sha256",
        )
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident reference hashes are invalid"
        ) from error
    payload = _verify_quarantine_incident(
        path,
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    if (
        type(reference.get("bytes")) is not int
        or reference.get("bytes") != path.stat().st_size
        or sha256_file(path) != file_sha256
        or payload.get("incidentSha256") != incident_sha256
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine incident reference changed"
        )
    return dict(reference)


def _e1_run_metadata(
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    source_sha256: str,
    native_sha256: str,
    quarantine_incident: object = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "phase": "4E1",
        "kind": REPORT_KIND,
        "calibrationContractSha256": contract.sha256,
        "nativeBackendSha256": native_sha256,
    }
    if quarantine_incident is not None:
        metadata["quarantineIncident"] = (
            _verify_quarantine_incident_reference(
                quarantine_incident,
                workspace=workspace,
                contract=contract,
                run_id=run_id,
                source_sha256=source_sha256,
                native_sha256=native_sha256,
            )
        )
    return metadata


def _existing_quarantine_incident(
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object] | None:
    root = _quarantine_root(workspace)
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine root is not a regular directory"
        )
    matches: list[dict[str, object]] = []
    for directory in sorted(root.glob(f"{run_id}-source-*-manifest-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise Phase4E1CalibrationError(
                "Phase 4E1 quarantine contains an invalid run directory"
            )
        incident_path = directory / QUARANTINE_INCIDENT_NAME
        payload = _load_canonical(
            incident_path,
            "existing Phase 4E1 quarantine incident",
        )
        _verify_self_hash(
            payload,
            field="incidentSha256",
            label="existing Phase 4E1 quarantine incident",
        )
        if payload.get("replacementSourceSha256") != source_sha256:
            continue
        matches.append(
            _quarantine_incident_reference(
                incident_path,
                workspace=workspace,
                contract=contract,
                run_id=run_id,
                source_sha256=source_sha256,
                native_sha256=native_sha256,
            )
        )
    if len(matches) > 1:
        raise Phase4E1CalibrationError(
            "multiple Phase 4E1 quarantine incidents target current source"
        )
    return None if not matches else matches[0]


def _quarantine_source_mismatched_run(
    *,
    workspace: Path,
    control_root: Path,
    run_root: Path,
    run_id: str,
    contract: Phase4E1CalibrationContract,
    native_sha256: str,
) -> dict[str, object] | None:
    if run_root.is_symlink() or not run_root.is_dir():
        raise Phase4E1CalibrationError(
            "Phase 4E1 run root is not a regular directory"
        )
    manifest_path = run_root / RUN_MANIFEST_NAME
    if not os.path.lexists(manifest_path):
        if any(run_root.iterdir()):
            raise Phase4E1CalibrationError(
                "Phase 4E1 run root has artifacts but no run manifest"
            )
        quarantine_root = _quarantine_root(workspace)
        if not quarantine_root.exists():
            return None
        current_source_sha256 = fingerprint_sources(workspace).sha256
        return _existing_quarantine_incident(
            workspace=workspace,
            contract=contract,
            run_id=run_id,
            source_sha256=current_source_sha256,
            native_sha256=native_sha256,
        )
    try:
        record = load_run_manifest(manifest_path)
    except (OSError, ValueError, RunManifestError) as error:
        raise Phase4E1CalibrationError(
            "source-mismatched E1 run manifest does not verify"
        ) from error
    if manifest_path.read_bytes() != canonical_json_bytes(record.payload):
        raise Phase4E1CalibrationError(
            "source-mismatched E1 run manifest is not canonical JSON"
        )
    provenance = record.payload.get("provenance")
    source = (
        provenance.get("source") if isinstance(provenance, Mapping) else None
    )
    try:
        original_source_sha256 = validate_sha256(
            source.get("sha256") if isinstance(source, Mapping) else None,
            "source-mismatched E1 source sha256",
        )
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "source-mismatched E1 run source is invalid"
        ) from error
    current_source_sha256 = fingerprint_sources(workspace).sha256
    if original_source_sha256 == current_source_sha256:
        return None
    if os.path.lexists(run_root / REPORT_NAME):
        raise Phase4E1CalibrationError(
            "completed Phase 4E1 evidence cannot be quarantined"
        )
    _verify_e1_run_manifest(
        manifest_path,
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        expected_sha256=record.sha256,
        source_sha256=original_source_sha256,
        native_sha256=native_sha256,
    )

    ledger_path = control_root / GPU_LEDGER_NAME
    expected_ledger_path = workspace / contract.resources.initial_ledger.path
    if (
        ledger_path != expected_ledger_path
        or ledger_path.is_symlink()
        or not ledger_path.is_file()
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine GPU ledger path is invalid"
        )
    ledger_bytes = ledger_path.read_bytes()
    ledger_payload = _load_canonical(
        ledger_path,
        "Phase 4E1 quarantine GPU ledger",
    )
    try:
        ledger_self_sha256 = validate_sha256(
            ledger_payload.get("ledgerSha256"),
            "Phase 4E1 quarantine GPU ledger self sha256",
        )
        ledger = GpuBudgetLedger(ledger_path)
    except (OSError, ValueError, RuntimeError) as error:
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine GPU ledger does not verify"
        ) from error
    e1_runs = {
        key: dict(value)
        for key, value in sorted(ledger.runs.items())
        if key.startswith("phase4e1-calibration-")
    }
    if any(value.get("status") == "running" for value in e1_runs.values()):
        raise Phase4E1CalibrationError(
            "running Phase 4E1 GPU ledger entry blocks quarantine"
        )

    quarantine_root = _quarantine_root(workspace)
    if quarantine_root.is_symlink() or (
        quarantine_root.exists() and not quarantine_root.is_dir()
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine root is invalid"
        )
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / (
        f"{run_id}-source-{original_source_sha256}-manifest-{record.sha256}"
    )
    if os.path.lexists(destination):
        raise Phase4E1CalibrationError(
            "Phase 4E1 quarantine destination already exists"
        )
    inventory = _run_tree_inventory(run_root)
    manifest_reference = _quarantine_manifest_reference(
        manifest_path,
        record.sha256,
    )
    incident_core = {
        "schemaVersion": QUARANTINE_INCIDENT_SCHEMA_VERSION,
        "kind": QUARANTINE_INCIDENT_KIND,
        "status": "sealed",
        "reason": "run-manifest-source-mismatch",
        "runId": run_id,
        "calibrationContractSha256": contract.sha256,
        "nativeBackendSha256": native_sha256,
        "originalSourceSha256": original_source_sha256,
        "replacementSourceSha256": current_source_sha256,
        "originalRunPath": run_root.relative_to(workspace).as_posix(),
        "quarantinePath": destination.relative_to(workspace).as_posix(),
        "originalRunManifest": manifest_reference,
        "ledgerSnapshot": {
            "path": ledger_path.relative_to(workspace).as_posix(),
            "bytes": len(ledger_bytes),
            "sha256": sha256_bytes(ledger_bytes),
            "ledgerSha256": ledger_self_sha256,
            "e1Runs": e1_runs,
        },
        "files": inventory,
    }
    incident = _self_hashed(incident_core, "incidentSha256")
    incident_path = run_root / QUARANTINE_INCIDENT_NAME
    _commit_immutable_json(
        incident_path,
        incident,
        label="Phase 4E1 quarantine incident",
    )
    try:
        os.replace(run_root, destination)
    except OSError as error:
        raise Phase4E1CalibrationError(
            "could not atomically quarantine source-mismatched E1 run"
        ) from error
    quarantined_incident_path = destination / QUARANTINE_INCIDENT_NAME
    reference = _quarantine_incident_reference(
        quarantined_incident_path,
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        source_sha256=current_source_sha256,
        native_sha256=native_sha256,
    )
    if ledger_path.read_bytes() != ledger_bytes:
        raise Phase4E1CalibrationError(
            "GPU ledger changed during Phase 4E1 quarantine"
        )
    _emit_progress(
        "recovery",
        "quarantined",
        incidentPath=reference["path"],
        incidentSha256=reference["incidentSha256"],
        originalSourceSha256=original_source_sha256,
        replacementSourceSha256=current_source_sha256,
    )
    return reference


def _prepare_run_manifest(
    *,
    workspace: Path,
    run_root: Path,
    run_id: str,
    contract: Phase4E1CalibrationContract,
    native_sha256: str,
    quarantine_incident: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    path = run_root / RUN_MANIFEST_NAME
    current_source = fingerprint_sources(workspace)
    if path.exists():
        record = load_run_manifest(path)
        stored_metadata = record.payload.get("metadata")
        stored_incident = (
            stored_metadata.get("quarantineIncident")
            if isinstance(stored_metadata, Mapping)
            else None
        )
        if (
            quarantine_incident is not None
            and stored_incident != quarantine_incident
        ):
            raise Phase4E1CalibrationError(
                "existing E1 run manifest binds another quarantine incident"
            )
        metadata = _e1_run_metadata(
            workspace=workspace,
            contract=contract,
            run_id=run_id,
            source_sha256=current_source.sha256,
            native_sha256=native_sha256,
            quarantine_incident=stored_incident,
        )
        source = record.payload.get("provenance", {}).get("source")
        if (
            stored_metadata != metadata
            or not isinstance(source, Mapping)
            or source.get("sha256") != current_source.sha256
        ):
            raise Phase4E1CalibrationError(
                "existing E1 run manifest differs from current source/runtime"
            )
        return record.sha256, current_source.sha256

    metadata = _e1_run_metadata(
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        source_sha256=current_source.sha256,
        native_sha256=native_sha256,
        quarantine_incident=quarantine_incident,
    )
    budget = ResourceBudget(
        artifact_limit_bytes=contract.resources.artifact_limit_bytes,
        minimum_free_disk_bytes=contract.resources.minimum_free_disk_bytes,
        gpu_hours=contract.resources.calibration_gpu_hour_limit,
        metadata={
            "phase": "4E1",
            "trainingAllowed": False,
            "checkpointWritesAllowed": False,
            "formalPromotion": False,
        },
    )
    record = build_run_manifest(
        workspace_root=workspace,
        config=contract.to_payload(),
        budget=budget,
        run_id=run_id,
        metadata=metadata,
    )
    save_run_manifest(record, path)
    return record.sha256, current_source.sha256


def _verify_selfplay_report(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    payload = _load_canonical(path, "E1 self-play report")
    _verify_self_hash(payload, field="reportSha256", label="E1 self-play report")
    result = payload.get("result")
    replay = result.get("replay") if isinstance(result, Mapping) else None
    expected_scenario = CalibrationScenario(
        network_name="192x16",
        channels=contract.self_play.channels,
        blocks=contract.self_play.residual_blocks,
        actors=contract.self_play.actors,
        simulations=contract.self_play.simulations,
        games=contract.self_play.games,
        seed=contract.self_play.seed,
    )
    expected_network = {
        "name": expected_scenario.network_name,
        "inputChannels": PHASE4_INPUT_CHANNELS,
        "channels": expected_scenario.channels,
        "residualBlocks": expected_scenario.blocks,
    }
    if (
        payload.get("schemaVersion") != COMPONENT_SCHEMA_VERSION
        or payload.get("kind") != SELFPLAY_KIND
        or payload.get("status") != "completed"
        or payload.get("calibrationOnly") is not True
        or payload.get("trainingPerformed") is not False
        or payload.get("checkpointsWritten") != 0
        or payload.get("calibrationContractSha256") != contract.sha256
        or payload.get("sourceSha256") != source_sha256
        or payload.get("nativeBackendSha256") != native_sha256
        or payload.get("baselineModelSha256")
        != contract.provenance.baseline_model_sha256
        or not isinstance(result, Mapping)
        or result.get("status") != "completed"
        or result.get("scenarioId") != expected_scenario.scenario_id
        or result.get("network") != expected_network
        or result.get("games") != contract.self_play.games
        or result.get("actors") != contract.self_play.actors
        or result.get("simulations") != contract.self_play.simulations
        or result.get("seed") != contract.self_play.seed
        or result.get("temperatureMoves")
        != contract.self_play.temperature_moves
        or not _has_decision_metrics(result)
        or not isinstance(replay, Mapping)
    ):
        raise Phase4E1CalibrationError("E1 self-play report contract mismatch")
    replay_root = _workspace_path(
        workspace,
        str(payload.get("replayRoot")),
        label="E1 self-play replay",
        file=False,
    )
    expected_attempts_root = path.parent / "selfplay-attempts"
    try:
        replay_parts = replay_root.relative_to(expected_attempts_root).parts
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "E1 self-play replay is outside its deterministic run directory"
        ) from error
    if (
        len(replay_parts) != 2
        or replay_parts[1] != "replay"
        or not replay_parts[0].startswith("phase4e1-calibration-")
    ):
        raise Phase4E1CalibrationError(
            "E1 self-play replay path is not a calibration attempt"
        )
    store = ReplayStore(
        replay_root,
        max_games=contract.self_play.games,
        max_plies=contract.self_play.games * 256,
        max_bytes=max(
            int(replay.get("committedBytes", 0)) + 1024**2,
            512 * 1024**2,
        ),
    )
    expected_model_id = (
        f"phase4e1-bootstrap-{contract.provenance.baseline_model_sha256}"
    )
    expected_game_ids = {
        f"{expected_scenario.scenario_id}-{index:06d}"
        for index in range(contract.self_play.games)
    }
    expected_game_metadata = {
        "calibrationOnly": True,
        "scenarioId": expected_scenario.scenario_id,
        "seed": contract.self_play.seed,
    }
    expected_ply_metadata = {
        "calibrationOnly": True,
        "requestedSimulations": contract.self_play.simulations,
    }
    initial_state = create_initial_state()
    verified_game_ids: set[str] = set()
    verified_winners: Counter[int] = Counter()
    verified_terminations: Counter[str] = Counter()
    verified_game_lengths: list[int] = []

    def verify_game(game: ReplayGame, states: tuple[State, ...]) -> None:
        game_id = game.game_id
        plies = game.plies
        final_state = states[-1]
        if (
            game_id not in expected_game_ids
            or game_id in verified_game_ids
            or game.initial_state != initial_state
            or dict(game.metadata) != expected_game_metadata
            or not plies
            or len(states) != len(plies) + 1
            or final_state.winner not in (0, 1)
            or final_state.termination_reason
            not in {"goal", "repetition", "move-limit"}
            or any(
                ply.model_generation != 0
                or ply.model_id != expected_model_id
                or dict(ply.metadata) != expected_ply_metadata
                or sum(ply.visit_counts) < contract.self_play.simulations
                for ply in plies
            )
        ):
            raise Phase4E1CalibrationError(
                "E1 self-play replay workload metadata does not verify"
            )
        verified_game_ids.add(str(game_id))
        verified_winners[int(final_state.winner)] += 1
        verified_terminations[str(final_state.termination_reason)] += 1
        verified_game_lengths.append(len(plies))

    verification = store.verify(deep=True, game_validator=verify_game)
    moves = int(result["moves"])
    games = contract.self_play.games
    elapsed = float(result["elapsedSeconds"])
    expected_winners = {
        "player0": verified_winners.get(0, 0),
        "player1": verified_winners.get(1, 0),
    }
    expected_terminations = dict(sorted(verified_terminations.items()))
    expected_game_length = {
        "minimum": min(verified_game_lengths),
        "maximum": max(verified_game_lengths),
        "mean": moves / games,
    }
    if (
        verification.game_count != games
        or verification.ply_count != moves
        or verified_game_ids != expected_game_ids
        or replay.get("committedBytes") != verification.committed_bytes
        or replay.get("shards") != games
        or result.get("winners") != expected_winners
        or result.get("terminationReasons") != expected_terminations
        or result.get("gameLength") != expected_game_length
        or not math.isclose(
            float(result["gamesPerHour"]),
            games * 3600.0 / elapsed,
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(result["movesPerSecond"]),
            moves / elapsed,
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(result["scheduledSimulationsPerSecond"]),
            moves * contract.self_play.simulations / elapsed,
            rel_tol=1e-12,
        )
    ):
        raise Phase4E1CalibrationError("E1 self-play replay does not verify")
    return payload


def _recover_component_candidate(
    report_path: Path,
    *,
    label: str,
    verifier: Callable[[Path], dict[str, object]],
) -> dict[str, object] | None:
    candidate_path = report_path.with_name(f".{report_path.name}.candidate")
    if not candidate_path.exists():
        return None
    try:
        payload = verifier(candidate_path)
    except KeyboardInterrupt:
        raise
    except Exception:
        candidate_path.unlink(missing_ok=True)
        raise
    _commit_immutable_json(report_path, payload, label=label)
    candidate_path.unlink(missing_ok=True)
    return verifier(report_path)


def _run_selfplay(
    *,
    workspace: Path,
    run_root: Path,
    contract: Phase4E1CalibrationContract,
    source_sha256: str,
    native_sha256: str,
    native_module: object,
    device: torch.device,
    guard: _ResourceGuard,
    invocation_id: str,
) -> dict[str, object]:
    report_path = run_root / SELFPLAY_REPORT_NAME
    candidate_path = report_path.with_name(f".{report_path.name}.candidate")
    if report_path.exists():
        return _verify_selfplay_report(
            report_path,
            workspace=workspace,
            contract=contract,
            source_sha256=source_sha256,
            native_sha256=native_sha256,
        )
    state = load_bootstrap_baseline_state(
        workspace_root=workspace,
        baseline_root=workspace
        / Path(contract.provenance.baseline_manifest_path).parent,
        verify_lineage=False,
    )
    standard_path = workspace / contract.provenance.standard_config_path
    model = _model(standard_path, state, device)
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
    evaluator.enable_native_encoder(native_module)
    scenario = CalibrationScenario(
        network_name="192x16",
        channels=contract.self_play.channels,
        blocks=contract.self_play.residual_blocks,
        actors=contract.self_play.actors,
        simulations=contract.self_play.simulations,
        games=contract.self_play.games,
        seed=contract.self_play.seed,
    )
    replay_root = run_root / "selfplay-attempts" / invocation_id / "replay"
    last_progress = 0.0

    def report_progress(moves: int, games: int) -> None:
        nonlocal last_progress
        guard.check()
        now = time.monotonic()
        if (
            games == contract.self_play.games
            or now - last_progress >= 30.0
        ):
            _emit_progress(
                "self-play",
                "running",
                games=games,
                gamesTotal=contract.self_play.games,
                moves=moves,
            )
            last_progress = now

    _emit_progress(
        "self-play",
        "started",
        games=0,
        gamesTotal=contract.self_play.games,
    )
    payload: dict[str, object] | None = None
    try:
        result = run_scenario(
            scenario,
            evaluator,
            search_factory=NativeMCTS,
            replay_root=replay_root,
            replay_max_bytes=512 * 1024**2,
            model_id=(
                f"phase4e1-bootstrap-{contract.provenance.baseline_model_sha256}"
            ),
            temperature_moves=contract.self_play.temperature_moves,
            progress_callback=report_progress,
        )
        guard.check(force=True)
        core = {
            "schemaVersion": COMPONENT_SCHEMA_VERSION,
            "kind": SELFPLAY_KIND,
            "status": "completed",
            "createdAt": utc_timestamp(),
            "calibrationOnly": True,
            "trainingPerformed": False,
            "checkpointsWritten": 0,
            "calibrationContractSha256": contract.sha256,
            "sourceSha256": source_sha256,
            "nativeBackendSha256": native_sha256,
            "baselineModelSha256": (
                contract.provenance.baseline_model_sha256
            ),
            "replayRoot": _relative(replay_root, workspace),
            "result": result,
        }
        payload = _self_hashed(core, "reportSha256")
        atomic_replace_bytes(candidate_path, canonical_json_bytes(payload))
    finally:
        del evaluator
        model.to("cpu")
        torch.cuda.empty_cache()
    if payload is None:
        raise Phase4E1CalibrationError(
            "self-play completed without a candidate report"
        )
    committed = _recover_component_candidate(
        report_path,
        label="E1 self-play report",
        verifier=lambda candidate: _verify_selfplay_report(
            candidate,
            workspace=workspace,
            contract=contract,
            source_sha256=source_sha256,
            native_sha256=native_sha256,
        ),
    )
    if committed is None:
        raise Phase4E1CalibrationError(
            "self-play candidate disappeared before commit"
        )
    _emit_progress(
        "self-play",
        "completed",
        games=contract.self_play.games,
        reportPath=_relative(report_path, workspace),
    )
    return committed


def _evaluation_summary(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    calls = int(metrics["calls"])
    positions = int(metrics["positions"])
    batch_counts = {
        int(size): int(count)
        for size, count in dict(metrics["batchCounts"]).items()
    }
    latency_counts = {
        int(bucket): int(count)
        for bucket, count in dict(metrics["latencyLog2Microseconds"]).items()
    }

    def histogram_percentile(
        counts: Mapping[int, int],
        probability: float,
        *,
        transform: Callable[[int], float],
    ) -> float | None:
        total = sum(counts.values())
        if total <= 0:
            return None
        rank = max(1, math.ceil(total * probability))
        cumulative = 0
        for value, count in sorted(counts.items()):
            cumulative += count
            if cumulative >= rank:
                return transform(value)
        raise Phase4E1CalibrationError("evaluation histogram is inconsistent")

    return {
        "calls": calls,
        "positions": positions,
        "aggregation": "durable-histogram-v1",
        "batchSize": {
            "p50": histogram_percentile(
                batch_counts, 0.50, transform=float
            ),
            "p95": histogram_percentile(
                batch_counts, 0.95, transform=float
            ),
            "maximum": max(batch_counts, default=None),
        },
        "latencyMilliseconds": {
            "p50UpperBound": histogram_percentile(
                latency_counts,
                0.50,
                transform=lambda bucket: (2**bucket) / 1000.0,
            ),
            "p95UpperBound": histogram_percentile(
                latency_counts,
                0.95,
                transform=lambda bucket: (2**bucket) / 1000.0,
            ),
            "maximum": float(metrics["latencySecondsMaximum"]) * 1000.0,
            "total": float(metrics["latencySecondsTotal"]) * 1000.0,
        },
    }


def _empty_evaluation_metrics() -> dict[str, object]:
    return {
        "calls": 0,
        "positions": 0,
        "batchCounts": {},
        "latencyLog2Microseconds": {},
        "latencySecondsTotal": 0.0,
        "latencySecondsMaximum": 0.0,
    }


def _validate_evaluation_metrics(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "calls",
        "positions",
        "batchCounts",
        "latencyLog2Microseconds",
        "latencySecondsTotal",
        "latencySecondsMaximum",
    }:
        raise Phase4E1CalibrationError("E1 evaluation progress is invalid")
    calls = value.get("calls")
    positions = value.get("positions")
    batch_counts = value.get("batchCounts")
    latency_counts = value.get("latencyLog2Microseconds")
    latency_total = value.get("latencySecondsTotal")
    latency_maximum = value.get("latencySecondsMaximum")
    if (
        type(calls) is not int
        or calls < 0
        or type(positions) is not int
        or positions < 0
        or not isinstance(batch_counts, Mapping)
        or not isinstance(latency_counts, Mapping)
        or not isinstance(latency_total, (int, float))
        or isinstance(latency_total, bool)
        or not math.isfinite(float(latency_total))
        or float(latency_total) < 0
        or not isinstance(latency_maximum, (int, float))
        or isinstance(latency_maximum, bool)
        or not math.isfinite(float(latency_maximum))
        or float(latency_maximum) < 0
    ):
        raise Phase4E1CalibrationError("E1 evaluation progress is invalid")
    normalized_batches: dict[str, int] = {}
    normalized_latency: dict[str, int] = {}
    for source, destination, label in (
        (batch_counts, normalized_batches, "batch"),
        (latency_counts, normalized_latency, "latency"),
    ):
        for key, count in source.items():
            try:
                integer_key = int(key)
            except (TypeError, ValueError) as error:
                raise Phase4E1CalibrationError(
                    f"E1 evaluation {label} histogram is invalid"
                ) from error
            if (
                str(integer_key) != key
                or integer_key < (1 if label == "batch" else 0)
                or type(count) is not int
                or count <= 0
            ):
                raise Phase4E1CalibrationError(
                    f"E1 evaluation {label} histogram is invalid"
                )
            destination[key] = count
    if (
        sum(normalized_batches.values()) != calls
        or sum(normalized_latency.values()) != calls
        or sum(int(size) * count for size, count in normalized_batches.items())
        != positions
    ):
        raise Phase4E1CalibrationError(
            "E1 evaluation progress totals are inconsistent"
        )
    return {
        "calls": calls,
        "positions": positions,
        "batchCounts": normalized_batches,
        "latencyLog2Microseconds": normalized_latency,
        "latencySecondsTotal": float(latency_total),
        "latencySecondsMaximum": float(latency_maximum),
    }


def _append_evaluation_metrics(
    metrics: dict[str, object],
    profiler: EvaluationProfiler,
    start: int,
) -> int:
    if start < 0 or start > len(profiler.batch_sizes):
        raise Phase4E1CalibrationError("evaluation profiler cursor is invalid")
    batches = profiler.batch_sizes[start:]
    elapsed = profiler.elapsed_seconds[start:]
    if len(batches) != len(elapsed):
        raise Phase4E1CalibrationError("evaluation profiler is inconsistent")
    batch_counts = metrics["batchCounts"]
    latency_counts = metrics["latencyLog2Microseconds"]
    if not isinstance(batch_counts, dict) or not isinstance(latency_counts, dict):
        raise Phase4E1CalibrationError("evaluation aggregate is invalid")
    for batch_size, seconds in zip(batches, elapsed):
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0
        ):
            raise Phase4E1CalibrationError("evaluation sample is invalid")
        batch_key = str(batch_size)
        microseconds = max(1, math.ceil(float(seconds) * 1_000_000.0))
        latency_key = str(math.ceil(math.log2(microseconds)))
        batch_counts[batch_key] = int(batch_counts.get(batch_key, 0)) + 1
        latency_counts[latency_key] = int(latency_counts.get(latency_key, 0)) + 1
        metrics["calls"] = int(metrics["calls"]) + 1
        metrics["positions"] = int(metrics["positions"]) + batch_size
        metrics["latencySecondsTotal"] = (
            float(metrics["latencySecondsTotal"]) + float(seconds)
        )
        metrics["latencySecondsMaximum"] = max(
            float(metrics["latencySecondsMaximum"]),
            float(seconds),
        )
    return len(profiler.batch_sizes)


def _validate_gpu_samples(value: object) -> list[list[float | None]]:
    if not isinstance(value, list):
        raise Phase4E1CalibrationError("E1 GPU samples are invalid")
    normalized: list[list[float | None]] = []
    for sample in value:
        if not isinstance(sample, list) or len(sample) != 4:
            raise Phase4E1CalibrationError("E1 GPU sample is invalid")
        converted: list[float | None] = []
        for index, item in enumerate(sample):
            if index == 3 and item is None:
                converted.append(None)
                continue
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or float(item) < 0
                or (index < 2 and float(item) > 100)
            ):
                raise Phase4E1CalibrationError("E1 GPU sample is invalid")
            converted.append(float(item))
        normalized.append(converted)
    return normalized


def _gpu_summary(
    samples: Sequence[Sequence[float | None]],
    errors: Sequence[str],
) -> dict[str, object]:
    gpu = [float(sample[0]) for sample in samples]
    memory = [float(sample[1]) for sample in samples]
    used_mib = [float(sample[2]) for sample in samples]
    power = [
        float(sample[3]) for sample in samples if sample[3] is not None
    ]

    def metric(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "maximum": max(values, default=None),
        }

    return {
        "available": bool(gpu),
        "backend": "nvidia-smi",
        "sampleCount": len(gpu),
        "intervalSeconds": 1.0,
        "error": None if not errors else "; ".join(errors),
        "gpuPercent": metric(gpu),
        "memoryControllerPercent": metric(memory),
        "memoryUsedMiB": metric(used_mib),
        "powerWatts": metric(power),
    }


def _load_arena_progress(
    path: Path,
    contract_sha256: str,
    calibration_contract_sha256: str,
) -> dict[str, object]:
    if not path.exists():
        return {
            "schemaVersion": PROGRESS_SCHEMA_VERSION,
            "contractSha256": contract_sha256,
            "calibrationContractSha256": calibration_contract_sha256,
            "attempts": 0,
            "cumulativeSeconds": 0.0,
            "completedPairs": 0,
            "actorRefillObserved": False,
            "evaluation": _empty_evaluation_metrics(),
            "gpuSamples": [],
            "gpuTelemetryErrors": [],
            "peakAllocatedBytes": 0,
            "peakReservedBytes": 0,
        }
    payload = _load_canonical(path, "E1 arena progress")
    _verify_self_hash(payload, field="progressSha256", label="E1 arena progress")
    if (
        set(payload)
        != {
            "schemaVersion",
            "contractSha256",
            "calibrationContractSha256",
            "attempts",
            "cumulativeSeconds",
            "completedPairs",
            "actorRefillObserved",
            "evaluation",
            "gpuSamples",
            "gpuTelemetryErrors",
            "peakAllocatedBytes",
            "peakReservedBytes",
            "progressSha256",
        }
        or payload.get("schemaVersion") != PROGRESS_SCHEMA_VERSION
        or payload.get("contractSha256") != contract_sha256
        or payload.get("calibrationContractSha256")
        != calibration_contract_sha256
        or type(payload.get("attempts")) is not int
        or payload.get("attempts") < 0
        or not isinstance(payload.get("cumulativeSeconds"), (int, float))
        or isinstance(payload.get("cumulativeSeconds"), bool)
        or not math.isfinite(float(payload.get("cumulativeSeconds")))
        or float(payload.get("cumulativeSeconds")) < 0
        or type(payload.get("completedPairs")) is not int
        or payload.get("completedPairs") < 0
        or type(payload.get("actorRefillObserved")) is not bool
        or type(payload.get("peakAllocatedBytes")) is not int
        or payload.get("peakAllocatedBytes") < 0
        or type(payload.get("peakReservedBytes")) is not int
        or payload.get("peakReservedBytes") < 0
        or not isinstance(payload.get("gpuTelemetryErrors"), list)
        or any(
            not isinstance(error, str) or not error
            for error in payload.get("gpuTelemetryErrors", [])
        )
    ):
        raise Phase4E1CalibrationError("E1 arena progress is invalid")
    normalized = dict(payload)
    normalized["evaluation"] = _validate_evaluation_metrics(
        payload.get("evaluation")
    )
    normalized["gpuSamples"] = _validate_gpu_samples(
        payload.get("gpuSamples")
    )
    return normalized


def _write_arena_progress(path: Path, core: Mapping[str, object]) -> None:
    payload = dict(core)
    payload.pop("progressSha256", None)
    atomic_replace_bytes(
        path,
        canonical_json_bytes(_self_hashed(payload, "progressSha256")),
    )


def _arena_search_config(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> MCTSConfig:
    settings = load_experiment_config(
        workspace / contract.provenance.standard_config_path
    ).mcts
    return MCTSConfig(
        simulations=contract.arena.simulations,
        c_base=settings.c_base,
        c_init=settings.c_init,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=settings.dirichlet_alpha,
        temperature=0.0,
        add_root_noise=False,
        seed=contract.arena.seed,
    )


def _arena_search_payload(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> dict[str, object]:
    search = _arena_search_config(workspace, contract)
    return {
        "implementation": "native-mcts-batched-sessions",
        "standardConfigSemanticSha256": (
            contract.provenance.standard_config_sha256
        ),
        "standardConfigEffectiveSha256": _effective_config_sha256(
            workspace / contract.provenance.standard_config_path,
            workspace,
        ),
        "simulations": search.simulations,
        "cBase": search.c_base,
        "cInit": search.c_init,
        "dirichletEpsilon": search.dirichlet_epsilon,
        "dirichletAlpha": search.dirichlet_alpha,
        "temperature": search.temperature,
        "addRootNoise": search.add_root_noise,
        "seed": search.seed,
    }


def _arena_search_sha256(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(_arena_search_payload(workspace, contract))
    )


def _arena_identities(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    *,
    source_sha256: str,
) -> tuple[SelectorIdentity, SelectorIdentity]:
    search_sha256 = _arena_search_sha256(workspace, contract)
    common = {
        "artifact_hash": contract.provenance.baseline_model_sha256,
        "config_hash": search_sha256,
        "source_hash": source_sha256,
        "engine": "native-mcts",
    }
    return (
        SelectorIdentity(
            selector_id="phase4e1-bootstrap-candidate",
            **common,
        ),
        SelectorIdentity(
            selector_id="phase4e1-bootstrap-opponent",
            **common,
        ),
    )


def _arena_selector_runtime(
    contract: Phase4E1CalibrationContract,
    *,
    native_sha256: str,
) -> SelectorRuntime:
    return SelectorRuntime(
        implementation=(
            f"{MCTSSelector.__module__}.{MCTSSelector.__qualname__}"
            f"[{NativeMCTS.__module__}.{NativeMCTS.__qualname__}]"
        ),
        kind="native-mcts",
        configured_simulations=contract.arena.simulations,
        budget_kind=None,
        budget_limit=None,
        max_depth=None,
        identity_bound=True,
        backend_hash=native_sha256,
    )


def _arena_contract_sha256(
    contract: Phase4E1CalibrationContract,
    *,
    workspace: Path,
    source_sha256: str,
    native_sha256: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "kind": ARENA_KIND,
                "calibrationContractSha256": contract.sha256,
                "sourceSha256": source_sha256,
                "nativeBackendSha256": native_sha256,
                "suiteHash": contract.arena.suite_hash,
                "actors": contract.arena.actors,
                "simulations": contract.arena.simulations,
                "modelSha256": contract.provenance.baseline_model_sha256,
                "searchConfig": _arena_search_payload(workspace, contract),
            }
        )
    )


def _verify_partial_arena_store(
    *,
    workspace: Path,
    run_root: Path,
    contract: Phase4E1CalibrationContract,
    source_sha256: str,
    native_sha256: str,
) -> int:
    store_root = run_root / "arena-store"
    if not store_root.exists():
        return 0
    store_root = _workspace_path(
        workspace,
        store_root.relative_to(workspace).as_posix(),
        label="E1 partial arena store",
        file=False,
    )
    dev = load_opening_suite(
        workspace / contract.provenance.dev_suite_path,
        expected_suite="dev",
    )
    suite = build_opening_suite(
        dev.openings[: contract.arena.opening_prefix_count],
        suite="dev",
    )
    candidate_identity, opponent_identity = _arena_identities(
        workspace,
        contract,
        source_sha256=source_sha256,
    )
    runtime = _arena_selector_runtime(
        contract,
        native_sha256=native_sha256,
    )
    contract_sha256 = _arena_contract_sha256(
        contract,
        workspace=workspace,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    store = OpeningPairStore(
        store_root,
        contract_sha256=contract_sha256,
        suite=suite,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=runtime,
        opponent_runtime=runtime,
    )
    progress = _load_arena_progress(
        run_root / ARENA_PROGRESS_NAME,
        contract_sha256,
        contract.sha256,
    )
    if progress.get("completedPairs") != store.completed_pairs:
        raise Phase4E1CalibrationError(
            "E1 arena progress differs from the deeply verified pair store"
        )
    return store.completed_pairs


def _arena_counts(store: object) -> tuple[int, int, Counter[str]]:
    games = 0
    moves = 0
    terminations: Counter[str] = Counter()
    for ordinal in store.completed_ordinals:
        payload = store.pair_payload(ordinal)
        for game in payload["games"]:
            games += 1
            plies = game["plies"]
            moves += len(plies)
            terminations[str(game["terminationReason"])] += 1
    return games, moves, terminations


def _verify_arena_report(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    payload = _load_canonical(path, "E1 arena report")
    _verify_self_hash(payload, field="reportSha256", label="E1 arena report")
    result_path = _workspace_path(
        workspace,
        str(payload.get("arenaResultPath")),
        label="E1 arena result",
    )
    expected_result_path = (path.parent / "arena-store" / "result.json").resolve()
    if result_path != expected_result_path:
        raise Phase4E1CalibrationError(
            "E1 arena result is outside its deterministic run directory"
        )
    result = verify_completed_arena_result(result_path)
    expected_contract_sha256 = _arena_contract_sha256(
        contract,
        workspace=workspace,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    candidate_identity, opponent_identity = _arena_identities(
        workspace,
        contract,
        source_sha256=source_sha256,
    )
    expected_candidate = candidate_identity.to_dict()
    expected_opponent = opponent_identity.to_dict()
    expected_runtime = _arena_selector_runtime(
        contract,
        native_sha256=native_sha256,
    ).to_dict()
    expected_search = _arena_search_payload(workspace, contract)
    expected_search_sha256 = sha256_bytes(
        canonical_json_bytes(expected_search)
    )
    progress = _load_arena_progress(
        path.parent / ARENA_PROGRESS_NAME,
        expected_contract_sha256,
        contract.sha256,
    )
    recomputed_moves = 0
    recomputed_terminations: Counter[str] = Counter()
    for artifact in result["pairArtifacts"]:
        pair = _load_canonical(
            result_path.parent / str(artifact["file"]),
            "E1 arena pair",
        )
        for game in pair["games"]:
            recomputed_moves += len(game["plies"])
            recomputed_terminations[str(game["terminationReason"])] += 1
    expected_evaluation = _evaluation_summary(progress["evaluation"])
    expected_gpu = _gpu_summary(
        progress["gpuSamples"],
        progress["gpuTelemetryErrors"],
    )
    evaluation = payload.get("evaluation")
    batch = (
        evaluation.get("batchSize") if isinstance(evaluation, Mapping) else None
    )
    telemetry = payload.get("gpuUtilization")
    cuda_memory = payload.get("cudaMemory")
    if (
        payload.get("schemaVersion") != COMPONENT_SCHEMA_VERSION
        or payload.get("kind") != ARENA_KIND
        or payload.get("status") != "completed"
        or payload.get("smokeOnly") is not True
        or payload.get("formalPromotion") is not False
        or payload.get("trainingPerformed") is not False
        or payload.get("checkpointsWritten") != 0
        or payload.get("calibrationContractSha256") != contract.sha256
        or payload.get("arenaContractSha256") != expected_contract_sha256
        or payload.get("sourceSha256") != source_sha256
        or payload.get("nativeBackendSha256") != native_sha256
        or payload.get("baselineModelSha256")
        != contract.provenance.baseline_model_sha256
        or payload.get("searchConfig") != expected_search
        or payload.get("searchConfigSha256") != expected_search_sha256
        or payload.get("suiteHash") != contract.arena.suite_hash
        or payload.get("openingPairs") != contract.arena.opening_prefix_count
        or payload.get("games") != contract.arena.games
        or payload.get("actors") != contract.arena.actors
        or payload.get("simulations") != contract.arena.simulations
        or payload.get("actorRefillObserved")
        is not contract.arena.require_actor_refill
        or result.get("games") != contract.arena.games
        or result.get("suiteHash") != contract.arena.suite_hash
        or result.get("contractSha256") != expected_contract_sha256
        or result.get("candidateIdentity") != expected_candidate
        or result.get("opponentIdentity") != expected_opponent
        or result.get("candidateRuntime") != expected_runtime
        or result.get("opponentRuntime") != expected_runtime
        or result.get("resultSha256") != payload.get("arenaResultSha256")
        or progress.get("completedPairs")
        != contract.arena.opening_prefix_count
        or progress.get("actorRefillObserved")
        is not contract.arena.require_actor_refill
        or payload.get("attempts") != progress.get("attempts")
        or payload.get("elapsedSeconds")
        != progress.get("cumulativeSeconds")
        or evaluation != expected_evaluation
        or telemetry != expected_gpu
        or cuda_memory
        != {
            "peakAllocatedBytes": progress.get("peakAllocatedBytes"),
            "peakReservedBytes": progress.get("peakReservedBytes"),
        }
        or not isinstance(evaluation, Mapping)
        or not isinstance(batch, Mapping)
        or not isinstance(batch.get("p50"), (int, float))
        or float(batch.get("p50")) <= 0
        or not isinstance(batch.get("p95"), (int, float))
        or float(batch.get("p95")) <= 0
        or not isinstance(telemetry, Mapping)
        or telemetry.get("available") is not True
        or telemetry.get("error") is not None
        or type(telemetry.get("sampleCount")) is not int
        or telemetry.get("sampleCount") < 2
        or not isinstance(cuda_memory, Mapping)
        or type(cuda_memory.get("peakAllocatedBytes")) is not int
        or cuda_memory.get("peakAllocatedBytes") <= 0
        or type(cuda_memory.get("peakReservedBytes")) is not int
        or cuda_memory.get("peakReservedBytes") <= 0
    ):
        raise Phase4E1CalibrationError("E1 arena report contract mismatch")
    for key in (
        "elapsedSeconds",
        "gamesPerHour",
        "movesPerSecond",
        "scheduledSimulationsPerSecond",
    ):
        value = payload.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise Phase4E1CalibrationError(f"E1 arena {key} is invalid")
    elapsed = float(payload["elapsedSeconds"])
    games = int(payload["games"])
    moves = payload.get("moves")
    scheduled = payload.get("scheduledSimulations")
    terminations = payload.get("terminationReasons")
    if (
        type(moves) is not int
        or moves <= 0
        or moves != recomputed_moves
        or type(scheduled) is not int
        or scheduled != moves * contract.arena.simulations
        or not isinstance(terminations, Mapping)
        or any(type(count) is not int or count < 0 for count in terminations.values())
        or sum(terminations.values()) != games
        or dict(sorted(terminations.items()))
        != dict(sorted(recomputed_terminations.items()))
        or not math.isclose(
            float(payload["gamesPerHour"]),
            games * 3600.0 / elapsed,
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(payload["movesPerSecond"]),
            moves / elapsed,
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(payload["scheduledSimulationsPerSecond"]),
            scheduled / elapsed,
            rel_tol=1e-12,
        )
    ):
        raise Phase4E1CalibrationError("E1 arena throughput evidence is invalid")
    return payload


def _raise_primary_or_cleanup(
    primary_error: BaseException | None,
    cleanup_errors: Sequence[tuple[str, BaseException]],
) -> None:
    if primary_error is not None:
        if cleanup_errors:
            raise primary_error.with_traceback(
                primary_error.__traceback__
            ) from cleanup_errors[0][1]
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_errors:
        raise cleanup_errors[0][1]


def _run_arena(
    *,
    workspace: Path,
    run_root: Path,
    contract: Phase4E1CalibrationContract,
    source_sha256: str,
    native_sha256: str,
    native_module: object,
    device: torch.device,
    guard: _ResourceGuard,
) -> dict[str, object]:
    report_path = run_root / ARENA_REPORT_NAME
    if report_path.exists():
        return _verify_arena_report(
            report_path,
            workspace=workspace,
            contract=contract,
            source_sha256=source_sha256,
            native_sha256=native_sha256,
        )

    arena_contract_sha256 = _arena_contract_sha256(
        contract,
        workspace=workspace,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    progress_path = run_root / ARENA_PROGRESS_NAME
    progress = _load_arena_progress(
        progress_path,
        arena_contract_sha256,
        contract.sha256,
    )
    prior_seconds = float(progress["cumulativeSeconds"])
    progress["attempts"] = int(progress["attempts"]) + 1
    evaluation_metrics = _validate_evaluation_metrics(progress["evaluation"])
    gpu_samples = _validate_gpu_samples(progress["gpuSamples"])
    gpu_errors = list(progress["gpuTelemetryErrors"])
    arena_started = time.monotonic()
    _write_arena_progress(progress_path, progress)

    dev = load_opening_suite(
        workspace / contract.provenance.dev_suite_path,
        expected_suite="dev",
    )
    suite = build_opening_suite(
        dev.openings[: contract.arena.opening_prefix_count],
        suite="dev",
    )
    state = load_bootstrap_baseline_state(
        workspace_root=workspace,
        baseline_root=workspace
        / Path(contract.provenance.baseline_manifest_path).parent,
        verify_lineage=False,
    )
    standard_path = workspace / contract.provenance.standard_config_path
    candidate_model = _model(standard_path, state, device)
    opponent_model = _model(standard_path, state, device)
    candidate_profiler = EvaluationProfiler(
        TorchEvaluator(candidate_model, device=device, use_amp=True)
    )
    opponent_profiler = EvaluationProfiler(
        TorchEvaluator(opponent_model, device=device, use_amp=True)
    )
    candidate_profiler.enable_native_encoder(native_module)
    opponent_profiler.enable_native_encoder(native_module)
    search_config = _arena_search_config(workspace, contract)
    candidate_identity, opponent_identity = _arena_identities(
        workspace,
        contract,
        source_sha256=source_sha256,
    )
    selector_runtime = _arena_selector_runtime(
        contract,
        native_sha256=native_sha256,
    )
    candidate = BoundMoveSelector(
        MCTSSelector(
            candidate_profiler,
            search_config,
            search_factory=NativeMCTS,
            engine="native-mcts",
        ),
        candidate_identity,
    )
    opponent = BoundMoveSelector(
        MCTSSelector(
            opponent_profiler,
            search_config,
            search_factory=NativeMCTS,
            engine="native-mcts",
        ),
        opponent_identity,
    )
    telemetry = NvidiaSmiProfiler(int(device.index or 0))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    telemetry.start()
    store = None
    profiler_cursors = [0, 0]
    telemetry_cursor = 0
    last_progress_write = arena_started
    last_completed_pairs = int(progress["completedPairs"])
    last_progress_emit = arena_started

    def persist_progress(*, completed_pairs: int, force: bool) -> None:
        nonlocal telemetry_cursor, last_progress_write, last_completed_pairs
        now = time.monotonic()
        if (
            not force
            and completed_pairs == last_completed_pairs
            and now - last_progress_write < HEARTBEAT_SECONDS
        ):
            guard.check()
            return
        for index, profiler in enumerate(
            (candidate_profiler, opponent_profiler)
        ):
            profiler_cursors[index] = _append_evaluation_metrics(
                evaluation_metrics,
                profiler,
                profiler_cursors[index],
            )
        for sample in telemetry.samples[telemetry_cursor:]:
            gpu_samples.append(
                [
                    float(sample[0]),
                    float(sample[1]),
                    float(sample[2]),
                    None if sample[3] is None else float(sample[3]),
                ]
            )
        telemetry_cursor = len(telemetry.samples)
        if telemetry.error is not None and telemetry.error not in gpu_errors:
            gpu_errors.append(telemetry.error)
        progress["cumulativeSeconds"] = prior_seconds + (
            now - arena_started
        )
        progress["completedPairs"] = completed_pairs
        progress["evaluation"] = evaluation_metrics
        progress["gpuSamples"] = gpu_samples
        progress["gpuTelemetryErrors"] = gpu_errors
        progress["peakAllocatedBytes"] = max(
            int(progress["peakAllocatedBytes"]),
            int(torch.cuda.max_memory_allocated(device)),
        )
        progress["peakReservedBytes"] = max(
            int(progress["peakReservedBytes"]),
            int(torch.cuda.max_memory_reserved(device)),
        )
        # The ledger timestamp must cover every durable progress timestamp.
        guard.check(force=True)
        _write_arena_progress(progress_path, progress)
        last_progress_write = now
        last_completed_pairs = completed_pairs

    def heartbeat(_moves: int, completed_pairs: int) -> None:
        nonlocal last_progress_emit
        pair_changed = completed_pairs != last_completed_pairs
        persist_progress(completed_pairs=completed_pairs, force=False)
        now = time.monotonic()
        if (
            pair_changed
            or now - last_progress_emit >= 30.0
        ):
            _emit_progress(
                "arena",
                "running",
                completedPairs=completed_pairs,
                openingPairs=contract.arena.opening_prefix_count,
            )
            last_progress_emit = now

    _emit_progress(
        "arena",
        "started",
        completedPairs=int(progress["completedPairs"]),
        openingPairs=contract.arena.opening_prefix_count,
    )
    primary_error: BaseException | None = None
    try:
        store = run_resumable_paired_arena(
            suite,
            candidate,
            opponent,
            store_root=run_root / "arena-store",
            contract_sha256=arena_contract_sha256,
            actors=contract.arena.actors,
            heartbeat=heartbeat,
        )
        result = store.finalize(
            bootstrap_seed=contract.arena.seed,
            bootstrap_samples=10_000,
            confidence=0.90,
        )
        torch.cuda.synchronize(device)
    except BaseException as error:
        primary_error = error
    cleanup_errors: list[tuple[str, BaseException]] = []

    def cleanup(label: str, action: Callable[[], object]) -> None:
        try:
            action()
        except BaseException as error:
            cleanup_errors.append((label, error))

    cleanup("stop GPU telemetry", telemetry.stop)
    completed_pairs = last_completed_pairs
    if store is not None:
        completed_pairs = store.completed_pairs
    else:
        try:
            completed_pairs = OpeningPairStore(
                run_root / "arena-store",
                contract_sha256=arena_contract_sha256,
                suite=suite,
                candidate_identity=candidate_identity,
                opponent_identity=opponent_identity,
                candidate_runtime=selector_runtime,
                opponent_runtime=selector_runtime,
            ).completed_pairs
        except BaseException as error:
            cleanup_errors.append(("reconcile arena store", error))
    cleanup(
        "persist arena progress",
        lambda: persist_progress(
            completed_pairs=completed_pairs,
            force=True,
        ),
    )
    cleanup("close candidate selector", candidate.close)
    cleanup("close opponent selector", opponent.close)
    cleanup("move candidate model to CPU", lambda: candidate_model.to("cpu"))
    cleanup("move opponent model to CPU", lambda: opponent_model.to("cpu"))
    cleanup("empty CUDA cache", torch.cuda.empty_cache)
    _raise_primary_or_cleanup(primary_error, cleanup_errors)

    if store is None:
        raise Phase4E1CalibrationError("arena store was not created")
    verified_result = verify_completed_arena_result(
        run_root / "arena-store" / "result.json"
    )
    if verified_result != result:
        raise Phase4E1CalibrationError("arena result changed during verification")
    games, moves, terminations = _arena_counts(store)
    elapsed = float(progress["cumulativeSeconds"])
    scheduled = moves * contract.arena.simulations
    actor_refill = (
        contract.arena.require_actor_refill
        and contract.arena.games > contract.arena.actors
        and games == contract.arena.games
    )
    progress["actorRefillObserved"] = actor_refill
    guard.check(force=True)
    _write_arena_progress(progress_path, progress)
    core = {
        "schemaVersion": COMPONENT_SCHEMA_VERSION,
        "kind": ARENA_KIND,
        "status": "completed",
        "createdAt": utc_timestamp(),
        "smokeOnly": True,
        "formalPromotion": False,
        "trainingPerformed": False,
        "checkpointsWritten": 0,
        "calibrationContractSha256": contract.sha256,
        "arenaContractSha256": arena_contract_sha256,
        "sourceSha256": source_sha256,
        "nativeBackendSha256": native_sha256,
        "baselineModelSha256": contract.provenance.baseline_model_sha256,
        "searchConfig": _arena_search_payload(workspace, contract),
        "searchConfigSha256": _arena_search_sha256(workspace, contract),
        "suiteHash": suite.content_hash,
        "openingPairs": len(suite.openings),
        "games": games,
        "actors": contract.arena.actors,
        "simulations": contract.arena.simulations,
        "actorRefillObserved": actor_refill,
        "attempts": progress["attempts"],
        "elapsedSeconds": elapsed,
        "gamesPerHour": games * 3600.0 / elapsed,
        "moves": moves,
        "movesPerSecond": moves / elapsed,
        "scheduledSimulations": scheduled,
        "scheduledSimulationsPerSecond": scheduled / elapsed,
        "evaluation": _evaluation_summary(evaluation_metrics),
        "gpuUtilization": _gpu_summary(gpu_samples, gpu_errors),
        "cudaMemory": {
            "peakAllocatedBytes": int(progress["peakAllocatedBytes"]),
            "peakReservedBytes": int(progress["peakReservedBytes"]),
        },
        "terminationReasons": dict(sorted(terminations.items())),
        "arenaResultPath": _relative(
            run_root / "arena-store" / "result.json",
            workspace,
        ),
        "arenaResultSha256": verified_result["resultSha256"],
    }
    payload = _self_hashed(core, "reportSha256")
    candidate_path = report_path.with_name(f".{report_path.name}.candidate")
    atomic_replace_bytes(candidate_path, canonical_json_bytes(payload))
    committed = _recover_component_candidate(
        report_path,
        label="E1 arena report",
        verifier=lambda candidate: _verify_arena_report(
            candidate,
            workspace=workspace,
            contract=contract,
            source_sha256=source_sha256,
            native_sha256=native_sha256,
        ),
    )
    if committed is None:
        raise Phase4E1CalibrationError(
            "arena candidate disappeared before commit"
        )
    _emit_progress(
        "arena",
        "completed",
        games=games,
        reportPath=_relative(report_path, workspace),
    )
    return committed


def _recommend_budget(
    contract: Phase4E1CalibrationContract,
    *,
    global_gpu_hours: float,
) -> dict[str, object]:
    raw = (
        contract.resources.global_gpu_hour_limit
        - global_gpu_hours
        - contract.freeze.reserve_gpu_hours
    )
    limit = math.floor(raw * 1_000_000.0) / 1_000_000.0
    if limit < contract.freeze.minimum_stage_gpu_hours:
        raise Phase4E1CalibrationError(
            "less than the minimum 48 GPU-hours remains for Phase 4E"
        )
    return {
        "status": "recommended",
        "baselineGlobalHours": global_gpu_hours,
        "limitHours": limit,
        "reservedGpuHours": contract.freeze.reserve_gpu_hours,
        "globalGpuHourLimit": contract.resources.global_gpu_hour_limit,
        "maximumGenerations": contract.freeze.maximum_generations,
    }


def _approved_workloads(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> list[str]:
    bootstrap = _workspace_path(
        workspace,
        contract.provenance.bootstrap_config_path,
        label="Phase 4E Bootstrap config",
    )
    standard = _workspace_path(
        workspace,
        contract.provenance.standard_config_path,
        label="Phase 4E Standard config",
    )
    if (
        fingerprint_config(bootstrap).sha256
        != contract.provenance.bootstrap_config_sha256
        or fingerprint_config(standard).sha256
        != contract.provenance.standard_config_sha256
    ):
        raise Phase4E1CalibrationError(
            "approved Phase 4E workload source changed"
        )
    return [
        _effective_config_sha256(bootstrap, workspace),
        _effective_config_sha256(standard, workspace),
    ]


def _verify_e1_run_manifest(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_id: str,
    expected_sha256: object,
    source_sha256: str,
    native_sha256: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise Phase4E1CalibrationError("E1 run manifest is missing or a symlink")
    try:
        record = load_run_manifest(path)
    except (OSError, ValueError, RunManifestError) as error:
        raise Phase4E1CalibrationError("E1 run manifest does not verify") from error
    if path.read_bytes() != canonical_json_bytes(record.payload):
        raise Phase4E1CalibrationError("E1 run manifest must be canonical JSON")
    provenance = record.payload.get("provenance")
    source = (
        provenance.get("source") if isinstance(provenance, Mapping) else None
    )
    config = (
        provenance.get("config") if isinstance(provenance, Mapping) else None
    )
    expected_budget = ResourceBudget(
        artifact_limit_bytes=contract.resources.artifact_limit_bytes,
        minimum_free_disk_bytes=contract.resources.minimum_free_disk_bytes,
        gpu_hours=contract.resources.calibration_gpu_hour_limit,
        metadata={
            "phase": "4E1",
            "trainingAllowed": False,
            "checkpointWritesAllowed": False,
            "formalPromotion": False,
        },
    ).to_payload()
    stored_metadata = record.payload.get("metadata")
    stored_incident = (
        stored_metadata.get("quarantineIncident")
        if isinstance(stored_metadata, Mapping)
        else None
    )
    expected_metadata = _e1_run_metadata(
        workspace=workspace,
        contract=contract,
        run_id=run_id,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
        quarantine_incident=stored_incident,
    )
    if (
        record.sha256 != expected_sha256
        or record.payload.get("runId") != run_id
        or record.payload.get("workspace") != str(workspace)
        or record.payload.get("resourceBudget") != expected_budget
        or stored_metadata != expected_metadata
        or not isinstance(source, Mapping)
        or source.get("sha256") != source_sha256
        or not isinstance(config, Mapping)
        or config.get("sha256") != fingerprint_config(contract.to_payload()).sha256
    ):
        raise Phase4E1CalibrationError("E1 run manifest contract mismatch")


def _verify_final_report(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    verify_ledger: bool,
    candidate_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected_run_id = f"phase4e1-calibration-{contract.sha256[:16]}"
    expected_path = Path(
        os.path.abspath(
            workspace / DEFAULT_OUTPUT_ROOT / expected_run_id / REPORT_NAME
        )
    )
    path = Path(os.path.abspath(path))
    if path != expected_path:
        raise Phase4E1CalibrationError(
            "Phase 4E1 report is outside its deterministic contract address"
        )
    relative_parts = path.relative_to(workspace).parts
    cursor = workspace
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise Phase4E1CalibrationError(
                "Phase 4E1 report path contains a symlink"
            )
    payload = (
        _load_canonical(path, "Phase 4E1 report")
        if candidate_payload is None
        else dict(candidate_payload)
    )
    _verify_self_hash(payload, field="reportSha256", label="Phase 4E1 report")
    if (
        set(payload)
        != {
            "schemaVersion",
            "kind",
            "status",
            "createdAt",
            "runId",
            "decisionEligible",
            "formalPromotion",
            "trainingPerformed",
            "checkpointsWritten",
            "trainingAuthorized",
            "calibrationContractSha256",
            "preparedExecutionContractSha256",
            "runManifestSha256",
            "sourceSha256",
            "nativeBackendSha256",
            "approvedWorkloads",
            "bootstrapBaseline",
            "selfPlay",
            "arena",
            "resourceUsage",
            "recommendedStageBudget",
            "reportSha256",
        }
        or payload.get("schemaVersion") != REPORT_SCHEMA_VERSION
        or payload.get("kind") != REPORT_KIND
        or payload.get("status") != "completed"
        or not isinstance(payload.get("createdAt"), str)
        or not payload.get("createdAt")
        or payload.get("decisionEligible") is not True
        or payload.get("formalPromotion") is not False
        or payload.get("trainingPerformed") is not False
        or payload.get("checkpointsWritten") != 0
        or payload.get("trainingAuthorized") is not False
        or payload.get("calibrationContractSha256") != contract.sha256
        or payload.get("preparedExecutionContractSha256")
        != contract.provenance.prepared_execution_contract_sha256
        or payload.get("runId") != expected_run_id
    ):
        raise Phase4E1CalibrationError("Phase 4E1 report contract mismatch")
    source_sha256 = validate_sha256(
        payload.get("sourceSha256"), "Phase 4E1 source sha256"
    )
    native_sha256 = validate_sha256(
        payload.get("nativeBackendSha256"),
        "Phase 4E1 native backend sha256",
    )
    _verify_e1_run_manifest(
        path.parent / RUN_MANIFEST_NAME,
        workspace=workspace,
        contract=contract,
        run_id=expected_run_id,
        expected_sha256=payload.get("runManifestSha256"),
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    selfplay_path, selfplay = _validate_reference(
        payload.get("selfPlay"),
        workspace=workspace,
        label="E1 self-play report",
    )
    arena_path, arena = _validate_reference(
        payload.get("arena"),
        workspace=workspace,
        label="E1 arena report",
    )
    if (
        selfplay_path != path.parent / SELFPLAY_REPORT_NAME
        or arena_path != path.parent / ARENA_REPORT_NAME
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 component is outside its deterministic run directory"
        )
    _verify_selfplay_report(
        workspace / str(payload["selfPlay"]["path"]),
        workspace=workspace,
        contract=contract,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    _verify_arena_report(
        workspace / str(payload["arena"]["path"]),
        workspace=workspace,
        contract=contract,
        source_sha256=source_sha256,
        native_sha256=native_sha256,
    )
    if selfplay.get("status") != "completed" or arena.get("status") != "completed":
        raise Phase4E1CalibrationError("Phase 4E1 component is incomplete")

    approved = payload.get("approvedWorkloads")
    expected_approved = _approved_workloads(workspace, contract)
    if approved != expected_approved or len(set(expected_approved)) != 2:
        raise Phase4E1CalibrationError("approved workload list is invalid")
    baseline_root = (
        workspace / Path(contract.provenance.baseline_manifest_path).parent
    )
    if payload.get("bootstrapBaseline") != verify_bootstrap_baseline(
        workspace_root=workspace,
        baseline_root=baseline_root,
    ):
        raise Phase4E1CalibrationError(
            "Phase 4E1 bootstrap baseline evidence changed"
        )
    usage = payload.get("resourceUsage")
    budget = payload.get("recommendedStageBudget")
    if not isinstance(usage, Mapping) or not isinstance(budget, Mapping):
        raise Phase4E1CalibrationError("Phase 4E1 budget evidence is missing")
    global_hours = usage.get("globalGpuHours")
    if (
        set(usage)
        != {
            "initialGlobalGpuHours",
            "globalGpuHours",
            "phase4e1GpuHours",
            "calibrationLimitHours",
            "artifactBytes",
            "freeDiskBytes",
            "ledgerEvidence",
        }
        or usage.get("initialGlobalGpuHours")
        != contract.resources.initial_ledger.total_gpu_hours
        or usage.get("calibrationLimitHours")
        != contract.resources.calibration_gpu_hour_limit
        or not isinstance(global_hours, (int, float))
        or isinstance(global_hours, bool)
        or not math.isfinite(float(global_hours))
        or float(global_hours) < contract.resources.initial_ledger.total_gpu_hours
        or float(global_hours)
        - contract.resources.initial_ledger.total_gpu_hours
        > contract.resources.calibration_gpu_hour_limit
        or usage.get("phase4e1GpuHours")
        != float(global_hours)
        - contract.resources.initial_ledger.total_gpu_hours
        or type(usage.get("artifactBytes")) is not int
        or usage.get("artifactBytes") < 0
        or usage.get("artifactBytes")
        >= contract.resources.artifact_limit_bytes
        or type(usage.get("freeDiskBytes")) is not int
        or usage.get("freeDiskBytes")
        < contract.resources.minimum_free_disk_bytes
    ):
        raise Phase4E1CalibrationError("Phase 4E1 GPU usage is invalid")
    expected_budget = _recommend_budget(
        contract, global_gpu_hours=float(global_hours)
    )
    if budget != expected_budget:
        raise Phase4E1CalibrationError("recommended stage budget was not recomputed")
    _verify_ledger_snapshot_evidence(
        usage.get("ledgerEvidence"),
        workspace=workspace,
        contract=contract,
        reported_global_hours=float(global_hours),
        verify_current=verify_ledger,
    )
    return payload


def verify_phase4e1_decision_report(
    *,
    workspace_root: str | Path,
    report_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT,
    verify_ledger: bool = True,
) -> dict[str, object]:
    """Verify the complete E1 evidence tree against the current local runtime."""

    workspace = Path(workspace_root).resolve()
    source_contract_path = Path(contract_path)
    if not source_contract_path.is_absolute():
        source_contract_path = workspace / source_contract_path
    contract = load_phase4e1_contract(source_contract_path)
    _preflight(
        workspace=workspace,
        contract=contract,
        contract_path=source_contract_path,
    )
    report = Path(report_path)
    if not report.is_absolute():
        report = workspace / report
    payload = _verify_final_report(
        report,
        workspace=workspace,
        contract=contract,
        verify_ledger=verify_ledger,
    )
    if fingerprint_sources(workspace).sha256 != payload.get("sourceSha256"):
        raise Phase4E1CalibrationError(
            "current source differs from the Phase 4E1 decision report"
        )
    if loaded_native_backend_hash() != payload.get("nativeBackendSha256"):
        raise Phase4E1CalibrationError(
            "current native backend differs from the Phase 4E1 decision report"
        )
    return payload


def run_phase4e1_calibration(
    *,
    workspace_root: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT,
    device_index: int = 0,
    external_gpu_hours: float = 0.0,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    source_contract_path = Path(contract_path)
    if not source_contract_path.is_absolute():
        source_contract_path = workspace / source_contract_path
    source_contract_path = source_contract_path.resolve()
    contract = load_phase4e1_contract(source_contract_path)
    preflight = _preflight(
        workspace=workspace,
        contract=contract,
        contract_path=source_contract_path,
    )
    run_id = f"phase4e1-calibration-{contract.sha256[:16]}"
    run_root = (
        workspace / DEFAULT_OUTPUT_ROOT / run_id
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    final_path = run_root / REPORT_NAME
    if final_path.exists():
        return verify_phase4e1_decision_report(
            workspace_root=workspace,
            report_path=final_path,
            contract_path=source_contract_path,
            verify_ledger=True,
        )
    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    native_sha256 = loaded_native_backend_hash()
    control_root = workspace / "training/runs/phase4"
    invocation_id = (
        f"phase4e1-calibration-{os.getpid()}-{time.time_ns()}"
    )
    lock_path = control_root / GPU_LOCK_NAME
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_for_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_for_sigterm)
    try:
        with RunLock(lock_path, run_id=invocation_id):
            if final_path.exists():
                return verify_phase4e1_decision_report(
                    workspace_root=workspace,
                    report_path=final_path,
                    contract_path=source_contract_path,
                    verify_ledger=True,
                )
            quarantine_incident = _quarantine_source_mismatched_run(
                workspace=workspace,
                control_root=control_root,
                run_root=run_root,
                run_id=run_id,
                contract=contract,
                native_sha256=native_sha256,
            )
            run_root.mkdir(parents=True, exist_ok=True)
            manifest_sha256, source_sha256 = _prepare_run_manifest(
                workspace=workspace,
                run_root=run_root,
                run_id=run_id,
                contract=contract,
                native_sha256=native_sha256,
                quarantine_incident=quarantine_incident,
            )
            selfplay_path = run_root / SELFPLAY_REPORT_NAME
            arena_path = run_root / ARENA_REPORT_NAME
            if not selfplay_path.exists():
                recovered = _recover_component_candidate(
                    selfplay_path,
                    label="E1 self-play report",
                    verifier=lambda candidate: _verify_selfplay_report(
                        candidate,
                        workspace=workspace,
                        contract=contract,
                        source_sha256=source_sha256,
                        native_sha256=native_sha256,
                    ),
                )
                if recovered is not None:
                    _emit_progress(
                        "self-play",
                        "recovered",
                        reportPath=_relative(selfplay_path, workspace),
                    )
            if not arena_path.exists():
                recovered = _recover_component_candidate(
                    arena_path,
                    label="E1 arena report",
                    verifier=lambda candidate: _verify_arena_report(
                        candidate,
                        workspace=workspace,
                        contract=contract,
                        source_sha256=source_sha256,
                        native_sha256=native_sha256,
                    ),
                )
                if recovered is not None:
                    _emit_progress(
                        "arena",
                        "recovered",
                        reportPath=_relative(arena_path, workspace),
                    )
            if selfplay_path.exists():
                _verify_selfplay_report(
                    selfplay_path,
                    workspace=workspace,
                    contract=contract,
                    source_sha256=source_sha256,
                    native_sha256=native_sha256,
                )
            if arena_path.exists():
                _verify_arena_report(
                    arena_path,
                    workspace=workspace,
                    contract=contract,
                    source_sha256=source_sha256,
                    native_sha256=native_sha256,
                )
            else:
                verified_pairs = _verify_partial_arena_store(
                    workspace=workspace,
                    run_root=run_root,
                    contract=contract,
                    source_sha256=source_sha256,
                    native_sha256=native_sha256,
                )
                if verified_pairs:
                    _emit_progress(
                        "arena",
                        "partial-store-verified",
                        completedPairs=verified_pairs,
                        openingPairs=contract.arena.opening_prefix_count,
                    )
            missing_selfplay = not selfplay_path.exists()
            missing_arena = not arena_path.exists()
            ledger = GpuBudgetLedger(control_root / GPU_LEDGER_NAME)
            if (
                isinstance(external_gpu_hours, bool)
                or not isinstance(external_gpu_hours, (int, float))
                or not math.isfinite(float(external_gpu_hours))
                or float(external_gpu_hours) < 0
            ):
                raise ValueError(
                    "external_gpu_hours must be finite and non-negative"
                )
            projected_external = max(
                ledger.external_hours_floor,
                float(external_gpu_hours),
            )
            run_hours = sum(
                float(run["accountedSeconds"])
                for run in ledger.runs.values()
            ) / 3600.0
            global_before = projected_external + run_hours
            if missing_selfplay or missing_arena:
                if (
                    global_before
                    - contract.resources.initial_ledger.total_gpu_hours
                    >= contract.resources.calibration_gpu_hour_limit
                ):
                    raise Phase4E1CalibrationError(
                        "Phase 4E1 GPU budget is exhausted"
                    )
                if (
                    not torch.cuda.is_available()
                    or device_index >= torch.cuda.device_count()
                ):
                    raise Phase4E1CalibrationError(
                        "requested CUDA device is unavailable"
                    )
                native_module = load_native_module()
                if not callable(getattr(native_module, "encode_states", None)):
                    raise Phase4E1CalibrationError(
                        "native batch encoder is unavailable"
                    )
                device = torch.device(f"cuda:{device_index}")
                started = time.monotonic()
                ledger.begin(invocation_id, external_hours=external_gpu_hours)
                guard = _ResourceGuard(
                    control_root=control_root,
                    contract=contract,
                    ledger=ledger,
                    ledger_run_id=invocation_id,
                    started=started,
                    global_at_start=global_before,
                )
                status = "failed"
                try:
                    guard.check(force=True)
                    if missing_selfplay:
                        _run_selfplay(
                            workspace=workspace,
                            run_root=run_root,
                            contract=contract,
                            source_sha256=source_sha256,
                            native_sha256=native_sha256,
                            native_module=native_module,
                            device=device,
                            guard=guard,
                            invocation_id=invocation_id,
                        )
                    if missing_arena:
                        _run_arena(
                            workspace=workspace,
                            run_root=run_root,
                            contract=contract,
                            source_sha256=source_sha256,
                            native_sha256=native_sha256,
                            native_module=native_module,
                            device=device,
                            guard=guard,
                        )
                    status = "completed"
                except KeyboardInterrupt:
                    status = "interrupted"
                    raise
                finally:
                    final_global = ledger.update(
                        invocation_id,
                        elapsed_seconds=time.monotonic() - started,
                        status=status,
                    )
            else:
                if float(external_gpu_hours) > ledger.external_hours_floor:
                    ledger.record_external_hours(float(external_gpu_hours))
                final_global = ledger.total_hours
            recommended = _recommend_budget(
                contract, global_gpu_hours=final_global
            )
            approved = _approved_workloads(workspace, contract)
            usage = {
                "artifactBytes": _directory_size(control_root),
                "freeDiskBytes": int(shutil.disk_usage(control_root).free),
                "ledgerEvidence": _ledger_snapshot_evidence(
                    workspace,
                    contract,
                ),
            }
            core = {
                "schemaVersion": REPORT_SCHEMA_VERSION,
                "kind": REPORT_KIND,
                "status": "completed",
                "createdAt": utc_timestamp(),
                "runId": run_id,
                "decisionEligible": True,
                "formalPromotion": False,
                "trainingPerformed": False,
                "checkpointsWritten": 0,
                "trainingAuthorized": False,
                "calibrationContractSha256": contract.sha256,
                "preparedExecutionContractSha256": preflight[
                    "preparedExecutionContractSha256"
                ],
                "runManifestSha256": manifest_sha256,
                "sourceSha256": source_sha256,
                "nativeBackendSha256": native_sha256,
                "approvedWorkloads": approved,
                "bootstrapBaseline": preflight["bootstrapBaseline"],
                "selfPlay": _component_reference(
                    run_root / SELFPLAY_REPORT_NAME, workspace
                ),
                "arena": _component_reference(
                    run_root / ARENA_REPORT_NAME, workspace
                ),
                "resourceUsage": {
                    "initialGlobalGpuHours": (
                        contract.resources.initial_ledger.total_gpu_hours
                    ),
                    "globalGpuHours": final_global,
                    "phase4e1GpuHours": (
                        final_global
                        - contract.resources.initial_ledger.total_gpu_hours
                    ),
                    "calibrationLimitHours": (
                        contract.resources.calibration_gpu_hour_limit
                    ),
                    "artifactBytes": usage["artifactBytes"],
                    "freeDiskBytes": usage["freeDiskBytes"],
                    "ledgerEvidence": usage["ledgerEvidence"],
                },
                "recommendedStageBudget": recommended,
            }
            payload = _self_hashed(core, "reportSha256")
            _verify_final_report(
                final_path,
                workspace=workspace,
                contract=contract,
                verify_ledger=True,
                candidate_payload=payload,
            )
            _commit_immutable_json(
                final_path,
                payload,
                label="Phase 4E1 final report",
            )
            return _verify_final_report(
                final_path,
                workspace=workspace,
                contract=contract,
                verify_ledger=True,
            )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def freeze_phase4e_execution_contract(
    *,
    workspace_root: str | Path,
    calibration_contract_path: str | Path,
    report_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    contract_path = Path(calibration_contract_path)
    if not contract_path.is_absolute():
        contract_path = workspace / contract_path
    contract = load_phase4e1_contract(contract_path)
    report = Path(report_path)
    if not report.is_absolute():
        report = workspace / report
    payload = verify_phase4e1_decision_report(
        workspace_root=workspace,
        report_path=report,
        contract_path=contract_path,
        verify_ledger=True,
    )
    recommended = payload["recommendedStageBudget"]
    execution = Phase4ExecutionContract(
        schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
        stage="phase4e",
        status=PHASE4_EXECUTION_STATUS_FROZEN,
        purpose="phase4e-bootstrap-and-formal-promotion",
        formal_promotion=True,
        maximum_generations=contract.freeze.maximum_generations,
        stage_budget=Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=float(recommended["baselineGlobalHours"]),
            limit_hours=float(recommended["limitHours"]),
            baseline_run_id=str(payload["runId"]),
            baseline_report_sha256=str(payload["reportSha256"]),
        ),
    )
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = workspace / destination
    destination = Path(os.path.abspath(destination))
    try:
        destination_parts = destination.relative_to(workspace).parts
    except ValueError as error:
        raise Phase4E1CalibrationError(
            "frozen execution contract must stay inside the workspace"
        ) from error
    cursor = workspace
    for part in destination_parts:
        cursor /= part
        if cursor.is_symlink():
            raise Phase4E1CalibrationError(
                "frozen execution contract path contains a symlink"
            )
    _commit_immutable_json(
        destination,
        execution.to_payload(),
        label="frozen Phase 4E execution contract",
    )
    return {
        "status": "frozen",
        "path": _relative(destination, workspace),
        "executionContractSha256": execution.sha256,
        "baselineRunId": payload["runId"],
        "baselineReportSha256": payload["reportSha256"],
        "baselineGlobalHours": execution.stage_budget.baseline_global_hours,
        "limitHours": execution.stage_budget.limit_hours,
        "maximumGenerations": execution.maximum_generations,
    }


def _summary(
    payload: Mapping[str, object],
    *,
    report_path: Path,
) -> dict[str, object]:
    selfplay = payload.get("selfPlay")
    arena = payload.get("arena")
    usage = payload.get("resourceUsage")
    budget = payload.get("recommendedStageBudget")
    return {
        "status": payload.get("status"),
        "decisionEligible": payload.get("decisionEligible"),
        "runId": payload.get("runId"),
        "reportPath": str(report_path),
        "reportSha256": payload.get("reportSha256"),
        "selfPlayReportPath": (
            selfplay.get("path") if isinstance(selfplay, Mapping) else None
        ),
        "arenaReportPath": (
            arena.get("path") if isinstance(arena, Mapping) else None
        ),
        "globalGpuHours": (
            usage.get("globalGpuHours") if isinstance(usage, Mapping) else None
        ),
        "phase4e1GpuHours": (
            usage.get("phase4e1GpuHours") if isinstance(usage, Mapping) else None
        ),
        "recommendedPhase4eGpuHours": (
            budget.get("limitHours") if isinstance(budget, Mapping) else None
        ),
        "trainingAuthorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4E1 Standard-shaped CUDA calibration and arena smoke."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--workspace-root", default=".")
    run.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--gpu-hours-used", type=float, default=0.0)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--workspace-root", default=".")
    verify.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    verify.add_argument("--report", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--workspace-root", default=".")
    freeze.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    freeze.add_argument("--report", required=True)
    freeze.add_argument("--output", default=str(FROZEN_EXECUTION_DEFAULT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = Path(args.workspace_root).resolve()
    if args.command == "run":
        contract = load_phase4e1_contract(
            workspace / args.contract
            if not Path(args.contract).is_absolute()
            else args.contract
        )
        run_id = f"phase4e1-calibration-{contract.sha256[:16]}"
        report_path = workspace / DEFAULT_OUTPUT_ROOT / run_id / REPORT_NAME
        try:
            payload = run_phase4e1_calibration(
                workspace_root=workspace,
                contract_path=args.contract,
                device_index=args.device_index,
                external_gpu_hours=args.gpu_hours_used,
            )
        except KeyboardInterrupt:
            return 130
        print(
            canonical_json_bytes(
                _summary(payload, report_path=report_path)
            ).decode(),
            end="",
        )
        return 0
    if args.command == "verify":
        contract = load_phase4e1_contract(
            workspace / args.contract
            if not Path(args.contract).is_absolute()
            else args.contract
        )
        report_path = (
            Path(args.report)
            if Path(args.report).is_absolute()
            else workspace / args.report
        )
        payload = verify_phase4e1_decision_report(
            workspace_root=workspace,
            report_path=report_path,
            contract_path=args.contract,
            verify_ledger=True,
        )
        print(
            canonical_json_bytes(
                _summary(payload, report_path=report_path)
            ).decode(),
            end="",
        )
        return 0
    result = freeze_phase4e_execution_contract(
        workspace_root=workspace,
        calibration_contract_path=args.contract,
        report_path=args.report,
        output_path=args.output,
    )
    print(canonical_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_KIND",
    "SELFPLAY_KIND",
    "ARENA_KIND",
    "Phase4E1CalibrationError",
    "run_phase4e1_calibration",
    "freeze_phase4e_execution_contract",
    "main",
]
