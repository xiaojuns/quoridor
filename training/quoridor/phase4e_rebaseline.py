"""CPU-only provenance repair for a superseded Phase 4E1 baseline.

This module deliberately keeps the existing Phase 4E1 quarantine contract.
The completed run is fully verified first, then an additional immutable
attestation is added to the run tree before the normal quarantine incident is
sealed.  The existing Phase 4E1 recovery path can therefore discover the
incident and bind it into the replacement run manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .calibration import GpuBudgetLedger
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import (
    PHASE4E_STAGE,
    PHASE4_STAGE_LEDGER_NAME,
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
from .phase4e1_contract import (
    Phase4E1CalibrationContract,
    load_phase4e1_contract,
)
from .run_manifest import (
    RunLock,
    RunManifestError,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    load_run_manifest,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)
SUPERSESSION_SCHEMA_VERSION = 1
SUPERSESSION_KIND = "phase4e1-completed-evidence-supersession"
SUPERSESSION_STATUS = "verified"
SUPERSESSION_REASON = "source-fingerprint-correctness-repair"
SUPERSESSION_ATTESTATION_NAME = "completed-evidence-supersession.json"

DEFAULT_CONTROL_ROOT = Path("training/runs/phase4")
DEFAULT_CALIBRATION_CONTRACT = Path(
    "training/configs/phase4e1-calibration-v1.json"
)
DEFAULT_OLD_EXECUTION_CONTRACT = Path(
    "training/configs/phase4e-execution-frozen-v1.json"
)
DEFAULT_NEW_EXECUTION_CONTRACT = Path(
    "training/configs/phase4e-execution-frozen-v2.json"
)
REBASE_PREPARED_KIND = "phase4e-stage-ledger-rebase-prepared"
REBASE_COMPLETED_KIND = "phase4e-stage-ledger-rebase-completed"
REBASE_RECEIPT_ROOT = Path(
    "training/runs/phase4/phase4e1/stage-rebases"
)
CONTROL_WRITE_RESERVE_BYTES = 16 * 1024**2


class Phase4ERebaselineError(RuntimeError):
    """Raised when historical evidence cannot be safely superseded."""


HistoricalVerifier = Callable[
    [Path, Path, Phase4E1CalibrationContract], Mapping[str, object]
]
BootstrapIncidentVerifier = Callable[
    [Path, Path, str, str, str], Mapping[str, object]
]


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError as error:
        raise Phase4ERebaselineError(
            f"cannot open directory for durability sync: {path}"
        ) from error
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise Phase4ERebaselineError(
                f"cannot durability-sync directory: {path}"
            ) from error
    finally:
        os.close(descriptor)


def _resource_preflight(
    control_root: Path,
    *,
    artifact_limit_bytes: int,
    minimum_free_disk_bytes: int,
) -> dict[str, int]:
    def strict_directory_size(path: Path) -> int:
        total = 0
        try:
            entries = list(os.scandir(path))
        except OSError as error:
            raise Phase4ERebaselineError(
                f"cannot inspect Phase 4 artifact path: {path}"
            ) from error
        for entry in entries:
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise Phase4ERebaselineError(
                    "Phase 4 artifact tree contains a symlink"
                )
            if entry.is_dir(follow_symlinks=False):
                total += strict_directory_size(entry_path)
            elif entry.is_file(follow_symlinks=False):
                try:
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError as error:
                    raise Phase4ERebaselineError(
                        f"cannot stat Phase 4 artifact: {entry_path}"
                    ) from error
            else:
                raise Phase4ERebaselineError(
                    "Phase 4 artifact tree contains a non-regular entry"
                )
        return total

    artifact_bytes = strict_directory_size(control_root)
    free_bytes = shutil.disk_usage(control_root).free
    if artifact_bytes + CONTROL_WRITE_RESERVE_BYTES >= artifact_limit_bytes:
        raise Phase4ERebaselineError(
            "Phase 4 artifact limit cannot reserve rebaseline control writes"
        )
    if free_bytes - CONTROL_WRITE_RESERVE_BYTES < minimum_free_disk_bytes:
        raise Phase4ERebaselineError(
            "Phase 4 disk stop line cannot reserve rebaseline control writes"
        )
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": artifact_limit_bytes,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": minimum_free_disk_bytes,
        "reservedWriteBytes": CONTROL_WRITE_RESERVE_BYTES,
    }


def _workspace_path(
    workspace: Path,
    source: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = _absolute_without_resolving(candidate)
    try:
        parts = candidate.relative_to(workspace).parts
    except ValueError as error:
        raise Phase4ERebaselineError(f"{label} escapes the workspace") from error
    cursor = workspace
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise Phase4ERebaselineError(f"{label} path contains a symlink")
    if must_exist and not candidate.exists():
        raise Phase4ERebaselineError(f"{label} is missing")
    return candidate


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4ERebaselineError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4ERebaselineError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4ERebaselineError(f"{label} must be canonical JSON")
    return dict(payload)


def _self_hashed(
    core: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    return {
        **core,
        field: sha256_bytes(canonical_json_bytes(core)),
    }


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    try:
        digest = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise Phase4ERebaselineError(f"{label} self hash is invalid") from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4ERebaselineError(f"{label} self hash mismatch")
    return digest


def _file_reference(
    path: Path,
    *,
    workspace: Path,
    self_hash_field: str | None = None,
) -> dict[str, object]:
    reference: dict[str, object] = {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if self_hash_field is not None:
        payload = _load_canonical(path, path.name)
        reference[self_hash_field] = _verify_self_hash(
            payload,
            field=self_hash_field,
            label=path.name,
        )
    return reference


def _execution_reference(
    path: Path,
    *,
    workspace: Path,
) -> tuple[Phase4ExecutionContract, dict[str, object]]:
    contract = load_phase4_execution_contract(path)
    if (
        contract.stage != PHASE4E_STAGE
        or contract.status != PHASE4_EXECUTION_STATUS_FROZEN
        or not contract.formal_promotion
    ):
        raise Phase4ERebaselineError(
            "old execution contract is not a frozen formal Phase 4E contract"
        )
    return contract, {
        **_file_reference(path, workspace=workspace),
        "executionContractSha256": contract.sha256,
    }


def _stage_ledger_reference(
    path: Path,
    *,
    workspace: Path,
    old_execution: Phase4ExecutionContract,
) -> dict[str, object]:
    ledger = Phase4StageBudgetLedger(path)
    try:
        record = ledger.stages[PHASE4E_STAGE]
    except KeyError as error:
        raise Phase4ERebaselineError(
            "Phase 4E stage ledger record is missing"
        ) from error
    budget = old_execution.require_frozen_budget()
    if (
        record.get("baselineGlobalHours") != budget.baseline_global_hours
        or record.get("limitHours") != budget.limit_hours
        or record.get("baselineRunId") != budget.baseline_run_id
        or record.get("baselineReportSha256")
        != budget.baseline_report_sha256
    ):
        raise Phase4ERebaselineError(
            "Phase 4E stage ledger differs from the old frozen contract"
        )
    payload = _load_canonical(path, "Phase 4 stage budget ledger")
    return {
        **_file_reference(path, workspace=workspace),
        "ledgerSha256": validate_sha256(
            payload.get("ledgerSha256"),
            "Phase 4 stage ledger self hash",
        ),
        "phase4eRecord": dict(record),
        "payload": payload,
    }


def _gpu_ledger_reference(
    path: Path,
    *,
    workspace: Path,
) -> tuple[bytes, dict[str, object]]:
    payload = _load_canonical(path, "Phase 4 GPU ledger")
    try:
        ledger = GpuBudgetLedger(path)
        ledger_sha256 = validate_sha256(
            payload.get("ledgerSha256"),
            "Phase 4 GPU ledger self hash",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise Phase4ERebaselineError(
            "Phase 4 GPU ledger does not verify"
        ) from error
    if any(run.get("status") == "running" for run in ledger.runs.values()):
        raise Phase4ERebaselineError(
            "a running GPU ledger entry blocks completed evidence supersession"
        )
    raw = path.read_bytes()
    return raw, {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "ledgerSha256": ledger_sha256,
        "totalGpuHours": ledger.total_hours,
        "runs": {
            key: dict(value)
            for key, value in sorted(ledger.runs.items())
        },
        "payload": payload,
    }


def _verify_gpu_ledger_append_only(
    snapshot: Mapping[str, object],
    *,
    workspace: Path,
) -> None:
    required = {
        "path",
        "bytes",
        "sha256",
        "ledgerSha256",
        "totalGpuHours",
        "runs",
        "payload",
    }
    if set(snapshot) != required or not isinstance(snapshot.get("path"), str):
        raise Phase4ERebaselineError(
            "attested global GPU ledger snapshot contract is invalid"
        )
    path = _workspace_path(
        workspace,
        snapshot["path"],
        label="current global GPU ledger",
        must_exist=True,
    )
    current_payload = _load_canonical(path, "current global GPU ledger")
    current = GpuBudgetLedger(path)
    stored_payload = snapshot.get("payload")
    stored_runs = snapshot.get("runs")
    if not isinstance(stored_payload, Mapping) or not isinstance(
        stored_runs, Mapping
    ):
        raise Phase4ERebaselineError(
            "attested global GPU ledger snapshot is invalid"
        )
    stored_core = dict(stored_payload)
    stored_self_hash = stored_core.pop("ledgerSha256", None)
    stored_raw = canonical_json_bytes(stored_payload)
    total_gpu_hours = snapshot.get("totalGpuHours")
    if (
        isinstance(snapshot.get("bytes"), bool)
        or not isinstance(snapshot.get("bytes"), int)
        or snapshot.get("bytes") != len(stored_raw)
        or snapshot.get("sha256") != sha256_bytes(stored_raw)
        or snapshot.get("ledgerSha256") != stored_self_hash
        or sha256_bytes(canonical_json_bytes(stored_core)) != stored_self_hash
        or stored_payload.get("runs") != stored_runs
        or isinstance(total_gpu_hours, bool)
        or not isinstance(total_gpu_hours, (int, float))
        or not math.isfinite(float(total_gpu_hours))
        or float(total_gpu_hours) < 0
    ):
        raise Phase4ERebaselineError(
            "attested global GPU ledger snapshot does not verify"
        )
    try:
        validate_sha256(
            snapshot.get("sha256"),
            "attested global GPU ledger file hash",
        )
        validate_sha256(
            stored_self_hash,
            "attested global GPU ledger self hash",
        )
    except ValueError as error:
        raise Phase4ERebaselineError(
            "attested global GPU ledger snapshot hash is invalid"
        ) from error
    stored_seconds = 0.0
    for run_id, stored_run in stored_runs.items():
        seconds = (
            stored_run.get("accountedSeconds")
            if isinstance(stored_run, Mapping)
            else None
        )
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(stored_run, Mapping)
            or stored_run.get("status") == "running"
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) < 0
        ):
            raise Phase4ERebaselineError(
                "attested global GPU ledger run is invalid"
            )
        stored_seconds += float(seconds)
    stored_external = stored_payload.get("externalHoursFloor")
    if (
        isinstance(stored_external, bool)
        or not isinstance(stored_external, (int, float))
        or not math.isfinite(float(stored_external))
        or float(stored_external) < 0
        or float(stored_external) + stored_seconds / 3600.0
        != float(total_gpu_hours)
    ):
        raise Phase4ERebaselineError(
            "attested global GPU ledger total does not verify"
        )
    current_runs = current_payload.get("runs")
    if not isinstance(current_runs, Mapping):
        raise Phase4ERebaselineError(
            "current global GPU ledger has no run mapping"
        )
    for run_id, stored_run in stored_runs.items():
        if (
            not isinstance(run_id, str)
            or not isinstance(stored_run, Mapping)
            or stored_run.get("status") == "running"
            or current_runs.get(run_id) != stored_run
        ):
            raise Phase4ERebaselineError(
                "global GPU ledger is not an append-only extension "
                "of the supersession snapshot"
            )
    if (
        current.external_hours_floor < float(stored_external)
        or current.total_hours < float(total_gpu_hours)
    ):
        raise Phase4ERebaselineError(
            "global GPU ledger regressed from the supersession snapshot"
        )


def _default_historical_verifier(
    report_path: Path,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> Mapping[str, object]:
    # This is the complete verifier except for the public wrapper's final
    # current-source equality check. All historical lineage, replay, arena,
    # baseline, resource, and append-only GPU-ledger checks remain enabled.
    from . import phase4e1_calibration as calibration

    return calibration._verify_final_report(
        report_path,
        workspace=workspace,
        contract=contract,
        verify_ledger=True,
    )


def _default_bootstrap_incident_verifier(
    incident_path: Path,
    workspace: Path,
    old_execution_sha256: str,
    original_source_sha256: str,
    replacement_source_sha256: str,
) -> Mapping[str, object]:
    from .phase4e_bootstrap_quarantine import (
        verify_quarantine_incident_attestation,
    )

    return verify_quarantine_incident_attestation(
        workspace_root=workspace,
        incident_path=incident_path,
        expected_old_execution_contract_sha256=old_execution_sha256,
        expected_original_source_sha256=original_source_sha256,
        expected_replacement_source_sha256=replacement_source_sha256,
    )


def _verify_pre_move_incident(
    path: Path,
    *,
    workspace: Path,
    contract: Phase4E1CalibrationContract,
    run_root: Path,
    destination: Path,
    run_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    original_source_sha256: str,
    replacement_source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    from . import phase4e1_calibration as calibration

    payload = calibration._load_canonical(
        path,
        "sealed pre-move Phase 4E1 quarantine incident",
    )
    calibration._verify_self_hash(
        payload,
        field="incidentSha256",
        label="sealed pre-move Phase 4E1 quarantine incident",
    )
    manifest_reference = payload.get("originalRunManifest")
    ledger_snapshot = payload.get("ledgerSnapshot")
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
        or payload.get("schemaVersion")
        != calibration.QUARANTINE_INCIDENT_SCHEMA_VERSION
        or payload.get("kind") != calibration.QUARANTINE_INCIDENT_KIND
        or payload.get("status") != "sealed"
        or payload.get("reason") != "run-manifest-source-mismatch"
        or payload.get("runId") != run_id
        or payload.get("calibrationContractSha256") != contract.sha256
        or payload.get("nativeBackendSha256") != native_sha256
        or payload.get("originalSourceSha256") != original_source_sha256
        or payload.get("replacementSourceSha256")
        != replacement_source_sha256
        or payload.get("originalRunPath")
        != run_root.relative_to(workspace).as_posix()
        or payload.get("quarantinePath")
        != destination.relative_to(workspace).as_posix()
        or not isinstance(manifest_reference, Mapping)
        or not isinstance(ledger_snapshot, Mapping)
        or not isinstance(files, list)
        or files != calibration._run_tree_inventory(run_root)
    ):
        raise Phase4ERebaselineError(
            "sealed pre-move quarantine incident contract mismatch"
        )
    expected_manifest_reference = calibration._quarantine_manifest_reference(
        manifest_path,
        manifest_sha256,
    )
    if manifest_reference != expected_manifest_reference:
        raise Phase4ERebaselineError(
            "sealed pre-move incident manifest reference changed"
        )
    e1_runs = ledger_snapshot.get("e1Runs")
    try:
        validate_sha256(
            ledger_snapshot.get("sha256"),
            "sealed incident GPU ledger file hash",
        )
        validate_sha256(
            ledger_snapshot.get("ledgerSha256"),
            "sealed incident GPU ledger self hash",
        )
    except ValueError as error:
        raise Phase4ERebaselineError(
            "sealed pre-move incident ledger hashes are invalid"
        ) from error
    if (
        not isinstance(e1_runs, Mapping)
        or any(
            not isinstance(value, Mapping)
            or value.get("status") == "running"
            for value in e1_runs.values()
        )
    ):
        raise Phase4ERebaselineError(
            "sealed pre-move incident contains a running E1 ledger entry"
        )
    return payload


def _verify_completed_attestation(
    path: Path,
    *,
    workspace: Path,
    run_id: str,
    contract_sha256: str,
    old_execution_sha256: str,
    bootstrap_trigger_reference: Mapping[str, object],
    old_report_sha256: str,
    original_source_sha256: str,
    replacement_source_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    payload = _load_canonical(path, "completed E1 supersession attestation")
    _verify_self_hash(
        payload,
        field="attestationSha256",
        label="completed E1 supersession attestation",
    )
    if (
        set(payload)
        != {
            "schemaVersion",
            "kind",
            "status",
            "reason",
            "verifiedAt",
            "runId",
            "calibrationContractSha256",
            "oldExecutionContract",
            "bootstrapFailureTrigger",
            "oldDecisionReport",
            "oldRunManifest",
            "originalSourceSha256",
            "replacementSourceSha256",
            "nativeBackendSha256",
            "stageLedgerBefore",
            "globalGpuLedger",
            "attestationSha256",
        }
        or payload.get("schemaVersion") != SUPERSESSION_SCHEMA_VERSION
        or payload.get("kind") != SUPERSESSION_KIND
        or payload.get("status") != SUPERSESSION_STATUS
        or payload.get("reason") != SUPERSESSION_REASON
        or not isinstance(payload.get("verifiedAt"), str)
        or not payload.get("verifiedAt")
        or payload.get("runId") != run_id
        or payload.get("calibrationContractSha256") != contract_sha256
        or payload.get("originalSourceSha256") != original_source_sha256
        or payload.get("replacementSourceSha256") != replacement_source_sha256
        or payload.get("nativeBackendSha256") != native_sha256
    ):
        raise Phase4ERebaselineError(
            "completed E1 supersession attestation contract mismatch"
        )
    old_execution = payload.get("oldExecutionContract")
    bootstrap_trigger = payload.get("bootstrapFailureTrigger")
    old_report = payload.get("oldDecisionReport")
    old_manifest = payload.get("oldRunManifest")
    stage = payload.get("stageLedgerBefore")
    global_ledger = payload.get("globalGpuLedger")
    if (
        not isinstance(old_execution, Mapping)
        or old_execution.get("executionContractSha256")
        != old_execution_sha256
        or not isinstance(bootstrap_trigger, Mapping)
        or dict(bootstrap_trigger) != dict(bootstrap_trigger_reference)
        or not isinstance(old_report, Mapping)
        or old_report.get("reportSha256") != old_report_sha256
        or not isinstance(old_manifest, Mapping)
        or not isinstance(stage, Mapping)
        or not isinstance(global_ledger, Mapping)
    ):
        raise Phase4ERebaselineError(
            "completed E1 supersession provenance is incomplete"
        )
    for reference, label in (
        (old_execution, "old execution contract"),
        (bootstrap_trigger, "bootstrap failure trigger"),
        (old_report, "old decision report"),
        (old_manifest, "old run manifest"),
        (stage, "old stage ledger"),
        (global_ledger, "global GPU ledger"),
    ):
        try:
            validate_sha256(reference.get("sha256"), f"{label} file hash")
        except ValueError as error:
            raise Phase4ERebaselineError(
                "completed E1 supersession contains an invalid file hash"
            ) from error
    execution_path = _workspace_path(
        workspace,
        str(old_execution["path"]),
        label="attested old execution contract",
        must_exist=True,
    )
    execution_raw = execution_path.read_bytes()
    if (
        old_execution.get("bytes") != len(execution_raw)
        or old_execution.get("sha256") != sha256_bytes(execution_raw)
        or load_phase4_execution_contract(execution_path).sha256
        != old_execution_sha256
    ):
        raise Phase4ERebaselineError(
            "attested old execution contract changed"
        )
    trigger_path = _workspace_path(
        workspace,
        str(bootstrap_trigger["path"]),
        label="attested Bootstrap quarantine incident",
        must_exist=True,
    )
    trigger_raw = trigger_path.read_bytes()
    trigger_payload = _load_canonical(
        trigger_path,
        "attested Bootstrap quarantine incident",
    )
    if (
        bootstrap_trigger.get("bytes") != len(trigger_raw)
        or bootstrap_trigger.get("sha256") != sha256_bytes(trigger_raw)
        or trigger_payload.get("incidentSha256")
        != bootstrap_trigger.get("incidentSha256")
    ):
        raise Phase4ERebaselineError(
            "attested Bootstrap quarantine incident changed"
        )

    archive_root = path.parent
    archived_report = archive_root / Path(str(old_report["path"])).name
    archived_manifest = archive_root / Path(str(old_manifest["path"])).name
    report_payload = _load_canonical(
        archived_report,
        "archived old E1 decision report",
    )
    manifest_payload = _load_canonical(
        archived_manifest,
        "archived old E1 run manifest",
    )
    report_raw = archived_report.read_bytes()
    manifest_raw = archived_manifest.read_bytes()
    if (
        old_report.get("bytes") != len(report_raw)
        or old_report.get("sha256") != sha256_bytes(report_raw)
        or report_payload.get("reportSha256") != old_report_sha256
        or old_manifest.get("bytes") != len(manifest_raw)
        or old_manifest.get("sha256") != sha256_bytes(manifest_raw)
        or manifest_payload.get("manifestSha256")
        != old_manifest.get("manifestSha256")
    ):
        raise Phase4ERebaselineError(
            "archived completed E1 report or manifest changed"
        )
    _verify_self_hash(
        report_payload,
        field="reportSha256",
        label="archived old E1 decision report",
    )
    _verify_self_hash(
        manifest_payload,
        field="manifestSha256",
        label="archived old E1 run manifest",
    )

    stage_payload = stage.get("payload")
    global_payload = global_ledger.get("payload")
    if not isinstance(stage_payload, Mapping) or not isinstance(
        global_payload, Mapping
    ):
        raise Phase4ERebaselineError(
            "completed E1 supersession omitted immutable ledger snapshots"
        )
    stage_stages = stage_payload.get("stages")
    if not isinstance(stage_stages, Mapping):
        raise Phase4ERebaselineError(
            "completed E1 supersession stage snapshot is invalid"
        )
    stage_core = dict(stage_payload)
    stage_self = stage_core.pop("ledgerSha256", None)
    global_core = dict(global_payload)
    global_self = global_core.pop("ledgerSha256", None)
    if (
        stage.get("ledgerSha256") != stage_self
        or stage.get("sha256")
        != sha256_bytes(canonical_json_bytes(stage_payload))
        or sha256_bytes(canonical_json_bytes(stage_core)) != stage_self
        or stage_stages.get(PHASE4E_STAGE)
        != stage.get("phase4eRecord")
        or global_ledger.get("ledgerSha256") != global_self
        or global_ledger.get("sha256")
        != sha256_bytes(canonical_json_bytes(global_payload))
        or sha256_bytes(canonical_json_bytes(global_core)) != global_self
        or global_payload.get("runs") != global_ledger.get("runs")
    ):
        raise Phase4ERebaselineError(
            "completed E1 supersession ledger snapshot does not verify"
        )
    _verify_gpu_ledger_append_only(global_ledger, workspace=workspace)
    return payload


def _attestation_report_sha256(
    incident_path: Path,
    *,
    workspace: Path,
    expected: Mapping[str, object],
    bootstrap_trigger_reference: Mapping[str, object],
) -> str:
    attestation_path = incident_path.parent / SUPERSESSION_ATTESTATION_NAME
    payload = _verify_completed_attestation(
        attestation_path,
        workspace=workspace,
        run_id=str(expected["runId"]),
        contract_sha256=str(expected["calibrationContractSha256"]),
        old_execution_sha256=str(expected["oldExecutionContractSha256"]),
        bootstrap_trigger_reference=bootstrap_trigger_reference,
        old_report_sha256=str(expected["oldReportSha256"]),
        original_source_sha256=str(expected["originalSourceSha256"]),
        replacement_source_sha256=str(expected["replacementSourceSha256"]),
        native_sha256=str(expected["nativeBackendSha256"]),
    )
    return str(payload["oldDecisionReport"]["reportSha256"])


def supersede_completed_e1_evidence(
    *,
    workspace_root: str | Path,
    calibration_contract_path: str | Path = DEFAULT_CALIBRATION_CONTRACT,
    report_path: str | Path,
    old_execution_contract_path: str | Path = DEFAULT_OLD_EXECUTION_CONTRACT,
    bootstrap_quarantine_incident_path: str | Path,
    expected_old_source_sha256: str,
    expected_replacement_source_sha256: str,
    execute: bool = False,
    historical_verifier: HistoricalVerifier | None = None,
    bootstrap_incident_verifier: BootstrapIncidentVerifier | None = None,
) -> dict[str, object]:
    """Deep-verify and atomically quarantine one completed historical E1 run."""

    from . import phase4e1_calibration as calibration

    workspace = _absolute_without_resolving(Path(workspace_root))
    if workspace.is_symlink() or not workspace.is_dir():
        raise Phase4ERebaselineError(
            "workspace root is missing or is a symlink"
        )
    try:
        expected_old_source = validate_sha256(
            expected_old_source_sha256,
            "expected old source sha256",
        )
        expected_replacement_source = validate_sha256(
            expected_replacement_source_sha256,
            "expected replacement source sha256",
        )
    except ValueError as error:
        raise Phase4ERebaselineError("source expectation is invalid") from error
    if expected_old_source == expected_replacement_source:
        raise Phase4ERebaselineError(
            "replacement source must differ from the superseded source"
        )
    current_source = fingerprint_sources(workspace).sha256
    if current_source != expected_replacement_source:
        raise Phase4ERebaselineError(
            "current source differs from the explicitly reviewed replacement"
        )

    calibration_path = _workspace_path(
        workspace,
        calibration_contract_path,
        label="Phase 4E1 calibration contract",
        must_exist=True,
    )
    contract = load_phase4e1_contract(calibration_path)
    old_execution_path = _workspace_path(
        workspace,
        old_execution_contract_path,
        label="old Phase 4E execution contract",
        must_exist=True,
    )
    old_execution, old_execution_reference = _execution_reference(
        old_execution_path,
        workspace=workspace,
    )
    bootstrap_incident_path = _workspace_path(
        workspace,
        bootstrap_quarantine_incident_path,
        label="failed Bootstrap quarantine incident",
        must_exist=True,
    )
    trigger_verifier = (
        bootstrap_incident_verifier or _default_bootstrap_incident_verifier
    )
    bootstrap_trigger_payload = dict(
        trigger_verifier(
            bootstrap_incident_path,
            workspace,
            old_execution.sha256,
            expected_old_source,
            expected_replacement_source,
        )
    )
    try:
        bootstrap_trigger_self_hash = validate_sha256(
            bootstrap_trigger_payload.get("incidentSha256"),
            "failed Bootstrap quarantine incident self hash",
        )
    except ValueError as error:
        raise Phase4ERebaselineError(
            "failed Bootstrap quarantine verifier returned invalid evidence"
        ) from error
    bootstrap_trigger_reference = {
        **_file_reference(
            bootstrap_incident_path,
            workspace=workspace,
        ),
        "incidentSha256": bootstrap_trigger_self_hash,
    }
    old_budget = old_execution.require_frozen_budget()
    run_id = f"phase4e1-calibration-{contract.sha256[:16]}"
    if old_budget.baseline_run_id != run_id:
        raise Phase4ERebaselineError(
            "old frozen contract does not name the deterministic E1 run"
        )
    report = _workspace_path(
        workspace,
        report_path,
        label="old Phase 4E1 report",
        must_exist=False,
    )
    run_root = _workspace_path(
        workspace,
        calibration.DEFAULT_OUTPUT_ROOT / run_id,
        label="completed Phase 4E1 run root",
        must_exist=False,
    )
    expected_report = run_root / calibration.REPORT_NAME
    if report != expected_report:
        raise Phase4ERebaselineError(
            "old E1 report is outside its deterministic run directory"
        )
    control_root = _workspace_path(
        workspace,
        DEFAULT_CONTROL_ROOT,
        label="Phase 4 control root",
        must_exist=True,
    )
    lock_path = control_root / calibration.GPU_LOCK_NAME
    if lock_path.is_symlink():
        raise Phase4ERebaselineError("Phase 4 global lock is a symlink")

    invocation_id = f"phase4e1-completed-supersede-{os.getpid()}"
    with RunLock(
        lock_path,
        run_id=invocation_id,
    ):
        if fingerprint_sources(workspace).sha256 != expected_replacement_source:
            raise Phase4ERebaselineError(
                "replacement source changed while acquiring the global lock"
            )
        resources = _resource_preflight(
            control_root,
            artifact_limit_bytes=contract.resources.artifact_limit_bytes,
            minimum_free_disk_bytes=(
                contract.resources.minimum_free_disk_bytes
            ),
        )
        # If the atomic rename already completed, use the existing contract's
        # discovery path and then verify the completed-evidence attestation.
        if not os.path.lexists(run_root):
            replacement_native = None
            quarantine_root = _workspace_path(
                workspace,
                calibration._quarantine_root(workspace),
                label="Phase 4E1 quarantine root",
                must_exist=False,
            )
            if quarantine_root.exists():
                if not quarantine_root.is_dir():
                    raise Phase4ERebaselineError(
                        "Phase 4E1 quarantine root is invalid"
                    )
                for incident in sorted(
                    quarantine_root.glob(
                        f"{run_id}-source-{expected_old_source}-manifest-*/"
                        f"{calibration.QUARANTINE_INCIDENT_NAME}"
                    )
                ):
                    candidate = calibration._load_canonical(
                        incident,
                        "completed E1 quarantine incident",
                    )
                    if (
                        candidate.get("replacementSourceSha256")
                        == expected_replacement_source
                    ):
                        replacement_native = candidate.get(
                            "nativeBackendSha256"
                        )
                        break
            if not isinstance(replacement_native, str):
                raise Phase4ERebaselineError(
                    "completed E1 run is absent and no matching quarantine exists"
                )
            try:
                replacement_native = validate_sha256(
                    replacement_native,
                    "completed E1 quarantine native backend sha256",
                )
            except ValueError as error:
                raise Phase4ERebaselineError(
                    "completed E1 quarantine native backend is invalid"
                ) from error
            if loaded_native_backend_hash() != replacement_native:
                raise Phase4ERebaselineError(
                    "loaded native backend differs from quarantined E1 evidence"
                )
            reference = calibration._existing_quarantine_incident(
                workspace=workspace,
                contract=contract,
                run_id=run_id,
                source_sha256=expected_replacement_source,
                native_sha256=replacement_native,
            )
            if reference is None:
                raise Phase4ERebaselineError(
                    "completed E1 quarantine is not discoverable"
                )
            incident_path = _workspace_path(
                workspace,
                str(reference["path"]),
                label="completed E1 quarantine incident",
                must_exist=True,
            )
            incident_payload = calibration._load_canonical(
                incident_path,
                "completed E1 quarantine incident",
            )
            _attestation_report_sha256(
                incident_path,
                workspace=workspace,
                bootstrap_trigger_reference=bootstrap_trigger_reference,
                expected={
                    "runId": run_id,
                    "calibrationContractSha256": contract.sha256,
                    "oldExecutionContractSha256": old_execution.sha256,
                    "oldReportSha256": old_budget.baseline_report_sha256,
                    "originalSourceSha256": expected_old_source,
                    "replacementSourceSha256": expected_replacement_source,
                    "nativeBackendSha256": incident_payload[
                        "nativeBackendSha256"
                    ],
                },
            )
            if execute:
                _fsync_directory(run_root.parent)
                _fsync_directory(incident_path.parent.parent)
            return {
                "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                "status": "already-superseded",
                "runId": run_id,
                "oldReportSha256": old_budget.baseline_report_sha256,
                "oldSourceSha256": expected_old_source,
                "replacementSourceSha256": expected_replacement_source,
                "quarantineIncident": reference,
                "resources": resources,
            }

        if run_root.is_symlink() or not run_root.is_dir():
            raise Phase4ERebaselineError(
                "completed E1 run root is not a regular directory"
            )
        if not report.is_file() or report.is_symlink():
            raise Phase4ERebaselineError(
                "completed E1 report is missing or is a symlink"
            )
        verifier = historical_verifier or _default_historical_verifier
        verified = dict(verifier(report, workspace, contract))
        try:
            report_sha256 = validate_sha256(
                verified.get("reportSha256"),
                "historical E1 report self hash",
            )
            original_source_sha256 = validate_sha256(
                verified.get("sourceSha256"),
                "historical E1 source sha256",
            )
            native_sha256 = validate_sha256(
                verified.get("nativeBackendSha256"),
                "historical E1 native sha256",
            )
        except ValueError as error:
            raise Phase4ERebaselineError(
                "historical E1 verifier returned invalid provenance"
            ) from error
        if (
            verified.get("status") != "completed"
            or verified.get("decisionEligible") is not True
            or verified.get("runId") != run_id
            or report_sha256 != old_budget.baseline_report_sha256
            or original_source_sha256 != expected_old_source
        ):
            raise Phase4ERebaselineError(
                "historical E1 report differs from the reviewed supersession"
            )
        recommended = verified.get("recommendedStageBudget")
        if not isinstance(recommended, Mapping) or (
            old_budget.baseline_global_hours
            != recommended.get("baselineGlobalHours")
            or old_budget.limit_hours != recommended.get("limitHours")
            or old_execution.maximum_generations
            != contract.freeze.maximum_generations
        ):
            raise Phase4ERebaselineError(
                "old execution budget differs from the historical E1 report"
            )
        current_native_sha256 = loaded_native_backend_hash()
        if current_native_sha256 != native_sha256:
            raise Phase4ERebaselineError(
                "loaded native backend differs from historical E1 evidence"
            )

        manifest_path = run_root / calibration.RUN_MANIFEST_NAME
        try:
            manifest = load_run_manifest(manifest_path)
        except (OSError, ValueError, RunManifestError) as error:
            raise Phase4ERebaselineError(
                "historical E1 run manifest does not verify"
            ) from error
        if manifest_path.read_bytes() != canonical_json_bytes(manifest.payload):
            raise Phase4ERebaselineError(
                "historical E1 run manifest is not canonical JSON"
            )
        provenance = manifest.payload.get("provenance")
        manifest_source = (
            provenance.get("source")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            not isinstance(manifest_source, Mapping)
            or manifest_source.get("sha256") != original_source_sha256
            or manifest.sha256 != verified.get("runManifestSha256")
        ):
            raise Phase4ERebaselineError(
                "historical E1 run manifest identity changed"
            )

        stage_path = control_root / PHASE4_STAGE_LEDGER_NAME
        stage_reference = _stage_ledger_reference(
            stage_path,
            workspace=workspace,
            old_execution=old_execution,
        )
        ledger_path = control_root / calibration.GPU_LEDGER_NAME
        expected_ledger_path = _workspace_path(
            workspace,
            contract.resources.initial_ledger.path,
            label="calibration GPU ledger",
            must_exist=True,
        )
        if ledger_path != expected_ledger_path:
            raise Phase4ERebaselineError(
                "GPU ledger path differs from the calibration contract"
            )
        ledger_before, global_ledger_reference = _gpu_ledger_reference(
            ledger_path,
            workspace=workspace,
        )
        locked_trigger_payload = dict(
            trigger_verifier(
                bootstrap_incident_path,
                workspace,
                old_execution.sha256,
                expected_old_source,
                expected_replacement_source,
            )
        )
        if (
            locked_trigger_payload.get("incidentSha256")
            != bootstrap_trigger_self_hash
            or _file_reference(
                bootstrap_incident_path,
                workspace=workspace,
            )
            != {
                key: value
                for key, value in bootstrap_trigger_reference.items()
                if key != "incidentSha256"
            }
        ):
            raise Phase4ERebaselineError(
                "failed Bootstrap quarantine incident changed under lock"
            )

        quarantine_root = _workspace_path(
            workspace,
            calibration._quarantine_root(workspace),
            label="Phase 4E1 quarantine root",
            must_exist=False,
        )
        if quarantine_root.exists() and not quarantine_root.is_dir():
            raise Phase4ERebaselineError(
                "Phase 4E1 quarantine root is invalid"
            )
        destination = quarantine_root / (
            f"{run_id}-source-{original_source_sha256}-manifest-{manifest.sha256}"
        )
        if os.path.lexists(destination):
            raise Phase4ERebaselineError(
                "completed E1 quarantine destination already exists"
            )
        attestation_path = run_root / SUPERSESSION_ATTESTATION_NAME
        incident_path = run_root / calibration.QUARANTINE_INCIDENT_NAME
        if not execute:
            pending_state = None
            if os.path.lexists(attestation_path) or os.path.lexists(
                incident_path
            ):
                _verify_completed_attestation(
                    attestation_path,
                    workspace=workspace,
                    run_id=run_id,
                    contract_sha256=contract.sha256,
                    old_execution_sha256=old_execution.sha256,
                    bootstrap_trigger_reference=bootstrap_trigger_reference,
                    old_report_sha256=report_sha256,
                    original_source_sha256=original_source_sha256,
                    replacement_source_sha256=expected_replacement_source,
                    native_sha256=native_sha256,
                )
                pending_state = "attested-before-incident"
                if os.path.lexists(incident_path):
                    _verify_pre_move_incident(
                        incident_path,
                        workspace=workspace,
                        contract=contract,
                        run_root=run_root,
                        destination=destination,
                        run_id=run_id,
                        manifest_path=manifest_path,
                        manifest_sha256=manifest.sha256,
                        original_source_sha256=original_source_sha256,
                        replacement_source_sha256=(
                            expected_replacement_source
                        ),
                        native_sha256=native_sha256,
                    )
                    pending_state = "sealed-before-move"
            return {
                "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                "status": (
                    "verified-dry-run-pending"
                    if pending_state is not None
                    else "verified-dry-run"
                ),
                "pendingState": pending_state,
                "runId": run_id,
                "oldReportSha256": report_sha256,
                "oldSourceSha256": original_source_sha256,
                "replacementSourceSha256": expected_replacement_source,
                "quarantinePath": destination.relative_to(workspace).as_posix(),
                "bootstrapFailureTrigger": bootstrap_trigger_reference,
                "oldStageLedgerSha256": stage_reference["ledgerSha256"],
                "globalGpuLedgerSha256": global_ledger_reference[
                    "ledgerSha256"
                ],
                "resources": resources,
            }

        if attestation_path.exists():
            attestation = _verify_completed_attestation(
                attestation_path,
                workspace=workspace,
                run_id=run_id,
                contract_sha256=contract.sha256,
                old_execution_sha256=old_execution.sha256,
                bootstrap_trigger_reference=bootstrap_trigger_reference,
                old_report_sha256=report_sha256,
                original_source_sha256=original_source_sha256,
                replacement_source_sha256=expected_replacement_source,
                native_sha256=native_sha256,
            )
        else:
            attestation = _self_hashed(
                {
                    "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                    "kind": SUPERSESSION_KIND,
                    "status": SUPERSESSION_STATUS,
                    "reason": SUPERSESSION_REASON,
                    "verifiedAt": utc_timestamp(),
                    "runId": run_id,
                    "calibrationContractSha256": contract.sha256,
                    "oldExecutionContract": old_execution_reference,
                    "bootstrapFailureTrigger": bootstrap_trigger_reference,
                    "oldDecisionReport": {
                        **_file_reference(report, workspace=workspace),
                        "reportSha256": report_sha256,
                    },
                    "oldRunManifest": {
                        **_file_reference(
                            manifest_path,
                            workspace=workspace,
                        ),
                        "manifestSha256": manifest.sha256,
                    },
                    "originalSourceSha256": original_source_sha256,
                    "replacementSourceSha256": expected_replacement_source,
                    "nativeBackendSha256": native_sha256,
                    "stageLedgerBefore": stage_reference,
                    "globalGpuLedger": global_ledger_reference,
                },
                field="attestationSha256",
            )
            calibration._commit_immutable_json(
                attestation_path,
                attestation,
                label="completed E1 supersession attestation",
            )
            _verify_completed_attestation(
                attestation_path,
                workspace=workspace,
                run_id=run_id,
                contract_sha256=contract.sha256,
                old_execution_sha256=old_execution.sha256,
                bootstrap_trigger_reference=bootstrap_trigger_reference,
                old_report_sha256=report_sha256,
                original_source_sha256=original_source_sha256,
                replacement_source_sha256=expected_replacement_source,
                native_sha256=native_sha256,
            )

        quarantine_root.mkdir(parents=True, exist_ok=True)
        inventory = calibration._run_tree_inventory(run_root)
        if SUPERSESSION_ATTESTATION_NAME not in {
            str(item.get("path")) for item in inventory
        }:
            raise Phase4ERebaselineError(
                "completed supersession attestation is absent from inventory"
            )
        ledger_payload = calibration._load_canonical(
            ledger_path,
            "Phase 4E1 quarantine GPU ledger",
        )
        ledger = GpuBudgetLedger(ledger_path)
        e1_runs = {
            key: dict(value)
            for key, value in sorted(ledger.runs.items())
            if key.startswith("phase4e1-calibration-")
        }
        if any(run.get("status") == "running" for run in e1_runs.values()):
            raise Phase4ERebaselineError(
                "running Phase 4E1 GPU ledger entry blocks supersession"
            )
        if incident_path.exists():
            incident = _verify_pre_move_incident(
                incident_path,
                workspace=workspace,
                contract=contract,
                run_root=run_root,
                destination=destination,
                run_id=run_id,
                manifest_path=manifest_path,
                manifest_sha256=manifest.sha256,
                original_source_sha256=original_source_sha256,
                replacement_source_sha256=expected_replacement_source,
                native_sha256=native_sha256,
            )
        else:
            incident = calibration._self_hashed(
                {
                    "schemaVersion": (
                        calibration.QUARANTINE_INCIDENT_SCHEMA_VERSION
                    ),
                    "kind": calibration.QUARANTINE_INCIDENT_KIND,
                    "status": "sealed",
                    "reason": "run-manifest-source-mismatch",
                    "runId": run_id,
                    "calibrationContractSha256": contract.sha256,
                    "nativeBackendSha256": native_sha256,
                    "originalSourceSha256": original_source_sha256,
                    "replacementSourceSha256": expected_replacement_source,
                    "originalRunPath": (
                        run_root.relative_to(workspace).as_posix()
                    ),
                    "quarantinePath": (
                        destination.relative_to(workspace).as_posix()
                    ),
                    "originalRunManifest": (
                        calibration._quarantine_manifest_reference(
                            manifest_path,
                            manifest.sha256,
                        )
                    ),
                    "ledgerSnapshot": {
                        "path": ledger_path.relative_to(workspace).as_posix(),
                        "bytes": len(ledger_before),
                        "sha256": sha256_bytes(ledger_before),
                        "ledgerSha256": validate_sha256(
                            ledger_payload.get("ledgerSha256"),
                            "Phase 4 GPU ledger self hash",
                        ),
                        "e1Runs": e1_runs,
                    },
                    "files": inventory,
                },
                "incidentSha256",
            )
            calibration._commit_immutable_json(
                incident_path,
                incident,
                label="completed Phase 4E1 quarantine incident",
            )
        if fingerprint_sources(workspace).sha256 != expected_replacement_source:
            raise Phase4ERebaselineError(
                "replacement source changed before completed evidence move"
            )
        try:
            os.replace(run_root, destination)
        except OSError as error:
            raise Phase4ERebaselineError(
                "could not atomically quarantine completed E1 evidence"
            ) from error
        _fsync_directory(run_root.parent)
        _fsync_directory(destination.parent)
        moved_incident_path = destination / calibration.QUARANTINE_INCIDENT_NAME
        reference = calibration._quarantine_incident_reference(
            moved_incident_path,
            workspace=workspace,
            contract=contract,
            run_id=run_id,
            source_sha256=expected_replacement_source,
            native_sha256=native_sha256,
        )
        _verify_completed_attestation(
            destination / SUPERSESSION_ATTESTATION_NAME,
            workspace=workspace,
            run_id=run_id,
            contract_sha256=contract.sha256,
            old_execution_sha256=old_execution.sha256,
            bootstrap_trigger_reference=bootstrap_trigger_reference,
            old_report_sha256=report_sha256,
            original_source_sha256=original_source_sha256,
            replacement_source_sha256=expected_replacement_source,
            native_sha256=native_sha256,
        )
        discovered = calibration._existing_quarantine_incident(
            workspace=workspace,
            contract=contract,
            run_id=run_id,
            source_sha256=expected_replacement_source,
            native_sha256=native_sha256,
        )
        if discovered != reference:
            raise Phase4ERebaselineError(
                "replacement E1 manifest cannot discover superseded evidence"
            )
        if ledger_path.read_bytes() != ledger_before:
            raise Phase4ERebaselineError(
                "GPU ledger changed during completed E1 supersession"
            )
        if fingerprint_sources(workspace).sha256 != expected_replacement_source:
            raise Phase4ERebaselineError(
                "replacement source changed during completed evidence move"
            )
        return {
            "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
            "status": "superseded",
            "runId": run_id,
            "oldReportSha256": report_sha256,
            "oldSourceSha256": original_source_sha256,
            "replacementSourceSha256": expected_replacement_source,
            "attestationSha256": attestation["attestationSha256"],
            "quarantineIncident": reference,
            "resources": resources,
        }


def _phase4e_record_matches(
    record: Mapping[str, object] | None,
    contract: Phase4ExecutionContract,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    budget = contract.require_frozen_budget()
    return (
        record.get("baselineGlobalHours") == budget.baseline_global_hours
        and record.get("limitHours") == budget.limit_hours
        and record.get("baselineRunId") == budget.baseline_run_id
        and record.get("baselineReportSha256")
        == budget.baseline_report_sha256
        and isinstance(record.get("createdAt"), str)
        and bool(record.get("createdAt"))
    )


def _verify_expected_execution_file(
    path: Path,
    *,
    workspace: Path,
    expected_contract: Phase4ExecutionContract,
    expected_reference: Mapping[str, object],
) -> None:
    if not os.path.lexists(path):
        raise Phase4ERebaselineError(
            "replacement frozen Phase 4E execution contract is missing"
        )
    try:
        actual_contract, actual_reference = _execution_reference(
            path,
            workspace=workspace,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise Phase4ERebaselineError(
            "replacement frozen Phase 4E execution contract does not verify"
        ) from error
    if (
        actual_contract != expected_contract
        or actual_reference != dict(expected_reference)
    ):
        raise Phase4ERebaselineError(
            "replacement frozen Phase 4E execution contract changed"
        )


def _bootstrap_run_path(
    workspace: Path,
    contract: Phase4E1CalibrationContract,
) -> Path:
    from .config import load_experiment_config

    config_path = _workspace_path(
        workspace,
        contract.provenance.bootstrap_config_path,
        label="Phase 4E Bootstrap config",
        must_exist=True,
    )
    try:
        configured = load_experiment_config(config_path).run_dir
    except (OSError, TypeError, ValueError) as error:
        raise Phase4ERebaselineError(
            "Phase 4E Bootstrap config does not verify"
        ) from error
    run_path = _workspace_path(
        workspace,
        configured,
        label="canonical Phase 4E Bootstrap run",
        must_exist=False,
    )
    expected = _workspace_path(
        workspace,
        "training/runs/phase4/phase4e/bootstrap-v1",
        label="expected Phase 4E Bootstrap run",
        must_exist=False,
    )
    if run_path != expected:
        raise Phase4ERebaselineError(
            "Phase 4E Bootstrap config names a non-canonical run directory"
        )
    return run_path


def _build_frozen_execution(
    report: Mapping[str, object],
    contract: Phase4E1CalibrationContract,
) -> Phase4ExecutionContract:
    recommended = report.get("recommendedStageBudget")
    if not isinstance(recommended, Mapping):
        raise Phase4ERebaselineError(
            "replacement E1 report has no stage recommendation"
        )
    return Phase4ExecutionContract(
        schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
        stage=PHASE4E_STAGE,
        status=PHASE4_EXECUTION_STATUS_FROZEN,
        purpose="phase4e-bootstrap-and-formal-promotion",
        formal_promotion=True,
        maximum_generations=contract.freeze.maximum_generations,
        stage_budget=Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=float(recommended["baselineGlobalHours"]),
            limit_hours=float(recommended["limitHours"]),
            baseline_run_id=str(report["runId"]),
            baseline_report_sha256=str(report["reportSha256"]),
        ),
    )


def _ledger_payload_with_rebased_phase4e(
    payload: Mapping[str, object],
    *,
    new_execution: Phase4ExecutionContract,
    created_at: str,
) -> tuple[dict[str, object], dict[str, object]]:
    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        raise Phase4ERebaselineError("stage ledger has no stage mapping")
    copied_stages = {
        str(stage): dict(record)
        for stage, record in stages.items()
        if isinstance(stage, str) and isinstance(record, Mapping)
    }
    if len(copied_stages) != len(stages):
        raise Phase4ERebaselineError("stage ledger contains an invalid record")
    budget = new_execution.require_frozen_budget()
    new_record = {
        "baselineGlobalHours": budget.baseline_global_hours,
        "limitHours": budget.limit_hours,
        "baselineRunId": budget.baseline_run_id,
        "baselineReportSha256": budget.baseline_report_sha256,
        "createdAt": created_at,
    }
    copied_stages[PHASE4E_STAGE] = new_record
    core = {
        "schemaVersion": payload.get("schemaVersion"),
        "stages": copied_stages,
        "updatedAt": created_at,
    }
    after = {
        **core,
        "ledgerSha256": sha256_bytes(canonical_json_bytes(core)),
    }
    return after, new_record


def _ledger_receipt_reference(
    payload: Mapping[str, object],
    *,
    path: Path,
    workspace: Path,
) -> dict[str, object]:
    raw = canonical_json_bytes(payload)
    return {
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "ledgerSha256": validate_sha256(
            payload.get("ledgerSha256"),
            "stage ledger self hash",
        ),
        "payload": dict(payload),
    }


def _verify_ledger_receipt_reference(
    reference: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    payload = reference.get("payload")
    if not isinstance(payload, Mapping):
        raise Phase4ERebaselineError(f"{label} omitted its ledger payload")
    raw = canonical_json_bytes(payload)
    core = dict(payload)
    stored = core.pop("ledgerSha256", None)
    try:
        stored = validate_sha256(stored, f"{label} ledger self hash")
        file_sha = validate_sha256(reference.get("sha256"), f"{label} file hash")
    except ValueError as error:
        raise Phase4ERebaselineError(f"{label} hash is invalid") from error
    if (
        sha256_bytes(canonical_json_bytes(core)) != stored
        or reference.get("ledgerSha256") != stored
        or reference.get("bytes") != len(raw)
        or file_sha != sha256_bytes(raw)
    ):
        raise Phase4ERebaselineError(f"{label} does not verify")
    return dict(payload)


def _verify_rebase_prepared(
    path: Path,
    *,
    old_execution_reference: Mapping[str, object],
    new_execution_reference: Mapping[str, object],
    new_report_reference: Mapping[str, object],
    completed_supersession: Mapping[str, object],
    bootstrap_trigger: Mapping[str, object],
    workspace: Path,
    old_execution: Phase4ExecutionContract,
    new_execution: Phase4ExecutionContract,
    supersession_attestation: Mapping[str, object],
) -> dict[str, object]:
    payload = _load_canonical(path, "Phase 4E stage rebase preparation")
    _verify_self_hash(
        payload,
        field="receiptSha256",
        label="Phase 4E stage rebase preparation",
    )
    if (
        set(payload)
        != {
            "schemaVersion",
            "kind",
            "status",
            "reason",
            "createdAt",
            "oldExecutionContract",
            "newExecutionContract",
            "newDecisionReport",
            "completedE1Supersession",
            "bootstrapFailureTrigger",
            "globalGpuLedger",
            "stageLedgerBefore",
            "stageLedgerAfter",
            "receiptSha256",
        }
        or payload.get("schemaVersion") != SUPERSESSION_SCHEMA_VERSION
        or payload.get("kind") != REBASE_PREPARED_KIND
        or payload.get("status") != "prepared"
        or payload.get("reason") != SUPERSESSION_REASON
        or not isinstance(payload.get("createdAt"), str)
        or not payload.get("createdAt")
        or payload.get("oldExecutionContract") != old_execution_reference
        or payload.get("newExecutionContract") != new_execution_reference
        or payload.get("newDecisionReport") != new_report_reference
        or payload.get("completedE1Supersession") != completed_supersession
        or payload.get("bootstrapFailureTrigger") != bootstrap_trigger
        or not isinstance(payload.get("globalGpuLedger"), Mapping)
        or not isinstance(payload.get("stageLedgerBefore"), Mapping)
        or not isinstance(payload.get("stageLedgerAfter"), Mapping)
    ):
        raise Phase4ERebaselineError(
            "Phase 4E stage rebase preparation contract mismatch"
        )
    before = _verify_ledger_receipt_reference(
        payload["stageLedgerBefore"],
        label="stage ledger before rebase",
    )
    after = _verify_ledger_receipt_reference(
        payload["stageLedgerAfter"],
        label="stage ledger after rebase",
    )
    before_stages = before.get("stages")
    after_stages = after.get("stages")
    if not isinstance(before_stages, Mapping) or not isinstance(
        after_stages, Mapping
    ):
        raise Phase4ERebaselineError("stage rebase receipts have no stages")
    if {
        key: value
        for key, value in before_stages.items()
        if key != PHASE4E_STAGE
    } != {
        key: value
        for key, value in after_stages.items()
        if key != PHASE4E_STAGE
    }:
        raise Phase4ERebaselineError(
            "stage rebase would alter a non-Phase 4E record"
        )
    attested_stage = supersession_attestation.get("stageLedgerBefore")
    if not isinstance(attested_stage, Mapping):
        raise Phase4ERebaselineError(
            "completed supersession omitted the old stage ledger"
        )
    if (
        before != attested_stage.get("payload")
        or payload["stageLedgerBefore"].get("sha256")
        != attested_stage.get("sha256")
        or payload["stageLedgerBefore"].get("ledgerSha256")
        != attested_stage.get("ledgerSha256")
        or before_stages.get(PHASE4E_STAGE)
        != attested_stage.get("phase4eRecord")
        or not _phase4e_record_matches(
            before_stages.get(PHASE4E_STAGE),
            old_execution,
        )
        or not _phase4e_record_matches(
            after_stages.get(PHASE4E_STAGE),
            new_execution,
        )
        or after_stages[PHASE4E_STAGE].get("createdAt")
        != payload.get("createdAt")
    ):
        raise Phase4ERebaselineError(
            "Phase 4E stage rebase is not anchored to attested old/new records"
        )
    _verify_gpu_ledger_append_only(
        payload["globalGpuLedger"],
        workspace=workspace,
    )
    return payload


def _verify_rebase_completed(
    path: Path,
    *,
    prepared_reference: Mapping[str, object],
    stage_after: Mapping[str, object],
) -> dict[str, object]:
    payload = _load_canonical(path, "completed Phase 4E stage rebase")
    _verify_self_hash(
        payload,
        field="receiptSha256",
        label="completed Phase 4E stage rebase",
    )
    if (
        set(payload)
        != {
            "schemaVersion",
            "kind",
            "status",
            "completedAt",
            "preparedReceipt",
            "stageLedgerAfter",
            "receiptSha256",
        }
        or payload.get("schemaVersion") != SUPERSESSION_SCHEMA_VERSION
        or payload.get("kind") != REBASE_COMPLETED_KIND
        or payload.get("status") != "completed"
        or not isinstance(payload.get("completedAt"), str)
        or not payload.get("completedAt")
        or payload.get("preparedReceipt") != prepared_reference
        or payload.get("stageLedgerAfter") != stage_after
    ):
        raise Phase4ERebaselineError(
            "completed Phase 4E stage rebase contract mismatch"
        )
    return payload


def freeze_and_rebase_phase4e(
    *,
    workspace_root: str | Path,
    calibration_contract_path: str | Path = DEFAULT_CALIBRATION_CONTRACT,
    report_path: str | Path,
    old_execution_contract_path: str | Path = DEFAULT_OLD_EXECUTION_CONTRACT,
    new_execution_contract_path: str | Path = DEFAULT_NEW_EXECUTION_CONTRACT,
    bootstrap_quarantine_incident_path: str | Path,
    expected_old_source_sha256: str,
    expected_replacement_source_sha256: str,
    execute: bool = False,
    decision_verifier: Callable[
        [Path, Path, Phase4E1CalibrationContract], Mapping[str, object]
    ]
    | None = None,
    bootstrap_incident_verifier: BootstrapIncidentVerifier | None = None,
) -> dict[str, object]:
    """Freeze a replacement E1 report and journal an atomic stage-ledger rebase."""

    from . import phase4e1_calibration as calibration

    workspace = _absolute_without_resolving(Path(workspace_root))
    if workspace.is_symlink() or not workspace.is_dir():
        raise Phase4ERebaselineError(
            "workspace root is missing or is a symlink"
        )
    try:
        expected_old_source = validate_sha256(
            expected_old_source_sha256,
            "expected old source sha256",
        )
        expected_replacement_source = validate_sha256(
            expected_replacement_source_sha256,
            "expected replacement source sha256",
        )
    except ValueError as error:
        raise Phase4ERebaselineError("source expectation is invalid") from error
    if fingerprint_sources(workspace).sha256 != expected_replacement_source:
        raise Phase4ERebaselineError(
            "current source differs from the replacement E1 source"
        )
    calibration_path = _workspace_path(
        workspace,
        calibration_contract_path,
        label="Phase 4E1 calibration contract",
        must_exist=True,
    )
    calibration_contract = load_phase4e1_contract(calibration_path)
    old_execution_path = _workspace_path(
        workspace,
        old_execution_contract_path,
        label="old Phase 4E execution contract",
        must_exist=True,
    )
    old_execution, old_execution_reference = _execution_reference(
        old_execution_path,
        workspace=workspace,
    )
    old_budget = old_execution.require_frozen_budget()
    report = _workspace_path(
        workspace,
        report_path,
        label="replacement Phase 4E1 report",
        must_exist=True,
    )
    verifier = decision_verifier
    if verifier is None:
        def verifier(
            target: Path,
            root: Path,
            contract: Phase4E1CalibrationContract,
        ) -> Mapping[str, object]:
            return calibration.verify_phase4e1_decision_report(
                workspace_root=root,
                report_path=target,
                contract_path=calibration_path,
                verify_ledger=True,
            )
    replacement_report = dict(
        verifier(report, workspace, calibration_contract)
    )
    if (
        replacement_report.get("status") != "completed"
        or replacement_report.get("decisionEligible") is not True
        or replacement_report.get("sourceSha256")
        != expected_replacement_source
        or replacement_report.get("reportSha256")
        == old_budget.baseline_report_sha256
    ):
        raise Phase4ERebaselineError(
            "replacement E1 report is not a new eligible source-bound decision"
        )
    native_sha256 = validate_sha256(
        replacement_report.get("nativeBackendSha256"),
        "replacement E1 native backend sha256",
    )
    if loaded_native_backend_hash() != native_sha256:
        raise Phase4ERebaselineError(
            "loaded native backend differs from replacement E1 evidence"
        )
    new_execution = _build_frozen_execution(
        replacement_report,
        calibration_contract,
    )
    new_execution_path = _workspace_path(
        workspace,
        new_execution_contract_path,
        label="new Phase 4E execution contract",
        must_exist=False,
    )
    new_execution_reference = {
        "path": new_execution_path.relative_to(workspace).as_posix(),
        "bytes": len(canonical_json_bytes(new_execution.to_payload())),
        "sha256": sha256_bytes(
            canonical_json_bytes(new_execution.to_payload())
        ),
        "executionContractSha256": new_execution.sha256,
    }
    new_report_reference = {
        **_file_reference(report, workspace=workspace),
        "reportSha256": replacement_report["reportSha256"],
    }

    bootstrap_incident_path = _workspace_path(
        workspace,
        bootstrap_quarantine_incident_path,
        label="failed Bootstrap quarantine incident",
        must_exist=True,
    )
    trigger_verifier = (
        bootstrap_incident_verifier or _default_bootstrap_incident_verifier
    )
    bootstrap_payload = dict(
        trigger_verifier(
            bootstrap_incident_path,
            workspace,
            old_execution.sha256,
            expected_old_source,
            expected_replacement_source,
        )
    )
    bootstrap_reference = {
        **_file_reference(
            bootstrap_incident_path,
            workspace=workspace,
        ),
        "incidentSha256": validate_sha256(
            bootstrap_payload.get("incidentSha256"),
            "Bootstrap quarantine incident self hash",
        ),
    }

    manifest = load_run_manifest(report.parent / calibration.RUN_MANIFEST_NAME)
    metadata = manifest.payload.get("metadata")
    supersession_reference = (
        metadata.get("quarantineIncident")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(supersession_reference, Mapping):
        raise Phase4ERebaselineError(
            "replacement E1 manifest did not bind completed supersession"
        )
    supersession_reference = dict(supersession_reference)
    supersession_path = _workspace_path(
        workspace,
        str(supersession_reference.get("path")),
        label="completed E1 supersession incident",
        must_exist=True,
    )
    verified_supersession = calibration._verify_quarantine_incident_reference(
        supersession_reference,
        workspace=workspace,
        contract=calibration_contract,
        run_id=str(replacement_report["runId"]),
        source_sha256=expected_replacement_source,
        native_sha256=native_sha256,
    )
    if verified_supersession != supersession_reference:
        raise Phase4ERebaselineError(
            "replacement E1 supersession reference changed"
        )
    supersession_attestation = _verify_completed_attestation(
        supersession_path.parent / SUPERSESSION_ATTESTATION_NAME,
        workspace=workspace,
        run_id=str(replacement_report["runId"]),
        contract_sha256=calibration_contract.sha256,
        old_execution_sha256=old_execution.sha256,
        bootstrap_trigger_reference=bootstrap_reference,
        old_report_sha256=old_budget.baseline_report_sha256,
        original_source_sha256=expected_old_source,
        replacement_source_sha256=expected_replacement_source,
        native_sha256=native_sha256,
    )
    bootstrap_run_path = _bootstrap_run_path(
        workspace,
        calibration_contract,
    )

    control_root = _workspace_path(
        workspace,
        DEFAULT_CONTROL_ROOT,
        label="Phase 4 control root",
        must_exist=True,
    )
    lock_path = control_root / calibration.GPU_LOCK_NAME
    if lock_path.is_symlink():
        raise Phase4ERebaselineError("Phase 4 global lock is a symlink")
    with RunLock(
        lock_path,
        run_id=f"phase4e-stage-rebase-{os.getpid()}",
    ):
        if fingerprint_sources(workspace).sha256 != expected_replacement_source:
            raise Phase4ERebaselineError(
                "replacement source changed while acquiring rebase lock"
            )
        locked_attestation = _verify_completed_attestation(
            supersession_path.parent / SUPERSESSION_ATTESTATION_NAME,
            workspace=workspace,
            run_id=str(replacement_report["runId"]),
            contract_sha256=calibration_contract.sha256,
            old_execution_sha256=old_execution.sha256,
            bootstrap_trigger_reference=bootstrap_reference,
            old_report_sha256=old_budget.baseline_report_sha256,
            original_source_sha256=expected_old_source,
            replacement_source_sha256=expected_replacement_source,
            native_sha256=native_sha256,
        )
        if locked_attestation != supersession_attestation:
            raise Phase4ERebaselineError(
                "completed E1 supersession changed while acquiring rebase lock"
            )
        resources = _resource_preflight(
            control_root,
            artifact_limit_bytes=(
                calibration_contract.resources.artifact_limit_bytes
            ),
            minimum_free_disk_bytes=(
                calibration_contract.resources.minimum_free_disk_bytes
            ),
        )
        locked_report = dict(
            verifier(report, workspace, calibration_contract)
        )
        if locked_report != replacement_report:
            raise Phase4ERebaselineError(
                "replacement E1 report changed while acquiring rebase lock"
            )
        global_path = control_root / calibration.GPU_LEDGER_NAME
        global_before, global_reference = _gpu_ledger_reference(
            global_path,
            workspace=workspace,
        )
        if GpuBudgetLedger(global_path).total_hours < float(
            new_execution.stage_budget.baseline_global_hours
        ):
            raise Phase4ERebaselineError(
                "global GPU ledger regressed below replacement E1 baseline"
            )
        stage_path = control_root / PHASE4_STAGE_LEDGER_NAME
        stage_payload = _load_canonical(
            stage_path,
            "Phase 4 stage budget ledger",
        )
        Phase4StageBudgetLedger(stage_path)
        stages = stage_payload.get("stages")
        old_record = (
            stages.get(PHASE4E_STAGE)
            if isinstance(stages, Mapping)
            else None
        )

        transaction_root = _workspace_path(
            workspace,
            REBASE_RECEIPT_ROOT / new_execution.sha256,
            label="Phase 4E stage rebase transaction",
            must_exist=False,
        )
        prepared_path = transaction_root / "prepared.json"
        completed_path = transaction_root / "completed.json"
        if os.path.lexists(new_execution_path):
            _verify_expected_execution_file(
                new_execution_path,
                workspace=workspace,
                expected_contract=new_execution,
                expected_reference=new_execution_reference,
            )
        if completed_path.exists():
            _verify_expected_execution_file(
                new_execution_path,
                workspace=workspace,
                expected_contract=new_execution,
                expected_reference=new_execution_reference,
            )
            prepared = _verify_rebase_prepared(
                prepared_path,
                old_execution_reference=old_execution_reference,
                new_execution_reference=new_execution_reference,
                new_report_reference=new_report_reference,
                completed_supersession=supersession_reference,
                bootstrap_trigger=bootstrap_reference,
                workspace=workspace,
                old_execution=old_execution,
                new_execution=new_execution,
                supersession_attestation=supersession_attestation,
            )
            prepared_reference = {
                **_file_reference(prepared_path, workspace=workspace),
                "receiptSha256": prepared["receiptSha256"],
            }
            completed = _verify_rebase_completed(
                completed_path,
                prepared_reference=prepared_reference,
                stage_after=prepared["stageLedgerAfter"],
            )
            current_stage = _load_canonical(
                stage_path,
                "rebased Phase 4 stage ledger",
            )
            if current_stage != prepared["stageLedgerAfter"]["payload"]:
                raise Phase4ERebaselineError(
                    "completed stage rebase no longer matches the ledger"
                )
            return {
                "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                "status": "already-completed",
                "newExecutionContractSha256": new_execution.sha256,
                "newReportSha256": replacement_report["reportSha256"],
                "preparedReceiptSha256": prepared["receiptSha256"],
                "completedReceiptSha256": completed["receiptSha256"],
                "resources": resources,
            }

        if execute and os.path.lexists(bootstrap_run_path):
            raise Phase4ERebaselineError(
                "canonical Phase 4E Bootstrap run still exists; "
                "failed lineage must remain quarantined before rebase"
            )
        if prepared_path.exists():
            _verify_expected_execution_file(
                new_execution_path,
                workspace=workspace,
                expected_contract=new_execution,
                expected_reference=new_execution_reference,
            )
            prepared = _verify_rebase_prepared(
                prepared_path,
                old_execution_reference=old_execution_reference,
                new_execution_reference=new_execution_reference,
                new_report_reference=new_report_reference,
                completed_supersession=supersession_reference,
                bootstrap_trigger=bootstrap_reference,
                workspace=workspace,
                old_execution=old_execution,
                new_execution=new_execution,
                supersession_attestation=supersession_attestation,
            )
            before_payload = prepared["stageLedgerBefore"]["payload"]
            after_payload = prepared["stageLedgerAfter"]["payload"]
            current_stage = _load_canonical(
                stage_path,
                "Phase 4 stage budget ledger",
            )
            if current_stage == before_payload:
                pending_state = "prepared-before-ledger"
            elif current_stage == after_payload:
                pending_state = "prepared-after-ledger"
            else:
                raise Phase4ERebaselineError(
                    "stage ledger changed outside the prepared rebase transaction"
                )
            if not execute:
                return {
                    "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                    "status": "verified-dry-run-pending",
                    "pendingState": pending_state,
                    "newExecutionContractSha256": new_execution.sha256,
                    "newReportSha256": replacement_report["reportSha256"],
                    "oldStageLedgerSha256": before_payload[
                        "ledgerSha256"
                    ],
                    "newStageLedgerSha256": after_payload[
                        "ledgerSha256"
                    ],
                    "resources": resources,
                }
        else:
            attested_stage = supersession_attestation.get(
                "stageLedgerBefore"
            )
            if (
                not isinstance(attested_stage, Mapping)
                or not _phase4e_record_matches(old_record, old_execution)
                or stage_payload != attested_stage.get("payload")
            ):
                raise Phase4ERebaselineError(
                    "stage ledger is not anchored to the attested old contract"
                )
            created_at = utc_timestamp()
            after_payload, new_record = _ledger_payload_with_rebased_phase4e(
                stage_payload,
                new_execution=new_execution,
                created_at=created_at,
            )
            if not _phase4e_record_matches(new_record, new_execution):
                raise Phase4ERebaselineError(
                    "proposed Phase 4E stage record is invalid"
                )
            prepared = _self_hashed(
                {
                    "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                    "kind": REBASE_PREPARED_KIND,
                    "status": "prepared",
                    "reason": SUPERSESSION_REASON,
                    "createdAt": created_at,
                    "oldExecutionContract": old_execution_reference,
                    "newExecutionContract": new_execution_reference,
                    "newDecisionReport": new_report_reference,
                    "completedE1Supersession": supersession_reference,
                    "bootstrapFailureTrigger": bootstrap_reference,
                    "globalGpuLedger": global_reference,
                    "stageLedgerBefore": _ledger_receipt_reference(
                        stage_payload,
                        path=stage_path,
                        workspace=workspace,
                    ),
                    "stageLedgerAfter": _ledger_receipt_reference(
                        after_payload,
                        path=stage_path,
                        workspace=workspace,
                    ),
                },
                field="receiptSha256",
            )
            if not execute:
                return {
                    "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                    "status": "verified-dry-run",
                    "newExecutionContractSha256": new_execution.sha256,
                    "newReportSha256": replacement_report["reportSha256"],
                    "oldStageLedgerSha256": stage_payload["ledgerSha256"],
                    "newStageLedgerSha256": after_payload["ledgerSha256"],
                    "remainingGpuHours": (
                        new_execution.stage_budget.limit_hours
                        - (
                            GpuBudgetLedger(global_path).total_hours
                            - new_execution.stage_budget.baseline_global_hours
                        )
                    ),
                    "resources": resources,
                }
            calibration._commit_immutable_json(
                new_execution_path,
                new_execution.to_payload(),
                label="replacement frozen Phase 4E execution contract",
            )
            _verify_expected_execution_file(
                new_execution_path,
                workspace=workspace,
                expected_contract=new_execution,
                expected_reference=new_execution_reference,
            )
            calibration._commit_immutable_json(
                prepared_path,
                prepared,
                label="Phase 4E stage rebase preparation",
            )
            prepared = _verify_rebase_prepared(
                prepared_path,
                old_execution_reference=old_execution_reference,
                new_execution_reference=new_execution_reference,
                new_report_reference=new_report_reference,
                completed_supersession=supersession_reference,
                bootstrap_trigger=bootstrap_reference,
                workspace=workspace,
                old_execution=old_execution,
                new_execution=new_execution,
                supersession_attestation=supersession_attestation,
            )

        before_payload = prepared["stageLedgerBefore"]["payload"]
        after_payload = prepared["stageLedgerAfter"]["payload"]
        current_stage = _load_canonical(
            stage_path,
            "Phase 4 stage budget ledger",
        )
        if current_stage == before_payload:
            atomic_replace_bytes(
                stage_path,
                canonical_json_bytes(after_payload),
            )
        elif current_stage != after_payload:
            raise Phase4ERebaselineError(
                "stage ledger changed outside the prepared rebase transaction"
            )
        reopened = Phase4StageBudgetLedger(stage_path)
        reopened.ensure_stage(
            PHASE4E_STAGE,
            float(new_execution.stage_budget.baseline_global_hours),
            float(new_execution.stage_budget.limit_hours),
            baseline_run_id=new_execution.stage_budget.baseline_run_id,
            baseline_report_sha256=(
                new_execution.stage_budget.baseline_report_sha256
            ),
        )
        if global_path.read_bytes() != global_before:
            raise Phase4ERebaselineError(
                "global GPU ledger changed during stage rebase"
            )
        prepared_reference = {
            **_file_reference(prepared_path, workspace=workspace),
            "receiptSha256": prepared["receiptSha256"],
        }
        completed_at = utc_timestamp()
        if not completed_at:
            raise Phase4ERebaselineError(
                "completed Phase 4E stage rebase timestamp is empty"
            )
        completed = _self_hashed(
            {
                "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
                "kind": REBASE_COMPLETED_KIND,
                "status": "completed",
                "completedAt": completed_at,
                "preparedReceipt": prepared_reference,
                "stageLedgerAfter": prepared["stageLedgerAfter"],
            },
            field="receiptSha256",
        )
        calibration._commit_immutable_json(
            completed_path,
            completed,
            label="completed Phase 4E stage rebase",
        )
        _verify_rebase_completed(
            completed_path,
            prepared_reference=prepared_reference,
            stage_after=prepared["stageLedgerAfter"],
        )
        if global_path.read_bytes() != global_before:
            raise Phase4ERebaselineError(
                "global GPU ledger changed while sealing stage rebase"
            )
        return {
            "schemaVersion": SUPERSESSION_SCHEMA_VERSION,
            "status": "completed",
            "newExecutionContractPath": (
                new_execution_path.relative_to(workspace).as_posix()
            ),
            "newExecutionContractSha256": new_execution.sha256,
            "newReportSha256": replacement_report["reportSha256"],
            "stageLedgerSha256": after_payload["ledgerSha256"],
            "preparedReceiptSha256": prepared["receiptSha256"],
            "completedReceiptSha256": completed["receiptSha256"],
            "resources": resources,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Phase 4E completed-evidence supersession."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    supersede = subparsers.add_parser("supersede-e1")
    supersede.add_argument("--workspace-root", default=".")
    supersede.add_argument(
        "--calibration-contract",
        default=str(DEFAULT_CALIBRATION_CONTRACT),
    )
    supersede.add_argument("--report", required=True)
    supersede.add_argument(
        "--old-execution-contract",
        default=str(DEFAULT_OLD_EXECUTION_CONTRACT),
    )
    supersede.add_argument(
        "--bootstrap-quarantine-incident",
        required=True,
    )
    supersede.add_argument("--expected-old-source-sha256", required=True)
    supersede.add_argument(
        "--expected-replacement-source-sha256",
        required=True,
    )
    supersede.add_argument(
        "--execute",
        action="store_true",
        help="perform the sealed attestation write and atomic quarantine move",
    )
    rebase = subparsers.add_parser("freeze-rebase")
    rebase.add_argument("--workspace-root", default=".")
    rebase.add_argument(
        "--calibration-contract",
        default=str(DEFAULT_CALIBRATION_CONTRACT),
    )
    rebase.add_argument("--report", required=True)
    rebase.add_argument(
        "--old-execution-contract",
        default=str(DEFAULT_OLD_EXECUTION_CONTRACT),
    )
    rebase.add_argument(
        "--new-execution-contract",
        default=str(DEFAULT_NEW_EXECUTION_CONTRACT),
    )
    rebase.add_argument(
        "--bootstrap-quarantine-incident",
        required=True,
    )
    rebase.add_argument("--expected-old-source-sha256", required=True)
    rebase.add_argument(
        "--expected-replacement-source-sha256",
        required=True,
    )
    rebase.add_argument(
        "--execute",
        action="store_true",
        help="write frozen-v2 and atomically rebase the Phase 4E stage ledger",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "workspace_root": args.workspace_root,
        "calibration_contract_path": args.calibration_contract,
        "report_path": args.report,
        "old_execution_contract_path": args.old_execution_contract,
        "bootstrap_quarantine_incident_path": (
            args.bootstrap_quarantine_incident
        ),
        "expected_old_source_sha256": args.expected_old_source_sha256,
        "expected_replacement_source_sha256": (
            args.expected_replacement_source_sha256
        ),
        "execute": args.execute,
    }
    if args.command == "supersede-e1":
        result = supersede_completed_e1_evidence(**common)
    elif args.command == "freeze-rebase":
        result = freeze_and_rebase_phase4e(
            **common,
            new_execution_contract_path=args.new_execution_contract,
        )
    else:
        raise AssertionError("unsupported Phase 4E rebaseline command")
    print(canonical_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SUPERSESSION_SCHEMA_VERSION",
    "SUPERSESSION_KIND",
    "SUPERSESSION_STATUS",
    "SUPERSESSION_REASON",
    "SUPERSESSION_ATTESTATION_NAME",
    "Phase4ERebaselineError",
    "supersede_completed_e1_evidence",
    "freeze_and_rebase_phase4e",
    "main",
]
