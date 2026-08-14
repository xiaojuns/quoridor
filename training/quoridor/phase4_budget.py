"""Shared, self-validated stage budgets for local Phase 4 GPU work."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping

from .run_manifest import (
    atomic_replace_bytes,
    canonical_json_bytes,
    sha256_bytes,
    utc_timestamp,
    validate_sha256,
)


PHASE4_STAGE_LEDGER_SCHEMA_VERSION = 1
PHASE4_CONTROL_RELATIVE = Path("training/runs/phase4")
PHASE4_STAGE_LEDGER_NAME = "stage-budget-ledger.json"
PHASE4D_STAGE = "phase4d"
PHASE4E_STAGE = "phase4e"
PHASE4F5_STAGE = "phase4f5"
PHASE4F_RECOVERY_STAGE = "phase4f-recovery"
PHASE4F18_PILOT_STAGE = "phase4f18-pilot"
PHASE4F35_PILOT_STAGE = "phase4f35-stage-a2-pilot"
PHASE4F36_PILOT_STAGE = "phase4f36-stage-a2-pilot-v2"
PHASE4F38_PILOT_STAGE = "phase4f38-stage-a2-policy-only-pilot"
PHASE4F_STAGE_A = "phase4f-stage-a"
PHASE4F_STAGE_A_CONTINUATION = "phase4f-stage-a-continuation"
PHASE4F_STAGE_B_PILOT = "phase4f-stage-b-pilot"
PHASE4F_STAGE_B = "phase4f-stage-b"
PHASE4F_STAGE_B_CONTINUATION = "phase4f-stage-b-continuation"
PHASE4F_STAGE_B_EXTENDED = "phase4f-stage-b-extended"
PHASE4D_BASELINE_GLOBAL_GPU_HOURS = 4.038853156647076
PHASE4D_GPU_HOUR_LIMIT = 10.0
PHASE4D_BASELINE_RUN_ID = "phase4-calibration-20260721-171946-972095"
PHASE4D_BASELINE_REPORT_SHA256 = (
    "0ab2211d1e890154d029a27e75328a693a711dd5fa59d13bc24199f6c840df74"
)
PHASE4D_BASELINE_REPORT_RELATIVE = (
    Path("calibration") / PHASE4D_BASELINE_RUN_ID / "report.json"
)
PHASE4E1_BASELINE_SCHEMA_VERSION = 1
PHASE4E1_BASELINE_KIND = "phase4e1-standard-calibration-and-arena-smoke"
PHASE4E1_BASELINE_RELATIVE_ROOT = Path("phase4e1") / "calibration"
_SAFE_BASELINE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class Phase4StageBudgetError(RuntimeError):
    """Raised when a persisted stage contract is invalid or exhausted."""


def _non_negative_hours(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _positive_hours(value: object, label: str) -> float:
    hours = _non_negative_hours(value, label)
    if hours <= 0:
        raise ValueError(f"{label} must be positive")
    return hours


class Phase4StageBudgetLedger:
    """Freeze stage baselines while deriving usage from the global GPU ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.stages: dict[str, dict[str, object]] = {}
        if self.path.exists():
            self._load()

    def ensure_stage(
        self,
        stage: str,
        baseline_global_hours: float,
        limit_hours: float,
        *,
        baseline_run_id: str | None = None,
        baseline_report_sha256: str | None = None,
    ) -> dict[str, object]:
        """Create an immutable stage contract, or validate the existing one."""

        if not isinstance(stage, str) or not stage or len(stage) > 64:
            raise ValueError("stage must be a non-empty string of at most 64 characters")
        baseline = _non_negative_hours(
            baseline_global_hours, "baseline_global_hours"
        )
        limit = _positive_hours(limit_hours, "limit_hours")
        if stage == PHASE4D_STAGE:
            if baseline_run_id is None and baseline_report_sha256 is None:
                baseline_run_id = PHASE4D_BASELINE_RUN_ID
                baseline_report_sha256 = PHASE4D_BASELINE_REPORT_SHA256
            if (
                baseline != PHASE4D_BASELINE_GLOBAL_GPU_HOURS
                or limit != PHASE4D_GPU_HOUR_LIMIT
                or baseline_run_id != PHASE4D_BASELINE_RUN_ID
                or baseline_report_sha256 != PHASE4D_BASELINE_REPORT_SHA256
            ):
                raise Phase4StageBudgetError(
                    "Phase 4D baseline and limit do not match the frozen contract"
                )
        self._validate_baseline_provenance(
            baseline_run_id,
            baseline_report_sha256,
        )
        existing = self.stages.get(stage)
        if existing is not None:
            if (
                float(existing["baselineGlobalHours"]) != baseline
                or float(existing["limitHours"]) != limit
                or existing.get("baselineRunId") != baseline_run_id
                or existing.get("baselineReportSha256")
                != baseline_report_sha256
            ):
                raise Phase4StageBudgetError(
                    f"persisted {stage} stage budget does not match the frozen contract"
                )
            return dict(existing)

        record: dict[str, object] = {
            "baselineGlobalHours": baseline,
            "limitHours": limit,
            "baselineRunId": baseline_run_id,
            "baselineReportSha256": baseline_report_sha256,
            "createdAt": utc_timestamp(),
        }
        self.stages[stage] = record
        self._save()
        return dict(record)

    @staticmethod
    def _validate_baseline_provenance(
        baseline_run_id: object,
        baseline_report_sha256: object,
    ) -> None:
        if (baseline_run_id is None) != (baseline_report_sha256 is None):
            raise Phase4StageBudgetError(
                "Phase 4 stage baseline provenance must be complete"
            )
        if baseline_run_id is not None and (
            not isinstance(baseline_run_id, str) or not baseline_run_id
        ):
            raise Phase4StageBudgetError(
                "Phase 4 stage baseline run must be a non-empty string"
            )
        if baseline_report_sha256 is not None and (
            not isinstance(baseline_report_sha256, str)
            or len(baseline_report_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in baseline_report_sha256
            )
        ):
            raise Phase4StageBudgetError(
                "Phase 4 stage baseline report hash is invalid"
            )

    def assert_available(
        self,
        stage: str,
        global_hours: float,
    ) -> dict[str, object]:
        """Return current stage usage, failing at the frozen stop line."""

        usage = self.usage(stage, global_hours)
        if not usage["available"]:
            raise Phase4StageBudgetError(
                f"{stage} {usage['limitHours']:g} GPU-hour stage budget is exhausted"
            )
        return usage

    def usage(self, stage: str, global_hours: float) -> dict[str, object]:
        """Derive stage usage from the authoritative cumulative global hours."""

        total = _non_negative_hours(global_hours, "global_hours")
        try:
            record = self.stages[stage]
        except KeyError as error:
            raise Phase4StageBudgetError(
                f"stage budget has not been initialized: {stage}"
            ) from error
        baseline = float(record["baselineGlobalHours"])
        limit = float(record["limitHours"])
        if total < baseline:
            raise Phase4StageBudgetError(
                f"global GPU ledger regressed below the {stage} baseline"
            )
        used = total - baseline
        return {
            "stage": stage,
            "baselineGlobalHours": baseline,
            "globalHours": total,
            "usedHours": used,
            "limitHours": limit,
            "remainingHours": max(0.0, limit - used),
            "available": used < limit,
        }

    def _core(self) -> dict[str, object]:
        return {
            "schemaVersion": PHASE4_STAGE_LEDGER_SCHEMA_VERSION,
            "stages": self.stages,
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
            raise Phase4StageBudgetError(
                "Phase 4 stage budget ledger is unreadable"
            ) from error
        if not isinstance(payload, Mapping):
            raise Phase4StageBudgetError(
                "Phase 4 stage budget ledger must be a mapping"
            )
        stored_hash = payload.get("ledgerSha256")
        core = dict(payload)
        core.pop("ledgerSha256", None)
        if (
            not isinstance(stored_hash, str)
            or sha256_bytes(canonical_json_bytes(core)) != stored_hash
        ):
            raise Phase4StageBudgetError(
                "Phase 4 stage budget ledger checksum mismatch"
            )
        stages = core.get("stages")
        if (
            core.get("schemaVersion") != PHASE4_STAGE_LEDGER_SCHEMA_VERSION
            or not isinstance(stages, Mapping)
        ):
            raise Phase4StageBudgetError(
                "Phase 4 stage budget ledger contract mismatch"
            )

        normalized: dict[str, dict[str, object]] = {}
        for stage, raw_record in stages.items():
            if not isinstance(stage, str) or not stage or not isinstance(
                raw_record, Mapping
            ):
                raise Phase4StageBudgetError(
                    "Phase 4 stage budget ledger contains an invalid stage"
                )
            try:
                baseline = _non_negative_hours(
                    raw_record.get("baselineGlobalHours"),
                    "baselineGlobalHours",
                )
                limit = _positive_hours(raw_record.get("limitHours"), "limitHours")
            except (TypeError, ValueError) as error:
                raise Phase4StageBudgetError(
                    "Phase 4 stage budget ledger contains invalid hours"
                ) from error
            created_at = raw_record.get("createdAt")
            if not isinstance(created_at, str) or not created_at:
                raise Phase4StageBudgetError(
                    "Phase 4 stage budget ledger contains an invalid timestamp"
                )
            baseline_run_id = raw_record.get("baselineRunId")
            baseline_report_sha256 = raw_record.get("baselineReportSha256")
            try:
                self._validate_baseline_provenance(
                    baseline_run_id,
                    baseline_report_sha256,
                )
            except Phase4StageBudgetError as error:
                raise Phase4StageBudgetError(
                    "Phase 4 stage budget ledger contains invalid baseline provenance"
                ) from error
            if stage == PHASE4D_STAGE and (
                baseline_run_id != PHASE4D_BASELINE_RUN_ID
                or baseline_report_sha256 != PHASE4D_BASELINE_REPORT_SHA256
                or baseline != PHASE4D_BASELINE_GLOBAL_GPU_HOURS
                or limit != PHASE4D_GPU_HOUR_LIMIT
            ):
                raise Phase4StageBudgetError(
                    "Phase 4D stage budget does not match the frozen contract"
                )
            normalized[stage] = {
                "baselineGlobalHours": baseline,
                "limitHours": limit,
                "baselineRunId": baseline_run_id,
                "baselineReportSha256": baseline_report_sha256,
                "createdAt": created_at,
            }
        self.stages = normalized


def verify_phase4d_baseline_report(control_root: str | Path) -> dict[str, object]:
    """Verify the reviewed full-calibration report that anchors Phase 4D."""

    path = Path(control_root).resolve() / PHASE4D_BASELINE_REPORT_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4StageBudgetError(
            "the frozen Phase 4D baseline report is unreadable"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase4StageBudgetError(
            "the frozen Phase 4D baseline report must be a mapping"
        )
    stored_hash = payload.get("reportSha256")
    core = dict(payload)
    core.pop("reportSha256", None)
    resource_usage = payload.get("resourceUsage")
    total_hours = (
        resource_usage.get("totalGpuHours")
        if isinstance(resource_usage, Mapping)
        else None
    )
    if (
        stored_hash != PHASE4D_BASELINE_REPORT_SHA256
        or sha256_bytes(canonical_json_bytes(core)) != stored_hash
        or payload.get("runId") != PHASE4D_BASELINE_RUN_ID
        or payload.get("status") != "completed"
        or total_hours != PHASE4D_BASELINE_GLOBAL_GPU_HOURS
    ):
        raise Phase4StageBudgetError(
            "the frozen Phase 4D baseline report failed provenance validation"
        )
    return {
        "path": str(path),
        "runId": PHASE4D_BASELINE_RUN_ID,
        "reportSha256": PHASE4D_BASELINE_REPORT_SHA256,
        "globalGpuHours": PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
        "status": "verified",
    }


def _phase4e1_report_path(
    control_root: str | Path,
    baseline_run_id: object,
) -> Path:
    if (
        not isinstance(baseline_run_id, str)
        or _SAFE_BASELINE_RUN_ID.fullmatch(baseline_run_id) is None
    ):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline run id is not a safe path component"
        )
    root = Path(control_root)
    if root.is_symlink() or not root.is_dir():
        raise Phase4StageBudgetError(
            "the Phase 4 control root is missing or is a symlink"
        )
    root = root.resolve()
    relative = PHASE4E1_BASELINE_RELATIVE_ROOT / baseline_run_id / "report.json"
    path = root / relative
    for component in (
        root / "phase4e1",
        root / PHASE4E1_BASELINE_RELATIVE_ROOT,
        path.parent,
        path,
    ):
        if component.is_symlink():
            raise Phase4StageBudgetError(
                "the Phase 4E baseline report path contains a symlink"
            )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report is missing or escapes the control root"
        ) from error
    if resolved != path or not resolved.is_file():
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report path is not a regular file"
        )
    return resolved


def verify_phase4e_baseline_report(
    control_root: str | Path,
    *,
    baseline_run_id: object,
    baseline_report_sha256: object,
    baseline_global_hours: object,
    limit_hours: object,
) -> dict[str, object]:
    """Verify the reviewed Phase 4E1 decision report bound by a frozen budget."""

    try:
        expected_report_sha256 = validate_sha256(
            baseline_report_sha256,
            "Phase 4E baseline report sha256",
        )
        expected_global_hours = _non_negative_hours(
            baseline_global_hours,
            "baseline_global_hours",
        )
        expected_limit_hours = _positive_hours(limit_hours, "limit_hours")
    except ValueError as error:
        raise Phase4StageBudgetError(
            "the Phase 4E execution budget is invalid"
        ) from error

    path = _phase4e1_report_path(control_root, baseline_run_id)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report is unreadable"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report must be an object"
        )
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError) as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report contains non-canonical values"
        ) from error
    if canonical != raw:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report must be canonical JSON"
        )

    stored_hash = payload.get("reportSha256")
    try:
        stored_hash = validate_sha256(
            stored_hash,
            "Phase 4E baseline report self hash",
        )
    except ValueError as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report has an invalid self hash"
        ) from error
    core = dict(payload)
    core.pop("reportSha256", None)
    if sha256_bytes(canonical_json_bytes(core)) != stored_hash:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report self hash does not verify"
        )
    if stored_hash != expected_report_sha256:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report differs from the execution contract"
        )

    if (
        type(payload.get("schemaVersion")) is not int
        or payload.get("schemaVersion") != PHASE4E1_BASELINE_SCHEMA_VERSION
        or payload.get("kind") != PHASE4E1_BASELINE_KIND
        or payload.get("status") != "completed"
        or payload.get("decisionEligible") is not True
        or payload.get("formalPromotion") is not False
        or payload.get("trainingPerformed") is not False
        or type(payload.get("checkpointsWritten")) is not int
        or payload.get("checkpointsWritten") != 0
        or payload.get("trainingAuthorized") is not False
        or payload.get("runId") != baseline_run_id
    ):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report is not an eligible calibration decision"
        )

    resource_usage = payload.get("resourceUsage")
    recommended_budget = payload.get("recommendedStageBudget")
    if not isinstance(resource_usage, Mapping) or not isinstance(
        recommended_budget, Mapping
    ):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report budget evidence is missing"
        )
    try:
        reported_global_hours = _non_negative_hours(
            resource_usage.get("globalGpuHours"),
            "resourceUsage.globalGpuHours",
        )
        recommended_baseline = _non_negative_hours(
            recommended_budget.get("baselineGlobalHours"),
            "recommendedStageBudget.baselineGlobalHours",
        )
        recommended_limit = _positive_hours(
            recommended_budget.get("limitHours"),
            "recommendedStageBudget.limitHours",
        )
    except ValueError as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report contains invalid budget values"
        ) from error
    if (
        reported_global_hours != expected_global_hours
        or recommended_baseline != expected_global_hours
        or recommended_limit != expected_limit_hours
    ):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report budget differs from the execution contract"
        )

    approved = payload.get("approvedWorkloads")
    if not isinstance(approved, list) or not approved:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report has no approved workloads"
        )
    try:
        approved_workloads = [
            validate_sha256(value, "approved Phase 4E workload sha256")
            for value in approved
        ]
        source_sha256 = validate_sha256(
            payload.get("sourceSha256"),
            "Phase 4E source sha256",
        )
        native_backend_sha256 = validate_sha256(
            payload.get("nativeBackendSha256"),
            "Phase 4E native backend sha256",
        )
    except ValueError as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report has invalid provenance hashes"
        ) from error
    if len(set(approved_workloads)) != len(approved_workloads):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report repeats an approved workload"
        )

    # The frozen execution contract is writable local input, so its report hash
    # cannot serve as the sole trust root. Re-run the complete E1 verifier,
    # including the immutable run manifest, replay, arena, and fixed contract.
    try:
        from .phase4e1_calibration import verify_phase4e1_decision_report

        control = Path(control_root).resolve()
        workspace = control.parents[2]
        verified = verify_phase4e1_decision_report(
            workspace_root=workspace,
            report_path=path,
            verify_ledger=True,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report failed complete E1 verification"
        ) from error
    if (
        verified.get("reportSha256") != stored_hash
        or verified.get("runId") != baseline_run_id
    ):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report changed during complete verification"
        )
    bootstrap_baseline = verified.get("bootstrapBaseline")
    if not isinstance(bootstrap_baseline, Mapping):
        raise Phase4StageBudgetError(
            "the Phase 4E baseline report lost its Bootstrap evidence"
        )

    return {
        "path": str(path),
        "runId": baseline_run_id,
        "reportSha256": stored_hash,
        "reportFileSha256": sha256_bytes(raw),
        "globalGpuHours": reported_global_hours,
        "limitHours": recommended_limit,
        "approvedWorkloads": approved_workloads,
        "sourceSha256": source_sha256,
        "nativeBackendSha256": native_backend_sha256,
        "bootstrapBaseline": dict(bootstrap_baseline),
        "status": "verified",
    }


__all__ = [
    "PHASE4_STAGE_LEDGER_SCHEMA_VERSION",
    "PHASE4_CONTROL_RELATIVE",
    "PHASE4_STAGE_LEDGER_NAME",
    "PHASE4D_STAGE",
    "PHASE4E_STAGE",
    "PHASE4F5_STAGE",
    "PHASE4F_RECOVERY_STAGE",
    "PHASE4D_BASELINE_GLOBAL_GPU_HOURS",
    "PHASE4D_GPU_HOUR_LIMIT",
    "PHASE4D_BASELINE_RUN_ID",
    "PHASE4D_BASELINE_REPORT_SHA256",
    "PHASE4D_BASELINE_REPORT_RELATIVE",
    "PHASE4E1_BASELINE_SCHEMA_VERSION",
    "PHASE4E1_BASELINE_KIND",
    "PHASE4E1_BASELINE_RELATIVE_ROOT",
    "Phase4StageBudgetError",
    "Phase4StageBudgetLedger",
    "verify_phase4d_baseline_report",
    "verify_phase4e_baseline_report",
]
