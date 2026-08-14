"""Versioned execution contracts for durable Phase 4 training stages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .phase4_budget import (
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_BASELINE_REPORT_SHA256,
    PHASE4D_BASELINE_RUN_ID,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
)
from .phase4e1_contract import (
    FREEZE_MAXIMUM_GENERATIONS,
    FREEZE_MINIMUM_STAGE_GPU_HOURS,
    FREEZE_RESERVE_GPU_HOURS,
)
from .run_manifest import canonical_json_bytes, sha256_bytes


PHASE4_EXECUTION_SCHEMA_VERSION = 1
PHASE4_EXECUTION_STATUS_PREPARED = "prepared"
PHASE4_EXECUTION_STATUS_FROZEN = "frozen"
PHASE4_BUDGET_STATUS_PENDING = "pending"
PHASE4_BUDGET_STATUS_FROZEN = "frozen"
PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT = 72.0
PHASE4_GLOBAL_GPU_HOUR_LIMIT = 120.0


class Phase4ExecutionContractError(ValueError):
    """Raised when a Phase 4 execution contract is malformed or incomplete."""


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4ExecutionContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _validate_stage(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise Phase4ExecutionContractError(
            "stage must be a non-empty string of at most 64 characters"
        )
    return value


def _non_negative_hours(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise Phase4ExecutionContractError(
            f"{label} must be finite and non-negative"
        )
    return float(value)


def _positive_hours(value: object, label: str) -> float:
    hours = _non_negative_hours(value, label)
    if hours <= 0:
        raise Phase4ExecutionContractError(f"{label} must be positive")
    return hours


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise Phase4ExecutionContractError(f"{label} fields are invalid")


@dataclass(frozen=True, slots=True)
class Phase4ExecutionBudget:
    """A pending or immutable stage-local GPU budget."""

    status: str
    baseline_global_hours: float | None = None
    limit_hours: float | None = None
    baseline_run_id: str | None = None
    baseline_report_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            PHASE4_BUDGET_STATUS_PENDING,
            PHASE4_BUDGET_STATUS_FROZEN,
        }:
            raise Phase4ExecutionContractError(
                "stageBudget.status must be pending or frozen"
            )
        if (self.baseline_run_id is None) != (
            self.baseline_report_sha256 is None
        ):
            raise Phase4ExecutionContractError(
                "stage budget baseline provenance must be complete"
            )
        if self.baseline_run_id is not None:
            if not isinstance(self.baseline_run_id, str) or not self.baseline_run_id:
                raise Phase4ExecutionContractError(
                    "stageBudget.baselineRunId must be a non-empty string"
                )
            _validate_sha256(
                self.baseline_report_sha256,
                "stageBudget.baselineReportSha256",
            )

        values = (
            self.baseline_global_hours,
            self.limit_hours,
            self.baseline_run_id,
            self.baseline_report_sha256,
        )
        if self.status == PHASE4_BUDGET_STATUS_PENDING:
            if any(value is not None for value in values):
                raise Phase4ExecutionContractError(
                    "pending stage budget must not freeze GPU values or provenance"
                )
            return

        if self.baseline_global_hours is None or self.limit_hours is None:
            raise Phase4ExecutionContractError(
                "frozen stage budget requires baselineGlobalHours and limitHours"
            )
        if self.baseline_run_id is None or self.baseline_report_sha256 is None:
            raise Phase4ExecutionContractError(
                "frozen stage budget requires baseline provenance"
            )
        object.__setattr__(
            self,
            "baseline_global_hours",
            _non_negative_hours(
                self.baseline_global_hours,
                "stageBudget.baselineGlobalHours",
            ),
        )
        object.__setattr__(
            self,
            "limit_hours",
            _positive_hours(self.limit_hours, "stageBudget.limitHours"),
        )
        if self.baseline_global_hours + self.limit_hours > (
            PHASE4_GLOBAL_GPU_HOUR_LIMIT
        ):
            raise Phase4ExecutionContractError(
                "frozen stage budget exceeds the global GPU-hour limit"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Phase4ExecutionBudget":
        if not isinstance(data, Mapping):
            raise Phase4ExecutionContractError("stageBudget must be an object")
        _require_exact_fields(
            data,
            {
                "status",
                "baselineGlobalHours",
                "limitHours",
                "baselineRunId",
                "baselineReportSha256",
            },
            "stageBudget",
        )
        return cls(
            status=data.get("status"),
            baseline_global_hours=data.get("baselineGlobalHours"),
            limit_hours=data.get("limitHours"),
            baseline_run_id=data.get("baselineRunId"),
            baseline_report_sha256=data.get("baselineReportSha256"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "baselineGlobalHours": self.baseline_global_hours,
            "limitHours": self.limit_hours,
            "baselineRunId": self.baseline_run_id,
            "baselineReportSha256": self.baseline_report_sha256,
        }


@dataclass(frozen=True, slots=True)
class Phase4ExecutionContract:
    """Immutable semantics for one Phase 4 stage.

    ``maximum_generations`` is a committed-generation count. A ceiling of N
    permits generation indexes 0 through N-1 and absolute targets in [0, N].
    """

    schema_version: int
    stage: str
    status: str
    purpose: str
    formal_promotion: bool
    maximum_generations: int
    stage_budget: Phase4ExecutionBudget

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PHASE4_EXECUTION_SCHEMA_VERSION
        ):
            raise Phase4ExecutionContractError(
                "unsupported Phase 4 execution contract schema"
            )
        object.__setattr__(self, "stage", _validate_stage(self.stage))
        if self.status not in {
            PHASE4_EXECUTION_STATUS_PREPARED,
            PHASE4_EXECUTION_STATUS_FROZEN,
        }:
            raise Phase4ExecutionContractError(
                "status must be prepared or frozen"
            )
        if not isinstance(self.purpose, str) or not self.purpose:
            raise Phase4ExecutionContractError("purpose must be a non-empty string")
        if not isinstance(self.formal_promotion, bool):
            raise Phase4ExecutionContractError("formalPromotion must be boolean")
        if (
            type(self.maximum_generations) is not int
            or self.maximum_generations <= 0
        ):
            raise Phase4ExecutionContractError(
                "maximumGenerations must be a positive integer"
            )
        if not isinstance(self.stage_budget, Phase4ExecutionBudget):
            raise Phase4ExecutionContractError(
                "stageBudget must be a Phase4ExecutionBudget"
            )
        expected_budget_status = (
            PHASE4_BUDGET_STATUS_PENDING
            if self.status == PHASE4_EXECUTION_STATUS_PREPARED
            else PHASE4_BUDGET_STATUS_FROZEN
        )
        if self.stage_budget.status != expected_budget_status:
            raise Phase4ExecutionContractError(
                "execution status and stage budget status do not agree"
            )
        if self.stage == "phase4e" and self.status == PHASE4_EXECUTION_STATUS_FROZEN:
            budget = self.stage_budget
            expected_limit = math.floor(
                (
                    PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT
                    - float(budget.baseline_global_hours)
                    - FREEZE_RESERVE_GPU_HOURS
                )
                * 1_000_000.0
            ) / 1_000_000.0
            if (
                self.purpose != "phase4e-bootstrap-and-formal-promotion"
                or self.formal_promotion is not True
                or self.maximum_generations != FREEZE_MAXIMUM_GENERATIONS
                or float(budget.limit_hours) != expected_limit
                or expected_limit < FREEZE_MINIMUM_STAGE_GPU_HOURS
            ):
                raise Phase4ExecutionContractError(
                    "frozen Phase 4E contract differs from the reviewed E1 policy"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Phase4ExecutionContract":
        if not isinstance(data, Mapping):
            raise Phase4ExecutionContractError(
                "Phase 4 execution contract must be an object"
            )
        _require_exact_fields(
            data,
            {
                "schemaVersion",
                "stage",
                "status",
                "purpose",
                "formalPromotion",
                "maximumGenerations",
                "stageBudget",
            },
            "Phase 4 execution contract",
        )
        stage_budget = data.get("stageBudget")
        if not isinstance(stage_budget, Mapping):
            raise Phase4ExecutionContractError("stageBudget must be an object")
        return cls(
            schema_version=data.get("schemaVersion"),
            stage=data.get("stage"),
            status=data.get("status"),
            purpose=data.get("purpose"),
            formal_promotion=data.get("formalPromotion"),
            maximum_generations=data.get("maximumGenerations"),
            stage_budget=Phase4ExecutionBudget.from_dict(stage_budget),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "purpose": self.purpose,
            "formalPromotion": self.formal_promotion,
            "maximumGenerations": self.maximum_generations,
            "stageBudget": self.stage_budget.to_payload(),
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_payload()))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256

    @property
    def budget_frozen(self) -> bool:
        return self.stage_budget.status == PHASE4_BUDGET_STATUS_FROZEN

    def validate_target_generations(self, target: object) -> int:
        if (
            type(target) is not int
            or not 0 <= target <= self.maximum_generations
        ):
            raise Phase4ExecutionContractError(
                "target generations must be an integer in "
                f"[0, {self.maximum_generations}]"
            )
        return target

    def remaining_generations(self, next_generation: object) -> int:
        next_value = self.validate_target_generations(next_generation)
        return self.maximum_generations - next_value

    def require_frozen_budget(self) -> Phase4ExecutionBudget:
        if not self.budget_frozen:
            raise Phase4ExecutionContractError(
                "stage GPU budget is pending and cannot authorize training"
            )
        return self.stage_budget


def load_phase4_execution_contract(
    path: str | Path,
) -> Phase4ExecutionContract:
    """Load and strictly validate a Phase 4 execution contract JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4ExecutionContractError(
            f"cannot load Phase 4 execution contract {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase4ExecutionContractError(
            "Phase 4 execution contract must be an object"
        )
    return Phase4ExecutionContract.from_dict(payload)


def phase4d_execution_contract() -> Phase4ExecutionContract:
    """Return the existing frozen five-generation Phase 4D contract."""

    return Phase4ExecutionContract(
        schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
        stage=PHASE4D_STAGE,
        status=PHASE4_EXECUTION_STATUS_FROZEN,
        purpose="warm-cold-pilot-training",
        formal_promotion=False,
        maximum_generations=5,
        stage_budget=Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
            limit_hours=PHASE4D_GPU_HOUR_LIMIT,
            baseline_run_id=PHASE4D_BASELINE_RUN_ID,
            baseline_report_sha256=PHASE4D_BASELINE_REPORT_SHA256,
        ),
    )


PHASE4D_EXECUTION_CONTRACT = phase4d_execution_contract()


__all__ = [
    "PHASE4_EXECUTION_SCHEMA_VERSION",
    "PHASE4_EXECUTION_STATUS_PREPARED",
    "PHASE4_EXECUTION_STATUS_FROZEN",
    "PHASE4_BUDGET_STATUS_PENDING",
    "PHASE4_BUDGET_STATUS_FROZEN",
    "PHASE4_GLOBAL_GPU_HOUR_LIMIT",
    "PHASE4D_EXECUTION_CONTRACT",
    "Phase4ExecutionBudget",
    "Phase4ExecutionContract",
    "Phase4ExecutionContractError",
    "load_phase4_execution_contract",
    "phase4d_execution_contract",
]
