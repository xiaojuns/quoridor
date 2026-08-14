"""Immutable Phase 4E bootstrap baseline extracted from the Phase 4D pilot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_pilot import (
    PHASE4D_ARENA_GAMES,
    PHASE4D_ARENA_OPENINGS,
    PHASE4D_ARENA_SIMULATIONS,
    PHASE4D_CONTRACT_NAME,
    PHASE4D_ONE_SIDED_LOWER_THRESHOLD,
    PHASE4D_PILOT_GENERATIONS,
    PHASE4D_RAW_WIN_RATE_THRESHOLD,
    PHASE4D_RESULT_MANIFEST_NAME,
    Phase4PilotContractError,
    load_lane_state,
    load_pilot_contract,
)
from .run_manifest import (
    ImmutableArtifactExistsError,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_config,
    normalize_json_mapping,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)


BASELINE_SCHEMA_VERSION = 1
BASELINE_MODEL_SCHEMA_VERSION = 1
BASELINE_KIND = "phase4e-bootstrap-baseline"
BASELINE_MODEL_FORMAT = "quoridor-phase4e-bootstrap-model-only"
BASELINE_MODEL_NAME = "model.pt"
BASELINE_MANIFEST_NAME = "manifest.json"
DEFAULT_PILOT_ROOT = Path("training/runs/phase4/phase4d-pilot-v2")
DEFAULT_BASELINE_ROOT = Path(
    "training/runs/phase4/baselines/bootstrap-warm-v2"
)
_FINAL_GENERATION = PHASE4D_PILOT_GENERATIONS - 1

_CHECKPOINT_KEYS = frozenset(
    {
        "schemaVersion",
        "generation",
        "step",
        "identity",
        "model",
        "optimizer",
        "trainerRuntime",
        "rng",
        "generationData",
        "replaySnapshot",
    }
)
_MODEL_KEYS = frozenset(
    {
        "schemaVersion",
        "kind",
        "format",
        "generation",
        "step",
        "formalPromotion",
        "provenance",
        "modelStateSha256",
        "modelState",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schemaVersion",
        "kind",
        "status",
        "createdAt",
        "formalPromotion",
        "championStorePublished",
        "model",
        "lineage",
        "manifestSha256",
    }
)


class Phase4BaselineError(RuntimeError):
    """Base class for bootstrap baseline preparation failures."""


class Phase4BaselineLineageError(Phase4BaselineError):
    """Raised when the source pilot lineage is incomplete or inconsistent."""


class Phase4BaselineCorruptionError(Phase4BaselineError):
    """Raised when an existing baseline archive fails verification."""


@dataclass(frozen=True, slots=True)
class BaselineVerification:
    manifest_path: Path
    model_path: Path
    manifest_sha256: str
    model_sha256: str
    model_state_sha256: str
    generation: int
    step: int
    lineage_verified: bool

    def to_payload(self, workspace: Path) -> dict[str, object]:
        return {
            "schemaVersion": BASELINE_SCHEMA_VERSION,
            "status": "completed",
            "kind": BASELINE_KIND,
            "formalPromotion": False,
            "manifestPath": _relative(self.manifest_path, workspace),
            "manifestSha256": self.manifest_sha256,
            "modelPath": _relative(self.model_path, workspace),
            "modelSha256": self.model_sha256,
            "modelStateSha256": self.model_state_sha256,
            "generation": self.generation,
            "step": self.step,
            "lineageVerified": self.lineage_verified,
        }


@dataclass(frozen=True, slots=True)
class _LaneArtifact:
    lane: str
    generation: int
    step: int
    config_path: Path
    contract_config_sha256: str
    runtime_config_sha256: str
    run_manifest_path: Path
    run_manifest_sha256: str
    generation_report_path: Path
    generation_report_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_metadata_path: Path
    checkpoint_index_path: Path
    model_state_sha256: str
    source_sha256: str
    native_backend_sha256: str
    model_state: OrderedDict[str, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class _PilotLineage:
    payload: Mapping[str, object]
    selected_state: OrderedDict[str, torch.Tensor]
    selected_model_state_sha256: str
    selected_checkpoint_sha256: str
    selected_config_sha256: str
    source_sha256: str
    native_backend_sha256: str
    result_manifest_sha256: str
    contract_sha256: str
    generation: int
    step: int


def _checked_digest(value: object, label: str, error_type: type[Phase4BaselineError]) -> str:
    try:
        return validate_sha256(value, label)
    except ValueError as error:
        raise error_type(str(error)) from error


def _load_json(
    path: Path,
    label: str,
    error_type: type[Phase4BaselineError],
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise error_type(f"{label} is missing or is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise error_type(f"{label} must contain a JSON object")
    try:
        return normalize_json_mapping(payload, label)
    except (TypeError, ValueError) as error:
        raise error_type(f"{label} is not canonical JSON data") from error


def _verify_self_hash(
    payload: Mapping[str, object],
    field: str,
    label: str,
    error_type: type[Phase4BaselineError],
) -> str:
    digest = _checked_digest(payload.get(field), f"{label}.{field}", error_type)
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise error_type(f"{label} self hash mismatch")
    return digest


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4BaselineError(f"path is outside workspace: {path}") from error


def _workspace_directory(
    workspace: Path,
    value: str | Path,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    raw = Path(value)
    path = raw if raw.is_absolute() else workspace / raw
    if path.is_symlink():
        raise Phase4BaselineError(f"{label} must not be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise Phase4BaselineError(f"{label} must be inside the workspace") from error
    if must_exist and (not resolved.is_dir() or resolved.is_symlink()):
        raise Phase4BaselineError(f"{label} does not name a directory")
    return resolved


def _workspace_file(
    workspace: Path,
    value: object,
    label: str,
    error_type: type[Phase4BaselineError],
) -> Path:
    if not isinstance(value, str) or not value:
        raise error_type(f"{label} must be a non-empty path")
    raw = Path(value)
    path = raw if raw.is_absolute() else workspace / raw
    if path.is_symlink():
        raise error_type(f"{label} must not be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise error_type(f"{label} must be inside the workspace") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise error_type(f"{label} does not name a regular file")
    return resolved


def _child_file(
    root: Path,
    value: object,
    label: str,
    error_type: type[Phase4BaselineError],
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
    ):
        raise error_type(f"{label} must be a plain file name")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise error_type(f"{label} is missing or is not a regular file")
    return path.resolve()


def _file_reference(path: Path, workspace: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4BaselineLineageError(
            f"lineage artifact is missing or is not a regular file: {path}"
        )
    return {
        "path": _relative(path, workspace),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _normalized_model_state(
    state: object,
    error_type: type[Phase4BaselineError],
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise error_type("model state must be a non-empty mapping")
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in sorted(state.items(), key=lambda item: str(item[0])):
        if not isinstance(key, str) or not key:
            raise error_type("model state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise error_type("model state values must be tensors")
        tensor = value.detach().to(device="cpu").contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise error_type(f"model state tensor {key!r} contains NaN or Inf")
        normalized[key] = tensor
    metadata = getattr(state, "_metadata", None)
    if isinstance(metadata, Mapping):
        normalized._metadata = deepcopy(metadata)  # type: ignore[attr-defined]
    return normalized


def _resolve_contract_config_path(
    workspace: Path,
    pilot_root: Path,
    contract: Mapping[str, object],
    lane: str,
) -> tuple[Path, Mapping[str, object]]:
    configs = contract.get("configs")
    if not isinstance(configs, Mapping):
        raise Phase4BaselineLineageError("pilot config records are missing")
    record = configs.get(lane)
    if not isinstance(record, Mapping):
        raise Phase4BaselineLineageError(f"{lane} config record is missing")
    path = _workspace_file(
        workspace,
        record.get("path"),
        f"{lane} config",
        Phase4BaselineLineageError,
    )
    expected = (pilot_root / "effective-configs" / f"{lane}.json").resolve()
    if path != expected:
        raise Phase4BaselineLineageError(
            f"{lane} config is outside the selected pilot root"
        )
    if sha256_file(path) != record.get("fileSha256"):
        raise Phase4BaselineLineageError(f"{lane} config file hash mismatch")
    _checked_digest(
        record.get("configSha256"),
        f"{lane} contract config sha256",
        Phase4BaselineLineageError,
    )
    return path, record


def _load_checkpoint(
    path: Path,
    lane: str,
) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise Phase4BaselineLineageError(
            f"{lane} final checkpoint cannot be deserialized"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
        raise Phase4BaselineLineageError(
            f"{lane} final checkpoint schema fields are invalid"
        )
    return payload


def _validate_lane_final(
    *,
    workspace: Path,
    pilot_root: Path,
    contract: Mapping[str, object],
    lane: str,
    retain_model: bool,
) -> _LaneArtifact:
    try:
        state = load_lane_state(
            workspace_root=workspace,
            contract=contract,
            lane=lane,
        )
    except Exception as error:
        raise Phase4BaselineLineageError(
            f"{lane} lane manifest/report lineage does not verify: {error}"
        ) from error
    expected_run_dir = (pilot_root / "lanes" / lane).resolve()
    if state.config.run_dir.resolve() != expected_run_dir:
        raise Phase4BaselineLineageError(
            f"{lane} run directory is outside the selected pilot root"
        )
    if (
        not state.complete
        or state.next_generation != PHASE4D_PILOT_GENERATIONS
        or len(state.reports) != PHASE4D_PILOT_GENERATIONS
        or tuple(record.generation for record in state.reports)
        != tuple(range(PHASE4D_PILOT_GENERATIONS))
    ):
        raise Phase4BaselineLineageError(
            f"{lane} lane is incomplete or does not end at generation 4"
        )

    runtime = contract.get("runtimeProvenance")
    if not isinstance(runtime, Mapping):
        raise Phase4BaselineLineageError("pilot runtime provenance is missing")
    source_sha256 = _checked_digest(
        runtime.get("sourceSha256"),
        "pilot source sha256",
        Phase4BaselineLineageError,
    )
    native_sha256 = _checked_digest(
        runtime.get("nativeBackendSha256"),
        "pilot native backend sha256",
        Phase4BaselineLineageError,
    )
    if (
        state.source_sha256 != source_sha256
        or state.native_backend_sha256 != native_sha256
    ):
        raise Phase4BaselineLineageError(
            f"{lane} source/native lineage differs from the pilot contract"
        )

    config_path, config_record = _resolve_contract_config_path(
        workspace, pilot_root, contract, lane
    )
    runtime_config_sha256 = fingerprint_config(state.config).sha256
    run_manifest_path = expected_run_dir / "run-manifest.json"
    run_manifest = _load_json(
        run_manifest_path,
        f"{lane} run manifest",
        Phase4BaselineLineageError,
    )
    run_manifest_sha256 = _verify_self_hash(
        run_manifest,
        "manifestSha256",
        f"{lane} run manifest",
        Phase4BaselineLineageError,
    )
    provenance = run_manifest.get("provenance")
    config_provenance = (
        provenance.get("config") if isinstance(provenance, Mapping) else None
    )
    source_provenance = (
        provenance.get("source") if isinstance(provenance, Mapping) else None
    )
    budget = run_manifest.get("resourceBudget")
    budget_metadata = budget.get("metadata") if isinstance(budget, Mapping) else None
    metadata = run_manifest.get("metadata")
    if (
        not isinstance(config_provenance, Mapping)
        or config_provenance.get("sha256") != runtime_config_sha256
        or not isinstance(source_provenance, Mapping)
        or source_provenance.get("sha256") != source_sha256
        or not isinstance(budget_metadata, Mapping)
        or budget_metadata.get("nativeBackendSha256") != native_sha256
        or not isinstance(metadata, Mapping)
        or metadata.get("formalPromotion") is not False
    ):
        raise Phase4BaselineLineageError(
            f"{lane} run manifest config/source/native lineage is invalid"
        )
    if state.manifest_sha256 != run_manifest_sha256:
        raise Phase4BaselineLineageError(
            f"{lane} lane state uses a different run manifest"
        )

    record = state.reports[-1]
    generation_report = record.payload
    generation_report_sha256 = _verify_self_hash(
        generation_report,
        "reportSha256",
        f"{lane} generation report",
        Phase4BaselineLineageError,
    )
    body = generation_report.get("report")
    checkpoint_record = (
        body.get("checkpoint") if isinstance(body, Mapping) else None
    )
    generation_data = (
        body.get("generationData") if isinstance(body, Mapping) else None
    )
    if (
        generation_report.get("schemaVersion") != 1
        or generation_report.get("generation") != _FINAL_GENERATION
        or generation_report.get("runManifestSha256") != run_manifest_sha256
        or not isinstance(checkpoint_record, Mapping)
        or not isinstance(generation_data, Mapping)
        or generation_data.get("schemaVersion") != 1
        or generation_data.get("generation") != _FINAL_GENERATION
    ):
        raise Phase4BaselineLineageError(
            f"{lane} final generation report is invalid"
        )
    promotion = generation_data.get("promotion")
    training = generation_data.get("training")
    if (
        not isinstance(promotion, Mapping)
        or promotion.get("status") != "not-run"
        or not isinstance(training, Mapping)
        or training.get("trainerStep") != 320
    ):
        raise Phase4BaselineLineageError(
            f"{lane} final generation is not a non-promotion pilot step"
        )

    checkpoint_name = checkpoint_record.get("path")
    checkpoint_path = _child_file(
        expected_run_dir / "checkpoints",
        checkpoint_name,
        f"{lane} checkpoint",
        Phase4BaselineLineageError,
    )
    checkpoint_sha256 = _checked_digest(
        checkpoint_record.get("sha256"),
        f"{lane} checkpoint sha256",
        Phase4BaselineLineageError,
    )
    step = checkpoint_record.get("step")
    if (
        type(step) is not int
        or step != 320
        or checkpoint_record.get("bytes") != checkpoint_path.stat().st_size
        or sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise Phase4BaselineLineageError(
            f"{lane} checkpoint file does not match its generation report"
        )

    expected_identity = {
        "runManifestSha256": run_manifest_sha256,
        "configSha256": runtime_config_sha256,
        "sourceSha256": source_sha256,
    }
    checkpoint = _load_checkpoint(checkpoint_path, lane)
    if (
        checkpoint.get("schemaVersion") != 1
        or checkpoint.get("generation") != _FINAL_GENERATION
        or checkpoint.get("step") != step
        or checkpoint.get("identity") != expected_identity
        or checkpoint.get("generationData") != generation_data
    ):
        raise Phase4BaselineLineageError(
            f"{lane} checkpoint payload/report binding is invalid"
        )
    model_state = _normalized_model_state(
        checkpoint.get("model"),
        Phase4BaselineLineageError,
    )
    try:
        model_state_sha256 = canonical_model_state_sha256(model_state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4BaselineLineageError(
            f"{lane} checkpoint model state is invalid"
        ) from error

    checkpoint_metadata_path = checkpoint_path.with_suffix(".json")
    checkpoint_metadata = _load_json(
        checkpoint_metadata_path,
        f"{lane} checkpoint metadata",
        Phase4BaselineLineageError,
    )
    if checkpoint_metadata != {
        "schemaVersion": 1,
        "generation": _FINAL_GENERATION,
        "step": step,
        "file": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint_sha256,
        "identity": expected_identity,
    }:
        raise Phase4BaselineLineageError(
            f"{lane} checkpoint metadata does not verify"
        )
    checkpoint_index_path = checkpoint_path.parent / "latest.json"
    checkpoint_index = _load_json(
        checkpoint_index_path,
        f"{lane} checkpoint index",
        Phase4BaselineLineageError,
    )
    if checkpoint_index != {
        "schemaVersion": 1,
        "generation": _FINAL_GENERATION,
        "step": step,
        "metadata": checkpoint_metadata_path.name,
        "sha256": checkpoint_sha256,
    }:
        raise Phase4BaselineLineageError(
            f"{lane} latest checkpoint index does not select generation 4"
        )

    return _LaneArtifact(
        lane=lane,
        generation=_FINAL_GENERATION,
        step=step,
        config_path=config_path,
        contract_config_sha256=str(config_record["configSha256"]),
        runtime_config_sha256=runtime_config_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        generation_report_path=record.path,
        generation_report_sha256=generation_report_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_metadata_path=checkpoint_metadata_path,
        checkpoint_index_path=checkpoint_index_path,
        model_state_sha256=model_state_sha256,
        source_sha256=source_sha256,
        native_backend_sha256=native_sha256,
        model_state=model_state if retain_model else None,
    )


def _lane_reference(
    lane: _LaneArtifact,
    workspace: Path,
) -> dict[str, object]:
    return {
        "lane": lane.lane,
        "generation": lane.generation,
        "step": lane.step,
        "sourceSha256": lane.source_sha256,
        "nativeBackendSha256": lane.native_backend_sha256,
        "modelStateSha256": lane.model_state_sha256,
        "config": {
            **_file_reference(lane.config_path, workspace),
            "contractConfigSha256": lane.contract_config_sha256,
            "runtimeConfigSha256": lane.runtime_config_sha256,
        },
        "runManifest": {
            **_file_reference(lane.run_manifest_path, workspace),
            "manifestSha256": lane.run_manifest_sha256,
        },
        "generationReport": {
            **_file_reference(lane.generation_report_path, workspace),
            "reportSha256": lane.generation_report_sha256,
        },
        "checkpoint": _file_reference(lane.checkpoint_path, workspace),
        "checkpointMetadata": _file_reference(
            lane.checkpoint_metadata_path, workspace
        ),
        "checkpointIndex": _file_reference(lane.checkpoint_index_path, workspace),
    }


def _validate_decision(
    decision: object,
    summary: Mapping[str, object],
) -> None:
    if not isinstance(decision, Mapping):
        raise Phase4BaselineLineageError("pilot decision is missing")
    checks = decision.get("checks")
    raw_check = checks.get("rawWinRate") if isinstance(checks, Mapping) else None
    lower_check = (
        checks.get("oneSided95Lower") if isinstance(checks, Mapping) else None
    )
    wins = summary.get("candidateWins")
    losses = summary.get("candidateLosses")
    games = summary.get("games")
    interval = summary.get("winRateCI")
    if (
        decision.get("candidateLane") != "warm"
        or decision.get("opponentLane") != "cold"
        or decision.get("selectedLane") != "warm"
        or decision.get("warmSelected") is not True
        or decision.get("screenOnly") is not True
        or decision.get("formalChampionDeclaration") is not False
        or not isinstance(raw_check, Mapping)
        or raw_check.get("passed") is not True
        or not isinstance(lower_check, Mapping)
        or lower_check.get("passed") is not True
        or type(wins) is not int
        or type(losses) is not int
        or games != PHASE4D_ARENA_GAMES
        or wins + losses != games
        or not isinstance(interval, Mapping)
    ):
        raise Phase4BaselineLineageError(
            "pilot did not select warm through the completed screen-only gate"
        )
    raw = wins / games
    lower = interval.get("lower")
    if (
        not isinstance(lower, (int, float))
        or isinstance(lower, bool)
        or summary.get("winRate") != raw
        or decision.get("rawWinRate") != raw
        or decision.get("oneSided95Lower") != lower
        or raw < PHASE4D_RAW_WIN_RATE_THRESHOLD
        or float(lower) <= PHASE4D_ONE_SIDED_LOWER_THRESHOLD
    ):
        raise Phase4BaselineLineageError(
            "pilot warm decision statistics do not satisfy the frozen screen"
        )


def _validate_arena_and_result(
    *,
    workspace: Path,
    pilot_root: Path,
    contract: Mapping[str, object],
    contract_path: Path,
    lanes: Mapping[str, _LaneArtifact],
) -> tuple[dict[str, object], str]:
    arena_root = pilot_root / "arena"
    result_path = arena_root / PHASE4D_RESULT_MANIFEST_NAME
    result = _load_json(
        result_path,
        "pilot result manifest",
        Phase4BaselineLineageError,
    )
    result_manifest_sha256 = _verify_self_hash(
        result,
        "resultManifestSha256",
        "pilot result manifest",
        Phase4BaselineLineageError,
    )
    contract_sha256 = str(contract["contractSha256"])
    if (
        result.get("schemaVersion") != 1
        or result.get("status") != "completed"
        or result.get("contractSha256") != contract_sha256
    ):
        raise Phase4BaselineLineageError(
            "pilot result manifest is incomplete or belongs to another contract"
        )

    report_record = result.get("report")
    games_record = result.get("gamesJsonl")
    if not isinstance(report_record, Mapping) or not isinstance(
        games_record, Mapping
    ):
        raise Phase4BaselineLineageError("pilot result artifact records are invalid")
    report_path = _child_file(
        arena_root,
        report_record.get("file"),
        "pilot arena report",
        Phase4BaselineLineageError,
    )
    games_path = _child_file(
        arena_root,
        games_record.get("file"),
        "pilot games JSONL",
        Phase4BaselineLineageError,
    )
    for path, record, label in (
        (report_path, report_record, "pilot arena report"),
        (games_path, games_record, "pilot games JSONL"),
    ):
        digest = _checked_digest(
            record.get("sha256"),
            f"{label} sha256",
            Phase4BaselineLineageError,
        )
        if (
            record.get("bytes") != path.stat().st_size
            or sha256_file(path) != digest
        ):
            raise Phase4BaselineLineageError(f"{label} does not verify")
    if games_record.get("completedGames") != PHASE4D_ARENA_GAMES:
        raise Phase4BaselineLineageError("pilot games JSONL is incomplete")

    report = _load_json(
        report_path,
        "pilot arena report",
        Phase4BaselineLineageError,
    )
    decision = report.get("decision")
    if (
        report.get("schemaVersion") != 1
        or report.get("recordType") != "phase4d-warm-cold-screen"
        or report.get("status") != "completed"
        or report.get("contractSha256") != contract_sha256
        or report.get("contractPath") != _relative(contract_path, workspace)
        or result.get("decision") != decision
    ):
        raise Phase4BaselineLineageError(
            "pilot arena report/result manifest binding is invalid"
        )

    runtime = contract.get("runtimeProvenance")
    if not isinstance(runtime, Mapping):
        raise Phase4BaselineLineageError("pilot runtime provenance is missing")
    source_sha256 = runtime["sourceSha256"]
    native_sha256 = runtime["nativeBackendSha256"]
    if report.get("nativeBackendSha256") != native_sha256:
        raise Phase4BaselineLineageError(
            "pilot arena native backend lineage changed"
        )

    expected_schedule = []
    for generation in range(PHASE4D_PILOT_GENERATIONS):
        for lane_name in ("cold", "warm"):
            lane_state = load_lane_state(
                workspace_root=workspace,
                contract=contract,
                lane=lane_name,
            )
            created_at = lane_state.reports[generation].payload.get("createdAt")
            if not isinstance(created_at, str) or not created_at:
                raise Phase4BaselineLineageError(
                    f"{lane_name} generation timestamp is invalid"
                )
            expected_schedule.append(
                {
                    "lane": lane_name,
                    "generation": generation,
                    "createdAt": created_at,
                }
            )
    if report.get("schedule") != expected_schedule:
        raise Phase4BaselineLineageError(
            "pilot report does not preserve the frozen alternating schedule"
        )

    report_lanes = report.get("lanes")
    artifacts = report.get("artifacts")
    if not isinstance(report_lanes, Mapping) or not isinstance(artifacts, Mapping):
        raise Phase4BaselineLineageError("pilot report lane artifacts are missing")
    for lane_name, lane in lanes.items():
        lane_report = report_lanes.get(lane_name)
        artifact = artifacts.get(lane_name)
        latest = (
            lane_report.get("latestCheckpoint")
            if isinstance(lane_report, Mapping)
            else None
        )
        if (
            not isinstance(lane_report, Mapping)
            or lane_report.get("lane") != lane_name
            or lane_report.get("status") != "completed"
            or lane_report.get("nextGeneration") != PHASE4D_PILOT_GENERATIONS
            or lane_report.get("manifestSha256") != lane.run_manifest_sha256
            or lane_report.get("sourceSha256") != source_sha256
            or not isinstance(latest, Mapping)
            or latest.get("generation") != _FINAL_GENERATION
            or latest.get("step") != lane.step
            or latest.get("sha256") != lane.checkpoint_sha256
            or latest.get("modelStateSha256") != lane.model_state_sha256
            or _workspace_file(
                workspace,
                latest.get("path"),
                f"{lane_name} reported checkpoint",
                Phase4BaselineLineageError,
            )
            != lane.checkpoint_path
            or not isinstance(artifact, Mapping)
            or artifact.get("checkpointSha256") != lane.checkpoint_sha256
            or artifact.get("modelStateSha256") != lane.model_state_sha256
            or _workspace_file(
                workspace,
                artifact.get("checkpointPath"),
                f"{lane_name} arena artifact",
                Phase4BaselineLineageError,
            )
            != lane.checkpoint_path
        ):
            raise Phase4BaselineLineageError(
                f"{lane_name} final checkpoint is not bound to the pilot report"
            )

    arena = report.get("arena")
    if not isinstance(arena, Mapping):
        raise Phase4BaselineLineageError("paired arena result is missing")
    arena_contract = contract.get("arena")
    if not isinstance(arena_contract, Mapping):
        raise Phase4BaselineLineageError("pilot arena contract is missing")
    candidate_identity = arena.get("candidateIdentity")
    opponent_identity = arena.get("opponentIdentity")
    candidate_runtime = arena.get("candidateRuntime")
    opponent_runtime = arena.get("opponentRuntime")
    warm = lanes["warm"]
    cold = lanes["cold"]
    expected_candidate = {
        "selectorId": "phase4d-warm-g4",
        "engine": "native-mcts",
        "artifactHash": warm.checkpoint_sha256,
        "configHash": warm.runtime_config_sha256,
        "sourceHash": source_sha256,
        "verified": True,
    }
    expected_opponent = {
        "selectorId": "phase4d-cold-g4",
        "engine": "native-mcts",
        "artifactHash": cold.checkpoint_sha256,
        "configHash": cold.runtime_config_sha256,
        "sourceHash": source_sha256,
        "verified": True,
    }
    if (
        arena.get("schemaVersion") != 1
        or arena.get("recordType") != "paired-arena-result"
        or arena.get("suite") != "dev"
        or arena.get("suiteHash") != arena_contract.get("screenSuiteHash")
        or candidate_identity != expected_candidate
        or opponent_identity != expected_opponent
    ):
        raise Phase4BaselineLineageError(
            "paired arena candidate/opponent identity is invalid"
        )
    for role, runtime_record in (
        ("candidate", candidate_runtime),
        ("opponent", opponent_runtime),
    ):
        if (
            not isinstance(runtime_record, Mapping)
            or runtime_record.get("kind") != "native-mcts"
            or runtime_record.get("trusted") is not True
            or runtime_record.get("identityBound") is not True
            or runtime_record.get("configuredSimulations")
            != PHASE4D_ARENA_SIMULATIONS
            or runtime_record.get("backendHash") != native_sha256
        ):
            raise Phase4BaselineLineageError(
                f"paired arena {role} runtime is not provenance-bound"
            )

    games = arena.get("games")
    summary = arena.get("summary")
    if (
        not isinstance(games, list)
        or len(games) != PHASE4D_ARENA_GAMES
        or not isinstance(summary, Mapping)
    ):
        raise Phase4BaselineLineageError("paired arena workload is incomplete")
    try:
        jsonl_games = [
            json.loads(line)
            for line in games_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4BaselineLineageError(
            "pilot games JSONL cannot be decoded"
        ) from error
    if jsonl_games != games:
        raise Phase4BaselineLineageError(
            "pilot games JSONL differs from the arena report"
        )
    game_ids: set[str] = set()
    opening_legs: dict[str, set[int]] = {}
    candidate_wins = 0
    for game in games:
        if not isinstance(game, Mapping):
            raise Phase4BaselineLineageError("pilot game record is invalid")
        game_id = game.get("gameId")
        opening_id = game.get("openingId")
        leg = game.get("leg")
        candidate_won = game.get("candidateWon")
        if (
            not isinstance(game_id, str)
            or not game_id
            or game_id in game_ids
            or not isinstance(opening_id, str)
            or type(leg) is not int
            or leg not in (0, 1)
            or type(candidate_won) is not bool
            or game.get("candidateIdentity") != expected_candidate
            or game.get("opponentIdentity") != expected_opponent
        ):
            raise Phase4BaselineLineageError(
                "pilot game identity or outcome record is invalid"
            )
        game_ids.add(game_id)
        opening_legs.setdefault(opening_id, set()).add(leg)
        candidate_wins += int(candidate_won)
    opening_ids = arena_contract.get("openingIds")
    if (
        not isinstance(opening_ids, list)
        or set(opening_legs) != set(opening_ids)
        or len(opening_legs) != PHASE4D_ARENA_OPENINGS
        or any(legs != {0, 1} for legs in opening_legs.values())
        or summary.get("openingPairs") != PHASE4D_ARENA_OPENINGS
        or summary.get("games") != PHASE4D_ARENA_GAMES
        or summary.get("candidateWins") != candidate_wins
        or summary.get("candidateLosses")
        != PHASE4D_ARENA_GAMES - candidate_wins
        or summary.get("draws") != 0
    ):
        raise Phase4BaselineLineageError(
            "paired arena opening pairs or summary outcomes are invalid"
        )
    _validate_decision(decision, summary)

    result_ref = {
        **_file_reference(result_path, workspace),
        "resultManifestSha256": result_manifest_sha256,
    }
    report_ref = _file_reference(report_path, workspace)
    games_ref = {
        **_file_reference(games_path, workspace),
        "completedGames": PHASE4D_ARENA_GAMES,
    }
    return {
        "resultManifest": result_ref,
        "arenaReport": report_ref,
        "gamesJsonl": games_ref,
        "selectedLane": "warm",
        "screenOnly": True,
        "formalChampionDeclaration": False,
        "rawWinRate": decision["rawWinRate"],
        "oneSided95Lower": decision["oneSided95Lower"],
    }, result_manifest_sha256


def _collect_pilot_lineage(
    *,
    workspace: Path,
    pilot_root: Path,
) -> _PilotLineage:
    contract_path = pilot_root / PHASE4D_CONTRACT_NAME
    try:
        contract = load_pilot_contract(
            contract_path,
            workspace_root=workspace,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4BaselineLineageError(
            f"pilot contract does not verify: {error}"
        ) from error
    contract_sha256 = _checked_digest(
        contract.get("contractSha256"),
        "pilot contract sha256",
        Phase4BaselineLineageError,
    )
    if contract.get("formalChampionDeclaration") is not False:
        raise Phase4BaselineLineageError(
            "pilot contract must not claim formal promotion"
        )

    cold = _validate_lane_final(
        workspace=workspace,
        pilot_root=pilot_root,
        contract=contract,
        lane="cold",
        retain_model=False,
    )
    warm = _validate_lane_final(
        workspace=workspace,
        pilot_root=pilot_root,
        contract=contract,
        lane="warm",
        retain_model=True,
    )
    if warm.model_state is None:
        raise Phase4BaselineLineageError(
            "warm final checkpoint model state was not retained"
        )
    if (
        cold.source_sha256 != warm.source_sha256
        or cold.native_backend_sha256 != warm.native_backend_sha256
    ):
        raise Phase4BaselineLineageError(
            "pilot lanes do not share source/native provenance"
        )
    arena_lineage, result_manifest_sha256 = _validate_arena_and_result(
        workspace=workspace,
        pilot_root=pilot_root,
        contract=contract,
        contract_path=contract_path,
        lanes={"cold": cold, "warm": warm},
    )
    payload = {
        "pilot": {
            "root": _relative(pilot_root, workspace),
            "contract": {
                **_file_reference(contract_path, workspace),
                "contractSha256": contract_sha256,
            },
            **arena_lineage,
        },
        "lanes": {
            "cold": _lane_reference(cold, workspace),
            "warm": _lane_reference(warm, workspace),
        },
        "selection": {
            "lane": "warm",
            "generation": warm.generation,
            "step": warm.step,
            "checkpointSha256": warm.checkpoint_sha256,
            "modelStateSha256": warm.model_state_sha256,
            "configSha256": warm.runtime_config_sha256,
            "sourceSha256": warm.source_sha256,
            "nativeBackendSha256": warm.native_backend_sha256,
            "formalPromotion": False,
        },
    }
    return _PilotLineage(
        payload=normalize_json_mapping(payload, "baseline lineage"),
        selected_state=warm.model_state,
        selected_model_state_sha256=warm.model_state_sha256,
        selected_checkpoint_sha256=warm.checkpoint_sha256,
        selected_config_sha256=warm.runtime_config_sha256,
        source_sha256=warm.source_sha256,
        native_backend_sha256=warm.native_backend_sha256,
        result_manifest_sha256=result_manifest_sha256,
        contract_sha256=contract_sha256,
        generation=warm.generation,
        step=warm.step,
    )


def _model_payload(lineage: _PilotLineage) -> dict[str, object]:
    return {
        "schemaVersion": BASELINE_MODEL_SCHEMA_VERSION,
        "kind": BASELINE_KIND,
        "format": BASELINE_MODEL_FORMAT,
        "generation": lineage.generation,
        "step": lineage.step,
        "formalPromotion": False,
        "provenance": {
            "contractSha256": lineage.contract_sha256,
            "resultManifestSha256": lineage.result_manifest_sha256,
            "checkpointSha256": lineage.selected_checkpoint_sha256,
            "configSha256": lineage.selected_config_sha256,
            "sourceSha256": lineage.source_sha256,
            "nativeBackendSha256": lineage.native_backend_sha256,
        },
        "modelStateSha256": lineage.selected_model_state_sha256,
        "modelState": lineage.selected_state,
    }


def _load_model_payload(
    path: Path,
    error_type: type[Phase4BaselineError],
) -> tuple[dict[str, object], OrderedDict[str, torch.Tensor]]:
    if path.is_symlink() or not path.is_file():
        raise error_type("baseline model artifact is missing or is a symlink")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise error_type("baseline model artifact cannot be loaded safely") from error
    if not isinstance(payload, Mapping) or set(payload) != _MODEL_KEYS:
        raise error_type("baseline model artifact schema fields are invalid")
    if (
        payload.get("schemaVersion") != BASELINE_MODEL_SCHEMA_VERSION
        or payload.get("kind") != BASELINE_KIND
        or payload.get("format") != BASELINE_MODEL_FORMAT
        or payload.get("formalPromotion") is not False
        or type(payload.get("generation")) is not int
        or type(payload.get("step")) is not int
    ):
        raise error_type("baseline model artifact metadata is invalid")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "contractSha256",
        "resultManifestSha256",
        "checkpointSha256",
        "configSha256",
        "sourceSha256",
        "nativeBackendSha256",
    }:
        raise error_type("baseline model provenance is invalid")
    for key, value in provenance.items():
        _checked_digest(value, f"baseline model provenance.{key}", error_type)
    state = _normalized_model_state(payload.get("modelState"), error_type)
    expected_model_sha256 = _checked_digest(
        payload.get("modelStateSha256"),
        "baseline modelStateSha256",
        error_type,
    )
    if canonical_model_state_sha256(state) != expected_model_sha256:
        raise error_type("baseline model tensor-state hash mismatch")
    return dict(payload), state


def _model_matches_lineage(
    payload: Mapping[str, object],
    lineage: _PilotLineage,
) -> bool:
    return (
        payload.get("generation") == lineage.generation
        and payload.get("step") == lineage.step
        and payload.get("modelStateSha256")
        == lineage.selected_model_state_sha256
        and payload.get("provenance")
        == _model_payload(lineage)["provenance"]
    )


def _atomic_create_model(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary: Path | None = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(dict(payload), output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ImmutableArtifactExistsError(
                f"immutable artifact already exists: {path}"
            ) from error
        temporary.unlink()
        temporary = None
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest_payload(
    *,
    workspace: Path,
    model_path: Path,
    lineage: _PilotLineage,
) -> dict[str, object]:
    core = {
        "schemaVersion": BASELINE_SCHEMA_VERSION,
        "kind": BASELINE_KIND,
        "status": "completed",
        "createdAt": utc_timestamp(),
        "formalPromotion": False,
        "championStorePublished": False,
        "model": {
            **_file_reference(model_path, workspace),
            "format": BASELINE_MODEL_FORMAT,
            "generation": lineage.generation,
            "step": lineage.step,
            "modelStateSha256": lineage.selected_model_state_sha256,
        },
        "lineage": dict(lineage.payload),
    }
    return {
        **core,
        "manifestSha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _verify_archive(
    *,
    workspace: Path,
    baseline_root: Path,
    lineage: _PilotLineage | None,
) -> BaselineVerification:
    manifest_path = baseline_root / BASELINE_MANIFEST_NAME
    model_path = baseline_root / BASELINE_MODEL_NAME
    manifest = _load_json(
        manifest_path,
        "baseline manifest",
        Phase4BaselineCorruptionError,
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise Phase4BaselineCorruptionError(
            "baseline manifest schema fields are invalid"
        )
    manifest_sha256 = _verify_self_hash(
        manifest,
        "manifestSha256",
        "baseline manifest",
        Phase4BaselineCorruptionError,
    )
    if (
        manifest.get("schemaVersion") != BASELINE_SCHEMA_VERSION
        or manifest.get("kind") != BASELINE_KIND
        or manifest.get("status") != "completed"
        or manifest.get("formalPromotion") is not False
        or manifest.get("championStorePublished") is not False
        or not isinstance(manifest.get("createdAt"), str)
        or not manifest.get("createdAt")
    ):
        raise Phase4BaselineCorruptionError(
            "baseline manifest status/role is invalid"
        )
    model_record = manifest.get("model")
    stored_lineage = manifest.get("lineage")
    if not isinstance(model_record, Mapping) or not isinstance(
        stored_lineage, Mapping
    ):
        raise Phase4BaselineCorruptionError(
            "baseline manifest model/lineage record is invalid"
        )
    if model_record.get("path") != _relative(model_path, workspace):
        raise Phase4BaselineCorruptionError(
            "baseline manifest points to an unexpected model path"
        )
    model_sha256 = _checked_digest(
        model_record.get("sha256"),
        "baseline model file sha256",
        Phase4BaselineCorruptionError,
    )
    if (
        model_path.is_symlink()
        or not model_path.is_file()
        or model_record.get("bytes") != model_path.stat().st_size
        or sha256_file(model_path) != model_sha256
        or model_record.get("format") != BASELINE_MODEL_FORMAT
    ):
        raise Phase4BaselineCorruptionError(
            "baseline model file does not verify"
        )
    model_payload, _state = _load_model_payload(
        model_path,
        Phase4BaselineCorruptionError,
    )
    model_state_sha256 = str(model_payload["modelStateSha256"])
    if (
        model_record.get("generation") != model_payload.get("generation")
        or model_record.get("step") != model_payload.get("step")
        or model_record.get("modelStateSha256") != model_state_sha256
    ):
        raise Phase4BaselineCorruptionError(
            "baseline manifest/model metadata differs"
        )
    if lineage is not None:
        if (
            stored_lineage != lineage.payload
            or not _model_matches_lineage(model_payload, lineage)
        ):
            raise Phase4BaselineCorruptionError(
                "baseline archive differs from the verified pilot lineage"
            )
    return BaselineVerification(
        manifest_path=manifest_path,
        model_path=model_path,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
        model_state_sha256=model_state_sha256,
        generation=int(model_payload["generation"]),
        step=int(model_payload["step"]),
        lineage_verified=lineage is not None,
    )


def prepare_bootstrap_baseline(
    *,
    workspace_root: str | Path,
    pilot_root: str | Path = DEFAULT_PILOT_ROOT,
    output_root: str | Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, object]:
    """Validate pilot-v2 and atomically create the warm-g4 model-only baseline."""

    workspace = Path(workspace_root).resolve()
    pilot = _workspace_directory(
        workspace, pilot_root, "pilot root", must_exist=True
    )
    output = _workspace_directory(
        workspace, output_root, "baseline output root", must_exist=False
    )
    lineage = _collect_pilot_lineage(workspace=workspace, pilot_root=pilot)
    manifest_path = output / BASELINE_MANIFEST_NAME
    model_path = output / BASELINE_MODEL_NAME

    if manifest_path.exists():
        verification = _verify_archive(
            workspace=workspace,
            baseline_root=output,
            lineage=lineage,
        )
        return {
            **verification.to_payload(workspace),
            "created": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise Phase4BaselineCorruptionError(
            "baseline output root must not be a symlink"
        )

    expected_model = _model_payload(lineage)
    if model_path.exists():
        existing, _state = _load_model_payload(
            model_path,
            Phase4BaselineCorruptionError,
        )
        if not _model_matches_lineage(existing, lineage):
            raise Phase4BaselineCorruptionError(
                "existing orphan baseline model differs from the pilot lineage"
            )
    else:
        try:
            _atomic_create_model(model_path, expected_model)
        except ImmutableArtifactExistsError:
            existing, _state = _load_model_payload(
                model_path,
                Phase4BaselineCorruptionError,
            )
            if not _model_matches_lineage(existing, lineage):
                raise Phase4BaselineCorruptionError(
                    "concurrent baseline model differs from the pilot lineage"
                )

    manifest = _manifest_payload(
        workspace=workspace,
        model_path=model_path,
        lineage=lineage,
    )
    try:
        atomic_create_bytes(manifest_path, canonical_json_bytes(manifest))
        created = True
    except ImmutableArtifactExistsError:
        created = False
    verification = _verify_archive(
        workspace=workspace,
        baseline_root=output,
        lineage=lineage,
    )
    return {
        **verification.to_payload(workspace),
        "created": created,
    }


def verify_bootstrap_baseline(
    *,
    workspace_root: str | Path,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, object]:
    """Deep-verify the baseline model and every referenced pilot artifact."""

    workspace = Path(workspace_root).resolve()
    root = _workspace_directory(
        workspace, baseline_root, "baseline root", must_exist=True
    )
    manifest = _load_json(
        root / BASELINE_MANIFEST_NAME,
        "baseline manifest",
        Phase4BaselineCorruptionError,
    )
    lineage_record = manifest.get("lineage")
    pilot_record = (
        lineage_record.get("pilot")
        if isinstance(lineage_record, Mapping)
        else None
    )
    if not isinstance(pilot_record, Mapping):
        raise Phase4BaselineCorruptionError(
            "baseline manifest does not name its pilot lineage"
        )
    pilot_root = _workspace_directory(
        workspace,
        pilot_record.get("root"),
        "baseline pilot root",
        must_exist=True,
    )
    lineage = _collect_pilot_lineage(
        workspace=workspace,
        pilot_root=pilot_root,
    )
    verification = _verify_archive(
        workspace=workspace,
        baseline_root=root,
        lineage=lineage,
    )
    return verification.to_payload(workspace)


def load_bootstrap_baseline_state(
    *,
    workspace_root: str | Path,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
    verify_lineage: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Load a verified CPU state dict without exposing trainer state."""

    workspace = Path(workspace_root).resolve()
    root = _workspace_directory(
        workspace, baseline_root, "baseline root", must_exist=True
    )
    if verify_lineage:
        verify_bootstrap_baseline(
            workspace_root=workspace,
            baseline_root=root,
        )
    else:
        _verify_archive(
            workspace=workspace,
            baseline_root=root,
            lineage=None,
        )
    _payload, state = _load_model_payload(
        root / BASELINE_MODEL_NAME,
        Phase4BaselineCorruptionError,
    )
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the Phase 4E bootstrap baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Extract warm generation 4 after verifying the Phase 4D pilot.",
    )
    prepare.add_argument("--workspace-root", default=".")
    prepare.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    prepare.add_argument("--output-root", default=str(DEFAULT_BASELINE_ROOT))

    verify = subparsers.add_parser(
        "verify",
        help="Verify the baseline model and its complete source lineage.",
    )
    verify.add_argument("--workspace-root", default=".")
    verify.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_bootstrap_baseline(
                workspace_root=args.workspace_root,
                pilot_root=args.pilot_root,
                output_root=args.output_root,
            )
        else:
            result = verify_bootstrap_baseline(
                workspace_root=args.workspace_root,
                baseline_root=args.baseline_root,
            )
    except Phase4BaselineError as error:
        raise SystemExit(f"phase4-baseline: {error}") from error
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_KIND",
    "BASELINE_MANIFEST_NAME",
    "BASELINE_MODEL_FORMAT",
    "BASELINE_MODEL_NAME",
    "BASELINE_SCHEMA_VERSION",
    "BaselineVerification",
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_PILOT_ROOT",
    "Phase4BaselineCorruptionError",
    "Phase4BaselineError",
    "Phase4BaselineLineageError",
    "load_bootstrap_baseline_state",
    "main",
    "prepare_bootstrap_baseline",
    "verify_bootstrap_baseline",
]
