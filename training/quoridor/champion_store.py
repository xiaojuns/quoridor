"""Independent, model-only champion archive with audited retention."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    RunLock,
    atomic_replace_bytes,
    canonical_json_bytes,
    normalize_json_mapping,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)
from .storage import directory_size


CHAMPION_STORE_SCHEMA_VERSION = 3
CHAMPION_MODEL_SCHEMA_VERSION = 1
DEFAULT_RECENT_CHAMPIONS = 8
DEFAULT_MILESTONES = 4
_MANIFEST_NAME = "manifest.json"
_MODEL_GLOB = "champion-*.pt"
_LOCK_NAME = ".champion-store.lock"
_MODEL_WRITE_OVERHEAD_BYTES = 16 * 1024 * 1024


class ChampionStoreError(RuntimeError):
    """Base class for champion archive failures."""


class ChampionBudgetError(ChampionStoreError):
    """Raised when protected champion artifacts cannot fit the 8 GiB cap."""


class ChampionCorruptionError(ChampionStoreError):
    """Raised when a champion model or manifest fails validation."""


class ChampionConfigurationError(ChampionStoreError):
    """Raised when an archive is opened with a different retention contract."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("artifact reference path must be a non-empty string")
        path = Path(self.path)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            raise ValueError("artifact reference path must be a safe relative path")
        object.__setattr__(
            self,
            "sha256",
            validate_sha256(self.sha256, "artifact reference sha256"),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> "ArtifactReference":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        reference = source
        if relative_to is not None:
            try:
                reference = source.resolve().relative_to(Path(relative_to).resolve())
            except ValueError as error:
                raise ValueError("artifact path is outside relative_to") from error
        return cls(path=reference.as_posix(), sha256=sha256_file(source))

    def to_payload(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ChampionRecord:
    sequence: int
    champion_id: str
    generation: int
    parent_id: str | None
    accepted_at: str
    model_sha256: str
    config_sha256: str
    source_sha256: str
    opening_suite_sha256: str
    arena_report: ArtifactReference
    milestone: bool
    retained: bool
    file_name: str | None
    byte_size: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "id": self.champion_id,
            "generation": self.generation,
            "parentId": self.parent_id,
            "acceptedAt": self.accepted_at,
            "modelSha256": self.model_sha256,
            "configSha256": self.config_sha256,
            "sourceSha256": self.source_sha256,
            "openingSuiteSha256": self.opening_suite_sha256,
            "arenaReport": self.arena_report.to_payload(),
            "milestone": self.milestone,
            "retained": self.retained,
            "file": self.file_name,
            "bytes": self.byte_size,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ChampionVerification:
    accepted_count: int
    retained_count: int
    committed_bytes: int
    deep: bool


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _record_from_payload(payload: object) -> ChampionRecord:
    if not isinstance(payload, Mapping):
        raise TypeError("champion entry must be a mapping")
    champion_id = payload.get("id")
    parent_id = payload.get("parentId")
    accepted_at = payload.get("acceptedAt")
    file_name = payload.get("file")
    if not isinstance(champion_id, str) or not champion_id:
        raise ValueError("champion id must be a non-empty string")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise ValueError("parentId must be a non-empty string or null")
    if not isinstance(accepted_at, str) or not accepted_at:
        raise ValueError("acceptedAt must be a non-empty string")
    if file_name is not None and (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not file_name.startswith("champion-")
        or not file_name.endswith(".pt")
    ):
        raise ValueError("champion file must be null or a plain champion-*.pt name")
    arena_payload = payload.get("arenaReport")
    if not isinstance(arena_payload, Mapping):
        raise TypeError("arenaReport must be a mapping")
    milestone = payload.get("milestone")
    retained = payload.get("retained")
    if type(milestone) is not bool or type(retained) is not bool:
        raise TypeError("milestone and retained must be booleans")
    byte_size = _require_int(payload.get("bytes"), "champion bytes")
    if retained and (file_name is None or byte_size <= 0):
        raise ValueError("retained champion must name a non-empty model file")
    if not retained and (file_name is not None or byte_size != 0):
        raise ValueError("pruned champion must not retain file metadata")
    return ChampionRecord(
        sequence=_require_int(payload.get("sequence"), "champion sequence"),
        champion_id=champion_id,
        generation=_require_int(payload.get("generation"), "champion generation"),
        parent_id=parent_id,
        accepted_at=accepted_at,
        model_sha256=validate_sha256(
            payload.get("modelSha256"), "champion model sha256"
        ),
        config_sha256=validate_sha256(
            payload.get("configSha256"), "champion config sha256"
        ),
        source_sha256=validate_sha256(
            payload.get("sourceSha256"), "champion source sha256"
        ),
        opening_suite_sha256=validate_sha256(
            payload.get("openingSuiteSha256"), "opening suite sha256"
        ),
        arena_report=ArtifactReference(
            path=arena_payload.get("path"),
            sha256=arena_payload.get("sha256"),
        ),
        milestone=milestone,
        retained=retained,
        file_name=file_name,
        byte_size=byte_size,
        metadata=normalize_json_mapping(
            payload.get("metadata", {}), "champion metadata"
        ),
    )


def _model_state(
    model_or_state: nn.Module | Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    source = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    if not isinstance(source, Mapping) or not source:
        raise TypeError("model_or_state must provide a non-empty state mapping")
    items = list(source.items())
    if any(not isinstance(key, str) or not key for key, _ in items):
        raise TypeError("model state keys must be non-empty strings")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in sorted(items, key=lambda item: item[0]):
        if not isinstance(value, torch.Tensor):
            raise TypeError("model-only champion state values must be tensors")
        tensor = value.detach().to(device="cpu").contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"model state tensor {key!r} contains non-finite values")
        state[key] = tensor
    source_metadata = getattr(source, "_metadata", None)
    if isinstance(source_metadata, Mapping):
        state._metadata = deepcopy(source_metadata)  # type: ignore[attr-defined]
    return state


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            return True
    return False


def _model_write_reserve_bytes(state: Mapping[str, torch.Tensor]) -> int:
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state.values()
    )
    return max(_MODEL_WRITE_OVERHEAD_BYTES, tensor_bytes * 2)


class ChampionStore:
    """Model-only archive independent from train-resume checkpoint pruning."""

    def __init__(
        self,
        root: str | Path,
        *,
        budget_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
        recent_to_keep: int = DEFAULT_RECENT_CHAMPIONS,
        milestones_to_keep: int = DEFAULT_MILESTONES,
        artifact_root: str | Path | None = None,
        global_artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
        minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ) -> None:
        if (
            type(budget_bytes) is not int
            or not 0 < budget_bytes <= DEFAULT_ARTIFACT_LIMIT_BYTES
        ):
            raise ValueError("budget_bytes must be in (0, 8 GiB]")
        if (
            type(global_artifact_limit_bytes) is not int
            or not 0
            < global_artifact_limit_bytes
            <= DEFAULT_ARTIFACT_LIMIT_BYTES
        ):
            raise ValueError(
                "global_artifact_limit_bytes must be in (0, 8 GiB]"
            )
        if (
            type(minimum_free_disk_bytes) is not int
            or minimum_free_disk_bytes <= 0
        ):
            raise ValueError(
                "minimum_free_disk_bytes must be a positive integer"
            )
        for value, label in (
            (recent_to_keep, "recent_to_keep"),
            (milestones_to_keep, "milestones_to_keep"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        requested_root = Path(root).absolute()
        requested_artifact_root = (
            Path(artifact_root).absolute()
            if artifact_root is not None
            else requested_root.parent
        )
        if _has_symlink_component(
            requested_root
        ) or _has_symlink_component(requested_artifact_root):
            raise ValueError(
                "champion and artifact roots must not traverse symbolic links"
            )
        resolved_root = requested_root.resolve()
        resolved_artifact_root = requested_artifact_root.resolve()
        try:
            resolved_root.relative_to(resolved_artifact_root)
        except ValueError as error:
            raise ValueError(
                "champion root must be inside the global artifact root"
            ) from error
        resolved_root.mkdir(parents=True, exist_ok=True)
        if (
            _has_symlink_component(resolved_root)
            or _has_symlink_component(resolved_artifact_root)
            or resolved_root.resolve() != resolved_root
        ):
            raise ValueError(
                "champion and artifact roots changed through symbolic links"
            )
        self.root = resolved_root
        self.artifact_root = resolved_artifact_root
        self.manifest_path = self.root / _MANIFEST_NAME
        self.lock_path = self.root / _LOCK_NAME
        self.budget_bytes = budget_bytes
        self.global_artifact_limit_bytes = global_artifact_limit_bytes
        self.minimum_free_disk_bytes = minimum_free_disk_bytes
        self.recent_to_keep = recent_to_keep
        self.milestones_to_keep = milestones_to_keep
        self._records: list[ChampionRecord] = []
        self._pointers: dict[str, str | None] = {
            "latest": None,
            "best": None,
            "product": None,
        }
        with self._lock():
            self._reload_unlocked(allow_initialize=True)
            self._assert_global_write_budget(reserve_bytes=0)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[ChampionRecord, ...]:
        return tuple(self._records)

    @property
    def retained_records(self) -> tuple[ChampionRecord, ...]:
        return tuple(record for record in self._records if record.retained)

    @property
    def pointers(self) -> dict[str, str | None]:
        return dict(self._pointers)

    @property
    def committed_bytes(self) -> int:
        manifest_size = self.manifest_path.stat().st_size if self.manifest_path.exists() else 0
        return manifest_size + sum(record.byte_size for record in self._records)

    def _assert_global_write_budget(self, *, reserve_bytes: int) -> None:
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a non-negative integer")
        committed = directory_size(self.artifact_root)
        projected = committed + reserve_bytes
        if projected > self.global_artifact_limit_bytes:
            raise ChampionBudgetError(
                "global Phase 4 artifact budget would be exceeded: "
                f"{projected} > {self.global_artifact_limit_bytes} bytes"
            )
        free = shutil.disk_usage(self.artifact_root).free
        if free - reserve_bytes < self.minimum_free_disk_bytes:
            raise ChampionBudgetError(
                "global Phase 4 free-disk stop line would be crossed: "
                f"{free - reserve_bytes} < "
                f"{self.minimum_free_disk_bytes} bytes"
            )

    def publish(
        self,
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        *,
        generation: int,
        parent_id: str | None,
        config_sha256: str,
        source_sha256: str,
        opening_suite_sha256: str,
        arena_report: ArtifactReference,
        milestone: bool = False,
        best: bool = False,
        product: bool = False,
        metadata: Mapping[str, object] | None = None,
        accepted_at: str | None = None,
    ) -> ChampionRecord:
        with self._lock():
            self._reload_unlocked()
            return self._publish_unlocked(
                model_or_state,
                generation=generation,
                parent_id=parent_id,
                config_sha256=config_sha256,
                source_sha256=source_sha256,
                opening_suite_sha256=opening_suite_sha256,
                arena_report=arena_report,
                milestone=milestone,
                best=best,
                product=product,
                metadata=metadata,
                accepted_at=accepted_at,
            )

    def _publish_unlocked(
        self,
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        *,
        generation: int,
        parent_id: str | None,
        config_sha256: str,
        source_sha256: str,
        opening_suite_sha256: str,
        arena_report: ArtifactReference,
        milestone: bool,
        best: bool,
        product: bool,
        metadata: Mapping[str, object] | None,
        accepted_at: str | None,
    ) -> ChampionRecord:
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(arena_report, ArtifactReference):
            raise TypeError("arena_report must be an ArtifactReference")
        self._verify_arena_report(arena_report)
        if accepted_at is not None and (
            not isinstance(accepted_at, str) or not accepted_at
        ):
            raise ValueError("accepted_at must be a non-empty string or None")
        for value, label in (
            (milestone, "milestone"),
            (best, "best"),
            (product, "product"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{label} must be a boolean")
        latest_id = self._pointers["latest"]
        if latest_id is None:
            if parent_id is not None:
                raise ValueError("the first champion must not have a parent")
        else:
            if parent_id != latest_id:
                raise ValueError("new champion parent_id must be the current latest champion")
            parent = self._record(parent_id)
            if generation <= parent.generation:
                raise ValueError("champion generation must advance beyond its parent")

        state = _model_state(model_or_state)
        self._assert_global_write_budget(
            reserve_bytes=_model_write_reserve_bytes(state)
        )
        temporary_path = self._save_model_temporary(generation, state)
        try:
            byte_size = temporary_path.stat().st_size
            model_sha256 = sha256_file(temporary_path)
            sequence = len(self._records)
            champion_id = (
                f"g{generation:08d}-s{sequence:08d}-{model_sha256[:12]}"
            )
            file_name = f"champion-{champion_id}.pt"
            record = ChampionRecord(
                sequence=sequence,
                champion_id=champion_id,
                generation=generation,
                parent_id=parent_id,
                accepted_at=accepted_at or utc_timestamp(),
                model_sha256=model_sha256,
                config_sha256=validate_sha256(config_sha256, "config_sha256"),
                source_sha256=validate_sha256(source_sha256, "source_sha256"),
                opening_suite_sha256=validate_sha256(
                    opening_suite_sha256, "opening_suite_sha256"
                ),
                arena_report=arena_report,
                milestone=milestone,
                retained=True,
                file_name=file_name,
                byte_size=byte_size,
                metadata=normalize_json_mapping(metadata or {}, "champion metadata"),
            )
            pointers = dict(self._pointers)
            pointers["latest"] = champion_id
            if best:
                pointers["best"] = champion_id
            if product:
                pointers["product"] = champion_id
            planned, pruned = self._retention_plan([*self._records, record], pointers)
            manifest_bytes = self._manifest_bytes(planned, pointers)
            self._assert_global_write_budget(
                reserve_bytes=len(manifest_bytes)
            )
            destination = self.root / file_name
            if destination.exists():
                raise ChampionCorruptionError(f"champion model already exists: {file_name}")
            previous_manifest = self.manifest_path.read_bytes()
            os.replace(temporary_path, destination)
            temporary_path = None
            _fsync_directory(self.root)
            manifest_replaced = False
            try:
                atomic_replace_bytes(self.manifest_path, manifest_bytes)
                manifest_replaced = True
                self._assert_global_write_budget(reserve_bytes=0)
            except Exception:
                if manifest_replaced:
                    atomic_replace_bytes(
                        self.manifest_path,
                        previous_manifest,
                    )
                destination.unlink(missing_ok=True)
                _fsync_directory(self.root)
                raise
            self._records = planned
            self._pointers = pointers
            self._delete_pruned_models(pruned)
            return self._record(champion_id)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def set_role(self, champion_id: str, role: str) -> ChampionRecord:
        """Protect a retained champion as ``best``, ``product``, or milestone."""

        with self._lock():
            self._reload_unlocked()
            return self._set_role_unlocked(champion_id, role)

    def _set_role_unlocked(self, champion_id: str, role: str) -> ChampionRecord:
        record = self._record(champion_id)
        if not record.retained:
            raise ValueError("cannot assign a role to a pruned model artifact")
        records = list(self._records)
        pointers = dict(self._pointers)
        if role == "milestone":
            records[record.sequence] = replace(record, milestone=True)
        elif role in ("best", "product"):
            pointers[role] = champion_id
        else:
            raise ValueError("role must be best, product, or milestone")
        planned, pruned = self._retention_plan(records, pointers)
        self._commit_manifest(planned, pointers)
        self._delete_pruned_models(pruned)
        return self._record(champion_id)

    def resolve(self, champion_id: str | None = None, *, role: str = "latest") -> ChampionRecord:
        if champion_id is not None:
            return self._record(champion_id)
        if role not in self._pointers:
            raise ValueError("role must be latest, best, or product")
        selected = self._pointers[role]
        if selected is None:
            raise KeyError(f"no {role} champion is available")
        return self._record(selected)

    def load_state_dict(
        self,
        champion_id: str | None = None,
        *,
        role: str = "latest",
        map_location: str | torch.device = "cpu",
    ) -> OrderedDict[str, torch.Tensor]:
        record = self.resolve(champion_id, role=role)
        if not record.retained or record.file_name is None:
            raise KeyError(f"champion model was pruned: {record.champion_id}")
        path = self.root / record.file_name
        self._verify_model_file(record)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as error:
            raise ChampionCorruptionError(
                f"cannot deserialize champion model {record.champion_id}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ChampionCorruptionError("champion model payload must be a mapping")
        if payload.get("schemaVersion") != CHAMPION_MODEL_SCHEMA_VERSION:
            raise ChampionCorruptionError("unsupported champion model schema")
        if payload.get("format") != "pytorch-state-dict-only":
            raise ChampionCorruptionError("champion artifact is not model-only")
        if payload.get("generation") != record.generation:
            raise ChampionCorruptionError("champion model generation mismatch")
        raw_state = payload.get("modelState")
        try:
            state = _model_state(raw_state)
        except (TypeError, ValueError) as error:
            raise ChampionCorruptionError("champion model state is invalid") from error
        mapped: OrderedDict[str, torch.Tensor] = OrderedDict(
            (key, value.to(map_location)) for key, value in state.items()
        )
        source_metadata = getattr(state, "_metadata", None)
        if isinstance(source_metadata, Mapping):
            mapped._metadata = deepcopy(source_metadata)  # type: ignore[attr-defined]
        return mapped

    def load_into(
        self,
        model: nn.Module,
        champion_id: str | None = None,
        *,
        role: str = "latest",
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> ChampionRecord:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        record = self.resolve(champion_id, role=role)
        state = self.load_state_dict(
            record.champion_id,
            map_location=map_location,
        )
        model.load_state_dict(state, strict=strict)
        return record

    def verify(self, *, deep: bool = True) -> ChampionVerification:
        for record in self._records:
            self._verify_arena_report(record.arena_report)
        for record in self.retained_records:
            self._verify_model_file(record)
            if deep:
                self.load_state_dict(record.champion_id, map_location="cpu")
        return ChampionVerification(
            accepted_count=len(self._records),
            retained_count=len(self.retained_records),
            committed_bytes=self.committed_bytes,
            deep=deep,
        )

    def _record(self, champion_id: str) -> ChampionRecord:
        for record in self._records:
            if record.champion_id == champion_id:
                return record
        raise KeyError(champion_id)

    def _save_model_temporary(
        self, generation: int, state: Mapping[str, torch.Tensor]
    ) -> Path:
        handle, name = tempfile.mkstemp(
            dir=self.root,
            prefix=".champion-model.",
            suffix=".tmp",
        )
        path = Path(name)
        try:
            with os.fdopen(handle, "wb") as output:
                torch.save(
                    {
                        "schemaVersion": CHAMPION_MODEL_SCHEMA_VERSION,
                        "format": "pytorch-state-dict-only",
                        "generation": generation,
                        "modelState": state,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def _protected_ids(
        self,
        records: list[ChampionRecord],
        pointers: Mapping[str, str | None],
    ) -> set[str]:
        protected = {
            record.champion_id for record in records[-self.recent_to_keep :]
        }
        milestones = [record for record in records if record.milestone]
        protected.update(
            record.champion_id
            for record in milestones[-self.milestones_to_keep :]
        )
        protected.update(
            champion_id
            for champion_id in (pointers.get("best"), pointers.get("product"))
            if champion_id is not None
        )
        latest = pointers.get("latest")
        if latest is not None:
            protected.add(latest)
        return protected

    def _retention_plan(
        self,
        records: list[ChampionRecord],
        pointers: Mapping[str, str | None],
    ) -> tuple[list[ChampionRecord], list[ChampionRecord]]:
        protected = self._protected_ids(records, pointers)
        planned: list[ChampionRecord] = []
        pruned: list[ChampionRecord] = []
        for record in records:
            should_retain = record.champion_id in protected
            if should_retain and not record.retained:
                raise ChampionBudgetError(
                    f"protected champion model is unavailable: {record.champion_id}"
                )
            if should_retain:
                planned.append(record)
            elif record.retained:
                pruned.append(record)
                planned.append(
                    replace(record, retained=False, file_name=None, byte_size=0)
                )
            else:
                planned.append(record)
        manifest_bytes = self._manifest_bytes(planned, pointers)
        committed = len(manifest_bytes) + sum(record.byte_size for record in planned)
        if committed > self.budget_bytes:
            raise ChampionBudgetError(
                f"protected champion archive exceeds budget: "
                f"{committed} > {self.budget_bytes} bytes"
            )
        return planned, pruned

    def _manifest_core(
        self,
        records: list[ChampionRecord],
        pointers: Mapping[str, str | None],
    ) -> dict[str, object]:
        return {
            "schemaVersion": CHAMPION_STORE_SCHEMA_VERSION,
            "modelSchemaVersion": CHAMPION_MODEL_SCHEMA_VERSION,
            "artifactRoot": str(self.artifact_root),
            "budgetBytes": self.budget_bytes,
            "globalArtifactBudget": {
                "limitBytes": self.global_artifact_limit_bytes,
                "minimumFreeDiskBytes": self.minimum_free_disk_bytes,
            },
            "retention": {
                "recent": self.recent_to_keep,
                "milestones": self.milestones_to_keep,
                "protectedRoles": ["best", "product"],
            },
            "pointers": dict(pointers),
            "totals": {
                "accepted": len(records),
                "retained": sum(record.retained for record in records),
                "modelBytes": sum(record.byte_size for record in records),
            },
            "champions": [record.to_payload() for record in records],
        }

    def _manifest_bytes(
        self,
        records: list[ChampionRecord],
        pointers: Mapping[str, str | None],
    ) -> bytes:
        core = self._manifest_core(records, pointers)
        digest = sha256_bytes(canonical_json_bytes(core))
        return canonical_json_bytes({**core, "manifestSha256": digest})

    def _commit_manifest(
        self,
        records: list[ChampionRecord],
        pointers: Mapping[str, str | None],
    ) -> None:
        manifest_bytes = self._manifest_bytes(records, pointers)
        committed = len(manifest_bytes) + sum(record.byte_size for record in records)
        if committed > self.budget_bytes:
            raise ChampionBudgetError(
                f"champion archive exceeds budget: {committed} > {self.budget_bytes}"
            )
        atomic_replace_bytes(self.manifest_path, manifest_bytes)
        self._records = records
        self._pointers = dict(pointers)

    def _delete_pruned_models(self, records: list[ChampionRecord]) -> None:
        for record in records:
            if record.file_name is not None:
                (self.root / record.file_name).unlink(missing_ok=True)
        if records:
            _fsync_directory(self.root)

    def _load_manifest(self) -> None:
        try:
            raw = self.manifest_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ChampionCorruptionError("champion manifest is unreadable") from error
        if not isinstance(payload, Mapping):
            raise ChampionCorruptionError("champion manifest must be a mapping")
        try:
            if payload.get("schemaVersion") != CHAMPION_STORE_SCHEMA_VERSION:
                raise ValueError("unsupported champion store schema")
            if payload.get("modelSchemaVersion") != CHAMPION_MODEL_SCHEMA_VERSION:
                raise ValueError("unsupported champion model schema")
            if payload.get("artifactRoot") != str(self.artifact_root):
                raise ChampionConfigurationError(
                    "champion artifact root does not match manifest"
                )
            stored_digest = validate_sha256(
                payload.get("manifestSha256"), "champion manifest sha256"
            )
            core = dict(payload)
            del core["manifestSha256"]
            if sha256_bytes(canonical_json_bytes(core)) != stored_digest:
                raise ValueError("champion manifest self hash mismatch")
            if payload.get("budgetBytes") != self.budget_bytes:
                raise ChampionConfigurationError("champion budget does not match manifest")
            global_budget = payload.get("globalArtifactBudget")
            if (
                not isinstance(global_budget, Mapping)
                or dict(global_budget)
                != {
                    "limitBytes": self.global_artifact_limit_bytes,
                    "minimumFreeDiskBytes": self.minimum_free_disk_bytes,
                }
            ):
                raise ChampionConfigurationError(
                    "global artifact budget does not match manifest"
                )
            retention = payload.get("retention")
            if not isinstance(retention, Mapping):
                raise TypeError("retention must be a mapping")
            if (
                retention.get("recent") != self.recent_to_keep
                or retention.get("milestones") != self.milestones_to_keep
            ):
                raise ChampionConfigurationError(
                    "champion retention settings do not match manifest"
                )
            raw_records = payload.get("champions")
            if not isinstance(raw_records, list):
                raise TypeError("champions must be a list")
            records = [_record_from_payload(item) for item in raw_records]
            pointers = payload.get("pointers")
            if not isinstance(pointers, Mapping) or set(pointers) != {
                "latest",
                "best",
                "product",
            }:
                raise ValueError("champion pointers are invalid")
            normalized_pointers: dict[str, str | None] = {}
            ids = {record.champion_id for record in records}
            for role in ("latest", "best", "product"):
                selected = pointers.get(role)
                if selected is not None and selected not in ids:
                    raise ValueError(f"{role} pointer references an unknown champion")
                normalized_pointers[role] = selected
            if [record.sequence for record in records] != list(range(len(records))):
                raise ValueError("champion sequences must be contiguous and ordered")
            if len(ids) != len(records):
                raise ValueError("champion IDs must be unique")
            for index, record in enumerate(records):
                if index == 0:
                    if record.parent_id is not None:
                        raise ValueError("first champion must not have a parent")
                else:
                    parent = records[index - 1]
                    if record.parent_id != parent.champion_id:
                        raise ValueError("champion lineage must follow accepted order")
                    if record.generation <= parent.generation:
                        raise ValueError("champion generations must increase")
            expected_latest = records[-1].champion_id if records else None
            if normalized_pointers["latest"] != expected_latest:
                raise ValueError("latest pointer does not reference newest champion")
            expected_retained = self._protected_ids(records, normalized_pointers)
            actual_retained = {
                record.champion_id for record in records if record.retained
            }
            if actual_retained != expected_retained:
                raise ValueError("retained models do not match retention policy")
            totals = payload.get("totals")
            expected_totals = {
                "accepted": len(records),
                "retained": len(actual_retained),
                "modelBytes": sum(record.byte_size for record in records),
            }
            if not isinstance(totals, Mapping) or dict(totals) != expected_totals:
                raise ValueError("champion manifest totals are invalid")
            if len(raw) + expected_totals["modelBytes"] > self.budget_bytes:
                raise ValueError("champion archive exceeds configured budget")
        except ChampionConfigurationError:
            raise
        except (TypeError, ValueError) as error:
            raise ChampionCorruptionError(f"invalid champion manifest: {error}") from error
        self._records = records
        self._pointers = normalized_pointers
        for record in self.retained_records:
            if record.file_name is None or not (self.root / record.file_name).is_file():
                raise ChampionCorruptionError(
                    f"retained champion model is missing: {record.champion_id}"
                )
            if (self.root / record.file_name).stat().st_size != record.byte_size:
                raise ChampionCorruptionError(
                    f"champion model size mismatch: {record.champion_id}"
                )
        for record in self._records:
            self._verify_arena_report(record.arena_report)

    def _verify_arena_report(self, reference: ArtifactReference) -> None:
        path = (self.artifact_root / reference.path).resolve()
        try:
            path.relative_to(self.artifact_root)
        except ValueError as error:
            raise ChampionCorruptionError(
                "arena report reference escapes the artifact root"
            ) from error
        if not path.is_file():
            raise ChampionCorruptionError(f"arena report is missing: {reference.path}")
        if sha256_file(path) != reference.sha256:
            raise ChampionCorruptionError(
                f"arena report checksum mismatch: {reference.path}"
            )

    def _verify_model_file(self, record: ChampionRecord) -> None:
        if not record.retained or record.file_name is None:
            raise ChampionCorruptionError(
                f"champion artifact is not retained: {record.champion_id}"
            )
        path = self.root / record.file_name
        if not path.is_file():
            raise ChampionCorruptionError(
                f"champion model is missing: {record.champion_id}"
            )
        if path.stat().st_size != record.byte_size:
            raise ChampionCorruptionError(
                f"champion model size mismatch: {record.champion_id}"
            )
        if sha256_file(path) != record.model_sha256:
            raise ChampionCorruptionError(
                f"champion model checksum mismatch: {record.champion_id}"
            )

    def _lock(self) -> RunLock:
        return RunLock(
            self.lock_path,
            run_id="champion-store",
            blocking=True,
        )

    def _reload_unlocked(self, *, allow_initialize: bool = False) -> None:
        self._remove_temporary_files()
        if self.manifest_path.exists():
            self._load_manifest()
        elif tuple(self.root.glob(_MODEL_GLOB)):
            raise ChampionCorruptionError("champion models exist without a manifest")
        elif allow_initialize:
            self._records = []
            self._pointers = {
                "latest": None,
                "best": None,
                "product": None,
            }
            self._commit_manifest([], self._pointers)
        else:
            raise ChampionCorruptionError("champion manifest is missing")
        self._remove_orphan_models()

    def _remove_temporary_files(self) -> None:
        for path in self.root.glob(".*.tmp"):
            if path.is_file():
                path.unlink(missing_ok=True)

    def _remove_orphan_models(self) -> None:
        committed = {
            record.file_name
            for record in self._records
            if record.retained and record.file_name is not None
        }
        removed = False
        for path in self.root.glob(_MODEL_GLOB):
            if path.name not in committed:
                path.unlink(missing_ok=True)
                removed = True
        if removed:
            _fsync_directory(self.root)


__all__ = [
    "CHAMPION_STORE_SCHEMA_VERSION",
    "CHAMPION_MODEL_SCHEMA_VERSION",
    "DEFAULT_RECENT_CHAMPIONS",
    "DEFAULT_MILESTONES",
    "ChampionStoreError",
    "ChampionBudgetError",
    "ChampionCorruptionError",
    "ChampionConfigurationError",
    "ArtifactReference",
    "ChampionRecord",
    "ChampionVerification",
    "ChampionStore",
]
