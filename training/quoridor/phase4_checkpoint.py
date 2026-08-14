"""Checksummed, lineage-bound checkpoints for the Phase 4 pipeline."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .checkpoint import capture_rng_state, restore_rng_state
from .run_manifest import (
    ImmutableArtifactExistsError,
    atomic_create_bytes,
    atomic_replace_bytes,
    canonical_json_bytes,
    normalize_json_mapping,
    sha256_file,
    validate_sha256,
)
from .storage import ArtifactBudgetError, directory_size
from .trainer import AlphaZeroTrainer


PHASE4_CHECKPOINT_SCHEMA_VERSION = 1
PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION = 2
PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION = 1
PHASE4_CANDIDATE_MODEL_FORMAT = "quoridor-phase4-candidate-model-only"
_CHECKPOINT_GLOB = "checkpoint-g*-s*.json"
_CANDIDATE_MODEL_GLOB = "candidate-model-checkpoint-g*-s*.pt"
_INDEX_WRITE_RESERVE_BYTES = 4096


class Phase4CheckpointError(RuntimeError):
    """Base class for Phase 4 checkpoint failures."""


class Phase4CheckpointCorruptionError(Phase4CheckpointError):
    """Raised when checkpoint bytes, metadata, or payload are invalid."""


class Phase4CheckpointIdentityError(Phase4CheckpointError):
    """Raised when a checkpoint belongs to a different immutable lineage."""


def canonical_model_state_sha256(
    model_or_state: nn.Module | Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor keys, dtypes, shapes, and canonical CPU bytes."""

    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise TypeError("model state must be a non-empty mapping")
    digest = hashlib.sha256(b"quoridor-model-tensor-state-v1\x00")
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not key:
            raise TypeError("model state keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("model state values must be tensors")
        canonical = tensor.detach().to(device="cpu").contiguous()
        if (canonical.is_floating_point() or canonical.is_complex()) and not bool(
            torch.isfinite(canonical).all().item()
        ):
            raise ValueError(f"model state tensor {key!r} contains NaN or Inf")
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(canonical.dtype).encode("ascii")
        shape_bytes = canonical_json_bytes(list(canonical.shape))
        content = canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        for block in (key_bytes, dtype_bytes, shape_bytes, content):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def _candidate_model_state(
    model_or_state: nn.Module | Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    source = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    if not isinstance(source, Mapping) or not source:
        raise TypeError("model state must be a non-empty mapping")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(source):
        value = source[key]
        if not isinstance(key, str) or not key:
            raise TypeError("model state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("model state values must be tensors")
        tensor = value.detach().to(device="cpu").contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"model state tensor {key!r} contains NaN or Inf")
        state[key] = tensor
    source_metadata = getattr(source, "_metadata", None)
    if isinstance(source_metadata, Mapping):
        state._metadata = deepcopy(source_metadata)  # type: ignore[attr-defined]
    return state


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    run_manifest_sha256: str
    config_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_manifest_sha256, "run_manifest_sha256"),
            (self.config_sha256, "config_sha256"),
            (self.source_sha256, "source_sha256"),
        ):
            validate_sha256(value, label)

    def to_payload(self) -> dict[str, str]:
        return {
            "runManifestSha256": self.run_manifest_sha256,
            "configSha256": self.config_sha256,
            "sourceSha256": self.source_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CheckpointIdentity":
        if not isinstance(payload, Mapping):
            raise Phase4CheckpointCorruptionError(
                "checkpoint identity must be a mapping"
            )
        try:
            return cls(
                run_manifest_sha256=payload.get("runManifestSha256"),
                config_sha256=payload.get("configSha256"),
                source_sha256=payload.get("sourceSha256"),
            )
        except (TypeError, ValueError) as error:
            raise Phase4CheckpointCorruptionError(
                "checkpoint identity is invalid"
            ) from error


@dataclass(frozen=True, slots=True)
class Phase4CheckpointRecord:
    generation: int
    step: int
    data_path: Path
    metadata_path: Path
    byte_size: int
    sha256: str
    candidate_model_path: Path | None = None
    candidate_model_byte_size: int = 0
    candidate_model_sha256: str | None = None
    model_state_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Phase4CheckpointLoad:
    record: Phase4CheckpointRecord
    payload: Mapping[str, Any]
    rejected_newer: tuple[str, ...]


def _require_non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase4CheckpointCorruptionError(
            f"{label} must be a non-negative integer"
        )
    return value


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


def _atomic_create_torch(
    payload: object,
    destination: Path,
    *,
    artifact_root: Path,
    artifact_limit_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as source:
            os.fsync(source.fileno())
        temporary_size = temporary_path.stat().st_size
        committed_size = directory_size(artifact_root) - temporary_size
        if committed_size + temporary_size > artifact_limit_bytes:
            raise ArtifactBudgetError(
                "artifact budget exceeded: "
                f"{committed_size + temporary_size} > {artifact_limit_bytes} bytes"
            )
        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise ImmutableArtifactExistsError(
                f"immutable artifact already exists: {destination}"
            ) from error
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class Phase4CheckpointStore:
    """Immutable checkpoints with pre-deserialization checksum validation."""

    def __init__(
        self,
        root: str | Path,
        *,
        identity: CheckpointIdentity,
        artifact_root: str | Path,
        artifact_limit_bytes: int,
        max_to_keep: int,
    ) -> None:
        if not isinstance(identity, CheckpointIdentity):
            raise TypeError("identity must be a CheckpointIdentity")
        if type(artifact_limit_bytes) is not int or artifact_limit_bytes <= 0:
            raise ValueError("artifact_limit_bytes must be a positive integer")
        if type(max_to_keep) is not int or max_to_keep < 2:
            raise ValueError("max_to_keep must be at least two for fallback")
        self.root = Path(root)
        self.identity = identity
        self.artifact_root = Path(artifact_root)
        self.artifact_limit_bytes = artifact_limit_bytes
        self.max_to_keep = max_to_keep
        self.root.mkdir(parents=True, exist_ok=True)
        self._remove_orphans()

    @staticmethod
    def _stem(generation: int, step: int) -> str:
        if type(generation) is not int or generation < -1:
            raise ValueError("generation must be an integer >= -1")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        generation_token = (
            f"-{abs(generation):07d}" if generation < 0 else f"{generation:08d}"
        )
        return f"checkpoint-g{generation_token}-s{step:012d}"

    def save(
        self,
        *,
        generation: int,
        model: nn.Module,
        trainer: AlphaZeroTrainer,
        generation_data: Mapping[str, object],
        replay_snapshot: Mapping[str, object],
    ) -> Phase4CheckpointRecord:
        if trainer.step < 0:
            raise ValueError("trainer step must be non-negative")
        normalized_generation_data = normalize_json_mapping(
            generation_data, "checkpoint generation data"
        )
        normalized_replay = normalize_json_mapping(
            replay_snapshot, "checkpoint replay snapshot"
        )
        self._validate_generation_contract(
            generation,
            trainer.step,
            normalized_generation_data,
            normalized_replay,
        )
        stem = self._stem(generation, trainer.step)
        data_path = self.root / f"{stem}.pt"
        metadata_path = self.root / f"{stem}.json"
        candidate_model_path = self.root / f"candidate-model-{stem}.pt"
        if (
            data_path.exists()
            or metadata_path.exists()
            or candidate_model_path.exists()
        ):
            try:
                self._validate_record(metadata_path)
            except (OSError, Phase4CheckpointError):
                data_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                candidate_model_path.unlink(missing_ok=True)
            else:
                raise ImmutableArtifactExistsError(
                    f"immutable checkpoint already exists: {stem}"
                )
        candidate_state = _candidate_model_state(model)
        model_state_sha256 = canonical_model_state_sha256(candidate_state)
        payload = {
            "schemaVersion": PHASE4_CHECKPOINT_SCHEMA_VERSION,
            "generation": generation,
            "step": trainer.step,
            "identity": self.identity.to_payload(),
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "trainerRuntime": trainer.runtime_state_dict(),
            "rng": capture_rng_state(),
            "generationData": normalized_generation_data,
            "replaySnapshot": normalized_replay,
        }
        prunable_bytes = self._prunable_bytes_after_next_save()
        data_write_limit = (
            self.artifact_limit_bytes
            + prunable_bytes
            - _INDEX_WRITE_RESERVE_BYTES
        )
        if data_write_limit <= 0:
            raise ArtifactBudgetError(
                "artifact budget is too small for checkpoint indexes"
            )
        _atomic_create_torch(
            payload,
            data_path,
            artifact_root=self.artifact_root,
            artifact_limit_bytes=data_write_limit,
        )
        try:
            byte_size = data_path.stat().st_size
            digest = sha256_file(data_path)
            candidate_payload = {
                "schemaVersion": PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
                "format": PHASE4_CANDIDATE_MODEL_FORMAT,
                "generation": generation,
                "step": trainer.step,
                "identity": self.identity.to_payload(),
                "checkpointSha256": digest,
                "modelStateSha256": model_state_sha256,
                "modelState": candidate_state,
            }
            _atomic_create_torch(
                candidate_payload,
                candidate_model_path,
                artifact_root=self.artifact_root,
                artifact_limit_bytes=data_write_limit,
            )
            candidate_model_byte_size = candidate_model_path.stat().st_size
            candidate_model_sha256 = sha256_file(candidate_model_path)
            metadata = {
                "schemaVersion": PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION,
                "generation": generation,
                "step": trainer.step,
                "file": data_path.name,
                "bytes": byte_size,
                "sha256": digest,
                "identity": self.identity.to_payload(),
                "candidateModel": {
                    "schemaVersion": PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
                    "format": PHASE4_CANDIDATE_MODEL_FORMAT,
                    "file": candidate_model_path.name,
                    "bytes": candidate_model_byte_size,
                    "sha256": candidate_model_sha256,
                    "modelStateSha256": model_state_sha256,
                },
            }
            atomic_create_bytes(metadata_path, canonical_json_bytes(metadata))
            if directory_size(self.artifact_root) > (
                self.artifact_limit_bytes + prunable_bytes
            ):
                raise ArtifactBudgetError(
                    "checkpoint metadata would exceed the artifact budget"
                )
        except Exception:
            metadata_path.unlink(missing_ok=True)
            candidate_model_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
            _fsync_directory(self.root)
            raise
        record = Phase4CheckpointRecord(
            generation=generation,
            step=trainer.step,
            data_path=data_path,
            metadata_path=metadata_path,
            byte_size=byte_size,
            sha256=digest,
            candidate_model_path=candidate_model_path,
            candidate_model_byte_size=candidate_model_byte_size,
            candidate_model_sha256=candidate_model_sha256,
            model_state_sha256=model_state_sha256,
        )
        self._write_latest(record)
        self._prune()
        if directory_size(self.artifact_root) > self.artifact_limit_bytes:
            raise ArtifactBudgetError(
                "retained checkpoint set exceeds the artifact budget"
            )
        return record

    def candidates(self) -> tuple[Path, ...]:
        return tuple(sorted(self.root.glob(_CHECKPOINT_GLOB), reverse=True))

    def latest_valid(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> Phase4CheckpointLoad | None:
        rejected: list[str] = []
        for metadata_path in self.candidates():
            try:
                record = self._validate_record(metadata_path)
                payload = torch.load(
                    record.data_path,
                    map_location=map_location,
                    weights_only=False,
                )
                self._validate_payload(payload, record)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                rejected.append(f"{metadata_path.name}: {type(error).__name__}: {error}")
                continue
            return Phase4CheckpointLoad(
                record=record,
                payload=payload,
                rejected_newer=tuple(rejected),
            )
        if rejected:
            raise Phase4CheckpointCorruptionError(
                "no valid Phase 4 checkpoint remains; " + "; ".join(rejected)
            )
        return None

    def load_generation(
        self,
        generation: int,
        *,
        map_location: str | torch.device = "cpu",
    ) -> Phase4CheckpointLoad:
        """Load one exact generation after index, checksum, and identity checks."""

        matches: list[Phase4CheckpointLoad] = []
        for metadata_path in self.candidates():
            record = self._validate_record(metadata_path)
            if record.generation != generation:
                continue
            payload = torch.load(
                record.data_path,
                map_location=map_location,
                weights_only=False,
            )
            self._validate_payload(payload, record)
            matches.append(
                Phase4CheckpointLoad(
                    record=record,
                    payload=payload,
                    rejected_newer=(),
                )
            )
        if len(matches) != 1:
            raise Phase4CheckpointCorruptionError(
                f"expected exactly one valid checkpoint for generation {generation}"
            )
        return matches[0]

    def restore(
        self,
        model: nn.Module,
        trainer: AlphaZeroTrainer,
        *,
        map_location: str | torch.device,
        restore_rng: bool = True,
    ) -> Phase4CheckpointLoad | None:
        rejected: list[str] = []
        for metadata_path in self.candidates():
            rng_before = capture_rng_state() if restore_rng else None
            try:
                record = self._validate_record(metadata_path)
                payload = torch.load(
                    record.data_path,
                    map_location=map_location,
                    weights_only=False,
                )
                self._validate_payload(payload, record)
                model.load_state_dict(payload["model"])
                trainer.optimizer.load_state_dict(payload["optimizer"])
                trainer.load_runtime_state_dict(payload["trainerRuntime"])
                if trainer.step != record.step:
                    raise Phase4CheckpointCorruptionError(
                        "trainer runtime step does not match checkpoint metadata"
                    )
                if restore_rng:
                    restore_rng_state(payload["rng"])
            except Exception as error:
                if rng_before is not None:
                    try:
                        restore_rng_state(rng_before)
                    except Exception as rollback_error:
                        raise Phase4CheckpointCorruptionError(
                            "failed to restore RNG state after rejecting a checkpoint"
                        ) from rollback_error
                rejected.append(
                    f"{metadata_path.name}: {type(error).__name__}: {error}"
                )
                continue
            return Phase4CheckpointLoad(
                record=record,
                payload=payload,
                rejected_newer=tuple(rejected),
            )
        if rejected:
            raise Phase4CheckpointCorruptionError(
                "no restorable Phase 4 checkpoint remains; " + "; ".join(rejected)
            )
        return None

    def _validate_record(self, metadata_path: Path) -> Phase4CheckpointRecord:
        import json

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Phase4CheckpointCorruptionError(
                f"cannot read checkpoint metadata {metadata_path.name}"
            ) from error
        if not isinstance(metadata, Mapping):
            raise Phase4CheckpointCorruptionError(
                "checkpoint metadata must be a mapping"
            )
        metadata_schema = metadata.get("schemaVersion")
        if metadata_schema not in (1, PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION):
            raise Phase4CheckpointCorruptionError(
                "unsupported checkpoint metadata schema"
            )
        identity = CheckpointIdentity.from_payload(metadata.get("identity"))
        if identity != self.identity:
            raise Phase4CheckpointIdentityError(
                "checkpoint metadata belongs to another lineage"
            )
        generation_value = metadata.get("generation")
        if type(generation_value) is not int or generation_value < -1:
            raise Phase4CheckpointCorruptionError(
                "checkpoint generation must be an integer >= -1"
            )
        generation = generation_value
        step = _require_non_negative_int(metadata.get("step"), "checkpoint step")
        expected_stem = self._stem(generation, step)
        if metadata_path.stem != expected_stem:
            raise Phase4CheckpointCorruptionError(
                "checkpoint metadata path/content mismatch"
            )
        file_name = metadata.get("file")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or file_name != f"{expected_stem}.pt"
        ):
            raise Phase4CheckpointCorruptionError("checkpoint file name is invalid")
        data_path = self.root / file_name
        if not data_path.is_file() or data_path.is_symlink():
            raise Phase4CheckpointCorruptionError(
                "checkpoint data file is missing or is a symlink"
            )
        byte_size = _require_non_negative_int(
            metadata.get("bytes"), "checkpoint byte size"
        )
        if byte_size <= 0 or data_path.stat().st_size != byte_size:
            raise Phase4CheckpointCorruptionError("checkpoint byte size mismatch")
        try:
            expected_digest = validate_sha256(
                metadata.get("sha256"), "checkpoint sha256"
            )
        except ValueError as error:
            raise Phase4CheckpointCorruptionError(
                "checkpoint SHA-256 metadata is invalid"
            ) from error
        if sha256_file(data_path) != expected_digest:
            raise Phase4CheckpointCorruptionError("checkpoint SHA-256 mismatch")
        candidate_model_path: Path | None = None
        candidate_model_byte_size = 0
        candidate_model_sha256: str | None = None
        model_state_sha256: str | None = None
        if metadata_schema == PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION:
            candidate = metadata.get("candidateModel")
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "schemaVersion",
                "format",
                "file",
                "bytes",
                "sha256",
                "modelStateSha256",
            }:
                raise Phase4CheckpointCorruptionError(
                    "candidate model metadata is invalid"
                )
            candidate_file = candidate.get("file")
            expected_candidate_file = f"candidate-model-{expected_stem}.pt"
            if (
                candidate.get("schemaVersion")
                != PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION
                or candidate.get("format") != PHASE4_CANDIDATE_MODEL_FORMAT
                or not isinstance(candidate_file, str)
                or Path(candidate_file).name != candidate_file
                or candidate_file != expected_candidate_file
            ):
                raise Phase4CheckpointCorruptionError(
                    "candidate model metadata contract is invalid"
                )
            candidate_model_path = self.root / candidate_file
            if (
                not candidate_model_path.is_file()
                or candidate_model_path.is_symlink()
            ):
                raise Phase4CheckpointCorruptionError(
                    "candidate model artifact is missing or is a symlink"
                )
            candidate_model_byte_size = _require_non_negative_int(
                candidate.get("bytes"), "candidate model byte size"
            )
            if (
                candidate_model_byte_size <= 0
                or candidate_model_path.stat().st_size
                != candidate_model_byte_size
            ):
                raise Phase4CheckpointCorruptionError(
                    "candidate model byte size mismatch"
                )
            try:
                candidate_model_sha256 = validate_sha256(
                    candidate.get("sha256"), "candidate model sha256"
                )
                model_state_sha256 = validate_sha256(
                    candidate.get("modelStateSha256"),
                    "candidate model state sha256",
                )
            except ValueError as error:
                raise Phase4CheckpointCorruptionError(
                    "candidate model hashes are invalid"
                ) from error
            if sha256_file(candidate_model_path) != candidate_model_sha256:
                raise Phase4CheckpointCorruptionError(
                    "candidate model SHA-256 mismatch"
                )
        return Phase4CheckpointRecord(
            generation=generation,
            step=step,
            data_path=data_path,
            metadata_path=metadata_path,
            byte_size=byte_size,
            sha256=expected_digest,
            candidate_model_path=candidate_model_path,
            candidate_model_byte_size=candidate_model_byte_size,
            candidate_model_sha256=candidate_model_sha256,
            model_state_sha256=model_state_sha256,
        )

    def _validate_payload(
        self, payload: object, record: Phase4CheckpointRecord
    ) -> None:
        if not isinstance(payload, Mapping):
            raise Phase4CheckpointCorruptionError(
                "checkpoint payload must be a mapping"
            )
        if payload.get("schemaVersion") != PHASE4_CHECKPOINT_SCHEMA_VERSION:
            raise Phase4CheckpointCorruptionError(
                "unsupported checkpoint payload schema"
            )
        if payload.get("generation") != record.generation:
            raise Phase4CheckpointCorruptionError(
                "checkpoint generation metadata mismatch"
            )
        if payload.get("step") != record.step:
            raise Phase4CheckpointCorruptionError("checkpoint step metadata mismatch")
        if CheckpointIdentity.from_payload(payload.get("identity")) != self.identity:
            raise Phase4CheckpointIdentityError(
                "checkpoint payload belongs to another lineage"
            )
        for field in (
            "model",
            "optimizer",
            "trainerRuntime",
            "rng",
            "generationData",
            "replaySnapshot",
        ):
            if field not in payload:
                raise Phase4CheckpointCorruptionError(
                    f"checkpoint payload is missing {field}"
                )
        for field in (
            "model",
            "optimizer",
            "trainerRuntime",
            "rng",
            "generationData",
            "replaySnapshot",
        ):
            if not isinstance(payload[field], Mapping):
                raise Phase4CheckpointCorruptionError(
                    f"checkpoint {field} must be a mapping"
                )
        self._validate_generation_contract(
            record.generation,
            record.step,
            payload["generationData"],
            payload["replaySnapshot"],
        )
        if record.generation == -1:
            optimizer = payload["optimizer"]
            trainer_runtime = payload["trainerRuntime"]
            if (
                optimizer.get("state") != {}
                or trainer_runtime.get("step") != 0
                or record.step != 0
            ):
                raise Phase4CheckpointCorruptionError(
                    "initialization checkpoint must contain an empty optimizer "
                    "and zero-step trainer runtime"
                )

    @staticmethod
    def _validate_generation_contract(
        generation: int,
        step: int,
        generation_data: Mapping[str, object],
        replay_snapshot: Mapping[str, object],
    ) -> None:
        game_ids = replay_snapshot.get("gameIds")
        replay_bytes = replay_snapshot.get("bytes")
        manifest_bytes = replay_snapshot.get("manifestBytes")
        committed_bytes = replay_snapshot.get("committedBytes")
        if (
            not isinstance(game_ids, list)
            or any(not isinstance(game_id, str) or not game_id for game_id in game_ids)
            or replay_snapshot.get("games") != len(game_ids)
            or type(replay_snapshot.get("plies")) is not int
            or replay_snapshot.get("plies") < 0
            or type(replay_bytes) is not int
            or replay_bytes < 0
            or type(manifest_bytes) is not int
            or manifest_bytes <= 0
            or type(committed_bytes) is not int
            or committed_bytes != replay_bytes + manifest_bytes
            or replay_snapshot.get("deepVerified") is not True
        ):
            raise Phase4CheckpointCorruptionError(
                "checkpoint replay snapshot totals are invalid"
            )
        try:
            validate_sha256(
                replay_snapshot.get("manifestSha256"),
                "replay manifest sha256",
            )
        except ValueError as error:
            raise Phase4CheckpointCorruptionError(
                "checkpoint replay manifest hash is invalid"
            ) from error
        if generation == -1:
            required = {
                "schemaVersion",
                "kind",
                "nextGeneration",
                "mode",
                "seed",
                "baseModelSha256",
                "initializedModelSha256",
                "optimizerReset",
                "artifact",
            }
            if (
                step != 0
                or set(generation_data) != required
                or generation_data.get("schemaVersion") != 1
                or generation_data.get("kind") != "initialization"
                or generation_data.get("nextGeneration") != 0
                or generation_data.get("mode")
                not in {"random", "model", "champion", "recovery"}
                or type(generation_data.get("seed")) is not int
                or generation_data.get("seed") < 0
                or generation_data.get("optimizerReset") is not True
                or replay_snapshot.get("games") != 0
                or replay_snapshot.get("plies") != 0
                or replay_snapshot.get("bytes") != 0
                or replay_snapshot.get("gameIds") != []
            ):
                raise Phase4CheckpointCorruptionError(
                    "generation -1 must be an empty-replay initialization checkpoint"
                )
            try:
                validate_sha256(
                    generation_data.get("baseModelSha256"),
                    "initialization base model sha256",
                )
                validate_sha256(
                    generation_data.get("initializedModelSha256"),
                    "initialized model sha256",
                )
            except ValueError as error:
                raise Phase4CheckpointCorruptionError(
                    "initialization checkpoint model hashes are invalid"
                ) from error
            artifact = generation_data.get("artifact")
            if (generation_data.get("mode") == "random" and artifact is not None) or (
                generation_data.get("mode") in {"model", "champion", "recovery"}
                and not isinstance(artifact, Mapping)
            ):
                raise Phase4CheckpointCorruptionError(
                    "initialization checkpoint artifact metadata is invalid"
                )
            if (
                generation_data.get("mode") == "random"
                and generation_data.get("baseModelSha256")
                != generation_data.get("initializedModelSha256")
            ):
                raise Phase4CheckpointCorruptionError(
                    "cold initialization must retain the seeded base model"
                )
            if isinstance(artifact, Mapping):
                baseline = artifact.get("kind") == "phase4e-bootstrap-baseline"
                champion = artifact.get("kind") == "published-champion"
                recovery = artifact.get("kind") == "recovery-candidate"
                mode = generation_data.get("mode")
                if (
                    (mode == "champion") != champion
                    or (mode == "recovery") != recovery
                    or (mode == "model" and (champion or recovery))
                ):
                    raise Phase4CheckpointCorruptionError(
                        "initialization mode and artifact provenance disagree"
                    )
                if champion:
                    expected_artifact_fields = {
                        "kind",
                        "championId",
                        "generation",
                        "artifactPath",
                        "artifactSha256",
                        "modelStateSha256",
                        "championManifestPath",
                        "championManifestSha256",
                        "championManifestFileSha256",
                        "sourceSha256",
                        "configSha256",
                        "openingSuiteSha256",
                        "formalPromotion",
                    }
                elif baseline:
                    expected_artifact_fields = {
                        "kind",
                        "schemaVersion",
                        "baselineRoot",
                        "manifestSha256",
                        "modelSha256",
                        "modelStateSha256",
                        "formalPromotion",
                    }
                elif recovery:
                    expected_artifact_fields = {
                        "kind",
                        "generation",
                        "step",
                        "artifactPath",
                        "artifactSha256",
                        "modelStateSha256",
                        "recoveryManifestPath",
                        "recoveryManifestSha256",
                        "recoveryManifestFileSha256",
                        "sourceSha256",
                        "configSha256",
                        "runManifestSha256",
                        "checkpointSha256",
                    }
                else:
                    expected_artifact_fields = {
                        "artifactPath",
                        "artifactSha256",
                        "modelStateSha256",
                        "teacherDatasetSha256",
                        "pretrainingReportSha256",
                    }
                if (
                    set(artifact) != expected_artifact_fields
                    or artifact.get("modelStateSha256")
                    != generation_data.get("initializedModelSha256")
                    or (
                        baseline
                        and (
                            artifact.get("schemaVersion") != 1
                            or artifact.get("formalPromotion") is not False
                        )
                    )
                    or (
                        champion
                        and (
                            not isinstance(artifact.get("championId"), str)
                            or not artifact.get("championId")
                            or type(artifact.get("generation")) is not int
                            or artifact.get("generation") < 0
                            or not isinstance(artifact.get("formalPromotion"), bool)
                        )
                    )
                    or (
                        recovery
                        and (
                            type(artifact.get("generation")) is not int
                            or artifact.get("generation") < 0
                            or type(artifact.get("step")) is not int
                            or artifact.get("step") < 0
                        )
                    )
                ):
                    raise Phase4CheckpointCorruptionError(
                        "initialization provenance is incomplete"
                    )
                path_fields = (
                    ("artifactPath", "championManifestPath")
                    if champion
                    else (
                        ("baselineRoot",)
                        if baseline
                        else (
                            ("artifactPath", "recoveryManifestPath")
                            if recovery
                            else ("artifactPath",)
                        )
                    )
                )
                paths: dict[str, Path] = {}
                for field in path_fields:
                    path = artifact.get(field)
                    if (
                        not isinstance(path, str)
                        or not path
                        or Path(path).is_absolute()
                        or ".." in Path(path).parts
                    ):
                        raise Phase4CheckpointCorruptionError(
                            "initialization artifact path is unsafe"
                        )
                    paths[field] = Path(path)
                if champion and (
                    paths["artifactPath"].parent
                    != paths["championManifestPath"].parent
                    or paths["championManifestPath"].name != "manifest.json"
                ):
                    raise Phase4CheckpointCorruptionError(
                        "champion artifact and manifest paths disagree"
                    )
                try:
                    if champion:
                        digest_fields = (
                            "artifactSha256",
                            "modelStateSha256",
                            "championManifestSha256",
                            "championManifestFileSha256",
                            "sourceSha256",
                            "configSha256",
                            "openingSuiteSha256",
                        )
                    elif baseline:
                        digest_fields = (
                            "manifestSha256",
                            "modelSha256",
                            "modelStateSha256",
                        )
                    elif recovery:
                        digest_fields = (
                            "artifactSha256",
                            "modelStateSha256",
                            "recoveryManifestSha256",
                            "recoveryManifestFileSha256",
                            "sourceSha256",
                            "configSha256",
                            "runManifestSha256",
                            "checkpointSha256",
                        )
                    else:
                        digest_fields = (
                            "artifactSha256",
                            "modelStateSha256",
                            "teacherDatasetSha256",
                            "pretrainingReportSha256",
                        )
                    for field in digest_fields:
                        validate_sha256(artifact.get(field), field)
                except ValueError as error:
                    raise Phase4CheckpointCorruptionError(
                        "initialization provenance hashes are invalid"
                    ) from error
            return
        if generation_data.get("generation") != generation:
            raise Phase4CheckpointCorruptionError(
                "training checkpoint generationData does not match generation"
            )

    def _write_latest(self, record: Phase4CheckpointRecord) -> None:
        atomic_replace_bytes(
            self.root / "latest.json",
            canonical_json_bytes(
                {
                    "schemaVersion": PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION,
                    "generation": record.generation,
                    "step": record.step,
                    "metadata": record.metadata_path.name,
                    "sha256": record.sha256,
                    "candidateModelSha256": record.candidate_model_sha256,
                    "modelStateSha256": record.model_state_sha256,
                }
            ),
        )

    def quarantine_rejected(self, rejected: Sequence[str]) -> tuple[Path, ...]:
        """Move rejected committed artifacts aside so their generation can rerun."""

        if not rejected:
            return ()
        quarantine = self.root / "rejected"
        quarantine.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        token = str(time.time_ns())
        for rejection in rejected:
            metadata_name = rejection.split(":", 1)[0]
            metadata_path = self.root / metadata_name
            if (
                metadata_path.parent != self.root
                or not metadata_name.startswith("checkpoint-g")
                or not metadata_name.endswith(".json")
            ):
                raise Phase4CheckpointCorruptionError(
                    "rejected checkpoint metadata name is unsafe"
                )
            data_path = metadata_path.with_suffix(".pt")
            candidate_model_path = (
                self.root / f"candidate-model-{metadata_path.stem}.pt"
            )
            for source in (
                metadata_path,
                data_path,
                candidate_model_path,
            ):
                if not source.exists():
                    continue
                destination = quarantine / f"{source.stem}-{token}{source.suffix}"
                if destination.exists():
                    raise ImmutableArtifactExistsError(
                        f"checkpoint quarantine collision: {destination.name}"
                    )
                os.replace(source, destination)
                moved.append(destination)
        _fsync_directory(self.root)
        _fsync_directory(quarantine)
        return tuple(moved)

    def _prune(self) -> None:
        records: list[Phase4CheckpointRecord] = []
        for metadata_path in self.candidates():
            try:
                records.append(self._validate_record(metadata_path))
            except Phase4CheckpointError:
                continue
        records.sort(key=lambda item: (item.generation, item.step), reverse=True)
        for record in records[self.max_to_keep :]:
            record.metadata_path.unlink(missing_ok=True)
            record.data_path.unlink(missing_ok=True)
            if record.candidate_model_path is not None:
                record.candidate_model_path.unlink(missing_ok=True)
        _fsync_directory(self.root)

    def _prunable_bytes_after_next_save(self) -> int:
        records: list[Phase4CheckpointRecord] = []
        for metadata_path in self.candidates():
            try:
                records.append(self._validate_record(metadata_path))
            except Phase4CheckpointError:
                continue
        records.sort(key=lambda item: (item.generation, item.step))
        prune_count = max(0, len(records) + 1 - self.max_to_keep)
        return sum(
            record.byte_size
            + record.candidate_model_byte_size
            + record.metadata_path.stat().st_size
            for record in records[:prune_count]
        )

    def _remove_orphans(self) -> None:
        """Remove files that could not have reached the sidecar commit point."""

        for path in self.root.glob(".*.tmp"):
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)
        for data_path in self.root.glob("checkpoint-g*-s*.pt"):
            if data_path.is_symlink():
                continue
            if not data_path.with_suffix(".json").exists():
                data_path.unlink(missing_ok=True)
        for model_path in self.root.glob(_CANDIDATE_MODEL_GLOB):
            if model_path.is_symlink():
                continue
            stem = model_path.name.removeprefix("candidate-model-").removesuffix(
                ".pt"
            )
            if not (self.root / f"{stem}.json").exists():
                model_path.unlink(missing_ok=True)
        _fsync_directory(self.root)


__all__ = [
    "PHASE4_CHECKPOINT_SCHEMA_VERSION",
    "PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION",
    "PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION",
    "PHASE4_CANDIDATE_MODEL_FORMAT",
    "Phase4CheckpointError",
    "Phase4CheckpointCorruptionError",
    "Phase4CheckpointIdentityError",
    "canonical_model_state_sha256",
    "CheckpointIdentity",
    "Phase4CheckpointRecord",
    "Phase4CheckpointLoad",
    "Phase4CheckpointStore",
]
