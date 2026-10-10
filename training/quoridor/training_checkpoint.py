"""Versioned, lineage-bound training checkpoints for the M4 player.

This module is intentionally separate from :mod:`model_checkpoint`.  The M2
artifact is weights-only and suitable for inference; an M4 checkpoint contains
the optimizer, optional scheduler, replay state and all random streams needed
to continue a run.  Loading uses PyTorch's ``weights_only`` unpickler and
therefore never executes an arbitrary pickle payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from .encoder import ENCODER_VERSION
from .feature_spec import MODEL_VERSION
from .policy_value import PolicyValueNetwork
from .rules_contract import CODEC_VERSION, RULES_VERSION, VALUE_PERSPECTIVE
from .trainer import AlphaZeroTrainer


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "quoridor-training-resume"
SEARCH_VERSION = "puct-v1"


class TrainingCheckpointError(ValueError):
    """Base error for malformed or incompatible M4 checkpoints."""


class IncompatibleTrainingCheckpointError(TrainingCheckpointError):
    """The checkpoint is validly encoded but belongs to another lineage."""


@dataclass(frozen=True)
class TrainingCheckpointArtifact:
    path: Path
    sha256: str
    step: int
    lineage: dict[str, object]
    replay_sha256: str | None


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    artifact: TrainingCheckpointArtifact
    payload: dict[str, object]


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise TrainingCheckpointError(f"metadata is not JSON-compatible: {error}") from error


def _clone_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        # Dataclass instances (notably replay samples) cannot be loaded by
        # torch's safe unpickler.  Preserve an explicit type marker and only
        # JSON/tensor fields in the checkpoint instead.
        return {
            "__quoridor_dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            **{field.name: _clone_cpu(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {key: _clone_cpu(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu(child) for child in value)
    if isinstance(value, list):
        return [_clone_cpu(child) for child in value]
    return value


def _rng_payload() -> dict[str, object]:
    # Convert numpy's ndarray to a plain list so the checkpoint remains
    # loadable by weights_only=True on every supported NumPy version.
    py_state = random.getstate()
    np_state = np.random.get_state()
    return {
        "python": [py_state[0], list(py_state[1]), py_state[2]],
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state().cpu().clone(),
        "cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def capture_rng_state() -> dict[str, object]:
    """Capture all supported random streams in a safe serializable form."""
    return _rng_payload()


def _as_tuple(value: object) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise TrainingCheckpointError("RNG state has invalid sequence type")
    return tuple(value)


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a state created by :func:`capture_rng_state` exactly."""
    if not isinstance(state, Mapping):
        raise TrainingCheckpointError("RNG state must be a mapping")
    try:
        py = _as_tuple(state["python"])
        if len(py) != 3:
            raise ValueError("python RNG state length")
        random.setstate((py[0], _as_tuple(py[1]), py[2]))
        np_state = _as_tuple(state["numpy"])
        if len(np_state) != 5:
            raise ValueError("numpy RNG state length")
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), int(np_state[2]), int(np_state[3]), float(np_state[4])))
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            raise TypeError("torch RNG state must be a tensor")
        torch.set_rng_state(torch_state.detach().cpu())
        cuda = state.get("cuda", [])
        if torch.cuda.is_available() and cuda:
            if not isinstance(cuda, (list, tuple)) or not all(isinstance(item, torch.Tensor) for item in cuda):
                raise TypeError("CUDA RNG state must be a tensor list")
            torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingCheckpointError(f"invalid RNG state: {error}") from error


def _hash_value(value: object, digest: hashlib._Hash | None = None) -> str:
    """Stable digest for replay state, including tensor bytes and metadata."""
    digest = digest or hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(_json(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            digest.update(_json(str(key)).encode())
            _hash_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(item, digest)
    else:
        digest.update(type(value).__name__.encode() + b"\0" + _json(value).encode())
    return digest.hexdigest()


def _validate_finite_tensors(value: object, label: str) -> None:
    """Reject non-finite checkpoint tensors before touching destination state."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or not bool(torch.isfinite(value).all().item()):
            raise TrainingCheckpointError(f"{label} contains a non-finite or sparse tensor")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tensors(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_tensors(child, f"{label}[{index}]")


def replay_state_and_hash(replay: object | None) -> tuple[object | None, str | None]:
    """Snapshot a replay object and derive an immutable lineage digest."""
    if replay is None:
        return None, None
    state = replay.state_dict() if hasattr(replay, "state_dict") else replay
    state = _clone_cpu(state)
    if not isinstance(state, Mapping):
        raise TypeError("replay state must be a mapping")
    return state, _hash_value(state)


def _restore_replay_dataclasses(value: object) -> object:
    """Rehydrate the two safe replay sample records before buffer loading."""
    if isinstance(value, list):
        return [_restore_replay_dataclasses(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_replay_dataclasses(item) for item in value)
    if isinstance(value, Mapping):
        marker = value.get("__quoridor_dataclass__")
        fields_value = {key: _restore_replay_dataclasses(item) for key, item in value.items() if key != "__quoridor_dataclass__"}
        if marker == "quoridor.replay.ReplaySample":
            from .replay import ReplaySample
            return ReplaySample(**fields_value)
        if marker == "quoridor.replay.TrainingSample":
            from .replay import TrainingSample
            return TrainingSample(**fields_value)
        return fields_value
    return value


def _git_value(*args: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        output = subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output.stdout.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def default_lineage(*, config_sha256: str | None = None, code_revision: str | None = None, code_dirty: bool | None = None) -> dict[str, object]:
    """Construct the mandatory M4 identity fields for a fresh run."""
    root = Path(__file__).resolve().parents[2]
    lock_hash = hashlib.sha256((root / "training/uv.lock").read_bytes()).hexdigest()
    if code_revision is None:
        code_revision = _git_value("rev-parse", "HEAD")
    if code_dirty is None:
        status = _git_value("status", "--porcelain")
        code_dirty = bool(status)
    return {
        "config_sha256": config_sha256,
        "lock_sha256": lock_hash,
        "rules_version": RULES_VERSION,
        "codec_version": CODEC_VERSION,
        "value_perspective": VALUE_PERSPECTIVE,
        "encoder_version": ENCODER_VERSION,
        "model_version": MODEL_VERSION,
        "search_version": SEARCH_VERSION,
        "code_revision": code_revision,
        "code_dirty": bool(code_dirty),
    }


def _normalize_lineage(lineage: Mapping[str, object] | None) -> dict[str, object]:
    if lineage is None:
        return default_lineage()
    if not isinstance(lineage, Mapping):
        raise TypeError("lineage must be a mapping")
    # JSON roundtrip validates primitive types and gives an independent copy.
    try:
        value = json.loads(_json(dict(lineage)))
    except TrainingCheckpointError:
        raise
    required = {
        "lock_sha256", "rules_version", "codec_version",
        "value_perspective", "encoder_version", "model_version", "search_version",
        "code_revision", "code_dirty",
    }
    if not required.issubset(value):
        raise TrainingCheckpointError(f"lineage is missing fields: {sorted(required - set(value))}")
    if value["rules_version"] != RULES_VERSION or value["codec_version"] != CODEC_VERSION or value["value_perspective"] != VALUE_PERSPECTIVE:
        raise IncompatibleTrainingCheckpointError("lineage rules contract is incompatible")
    if value["encoder_version"] != ENCODER_VERSION or value["model_version"] != MODEL_VERSION or value["search_version"] != SEARCH_VERSION:
        raise IncompatibleTrainingCheckpointError("lineage model/search version is incompatible")
    if not isinstance(value["lock_sha256"], str) or len(value["lock_sha256"]) != 64 or any(character not in "0123456789abcdef" for character in value["lock_sha256"]):
        raise TrainingCheckpointError("lineage lock_sha256 must be a SHA-256 digest")
    config_hash = value.get("config_sha256")
    if config_hash is not None and (not isinstance(config_hash, str) or len(config_hash) != 64 or any(character not in "0123456789abcdef" for character in config_hash)):
        raise TrainingCheckpointError("lineage config_sha256 must be a SHA-256 digest or null")
    if value["code_revision"] is not None and not isinstance(value["code_revision"], str):
        raise TrainingCheckpointError("lineage code_revision must be a string or null")
    if not isinstance(value["code_dirty"], bool):
        raise TrainingCheckpointError("lineage code_dirty must be boolean")
    return value


def _validate_model(trainer: AlphaZeroTrainer) -> None:
    if not isinstance(trainer, AlphaZeroTrainer):
        raise TypeError("trainer must be an AlphaZeroTrainer")
    if not isinstance(trainer.model, PolicyValueNetwork):
        raise TypeError("training checkpoints require PolicyValueNetwork")


def save_training_checkpoint(
    path: str | Path,
    trainer: AlphaZeroTrainer,
    *,
    replay: object | None = None,
    lineage: Mapping[str, object] | None = None,
    config_sha256: str | None = None,
    step: int | None = None,
    counters: Mapping[str, object] | None = None,
) -> TrainingCheckpointArtifact:
    """Atomically save model, optimizer, runtime, replay, RNG and lineage."""
    _validate_model(trainer)
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("step must be a non-negative integer")
    actual_step = trainer.step if step is None else step
    normalized_lineage = _normalize_lineage(lineage)
    if config_sha256 is not None:
        if not isinstance(config_sha256, str) or len(config_sha256) != 64 or any(character not in "0123456789abcdef" for character in config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        if normalized_lineage.get("config_sha256") not in (None, config_sha256):
            raise IncompatibleTrainingCheckpointError("config hash disagrees with lineage")
        normalized_lineage["config_sha256"] = config_sha256
    if normalized_lineage.get("config_sha256") is None:
        raise TrainingCheckpointError("training checkpoints require a config_sha256 lineage field")
    replay_state, replay_hash = replay_state_and_hash(replay)
    if replay_hash is not None:
        normalized_lineage["replay_sha256"] = replay_hash
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "step": actual_step,
        "model_version": MODEL_VERSION,
        "architecture": trainer.model.config.to_dict(),
        "trainer_config": trainer.config,
        "model": _clone_cpu(trainer.model.state_dict()),
        "optimizer": _clone_cpu(trainer.optimizer.state_dict()),
        "runtime": _clone_cpu(trainer.runtime_state_dict()),
        "rng": _rng_payload(),
        "replay": replay_state,
        "replay_sha256": replay_hash,
        "lineage": normalized_lineage,
        "counters": dict(counters or {}),
    }
    try:
        _json(payload["lineage"])
        _json(payload["counters"])
    except TrainingCheckpointError:
        raise
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        raw = temporary.read_bytes()
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return TrainingCheckpointArtifact(target, hashlib.sha256(raw).hexdigest(), actual_step, normalized_lineage, replay_hash)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    if _json(actual) != _json(expected):
        raise IncompatibleTrainingCheckpointError(f"checkpoint {label} mismatch")


def load_training_checkpoint(
    path: str | Path,
    trainer: AlphaZeroTrainer,
    *,
    replay: object | None = None,
    expected_lineage: Mapping[str, object] | None = None,
    expected_config_sha256: str | None = None,
    expected_replay_sha256: str | None = None,
    restore_rng: bool = True,
) -> LoadedTrainingCheckpoint:
    """Validate and restore a checkpoint into an existing trainer/replay."""
    _validate_model(trainer)
    source = Path(path).absolute()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TrainingCheckpointError(f"cannot read weights-only checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise TrainingCheckpointError("checkpoint payload must be a mapping")
    expected_fields = {"schema_version", "kind", "step", "model_version", "architecture", "trainer_config", "model", "optimizer", "runtime", "rng", "replay", "replay_sha256", "lineage", "counters"}
    if set(payload) != expected_fields:
        raise TrainingCheckpointError("checkpoint fields mismatch; legacy checkpoints require explicit migration")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION or payload["kind"] != CHECKPOINT_KIND:
        raise TrainingCheckpointError("unsupported training checkpoint schema")
    if payload["model_version"] != MODEL_VERSION:
        raise IncompatibleTrainingCheckpointError("model version mismatch")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise TrainingCheckpointError("checkpoint step must be non-negative")
    _validate_finite_tensors(payload["model"], "model")
    _validate_finite_tensors(payload["optimizer"], "optimizer")
    _expect_equal(payload["architecture"], trainer.model.config.to_dict(), "architecture")
    _expect_equal(payload["trainer_config"], trainer.config, "trainer configuration")
    stored_lineage = _normalize_lineage(payload["lineage"])
    if stored_lineage.get("config_sha256") is None:
        raise TrainingCheckpointError("checkpoint is missing config_sha256 lineage")
    current_lock_hash = default_lineage()["lock_sha256"]
    if stored_lineage.get("lock_sha256") != current_lock_hash:
        raise IncompatibleTrainingCheckpointError("dependency lock hash mismatch")
    if expected_config_sha256 is not None and (not isinstance(expected_config_sha256, str) or len(expected_config_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_config_sha256)):
        raise ValueError("expected_config_sha256 must be a lowercase SHA-256 digest")
    if expected_config_sha256 is not None and stored_lineage.get("config_sha256") != expected_config_sha256:
        raise IncompatibleTrainingCheckpointError("configuration hash mismatch")
    if expected_lineage is not None:
        if not isinstance(expected_lineage, Mapping):
            raise TypeError("expected_lineage must be a mapping")
        for key, expected in expected_lineage.items():
            if key not in stored_lineage or _json(stored_lineage[key]) != _json(expected):
                raise IncompatibleTrainingCheckpointError(f"lineage mismatch for {key}")
    replay_state = payload["replay"]
    stored_replay_hash = payload["replay_sha256"]
    if expected_replay_sha256 is not None and (not isinstance(expected_replay_sha256, str) or len(expected_replay_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_replay_sha256)):
        raise ValueError("expected_replay_sha256 must be a lowercase SHA-256 digest")
    if expected_replay_sha256 is not None and stored_replay_hash != expected_replay_sha256:
        raise IncompatibleTrainingCheckpointError("replay lineage hash mismatch")
    if replay_state is not None and _hash_value(replay_state) != stored_replay_hash:
        raise TrainingCheckpointError("replay payload hash mismatch")
    if stored_lineage.get("replay_sha256") not in (None, stored_replay_hash):
        raise TrainingCheckpointError("replay lineage hash disagrees with payload")
    if replay is not None and replay_state is None:
        raise IncompatibleTrainingCheckpointError("checkpoint has no replay for requested restore")
    # Restore as one fail-closed transaction. A malformed optimizer or replay
    # payload must not leave the caller with half of an unrelated run loaded.
    replay_before = replay.state_dict() if replay is not None else None
    model_before = _clone_cpu(trainer.model.state_dict())
    optimizer_before = _clone_cpu(trainer.optimizer.state_dict())
    runtime_before = _clone_cpu(trainer.runtime_state_dict())
    step_before = trainer.step
    rng_before = _rng_payload()
    try:
        if replay is not None:
            if not hasattr(replay, "load_state_dict"):
                raise TypeError("replay must provide load_state_dict")
            replay.load_state_dict(_restore_replay_dataclasses(replay_state))
        trainer.model.load_state_dict(payload["model"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.load_runtime_state_dict(payload["runtime"])
        trainer.step = int(payload["step"])
        if restore_rng:
            restore_rng_state(payload["rng"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        try:
            if replay is not None and replay_before is not None:
                replay.load_state_dict(replay_before)
            trainer.model.load_state_dict(model_before, strict=True)
            trainer.optimizer.load_state_dict(optimizer_before)
            trainer.load_runtime_state_dict(runtime_before)
            trainer.step = step_before
            restore_rng_state(rng_before)
        except Exception as rollback_error:
            raise TrainingCheckpointError(f"checkpoint restore failed and rollback failed: {rollback_error}") from error
        raise IncompatibleTrainingCheckpointError(f"trainer/replay state mismatch: {error}") from error
    artifact = TrainingCheckpointArtifact(source, digest, int(payload["step"]), stored_lineage, stored_replay_hash)
    return LoadedTrainingCheckpoint(artifact, payload)


__all__ = [
    "CHECKPOINT_KIND", "CHECKPOINT_SCHEMA_VERSION", "SEARCH_VERSION", "IncompatibleTrainingCheckpointError",
    "LoadedTrainingCheckpoint", "TrainingCheckpointArtifact", "TrainingCheckpointError",
    "capture_rng_state", "default_lineage", "load_training_checkpoint", "replay_state_and_hash",
    "restore_rng_state", "save_training_checkpoint",
]
