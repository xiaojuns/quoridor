"""Versioned M2 model weights, deliberately separate from legacy checkpoints.

This artifact contains inference weights and compatibility metadata only. It is
not a training-resume checkpoint: optimizer, replay, RNG and lineage belong to
M4. Loading always creates a fresh CPU/eval model and never mutates a caller's
model. Float32 weights and every buffer must exactly match the declared network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .feature_spec import MODEL_VERSION, encoder_metadata
from .policy_value import NetworkConfig, PolicyValueNetwork
from .rules_contract import contract_metadata, require_compatible_artifact


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "quoridor-model-weights"
_FIELDS = {
    "schema_version", "kind", "rules_contract", "encoder", "model_version",
    "architecture", "state_dict",
}


class IncompatibleCheckpointError(ValueError):
    """Model bytes, metadata or weights do not satisfy the M2 contract."""


@dataclass(frozen=True)
class CheckpointArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    model: PolicyValueNetwork
    artifact: CheckpointArtifact
    _metadata_json: str = field(repr=False)

    @property
    def metadata(self) -> dict[str, object]:
        """Return an independent metadata copy, excluding tensor weights."""
        return json.loads(self._metadata_json)


def _json(value: object) -> str:
    def validate(item: object) -> None:
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError("metadata keys must be strings")
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise TypeError("metadata must contain only JSON primitives")

    try:
        validate(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise IncompatibleCheckpointError(f"invalid checkpoint metadata: {exc}") from exc


def _require_metadata(actual: object, expected: object, field_name: str) -> None:
    # JSON comparison is type-sensitive: True, 1 and 1.0 are different metadata.
    if _json(actual) != _json(expected):
        raise IncompatibleCheckpointError(f"incompatible checkpoint {field_name}")


def _metadata(config: NetworkConfig) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "rules_contract": contract_metadata(),
        "encoder": encoder_metadata(),
        "model_version": MODEL_VERSION,
        "architecture": config.to_dict(),
    }


def _new_cpu_model(config: NetworkConfig) -> PolicyValueNetwork:
    # Loading/validating weights must not consume the caller's random stream.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        return PolicyValueNetwork(config).to(device="cpu", dtype=torch.float32)


def _validate_weights(weights: object, config: NetworkConfig) -> None:
    if not isinstance(weights, Mapping) or any(type(key) is not str for key in weights):
        raise IncompatibleCheckpointError("state_dict must map string keys to tensors")
    if config.blocks > len(weights):
        raise IncompatibleCheckpointError("architecture has more residual blocks than state_dict entries")
    try:
        # Metadata may lie about tensor sizes: check shapes without allocating
        # those sizes in CPU memory. Each residual block has registered weights.
        with torch.device("meta"):
            model = PolicyValueNetwork(config).to(dtype=torch.float32)
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise IncompatibleCheckpointError(f"invalid architecture shapes: {exc}") from exc
    expected = model.state_dict()
    if set(weights) != set(expected):
        missing = sorted(set(expected) - set(weights))
        extra = sorted(set(weights) - set(expected))
        raise IncompatibleCheckpointError(f"state_dict keys mismatch: missing={missing}, extra={extra}")
    for name, reference in expected.items():
        value = weights[name]
        if type(value) is not torch.Tensor:
            raise IncompatibleCheckpointError(f"state_dict {name} must be a tensor")
        if value.layout != torch.strided or value.is_quantized:
            raise IncompatibleCheckpointError(f"state_dict {name} must be a dense tensor")
        if value.device.type != "cpu":
            raise IncompatibleCheckpointError(f"state_dict {name} must be a CPU tensor")
        if value.shape != reference.shape:
            raise IncompatibleCheckpointError(f"state_dict {name} shape mismatch")
        if value.dtype != reference.dtype:
            raise IncompatibleCheckpointError(f"state_dict {name} dtype mismatch")
        if not bool(torch.isfinite(value).all()):
            raise IncompatibleCheckpointError(f"state_dict {name} contains nonfinite values")
        if name.endswith(("running_var", "num_batches_tracked")) and bool((value < 0).any()):
            raise IncompatibleCheckpointError(f"state_dict {name} contains negative batchnorm state")


def save_model_checkpoint(path: str | Path, model: PolicyValueNetwork) -> CheckpointArtifact:
    """Atomically publish a new model artifact; parent directory must exist.

    Existing files and symlinks are never overwritten. Validation and an
    independent CPU snapshot precede publication. The caller owns synchronization
    with any optimizer updating ``model``; concurrent training snapshots are not
    part of this M2 API.
    """
    if type(model) is not PolicyValueNetwork:
        raise TypeError("model must be a PolicyValueNetwork")
    config = NetworkConfig.from_dict(model.config.to_dict())
    weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _validate_weights(weights, config)
    payload = {**_metadata(config), "state_dict": weights}
    target = Path(path).absolute()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # Unlike replace(), linking fails atomically if the destination exists.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return CheckpointArtifact(path=target, sha256=digest)


def load_model_checkpoint(
    path: str | Path, *, expected_sha256: str | None = None,
    expected_config: NetworkConfig | None = None,
) -> LoadedModelCheckpoint:
    """Validate exact bytes and contracts, then return a fresh CPU/eval model.

    Pass ``expected_sha256`` to bind an immutable identity supplied by a run or
    arena. Without it, the returned digest identifies the bytes actually loaded;
    computing a digest alone does not authenticate their external provenance.
    """
    if expected_sha256 is not None and (
        type(expected_sha256) is not str or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    if expected_config is not None and type(expected_config) is not NetworkConfig:
        raise TypeError("expected_config must be a NetworkConfig")
    source = Path(path).absolute()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise IncompatibleCheckpointError("checkpoint SHA-256 mismatch")
    try:
        # Hash and deserialize the same snapshot, avoiding path replacement races.
        payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    except Exception as exc:
        raise IncompatibleCheckpointError(f"cannot read weights-only checkpoint: {exc}") from exc
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise IncompatibleCheckpointError("checkpoint fields mismatch; legacy checkpoints require explicit migration")
    for key, expected in (
        ("schema_version", CHECKPOINT_SCHEMA_VERSION), ("kind", CHECKPOINT_KIND),
        ("model_version", MODEL_VERSION), ("encoder", encoder_metadata()),
        ("rules_contract", contract_metadata()),
    ):
        _require_metadata(payload[key], expected, key)
    require_compatible_artifact(payload["rules_contract"])
    try:
        config = NetworkConfig.from_dict(payload["architecture"])
    except (TypeError, ValueError) as exc:
        raise IncompatibleCheckpointError(f"incompatible checkpoint architecture: {exc}") from exc
    _require_metadata(payload["architecture"], config.to_dict(), "architecture")
    if expected_config is not None:
        _require_metadata(config.to_dict(), expected_config.to_dict(), "expected architecture")
    _validate_weights(payload["state_dict"], config)
    model = _new_cpu_model(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return LoadedModelCheckpoint(model, CheckpointArtifact(source, digest), _json(metadata))
