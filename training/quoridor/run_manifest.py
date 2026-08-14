"""Immutable run provenance, generation reports, and a single-writer lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


RUN_MANIFEST_SCHEMA_VERSION = 1
GENERATION_REPORT_SCHEMA_VERSION = 1
SOURCE_HASH_ALGORITHM = "sha256-path-content-v1"
DEFAULT_ARTIFACT_LIMIT_BYTES = 8 * 1024**3
MAXIMUM_ARTIFACT_LIMIT_BYTES = 12 * 1024**3
DEFAULT_MINIMUM_FREE_DISK_BYTES = 20 * 1024**3

_IGNORED_SOURCE_PARTS = frozenset(
    {".git", "__pycache__", "build", "runs", ".venv"}
)
_IGNORED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo", ".tmp"})
_DEFAULT_SOURCE_PATHS = (
    "training/quoridor",
    "training/pyproject.toml",
    "native/include",
    "native/src",
    "native/CMakeLists.txt",
    "src/game.js",
    "src/ai.js",
    "src/alphazero",
    "scripts",
)


class RunManifestError(RuntimeError):
    """Base class for run provenance failures."""


class ManifestCorruptionError(RunManifestError):
    """Raised when a stored manifest/report fails validation."""


class ImmutableArtifactExistsError(RunManifestError):
    """Raised when an immutable manifest/report path already exists."""


class RunAlreadyLockedError(RunManifestError):
    """Raised when another writer already holds the run lock."""


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Recorded local resource limits for a training run."""

    artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES
    gpu_hours: float | None = None
    maximum_generations: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.artifact_limit_bytes, "artifact_limit_bytes"),
            (self.minimum_free_disk_bytes, "minimum_free_disk_bytes"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.artifact_limit_bytes > MAXIMUM_ARTIFACT_LIMIT_BYTES:
            raise ValueError(
                "artifact_limit_bytes must not exceed the Phase 4 12 GiB cap"
            )
        if self.minimum_free_disk_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
            raise ValueError(
                "minimum_free_disk_bytes must preserve the Phase 4 20 GiB stop line"
            )
        if self.gpu_hours is not None:
            if isinstance(self.gpu_hours, bool) or not isinstance(
                self.gpu_hours, (int, float)
            ):
                raise TypeError("gpu_hours must be numeric or None")
            gpu_hours = float(self.gpu_hours)
            if not math.isfinite(gpu_hours) or gpu_hours <= 0:
                raise ValueError("gpu_hours must be finite and positive")
            object.__setattr__(self, "gpu_hours", gpu_hours)
        if self.maximum_generations is not None and (
            type(self.maximum_generations) is not int
            or self.maximum_generations <= 0
        ):
            raise ValueError("maximum_generations must be a positive integer or None")
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(self.metadata, "resource budget metadata"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifactLimitBytes": self.artifact_limit_bytes,
            "minimumFreeDiskBytes": self.minimum_free_disk_bytes,
            "gpuHours": self.gpu_hours,
            "maximumGenerations": self.maximum_generations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SourceFileDigest:
    path: str
    byte_size: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    sha256: str
    files: tuple[SourceFileDigest, ...]
    byte_size: int

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithm": SOURCE_HASH_ALGORITHM,
            "sha256": self.sha256,
            "fileCount": len(self.files),
            "bytes": self.byte_size,
            "files": [item.to_payload() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class ConfigFingerprint:
    sha256: str
    payload: object

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithm": "sha256-canonical-json-v1",
            "sha256": self.sha256,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class RunManifestRecord:
    payload: Mapping[str, object]
    sha256: str


@dataclass(frozen=True, slots=True)
class GenerationReportRecord:
    generation: int
    path: Path
    sha256: str
    payload: Mapping[str, object]


def normalize_json_value(value: object, label: str = "value") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} mapping keys must be strings")
            normalized[key] = normalize_json_value(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            normalize_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} must contain only JSON-compatible values")


def normalize_json_mapping(
    value: Mapping[str, object], label: str = "mapping"
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = normalize_json_value(value, label)
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a mapping")
    return normalized


def canonical_json_bytes(payload: object) -> bytes:
    normalized = normalize_json_value(payload, "payload")
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def validate_sha256(value: object, label: str = "sha256") -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


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


def atomic_replace_bytes(destination: str | Path, payload: bytes) -> Path:
    """Atomically create or replace a mutable index such as a store manifest."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_create_bytes(destination: str | Path, payload: bytes) -> Path:
    """Atomically create an immutable file without a check-then-write race."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise ImmutableArtifactExistsError(
                f"immutable artifact already exists: {path}"
            ) from error
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(path.parent)
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _is_ignored_source(path: Path) -> bool:
    return bool(_IGNORED_SOURCE_PARTS.intersection(path.parts)) or (
        path.suffix in _IGNORED_SOURCE_SUFFIXES
    )


def _source_files(
    workspace_root: Path, paths: Iterable[str | Path]
) -> tuple[Path, ...]:
    workspace = workspace_root.resolve()
    files: set[Path] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if candidate.is_symlink():
            raise ValueError(f"source paths must not be symlinks: {candidate}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as error:
            raise ValueError(f"source path is outside workspace: {candidate}") from error
        candidates = (resolved,) if resolved.is_file() else resolved.rglob("*")
        for item in candidates:
            if item.is_symlink():
                raise ValueError(f"source trees must not contain symlinks: {item}")
            if item.is_file():
                relative = item.relative_to(workspace)
                if not _is_ignored_source(relative):
                    files.add(item)
    if not files:
        raise ValueError("source manifest must include at least one file")
    return tuple(sorted(files, key=lambda path: path.relative_to(workspace).as_posix()))


def fingerprint_sources(
    workspace_root: str | Path,
    paths: Iterable[str | Path] | None = None,
) -> SourceFingerprint:
    """Hash source path names and bytes without relying on Git metadata."""

    workspace = Path(workspace_root).resolve()
    selected_paths: Sequence[str | Path] = (
        _DEFAULT_SOURCE_PATHS if paths is None else tuple(paths)
    )
    files = _source_files(workspace, selected_paths)
    aggregate = hashlib.sha256()
    aggregate.update(b"quoridor-source-manifest-v1\x00")
    items: list[SourceFileDigest] = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(workspace).as_posix()
        content = path.read_bytes()
        path_bytes = relative.encode("utf-8")
        aggregate.update(len(path_bytes).to_bytes(8, "big"))
        aggregate.update(path_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        items.append(
            SourceFileDigest(
                path=relative,
                byte_size=len(content),
                sha256=sha256_bytes(content),
            )
        )
        total_bytes += len(content)
    return SourceFingerprint(
        sha256=aggregate.hexdigest(),
        files=tuple(items),
        byte_size=total_bytes,
    )


def fingerprint_config(config: object) -> ConfigFingerprint:
    """Hash semantic canonical JSON, independent of key order or whitespace."""

    if isinstance(config, (str, Path)):
        source = Path(config)
        try:
            config = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load config {source}: {error}") from error
    elif is_dataclass(config) and not isinstance(config, type):
        payload_builder = getattr(config, "to_fingerprint_payload", None)
        config = payload_builder() if callable(payload_builder) else asdict(config)
    normalized = normalize_json_value(config, "config")
    encoded = canonical_json_bytes(normalized)
    return ConfigFingerprint(sha256=sha256_bytes(encoded), payload=normalized)


def collect_runtime_environment() -> dict[str, object]:
    """Collect local runtime facts without requiring Git or network access."""

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, object]] = []
    if cuda_available:
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                device: dict[str, object] = {
                    "index": index,
                    "name": properties.name,
                    "totalMemoryBytes": properties.total_memory,
                    "computeCapability": [properties.major, properties.minor],
                }
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    device["freeMemoryBytesAtStart"] = free_bytes
                    device["visibleMemoryBytes"] = total_bytes
                except (RuntimeError, TypeError):
                    pass
                devices.append(device)
        except RuntimeError:
            cuda_available = False
            devices = []
    cudnn_version = torch.backends.cudnn.version() if cuda_available else None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "compiledCudaVersion": torch.version.cuda,
            "cudaAvailable": cuda_available,
            "cudnnVersion": cudnn_version,
            "devices": devices,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpuCount": os.cpu_count(),
        },
    }


def _with_self_hash(payload: Mapping[str, object], field_name: str) -> dict[str, object]:
    core = normalize_json_mapping(payload, "self-hashed payload")
    digest = sha256_bytes(canonical_json_bytes(core))
    return {**core, field_name: digest}


def _verify_self_hash(
    payload: Mapping[str, object], field_name: str, label: str
) -> str:
    stored = validate_sha256(payload.get(field_name), f"{label} {field_name}")
    core = dict(payload)
    del core[field_name]
    expected = sha256_bytes(canonical_json_bytes(core))
    if stored != expected:
        raise ManifestCorruptionError(f"{label} self hash mismatch")
    return stored


def build_run_manifest(
    *,
    workspace_root: str | Path,
    config: object,
    budget: ResourceBudget,
    run_id: str,
    source_paths: Iterable[str | Path] | None = None,
    environment: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    created_at: str | None = None,
) -> RunManifestRecord:
    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        raise ValueError("run_id must be a non-empty string of at most 128 characters")
    if not isinstance(budget, ResourceBudget):
        raise TypeError("budget must be a ResourceBudget")
    if created_at is not None and (not isinstance(created_at, str) or not created_at):
        raise ValueError("created_at must be a non-empty string or None")
    workspace = Path(workspace_root).resolve()
    source = fingerprint_sources(workspace, source_paths)
    config_fingerprint = fingerprint_config(config)
    runtime = (
        collect_runtime_environment()
        if environment is None
        else normalize_json_mapping(environment, "environment")
    )
    core = {
        "schemaVersion": RUN_MANIFEST_SCHEMA_VERSION,
        "runId": run_id,
        "createdAt": created_at or utc_timestamp(),
        "executionPolicy": {
            "scope": "local-machine-only",
            "localOnly": True,
            "networkRequired": False,
            "cloudServices": False,
        },
        "workspace": str(workspace),
        "provenance": {
            "gitMetadataUsed": False,
            "source": source.to_payload(),
            "config": config_fingerprint.to_payload(),
        },
        "environment": runtime,
        "resourceBudget": budget.to_payload(),
        "metadata": normalize_json_mapping(metadata or {}, "run metadata"),
    }
    payload = _with_self_hash(core, "manifestSha256")
    return RunManifestRecord(payload=payload, sha256=payload["manifestSha256"])


def save_run_manifest(record: RunManifestRecord, path: str | Path) -> Path:
    if not isinstance(record, RunManifestRecord):
        raise TypeError("record must be a RunManifestRecord")
    if record.payload.get("manifestSha256") != record.sha256:
        raise ValueError("run manifest record hash does not match its payload")
    _verify_self_hash(record.payload, "manifestSha256", "run manifest")
    return atomic_create_bytes(path, canonical_json_bytes(record.payload))


def load_run_manifest(path: str | Path) -> RunManifestRecord:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestCorruptionError(f"cannot load run manifest {source}") from error
    if not isinstance(payload, Mapping):
        raise ManifestCorruptionError("run manifest must be a mapping")
    if payload.get("schemaVersion") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ManifestCorruptionError("unsupported run manifest schema version")
    execution_policy = payload.get("executionPolicy")
    if not isinstance(execution_policy, Mapping) or execution_policy.get(
        "localOnly"
    ) is not True:
        raise ManifestCorruptionError("run manifest must declare local-only execution")
    digest = _verify_self_hash(payload, "manifestSha256", "run manifest")
    return RunManifestRecord(payload=dict(payload), sha256=digest)


class GenerationReportStore:
    """Immutable, generation-addressed JSON reports for one run manifest."""

    def __init__(self, root: str | Path, *, run_manifest_sha256: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_manifest_sha256 = validate_sha256(
            run_manifest_sha256, "run_manifest_sha256"
        )

    def path_for(self, generation: int) -> Path:
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        return self.root / f"generation-{generation:08d}.json"

    def write(
        self,
        generation: int,
        report: Mapping[str, object],
        *,
        created_at: str | None = None,
    ) -> GenerationReportRecord:
        path = self.path_for(generation)
        if created_at is not None and (
            not isinstance(created_at, str) or not created_at
        ):
            raise ValueError("created_at must be a non-empty string or None")
        core = {
            "schemaVersion": GENERATION_REPORT_SCHEMA_VERSION,
            "generation": generation,
            "createdAt": created_at or utc_timestamp(),
            "runManifestSha256": self.run_manifest_sha256,
            "report": normalize_json_mapping(report, "generation report"),
        }
        payload = _with_self_hash(core, "reportSha256")
        atomic_create_bytes(path, canonical_json_bytes(payload))
        return GenerationReportRecord(
            generation=generation,
            path=path,
            sha256=payload["reportSha256"],
            payload=payload,
        )

    def load(self, generation: int) -> GenerationReportRecord:
        path = self.path_for(generation)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ManifestCorruptionError(
                f"cannot load generation report {path}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ManifestCorruptionError("generation report must be a mapping")
        if payload.get("schemaVersion") != GENERATION_REPORT_SCHEMA_VERSION:
            raise ManifestCorruptionError("unsupported generation report schema")
        if payload.get("generation") != generation:
            raise ManifestCorruptionError("generation report path/content mismatch")
        if payload.get("runManifestSha256") != self.run_manifest_sha256:
            raise ManifestCorruptionError("generation report belongs to another run")
        digest = _verify_self_hash(payload, "reportSha256", "generation report")
        return GenerationReportRecord(
            generation=generation,
            path=path,
            sha256=digest,
            payload=dict(payload),
        )

    def generations(self) -> tuple[int, ...]:
        generations: list[int] = []
        for path in self.root.glob("generation-*.json"):
            suffix = path.stem.removeprefix("generation-")
            if len(suffix) != 8 or not suffix.isdigit():
                raise ManifestCorruptionError(
                    f"unexpected generation report name: {path.name}"
                )
            generations.append(int(suffix))
        return tuple(sorted(generations))


class RunLock:
    """Advisory Linux ``flock`` protecting one run from multiple writers."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        blocking: bool = False,
    ) -> None:
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be a non-empty string or None")
        self.path = Path(path)
        self.run_id = run_id
        if type(blocking) is not bool:
            raise TypeError("blocking must be a boolean")
        self.blocking = blocking
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> "RunLock":
        if self.acquired:
            raise RuntimeError("run lock is already acquired by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        flags = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RunAlreadyLockedError(f"run is already locked: {self.path}") from error
        try:
            metadata = canonical_json_bytes(
                {
                    "schemaVersion": 1,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "runId": self.run_id,
                    "acquiredAt": utc_timestamp(),
                }
            )
            os.ftruncate(descriptor, 0)
            view = memoryview(metadata)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("failed to write run lock metadata")
                view = view[written:]
            os.fsync(descriptor)
        except Exception:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "GENERATION_REPORT_SCHEMA_VERSION",
    "SOURCE_HASH_ALGORITHM",
    "DEFAULT_ARTIFACT_LIMIT_BYTES",
    "MAXIMUM_ARTIFACT_LIMIT_BYTES",
    "DEFAULT_MINIMUM_FREE_DISK_BYTES",
    "RunManifestError",
    "ManifestCorruptionError",
    "ImmutableArtifactExistsError",
    "RunAlreadyLockedError",
    "ResourceBudget",
    "SourceFileDigest",
    "SourceFingerprint",
    "ConfigFingerprint",
    "RunManifestRecord",
    "GenerationReportRecord",
    "normalize_json_value",
    "normalize_json_mapping",
    "canonical_json_bytes",
    "validate_sha256",
    "sha256_bytes",
    "sha256_file",
    "utc_timestamp",
    "atomic_replace_bytes",
    "atomic_create_bytes",
    "fingerprint_sources",
    "fingerprint_config",
    "collect_runtime_environment",
    "build_run_manifest",
    "save_run_manifest",
    "load_run_manifest",
    "GenerationReportStore",
    "RunLock",
]
