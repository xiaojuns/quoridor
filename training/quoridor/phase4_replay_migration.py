"""Migrate the verified Phase 4D warm replay into an empty Phase 4E store."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .codec import ACTION_SPACE_SIZE, CODEC_VERSION
from .phase4_baseline import (
    BASELINE_KIND,
    BASELINE_MANIFEST_NAME,
    DEFAULT_BASELINE_ROOT,
    Phase4BaselineError,
    verify_bootstrap_baseline,
)
from .replay_store import (
    REPLAY_SHARD_SCHEMA_VERSION,
    REPLAY_STORE_SCHEMA_VERSION,
    ReplayCorruptionError,
    ReplayShardInfo,
    ReplayStore,
)
from .run_manifest import (
    ImmutableArtifactExistsError,
    atomic_create_bytes,
    canonical_json_bytes,
    normalize_json_mapping,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)


MIGRATION_SCHEMA_VERSION = 1
MIGRATION_KIND = "phase4e-warm-replay-migration"
MIGRATION_MANIFEST_NAME = "migration-manifest.json"
DEFAULT_SOURCE_ROOT = Path(
    "training/runs/phase4/phase4d-pilot-v2/lanes/warm/replay"
)
_REPLAY_MANIFEST_NAME = "manifest.json"
_REPLAY_MANIFEST_KEYS = frozenset(
    {
        "schemaVersion",
        "shardSchemaVersion",
        "codecVersion",
        "actionSpaceSize",
        "capacity",
        "nextSequence",
        "totals",
        "shards",
    }
)
_SHARD_KEYS = frozenset(
    {
        "sequence",
        "gameId",
        "file",
        "bytes",
        "sha256",
        "plies",
        "minimumGeneration",
        "maximumGeneration",
    }
)
_MIGRATION_KEYS = frozenset(
    {
        "schemaVersion",
        "kind",
        "status",
        "createdAt",
        "formalPromotion",
        "baseline",
        "source",
        "target",
        "transfer",
        "migrationManifestSha256",
    }
)
_REPLAY_SNAPSHOT_KEYS = frozenset(
    {
        "manifestSha256",
        "games",
        "plies",
        "bytes",
        "manifestBytes",
        "committedBytes",
        "deepVerified",
        "gameIds",
        "prefixSha256",
    }
)
_TRANSFER_MODES = frozenset({"copy", "hardlink", "auto"})


class Phase4ReplayMigrationError(RuntimeError):
    """Base class for warm replay migration failures."""


class Phase4ReplayMigrationLineageError(Phase4ReplayMigrationError):
    """Raised when baseline, checkpoint, and source replay do not bind."""


class Phase4ReplayMigrationConfigurationError(Phase4ReplayMigrationError):
    """Raised for an unsafe target or insufficient target capacity."""


class Phase4ReplayMigrationCorruptionError(Phase4ReplayMigrationError):
    """Raised when an existing migration or replay file was altered."""


@dataclass(frozen=True, slots=True)
class ReplayCapacity:
    games: int
    plies: int
    bytes: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.games, "games"),
            (self.plies, "plies"),
            (self.bytes, "bytes"),
        ):
            if type(value) is not int or value <= 0:
                raise Phase4ReplayMigrationConfigurationError(
                    f"target {label} capacity must be a positive integer"
                )

    def to_payload(self) -> dict[str, int]:
        return {
            "games": self.games,
            "plies": self.plies,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class _ReadOnlyStore:
    root: Path
    manifest_path: Path
    manifest_payload: Mapping[str, object]
    manifest_bytes: bytes
    manifest_sha256: str
    capacity: ReplayCapacity
    next_sequence: int
    entries: tuple[ReplayShardInfo, ...]
    committed_bytes: int
    prefix_sha256: str

    @property
    def game_count(self) -> int:
        return len(self.entries)

    @property
    def ply_count(self) -> int:
        return sum(entry.ply_count for entry in self.entries)

    @property
    def shard_bytes(self) -> int:
        return sum(entry.byte_size for entry in self.entries)

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(entry.game_id for entry in self.entries)


@dataclass(frozen=True, slots=True)
class _BaselineBinding:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    model_state_sha256: str
    pilot_root: Path
    result_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    replay_snapshot: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    baseline: _BaselineBinding
    store: _ReadOnlyStore
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayMigrationVerification:
    manifest_path: Path
    manifest_sha256: str
    source_root: Path
    target_root: Path
    game_count: int
    ply_count: int
    prefix_sha256: str
    capacity: ReplayCapacity
    hard_linked: int
    copied: int
    lineage_verified: bool

    def to_payload(self, workspace: Path) -> dict[str, object]:
        return {
            "schemaVersion": MIGRATION_SCHEMA_VERSION,
            "kind": MIGRATION_KIND,
            "status": "completed",
            "formalPromotion": False,
            "manifestPath": _relative(self.manifest_path, workspace),
            "manifestSha256": self.manifest_sha256,
            "sourceRoot": _relative(self.source_root, workspace),
            "targetRoot": _relative(self.target_root, workspace),
            "games": self.game_count,
            "plies": self.ply_count,
            "prefixSha256": self.prefix_sha256,
            "capacity": self.capacity.to_payload(),
            "hardLinked": self.hard_linked,
            "copied": self.copied,
            "lineageVerified": self.lineage_verified,
        }


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Phase4ReplayMigrationCorruptionError(
            f"{label} must be a positive integer"
        )
    return value


def _non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase4ReplayMigrationCorruptionError(
            f"{label} must be a non-negative integer"
        )
    return value


def _digest(
    value: object,
    label: str,
    error_type: type[Phase4ReplayMigrationError],
) -> str:
    try:
        return validate_sha256(value, label)
    except ValueError as error:
        raise error_type(str(error)) from error


def _load_json(
    path: Path,
    label: str,
    error_type: type[Phase4ReplayMigrationError],
) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise error_type(f"{label} is missing or is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise error_type(f"{label} must contain a JSON object")
    try:
        normalized = normalize_json_mapping(payload, label)
    except (TypeError, ValueError) as error:
        raise error_type(f"{label} is not canonical JSON data") from error
    return normalized, raw


def _verify_self_hash(
    payload: Mapping[str, object],
    field: str,
    label: str,
    error_type: type[Phase4ReplayMigrationError],
) -> str:
    digest = _digest(payload.get(field), f"{label}.{field}", error_type)
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise error_type(f"{label} self hash mismatch")
    return digest


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4ReplayMigrationConfigurationError(
            f"path is outside workspace: {path}"
        ) from error


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
        raise Phase4ReplayMigrationConfigurationError(
            f"{label} must not be a symlink"
        )
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise Phase4ReplayMigrationConfigurationError(
            f"{label} must be inside the workspace"
        ) from error
    if must_exist and (not resolved.is_dir() or resolved.is_symlink()):
        raise Phase4ReplayMigrationConfigurationError(
            f"{label} does not name a directory"
        )
    return resolved


def _workspace_file(
    workspace: Path,
    value: object,
    label: str,
    error_type: type[Phase4ReplayMigrationError],
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


def _parse_shard(value: object) -> ReplayShardInfo:
    if not isinstance(value, Mapping) or set(value) != _SHARD_KEYS:
        raise Phase4ReplayMigrationCorruptionError(
            "replay shard manifest entry fields are invalid"
        )
    file_name = value.get("file")
    game_id = value.get("gameId")
    if (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not file_name.startswith("game-")
        or not file_name.endswith(".qrp")
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "replay shard file name is unsafe"
        )
    if not isinstance(game_id, str) or not game_id:
        raise Phase4ReplayMigrationCorruptionError(
            "replay shard game ID is invalid"
        )
    minimum_generation = _non_negative_int(
        value.get("minimumGeneration"), "minimum generation"
    )
    maximum_generation = _non_negative_int(
        value.get("maximumGeneration"), "maximum generation"
    )
    if maximum_generation < minimum_generation:
        raise Phase4ReplayMigrationCorruptionError(
            "replay shard generation range is invalid"
        )
    return ReplayShardInfo(
        sequence=_non_negative_int(value.get("sequence"), "shard sequence"),
        game_id=game_id,
        file_name=file_name,
        byte_size=_positive_int(value.get("bytes"), "shard bytes"),
        sha256=_digest(
            value.get("sha256"),
            "shard sha256",
            Phase4ReplayMigrationCorruptionError,
        ),
        ply_count=_positive_int(value.get("plies"), "shard plies"),
        minimum_generation=minimum_generation,
        maximum_generation=maximum_generation,
    )


def _shard_payload(entry: ReplayShardInfo) -> dict[str, object]:
    return {
        "sequence": entry.sequence,
        "gameId": entry.game_id,
        "file": entry.file_name,
        "bytes": entry.byte_size,
        "sha256": entry.sha256,
        "plies": entry.ply_count,
        "minimumGeneration": entry.minimum_generation,
        "maximumGeneration": entry.maximum_generation,
    }


def _prefix_sha256(entries: Sequence[ReplayShardInfo]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "shards": [_shard_payload(entry) for entry in entries],
            }
        )
    )


def _load_store_readonly(
    root: Path,
    *,
    expected_capacity: ReplayCapacity | None = None,
    allowed_extra_names: frozenset[str] | set[str] = frozenset(),
    deep: bool = True,
) -> _ReadOnlyStore:
    if type(deep) is not bool:
        raise Phase4ReplayMigrationConfigurationError(
            "deep replay verification flag must be a boolean"
        )
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        for name in allowed_extra_names
    ):
        raise Phase4ReplayMigrationConfigurationError(
            "allowed replay directory names are unsafe"
        )
    manifest_path = root / _REPLAY_MANIFEST_NAME
    payload, raw = _load_json(
        manifest_path,
        "replay manifest",
        Phase4ReplayMigrationCorruptionError,
    )
    if set(payload) != _REPLAY_MANIFEST_KEYS:
        raise Phase4ReplayMigrationCorruptionError(
            "replay manifest schema fields are invalid"
        )
    if canonical_json_bytes(payload) != raw:
        raise Phase4ReplayMigrationCorruptionError(
            "replay manifest is not canonical JSON"
        )
    if (
        payload.get("schemaVersion") != REPLAY_STORE_SCHEMA_VERSION
        or payload.get("shardSchemaVersion") != REPLAY_SHARD_SCHEMA_VERSION
        or payload.get("codecVersion") != CODEC_VERSION
        or payload.get("actionSpaceSize") != ACTION_SPACE_SIZE
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "replay manifest codec/schema is incompatible"
        )
    capacity_payload = payload.get("capacity")
    if not isinstance(capacity_payload, Mapping) or set(capacity_payload) != {
        "games",
        "plies",
        "bytes",
    }:
        raise Phase4ReplayMigrationCorruptionError(
            "replay capacity record is invalid"
        )
    capacity = ReplayCapacity(
        games=_positive_int(capacity_payload.get("games"), "capacity.games"),
        plies=_positive_int(capacity_payload.get("plies"), "capacity.plies"),
        bytes=_positive_int(capacity_payload.get("bytes"), "capacity.bytes"),
    )
    if expected_capacity is not None and capacity != expected_capacity:
        raise Phase4ReplayMigrationConfigurationError(
            "target replay capacity differs from the requested capacity"
        )
    raw_entries = payload.get("shards")
    if not isinstance(raw_entries, list):
        raise Phase4ReplayMigrationCorruptionError(
            "replay manifest shards must be a list"
        )
    entries = tuple(_parse_shard(value) for value in raw_entries)
    sequences = tuple(entry.sequence for entry in entries)
    game_ids = tuple(entry.game_id for entry in entries)
    file_names = tuple(entry.file_name for entry in entries)
    if (
        sequences != tuple(sorted(sequences))
        or len(set(sequences)) != len(sequences)
        or len(set(game_ids)) != len(game_ids)
        or len(set(file_names)) != len(file_names)
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "replay shard order or identities are invalid"
        )
    next_sequence = _non_negative_int(
        payload.get("nextSequence"), "nextSequence"
    )
    if entries and next_sequence <= entries[-1].sequence:
        raise Phase4ReplayMigrationCorruptionError(
            "nextSequence does not follow committed shards"
        )
    totals = payload.get("totals")
    expected_totals = {
        "games": len(entries),
        "plies": sum(entry.ply_count for entry in entries),
        "shardBytes": sum(entry.byte_size for entry in entries),
    }
    if not isinstance(totals, Mapping) or dict(totals) != expected_totals:
        raise Phase4ReplayMigrationCorruptionError(
            "replay manifest totals do not match shards"
        )
    committed_bytes = len(raw) + expected_totals["shardBytes"]
    if (
        len(entries) > capacity.games
        or expected_totals["plies"] > capacity.plies
        or committed_bytes > capacity.bytes
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "replay data exceeds its declared capacity"
        )

    expected_names = {
        _REPLAY_MANIFEST_NAME,
        *file_names,
        *allowed_extra_names,
    }
    try:
        actual_names = {path.name for path in root.iterdir()}
    except OSError as error:
        raise Phase4ReplayMigrationCorruptionError(
            f"replay directory cannot be listed: {root}"
        ) from error
    if actual_names != expected_names:
        raise Phase4ReplayMigrationCorruptionError(
            "replay directory contains uncommitted or unexpected files"
        )
    for entry in entries:
        path = root / entry.file_name
        if path.is_symlink() or not path.is_file():
            raise Phase4ReplayMigrationCorruptionError(
                f"replay shard is missing or a symlink: {entry.file_name}"
            )
        try:
            matches = (
                path.stat().st_size == entry.byte_size
                and sha256_file(path) == entry.sha256
            )
        except OSError as error:
            raise Phase4ReplayMigrationCorruptionError(
                f"replay shard cannot be read: {entry.file_name}"
            ) from error
        if not matches:
            raise Phase4ReplayMigrationCorruptionError(
                f"replay shard hash/size mismatch: {entry.file_name}"
            )

    # Construct a read-only view without ReplayStore.__init__, which removes
    # temporary and uncommitted files and therefore must not touch the pilot.
    store = ReplayStore.__new__(ReplayStore)
    store.root = root
    store.max_games = capacity.games
    store.max_plies = capacity.plies
    store.max_bytes = capacity.bytes
    store.manifest_path = manifest_path
    store._entries = list(entries)
    store._next_sequence = next_sequence
    try:
        verification = store.verify(deep=deep)
    except ReplayCorruptionError as error:
        raise Phase4ReplayMigrationCorruptionError(
            f"deep replay verification failed: {error}"
        ) from error
    if (
        verification.game_count != len(entries)
        or verification.ply_count != expected_totals["plies"]
        or verification.committed_bytes != committed_bytes
        or verification.deep is not deep
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "deep replay verification totals changed"
        )
    return _ReadOnlyStore(
        root=root,
        manifest_path=manifest_path,
        manifest_payload=payload,
        manifest_bytes=raw,
        manifest_sha256=sha256_bytes(raw),
        capacity=capacity,
        next_sequence=next_sequence,
        entries=entries,
        committed_bytes=committed_bytes,
        prefix_sha256=_prefix_sha256(entries),
    )


def _load_baseline_binding(
    *,
    workspace: Path,
    baseline_root: Path,
) -> _BaselineBinding:
    try:
        verified = verify_bootstrap_baseline(
            workspace_root=workspace,
            baseline_root=baseline_root,
        )
    except Phase4BaselineError as error:
        raise Phase4ReplayMigrationLineageError(
            f"bootstrap baseline does not verify: {error}"
        ) from error
    manifest_path = baseline_root / BASELINE_MANIFEST_NAME
    manifest, _raw = _load_json(
        manifest_path,
        "bootstrap baseline manifest",
        Phase4ReplayMigrationLineageError,
    )
    manifest_sha256 = _verify_self_hash(
        manifest,
        "manifestSha256",
        "bootstrap baseline manifest",
        Phase4ReplayMigrationLineageError,
    )
    if (
        manifest.get("kind") != BASELINE_KIND
        or manifest.get("status") != "completed"
        or manifest.get("formalPromotion") is not False
        or manifest.get("championStorePublished") is not False
        or verified.get("manifestSha256") != manifest_sha256
    ):
        raise Phase4ReplayMigrationLineageError(
            "bootstrap baseline role/status is invalid"
        )
    lineage = manifest.get("lineage")
    pilot = lineage.get("pilot") if isinstance(lineage, Mapping) else None
    lanes = lineage.get("lanes") if isinstance(lineage, Mapping) else None
    selection = (
        lineage.get("selection") if isinstance(lineage, Mapping) else None
    )
    warm = lanes.get("warm") if isinstance(lanes, Mapping) else None
    checkpoint_record = (
        warm.get("checkpoint") if isinstance(warm, Mapping) else None
    )
    result_record = (
        pilot.get("resultManifest") if isinstance(pilot, Mapping) else None
    )
    if (
        not isinstance(pilot, Mapping)
        or not isinstance(selection, Mapping)
        or selection.get("lane") != "warm"
        or selection.get("generation") != 4
        or selection.get("step") != 320
        or selection.get("formalPromotion") is not False
        or not isinstance(warm, Mapping)
        or not isinstance(checkpoint_record, Mapping)
        or not isinstance(result_record, Mapping)
    ):
        raise Phase4ReplayMigrationLineageError(
            "baseline does not bind a warm generation-4 pilot checkpoint"
        )
    pilot_root = _workspace_directory(
        workspace,
        pilot.get("root"),
        "baseline pilot root",
        must_exist=True,
    )
    checkpoint_path = _workspace_file(
        workspace,
        checkpoint_record.get("path"),
        "baseline warm checkpoint",
        Phase4ReplayMigrationLineageError,
    )
    checkpoint_sha256 = _digest(
        checkpoint_record.get("sha256"),
        "baseline warm checkpoint sha256",
        Phase4ReplayMigrationLineageError,
    )
    if (
        checkpoint_record.get("sha256") != selection.get("checkpointSha256")
        or checkpoint_path
        != (pilot_root / "lanes/warm/checkpoints" / checkpoint_path.name).resolve()
        or sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise Phase4ReplayMigrationLineageError(
            "baseline warm checkpoint reference is inconsistent"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise Phase4ReplayMigrationLineageError(
            "baseline warm checkpoint cannot be loaded"
        ) from error
    snapshot = (
        checkpoint.get("replaySnapshot")
        if isinstance(checkpoint, Mapping)
        else None
    )
    if (
        not isinstance(snapshot, Mapping)
        or set(snapshot) != _REPLAY_SNAPSHOT_KEYS
        or snapshot.get("deepVerified") is not True
    ):
        raise Phase4ReplayMigrationLineageError(
            "warm checkpoint replay snapshot is incomplete"
        )
    result_manifest_sha256 = _digest(
        result_record.get("resultManifestSha256"),
        "baseline pilot result manifest sha256",
        Phase4ReplayMigrationLineageError,
    )
    model_state_sha256 = _digest(
        selection.get("modelStateSha256"),
        "baseline model state sha256",
        Phase4ReplayMigrationLineageError,
    )
    if verified.get("modelStateSha256") != model_state_sha256:
        raise Phase4ReplayMigrationLineageError(
            "baseline verification selected another model state"
        )
    return _BaselineBinding(
        root=baseline_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        model_state_sha256=model_state_sha256,
        pilot_root=pilot_root,
        result_manifest_sha256=result_manifest_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        replay_snapshot=normalize_json_mapping(
            snapshot, "warm checkpoint replay snapshot"
        ),
    )


def _load_source_binding(
    *,
    workspace: Path,
    baseline_root: Path,
    source_root: Path,
    deep: bool = True,
) -> _SourceBinding:
    baseline = _load_baseline_binding(
        workspace=workspace,
        baseline_root=baseline_root,
    )
    expected_source = (baseline.pilot_root / "lanes/warm/replay").resolve()
    if source_root != expected_source:
        raise Phase4ReplayMigrationLineageError(
            "source replay is not the verified pilot-v2 warm ReplayStore"
        )
    store = _load_store_readonly(source_root, deep=deep)
    snapshot = baseline.replay_snapshot
    expected_snapshot = {
        "manifestSha256": store.manifest_sha256,
        "games": store.game_count,
        "plies": store.ply_count,
        "bytes": store.shard_bytes,
        "manifestBytes": len(store.manifest_bytes),
        "committedBytes": store.committed_bytes,
        "deepVerified": True,
        "gameIds": list(store.game_ids),
        "prefixSha256": store.prefix_sha256,
    }
    if snapshot != expected_snapshot:
        raise Phase4ReplayMigrationLineageError(
            "warm ReplayStore differs from the baseline checkpoint snapshot"
        )
    return _SourceBinding(
        baseline=baseline,
        store=store,
        snapshot_sha256=sha256_bytes(canonical_json_bytes(snapshot)),
    )


def _capacity_covers(source: ReplayCapacity, target: ReplayCapacity) -> None:
    if (
        target.games < source.games
        or target.plies < source.plies
        or target.bytes < source.bytes
    ):
        raise Phase4ReplayMigrationConfigurationError(
            "target capacity must be at least the source ReplayStore capacity"
        )


def _target_manifest_payload(
    source: _ReadOnlyStore,
    capacity: ReplayCapacity,
) -> dict[str, object]:
    return {
        **dict(source.manifest_payload),
        "capacity": capacity.to_payload(),
    }


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


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise Phase4ReplayMigrationConfigurationError(
                f"target file appeared during migration: {destination.name}"
            ) from error
        temporary.unlink()
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _transfer_shard(
    source: Path,
    destination: Path,
    mode: str,
) -> str:
    if mode not in _TRANSFER_MODES:
        raise Phase4ReplayMigrationConfigurationError(
            "transfer mode must be copy, hardlink, or auto"
        )
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            _fsync_directory(destination.parent)
            return "hardlink"
        except FileExistsError as error:
            raise Phase4ReplayMigrationConfigurationError(
                f"target file appeared during migration: {destination.name}"
            ) from error
        except OSError as error:
            if mode == "hardlink" or error.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                getattr(errno, "EOPNOTSUPP", 95),
                getattr(errno, "ENOTSUP", 95),
            }:
                raise Phase4ReplayMigrationConfigurationError(
                    f"cannot hard-link replay shard {source.name}: {error}"
                ) from error
    _atomic_copy(source, destination)
    return "copy"


def _store_reference(
    store: _ReadOnlyStore,
    workspace: Path,
) -> dict[str, object]:
    return {
        "root": _relative(store.root, workspace),
        "replayManifest": {
            "path": _relative(store.manifest_path, workspace),
            "bytes": len(store.manifest_bytes),
            "sha256": store.manifest_sha256,
        },
        "capacity": store.capacity.to_payload(),
        "nextSequence": store.next_sequence,
        "totals": {
            "games": store.game_count,
            "plies": store.ply_count,
            "shardBytes": store.shard_bytes,
            "committedBytes": store.committed_bytes,
        },
        "prefixSha256": store.prefix_sha256,
    }


def _baseline_reference(
    binding: _BaselineBinding,
    workspace: Path,
) -> dict[str, object]:
    return {
        "root": _relative(binding.root, workspace),
        "manifestPath": _relative(binding.manifest_path, workspace),
        "manifestSha256": binding.manifest_sha256,
        "modelStateSha256": binding.model_state_sha256,
        "pilotResultManifestSha256": binding.result_manifest_sha256,
        "checkpointPath": _relative(binding.checkpoint_path, workspace),
        "checkpointSha256": binding.checkpoint_sha256,
    }


def _migration_payload(
    *,
    workspace: Path,
    source: _SourceBinding,
    target: _ReadOnlyStore,
    transfer_mode: str,
    hard_linked: int,
    copied: int,
) -> dict[str, object]:
    core = {
        "schemaVersion": MIGRATION_SCHEMA_VERSION,
        "kind": MIGRATION_KIND,
        "status": "completed",
        "createdAt": utc_timestamp(),
        "formalPromotion": False,
        "baseline": _baseline_reference(source.baseline, workspace),
        "source": {
            **_store_reference(source.store, workspace),
            "checkpointReplaySnapshotSha256": source.snapshot_sha256,
        },
        "target": {
            **_store_reference(target, workspace),
            "shardsPreserved": True,
        },
        "transfer": {
            "requestedMode": transfer_mode,
            "hardLinked": hard_linked,
            "copied": copied,
        },
    }
    return {
        **core,
        "migrationManifestSha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _actual_transfer_counts(
    source: _ReadOnlyStore,
    target: _ReadOnlyStore,
) -> tuple[int, int]:
    hard_linked = 0
    copied = 0
    for source_entry, target_entry in zip(source.entries, target.entries):
        source_path = source.root / source_entry.file_name
        target_path = target.root / target_entry.file_name
        try:
            same = os.path.samefile(source_path, target_path)
        except OSError as error:
            raise Phase4ReplayMigrationCorruptionError(
                f"cannot compare migrated shard identity: {target_entry.file_name}"
            ) from error
        if same:
            hard_linked += 1
        else:
            copied += 1
    return hard_linked, copied


def _verify_completed_migration(
    *,
    workspace: Path,
    source: _SourceBinding,
    target_root: Path,
    capacity: ReplayCapacity,
    requested_mode: str | None,
    deep_target: bool = True,
) -> ReplayMigrationVerification:
    migration_path = target_root / MIGRATION_MANIFEST_NAME
    migration, _raw = _load_json(
        migration_path,
        "replay migration manifest",
        Phase4ReplayMigrationCorruptionError,
    )
    if set(migration) != _MIGRATION_KEYS:
        raise Phase4ReplayMigrationCorruptionError(
            "replay migration manifest schema fields are invalid"
        )
    if canonical_json_bytes(migration) != _raw:
        raise Phase4ReplayMigrationCorruptionError(
            "replay migration manifest is not canonical JSON"
        )
    migration_sha256 = _verify_self_hash(
        migration,
        "migrationManifestSha256",
        "replay migration manifest",
        Phase4ReplayMigrationCorruptionError,
    )
    if (
        migration.get("schemaVersion") != MIGRATION_SCHEMA_VERSION
        or migration.get("kind") != MIGRATION_KIND
        or migration.get("status") != "completed"
        or migration.get("formalPromotion") is not False
        or not isinstance(migration.get("createdAt"), str)
        or not migration.get("createdAt")
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "replay migration manifest status is invalid"
        )
    target = _load_store_readonly(
        target_root,
        expected_capacity=capacity,
        allowed_extra_names={MIGRATION_MANIFEST_NAME},
        deep=deep_target,
    )
    if tuple(_shard_payload(entry) for entry in target.entries) != tuple(
        _shard_payload(entry) for entry in source.store.entries
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "target replay shard order/hash differs from source"
        )
    if (
        target.next_sequence != source.store.next_sequence
        or target.prefix_sha256 != source.store.prefix_sha256
        or target.game_ids != source.store.game_ids
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "target replay identity differs from source"
        )
    transfer = migration.get("transfer")
    if not isinstance(transfer, Mapping) or set(transfer) != {
        "requestedMode",
        "hardLinked",
        "copied",
    }:
        raise Phase4ReplayMigrationCorruptionError(
            "migration transfer record is invalid"
        )
    stored_mode = transfer.get("requestedMode")
    if stored_mode not in _TRANSFER_MODES:
        raise Phase4ReplayMigrationCorruptionError(
            "migration transfer mode is invalid"
        )
    if requested_mode is not None and stored_mode != requested_mode:
        raise Phase4ReplayMigrationConfigurationError(
            "completed migration used another transfer mode"
        )
    actual_hard_linked, actual_copied = _actual_transfer_counts(
        source.store, target
    )
    if (
        transfer.get("hardLinked") != actual_hard_linked
        or transfer.get("copied") != actual_copied
        or actual_hard_linked + actual_copied != source.store.game_count
        or (stored_mode == "copy" and actual_hard_linked != 0)
        or (stored_mode == "hardlink" and actual_copied != 0)
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "migration transfer counts no longer match shard files"
        )
    expected_baseline = _baseline_reference(source.baseline, workspace)
    expected_source = {
        **_store_reference(source.store, workspace),
        "checkpointReplaySnapshotSha256": source.snapshot_sha256,
    }
    expected_target = {
        **_store_reference(target, workspace),
        "shardsPreserved": True,
    }
    if (
        migration.get("baseline") != expected_baseline
        or migration.get("source") != expected_source
        or migration.get("target") != expected_target
    ):
        raise Phase4ReplayMigrationCorruptionError(
            "migration manifest lineage/store references do not verify"
        )
    return ReplayMigrationVerification(
        manifest_path=migration_path,
        manifest_sha256=migration_sha256,
        source_root=source.store.root,
        target_root=target.root,
        game_count=target.game_count,
        ply_count=target.ply_count,
        prefix_sha256=target.prefix_sha256,
        capacity=target.capacity,
        hard_linked=actual_hard_linked,
        copied=actual_copied,
        lineage_verified=True,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _resolve_inputs(
    *,
    workspace_root: str | Path,
    baseline_root: str | Path,
    source_root: str | Path,
    target_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    workspace = Path(workspace_root).resolve()
    if (
        workspace.is_symlink()
        or not workspace.is_dir()
        or workspace.resolve() != workspace
    ):
        raise Phase4ReplayMigrationConfigurationError(
            "workspace root must be a real directory"
        )
    baseline = _workspace_directory(
        workspace,
        baseline_root,
        "baseline root",
        must_exist=True,
    )
    source = _workspace_directory(
        workspace,
        source_root,
        "source replay root",
        must_exist=True,
    )
    target = _workspace_directory(
        workspace,
        target_root,
        "target replay root",
        must_exist=False,
    )
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise Phase4ReplayMigrationConfigurationError(
            "target replay root must be absent or a directory"
        )
    if _paths_overlap(target, source):
        raise Phase4ReplayMigrationConfigurationError(
            "target replay root must not overlap the source replay"
        )
    if _paths_overlap(target, baseline):
        raise Phase4ReplayMigrationConfigurationError(
            "target replay root must not overlap the bootstrap baseline"
        )
    return workspace, baseline, source, target


def _assert_target_outside_pilot(
    target_root: Path,
    source: _SourceBinding,
) -> None:
    if _paths_overlap(target_root, source.baseline.pilot_root):
        raise Phase4ReplayMigrationConfigurationError(
            "target replay root must not modify the verified pilot"
        )


def _same_source_binding(
    before: _SourceBinding,
    after: _SourceBinding,
) -> bool:
    return (
        before.baseline == after.baseline
        and before.snapshot_sha256 == after.snapshot_sha256
        and before.store.manifest_sha256 == after.store.manifest_sha256
        and before.store.capacity == after.store.capacity
        and before.store.next_sequence == after.store.next_sequence
        and before.store.entries == after.store.entries
        and before.store.prefix_sha256 == after.store.prefix_sha256
    )


def _rollback_created(
    paths: Sequence[Path],
    *,
    target_root: Path,
    created_target_root: bool,
) -> None:
    for path in reversed(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    _fsync_directory(target_root)
    if created_target_root:
        try:
            target_root.rmdir()
        except OSError:
            pass


def _capacity(
    *,
    target_max_games: int,
    target_max_plies: int,
    target_max_bytes: int,
) -> ReplayCapacity:
    return ReplayCapacity(
        games=target_max_games,
        plies=target_max_plies,
        bytes=target_max_bytes,
    )


def verify_warm_replay_migration(
    *,
    workspace_root: str | Path,
    target_root: str | Path,
    target_max_games: int,
    target_max_plies: int,
    target_max_bytes: int,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, object]:
    """Deeply verify a completed warm replay migration and all source lineage."""

    workspace, baseline, source_path, target = _resolve_inputs(
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        source_root=source_root,
        target_root=target_root,
    )
    capacity = _capacity(
        target_max_games=target_max_games,
        target_max_plies=target_max_plies,
        target_max_bytes=target_max_bytes,
    )
    source = _load_source_binding(
        workspace=workspace,
        baseline_root=baseline,
        source_root=source_path,
    )
    _assert_target_outside_pilot(target, source)
    _capacity_covers(source.store.capacity, capacity)
    if not target.is_dir():
        raise Phase4ReplayMigrationConfigurationError(
            "target replay root does not exist"
        )
    verification = _verify_completed_migration(
        workspace=workspace,
        source=source,
        target_root=target,
        capacity=capacity,
        requested_mode=None,
    )
    return verification.to_payload(workspace)


def prepare_warm_replay_migration(
    *,
    workspace_root: str | Path,
    target_root: str | Path,
    target_max_games: int,
    target_max_plies: int,
    target_max_bytes: int,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    transfer_mode: str = "copy",
) -> dict[str, object]:
    """Migrate the verified pilot-v2 warm replay into a new empty store."""

    if transfer_mode not in _TRANSFER_MODES:
        raise Phase4ReplayMigrationConfigurationError(
            "transfer mode must be copy, hardlink, or auto"
        )
    workspace, baseline, source_path, target = _resolve_inputs(
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        source_root=source_root,
        target_root=target_root,
    )
    capacity = _capacity(
        target_max_games=target_max_games,
        target_max_plies=target_max_plies,
        target_max_bytes=target_max_bytes,
    )
    source = _load_source_binding(
        workspace=workspace,
        baseline_root=baseline,
        source_root=source_path,
    )
    _assert_target_outside_pilot(target, source)
    _capacity_covers(source.store.capacity, capacity)

    if target.exists():
        try:
            target_entries = tuple(target.iterdir())
        except OSError as error:
            raise Phase4ReplayMigrationConfigurationError(
                "target replay root cannot be listed"
            ) from error
        if target_entries:
            if (target / MIGRATION_MANIFEST_NAME).is_file():
                verification = _verify_completed_migration(
                    workspace=workspace,
                    source=source,
                    target_root=target,
                    capacity=capacity,
                    requested_mode=transfer_mode,
                )
                return {
                    **verification.to_payload(workspace),
                    "created": False,
                }
            raise Phase4ReplayMigrationConfigurationError(
                "target replay directory must be empty"
            )

    created_target_root = False
    created_paths: list[Path] = []
    hard_linked = 0
    copied = 0
    try:
        if not target.exists():
            try:
                target.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                raise Phase4ReplayMigrationConfigurationError(
                    f"cannot create target replay root: {target}"
                ) from error
            created_target_root = True

        for entry in source.store.entries:
            source_shard = source.store.root / entry.file_name
            target_shard = target / entry.file_name
            transferred_as = _transfer_shard(
                source_shard,
                target_shard,
                transfer_mode,
            )
            created_paths.append(target_shard)
            if transferred_as == "hardlink":
                hard_linked += 1
            else:
                copied += 1

        target_manifest_payload = _target_manifest_payload(
            source.store,
            capacity,
        )
        target_manifest_bytes = canonical_json_bytes(target_manifest_payload)
        if (
            len(target_manifest_bytes) + source.store.shard_bytes
            > capacity.bytes
        ):
            raise Phase4ReplayMigrationConfigurationError(
                "target replay data exceeds target byte capacity"
            )
        target_manifest_path = target / _REPLAY_MANIFEST_NAME
        try:
            atomic_create_bytes(target_manifest_path, target_manifest_bytes)
        except ImmutableArtifactExistsError as error:
            raise Phase4ReplayMigrationConfigurationError(
                "target replay manifest appeared during migration"
            ) from error
        created_paths.append(target_manifest_path)

        target_store = _load_store_readonly(
            target,
            expected_capacity=capacity,
        )
        if tuple(
            _shard_payload(entry) for entry in target_store.entries
        ) != tuple(
            _shard_payload(entry) for entry in source.store.entries
        ):
            raise Phase4ReplayMigrationCorruptionError(
                "migrated replay shard order/hash differs from source"
            )
        if (
            target_store.next_sequence != source.store.next_sequence
            or target_store.prefix_sha256 != source.store.prefix_sha256
            or target_store.game_ids != source.store.game_ids
        ):
            raise Phase4ReplayMigrationCorruptionError(
                "migrated replay identity differs from source"
            )

        current_source = _load_source_binding(
            workspace=workspace,
            baseline_root=baseline,
            source_root=source_path,
            deep=False,
        )
        if not _same_source_binding(source, current_source):
            raise Phase4ReplayMigrationLineageError(
                "source lineage changed during replay migration"
            )

        migration = _migration_payload(
            workspace=workspace,
            source=current_source,
            target=target_store,
            transfer_mode=transfer_mode,
            hard_linked=hard_linked,
            copied=copied,
        )
        migration_path = target / MIGRATION_MANIFEST_NAME
        try:
            atomic_create_bytes(
                migration_path,
                canonical_json_bytes(migration),
            )
        except ImmutableArtifactExistsError as error:
            raise Phase4ReplayMigrationConfigurationError(
                "migration manifest appeared during migration"
            ) from error
        created_paths.append(migration_path)

        verification = _verify_completed_migration(
            workspace=workspace,
            source=current_source,
            target_root=target,
            capacity=capacity,
            requested_mode=transfer_mode,
            deep_target=False,
        )
    except BaseException:
        _rollback_created(
            created_paths,
            target_root=target,
            created_target_root=created_target_root,
        )
        raise
    return {
        **verification.to_payload(workspace),
        "created": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or verify the verified Phase 4D warm replay migration."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--workspace-root", default=".")
        command.add_argument(
            "--baseline-root",
            default=str(DEFAULT_BASELINE_ROOT),
        )
        command.add_argument(
            "--source-root",
            default=str(DEFAULT_SOURCE_ROOT),
        )
        command.add_argument("--target-root", required=True)
        command.add_argument(
            "--target-max-games",
            type=int,
            required=True,
        )
        command.add_argument(
            "--target-max-plies",
            type=int,
            required=True,
        )
        command.add_argument(
            "--target-max-bytes",
            type=int,
            required=True,
        )

    prepare = subparsers.add_parser(
        "prepare",
        help="Migrate into a new empty target after full lineage verification.",
    )
    add_common_arguments(prepare)
    prepare.add_argument(
        "--transfer-mode",
        choices=sorted(_TRANSFER_MODES),
        default="copy",
    )

    verify = subparsers.add_parser(
        "verify",
        help="Deeply verify an existing migration and its source lineage.",
    )
    add_common_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "workspace_root": args.workspace_root,
        "baseline_root": args.baseline_root,
        "source_root": args.source_root,
        "target_root": args.target_root,
        "target_max_games": args.target_max_games,
        "target_max_plies": args.target_max_plies,
        "target_max_bytes": args.target_max_bytes,
    }
    try:
        if args.command == "prepare":
            result = prepare_warm_replay_migration(
                **common,
                transfer_mode=args.transfer_mode,
            )
        else:
            result = verify_warm_replay_migration(**common)
    except Phase4ReplayMigrationError as error:
        raise SystemExit(f"phase4-replay-migration: {error}") from error
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
    "DEFAULT_SOURCE_ROOT",
    "MIGRATION_KIND",
    "MIGRATION_MANIFEST_NAME",
    "MIGRATION_SCHEMA_VERSION",
    "Phase4ReplayMigrationConfigurationError",
    "Phase4ReplayMigrationCorruptionError",
    "Phase4ReplayMigrationError",
    "Phase4ReplayMigrationLineageError",
    "ReplayCapacity",
    "ReplayMigrationVerification",
    "main",
    "prepare_warm_replay_migration",
    "verify_warm_replay_migration",
]
