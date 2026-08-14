"""Deterministic, bounded, disk-backed tensor cache for raw Phase 4 replay.

``ReplayStore`` remains the durable source of truth.  This module publishes
immutable, checksummed tensor chunks derived from its active FIFO window.  The
cache can always be discarded and reconstructed from the raw game shards.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .codec import mirror_action_index, mirror_state, to_canonical_action_index
from .game import State, get_winner
from .network import ACTION_SIZE, PHASE4_INPUT_CHANNELS, encode_state
from .replay_store import ReplayGame, ReplayShardInfo, ReplayStore
from .run_manifest import (
    atomic_replace_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)


REPLAY_TENSOR_CACHE_SCHEMA_VERSION = 1
REPLAY_TENSOR_CACHE_ALGORITHM_VERSION = 1
REPLAY_TENSOR_CACHE_MAX_BYTES = 4 * 1024**3
REPLAY_TENSOR_CACHE_CHUNK_POSITIONS = 4096
REPLAY_TENSOR_CACHE_BYTES_PER_POSITION = (
    2 * PHASE4_INPUT_CHANNELS * 9 * 9 * 4
    + 2 * ACTION_SIZE * 4
    + 4
)
REPLAY_SAMPLING_ALGORITHM = "python-random-sample-v1"
REPLAY_MIRROR_RNG_ALGORITHM = "shared-one-draw-per-sample-v1"

_MANIFEST_NAME = "manifest.json"
_CHUNKS_DIRECTORY = "chunks"
_CHUNK_MANIFEST_NAME = "chunk.json"
_ARRAY_NAMES = ("states", "policies", "values")
_ARRAY_FILES = {
    "states": "states.npy",
    "policies": "policies.npy",
    "values": "values.npy",
}
_FLOAT32_NAME = np.dtype(np.float32).str
_CHUNK_OVERHEAD_RESERVE_BYTES = 16 * 1024
_MANIFEST_RESERVE_BYTES = 4 * 1024 * 1024


class ReplayTensorCacheError(RuntimeError):
    """Base class for replay tensor cache failures."""


class ReplayTensorCacheCapacityError(ReplayTensorCacheError):
    """Raised before a refresh would exceed the configured cache byte cap."""


class ReplayTensorCacheCorruptionError(ReplayTensorCacheError):
    """Raised when a committed cache artifact fails validation."""


class ReplayTensorCacheConfigurationError(ReplayTensorCacheError):
    """Raised when a cache is opened under a different immutable contract."""


class ReplayTensorCacheStaleError(ReplayTensorCacheError):
    """Raised when replay or cache state changes during a cache operation."""


@dataclass(frozen=True, slots=True)
class ReplayTensorSamplePlan:
    """RNG-complete sample coordinates bound to one immutable cache manifest."""

    manifest_sha256: str
    raw_window_sha256: str
    total_positions: int
    global_indices: tuple[int, ...]
    mirrored: tuple[bool, ...]
    sampling_algorithm: str = REPLAY_SAMPLING_ALGORITHM
    mirror_rng_algorithm: str = REPLAY_MIRROR_RNG_ALGORITHM

    def __post_init__(self) -> None:
        validate_sha256(self.manifest_sha256, "sample plan manifest sha256")
        validate_sha256(self.raw_window_sha256, "sample plan raw window sha256")
        if type(self.total_positions) is not int or self.total_positions <= 0:
            raise ValueError("sample plan total_positions must be positive")
        if not self.global_indices:
            raise ValueError("sample plan cannot be empty")
        if len(self.global_indices) != len(self.mirrored):
            raise ValueError("sample plan coordinates/orientations differ")
        if any(
            type(index) is not int or not 0 <= index < self.total_positions
            for index in self.global_indices
        ):
            raise IndexError("sample plan index is outside replay")
        if len(set(self.global_indices)) != len(self.global_indices):
            raise ValueError("sample plan must sample without replacement")
        if any(type(value) is not bool for value in self.mirrored):
            raise TypeError("sample plan orientations must be bool")
        if self.sampling_algorithm != REPLAY_SAMPLING_ALGORITHM:
            raise ValueError("unsupported replay sampling algorithm")
        if self.mirror_rng_algorithm != REPLAY_MIRROR_RNG_ALGORITHM:
            raise ValueError("unsupported replay mirror RNG algorithm")


@dataclass(frozen=True, slots=True)
class ReplayTensorCacheRefresh:
    algorithm_version: int
    manifest_sha256: str
    manifest_file_sha256: str
    raw_manifest_sha256: str
    raw_window_sha256: str
    games: int
    plies: int
    chunks: int
    cache_bytes: int
    reused_games: int
    encoded_games: int
    encoded_plies: int
    refresh_seconds: float

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithmVersion": self.algorithm_version,
            "manifestSha256": self.manifest_sha256,
            "manifestFileSha256": self.manifest_file_sha256,
            "rawManifestSha256": self.raw_manifest_sha256,
            "rawWindowSha256": self.raw_window_sha256,
            "games": self.games,
            "plies": self.plies,
            "chunks": self.chunks,
            "cacheBytes": self.cache_bytes,
            "reusedGames": self.reused_games,
            "encodedGames": self.encoded_games,
            "encodedPlies": self.encoded_plies,
            "refreshSeconds": self.refresh_seconds,
        }


@dataclass(frozen=True, slots=True)
class ReplayTensorCacheVerification:
    manifest_sha256: str
    manifest_file_sha256: str
    raw_manifest_sha256: str
    raw_window_sha256: str
    games: int
    plies: int
    chunks: int
    cache_bytes: int
    deep: bool
    memory_mapped: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "manifestSha256": self.manifest_sha256,
            "manifestFileSha256": self.manifest_file_sha256,
            "rawManifestSha256": self.raw_manifest_sha256,
            "rawWindowSha256": self.raw_window_sha256,
            "games": self.games,
            "plies": self.plies,
            "chunks": self.chunks,
            "cacheBytes": self.cache_bytes,
            "deep": self.deep,
            "memoryMapped": self.memory_mapped,
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


def _shard_payload(shard: ReplayShardInfo) -> dict[str, object]:
    return {
        "sequence": shard.sequence,
        "gameId": shard.game_id,
        "file": shard.file_name,
        "bytes": shard.byte_size,
        "sha256": shard.sha256,
        "plies": shard.ply_count,
        "minimumGeneration": shard.minimum_generation,
        "maximumGeneration": shard.maximum_generation,
    }


def _raw_signature(payload: Mapping[str, object]) -> tuple[object, ...]:
    return (
        payload.get("sequence"),
        payload.get("gameId"),
        payload.get("file"),
        payload.get("bytes"),
        payload.get("sha256"),
        payload.get("plies"),
        payload.get("minimumGeneration"),
        payload.get("maximumGeneration"),
    )


def _raw_window_sha256(shards: Sequence[ReplayShardInfo]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "shards": [_shard_payload(shard) for shard in shards],
            }
        )
    )


def _cache_directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _array_contract(position_count: int) -> dict[str, tuple[int, ...]]:
    return {
        "states": (position_count, 2, PHASE4_INPUT_CHANNELS, 9, 9),
        "policies": (position_count, 2, ACTION_SIZE),
        "values": (position_count,),
    }


def _validate_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("mirror_probability must be a number")
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("mirror_probability must be in [0, 1]")
    return probability


class ReplayTensorCache:
    """A checksummed, refreshable tensor view of one ``ReplayStore`` window."""

    def __init__(
        self,
        root: str | Path,
        replay: ReplayStore,
        *,
        source_sha256: str,
        encoder_sha256: str,
        native_encoder: Callable[..., object] | None = None,
        maximum_cache_bytes: int = REPLAY_TENSOR_CACHE_MAX_BYTES,
        chunk_positions: int = REPLAY_TENSOR_CACHE_CHUNK_POSITIONS,
    ) -> None:
        if not isinstance(replay, ReplayStore):
            raise TypeError("replay must be ReplayStore")
        self.source_sha256 = validate_sha256(source_sha256, "source_sha256")
        self.encoder_sha256 = validate_sha256(encoder_sha256, "encoder_sha256")
        if native_encoder is not None and not callable(native_encoder):
            raise TypeError("native_encoder must be callable or None")
        if type(maximum_cache_bytes) is not int or maximum_cache_bytes <= 0:
            raise ValueError("maximum_cache_bytes must be positive")
        if type(chunk_positions) is not int or chunk_positions <= 0:
            raise ValueError("chunk_positions must be positive")
        self.root = Path(root)
        self.replay = replay
        self.native_encoder = native_encoder
        self.maximum_cache_bytes = maximum_cache_bytes
        self.chunk_positions = chunk_positions
        if self.root.resolve() == replay.root.resolve():
            raise ValueError("tensor cache root must differ from replay root")
        self.manifest_path = self.root / _MANIFEST_NAME
        self.chunks_root = self.root / _CHUNKS_DIRECTORY
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, object] | None = None
        self._manifest_file_sha256: str | None = None
        self._stale = False
        self._verified_chunk_stats: dict[
            str, tuple[tuple[int, int, int], ...]
        ] = {}
        self._raw_shard_stats: tuple[tuple[int, int, int], ...] | None = None
        self._remove_temporary_artifacts()
        if self.manifest_path.exists():
            self._manifest = self._load_manifest()
            self._manifest_file_sha256 = sha256_file(self.manifest_path)
            self._remove_unreferenced_chunks(self._manifest)
        else:
            self._remove_unreferenced_chunks(None)

    @property
    def is_built(self) -> bool:
        return self._manifest is not None and not self._stale

    def _remove_temporary_artifacts(self) -> None:
        for path in self.root.glob(".*.tmp"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for path in self.chunks_root.glob(".chunk-*.tmp"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def _remove_unreferenced_chunks(
        self, manifest: Mapping[str, object] | None
    ) -> None:
        referenced = set()
        if manifest is not None:
            chunks = manifest.get("chunks")
            if isinstance(chunks, list):
                referenced = {
                    str(chunk.get("directory"))
                    for chunk in chunks
                    if isinstance(chunk, Mapping)
                }
        for path in self.chunks_root.iterdir():
            if path.name.startswith(".chunk-") and path.name.endswith(".tmp"):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                continue
            if path.name not in referenced:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        _fsync_directory(self.chunks_root)

    def _manifest_contract(self) -> dict[str, object]:
        return {
            "maximumCacheBytes": self.maximum_cache_bytes,
            "chunkPositions": self.chunk_positions,
            "inputChannels": PHASE4_INPUT_CHANNELS,
            "actionSize": ACTION_SIZE,
            "bytesPerPosition": REPLAY_TENSOR_CACHE_BYTES_PER_POSITION,
            "sourceSha256": self.source_sha256,
            "encoderSha256": self.encoder_sha256,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "replayCapacity": {
                "games": self.replay.max_games,
                "plies": self.replay.max_plies,
                "bytes": self.replay.max_bytes,
            },
        }

    def _load_manifest(self) -> dict[str, object]:
        try:
            raw = self.manifest_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest cannot be decoded"
            ) from error
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest is not canonical"
            )
        expected_keys = {
            "schemaVersion",
            "algorithmVersion",
            "samplingAlgorithm",
            "mirrorRngAlgorithm",
            "contract",
            "rawManifestSha256",
            "rawWindowSha256",
            "totals",
            "entries",
            "chunks",
            "manifestSha256",
        }
        if set(payload) != expected_keys:
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest schema changed"
            )
        core = dict(payload)
        stored_digest = core.pop("manifestSha256")
        try:
            validate_sha256(stored_digest, "tensor cache manifest sha256")
        except ValueError as error:
            raise ReplayTensorCacheCorruptionError(str(error)) from error
        if sha256_bytes(canonical_json_bytes(core)) != stored_digest:
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest self hash mismatch"
            )
        if (
            payload.get("schemaVersion") != REPLAY_TENSOR_CACHE_SCHEMA_VERSION
            or payload.get("algorithmVersion")
            != REPLAY_TENSOR_CACHE_ALGORITHM_VERSION
            or payload.get("samplingAlgorithm") != REPLAY_SAMPLING_ALGORITHM
            or payload.get("mirrorRngAlgorithm") != REPLAY_MIRROR_RNG_ALGORITHM
        ):
            raise ReplayTensorCacheConfigurationError(
                "tensor cache algorithm contract changed"
            )
        if payload.get("contract") != self._manifest_contract():
            raise ReplayTensorCacheConfigurationError(
                "tensor cache immutable contract differs"
            )
        try:
            validate_sha256(payload.get("rawManifestSha256"), "raw manifest sha256")
            validate_sha256(payload.get("rawWindowSha256"), "raw window sha256")
        except ValueError as error:
            raise ReplayTensorCacheCorruptionError(str(error)) from error
        entries = payload.get("entries")
        chunks = payload.get("chunks")
        totals = payload.get("totals")
        if (
            not isinstance(entries, list)
            or not isinstance(chunks, list)
            or not isinstance(totals, Mapping)
            or set(totals) != {"games", "plies", "chunks", "cacheBytes"}
        ):
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest collections are invalid"
            )
        chunk_by_id: dict[str, Mapping[str, object]] = {}
        for chunk in chunks:
            self._validate_chunk_record(chunk, deep=False)
            if not isinstance(chunk, Mapping):
                raise ReplayTensorCacheCorruptionError("chunk record is invalid")
            chunk_id = chunk.get("id")
            if not isinstance(chunk_id, str) or chunk_id in chunk_by_id:
                raise ReplayTensorCacheCorruptionError(
                    "chunk IDs must be unique strings"
                )
            chunk_by_id[chunk_id] = chunk
        previous_sequence = -1
        game_ids: set[str] = set()
        position_total = 0
        for entry in entries:
            self._validate_entry(entry, chunk_by_id)
            if not isinstance(entry, Mapping):
                raise ReplayTensorCacheCorruptionError("cache entry is invalid")
            sequence = entry.get("sequence")
            game_id = entry.get("gameId")
            if (
                not isinstance(sequence, int)
                or sequence <= previous_sequence
                or not isinstance(game_id, str)
                or game_id in game_ids
            ):
                raise ReplayTensorCacheCorruptionError(
                    "cache entry order/identity is invalid"
                )
            previous_sequence = sequence
            game_ids.add(game_id)
            position_total += int(entry["plies"])
        actual_cache_bytes = sum(int(chunk["bytes"]) for chunk in chunks)
        expected_totals = {
            "games": len(entries),
            "plies": position_total,
            "chunks": len(chunks),
            "cacheBytes": actual_cache_bytes,
        }
        if dict(totals) != expected_totals:
            raise ReplayTensorCacheCorruptionError(
                "tensor cache manifest totals differ"
            )
        if actual_cache_bytes > self.maximum_cache_bytes:
            raise ReplayTensorCacheCapacityError(
                "committed tensor cache exceeds its configured byte cap"
            )
        return payload

    def _validate_entry(
        self,
        entry: object,
        chunk_by_id: Mapping[str, Mapping[str, object]],
    ) -> None:
        expected = {
            "sequence",
            "gameId",
            "file",
            "bytes",
            "sha256",
            "plies",
            "minimumGeneration",
            "maximumGeneration",
            "chunkId",
            "chunkOffset",
        }
        if not isinstance(entry, Mapping) or set(entry) != expected:
            raise ReplayTensorCacheCorruptionError("tensor cache entry schema changed")
        raw = {key: entry[key] for key in expected - {"chunkId", "chunkOffset"}}
        if (
            type(raw["sequence"]) is not int
            or raw["sequence"] < 0
            or not isinstance(raw["gameId"], str)
            or not raw["gameId"]
            or not isinstance(raw["file"], str)
            or Path(raw["file"]).name != raw["file"]
            or type(raw["bytes"]) is not int
            or raw["bytes"] <= 0
            or type(raw["plies"]) is not int
            or raw["plies"] <= 0
            or type(raw["minimumGeneration"]) is not int
            or raw["minimumGeneration"] < 0
            or type(raw["maximumGeneration"]) is not int
            or raw["maximumGeneration"] < raw["minimumGeneration"]
        ):
            raise ReplayTensorCacheCorruptionError("tensor cache entry values are invalid")
        try:
            validate_sha256(raw["sha256"], "tensor cache entry sha256")
        except ValueError as error:
            raise ReplayTensorCacheCorruptionError(str(error)) from error
        chunk_id = entry["chunkId"]
        offset = entry["chunkOffset"]
        if (
            not isinstance(chunk_id, str)
            or chunk_id not in chunk_by_id
            or type(offset) is not int
            or offset < 0
            or offset + raw["plies"] > chunk_by_id[chunk_id]["positions"]
        ):
            raise ReplayTensorCacheCorruptionError(
                "tensor cache entry chunk mapping is invalid"
            )

    def _validate_chunk_record(self, chunk: object, *, deep: bool) -> None:
        expected = {
            "id",
            "directory",
            "chunkSha256",
            "firstSequence",
            "lastSequence",
            "positions",
            "bytes",
            "files",
        }
        if not isinstance(chunk, Mapping) or set(chunk) != expected:
            raise ReplayTensorCacheCorruptionError("tensor cache chunk schema changed")
        chunk_id = chunk.get("id")
        directory = chunk.get("directory")
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or not isinstance(directory, str)
            or Path(directory).name != directory
            or not directory.startswith("chunk-")
            or type(chunk.get("firstSequence")) is not int
            or type(chunk.get("lastSequence")) is not int
            or chunk["firstSequence"] < 0
            or chunk["lastSequence"] < chunk["firstSequence"]
            or type(chunk.get("positions")) is not int
            or chunk["positions"] <= 0
            or type(chunk.get("bytes")) is not int
            or chunk["bytes"] <= 0
        ):
            raise ReplayTensorCacheCorruptionError("tensor cache chunk values are invalid")
        try:
            validate_sha256(chunk.get("chunkSha256"), "chunk sha256")
        except ValueError as error:
            raise ReplayTensorCacheCorruptionError(str(error)) from error
        files = chunk.get("files")
        if not isinstance(files, Mapping) or set(files) != set(_ARRAY_NAMES):
            raise ReplayTensorCacheCorruptionError("tensor cache chunk files are invalid")
        expected_shapes = _array_contract(int(chunk["positions"]))
        file_total = 0
        for name in _ARRAY_NAMES:
            record = files[name]
            if (
                not isinstance(record, Mapping)
                or set(record) != {"file", "bytes", "sha256", "dtype", "shape"}
                or record.get("file") != _ARRAY_FILES[name]
                or type(record.get("bytes")) is not int
                or record["bytes"] <= 0
                or record.get("dtype") != _FLOAT32_NAME
                or record.get("shape") != list(expected_shapes[name])
            ):
                raise ReplayTensorCacheCorruptionError(
                    f"tensor cache {name} file contract is invalid"
                )
            try:
                validate_sha256(record.get("sha256"), f"{name} file sha256")
            except ValueError as error:
                raise ReplayTensorCacheCorruptionError(str(error)) from error
            file_total += int(record["bytes"])
        chunk_root = self.chunks_root / directory
        metadata_path = chunk_root / _CHUNK_MANIFEST_NAME
        if (
            not chunk_root.is_dir()
            or chunk_root.is_symlink()
            or not metadata_path.is_file()
            or metadata_path.is_symlink()
        ):
            raise ReplayTensorCacheCorruptionError(
                f"tensor cache chunk is missing: {directory}"
            )
        try:
            raw_metadata = metadata_path.read_bytes()
            metadata = json.loads(raw_metadata)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayTensorCacheCorruptionError(
                f"tensor cache chunk metadata is corrupt: {directory}"
            ) from error
        if (
            not isinstance(metadata, Mapping)
            or canonical_json_bytes(metadata) != raw_metadata
            or metadata != chunk
            or sha256_bytes(canonical_json_bytes(
                {key: metadata[key] for key in metadata if key != "chunkSha256"}
            ))
            != chunk["chunkSha256"]
        ):
            raise ReplayTensorCacheCorruptionError(
                f"tensor cache chunk metadata differs: {directory}"
            )
        actual_total = metadata_path.stat().st_size
        for name in _ARRAY_NAMES:
            record = files[name]
            path = chunk_root / str(record["file"])
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record["bytes"]
            ):
                raise ReplayTensorCacheCorruptionError(
                    f"tensor cache {name} file differs: {directory}"
                )
            if deep and sha256_file(path) != record["sha256"]:
                raise ReplayTensorCacheCorruptionError(
                    f"tensor cache {name} checksum differs: {directory}"
                )
            actual_total += path.stat().st_size
        if (
            actual_total != chunk["bytes"]
            or file_total + metadata_path.stat().st_size != chunk["bytes"]
        ):
            raise ReplayTensorCacheCorruptionError(
                f"tensor cache chunk byte total differs: {directory}"
            )

    def _current_replay_identity(
        self,
    ) -> tuple[tuple[ReplayShardInfo, ...], str, str]:
        shards = self.replay.shards
        try:
            raw_manifest_sha256 = sha256_file(self.replay.manifest_path)
        except OSError as error:
            raise ReplayTensorCacheStaleError(
                "raw replay manifest cannot be hashed"
            ) from error
        return shards, raw_manifest_sha256, _raw_window_sha256(shards)

    def _raw_file_stats(
        self, shards: Sequence[ReplayShardInfo]
    ) -> tuple[tuple[int, int, int], ...]:
        try:
            stats = []
            for shard in shards:
                value = (self.replay.root / shard.file_name).stat()
                stats.append((value.st_size, value.st_mtime_ns, value.st_ino))
            return tuple(stats)
        except OSError as error:
            raise ReplayTensorCacheCorruptionError(
                "raw replay shard metadata cannot be read"
            ) from error

    def _verify_raw_shards(
        self, shards: Sequence[ReplayShardInfo]
    ) -> tuple[tuple[int, int, int], ...]:
        stats_before = self._raw_file_stats(shards)
        for shard, stat in zip(shards, stats_before, strict=True):
            path = self.replay.root / shard.file_name
            if stat[0] != shard.byte_size or sha256_file(path) != shard.sha256:
                raise ReplayTensorCacheCorruptionError(
                    f"raw replay shard differs from its manifest: {shard.file_name}"
                )
        stats_after = self._raw_file_stats(shards)
        if stats_after != stats_before:
            raise ReplayTensorCacheStaleError(
                "raw replay shard changed during checksum verification"
            )
        return stats_after

    def _assert_manifest_current(self) -> None:
        if self._stale or self._manifest is None or self._manifest_file_sha256 is None:
            raise ReplayTensorCacheStaleError(
                "tensor cache is stale or has not been built"
            )
        try:
            if sha256_file(self.manifest_path) != self._manifest_file_sha256:
                raise ReplayTensorCacheStaleError(
                    "tensor cache manifest changed after publication"
                )
            shards, raw_manifest_sha256, raw_window_sha256 = (
                self._current_replay_identity()
            )
        except Exception:
            self._stale = True
            raise
        if (
            self._manifest.get("rawManifestSha256") != raw_manifest_sha256
            or self._manifest.get("rawWindowSha256") != raw_window_sha256
            or [_raw_signature(entry) for entry in self._manifest["entries"]]
            != [_raw_signature(_shard_payload(shard)) for shard in shards]
        ):
            self._stale = True
            raise ReplayTensorCacheStaleError(
                "raw replay changed after tensor cache publication"
            )
        try:
            current_stats = self._raw_file_stats(shards)
            if self._raw_shard_stats is None:
                self._raw_shard_stats = self._verify_raw_shards(shards)
            elif current_stats != self._raw_shard_stats:
                raise ReplayTensorCacheCorruptionError(
                    "raw replay shard changed after tensor cache publication"
                )
        except Exception:
            self._stale = True
            raise

    @staticmethod
    def _validate_window_transition(
        old_entries: Sequence[Mapping[str, object]],
        current_shards: Sequence[ReplayShardInfo],
    ) -> None:
        if not old_entries:
            return
        old = [_raw_signature(entry) for entry in old_entries]
        current = [_raw_signature(_shard_payload(shard)) for shard in current_shards]
        if current == old:
            return
        old_sequences = [int(signature[0]) for signature in old]
        current_sequences = [int(signature[0]) for signature in current]
        if not current:
            raise ReplayTensorCacheStaleError(
                "non-empty replay window cannot transition to empty"
            )
        overlap = set(old_sequences).intersection(current_sequences)
        if overlap:
            first_overlap = current_sequences[0]
            if first_overlap not in old_sequences:
                raise ReplayTensorCacheStaleError("replay overlap is not a suffix/prefix")
            old_start = old_sequences.index(first_overlap)
            overlap_count = len(old) - old_start
            if current[:overlap_count] != old[old_start:]:
                raise ReplayTensorCacheStaleError(
                    "replay FIFO overlap differs from the cached suffix"
                )
            appended = current[overlap_count:]
        else:
            if current_sequences[0] <= old_sequences[-1]:
                raise ReplayTensorCacheStaleError(
                    "replay window was replaced without monotonic append"
                )
            appended = current
        if not appended or any(
            int(signature[0]) <= old_sequences[-1] for signature in appended
        ):
            raise ReplayTensorCacheStaleError(
                "replay eviction is not accompanied by a monotonic append"
            )

    def _group_missing_shards(
        self, shards: Sequence[ReplayShardInfo]
    ) -> tuple[tuple[ReplayShardInfo, ...], ...]:
        groups: list[list[ReplayShardInfo]] = []
        current: list[ReplayShardInfo] = []
        positions = 0
        for shard in shards:
            if current and positions + shard.ply_count > self.chunk_positions:
                groups.append(current)
                current = []
                positions = 0
            current.append(shard)
            positions += shard.ply_count
        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)

    def _encode_states(self, states: Sequence[State]) -> np.ndarray:
        if self.native_encoder is None:
            tensor = torch.stack(
                [
                    encode_state(state, input_channels=PHASE4_INPUT_CHANNELS)
                    for state in states
                ]
            )
            return tensor.numpy()
        encoded = np.asarray(
            self.native_encoder(
                states,
                input_channels=PHASE4_INPUT_CHANNELS,
            )
        )
        expected = (len(states), PHASE4_INPUT_CHANNELS, 9, 9)
        if encoded.shape != expected or encoded.dtype != np.float32:
            raise ReplayTensorCacheError(
                f"native encoder must return float32 shape {expected}"
            )
        return np.ascontiguousarray(encoded)

    @staticmethod
    def _policy(game: ReplayGame, states: Sequence[State], mirrored: bool) -> torch.Tensor:
        policies = []
        for state, ply in zip(states, game.plies, strict=True):
            transformed_state = mirror_state(state) if mirrored else state
            policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
            for raw_index, count in enumerate(ply.visit_counts):
                if count <= 0:
                    continue
                transformed_index = (
                    mirror_action_index(raw_index) if mirrored else raw_index
                )
                canonical_index = to_canonical_action_index(
                    transformed_index,
                    transformed_state.current_player,
                )
                policy[canonical_index] += float(count)
            total = policy.sum()
            if not bool(torch.isfinite(total).item()) or total <= 0:
                raise ReplayTensorCacheError(
                    f"replay game {game.game_id} has invalid visit mass"
                )
            policies.append(policy / total)
        return torch.stack(policies)

    def _encode_game(
        self,
        game: ReplayGame,
        states: Sequence[State],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(states) != len(game.plies) + 1:
            raise ReplayTensorCacheError(
                f"reconstructed state count differs for {game.game_id}"
            )
        winner = get_winner(states[-1])
        if winner not in (0, 1):
            raise ReplayTensorCacheError(
                f"replay game has no winner: {game.game_id}"
            )
        positions = states[:-1]
        identity = self._encode_states(positions)
        mirrored_states = self._encode_states(
            tuple(mirror_state(state) for state in positions)
        )
        encoded_states = np.stack((identity, mirrored_states), axis=1)
        encoded_policies = np.stack(
            (
                self._policy(game, positions, False).numpy(),
                self._policy(game, positions, True).numpy(),
            ),
            axis=1,
        )
        values = np.asarray(
            [
                1.0 if state.current_player == winner else -1.0
                for state in positions
            ],
            dtype=np.float32,
        )
        return encoded_states, encoded_policies, values

    def _build_chunk(
        self, shards: Sequence[ReplayShardInfo]
    ) -> tuple[dict[str, object], dict[tuple[object, ...], tuple[str, int]]]:
        positions = sum(shard.ply_count for shard in shards)
        temporary_root = Path(
            tempfile.mkdtemp(
                dir=self.chunks_root,
                prefix=".chunk-",
                suffix=".tmp",
            )
        )
        shapes = _array_contract(positions)
        arrays: dict[str, np.memmap] = {}
        mapping: dict[tuple[object, ...], tuple[str, int]] = {}
        try:
            for name in _ARRAY_NAMES:
                arrays[name] = np.lib.format.open_memmap(
                    temporary_root / _ARRAY_FILES[name],
                    mode="w+",
                    dtype=np.float32,
                    shape=shapes[name],
                )
            offset = 0
            for shard in shards:
                game, states = self.replay.load_game_with_states(shard.game_id)
                encoded_states, encoded_policies, values = self._encode_game(
                    game,
                    states,
                )
                end = offset + shard.ply_count
                arrays["states"][offset:end] = encoded_states
                arrays["policies"][offset:end] = encoded_policies
                arrays["values"][offset:end] = values
                mapping[_raw_signature(_shard_payload(shard))] = ("", offset)
                offset = end
            if offset != positions:
                raise ReplayTensorCacheError("tensor chunk coverage is incomplete")
            for array in arrays.values():
                array.flush()
            arrays.clear()
            for name in _ARRAY_NAMES:
                path = temporary_root / _ARRAY_FILES[name]
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            files: dict[str, object] = {}
            for name in _ARRAY_NAMES:
                path = temporary_root / _ARRAY_FILES[name]
                files[name] = {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "dtype": _FLOAT32_NAME,
                    "shape": list(shapes[name]),
                }
            chunk_core: dict[str, object] = {
                "id": (
                    f"{shards[0].sequence:012d}-{shards[-1].sequence:012d}"
                ),
                "directory": "",
                "firstSequence": shards[0].sequence,
                "lastSequence": shards[-1].sequence,
                "positions": positions,
                "bytes": 0,
                "files": files,
            }
            preliminary_sha = sha256_bytes(canonical_json_bytes(chunk_core))
            directory = (
                f"chunk-{shards[0].sequence:012d}-{shards[-1].sequence:012d}-"
                f"{preliminary_sha[:16]}"
            )
            chunk_core["directory"] = directory
            metadata_size = 0
            for _ in range(4):
                chunk_core["bytes"] = (
                    sum(int(record["bytes"]) for record in files.values())
                    + metadata_size
                )
                digest = sha256_bytes(canonical_json_bytes(chunk_core))
                payload = {**chunk_core, "chunkSha256": digest}
                next_size = len(canonical_json_bytes(payload))
                if next_size == metadata_size:
                    break
                metadata_size = next_size
            chunk_core["bytes"] = (
                sum(int(record["bytes"]) for record in files.values())
                + metadata_size
            )
            digest = sha256_bytes(canonical_json_bytes(chunk_core))
            chunk = {**chunk_core, "chunkSha256": digest}
            metadata = canonical_json_bytes(chunk)
            if len(metadata) != metadata_size:
                raise ReplayTensorCacheError(
                    "tensor chunk metadata size did not converge"
                )
            metadata_path = temporary_root / _CHUNK_MANIFEST_NAME
            with metadata_path.open("xb") as destination:
                destination.write(metadata)
                destination.flush()
                os.fsync(destination.fileno())
            _fsync_directory(temporary_root)
            final_root = self.chunks_root / directory
            if final_root.exists():
                existing = self._chunk_from_directory(final_root)
                if existing != chunk:
                    raise ReplayTensorCacheCorruptionError(
                        f"immutable tensor chunk collision: {directory}"
                    )
                shutil.rmtree(temporary_root)
            else:
                os.replace(temporary_root, final_root)
                _fsync_directory(self.chunks_root)
            chunk_id = str(chunk["id"])
            mapping = {
                signature: (chunk_id, offset)
                for signature, (_, offset) in mapping.items()
            }
            return chunk, mapping
        except Exception:
            arrays.clear()
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            raise

    @staticmethod
    def _chunk_from_directory(root: Path) -> dict[str, object]:
        try:
            payload = json.loads((root / _CHUNK_MANIFEST_NAME).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayTensorCacheCorruptionError(
                f"cannot read existing immutable chunk: {root.name}"
            ) from error
        if not isinstance(payload, dict):
            raise ReplayTensorCacheCorruptionError(
                f"existing immutable chunk is invalid: {root.name}"
            )
        return payload

    def _entry_with_mapping(
        self,
        shard: ReplayShardInfo,
        chunk_id: str,
        offset: int,
    ) -> dict[str, object]:
        return {
            **_shard_payload(shard),
            "chunkId": chunk_id,
            "chunkOffset": offset,
        }

    def refresh(self) -> ReplayTensorCacheRefresh:
        """Publish a tensor manifest for the replay's current active FIFO window."""

        if self._stale:
            raise ReplayTensorCacheStaleError(
                "stale tensor cache must be reopened before refresh"
            )
        started = time.monotonic()
        self._remove_temporary_artifacts()
        old_manifest = self._manifest
        old_entries = (
            tuple(old_manifest["entries"]) if old_manifest is not None else ()
        )
        shards_before, raw_manifest_before, raw_window_before = (
            self._current_replay_identity()
        )
        try:
            if old_manifest is not None and (
                self._manifest_file_sha256 is None
                or not self.manifest_path.is_file()
                or sha256_file(self.manifest_path) != self._manifest_file_sha256
            ):
                raise ReplayTensorCacheStaleError(
                    "tensor cache manifest changed before refresh"
                )
            self._validate_window_transition(old_entries, shards_before)
            raw_stats_before = self._verify_raw_shards(shards_before)
            expected_tensor_bytes = (
                sum(shard.ply_count for shard in shards_before)
                * REPLAY_TENSOR_CACHE_BYTES_PER_POSITION
            )
            if expected_tensor_bytes > self.maximum_cache_bytes:
                raise ReplayTensorCacheCapacityError(
                    "active replay tensor payload exceeds the 4 GiB cache cap"
                )
            old_mapping: dict[
                tuple[object, ...], tuple[str, int]
            ] = {}
            old_chunks: dict[str, Mapping[str, object]] = {}
            if old_manifest is not None:
                old_chunks = {
                    str(chunk["id"]): chunk for chunk in old_manifest["chunks"]
                }
                old_mapping = {
                    _raw_signature(entry): (
                        str(entry["chunkId"]),
                        int(entry["chunkOffset"]),
                    )
                    for entry in old_entries
                }
            missing = [
                shard
                for shard in shards_before
                if _raw_signature(_shard_payload(shard)) not in old_mapping
            ]
            missing_groups = self._group_missing_shards(missing)
            estimated_new_bytes = (
                sum(shard.ply_count for shard in missing)
                * REPLAY_TENSOR_CACHE_BYTES_PER_POSITION
                + len(missing_groups) * _CHUNK_OVERHEAD_RESERVE_BYTES
            )
            current_cache_bytes = _cache_directory_bytes(self.root)
            if (
                current_cache_bytes
                + estimated_new_bytes
                + _MANIFEST_RESERVE_BYTES
                > self.maximum_cache_bytes
            ):
                raise ReplayTensorCacheCapacityError(
                    "tensor cache refresh transient bytes exceed the configured cap"
                )
            if shutil.disk_usage(self.root).free < (
                estimated_new_bytes + _MANIFEST_RESERVE_BYTES
            ):
                raise ReplayTensorCacheCapacityError(
                    "insufficient free disk for atomic tensor cache refresh"
                )
            new_chunks: dict[str, Mapping[str, object]] = dict(old_chunks)
            encoded_mapping: dict[
                tuple[object, ...], tuple[str, int]
            ] = {}
            for group in missing_groups:
                chunk, mapping = self._build_chunk(group)
                new_chunks[str(chunk["id"])] = chunk
                encoded_mapping.update(mapping)
            all_mapping = {**old_mapping, **encoded_mapping}
            entries = [
                self._entry_with_mapping(
                    shard,
                    *all_mapping[_raw_signature(_shard_payload(shard))],
                )
                for shard in shards_before
            ]
            referenced_chunk_ids = {str(entry["chunkId"]) for entry in entries}
            chunks = [
                dict(new_chunks[chunk_id])
                for chunk_id in sorted(
                    referenced_chunk_ids,
                    key=lambda value: (
                        int(new_chunks[value]["firstSequence"]),
                        value,
                    ),
                )
            ]
            for chunk in chunks:
                self._validate_chunk_record(chunk, deep=False)
            shards_after, raw_manifest_after, raw_window_after = (
                self._current_replay_identity()
            )
            if (
                shards_after != shards_before
                or raw_manifest_after != raw_manifest_before
                or raw_window_after != raw_window_before
                or self._raw_file_stats(shards_after) != raw_stats_before
            ):
                raise ReplayTensorCacheStaleError(
                    "raw replay changed during tensor cache refresh"
                )
            cache_bytes = sum(int(chunk["bytes"]) for chunk in chunks)
            core: dict[str, object] = {
                "schemaVersion": REPLAY_TENSOR_CACHE_SCHEMA_VERSION,
                "algorithmVersion": REPLAY_TENSOR_CACHE_ALGORITHM_VERSION,
                "samplingAlgorithm": REPLAY_SAMPLING_ALGORITHM,
                "mirrorRngAlgorithm": REPLAY_MIRROR_RNG_ALGORITHM,
                "contract": self._manifest_contract(),
                "rawManifestSha256": raw_manifest_before,
                "rawWindowSha256": raw_window_before,
                "totals": {
                    "games": len(entries),
                    "plies": sum(shard.ply_count for shard in shards_before),
                    "chunks": len(chunks),
                    "cacheBytes": cache_bytes,
                },
                "entries": entries,
                "chunks": chunks,
            }
            manifest_sha256 = sha256_bytes(canonical_json_bytes(core))
            manifest = {**core, "manifestSha256": manifest_sha256}
            manifest_bytes = canonical_json_bytes(manifest)
            if cache_bytes + len(manifest_bytes) > self.maximum_cache_bytes:
                raise ReplayTensorCacheCapacityError(
                    "committed tensor cache exceeds the configured byte cap"
                )
            atomic_replace_bytes(self.manifest_path, manifest_bytes)
            manifest_file_sha256 = sha256_file(self.manifest_path)
            # Publication happens before garbage collection; a crash here leaves
            # a valid manifest and harmless orphan chunks recoverable on reopen.
            self._manifest = manifest
            self._manifest_file_sha256 = manifest_file_sha256
            self._raw_shard_stats = raw_stats_before
            self._remove_unreferenced_chunks(manifest)
            self._verified_chunk_stats.clear()
            elapsed = max(0.0, time.monotonic() - started)
            return ReplayTensorCacheRefresh(
                algorithm_version=REPLAY_TENSOR_CACHE_ALGORITHM_VERSION,
                manifest_sha256=manifest_sha256,
                manifest_file_sha256=manifest_file_sha256,
                raw_manifest_sha256=raw_manifest_before,
                raw_window_sha256=raw_window_before,
                games=len(entries),
                plies=sum(shard.ply_count for shard in shards_before),
                chunks=len(chunks),
                cache_bytes=cache_bytes,
                reused_games=len(entries) - len(missing),
                encoded_games=len(missing),
                encoded_plies=sum(shard.ply_count for shard in missing),
                refresh_seconds=elapsed,
            )
        except Exception:
            self._stale = True
            raise

    def _chunk_stats(
        self, chunk: Mapping[str, object]
    ) -> tuple[tuple[int, int, int], ...]:
        root = self.chunks_root / str(chunk["directory"])
        paths = [root / _CHUNK_MANIFEST_NAME] + [
            root / _ARRAY_FILES[name] for name in _ARRAY_NAMES
        ]
        return tuple(
            (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ino,
            )
            for path in paths
        )

    def _verify_chunk_for_read(self, chunk: Mapping[str, object]) -> None:
        chunk_id = str(chunk["id"])
        try:
            stats = self._chunk_stats(chunk)
            cached = self._verified_chunk_stats.get(chunk_id)
            if cached is None:
                self._validate_chunk_record(chunk, deep=True)
                stats = self._chunk_stats(chunk)
                self._verified_chunk_stats[chunk_id] = stats
            elif cached != stats:
                raise ReplayTensorCacheCorruptionError(
                    f"tensor cache chunk changed after verification: {chunk_id}"
                )
        except (OSError, ReplayTensorCacheError):
            self._stale = True
            raise

    def sample_plan(
        self,
        count: int,
        *,
        rng: random.Random | None = None,
        mirror_probability: float = 0.5,
    ) -> ReplayTensorSamplePlan:
        """Consume the exact legacy sampling and mirror RNG sequence."""

        self._assert_manifest_current()
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        probability = _validate_probability(mirror_probability)
        if self._manifest is None:
            raise ReplayTensorCacheStaleError("tensor cache has not been built")
        total = int(self._manifest["totals"]["plies"])
        if total == 0:
            raise ValueError("cannot sample an empty replay cache")
        generator = rng or random.SystemRandom()
        selected = tuple(generator.sample(range(total), min(count, total)))
        # This draw is intentionally unconditional, including p=0 and p=1.
        mirrored = tuple(generator.random() < probability for _ in selected)
        return ReplayTensorSamplePlan(
            manifest_sha256=str(self._manifest["manifestSha256"]),
            raw_window_sha256=str(self._manifest["rawWindowSha256"]),
            total_positions=total,
            global_indices=selected,
            mirrored=mirrored,
        )

    def load_plan(
        self,
        plan: ReplayTensorSamplePlan,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Gather one planned batch via read-only NumPy memory maps."""

        if not isinstance(plan, ReplayTensorSamplePlan):
            raise TypeError("plan must be ReplayTensorSamplePlan")
        self._assert_manifest_current()
        if self._manifest is None:
            raise ReplayTensorCacheStaleError("tensor cache has not been built")
        if (
            plan.manifest_sha256 != self._manifest["manifestSha256"]
            or plan.raw_window_sha256 != self._manifest["rawWindowSha256"]
            or plan.total_positions != self._manifest["totals"]["plies"]
        ):
            raise ReplayTensorCacheStaleError(
                "sample plan belongs to another tensor cache snapshot"
            )
        entries = self._manifest["entries"]
        chunks = {
            str(chunk["id"]): chunk for chunk in self._manifest["chunks"]
        }
        cumulative: list[int] = []
        running = 0
        for entry in entries:
            running += int(entry["plies"])
            cumulative.append(running)
        grouped: dict[str, list[tuple[int, int, bool]]] = {}
        for output_index, (global_index, mirrored) in enumerate(
            zip(plan.global_indices, plan.mirrored, strict=True)
        ):
            entry_index = bisect.bisect_right(cumulative, global_index)
            previous = cumulative[entry_index - 1] if entry_index else 0
            entry = entries[entry_index]
            chunk_row = int(entry["chunkOffset"]) + global_index - previous
            grouped.setdefault(str(entry["chunkId"]), []).append(
                (output_index, chunk_row, mirrored)
            )
        batch_size = len(plan.global_indices)
        states = torch.empty(
            (batch_size, PHASE4_INPUT_CHANNELS, 9, 9),
            dtype=torch.float32,
        )
        policies = torch.empty((batch_size, ACTION_SIZE), dtype=torch.float32)
        values = torch.empty(batch_size, dtype=torch.float32)
        try:
            for chunk_id, requested in grouped.items():
                chunk = chunks[chunk_id]
                self._verify_chunk_for_read(chunk)
                root = self.chunks_root / str(chunk["directory"])
                state_map = np.load(
                    root / _ARRAY_FILES["states"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                policy_map = np.load(
                    root / _ARRAY_FILES["policies"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                value_map = np.load(
                    root / _ARRAY_FILES["values"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                rows = np.asarray([row for _, row, _ in requested], dtype=np.int64)
                orientations = np.asarray(
                    [int(mirrored) for _, _, mirrored in requested],
                    dtype=np.int64,
                )
                outputs = torch.tensor(
                    [output for output, _, _ in requested], dtype=torch.long
                )
                gathered_states = np.ascontiguousarray(
                    state_map[rows, orientations]
                )
                gathered_policies = np.ascontiguousarray(
                    policy_map[rows, orientations]
                )
                gathered_values = np.ascontiguousarray(value_map[rows])
                states[outputs] = torch.from_numpy(gathered_states)
                policies[outputs] = torch.from_numpy(gathered_policies)
                values[outputs] = torch.from_numpy(gathered_values)
            self._assert_manifest_current()
        except Exception:
            self._stale = True
            raise
        target = torch.device(device)
        return {
            "states": states.to(target),
            "policies": policies.to(target),
            "values": values.to(target),
        }

    def load_batch(
        self,
        count: int,
        *,
        rng: random.Random | None = None,
        mirror_probability: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        plan = self.sample_plan(
            count,
            rng=rng,
            mirror_probability=mirror_probability,
        )
        return self.load_plan(plan, device=device)

    def verify(self, *, deep: bool = True) -> ReplayTensorCacheVerification:
        self._assert_manifest_current()
        if self._manifest is None or self._manifest_file_sha256 is None:
            raise ReplayTensorCacheStaleError("tensor cache has not been built")
        try:
            for chunk in self._manifest["chunks"]:
                self._validate_chunk_record(chunk, deep=deep)
            if deep:
                shards, _, _ = self._current_replay_identity()
                self._raw_shard_stats = self._verify_raw_shards(shards)
        except Exception:
            self._stale = True
            raise
        totals = self._manifest["totals"]
        return ReplayTensorCacheVerification(
            manifest_sha256=str(self._manifest["manifestSha256"]),
            manifest_file_sha256=self._manifest_file_sha256,
            raw_manifest_sha256=str(self._manifest["rawManifestSha256"]),
            raw_window_sha256=str(self._manifest["rawWindowSha256"]),
            games=int(totals["games"]),
            plies=int(totals["plies"]),
            chunks=int(totals["chunks"]),
            cache_bytes=int(totals["cacheBytes"]),
            deep=deep,
        )

    def snapshot(self, *, deep: bool = False) -> dict[str, object]:
        verification = self.verify(deep=deep)
        return {
            "schemaVersion": REPLAY_TENSOR_CACHE_SCHEMA_VERSION,
            "algorithmVersion": REPLAY_TENSOR_CACHE_ALGORITHM_VERSION,
            "samplingAlgorithm": REPLAY_SAMPLING_ALGORITHM,
            "mirrorRngAlgorithm": REPLAY_MIRROR_RNG_ALGORITHM,
            **verification.to_payload(),
            "maximumCacheBytes": self.maximum_cache_bytes,
            "chunkPositions": self.chunk_positions,
            "sourceSha256": self.source_sha256,
            "encoderSha256": self.encoder_sha256,
        }


__all__ = [
    "REPLAY_MIRROR_RNG_ALGORITHM",
    "REPLAY_SAMPLING_ALGORITHM",
    "REPLAY_TENSOR_CACHE_ALGORITHM_VERSION",
    "REPLAY_TENSOR_CACHE_BYTES_PER_POSITION",
    "REPLAY_TENSOR_CACHE_CHUNK_POSITIONS",
    "REPLAY_TENSOR_CACHE_MAX_BYTES",
    "REPLAY_TENSOR_CACHE_SCHEMA_VERSION",
    "ReplayTensorCache",
    "ReplayTensorCacheCapacityError",
    "ReplayTensorCacheConfigurationError",
    "ReplayTensorCacheCorruptionError",
    "ReplayTensorCacheError",
    "ReplayTensorCacheRefresh",
    "ReplayTensorCacheStaleError",
    "ReplayTensorCacheVerification",
    "ReplayTensorSamplePlan",
]
