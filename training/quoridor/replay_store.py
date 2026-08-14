"""Durable, game-sharded replay storage for Phase 4 training.

The existing in-memory :mod:`quoridor.replay` buffer remains the Phase 2/3
compatibility path.  This module stores the source game instead of encoded
training tensors so future encoders, reanalysis, and sample-time augmentation
can reconstruct an exact rule state.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from .codec import (
    ACTION_SPACE_SIZE,
    CODEC_VERSION,
    state_from_serializable,
    state_to_serializable,
    position_hash,
    validate_position_hash,
)
from .game import State, apply_action_index, get_winner, is_terminal, legal_action_indices


REPLAY_STORE_SCHEMA_VERSION = 1
REPLAY_SHARD_SCHEMA_VERSION = 1
_SHARD_MAGIC = b"QRP\x00\x01"
_DIGEST_BYTES = hashlib.sha256().digest_size
_SHARD_GLOB = "game-*.qrp"
_MANIFEST_NAME = "manifest.json"


class ReplayStoreError(RuntimeError):
    """Base class for durable replay failures."""


class ReplayCapacityError(ReplayStoreError):
    """Raised when one game cannot fit the configured replay capacity."""


class ReplayCorruptionError(ReplayStoreError):
    """Raised when committed replay data fails structural or hash validation."""


class ReplayConfigurationError(ReplayStoreError):
    """Raised when an existing store is opened with a different capacity."""


@dataclass(frozen=True, slots=True)
class ReplayPly:
    """One raw-board MCTS decision.

    ``visit_counts`` uses the fixed raw 209-action coordinate system.  It is
    sparse on disk but exposed densely so reconstruction is unambiguous.
    ``model_generation`` and ``model_id`` are per-ply because pool self-play
    may use different model generations for the two players.
    """

    action_index: int
    visit_counts: tuple[int, ...]
    root_value: float
    model_generation: int
    model_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.action_index) is not int or not 0 <= self.action_index < ACTION_SPACE_SIZE:
            raise ValueError(f"action_index must be in [0, {ACTION_SPACE_SIZE})")
        counts = tuple(self.visit_counts)
        if len(counts) != ACTION_SPACE_SIZE:
            raise ValueError(f"visit_counts must contain {ACTION_SPACE_SIZE} entries")
        for count in counts:
            if type(count) is not int or count < 0:
                raise ValueError("visit counts must be non-negative integers")
        if not any(counts):
            raise ValueError("visit_counts must contain positive search mass")
        if counts[self.action_index] <= 0:
            raise ValueError("the selected action must have a positive visit count")
        if isinstance(self.root_value, bool) or not isinstance(
            self.root_value, (int, float)
        ):
            raise TypeError("root_value must be a number")
        root_value = float(self.root_value)
        if not math.isfinite(root_value) or not -1.0 <= root_value <= 1.0:
            raise ValueError("root_value must be finite and in [-1, 1]")
        if type(self.model_generation) is not int or self.model_generation < 0:
            raise ValueError("model_generation must be a non-negative integer")
        if not isinstance(self.model_id, str) or not self.model_id or len(self.model_id) > 256:
            raise ValueError("model_id must be a non-empty string of at most 256 characters")
        metadata = _normalize_json_mapping(self.metadata, "ply metadata")
        object.__setattr__(self, "visit_counts", counts)
        object.__setattr__(self, "root_value", root_value)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class ReplayGame:
    """A complete game beginning at an exact, potentially non-standard state."""

    game_id: str
    initial_state: State
    plies: tuple[ReplayPly, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.game_id, str) or not self.game_id or len(self.game_id) > 128:
            raise ValueError("game_id must be a non-empty string of at most 128 characters")
        if not isinstance(self.initial_state, State):
            raise TypeError("initial_state must be a quoridor.game.State")
        plies = tuple(self.plies)
        if not plies:
            raise ValueError("a replay game must contain at least one ply")
        if any(not isinstance(ply, ReplayPly) for ply in plies):
            raise TypeError("plies must contain ReplayPly values")
        counts = self.initial_state.position_counts
        if not isinstance(counts, Mapping) or not counts:
            raise ValueError("initial_state must contain complete position history")
        for historical_hash, count in counts.items():
            validate_position_hash(historical_hash)
            if type(count) is not int or not 1 <= count < 3:
                raise ValueError("non-terminal position history counts must be 1 or 2")
        if position_hash(self.initial_state) not in counts:
            raise ValueError("position history must contain the initial position")
        if sum(counts.values()) != self.initial_state.move_number + 1:
            raise ValueError(
                "position history occurrences must equal initial move_number + 1"
            )
        metadata = _normalize_json_mapping(self.metadata, "game metadata")
        object.__setattr__(self, "plies", plies)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """A reconstructed training position before encoding or augmentation."""

    game_id: str
    ply_index: int
    state: State
    action_index: int
    visit_counts: tuple[int, ...]
    root_value: float
    outcome: float
    model_generation: int
    model_id: str
    game_metadata: Mapping[str, object]
    ply_metadata: Mapping[str, object]
    augmentation: str = "identity"


@dataclass(frozen=True, slots=True)
class ReplayShardInfo:
    sequence: int
    game_id: str
    file_name: str
    byte_size: int
    sha256: str
    ply_count: int
    minimum_generation: int
    maximum_generation: int


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    game_count: int
    ply_count: int
    committed_bytes: int
    deep: bool


SampleTransform = Callable[[ReplaySample], ReplaySample]


def _normalize_json_value(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
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
            normalized[key] = _normalize_json_value(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} must contain only JSON-compatible values")


def _normalize_json_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = _normalize_json_value(value, label)
    if not isinstance(normalized, dict):  # Defensive narrowing for type checkers.
        raise TypeError(f"{label} must be a mapping")
    return normalized


def _canonical_json(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("replay payload is not canonical JSON") from error
    return encoded.encode("utf-8") + b"\n"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # The rename is already committed. Some filesystems do not support
            # directory fsync, so this durability enhancement must not turn a
            # successful replace into an apparent transaction failure.
            pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _state_to_full_payload(state: State) -> dict[str, object]:
    counts: list[list[object]] = []
    for key, count in sorted(state.position_counts.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("position_counts keys must be non-empty strings")
        if type(count) is not int or count <= 0:
            raise ValueError("position_counts values must be positive integers")
        counts.append([key, count])
    return {
        "state": state_to_serializable(state),
        "positionCounts": counts,
    }


def _state_from_full_payload(payload: object) -> State:
    if not isinstance(payload, Mapping):
        raise TypeError("full state payload must be a mapping")
    dto = payload.get("state")
    counts_payload = payload.get("positionCounts")
    if not isinstance(dto, Mapping):
        raise TypeError("full state payload is missing state DTO")
    if not isinstance(counts_payload, list):
        raise TypeError("positionCounts must be a list")
    counts: dict[str, int] = {}
    for item in counts_payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("positionCounts entries must be [hash, count] pairs")
        key, count = item
        if not isinstance(key, str) or not key:
            raise ValueError("positionCounts hashes must be non-empty strings")
        if key in counts:
            raise ValueError("positionCounts must not contain duplicate hashes")
        if type(count) is not int or count <= 0:
            raise ValueError("positionCounts values must be positive integers")
        counts[key] = count
    state = state_from_serializable(dto)
    return replace(state, position_counts=counts)


def _state_digest(state: State) -> str:
    return hashlib.sha256(_canonical_json(_state_to_full_payload(state))).hexdigest()


def reconstruct_game(game: ReplayGame) -> tuple[State, ...]:
    """Replay and validate a complete game, returning every state including final."""

    state = game.initial_state
    if is_terminal(state):
        raise ValueError("replay initial_state must be non-terminal")
    states = [state]
    for ply_index, ply in enumerate(game.plies):
        if is_terminal(state):
            raise ValueError(f"game contains plies after terminal state at ply {ply_index}")
        legal = frozenset(legal_action_indices(state))
        if ply.action_index not in legal:
            raise ValueError(f"selected action is illegal at ply {ply_index}")
        searched = {
            action_index
            for action_index, count in enumerate(ply.visit_counts)
            if count > 0
        }
        if not searched.issubset(legal):
            raise ValueError(f"visit_counts include illegal actions at ply {ply_index}")
        state = apply_action_index(state, ply.action_index)
        states.append(state)
    if not is_terminal(state) or get_winner(state) is None:
        raise ValueError("replay action sequence must end in a terminal state")
    return tuple(states)


def _sparse_counts(counts: Sequence[int]) -> list[list[int]]:
    return [
        [action_index, count]
        for action_index, count in enumerate(counts)
        if count > 0
    ]


def _dense_counts(payload: object) -> tuple[int, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("visitCounts must be a non-empty sparse list")
    counts = [0] * ACTION_SPACE_SIZE
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("visitCounts entries must be [action, count] pairs")
        action, count = item
        if type(action) is not int or not 0 <= action < ACTION_SPACE_SIZE:
            raise ValueError("visitCounts action is outside the action space")
        if type(count) is not int or count <= 0:
            raise ValueError("sparse visit counts must be positive integers")
        if counts[action] != 0:
            raise ValueError("visitCounts contains a duplicate action")
        counts[action] = count
    return tuple(counts)


def _pack_payload(payload: Mapping[str, object]) -> bytes:
    raw = _canonical_json(payload)
    digest = hashlib.sha256(raw).digest()
    return _SHARD_MAGIC + digest + zlib.compress(raw, level=9)


def _unpack_payload(data: bytes) -> dict[str, object]:
    if len(data) <= len(_SHARD_MAGIC) + _DIGEST_BYTES or not data.startswith(_SHARD_MAGIC):
        raise ReplayCorruptionError("replay shard has an invalid magic header")
    expected_digest = data[len(_SHARD_MAGIC) : len(_SHARD_MAGIC) + _DIGEST_BYTES]
    compressed = data[len(_SHARD_MAGIC) + _DIGEST_BYTES :]
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as error:
        raise ReplayCorruptionError("replay shard compression is corrupt") from error
    if hashlib.sha256(raw).digest() != expected_digest:
        raise ReplayCorruptionError("replay shard payload checksum mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayCorruptionError("replay shard payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ReplayCorruptionError("replay shard payload must be a JSON object")
    return payload


def _payload_from_game(game: ReplayGame, sequence: int, final_state: State) -> dict[str, object]:
    return {
        "schemaVersion": REPLAY_SHARD_SCHEMA_VERSION,
        "codecVersion": CODEC_VERSION,
        "actionSpaceSize": ACTION_SPACE_SIZE,
        "sequence": sequence,
        "gameId": game.game_id,
        "initialState": _state_to_full_payload(game.initial_state),
        "metadata": dict(game.metadata),
        "plies": [
            {
                "actionIndex": ply.action_index,
                "visitCounts": _sparse_counts(ply.visit_counts),
                "rootValue": ply.root_value,
                "model": {
                    "generation": ply.model_generation,
                    "id": ply.model_id,
                },
                "metadata": dict(ply.metadata),
            }
            for ply in game.plies
        ],
        "result": {
            "winner": get_winner(final_state),
            "terminationReason": final_state.termination_reason,
            "moveNumber": final_state.move_number,
            "fullStateSha256": _state_digest(final_state),
        },
    }


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _game_from_payload(
    payload: Mapping[str, object],
) -> tuple[ReplayGame, int, tuple[State, ...]]:
    if payload.get("schemaVersion") != REPLAY_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported replay shard schema version")
    if payload.get("codecVersion") != CODEC_VERSION:
        raise ValueError("unsupported replay shard codec version")
    if payload.get("actionSpaceSize") != ACTION_SPACE_SIZE:
        raise ValueError("replay shard action space does not match this engine")
    sequence = _require_int(payload.get("sequence"), "sequence")
    game_id = payload.get("gameId")
    if not isinstance(game_id, str):
        raise TypeError("gameId must be a string")
    raw_plies = payload.get("plies")
    if not isinstance(raw_plies, list):
        raise TypeError("plies must be a list")
    plies: list[ReplayPly] = []
    for raw_ply in raw_plies:
        if not isinstance(raw_ply, Mapping):
            raise TypeError("ply payload must be a mapping")
        model = raw_ply.get("model")
        if not isinstance(model, Mapping):
            raise TypeError("ply model metadata must be a mapping")
        plies.append(
            ReplayPly(
                action_index=_require_int(raw_ply.get("actionIndex"), "actionIndex"),
                visit_counts=_dense_counts(raw_ply.get("visitCounts")),
                root_value=_require_float(raw_ply.get("rootValue"), "rootValue"),
                model_generation=_require_int(model.get("generation"), "model generation"),
                model_id=model.get("id"),
                metadata=_normalize_json_mapping(raw_ply.get("metadata", {}), "ply metadata"),
            )
        )
    game = ReplayGame(
        game_id=game_id,
        initial_state=_state_from_full_payload(payload.get("initialState")),
        plies=tuple(plies),
        metadata=_normalize_json_mapping(payload.get("metadata", {}), "game metadata"),
    )
    states = reconstruct_game(game)
    final_state = states[-1]
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    if result.get("winner") != get_winner(final_state):
        raise ValueError("stored winner does not match reconstructed game")
    if result.get("terminationReason") != final_state.termination_reason:
        raise ValueError("stored termination reason does not match reconstructed game")
    if result.get("moveNumber") != final_state.move_number:
        raise ValueError("stored move number does not match reconstructed game")
    if result.get("fullStateSha256") != _state_digest(final_state):
        raise ValueError("stored final state digest does not match reconstructed game")
    return game, sequence, states


def _entry_to_payload(entry: ReplayShardInfo) -> dict[str, object]:
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


def _entry_from_payload(payload: object) -> ReplayShardInfo:
    if not isinstance(payload, Mapping):
        raise TypeError("manifest shard entry must be a mapping")
    file_name = payload.get("file")
    game_id = payload.get("gameId")
    digest = payload.get("sha256")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError("manifest shard file must be a plain file name")
    if not file_name.startswith("game-") or not file_name.endswith(".qrp"):
        raise ValueError("manifest shard file has an invalid name")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("manifest gameId must be a non-empty string")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("manifest shard sha256 is invalid")
    return ReplayShardInfo(
        sequence=_require_int(payload.get("sequence"), "manifest sequence"),
        game_id=game_id,
        file_name=file_name,
        byte_size=_require_int(payload.get("bytes"), "manifest byte size", minimum=1),
        sha256=digest,
        ply_count=_require_int(payload.get("plies"), "manifest ply count", minimum=1),
        minimum_generation=_require_int(
            payload.get("minimumGeneration"), "minimum generation"
        ),
        maximum_generation=_require_int(
            payload.get("maximumGeneration"), "maximum generation"
        ),
    )


class ReplayStore:
    """A bounded collection of atomic, self-validating complete-game shards.

    Capacity eviction is FIFO and always removes whole games.  ``max_bytes``
    covers committed shard files plus the manifest, not transient atomic-write
    files.  A crash before manifest replacement leaves the previous manifest
    authoritative; the uncommitted shard is discarded on the next open.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_games: int,
        max_plies: int,
        max_bytes: int,
        recover: bool = False,
    ) -> None:
        for value, label in (
            (max_games, "max_games"),
            (max_plies, "max_plies"),
            (max_bytes, "max_bytes"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.root = Path(root)
        self.max_games = max_games
        self.max_plies = max_plies
        self.max_bytes = max_bytes
        self.manifest_path = self.root / _MANIFEST_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: list[ReplayShardInfo] = []
        self._next_sequence = 0
        self._remove_temporary_files()

        if self.manifest_path.exists():
            try:
                self._load_manifest()
            except ReplayCorruptionError:
                if not recover:
                    raise
                self._recover_from_shards()
        else:
            shards = tuple(self.root.glob(_SHARD_GLOB))
            if shards:
                if not recover:
                    raise ReplayCorruptionError(
                        "replay shards exist without a manifest; reopen with recover=True"
                    )
                self._recover_from_shards()
            else:
                self._write_manifest([], 0)
        self._remove_uncommitted_shards()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def game_count(self) -> int:
        return len(self._entries)

    @property
    def ply_count(self) -> int:
        return sum(entry.ply_count for entry in self._entries)

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(entry.game_id for entry in self._entries)

    @property
    def shards(self) -> tuple[ReplayShardInfo, ...]:
        return tuple(self._entries)

    @property
    def committed_bytes(self) -> int:
        manifest_size = self.manifest_path.stat().st_size if self.manifest_path.exists() else 0
        return manifest_size + sum(entry.byte_size for entry in self._entries)

    def append(self, game: ReplayGame) -> ReplayShardInfo:
        """Validate and atomically commit one complete game, pruning FIFO."""

        if not isinstance(game, ReplayGame):
            raise TypeError("game must be a ReplayGame")
        if game.game_id in self.game_ids:
            raise ValueError(f"duplicate replay game_id: {game.game_id}")
        final_state = reconstruct_game(game)[-1]
        sequence = self._next_sequence
        payload = _payload_from_game(game, sequence, final_state)
        shard_bytes = _pack_payload(payload)
        file_token = hashlib.sha256(game.game_id.encode("utf-8")).hexdigest()[:12]
        file_name = f"game-{sequence:012d}-{file_token}.qrp"
        generations = [ply.model_generation for ply in game.plies]
        entry = ReplayShardInfo(
            sequence=sequence,
            game_id=game.game_id,
            file_name=file_name,
            byte_size=len(shard_bytes),
            sha256=hashlib.sha256(shard_bytes).hexdigest(),
            ply_count=len(game.plies),
            minimum_generation=min(generations),
            maximum_generation=max(generations),
        )
        kept, evicted = self._bounded_entries(
            [*self._entries, entry],
            next_sequence=sequence + 1,
            required_entry=entry,
        )
        manifest_bytes = self._manifest_bytes(kept, sequence + 1)
        required_free = len(shard_bytes) + len(manifest_bytes)
        if shutil.disk_usage(self.root).free < required_free:
            raise ReplayCapacityError(
                f"insufficient free disk for atomic replay commit: need {required_free} bytes"
            )
        destination = self.root / file_name
        if destination.exists():
            raise ReplayCorruptionError(f"replay shard already exists: {file_name}")

        _atomic_write_bytes(destination, shard_bytes)
        try:
            _atomic_write_bytes(self.manifest_path, manifest_bytes)
        except Exception:
            destination.unlink(missing_ok=True)
            _fsync_directory(self.root)
            raise
        self._entries = kept
        self._next_sequence = sequence + 1
        for old_entry in evicted:
            (self.root / old_entry.file_name).unlink(missing_ok=True)
        _fsync_directory(self.root)
        return entry

    def load_game(self, game_id: str) -> ReplayGame:
        entry = self._entry_for_game(game_id)
        game, _, _ = self._read_entry(entry)
        return game

    def load_game_with_states(
        self, game_id: str
    ) -> tuple[ReplayGame, tuple[State, ...]]:
        """Decode and deeply validate one shard, retaining its reconstructed states."""

        entry = self._entry_for_game(game_id)
        game, _, states = self._read_entry(entry)
        return game, states

    def iter_games(self) -> Iterator[ReplayGame]:
        for entry in self._entries:
            game, _, _ = self._read_entry(entry)
            yield game

    def sample_plies(
        self,
        count: int,
        *,
        rng: random.Random | None = None,
        transform: SampleTransform | None = None,
    ) -> tuple[ReplaySample, ...]:
        """Uniformly sample plies without replacement.

        ``transform`` is the sample-time augmentation extension point.  A
        future mirror transform must update state, action, and visit counts
        together and can mark ``augmentation="mirror"`` on its result.
        """

        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable")
        total = self.ply_count
        if total == 0:
            raise ValueError("cannot sample an empty replay store")
        selected_count = min(count, total)
        generator = rng or random.SystemRandom()
        selected = generator.sample(range(total), selected_count)
        cumulative: list[int] = []
        running = 0
        for entry in self._entries:
            running += entry.ply_count
            cumulative.append(running)

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, global_index in enumerate(selected):
            entry_index = bisect.bisect_right(cumulative, global_index)
            previous = cumulative[entry_index - 1] if entry_index else 0
            grouped.setdefault(entry_index, []).append(
                (output_index, global_index - previous)
            )
        samples: list[ReplaySample | None] = [None] * selected_count
        for entry_index, requested in grouped.items():
            game, _, states = self._read_entry(self._entries[entry_index])
            final_state = states[-1]
            winner = get_winner(final_state)
            if winner is None:
                raise ReplayCorruptionError("reconstructed replay has no winner")
            for output_index, ply_index in requested:
                state = states[ply_index]
                ply = game.plies[ply_index]
                sample = ReplaySample(
                    game_id=game.game_id,
                    ply_index=ply_index,
                    state=state,
                    action_index=ply.action_index,
                    visit_counts=ply.visit_counts,
                    root_value=ply.root_value,
                    outcome=1.0 if state.current_player == winner else -1.0,
                    model_generation=ply.model_generation,
                    model_id=ply.model_id,
                    game_metadata=game.metadata,
                    ply_metadata=ply.metadata,
                )
                if transform is not None:
                    sample = transform(sample)
                    if not isinstance(sample, ReplaySample):
                        raise TypeError("sample transform must return ReplaySample")
                samples[output_index] = sample
        if any(sample is None for sample in samples):
            raise ReplayCorruptionError("failed to reconstruct every sampled replay ply")
        return tuple(sample for sample in samples if sample is not None)

    def verify(
        self,
        *,
        deep: bool = True,
        game_validator: (
            Callable[[ReplayGame, tuple[State, ...]], None] | None
        ) = None,
    ) -> ReplayVerification:
        """Verify the store and optionally validate each deeply decoded game."""

        if game_validator is not None and not callable(game_validator):
            raise TypeError("game_validator must be callable")
        if game_validator is not None and not deep:
            raise ValueError("game_validator requires deep verification")
        self._verify_entry_files(deep=deep, game_validator=game_validator)
        return ReplayVerification(
            game_count=self.game_count,
            ply_count=self.ply_count,
            committed_bytes=self.committed_bytes,
            deep=deep,
        )

    def _entry_for_game(self, game_id: str) -> ReplayShardInfo:
        for entry in self._entries:
            if entry.game_id == game_id:
                return entry
        raise KeyError(game_id)

    def _read_entry(
        self, entry: ReplayShardInfo
    ) -> tuple[ReplayGame, int, tuple[State, ...]]:
        path = self.root / entry.file_name
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ReplayCorruptionError(f"cannot read replay shard {entry.file_name}") from error
        if len(data) != entry.byte_size:
            raise ReplayCorruptionError(f"replay shard size mismatch: {entry.file_name}")
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise ReplayCorruptionError(f"replay shard checksum mismatch: {entry.file_name}")
        payload = _unpack_payload(data)
        try:
            game, sequence, states = _game_from_payload(payload)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ReplayCorruptionError(
                f"replay shard validation failed: {entry.file_name}: {error}"
            ) from error
        if sequence != entry.sequence or game.game_id != entry.game_id:
            raise ReplayCorruptionError(f"replay shard identity mismatch: {entry.file_name}")
        if len(game.plies) != entry.ply_count:
            raise ReplayCorruptionError(f"replay shard ply count mismatch: {entry.file_name}")
        generations = [ply.model_generation for ply in game.plies]
        if (
            min(generations) != entry.minimum_generation
            or max(generations) != entry.maximum_generation
        ):
            raise ReplayCorruptionError(
                f"replay shard generation metadata mismatch: {entry.file_name}"
            )
        return game, sequence, states

    def _capacity_payload(self) -> dict[str, int]:
        return {
            "games": self.max_games,
            "plies": self.max_plies,
            "bytes": self.max_bytes,
        }

    def _manifest_payload(
        self, entries: Sequence[ReplayShardInfo], next_sequence: int
    ) -> dict[str, object]:
        return {
            "schemaVersion": REPLAY_STORE_SCHEMA_VERSION,
            "shardSchemaVersion": REPLAY_SHARD_SCHEMA_VERSION,
            "codecVersion": CODEC_VERSION,
            "actionSpaceSize": ACTION_SPACE_SIZE,
            "capacity": self._capacity_payload(),
            "nextSequence": next_sequence,
            "totals": {
                "games": len(entries),
                "plies": sum(entry.ply_count for entry in entries),
                "shardBytes": sum(entry.byte_size for entry in entries),
            },
            "shards": [_entry_to_payload(entry) for entry in entries],
        }

    def _manifest_bytes(
        self, entries: Sequence[ReplayShardInfo], next_sequence: int
    ) -> bytes:
        return _canonical_json(self._manifest_payload(entries, next_sequence))

    def _write_manifest(
        self, entries: Sequence[ReplayShardInfo], next_sequence: int
    ) -> None:
        manifest_bytes = self._manifest_bytes(entries, next_sequence)
        committed = len(manifest_bytes) + sum(entry.byte_size for entry in entries)
        if committed > self.max_bytes:
            raise ReplayCapacityError(
                f"replay byte capacity exceeded: {committed} > {self.max_bytes}"
            )
        _atomic_write_bytes(self.manifest_path, manifest_bytes)
        self._entries = list(entries)
        self._next_sequence = next_sequence

    def _bounded_entries(
        self,
        entries: Sequence[ReplayShardInfo],
        *,
        next_sequence: int,
        required_entry: ReplayShardInfo,
    ) -> tuple[list[ReplayShardInfo], list[ReplayShardInfo]]:
        kept = list(entries)
        evicted: list[ReplayShardInfo] = []
        while kept:
            manifest_size = len(self._manifest_bytes(kept, next_sequence))
            over_capacity = (
                len(kept) > self.max_games
                or sum(entry.ply_count for entry in kept) > self.max_plies
                or manifest_size + sum(entry.byte_size for entry in kept) > self.max_bytes
            )
            if not over_capacity:
                break
            oldest = kept.pop(0)
            evicted.append(oldest)
        if required_entry not in kept:
            raise ReplayCapacityError(
                "one replay game exceeds the configured game, ply, or byte capacity"
            )
        return kept, evicted

    def _load_manifest(self) -> None:
        try:
            raw = self.manifest_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayCorruptionError("replay manifest is unreadable") from error
        if not isinstance(payload, Mapping):
            raise ReplayCorruptionError("replay manifest must be a JSON object")
        try:
            if payload.get("schemaVersion") != REPLAY_STORE_SCHEMA_VERSION:
                raise ValueError("unsupported replay manifest schema version")
            if payload.get("shardSchemaVersion") != REPLAY_SHARD_SCHEMA_VERSION:
                raise ValueError("unsupported replay shard schema version")
            if payload.get("codecVersion") != CODEC_VERSION:
                raise ValueError("replay codec version mismatch")
            if payload.get("actionSpaceSize") != ACTION_SPACE_SIZE:
                raise ValueError("replay action space mismatch")
            capacity = payload.get("capacity")
            if not isinstance(capacity, Mapping):
                raise TypeError("manifest capacity must be a mapping")
            stored_capacity = {
                "games": _require_int(capacity.get("games"), "capacity.games", minimum=1),
                "plies": _require_int(capacity.get("plies"), "capacity.plies", minimum=1),
                "bytes": _require_int(capacity.get("bytes"), "capacity.bytes", minimum=1),
            }
            if stored_capacity != self._capacity_payload():
                raise ReplayConfigurationError(
                    f"replay capacity mismatch: stored={stored_capacity}, "
                    f"requested={self._capacity_payload()}"
                )
            raw_entries = payload.get("shards")
            if not isinstance(raw_entries, list):
                raise TypeError("manifest shards must be a list")
            entries = [_entry_from_payload(entry) for entry in raw_entries]
            next_sequence = _require_int(payload.get("nextSequence"), "nextSequence")
            totals = payload.get("totals")
            if not isinstance(totals, Mapping):
                raise TypeError("manifest totals must be a mapping")
            expected_totals = {
                "games": len(entries),
                "plies": sum(entry.ply_count for entry in entries),
                "shardBytes": sum(entry.byte_size for entry in entries),
            }
            if dict(totals) != expected_totals:
                raise ValueError("manifest totals do not match shard entries")
            sequences = [entry.sequence for entry in entries]
            game_ids = [entry.game_id for entry in entries]
            file_names = [entry.file_name for entry in entries]
            if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
                raise ValueError("manifest shard sequences must be unique and sorted")
            if len(set(game_ids)) != len(game_ids) or len(set(file_names)) != len(file_names):
                raise ValueError("manifest game IDs and file names must be unique")
            if entries and next_sequence <= entries[-1].sequence:
                raise ValueError("nextSequence must follow every committed shard")
            if len(raw) + sum(entry.byte_size for entry in entries) > self.max_bytes:
                raise ValueError("committed replay exceeds configured byte capacity")
        except ReplayConfigurationError:
            raise
        except (TypeError, ValueError) as error:
            raise ReplayCorruptionError(f"invalid replay manifest: {error}") from error
        self._entries = entries
        self._next_sequence = next_sequence
        self._verify_entry_files(deep=False)

    def _verify_entry_files(
        self,
        *,
        deep: bool,
        game_validator: (
            Callable[[ReplayGame, tuple[State, ...]], None] | None
        ) = None,
    ) -> None:
        for entry in self._entries:
            path = self.root / entry.file_name
            if not path.is_file():
                raise ReplayCorruptionError(f"replay shard is missing: {entry.file_name}")
            if path.stat().st_size != entry.byte_size:
                raise ReplayCorruptionError(f"replay shard size mismatch: {entry.file_name}")
            if deep:
                game, _, states = self._read_entry(entry)
                if game_validator is not None:
                    game_validator(game, states)

    def _recover_from_shards(self) -> None:
        recovered: list[ReplayShardInfo] = []
        for path in sorted(self.root.glob(_SHARD_GLOB)):
            try:
                data = path.read_bytes()
                payload = _unpack_payload(data)
                game, sequence, _ = _game_from_payload(payload)
            except (OSError, KeyError, TypeError, ValueError, ReplayCorruptionError) as error:
                raise ReplayCorruptionError(
                    f"cannot recover replay shard {path.name}: {error}"
                ) from error
            generations = [ply.model_generation for ply in game.plies]
            recovered.append(
                ReplayShardInfo(
                    sequence=sequence,
                    game_id=game.game_id,
                    file_name=path.name,
                    byte_size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    ply_count=len(game.plies),
                    minimum_generation=min(generations),
                    maximum_generation=max(generations),
                )
            )
        recovered.sort(key=lambda entry: entry.sequence)
        if len({entry.sequence for entry in recovered}) != len(recovered):
            raise ReplayCorruptionError("cannot recover duplicate replay sequences")
        if len({entry.game_id for entry in recovered}) != len(recovered):
            raise ReplayCorruptionError("cannot recover duplicate replay game IDs")
        next_sequence = recovered[-1].sequence + 1 if recovered else 0
        if recovered:
            kept, evicted = self._bounded_entries(
                recovered,
                next_sequence=next_sequence,
                required_entry=recovered[-1],
            )
            if evicted:
                raise ReplayConfigurationError(
                    "recovered replay exceeds the requested capacity; "
                    "reopen with the original or a larger capacity"
                )
        self._write_manifest(recovered, next_sequence)
        _fsync_directory(self.root)

    def _remove_temporary_files(self) -> None:
        for path in self.root.glob(".*.tmp"):
            if path.is_file():
                path.unlink(missing_ok=True)

    def _remove_uncommitted_shards(self) -> None:
        committed = {entry.file_name for entry in self._entries}
        removed = False
        for path in self.root.glob(_SHARD_GLOB):
            if path.name not in committed:
                path.unlink(missing_ok=True)
                removed = True
        if removed:
            _fsync_directory(self.root)


__all__ = [
    "REPLAY_STORE_SCHEMA_VERSION",
    "REPLAY_SHARD_SCHEMA_VERSION",
    "ReplayStoreError",
    "ReplayCapacityError",
    "ReplayCorruptionError",
    "ReplayConfigurationError",
    "ReplayPly",
    "ReplayGame",
    "ReplaySample",
    "ReplayShardInfo",
    "ReplayVerification",
    "SampleTransform",
    "ReplayStore",
    "reconstruct_game",
]
