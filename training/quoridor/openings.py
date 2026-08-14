"""Versioned, replay-verifiable opening suites for strength evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .codec import (
    CODEC_VERSION,
    current_repetition_count,
    mirror_state,
    position_hash,
    state_from_serializable,
    state_to_serializable,
)
from .game import (
    BOARD_SIZE,
    MAX_HALF_MOVES,
    REPETITION_LIMIT,
    STARTING_WALLS,
    WALL_GRID_SIZE,
    State,
    apply_action_index,
    create_initial_state,
    is_terminal,
)
from .storage import atomic_json_save


OPENING_SCHEMA_VERSION = 1
PAIRING_CONTRACT = "candidate-player-swap-v1"
SUITE_NAMES = frozenset(
    {
        "dev",
        "final",
        "recovery-dev-v1",
        "recovery-final-v1",
        "recovery-fresh-dev-v1",
    }
)

_HASH_PREFIX = "sha256:"
_STATE_KEYS = frozenset(
    {
        "codecVersion",
        "pawns",
        "wallsRemaining",
        "horizontalWalls",
        "verticalWalls",
        "currentPlayer",
        "winner",
        "moveNumber",
        "terminationReason",
        "repetitionCount",
        "positionCounts",
    }
)


class OpeningValidationError(ValueError):
    """Raised when an opening asset does not satisfy its frozen contract."""


@dataclass(frozen=True, slots=True)
class OpeningLeg:
    """One half of a paired match from the candidate's perspective."""

    opening_id: str
    leg: int
    candidate_player: int
    opponent_player: int
    candidate_to_move: bool


@dataclass(frozen=True, slots=True)
class Opening:
    """A reachable non-terminal state and the prefix that proves its history."""

    id: str
    content_hash: str
    suite: str
    starting_player: int
    actions: tuple[int, ...]
    state: State

    def paired_legs(self) -> tuple[OpeningLeg, OpeningLeg]:
        """Return the mandatory two games with candidate ownership exchanged."""

        side_to_move = self.state.current_player
        return (
            OpeningLeg(
                opening_id=self.id,
                leg=0,
                candidate_player=side_to_move,
                opponent_player=1 - side_to_move,
                candidate_to_move=True,
            ),
            OpeningLeg(
                opening_id=self.id,
                leg=1,
                candidate_player=1 - side_to_move,
                opponent_player=side_to_move,
                candidate_to_move=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class OpeningSuite:
    """A content-addressed dev or final collection with one pairing contract."""

    name: str
    openings: tuple[Opening, ...]
    content_hash: str
    schema_version: int = OPENING_SCHEMA_VERSION
    pairing_contract: str = PAIRING_CONTRACT


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_hash(payload: object) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def _validate_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(_HASH_PREFIX):
        raise OpeningValidationError(f"{label} must be a sha256 hash")
    digest = value[len(_HASH_PREFIX) :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OpeningValidationError(f"{label} must contain 64 lowercase hex digits")
    return value


def _validate_suite_name(value: object) -> str:
    if not isinstance(value, str) or value not in SUITE_NAMES:
        raise OpeningValidationError(f"suite must be one of {sorted(SUITE_NAMES)}")
    return str(value)


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise OpeningValidationError(
            f"{label} keys do not match schema; missing={missing}, extra={extra}"
        )


def _require_player(value: object, label: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise OpeningValidationError(f"{label} must be 0 or 1")
    return value


def _normalize_actions(actions: Iterable[int]) -> tuple[int, ...]:
    if isinstance(actions, (str, bytes)):
        raise OpeningValidationError("actions must be an iterable of action indices")
    try:
        normalized = tuple(actions)
    except TypeError as error:
        raise OpeningValidationError("actions must be iterable") from error
    for ply, action in enumerate(normalized):
        if type(action) is not int or not 0 <= action < 209:
            raise OpeningValidationError(
                f"action at ply {ply} must be an integer in [0, 208]"
            )
    return normalized


def _parse_wall_key(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise OpeningValidationError(f"{label} must be a string")
    parts = value.split(",")
    if len(parts) != 2:
        raise OpeningValidationError(f"invalid wall key in {label}: {value!r}")
    try:
        row, col = (int(part) for part in parts)
    except ValueError as error:
        raise OpeningValidationError(f"invalid wall key in {label}: {value!r}") from error
    if not 0 <= row < WALL_GRID_SIZE or not 0 <= col < WALL_GRID_SIZE:
        raise OpeningValidationError(f"wall key outside board in {label}: {value!r}")
    if value != f"{row},{col}":
        raise OpeningValidationError(f"wall key is not canonical in {label}: {value!r}")
    return row, col


def _validate_position_hash(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise OpeningValidationError(f"{label} must be a string")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise OpeningValidationError(f"{label} is not valid JSON") from error
    if not isinstance(payload, list) or len(payload) != 10:
        raise OpeningValidationError(f"{label} has an invalid codec shape")
    if _canonical_json(payload) != value:
        raise OpeningValidationError(f"{label} is not canonically encoded")

    version, player, p0_row, p0_col, p1_row, p1_col, walls0, walls1, horizontal, vertical = payload
    if version != CODEC_VERSION:
        raise OpeningValidationError(f"{label} has an unsupported codec version")
    _require_player(player, f"{label} current player")
    for coordinate in (p0_row, p0_col, p1_row, p1_col):
        if type(coordinate) is not int or not 0 <= coordinate < BOARD_SIZE:
            raise OpeningValidationError(f"{label} contains an invalid pawn coordinate")
    if (p0_row, p0_col) == (p1_row, p1_col):
        raise OpeningValidationError(f"{label} places both pawns on one square")
    for remaining in (walls0, walls1):
        if type(remaining) is not int or not 0 <= remaining <= STARTING_WALLS:
            raise OpeningValidationError(f"{label} contains an invalid wall count")
    for walls, orientation in ((horizontal, "horizontal"), (vertical, "vertical")):
        if (
            not isinstance(walls, list)
            or any(not isinstance(wall, str) for wall in walls)
            or walls != sorted(walls)
            or len(walls) != len(set(walls))
        ):
            raise OpeningValidationError(f"{label} {orientation} walls are not canonical")
        for index, wall in enumerate(walls):
            _parse_wall_key(wall, f"{label} {orientation} wall {index}")
    return value


def state_to_opening_payload(state: State) -> dict[str, object]:
    """Serialize every State field, including the complete repetition map."""

    payload = state_to_serializable(state)
    try:
        counts = sorted(state.position_counts.items())
    except (AttributeError, TypeError) as error:
        raise OpeningValidationError("state.position_counts must be a mapping") from error
    payload["positionCounts"] = [
        {"hash": position, "count": count} for position, count in counts
    ]
    return payload


def state_from_opening_payload(payload: Mapping[str, object]) -> State:
    """Restore an opening state without truncating its repetition history."""

    if not isinstance(payload, Mapping):
        raise OpeningValidationError("opening state must be a mapping")
    _require_exact_keys(payload, _STATE_KEYS, "opening state")
    try:
        state = state_from_serializable(payload)
    except (TypeError, ValueError) as error:
        raise OpeningValidationError(f"invalid opening state: {error}") from error

    raw_counts = payload["positionCounts"]
    if not isinstance(raw_counts, list) or not raw_counts:
        raise OpeningValidationError("positionCounts must be a non-empty list")
    counts: dict[str, int] = {}
    ordered_hashes: list[str] = []
    for index, entry in enumerate(raw_counts):
        if not isinstance(entry, Mapping):
            raise OpeningValidationError(f"positionCounts[{index}] must be a mapping")
        _require_exact_keys(
            entry,
            frozenset({"hash", "count"}),
            f"positionCounts[{index}]",
        )
        historical_hash = _validate_position_hash(
            entry["hash"], f"positionCounts[{index}].hash"
        )
        count = entry["count"]
        if type(count) is not int or not 1 <= count < REPETITION_LIMIT:
            raise OpeningValidationError(
                f"positionCounts[{index}].count must be 1 or 2"
            )
        if historical_hash in counts:
            raise OpeningValidationError("positionCounts contains duplicate hashes")
        counts[historical_hash] = count
        ordered_hashes.append(historical_hash)
    if ordered_hashes != sorted(ordered_hashes):
        raise OpeningValidationError("positionCounts must be sorted by hash")

    state = replace(state, position_counts=counts)
    current_hash = position_hash(state)
    if current_hash not in counts:
        raise OpeningValidationError("positionCounts omits the current position")
    if current_repetition_count(state) != payload["repetitionCount"]:
        raise OpeningValidationError(
            "repetitionCount does not match the complete positionCounts map"
        )
    if state.winner is not None or state.termination_reason is not None or is_terminal(state):
        raise OpeningValidationError("evaluation openings must be non-terminal")
    if state.move_number >= MAX_HALF_MOVES:
        raise OpeningValidationError("opening has reached the half-move limit")
    if sum(counts.values()) != state.move_number + 1:
        raise OpeningValidationError(
            "positionCounts occurrence total must equal moveNumber + 1"
        )
    return state


def _replay(starting_player: int, actions: Sequence[int]) -> State:
    try:
        state = create_initial_state(starting_player)
    except ValueError as error:
        raise OpeningValidationError(str(error)) from error
    for ply, action in enumerate(actions):
        try:
            state = apply_action_index(state, action)
        except (TypeError, ValueError) as error:
            raise OpeningValidationError(
                f"opening action at ply {ply} is invalid: {error}"
            ) from error
        if is_terminal(state):
            raise OpeningValidationError(
                f"opening action history becomes terminal at ply {ply}"
            )
    return state


def _opening_core(
    starting_player: int,
    actions: Sequence[int],
    state: State,
) -> dict[str, object]:
    return {
        "startingPlayer": starting_player,
        "actions": list(actions),
        "state": state_to_opening_payload(state),
    }


def _opening_id(content_hash: str) -> str:
    return f"opening-{content_hash[len(_HASH_PREFIX):len(_HASH_PREFIX) + 24]}"


def _opening_entry(opening: Opening) -> dict[str, object]:
    return {
        "id": opening.id,
        "hash": opening.content_hash,
        **_opening_core(opening.starting_player, opening.actions, opening.state),
    }


def generate_opening(
    actions: Iterable[int],
    *,
    suite: str,
    starting_player: int = 0,
) -> Opening:
    """Replay an action prefix and return a content-addressed opening."""

    suite = _validate_suite_name(suite)
    starting_player = _require_player(starting_player, "starting_player")
    normalized_actions = _normalize_actions(actions)
    state = _replay(starting_player, normalized_actions)
    content_hash = _content_hash(
        _opening_core(starting_player, normalized_actions, state)
    )
    opening = Opening(
        id=_opening_id(content_hash),
        content_hash=content_hash,
        suite=suite,
        starting_player=starting_player,
        actions=normalized_actions,
        state=state,
    )
    validate_opening(opening)
    return opening


def validate_opening(opening: Opening) -> None:
    """Verify identifiers, replay provenance, and the complete stored State."""

    if not isinstance(opening, Opening):
        raise OpeningValidationError("opening must be an Opening")
    _validate_suite_name(opening.suite)
    starting_player = _require_player(opening.starting_player, "starting_player")
    actions = _normalize_actions(opening.actions)
    content_hash = _validate_hash(opening.content_hash, "opening hash")
    if opening.id != _opening_id(content_hash):
        raise OpeningValidationError("opening id does not match its content hash")

    replayed = _replay(starting_player, actions)
    try:
        stored_payload = state_to_opening_payload(opening.state)
        restored = state_from_opening_payload(stored_payload)
    except (TypeError, ValueError) as error:
        if isinstance(error, OpeningValidationError):
            raise
        raise OpeningValidationError(f"invalid stored opening state: {error}") from error
    if state_to_opening_payload(restored) != state_to_opening_payload(replayed):
        raise OpeningValidationError(
            "stored opening state/history does not match replayed actions"
        )
    expected_hash = _content_hash(_opening_core(starting_player, actions, replayed))
    if content_hash != expected_hash:
        raise OpeningValidationError("opening hash does not match its content")


def _suite_core(name: str, openings: Sequence[Opening]) -> dict[str, object]:
    return {
        "schemaVersion": OPENING_SCHEMA_VERSION,
        "suite": name,
        "pairingContract": PAIRING_CONTRACT,
        "openings": [_opening_entry(opening) for opening in openings],
    }


def build_opening_suite(
    openings: Iterable[Opening],
    *,
    suite: str,
) -> OpeningSuite:
    """Build a deterministically ordered, content-addressed suite manifest."""

    suite = _validate_suite_name(suite)
    normalized = tuple(sorted(tuple(openings), key=lambda opening: opening.id))
    if not normalized:
        raise OpeningValidationError("an opening suite must not be empty")
    content_hash = _content_hash(_suite_core(suite, normalized))
    result = OpeningSuite(
        name=suite,
        openings=normalized,
        content_hash=content_hash,
    )
    validate_opening_suite(result)
    return result


def generate_opening_suite(
    action_histories: Iterable[Iterable[int]],
    *,
    suite: str,
    starting_player: int = 0,
) -> OpeningSuite:
    """Generate a suite from replayable action prefixes."""

    return build_opening_suite(
        (
            generate_opening(
                actions,
                suite=suite,
                starting_player=starting_player,
            )
            for actions in action_histories
        ),
        suite=suite,
    )


def validate_opening_suite(suite: OpeningSuite) -> None:
    if not isinstance(suite, OpeningSuite):
        raise OpeningValidationError("suite must be an OpeningSuite")
    name = _validate_suite_name(suite.name)
    if suite.schema_version != OPENING_SCHEMA_VERSION:
        raise OpeningValidationError("unsupported opening schema version")
    if suite.pairing_contract != PAIRING_CONTRACT:
        raise OpeningValidationError("unsupported opening pairing contract")
    if not isinstance(suite.openings, tuple) or not suite.openings:
        raise OpeningValidationError("suite openings must be a non-empty tuple")
    _validate_hash(suite.content_hash, "suite hash")

    identifiers: set[str] = set()
    hashes: set[str] = set()
    for opening in suite.openings:
        validate_opening(opening)
        if opening.suite != name:
            raise OpeningValidationError("opening suite label does not match manifest")
        if opening.id in identifiers or opening.content_hash in hashes:
            raise OpeningValidationError("suite contains a duplicate opening")
        identifiers.add(opening.id)
        hashes.add(opening.content_hash)
    if tuple(sorted(suite.openings, key=lambda opening: opening.id)) != suite.openings:
        raise OpeningValidationError("suite openings must be sorted by id")
    expected_hash = _content_hash(_suite_core(name, suite.openings))
    if suite.content_hash != expected_hash:
        raise OpeningValidationError("suite hash does not match its content")


def validate_suite_partition(dev: OpeningSuite, final: OpeningSuite) -> None:
    """Require valid, disjoint dev/final assets to prevent evaluation leakage."""

    validate_opening_suite(dev)
    validate_opening_suite(final)
    if dev.name != "dev" or final.name != "final":
        raise OpeningValidationError("suite partition must be ordered as dev, final")
    dev_ids = {opening.id for opening in dev.openings}
    final_ids = {opening.id for opening in final.openings}
    dev_hashes = {opening.content_hash for opening in dev.openings}
    final_hashes = {opening.content_hash for opening in final.openings}
    if dev_ids & final_ids or dev_hashes & final_hashes:
        raise OpeningValidationError("dev and final suites must be content-disjoint")


def opening_position_orbit_digest(opening: Opening) -> str:
    """Return a horizontal-mirror-invariant digest of an opening position."""

    validate_opening(opening)
    raw = position_hash(opening.state).encode("ascii")
    mirrored = position_hash(mirror_state(opening.state)).encode("ascii")
    return hashlib.sha256(min(raw, mirrored)).hexdigest()


def validate_suite_collection(*suites: OpeningSuite) -> None:
    """Require any evaluation suites to be content and mirror-orbit disjoint."""

    if len(suites) < 2:
        raise OpeningValidationError("suite collection requires at least two suites")
    names: set[str] = set()
    identities: set[str] = set()
    hashes: set[str] = set()
    orbits: set[str] = set()
    for suite in suites:
        validate_opening_suite(suite)
        if suite.name in names:
            raise OpeningValidationError("suite collection contains a duplicate name")
        names.add(suite.name)
        for opening in suite.openings:
            orbit = opening_position_orbit_digest(opening)
            if (
                opening.id in identities
                or opening.content_hash in hashes
                or orbit in orbits
            ):
                raise OpeningValidationError(
                    "evaluation suites must be content and mirror-orbit disjoint"
                )
            identities.add(opening.id)
            hashes.add(opening.content_hash)
            orbits.add(orbit)


def opening_suite_to_payload(suite: OpeningSuite) -> dict[str, object]:
    validate_opening_suite(suite)
    return {**_suite_core(suite.name, suite.openings), "suiteHash": suite.content_hash}


def save_opening_suite(suite: OpeningSuite, path: str | Path) -> Path:
    """Atomically save a validated suite. This does not mark it as locked."""

    return atomic_json_save(opening_suite_to_payload(suite), path)


def load_opening_suite(
    path: str | Path,
    *,
    expected_suite: str | None = None,
) -> OpeningSuite:
    """Load and fully replay every opening before returning the suite."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OpeningValidationError(f"cannot load opening suite {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise OpeningValidationError("opening suite root must be a mapping")
    _require_exact_keys(
        payload,
        frozenset(
            {"schemaVersion", "suite", "pairingContract", "openings", "suiteHash"}
        ),
        "opening suite",
    )
    if payload["schemaVersion"] != OPENING_SCHEMA_VERSION:
        raise OpeningValidationError("unsupported opening schema version")
    if payload["pairingContract"] != PAIRING_CONTRACT:
        raise OpeningValidationError("unsupported opening pairing contract")
    name = _validate_suite_name(payload["suite"])
    if expected_suite is not None and name != _validate_suite_name(expected_suite):
        raise OpeningValidationError(
            f"expected {expected_suite!r} suite, loaded {name!r}"
        )
    raw_openings = payload["openings"]
    if not isinstance(raw_openings, list) or not raw_openings:
        raise OpeningValidationError("openings must be a non-empty list")

    openings: list[Opening] = []
    entry_keys = frozenset({"id", "hash", "startingPlayer", "actions", "state"})
    for index, entry in enumerate(raw_openings):
        if not isinstance(entry, Mapping):
            raise OpeningValidationError(f"openings[{index}] must be a mapping")
        _require_exact_keys(entry, entry_keys, f"openings[{index}]")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier:
            raise OpeningValidationError(f"openings[{index}].id must be non-empty")
        content_hash = _validate_hash(entry["hash"], f"openings[{index}].hash")
        starting_player = _require_player(
            entry["startingPlayer"], f"openings[{index}].startingPlayer"
        )
        raw_actions = entry["actions"]
        if not isinstance(raw_actions, list):
            raise OpeningValidationError(f"openings[{index}].actions must be a list")
        actions = _normalize_actions(raw_actions)
        raw_state = entry["state"]
        if not isinstance(raw_state, Mapping):
            raise OpeningValidationError(f"openings[{index}].state must be a mapping")
        state = state_from_opening_payload(raw_state)
        openings.append(
            Opening(
                id=identifier,
                content_hash=content_hash,
                suite=name,
                starting_player=starting_player,
                actions=actions,
                state=state,
            )
        )

    suite = OpeningSuite(
        name=name,
        openings=tuple(openings),
        content_hash=_validate_hash(payload["suiteHash"], "suite hash"),
    )
    validate_opening_suite(suite)
    return suite


__all__ = [
    "OPENING_SCHEMA_VERSION",
    "PAIRING_CONTRACT",
    "SUITE_NAMES",
    "OpeningValidationError",
    "OpeningLeg",
    "Opening",
    "OpeningSuite",
    "state_to_opening_payload",
    "state_from_opening_payload",
    "generate_opening",
    "generate_opening_suite",
    "build_opening_suite",
    "validate_opening",
    "validate_opening_suite",
    "validate_suite_partition",
    "opening_position_orbit_digest",
    "validate_suite_collection",
    "opening_suite_to_payload",
    "save_opening_suite",
    "load_opening_suite",
]
