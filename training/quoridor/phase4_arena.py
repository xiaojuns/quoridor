"""Resumable opening-pair arena for Phase 4E promotion gates.

The durable unit is an opening pair, never an individual game.  Neural
participants with the persistent-session selector API are scheduled in batches;
other selectors use the existing serial referee.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .codec import position_hash
from .game import (
    State,
    apply_action_index,
    get_winner,
    is_terminal,
    legal_action_indices,
)
from .openings import (
    Opening,
    OpeningLeg,
    OpeningSuite,
    OpeningValidationError,
    load_opening_suite,
    opening_suite_to_payload,
    validate_opening_suite,
)
from .paired_arena import (
    BoundMoveSelector,
    GameRecord,
    PlyRecord,
    SelectorContractError,
    SelectorIdentity,
    SelectorRuntime,
    TERMINATION_REASONS,
    TerminalResultError,
    _metadata_json,
    _play_leg,
    _referee_state,
    _resolve_participant,
    _selector_view,
    _snapshot_opening,
    _validated_legs,
)
from .run_manifest import (
    ImmutableArtifactExistsError,
    RunLock,
    atomic_create_bytes,
    atomic_replace_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .selectors import (
    ClassicHardSelector,
    MCTS_ROOT_VISIT_SEMANTICS,
    MCTSSelector,
    MoveSelector,
    SessionMoveSelector,
    SelectionResult,
)
from .statistics import PairedGameResult, summarize_paired_results


PHASE4_ARENA_SCHEMA_VERSION = 1
PHASE4_ARENA_SCHEDULER = "opening-pair-actor-refill-v1"
PHASE4_ARENA_COMMIT_UNIT = "complete-opening-pair-v1"
_PAIR_GLOB = "pair-*.json"
_PROGRESS_NAME = "progress.json"
_RESULT_NAME = "result.json"
_SUITE_NAME = "opening-suite.json"
_LOCK_NAME = ".arena.lock"
_PLY_FIELDS = frozenset(
    {
        "ply",
        "moveNumberBefore",
        "player",
        "selectorRole",
        "positionHashBefore",
        "actionIndex",
        "selection",
        "positionHashAfter",
    }
)
_GAME_FIELDS = frozenset(
    {
        "schemaVersion",
        "recordType",
        "gameId",
        "suite",
        "suiteHash",
        "openingId",
        "openingHash",
        "leg",
        "candidatePlayer",
        "opponentPlayer",
        "candidateToMove",
        "candidateIdentity",
        "opponentIdentity",
        "candidateRuntime",
        "opponentRuntime",
        "openingMoveNumber",
        "initialPositionHash",
        "plies",
        "finalPositionHash",
        "finalMoveNumber",
        "winner",
        "terminationReason",
        "candidateWon",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "contractSha256",
        "suiteHash",
        "openingId",
        "openingHash",
        "ordinal",
        "commitUnit",
        "games",
        "pairSha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schemaVersion",
        "kind",
        "status",
        "contractSha256",
        "suite",
        "suiteHash",
        "scheduler",
        "commitUnit",
        "candidateIdentity",
        "opponentIdentity",
        "candidateRuntime",
        "opponentRuntime",
        "openingSuite",
        "openingPairs",
        "games",
        "statistics",
        "pairArtifacts",
        "resultSha256",
    }
)


class Phase4ArenaError(RuntimeError):
    """Base class for resumable Phase 4E arena failures."""


class Phase4ArenaCorruptionError(Phase4ArenaError):
    """Raised when committed pair evidence is corrupt or ambiguous."""


@dataclass(frozen=True, slots=True)
class PairArtifact:
    ordinal: int
    opening_id: str
    file_name: str
    sha256: str
    byte_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "openingId": self.opening_id,
            "file": self.file_name,
            "sha256": self.sha256,
            "bytes": self.byte_size,
        }


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(core)
    payload[field] = sha256_bytes(canonical_json_bytes(core))
    return payload


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    try:
        digest = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise Phase4ArenaCorruptionError(f"{label} has an invalid self hash") from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4ArenaCorruptionError(f"{label} self hash mismatch")
    return digest


def _safe_mapping(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise Phase4ArenaCorruptionError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4ArenaCorruptionError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4ArenaCorruptionError(f"{label} is not canonical JSON")
    return dict(payload)


def _identity_from_payload(
    value: object,
    label: str,
) -> SelectorIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "selectorId",
        "artifactHash",
        "configHash",
        "sourceHash",
        "engine",
        "verified",
    }:
        raise Phase4ArenaCorruptionError(f"{label} fields are invalid")
    try:
        identity = SelectorIdentity(
            selector_id=value.get("selectorId"),
            artifact_hash=value.get("artifactHash"),
            config_hash=value.get("configHash"),
            source_hash=value.get("sourceHash"),
            engine=value.get("engine"),
        )
    except (TypeError, ValueError) as error:
        raise Phase4ArenaCorruptionError(f"{label} is invalid") from error
    if identity.to_dict() != dict(value):
        raise Phase4ArenaCorruptionError(f"{label} verification flag is invalid")
    return identity


def _runtime_from_payload(
    value: object,
    label: str,
) -> SelectorRuntime:
    if not isinstance(value, Mapping) or set(value) != {
        "implementation",
        "kind",
        "trusted",
        "configuredSimulations",
        "budgetKind",
        "budgetLimit",
        "maxDepth",
        "identityBound",
        "backendHash",
    }:
        raise Phase4ArenaCorruptionError(f"{label} fields are invalid")
    try:
        runtime = SelectorRuntime(
            implementation=value.get("implementation"),
            kind=value.get("kind"),
            configured_simulations=value.get("configuredSimulations"),
            budget_kind=value.get("budgetKind"),
            budget_limit=value.get("budgetLimit"),
            max_depth=value.get("maxDepth"),
            identity_bound=value.get("identityBound"),
            backend_hash=value.get("backendHash"),
        )
    except (TypeError, ValueError) as error:
        raise Phase4ArenaCorruptionError(f"{label} is invalid") from error
    if runtime.to_dict() != dict(value):
        raise Phase4ArenaCorruptionError(f"{label} trusted flag is invalid")
    return runtime


def _load_suite_snapshot(
    path: Path,
    *,
    reference: object | None = None,
) -> OpeningSuite:
    payload = _safe_mapping(path, "opening-suite snapshot")
    if reference is not None:
        if not isinstance(reference, Mapping) or set(reference) != {
            "file",
            "sha256",
            "bytes",
        }:
            raise Phase4ArenaCorruptionError(
                "opening-suite snapshot reference is invalid"
            )
        try:
            digest = validate_sha256(
                reference.get("sha256"),
                "opening-suite snapshot sha256",
            )
        except ValueError as error:
            raise Phase4ArenaCorruptionError(
                "opening-suite snapshot hash is invalid"
            ) from error
        if (
            reference.get("file") != _SUITE_NAME
            or reference.get("bytes") != path.stat().st_size
            or sha256_file(path) != digest
        ):
            raise Phase4ArenaCorruptionError(
                "opening-suite snapshot reference does not verify"
            )
    try:
        suite = load_opening_suite(path)
    except (OSError, TypeError, OpeningValidationError) as error:
        raise Phase4ArenaCorruptionError(
            "opening-suite snapshot does not replay"
        ) from error
    if opening_suite_to_payload(suite) != payload:
        raise Phase4ArenaCorruptionError(
            "opening-suite snapshot is not canonical"
        )
    return suite


def _suite_reference(path: Path) -> dict[str, object]:
    return {
        "file": _SUITE_NAME,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _raw_selector(selector: MoveSelector) -> MoveSelector:
    return selector.selector if type(selector) is BoundMoveSelector else selector


def _session_selector(selector: MoveSelector) -> SessionMoveSelector | None:
    raw = _raw_selector(selector)
    if not isinstance(raw, (MCTSSelector, ClassicHardSelector)):
        return None
    return raw if bool(raw.supports_sessions) else None


def _validate_identity_payload(
    game: Mapping[str, object],
    *,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> None:
    expected = {
        "candidateIdentity": candidate_identity.to_dict(),
        "opponentIdentity": opponent_identity.to_dict(),
        "candidateRuntime": candidate_runtime.to_dict(),
        "opponentRuntime": opponent_runtime.to_dict(),
    }
    for key, value in expected.items():
        if game.get(key) != value:
            raise Phase4ArenaCorruptionError(
                f"committed game {key} differs from the arena contract"
            )


def _validate_selection_payload(
    value: object,
    *,
    identity: SelectorIdentity,
    runtime: SelectorRuntime,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "engine",
        "value",
        "score",
        "simulations",
        "nodes",
        "elapsedMs",
        "wallTimeMs",
        "depth",
        "metadata",
    }:
        raise Phase4ArenaCorruptionError(f"{label} selection fields are invalid")
    engine = value.get("engine")
    if (
        not isinstance(engine, str)
        or not engine
        or (identity.engine is not None and engine != identity.engine)
    ):
        raise Phase4ArenaCorruptionError(f"{label} selector engine is invalid")
    for field in ("simulations", "nodes", "depth"):
        item = value.get(field)
        if type(item) is not int or item < 0:
            raise Phase4ArenaCorruptionError(
                f"{label} selection {field} is invalid"
            )
    for field in ("elapsedMs", "wallTimeMs"):
        item = value.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
        ):
            raise Phase4ArenaCorruptionError(
                f"{label} selection {field} is invalid"
            )
    for field in ("value", "score"):
        item = value.get(field)
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise Phase4ArenaCorruptionError(
                f"{label} selection {field} is invalid"
            )
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Phase4ArenaCorruptionError(
            f"{label} selection metadata is invalid"
        )
    if runtime.kind in {"native-mcts", "python-mcts"}:
        simulations = value.get("simulations")
        nodes = value.get("nodes")
        configured = runtime.configured_simulations
        if simulations != configured:
            raise Phase4ArenaCorruptionError(
                f"{label} simulations differ from the configured MCTS runtime"
            )
        if set(metadata) != {
            "configuredSimulations",
            "batchSize",
            "rootVisits",
            "retainedRootVisits",
            "visitCountSemantics",
        }:
            raise Phase4ArenaCorruptionError(
                f"{label} MCTS metadata fields are invalid"
            )
        batch_size = metadata.get("batchSize")
        root_visits = metadata.get("rootVisits")
        retained_visits = metadata.get("retainedRootVisits")
        if (
            metadata.get("configuredSimulations") != configured
            or type(batch_size) is not int
            or batch_size <= 0
            or type(root_visits) is not int
            or root_visits < simulations
            or type(retained_visits) is not int
            or retained_visits < 0
            or retained_visits != root_visits - simulations
            or metadata.get("visitCountSemantics")
            != MCTS_ROOT_VISIT_SEMANTICS
            or nodes != root_visits
        ):
            raise Phase4ArenaCorruptionError(
                f"{label} MCTS visit accounting is invalid"
            )


def _validate_game_payload(
    game: Mapping[str, object],
    *,
    suite: OpeningSuite,
    opening: Opening,
    leg: OpeningLeg,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> None:
    if set(game) != _GAME_FIELDS:
        raise Phase4ArenaCorruptionError(
            f"committed game fields are invalid for {opening.id} leg {leg.leg}"
        )
    _validate_identity_payload(
        game,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
    )
    if (
        game.get("schemaVersion") != 1
        or game.get("recordType") != "paired-arena-game"
        or game.get("gameId") != f"{opening.id}:leg-{leg.leg}"
        or game.get("suite") != suite.name
        or game.get("suiteHash") != suite.content_hash
        or game.get("openingId") != opening.id
        or game.get("openingHash") != opening.content_hash
        or game.get("leg") != leg.leg
        or game.get("candidatePlayer") != leg.candidate_player
        or game.get("opponentPlayer") != leg.opponent_player
        or game.get("candidateToMove") != leg.candidate_to_move
        or game.get("openingMoveNumber") != opening.state.move_number
        or game.get("initialPositionHash") != position_hash(opening.state)
    ):
        raise Phase4ArenaCorruptionError(
            f"committed game identity mismatch for {opening.id} leg {leg.leg}"
        )
    raw_plies = game.get("plies")
    if not isinstance(raw_plies, list) or not raw_plies:
        raise Phase4ArenaCorruptionError("committed game has no plies")
    state = _referee_state(opening)
    for index, raw_ply in enumerate(raw_plies):
        if not isinstance(raw_ply, Mapping):
            raise Phase4ArenaCorruptionError("committed game ply is not a mapping")
        player = state.current_player
        expected_role = (
            "candidate" if player == leg.candidate_player else "opponent"
        )
        identity = (
            candidate_identity
            if expected_role == "candidate"
            else opponent_identity
        )
        runtime = (
            candidate_runtime
            if expected_role == "candidate"
            else opponent_runtime
        )
        action = raw_ply.get("actionIndex")
        if (
            set(raw_ply) != _PLY_FIELDS
            or raw_ply.get("ply") != index
            or raw_ply.get("moveNumberBefore") != state.move_number
            or raw_ply.get("player") != player
            or raw_ply.get("selectorRole") != expected_role
            or raw_ply.get("positionHashBefore") != position_hash(state)
            or type(action) is not int
            or action not in frozenset(legal_action_indices(state))
        ):
            raise Phase4ArenaCorruptionError(
                f"committed game ply {index} violates the referee transcript"
            )
        _validate_selection_payload(
            raw_ply.get("selection"),
            identity=identity,
            runtime=runtime,
            label=f"committed game ply {index}",
        )
        state = apply_action_index(state, action)
        if raw_ply.get("positionHashAfter") != position_hash(state):
            raise Phase4ArenaCorruptionError(
                f"committed game ply {index} has an invalid after hash"
            )
    winner = get_winner(state)
    if (
        not is_terminal(state)
        or winner not in (0, 1)
        or game.get("finalPositionHash") != position_hash(state)
        or game.get("finalMoveNumber") != state.move_number
        or game.get("winner") != winner
        or game.get("terminationReason") != state.termination_reason
        or game.get("candidateWon") != (winner == leg.candidate_player)
    ):
        raise Phase4ArenaCorruptionError(
            f"committed game result mismatch for {opening.id} leg {leg.leg}"
        )


def _validate_pair_payload_core(
    payload: Mapping[str, object],
    *,
    file_name: str,
    contract_sha256: str,
    suite: OpeningSuite,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> tuple[int, Opening]:
    digest = _verify_self_hash(payload, field="pairSha256", label=file_name)
    ordinal = payload.get("ordinal")
    if type(ordinal) is not int or not 0 <= ordinal < len(suite.openings):
        raise Phase4ArenaCorruptionError("pair ordinal is outside the suite")
    opening = suite.openings[ordinal]
    legs = _validated_legs(opening)
    games = payload.get("games")
    if (
        set(payload) != _PAIR_FIELDS
        or payload.get("schemaVersion") != PHASE4_ARENA_SCHEMA_VERSION
        or payload.get("kind") != "phase4-arena-opening-pair"
        or payload.get("contractSha256") != contract_sha256
        or payload.get("suiteHash") != suite.content_hash
        or payload.get("openingId") != opening.id
        or payload.get("openingHash") != opening.content_hash
        or payload.get("commitUnit") != PHASE4_ARENA_COMMIT_UNIT
        or not isinstance(games, list)
        or len(games) != 2
        or any(not isinstance(game, Mapping) for game in games)
    ):
        raise Phase4ArenaCorruptionError(
            f"pair contract mismatch in {file_name}"
        )
    ordered = sorted(games, key=lambda game: game.get("leg", -1))
    for game, leg in zip(ordered, legs):
        _validate_game_payload(
            game,
            suite=suite,
            opening=opening,
            leg=leg,
            candidate_identity=candidate_identity,
            opponent_identity=opponent_identity,
            candidate_runtime=candidate_runtime,
            opponent_runtime=opponent_runtime,
        )
    expected_prefix = f"pair-{ordinal:06d}-{opening.id}-"
    if file_name != f"{expected_prefix}{digest[:12]}.json":
        raise Phase4ArenaCorruptionError("pair file name/content mismatch")
    return ordinal, opening


def _validate_pair_payload_against_contract(
    payload: Mapping[str, object],
    *,
    path: Path,
    contract_sha256: str,
    suite: OpeningSuite,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> PairArtifact:
    ordinal, opening = _validate_pair_payload_core(
        payload,
        file_name=path.name,
        contract_sha256=contract_sha256,
        suite=suite,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
    )
    return PairArtifact(
        ordinal=ordinal,
        opening_id=opening.id,
        file_name=path.name,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
    )


class OpeningPairStore:
    """Immutable pair files plus a reconstructable self-hashed progress index."""

    def __init__(
        self,
        root: str | Path,
        *,
        contract_sha256: str,
        suite: OpeningSuite,
        candidate_identity: SelectorIdentity,
        opponent_identity: SelectorIdentity,
        candidate_runtime: SelectorRuntime,
        opponent_runtime: SelectorRuntime,
    ) -> None:
        validate_opening_suite(suite)
        self.root = Path(root)
        self.contract_sha256 = validate_sha256(
            contract_sha256, "arena contract sha256"
        )
        self.suite = suite
        self.candidate_identity = candidate_identity
        self.opponent_identity = opponent_identity
        self.candidate_runtime = candidate_runtime
        self.opponent_runtime = opponent_runtime
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / _LOCK_NAME
        self.suite_path = self.root / _SUITE_NAME
        self._artifacts: dict[int, PairArtifact] = {}
        with self._lock():
            self._cleanup_temporaries_unlocked()
            self._ensure_suite_snapshot_unlocked()
            self._reconcile_unlocked()

    def _lock(self) -> RunLock:
        return RunLock(
            self.lock_path,
            run_id=f"phase4-arena:{self.contract_sha256[:16]}",
            blocking=True,
        )

    def _cleanup_temporaries_unlocked(self) -> None:
        for temporary in self.root.glob(".*.tmp"):
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink(missing_ok=True)

    def _ensure_suite_snapshot_unlocked(self) -> None:
        encoded = canonical_json_bytes(opening_suite_to_payload(self.suite))
        if self.suite_path.exists():
            if (
                self.suite_path.is_symlink()
                or not self.suite_path.is_file()
                or self.suite_path.read_bytes() != encoded
            ):
                raise Phase4ArenaCorruptionError(
                    "opening-suite snapshot differs from the arena contract"
                )
        else:
            try:
                atomic_create_bytes(self.suite_path, encoded)
            except ImmutableArtifactExistsError:
                if (
                    self.suite_path.is_symlink()
                    or not self.suite_path.is_file()
                    or self.suite_path.read_bytes() != encoded
                ):
                    raise Phase4ArenaCorruptionError(
                        "concurrent opening-suite snapshot differs"
                    )
        loaded = _load_suite_snapshot(self.suite_path)
        if loaded != self.suite:
            raise Phase4ArenaCorruptionError(
                "opening-suite snapshot changed after creation"
            )

    @property
    def completed_ordinals(self) -> tuple[int, ...]:
        return tuple(sorted(self._artifacts))

    @property
    def completed_pairs(self) -> int:
        return len(self._artifacts)

    def _opening(self, ordinal: int) -> Opening:
        if type(ordinal) is not int or not 0 <= ordinal < len(self.suite.openings):
            raise ValueError("opening ordinal is outside the suite")
        return self.suite.openings[ordinal]

    def _validate_pair_payload(
        self,
        payload: Mapping[str, object],
        *,
        path: Path,
    ) -> PairArtifact:
        return _validate_pair_payload_against_contract(
            payload,
            path=path,
            contract_sha256=self.contract_sha256,
            suite=self.suite,
            candidate_identity=self.candidate_identity,
            opponent_identity=self.opponent_identity,
            candidate_runtime=self.candidate_runtime,
            opponent_runtime=self.opponent_runtime,
        )

    def _reconcile_unlocked(self) -> tuple[PairArtifact, ...]:
        artifacts: dict[int, PairArtifact] = {}
        for path in sorted(self.root.glob(_PAIR_GLOB)):
            artifact = self._validate_pair_payload(
                _safe_mapping(path, f"pair file {path.name}"),
                path=path,
            )
            previous = artifacts.get(artifact.ordinal)
            if previous is not None and previous.sha256 != artifact.sha256:
                raise Phase4ArenaCorruptionError(
                    f"conflicting pair files for ordinal {artifact.ordinal}"
                )
            artifacts[artifact.ordinal] = artifact
        self._artifacts = artifacts
        self._write_progress()
        return tuple(artifacts[index] for index in sorted(artifacts))

    def reconcile(self) -> tuple[PairArtifact, ...]:
        with self._lock():
            self._cleanup_temporaries_unlocked()
            self._ensure_suite_snapshot_unlocked()
            return self._reconcile_unlocked()

    def _progress_core(self) -> dict[str, object]:
        pairs = [
            self._artifacts[index].to_dict() for index in sorted(self._artifacts)
        ]
        return {
            "schemaVersion": PHASE4_ARENA_SCHEMA_VERSION,
            "kind": "phase4-arena-progress",
            "status": (
                "completed"
                if len(pairs) == len(self.suite.openings)
                else "in-progress"
            ),
            "contractSha256": self.contract_sha256,
            "suiteHash": self.suite.content_hash,
            "openingSuite": _suite_reference(self.suite_path),
            "scheduler": PHASE4_ARENA_SCHEDULER,
            "commitUnit": PHASE4_ARENA_COMMIT_UNIT,
            "completedPairs": len(pairs),
            "completedGames": len(pairs) * 2,
            "pairs": pairs,
        }

    def _write_progress(self) -> None:
        payload = _self_hashed(self._progress_core(), "progressSha256")
        atomic_replace_bytes(
            self.root / _PROGRESS_NAME,
            canonical_json_bytes(payload),
        )

    def commit_pair(
        self,
        ordinal: int,
        games: Sequence[GameRecord],
    ) -> PairArtifact:
        opening = self._opening(ordinal)
        games = tuple(sorted(games, key=lambda game: game.leg))
        if (
            len(games) != 2
            or tuple(game.leg for game in games) != (0, 1)
            or any(game.opening_id != opening.id for game in games)
        ):
            raise ValueError("commit_pair requires both legs of one opening")
        core = {
            "schemaVersion": PHASE4_ARENA_SCHEMA_VERSION,
            "kind": "phase4-arena-opening-pair",
            "contractSha256": self.contract_sha256,
            "suiteHash": self.suite.content_hash,
            "openingId": opening.id,
            "openingHash": opening.content_hash,
            "ordinal": ordinal,
            "commitUnit": PHASE4_ARENA_COMMIT_UNIT,
            "games": [game.to_dict() for game in games],
        }
        payload = _self_hashed(core, "pairSha256")
        digest = str(payload["pairSha256"])
        destination = (
            self.root
            / f"pair-{ordinal:06d}-{opening.id}-{digest[:12]}.json"
        )
        encoded = canonical_json_bytes(payload)
        expected_file_sha256 = sha256_bytes(encoded)
        with self._lock():
            self._cleanup_temporaries_unlocked()
            self._ensure_suite_snapshot_unlocked()
            self._reconcile_unlocked()
            previous = self._artifacts.get(ordinal)
            if previous is not None:
                if (
                    previous.file_name != destination.name
                    or previous.sha256 != expected_file_sha256
                ):
                    raise Phase4ArenaCorruptionError(
                        f"opening ordinal {ordinal} was claimed by another runner"
                    )
                return previous
            _validate_pair_payload_core(
                payload,
                file_name=destination.name,
                contract_sha256=self.contract_sha256,
                suite=self.suite,
                candidate_identity=self.candidate_identity,
                opponent_identity=self.opponent_identity,
                candidate_runtime=self.candidate_runtime,
                opponent_runtime=self.opponent_runtime,
            )
            created = False
            try:
                atomic_create_bytes(destination, encoded)
                created = True
            except ImmutableArtifactExistsError:
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or destination.read_bytes() != encoded
                ):
                    raise Phase4ArenaCorruptionError(
                        f"immutable pair already differs: {destination.name}"
                    )
            try:
                durable_payload = _safe_mapping(
                    destination,
                    f"pair file {destination.name}",
                )
                artifact = self._validate_pair_payload(
                    durable_payload,
                    path=destination,
                )
                if artifact.sha256 != expected_file_sha256:
                    raise Phase4ArenaCorruptionError(
                        f"durable pair differs after creation: {destination.name}"
                    )
            except Exception:
                if created:
                    try:
                        destination.unlink()
                    except OSError:
                        pass
                raise
            self._artifacts[ordinal] = artifact
            self._write_progress()
            return artifact

    def pair_payload(self, ordinal: int) -> dict[str, object]:
        artifact = self._artifacts[ordinal]
        return _safe_mapping(
            self.root / artifact.file_name,
            f"pair file {artifact.file_name}",
        )

    def finalize(
        self,
        *,
        bootstrap_seed: int,
        bootstrap_samples: int,
        confidence: float = 0.90,
    ) -> dict[str, object]:
        with self._lock():
            self._cleanup_temporaries_unlocked()
            self._ensure_suite_snapshot_unlocked()
            self._reconcile_unlocked()
            if self.completed_pairs != len(self.suite.openings):
                raise Phase4ArenaError("arena cannot finalize with incomplete pairs")
            games = [
                game
                for ordinal in self.completed_ordinals
                for game in self.pair_payload(ordinal)["games"]
            ]
            results = tuple(
                PairedGameResult(
                    opening_id=str(game["openingId"]),
                    leg=int(game["leg"]),
                    candidate_player=int(game["candidatePlayer"]),
                    winner=int(game["winner"]),
                )
                for game in games
            )
            statistics = summarize_paired_results(
                results,
                seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
            )
            core = {
                "schemaVersion": PHASE4_ARENA_SCHEMA_VERSION,
                "kind": "phase4-resumable-paired-arena-result",
                "status": "completed",
                "contractSha256": self.contract_sha256,
                "suite": self.suite.name,
                "suiteHash": self.suite.content_hash,
                "scheduler": PHASE4_ARENA_SCHEDULER,
                "commitUnit": PHASE4_ARENA_COMMIT_UNIT,
                "candidateIdentity": self.candidate_identity.to_dict(),
                "opponentIdentity": self.opponent_identity.to_dict(),
                "candidateRuntime": self.candidate_runtime.to_dict(),
                "opponentRuntime": self.opponent_runtime.to_dict(),
                "openingSuite": _suite_reference(self.suite_path),
                "openingPairs": len(self.suite.openings),
                "games": len(games),
                "statistics": statistics.to_dict(),
                "pairArtifacts": [
                    self._artifacts[index].to_dict()
                    for index in self.completed_ordinals
                ],
            }
            payload = _self_hashed(core, "resultSha256")
            atomic_replace_bytes(
                self.root / _RESULT_NAME,
                canonical_json_bytes(payload),
            )
            return payload


def verify_completed_arena_result(
    result_path: str | Path,
) -> dict[str, object]:
    """Deep-verify a completed arena result and return its canonical payload.

    Verification reloads the persisted opening suite, replays every recorded
    action through the rules engine, checks terminal outcomes, enforces each
    MCTS runtime's configured simulation count, and recomputes all statistics.
    """

    path = Path(result_path)
    if path.name != _RESULT_NAME:
        raise Phase4ArenaCorruptionError(
            f"completed arena result must be named {_RESULT_NAME}"
        )
    root = path.parent
    with RunLock(
        root / _LOCK_NAME,
        run_id="phase4-arena:completed-result-verification",
        blocking=True,
    ):
        result = _safe_mapping(path, "completed arena result")
        result_sha256 = _verify_self_hash(
            result,
            field="resultSha256",
            label="completed arena result",
        )
        if (
            set(result) != _RESULT_FIELDS
            or result.get("schemaVersion") != PHASE4_ARENA_SCHEMA_VERSION
            or result.get("kind")
            != "phase4-resumable-paired-arena-result"
            or result.get("status") != "completed"
            or result.get("scheduler") != PHASE4_ARENA_SCHEDULER
            or result.get("commitUnit") != PHASE4_ARENA_COMMIT_UNIT
        ):
            raise Phase4ArenaCorruptionError(
                "completed arena result schema or status is invalid"
            )
        try:
            contract_sha256 = validate_sha256(
                result.get("contractSha256"),
                "arena contract sha256",
            )
        except ValueError as error:
            raise Phase4ArenaCorruptionError(
                "completed arena contract hash is invalid"
            ) from error

        suite_path = root / _SUITE_NAME
        suite = _load_suite_snapshot(
            suite_path,
            reference=result.get("openingSuite"),
        )
        if (
            result.get("suite") != suite.name
            or result.get("suiteHash") != suite.content_hash
        ):
            raise Phase4ArenaCorruptionError(
                "completed arena result differs from its opening suite"
            )
        candidate_identity = _identity_from_payload(
            result.get("candidateIdentity"),
            "candidate identity",
        )
        opponent_identity = _identity_from_payload(
            result.get("opponentIdentity"),
            "opponent identity",
        )
        candidate_runtime = _runtime_from_payload(
            result.get("candidateRuntime"),
            "candidate runtime",
        )
        opponent_runtime = _runtime_from_payload(
            result.get("opponentRuntime"),
            "opponent runtime",
        )

        raw_artifacts = result.get("pairArtifacts")
        expected_pairs = len(suite.openings)
        if (
            not isinstance(raw_artifacts, list)
            or len(raw_artifacts) != expected_pairs
            or result.get("openingPairs") != expected_pairs
            or result.get("games") != expected_pairs * 2
        ):
            raise Phase4ArenaCorruptionError(
                "completed arena pair/game totals are invalid"
            )
        artifacts_by_ordinal: dict[int, Mapping[str, object]] = {}
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != {
                "ordinal",
                "openingId",
                "file",
                "sha256",
                "bytes",
            }:
                raise Phase4ArenaCorruptionError(
                    "completed arena pair reference fields are invalid"
                )
            ordinal = raw_artifact.get("ordinal")
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < expected_pairs
                or ordinal in artifacts_by_ordinal
            ):
                raise Phase4ArenaCorruptionError(
                    "completed arena pair ordinals are invalid"
                )
            artifacts_by_ordinal[ordinal] = raw_artifact
        if set(artifacts_by_ordinal) != set(range(expected_pairs)):
            raise Phase4ArenaCorruptionError(
                "completed arena omits an opening pair"
            )

        paired_results: list[PairedGameResult] = []
        expected_files: set[str] = set()
        for ordinal in range(expected_pairs):
            reference = artifacts_by_ordinal[ordinal]
            file_name = reference.get("file")
            if not isinstance(file_name, str) or Path(file_name).name != file_name:
                raise Phase4ArenaCorruptionError(
                    "completed arena pair file name is unsafe"
                )
            pair_path = root / file_name
            pair_payload = _safe_mapping(
                pair_path,
                f"completed pair {ordinal}",
            )
            artifact = _validate_pair_payload_against_contract(
                pair_payload,
                path=pair_path,
                contract_sha256=contract_sha256,
                suite=suite,
                candidate_identity=candidate_identity,
                opponent_identity=opponent_identity,
                candidate_runtime=candidate_runtime,
                opponent_runtime=opponent_runtime,
            )
            if artifact.to_dict() != dict(reference):
                raise Phase4ArenaCorruptionError(
                    f"completed pair reference differs for ordinal {ordinal}"
                )
            expected_files.add(file_name)
            for game in sorted(
                pair_payload["games"],
                key=lambda item: item["leg"],
            ):
                paired_results.append(
                    PairedGameResult(
                        opening_id=str(game["openingId"]),
                        leg=int(game["leg"]),
                        candidate_player=int(game["candidatePlayer"]),
                        winner=int(game["winner"]),
                    )
                )
        actual_files = {item.name for item in root.glob(_PAIR_GLOB)}
        if actual_files != expected_files:
            raise Phase4ArenaCorruptionError(
                "completed arena contains unreferenced or missing pair files"
            )

        statistics = result.get("statistics")
        interval = (
            statistics.get("winRateCI")
            if isinstance(statistics, Mapping)
            else None
        )
        bootstrap = (
            statistics.get("bootstrap")
            if isinstance(statistics, Mapping)
            else None
        )
        if not isinstance(interval, Mapping) or not isinstance(
            bootstrap, Mapping
        ):
            raise Phase4ArenaCorruptionError(
                "completed arena statistics metadata is invalid"
            )
        try:
            recomputed = summarize_paired_results(
                tuple(paired_results),
                seed=bootstrap.get("seed"),
                bootstrap_samples=bootstrap.get("samples"),
                confidence=interval.get("confidence"),
            ).to_dict()
        except (TypeError, ValueError) as error:
            raise Phase4ArenaCorruptionError(
                "completed arena statistics parameters are invalid"
            ) from error
        if recomputed != dict(statistics):
            raise Phase4ArenaCorruptionError(
                "completed arena statistics differ from replayed outcomes"
            )
        if result.get("resultSha256") != result_sha256:
            raise Phase4ArenaCorruptionError(
                "completed arena result hash changed during verification"
            )
        return result


@dataclass(slots=True)
class _ActiveLeg:
    ordinal: int
    opening: Opening
    leg: OpeningLeg
    state: State
    initial_position_hash: str
    candidate_session: int
    opponent_session: int
    plies: list[PlyRecord] = field(default_factory=list)


def _finished_record(
    active: _ActiveLeg,
    *,
    suite: OpeningSuite,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> GameRecord:
    winner = get_winner(active.state)
    if winner not in (0, 1) or active.state.termination_reason not in TERMINATION_REASONS:
        raise TerminalResultError(
            f"opening {active.opening.id!r}, leg {active.leg.leg} ended without a winner"
        )
    return GameRecord(
        suite_name=suite.name,
        suite_hash=suite.content_hash,
        opening_id=active.opening.id,
        opening_hash=active.opening.content_hash,
        leg=active.leg.leg,
        candidate_player=active.leg.candidate_player,
        opponent_player=active.leg.opponent_player,
        candidate_to_move=active.leg.candidate_to_move,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
        opening_move_number=active.opening.state.move_number,
        initial_position_hash=active.initial_position_hash,
        plies=tuple(active.plies),
        final_position_hash=position_hash(active.state),
        final_move_number=active.state.move_number,
        winner=winner,
        termination_reason=active.state.termination_reason,
    )


def _run_batched_missing_pairs(
    *,
    store: OpeningPairStore,
    suite: OpeningSuite,
    candidate: SessionMoveSelector,
    opponent: SessionMoveSelector,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
    missing_ordinals: Sequence[int],
    actors: int,
    heartbeat: Callable[[int, int], None] | None,
) -> None:
    schedule = [
        (ordinal, _snapshot_opening(suite.openings[ordinal]), leg)
        for ordinal in missing_ordinals
        for leg in _validated_legs(suite.openings[ordinal])
    ]
    active: list[_ActiveLeg] = []
    completed: dict[int, list[GameRecord]] = {}
    open_candidate: set[int] = set()
    open_opponent: set[int] = set()
    cursor = 0
    moves = 0

    def fill() -> None:
        nonlocal cursor
        while cursor < len(schedule) and len(active) < actors:
            ordinal, opening, leg = schedule[cursor]
            cursor += 1
            state = _referee_state(opening)
            candidate_session = candidate.create_session(_selector_view(state))
            opponent_session = opponent.create_session(_selector_view(state))
            open_candidate.add(candidate_session)
            open_opponent.add(opponent_session)
            active.append(
                _ActiveLeg(
                    ordinal=ordinal,
                    opening=opening,
                    leg=leg,
                    state=state,
                    initial_position_hash=position_hash(state),
                    candidate_session=candidate_session,
                    opponent_session=opponent_session,
                )
            )

    try:
        fill()
        while active:
            if heartbeat is not None:
                heartbeat(moves, store.completed_pairs)
            selections: dict[int, tuple[SelectionResult, float]] = {}
            for role, selector in (("candidate", candidate), ("opponent", opponent)):
                indices = [
                    index
                    for index, game in enumerate(active)
                    if (
                        game.state.current_player == game.leg.candidate_player
                    )
                    == (role == "candidate")
                ]
                if not indices:
                    continue
                session_ids = [
                    (
                        active[index].candidate_session
                        if role == "candidate"
                        else active[index].opponent_session
                    )
                    for index in indices
                ]
                states = [_selector_view(active[index].state) for index in indices]
                started = time.monotonic()
                batch = selector.choose_sessions(session_ids, states)
                wall_time_ms = (time.monotonic() - started) * 1_000.0
                if len(batch) != len(indices):
                    raise SelectorContractError(
                        f"{role} batched selector returned the wrong result count"
                    )
                for index, selection in zip(indices, batch):
                    selections[index] = (selection, wall_time_ms)

            finished: list[int] = []
            for index, game in enumerate(active):
                selection, wall_time_ms = selections[index]
                state = game.state
                player = state.current_player
                is_candidate = player == game.leg.candidate_player
                identity = candidate_identity if is_candidate else opponent_identity
                if not isinstance(selection, SelectionResult):
                    raise SelectorContractError(
                        "batched selector must return SelectionResult"
                    )
                if identity.engine is not None and selection.engine != identity.engine:
                    raise SelectorContractError("batched selector engine mismatch")
                legal = frozenset(legal_action_indices(state))
                if selection.action_index not in legal:
                    raise SelectorContractError("batched selector returned an illegal action")
                before_hash = position_hash(state)
                next_state = apply_action_index(state, selection.action_index)
                game.plies.append(
                    PlyRecord(
                        ply=len(game.plies),
                        move_number_before=state.move_number,
                        player=player,
                        selector_role="candidate" if is_candidate else "opponent",
                        position_hash_before=before_hash,
                        action_index=selection.action_index,
                        engine=selection.engine,
                        value=selection.value,
                        score=selection.score,
                        simulations=selection.simulations,
                        nodes=selection.nodes,
                        elapsed_ms=selection.elapsed_ms,
                        wall_time_ms=wall_time_ms,
                        depth=selection.depth,
                        metadata_json=_metadata_json(selection.metadata),
                        position_hash_after=position_hash(next_state),
                    )
                )
                candidate.advance_session(
                    game.candidate_session,
                    selection.action_index,
                    _selector_view(next_state),
                )
                opponent.advance_session(
                    game.opponent_session,
                    selection.action_index,
                    _selector_view(next_state),
                )
                game.state = next_state
                moves += 1
                if is_terminal(next_state):
                    record = _finished_record(
                        game,
                        suite=suite,
                        candidate_identity=candidate_identity,
                        opponent_identity=opponent_identity,
                        candidate_runtime=candidate_runtime,
                        opponent_runtime=opponent_runtime,
                    )
                    completed.setdefault(game.ordinal, []).append(record)
                    candidate.release_session(game.candidate_session)
                    opponent.release_session(game.opponent_session)
                    open_candidate.discard(game.candidate_session)
                    open_opponent.discard(game.opponent_session)
                    finished.append(index)
            for index in reversed(finished):
                del active[index]
            for ordinal in tuple(sorted(completed)):
                if len(completed[ordinal]) == 2:
                    store.commit_pair(ordinal, completed.pop(ordinal))
                    if heartbeat is not None:
                        heartbeat(moves, store.completed_pairs)
            fill()
    finally:
        for session_id in tuple(open_candidate):
            try:
                candidate.release_session(session_id)
            except Exception:
                pass
        for session_id in tuple(open_opponent):
            try:
                opponent.release_session(session_id)
            except Exception:
                pass


def run_resumable_paired_arena(
    suite: OpeningSuite,
    candidate: MoveSelector,
    opponent: MoveSelector,
    *,
    store_root: str | Path,
    contract_sha256: str,
    actors: int = 64,
    heartbeat: Callable[[int, int], None] | None = None,
) -> OpeningPairStore:
    """Run only missing pairs and return a fully reconciled durable store."""

    validate_opening_suite(suite)
    if candidate is opponent:
        raise ValueError("candidate and opponent selectors must be distinct")
    if type(actors) is not int or actors <= 0:
        raise ValueError("actors must be a positive integer")
    candidate_identity, candidate_runtime = _resolve_participant(
        candidate, None, "candidate"
    )
    opponent_identity, opponent_runtime = _resolve_participant(
        opponent, None, "opponent"
    )
    store = OpeningPairStore(
        store_root,
        contract_sha256=contract_sha256,
        suite=suite,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
    )
    missing = tuple(
        ordinal
        for ordinal in range(len(suite.openings))
        if ordinal not in store.completed_ordinals
    )
    if not missing:
        return store
    candidate_session = _session_selector(candidate)
    opponent_session = _session_selector(opponent)
    if (
        candidate_session is not None
        and opponent_session is not None
        and candidate_session is not opponent_session
    ):
        _run_batched_missing_pairs(
            store=store,
            suite=suite,
            candidate=candidate_session,
            opponent=opponent_session,
            candidate_identity=candidate_identity,
            opponent_identity=opponent_identity,
            candidate_runtime=candidate_runtime,
            opponent_runtime=opponent_runtime,
            missing_ordinals=missing,
            actors=min(actors, len(missing) * 2),
            heartbeat=heartbeat,
        )
    else:
        moves = 0
        for ordinal in missing:
            opening = _snapshot_opening(suite.openings[ordinal])
            records = []
            for leg in _validated_legs(opening):
                record = _play_leg(
                    suite,
                    opening,
                    leg,
                    candidate,
                    opponent,
                    candidate_identity,
                    opponent_identity,
                    candidate_runtime,
                    opponent_runtime,
                )
                moves += len(record.plies)
                records.append(record)
                if heartbeat is not None:
                    heartbeat(moves, store.completed_pairs)
            store.commit_pair(ordinal, records)
            if heartbeat is not None:
                heartbeat(moves, store.completed_pairs)
    store.reconcile()
    return store


__all__ = [
    "PHASE4_ARENA_SCHEMA_VERSION",
    "PHASE4_ARENA_SCHEDULER",
    "PHASE4_ARENA_COMMIT_UNIT",
    "Phase4ArenaError",
    "Phase4ArenaCorruptionError",
    "PairArtifact",
    "OpeningPairStore",
    "verify_completed_arena_result",
    "run_resumable_paired_arena",
]
