"""Auditable paired-opening arena with Python rules as the sole referee."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from types import ModuleType
from typing import Mapping, Protocol, TextIO

from .classic_bridge import CLASSIC_ENGINE, HARD_MAX_DEPTH, ClassicAIBridge
from .codec import position_hash
from .game import State, apply_action_index, get_winner, is_terminal, legal_action_indices
from .mcts import MCTS
from .native_mcts import NativeMCTS, NativeMCTSUnavailable, load_native_module
from .openings import (
    Opening,
    OpeningLeg,
    OpeningSuite,
    validate_opening,
    validate_opening_suite,
)
from .selectors import ClassicHardSelector, MCTSSelector, MoveSelector, SelectionResult
from .statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    PairedEvaluationStatistics,
    PairedGameResult,
    aggregate_opening_pairs,
    summarize_paired_results,
)


PAIRED_ARENA_SCHEMA_VERSION = 1
TERMINATION_REASONS = frozenset({"goal", "repetition", "move-limit"})
GATE_BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED
GATE_BOOTSTRAP_SAMPLES = DEFAULT_BOOTSTRAP_SAMPLES
FINAL_WALL_CLOCK_LIMIT_MS = 850


class PairedArenaError(RuntimeError):
    """Base class for arena contract and adjudication failures."""


class SelectorContractError(PairedArenaError):
    """Raised when a selector returns a malformed result."""


class IllegalSelectionError(PairedArenaError):
    """Raised when a selector chooses an action rejected by the Python rules."""


class TerminalResultError(PairedArenaError):
    """Raised when a terminal game has no decisive, internally valid result."""


class JsonlSink(Protocol):
    """Caller-owned text sink used for durable, per-game streaming."""

    def write(self, text: str) -> int: ...


def _artifact_digest(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must contain 64 lowercase SHA-256 hex digits")
    return value


@dataclass(frozen=True, slots=True)
class SelectorIdentity:
    """Artifact and configuration identity claimed by one arena participant."""

    selector_id: str
    artifact_hash: str | None
    config_hash: str | None
    source_hash: str | None
    engine: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.selector_id, str) or not self.selector_id:
            raise ValueError("selector_id must be a non-empty string")
        for value, label in (
            (self.artifact_hash, "artifact_hash"),
            (self.config_hash, "config_hash"),
            (self.source_hash, "source_hash"),
        ):
            _artifact_digest(value, label)
        if self.engine is not None and (
            not isinstance(self.engine, str) or not self.engine
        ):
            raise ValueError("engine must be a non-empty string or None")

    @property
    def verified(self) -> bool:
        return all(
            value is not None
            for value in (
                self.artifact_hash,
                self.config_hash,
                self.source_hash,
                self.engine,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "selectorId": self.selector_id,
            "artifactHash": self.artifact_hash,
            "configHash": self.config_hash,
            "sourceHash": self.source_hash,
            "engine": self.engine,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class SelectorRuntime:
    """Arena-derived selector implementation and configured search workload."""

    implementation: str
    kind: str
    configured_simulations: int | None
    budget_kind: str | None
    budget_limit: int | None
    max_depth: int | None
    identity_bound: bool
    backend_hash: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("implementation must be a non-empty string")
        if self.kind not in (
            "native-mcts",
            "python-mcts",
            "classic-hard",
            "untrusted",
        ):
            raise ValueError("unsupported selector runtime kind")
        if type(self.identity_bound) is not bool:
            raise ValueError("identity_bound must be a bool")
        _artifact_digest(self.backend_hash, "backend_hash")
        if self.kind in ("native-mcts", "python-mcts") and (
            type(self.configured_simulations) is not int
            or self.configured_simulations <= 0
            or self.budget_kind is not None
            or self.budget_limit is not None
            or self.max_depth is not None
        ):
            raise ValueError("mcts runtime must contain only configured simulations")
        if self.kind == "classic-hard" and (
            self.configured_simulations != 0
            or self.budget_kind not in ("nodes", "time")
            or type(self.budget_limit) is not int
            or self.budget_limit < 0
            or type(self.max_depth) is not int
            or self.max_depth <= 0
        ):
            raise ValueError("classic runtime must contain budget and max depth")
        if self.kind == "untrusted" and any(
            value is not None
            for value in (
                self.configured_simulations,
                self.budget_kind,
                self.budget_limit,
                self.max_depth,
                self.backend_hash,
            )
        ):
            raise ValueError("untrusted runtime must not claim a search configuration")
        if self.kind in ("native-mcts", "classic-hard") and self.backend_hash is None:
            raise ValueError("formal selector runtime must include a backend hash")

    @property
    def trusted(self) -> bool:
        return self.kind in ("native-mcts", "classic-hard")

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "kind": self.kind,
            "trusted": self.trusted,
            "configuredSimulations": self.configured_simulations,
            "budgetKind": self.budget_kind,
            "budgetLimit": self.budget_limit,
            "maxDepth": self.max_depth,
            "identityBound": self.identity_bound,
            "backendHash": self.backend_hash,
        }


class BoundMoveSelector:
    """Caller-owned selector coupled to a store-verified artifact identity."""

    def __init__(self, selector: MoveSelector, identity: SelectorIdentity) -> None:
        if not callable(getattr(selector, "choose", None)):
            raise TypeError("selector must define choose(state)")
        if not callable(getattr(selector, "close", None)):
            raise TypeError("selector must define close()")
        if not isinstance(identity, SelectorIdentity) or not identity.verified:
            raise ValueError("identity must be a verified SelectorIdentity")
        self.selector = selector
        self.identity = identity

    def choose(self, state: State) -> SelectionResult:
        return self.selector.choose(state)

    def close(self) -> None:
        self.selector.close()


@dataclass(frozen=True, slots=True)
class ArenaGatePolicy:
    """Frozen provenance and workload expected by one formal gate run."""

    suite_hash: str
    candidate_identity: SelectorIdentity
    opponent_identity: SelectorIdentity
    candidate_backend_hash: str
    opponent_backend_hash: str
    candidate_simulations: int | None = None
    opponent_budget_kind: str | None = None
    opponent_budget_limit: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.suite_hash, str)
            or not self.suite_hash.startswith("sha256:")
            or _artifact_digest(self.suite_hash.removeprefix("sha256:"), "suite_hash")
            is None
        ):
            raise ValueError("suite_hash must be a content-addressed SHA-256 hash")
        if not isinstance(self.candidate_identity, SelectorIdentity):
            raise TypeError("candidate_identity must be a SelectorIdentity")
        if not isinstance(self.opponent_identity, SelectorIdentity):
            raise TypeError("opponent_identity must be a SelectorIdentity")
        if not self.candidate_identity.verified or not self.opponent_identity.verified:
            raise ValueError("formal gate identities must include artifact/config/source/engine")
        _artifact_digest(self.candidate_backend_hash, "candidate_backend_hash")
        _artifact_digest(self.opponent_backend_hash, "opponent_backend_hash")
        if self.candidate_simulations is not None and (
            type(self.candidate_simulations) is not int
            or self.candidate_simulations <= 0
        ):
            raise ValueError("candidate_simulations must be positive or None")
        if self.opponent_budget_kind not in (None, "nodes", "time"):
            raise ValueError('opponent_budget_kind must be "nodes", "time", or None')
        if (self.opponent_budget_kind is None) != (self.opponent_budget_limit is None):
            raise ValueError("opponent budget kind and limit must be set together")
        if self.opponent_budget_limit is not None and (
            type(self.opponent_budget_limit) is not int
            or self.opponent_budget_limit <= 0
        ):
            raise ValueError("opponent_budget_limit must be positive or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "suiteHash": self.suite_hash,
            "candidate": self.candidate_identity.to_dict(),
            "opponent": self.opponent_identity.to_dict(),
            "candidateBackendHash": self.candidate_backend_hash,
            "opponentBackendHash": self.opponent_backend_hash,
            "candidateSimulations": self.candidate_simulations,
            "opponentBudget": (
                None
                if self.opponent_budget_kind is None
                else {
                    "kind": self.opponent_budget_kind,
                    "limit": self.opponent_budget_limit,
                }
            ),
            "bootstrap": {
                "seed": GATE_BOOTSTRAP_SEED,
                "samples": GATE_BOOTSTRAP_SAMPLES,
            },
        }


@dataclass(frozen=True, slots=True)
class PlyRecord:
    """Immutable selector telemetry and rule transition for one half-move."""

    ply: int
    move_number_before: int
    player: int
    selector_role: str
    position_hash_before: str
    action_index: int
    engine: str
    value: float | None
    score: float | None
    simulations: int
    nodes: int
    elapsed_ms: float
    wall_time_ms: float
    depth: int
    metadata_json: str
    position_hash_after: str

    def __post_init__(self) -> None:
        if type(self.ply) is not int or self.ply < 0:
            raise ValueError("ply must be a non-negative integer")
        if type(self.move_number_before) is not int or self.move_number_before < 0:
            raise ValueError("move_number_before must be a non-negative integer")
        if type(self.player) is not int or self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if self.selector_role not in ("candidate", "opponent"):
            raise ValueError("selector_role must be candidate or opponent")
        if type(self.action_index) is not int or not 0 <= self.action_index < 209:
            raise ValueError("action_index must be an integer in [0, 208]")
        if not isinstance(self.engine, str) or not self.engine:
            raise ValueError("engine must be a non-empty string")
        for value, label in ((self.value, "value"), (self.score, "score")):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite or None")
        for value, label in (
            (self.simulations, "simulations"),
            (self.nodes, "nodes"),
            (self.depth, "depth"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        for value, label in (
            (self.elapsed_ms, "elapsed_ms"),
            (self.wall_time_ms, "wall_time_ms"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        for value, label in (
            (self.position_hash_before, "position_hash_before"),
            (self.position_hash_after, "position_hash_after"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if not isinstance(self.metadata_json, str):
            raise ValueError("metadata_json must be a string")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must contain valid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must contain a JSON object")

    def to_dict(self) -> dict[str, object]:
        return {
            "ply": self.ply,
            "moveNumberBefore": self.move_number_before,
            "player": self.player,
            "selectorRole": self.selector_role,
            "positionHashBefore": self.position_hash_before,
            "actionIndex": self.action_index,
            "selection": {
                "engine": self.engine,
                "value": self.value,
                "score": self.score,
                "simulations": self.simulations,
                "nodes": self.nodes,
                "elapsedMs": self.elapsed_ms,
                "wallTimeMs": self.wall_time_ms,
                "depth": self.depth,
                "metadata": json.loads(self.metadata_json),
            },
            "positionHashAfter": self.position_hash_after,
        }


@dataclass(frozen=True, slots=True)
class OpeningReference:
    """Frozen opening asset fields required to audit both paired legs."""

    opening_id: str
    opening_hash: str
    move_number: int
    initial_position_hash: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.opening_id, "opening_id"),
            (self.opening_hash, "opening_hash"),
            (self.initial_position_hash, "initial_position_hash"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if type(self.move_number) is not int or self.move_number < 0:
            raise ValueError("move_number must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "openingId": self.opening_id,
            "openingHash": self.opening_hash,
            "moveNumber": self.move_number,
            "initialPositionHash": self.initial_position_hash,
        }


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One decisive leg of an opening pair."""

    suite_name: str
    suite_hash: str
    opening_id: str
    opening_hash: str
    leg: int
    candidate_player: int
    opponent_player: int
    candidate_to_move: bool
    candidate_identity: SelectorIdentity
    opponent_identity: SelectorIdentity
    candidate_runtime: SelectorRuntime
    opponent_runtime: SelectorRuntime
    opening_move_number: int
    initial_position_hash: str
    plies: tuple[PlyRecord, ...]
    final_position_hash: str
    final_move_number: int
    winner: int
    termination_reason: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.suite_name, "suite_name"),
            (self.suite_hash, "suite_hash"),
            (self.opening_id, "opening_id"),
            (self.opening_hash, "opening_hash"),
            (self.initial_position_hash, "initial_position_hash"),
            (self.final_position_hash, "final_position_hash"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if type(self.leg) is not int or self.leg not in (0, 1):
            raise ValueError("leg must be 0 or 1")
        if (
            type(self.candidate_player) is not int
            or type(self.opponent_player) is not int
            or {self.candidate_player, self.opponent_player} != {0, 1}
        ):
            raise ValueError("candidate and opponent must own different players")
        if type(self.candidate_to_move) is not bool:
            raise ValueError("candidate_to_move must be a bool")
        if not isinstance(self.candidate_identity, SelectorIdentity):
            raise ValueError("candidate_identity must be a SelectorIdentity")
        if not isinstance(self.opponent_identity, SelectorIdentity):
            raise ValueError("opponent_identity must be a SelectorIdentity")
        if not isinstance(self.candidate_runtime, SelectorRuntime):
            raise ValueError("candidate_runtime must be a SelectorRuntime")
        if not isinstance(self.opponent_runtime, SelectorRuntime):
            raise ValueError("opponent_runtime must be a SelectorRuntime")
        if type(self.winner) is not int or self.winner not in (0, 1):
            raise ValueError("winner must be player 0 or 1")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError("unsupported termination reason")
        if not isinstance(self.plies, tuple) or not self.plies:
            raise ValueError("a completed game must contain at least one ply")
        if any(not isinstance(ply, PlyRecord) for ply in self.plies):
            raise ValueError("plies must contain only PlyRecord values")
        if type(self.opening_move_number) is not int or self.opening_move_number < 0:
            raise ValueError("opening_move_number must be a non-negative integer")
        if type(self.final_move_number) is not int or self.final_move_number < 0:
            raise ValueError("final_move_number must be a non-negative integer")
        if self.final_move_number - self.opening_move_number != len(self.plies):
            raise ValueError("move-number delta must equal the number of recorded plies")
        for index, ply in enumerate(self.plies):
            if ply.ply != index:
                raise ValueError("ply indices must be contiguous from zero")
            if ply.move_number_before != self.opening_move_number + index:
                raise ValueError("ply move numbers must be contiguous")
            if index and ply.player == self.plies[index - 1].player:
                raise ValueError("players must alternate on every ply")
            if index and ply.position_hash_before != self.plies[index - 1].position_hash_after:
                raise ValueError("ply position hashes must form a continuous chain")
            expected_role = (
                "candidate" if ply.player == self.candidate_player else "opponent"
            )
            if ply.selector_role != expected_role:
                raise ValueError("ply selector role disagrees with player ownership")
        first_role = "candidate" if self.candidate_to_move else "opponent"
        if self.plies[0].selector_role != first_role:
            raise ValueError("candidate_to_move disagrees with the first ply")
        if self.plies[0].position_hash_before != self.initial_position_hash:
            raise ValueError("initial position hash disagrees with the first ply")
        if self.plies[-1].position_hash_after != self.final_position_hash:
            raise ValueError("final position hash disagrees with the last ply")

    @property
    def game_id(self) -> str:
        return f"{self.opening_id}:leg-{self.leg}"

    @property
    def candidate_won(self) -> bool:
        return self.winner == self.candidate_player

    def to_paired_result(self) -> PairedGameResult:
        return PairedGameResult(
            opening_id=self.opening_id,
            leg=self.leg,
            candidate_player=self.candidate_player,
            winner=self.winner,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": PAIRED_ARENA_SCHEMA_VERSION,
            "recordType": "paired-arena-game",
            "gameId": self.game_id,
            "suite": self.suite_name,
            "suiteHash": self.suite_hash,
            "openingId": self.opening_id,
            "openingHash": self.opening_hash,
            "leg": self.leg,
            "candidatePlayer": self.candidate_player,
            "opponentPlayer": self.opponent_player,
            "candidateToMove": self.candidate_to_move,
            "candidateIdentity": self.candidate_identity.to_dict(),
            "opponentIdentity": self.opponent_identity.to_dict(),
            "candidateRuntime": self.candidate_runtime.to_dict(),
            "opponentRuntime": self.opponent_runtime.to_dict(),
            "openingMoveNumber": self.opening_move_number,
            "initialPositionHash": self.initial_position_hash,
            "plies": [ply.to_dict() for ply in self.plies],
            "finalPositionHash": self.final_position_hash,
            "finalMoveNumber": self.final_move_number,
            "winner": self.winner,
            "terminationReason": self.termination_reason,
            "candidateWon": self.candidate_won,
        }


@dataclass(frozen=True, slots=True)
class ResultBreakdown:
    games: int
    wins: int
    losses: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.games, self.wins, self.losses)
        ):
            raise ValueError("breakdown counts must be non-negative integers")
        if self.wins + self.losses != self.games:
            raise ValueError("wins plus losses must equal games")

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "winRate": self.win_rate,
        }


@dataclass(frozen=True, slots=True)
class SelectionSummary:
    """Aggregate one selector role without discarding its per-ply records."""

    moves: int
    engines: tuple[tuple[str, int], ...]
    simulations: int
    nodes: int
    simulations_per_move: float
    nodes_per_move: float
    elapsed_ms_p50: float
    elapsed_ms_p95: float
    elapsed_ms_max: float
    wall_time_ms_p50: float
    wall_time_ms_p95: float
    wall_time_ms_max: float
    max_depth: int

    def to_dict(self) -> dict[str, object]:
        return {
            "moves": self.moves,
            "engines": dict(self.engines),
            "simulations": self.simulations,
            "nodes": self.nodes,
            "simulationsPerMove": self.simulations_per_move,
            "nodesPerMove": self.nodes_per_move,
            "elapsedMs": {
                "p50": self.elapsed_ms_p50,
                "p95": self.elapsed_ms_p95,
                "max": self.elapsed_ms_max,
            },
            "wallTimeMs": {
                "p50": self.wall_time_ms_p50,
                "p95": self.wall_time_ms_p95,
                "max": self.wall_time_ms_max,
            },
            "maxDepth": self.max_depth,
        }


@dataclass(frozen=True, slots=True)
class PairedArenaResult:
    """Immutable game records plus paired and side-aware summaries."""

    suite_name: str
    suite_hash: str
    candidate_identity: SelectorIdentity
    opponent_identity: SelectorIdentity
    candidate_runtime: SelectorRuntime
    opponent_runtime: SelectorRuntime
    opening_manifest: tuple[OpeningReference, ...]
    games: tuple[GameRecord, ...]
    statistics: PairedEvaluationStatistics
    candidate_to_move: ResultBreakdown
    candidate_not_to_move: ResultBreakdown
    candidate_as_player_zero: ResultBreakdown
    candidate_as_player_one: ResultBreakdown
    termination_counts: tuple[tuple[str, int], ...]
    candidate_selection: SelectionSummary
    opponent_selection: SelectionSummary

    def __post_init__(self) -> None:
        if not isinstance(self.games, tuple) or not self.games:
            raise ValueError("arena result must contain completed games")
        if any(not isinstance(game, GameRecord) for game in self.games):
            raise ValueError("games must contain only GameRecord values")
        if not isinstance(self.candidate_identity, SelectorIdentity):
            raise ValueError("candidate_identity must be a SelectorIdentity")
        if not isinstance(self.opponent_identity, SelectorIdentity):
            raise ValueError("opponent_identity must be a SelectorIdentity")
        if not isinstance(self.candidate_runtime, SelectorRuntime):
            raise ValueError("candidate_runtime must be a SelectorRuntime")
        if not isinstance(self.opponent_runtime, SelectorRuntime):
            raise ValueError("opponent_runtime must be a SelectorRuntime")
        if (
            not isinstance(self.opening_manifest, tuple)
            or not self.opening_manifest
            or any(
                not isinstance(reference, OpeningReference)
                for reference in self.opening_manifest
            )
        ):
            raise ValueError("opening_manifest must contain OpeningReference values")
        if len({reference.opening_id for reference in self.opening_manifest}) != len(
            self.opening_manifest
        ):
            raise ValueError("opening manifest identifiers must be unique")
        if any(
            game.suite_name != self.suite_name
            or game.suite_hash != self.suite_hash
            or game.candidate_identity != self.candidate_identity
            or game.opponent_identity != self.opponent_identity
            or game.candidate_runtime != self.candidate_runtime
            or game.opponent_runtime != self.opponent_runtime
            for game in self.games
        ):
            raise ValueError("every game must reference the result's opening suite")
        if len({game.game_id for game in self.games}) != len(self.games):
            raise ValueError("game identifiers must be unique")
        if len(self.games) != self.statistics.games:
            raise ValueError("game records and paired statistics disagree")
        pairs = aggregate_opening_pairs(
            game.to_paired_result() for game in self.games
        )
        references = {
            reference.opening_id: reference for reference in self.opening_manifest
        }
        if set(references) != {pair.opening_id for pair in pairs}:
            raise ValueError("opening manifest does not match paired game identifiers")
        for game in self.games:
            reference = references[game.opening_id]
            if (
                game.opening_hash != reference.opening_hash
                or game.opening_move_number != reference.move_number
                or game.initial_position_hash != reference.initial_position_hash
            ):
                raise ValueError("game opening fields do not match the opening manifest")
        games_by_opening: dict[str, list[GameRecord]] = {}
        for game in self.games:
            games_by_opening.setdefault(game.opening_id, []).append(game)
        for opening_id, opening_games in games_by_opening.items():
            leg_zero, leg_one = sorted(opening_games, key=lambda game: game.leg)
            try:
                side_to_move = json.loads(
                    references[opening_id].initial_position_hash
                )[1]
            except (json.JSONDecodeError, IndexError, TypeError) as error:
                raise ValueError("opening manifest contains an invalid position hash") from error
            if (
                side_to_move not in (0, 1)
                or not leg_zero.candidate_to_move
                or leg_one.candidate_to_move
                or leg_zero.candidate_player != side_to_move
                or leg_one.candidate_player != 1 - side_to_move
            ):
                raise ValueError("paired games do not implement the opening side swap")
        candidate_wins = sum(game.candidate_won for game in self.games)
        if (
            len(pairs) != self.statistics.opening_pairs
            or candidate_wins != self.statistics.candidate_wins
            or len(self.games) - candidate_wins != self.statistics.candidate_losses
            or candidate_wins / len(self.games) != self.statistics.win_rate
        ):
            raise ValueError("paired statistics do not match the game records")
        expected_terminations = tuple(
            (reason, sum(game.termination_reason == reason for game in self.games))
            for reason in sorted(TERMINATION_REASONS)
        )
        if self.termination_counts != expected_terminations:
            raise ValueError("termination counts do not match the game records")
        if self.candidate_to_move != _breakdown(
            tuple(game for game in self.games if game.candidate_to_move)
        ):
            raise ValueError("candidate-to-move breakdown does not match games")
        if self.candidate_not_to_move != _breakdown(
            tuple(game for game in self.games if not game.candidate_to_move)
        ):
            raise ValueError("candidate-not-to-move breakdown does not match games")
        if self.candidate_as_player_zero != _breakdown(
            tuple(game for game in self.games if game.candidate_player == 0)
        ):
            raise ValueError("player-zero breakdown does not match games")
        if self.candidate_as_player_one != _breakdown(
            tuple(game for game in self.games if game.candidate_player == 1)
        ):
            raise ValueError("player-one breakdown does not match games")
        if self.candidate_selection != _selection_summary(self.games, "candidate"):
            raise ValueError("candidate selection summary does not match plies")
        if self.opponent_selection != _selection_summary(self.games, "opponent"):
            raise ValueError("opponent selection summary does not match plies")
        if self.candidate_identity.engine is not None and any(
            ply.engine != self.candidate_identity.engine
            for game in self.games
            for ply in game.plies
            if ply.selector_role == "candidate"
        ):
            raise ValueError("candidate engine identity does not match plies")
        if self.opponent_identity.engine is not None and any(
            ply.engine != self.opponent_identity.engine
            for game in self.games
            for ply in game.plies
            if ply.selector_role == "opponent"
        ):
            raise ValueError("opponent engine identity does not match plies")

    def to_dict(self) -> dict[str, object]:
        paired = self.statistics.to_dict()
        paired.update(
            {
                "candidateToMove": self.candidate_to_move.to_dict(),
                "candidateNotToMove": self.candidate_not_to_move.to_dict(),
                "candidateAsPlayer0": self.candidate_as_player_zero.to_dict(),
                "candidateAsPlayer1": self.candidate_as_player_one.to_dict(),
                "terminationCounts": dict(self.termination_counts),
                "selectionStats": {
                    "candidate": self.candidate_selection.to_dict(),
                    "opponent": self.opponent_selection.to_dict(),
                },
            }
        )
        return {
            "schemaVersion": PAIRED_ARENA_SCHEMA_VERSION,
            "recordType": "paired-arena-result",
            "suite": self.suite_name,
            "suiteHash": self.suite_hash,
            "candidateIdentity": self.candidate_identity.to_dict(),
            "opponentIdentity": self.opponent_identity.to_dict(),
            "candidateRuntime": self.candidate_runtime.to_dict(),
            "opponentRuntime": self.opponent_runtime.to_dict(),
            "openingManifest": [
                reference.to_dict() for reference in self.opening_manifest
            ],
            "summary": paired,
            "games": [game.to_dict() for game in self.games],
        }


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    actual: str | int | float | bool
    requirement: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class GateDecision:
    gate: str
    passed: bool
    interval_kind: str
    statistics: PairedEvaluationStatistics
    checks: tuple[GateCheck, ...]
    policy: ArenaGatePolicy | None

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "intervalKind": self.interval_kind,
            "policy": None if self.policy is None else self.policy.to_dict(),
            "statistics": self.statistics.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
        }


def _normalize_json_value(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SelectorContractError(f"{label} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise SelectorContractError(f"{label} keys must be strings")
            normalized[key] = _normalize_json_value(nested, f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(nested, f"{label}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise SelectorContractError(f"{label} contains a non-JSON value")


def _metadata_json(metadata: Mapping[str, object]) -> str:
    normalized = _normalize_json_value(metadata, "selection metadata")
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_participants(
    candidate: MoveSelector,
    opponent: MoveSelector,
    jsonl_sink: JsonlSink | None,
) -> None:
    for selector, label in ((candidate, "candidate"), (opponent, "opponent")):
        if not callable(getattr(selector, "choose", None)):
            raise TypeError(f"{label} must define choose(state)")
    if jsonl_sink is not None and not callable(getattr(jsonl_sink, "write", None)):
        raise TypeError("jsonl_sink must define write(text)")


def _unverified_identity(selector: MoveSelector, role: str) -> SelectorIdentity:
    selector_type = type(selector)
    return SelectorIdentity(
        selector_id=(
            f"unverified:{role}:{selector_type.__module__}.{selector_type.__qualname__}"
        ),
        artifact_hash=None,
        config_hash=None,
        source_hash=None,
        engine=None,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_classic_backend_hash() -> str:
    """Hash the canonical bridge and every JavaScript rules/search dependency."""

    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        Path("scripts/classic-ai-bridge.mjs"),
        Path("src/ai.js"),
        Path("src/game.js"),
        Path("src/alphazero/state-codec.js"),
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.as_posix().encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def loaded_native_backend_hash() -> str:
    """Hash the extension binary loaded by the canonical NativeMCTS adapter."""

    module = load_native_module()
    specification = getattr(module, "__spec__", None)
    loader = getattr(specification, "loader", None)
    origin_value = getattr(specification, "origin", None)
    discovered = importlib.machinery.PathFinder.find_spec("quoridor_native")
    discovered_loader = getattr(discovered, "loader", None)
    discovered_origin = getattr(discovered, "origin", None)
    path_value = getattr(module, "__file__", None)
    if (
        not isinstance(loader, importlib.machinery.ExtensionFileLoader)
        or not isinstance(
            discovered_loader,
            importlib.machinery.ExtensionFileLoader,
        )
        or getattr(module, "__loader__", None) is not loader
        or getattr(specification, "name", None) != "quoridor_native"
        or not isinstance(origin_value, str)
        or not isinstance(discovered_origin, str)
        or not isinstance(path_value, str)
    ):
        raise NativeMCTSUnavailable(
            "quoridor_native must be loaded by CPython as a native extension"
        )
    if not path_value.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise NativeMCTSUnavailable(
            "quoridor_native path does not have a native extension suffix"
        )
    if Path(origin_value).resolve() != Path(path_value).resolve():
        raise NativeMCTSUnavailable("quoridor_native does not expose an extension path")
    if Path(discovered_origin).resolve() != Path(path_value).resolve():
        raise NativeMCTSUnavailable(
            "quoridor_native is not the native extension selected from sys.path"
        )
    path = Path(path_value).resolve()
    if not path.is_file():
        raise NativeMCTSUnavailable("quoridor_native extension path does not exist")
    return _file_sha256(path)


def _native_runtime_backend_hash(search: NativeMCTS) -> str | None:
    module = search.module
    try:
        canonical_module = load_native_module()
        binary_hash = loaded_native_backend_hash()
    except NativeMCTSUnavailable:
        return None
    backend = getattr(search, "_search", None)
    if (
        not isinstance(module, ModuleType)
        or module is not canonical_module
        or getattr(module, "__name__", None) != "quoridor_native"
        or getattr(module, "MCTS", None) is not type(backend)
        or getattr(type(backend), "__module__", None) != "quoridor_native"
    ):
        return None
    return binary_hash


def _classic_runtime_backend_hash(bridge: ClassicAIBridge) -> str | None:
    process = getattr(bridge, "_process", None)
    raw_args = getattr(process, "args", None)
    if isinstance(raw_args, (str, bytes)):
        return None
    try:
        args = tuple(raw_args)
    except TypeError:
        return None
    root = Path(__file__).resolve().parents[2]
    expected_script = (root / "scripts" / "classic-ai-bridge.mjs").resolve()
    node = shutil.which("node")
    if (
        len(args) != 2
        or node is None
        or not all(isinstance(part, str) and part for part in args)
        or Path(args[0]).resolve() != Path(node).resolve()
        or Path(args[1]).resolve() != expected_script
        or Path(getattr(bridge, "_root", "")).resolve() != root
    ):
        return None
    return canonical_classic_backend_hash()


def _selector_runtime(
    selector: MoveSelector,
    *,
    identity_bound: bool,
) -> SelectorRuntime:
    selector_type = type(selector)
    implementation = f"{selector_type.__module__}.{selector_type.__qualname__}"
    if selector_type is MCTSSelector:
        search_type = type(selector._search)
        if search_type is NativeMCTS:
            backend_hash = _native_runtime_backend_hash(selector._search)
            if backend_hash is not None:
                search_implementation = (
                    f"{search_type.__module__}.{search_type.__qualname__}"
                )
                return SelectorRuntime(
                    implementation=f"{implementation}[{search_implementation}]",
                    kind="native-mcts",
                    configured_simulations=selector.config.simulations,
                    budget_kind=None,
                    budget_limit=None,
                    max_depth=None,
                    identity_bound=identity_bound,
                    backend_hash=backend_hash,
                )
        if search_type is MCTS:
            search_implementation = (
                f"{search_type.__module__}.{search_type.__qualname__}"
            )
            return SelectorRuntime(
                implementation=f"{implementation}[{search_implementation}]",
                kind="python-mcts",
                configured_simulations=selector.config.simulations,
                budget_kind=None,
                budget_limit=None,
                max_depth=None,
                identity_bound=identity_bound,
                backend_hash=None,
            )
    if (
        selector_type is ClassicHardSelector
        and type(selector._bridge) is ClassicAIBridge
    ):
        backend_hash = _classic_runtime_backend_hash(selector._bridge)
        if backend_hash is not None:
            return SelectorRuntime(
                implementation=implementation,
                kind="classic-hard",
                configured_simulations=0,
                budget_kind=selector.budget.kind,
                budget_limit=selector.budget.limit,
                max_depth=selector.max_depth,
                identity_bound=identity_bound,
                backend_hash=backend_hash,
            )
    return SelectorRuntime(
        implementation=implementation,
        kind="untrusted",
        configured_simulations=None,
        budget_kind=None,
        budget_limit=None,
        max_depth=None,
        identity_bound=identity_bound,
        backend_hash=None,
    )


def _resolve_participant(
    selector: MoveSelector,
    supplied_identity: SelectorIdentity | None,
    role: str,
) -> tuple[SelectorIdentity, SelectorRuntime]:
    if type(selector) is BoundMoveSelector:
        bound = selector
        if supplied_identity is not None and supplied_identity != bound.identity:
            raise ValueError(f"{role}_identity disagrees with the bound selector")
        return bound.identity, _selector_runtime(
            bound.selector,
            identity_bound=True,
        )
    identity = supplied_identity or _unverified_identity(selector, role)
    return identity, _selector_runtime(selector, identity_bound=False)


def inspect_selector_runtime(selector: MoveSelector) -> SelectorRuntime:
    """Return the arena-derived backend attestation without taking ownership."""

    if type(selector) is BoundMoveSelector:
        return _selector_runtime(selector.selector, identity_bound=True)
    return _selector_runtime(selector, identity_bound=False)


def _validate_run_settings(
    bootstrap_seed: object,
    bootstrap_samples: object,
    confidence: object,
) -> None:
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**63:
        raise ValueError("bootstrap_seed must be an integer in [0, 2^63)")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 < float(confidence) < 1
    ):
        raise ValueError("confidence must be between 0 and 1")


def _validate_leg(opening: Opening, leg: OpeningLeg) -> None:
    if leg.opening_id != opening.id:
        raise PairedArenaError("opening leg identifier mismatch")
    if {leg.candidate_player, leg.opponent_player} != {0, 1}:
        raise PairedArenaError("opening leg does not assign both players")
    expected_to_move = leg.candidate_player == opening.state.current_player
    if leg.candidate_to_move != expected_to_move:
        raise PairedArenaError("opening leg candidate-to-move flag is inconsistent")


def _validated_legs(opening: Opening) -> tuple[OpeningLeg, OpeningLeg]:
    legs = Opening.paired_legs(opening)
    if len(legs) != 2 or any(not isinstance(leg, OpeningLeg) for leg in legs):
        raise PairedArenaError("each opening must define exactly two OpeningLeg values")
    ordered = tuple(sorted(legs, key=lambda leg: leg.leg))
    if tuple(leg.leg for leg in ordered) != (0, 1):
        raise PairedArenaError("each opening must define exactly legs 0 and 1")
    for leg in ordered:
        _validate_leg(opening, leg)
    side_to_move = opening.state.current_player
    if (
        ordered[0].candidate_player != side_to_move
        or not ordered[0].candidate_to_move
        or ordered[1].candidate_player != 1 - side_to_move
        or ordered[1].candidate_to_move
        or {leg.candidate_player for leg in ordered} != {0, 1}
    ):
        raise PairedArenaError(
            "opening legs must exchange candidate ownership with leg 0 to move"
        )
    return ordered[0], ordered[1]


def _snapshot_opening(opening: Opening) -> Opening:
    snapshot = Opening(
        id=opening.id,
        content_hash=opening.content_hash,
        suite=opening.suite,
        starting_player=opening.starting_player,
        actions=tuple(opening.actions),
        state=replace(
            opening.state,
            position_counts=MappingProxyType(dict(opening.state.position_counts)),
        ),
    )
    validate_opening(snapshot)
    return snapshot


def _referee_state(opening: Opening) -> State:
    return replace(
        opening.state,
        position_counts=dict(opening.state.position_counts),
    )


def _selector_view(state: State) -> State:
    return replace(
        state,
        position_counts=MappingProxyType(dict(state.position_counts)),
    )


def _play_leg(
    suite: OpeningSuite,
    opening: Opening,
    leg: OpeningLeg,
    candidate: MoveSelector,
    opponent: MoveSelector,
    candidate_identity: SelectorIdentity,
    opponent_identity: SelectorIdentity,
    candidate_runtime: SelectorRuntime,
    opponent_runtime: SelectorRuntime,
) -> GameRecord:
    _validate_leg(opening, leg)
    state = _referee_state(opening)
    initial_hash = position_hash(state)
    plies: list[PlyRecord] = []

    while not is_terminal(state):
        player = state.current_player
        is_candidate = player == leg.candidate_player
        selector = candidate if is_candidate else opponent

        # Selector exceptions deliberately propagate unchanged to their owner.
        started_at = time.monotonic()
        selection = selector.choose(_selector_view(state))
        wall_time_ms = (time.monotonic() - started_at) * 1_000.0
        if not isinstance(selection, SelectionResult):
            raise SelectorContractError(
                f"{('candidate' if is_candidate else 'opponent')} selector must "
                "return SelectionResult"
            )
        identity = candidate_identity if is_candidate else opponent_identity
        if identity.engine is not None and selection.engine != identity.engine:
            raise SelectorContractError(
                f"{('candidate' if is_candidate else 'opponent')} engine mismatch: "
                f"expected {identity.engine!r}, received {selection.engine!r}"
            )

        legal = frozenset(legal_action_indices(state))
        if selection.action_index not in legal:
            raise IllegalSelectionError(
                f"{('candidate' if is_candidate else 'opponent')} selected illegal action "
                f"{selection.action_index} at opening {opening.id!r}, leg {leg.leg}, "
                f"move {state.move_number}"
            )
        metadata_json = _metadata_json(selection.metadata)
        before_hash = position_hash(state)
        next_state = apply_action_index(state, selection.action_index)
        after_hash = position_hash(next_state)
        plies.append(
            PlyRecord(
                ply=len(plies),
                move_number_before=state.move_number,
                player=player,
                selector_role="candidate" if is_candidate else "opponent",
                position_hash_before=before_hash,
                action_index=selection.action_index,
                engine=selection.engine,
                value=None if selection.value is None else float(selection.value),
                score=None if selection.score is None else float(selection.score),
                simulations=selection.simulations,
                nodes=selection.nodes,
                elapsed_ms=float(selection.elapsed_ms),
                wall_time_ms=wall_time_ms,
                depth=selection.depth,
                metadata_json=metadata_json,
                position_hash_after=after_hash,
            )
        )
        state = next_state

    winner = get_winner(state)
    if type(winner) is not int or winner not in (0, 1):
        raise TerminalResultError(
            f"opening {opening.id!r}, leg {leg.leg} ended without a winner"
        )
    if state.termination_reason not in TERMINATION_REASONS:
        raise TerminalResultError(
            f"opening {opening.id!r}, leg {leg.leg} has invalid termination reason "
            f"{state.termination_reason!r}"
        )

    return GameRecord(
        suite_name=suite.name,
        suite_hash=suite.content_hash,
        opening_id=opening.id,
        opening_hash=opening.content_hash,
        leg=leg.leg,
        candidate_player=leg.candidate_player,
        opponent_player=leg.opponent_player,
        candidate_to_move=leg.candidate_to_move,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
        opening_move_number=opening.state.move_number,
        initial_position_hash=initial_hash,
        plies=tuple(plies),
        final_position_hash=position_hash(state),
        final_move_number=state.move_number,
        winner=winner,
        termination_reason=state.termination_reason,
    )


def _breakdown(games: tuple[GameRecord, ...]) -> ResultBreakdown:
    wins = sum(game.candidate_won for game in games)
    return ResultBreakdown(games=len(games), wins=wins, losses=len(games) - wins)


def _percentile(values: tuple[float, ...], probability: float) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _selection_summary(
    games: tuple[GameRecord, ...],
    role: str,
) -> SelectionSummary:
    plies = tuple(
        ply for game in games for ply in game.plies if ply.selector_role == role
    )
    engine_names = sorted({ply.engine for ply in plies})
    engines = tuple(
        (engine, sum(ply.engine == engine for ply in plies))
        for engine in engine_names
    )
    moves = len(plies)
    simulations = sum(ply.simulations for ply in plies)
    nodes = sum(ply.nodes for ply in plies)
    elapsed = tuple(ply.elapsed_ms for ply in plies)
    wall_time = tuple(ply.wall_time_ms for ply in plies)
    return SelectionSummary(
        moves=moves,
        engines=engines,
        simulations=simulations,
        nodes=nodes,
        simulations_per_move=simulations / moves if moves else 0.0,
        nodes_per_move=nodes / moves if moves else 0.0,
        elapsed_ms_p50=_percentile(elapsed, 0.50),
        elapsed_ms_p95=_percentile(elapsed, 0.95),
        elapsed_ms_max=max(elapsed, default=0.0),
        wall_time_ms_p50=_percentile(wall_time, 0.50),
        wall_time_ms_p95=_percentile(wall_time, 0.95),
        wall_time_ms_max=max(wall_time, default=0.0),
        max_depth=max((ply.depth for ply in plies), default=0),
    )


def _write_game(sink: JsonlSink, game: GameRecord) -> None:
    payload = json.dumps(
        game.to_dict(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    line = f"{payload}\n"
    written = sink.write(line)
    if written != len(line):
        raise OSError(
            f"jsonl_sink short write: expected {len(line)} characters, wrote {written!r}"
        )
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()


def run_paired_arena(
    suite: OpeningSuite,
    candidate: MoveSelector,
    opponent: MoveSelector,
    *,
    jsonl_sink: TextIO | JsonlSink | None = None,
    candidate_identity: SelectorIdentity | None = None,
    opponent_identity: SelectorIdentity | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedArenaResult:
    """Play exactly two exchanged-player legs for every validated opening.

    ``candidate``, ``opponent``, and ``jsonl_sink`` remain caller-owned.  This
    function never closes them, including when a game fails.
    """

    validate_opening_suite(suite)
    _validate_participants(candidate, opponent, jsonl_sink)
    if candidate_identity is not None and not isinstance(
        candidate_identity, SelectorIdentity
    ):
        raise TypeError("candidate_identity must be a SelectorIdentity or None")
    if opponent_identity is not None and not isinstance(
        opponent_identity, SelectorIdentity
    ):
        raise TypeError("opponent_identity must be a SelectorIdentity or None")
    candidate_identity, candidate_runtime = _resolve_participant(
        candidate,
        candidate_identity,
        "candidate",
    )
    opponent_identity, opponent_runtime = _resolve_participant(
        opponent,
        opponent_identity,
        "opponent",
    )
    _validate_run_settings(bootstrap_seed, bootstrap_samples, confidence)
    records: list[GameRecord] = []
    opening_snapshots = tuple(_snapshot_opening(opening) for opening in suite.openings)
    schedule = tuple(
        (opening, _validated_legs(opening)) for opening in opening_snapshots
    )

    for opening, legs in schedule:
        for leg in legs:
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
            records.append(record)
            if jsonl_sink is not None:
                _write_game(jsonl_sink, record)

    games = tuple(records)
    statistics = summarize_paired_results(
        (game.to_paired_result() for game in games),
        seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    candidate_to_move = tuple(game for game in games if game.candidate_to_move)
    candidate_not_to_move = tuple(game for game in games if not game.candidate_to_move)
    candidate_as_zero = tuple(game for game in games if game.candidate_player == 0)
    candidate_as_one = tuple(game for game in games if game.candidate_player == 1)
    terminations = tuple(
        (reason, sum(game.termination_reason == reason for game in games))
        for reason in sorted(TERMINATION_REASONS)
    )
    return PairedArenaResult(
        suite_name=suite.name,
        suite_hash=suite.content_hash,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        candidate_runtime=candidate_runtime,
        opponent_runtime=opponent_runtime,
        opening_manifest=tuple(
            OpeningReference(
                opening_id=opening.id,
                opening_hash=opening.content_hash,
                move_number=opening.state.move_number,
                initial_position_hash=position_hash(opening.state),
            )
            for opening in opening_snapshots
        ),
        games=games,
        statistics=statistics,
        candidate_to_move=_breakdown(candidate_to_move),
        candidate_not_to_move=_breakdown(candidate_not_to_move),
        candidate_as_player_zero=_breakdown(candidate_as_zero),
        candidate_as_player_one=_breakdown(candidate_as_one),
        termination_counts=terminations,
        candidate_selection=_selection_summary(games, "candidate"),
        opponent_selection=_selection_summary(games, "opponent"),
    )


def _gate_statistics(
    result: PairedArenaResult,
    *,
    one_sided: bool,
) -> PairedEvaluationStatistics:
    # A central 90% percentile interval has the same 5% lower tail as a
    # one-sided 95% percentile lower bound.
    confidence = 0.90 if one_sided else 0.95
    return summarize_paired_results(
        (game.to_paired_result() for game in result.games),
        seed=GATE_BOOTSTRAP_SEED,
        bootstrap_samples=GATE_BOOTSTRAP_SAMPLES,
        confidence=confidence,
    )


def _decision(
    gate: str,
    interval_kind: str,
    statistics: PairedEvaluationStatistics,
    checks: tuple[GateCheck, ...],
    policy: ArenaGatePolicy | None,
) -> GateDecision:
    return GateDecision(
        gate=gate,
        passed=all(check.passed for check in checks),
        interval_kind=interval_kind,
        statistics=statistics,
        checks=checks,
        policy=policy,
    )


def _provenance_checks(
    result: PairedArenaResult,
    policy: ArenaGatePolicy | None,
) -> tuple[GateCheck, ...]:
    if policy is None:
        return (
            GateCheck(
                "gatePolicy",
                False,
                False,
                "a frozen suite and verified participant policy is required",
            ),
        )
    return (
        GateCheck("gatePolicy", True, True, "configured"),
        GateCheck(
            "suiteHash",
            result.suite_hash == policy.suite_hash,
            result.suite_hash,
            policy.suite_hash,
        ),
        GateCheck(
            "candidateIdentity",
            result.candidate_identity == policy.candidate_identity,
            result.candidate_identity.selector_id,
            policy.candidate_identity.selector_id,
        ),
        GateCheck(
            "opponentIdentity",
            result.opponent_identity == policy.opponent_identity,
            result.opponent_identity.selector_id,
            policy.opponent_identity.selector_id,
        ),
        GateCheck(
            "candidateIdentityBinding",
            result.candidate_runtime.identity_bound,
            result.candidate_runtime.identity_bound,
            "identity is coupled through BoundMoveSelector",
        ),
        GateCheck(
            "opponentIdentityBinding",
            result.opponent_runtime.identity_bound,
            result.opponent_runtime.identity_bound,
            "identity is coupled through BoundMoveSelector",
        ),
        GateCheck(
            "candidateBackendHash",
            result.candidate_runtime.backend_hash
            == policy.candidate_backend_hash,
            result.candidate_runtime.backend_hash or "untrusted",
            policy.candidate_backend_hash,
        ),
        GateCheck(
            "opponentBackendHash",
            result.opponent_runtime.backend_hash
            == policy.opponent_backend_hash,
            result.opponent_runtime.backend_hash or "untrusted",
            policy.opponent_backend_hash,
        ),
    )


def _bootstrap_checks(
    statistics: PairedEvaluationStatistics,
) -> tuple[GateCheck, GateCheck]:
    return (
        GateCheck(
            "bootstrapSeed",
            statistics.bootstrap_seed == GATE_BOOTSTRAP_SEED,
            statistics.bootstrap_seed,
            f"exactly {GATE_BOOTSTRAP_SEED}",
        ),
        GateCheck(
            "bootstrapSamples",
            statistics.bootstrap_samples == GATE_BOOTSTRAP_SAMPLES,
            statistics.bootstrap_samples,
            f"exactly {GATE_BOOTSTRAP_SAMPLES}",
        ),
    )


def _simulation_check(
    result: PairedArenaResult,
    expected: int,
) -> GateCheck:
    observed = tuple(
        sorted({ply.simulations for game in result.games for ply in game.plies})
    )
    actual: int | str = observed[0] if len(observed) == 1 else repr(observed)
    return GateCheck(
        "simulationsPerMove",
        observed == (expected,),
        actual,
        f"exactly {expected}",
    )


def _neural_runtime_checks(
    result: PairedArenaResult,
    expected_simulations: int,
) -> tuple[GateCheck, GateCheck]:
    return (
        GateCheck(
            "candidateRuntime",
            result.candidate_runtime.kind == "native-mcts"
            and result.candidate_runtime.configured_simulations
            == expected_simulations,
            (
                f"{result.candidate_runtime.kind}:"
                f"{result.candidate_runtime.configured_simulations}"
            ),
            f"trusted native-mcts:{expected_simulations}",
        ),
        GateCheck(
            "opponentRuntime",
            result.opponent_runtime.kind == "native-mcts"
            and result.opponent_runtime.configured_simulations
            == expected_simulations,
            (
                f"{result.opponent_runtime.kind}:"
                f"{result.opponent_runtime.configured_simulations}"
            ),
            f"trusted native-mcts:{expected_simulations}",
        ),
    )
def _role_simulation_check(
    result: PairedArenaResult,
    role: str,
    expected: int,
) -> GateCheck:
    observed = tuple(
        sorted(
            {
                ply.simulations
                for game in result.games
                for ply in game.plies
                if ply.selector_role == role
            }
        )
    )
    actual: int | str = observed[0] if len(observed) == 1 else repr(observed)
    return GateCheck(
        f"{role}SimulationsPerMove",
        observed == (expected,),
        actual,
        f"exactly {expected}",
    )


def _opponent_budget_check(
    result: PairedArenaResult,
    kind: str,
    limit: int,
) -> GateCheck:
    observed = tuple(
        (
            json.loads(ply.metadata_json).get("budgetKind"),
            json.loads(ply.metadata_json).get("budgetLimit"),
        )
        for game in result.games
        for ply in game.plies
        if ply.selector_role == "opponent"
    )
    expected = (kind, limit)
    return GateCheck(
        "opponentBudget",
        bool(observed) and all(value == expected for value in observed),
        repr(observed),
        f"every opponent move uses {expected!r}",
    )


def _opponent_node_count_check(
    result: PairedArenaResult,
    limit: int,
) -> GateCheck:
    observed = tuple(
        ply.nodes
        for game in result.games
        for ply in game.plies
        if ply.selector_role == "opponent"
    )
    within_budget = bool(observed) and all(0 < nodes <= limit for nodes in observed)
    return GateCheck(
        "opponentNodeCount",
        within_budget,
        f"min={min(observed, default=0)},max={max(observed, default=0)}",
        f"every move in [1, {limit}]",
    )


def evaluate_screen_gate(
    result: PairedArenaResult,
    *,
    policy: ArenaGatePolicy | None = None,
) -> GateDecision:
    """Validate the 32-pair dev screening workload; it asserts no strength claim."""

    checks = (
        *_provenance_checks(result, policy),
        GateCheck("suite", result.suite_name == "dev", result.suite_name, "dev"),
        GateCheck(
            "openingPairs",
            result.statistics.opening_pairs == 32,
            result.statistics.opening_pairs,
            "exactly 32",
        ),
        GateCheck(
            "games",
            result.statistics.games == 64,
            result.statistics.games,
            "exactly 64",
        ),
        *_neural_runtime_checks(result, 256),
        _simulation_check(result, 256),
    )
    return _decision(
        "screen",
        "not-applicable",
        result.statistics,
        checks,
        policy,
    )


def evaluate_confirmation_gate(
    result: PairedArenaResult,
    *,
    policy: ArenaGatePolicy | None = None,
) -> GateDecision:
    """Apply the formal champion promotion thresholds from Phase 4."""

    statistics = _gate_statistics(
        result,
        one_sided=True,
    )
    checks = (
        *_provenance_checks(result, policy),
        *_bootstrap_checks(statistics),
        GateCheck("suite", result.suite_name == "dev", result.suite_name, "dev"),
        GateCheck(
            "openingPairs",
            statistics.opening_pairs == 200,
            statistics.opening_pairs,
            "exactly 200",
        ),
        GateCheck("games", statistics.games == 400, statistics.games, "exactly 400"),
        GateCheck("winRate", statistics.win_rate >= 0.55, statistics.win_rate, ">= 0.55"),
        GateCheck(
            "oneSided95Lower",
            statistics.win_rate_ci.lower > 0.50,
            statistics.win_rate_ci.lower,
            "> 0.50",
        ),
        *_neural_runtime_checks(result, 800),
        _simulation_check(result, 800),
    )
    return _decision(
        "champion-confirmation",
        "one-sided-95-percent-lower",
        statistics,
        checks,
        policy,
    )


def evaluate_noninferiority_gate(
    result: PairedArenaResult,
    *,
    policy: ArenaGatePolicy | None = None,
) -> GateDecision:
    """Apply the historical/product champion non-inferiority thresholds."""

    statistics = _gate_statistics(
        result,
        one_sided=True,
    )
    checks = (
        *_provenance_checks(result, policy),
        *_bootstrap_checks(statistics),
        GateCheck("suite", result.suite_name == "dev", result.suite_name, "dev"),
        GateCheck(
            "openingPairs",
            statistics.opening_pairs == 200,
            statistics.opening_pairs,
            "exactly 200",
        ),
        GateCheck("games", statistics.games == 400, statistics.games, "exactly 400"),
        GateCheck("winRate", statistics.win_rate >= 0.50, statistics.win_rate, ">= 0.50"),
        GateCheck(
            "oneSided95Lower",
            statistics.win_rate_ci.lower > 0.45,
            statistics.win_rate_ci.lower,
            "> 0.45",
        ),
        *_neural_runtime_checks(result, 800),
        _simulation_check(result, 800),
    )
    return _decision(
        "champion-noninferiority",
        "one-sided-95-percent-lower",
        statistics,
        checks,
        policy,
    )


def evaluate_final_hard_gate(
    result: PairedArenaResult,
    *,
    track: str,
    policy: ArenaGatePolicy | None = None,
) -> GateDecision:
    """Apply the locked-final fixed-workload or wall-clock hard-AI gate."""

    if track not in ("fixed", "wall-clock"):
        raise ValueError('track must be "fixed" or "wall-clock"')
    statistics = _gate_statistics(
        result,
        one_sided=False,
    )
    checks: list[GateCheck] = [
        *_provenance_checks(result, policy),
        *_bootstrap_checks(statistics),
        GateCheck("suite", result.suite_name == "final", result.suite_name, "final"),
        GateCheck(
            "openingPairs",
            statistics.opening_pairs == 200,
            statistics.opening_pairs,
            "exactly 200",
        ),
        GateCheck("games", statistics.games == 400, statistics.games, "exactly 400"),
        GateCheck("winRate", statistics.win_rate >= 0.60, statistics.win_rate, ">= 0.60"),
        GateCheck(
            "twoSided95Lower",
            statistics.win_rate_ci.lower >= 0.55,
            statistics.win_rate_ci.lower,
            ">= 0.55",
        ),
        GateCheck(
            "opponentEngine",
            result.opponent_identity.engine == CLASSIC_ENGINE,
            result.opponent_identity.engine or "unverified",
            CLASSIC_ENGINE,
        ),
    ]
    expected_budget_kind = "nodes" if track == "fixed" else "time"
    policy_has_workload = (
        policy is not None
        and policy.candidate_simulations is not None
        and policy.opponent_budget_kind == expected_budget_kind
        and policy.opponent_budget_limit is not None
        and (
            track != "wall-clock"
            or policy.opponent_budget_limit <= FINAL_WALL_CLOCK_LIMIT_MS
        )
    )
    checks.append(
        GateCheck(
            "trackWorkloadPolicy",
            policy_has_workload,
            policy_has_workload,
            (
                "candidate simulations plus exact classic nodes budget"
                if track == "fixed"
                else "candidate simulations plus classic time budget <= 850 ms"
            ),
        )
    )
    if policy_has_workload and policy is not None:
        candidate_runtime_matches = (
            result.candidate_runtime.kind == "native-mcts"
            and result.candidate_runtime.configured_simulations
            == policy.candidate_simulations
        )
        opponent_runtime_matches = (
            result.opponent_runtime.kind == "classic-hard"
            and result.opponent_runtime.budget_kind
            == policy.opponent_budget_kind
            and result.opponent_runtime.budget_limit
            == policy.opponent_budget_limit
            and result.opponent_runtime.max_depth == HARD_MAX_DEPTH
        )
        checks.extend(
            (
                GateCheck(
                    "candidateRuntime",
                    candidate_runtime_matches,
                    result.candidate_runtime.kind,
                    "trusted native-mcts with the policy simulation count",
                ),
                GateCheck(
                    "opponentRuntime",
                    opponent_runtime_matches,
                    result.opponent_runtime.kind,
                    (
                        f"trusted classic-hard {policy.opponent_budget_kind}:"
                        f"{policy.opponent_budget_limit}, depth={HARD_MAX_DEPTH}"
                    ),
                ),
                _role_simulation_check(
                    result,
                    "candidate",
                    policy.candidate_simulations,
                ),
                _role_simulation_check(result, "opponent", 0),
                _opponent_budget_check(
                    result,
                    policy.opponent_budget_kind,
                    policy.opponent_budget_limit,
                ),
            )
        )
        if track == "fixed":
            checks.append(
                _opponent_node_count_check(
                    result,
                    policy.opponent_budget_limit,
                )
            )
    if track == "wall-clock":
        observed_max = max(
            ply.wall_time_ms for game in result.games for ply in game.plies
        )
        checks.append(
            GateCheck(
                "maxMoveMs",
                observed_max <= FINAL_WALL_CLOCK_LIMIT_MS,
                observed_max,
                f"<= {FINAL_WALL_CLOCK_LIMIT_MS}",
            )
        )
    return _decision(
        f"final-hard-{track}",
        "two-sided-95-percent",
        statistics,
        tuple(checks),
        policy,
    )


__all__ = [
    "PAIRED_ARENA_SCHEMA_VERSION",
    "TERMINATION_REASONS",
    "GATE_BOOTSTRAP_SEED",
    "GATE_BOOTSTRAP_SAMPLES",
    "FINAL_WALL_CLOCK_LIMIT_MS",
    "PairedArenaError",
    "SelectorContractError",
    "IllegalSelectionError",
    "TerminalResultError",
    "JsonlSink",
    "SelectorIdentity",
    "SelectorRuntime",
    "BoundMoveSelector",
    "ArenaGatePolicy",
    "canonical_classic_backend_hash",
    "loaded_native_backend_hash",
    "inspect_selector_runtime",
    "PlyRecord",
    "OpeningReference",
    "GameRecord",
    "ResultBreakdown",
    "SelectionSummary",
    "PairedArenaResult",
    "GateCheck",
    "GateDecision",
    "run_paired_arena",
    "evaluate_screen_gate",
    "evaluate_confirmation_gate",
    "evaluate_noninferiority_gate",
    "evaluate_final_hard_gate",
]
