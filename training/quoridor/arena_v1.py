"""Frozen, paired arena execution for the ``quoridor-v1`` player.

This module is deliberately separate from :mod:`quoridor.paired_arena`, which
is the legacy ``game.State`` arena.  M5 consumes the immutable opening bytes
and config validated by :mod:`project_config`, replays every prefix through
the new engine/episode APIs, and runs deterministic PUCT games.  A game that
does not reach a goal or repetition result is retained as an incomplete
record and can never make a promotion eligible.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from . import engine
from .episode import Episode, advance_episode, start_episode
from .model_checkpoint import (
    IncompatibleCheckpointError,
    load_model_checkpoint,
)
from .openings import Opening, OpeningSuite, state_from_opening_payload, validate_opening_suite
from .project_config import PROJECT_ROOT, ProjectConfig
from .project_runtime import BudgetExpired, Deadline, RunSession
from .puct import InvalidEvaluationError, SearchConfig, SearchError, run_search
from .search_config import search_config_from_project
from .search_evaluator import NeuralEvaluator


class ArenaError(RuntimeError):
    """Base class for frozen arena input or execution errors."""


class ArenaInputError(ArenaError, ValueError):
    """Candidate, champion, opening, or protocol input is not compatible."""


class ArenaExecutionError(ArenaError):
    """A game or evaluator failed without a trustworthy result."""


@dataclass(frozen=True, slots=True)
class ArenaArtifact:
    """One model loaded from immutable bytes and its compatibility metadata."""

    path: Path
    sha256: str
    model: Any
    metadata_json: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "architecture": self.model.config.to_dict(),
            "metadata": json.loads(self.metadata_json),
        }


@dataclass(frozen=True, slots=True)
class ArenaGame:
    """Auditable result for one leg of one opening pair."""

    opening_id: str
    leg: int
    candidate_player: int
    status: str
    reason: str
    winner: int | None
    plies: int
    search_simulations: int
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.opening_id, str) or not self.opening_id:
            raise ValueError("opening_id must be non-empty")
        if type(self.leg) is not int or self.leg not in (0, 1):
            raise ValueError("leg must be 0 or 1")
        if type(self.candidate_player) is not int or self.candidate_player not in (0, 1):
            raise ValueError("candidate_player must be 0 or 1")
        if self.status not in {"completed", "truncated", "failed"}:
            raise ValueError("unsupported arena game status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        if self.status == "completed":
            if self.reason not in {"goal", "repetition"}:
                raise ValueError("completed games require a terminal reason")
            if self.reason == "goal" and (type(self.winner) is not int or self.winner not in (0, 1)):
                raise ValueError("goal games require a winner")
            if self.reason == "repetition" and self.winner is not None:
                raise ValueError("repetition games are draws")
        elif self.winner is not None:
            raise ValueError("incomplete games cannot claim a winner")
        if type(self.plies) is not int or self.plies < 0:
            raise ValueError("plies must be nonnegative")
        if type(self.search_simulations) is not int or self.search_simulations <= 0:
            raise ValueError("search_simulations must be positive")

    @property
    def candidate_outcome(self) -> str:
        if self.status != "completed":
            return "incomplete"
        if self.winner is None:
            return "draw"
        return "win" if self.winner == self.candidate_player else "loss"

    def to_dict(self) -> dict[str, object]:
        return {
            "openingId": self.opening_id,
            "leg": self.leg,
            "candidatePlayer": self.candidate_player,
            "status": self.status,
            "reason": self.reason,
            "winner": self.winner,
            "plies": self.plies,
            "searchSimulations": self.search_simulations,
            "candidateOutcome": self.candidate_outcome,
            "detail": self.detail,
        }


def _sha256(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except (OSError, ValueError) as error:
        raise ArenaInputError(f"cannot read model artifact {path}: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def _resolve_model_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArenaInputError(f"{label} does not exist: {value}") from error
    if not path.is_file():
        raise ArenaInputError(f"{label} must be a regular file: {value}")
    return path


def load_arena_artifacts(
    candidate: str | Path,
    champion: str | Path,
    *,
    candidate_sha256: str | None = None,
    champion_sha256: str | None = None,
    event_callback: Callable[..., None] | None = None,
) -> tuple[ArenaArtifact, ArenaArtifact]:
    """Load two model-only checkpoints and require the exact same architecture.

    Optional expected digests are useful when an evaluation manifest binds the
    model bytes.  They are checked before deserialization; no legacy checkpoint
    or training-resume artifact is accepted by this entry point.
    """

    paths = (_resolve_model_path(candidate, "candidate"), _resolve_model_path(champion, "champion"))
    expected = (candidate_sha256, champion_sha256)
    loaded: list[ArenaArtifact] = []
    for path, expected_hash, label in zip(paths, expected, ("candidate", "champion")):
        digest = _sha256(path)
        if event_callback is not None:
            event_callback("arena_model_input", role=label, path=str(path), sha256=digest, expected_sha256=expected_hash)
        if expected_hash is not None:
            if type(expected_hash) is not str or len(expected_hash) != 64 or any(
                char not in "0123456789abcdef" for char in expected_hash
            ):
                raise ArenaInputError(f"{label}_sha256 must be 64 lowercase SHA-256 digits")
            if digest != expected_hash:
                raise ArenaInputError(f"{label} model SHA-256 mismatch")
        try:
            checkpoint = load_model_checkpoint(path, expected_sha256=digest)
        except (OSError, ValueError, IncompatibleCheckpointError) as error:
            raise ArenaInputError(f"incompatible {label} model artifact: {error}") from error
        loaded.append(ArenaArtifact(path, checkpoint.artifact.sha256, checkpoint.model, json.dumps(checkpoint.metadata, sort_keys=True)))
        if event_callback is not None:
            event_callback("arena_model_validated", role=label, **loaded[-1].to_dict())
    if loaded[0].model.config.to_dict() != loaded[1].model.config.to_dict():
        raise ArenaInputError("candidate and champion architectures must match exactly")
    return loaded[0], loaded[1]


def _opening_engine_episode(opening: Opening, *, max_game_plies: int) -> Episode:
    """Replay a frozen legacy opening prefix under the M1 engine.

    The old opening JSON is only an immutable input container.  Every action is
    committed through ``advance_episode`` so the new full history, side to
    move, wall inventory, and terminal precedence are revalidated before play.
    """

    try:
        episode = start_episode(
            engine.initial_state(opening.starting_player),
            max_game_plies=max_game_plies,
        )
        for index, action in enumerate(opening.actions):
            if episode.finished:
                raise ArenaInputError(f"opening {opening.id} becomes terminal at ply {index}")
            episode = advance_episode(episode, action)
            if episode.outcome is not None:
                raise ArenaInputError(f"opening {opening.id} becomes terminal at ply {index}")
    except ArenaInputError:
        raise
    except (TypeError, ValueError) as error:
        raise ArenaInputError(f"opening {opening.id} is not reachable under quoridor-v1: {error}") from error

    old = opening.state
    current = episode.state
    if (
        current.pawns != old.pawns
        or current.walls_remaining != old.walls_remaining
        or current.horizontal_walls != old.horizontal_walls
        or current.vertical_walls != old.vertical_walls
        or current.to_play != old.current_player
    ):
        raise ArenaInputError(f"opening {opening.id} stored state disagrees with quoridor-v1 replay")
    if episode.plies != len(opening.actions):
        raise ArenaInputError(f"opening {opening.id} history length mismatch")
    return episode


def load_frozen_openings(config: ProjectConfig, *, deadline: Deadline | None = None) -> tuple[Opening, ...]:
    """Load exactly the configured, digest-checked opening suite and revalidate it."""

    if type(config) is not ProjectConfig or config.mode != "arena":
        raise TypeError("load_frozen_openings requires a validated arena ProjectConfig")
    document = config.document
    arena = document["arena"]
    source = PROJECT_ROOT / arena["opening_suite"]
    try:
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != arena["opening_suite_sha256"]:
            raise ArenaInputError("frozen opening suite SHA-256 mismatch at load")
        # Parse the same bytes whose digest was checked; never reread a mutable
        # path after validating its identity.
        payload = json.loads(raw)
        if set(payload) != {"schemaVersion", "suite", "pairingContract", "openings", "suiteHash"}:
            raise ArenaInputError("opening suite schema mismatch")
        if payload["suite"] != Path(arena["opening_suite"]).stem:
            raise ArenaInputError("opening suite name mismatch")
        entries = []
        for entry in payload["openings"]:
            if deadline is not None:
                deadline.check()
            if set(entry) != {"id", "hash", "startingPlayer", "actions", "state"}:
                raise ArenaInputError("opening entry schema mismatch")
            entries.append(Opening(
                entry["id"], entry["hash"], payload["suite"], entry["startingPlayer"],
                tuple(entry["actions"]), state_from_opening_payload(entry["state"]),
            ))
        suite = OpeningSuite(payload["suite"], tuple(entries), payload["suiteHash"], payload["schemaVersion"], payload["pairingContract"])
        validate_opening_suite(suite)
    except BudgetExpired:
        raise
    except Exception as error:
        raise ArenaInputError(f"cannot validate frozen opening suite: {error}") from error
    if len(suite.openings) != arena["pairs"]:
        raise ArenaInputError(
            f"frozen opening suite has {len(suite.openings)} openings; expected {arena['pairs']}"
        )
    max_plies = document["budget"]["max_game_plies"]
    for opening in suite.openings:
        if deadline is not None:
            deadline.check()
        _opening_engine_episode(opening, max_game_plies=max_plies)
    if deadline is not None:
        deadline.check()
    return suite.openings


def _search_config(config: ProjectConfig, seed: int) -> SearchConfig:
    """Derive the one immutable evaluation workload from the frozen YAML."""

    base = search_config_from_project(config)
    # Seeds identify pair/leg scheduling and are not a workload override.
    return replace(base, seed=seed, mode="evaluation", root_noise=False, temperature=0.0)


def play_arena_game(
    config: ProjectConfig,
    opening: Opening,
    *,
    leg: int,
    candidate_player: int,
    candidate_model: Any,
    champion_model: Any,
    deadline: Deadline,
    pair_index: int = 0,
    event_callback: Callable[..., None] | None = None,
) -> ArenaGame:
    """Play one deterministic opening leg with a shared cooperative deadline."""

    if type(leg) is not int or leg not in (0, 1) or type(candidate_player) is not int or candidate_player not in (0, 1):
        raise ValueError("leg and candidate_player must be 0 or 1")
    if type(config) is not ProjectConfig or config.mode != "arena":
        raise TypeError("play_arena_game requires a validated arena config")
    episode = _opening_engine_episode(
        opening, max_game_plies=config.document["budget"]["max_game_plies"]
    )
    simulations = config.document["search"]["simulations"]
    evaluators = {
        candidate_player: NeuralEvaluator(candidate_model),
        1 - candidate_player: NeuralEvaluator(champion_model),
    }
    try:
        while not episode.finished:
            deadline.check()
            player = episode.state.to_play
            search = run_search(
                episode,
                evaluators[player],
                _search_config(config, config.document["seed"] + pair_index * 2 + leg),
                deadline=deadline,
            )
            if event_callback is not None:
                event_callback(
                    "arena_search", opening_id=opening.id, leg=leg,
                    ply=episode.plies, to_play=player, action=search.action,
                    search_status=search.status,
                    completed_simulations=search.completed_simulations,
                    expected_simulations=simulations,
                    visits=list(search.visits),
                    history_sha256=hashlib.sha256(json.dumps(episode.history).encode()).hexdigest(),
                )
            if search.status != "completed" or search.action is None:
                reason = {
                    "budget_exhausted": "time_budget",
                    "simulation_truncated": "search_truncated",
                }.get(search.status, "search_incomplete")
                return ArenaGame(opening.id, leg, candidate_player, "truncated", reason, None, episode.plies, simulations)
            if search.completed_simulations != simulations or sum(search.visits) != simulations:
                return ArenaGame(opening.id, leg, candidate_player, "failed", "search_contract_error", None, episode.plies, simulations)
            action = search.action
            legal = engine.legal_actions(episode.state)
            if action not in legal:
                return ArenaGame(opening.id, leg, candidate_player, "failed", "illegal_action", None, episode.plies, simulations)
            if (len(search.visits) != engine.ACTION_SPACE_SIZE
                    or any(type(count) is not int or count < 0 for count in search.visits)
                    or any(count for index, count in enumerate(search.visits) if index not in legal)
                    or search.action != max(legal, key=lambda choice: (search.visits[choice], -choice))):
                return ArenaGame(opening.id, leg, candidate_player, "failed", "search_contract_error", None, episode.plies, simulations)
            episode = advance_episode(episode, action, deadline=deadline)
            if episode.outcome is not None and episode.outcome.status == "failed":
                return ArenaGame(opening.id, leg, candidate_player, "failed", episode.outcome.reason, None, episode.plies, simulations)
        assert episode.outcome is not None
        return ArenaGame(
            opening.id,
            leg,
            candidate_player,
            "completed" if episode.outcome.terminated else "truncated",
            episode.outcome.reason,
            episode.outcome.winner,
            episode.plies,
            simulations,
        )
    except BudgetExpired:
        return ArenaGame(opening.id, leg, candidate_player, "truncated", "time_budget", None, episode.plies, simulations)
    except KeyboardInterrupt:
        return ArenaGame(opening.id, leg, candidate_player, "truncated", "interrupted", None, episode.plies, simulations)
    except InvalidEvaluationError as error:
        return ArenaGame(opening.id, leg, candidate_player, "failed", "numerical_error", None, episode.plies, simulations, str(error))
    except SearchError as error:
        reason = "numerical_error" if isinstance(error.__cause__, (ValueError, FloatingPointError)) else "infrastructure_error"
        return ArenaGame(opening.id, leg, candidate_player, "failed", reason, None, episode.plies, simulations, str(error))
    except Exception as error:
        return ArenaGame(opening.id, leg, candidate_player, "failed", "infrastructure_error", None, episode.plies, simulations, str(error))


def _statistics(
    games: list[ArenaGame], *, config: ProjectConfig,
    deadline: Deadline | None = None, compute_bootstrap: bool = True,
) -> dict[str, object]:
    """Convert game records through the M5 scoring contract when available."""

    from .arena_statistics import ArenaGameResult, summarize_arena
    results = []
    for game in games:
        if game.status == "completed":
            status = "completed"
            outcome = game.candidate_outcome
        elif game.reason == "illegal_action":
            status, outcome = "illegal", None
        elif game.reason == "numerical_error":
            status, outcome = "numerical", None
        else:
            status, outcome = game.status, None
        results.append(ArenaGameResult(
            opening_id=game.opening_id,
            leg=game.leg,
            candidate_player=game.candidate_player,
            outcome=outcome,
            status=status,
            reason=game.reason,
        ))
    stats = summarize_arena(
        results,
        expected_pairs=config.document["arena"]["pairs"],
        bootstrap_seed=config.document["seed"],
        bootstrap_samples=config.document["arena"]["bootstrap_samples"],
        bootstrap_confidence=config.document["arena"]["promotion_confidence"],
        promotion_score=config.document["arena"]["promotion_score"],
        check=deadline.check if deadline is not None else None,
        compute_bootstrap=compute_bootstrap,
    )
    result = stats.to_dict()
    result["truncationsByReason"] = dict(sorted(Counter(game.reason for game in games if game.status == "truncated").items()))
    result["failuresByReason"] = dict(sorted(Counter(game.reason for game in games if game.status == "failed").items()))
    return result


def _invalidate_result(summary: dict[str, object]) -> None:
    summary["decisionEligible"] = False
    summary["promotionEligible"] = False
    gate = summary.get("promotionGate")
    if isinstance(gate, dict):
        gate["decisionEligible"] = False
        gate["passed"] = False


def _save_result(path: Path, summary: dict[str, object]) -> None:
    """Publish one whole result; an observer never sees a half-written gate."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _log_finalization_invalidation(run: RunSession, summary: dict[str, object]) -> None:
    """Append the correction after the generic run's terminal budget event.

    The generic session is already closed, so its normal active-only ``log``
    API cannot be used.  This explicit arena correction follows its final
    sequence number and never changes the run's recorded terminal status.
    """
    event = {
        "schema_version": 1, "sequence": run._sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": run.deadline.elapsed_seconds,
        "event": "arena_finalization_invalidated",
        "status": "incomplete", "provisional": False,
        "decisionEligible": False, "promotionEligible": False,
        "promotionGate": summary["promotionGate"],
        "stop_reason": summary["stop_reason"],
    }
    with (run.path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    run._sequence += 1


@contextmanager
def _arena_run_session(config: ProjectConfig, *, run_root: Path | None):
    """Bind published eligibility to the final generic run status.

    The ``finally`` guard also handles an interruption or I/O exception after a
    result was published.  Consumers must require both ``summary.json`` and
    ``arena-result.json`` to be completed; a provisional event alone is never
    promotion evidence.
    """
    run = RunSession(config, run_root=run_root)
    original_error: BaseException | None = None
    try:
        with run:
            yield run
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            status_path = run.path / "summary.json"
            result_path = run.path / "arena-result.json"
            if status_path.exists() and result_path.exists():
                runtime_summary = json.loads(status_path.read_text(encoding="utf-8"))
                run_status = runtime_summary["status"]
                if run_status == "completed" and run.deadline.expired:
                    # The generic context checks before writing its final
                    # audit files. Account for an overrun inside those writes.
                    run_status = "budget_exhausted"
                    runtime_summary.update(status=run_status, elapsed_seconds=run.deadline.elapsed_seconds)
                    _save_result(status_path, runtime_summary)
                if run_status != "completed":
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                    summary.update(status="incomplete", stop_reason=run_status)
                    _invalidate_result(summary)
                    _save_result(result_path, summary)
                    _log_finalization_invalidation(run, summary)
                    if original_error is None:
                        message = f"arena {run_status} at finalization; result retained at {run.path.resolve()}"
                        if run_status == "budget_exhausted":
                            raise BudgetExpired(message)
                        raise ArenaExecutionError(message)
        except BaseException:
            # Cleanup cannot replace the interruption or infrastructure error
            # that caused the run to fail.  Even when the filesystem cannot
            # accept a correction, its non-completed run summary blocks use.
            if original_error is None:
                raise


def run_arena(
    config: ProjectConfig,
    candidate: str | Path,
    champion: str | Path,
    *,
    candidate_sha256: str | None = None,
    champion_sha256: str | None = None,
    run_root: Path | None = None,
) -> dict[str, object]:
    """Run all configured paired games and retain auditable JSONL results."""

    if type(config) is not ProjectConfig or config.mode != "arena":
        raise TypeError("run_arena requires a validated arena ProjectConfig")
    # The config loader has already checked raw YAML/opening digests.  Keep the
    # explicit call here so direct callers cannot accidentally bypass that gate.
    from .project_config import load_project_config

    frozen = load_project_config(config.source_path, expected_mode="arena")
    if frozen.yaml_bytes != config.yaml_bytes or frozen.canonical_json != config.canonical_json:
        raise ArenaInputError("arena config snapshot changed after validation")
    games: list[ArenaGame] = []
    summary: dict[str, object]
    interrupted = False
    exhausted = False
    with _arena_run_session(config, run_root=run_root) as run:
        run.log(
            "arena_inputs", candidate_path=str(candidate), champion_path=str(champion),
            candidate_expected_sha256=candidate_sha256,
            champion_expected_sha256=champion_sha256,
        )
        run.deadline.check()
        candidate_artifact, champion_artifact = load_arena_artifacts(
            candidate,
            champion,
            candidate_sha256=candidate_sha256,
            champion_sha256=champion_sha256,
            event_callback=run.log,
        )
        run.deadline.check()
        openings = load_frozen_openings(config, deadline=run.deadline)
        run.log(
            "m5_effective_config",
            protocol_id=config.document["arena"]["protocol_id"],
            search=config.document["search"],
            pairs=config.document["arena"]["pairs"],
            bootstrap_samples=config.document["arena"]["bootstrap_samples"],
            candidate=candidate_artifact.to_dict(),
            champion=champion_artifact.to_dict(),
            opening_suite_sha256=config.document["arena"]["opening_suite_sha256"],
            evaluation_scope="development-gate",
            prior_fixture_exposure="legacy dev opening suite, previously used in Phase 4 evaluations",
            final_acceptance=False,
        )
        try:
            for pair_index, opening in enumerate(openings):
                for leg, candidate_player in ((0, opening.state.current_player), (1, 1 - opening.state.current_player)):
                    game = play_arena_game(
                        config,
                        opening,
                        leg=leg,
                        candidate_player=candidate_player,
                        candidate_model=candidate_artifact.model,
                        champion_model=champion_artifact.model,
                        deadline=run.deadline,
                        pair_index=pair_index,
                        event_callback=run.log,
                    )
                    games.append(game)
                    run.log("arena_game", **game.to_dict())
                    if game.status != "completed":
                        # Continue only while the shared budget permits; an
                        # incomplete pair remains visible and blocks promotion.
                        run.deadline.check()
                        if game.reason == "interrupted":
                            raise KeyboardInterrupt()
        except BudgetExpired:
            exhausted = True
            run.log("arena_budget_exhausted", games=len(games))
        except KeyboardInterrupt:
            interrupted = True
            run.log("arena_interrupted", games=len(games))
        try:
            if interrupted or exhausted:
                summary = _statistics(games, config=config, compute_bootstrap=False)
            else:
                run.deadline.check()
                summary = _statistics(games, config=config, deadline=run.deadline)
                run.deadline.check()
        except BudgetExpired:
            exhausted = True
            summary = _statistics(games, config=config, compute_bootstrap=False)
        except KeyboardInterrupt:
            interrupted = True
            summary = _statistics(games, config=config, compute_bootstrap=False)
        exhausted = exhausted or run.deadline.expired
        summary = {
            **summary,
            "status": "completed" if not exhausted and not interrupted and len(games) == len(openings) * 2 and all(game.status == "completed" for game in games) else "incomplete",
            "games": len(games),
            "scheduledGames": len(openings) * 2,
            "candidate": candidate_artifact.to_dict(),
            "champion": champion_artifact.to_dict(),
            "stop_reason": "interrupted" if interrupted else ("time_budget" if exhausted else None),
        }
        if summary["status"] != "completed":
            _invalidate_result(summary)
        run.log("m5_arena_summary", provisional=True, **summary)
        _save_result(run.path / "arena-result.json", summary)
        if summary["status"] != "completed":
            message = f"arena incomplete; partial result retained at {run.path.resolve()}"
            if exhausted or run.deadline.expired:
                raise BudgetExpired(message)
            if interrupted or any(game.reason == "interrupted" for game in games):
                raise KeyboardInterrupt(message)
            raise ArenaExecutionError(message)
    summary["run_dir"] = str(run.path.resolve())
    return summary


__all__ = [
    "ArenaError", "ArenaInputError", "ArenaExecutionError", "ArenaArtifact",
    "ArenaGame", "load_arena_artifacts", "load_frozen_openings",
    "play_arena_game", "run_arena",
]
