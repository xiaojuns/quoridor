"""Deterministic, resumable classic-AI teacher-data generation for Phase 4D.

Teacher trajectories are stored directly in :class:`ReplayStore`.  This keeps
the warm-start input on the same raw-state path as self-play: every sample can
be reconstructed from an exact initial state and a complete action sequence.
The train/holdout split happens before a game is generated, so no trajectory is
ever shared across the two stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .classic_bridge import (
    CLASSIC_ENGINE,
    ClassicAIBridge,
    ClassicSearchBudget,
)
from .challenge_replay import arena_position_orbits
from .codec import ACTION_SPACE_SIZE, mirror_state, position_hash
from .game import (
    MAX_HALF_MOVES,
    State,
    apply_action_index,
    create_initial_state,
    get_winner,
    is_terminal,
    legal_action_indices,
)
from .openings import load_opening_suite, state_to_opening_payload
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_gates import find_valid_memory_smoke
from .replay_store import ReplayGame, ReplayPly, ReplayStore, reconstruct_game
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    ResourceBudget,
    RunLock,
    atomic_replace_bytes,
    build_run_manifest,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    save_run_manifest,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
    validate_sha256,
)
from .calibration import GPU_LOCK_NAME


TEACHER_DATASET_SCHEMA_VERSION = 1
TEACHER_POLICY_SCALE = 1_000_000
TEACHER_POLICY_FLOOR_UNITS = 100
TEACHER_MODEL_ID = "classic-negamax-hard-v1"
DEFAULT_OUTPUT_ROOT = Path("training/runs/phase4/teacher-data")
DEFAULT_ARTIFACT_ROOT = Path("training/runs/phase4")
ARTIFACT_WRITE_RESERVE_BYTES = 64 * 1024**2
HOLDOUT_DENOMINATOR = 10

_MANIFEST_NAME = "teacher-manifest.json"
_RUN_MANIFEST_NAME = "run-manifest.json"
_SOURCE_PATHS = (
    "training/quoridor/teacher_data.py",
    "training/quoridor/challenge_replay.py",
    "training/quoridor/classic_bridge.py",
    "training/quoridor/replay_store.py",
    "training/quoridor/openings.py",
    "training/quoridor/codec.py",
    "training/quoridor/game.py",
    "training/quoridor/run_manifest.py",
    "training/quoridor/phase4_budget.py",
    "training/quoridor/phase4_gates.py",
    "training/quoridor/training_memory_smoke.py",
    "training/quoridor/calibration.py",
    "scripts/generate-phase4-teacher-data.py",
    "scripts/generate-phase4f-challenge-data.sh",
    "scripts/classic-ai-bridge.mjs",
    "src/ai.js",
    "src/game.js",
    "src/alphazero/state-codec.js",
    "src/alphazero/action-codec.js",
)


class TeacherDataError(RuntimeError):
    """Base class for teacher dataset failures."""


class TeacherDataContractError(TeacherDataError):
    """Raised when persisted teacher data violates its immutable contract."""


class TeacherDataResourceError(TeacherDataError):
    """Raised when a Phase 4 artifact or free-disk stop line is reached."""


class DuplicateTeacherTrajectory(TeacherDataError):
    """Internal signal used to discard a trajectory containing a duplicate state."""


@dataclass(frozen=True, slots=True)
class TeacherDataConfig:
    """Semantic configuration hashed into the immutable run manifest."""

    target_positions: int = 50_000
    seed: int = 20260722
    node_budget: int = 512
    max_depth: int = 5
    exploration_probability: float = 0.15
    opening_depths: tuple[int, ...] = (8, 24, 48)
    request_timeout_seconds: float = 30.0
    maximum_attempts: int = 250_000
    policy_scale: int = TEACHER_POLICY_SCALE
    policy_floor_units: int = TEACHER_POLICY_FLOOR_UNITS
    exclusion_set_sha256: str | None = None
    exclusion_orbit_count: int = 0
    recovery_preparation_report_sha256: str | None = None
    recovery_challenge_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.target_positions) is not int or self.target_positions < 10:
            raise ValueError("target_positions must be an integer >= 10")
        if self.target_positions % HOLDOUT_DENOMINATOR != 0:
            raise ValueError("target_positions must be divisible by 10")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.node_budget) is not int or self.node_budget <= 0:
            raise ValueError("node_budget must be a positive integer")
        if type(self.max_depth) is not int or not 1 <= self.max_depth <= 12:
            raise ValueError("max_depth must be an integer in [1, 12]")
        if (
            isinstance(self.exploration_probability, bool)
            or not isinstance(self.exploration_probability, (int, float))
            or not math.isfinite(float(self.exploration_probability))
            or not 0.0 <= float(self.exploration_probability) < 1.0
        ):
            raise ValueError("exploration_probability must be finite and in [0, 1)")
        object.__setattr__(
            self, "exploration_probability", float(self.exploration_probability)
        )
        depths = tuple(self.opening_depths)
        if (
            not depths
            or len(depths) != len(set(depths))
            or any(type(depth) is not int or not 0 <= depth < MAX_HALF_MOVES for depth in depths)
        ):
            raise ValueError(
                "opening_depths must contain unique integers in [0, MAX_HALF_MOVES)"
            )
        object.__setattr__(self, "opening_depths", depths)
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or float(self.request_timeout_seconds) <= 0
        ):
            raise ValueError("request_timeout_seconds must be finite and positive")
        object.__setattr__(
            self, "request_timeout_seconds", float(self.request_timeout_seconds)
        )
        if type(self.maximum_attempts) is not int or self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be a positive integer")
        if type(self.policy_scale) is not int or self.policy_scale <= 0:
            raise ValueError("policy_scale must be a positive integer")
        if type(self.policy_floor_units) is not int or self.policy_floor_units <= 0:
            raise ValueError("policy_floor_units must be a positive integer")
        if self.policy_floor_units * ACTION_SPACE_SIZE >= self.policy_scale:
            raise ValueError("policy floor leaves no mass for the teacher action")
        if (self.exclusion_set_sha256 is None) != (self.exclusion_orbit_count == 0):
            raise ValueError(
                "teacher exclusion hash and positive orbit count must be declared together"
            )
        if self.exclusion_set_sha256 is not None:
            validate_sha256(self.exclusion_set_sha256, "teacher exclusion set sha256")
        if type(self.exclusion_orbit_count) is not int or self.exclusion_orbit_count < 0:
            raise ValueError("exclusion_orbit_count must be a non-negative integer")
        if (self.recovery_preparation_report_sha256 is None) != (
            self.recovery_challenge_contract_sha256 is None
        ):
            raise ValueError(
                "recovery preparation report and challenge contract hashes must be declared together"
            )
        if self.recovery_preparation_report_sha256 is not None:
            validate_sha256(
                self.recovery_preparation_report_sha256,
                "recovery preparation report sha256",
            )
            validate_sha256(
                self.recovery_challenge_contract_sha256,
                "recovery challenge contract sha256",
            )

    @property
    def holdout_target(self) -> int:
        return self.target_positions // HOLDOUT_DENOMINATOR

    @property
    def train_target(self) -> int:
        return self.target_positions - self.holdout_target

    def to_payload(self) -> dict[str, object]:
        payload = {
            "targetPositions": self.target_positions,
            "splitTargets": {
                "train": self.train_target,
                "holdout": self.holdout_target,
            },
            "holdoutProtocol": "whole-game-sha256-modulo-10-v1",
            "seed": self.seed,
            "classicSearch": {
                "engine": CLASSIC_ENGINE,
                "difficulty": "hard",
                "budgetKind": "nodes",
                "nodeBudget": self.node_budget,
                "maxDepth": self.max_depth,
            },
            "explorationProbability": self.exploration_probability,
            "openingDepths": list(self.opening_depths),
            "requestTimeoutSeconds": self.request_timeout_seconds,
            "maximumAttempts": self.maximum_attempts,
            "policy": {
                "scale": self.policy_scale,
                "floorUnits": self.policy_floor_units,
                "probabilityFloor": self.policy_floor_units / self.policy_scale,
                "legalSupport": "complete",
            },
            "deduplication": "sha256-full-state-history-horizontal-orbit-v1",
        }
        if self.exclusion_set_sha256 is not None:
            payload["evaluationExclusions"] = {
                "algorithm": "sha256-sorted-horizontal-position-orbits-v1",
                "sha256": self.exclusion_set_sha256,
                "orbits": self.exclusion_orbit_count,
            }
        if self.recovery_preparation_report_sha256 is not None:
            payload["recoveryAuthorization"] = {
                "kind": "phase4f2-cpu-preparation-gate-v1",
                "reportSha256": self.recovery_preparation_report_sha256,
                "challengeContractSha256": self.recovery_challenge_contract_sha256,
            }
        return payload


def exclusion_set_sha256(orbits: Sequence[str]) -> str:
    normalized = sorted(set(orbits))
    if len(normalized) != len(orbits):
        raise ValueError("teacher exclusion orbits must be unique")
    for value in normalized:
        validate_sha256(value, "teacher exclusion orbit")
    return sha256_bytes(canonical_json_bytes(normalized))


@dataclass(frozen=True, slots=True)
class TeacherDatasetSummary:
    status: str
    train_positions: int
    holdout_positions: int
    train_games: int
    holdout_games: int
    next_game_index: int
    duplicate_trajectories: int
    dataset_fingerprint: str | None
    manifest_path: Path
    run_manifest_path: Path

    @property
    def total_positions(self) -> int:
        return self.train_positions + self.holdout_positions

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "positions": {
                "total": self.total_positions,
                "train": self.train_positions,
                "holdout": self.holdout_positions,
            },
            "games": {
                "train": self.train_games,
                "holdout": self.holdout_games,
            },
            "nextGameIndex": self.next_game_index,
            "duplicateTrajectories": self.duplicate_trajectories,
            "datasetFingerprint": self.dataset_fingerprint,
            "manifestPath": str(self.manifest_path),
            "runManifestPath": str(self.run_manifest_path),
        }


@dataclass(slots=True)
class _Progress:
    next_game_index: int
    duplicate_trajectories: int
    seen_orbit_digests: set[str]
    histograms: dict[str, dict[str, int]]


ResourceChecker = Callable[[Path], Mapping[str, int]]
BridgeFactory = Callable[[], ClassicAIBridge]


def _full_state_digest(state: State) -> str:
    return sha256_bytes(canonical_json_bytes(state_to_opening_payload(state)))


def canonical_state_orbit_digest(state: State) -> str:
    """Identify a raw state and its horizontal mirror as one augmentation orbit."""

    raw = canonical_json_bytes(state_to_opening_payload(state))
    mirrored = canonical_json_bytes(state_to_opening_payload(mirror_state(state)))
    return sha256_bytes(min(raw, mirrored))


def evaluation_position_orbit_digest(state: State) -> str:
    """Identify a playable position and its horizontal mirror for leakage gates."""

    raw = position_hash(state).encode("ascii")
    mirrored = position_hash(mirror_state(state)).encode("ascii")
    return sha256_bytes(min(raw, mirrored))


def teacher_policy_counts(
    legal_actions: Sequence[int],
    teacher_action: int,
    *,
    scale: int = TEACHER_POLICY_SCALE,
    floor_units: int = TEACHER_POLICY_FLOOR_UNITS,
) -> tuple[int, ...]:
    """Return fixed-point policy weights with positive mass on every legal action."""

    legal = tuple(legal_actions)
    if not legal or len(legal) != len(set(legal)):
        raise ValueError("legal_actions must be non-empty and unique")
    if any(type(action) is not int or not 0 <= action < ACTION_SPACE_SIZE for action in legal):
        raise ValueError("legal action is outside the action space")
    if teacher_action not in legal:
        raise ValueError("teacher_action must be legal")
    if type(scale) is not int or type(floor_units) is not int:
        raise TypeError("scale and floor_units must be integers")
    if scale <= 0 or floor_units <= 0 or floor_units * len(legal) >= scale:
        raise ValueError("policy scale/floor cannot represent complete legal support")
    counts = [0] * ACTION_SPACE_SIZE
    for action in legal:
        counts[action] = floor_units
    counts[teacher_action] += scale - floor_units * len(legal)
    return tuple(counts)


def _derived_seed(seed: int, namespace: str, game_index: int, attempt: int = 0) -> int:
    payload = f"phase4-teacher-v1:{seed}:{namespace}:{game_index}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def split_for_game(seed: int, game_index: int) -> str:
    """Assign a complete game to a stable 90/10 train/holdout partition."""

    bucket = _derived_seed(seed, "split", game_index) % HOLDOUT_DENOMINATOR
    return "holdout" if bucket == 0 else "train"


def _opening_spec(config: TeacherDataConfig, game_index: int) -> tuple[int, int]:
    selector = _derived_seed(config.seed, "stratum", game_index)
    depth = config.opening_depths[selector % len(config.opening_depths)]
    starting_player = (selector // len(config.opening_depths)) % 2
    return depth, starting_player


def _random_opening(
    config: TeacherDataConfig,
    game_index: int,
    *,
    maximum_retries: int = 64,
) -> tuple[State, tuple[int, ...], int]:
    depth, starting_player = _opening_spec(config, game_index)
    for attempt in range(maximum_retries):
        rng = random.Random(_derived_seed(config.seed, "opening", game_index, attempt))
        state = create_initial_state(starting_player)
        actions: list[int] = []
        viable = True
        for _ in range(depth):
            candidates = list(legal_action_indices(state))
            rng.shuffle(candidates)
            next_state: State | None = None
            selected: int | None = None
            for action in candidates:
                candidate = apply_action_index(state, action)
                if not is_terminal(candidate):
                    selected = action
                    next_state = candidate
                    break
            if selected is None or next_state is None:
                viable = False
                break
            actions.append(selected)
            state = next_state
        if viable:
            return state, tuple(actions), starting_player
    raise TeacherDataError(
        f"could not construct a non-terminal depth-{depth} opening for game {game_index}"
    )


def _wall_bucket(total_remaining: int) -> str:
    if total_remaining <= 4:
        return "00-04"
    if total_remaining <= 9:
        return "05-09"
    if total_remaining <= 14:
        return "10-14"
    return "15-20"


def _choose_played_action(
    legal: Sequence[int],
    teacher_action: int,
    rng: random.Random,
    exploration_probability: float,
) -> tuple[int, bool]:
    alternatives = tuple(action for action in legal if action != teacher_action)
    if alternatives and rng.random() < exploration_probability:
        return alternatives[rng.randrange(len(alternatives))], True
    return teacher_action, False


def _teacher_value(score: float) -> float:
    return math.tanh(score / 600.0)


def _build_teacher_game(
    config: TeacherDataConfig,
    game_index: int,
    split: str,
    bridge: ClassicAIBridge,
    seen_orbit_digests: set[str],
    excluded_position_orbits: frozenset[str],
) -> tuple[ReplayGame, tuple[str, ...]]:
    initial_state, opening_actions, starting_player = _random_opening(
        config, game_index
    )
    opening_depth = len(opening_actions)
    rng = random.Random(_derived_seed(config.seed, "trajectory", game_index))
    state = initial_state
    plies: list[ReplayPly] = []
    orbit_digests: list[str] = []
    local_orbits: set[str] = set()
    budget = ClassicSearchBudget.nodes(config.node_budget)

    while not is_terminal(state):
        if evaluation_position_orbit_digest(state) in excluded_position_orbits:
            raise DuplicateTeacherTrajectory(
                f"game {game_index} intersects an excluded evaluation position"
            )
        state_digest = _full_state_digest(state)
        orbit_digest = canonical_state_orbit_digest(state)
        if orbit_digest in seen_orbit_digests or orbit_digest in local_orbits:
            raise DuplicateTeacherTrajectory(
                f"game {game_index} intersects an existing augmentation orbit"
            )
        local_orbits.add(orbit_digest)
        orbit_digests.append(orbit_digest)
        legal = legal_action_indices(state)
        result = bridge.choose(
            state,
            budget=budget,
            max_depth=config.max_depth,
            timeout=config.request_timeout_seconds,
        )
        counts = teacher_policy_counts(
            legal,
            result.action_index,
            scale=config.policy_scale,
            floor_units=config.policy_floor_units,
        )
        played_action, explored = _choose_played_action(
            legal,
            result.action_index,
            rng,
            config.exploration_probability,
        )
        total_walls = sum(state.walls_remaining)
        plies.append(
            ReplayPly(
                action_index=played_action,
                visit_counts=counts,
                root_value=_teacher_value(result.score),
                model_generation=0,
                model_id=TEACHER_MODEL_ID,
                metadata={
                    "teacherActionIndex": result.action_index,
                    "teacherScore": result.score,
                    "searchDepth": result.depth,
                    "searchNodes": result.nodes,
                    "budgetKind": budget.kind,
                    "budgetLimit": budget.limit,
                    "maxDepth": config.max_depth,
                    "legalActionCount": len(legal),
                    "policyScale": config.policy_scale,
                    "policyFloorUnits": config.policy_floor_units,
                    "teacherStateSha256": state_digest,
                    "teacherOrbitSha256": orbit_digest,
                    "positionHash": position_hash(state),
                    "sideToMove": state.current_player,
                    "wallsRemainingTotal": total_walls,
                    "wallCountBucket": _wall_bucket(total_walls),
                    "openingDepth": opening_depth,
                    "playedByExploration": explored,
                },
            )
        )
        state = apply_action_index(state, played_action)

    winner = get_winner(state)
    if winner is None:
        raise TeacherDataError("classic teacher trajectory did not terminate")
    game_id_digest = sha256_bytes(
        canonical_json_bytes(
            {
                "seed": config.seed,
                "gameIndex": game_index,
                "split": split,
                "opening": opening_actions,
                "actions": [ply.action_index for ply in plies],
            }
        )
    )
    game = ReplayGame(
        game_id=f"teacher-{game_index:08d}-{game_id_digest[:16]}",
        initial_state=initial_state,
        plies=tuple(plies),
        metadata={
            "source": "phase4d-classic-hard-teacher",
            "datasetSchemaVersion": TEACHER_DATASET_SCHEMA_VERSION,
            "datasetSplit": split,
            "gameIndex": game_index,
            "gameSeed": _derived_seed(config.seed, "trajectory", game_index),
            "startingPlayer": starting_player,
            "openingDepth": opening_depth,
            "openingActions": list(opening_actions),
            "winner": winner,
            "terminationReason": state.termination_reason,
            "teacherEngine": CLASSIC_ENGINE,
            "teacherModelId": TEACHER_MODEL_ID,
            "nodeBudget": config.node_budget,
            "maxDepth": config.max_depth,
            "policyScale": config.policy_scale,
            "policyFloorUnits": config.policy_floor_units,
            "explorationProbability": config.exploration_probability,
        },
    )
    return game, tuple(orbit_digests)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise TeacherDataResourceError(
                f"artifact tree must not contain symlinks: {item}"
            )
        if item.is_file():
            total += item.stat().st_size
    return total


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise TeacherDataResourceError(
                f"cannot find an existing filesystem ancestor for {path}"
            )
        candidate = parent
    return candidate


def assert_teacher_resources(artifact_root: Path) -> Mapping[str, int]:
    """Enforce the global Phase 4 artifact cap and free-disk stop line."""

    root = artifact_root.resolve()
    artifact_bytes = _directory_size(root)
    if artifact_bytes > DEFAULT_ARTIFACT_LIMIT_BYTES - ARTIFACT_WRITE_RESERVE_BYTES:
        raise TeacherDataResourceError(
            "Phase 4 artifacts are within the 64 MiB reserve below the 8 GiB cap"
        )
    free_bytes = int(shutil.disk_usage(_existing_ancestor(root)).free)
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES + ARTIFACT_WRITE_RESERVE_BYTES:
        raise TeacherDataResourceError(
            "free disk is within the 64 MiB reserve above the 20 GiB stop line"
        )
    return {"artifactBytes": artifact_bytes, "freeDiskBytes": free_bytes}


def _histogram_template() -> dict[str, dict[str, int]]:
    return {
        "startingPlayer": {},
        "winner": {},
        "sideToMove": {},
        "wallCountBucket": {},
        "openingDepth": {},
        "terminationReason": {},
    }


def _histogram_increment(
    histograms: dict[str, dict[str, int]], category: str, value: object, amount: int
) -> None:
    key = str(value)
    bucket = histograms[category]
    bucket[key] = bucket.get(key, 0) + amount


def _verify_teacher_game(
    game: ReplayGame,
    *,
    split: str,
    config: TeacherDataConfig,
) -> tuple[set[str], dict[str, dict[str, int]], int]:
    metadata = game.metadata
    if metadata.get("datasetSplit") != split:
        raise TeacherDataContractError("teacher game is stored in the wrong split")
    game_index = metadata.get("gameIndex")
    if type(game_index) is not int or game_index < 0:
        raise TeacherDataContractError("teacher game has an invalid gameIndex")
    if split_for_game(config.seed, game_index) != split:
        raise TeacherDataContractError("teacher game violates deterministic split")
    if (
        metadata.get("source") != "phase4d-classic-hard-teacher"
        or metadata.get("datasetSchemaVersion") != TEACHER_DATASET_SCHEMA_VERSION
        or metadata.get("teacherEngine") != CLASSIC_ENGINE
        or metadata.get("teacherModelId") != TEACHER_MODEL_ID
        or metadata.get("nodeBudget") != config.node_budget
        or metadata.get("maxDepth") != config.max_depth
        or metadata.get("policyScale") != config.policy_scale
        or metadata.get("policyFloorUnits") != config.policy_floor_units
        or metadata.get("explorationProbability")
        != config.exploration_probability
    ):
        raise TeacherDataContractError("teacher game source/config metadata mismatch")
    starting_player = metadata.get("startingPlayer")
    opening_actions = metadata.get("openingActions")
    if type(starting_player) is not int or starting_player not in (0, 1):
        raise TeacherDataContractError("teacher game has an invalid startingPlayer")
    if not isinstance(opening_actions, list) or any(
        type(action) is not int for action in opening_actions
    ):
        raise TeacherDataContractError("teacher game openingActions are invalid")
    opening_state = create_initial_state(starting_player)
    try:
        for action in opening_actions:
            opening_state = apply_action_index(opening_state, action)
    except ValueError as error:
        raise TeacherDataContractError("teacher opening cannot be replayed") from error
    if state_to_opening_payload(opening_state) != state_to_opening_payload(
        game.initial_state
    ):
        raise TeacherDataContractError("teacher initial state/opening history mismatch")

    states = reconstruct_game(game)
    final_state = states[-1]
    winner = get_winner(final_state)
    if winner != metadata.get("winner"):
        raise TeacherDataContractError("teacher winner metadata mismatch")
    if final_state.termination_reason != metadata.get("terminationReason"):
        raise TeacherDataContractError("teacher termination metadata mismatch")

    digests: set[str] = set()
    histograms = _histogram_template()
    for ply_index, (state, ply) in enumerate(zip(states, game.plies)):
        legal = frozenset(legal_action_indices(state))
        support = {
            action for action, count in enumerate(ply.visit_counts) if count > 0
        }
        if support != legal:
            raise TeacherDataContractError(
                f"teacher policy omits or adds legal support at ply {ply_index}"
            )
        if sum(ply.visit_counts) != config.policy_scale:
            raise TeacherDataContractError("teacher fixed-point policy scale mismatch")
        if any(
            ply.visit_counts[action] < config.policy_floor_units for action in legal
        ):
            raise TeacherDataContractError("teacher legal-action floor is missing")
        teacher_action = ply.metadata.get("teacherActionIndex")
        if type(teacher_action) is not int or teacher_action not in legal:
            raise TeacherDataContractError("teacher selected action is invalid")
        expected_counts = teacher_policy_counts(
            tuple(sorted(legal)),
            teacher_action,
            scale=config.policy_scale,
            floor_units=config.policy_floor_units,
        )
        if ply.visit_counts != expected_counts:
            raise TeacherDataContractError("teacher policy distribution mismatch")
        if ply.model_id != TEACHER_MODEL_ID or ply.model_generation != 0:
            raise TeacherDataContractError("teacher model identity mismatch")
        if (
            ply.metadata.get("budgetKind") != "nodes"
            or ply.metadata.get("budgetLimit") != config.node_budget
            or ply.metadata.get("maxDepth") != config.max_depth
            or ply.metadata.get("legalActionCount") != len(legal)
            or ply.metadata.get("policyScale") != config.policy_scale
            or ply.metadata.get("policyFloorUnits") != config.policy_floor_units
        ):
            raise TeacherDataContractError("teacher search budget metadata mismatch")
        teacher_score = ply.metadata.get("teacherScore")
        search_depth = ply.metadata.get("searchDepth")
        search_nodes = ply.metadata.get("searchNodes")
        if (
            isinstance(teacher_score, bool)
            or not isinstance(teacher_score, (int, float))
            or not math.isfinite(float(teacher_score))
            or type(search_depth) is not int
            or not 0 <= search_depth <= config.max_depth
            or type(search_nodes) is not int
            or not 0 <= search_nodes <= config.node_budget
            or not math.isclose(
                ply.root_value,
                _teacher_value(float(teacher_score)),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise TeacherDataContractError("teacher search result metadata mismatch")
        raw_digest = _full_state_digest(state)
        orbit_digest = canonical_state_orbit_digest(state)
        if ply.metadata.get("teacherStateSha256") != raw_digest:
            raise TeacherDataContractError("teacher state digest mismatch")
        if ply.metadata.get("teacherOrbitSha256") != orbit_digest:
            raise TeacherDataContractError("teacher augmentation-orbit digest mismatch")
        if ply.metadata.get("positionHash") != position_hash(state):
            raise TeacherDataContractError("teacher position hash mismatch")
        if orbit_digest in digests:
            raise TeacherDataContractError(
                "teacher game contains a duplicate augmentation orbit"
            )
        digests.add(orbit_digest)
        total_walls = sum(state.walls_remaining)
        if (
            ply.metadata.get("sideToMove") != state.current_player
            or ply.metadata.get("wallsRemainingTotal") != total_walls
            or ply.metadata.get("wallCountBucket") != _wall_bucket(total_walls)
        ):
            raise TeacherDataContractError("teacher stratum metadata mismatch")

        _histogram_increment(histograms, "startingPlayer", starting_player, 1)
        _histogram_increment(histograms, "winner", winner, 1)
        _histogram_increment(histograms, "sideToMove", state.current_player, 1)
        _histogram_increment(
            histograms, "wallCountBucket", _wall_bucket(total_walls), 1
        )
        _histogram_increment(
            histograms, "openingDepth", len(opening_actions), 1
        )
        _histogram_increment(
            histograms, "terminationReason", final_state.termination_reason, 1
        )
    return digests, histograms, game_index


def _merge_histograms(
    destination: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, int]]
) -> None:
    for category, values in source.items():
        target = destination[category]
        for key, value in values.items():
            target[key] = target.get(key, 0) + value


def _load_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TeacherDataContractError(f"cannot load {label}: {path}") from error
    if not isinstance(payload, dict):
        raise TeacherDataContractError(f"{label} must be a JSON object")
    return payload


def _verify_self_hash(payload: Mapping[str, object]) -> None:
    stored = payload.get("manifestSha256")
    core = dict(payload)
    core.pop("manifestSha256", None)
    if not isinstance(stored, str) or stored != sha256_bytes(
        canonical_json_bytes(core)
    ):
        raise TeacherDataContractError("teacher manifest checksum mismatch")


def _canonical_self_hashed_mapping(
    path: Path,
    *,
    field: str,
    label: str,
) -> tuple[dict[str, object], str]:
    payload = _load_json_mapping(path, label)
    if path.read_bytes() != canonical_json_bytes(payload):
        raise TeacherDataContractError(f"{label} must be canonical JSON")
    stored = payload.get(field)
    core = dict(payload)
    core.pop(field, None)
    if (
        not isinstance(stored, str)
        or stored != sha256_bytes(canonical_json_bytes(core))
    ):
        raise TeacherDataContractError(f"{label} self hash mismatch")
    return payload, stored


def verify_recovery_preparation_gate(
    *,
    workspace_root: str | Path,
    control_root: str | Path,
    report_path: str | Path,
    output_root: str | Path,
    config: TeacherDataConfig,
) -> dict[str, object]:
    """Authorize CPU-only recovery data without weakening the legacy CUDA gate."""

    workspace = Path(workspace_root).resolve()
    control = Path(control_root).resolve()
    report = Path(report_path)
    report = report if report.is_absolute() else workspace / report
    if report.is_symlink():
        raise TeacherDataContractError("recovery preparation report must not be a symlink")
    report = report.resolve()
    try:
        report.relative_to(control)
    except ValueError as error:
        raise TeacherDataContractError(
            "recovery preparation report must be inside Phase 4 artifacts"
        ) from error
    payload, report_sha256 = _canonical_self_hashed_mapping(
        report,
        field="reportSha256",
        label="recovery preparation report",
    )
    if (
        payload.get("schemaVersion") != 1
        or payload.get("kind") != "phase4f-recovery-preparation"
        or payload.get("status") != "completed"
        or payload.get("cpuOnly") is not True
        or payload.get("gpuAccessed") is not False
        or payload.get("trainingAuthorized") is not False
        or payload.get("source") != fingerprint_sources(workspace).to_payload()
    ):
        raise TeacherDataContractError(
            "recovery preparation report is stale or does not authorize CPU data generation"
        )
    resources = payload.get("resources")
    ledger_path = control / "gpu-budget-ledger.json"
    if (
        not isinstance(resources, Mapping)
        or resources.get("gpuLedgerFileSha256") != sha256_file(ledger_path)
    ):
        raise TeacherDataContractError(
            "GPU ledger changed since the CPU-only recovery preparation"
        )
    challenge_reference = payload.get("challengeContract")
    if not isinstance(challenge_reference, Mapping):
        raise TeacherDataContractError("recovery challenge reference is invalid")
    challenge_value = challenge_reference.get("path")
    if not isinstance(challenge_value, str) or Path(challenge_value).is_absolute():
        raise TeacherDataContractError("recovery challenge path is invalid")
    challenge_path = (workspace / challenge_value).resolve()
    try:
        challenge_path.relative_to(workspace)
    except ValueError as error:
        raise TeacherDataContractError(
            "recovery challenge contract is outside the workspace"
        ) from error
    challenge, challenge_sha256 = _canonical_self_hashed_mapping(
        challenge_path,
        field="contractSha256",
        label="recovery challenge contract",
    )
    if (
        challenge_reference.get("fileSha256") != sha256_file(challenge_path)
        or challenge_reference.get("contractSha256") != challenge_sha256
        or challenge_reference.get("status") != "prepared"
        or challenge.get("kind") != "phase4f-recovery-challenge-replay-contract"
        or challenge.get("status") != "prepared"
        or challenge.get("cpuOnly") is not True
        or challenge.get("trainingUseAllowed") is not False
    ):
        raise TeacherDataContractError("recovery challenge contract binding is invalid")
    try:
        relative_output = Path(output_root).resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise TeacherDataContractError(
            "recovery challenge output must be inside the workspace"
        ) from error
    expected_generator = {
        "script": "scripts/generate-phase4f-challenge-data.sh",
        "targetPositions": config.target_positions,
        "seed": config.seed,
        "nodeBudget": config.node_budget,
        "maxDepth": config.max_depth,
        "explorationProbability": config.exploration_probability,
        "openingDepths": list(config.opening_depths),
    }
    exclusion = challenge.get("exclusionSet")
    if (
        challenge.get("datasetRoot") != relative_output
        or challenge.get("generator") != expected_generator
        or not isinstance(exclusion, Mapping)
        or exclusion.get("sha256") != config.exclusion_set_sha256
        or exclusion.get("orbits") != config.exclusion_orbit_count
    ):
        raise TeacherDataContractError(
            "recovery challenge contract differs from the requested dataset"
        )
    return {
        "kind": "phase4f2-cpu-preparation-gate-v1",
        "reportPath": report.relative_to(workspace).as_posix(),
        "reportSha256": report_sha256,
        "challengeContractPath": challenge_value,
        "challengeContractSha256": challenge_sha256,
        "sourceSha256": payload["source"]["sha256"],
        "gpuLedgerFileSha256": resources["gpuLedgerFileSha256"],
    }


class TeacherDatasetGenerator:
    """Own the two ReplayStores and their recoverable generation cursor."""

    def __init__(
        self,
        config: TeacherDataConfig,
        *,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        workspace_root: str | Path | None = None,
        resource_checker: ResourceChecker = assert_teacher_resources,
        excluded_position_orbits: Sequence[str] = (),
    ) -> None:
        if not isinstance(config, TeacherDataConfig):
            raise TypeError("config must be TeacherDataConfig")
        self.config = config
        self.excluded_position_orbits = frozenset(excluded_position_orbits)
        if len(self.excluded_position_orbits) != len(tuple(excluded_position_orbits)):
            raise ValueError("excluded teacher position orbits must be unique")
        actual_exclusion_sha256 = (
            None
            if not self.excluded_position_orbits
            else exclusion_set_sha256(tuple(self.excluded_position_orbits))
        )
        if (
            actual_exclusion_sha256 != config.exclusion_set_sha256
            or len(self.excluded_position_orbits) != config.exclusion_orbit_count
        ):
            raise ValueError("teacher exclusion set differs from its config binding")
        self.output_root = Path(output_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root is not None
            else Path(__file__).resolve().parents[2]
        )
        try:
            self.output_root.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("output_root must be inside artifact_root") from error
        self.resource_checker = resource_checker
        self.manifest_path = self.output_root / _MANIFEST_NAME
        self.run_manifest_path = self.output_root / _RUN_MANIFEST_NAME
        self.config_fingerprint = fingerprint_config(config.to_payload())
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.resource_checker(self.artifact_root)
        self._initialize_run_manifest()
        max_bytes = DEFAULT_ARTIFACT_LIMIT_BYTES
        self.stores = {
            "train": ReplayStore(
                self.output_root / "train",
                max_games=self.config.train_target,
                max_plies=self.config.train_target + MAX_HALF_MOVES,
                max_bytes=max_bytes,
            ),
            "holdout": ReplayStore(
                self.output_root / "holdout",
                max_games=self.config.holdout_target,
                max_plies=self.config.holdout_target + MAX_HALF_MOVES,
                max_bytes=max_bytes,
            ),
        }
        self.run_manifest = load_run_manifest(self.run_manifest_path)
        self.progress = self._rebuild_progress()

    def _initialize_run_manifest(self) -> None:
        if self.run_manifest_path.exists():
            record = load_run_manifest(self.run_manifest_path)
            provenance = record.payload.get("provenance")
            if not isinstance(provenance, Mapping):
                raise TeacherDataContractError("run manifest provenance is invalid")
            stored_config = provenance.get("config")
            stored_source = provenance.get("source")
            if (
                not isinstance(stored_config, Mapping)
                or stored_config.get("sha256") != self.config_fingerprint.sha256
            ):
                raise TeacherDataContractError(
                    "teacher generation config differs from the persisted run"
                )
            current_source = fingerprint_sources(
                self.workspace_root, _SOURCE_PATHS
            ).sha256
            if (
                not isinstance(stored_source, Mapping)
                or stored_source.get("sha256") != current_source
            ):
                raise TeacherDataContractError(
                    "teacher source fingerprint changed; resume is refused"
                )
            return
        record = build_run_manifest(
            workspace_root=self.workspace_root,
            config=self.config.to_payload(),
            budget=ResourceBudget(
                artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
                minimum_free_disk_bytes=DEFAULT_MINIMUM_FREE_DISK_BYTES,
                gpu_hours=None,
                metadata={"cpuOnly": True, "phase": "4D", "artifactScope": "phase4"},
            ),
            run_id=f"phase4d-teacher-{self.config_fingerprint.sha256[:16]}",
            source_paths=_SOURCE_PATHS,
            environment={
                "execution": "local-cpu-only",
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            metadata={
                "datasetSchemaVersion": TEACHER_DATASET_SCHEMA_VERSION,
                "teacherEngine": CLASSIC_ENGINE,
                "teacherModelId": TEACHER_MODEL_ID,
            },
        )
        save_run_manifest(record, self.run_manifest_path)

    def _rebuild_progress(self) -> _Progress:
        persisted_next = 0
        persisted_duplicates = 0
        if self.manifest_path.exists():
            manifest = _load_json_mapping(self.manifest_path, "teacher manifest")
            _verify_self_hash(manifest)
            if (
                manifest.get("schemaVersion") != TEACHER_DATASET_SCHEMA_VERSION
                or manifest.get("configSha256") != self.config_fingerprint.sha256
                or manifest.get("runManifestSha256") != self.run_manifest.sha256
            ):
                raise TeacherDataContractError("teacher manifest contract mismatch")
            persisted_next = manifest.get("nextGameIndex", 0)
            persisted_duplicates = manifest.get("duplicateTrajectories", 0)
            if type(persisted_next) is not int or persisted_next < 0:
                raise TeacherDataContractError("teacher nextGameIndex is invalid")
            if type(persisted_duplicates) is not int or persisted_duplicates < 0:
                raise TeacherDataContractError(
                    "teacher duplicateTrajectories is invalid"
                )

        seen: set[str] = set()
        histograms = _histogram_template()
        maximum_stored_index = -1
        stored_game_indices: set[int] = set()
        for split, store in self.stores.items():
            # ``iter_games`` performs ReplayStore's checksum/decode/full-game
            # validation.  The teacher verifier below then enforces the
            # stronger policy, provenance, and mirror-orbit contracts.
            store.verify(deep=False)
            for game in store.iter_games():
                digests, game_histograms, game_index = _verify_teacher_game(
                    game, split=split, config=self.config
                )
                states = reconstruct_game(game)
                if any(
                    evaluation_position_orbit_digest(state)
                    in self.excluded_position_orbits
                    for state in states[:-1]
                ):
                    raise TeacherDataContractError(
                        "teacher data intersect an excluded evaluation position"
                    )
                if game_index in stored_game_indices:
                    raise TeacherDataContractError(
                        "one teacher game index appears more than once"
                    )
                stored_game_indices.add(game_index)
                overlap = seen.intersection(digests)
                if overlap:
                    raise TeacherDataContractError(
                        "train/holdout teacher data contain mirrored or duplicate states"
                    )
                seen.update(digests)
                _merge_histograms(histograms, game_histograms)
                maximum_stored_index = max(maximum_stored_index, game_index)
        return _Progress(
            next_game_index=max(persisted_next, maximum_stored_index + 1),
            duplicate_trajectories=persisted_duplicates,
            seen_orbit_digests=seen,
            histograms=histograms,
        )

    def _target_for(self, split: str) -> int:
        return (
            self.config.train_target
            if split == "train"
            else self.config.holdout_target
        )

    def is_complete(self) -> bool:
        return all(
            self.stores[split].ply_count >= self._target_for(split)
            for split in ("train", "holdout")
        )

    def _content_fingerprint(self) -> str:
        payload = {
            "schemaVersion": TEACHER_DATASET_SCHEMA_VERSION,
            "configSha256": self.config_fingerprint.sha256,
            "stores": {
                split: {
                    "manifestSha256": sha256_file(store.manifest_path),
                    "games": store.game_count,
                    "positions": store.ply_count,
                }
                for split, store in sorted(self.stores.items())
            },
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def _manifest_core(self, status: str) -> dict[str, object]:
        content_fingerprint = self._content_fingerprint()
        return {
            "schemaVersion": TEACHER_DATASET_SCHEMA_VERSION,
            "status": status,
            "updatedAt": utc_timestamp(),
            "configSha256": self.config_fingerprint.sha256,
            "runManifestSha256": self.run_manifest.sha256,
            "nextGameIndex": self.progress.next_game_index,
            "attemptedGameIndices": self.progress.next_game_index,
            "duplicateTrajectories": self.progress.duplicate_trajectories,
            "targets": {
                "totalPositions": self.config.target_positions,
                "trainPositions": self.config.train_target,
                "holdoutPositions": self.config.holdout_target,
            },
            "actual": {
                "totalPositions": sum(store.ply_count for store in self.stores.values()),
                "trainPositions": self.stores["train"].ply_count,
                "holdoutPositions": self.stores["holdout"].ply_count,
                "trainGames": self.stores["train"].game_count,
                "holdoutGames": self.stores["holdout"].game_count,
                "uniqueRawStates": len(self.progress.seen_orbit_digests),
                "uniqueAugmentationOrbits": len(self.progress.seen_orbit_digests),
            },
            "strata": self.progress.histograms,
            "stores": {
                split: {
                    "path": split,
                    "manifestSha256": sha256_file(store.manifest_path),
                    "committedBytes": store.committed_bytes,
                    "games": store.game_count,
                    "positions": store.ply_count,
                }
                for split, store in sorted(self.stores.items())
            },
            "contentSha256": content_fingerprint,
            "datasetFingerprint": content_fingerprint if status == "completed" else None,
        }

    def write_manifest(self, status: str) -> None:
        core = self._manifest_core(status)
        payload = {
            **core,
            "manifestSha256": sha256_bytes(canonical_json_bytes(core)),
        }
        atomic_replace_bytes(self.manifest_path, canonical_json_bytes(payload))

    def summary(self, status: str | None = None) -> TeacherDatasetSummary:
        resolved_status = status or ("completed" if self.is_complete() else "running")
        return TeacherDatasetSummary(
            status=resolved_status,
            train_positions=self.stores["train"].ply_count,
            holdout_positions=self.stores["holdout"].ply_count,
            train_games=self.stores["train"].game_count,
            holdout_games=self.stores["holdout"].game_count,
            next_game_index=self.progress.next_game_index,
            duplicate_trajectories=self.progress.duplicate_trajectories,
            dataset_fingerprint=(
                self._content_fingerprint() if self.is_complete() else None
            ),
            manifest_path=self.manifest_path,
            run_manifest_path=self.run_manifest_path,
        )

    def verify(self, *, require_complete: bool = False) -> TeacherDatasetSummary:
        self.progress = self._rebuild_progress()
        if require_complete and not self.is_complete():
            raise TeacherDataContractError("teacher dataset is incomplete")
        status = "completed" if self.is_complete() else "running"
        self.write_manifest(status)
        return self.summary(status)

    def generate(
        self,
        *,
        bridge_factory: BridgeFactory = ClassicAIBridge,
        maximum_new_games: int | None = None,
        progress_callback: Callable[[TeacherDatasetSummary], None] | None = None,
    ) -> TeacherDatasetSummary:
        if maximum_new_games is not None and (
            type(maximum_new_games) is not int or maximum_new_games <= 0
        ):
            raise ValueError("maximum_new_games must be a positive integer or None")
        if self.is_complete():
            self.write_manifest("completed")
            return self.summary("completed")
        generated_games = 0
        bridge = bridge_factory()
        try:
            while not self.is_complete():
                if self.progress.next_game_index >= self.config.maximum_attempts:
                    raise TeacherDataError(
                        "maximum teacher trajectory attempts reached before targets"
                    )
                if maximum_new_games is not None and generated_games >= maximum_new_games:
                    self.write_manifest("paused")
                    return self.summary("paused")
                self.resource_checker(self.artifact_root)
                game_index = self.progress.next_game_index
                split = split_for_game(self.config.seed, game_index)
                if self.stores[split].ply_count >= self._target_for(split):
                    self.progress.next_game_index += 1
                    self.write_manifest("running")
                    continue
                try:
                    game, digests = _build_teacher_game(
                        self.config,
                        game_index,
                        split,
                        bridge,
                        self.progress.seen_orbit_digests,
                        self.excluded_position_orbits,
                    )
                except DuplicateTeacherTrajectory:
                    self.progress.duplicate_trajectories += 1
                    self.progress.next_game_index += 1
                    self.write_manifest("running")
                    continue
                self.resource_checker(self.artifact_root)
                self.stores[split].append(game)
                verified_digests, histograms, verified_index = _verify_teacher_game(
                    game, split=split, config=self.config
                )
                if verified_index != game_index or verified_digests != set(digests):
                    raise TeacherDataContractError(
                        "new teacher shard failed post-commit identity validation"
                    )
                overlap = self.progress.seen_orbit_digests.intersection(
                    verified_digests
                )
                if overlap:
                    raise TeacherDataContractError(
                        "new teacher shard overlaps a committed augmentation orbit"
                    )
                self.progress.seen_orbit_digests.update(verified_digests)
                _merge_histograms(self.progress.histograms, histograms)
                self.progress.next_game_index += 1
                generated_games += 1
                status = "completed" if self.is_complete() else "running"
                self.write_manifest(status)
                if progress_callback is not None:
                    progress_callback(self.summary(status))
            return self.summary("completed")
        except KeyboardInterrupt:
            self.write_manifest("interrupted")
            return self.summary("interrupted")
        finally:
            bridge.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Phase 4D classic-hard teacher replay"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--target-positions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--node-budget", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--exploration-probability", type=float, default=0.15)
    parser.add_argument("--opening-depths", type=int, nargs="+", default=[8, 24, 48])
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-attempts", type=int, default=250_000)
    parser.add_argument(
        "--exclude-suite",
        action="append",
        type=Path,
        default=[],
        help="opening suite whose playable position orbits must be excluded",
    )
    parser.add_argument(
        "--exclude-arena-result",
        action="append",
        type=Path,
        default=[],
        help="completed arena result whose full trajectories must be excluded",
    )
    parser.add_argument(
        "--recovery-preparation-report",
        type=Path,
        default=None,
        help="Phase 4F2 CPU-only preparation report replacing the legacy CUDA gate",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="deep-verify existing shards without starting the Node bridge",
    )
    return parser


def _install_signal_handlers() -> None:
    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    workspace_root = Path(__file__).resolve().parents[2]
    excluded_position_orbits: set[str] = set()
    for raw_path in arguments.exclude_suite:
        suite_path = raw_path if raw_path.is_absolute() else workspace_root / raw_path
        suite = load_opening_suite(suite_path)
        excluded_position_orbits.update(
            evaluation_position_orbit_digest(opening.state)
            for opening in suite.openings
        )
    for raw_path in arguments.exclude_arena_result:
        arena_path = raw_path if raw_path.is_absolute() else workspace_root / raw_path
        arena_payload = json.loads(arena_path.read_bytes())
        excluded_position_orbits.update(
            arena_position_orbits(
                workspace_root,
                {
                    "path": arena_path.resolve().relative_to(workspace_root).as_posix(),
                    "fileSha256": sha256_file(arena_path),
                    "resultSha256": arena_payload.get("resultSha256"),
                },
            )
        )
    ordered_exclusions = tuple(sorted(excluded_position_orbits))
    config = TeacherDataConfig(
        target_positions=arguments.target_positions,
        seed=arguments.seed,
        node_budget=arguments.node_budget,
        max_depth=arguments.max_depth,
        exploration_probability=arguments.exploration_probability,
        opening_depths=tuple(arguments.opening_depths),
        request_timeout_seconds=arguments.request_timeout_seconds,
        maximum_attempts=arguments.maximum_attempts,
        exclusion_set_sha256=(
            exclusion_set_sha256(ordered_exclusions) if ordered_exclusions else None
        ),
        exclusion_orbit_count=len(ordered_exclusions),
    )
    output_root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else workspace_root / arguments.output_root
    )
    artifact_root = (
        arguments.artifact_root
        if arguments.artifact_root.is_absolute()
        else workspace_root / arguments.artifact_root
    ).resolve()
    control_root = (workspace_root / PHASE4_CONTROL_RELATIVE).resolve()
    if artifact_root != control_root:
        raise ValueError(
            "artifact_root must be the shared training/runs/phase4 control root"
        )
    _install_signal_handlers()
    lock_path = control_root / GPU_LOCK_NAME
    with RunLock(lock_path, run_id="phase4d-teacher-data"):
        if arguments.recovery_preparation_report is None:
            find_valid_memory_smoke(control_root, workspace_root)
        else:
            recovery_gate = verify_recovery_preparation_gate(
                workspace_root=workspace_root,
                control_root=control_root,
                report_path=arguments.recovery_preparation_report,
                output_root=output_root,
                config=config,
            )
            config = replace(
                config,
                recovery_preparation_report_sha256=str(
                    recovery_gate["reportSha256"]
                ),
                recovery_challenge_contract_sha256=str(
                    recovery_gate["challengeContractSha256"]
                ),
            )
        generator = TeacherDatasetGenerator(
            config,
            output_root=output_root,
            artifact_root=artifact_root,
            workspace_root=workspace_root,
            excluded_position_orbits=ordered_exclusions,
        )
        if arguments.verify_only:
            summary = generator.verify(require_complete=False)
        else:
            last_progress = 0.0

            def report_progress(progress: TeacherDatasetSummary) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if now - last_progress < 30.0 and progress.status != "completed":
                    return
                last_progress = now
                print(
                    json.dumps(
                        {
                            "event": "teacher-data-progress",
                            "status": progress.status,
                            "positions": {
                                "total": progress.total_positions,
                                "train": progress.train_positions,
                                "holdout": progress.holdout_positions,
                            },
                            "games": {
                                "train": progress.train_games,
                                "holdout": progress.holdout_games,
                            },
                            "nextGameIndex": progress.next_game_index,
                            "duplicateTrajectories": progress.duplicate_trajectories,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )

            summary = generator.generate(progress_callback=report_progress)
    print(
        json.dumps(
            summary.to_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary.status in {"completed", "running", "paused"} else 130


__all__ = [
    "TEACHER_DATASET_SCHEMA_VERSION",
    "TEACHER_POLICY_SCALE",
    "TEACHER_POLICY_FLOOR_UNITS",
    "TEACHER_MODEL_ID",
    "TeacherDataError",
    "TeacherDataContractError",
    "TeacherDataResourceError",
    "TeacherDataConfig",
    "TeacherDatasetSummary",
    "TeacherDatasetGenerator",
    "canonical_state_orbit_digest",
    "evaluation_position_orbit_digest",
    "exclusion_set_sha256",
    "teacher_policy_counts",
    "split_for_game",
    "assert_teacher_resources",
    "verify_recovery_preparation_gate",
    "create_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
