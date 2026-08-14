"""Provenance and evaluation-leakage gates for recovery challenge replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .codec import mirror_position_hash, mirror_state, position_hash, validate_position_hash
from .config import ChallengeReplaySettings, HealthHoldoutSettings
from .openings import load_opening_suite
from .replay_store import ReplayGame, ReplayStore, ReplayVerification
from .run_manifest import canonical_json_bytes, sha256_bytes, sha256_file


class ChallengeReplayError(RuntimeError):
    """Raised when external recovery replay violates its frozen contract."""


ProgressCallback = Callable[[Mapping[str, object]], None]


def _progress(
    callback: ProgressCallback | None,
    *,
    operation: str,
    status: str,
    games: int,
    games_total: int,
) -> None:
    if callback is not None:
        callback(
            {
                "operation": operation,
                "status": status,
                "games": games,
                "gamesTotal": games_total,
            }
        )


@dataclass(frozen=True, slots=True)
class BoundChallengeReplay:
    store: ReplayStore
    contract: Mapping[str, object]
    verification: ReplayVerification
    excluded_orbits: int
    position_orbits: frozenset[str]


@dataclass(frozen=True, slots=True)
class BoundHealthHoldout:
    store: ReplayStore
    contract: Mapping[str, object]
    verification: ReplayVerification
    excluded_orbits: int
    train_orbits: int
    holdout_orbits: int


def _bound_file(workspace: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else workspace / path
    if candidate.is_symlink():
        raise ChallengeReplayError(f"{label} must not be a symlink")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as error:
        raise ChallengeReplayError(f"{label} must be inside the workspace") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ChallengeReplayError(f"{label} is missing")
    return candidate


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChallengeReplayError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise ChallengeReplayError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    stored = payload.get(field)
    core = dict(payload)
    core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise ChallengeReplayError(f"{label} self hash mismatch")
    return stored


def position_orbit_digest(state: object) -> str:
    raw = position_hash(state).encode("ascii")
    mirrored = position_hash(mirror_state(state)).encode("ascii")
    return sha256_bytes(min(raw, mirrored))


def position_hash_orbit_digest(value: str) -> str:
    validate_position_hash(value)
    mirrored = mirror_position_hash(value)
    return sha256_bytes(min(value, mirrored).encode("ascii"))


def arena_position_orbits(
    workspace_root: str | Path,
    reference: Mapping[str, object],
) -> set[str]:
    """Deeply bind every position in one completed paired-arena result."""

    workspace = Path(workspace_root).resolve()
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise ChallengeReplayError("excluded arena result path is invalid")
    result_path = _bound_file(workspace, path_value, "excluded arena result")
    result = _mapping(result_path, "excluded arena result")
    if sha256_file(result_path) != reference.get("fileSha256"):
        raise ChallengeReplayError("excluded arena result file hash mismatch")
    result_core = dict(result)
    result_sha256 = result_core.pop("resultSha256", None)
    if (
        result.get("status") != "completed"
        or result_sha256 != reference.get("resultSha256")
        or result_sha256 != sha256_bytes(canonical_json_bytes(result_core))
    ):
        raise ChallengeReplayError("excluded arena result binding is invalid")
    artifacts = result.get("pairArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ChallengeReplayError("excluded arena result has no pair artifacts")
    orbits: set[str] = set()
    games = 0
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ChallengeReplayError("excluded arena pair reference is invalid")
        file_name = artifact.get("file")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not file_name.endswith(".json")
        ):
            raise ChallengeReplayError("excluded arena pair filename is invalid")
        pair_path = _bound_file(
            workspace,
            result_path.parent / file_name,
            "excluded arena pair",
        )
        if sha256_file(pair_path) != artifact.get("sha256"):
            raise ChallengeReplayError("excluded arena pair file hash mismatch")
        pair = _mapping(pair_path, "excluded arena pair")
        pair_core = dict(pair)
        pair_sha256 = pair_core.pop("pairSha256", None)
        if pair_sha256 != sha256_bytes(canonical_json_bytes(pair_core)):
            raise ChallengeReplayError("excluded arena pair self hash mismatch")
        pair_games = pair.get("games")
        if not isinstance(pair_games, list) or len(pair_games) != 2:
            raise ChallengeReplayError("excluded arena pair games are invalid")
        games += len(pair_games)
        for game in pair_games:
            if not isinstance(game, Mapping):
                raise ChallengeReplayError("excluded arena game is invalid")
            hashes: list[object] = [
                game.get("initialPositionHash"),
                game.get("finalPositionHash"),
            ]
            plies = game.get("plies")
            if not isinstance(plies, list):
                raise ChallengeReplayError("excluded arena game plies are invalid")
            for ply in plies:
                if not isinstance(ply, Mapping):
                    raise ChallengeReplayError("excluded arena ply is invalid")
                hashes.extend(
                    (ply.get("positionHashBefore"), ply.get("positionHashAfter"))
                )
            for value in hashes:
                if not isinstance(value, str):
                    raise ChallengeReplayError("excluded arena position hash is invalid")
                orbits.add(position_hash_orbit_digest(value))
    expected_games = result.get("games")
    if expected_games != games:
        raise ChallengeReplayError("excluded arena game total mismatch")
    return orbits


def _excluded_orbits(workspace: Path, references: object) -> set[str]:
    if not isinstance(references, list) or not references:
        raise ChallengeReplayError("challenge contract has no excluded suites")
    result: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ChallengeReplayError("excluded suite reference is invalid")
        path_value = reference.get("path")
        if not isinstance(path_value, str):
            raise ChallengeReplayError("excluded suite path is invalid")
        path = _bound_file(workspace, path_value, "excluded suite")
        if sha256_file(path) != reference.get("fileSha256"):
            raise ChallengeReplayError("excluded suite file hash mismatch")
        name = reference.get("name")
        if not isinstance(name, str):
            raise ChallengeReplayError("excluded suite name is invalid")
        suite = load_opening_suite(path, expected_suite=name)
        if suite.content_hash != reference.get("suiteHash"):
            raise ChallengeReplayError("excluded suite content hash mismatch")
        result.update(position_orbit_digest(opening.state) for opening in suite.openings)
    return result


def _excluded_trajectory_orbits(workspace: Path, references: object) -> set[str]:
    if not isinstance(references, list) or not references:
        raise ChallengeReplayError("challenge contract has no excluded trajectories")
    result: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ChallengeReplayError("excluded trajectory reference is invalid")
        result.update(arena_position_orbits(workspace, reference))
    return result


def _capacity(manifest: Mapping[str, object]) -> tuple[int, int, int]:
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise ChallengeReplayError("challenge train manifest capacity is invalid")
    values = tuple(capacity.get(key) for key in ("games", "plies", "bytes"))
    if any(type(value) is not int or value <= 0 for value in values):
        raise ChallengeReplayError("challenge train capacity values are invalid")
    return int(values[0]), int(values[1]), int(values[2])


def open_bound_challenge_replay(
    *,
    workspace_root: str | Path,
    settings: ChallengeReplaySettings,
    progress_callback: ProgressCallback | None = None,
) -> BoundChallengeReplay:
    workspace = Path(workspace_root).resolve()
    contract_path = _bound_file(workspace, settings.contract_path, "challenge contract")
    contract = _mapping(contract_path, "challenge contract")
    contract_sha256 = _verify_self_hash(contract, "contractSha256", "challenge contract")
    if contract_sha256 != settings.contract_sha256:
        raise ChallengeReplayError("challenge contract hash differs from config")
    if (
        contract.get("schemaVersion") != 1
        or contract.get("kind") != "phase4f-recovery-challenge-replay-contract"
        or contract.get("status") != "completed"
        or contract.get("trainingUseAllowed") is not True
    ):
        raise ChallengeReplayError("challenge contract is not training-authorized")

    root = settings.root if settings.root.is_absolute() else workspace / settings.root
    root = root.resolve()
    try:
        root.relative_to(workspace)
    except ValueError as error:
        raise ChallengeReplayError("challenge root must be inside the workspace") from error
    contract_root = contract.get("datasetRoot")
    if not isinstance(contract_root, str) or root != (workspace / contract_root).resolve():
        raise ChallengeReplayError("challenge dataset root differs from contract")
    dataset_manifest_path = _bound_file(
        workspace, root / "teacher-manifest.json", "challenge dataset manifest"
    )
    dataset_manifest = _mapping(dataset_manifest_path, "challenge dataset manifest")
    if (
        sha256_file(dataset_manifest_path) != settings.dataset_manifest_sha256
        or dataset_manifest.get("status") != "completed"
        or dataset_manifest.get("datasetFingerprint") != settings.dataset_fingerprint
        or contract.get("datasetFingerprint") != settings.dataset_fingerprint
        or contract.get("datasetManifestSha256") != settings.dataset_manifest_sha256
    ):
        raise ChallengeReplayError("challenge dataset manifest binding is invalid")

    train_root = root / "train"
    train_manifest_path = _bound_file(
        workspace, train_root / "manifest.json", "challenge train manifest"
    )
    if (
        sha256_file(train_manifest_path) != settings.train_manifest_sha256
        or contract.get("trainManifestSha256") != settings.train_manifest_sha256
    ):
        raise ChallengeReplayError("challenge train manifest binding is invalid")
    train_manifest = _mapping(train_manifest_path, "challenge train manifest")
    max_games, max_plies, max_bytes = _capacity(train_manifest)
    store = ReplayStore(
        train_root,
        max_games=max_games,
        max_plies=max_plies,
        max_bytes=max_bytes,
    )
    excluded = _excluded_orbits(workspace, contract.get("excludedSuites"))
    excluded.update(
        _excluded_trajectory_orbits(
            workspace,
            contract.get("excludedTrajectoryEvidence"),
        )
    )
    exclusion_set = contract.get("exclusionSet")
    if (
        not isinstance(exclusion_set, Mapping)
        or exclusion_set.get("algorithm")
        != "sha256-sorted-horizontal-position-orbits-v1"
        or exclusion_set.get("orbits") != len(excluded)
        or exclusion_set.get("sha256")
        != sha256_bytes(canonical_json_bytes(sorted(excluded)))
    ):
        raise ChallengeReplayError("challenge exclusion-set binding is invalid")

    verified_games = 0
    position_orbits: set[str] = set()
    _progress(
        progress_callback,
        operation="challenge-train-deep-verify",
        status="started",
        games=0,
        games_total=store.game_count,
    )

    def validate_game(game: ReplayGame, states: Sequence[object]) -> None:
        nonlocal verified_games
        if game.metadata.get("teacherEngine") != "classic-negamax":
            raise ChallengeReplayError("challenge replay is not classic-hard teacher data")
        for state in states[:-1]:
            orbit = position_orbit_digest(state)
            if orbit in excluded:
                raise ChallengeReplayError(
                    "challenge replay intersects an excluded evaluation orbit"
                )
            position_orbits.add(orbit)
        verified_games += 1
        if verified_games % 100 == 0 or verified_games == store.game_count:
            _progress(
                progress_callback,
                operation="challenge-train-deep-verify",
                status="progress",
                games=verified_games,
                games_total=store.game_count,
            )

    verification = store.verify(deep=True, game_validator=validate_game)
    if verification.ply_count <= 0:
        raise ChallengeReplayError("challenge replay is empty")
    _progress(
        progress_callback,
        operation="challenge-train-deep-verify",
        status="completed",
        games=verified_games,
        games_total=store.game_count,
    )
    return BoundChallengeReplay(
        store=store,
        contract=contract,
        verification=verification,
        excluded_orbits=len(excluded),
        position_orbits=frozenset(position_orbits),
    )


def open_bound_health_holdout(
    *,
    workspace_root: str | Path,
    settings: HealthHoldoutSettings,
    bound_train: BoundChallengeReplay | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BoundHealthHoldout:
    """Open a fixed teacher holdout and prove it is disjoint from training."""

    workspace = Path(workspace_root).resolve()
    if bound_train is None:
        bound_train = open_bound_challenge_replay(
            workspace_root=workspace,
            settings=ChallengeReplaySettings(
                root=settings.root,
                contract_path=settings.contract_path,
                contract_sha256=settings.contract_sha256,
                dataset_fingerprint=settings.dataset_fingerprint,
                dataset_manifest_sha256=settings.dataset_manifest_sha256,
                train_manifest_sha256=settings.train_manifest_sha256,
                fraction=0.5,
            ),
            progress_callback=progress_callback,
        )
    else:
        if (
            bound_train.contract.get("contractSha256")
            != settings.contract_sha256
            or bound_train.store.root.resolve()
            != (workspace / settings.root / "train").resolve()
        ):
            raise ChallengeReplayError(
                "prevalidated challenge train binding differs from health holdout"
            )
    root = settings.root if settings.root.is_absolute() else workspace / settings.root
    root = root.resolve()
    holdout_root = root / "holdout"
    holdout_manifest_path = _bound_file(
        workspace,
        holdout_root / "manifest.json",
        "health holdout manifest",
    )
    dataset_manifest_path = _bound_file(
        workspace,
        root / "teacher-manifest.json",
        "health holdout dataset manifest",
    )
    dataset_manifest = _mapping(
        dataset_manifest_path,
        "health holdout dataset manifest",
    )
    stores = dataset_manifest.get("stores")
    holdout_reference = (
        stores.get("holdout") if isinstance(stores, Mapping) else None
    )
    contract_evidence = bound_train.contract.get("datasetEvidence")
    if (
        sha256_file(holdout_manifest_path) != settings.holdout_manifest_sha256
        or not isinstance(holdout_reference, Mapping)
        or holdout_reference.get("manifestSha256")
        != settings.holdout_manifest_sha256
        or not isinstance(contract_evidence, Mapping)
        or contract_evidence.get("holdoutManifestSha256")
        != settings.holdout_manifest_sha256
    ):
        raise ChallengeReplayError("health holdout manifest binding is invalid")
    holdout_manifest = _mapping(holdout_manifest_path, "health holdout manifest")
    max_games, max_plies, max_bytes = _capacity(holdout_manifest)
    store = ReplayStore(
        holdout_root,
        max_games=max_games,
        max_plies=max_plies,
        max_bytes=max_bytes,
    )

    excluded = _excluded_orbits(workspace, bound_train.contract.get("excludedSuites"))
    excluded.update(
        _excluded_trajectory_orbits(
            workspace,
            bound_train.contract.get("excludedTrajectoryEvidence"),
        )
    )
    train_orbits = set(bound_train.position_orbits)

    holdout_orbits: set[str] = set()

    verified_holdout_games = 0
    _progress(
        progress_callback,
        operation="health-holdout-deep-verify",
        status="started",
        games=0,
        games_total=store.game_count,
    )

    def validate_game(game: ReplayGame, states: Sequence[object]) -> None:
        nonlocal verified_holdout_games
        if game.metadata.get("teacherEngine") != "classic-negamax":
            raise ChallengeReplayError("health holdout is not classic-hard teacher data")
        for state in states[:-1]:
            orbit = position_orbit_digest(state)
            if orbit in excluded:
                raise ChallengeReplayError(
                    "health holdout intersects an excluded evaluation orbit"
                )
            holdout_orbits.add(orbit)
        verified_holdout_games += 1
        if (
            verified_holdout_games % 100 == 0
            or verified_holdout_games == store.game_count
        ):
            _progress(
                progress_callback,
                operation="health-holdout-deep-verify",
                status="progress",
                games=verified_holdout_games,
                games_total=store.game_count,
            )

    verification = store.verify(deep=True, game_validator=validate_game)
    if verification.ply_count <= 0:
        raise ChallengeReplayError("health holdout is empty")
    if len(holdout_orbits) != verification.ply_count:
        raise ChallengeReplayError("health holdout contains duplicate symmetry orbits")
    if train_orbits.intersection(holdout_orbits):
        raise ChallengeReplayError("challenge train and health holdout overlap")
    _progress(
        progress_callback,
        operation="health-holdout-deep-verify",
        status="completed",
        games=verified_holdout_games,
        games_total=store.game_count,
    )
    return BoundHealthHoldout(
        store=store,
        contract=bound_train.contract,
        verification=verification,
        excluded_orbits=len(excluded),
        train_orbits=len(train_orbits),
        holdout_orbits=len(holdout_orbits),
    )


def mixed_batch_counts(batch_size: int, fraction: float) -> tuple[int, int]:
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("mixed batch size must be at least two")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < float(fraction) < 1
    ):
        raise ValueError("challenge fraction must be in (0, 1)")
    challenge = min(batch_size - 1, max(1, round(batch_size * float(fraction))))
    return batch_size - challenge, challenge


def merge_tensor_batches(
    selfplay: Mapping[str, torch.Tensor],
    challenge: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expected = {"states", "policies", "values"}
    if set(selfplay) != expected or set(challenge) != expected:
        raise ValueError("mixed replay batches have invalid fields")
    return {
        key: torch.cat((selfplay[key], challenge[key]), dim=0)
        for key in sorted(expected)
    }


__all__ = [
    "BoundChallengeReplay",
    "BoundHealthHoldout",
    "ChallengeReplayError",
    "merge_tensor_batches",
    "mixed_batch_counts",
    "open_bound_challenge_replay",
    "open_bound_health_holdout",
    "arena_position_orbits",
    "position_orbit_digest",
    "position_hash_orbit_digest",
]
