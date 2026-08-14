"""Freeze the CPU-only data and policy contract for staged recovery training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from .challenge_replay import (
    arena_position_orbits,
    open_bound_health_holdout,
    position_orbit_digest,
)
from .config import HealthHoldoutSettings
from .openings import load_opening_suite, validate_suite_collection
from .replay_store import ReplayStore, reconstruct_game
from .run_manifest import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .teacher_data import evaluation_position_orbit_digest


SCHEMA_VERSION = 1
SOURCE_DATASET_ROOT = Path(
    "training/runs/phase4/phase4f/recovery-challenge-data"
)
DATASET_ROOT = Path(
    "training/runs/phase4/phase4f/recovery-challenge-v2-data"
)
CONTRACT_PATH = Path(
    "training/configs/phase4f17-classic-guidance-dataset-frozen-v1.json"
)
PROTOCOL_PATH = Path(
    "training/configs/phase4f17-two-stage-protocol-frozen-v1.json"
)
REPORT_PATH = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "staged-learning-report.json"
)
STAGING_CONTRACT_PATH = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "staged-contract-validation.tmp.json"
)
SUITE_PATHS = (
    Path("training/fixtures/openings/dev.json"),
    Path("training/fixtures/openings/final.json"),
    Path("training/fixtures/openings/recovery-v1/recovery-dev-v1.json"),
    Path("training/fixtures/openings/recovery-v1/recovery-final-v1.json"),
)
ARENA_RESULT_PATHS = (
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1-g14/"
        "confirmation-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1-g14/"
        "screen-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1-g9/"
        "confirmation-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1-g9/"
        "screen-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1/"
        "screen-arena-v2/store/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e/promotion-bootstrap-v1/"
        "screen-arena/store/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e1/calibration/"
        "phase4e1-calibration-47fbe72edd9c75c1/arena-store/result.json"
    ),
    Path(
        "training/runs/phase4/phase4e1/quarantine/"
        "phase4e1-calibration-47fbe72edd9c75c1-source-"
        "7fb9fd3ecd0568e183204f77142189aeb77f175cdd4e4ac603919a18fbdd5acd-"
        "manifest-c590e84e90cb672590cc36f6b6f9156eb930f00987368af85024c2f6a43f7e85/"
        "arena-store/result.json"
    ),
    Path("training/runs/phase4/phase4f/fixed-standard-initial/result.json"),
    Path("training/runs/phase4/phase4f/fixed/result.json"),
    Path(
        "training/runs/phase4/phase4f/recovery-evaluation-v1/"
        "screen-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4f/recovery-evaluation-v2/"
        "screen-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4f/recovery-evaluation-v3/"
        "screen-arena/result.json"
    ),
    Path(
        "training/runs/phase4/phase4f/recovery-evaluation-v4/"
        "screen-arena/result.json"
    ),
)


class Phase4F17StagedError(RuntimeError):
    """Raised when staged recovery evidence is incomplete or inconsistent."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F17StagedError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F17StagedError(f"{label} must be canonical JSON: {path}")
    return dict(payload)


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _suite_evidence(workspace: Path) -> tuple[list[dict[str, object]], set[str]]:
    suites = [load_opening_suite(workspace / path) for path in SUITE_PATHS]
    validate_suite_collection(*suites)
    evidence: list[dict[str, object]] = []
    orbits: set[str] = set()
    for path, suite in zip(SUITE_PATHS, suites, strict=True):
        current = {
            evaluation_position_orbit_digest(opening.state)
            for opening in suite.openings
        }
        evidence.append(
            {
                "name": suite.name,
                "path": path.as_posix(),
                "fileSha256": sha256_file(workspace / path),
                "suiteHash": suite.content_hash,
                "openings": len(suite.openings),
                "positionOrbits": len(current),
            }
        )
        orbits.update(current)
    return evidence, orbits


def _arena_evidence(workspace: Path) -> tuple[list[dict[str, object]], set[str]]:
    evidence: list[dict[str, object]] = []
    orbits: set[str] = set()
    result_hashes: set[str] = set()
    for path in ARENA_RESULT_PATHS:
        full_path = workspace / path
        payload = _mapping(full_path, "arena result")
        result_sha256 = payload.get("resultSha256")
        if not isinstance(result_sha256, str) or result_sha256 in result_hashes:
            raise Phase4F17StagedError("arena result hashes must be unique")
        result_hashes.add(result_sha256)
        reference = {
            "path": path.as_posix(),
            "fileSha256": sha256_file(full_path),
            "resultSha256": result_sha256,
        }
        current = arena_position_orbits(workspace, reference)
        evidence.append(
            {
                **reference,
                "games": payload.get("games"),
                "positionOrbits": len(current),
            }
        )
        orbits.update(current)
    return evidence, orbits


def _dataset_evidence(workspace: Path) -> dict[str, object]:
    root = workspace / DATASET_ROOT
    dataset_path = root / "teacher-manifest.json"
    train_path = root / "train/manifest.json"
    holdout_path = root / "holdout/manifest.json"
    dataset = _mapping(dataset_path, "teacher dataset manifest")
    train = _mapping(train_path, "teacher train manifest")
    holdout = _mapping(holdout_path, "teacher holdout manifest")
    actual = dataset.get("actual")
    strata = dataset.get("strata")
    if (
        dataset.get("status") != "completed"
        or not isinstance(actual, Mapping)
        or not isinstance(strata, Mapping)
        or actual.get("trainPositions") != train.get("totals", {}).get("plies")
        or actual.get("holdoutPositions") != holdout.get("totals", {}).get("plies")
    ):
        raise Phase4F17StagedError("teacher dataset evidence is inconsistent")
    return {
        "root": DATASET_ROOT.as_posix(),
        "datasetFingerprint": dataset.get("datasetFingerprint"),
        "datasetManifestPath": (DATASET_ROOT / "teacher-manifest.json").as_posix(),
        "datasetManifestSha256": sha256_file(dataset_path),
        "datasetManifestSelfSha256": dataset.get("manifestSha256"),
        "trainManifestPath": (DATASET_ROOT / "train/manifest.json").as_posix(),
        "trainManifestSha256": sha256_file(train_path),
        "holdoutManifestPath": (
            DATASET_ROOT / "holdout/manifest.json"
        ).as_posix(),
        "holdoutManifestSha256": sha256_file(holdout_path),
        "trainPositions": actual.get("trainPositions"),
        "holdoutPositions": actual.get("holdoutPositions"),
        "trainGames": actual.get("trainGames"),
        "holdoutGames": actual.get("holdoutGames"),
        "strata": strata,
    }


def _replay_store(root: Path) -> ReplayStore:
    manifest = _mapping(root / "manifest.json", "replay manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F17StagedError("replay manifest capacity is invalid")
    try:
        return ReplayStore(
            root,
            max_games=int(capacity["games"]),
            max_plies=int(capacity["plies"]),
            max_bytes=int(capacity["bytes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4F17StagedError("replay manifest capacity is invalid") from error


def _prepare_derived_dataset(workspace: Path) -> dict[str, object]:
    """Reuse train bytes and remove whole holdout games with repeated positions."""

    source_root = workspace / SOURCE_DATASET_ROOT
    target_root = workspace / DATASET_ROOT
    source_manifest = _mapping(
        source_root / "teacher-manifest.json",
        "source teacher dataset manifest",
    )
    source_train = source_root / "train"
    target_train = target_root / "train"
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_train, target_train, dirs_exist_ok=True)

    source_holdout = _replay_store(source_root / "holdout")
    source_holdout_manifest = _mapping(
        source_root / "holdout/manifest.json",
        "source holdout manifest",
    )
    capacity = source_holdout_manifest["capacity"]
    if not isinstance(capacity, Mapping):
        raise Phase4F17StagedError("source holdout capacity is invalid")
    target_holdout = ReplayStore(
        target_root / "holdout",
        max_games=int(capacity["games"]),
        max_plies=int(capacity["plies"]),
        max_bytes=int(capacity["bytes"]),
    )

    seen: set[str] = set()
    kept_games = []
    dropped_games: list[dict[str, object]] = []
    dropped_strata: dict[str, dict[str, int]] = {
        key: {} for key in source_manifest["strata"]
    }
    for game in source_holdout.iter_games():
        states = reconstruct_game(game)
        digests = [position_orbit_digest(state) for state in states[:-1]]
        reasons: list[str] = []
        if len(set(digests)) != len(digests):
            reasons.append("within-game-position-orbit-repeat")
        if seen.intersection(digests):
            reasons.append("prior-game-position-orbit-repeat")
        if reasons:
            dropped_games.append(
                {
                    "gameId": game.game_id,
                    "positions": len(game.plies),
                    "reasons": reasons,
                }
            )
            for ply in game.plies:
                values = {
                    "startingPlayer": game.metadata.get("startingPlayer"),
                    "winner": game.metadata.get("winner"),
                    "sideToMove": ply.metadata.get("sideToMove"),
                    "wallCountBucket": ply.metadata.get("wallCountBucket"),
                    "openingDepth": ply.metadata.get("openingDepth"),
                    "terminationReason": game.metadata.get("terminationReason"),
                }
                for category, value in values.items():
                    key = str(value)
                    bucket = dropped_strata[category]
                    bucket[key] = bucket.get(key, 0) + 1
            continue
        seen.update(digests)
        kept_games.append(game)

    expected_ids = tuple(game.game_id for game in kept_games)
    if target_holdout.game_ids != expected_ids[: target_holdout.game_count]:
        raise Phase4F17StagedError(
            "partial derived holdout is not a prefix of deterministic selection"
        )
    for game in kept_games[target_holdout.game_count :]:
        target_holdout.append(game)
    target_holdout.verify(deep=True)

    source_strata = source_manifest.get("strata")
    if not isinstance(source_strata, Mapping):
        raise Phase4F17StagedError("source teacher strata are invalid")
    derived_strata: dict[str, dict[str, int]] = {}
    for category, raw_values in source_strata.items():
        if not isinstance(category, str) or not isinstance(raw_values, Mapping):
            raise Phase4F17StagedError("source teacher strata are invalid")
        derived_strata[category] = {
            str(key): int(value) - dropped_strata[category].get(str(key), 0)
            for key, value in raw_values.items()
        }
        if any(value < 0 for value in derived_strata[category].values()):
            raise Phase4F17StagedError("derived teacher strata underflow")

    train_manifest_path = target_train / "manifest.json"
    holdout_manifest_path = target_root / "holdout/manifest.json"
    train_manifest_sha256 = sha256_file(train_manifest_path)
    holdout_manifest_sha256 = sha256_file(holdout_manifest_path)
    fingerprint_core = {
        "schemaVersion": 1,
        "algorithm": "phase4f17-whole-game-unique-position-holdout-v1",
        "sourceDatasetFingerprint": source_manifest.get("datasetFingerprint"),
        "trainManifestSha256": train_manifest_sha256,
        "holdoutManifestSha256": holdout_manifest_sha256,
        "keptHoldoutGameIds": list(expected_ids),
    }
    dataset_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_core))
    source_actual = source_manifest.get("actual")
    if not isinstance(source_actual, Mapping):
        raise Phase4F17StagedError("source teacher totals are invalid")
    actual = {
        "trainGames": source_actual.get("trainGames"),
        "trainPositions": source_actual.get("trainPositions"),
        "holdoutGames": target_holdout.game_count,
        "holdoutPositions": target_holdout.ply_count,
        "totalPositions": int(source_actual["trainPositions"])
        + target_holdout.ply_count,
    }
    manifest = _self_hashed(
        {
            "schemaVersion": 1,
            "kind": "phase4f17-derived-classic-hard-dataset",
            "status": "completed",
            "datasetFingerprint": dataset_fingerprint,
            "sourceDataset": {
                "root": SOURCE_DATASET_ROOT.as_posix(),
                "datasetFingerprint": source_manifest.get("datasetFingerprint"),
                "datasetManifestSha256": sha256_file(
                    source_root / "teacher-manifest.json"
                ),
            },
            "derivation": {
                **fingerprint_core,
                "droppedHoldoutGames": dropped_games,
                "droppedHoldoutPositions": sum(
                    int(item["positions"]) for item in dropped_games
                ),
            },
            "actual": actual,
            "strata": derived_strata,
            "stores": {
                "train": {
                    "path": "train",
                    "games": actual["trainGames"],
                    "positions": actual["trainPositions"],
                    "manifestSha256": train_manifest_sha256,
                },
                "holdout": {
                    "path": "holdout",
                    "games": actual["holdoutGames"],
                    "positions": actual["holdoutPositions"],
                    "manifestSha256": holdout_manifest_sha256,
                },
            },
        },
        "manifestSha256",
    )
    atomic_replace_bytes(
        target_root / "teacher-manifest.json",
        canonical_json_bytes(manifest),
    )
    return manifest


def _protocol(contract: Mapping[str, object]) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f17-two-stage-classic-guided-recovery-protocol",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingAuthorized": False,
            "rewardContract": {
                "terminalWin": 1.0,
                "terminalLoss": -1.0,
                "drawReward": None,
                "classicHardRewardShaping": False,
                "classicHardUse": "supervised-policy-and-value-replay-only",
            },
            "datasetContract": {
                "path": CONTRACT_PATH.as_posix(),
                "contractSha256": contract["contractSha256"],
                "datasetFingerprint": contract["datasetFingerprint"],
            },
            "stageA": {
                "name": "classic-hard-guided",
                "initialization": {
                    "source": "recovery-v2-generation-11-archive",
                    "optimizer": "empty",
                    "selfPlayReplay": "empty",
                },
                "challengeReplayFraction": 0.4,
                "learningRate": 0.0001,
                "minimumLearningRate": 0.00001,
                "maximumGenerationsPerMicroBlock": 2,
                "selection": "strict-minimum-fixed-health-holdout-total-loss",
            },
            "stageB": {
                "name": "pure-self-play",
                "initialization": {
                    "source": "best-stage-a-fixed-health-candidate",
                    "optimizer": "empty",
                    "selfPlayReplay": "empty",
                },
                "challengeReplayFraction": 0.0,
                "classicHardOptimizationWeight": 0.0,
                "fixedClassicHoldoutDiagnosticOnly": True,
                "maximumGenerationsPerMicroBlock": 2,
                "selection": (
                    "latest-pure-self-play-generation-pending-external-screen"
                ),
            },
            "transitionGate": {
                "requiresSeparateUserConfirmation": True,
                "requiresStageABestArtifact": True,
                "resetsOptimizerAndReplay": True,
                "forbidsStageAReplayImport": True,
            },
            "screenPolicy": {
                "opponent": "classic-hard-matched-workload",
                "openingPairs": 32,
                "games": 64,
                "belowWinRate": 0.35,
                "continueTrainingFromWinRate": 0.35,
                "confirmationFromWinRate": 0.5,
                "screenTrajectoriesTrainingUseAllowed": False,
            },
        },
        "protocolSha256",
    )


def prepare_phase4f17_staged(
    workspace_root: str | Path,
    *,
    progress_callback=None,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    _prepare_derived_dataset(workspace)
    suites, suite_orbits = _suite_evidence(workspace)
    arenas, arena_orbits = _arena_evidence(workspace)
    excluded = suite_orbits | arena_orbits
    dataset = _dataset_evidence(workspace)
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f-recovery-challenge-replay-contract",
            "revision": "phase4f17-all-prior-arena-isolation-v2",
            "status": "completed",
            "cpuOnly": True,
            "trainingUseAllowed": True,
            "datasetRoot": DATASET_ROOT.as_posix(),
            "datasetFingerprint": dataset["datasetFingerprint"],
            "datasetManifestSha256": dataset["datasetManifestSha256"],
            "trainManifestSha256": dataset["trainManifestSha256"],
            "datasetEvidence": dataset,
            "excludedSuites": suites,
            "excludedTrajectoryEvidence": arenas,
            "exclusionSet": {
                "algorithm": "sha256-sorted-horizontal-position-orbits-v1",
                "orbits": len(excluded),
                "sha256": sha256_bytes(canonical_json_bytes(sorted(excluded))),
            },
            "reuseAudit": {
                "rawDatasetRegenerated": False,
                "reason": "existing classic-hard data passed the expanded orbit audit",
                "allPriorArenaResults": len(arenas),
                "playerRoleBalance": dataset["strata"],
            },
        },
        "contractSha256",
    )

    staging_path = workspace / STAGING_CONTRACT_PATH
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(staging_path, canonical_json_bytes(contract))
    settings = HealthHoldoutSettings(
        root=DATASET_ROOT,
        contract_path=STAGING_CONTRACT_PATH,
        contract_sha256=str(contract["contractSha256"]),
        dataset_fingerprint=str(dataset["datasetFingerprint"]),
        dataset_manifest_sha256=str(dataset["datasetManifestSha256"]),
        train_manifest_sha256=str(dataset["trainManifestSha256"]),
        holdout_manifest_sha256=str(dataset["holdoutManifestSha256"]),
    )
    try:
        bound = open_bound_health_holdout(
            workspace_root=workspace,
            settings=settings,
            progress_callback=progress_callback,
        )
    finally:
        staging_path.unlink(missing_ok=True)

    contract_path = workspace / CONTRACT_PATH
    protocol_path = workspace / PROTOCOL_PATH
    atomic_replace_bytes(contract_path, canonical_json_bytes(contract))
    protocol = _protocol(contract)
    atomic_replace_bytes(protocol_path, canonical_json_bytes(protocol))
    source = fingerprint_sources(workspace).to_payload()
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f17-staged-recovery-cpu-preparation",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingAuthorized": False,
            "source": source,
            "datasetAudit": {
                "evaluationOrbits": len(arena_orbits),
                "excludedOrbits": bound.excluded_orbits,
                "trainOrbits": bound.train_orbits,
                "holdoutOrbits": bound.holdout_orbits,
                "trainHoldoutOverlap": 0,
                "evaluationOverlap": 0,
            },
            "datasetContract": {
                "path": CONTRACT_PATH.as_posix(),
                "fileSha256": sha256_file(contract_path),
                "contractSha256": contract["contractSha256"],
            },
            "twoStageProtocol": {
                "path": PROTOCOL_PATH.as_posix(),
                "fileSha256": sha256_file(protocol_path),
                "protocolSha256": protocol["protocolSha256"],
            },
            "nextStage": {
                "name": "Phase 4F18 source rebaseline and CUDA pilot preparation",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "reportSha256",
    )
    report_path = workspace / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(report_path, canonical_json_bytes(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    def publish(payload: Mapping[str, object]) -> None:
        print(canonical_json_bytes(dict(payload)).decode("ascii"), flush=True)

    report = prepare_phase4f17_staged(
        args.workspace_root,
        progress_callback=publish,
    )
    print(canonical_json_bytes(report).decode("ascii"), flush=True)
    return 0


__all__ = [
    "ARENA_RESULT_PATHS",
    "CONTRACT_PATH",
    "PROTOCOL_PATH",
    "Phase4F17StagedError",
    "prepare_phase4f17_staged",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
