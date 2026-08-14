"""CPU-only preparation contracts for Phase 4F strength recovery."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from .champion_store import ChampionStore
from .challenge_replay import arena_position_orbits
from .openings import load_opening_suite, validate_suite_collection
from .phase4_training import canonical_model_state_sha256
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size
from .teacher_data import evaluation_position_orbit_digest, exclusion_set_sha256


RECOVERY_SCHEMA_VERSION = 1
CHALLENGE_CONTRACT = Path(
    "training/configs/phase4f-recovery-challenge-prepared-v1.json"
)
WORKLOAD_DRAFT = Path("training/configs/phase4f-recovery-workload-prepared-v1.json")
REPORT_PATH = Path("training/runs/phase4/phase4f/recovery-preparation/report.json")
CONTROL_ROOT = Path("training/runs/phase4")
SUITE_PATHS = (
    Path("training/fixtures/openings/dev.json"),
    Path("training/fixtures/openings/final.json"),
    Path("training/fixtures/openings/recovery-v1/recovery-dev-v1.json"),
    Path("training/fixtures/openings/recovery-v1/recovery-final-v1.json"),
)
TRAJECTORY_RESULT_PATHS = (
    Path("training/runs/phase4/phase4f/fixed/result.json"),
    Path("training/runs/phase4/phase4f/fixed-standard-initial/result.json"),
)


class Phase4FRecoveryError(RuntimeError):
    """Raised when recovery preparation evidence is inconsistent."""


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4FRecoveryError(f"path is outside workspace: {path}") from error


def _global_gpu_hours(ledger: Mapping[str, object]) -> float:
    runs = ledger.get("runs")
    floor = ledger.get("externalHoursFloor")
    if not isinstance(runs, Mapping) or not isinstance(floor, (int, float)):
        raise Phase4FRecoveryError("GPU ledger is invalid")
    seconds = 0.0
    for record in runs.values():
        if not isinstance(record, Mapping):
            raise Phase4FRecoveryError("GPU ledger run is invalid")
        value = record.get("accountedSeconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise Phase4FRecoveryError("GPU ledger accounted seconds are invalid")
        seconds += float(value)
    return max(float(floor), seconds / 3600.0)


def _suite_references(
    workspace: Path,
) -> tuple[list[dict[str, object]], set[str]]:
    suites = [load_opening_suite(workspace / path) for path in SUITE_PATHS]
    validate_suite_collection(*suites)
    references: list[dict[str, object]] = []
    orbits: set[str] = set()
    for suite, relative_path in zip(suites, SUITE_PATHS, strict=True):
        path = workspace / relative_path
        suite_orbits = {
            evaluation_position_orbit_digest(opening.state)
            for opening in suite.openings
        }
        orbits.update(suite_orbits)
        references.append(
            {
                "name": suite.name,
                "path": relative_path.as_posix(),
                "fileSha256": sha256_file(path),
                "suiteHash": suite.content_hash,
                "openings": len(suite.openings),
                "positionOrbits": len(suite_orbits),
            }
        )
    if len(orbits) != sum(int(item["positionOrbits"]) for item in references):
        raise Phase4FRecoveryError("evaluation suites share a position mirror orbit")
    return references, orbits


def _trajectory_references(
    workspace: Path,
) -> tuple[list[dict[str, object]], set[str]]:
    references: list[dict[str, object]] = []
    orbits: set[str] = set()
    for relative_path in TRAJECTORY_RESULT_PATHS:
        path = workspace / relative_path
        payload = json.loads(path.read_bytes())
        reference = {
            "path": relative_path.as_posix(),
            "fileSha256": sha256_file(path),
            "resultSha256": payload.get("resultSha256"),
        }
        result_orbits = arena_position_orbits(workspace, reference)
        references.append(
            {
                **reference,
                "games": payload.get("games"),
                "positionOrbits": len(result_orbits),
            }
        )
        orbits.update(result_orbits)
    return references, orbits


def _champion_initialization(workspace: Path) -> dict[str, object]:
    control_root = workspace / CONTROL_ROOT
    store = ChampionStore(control_root / "champions", artifact_root=control_root)
    verification = store.verify(deep=True)
    record = store.resolve(role="latest")
    if not record.retained or record.file_name is None:
        raise Phase4FRecoveryError("latest published champion is not retained")
    state = store.load_state_dict(record.champion_id, map_location="cpu")
    artifact_path = store.root / record.file_name
    manifest_path = store.manifest_path
    manifest = json.loads(manifest_path.read_bytes())
    return {
        "mode": "champion",
        "expectedBaseModelSha256": (
            "51e500bbc612222bdc4902438af9dbfcf367c95cc523d5384309403e8b3ba010"
        ),
        "artifactPath": _relative(artifact_path, workspace),
        "artifactSha256": record.model_sha256,
        "championId": record.champion_id,
        "championManifestPath": _relative(manifest_path, workspace),
        "championManifestSha256": manifest["manifestSha256"],
        "championManifestFileSha256": sha256_file(manifest_path),
        "modelStateSha256": canonical_model_state_sha256(state),
        "resetOptimizer": True,
        "deepVerification": {
            "accepted": verification.accepted_count,
            "retained": verification.retained_count,
            "committedBytes": verification.committed_bytes,
        },
    }


def prepare_phase4f_recovery(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = workspace / CONTROL_ROOT
    if shutil.disk_usage(workspace).free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4FRecoveryError("less than 20 GiB of free disk remains")
    if directory_size(control_root) > DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4FRecoveryError("Phase 4 artifacts exceed 8 GiB")

    suite_references, suite_orbits = _suite_references(workspace)
    trajectory_references, trajectory_orbits = _trajectory_references(workspace)
    excluded_orbits = suite_orbits | trajectory_orbits
    exclusion_sha256 = exclusion_set_sha256(tuple(sorted(excluded_orbits)))
    initialization = _champion_initialization(workspace)
    ledger_path = control_root / "gpu-budget-ledger.json"
    ledger = json.loads(ledger_path.read_bytes())
    gpu_hours = _global_gpu_hours(ledger)

    challenge_core = {
        "schemaVersion": RECOVERY_SCHEMA_VERSION,
        "kind": "phase4f-recovery-challenge-replay-contract",
        "status": "prepared",
        "cpuOnly": True,
        "trainingUseAllowed": False,
        "datasetRoot": "training/runs/phase4/phase4f/recovery-challenge-data",
        "generator": {
            "script": "scripts/generate-phase4f-challenge-data.sh",
            "targetPositions": 30_000,
            "seed": 20260802,
            "nodeBudget": 587,
            "maxDepth": 5,
            "explorationProbability": 0.10,
            "openingDepths": [0, 4, 8, 12, 16],
        },
        "excludedSuites": suite_references,
        "excludedTrajectoryEvidence": trajectory_references,
        "exclusionSet": {
            "algorithm": "sha256-sorted-horizontal-position-orbits-v1",
            "sha256": exclusion_sha256,
            "orbits": len(excluded_orbits),
        },
        "completionRequirements": [
            "completed canonical teacher manifest and train replay manifest",
            "deep replay reconstruction with exclusion-orbit recomputation",
            "dataset, manifest, train-manifest, and contract SHA-256 binding",
        ],
    }
    challenge = _self_hashed(challenge_core, "contractSha256")

    workload_core = {
        "schemaVersion": RECOVERY_SCHEMA_VERSION,
        "kind": "phase4f-recovery-workload-draft",
        "status": "prepared",
        "executable": False,
        "trainingAuthorized": False,
        "reason": "challenge replay and post-change CPU rebaseline are incomplete",
        "runDir": "training/runs/phase4/phase4f/recovery-v1",
        "initialization": initialization,
        "network": {"inputChannels": 16, "channels": 192, "residualBlocks": 16},
        "selfPlay": {"games": 128, "actors": 64, "simulations": 512},
        "training": {
            "batchSize": 512,
            "trainingStepsPerGeneration": 96,
            "challengeReplayFraction": 0.25,
            "optimizer": "reset",
        },
        "evaluation": {
            "suite": "recovery-dev-v1",
            "screenGames": 64,
            "confirmationGames": 400,
            "simulations": 800,
        },
        "budgetEnvelopeGpuHours": {
            "recoveryTrainingMaximum": 14.0,
            "developmentScreensMaximum": 3.0,
            "reservedForFreshFinalGate": 9.5,
        },
        "challengeContract": {
            "path": CHALLENGE_CONTRACT.as_posix(),
            "contractSha256": challenge["contractSha256"],
            "requiredStatus": "completed",
        },
    }
    workload = _self_hashed(workload_core, "workloadSha256")

    challenge_path = workspace / CHALLENGE_CONTRACT
    workload_path = workspace / WORKLOAD_DRAFT
    challenge_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(challenge_path, canonical_json_bytes(challenge))
    atomic_replace_bytes(workload_path, canonical_json_bytes(workload))

    report_core = {
        "schemaVersion": RECOVERY_SCHEMA_VERSION,
        "kind": "phase4f-recovery-preparation",
        "status": "completed",
        "cpuOnly": True,
        "gpuAccessed": False,
        "trainingAuthorized": False,
        "source": fingerprint_sources(workspace).to_payload(),
        "championInitialization": initialization,
        "openingIsolation": {
            "suites": suite_references,
            "historicalTrajectories": trajectory_references,
            "exclusionSetSha256": exclusion_sha256,
            "contentAndMirrorOrbitDisjoint": True,
        },
        "challengeContract": {
            "path": CHALLENGE_CONTRACT.as_posix(),
            "fileSha256": sha256_file(challenge_path),
            "contractSha256": challenge["contractSha256"],
            "status": "prepared",
        },
        "workloadDraft": {
            "path": WORKLOAD_DRAFT.as_posix(),
            "fileSha256": sha256_file(workload_path),
            "workloadSha256": workload["workloadSha256"],
        },
        "resources": {
            "globalGpuHoursBeforeAndAfter": gpu_hours,
            "gpuLedgerFileSha256": sha256_file(ledger_path),
            "artifactBytes": directory_size(control_root),
            "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
            "freeDiskBytes": shutil.disk_usage(workspace).free,
            "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        },
        "nextStage": {
            "name": "Phase 4F3 challenge-data generation",
            "requiresConfirmation": True,
            "command": "./scripts/generate-phase4f-challenge-data.sh",
        },
    }
    report = _self_hashed(report_core, "reportSha256")
    report_path = workspace / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(report_path, canonical_json_bytes(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    report = prepare_phase4f_recovery(args.workspace_root)
    print(canonical_json_bytes(report).decode("ascii"), flush=True)
    return 0


__all__ = [
    "Phase4FRecoveryError",
    "prepare_phase4f_recovery",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
