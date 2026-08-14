"""Freeze completed Phase 4F challenge replay and prepare its CUDA rebaseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .challenge_replay import (
    BoundChallengeReplay,
    arena_position_orbits,
    open_bound_challenge_replay,
    position_orbit_digest,
)
from .config import ChallengeReplaySettings, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_training import validate_phase4_config
from .phase4f_recovery import _champion_initialization
from .replay_store import ReplayGame, ReplayStore, ReplayVerification
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
CONTROL_ROOT = Path("training/runs/phase4")
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-preparation/report.json"
)
PREPARED_CHALLENGE_CONTRACT = Path(
    "training/configs/phase4f-recovery-challenge-prepared-v1.json"
)
FROZEN_CHALLENGE_CONTRACT = Path(
    "training/configs/phase4f-recovery-challenge-frozen-v1.json"
)
RECOVERY_CONFIG = Path("training/configs/phase4f-recovery-v1.json")
REBASELINE_PREPARATION = Path(
    "training/configs/phase4f-recovery-rebaseline-prepared-v1.json"
)
FREEZE_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-freeze/report.json"
)


class Phase4FRecoveryFreezeError(RuntimeError):
    """Raised when recovery artifacts cannot be safely frozen."""


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4FRecoveryFreezeError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4FRecoveryFreezeError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4FRecoveryFreezeError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    stored = payload.get(field)
    core = dict(payload)
    core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(
        canonical_json_bytes(core)
    ):
        raise Phase4FRecoveryFreezeError(f"{label} self hash mismatch")
    return stored


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4FRecoveryFreezeError("artifact path escapes the workspace") from error


def _write_immutable(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4FRecoveryFreezeError(
                f"immutable recovery artifact already differs: {path}"
            )
        return
    atomic_create_bytes(path, encoded)


def _global_gpu_hours(ledger: Mapping[str, object]) -> float:
    runs = ledger.get("runs")
    floor = ledger.get("externalHoursFloor")
    if not isinstance(runs, Mapping) or isinstance(floor, bool) or not isinstance(
        floor, (int, float)
    ):
        raise Phase4FRecoveryFreezeError("GPU ledger is invalid")
    seconds = 0.0
    for run in runs.values():
        if not isinstance(run, Mapping) or run.get("status") == "running":
            raise Phase4FRecoveryFreezeError(
                "a running or invalid GPU ledger entry blocks CPU freeze"
            )
        value = run.get("accountedSeconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise Phase4FRecoveryFreezeError("GPU ledger usage is invalid")
        seconds += float(value)
    return max(float(floor), seconds / 3600.0)


def _resource_preflight(workspace: Path, control_root: Path) -> dict[str, int]:
    artifact_bytes = directory_size(control_root)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4FRecoveryFreezeError("Phase 4 artifacts exceed the 8 GiB cap")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4FRecoveryFreezeError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _prepared_evidence(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    report_path = workspace / PREPARATION_REPORT
    contract_path = workspace / PREPARED_CHALLENGE_CONTRACT
    report = _load_canonical(report_path, "Phase 4F recovery preparation report")
    report_sha256 = _verify_self_hash(
        report, "reportSha256", "Phase 4F recovery preparation report"
    )
    contract = _load_canonical(contract_path, "prepared challenge contract")
    contract_sha256 = _verify_self_hash(
        contract, "contractSha256", "prepared challenge contract"
    )
    reference = report.get("challengeContract")
    if (
        report.get("schemaVersion") != SCHEMA_VERSION
        or report.get("kind") != "phase4f-recovery-preparation"
        or report.get("status") != "completed"
        or report.get("cpuOnly") is not True
        or report.get("gpuAccessed") is not False
        or report.get("trainingAuthorized") is not False
        or contract.get("schemaVersion") != SCHEMA_VERSION
        or contract.get("kind")
        != "phase4f-recovery-challenge-replay-contract"
        or contract.get("status") != "prepared"
        or contract.get("trainingUseAllowed") is not False
        or not isinstance(reference, Mapping)
        or reference.get("path") != PREPARED_CHALLENGE_CONTRACT.as_posix()
        or reference.get("fileSha256") != sha256_file(contract_path)
        or reference.get("contractSha256") != contract_sha256
        or reference.get("status") != "prepared"
    ):
        raise Phase4FRecoveryFreezeError("prepared recovery evidence is inconsistent")
    resources = report.get("resources")
    ledger_path = workspace / CONTROL_ROOT / "gpu-budget-ledger.json"
    if (
        not isinstance(resources, Mapping)
        or resources.get("gpuLedgerFileSha256") != sha256_file(ledger_path)
    ):
        raise Phase4FRecoveryFreezeError(
            "GPU ledger changed after CPU recovery preparation"
        )
    return report, contract


def _dataset_evidence(
    workspace: Path,
    prepared_report: Mapping[str, object],
    prepared_contract: Mapping[str, object],
) -> dict[str, object]:
    dataset_root_value = prepared_contract.get("datasetRoot")
    if not isinstance(dataset_root_value, str):
        raise Phase4FRecoveryFreezeError("prepared dataset root is invalid")
    dataset_root = (workspace / dataset_root_value).resolve()
    try:
        dataset_root.relative_to(workspace)
    except ValueError as error:
        raise Phase4FRecoveryFreezeError("challenge dataset root escapes workspace") from error
    manifest_path = dataset_root / "teacher-manifest.json"
    run_manifest_path = dataset_root / "run-manifest.json"
    train_manifest_path = dataset_root / "train" / "manifest.json"
    holdout_manifest_path = dataset_root / "holdout" / "manifest.json"
    manifest = _load_canonical(manifest_path, "challenge teacher manifest")
    manifest_self_hash = _verify_self_hash(
        manifest, "manifestSha256", "challenge teacher manifest"
    )
    run_record = load_run_manifest(run_manifest_path)
    if run_manifest_path.read_bytes() != canonical_json_bytes(run_record.payload):
        raise Phase4FRecoveryFreezeError("challenge run manifest is not canonical")
    provenance = run_record.payload.get("provenance")
    config = provenance.get("config") if isinstance(provenance, Mapping) else None
    config_payload = config.get("payload") if isinstance(config, Mapping) else None
    authorization = (
        config_payload.get("recoveryAuthorization")
        if isinstance(config_payload, Mapping)
        else None
    )
    stores = manifest.get("stores")
    actual = manifest.get("actual")
    targets = manifest.get("targets")
    generator = prepared_contract.get("generator")
    dataset_fingerprint = manifest.get("datasetFingerprint")
    if (
        manifest.get("status") != "completed"
        or not isinstance(dataset_fingerprint, str)
        or manifest.get("contentSha256") != dataset_fingerprint
        or manifest.get("runManifestSha256") != run_record.sha256
        or not isinstance(config, Mapping)
        or manifest.get("configSha256") != config.get("sha256")
        or not isinstance(authorization, Mapping)
        or authorization.get("kind") != "phase4f2-cpu-preparation-gate-v1"
        or authorization.get("reportSha256") != prepared_report.get("reportSha256")
        or authorization.get("challengeContractSha256")
        != prepared_contract.get("contractSha256")
        or not isinstance(stores, Mapping)
        or not isinstance(actual, Mapping)
        or not isinstance(targets, Mapping)
        or not isinstance(generator, Mapping)
        or targets.get("totalPositions") != generator.get("targetPositions")
        or actual.get("totalPositions")
        != int(actual.get("trainPositions", -1))
        + int(actual.get("holdoutPositions", -1))
        or actual.get("uniqueRawStates") != actual.get("totalPositions")
        or actual.get("uniqueAugmentationOrbits") != actual.get("totalPositions")
    ):
        raise Phase4FRecoveryFreezeError("completed challenge dataset is inconsistent")
    for split, path in (
        ("train", train_manifest_path),
        ("holdout", holdout_manifest_path),
    ):
        reference = stores.get(split)
        if (
            not isinstance(reference, Mapping)
            or reference.get("manifestSha256") != sha256_file(path)
        ):
            raise Phase4FRecoveryFreezeError(
                f"challenge {split} replay manifest binding is invalid"
            )
        _load_canonical(path, f"challenge {split} replay manifest")
    source = provenance.get("source") if isinstance(provenance, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str):
        raise Phase4FRecoveryFreezeError("challenge generation source is invalid")
    return {
        "root": dataset_root_value,
        "datasetFingerprint": dataset_fingerprint,
        "datasetManifestPath": _relative(manifest_path, workspace),
        "datasetManifestSha256": sha256_file(manifest_path),
        "datasetManifestSelfSha256": manifest_self_hash,
        "runManifestPath": _relative(run_manifest_path, workspace),
        "runManifestSha256": run_record.sha256,
        "generationSourceSha256": source["sha256"],
        "configSha256": manifest["configSha256"],
        "trainManifestPath": _relative(train_manifest_path, workspace),
        "trainManifestSha256": sha256_file(train_manifest_path),
        "holdoutManifestPath": _relative(holdout_manifest_path, workspace),
        "holdoutManifestSha256": sha256_file(holdout_manifest_path),
        "positions": {
            "total": actual["totalPositions"],
            "train": actual["trainPositions"],
            "holdout": actual["holdoutPositions"],
        },
        "games": {
            "train": actual["trainGames"],
            "holdout": actual["holdoutGames"],
        },
    }


def _completed_contract(
    prepared_contract: Mapping[str, object],
    prepared_report: Mapping[str, object],
    dataset: Mapping[str, object],
    workspace: Path,
) -> dict[str, object]:
    core = dict(prepared_contract)
    core.pop("contractSha256", None)
    core["status"] = "completed"
    core["trainingUseAllowed"] = True
    core.update(
        {
            "preparedContract": {
                "path": PREPARED_CHALLENGE_CONTRACT.as_posix(),
                "fileSha256": sha256_file(workspace / PREPARED_CHALLENGE_CONTRACT),
                "contractSha256": prepared_contract["contractSha256"],
            },
            "preparationReport": {
                "path": PREPARATION_REPORT.as_posix(),
                "fileSha256": sha256_file(workspace / PREPARATION_REPORT),
                "reportSha256": prepared_report["reportSha256"],
            },
            "datasetFingerprint": dataset["datasetFingerprint"],
            "datasetManifestSha256": dataset["datasetManifestSha256"],
            "trainManifestSha256": dataset["trainManifestSha256"],
            "datasetEvidence": dict(dataset),
        }
    )
    return _self_hashed(core, "contractSha256")


def _challenge_settings(
    workspace: Path,
    contract_path: Path,
    contract: Mapping[str, object],
    dataset: Mapping[str, object],
) -> ChallengeReplaySettings:
    return ChallengeReplaySettings(
        root=workspace / str(dataset["root"]),
        contract_path=contract_path,
        contract_sha256=str(contract["contractSha256"]),
        dataset_fingerprint=str(dataset["datasetFingerprint"]),
        dataset_manifest_sha256=str(dataset["datasetManifestSha256"]),
        train_manifest_sha256=str(dataset["trainManifestSha256"]),
        fraction=0.25,
    )


def _verify_holdout(
    workspace: Path,
    contract: Mapping[str, object],
    dataset: Mapping[str, object],
) -> ReplayVerification:
    excluded: set[str] = set()
    suites = contract.get("excludedSuites")
    trajectories = contract.get("excludedTrajectoryEvidence")
    if not isinstance(suites, list) or not isinstance(trajectories, list):
        raise Phase4FRecoveryFreezeError("challenge exclusion references are invalid")
    from .openings import load_opening_suite

    for reference in suites:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise Phase4FRecoveryFreezeError("excluded suite reference is invalid")
        suite = load_opening_suite(workspace / str(reference["path"]))
        excluded.update(position_orbit_digest(item.state) for item in suite.openings)
    for reference in trajectories:
        if not isinstance(reference, Mapping):
            raise Phase4FRecoveryFreezeError("excluded trajectory reference is invalid")
        excluded.update(arena_position_orbits(workspace, reference))
    exclusion = contract.get("exclusionSet")
    if (
        not isinstance(exclusion, Mapping)
        or exclusion.get("orbits") != len(excluded)
        or exclusion.get("sha256")
        != sha256_bytes(canonical_json_bytes(sorted(excluded)))
    ):
        raise Phase4FRecoveryFreezeError("challenge exclusion set changed")
    root = workspace / str(dataset["root"]) / "holdout"
    manifest = _load_canonical(root / "manifest.json", "holdout replay manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4FRecoveryFreezeError("holdout replay capacity is invalid")
    try:
        store = ReplayStore(
            root,
            max_games=int(capacity["games"]),
            max_plies=int(capacity["plies"]),
            max_bytes=int(capacity["bytes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4FRecoveryFreezeError("holdout replay capacity is invalid") from error

    def validate_game(game: ReplayGame, states: Sequence[object]) -> None:
        if game.metadata.get("teacherEngine") != "classic-negamax":
            raise Phase4FRecoveryFreezeError(
                "holdout replay is not classic-hard teacher data"
            )
        if any(position_orbit_digest(state) in excluded for state in states[:-1]):
            raise Phase4FRecoveryFreezeError(
                "holdout replay intersects an evaluation orbit"
            )

    return store.verify(deep=True, game_validator=validate_game)


def _deep_verify_candidate(
    workspace: Path,
    contract: Mapping[str, object],
    dataset: Mapping[str, object],
) -> tuple[BoundChallengeReplay, ReplayVerification]:
    final_path = workspace / FROZEN_CHALLENGE_CONTRACT
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".verify.json", dir=final_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(contract))
            stream.flush()
            os.fsync(stream.fileno())
        bound = open_bound_challenge_replay(
            workspace_root=workspace,
            settings=_challenge_settings(workspace, temporary, contract, dataset),
        )
        holdout = _verify_holdout(workspace, contract, dataset)
        return bound, holdout
    finally:
        temporary.unlink(missing_ok=True)


def _recovery_config(
    initialization: Mapping[str, object],
    contract: Mapping[str, object],
    dataset: Mapping[str, object],
) -> dict[str, object]:
    champion = {
        key: value
        for key, value in initialization.items()
        if key != "deepVerification"
    }
    return {
        "pipeline": "phase4",
        "name": "phase4f-recovery-v1",
        "seed": 20260803,
        "device": "cuda",
        "runDir": "training/runs/phase4/phase4f/recovery-v1",
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "initialization": champion,
        "network": {
            "inputChannels": 16,
            "channels": 192,
            "residualBlocks": 16,
            "seReduction": 8,
            "policyChannels": 32,
            "valueChannels": 32,
            "valueHidden": 256,
        },
        "mcts": {
            "simulations": 512,
            "cBase": 19652,
            "cInit": 1.25,
            "dirichletEpsilon": 0.25,
            "dirichletAlpha": None,
        },
        "selfPlay": {"games": 128, "temperatureMoves": 24, "actors": 64},
        "training": {
            "replayCapacity": 300000,
            "batchSize": 512,
            "learningRate": 0.0003,
            "minimumLearningRate": 0.00003,
            "scheduleSteps": 20000,
            "weightDecay": 0.0001,
            "maxGradNorm": 5.0,
            "useAmp": True,
            "trainingSteps": 96,
            "maxCheckpoints": 2,
            "challengeReplay": {
                "root": dataset["root"],
                "contractPath": FROZEN_CHALLENGE_CONTRACT.as_posix(),
                "contractSha256": contract["contractSha256"],
                "datasetFingerprint": dataset["datasetFingerprint"],
                "datasetManifestSha256": dataset["datasetManifestSha256"],
                "trainManifestSha256": dataset["trainManifestSha256"],
                "fraction": 0.25,
            },
        },
        "arena": {"games": 64, "simulations": 800},
    }


def _verification_payload(value: ReplayVerification) -> dict[str, object]:
    return {
        "games": value.game_count,
        "plies": value.ply_count,
        "committedBytes": value.committed_bytes,
        "deep": value.deep,
    }


def freeze_phase4f_recovery(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = workspace / CONTROL_ROOT
    _resource_preflight(workspace, control_root)
    prepared_report, prepared_contract = _prepared_evidence(workspace)
    dataset = _dataset_evidence(
        workspace, prepared_report, prepared_contract
    )
    completed_contract = _completed_contract(
        prepared_contract, prepared_report, dataset, workspace
    )
    bound, holdout = _deep_verify_candidate(
        workspace, completed_contract, dataset
    )
    if (
        bound.verification.game_count != dataset["games"]["train"]
        or bound.verification.ply_count != dataset["positions"]["train"]
        or holdout.game_count != dataset["games"]["holdout"]
        or holdout.ply_count != dataset["positions"]["holdout"]
    ):
        raise Phase4FRecoveryFreezeError("deep replay totals differ from manifest")
    contract_path = workspace / FROZEN_CHALLENGE_CONTRACT
    _write_immutable(contract_path, completed_contract)

    initialization = _champion_initialization(workspace)
    config_payload = _recovery_config(initialization, completed_contract, dataset)
    config_path = workspace / RECOVERY_CONFIG
    _write_immutable(config_path, config_payload)
    config = load_experiment_config(config_path)
    validate_phase4_config(config)
    semantic_config_sha256 = fingerprint_config(config).sha256
    native_sha256 = loaded_native_backend_hash()
    source = fingerprint_sources(workspace).to_payload()
    rebaseline_core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f-recovery-rebaseline-preparation",
        "status": "prepared",
        "cpuOnly": True,
        "cudaAccessAllowed": False,
        "trainingAuthorized": False,
        "reason": "current-source CUDA memory, mixed-input, and one-generation gates are incomplete",
        "source": source,
        "nativeBackendSha256": native_sha256,
        "workload": {
            "path": RECOVERY_CONFIG.as_posix(),
            "fileSha256": sha256_file(config_path),
            "semanticConfigSha256": semantic_config_sha256,
        },
        "challengeReplay": {
            "contractPath": FROZEN_CHALLENGE_CONTRACT.as_posix(),
            "contractFileSha256": sha256_file(contract_path),
            "contractSha256": completed_contract["contractSha256"],
            "datasetFingerprint": dataset["datasetFingerprint"],
        },
        "requiredNextStageGates": [
            "current-source native and Python correctness",
            "192x16 batch-512 CUDA AMP memory smoke",
            "mixed self-play/challenge replay input throughput",
            "resumable one-generation recovery pilot with zero illegal actions",
            "GPU ledger and Phase 4E remaining-budget freeze",
        ],
        "budgetEnvelopeGpuHours": {
            "recoveryTrainingMaximum": 14.0,
            "developmentScreensMaximum": 3.0,
            "reservedForFreshFinalGate": 9.5,
        },
    }
    rebaseline = _self_hashed(rebaseline_core, "rebaselineSha256")
    rebaseline_path = workspace / REBASELINE_PREPARATION
    _write_immutable(rebaseline_path, rebaseline)

    ledger_path = control_root / "gpu-budget-ledger.json"
    ledger = _load_canonical(ledger_path, "GPU budget ledger")
    ledger_self_hash = _verify_self_hash(ledger, "ledgerSha256", "GPU budget ledger")
    gpu_hours = _global_gpu_hours(ledger)
    resources = _resource_preflight(workspace, control_root)
    report_core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f-recovery-freeze-report",
        "status": "completed",
        "cpuOnly": True,
        "gpuAccessed": False,
        "trainingAuthorized": False,
        "source": source,
        "nativeBackendSha256": native_sha256,
        "preparedContractSha256": prepared_contract["contractSha256"],
        "challengeContract": {
            "path": FROZEN_CHALLENGE_CONTRACT.as_posix(),
            "fileSha256": sha256_file(contract_path),
            "contractSha256": completed_contract["contractSha256"],
            "status": "completed",
            "trainingUseAllowed": True,
        },
        "dataset": dict(dataset),
        "deepVerification": {
            "train": _verification_payload(bound.verification),
            "holdout": _verification_payload(holdout),
            "excludedOrbits": bound.excluded_orbits,
        },
        "workload": {
            "path": RECOVERY_CONFIG.as_posix(),
            "fileSha256": sha256_file(config_path),
            "semanticConfigSha256": semantic_config_sha256,
        },
        "rebaselinePreparation": {
            "path": REBASELINE_PREPARATION.as_posix(),
            "fileSha256": sha256_file(rebaseline_path),
            "rebaselineSha256": rebaseline["rebaselineSha256"],
        },
        "resources": {
            **resources,
            "globalGpuHoursBeforeAndAfter": gpu_hours,
            "gpuLedgerFileSha256": sha256_file(ledger_path),
            "gpuLedgerSha256": ledger_self_hash,
        },
        "nextStage": {
            "name": "Phase 4F5 CUDA recovery rebaseline",
            "requiresConfirmation": True,
            "trainingStarts": False,
        },
    }
    report = _self_hashed(report_core, "reportSha256")
    _write_immutable(workspace / FREEZE_REPORT, report)
    return report


def verify_phase4f_recovery_freeze(
    workspace_root: str | Path,
    *,
    deep: bool = True,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    control_root = workspace / CONTROL_ROOT
    _resource_preflight(workspace, control_root)
    report = _load_canonical(workspace / FREEZE_REPORT, "recovery freeze report")
    _verify_self_hash(report, "reportSha256", "recovery freeze report")
    contract = _load_canonical(
        workspace / FROZEN_CHALLENGE_CONTRACT, "frozen challenge contract"
    )
    _verify_self_hash(contract, "contractSha256", "frozen challenge contract")
    rebaseline = _load_canonical(
        workspace / REBASELINE_PREPARATION, "rebaseline preparation"
    )
    _verify_self_hash(rebaseline, "rebaselineSha256", "rebaseline preparation")
    config_path = workspace / RECOVERY_CONFIG
    config = load_experiment_config(config_path)
    validate_phase4_config(config)
    challenge = report.get("challengeContract")
    workload = report.get("workload")
    rebaseline_reference = report.get("rebaselinePreparation")
    dataset = report.get("dataset")
    source = fingerprint_sources(workspace).to_payload()
    native_sha256 = loaded_native_backend_hash()
    if (
        report.get("schemaVersion") != SCHEMA_VERSION
        or report.get("kind") != "phase4f-recovery-freeze-report"
        or report.get("status") != "completed"
        or report.get("cpuOnly") is not True
        or report.get("gpuAccessed") is not False
        or report.get("trainingAuthorized") is not False
        or report.get("source") != source
        or report.get("nativeBackendSha256") != native_sha256
        or not isinstance(challenge, Mapping)
        or challenge.get("fileSha256")
        != sha256_file(workspace / FROZEN_CHALLENGE_CONTRACT)
        or challenge.get("contractSha256") != contract.get("contractSha256")
        or contract.get("status") != "completed"
        or contract.get("trainingUseAllowed") is not True
        or not isinstance(workload, Mapping)
        or workload.get("fileSha256") != sha256_file(config_path)
        or workload.get("semanticConfigSha256") != fingerprint_config(config).sha256
        or not isinstance(rebaseline_reference, Mapping)
        or rebaseline_reference.get("fileSha256")
        != sha256_file(workspace / REBASELINE_PREPARATION)
        or rebaseline_reference.get("rebaselineSha256")
        != rebaseline.get("rebaselineSha256")
        or rebaseline.get("source") != source
        or rebaseline.get("nativeBackendSha256") != native_sha256
        or not isinstance(dataset, Mapping)
    ):
        raise Phase4FRecoveryFreezeError("recovery freeze output binding changed")
    resources = report.get("resources")
    ledger_path = control_root / "gpu-budget-ledger.json"
    ledger = _load_canonical(ledger_path, "GPU budget ledger")
    ledger_sha256 = _verify_self_hash(ledger, "ledgerSha256", "GPU budget ledger")
    if (
        not isinstance(resources, Mapping)
        or resources.get("gpuLedgerFileSha256") != sha256_file(ledger_path)
        or resources.get("gpuLedgerSha256") != ledger_sha256
        or resources.get("globalGpuHoursBeforeAndAfter") != _global_gpu_hours(ledger)
    ):
        raise Phase4FRecoveryFreezeError("GPU ledger changed during CPU freeze")
    if deep:
        bound = open_bound_challenge_replay(
            workspace_root=workspace,
            settings=config.training.challenge_replay,
        )
        holdout = _verify_holdout(workspace, contract, dataset)
        expected = report.get("deepVerification")
        if not isinstance(expected, Mapping) or expected != {
            "train": _verification_payload(bound.verification),
            "holdout": _verification_payload(holdout),
            "excludedOrbits": bound.excluded_orbits,
        }:
            raise Phase4FRecoveryFreezeError("deep replay evidence changed")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report_path = args.workspace_root.resolve() / FREEZE_REPORT
    if args.verify_only or report_path.exists():
        report = verify_phase4f_recovery_freeze(args.workspace_root, deep=True)
    else:
        report = freeze_phase4f_recovery(args.workspace_root)
    summary = {
        "status": report["status"],
        "reportPath": FREEZE_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "challengeContractSha256": report["challengeContract"]["contractSha256"],
        "datasetFingerprint": report["dataset"]["datasetFingerprint"],
        "globalGpuHours": report["resources"]["globalGpuHoursBeforeAndAfter"],
        "trainingAuthorized": report["trainingAuthorized"],
    }
    print(canonical_json_bytes(summary).decode("ascii"), flush=True)
    return 0


__all__ = [
    "Phase4FRecoveryFreezeError",
    "freeze_phase4f_recovery",
    "verify_phase4f_recovery_freeze",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
