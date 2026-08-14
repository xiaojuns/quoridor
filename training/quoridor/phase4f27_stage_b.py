"""Freeze and review the conservative Stage B continuation block."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config, seed_everything
from .network import NetworkConfig, PolicyValueNetwork
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F_STAGE_B_CONTINUATION
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_gates import find_valid_memory_smoke
from .phase4_training import (
    load_phase4_recovery_artifact,
    reconcile_best_fixed_holdout_candidate,
    validate_phase4_config,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    GenerationReportStore,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    load_run_manifest,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
F26_GATE = Path(
    "training/runs/phase4/phase4f/stage-b-lineage-screen-v1/screen-gate.json"
)
EXPECTED_F26_GATE_SHA256 = (
    "9c86bd9858a3d9ecdd3b45dc2a0675a6d0741d2ed0a1594994c8e016ada75e76"
)
EXPECTED_F26_RESULT_SHA256 = (
    "6f7bce435fd5c567deda7bbca20f60eb0cf29b5d362c44de55c50c76149f2e92"
)
F24_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f24-stage-b-review.json"
)
EXPECTED_F24_REVIEW_SHA256 = (
    "40810a17527d8db4c698d574f4cfa4add3cda82f3fcf051859e115e95aa1281a"
)
SOURCE_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-b-block-00-v1"
)
SOURCE_CONFIG = Path("training/configs/phase4f24-stage-b-block-00-v1.json")
SOURCE_ARCHIVE = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f27-stage-b-source-archive.json"
)
CONFIG = Path("training/configs/phase4f27-stage-b-block-01-v1.json")
EXECUTION = Path("training/configs/phase4f27-stage-b-execution-frozen-v1.json")
RUN_DIR = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-b-block-01-v1"
)
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f27-stage-b-freeze-report.json"
)
REVIEW_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f27-stage-b-review.json"
)
EXPECTED_CANDIDATE_SHA256 = (
    "c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "48b6717bd802071256a3cd3509105bea7e38eb497db1436d92e3d5f4f676b9a1"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "6a8ccf68cdb0a1e234eb66ecd9a7550ab19dc6791a1f00fed1f462e3b394682e"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "727da9e7b7b3493c82c877f837b5534b8cccc4f83f829b4f7233bf4ed26ecefe"
)
EXPECTED_SOURCE_RUN_MANIFEST_SHA256 = (
    "4c9b02eab9f2a6083bae429f9bd8a8a065890c4aa393e5eef0735607e014f8d9"
)
EXPECTED_SOURCE_SHA256 = (
    "e09046962a9af69bca7fb99e4410fe25dcbcb13a7b190716854cf110a8c54b81"
)
SEED = 20260813
LEARNING_RATE = 5e-5
MAXIMUM_GENERATIONS = 2
STAGE_GPU_HOURS = 2.0
ARTIFACT_WRITE_RESERVE_BYTES = 2 * 1024**3


class Phase4F27StageBError(RuntimeError):
    """Raised when conservative Stage B evidence or output is invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F27StageBError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F27StageBError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F27StageBError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F27StageBError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F27StageBError(
            "12 GiB cap cannot preserve the 2 GiB continuation write reserve"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F27StageBError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_f26_evidence(workspace: Path) -> dict[str, object]:
    gate = _mapping(workspace / F26_GATE, "Phase 4F26 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "Phase 4F26 gate")
    checks = gate.get("checks")
    statistics = gate.get("statistics")
    if (
        gate_sha256 != EXPECTED_F26_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "stage-b-lineage-retained"
        or gate.get("additionalStageBTrainingCandidate") is not True
        or gate.get("additionalStageBTrainingAuthorized") is not False
        or gate.get("stageBRestartCandidate") is not False
        or gate.get("nextStageRequiresConfirmation") is not True
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("resultSha256") != EXPECTED_F26_RESULT_SHA256
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(statistics, Mapping)
        or statistics.get("games") != 64
        or statistics.get("candidateWins") != 34
        or statistics.get("candidateLosses") != 30
        or statistics.get("draws") != 0
        or statistics.get("winRate") != 0.53125
    ):
        raise Phase4F27StageBError("Phase 4F26 does not permit continuation preparation")
    result_path = workspace / str(gate.get("resultPath"))
    result = _mapping(result_path, "Phase 4F26 result")
    if (
        _verify_self_hash(result, "resultSha256", "Phase 4F26 result")
        != EXPECTED_F26_RESULT_SHA256
        or result.get("candidateIdentity", {}).get("artifactHash")
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise Phase4F27StageBError("Phase 4F26 result lineage changed")
    return {
        "gatePath": F26_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": gate["resultPath"],
        "resultSha256": EXPECTED_F26_RESULT_SHA256,
        "outcomeBand": gate["outcomeBand"],
        "statistics": dict(statistics),
    }


def _source_archive(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    review = _mapping(workspace / F24_REVIEW, "Phase 4F24 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "Phase 4F24 review")
    best = review.get("latestCandidate")
    manifest = load_run_manifest(workspace / SOURCE_RUN / "run-manifest.json")
    marker = _mapping(workspace / SOURCE_RUN / "best-candidate.json", "F24 best index")
    if (
        review_sha256 != EXPECTED_F24_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("stageBPassed") is not True
        or not isinstance(best, Mapping)
        or marker != best
        or best.get("generation") != 1
        or best.get("step") != 192
        or best.get("candidateModelSha256") != EXPECTED_CANDIDATE_SHA256
        or best.get("checkpointSha256") != EXPECTED_CHECKPOINT_SHA256
        or best.get("modelStateSha256") != EXPECTED_MODEL_STATE_SHA256
        or manifest.sha256 != EXPECTED_SOURCE_RUN_MANIFEST_SHA256
    ):
        raise Phase4F27StageBError("Phase 4F24 candidate is not continuation eligible")
    candidate_name = best.get("candidateModelPath")
    checkpoint_name = best.get("checkpointPath")
    if (
        not isinstance(candidate_name, str)
        or Path(candidate_name).name != candidate_name
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise Phase4F27StageBError("Phase 4F24 candidate filenames are invalid")
    candidate_path = workspace / SOURCE_RUN / "checkpoints" / candidate_name
    checkpoint_path = workspace / SOURCE_RUN / "checkpoints" / checkpoint_name
    if (
        sha256_file(candidate_path) != EXPECTED_CANDIDATE_SHA256
        or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise Phase4F27StageBError("Phase 4F24 candidate files changed")
    provenance = manifest.payload.get("provenance", {})
    config_sha256 = provenance.get("config", {}).get("sha256")
    source_sha256 = provenance.get("source", {}).get("sha256")
    if (
        config_sha256 != EXPECTED_SOURCE_CONFIG_SHA256
        or source_sha256 != EXPECTED_SOURCE_SHA256
    ):
        raise Phase4F27StageBError("Phase 4F24 run provenance changed")
    candidate = {
        "candidateModelSha256": EXPECTED_CANDIDATE_SHA256,
        "checkpointSha256": EXPECTED_CHECKPOINT_SHA256,
        "configSha256": config_sha256,
        "generation": 1,
        "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
        "runManifestSha256": manifest.sha256,
        "sourceSha256": source_sha256,
        "step": 192,
    }
    files = {
        "candidateModel": {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": EXPECTED_CANDIDATE_SHA256,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": EXPECTED_CHECKPOINT_SHA256,
        },
    }
    archive = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4-recovery-candidate-archive",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "candidate": candidate,
            "files": files,
            "bytes": sum(int(item["bytes"]) for item in files.values()),
        },
        "archiveSha256",
    )
    source = {
        "reviewPath": F24_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "candidatePath": candidate_path.relative_to(workspace).as_posix(),
        "checkpointPath": checkpoint_path.relative_to(workspace).as_posix(),
        **candidate,
    }
    return archive, source


def _base_model_sha256(source_payload: Mapping[str, object]) -> str:
    network = source_payload.get("network")
    if not isinstance(network, Mapping):
        raise Phase4F27StageBError("source network config is invalid")
    seed_everything(SEED)
    model = PolicyValueNetwork(
        NetworkConfig(
            input_channels=int(network["inputChannels"]),
            channels=int(network["channels"]),
            blocks=int(network["residualBlocks"]),
            se_reduction=int(network["seReduction"]),
            policy_channels=int(network["policyChannels"]),
            value_channels=int(network["valueChannels"]),
            value_hidden=int(network["valueHidden"]),
        )
    )
    return canonical_model_state_sha256(model)


def _config_payload(
    workspace: Path,
    archive: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    payload = _mapping(workspace / SOURCE_CONFIG, "F24 Stage B config")
    payload.update(
        {
            "name": "phase4f27-stage-b-block-01-v1",
            "seed": SEED,
            "runDir": RUN_DIR.as_posix(),
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        }
    )
    training = dict(payload["training"])
    training["learningRate"] = LEARNING_RATE
    payload["training"] = training
    payload["initialization"] = {
        "mode": "recovery",
        "expectedBaseModelSha256": _base_model_sha256(payload),
        "artifactPath": source["candidatePath"],
        "artifactSha256": EXPECTED_CANDIDATE_SHA256,
        "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
        "recoveryManifestPath": SOURCE_ARCHIVE.as_posix(),
        "recoveryManifestSha256": archive["archiveSha256"],
        "recoveryManifestFileSha256": sha256_file(workspace / SOURCE_ARCHIVE),
        "recoverySourceSha256": EXPECTED_SOURCE_SHA256,
        "recoveryConfigSha256": EXPECTED_SOURCE_CONFIG_SHA256,
        "recoveryRunManifestSha256": EXPECTED_SOURCE_RUN_MANIFEST_SHA256,
        "recoveryCheckpointSha256": EXPECTED_CHECKPOINT_SHA256,
        "recoveryGeneration": 1,
        "recoveryStep": 192,
        "resetOptimizer": True,
    }
    return payload


def _execution_payload(*, baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_B_CONTINUATION,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f27-conservative-pure-self-play-two-generation-block",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": STAGE_GPU_HOURS,
            "baselineRunId": "phase4f26-stage-b-lineage-screen-gate",
            "baselineReportSha256": EXPECTED_F26_GATE_SHA256,
        },
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    return replace(config, run_dir=(workspace / config.run_dir).resolve())


def _verify_initialization(workspace: Path, config: ExperimentConfig) -> dict[str, object]:
    seed_everything(config.seed)
    network = PolicyValueNetwork(
        NetworkConfig(
            input_channels=config.network.input_channels,
            channels=config.network.channels,
            blocks=config.network.residual_blocks,
            se_reduction=config.network.se_reduction,
            policy_channels=config.network.policy_channels,
            value_channels=config.network.value_channels,
            value_hidden=config.network.value_hidden,
        )
    )
    base_sha256 = canonical_model_state_sha256(network)
    if base_sha256 != config.initialization.expected_base_model_sha256:
        raise Phase4F27StageBError("seeded F27 base-model hash changed")
    state, provenance = load_phase4_recovery_artifact(
        workspace_root=workspace,
        config=config,
    )
    network.load_state_dict(state, strict=True)
    if canonical_model_state_sha256(network) != EXPECTED_MODEL_STATE_SHA256:
        raise Phase4F27StageBError("F27 source model changed after load")
    return {"baseModelSha256": base_sha256, **provenance}


def prepare_phase4f27(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    if all((workspace / path).is_file() for path in (CONFIG, EXECUTION, PREPARATION_REPORT)):
        return _verified_preparation(workspace)
    transition = _verify_f26_evidence(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F27StageBError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + STAGE_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F27StageBError("insufficient global GPU budget for F27")
    run_dir = workspace / RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise Phase4F27StageBError("F27 run directory is not empty")
    archive, source = _source_archive(workspace)
    (workspace / SOURCE_ARCHIVE).parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(workspace / SOURCE_ARCHIVE, canonical_json_bytes(archive))
    for relative, payload in (
        (CONFIG, _config_payload(workspace, archive, source)),
        (EXECUTION, _execution_payload(baseline_hours=ledger.total_hours)),
    ):
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(workspace / relative, canonical_json_bytes(payload))
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    if (
        config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
        or config.training.health_holdout is None
        or config.training.learning_rate != LEARNING_RATE
        or config.initialization.artifact_sha256 != EXPECTED_CANDIDATE_SHA256
        or config.initialization.recovery_checkpoint_sha256
        != EXPECTED_CHECKPOINT_SHA256
        or not config.initialization.reset_optimizer
    ):
        raise Phase4F27StageBError("F27 optimizer contract is invalid")
    initialization = _verify_initialization(workspace, config)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f27-conservative-stage-b-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "conservativeStageBTrainingAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "transitionEvidence": transition,
            "initialization": initialization,
            "lineage": {
                "source": "f24-latest-stage-b-candidate",
                "sourceGeneration": 1,
                "sourceStep": 192,
                "f24ReplayInherited": False,
                "optimizerInherited": False,
                "screenTrajectoriesImported": False,
                "seedChanged": True,
            },
            "optimizerInput": {
                "learningStage": "pure-self-play",
                "learningRate": LEARNING_RATE,
                "previousLearningRate": 1e-4,
                "challengeReplayConfigured": False,
                "classicHardOptimizationWeight": 0.0,
                "fixedClassicHoldoutDiagnosticOnly": True,
            },
            "archive": {
                "path": SOURCE_ARCHIVE.as_posix(),
                "fileSha256": sha256_file(workspace / SOURCE_ARCHIVE),
                "archiveSha256": archive["archiveSha256"],
                "referencedBytes": archive["bytes"],
                "modelFilesCopied": False,
            },
            "config": {
                "path": CONFIG.as_posix(),
                "fileSha256": sha256_file(workspace / CONFIG),
                "semanticConfigSha256": fingerprint_config(config).sha256,
                "runDir": RUN_DIR.as_posix(),
            },
            "execution": {
                "path": EXECUTION.as_posix(),
                "fileSha256": sha256_file(workspace / EXECUTION),
                "sha256": execution.sha256,
                "stage": execution.stage,
                "maximumGenerations": execution.maximum_generations,
                "baselineGlobalHours": ledger.total_hours,
                "limitHours": STAGE_GPU_HOURS,
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "exactly two conservative pure-self-play generations",
                "zero classic-hard optimizer samples",
                "exactly 256 cumulative fresh self-play games",
                "fixed classic-hard holdout diagnostic-only",
                "latest-generation external-screen selection",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    atomic_replace_bytes(workspace / PREPARATION_REPORT, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F27 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F27 preparation")
    archive, _ = _source_archive(workspace)
    stored_archive = _mapping(workspace / SOURCE_ARCHIVE, "F27 source archive")
    _verify_self_hash(stored_archive, "archiveSha256", "F27 source archive")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("conservativeStageBTrainingAuthorized") is not True
        or report.get("additionalStageBTrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or stored_archive != archive
        or report.get("archive", {}).get("fileSha256")
        != sha256_file(workspace / SOURCE_ARCHIVE)
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256")
        != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256")
        != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
        or execution.stage != PHASE4F_STAGE_B_CONTINUATION
        or config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
        or config.training.learning_rate != LEARNING_RATE
    ):
        raise Phase4F27StageBError("Phase 4F27 preparation is stale")
    _verify_f26_evidence(workspace)
    validate_phase4_config(config, execution)
    return report


def verify_phase4f27_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract.stage != PHASE4F_STAGE_B_CONTINUATION
        or execution_contract != expected
        or fingerprint_config(config).sha256
        != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F27StageBError("F27 workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "f26GateSha256": EXPECTED_F26_GATE_SHA256,
        "conservativeStageBTrainingAuthorized": True,
        "additionalStageBTrainingAuthorized": False,
    }


def review_phase4f27(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    preparation = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    reports = GenerationReportStore(
        workspace / RUN_DIR / "reports", run_manifest_sha256=manifest.sha256
    )
    if reports.generations() != (0, 1):
        raise Phase4F27StageBError("F27 must contain generations 0 and 1")
    generation_evidence: list[dict[str, object]] = []
    for generation in reports.generations():
        record = reports.load(generation)
        data = record.payload.get("report", {}).get("generationData")
        training = data.get("training") if isinstance(data, Mapping) else None
        selection = data.get("modelSelection") if isinstance(data, Mapping) else None
        replay = data.get("replay") if isinstance(data, Mapping) else None
        metrics = training.get("metrics") if isinstance(training, Mapping) else None
        learning_rate = metrics.get("learningRate") if isinstance(metrics, Mapping) else None
        expected_replay_games = 128 * (generation + 1)
        if (
            not isinstance(data, Mapping)
            or data.get("generation") != generation
            or data.get("selfPlay", {}).get("games") != 128
            or not isinstance(training, Mapping)
            or training.get("learningStage") != "pure-self-play"
            or training.get("challengeReplay") is not None
            or not isinstance(learning_rate, Mapping)
            or float(learning_rate.get("maximum", math.inf)) > LEARNING_RATE
            or not isinstance(data.get("fixedHealthHoldout"), Mapping)
            or not isinstance(selection, Mapping)
            or selection.get("criterion")
            != "latest-pure-self-play-generation-pending-external-screen"
            or selection.get("classicHoldoutDiagnosticOnly") is not True
            or selection.get("bestGeneration") != generation
            or not isinstance(replay, Mapping)
            or replay.get("games") != expected_replay_games
        ):
            raise Phase4F27StageBError(
                f"generation {generation} failed conservative Stage B gates"
            )
        generation_evidence.append(
            {
                "generation": generation,
                "reportPath": record.path.relative_to(workspace).as_posix(),
                "reportSha256": record.sha256,
                "fixedClassicHoldoutDiagnostic": data["fixedHealthHoldout"],
                "modelSelection": selection,
                "learningRate": learning_rate,
                "selfPlayGames": 128,
                "cumulativeReplayGames": expected_replay_games,
            }
        )
    latest = reconcile_best_fixed_holdout_candidate(workspace / RUN_DIR, reports)
    if not isinstance(latest, Mapping) or latest.get("generation") != 1:
        raise Phase4F27StageBError("latest F27 candidate is unavailable")
    provenance = manifest.payload.get("provenance", {})
    if (
        provenance.get("source", {}).get("sha256")
        != preparation["source"]["sha256"]
        or provenance.get("config", {}).get("sha256")
        != preparation["config"]["semanticConfigSha256"]
    ):
        raise Phase4F27StageBError("F27 lineage differs from preparation")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f27-conservative-stage-b-two-generation-review",
            "status": "completed",
            "conservativeStageBPassed": True,
            "f28ScreenAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "preparationReportSha256": preparation["reportSha256"],
            "memorySmoke": memory,
            "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
            "runManifestSha256": manifest.sha256,
            "generations": generation_evidence,
            "latestCandidate": dict(latest),
            "globalGpuHours": GpuBudgetLedger(
                workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
            ).total_hours,
            "nextStage": {
                "name": "Phase 4F28 screen conservative Stage B candidate",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "reviewSha256",
    )
    atomic_replace_bytes(workspace / REVIEW_REPORT, canonical_json_bytes(review))
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-preparation", "review"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_phase4f27(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        result = review_phase4f27(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


__all__ = [
    "CONFIG",
    "EXECUTION",
    "PREPARATION_REPORT",
    "REVIEW_REPORT",
    "Phase4F27StageBError",
    "prepare_phase4f27",
    "review_phase4f27",
    "verify_phase4f27_training_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
