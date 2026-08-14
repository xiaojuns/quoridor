"""Freeze and review the isolated pure-self-play Phase 4F Stage B pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config, seed_everything
from .network import NetworkConfig, PolicyValueNetwork
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F_STAGE_B_PILOT
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_gates import find_valid_memory_smoke
from .phase4_training import load_phase4_recovery_artifact, validate_phase4_config
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
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
F22_GATE = Path("training/runs/phase4/phase4f/stage-a-screen-v2/screen-gate.json")
PROTOCOL = Path("training/configs/phase4f17-two-stage-protocol-frozen-v1.json")
SOURCE_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1"
)
SOURCE_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
ARCHIVE_MANIFEST = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-source-archive.json"
)
CONFIG = Path("training/configs/phase4f23-stage-b-pilot-v1.json")
EXECUTION = Path("training/configs/phase4f23-stage-b-pilot-execution-frozen-v1.json")
RUN_DIR = Path("training/runs/phase4/phase4f/recovery-v3-stage-b-pilot-v1")
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-pilot-freeze-report.json"
)
PILOT_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-pilot-review.json"
)
EXPECTED_F22_GATE_SHA256 = (
    "850b023a86d752d939a34ef0fb373e718d261f05ec0303883bd0852db3fad14b"
)
EXPECTED_F22_RESULT_SHA256 = (
    "4cf946cd840e094102ee56f096581e7ef5c315da62e542bde365bd8d1a4710f5"
)
EXPECTED_PROTOCOL_SHA256 = (
    "523bed24d52df9cfd221727be71aada538521a3f6743496573d61b5882ec9523"
)
EXPECTED_CANDIDATE_SHA256 = (
    "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
)
EXPECTED_BASE_MODEL_SHA256 = (
    "6389e0ebddbd03b804b8426e04a176804ffc10809eb017dc4edb12f316de48a8"
)
SEED = 20260811
MAXIMUM_GENERATIONS = 1
PILOT_GPU_HOURS = 2.0
ARTIFACT_WRITE_RESERVE_BYTES = 1024**3


class Phase4F23StageBError(RuntimeError):
    """Raised when the Stage B transition or pilot evidence is invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F23StageBError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F23StageBError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F23StageBError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F23StageBError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F23StageBError(
            "12 GiB cap cannot preserve the 1 GiB Stage B pilot write reserve"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F23StageBError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_transition(workspace: Path) -> dict[str, object]:
    gate = _mapping(workspace / F22_GATE, "Phase 4F22 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "Phase 4F22 gate")
    checks = gate.get("checks")
    statistics = gate.get("statistics")
    if (
        gate_sha256 != EXPECTED_F22_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "stage-b-transition-candidate"
        or gate.get("stageBTransitionCandidate") is not True
        or gate.get("stageBTrainingAuthorized") is not False
        or gate.get("nextStageRequiresConfirmation") is not True
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("resultSha256") != EXPECTED_F22_RESULT_SHA256
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(statistics, Mapping)
        or statistics.get("games") != 64
        or statistics.get("candidateWins") != 36
        or statistics.get("candidateLosses") != 28
        or statistics.get("draws") != 0
        or statistics.get("winRate") != 0.5625
    ):
        raise Phase4F23StageBError("Phase 4F22 does not authorize Stage B preparation")
    result_path = workspace / str(gate.get("resultPath"))
    result = _mapping(result_path, "Phase 4F22 result")
    if _verify_self_hash(result, "resultSha256", "Phase 4F22 result") != EXPECTED_F22_RESULT_SHA256:
        raise Phase4F23StageBError("Phase 4F22 result changed after screening")

    protocol = _mapping(workspace / PROTOCOL, "two-stage protocol")
    protocol_sha256 = _verify_self_hash(protocol, "protocolSha256", "two-stage protocol")
    stage_b = protocol.get("stageB")
    transition = protocol.get("transitionGate")
    if (
        protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or not isinstance(stage_b, Mapping)
        or stage_b.get("name") != "pure-self-play"
        or stage_b.get("challengeReplayFraction") != 0.0
        or stage_b.get("classicHardOptimizationWeight") != 0.0
        or stage_b.get("fixedClassicHoldoutDiagnosticOnly") is not True
        or stage_b.get("maximumGenerationsPerMicroBlock") != 2
        or stage_b.get("selection")
        != "latest-pure-self-play-generation-pending-external-screen"
        or not isinstance(transition, Mapping)
        or transition.get("forbidsStageAReplayImport") is not True
        or transition.get("resetsOptimizerAndReplay") is not True
        or transition.get("requiresSeparateUserConfirmation") is not True
    ):
        raise Phase4F23StageBError("two-stage protocol does not permit Stage B")
    return {
        "gatePath": F22_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultSha256": EXPECTED_F22_RESULT_SHA256,
        "outcomeBand": gate["outcomeBand"],
        "statistics": dict(statistics),
        "protocolPath": PROTOCOL.as_posix(),
        "protocolSha256": protocol_sha256,
    }


def _source_candidate(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    best = _mapping(workspace / SOURCE_RUN / "best-candidate.json", "F21 best index")
    manifest = load_run_manifest(workspace / SOURCE_RUN / "run-manifest.json")
    candidate_name = best.get("candidateModelPath")
    checkpoint_name = best.get("checkpointPath")
    if (
        best.get("generation") != 1
        or best.get("step") != 192
        or best.get("candidateModelSha256") != EXPECTED_CANDIDATE_SHA256
        or best.get("checkpointSha256") != EXPECTED_CHECKPOINT_SHA256
        or best.get("modelStateSha256") != EXPECTED_MODEL_STATE_SHA256
        or not isinstance(candidate_name, str)
        or Path(candidate_name).name != candidate_name
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise Phase4F23StageBError("F21 best candidate identity is invalid")
    candidate_path = workspace / SOURCE_RUN / "checkpoints" / candidate_name
    checkpoint_path = workspace / SOURCE_RUN / "checkpoints" / checkpoint_name
    expected = {
        "candidateModelSha256": EXPECTED_CANDIDATE_SHA256,
        "checkpointSha256": EXPECTED_CHECKPOINT_SHA256,
        "configSha256": manifest.payload.get("provenance", {}).get("config", {}).get("sha256"),
        "generation": 1,
        "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
        "runManifestSha256": manifest.sha256,
        "sourceSha256": manifest.payload.get("provenance", {}).get("source", {}).get("sha256"),
        "step": 192,
    }
    if (
        sha256_file(candidate_path) != EXPECTED_CANDIDATE_SHA256
        or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise Phase4F23StageBError("F21 candidate files failed SHA-256 validation")
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
            "candidate": expected,
            "files": files,
            "bytes": sum(int(item["bytes"]) for item in files.values()),
        },
        "archiveSha256",
    )
    return archive, {
        "candidatePath": candidate_path.relative_to(workspace).as_posix(),
        "checkpointPath": checkpoint_path.relative_to(workspace).as_posix(),
        **expected,
    }


def _config_payload(
    workspace: Path,
    archive: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    payload = _mapping(workspace / SOURCE_CONFIG, "F21 training config")
    payload.update(
        {
            "name": "phase4f23-stage-b-pilot-v1",
            "seed": SEED,
            "runDir": RUN_DIR.as_posix(),
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        }
    )
    payload["initialization"] = {
        "mode": "recovery",
        "expectedBaseModelSha256": EXPECTED_BASE_MODEL_SHA256,
        "artifactPath": source["candidatePath"],
        "artifactSha256": source["candidateModelSha256"],
        "recoveryManifestPath": ARCHIVE_MANIFEST.as_posix(),
        "recoveryManifestSha256": archive["archiveSha256"],
        "recoveryManifestFileSha256": sha256_bytes(canonical_json_bytes(archive)),
        "recoverySourceSha256": source["sourceSha256"],
        "recoveryConfigSha256": source["configSha256"],
        "recoveryRunManifestSha256": source["runManifestSha256"],
        "recoveryCheckpointSha256": source["checkpointSha256"],
        "recoveryGeneration": source["generation"],
        "recoveryStep": source["step"],
        "modelStateSha256": source["modelStateSha256"],
        "resetOptimizer": True,
    }
    training = dict(payload["training"])
    training.pop("challengeReplay", None)
    training["learningStage"] = "pure-self-play"
    payload["training"] = training
    return payload


def _execution_payload(*, baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_B_PILOT,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f23-isolated-pure-self-play-stage-b-pilot",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": PILOT_GPU_HOURS,
            "baselineRunId": "phase4f22-stage-a-screen-gate",
            "baselineReportSha256": EXPECTED_F22_GATE_SHA256,
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
    if base_sha256 != EXPECTED_BASE_MODEL_SHA256:
        raise Phase4F23StageBError("seeded Stage B base-model hash changed")
    state, provenance = load_phase4_recovery_artifact(
        workspace_root=workspace,
        config=config,
    )
    network.load_state_dict(state, strict=True)
    if canonical_model_state_sha256(network) != EXPECTED_MODEL_STATE_SHA256:
        raise Phase4F23StageBError("Stage B source model changed after load")
    return {"baseModelSha256": base_sha256, **provenance}


def prepare_phase4f23(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    transition = _verify_transition(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F23StageBError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + PILOT_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F23StageBError("insufficient global GPU budget for Stage B pilot")
    run_dir = workspace / RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise Phase4F23StageBError("Stage B pilot run directory is not empty")

    archive, source_candidate = _source_candidate(workspace)
    artifacts = {
        ARCHIVE_MANIFEST: archive,
        CONFIG: _config_payload(workspace, archive, source_candidate),
        EXECUTION: _execution_payload(baseline_hours=ledger.total_hours),
    }
    for relative, payload in artifacts.items():
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(workspace / relative, canonical_json_bytes(payload))

    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    if (
        config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
        or config.training.health_holdout is None
    ):
        raise Phase4F23StageBError("Stage B optimizer input contract is invalid")
    initialization = _verify_initialization(workspace, config)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f23-stage-b-pilot-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "stageBPilotAuthorized": True,
            "formalStageBTrainingAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "transitionEvidence": transition,
            "sourceCandidate": source_candidate,
            "initialization": initialization,
            "lineage": {
                "sourceGeneration": 1,
                "sourceStep": 192,
                "optimizerInherited": False,
                "selfPlayReplayInherited": False,
                "stageAReplayImported": False,
                "screenTrajectoriesImported": False,
                "seedChanged": True,
            },
            "optimizerInput": {
                "learningStage": "pure-self-play",
                "challengeReplayConfigured": False,
                "classicHardOptimizationWeight": 0.0,
                "fixedClassicHoldoutDiagnosticOnly": True,
            },
            "archive": {
                "path": ARCHIVE_MANIFEST.as_posix(),
                "fileSha256": sha256_file(workspace / ARCHIVE_MANIFEST),
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
                "limitHours": PILOT_GPU_HOURS,
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "one isolated pure-self-play generation",
                "zero classic-hard optimizer samples",
                "exactly 128 fresh self-play games in replay",
                "fixed classic-hard holdout diagnostic",
                "latest-generation external-screen selection",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    atomic_replace_bytes(workspace / PREPARATION_REPORT, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F23 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F23 preparation")
    archive = _mapping(workspace / ARCHIVE_MANIFEST, "F23 source archive")
    _verify_self_hash(archive, "archiveSha256", "F23 source archive")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("stageBPilotAuthorized") is not True
        or report.get("formalStageBTrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("archive", {}).get("fileSha256")
        != sha256_file(workspace / ARCHIVE_MANIFEST)
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256")
        != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256")
        != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
        or config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
        or config.training.health_holdout is None
    ):
        raise Phase4F23StageBError("Phase 4F23 preparation is stale")
    _verify_transition(workspace)
    return report


def verify_phase4f23_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected_execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract != expected_execution
        or execution_contract.stage != PHASE4F_STAGE_B_PILOT
        or fingerprint_config(config).sha256
        != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F23StageBError("Stage B pilot workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "stageBPilotAuthorized": True,
        "formalStageBTrainingAuthorized": False,
    }


def review_phase4f23(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    generation_path = workspace / RUN_DIR / "reports/generation-00000000.json"
    generation = _mapping(generation_path, "Stage B pilot generation report")
    generation_sha256 = _verify_self_hash(
        generation, "reportSha256", "Stage B pilot generation report"
    )
    data = generation.get("report", {}).get("generationData")
    training = data.get("training") if isinstance(data, Mapping) else None
    selection = data.get("modelSelection") if isinstance(data, Mapping) else None
    replay = data.get("replay") if isinstance(data, Mapping) else None
    if (
        manifest.payload.get("provenance", {}).get("source", {}).get("sha256")
        != report["source"]["sha256"]
        or manifest.payload.get("provenance", {}).get("config", {}).get("sha256")
        != report["config"]["semanticConfigSha256"]
        or not isinstance(data, Mapping)
        or data.get("generation") != 0
        or data.get("selfPlay", {}).get("games") != 128
        or not isinstance(training, Mapping)
        or training.get("learningStage") != "pure-self-play"
        or training.get("challengeReplay") is not None
        or not isinstance(data.get("fixedHealthHoldout"), Mapping)
        or not isinstance(selection, Mapping)
        or selection.get("criterion")
        != "latest-pure-self-play-generation-pending-external-screen"
        or selection.get("classicHoldoutDiagnosticOnly") is not True
        or selection.get("bestGeneration") != 0
        or not isinstance(replay, Mapping)
        or replay.get("games") != 128
    ):
        raise Phase4F23StageBError("Stage B pilot did not pass isolation gates")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f23-stage-b-pure-self-play-pilot-review",
            "status": "completed",
            "stageBPilotPassed": True,
            "formalStageBTrainingAuthorized": False,
            "preparationReportSha256": report["reportSha256"],
            "memorySmoke": memory,
            "pilot": {
                "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
                "runManifestSha256": manifest.sha256,
                "generationReportPath": generation_path.relative_to(workspace).as_posix(),
                "generationReportSha256": generation_sha256,
                "fixedClassicHoldoutDiagnostic": data["fixedHealthHoldout"],
                "modelSelection": selection,
                "replayGames": replay["games"],
            },
            "globalGpuHours": GpuBudgetLedger(
                workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
            ).total_hours,
            "nextStage": {
                "name": "Phase 4F24 freeze and run Stage B two-generation block",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "reviewSha256",
    )
    atomic_replace_bytes(workspace / PILOT_REVIEW, canonical_json_bytes(review))
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-preparation", "review"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_phase4f23(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        result = review_phase4f23(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


__all__ = [
    "CONFIG",
    "EXECUTION",
    "PILOT_REVIEW",
    "PREPARATION_REPORT",
    "Phase4F23StageBError",
    "prepare_phase4f23",
    "review_phase4f23",
    "verify_phase4f23_training_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
