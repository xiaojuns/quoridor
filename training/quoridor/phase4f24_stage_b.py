"""Freeze and review the formal two-generation pure-self-play Stage B block."""

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
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F_STAGE_B
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
F23_PREPARATION = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-pilot-freeze-report.json"
)
F23_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-pilot-review.json"
)
SOURCE_CONFIG = Path("training/configs/phase4f23-stage-b-pilot-v1.json")
SOURCE_ARCHIVE = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f23-stage-b-source-archive.json"
)
CONFIG = Path("training/configs/phase4f24-stage-b-block-00-v1.json")
EXECUTION = Path("training/configs/phase4f24-stage-b-execution-frozen-v1.json")
RUN_DIR = Path("training/runs/phase4/phase4f/recovery-v3-stage-b-block-00-v1")
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f24-stage-b-freeze-report.json"
)
REVIEW_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f24-stage-b-review.json"
)
EXPECTED_F23_PREPARATION_SHA256 = (
    "6a283ed446455368033cc3b65a8c29e193292cc9f34df4a27b55fb88b5518982"
)
EXPECTED_F23_REVIEW_SHA256 = (
    "dcabfac8af0d18048a77dbd3984805aad6ecdc4f9fd09bff7c949ff54e05a8c6"
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
    "1d57f00fba63985b6121571114a33aa905c88d0a6914d6a8f5d13a5f146cf472"
)
SEED = 20260812
MAXIMUM_GENERATIONS = 2
STAGE_GPU_HOURS = 2.0
ARTIFACT_WRITE_RESERVE_BYTES = 2 * 1024**3


class Phase4F24StageBError(RuntimeError):
    """Raised when formal Stage B evidence or output is invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F24StageBError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F24StageBError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F24StageBError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F24StageBError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F24StageBError(
            "12 GiB cap cannot preserve the 2 GiB Stage B write reserve"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F24StageBError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_f23_evidence(workspace: Path) -> dict[str, object]:
    preparation = _mapping(workspace / F23_PREPARATION, "Phase 4F23 preparation")
    preparation_sha256 = _verify_self_hash(
        preparation, "reportSha256", "Phase 4F23 preparation"
    )
    review = _mapping(workspace / F23_REVIEW, "Phase 4F23 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "Phase 4F23 review")
    baseline_hours = review.get("globalGpuHours")
    memory = review.get("memorySmoke")
    pilot = review.get("pilot")
    gates = memory.get("gates") if isinstance(memory, Mapping) else None
    selection = pilot.get("modelSelection") if isinstance(pilot, Mapping) else None
    if (
        preparation_sha256 != EXPECTED_F23_PREPARATION_SHA256
        or review_sha256 != EXPECTED_F23_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("stageBPilotPassed") is not True
        or review.get("formalStageBTrainingAuthorized") is not False
        or review.get("preparationReportSha256") != preparation_sha256
        or isinstance(baseline_hours, bool)
        or not isinstance(baseline_hours, (int, float))
        or not math.isfinite(float(baseline_hours))
        or float(baseline_hours) < 0
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(pilot, Mapping)
        or pilot.get("replayGames") != 128
        or not isinstance(pilot.get("fixedClassicHoldoutDiagnostic"), Mapping)
        or not isinstance(selection, Mapping)
        or selection.get("criterion")
        != "latest-pure-self-play-generation-pending-external-screen"
        or selection.get("bestGeneration") != 0
    ):
        raise Phase4F24StageBError("Phase 4F23 pilot evidence is not eligible")

    generation_path = workspace / str(pilot.get("generationReportPath"))
    generation = _mapping(generation_path, "Phase 4F23 generation report")
    if _verify_self_hash(
        generation, "reportSha256", "Phase 4F23 generation report"
    ) != pilot.get("generationReportSha256"):
        raise Phase4F24StageBError("Phase 4F23 generation report changed")
    manifest_path = workspace / str(pilot.get("runManifestPath"))
    if load_run_manifest(manifest_path).sha256 != pilot.get("runManifestSha256"):
        raise Phase4F24StageBError("Phase 4F23 run manifest changed")
    return {
        "preparationPath": F23_PREPARATION.as_posix(),
        "preparationSha256": preparation_sha256,
        "reviewPath": F23_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "baselineGlobalHours": float(baseline_hours),
        "pilotGenerationReportSha256": pilot["generationReportSha256"],
        "pilotRunManifestSha256": pilot["runManifestSha256"],
        "pilotMemorySmokeReportSha256": memory["reportCoreSha256"],
    }


def _config_payload(workspace: Path) -> dict[str, object]:
    payload = _mapping(workspace / SOURCE_CONFIG, "F23 pilot config")
    payload.update(
        {
            "name": "phase4f24-stage-b-block-00-v1",
            "seed": SEED,
            "runDir": RUN_DIR.as_posix(),
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        }
    )
    initialization = dict(payload["initialization"])
    initialization["expectedBaseModelSha256"] = EXPECTED_BASE_MODEL_SHA256
    payload["initialization"] = initialization
    return payload


def _execution_payload(*, baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_B,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f24-formal-pure-self-play-two-generation-block",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": STAGE_GPU_HOURS,
            "baselineRunId": "phase4f23-stage-b-pilot-review",
            "baselineReportSha256": EXPECTED_F23_REVIEW_SHA256,
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
        raise Phase4F24StageBError("seeded F24 base-model hash changed")
    state, provenance = load_phase4_recovery_artifact(
        workspace_root=workspace,
        config=config,
    )
    network.load_state_dict(state, strict=True)
    if canonical_model_state_sha256(network) != EXPECTED_MODEL_STATE_SHA256:
        raise Phase4F24StageBError("F24 source model changed after load")
    return {"baseModelSha256": base_sha256, **provenance}


def prepare_phase4f24(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    pilot = _verify_f23_evidence(workspace)
    baseline_hours = float(pilot["baselineGlobalHours"])
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F24StageBError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + STAGE_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F24StageBError("insufficient global GPU budget for Stage B")
    run_dir = workspace / RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise Phase4F24StageBError("formal Stage B run directory is not empty")

    archive = _mapping(workspace / SOURCE_ARCHIVE, "Stage B source archive")
    archive_sha256 = _verify_self_hash(
        archive, "archiveSha256", "Stage B source archive"
    )
    candidate = archive.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("candidateModelSha256") != EXPECTED_CANDIDATE_SHA256
        or candidate.get("checkpointSha256") != EXPECTED_CHECKPOINT_SHA256
        or candidate.get("modelStateSha256") != EXPECTED_MODEL_STATE_SHA256
        or candidate.get("generation") != 1
        or candidate.get("step") != 192
    ):
        raise Phase4F24StageBError("Stage B source archive candidate is invalid")

    for relative, payload in (
        (CONFIG, _config_payload(workspace)),
        (EXECUTION, _execution_payload(baseline_hours=baseline_hours)),
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
        or config.initialization.artifact_sha256 != EXPECTED_CANDIDATE_SHA256
        or config.initialization.recovery_checkpoint_sha256
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise Phase4F24StageBError("formal Stage B optimizer contract is invalid")
    initialization = _verify_initialization(workspace, config)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f24-formal-stage-b-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "formalStageBTrainingAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "pilotEvidence": pilot,
            "initialization": initialization,
            "lineage": {
                "source": "f21-best-stage-a-candidate",
                "sourceGeneration": 1,
                "sourceStep": 192,
                "f23PilotModelInherited": False,
                "f23PilotReplayInherited": False,
                "optimizerInherited": False,
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
                "path": SOURCE_ARCHIVE.as_posix(),
                "fileSha256": sha256_file(workspace / SOURCE_ARCHIVE),
                "archiveSha256": archive_sha256,
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
                "baselineGlobalHours": baseline_hours,
                "limitHours": STAGE_GPU_HOURS,
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "exactly two pure-self-play generations",
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
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F24 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F24 preparation")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("formalStageBTrainingAuthorized") is not True
        or report.get("additionalStageBTrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("archive", {}).get("fileSha256")
        != sha256_file(workspace / SOURCE_ARCHIVE)
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256")
        != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256")
        != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
        or config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
    ):
        raise Phase4F24StageBError("Phase 4F24 preparation is stale")
    _verify_f23_evidence(workspace)
    validate_phase4_config(config, execution)
    return report


def verify_phase4f24_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract.stage != PHASE4F_STAGE_B
        or execution_contract != expected
        or fingerprint_config(config).sha256
        != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F24StageBError("formal Stage B workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "pilotReviewSha256": EXPECTED_F23_REVIEW_SHA256,
        "formalStageBTrainingAuthorized": True,
        "additionalStageBTrainingAuthorized": False,
    }


def review_phase4f24(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    preparation = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    reports = GenerationReportStore(
        workspace / RUN_DIR / "reports", run_manifest_sha256=manifest.sha256
    )
    if reports.generations() != (0, 1):
        raise Phase4F24StageBError("formal Stage B must contain generations 0 and 1")
    generation_evidence: list[dict[str, object]] = []
    for generation in reports.generations():
        record = reports.load(generation)
        data = record.payload.get("report", {}).get("generationData")
        training = data.get("training") if isinstance(data, Mapping) else None
        selection = data.get("modelSelection") if isinstance(data, Mapping) else None
        replay = data.get("replay") if isinstance(data, Mapping) else None
        expected_replay_games = 128 * (generation + 1)
        if (
            not isinstance(data, Mapping)
            or data.get("generation") != generation
            or data.get("selfPlay", {}).get("games") != 128
            or not isinstance(training, Mapping)
            or training.get("learningStage") != "pure-self-play"
            or training.get("challengeReplay") is not None
            or not isinstance(data.get("fixedHealthHoldout"), Mapping)
            or not isinstance(selection, Mapping)
            or selection.get("criterion")
            != "latest-pure-self-play-generation-pending-external-screen"
            or selection.get("classicHoldoutDiagnosticOnly") is not True
            or selection.get("bestGeneration") != generation
            or not isinstance(replay, Mapping)
            or replay.get("games") != expected_replay_games
        ):
            raise Phase4F24StageBError(
                f"generation {generation} failed pure-self-play gates"
            )
        generation_evidence.append(
            {
                "generation": generation,
                "reportPath": record.path.relative_to(workspace).as_posix(),
                "reportSha256": record.sha256,
                "fixedClassicHoldoutDiagnostic": data["fixedHealthHoldout"],
                "modelSelection": selection,
                "selfPlayGames": 128,
                "cumulativeReplayGames": expected_replay_games,
            }
        )
    latest = reconcile_best_fixed_holdout_candidate(workspace / RUN_DIR, reports)
    if not isinstance(latest, Mapping) or latest.get("generation") != 1:
        raise Phase4F24StageBError("latest Stage B candidate is unavailable")
    provenance = manifest.payload.get("provenance", {})
    if (
        provenance.get("source", {}).get("sha256")
        != preparation["source"]["sha256"]
        or provenance.get("config", {}).get("sha256")
        != preparation["config"]["semanticConfigSha256"]
    ):
        raise Phase4F24StageBError("formal Stage B lineage differs from preparation")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f24-stage-b-two-generation-review",
            "status": "completed",
            "stageBPassed": True,
            "f25ScreenAuthorized": True,
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
                "name": "Phase 4F25 screen latest Stage B candidate",
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
        result = prepare_phase4f24(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        result = review_phase4f24(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


__all__ = [
    "CONFIG",
    "EXECUTION",
    "PREPARATION_REPORT",
    "REVIEW_REPORT",
    "Phase4F24StageBError",
    "prepare_phase4f24",
    "review_phase4f24",
    "verify_phase4f24_training_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
