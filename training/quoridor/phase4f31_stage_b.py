"""Freeze and review the extended pure-self-play Stage B continuation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F_STAGE_B_EXTENDED
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_gates import find_valid_memory_smoke
from .phase4_training import reconcile_best_fixed_holdout_candidate, validate_phase4_config
from .replay_store import ReplayStore
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
F30_GATE = Path("training/runs/phase4/phase4f/fresh-lineage-screen-v1/screen-gate.json")
EXPECTED_F30_GATE_SHA256 = "b7b840dddd825bf2288974df40821739726cf5ec92673695f85075d1a0d4a358"
EXPECTED_F30_RESULT_SHA256 = "13211b7b80eec337117dc314971c35a48b8d3ae421a3f4a882b84272fd4a3e08"
F27_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f27-stage-b-review.json"
)
EXPECTED_F27_REVIEW_SHA256 = "b1d70538c6756e8df192159f41a9fdd88d88064b0ccaf5331b1e68cc3800d8e7"
SOURCE_RUN = Path("training/runs/phase4/phase4f/recovery-v3-stage-b-block-01-v1")
SOURCE_CONFIG = Path("training/configs/phase4f27-stage-b-block-01-v1.json")
EXPECTED_SOURCE_RUN_MANIFEST_SHA256 = "f1717f7a385d60a57844c171fded3ac0becd73d4e83d3dc31ec1e576e93cda82"
EXPECTED_SOURCE_CONFIG_SHA256 = "c70c2d25843388aaafd5b9cd2cebd539476629ac56e10c3e7eb572776556315b"
EXPECTED_SOURCE_SHA256 = "0182019b2865af6d7000d41d61ba2c5045fabb05066707e1110900265a842083"
EXPECTED_CANDIDATE_SHA256 = "6c7756474c05e5ebe2a6a8acfbac6a2b35f693a297bd112898589141f337a996"
EXPECTED_CHECKPOINT_SHA256 = "61df04e21c499b558b20a57de2daa240671b72bae75d9fd6292714c042c6e0dd"
EXPECTED_MODEL_STATE_SHA256 = "5f5091c982b1be28faf508ccec6f4b106d7a248c3efa213d62d36701b432a2ba"
EXPECTED_REPLAY_MANIFEST_SHA256 = "849c9c4fc2d1d36f59a7d6dd0630125300b75b664a525fb0f9c0104d6248901e"
SOURCE_GENERATION = 1
SOURCE_STEP = 192
SOURCE_REPLAY_GAMES = 256
CONFIG = Path("training/configs/phase4f31-stage-b-extended-v1.json")
EXECUTION = Path("training/configs/phase4f31-stage-b-execution-frozen-v1.json")
RUN_DIR = Path("training/runs/phase4/phase4f/recovery-v3-stage-b-extended-v1")
ARCHIVE = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f31-stage-b-continuation-archive.json"
)
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f31-stage-b-freeze-report.json"
)
REVIEW_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f31-stage-b-review.json"
)
LEARNING_RATE = 5e-5
MAXIMUM_GENERATIONS = 6
NEW_GENERATIONS = (2, 3, 4, 5)
STAGE_GPU_HOURS = 5.0
ARTIFACT_WRITE_RESERVE_BYTES = 2 * 1024**3


class Phase4F31StageBError(RuntimeError):
    """Raised when F31 evidence, continuation state, or output is invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F31StageBError(f"{label} is missing or is a symlink")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F31StageBError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F31StageBError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F31StageBError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _verify_transition(workspace: Path) -> dict[str, object]:
    gate = _mapping(workspace / F30_GATE, "Phase 4F30 gate")
    digest = _verify_self_hash(gate, "gateSha256", "Phase 4F30 gate")
    statistics = gate.get("statistics")
    if (
        digest != EXPECTED_F30_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "stage-b-stronger-on-fresh-openings"
        or gate.get("stageBRetainedCandidate") is not True
        or gate.get("stageARollbackCandidate") is not False
        or gate.get("trainingAuthorized") is not False
        or gate.get("resultSha256") != EXPECTED_F30_RESULT_SHA256
        or not isinstance(statistics, Mapping)
        or statistics.get("games") != 64
        or statistics.get("candidateWins") != 26
        or statistics.get("candidateLosses") != 38
    ):
        raise Phase4F31StageBError("Phase 4F30 does not support the F31 decision")
    return {
        "gatePath": F30_GATE.as_posix(),
        "gateSha256": digest,
        "resultSha256": EXPECTED_F30_RESULT_SHA256,
        "outcomeBand": gate["outcomeBand"],
        "statistics": dict(statistics),
    }


def _verify_source(workspace: Path) -> dict[str, object]:
    review = _mapping(workspace / F27_REVIEW, "Phase 4F27 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "Phase 4F27 review")
    candidate = review.get("latestCandidate")
    marker = _mapping(workspace / SOURCE_RUN / "best-candidate.json", "F27 candidate")
    manifest = load_run_manifest(workspace / SOURCE_RUN / "run-manifest.json")
    provenance = manifest.payload.get("provenance", {})
    config_sha256 = provenance.get("config", {}).get("sha256")
    source_sha256 = provenance.get("source", {}).get("sha256")
    if (
        review_sha256 != EXPECTED_F27_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("conservativeStageBPassed") is not True
        or candidate != marker
        or not isinstance(candidate, Mapping)
        or candidate.get("generation") != SOURCE_GENERATION
        or candidate.get("step") != SOURCE_STEP
        or candidate.get("candidateModelSha256") != EXPECTED_CANDIDATE_SHA256
        or candidate.get("checkpointSha256") != EXPECTED_CHECKPOINT_SHA256
        or candidate.get("modelStateSha256") != EXPECTED_MODEL_STATE_SHA256
        or manifest.sha256 != EXPECTED_SOURCE_RUN_MANIFEST_SHA256
        or config_sha256 != EXPECTED_SOURCE_CONFIG_SHA256
        or source_sha256 != EXPECTED_SOURCE_SHA256
    ):
        raise Phase4F31StageBError("F27 source lineage changed")
    candidate_path = workspace / SOURCE_RUN / "checkpoints" / str(candidate["candidateModelPath"])
    checkpoint_path = workspace / SOURCE_RUN / "checkpoints" / str(candidate["checkpointPath"])
    replay_path = workspace / SOURCE_RUN / "replay" / "manifest.json"
    if (
        sha256_file(candidate_path) != EXPECTED_CANDIDATE_SHA256
        or sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256
        or sha256_file(replay_path) != EXPECTED_REPLAY_MANIFEST_SHA256
    ):
        raise Phase4F31StageBError("F27 source files changed")
    source_config = load_experiment_config(workspace / SOURCE_CONFIG)
    replay = ReplayStore(
        workspace / SOURCE_RUN / "replay",
        max_games=source_config.training.replay_capacity,
        max_plies=source_config.training.replay_capacity,
        max_bytes=max(1, source_config.artifact_limit_bytes * 3 // 4),
    )
    verification = replay.verify(deep=False)
    if verification.game_count != SOURCE_REPLAY_GAMES:
        raise Phase4F31StageBError("F27 replay game count changed")
    return {
        "reviewPath": F27_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "runManifestPath": (SOURCE_RUN / "run-manifest.json").as_posix(),
        "runManifestSha256": manifest.sha256,
        "configSha256": config_sha256,
        "sourceSha256": source_sha256,
        "candidatePath": candidate_path.relative_to(workspace).as_posix(),
        "candidateModelSha256": EXPECTED_CANDIDATE_SHA256,
        "checkpointPath": checkpoint_path.relative_to(workspace).as_posix(),
        "checkpointSha256": EXPECTED_CHECKPOINT_SHA256,
        "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
        "generation": SOURCE_GENERATION,
        "step": SOURCE_STEP,
        "replayManifestSha256": EXPECTED_REPLAY_MANIFEST_SHA256,
        "replayGames": verification.game_count,
        "replayPlies": verification.ply_count,
        "replayCommittedBytes": verification.committed_bytes,
    }


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F31StageBError("12 GiB cap cannot preserve the 2 GiB write reserve")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F31StageBError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _config_payload(workspace: Path, source: Mapping[str, object], archive: Mapping[str, object]) -> dict[str, object]:
    payload = _mapping(workspace / SOURCE_CONFIG, "F27 config")
    payload.update(
        {
            "name": "phase4f31-stage-b-extended-v1",
            "runDir": RUN_DIR.as_posix(),
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        }
    )
    training = dict(payload["training"])
    training["learningRate"] = LEARNING_RATE
    payload["training"] = training
    payload["initialization"] = {
        "mode": "recovery",
        "expectedBaseModelSha256": payload["initialization"]["expectedBaseModelSha256"],
        "artifactPath": source["candidatePath"],
        "artifactSha256": EXPECTED_CANDIDATE_SHA256,
        "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
        "recoveryManifestPath": ARCHIVE.as_posix(),
        "recoveryManifestSha256": archive["archiveSha256"],
        "recoveryManifestFileSha256": sha256_file(workspace / ARCHIVE),
        "recoverySourceSha256": EXPECTED_SOURCE_SHA256,
        "recoveryConfigSha256": EXPECTED_SOURCE_CONFIG_SHA256,
        "recoveryRunManifestSha256": EXPECTED_SOURCE_RUN_MANIFEST_SHA256,
        "recoveryCheckpointSha256": EXPECTED_CHECKPOINT_SHA256,
        "recoveryGeneration": SOURCE_GENERATION,
        "recoveryStep": SOURCE_STEP,
        "resetOptimizer": False,
    }
    return payload


def _execution_payload(baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_B_EXTENDED,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f31-exact-stage-b-checkpoint-continuation",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": STAGE_GPU_HOURS,
            "baselineRunId": "phase4f30-fresh-lineage-screen-gate",
            "baselineReportSha256": EXPECTED_F30_GATE_SHA256,
        },
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    return replace(config, run_dir=(workspace / config.run_dir).resolve())


def prepare_phase4f31(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    if all((workspace / path).is_file() for path in (CONFIG, EXECUTION, ARCHIVE, PREPARATION_REPORT)):
        return _verified_preparation(workspace)
    transition = _verify_transition(workspace)
    source = _verify_source(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F31StageBError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + STAGE_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F31StageBError("insufficient global GPU budget for F31")
    if (workspace / RUN_DIR).exists() and any((workspace / RUN_DIR).iterdir()):
        raise Phase4F31StageBError("F31 run directory is not empty")
    archive = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4-recovery-candidate-archive",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "candidate": {
                "candidateModelSha256": EXPECTED_CANDIDATE_SHA256,
                "checkpointSha256": EXPECTED_CHECKPOINT_SHA256,
                "configSha256": EXPECTED_SOURCE_CONFIG_SHA256,
                "generation": SOURCE_GENERATION,
                "modelStateSha256": EXPECTED_MODEL_STATE_SHA256,
                "runManifestSha256": EXPECTED_SOURCE_RUN_MANIFEST_SHA256,
                "sourceSha256": EXPECTED_SOURCE_SHA256,
                "step": SOURCE_STEP,
            },
            "files": {
                "candidateModel": {
                    "path": Path(str(source["candidatePath"])).name,
                    "bytes": (workspace / str(source["candidatePath"])).stat().st_size,
                    "sha256": EXPECTED_CANDIDATE_SHA256,
                },
                "checkpoint": {
                    "path": Path(str(source["checkpointPath"])).name,
                    "bytes": (workspace / str(source["checkpointPath"])).stat().st_size,
                    "sha256": EXPECTED_CHECKPOINT_SHA256,
                },
            },
        },
        "archiveSha256",
    )
    archive["bytes"] = sum(int(value["bytes"]) for value in archive["files"].values())
    core = dict(archive)
    core.pop("archiveSha256")
    archive["archiveSha256"] = sha256_bytes(canonical_json_bytes(core))
    (workspace / ARCHIVE).parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(workspace / ARCHIVE, canonical_json_bytes(archive))
    for path, payload in (
        (CONFIG, _config_payload(workspace, source, archive)),
        (EXECUTION, _execution_payload(ledger.total_hours)),
    ):
        (workspace / path).parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(workspace / path, canonical_json_bytes(payload))
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    if (
        config.seed != 20260813
        or config.training.learning_stage != "pure-self-play"
        or config.training.challenge_replay is not None
        or config.training.health_holdout is None
        or config.training.learning_rate != LEARNING_RATE
        or config.initialization.reset_optimizer is not False
        or execution.maximum_generations != MAXIMUM_GENERATIONS
    ):
        raise Phase4F31StageBError("F31 continuation semantics are invalid")
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f31-extended-stage-b-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "formalExtendedStageBTrainingAuthorized": True,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "transitionEvidence": transition,
            "continuation": {
                **source,
                "optimizerInherited": True,
                "schedulerInherited": True,
                "rngInherited": True,
                "trainerStepInherited": True,
                "replayInherited": True,
                "screenTrajectoriesImported": False,
                "nextGeneration": 2,
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
                "newGenerations": list(NEW_GENERATIONS),
                "baselineGlobalHours": ledger.total_hours,
                "limitHours": STAGE_GPU_HOURS,
            },
            "archive": {
                "path": ARCHIVE.as_posix(),
                "fileSha256": sha256_file(workspace / ARCHIVE),
                "archiveSha256": archive["archiveSha256"],
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "exact F27 optimizer scheduler RNG step replay migration",
                "exactly four additional pure-self-play generations",
                "zero classic-hard optimizer samples",
                "fixed classic-hard holdout diagnostic-only",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    atomic_replace_bytes(workspace / PREPARATION_REPORT, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F31 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F31 preparation")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("formalExtendedStageBTrainingAuthorized") is not True
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256") != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256") != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
        or report.get("archive", {}).get("fileSha256") != sha256_file(workspace / ARCHIVE)
        or execution.stage != PHASE4F_STAGE_B_EXTENDED
        or config.initialization.reset_optimizer is not False
    ):
        raise Phase4F31StageBError("Phase 4F31 preparation is stale")
    _verify_transition(workspace)
    _verify_source(workspace)
    validate_phase4_config(config, execution)
    return report


def verify_phase4f31_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract != expected
        or fingerprint_config(config).sha256 != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F31StageBError("F31 workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "formalExtendedStageBTrainingAuthorized": True,
    }


def review_phase4f31(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    preparation = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    reports = GenerationReportStore(workspace / RUN_DIR / "reports", run_manifest_sha256=manifest.sha256)
    if reports.generations() != tuple(range(MAXIMUM_GENERATIONS)):
        raise Phase4F31StageBError("F31 must contain generations 0 through 5")
    evidence: list[dict[str, object]] = []
    for generation in NEW_GENERATIONS:
        record = reports.load(generation)
        data = record.payload.get("report", {}).get("generationData")
        training = data.get("training") if isinstance(data, Mapping) else None
        selection = data.get("modelSelection") if isinstance(data, Mapping) else None
        replay = data.get("replay") if isinstance(data, Mapping) else None
        metrics = training.get("metrics") if isinstance(training, Mapping) else None
        learning_rate = metrics.get("learningRate") if isinstance(metrics, Mapping) else None
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
            or selection.get("criterion") != "latest-pure-self-play-generation-pending-external-screen"
            or selection.get("bestGeneration") != generation
            or not isinstance(replay, Mapping)
            or replay.get("games") != 128 * (generation + 1)
        ):
            raise Phase4F31StageBError(f"generation {generation} failed F31 gates")
        evidence.append(
            {
                "generation": generation,
                "reportPath": record.path.relative_to(workspace).as_posix(),
                "reportSha256": record.sha256,
                "selfPlayGames": 128,
                "cumulativeReplayGames": replay["games"],
                "learningRate": learning_rate,
                "fixedClassicHoldoutDiagnostic": data["fixedHealthHoldout"],
            }
        )
    latest = reconcile_best_fixed_holdout_candidate(workspace / RUN_DIR, reports)
    if not isinstance(latest, Mapping) or latest.get("generation") != 5:
        raise Phase4F31StageBError("latest F31 candidate is unavailable")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f31-extended-stage-b-review",
            "status": "completed",
            "extendedStageBPassed": True,
            "f32FreshScreenAuthorized": True,
            "additionalTrainingAuthorized": False,
            "preparationReportSha256": preparation["reportSha256"],
            "memorySmoke": memory,
            "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
            "runManifestSha256": manifest.sha256,
            "migratedGenerations": [0, 1],
            "newGenerations": evidence,
            "latestCandidate": dict(latest),
            "globalGpuHours": GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME).total_hours,
            "nextStage": {
                "name": "Phase 4F32 fresh-opening screen of generation 5",
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
        result = prepare_phase4f31(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(args.workspace_root.resolve())
    else:
        result = review_phase4f31(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
