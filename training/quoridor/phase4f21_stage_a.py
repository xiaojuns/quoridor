"""Prepare and review the final guided Phase 4F Stage A micro-block."""

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
from .phase4_budget import (
    PHASE4_CONTROL_RELATIVE,
    PHASE4F_STAGE_A_CONTINUATION,
)
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
    DEFAULT_ARTIFACT_LIMIT_BYTES,
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
F19_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f19-stage-a-review.json"
)
F20_GATE = Path("training/runs/phase4/phase4f/stage-a-screen-v1/screen-gate.json")
PROTOCOL = Path("training/configs/phase4f17-two-stage-protocol-frozen-v1.json")
SOURCE_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-00-v1"
)
SOURCE_CONFIG = Path("training/configs/phase4f18-stage-a-block-00-v1.json")
PREPARATION_ROOT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1"
)
ARCHIVE_MANIFEST = PREPARATION_ROOT / "phase4f21-stage-a-source-archive.json"
PREPARATION_REPORT = PREPARATION_ROOT / "phase4f21-stage-a-freeze-report.json"
REVIEW_REPORT = PREPARATION_ROOT / "phase4f21-stage-a-review.json"
CAP_EXTENSION = Path(
    "training/configs/phase4f21-artifact-cap-extension-frozen-v1.json"
)
CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
EXECUTION = Path("training/configs/phase4f21-stage-a-execution-frozen-v1.json")
RUN_DIR = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1"
)

EXPECTED_F19_REVIEW_SHA256 = (
    "edc1707f4c87d889c3facd5dd3ba7a8caed6f482e0e4c24c0caea603bb5d1cf4"
)
EXPECTED_F20_GATE_SHA256 = (
    "dce614190604d1bdabbcc86b335a5cdd45e93d95ae70847e67b0e27be3fc2999"
)
EXPECTED_PROTOCOL_SHA256 = (
    "523bed24d52df9cfd221727be71aada538521a3f6743496573d61b5882ec9523"
)
EXPECTED_BASE_MODEL_SHA256 = (
    "23182b7a4e8110d1e619a507e8acb1e8c1c8a0e5e17cc0d706f9cef97736d8eb"
)
SEED = 20260810
MAXIMUM_GENERATIONS = 2
STAGE_GPU_HOURS = 2.0
ARTIFACT_WRITE_RESERVE_BYTES = 2 * 1024**3


class Phase4F21StageAError(RuntimeError):
    """Raised when the final Stage A block is not safely reproducible."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F21StageAError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F21StageAError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F21StageAError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F21StageAError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F21StageAError(
            "12 GiB cap cannot preserve the 2 GiB F21 write reserve"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F21StageAError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_historical_evidence(workspace: Path) -> dict[str, object]:
    review = _mapping(workspace / F19_REVIEW, "Phase 4F19 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "Phase 4F19 review")
    best = review.get("bestCandidate")
    if (
        review_sha256 != EXPECTED_F19_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("stageAPassed") is not True
        or review.get("stageBTrainingAuthorized") is not False
        or not isinstance(best, Mapping)
        or best.get("generation") != 1
        or best.get("step") != 192
        or best.get("candidateModelSha256")
        != "ea7df3db48a0110ce2dcd7dde7d603e455bf7fc578e95418096eb09f016a02d1"
        or best.get("checkpointSha256")
        != "e82c5ac7b87128e1ee4eeac63a7c4d89aa8083879385e9bd63f079cab0de57a2"
        or best.get("modelStateSha256")
        != "12233718eb60fa43084449f9b0c5025b446b9ef090f282373cc73949555d8b63"
    ):
        raise Phase4F21StageAError("Phase 4F19 best candidate is not eligible")

    gate = _mapping(workspace / F20_GATE, "Phase 4F20 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "Phase 4F20 gate")
    checks = gate.get("checks")
    statistics = gate.get("statistics")
    if (
        gate_sha256 != EXPECTED_F20_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "continue-stage-a"
        or gate.get("stageBTrainingAuthorized") is not False
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(statistics, Mapping)
        or statistics.get("games") != 64
        or statistics.get("candidateWins") != 23
        or statistics.get("candidateLosses") != 41
        or statistics.get("draws") != 0
        or statistics.get("winRate") != 0.359375
    ):
        raise Phase4F21StageAError("Phase 4F20 gate does not authorize continuation")
    result_path = workspace / str(gate.get("resultPath"))
    result = _mapping(result_path, "Phase 4F20 result")
    if _verify_self_hash(result, "resultSha256", "Phase 4F20 result") != gate.get(
        "resultSha256"
    ):
        raise Phase4F21StageAError("Phase 4F20 result changed after screening")

    protocol = _mapping(workspace / PROTOCOL, "two-stage protocol")
    protocol_sha256 = _verify_self_hash(
        protocol, "protocolSha256", "two-stage protocol"
    )
    if (
        protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or protocol.get("stageA", {}).get("maximumGenerationsPerMicroBlock") != 2
        or protocol.get("stageA", {}).get("challengeReplayFraction") != 0.4
        or protocol.get("screenPolicy", {}).get("continueTrainingFromWinRate") != 0.35
        or protocol.get("screenPolicy", {}).get("confirmationFromWinRate") != 0.5
    ):
        raise Phase4F21StageAError("two-stage protocol changed")
    return {
        "f19ReviewPath": F19_REVIEW.as_posix(),
        "f19ReviewSha256": review_sha256,
        "f19BestCandidate": dict(best),
        "f20GatePath": F20_GATE.as_posix(),
        "f20GateSha256": gate_sha256,
        "f20ResultSha256": gate["resultSha256"],
        "f20OutcomeBand": gate["outcomeBand"],
        "protocolPath": PROTOCOL.as_posix(),
        "protocolSha256": protocol_sha256,
    }


def _source_candidate(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    best = _mapping(workspace / SOURCE_RUN / "best-candidate.json", "F19 best index")
    manifest = load_run_manifest(workspace / SOURCE_RUN / "run-manifest.json")
    candidate_name = best.get("candidateModelPath")
    checkpoint_name = best.get("checkpointPath")
    if (
        not isinstance(candidate_name, str)
        or Path(candidate_name).name != candidate_name
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise Phase4F21StageAError("F19 candidate filenames are invalid")
    candidate_path = workspace / SOURCE_RUN / "checkpoints" / candidate_name
    checkpoint_path = workspace / SOURCE_RUN / "checkpoints" / checkpoint_name
    config_sha256 = manifest.payload.get("provenance", {}).get("config", {}).get("sha256")
    source_sha256 = manifest.payload.get("provenance", {}).get("source", {}).get("sha256")
    expected = {
        "candidateModelSha256": best.get("candidateModelSha256"),
        "checkpointSha256": best.get("checkpointSha256"),
        "configSha256": config_sha256,
        "generation": best.get("generation"),
        "modelStateSha256": best.get("modelStateSha256"),
        "runManifestSha256": manifest.sha256,
        "sourceSha256": source_sha256,
        "step": best.get("step"),
    }
    if (
        sha256_file(candidate_path) != expected["candidateModelSha256"]
        or sha256_file(checkpoint_path) != expected["checkpointSha256"]
    ):
        raise Phase4F21StageAError("F19 candidate files failed SHA-256 validation")
    files = {
        "candidateModel": {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": expected["candidateModelSha256"],
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": expected["checkpointSha256"],
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


def _cap_payload() -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4-artifact-cap-extension",
            "status": "frozen",
            "authorizationBasis": "explicit-user-confirmation-2026-08-09",
            "historicalEightGiBContractsUnchanged": True,
            "previousDefaultArtifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
            "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            "scope": {
                "stage": PHASE4F_STAGE_A_CONTINUATION,
                "configPath": CONFIG.as_posix(),
                "runDir": RUN_DIR.as_posix(),
            },
        },
        "capContractSha256",
    )


def _config_payload(
    workspace: Path,
    archive: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    payload = _mapping(workspace / SOURCE_CONFIG, "F19 training config")
    payload.update(
        {
            "name": "phase4f-stage-a-block-01-v1",
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
    return payload


def _execution_payload(*, baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_A_CONTINUATION,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f21-final-classic-hard-guided-two-generation-block",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": STAGE_GPU_HOURS,
            "baselineRunId": "phase4f20-stage-a-screen-gate",
            "baselineReportSha256": EXPECTED_F20_GATE_SHA256,
        },
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return config


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
        raise Phase4F21StageAError("seeded F21 base-model hash changed")
    state, provenance = load_phase4_recovery_artifact(
        workspace_root=workspace,
        config=config,
    )
    network.load_state_dict(state, strict=True)
    if canonical_model_state_sha256(network) != config.initialization.model_state_sha256:
        raise Phase4F21StageAError("F21 source model changed after load")
    return {"baseModelSha256": base_sha256, **provenance}


def prepare_phase4f21(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    history = _verify_historical_evidence(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F21StageAError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + STAGE_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F21StageAError("insufficient global GPU budget for F21")
    run_dir = workspace / RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise Phase4F21StageAError("F21 run directory is not empty")

    archive, source = _source_candidate(workspace)
    cap = _cap_payload()
    config_payload = _config_payload(workspace, archive, source)
    execution_payload = _execution_payload(baseline_hours=ledger.total_hours)
    for relative, payload in (
        (ARCHIVE_MANIFEST, archive),
        (CAP_EXTENSION, cap),
        (CONFIG, config_payload),
        (EXECUTION, execution_payload),
    ):
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(workspace / relative, canonical_json_bytes(payload))

    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    initialization = _verify_initialization(workspace, config)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f21-final-stage-a-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "formalStageATrainingAuthorized": True,
            "additionalStageATrainingAuthorized": False,
            "stageBTrainingAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "historicalEvidence": history,
            "sourceCandidate": source,
            "initialization": initialization,
            "lineage": {
                "sourceGeneration": 1,
                "sourceStep": 192,
                "optimizerInherited": False,
                "selfPlayReplayInherited": False,
                "screenTrajectoriesImported": False,
                "seedChanged": True,
            },
            "artifactCapExtension": {
                "path": CAP_EXTENSION.as_posix(),
                "fileSha256": sha256_file(workspace / CAP_EXTENSION),
                "capContractSha256": cap["capContractSha256"],
                "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
                "historicalEightGiBContractsUnchanged": True,
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
                "limitHours": STAGE_GPU_HOURS,
            },
            "selection": {
                "criterion": "strict-minimum-fixed-health-holdout-total-loss",
                "baselineTotalLoss": 1.9608344801517223,
                "requiresStrictImprovementForF22Screen": True,
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "exactly two completed 40-percent guided generations",
                "fixed health holdout selection",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    atomic_replace_bytes(workspace / PREPARATION_REPORT, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F21 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F21 preparation")
    cap = _mapping(workspace / CAP_EXTENSION, "artifact cap extension")
    _verify_self_hash(cap, "capContractSha256", "artifact cap extension")
    archive = _mapping(workspace / ARCHIVE_MANIFEST, "F21 source archive")
    _verify_self_hash(archive, "archiveSha256", "F21 source archive")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("formalStageATrainingAuthorized") is not True
        or report.get("stageBTrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("artifactCapExtension", {}).get("fileSha256")
        != sha256_file(workspace / CAP_EXTENSION)
        or report.get("archive", {}).get("fileSha256")
        != sha256_file(workspace / ARCHIVE_MANIFEST)
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256")
        != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256")
        != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
        or cap.get("artifactLimitBytes") != MAXIMUM_ARTIFACT_LIMIT_BYTES
        or cap.get("historicalEightGiBContractsUnchanged") is not True
        or config.artifact_limit_bytes != MAXIMUM_ARTIFACT_LIMIT_BYTES
        or execution.stage != PHASE4F_STAGE_A_CONTINUATION
    ):
        raise Phase4F21StageAError("Phase 4F21 preparation is stale")
    _verify_historical_evidence(workspace)
    validate_phase4_config(config, execution)
    _verify_initialization(workspace, config)
    _resources(workspace)
    return report


def verify_phase4f21_training_baseline(
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
        or execution_contract.stage != PHASE4F_STAGE_A_CONTINUATION
        or fingerprint_config(config).sha256
        != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F21StageAError("F21 workload differs from CPU preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "f20GateSha256": EXPECTED_F20_GATE_SHA256,
        "artifactCapContractSha256": report["artifactCapExtension"][
            "capContractSha256"
        ],
        "formalStageATrainingAuthorized": True,
        "stageBTrainingAuthorized": False,
    }


def review_phase4f21(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    preparation = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    reports = GenerationReportStore(
        workspace / RUN_DIR / "reports", run_manifest_sha256=manifest.sha256
    )
    if reports.generations() != (0, 1):
        raise Phase4F21StageAError("F21 must contain exactly generations 0 and 1")
    generations: list[dict[str, object]] = []
    for generation in reports.generations():
        record = reports.load(generation)
        data = record.payload.get("report", {}).get("generationData")
        if (
            not isinstance(data, Mapping)
            or data.get("generation") != generation
            or data.get("selfPlay", {}).get("games") != 128
            or not isinstance(data.get("fixedHealthHoldout"), Mapping)
            or data.get("training", {}).get("learningStage") != "classic-hard-guided"
            or data.get("training", {}).get("challengeReplay", {}).get("fraction") != 0.4
        ):
            raise Phase4F21StageAError(f"generation {generation} failed F21 gates")
        generations.append(
            {
                "generation": generation,
                "reportPath": record.path.relative_to(workspace).as_posix(),
                "reportSha256": record.sha256,
                "fixedHealthHoldout": data["fixedHealthHoldout"],
                "selfPlayGames": data["selfPlay"]["games"],
            }
        )
    best = reconcile_best_fixed_holdout_candidate(workspace / RUN_DIR, reports)
    if not isinstance(best, Mapping) or best.get("generation") not in (0, 1):
        raise Phase4F21StageAError("F21 best candidate is unavailable")
    best_total = float(best["selection"]["bestTotalLoss"])
    baseline_total = float(preparation["selection"]["baselineTotalLoss"])
    improved = best_total < baseline_total
    provenance = manifest.payload.get("provenance", {})
    if (
        provenance.get("source", {}).get("sha256") != preparation["source"]["sha256"]
        or provenance.get("config", {}).get("sha256")
        != preparation["config"]["semanticConfigSha256"]
        or manifest.payload.get("resourceBudget", {}).get("artifactLimitBytes")
        != MAXIMUM_ARTIFACT_LIMIT_BYTES
    ):
        raise Phase4F21StageAError("F21 run lineage differs from preparation")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f21-final-stage-a-two-generation-review",
            "status": "completed",
            "stageAPassed": True,
            "stageAImproved": improved,
            "additionalStageATrainingAuthorized": False,
            "stageBTrainingAuthorized": False,
            "f22ScreenAuthorized": improved,
            "preparationReportSha256": preparation["reportSha256"],
            "memorySmoke": memory,
            "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
            "runManifestSha256": manifest.sha256,
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
            "baselineBest": {
                "candidate": preparation["historicalEvidence"]["f19BestCandidate"],
                "totalLoss": baseline_total,
            },
            "generations": generations,
            "bestCandidate": dict(best),
            "comparison": {
                "criterion": "strict-minimum-fixed-health-holdout-total-loss",
                "baselineTotalLoss": baseline_total,
                "f21BestTotalLoss": best_total,
                "strictImprovement": improved,
            },
            "globalGpuHours": GpuBudgetLedger(
                workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
            ).total_hours,
            "nextStage": {
                "name": (
                    "Phase 4F22 screen best F21 candidate"
                    if improved
                    else "Stage A exhausted without a new screen candidate"
                ),
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
        payload = prepare_phase4f21(args.workspace_root)
    elif args.command == "verify-preparation":
        payload = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        payload = review_phase4f21(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
