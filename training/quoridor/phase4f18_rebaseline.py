"""CPU rebaseline and isolated CUDA-pilot gates for staged Phase 4F recovery."""

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
    PHASE4F18_PILOT_STAGE,
    PHASE4F_STAGE_A,
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
    validate_phase4_config,
)
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
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
F17_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "staged-learning-report.json"
)
DATASET_CONTRACT = Path(
    "training/configs/phase4f17-classic-guidance-dataset-frozen-v1.json"
)
PROTOCOL = Path(
    "training/configs/phase4f17-two-stage-protocol-frozen-v1.json"
)
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f18-rebaseline-report.json"
)
PILOT_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f18-pilot-review.json"
)
PILOT_CONFIG = Path("training/configs/phase4f18-stage-a-pilot-v1.json")
STAGE_A_CONFIG = Path("training/configs/phase4f18-stage-a-block-00-v1.json")
PILOT_EXECUTION = Path(
    "training/configs/phase4f18-stage-a-pilot-execution-frozen-v1.json"
)
STAGE_A_EXECUTION = Path(
    "training/configs/phase4f18-stage-a-execution-prepared-v1.json"
)
PILOT_RUN_DIR = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-pilot-v1"
)
STAGE_A_RUN_DIR = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-00-v1"
)
ARCHIVE_MANIFEST = Path(
    "training/runs/phase4/phase4f/recovery-evaluation-v3/"
    "generation-11-archive-manifest.json"
)
ARCHIVE_ARTIFACT = Path(
    "training/runs/phase4/phase4f/recovery-evaluation-v3/"
    "generation-11-archive/candidate-model-checkpoint-"
    "g00000011-s000000001152.pt"
)
EXPECTED_BASE_MODEL_SHA256 = (
    "87e87ec6f4f77b6156d56e095b1c23685d10d33cf14038ca4623b8a89d0b0120"
)
PILOT_GPU_HOURS = 2.0
PILOT_MAXIMUM_GENERATIONS = 1
STAGE_A_MAXIMUM_GENERATIONS = 2


class Phase4F18RebaselineError(RuntimeError):
    """Raised when staged recovery preparation or pilot evidence is unsafe."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F18RebaselineError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F18RebaselineError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F18RebaselineError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(
    payload: Mapping[str, object], field: str, label: str
) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(
        canonical_json_bytes(core)
    ):
        raise Phase4F18RebaselineError(f"{label} self hash mismatch")
    return stored


def _self_hashed(
    core: Mapping[str, object], field: str
) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resource_preflight(workspace: Path) -> dict[str, int]:
    control = workspace / PHASE4_CONTROL_RELATIVE
    artifact_bytes = directory_size(control)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4F18RebaselineError("Phase 4 artifacts reached the 8 GiB cap")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F18RebaselineError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_f17(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / F17_REPORT, "Phase 4F17 report")
    report_sha256 = _verify_self_hash(
        report, "reportSha256", "Phase 4F17 report"
    )
    dataset = _mapping(workspace / DATASET_CONTRACT, "dataset contract")
    protocol = _mapping(workspace / PROTOCOL, "two-stage protocol")
    dataset_sha256 = _verify_self_hash(
        dataset, "contractSha256", "dataset contract"
    )
    protocol_sha256 = _verify_self_hash(
        protocol, "protocolSha256", "two-stage protocol"
    )
    if (
        report.get("status") != "completed"
        or report.get("trainingAuthorized") is not False
        or report.get("cpuOnly") is not True
        or report.get("cudaAccessed") is not False
        or report.get("datasetAudit", {}).get("evaluationOverlap") != 0
        or report.get("datasetAudit", {}).get("trainHoldoutOverlap") != 0
        or report.get("datasetContract", {}).get("contractSha256")
        != dataset_sha256
        or report.get("twoStageProtocol", {}).get("protocolSha256")
        != protocol_sha256
        or dataset.get("trainingUseAllowed") is not True
        or protocol.get("stageA", {}).get("maximumGenerationsPerMicroBlock")
        != STAGE_A_MAXIMUM_GENERATIONS
        or protocol.get("stageA", {}).get("challengeReplayFraction") != 0.4
        or protocol.get("rewardContract", {}).get("classicHardRewardShaping")
        is not False
    ):
        raise Phase4F18RebaselineError("Phase 4F17 evidence is not eligible")
    return {
        "path": F17_REPORT.as_posix(),
        "fileSha256": sha256_file(workspace / F17_REPORT),
        "reportSha256": report_sha256,
        "datasetContractSha256": dataset_sha256,
        "protocolSha256": protocol_sha256,
    }


def _initialization() -> dict[str, object]:
    return {
        "mode": "recovery",
        "expectedBaseModelSha256": EXPECTED_BASE_MODEL_SHA256,
        "artifactPath": ARCHIVE_ARTIFACT.as_posix(),
        "artifactSha256": (
            "019fa6a2fd5a2b4615de8eea892e0f040ad8ecc55637a5896ef4167b0449e556"
        ),
        "recoveryManifestPath": ARCHIVE_MANIFEST.as_posix(),
        "recoveryManifestSha256": (
            "655a22ac6d7cc7d130f0f409be5dfa4e55d4129d28b18974a3e3d096dddb26d2"
        ),
        "recoveryManifestFileSha256": (
            "199e416eecff64ca3519e26823f09d600f0d6672f38b0830935e2317b6c25c32"
        ),
        "recoverySourceSha256": (
            "ed8e6f9262c3e132f6771043de646ee6d32872abc7c0c6a10da4c958f3f242fa"
        ),
        "recoveryConfigSha256": (
            "5bc576f8287071b10cde82b9a2d5bf8937205050e0ec0998cd43f7c41018e86f"
        ),
        "recoveryRunManifestSha256": (
            "ac9b6495a7084f426df459905e24945f931652f37fbce877b3843ba75bf0edc2"
        ),
        "recoveryCheckpointSha256": (
            "d8220407b528ab9c0c50a782df86231f22643a0ca4646953cfc29ca1a636f257"
        ),
        "recoveryGeneration": 11,
        "recoveryStep": 1152,
        "modelStateSha256": (
            "85bc74fbf76cc01b76c63b6d0688644be9735ba1ba4ccc77ab08788cc79162e7"
        ),
        "resetOptimizer": True,
    }


def _dataset_training() -> dict[str, object]:
    common = {
        "root": "training/runs/phase4/phase4f/recovery-challenge-v2-data",
        "contractPath": DATASET_CONTRACT.as_posix(),
        "contractSha256": (
            "bbf62d4a84eb032c24e64314670587e0fea0a73a3f473c908f0ca57007e4ef36"
        ),
        "datasetFingerprint": (
            "fa3342b1e4775fb2e8af8b03ec3b53e14bcdf9d9f7b67230cc587999aa1bd581"
        ),
        "datasetManifestSha256": (
            "ffcc44797622cb73aeaea26025a8458d25c1caf79c20c94d15e933fd1f0576b1"
        ),
        "trainManifestSha256": (
            "e4556e9b4132de80c83f153b8a31180fa87e70c1f96c761a4c4a2be3d47ab98c"
        ),
    }
    return {
        "challengeReplay": {**common, "fraction": 0.4},
        "healthHoldout": {
            **common,
            "holdoutManifestSha256": (
                "61f8ec9d9a5927ee40adc330748c305f9fa0b4a091f296b4d7e88d7a532c4a83"
            ),
        },
    }


def _config_payload(*, name: str, run_dir: Path) -> dict[str, object]:
    dataset = _dataset_training()
    return {
        "pipeline": "phase4",
        "name": name,
        "seed": 20260803,
        "device": "cuda",
        "runDir": run_dir.as_posix(),
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "initialization": _initialization(),
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
            "learningRate": 0.0001,
            "minimumLearningRate": 0.00001,
            "weightDecay": 0.0001,
            "trainingSteps": 96,
            "maxCheckpoints": 3,
            "maxGradNorm": 5.0,
            "useAmp": True,
            "scheduleSteps": 20000,
            "learningStage": "classic-hard-guided",
            **dataset,
        },
        "arena": {"games": 64, "simulations": 800},
    }


def _prepared_execution() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "stage": PHASE4F_STAGE_A,
        "status": "prepared",
        "purpose": "phase4f-stage-a-classic-hard-guided-micro-block",
        "formalPromotion": False,
        "maximumGenerations": STAGE_A_MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": "pending",
            "baselineGlobalHours": None,
            "limitHours": None,
            "baselineRunId": None,
            "baselineReportSha256": None,
        },
    }


def _pilot_execution(
    *, baseline_hours: float, f17_report_sha256: str
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "stage": PHASE4F18_PILOT_STAGE,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f18-stage-a-isolated-cuda-pilot",
        "formalPromotion": False,
        "maximumGenerations": PILOT_MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": PILOT_GPU_HOURS,
            "baselineRunId": "phase4f17-staged-recovery-cpu-preparation",
            "baselineReportSha256": f17_report_sha256,
        },
    }


def _effective_config(workspace: Path, path: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / path)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return config


def _verify_base_and_recovery(
    workspace: Path, config: ExperimentConfig
) -> dict[str, object]:
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
        raise Phase4F18RebaselineError("seeded base-model hash changed")
    state, provenance = load_phase4_recovery_artifact(
        workspace_root=workspace,
        config=config,
    )
    network.load_state_dict(state, strict=True)
    if canonical_model_state_sha256(network) != _initialization()["modelStateSha256"]:
        raise Phase4F18RebaselineError("recovery model state changed after load")
    return {"baseModelSha256": base_sha256, **provenance}


def prepare_phase4f18(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    f17 = _verify_f17(workspace)
    resources = _resource_preflight(workspace)
    ledger = GpuBudgetLedger(
        workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    )
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F18RebaselineError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + PILOT_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F18RebaselineError("insufficient global GPU budget for pilot")

    artifacts = {
        PILOT_CONFIG: _config_payload(
            name="phase4f18-stage-a-pilot-v1", run_dir=PILOT_RUN_DIR
        ),
        STAGE_A_CONFIG: _config_payload(
            name="phase4f-stage-a-block-00-v1", run_dir=STAGE_A_RUN_DIR
        ),
        PILOT_EXECUTION: _pilot_execution(
            baseline_hours=ledger.total_hours,
            f17_report_sha256=str(f17["reportSha256"]),
        ),
        STAGE_A_EXECUTION: _prepared_execution(),
    }
    for relative, payload in artifacts.items():
        atomic_replace_bytes(workspace / relative, canonical_json_bytes(payload))

    pilot_execution = load_phase4_execution_contract(workspace / PILOT_EXECUTION)
    stage_a_execution = load_phase4_execution_contract(workspace / STAGE_A_EXECUTION)
    pilot_config = _effective_config(workspace, PILOT_CONFIG)
    stage_a_config = _effective_config(workspace, STAGE_A_CONFIG)
    validate_phase4_config(pilot_config, pilot_execution)
    validate_phase4_config(stage_a_config, stage_a_execution)
    recovery = _verify_base_and_recovery(workspace, pilot_config)
    if fingerprint_config(pilot_config).sha256 == fingerprint_config(stage_a_config).sha256:
        raise Phase4F18RebaselineError("pilot and formal config identities collided")

    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f18-stage-a-source-rebaseline-and-pilot-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "cudaPilotAuthorized": True,
            "formalStageATrainingAuthorized": False,
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "baselineGlobalGpuHours": ledger.total_hours,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "phase4f17": f17,
            "recoveryInitialization": recovery,
            "pilot": {
                "configPath": PILOT_CONFIG.as_posix(),
                "configFileSha256": sha256_file(workspace / PILOT_CONFIG),
                "semanticConfigSha256": fingerprint_config(pilot_config).sha256,
                "executionPath": PILOT_EXECUTION.as_posix(),
                "executionFileSha256": sha256_file(workspace / PILOT_EXECUTION),
                "executionSha256": pilot_execution.sha256,
                "maximumGenerations": PILOT_MAXIMUM_GENERATIONS,
                "stageGpuHours": PILOT_GPU_HOURS,
            },
            "stageA": {
                "configPath": STAGE_A_CONFIG.as_posix(),
                "configFileSha256": sha256_file(workspace / STAGE_A_CONFIG),
                "semanticConfigSha256": fingerprint_config(stage_a_config).sha256,
                "executionPath": STAGE_A_EXECUTION.as_posix(),
                "executionFileSha256": sha256_file(workspace / STAGE_A_EXECUTION),
                "executionSha256": stage_a_execution.sha256,
                "maximumGenerations": STAGE_A_MAXIMUM_GENERATIONS,
                "budgetStatus": "pending-cuda-pilot",
            },
            "requiredPilotGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "one isolated generation with zero illegal actions",
                "40-percent classic-hard mixed input gate",
                "fixed health holdout evaluation",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    report_path = workspace / PREPARATION_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(report_path, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F18 report")
    _verify_self_hash(report, "reportSha256", "Phase 4F18 report")
    if (
        report.get("status") != "prepared"
        or report.get("cpuOnly") is not True
        or report.get("cudaPilotAuthorized") is not True
        or report.get("formalStageATrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
    ):
        raise Phase4F18RebaselineError("Phase 4F18 preparation is stale")
    for key, config_path, execution_path in (
        ("pilot", PILOT_CONFIG, PILOT_EXECUTION),
        ("stageA", STAGE_A_CONFIG, STAGE_A_EXECUTION),
    ):
        evidence = report.get(key)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("configFileSha256") != sha256_file(workspace / config_path)
            or evidence.get("executionFileSha256")
            != sha256_file(workspace / execution_path)
        ):
            raise Phase4F18RebaselineError(f"Phase 4F18 {key} artifact changed")
    return report


def verify_phase4f18_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    if execution_contract.stage != PHASE4F18_PILOT_STAGE:
        raise Phase4F18RebaselineError(
            "formal Stage A remains unauthorized until the CUDA pilot is reviewed"
        )
    pilot = report["pilot"]
    expected_execution = load_phase4_execution_contract(workspace / PILOT_EXECUTION)
    if (
        execution_contract != expected_execution
        or fingerprint_config(config).sha256 != pilot["semanticConfigSha256"]
    ):
        raise Phase4F18RebaselineError("pilot workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": pilot["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "globalGpuHourLimit": GPU_HOUR_BUDGET,
        "formalStageATrainingAuthorized": False,
    }


def review_phase4f18_pilot(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(
        workspace / PHASE4_CONTROL_RELATIVE,
        workspace,
    )
    manifest_path = workspace / PILOT_RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    generation_path = workspace / PILOT_RUN_DIR / "reports/generation-00000000.json"
    generation = _mapping(generation_path, "pilot generation report")
    generation_sha256 = _verify_self_hash(
        generation, "reportSha256", "pilot generation report"
    )
    data = generation.get("report", {}).get("generationData")
    if (
        manifest.payload.get("provenance", {}).get("source", {}).get("sha256")
        != report["source"]["sha256"]
        or manifest.payload.get("provenance", {}).get("config", {}).get("sha256")
        != report["pilot"]["semanticConfigSha256"]
        or not isinstance(data, Mapping)
        or data.get("generation") != 0
        or data.get("selfPlay", {}).get("games") != 128
        or not isinstance(data.get("fixedHealthHoldout"), Mapping)
        or data.get("training", {})
        .get("challengeReplay", {})
        .get("fraction")
        != 0.4
    ):
        raise Phase4F18RebaselineError("pilot generation did not pass staged gates")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f18-stage-a-cuda-pilot-review",
            "status": "completed",
            "pilotPassed": True,
            "formalStageATrainingAuthorized": False,
            "preparationReportSha256": report["reportSha256"],
            "memorySmoke": memory,
            "pilot": {
                "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
                "runManifestSha256": manifest.sha256,
                "generationReportPath": generation_path.relative_to(workspace).as_posix(),
                "generationReportSha256": generation_sha256,
                "fixedHealthHoldout": data["fixedHealthHoldout"],
            },
            "globalGpuHours": GpuBudgetLedger(
                workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
            ).total_hours,
            "nextStage": {
                "name": "Phase 4F19 freeze and run Stage A two-generation block",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "reviewSha256",
    )
    atomic_replace_bytes(workspace / PILOT_REVIEW, canonical_json_bytes(review))
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "verify-preparation", "review-pilot"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_phase4f18(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        result = review_phase4f18_pilot(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


__all__ = [
    "PILOT_CONFIG",
    "PILOT_EXECUTION",
    "PILOT_REVIEW",
    "PREPARATION_REPORT",
    "STAGE_A_CONFIG",
    "STAGE_A_EXECUTION",
    "Phase4F18RebaselineError",
    "prepare_phase4f18",
    "review_phase4f18_pilot",
    "verify_phase4f18_training_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
