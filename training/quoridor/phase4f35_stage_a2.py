"""Prepare and verify the stratified Stage A2 one-generation CUDA pilot."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_sampler import StratifiedChallengeSampler
from .config import (
    ChallengeSamplingSettings,
    ExperimentConfig,
    load_experiment_config,
    seed_everything,
)
from .network import NetworkConfig, PolicyValueNetwork
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F35_PILOT_STAGE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_training import (
    load_phase4_recovery_artifact,
    validate_phase4_config,
)
from .replay_store import ReplayStore
from .replay_tensor_cache import ReplayTensorCache
from .run_manifest import (
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
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
F34_REPORT = Path(
    "training/runs/phase4/phase4f/f34-failure-review-v1/preparation-report.json"
)
F34_REPORT_SHA256 = "c00ecb38994a41467fb6d821888d039cab904c95548d911511e9d4b0765d22b0"
F34_PROTOCOL = Path("training/configs/phase4f34-stage-a2-protocol-frozen-v1.json")
F34_PROTOCOL_SHA256 = "6d008e952e6f2a07cb1260cfbce67a09cabc9b183c3c2115e6af7a30bae24f55"
SOURCE_RUN = Path("training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1")
SOURCE_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
DATA_ROOT = Path("training/runs/phase4/phase4f/recovery-challenge-v2-data")
OLD_CHALLENGE_CACHE = SOURCE_RUN / "challenge-replay-tensor-cache"
ROOT = Path("training/runs/phase4/phase4f/f35-stage-a2-preparation-v1")
INDEX = ROOT / "guidance-strata-index.json"
ARCHIVE = ROOT / "f21-source-archive.json"
REPORT = ROOT / "preparation-report.json"
CONFIG = Path("training/configs/phase4f35-stage-a2-pilot-v1.json")
EXECUTION = Path("training/configs/phase4f35-stage-a2-pilot-execution-frozen-v1.json")
RUN_DIR = Path("training/runs/phase4/phase4f/stage-a2-pilot-v1")
SEED = 20260820
PILOT_GPU_HOURS = 2.0
ARTIFACT_RESERVE_BYTES = 1250 * 1024**2


class Phase4F35Error(RuntimeError):
    """Raised when Stage A2 preparation is not reproducible."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F35Error(f"{label} is unreadable") from error
    if path.is_symlink() or not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F35Error(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F35Error(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise Phase4F35Error(f"existing {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, encoded)


def _historical_evidence(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    report = _mapping(workspace / F34_REPORT, "F34 report")
    if (
        _verify_self_hash(report, "reportSha256", "F34 report") != F34_REPORT_SHA256
        or report.get("trainingAuthorized") is not False
        or report.get("pilotAuthorized") is not False
    ):
        raise Phase4F35Error("F34 report does not authorize F35 engineering")
    protocol = _mapping(workspace / F34_PROTOCOL, "F34 Stage A2 protocol")
    if (
        _verify_self_hash(protocol, "protocolSha256", "F34 protocol")
        != F34_PROTOCOL_SHA256
        or protocol.get("stageA2", {}).get("selfPlayGamesPerGeneration") != 128
        or protocol.get("classicGuidance", {}).get("challengeReplayFraction") != 0.5
    ):
        raise Phase4F35Error("F34 Stage A2 protocol changed")
    return report, protocol


def _source_candidate(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    best = _mapping(workspace / SOURCE_RUN / "best-candidate.json", "F21 best candidate")
    manifest = load_run_manifest(workspace / SOURCE_RUN / "run-manifest.json")
    candidate_name = best.get("candidateModelPath")
    checkpoint_name = best.get("checkpointPath")
    if not isinstance(candidate_name, str) or not isinstance(checkpoint_name, str):
        raise Phase4F35Error("F21 candidate filenames are invalid")
    candidate_path = workspace / SOURCE_RUN / "checkpoints" / candidate_name
    checkpoint_path = workspace / SOURCE_RUN / "checkpoints" / checkpoint_name
    provenance = manifest.payload.get("provenance", {})
    source = {
        "candidateModelSha256": best.get("candidateModelSha256"),
        "checkpointSha256": best.get("checkpointSha256"),
        "configSha256": provenance.get("config", {}).get("sha256"),
        "generation": best.get("generation"),
        "modelStateSha256": best.get("modelStateSha256"),
        "runManifestSha256": manifest.sha256,
        "sourceSha256": provenance.get("source", {}).get("sha256"),
        "step": best.get("step"),
        "candidatePath": candidate_path.relative_to(workspace).as_posix(),
        "checkpointPath": checkpoint_path.relative_to(workspace).as_posix(),
    }
    if (
        source["candidateModelSha256"]
        != "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
        or source["checkpointSha256"]
        != "dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b"
        or source["modelStateSha256"]
        != "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
        or source["generation"] != 1
        or source["step"] != 192
        or sha256_file(candidate_path) != source["candidateModelSha256"]
        or sha256_file(checkpoint_path) != source["checkpointSha256"]
    ):
        raise Phase4F35Error("F21 rollback candidate differs")
    files = {
        "candidateModel": {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": source["candidateModelSha256"],
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": source["checkpointSha256"],
        },
    }
    archive = _self_hashed(
        {
            "schemaVersion": 1,
            "kind": "phase4-recovery-candidate-archive",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "candidate": {key: source[key] for key in (
                "candidateModelSha256", "checkpointSha256", "configSha256",
                "generation", "modelStateSha256", "runManifestSha256",
                "sourceSha256", "step",
            )},
            "files": files,
            "bytes": sum(int(value["bytes"]) for value in files.values()),
        },
        "archiveSha256",
    )
    return archive, source


def _sampling_settings(index_sha256: str) -> ChallengeSamplingSettings:
    return ChallengeSamplingSettings(
        algorithm="stratified-opening-depth-and-action-kind-v1",
        early_opening_depths=(8, 12),
        minimum_early_fraction=0.6,
        balance_side_to_move=True,
        balance_action_kind=("move", "wall"),
        index_path=INDEX,
        index_sha256=index_sha256,
    )


def _build_index(workspace: Path) -> dict[str, object]:
    manifest = _mapping(workspace / DATA_ROOT / "train/manifest.json", "guidance manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F35Error("guidance capacity is invalid")
    store = ReplayStore(
        workspace / DATA_ROOT / "train",
        max_games=int(capacity["games"]),
        max_plies=int(capacity["plies"]),
        max_bytes=int(capacity["bytes"]),
    )
    old_manifest = _mapping(workspace / OLD_CHALLENGE_CACHE / "manifest.json", "F21 challenge cache")
    contract = old_manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise Phase4F35Error("F21 challenge cache contract is invalid")
    cache = ReplayTensorCache(
        workspace / OLD_CHALLENGE_CACHE,
        store,
        source_sha256=str(contract["sourceSha256"]),
        encoder_sha256=str(contract["encoderSha256"]),
    )
    sampler = StratifiedChallengeSampler(store, cache, _sampling_settings("0" * 64))
    sample = sampler.sample_plan(256, rng=random.Random(SEED))
    if (
        sample.summary.get("earlyPositions") != 154
        or sample.summary.get("sideToMove") != {"0": 128, "1": 128}
        or sample.summary.get("actionKind") != {"move": 128, "wall": 128}
    ):
        raise Phase4F35Error("real guidance data failed the frozen 256-sample quota")
    _write_immutable(workspace / INDEX, sampler.index_payload, "guidance strata index")
    return {"index": sampler.index_payload, "sample": dict(sample.summary)}


def _base_model_sha256() -> str:
    seed_everything(SEED)
    return canonical_model_state_sha256(
        PolicyValueNetwork(NetworkConfig(input_channels=16, channels=192, blocks=16))
    )


def _config_payload(
    archive: Mapping[str, object], source: Mapping[str, object], index_sha256: str
) -> dict[str, object]:
    common = {
        "root": DATA_ROOT.as_posix(),
        "contractPath": "training/configs/phase4f17-classic-guidance-dataset-frozen-v1.json",
        "contractSha256": "bbf62d4a84eb032c24e64314670587e0fea0a73a3f473c908f0ca57007e4ef36",
        "datasetFingerprint": "fa3342b1e4775fb2e8af8b03ec3b53e14bcdf9d9f7b67230cc587999aa1bd581",
        "datasetManifestSha256": "ffcc44797622cb73aeaea26025a8458d25c1caf79c20c94d15e933fd1f0576b1",
        "trainManifestSha256": "e4556e9b4132de80c83f153b8a31180fa87e70c1f96c761a4c4a2be3d47ab98c",
    }
    return {
        "pipeline": "phase4", "name": "phase4f35-stage-a2-pilot-v1", "seed": SEED,
        "device": "cuda", "runDir": RUN_DIR.as_posix(),
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "initialization": {
            "mode": "recovery", "expectedBaseModelSha256": _base_model_sha256(),
            "artifactPath": source["candidatePath"], "artifactSha256": source["candidateModelSha256"],
            "recoveryManifestPath": ARCHIVE.as_posix(), "recoveryManifestSha256": archive["archiveSha256"],
            "recoveryManifestFileSha256": sha256_bytes(canonical_json_bytes(archive)),
            "recoverySourceSha256": source["sourceSha256"], "recoveryConfigSha256": source["configSha256"],
            "recoveryRunManifestSha256": source["runManifestSha256"],
            "recoveryCheckpointSha256": source["checkpointSha256"],
            "recoveryGeneration": source["generation"], "recoveryStep": source["step"],
            "modelStateSha256": source["modelStateSha256"], "resetOptimizer": True,
        },
        "network": {"inputChannels":16,"channels":192,"residualBlocks":16,"seReduction":8,"policyChannels":32,"valueChannels":32,"valueHidden":256},
        "mcts": {"simulations":512,"cBase":19652,"cInit":1.25,"dirichletEpsilon":0.25,"dirichletAlpha":None},
        "selfPlay": {"games":128,"temperatureMoves":24,"actors":64},
        "training": {
            "replayCapacity":300000,"batchSize":512,"learningRate":0.00005,
            "minimumLearningRate":0.000005,"weightDecay":0.0001,"trainingSteps":96,
            "maxCheckpoints":3,"maxGradNorm":5.0,"useAmp":True,"scheduleSteps":20000,
            "learningStage":"classic-hard-guided",
            "challengeReplay": {**common,"fraction":0.5,"stratifiedSampling":{
                "algorithm":"stratified-opening-depth-and-action-kind-v1",
                "earlyOpeningDepths":[8,12],"minimumEarlyFraction":0.6,
                "balanceSideToMove":True,"balanceActionKind":["move","wall"],
                "indexPath":INDEX.as_posix(),"indexSha256":index_sha256,
            }},
            "healthHoldout": {**common,"holdoutManifestSha256":"61f8ec9d9a5927ee40adc330748c305f9fa0b4a091f296b4d7e88d7a532c4a83"},
        },
        "arena": {"games":64,"simulations":800},
    }


def _execution_payload(baseline: float) -> dict[str, object]:
    return {
        "schemaVersion":1,"stage":PHASE4F35_PILOT_STAGE,"status":PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose":"phase4f35-stage-a2-stratified-guidance-isolated-pilot",
        "formalPromotion":False,"maximumGenerations":1,
        "stageBudget":{"status":PHASE4_BUDGET_STATUS_FROZEN,"baselineGlobalHours":baseline,
            "limitHours":PILOT_GPU_HOURS,"baselineRunId":"phase4f34-failure-review",
            "baselineReportSha256":F34_REPORT_SHA256},
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    return replace(config, run_dir=(workspace / config.run_dir).resolve())


def _resources(workspace: Path) -> dict[str, int]:
    artifact = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free = shutil.disk_usage(workspace).free
    if artifact + ARTIFACT_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F35Error("12 GiB cap cannot preserve the Stage A2 pilot reserve")
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F35Error("less than 20 GiB free disk remains")
    return {"artifactBytes":artifact,"artifactLimitBytes":MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes":ARTIFACT_RESERVE_BYTES,"freeDiskBytes":free,
        "minimumFreeDiskBytes":DEFAULT_MINIMUM_FREE_DISK_BYTES}


def prepare_phase4f35(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    f34, _ = _historical_evidence(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(value.get("status") == "running" for value in ledger.runs.values()):
        raise Phase4F35Error("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + PILOT_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F35Error("insufficient global GPU budget for the pilot")
    archive, source = _source_candidate(workspace)
    index = _build_index(workspace)
    _write_immutable(workspace / ARCHIVE, archive, "F21 source archive")
    config_payload = _config_payload(archive, source, str(index["index"]["indexSha256"]))
    execution_payload = _execution_payload(ledger.total_hours)
    _write_immutable(workspace / CONFIG, config_payload, "Stage A2 pilot config")
    _write_immutable(workspace / EXECUTION, execution_payload, "Stage A2 execution contract")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    validate_phase4_config(config, execution)
    state, recovery = load_phase4_recovery_artifact(workspace_root=workspace, config=config)
    if canonical_model_state_sha256(state) != source["modelStateSha256"]:
        raise Phase4F35Error("F21 recovery model state differs")
    report = _self_hashed({
        "schemaVersion":1,"kind":"phase4f35-stage-a2-sampler-and-pilot-preparation",
        "status":"prepared","cpuOnly":True,"cudaAccessed":False,
        "cudaPilotAuthorized":False,"formalStageA2TrainingAuthorized":False,
        "globalGpuHours":ledger.total_hours,"globalGpuHourLimit":GPU_HOUR_BUDGET,
        "source":fingerprint_sources(workspace).to_payload(),
        "nativeBackendSha256":loaded_native_backend_hash(),
        "phase4f34":{"path":F34_REPORT.as_posix(),"reportSha256":f34["reportSha256"]},
        "rollback":recovery,
        "samplingIndex":{"path":INDEX.as_posix(),"indexSha256":index["index"]["indexSha256"],
            "inventory":index["index"]["inventory"],"sampleAudit":index["sample"]},
        "pilot":{"configPath":CONFIG.as_posix(),"configFileSha256":sha256_file(workspace/CONFIG),
            "semanticConfigSha256":fingerprint_config(config).sha256,
            "executionPath":EXECUTION.as_posix(),"executionFileSha256":sha256_file(workspace/EXECUTION),
            "executionSha256":execution.sha256,"maximumGenerations":1,"stageGpuHours":PILOT_GPU_HOURS},
        "resources":_resources(workspace),
        "nextStage":{"name":"Phase 4F36 run and review isolated Stage A2 CUDA pilot","requiresSeparateUserConfirmation":True},
    },"reportSha256")
    _write_immutable(workspace / REPORT, report, "F35 preparation report")
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / REPORT, "F35 report")
    _verify_self_hash(report,"reportSha256","F35 report")
    if (report.get("status") != "prepared" or report.get("cudaPilotAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()):
        raise Phase4F35Error("F35 preparation is stale")
    if report.get("globalGpuHours") != GpuBudgetLedger(workspace/PHASE4_CONTROL_RELATIVE/GPU_LEDGER_NAME).total_hours:
        raise Phase4F35Error("GPU ledger changed after F35 preparation")
    for path, key, section in ((CONFIG,"configFileSha256","pilot"),(EXECUTION,"executionFileSha256","pilot")):
        if report[section][key] != sha256_file(workspace/path):
            raise Phase4F35Error("F35 frozen pilot artifact changed")
    index = _mapping(workspace / INDEX,"F35 sampling index")
    if _verify_self_hash(index,"indexSha256","F35 sampling index") != report["samplingIndex"]["indexSha256"]:
        raise Phase4F35Error("F35 sampling index changed")
    return report


def verify_phase4f35_training_baseline(*, workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract, config: ExperimentConfig) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if execution_contract != expected or fingerprint_config(config).sha256 != report["pilot"]["semanticConfigSha256"]:
        raise Phase4F35Error("Stage A2 pilot workload differs from F35 preparation")
    return {"status":"verified","sourceSha256":report["source"]["sha256"],
        "nativeBackendSha256":report["nativeBackendSha256"],
        "semanticConfigSha256":report["pilot"]["semanticConfigSha256"],
        "reportPath":REPORT.as_posix(),"reportSha256":report["reportSha256"],
        "formalStageA2TrainingAuthorized":False}


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify-preparation"))
    parser.add_argument("--workspace-root",type=Path,default=Path.cwd())
    args=parser.parse_args(argv)
    result=prepare_phase4f35(args.workspace_root) if args.command=="prepare" else _verified_preparation(Path(args.workspace_root).resolve())
    print(canonical_json_bytes({"status":result["status"],"reportPath":REPORT.as_posix(),
        "reportSha256":result["reportSha256"],"globalGpuHours":result["globalGpuHours"],
        "samplingIndexSha256":result["samplingIndex"]["indexSha256"],
        "cudaPilotAuthorized":result["cudaPilotAuthorized"],"nextStage":result["nextStage"]}).decode("ascii"),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
