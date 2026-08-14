"""Freeze and review the formal two-generation Phase 4F Stage A block."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE, PHASE4F_STAGE_A
from .phase4_execution import (
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_FROZEN,
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_gates import find_valid_memory_smoke
from .phase4_training import (
    reconcile_best_fixed_holdout_candidate,
    validate_phase4_config,
)
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
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
F18_PREPARATION = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f18-rebaseline-report.json"
)
F18_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f18-pilot-review.json"
)
CONFIG = Path("training/configs/phase4f18-stage-a-block-00-v1.json")
EXECUTION = Path("training/configs/phase4f19-stage-a-execution-frozen-v1.json")
RUN_DIR = Path("training/runs/phase4/phase4f/recovery-v3-stage-a-block-00-v1")
PREPARATION_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f19-stage-a-freeze-report.json"
)
REVIEW_REPORT = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f19-stage-a-review.json"
)
EXPECTED_F18_PREPARATION_SHA256 = (
    "ac8c447ae7ba32b64d71465ca524ba2484f54f12a557c0f0f0df3ba019748e90"
)
EXPECTED_F18_REVIEW_SHA256 = (
    "5068db9a47e164d8fffdc47494c850dcaa545026a53db6dd8e71c22e2f9aa98b"
)
MAXIMUM_GENERATIONS = 2
STAGE_GPU_HOURS = 2.0


class Phase4F19StageAError(RuntimeError):
    """Raised when formal Stage A evidence or output is invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F19StageAError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F19StageAError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F19StageAError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F19StageAError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4F19StageAError("Phase 4 artifacts reached the 8 GiB cap")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F19StageAError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _verify_f18_evidence(workspace: Path) -> dict[str, object]:
    preparation = _mapping(workspace / F18_PREPARATION, "Phase 4F18 preparation")
    preparation_sha256 = _verify_self_hash(
        preparation, "reportSha256", "Phase 4F18 preparation"
    )
    review = _mapping(workspace / F18_REVIEW, "Phase 4F18 pilot review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "Phase 4F18 pilot review")
    pilot = review.get("pilot")
    memory = review.get("memorySmoke")
    gates = memory.get("gates") if isinstance(memory, Mapping) else None
    if (
        preparation_sha256 != EXPECTED_F18_PREPARATION_SHA256
        or review_sha256 != EXPECTED_F18_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("pilotPassed") is not True
        or review.get("formalStageATrainingAuthorized") is not False
        or review.get("preparationReportSha256") != preparation_sha256
        or not isinstance(pilot, Mapping)
        or not isinstance(pilot.get("fixedHealthHoldout"), Mapping)
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise Phase4F19StageAError("Phase 4F18 pilot evidence is not eligible")
    return {
        "preparationPath": F18_PREPARATION.as_posix(),
        "preparationSha256": preparation_sha256,
        "reviewPath": F18_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "pilotGenerationReportSha256": pilot.get("generationReportSha256"),
        "pilotMemorySmokeReportSha256": memory.get("reportCoreSha256"),
    }


def _execution_payload(*, baseline_hours: float) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "stage": PHASE4F_STAGE_A,
        "status": PHASE4_EXECUTION_STATUS_FROZEN,
        "purpose": "phase4f19-stage-a-classic-hard-guided-two-generation-block",
        "formalPromotion": False,
        "maximumGenerations": MAXIMUM_GENERATIONS,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_FROZEN,
            "baselineGlobalHours": baseline_hours,
            "limitHours": STAGE_GPU_HOURS,
            "baselineRunId": "phase4f18-stage-a-cuda-pilot-review",
            "baselineReportSha256": EXPECTED_F18_REVIEW_SHA256,
        },
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return config


def prepare_phase4f19(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    pilot = _verify_f18_evidence(workspace)
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F19StageAError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + STAGE_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F19StageAError("insufficient global GPU budget for Stage A")
    run_dir = workspace / RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise Phase4F19StageAError("formal Stage A run directory is not empty")

    atomic_replace_bytes(
        workspace / EXECUTION,
        canonical_json_bytes(_execution_payload(baseline_hours=ledger.total_hours)),
    )
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    config = _effective_config(workspace)
    validate_phase4_config(config, execution)
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f19-stage-a-formal-freeze",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "formalStageATrainingAuthorized": True,
            "stageBTrainingAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "pilotEvidence": pilot,
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
            "independentLineage": {
                "pilotReplayInherited": False,
                "pilotOptimizerInherited": False,
                "recoveryGeneration": 11,
                "resetOptimizer": True,
            },
            "requiredRuntimeGates": [
                "current-source 192x16 batch-512 CUDA AMP memory smoke",
                "current-source CPU replay input gate",
                "exactly two completed guided generations",
                "fixed health holdout selection",
            ],
            "resources": resources,
        },
        "reportSha256",
    )
    path = workspace / PREPARATION_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, canonical_json_bytes(report))
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / PREPARATION_REPORT, "Phase 4F19 preparation")
    _verify_self_hash(report, "reportSha256", "Phase 4F19 preparation")
    config = _effective_config(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        report.get("status") != "prepared"
        or report.get("formalStageATrainingAuthorized") is not True
        or report.get("stageBTrainingAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("nativeBackendSha256") != loaded_native_backend_hash()
        or report.get("config", {}).get("fileSha256") != sha256_file(workspace / CONFIG)
        or report.get("config", {}).get("semanticConfigSha256")
        != fingerprint_config(config).sha256
        or report.get("execution", {}).get("fileSha256")
        != sha256_file(workspace / EXECUTION)
        or report.get("execution", {}).get("sha256") != execution.sha256
    ):
        raise Phase4F19StageAError("Phase 4F19 preparation is stale")
    _verify_f18_evidence(workspace)
    validate_phase4_config(config, execution)
    return report


def verify_phase4f19_training_baseline(
    *,
    workspace_root: str | Path,
    execution_contract: Phase4ExecutionContract,
    config: ExperimentConfig,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    report = _verified_preparation(workspace)
    expected = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution_contract.stage != PHASE4F_STAGE_A
        or execution_contract != expected
        or fingerprint_config(config).sha256
        != report["config"]["semanticConfigSha256"]
    ):
        raise Phase4F19StageAError("formal Stage A workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["config"]["semanticConfigSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "pilotReviewSha256": EXPECTED_F18_REVIEW_SHA256,
        "formalStageATrainingAuthorized": True,
        "stageBTrainingAuthorized": False,
    }


def review_phase4f19(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    preparation = _verified_preparation(workspace)
    memory = find_valid_memory_smoke(workspace / PHASE4_CONTROL_RELATIVE, workspace)
    manifest_path = workspace / RUN_DIR / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    reports = GenerationReportStore(
        workspace / RUN_DIR / "reports", run_manifest_sha256=manifest.sha256
    )
    if reports.generations() != (0, 1):
        raise Phase4F19StageAError("formal Stage A must contain exactly generations 0 and 1")
    generation_evidence: list[dict[str, object]] = []
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
            raise Phase4F19StageAError(f"generation {generation} failed Stage A gates")
        generation_evidence.append(
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
        raise Phase4F19StageAError("best Stage A candidate is unavailable")
    provenance = manifest.payload.get("provenance", {})
    if (
        provenance.get("source", {}).get("sha256") != preparation["source"]["sha256"]
        or provenance.get("config", {}).get("sha256")
        != preparation["config"]["semanticConfigSha256"]
    ):
        raise Phase4F19StageAError("formal run lineage differs from preparation")
    review = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f19-stage-a-two-generation-review",
            "status": "completed",
            "stageAPassed": True,
            "stageBTrainingAuthorized": False,
            "preparationReportSha256": preparation["reportSha256"],
            "memorySmoke": memory,
            "runManifestPath": manifest_path.relative_to(workspace).as_posix(),
            "runManifestSha256": manifest.sha256,
            "generations": generation_evidence,
            "bestCandidate": dict(best),
            "globalGpuHours": GpuBudgetLedger(
                workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
            ).total_hours,
            "nextStage": {
                "name": "Phase 4F20 screen best Stage A candidate against classic-hard",
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
        result = prepare_phase4f19(args.workspace_root)
    elif args.command == "verify-preparation":
        result = _verified_preparation(Path(args.workspace_root).resolve())
    else:
        result = review_phase4f19(args.workspace_root)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
