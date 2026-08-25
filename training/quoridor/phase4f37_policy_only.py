"""Review F36 and freeze the isolated policy-only Stage A2 pilot."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .config import ExperimentConfig, load_experiment_config, seed_everything
from .network import NetworkConfig, PolicyValueNetwork
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import (
    PHASE4_CONTROL_RELATIVE,
    PHASE4F38_PILOT_STAGE,
)
from .phase4_execution import (
    Phase4ExecutionContract,
    load_phase4_execution_contract,
)
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4_training import validate_phase4_config
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
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
F36_SOURCE_GIT_COMMIT = "8cd9129a8946c0758edd6afd5d146f88bf1d0169"
F36_SOURCE_SHA256 = "d52cbecdf05d51af9de8d874b0d678b79a1a2d6747215f41645277f57d9c8173"
F36_RUN_DIR = Path("training/runs/phase4/phase4f/stage-a2-pilot-v2")
F36_MANIFEST_SHA256 = "945c2344b7679bbd758e03c4e7f20b2bf04e5ac344983bfed992e1e9eb403697"
F36_REPORT_SHA256 = "953bd7b21363a2d6d2789516790958b91c22048f4c47554d9c44754ee8823753"
F36_CANDIDATE_SHA256 = "94cc1c3655c62ce1e4f2e2bb99567b07ed569f5a2a6ba58d4df4f701f6cc2014"
F36_CHECKPOINT_SHA256 = "a9041891cb277f6606f2215bcb0dd2d5449ab2063f53a582446c7725a6af5d95"
F21_RUN_DIR = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1"
)
F21_MANIFEST_SHA256 = "96fc1f063bbbab0716da03228578d2eb23b062e708823ed68ebd0a8ba77f2bac"
F21_REPORT_SHA256 = "d85b8ea361afb706991a18c943ead9c8477b93563cdbc4aaaee6a2921ea5f8e7"
F21_CANDIDATE_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
F36_CONFIG = Path("training/configs/phase4f36-stage-a2-pilot-v2.json")
ROOT = Path("training/runs/phase4/phase4f/f37-policy-only-preparation-v2")
REPORT = ROOT / "preparation-report.json"
CONFIG = Path("training/configs/phase4f38-stage-a2-policy-only-pilot-v2.json")
EXECUTION = Path(
    "training/configs/phase4f38-stage-a2-policy-only-pilot-execution-frozen-v1.json"
)
RUN_DIR = Path("training/runs/phase4/phase4f/stage-a2-policy-only-pilot-v2")
PILOT_SEED = 20260821
PILOT_GPU_HOURS = 2.0
ARTIFACT_RESERVE_BYTES = 1250 * 1024**2
EXPECTED_TRAINING_STEPS = 96
EXPECTED_CHALLENGE_POSITIONS = 256
EXPECTED_EARLY_POSITIONS = 154


class Phase4F37Error(RuntimeError):
    """Raised when the F37 review or policy-only freeze is inconsistent."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F37Error(f"{label} is unreadable") from error
    if path.is_symlink() or not isinstance(payload, Mapping):
        raise Phase4F37Error(f"{label} is invalid")
    if raw != canonical_json_bytes(payload):
        raise Phase4F37Error(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F37Error(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise Phase4F37Error(f"existing {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, encoded)


def _verify_historical_source(workspace: Path, source: Mapping[str, object]) -> None:
    files = source.get("files")
    if (
        source.get("sha256") != F36_SOURCE_SHA256
        or not isinstance(files, list)
        or source.get("fileCount") != len(files)
    ):
        raise Phase4F37Error("F36 source inventory is invalid")
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise Phase4F37Error("F36 source record is invalid")
        relative = str(record["path"])
        try:
            archived = subprocess.run(
                ["git", "show", f"{F36_SOURCE_GIT_COMMIT}:{relative}"],
                cwd=workspace,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise Phase4F37Error(
                f"historical F36 Git source is unavailable: {relative}"
            ) from error
        if (
            len(archived) != record.get("bytes")
            or sha256_bytes(archived) != record.get("sha256")
        ):
            raise Phase4F37Error(f"historical F36 source differs: {relative}")
        total_bytes += len(archived)
    if total_bytes != source.get("bytes"):
        raise Phase4F37Error("F36 source byte count changed")


def _generation_data(
    workspace: Path,
    run_dir: Path,
    manifest_sha256: str,
    generation: int,
    report_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_run_manifest(workspace / run_dir / "run-manifest.json")
    if manifest.sha256 != manifest_sha256:
        raise Phase4F37Error(f"{run_dir.name} run manifest changed")
    report = GenerationReportStore(
        workspace / run_dir / "reports",
        run_manifest_sha256=manifest.sha256,
    ).load(generation)
    if report.sha256 != report_sha256:
        raise Phase4F37Error(f"{run_dir.name} generation report changed")
    wrapped = report.payload.get("report")
    data = wrapped.get("generationData") if isinstance(wrapped, Mapping) else None
    checkpoint = wrapped.get("checkpoint") if isinstance(wrapped, Mapping) else None
    if not isinstance(data, Mapping) or not isinstance(checkpoint, Mapping):
        raise Phase4F37Error(f"{run_dir.name} generation data is invalid")
    return dict(manifest.payload), {**dict(data), "checkpoint": dict(checkpoint)}


def _losses(payload: Mapping[str, object]) -> dict[str, object]:
    holdout = payload.get("fixedHealthHoldout")
    if not isinstance(holdout, Mapping):
        raise Phase4F37Error("fixed health holdout is missing")
    bands = holdout.get("byPlyBand")
    sides = holdout.get("bySideToMove")
    if not isinstance(bands, Mapping) or not isinstance(sides, Mapping):
        raise Phase4F37Error("fixed health holdout strata are missing")
    return {
        "overall": {
            key: float(holdout[key])
            for key in ("policyLoss", "valueLoss", "totalLoss")
        },
        "byPlyBand": {
            band: {key: float(bands[band][key]) for key in ("policyLoss", "valueLoss", "totalLoss")}
            for band in ("early", "middle", "late")
        },
        "bySideToMove": {
            side: {key: float(sides[side][key]) for key in ("policyLoss", "valueLoss", "totalLoss")}
            for side in ("0", "1")
        },
    }


def _delta(current: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        key: float(current[key]) - float(baseline[key])
        for key in ("policyLoss", "valueLoss", "totalLoss")
    }
    result["totalLossPercent"] = (
        (float(current["totalLoss"]) / float(baseline["totalLoss"])) - 1.0
    ) * 100.0
    return result


def _audit_f36(workspace: Path) -> dict[str, object]:
    manifest, f36 = _generation_data(
        workspace, F36_RUN_DIR, F36_MANIFEST_SHA256, 0, F36_REPORT_SHA256
    )
    source = manifest.get("provenance", {}).get("source")
    if not isinstance(source, Mapping):
        raise Phase4F37Error("F36 source provenance is missing")
    _verify_historical_source(workspace, source)
    _, f21 = _generation_data(
        workspace, F21_RUN_DIR, F21_MANIFEST_SHA256, 1, F21_REPORT_SHA256
    )
    checkpoint = f36.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise Phase4F37Error("F36 checkpoint record is missing")
    candidate = checkpoint.get("candidateModel")
    if (
        checkpoint.get("sha256") != F36_CHECKPOINT_SHA256
        or not isinstance(candidate, Mapping)
        or candidate.get("sha256") != F36_CANDIDATE_SHA256
        or sha256_file(workspace / F36_RUN_DIR / "checkpoints" / str(checkpoint["path"]))
        != F36_CHECKPOINT_SHA256
        or sha256_file(workspace / F36_RUN_DIR / "checkpoints" / str(candidate["path"]))
        != F36_CANDIDATE_SHA256
    ):
        raise Phase4F37Error("F36 model artifacts changed")
    f21_candidate = f21.get("checkpoint", {}).get("candidateModel")
    if not isinstance(f21_candidate, Mapping) or f21_candidate.get("sha256") != F21_CANDIDATE_SHA256:
        raise Phase4F37Error("F21 rollback candidate changed")

    training = f36.get("training")
    challenge = training.get("challengeReplay") if isinstance(training, Mapping) else None
    sampling = challenge.get("sampling") if isinstance(challenge, Mapping) else None
    batches = sampling.get("batches") if isinstance(sampling, Mapping) else None
    if (
        not isinstance(batches, list)
        or len(batches) != EXPECTED_TRAINING_STEPS
        or challenge.get("fraction") != 0.5
        or challenge.get("valueTargetMode", "all-samples") != "all-samples"
    ):
        raise Phase4F37Error("F36 challenge replay summary is invalid")
    for batch in batches:
        if (
            not isinstance(batch, Mapping)
            or batch.get("positions") != EXPECTED_CHALLENGE_POSITIONS
            or batch.get("earlyPositions") != EXPECTED_EARLY_POSITIONS
            or batch.get("sideToMove") != {"0": 128, "1": 128}
            or batch.get("actionKind") != {"move": 128, "wall": 128}
        ):
            raise Phase4F37Error("F36 stratified batch contract was not met")
    selfplay = f36.get("selfPlay")
    if (
        not isinstance(selfplay, Mapping)
        or selfplay.get("games") != 128
        or selfplay.get("moves") != 5473
        or selfplay.get("terminationReasons") != {"goal": 128}
    ):
        raise Phase4F37Error("F36 self-play summary is invalid")

    baseline = _losses(f21)
    current = _losses(f36)
    comparison = {
        "overall": _delta(current["overall"], baseline["overall"]),
        "byPlyBand": {
            band: _delta(current["byPlyBand"][band], baseline["byPlyBand"][band])
            for band in ("early", "middle", "late")
        },
        "bySideToMove": {
            side: _delta(current["bySideToMove"][side], baseline["bySideToMove"][side])
            for side in ("0", "1")
        },
    }
    eligible = (
        float(comparison["overall"]["totalLoss"]) < 0.0
        and float(comparison["byPlyBand"]["early"]["totalLoss"]) < 0.0
    )
    if eligible:
        raise Phase4F37Error("F36 unexpectedly satisfies the frozen quality gate")
    return {
        "status": "rejected",
        "candidateEligible": False,
        "screenAuthorized": False,
        "reasons": [
            "fixed-health-holdout-total-loss-regressed",
            "late-value-loss-regressed",
        ],
        "f21": {
            "runManifestSha256": F21_MANIFEST_SHA256,
            "generationReportSha256": F21_REPORT_SHA256,
            "candidateModelSha256": F21_CANDIDATE_SHA256,
            "losses": baseline,
        },
        "f36": {
            "runManifestSha256": F36_MANIFEST_SHA256,
            "generationReportSha256": F36_REPORT_SHA256,
            "candidateModelSha256": F36_CANDIDATE_SHA256,
            "checkpointSha256": F36_CHECKPOINT_SHA256,
            "sourceSha256": F36_SOURCE_SHA256,
            "selfPlayGames": 128,
            "selfPlayMoves": 5473,
            "stratifiedBatches": len(batches),
            "losses": current,
        },
        "comparison": comparison,
    }


def _ledger_snapshot(workspace: Path) -> dict[str, object]:
    relative = PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    path = workspace / relative
    ledger = GpuBudgetLedger(path)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F37Error("a running GPU ledger entry blocks F37 preparation")
    return {
        "path": relative.as_posix(),
        "fileSha256": sha256_file(path),
        "totalGpuHours": ledger.total_hours,
        "externalHoursFloor": ledger.external_hours_floor,
        "runs": {key: dict(value) for key, value in sorted(ledger.runs.items())},
    }


def _verify_ledger_append_only(
    workspace: Path, snapshot: Mapping[str, object], maximum_hours: float
) -> float:
    current = GpuBudgetLedger(workspace / str(snapshot["path"]))
    if any(record.get("status") == "running" for record in current.runs.values()):
        raise Phase4F37Error("a running GPU ledger entry blocks F37 verification")
    runs = snapshot.get("runs")
    if not isinstance(runs, Mapping) or any(
        current.runs.get(key) != value for key, value in runs.items()
    ):
        raise Phase4F37Error("GPU ledger is not append-only from F37 preparation")
    if not float(snapshot["totalGpuHours"]) <= current.total_hours <= maximum_hours:
        raise Phase4F37Error("F38 pilot GPU-hour envelope was exceeded")
    return current.total_hours


def _resources(workspace: Path) -> dict[str, int]:
    artifacts = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free = shutil.disk_usage(workspace).free
    if artifacts + ARTIFACT_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F37Error("16 GiB cap cannot preserve the F38 pilot reserve")
    if free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F37Error("less than 20 GiB free disk remains")
    return {
        "artifactBytes": artifacts,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_RESERVE_BYTES,
        "freeDiskBytes": free,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _effective_config(workspace: Path) -> ExperimentConfig:
    config = load_experiment_config(workspace / CONFIG)
    return replace(config, run_dir=(workspace / config.run_dir).resolve())


def _base_model_sha256(seed: int) -> str:
    seed_everything(seed)
    return canonical_model_state_sha256(
        PolicyValueNetwork(
            NetworkConfig(input_channels=16, channels=192, blocks=16)
        )
    )


def prepare_phase4f37(workspace_root: str | Path) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    audit = _audit_f36(workspace)
    snapshot = _ledger_snapshot(workspace)
    execution = load_phase4_execution_contract(workspace / EXECUTION)
    if (
        execution.stage != PHASE4F38_PILOT_STAGE
        or execution.maximum_generations != 1
        or execution.formal_promotion
        or execution.stage_budget.baseline_run_id != "phase4f36-stage-a2-pilot-v2"
        or execution.stage_budget.baseline_report_sha256 != F36_REPORT_SHA256
        or execution.stage_budget.limit_hours != PILOT_GPU_HOURS
    ):
        raise Phase4F37Error("F38 execution contract is not the frozen pilot contract")
    maximum_global_hours = (
        execution.stage_budget.baseline_global_hours
        + execution.stage_budget.limit_hours
    )
    if float(snapshot["totalGpuHours"]) > maximum_global_hours:
        raise Phase4F37Error("the frozen F38 pilot GPU-hour envelope was exceeded")
    if maximum_global_hours > 120.0:
        raise Phase4F37Error("the global 120 GPU-hour budget cannot fit F38")

    config_payload = _mapping(workspace / F36_CONFIG, "F36 config")
    config_payload["name"] = "phase4f38-stage-a2-policy-only-pilot-v2"
    config_payload["runDir"] = RUN_DIR.as_posix()
    config_payload["seed"] = PILOT_SEED
    config_payload["artifactLimitBytes"] = EXTENDED_ARTIFACT_LIMIT_BYTES
    initialization = config_payload.get("initialization")
    if not isinstance(initialization, dict):
        raise Phase4F37Error("F36 initialization config is missing")
    initialization["expectedBaseModelSha256"] = _base_model_sha256(PILOT_SEED)
    challenge = config_payload.get("training", {}).get("challengeReplay")
    if not isinstance(challenge, dict):
        raise Phase4F37Error("F36 challenge config is missing")
    challenge["valueTargetMode"] = "self-play-only"
    _write_immutable(workspace / CONFIG, config_payload, "F38 pilot config")
    config = _effective_config(workspace)
    validate_phase4_config(config, execution)
    if (
        config.training.challenge_replay is None
        or config.training.challenge_replay.value_target_mode != "self-play-only"
        or config.artifact_limit_bytes != EXTENDED_ARTIFACT_LIMIT_BYTES
        or config.initialization.artifact_sha256 != F21_CANDIDATE_SHA256
    ):
        raise Phase4F37Error("F38 policy-only workload is not isolated")

    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f37-f36-review-and-policy-only-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "cudaPilotAuthorized": False,
            "formalPromotionAuthorized": False,
            "source": fingerprint_sources(workspace).to_payload(),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "f36Review": audit,
            "strategy": {
                "initialization": "phase4f21-stage-a-best-with-empty-optimizer-and-replay",
                "teacherPolicyTargets": True,
                "teacherValueTargets": False,
                "selfPlayPolicyTargets": True,
                "selfPlayValueTargets": True,
                "terminalRewardChanged": False,
                "challengeReplayFraction": 0.5,
                "minimumEarlyFractionWithinChallenge": 0.6,
                "selection": "strict-minimum-fixed-health-holdout-total-loss",
            },
            "gpuLedger": snapshot,
            "pilot": {
                "configPath": CONFIG.as_posix(),
                "configFileSha256": sha256_file(workspace / CONFIG),
                "semanticConfigSha256": fingerprint_config(config).sha256,
                "executionPath": EXECUTION.as_posix(),
                "executionFileSha256": sha256_file(workspace / EXECUTION),
                "executionSha256": execution.sha256,
                "maximumGenerations": 1,
                "stageGpuHours": PILOT_GPU_HOURS,
                "maximumGlobalHours": maximum_global_hours,
            },
            "resources": _resources(workspace),
            "nextStage": {
                "name": "Phase 4F38 run corrected isolated policy-only Stage A2 pilot",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "reportSha256",
    )
    _write_immutable(workspace / REPORT, report, "F37 preparation report")
    return report


def _verified_preparation(workspace: Path) -> dict[str, object]:
    report = _mapping(workspace / REPORT, "F37 preparation report")
    _verify_self_hash(report, "reportSha256", "F37 preparation report")
    if (
        report.get("status") != "prepared"
        or report.get("cudaPilotAuthorized") is not False
        or report.get("formalPromotionAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
    ):
        raise Phase4F37Error("F37 preparation is stale")
    maximum = float(report["pilot"]["maximumGlobalHours"])
    _verify_ledger_append_only(workspace, report["gpuLedger"], maximum)
    for path, key in ((CONFIG, "configFileSha256"), (EXECUTION, "executionFileSha256")):
        if sha256_file(workspace / path) != report["pilot"][key]:
            raise Phase4F37Error("F37 frozen pilot artifact changed")
    return report


def verify_phase4f37_training_baseline(
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
        or fingerprint_config(config).sha256 != report["pilot"]["semanticConfigSha256"]
    ):
        raise Phase4F37Error("F38 policy-only workload differs from preparation")
    return {
        "status": "verified",
        "sourceSha256": report["source"]["sha256"],
        "nativeBackendSha256": report["nativeBackendSha256"],
        "semanticConfigSha256": report["pilot"]["semanticConfigSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "formalStageA2TrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-preparation"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    workspace = args.workspace_root.resolve()
    report = (
        prepare_phase4f37(workspace)
        if args.command == "prepare"
        else _verified_preparation(workspace)
    )
    print(
        canonical_json_bytes(
            {
                "status": report["status"],
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f36CandidateEligible": report["f36Review"]["candidateEligible"],
                "artifactLimitBytes": report["resources"]["artifactLimitBytes"],
                "nextStage": report["nextStage"],
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
