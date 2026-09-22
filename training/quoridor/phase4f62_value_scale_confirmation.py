"""Prepare, run, and verify the F21 value-scale confirmation arena."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import arena_position_orbits
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS
from .network import NetworkConfig, PolicyValueNetwork
from .openings import (
    OpeningSuite,
    load_opening_suite,
    opening_position_orbit_digest,
    opening_suite_to_payload,
    validate_suite_collection,
)
from .paired_arena import BoundMoveSelector, SelectorIdentity, loaded_native_backend_hash
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    FROZEN_SUITES,
    REPLAY_ROOTS,
    _arena_references,
    _load_generator,
    _mapping,
    _relative,
    _replay_orbits,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f61_value_scale_screen import (
    EXPECTED_F21_CONFIG_SHA256,
    EXPECTED_F21_MODEL_SHA256,
    EXPECTED_F21_STATE_SHA256,
    F21_CONFIG,
    F21_MODEL,
    ValueScaledEvaluator,
    _load_f21_state,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .selectors import MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f62-f21-value-scale-confirmation-v1")
SUITE_PATH = ROOT / "fresh-value-scale-confirmation-200.json"
ISOLATION_AUDIT = ROOT / "isolation-audit.json"
CONTRACT = ROOT / "execution-contract.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
STORAGE_AMENDMENT = ROOT / "storage-amendment-v1.json"
ARENA_ROOT = ROOT / "confirmation-arena"
GATE = ROOT / "confirmation-gate.json"

F59_POOL = Path(
    "training/runs/phase4/phase4f/f59-kl-trust-region-review-v1/"
    "f60-fresh-lineage-pool-200.json"
)
F60_RESULT = Path(
    "training/runs/phase4/phase4f/f60-kl-lineage-screen-v1/screen-arena/result.json"
)
F61_GATE = Path(
    "training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-gate.json"
)
F61_RESULT = Path(
    "training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/"
    "screen-arena/result.json"
)

EXPECTED_F61_GATE_SHA256 = (
    "35ab6ddba6a387ecca274b6592f39c5427b8957000737172345de429abf9891d"
)
EXPECTED_F61_RESULT_SHA256 = (
    "d59263c33e52e1b8d54f4f1f5c171ad56cb6dd2740a626ac2b243ac17faeb8fb"
)

SUITE_NAME = "recovery-fresh-dev-v1"
OPENING_PAIRS = 200
GAMES = 400
SIMULATIONS = 800
ACTORS = 32
MCTS_SEED = 20260918
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
CANDIDATE_VALUE_SCALE = 0.5
OPPONENT_VALUE_SCALE = 1.0
CONFIRMATION_GPU_HOURS = 6.0
AMENDED_CONFIRMATION_GPU_HOURS = 7.5
WRITE_RESERVE_BYTES = 128 * 1024**2
MINIMUM_LOWER_BOUND = 0.50

SOURCE_PATHS = (
    "scripts/generate_phase4_openings.py",
    "training/quoridor/challenge_replay.py",
    "training/quoridor/evaluator.py",
    "training/quoridor/native_mcts.py",
    "training/quoridor/network.py",
    "training/quoridor/openings.py",
    "training/quoridor/paired_arena.py",
    "training/quoridor/phase4_arena.py",
    "training/quoridor/phase4f29_fresh_screen.py",
    "training/quoridor/phase4f61_value_scale_screen.py",
    "training/quoridor/phase4f62_value_scale_confirmation.py",
    "training/quoridor/selectors.py",
)


class Phase4F62Error(RuntimeError):
    """Raised when F62 preparation, execution, or verification is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F62Error("workspace does not contain training/quoridor")
    return workspace


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F62Error("F62 reserve exceeds the frozen 16 GiB artifact cap")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F62Error("F62 reserve violates the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _f61_evidence(workspace: Path) -> dict[str, object]:
    gate = _mapping(workspace / F61_GATE, "F61 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F61 gate")
    result = verify_completed_arena_result(workspace / F61_RESULT)
    if (
        gate_sha != EXPECTED_F61_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "value-scale-screen-positive"
        or gate.get("confirmationArenaCandidate") is not True
        or gate.get("candidateValueScale") != CANDIDATE_VALUE_SCALE
        or gate.get("opponentValueScale") != OPPONENT_VALUE_SCALE
        or gate.get("resultSha256") != EXPECTED_F61_RESULT_SHA256
        or result.get("resultSha256") != EXPECTED_F61_RESULT_SHA256
        or result.get("statistics") != gate.get("statistics")
    ):
        raise Phase4F62Error("F61 evidence is not eligible for confirmation")
    return {
        "gatePath": F61_GATE.as_posix(),
        "gateSha256": gate_sha,
        "resultPath": F61_RESULT.as_posix(),
        "resultFileSha256": sha256_file(workspace / F61_RESULT),
        "resultSha256": EXPECTED_F61_RESULT_SHA256,
        "statistics": gate["statistics"],
    }


def _fresh_confirmation_suite(
    workspace: Path,
) -> tuple[OpeningSuite, dict[str, object]]:
    frozen: list[OpeningSuite] = []
    suite_refs: list[dict[str, object]] = []
    for relative, name in FROZEN_SUITES:
        path = workspace / relative
        suite = load_opening_suite(path, expected_suite=name)
        frozen.append(suite)
        suite_refs.append(
            {
                "path": relative.as_posix(),
                "name": name,
                "fileSha256": sha256_file(path),
                "suiteHash": suite.content_hash,
                "openingPairs": len(suite.openings),
            }
        )
    validate_suite_collection(*frozen)

    pool = load_opening_suite(workspace / F59_POOL, expected_suite=SUITE_NAME)
    forbidden = {
        opening_position_orbit_digest(opening)
        for suite in (*frozen, pool)
        for opening in suite.openings
    }
    pool_ref = {
        "path": F59_POOL.as_posix(),
        "fileSha256": sha256_file(workspace / F59_POOL),
        "suiteHash": pool.content_hash,
        "openingPairs": len(pool.openings),
    }

    replay_refs: list[dict[str, object]] = []
    for relative in REPLAY_ROOTS:
        print(
            canonical_json_bytes(
                {
                    "kind": "phase4f62-preparation-progress",
                    "stage": "replay-isolation",
                    "path": relative.as_posix(),
                    "status": "started",
                }
            ).decode("ascii"),
            end="",
            flush=True,
        )
        try:
            orbits, reference = _replay_orbits(workspace, relative)
        except Exception as error:
            raise Phase4F62Error(f"cannot verify excluded replay {relative}") from error
        forbidden.update(orbits)
        replay_refs.append({**reference, "uniquePositionOrbits": len(orbits)})

    arena_refs = _arena_references(workspace)
    arena_orbits: set[str] = set()
    for reference in arena_refs:
        arena_orbits.update(arena_position_orbits(workspace, reference))
    forbidden.update(arena_orbits)
    historical_paths = {str(reference["path"]) for reference in arena_refs}
    if F60_RESULT.as_posix() not in historical_paths:
        raise Phase4F62Error("F60 trajectories are absent from the exclusion set")
    if F61_RESULT.as_posix() not in historical_paths:
        raise Phase4F62Error("F61 trajectories are absent from the exclusion set")

    fresh = _load_generator(workspace).generate_fresh_development_suite(
        tuple(frozen), forbidden_orbits=forbidden
    )
    fresh_orbits = {
        opening_position_orbit_digest(opening) for opening in fresh.openings
    }
    overlap = len(fresh_orbits & forbidden)
    if len(fresh.openings) != OPENING_PAIRS or len(fresh_orbits) != OPENING_PAIRS:
        raise Phase4F62Error("fresh F62 confirmation suite is incomplete")
    if overlap:
        raise Phase4F62Error("fresh F62 suite overlaps excluded position orbits")
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f62-value-scale-confirmation-isolation-audit",
            "status": "completed",
            "frozenSuites": suite_refs,
            "f59FullPool": pool_ref,
            "replayStores": replay_refs,
            "historicalScreenResults": arena_refs,
            "historicalScreenPositionOrbits": len(arena_orbits),
            "totalExcludedPositionOrbits": len(forbidden),
            "excludedPositionOrbitsSha256": sha256_bytes(
                canonical_json_bytes(sorted(forbidden))
            ),
            "freshOpeningPositionOrbits": len(fresh_orbits),
            "freshOverlapCount": overlap,
            "confirmationOpeningPairs": len(fresh.openings),
            "f59FullPoolExcluded": True,
            "f60TrajectoriesExcluded": True,
            "f61TrajectoriesExcluded": True,
        },
        "auditSha256",
    )
    return fresh, audit


def _stored_suite(workspace: Path) -> tuple[OpeningSuite, dict[str, object]]:
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    audit = _mapping(workspace / ISOLATION_AUDIT, "F62 isolation audit")
    _verify_self_hash(audit, "auditSha256", "F62 isolation audit")
    current_orbits = {
        opening_position_orbit_digest(opening) for opening in suite.openings
    }
    if (
        len(suite.openings) != OPENING_PAIRS
        or len(current_orbits) != OPENING_PAIRS
        or audit.get("kind")
        != "phase4f62-value-scale-confirmation-isolation-audit"
        or audit.get("freshOverlapCount") != 0
        or audit.get("confirmationOpeningPairs") != OPENING_PAIRS
        or audit.get("f59FullPoolExcluded") is not True
        or audit.get("f60TrajectoriesExcluded") is not True
        or audit.get("f61TrajectoriesExcluded") is not True
    ):
        raise Phase4F62Error("stored F62 opening evidence is invalid")
    return suite, audit


def _runtime_binding(
    value_scale: float, source_sha: str, native_sha: str
) -> dict[str, object]:
    return _self_hashed(
        {
            "kind": "native-mcts",
            "sourceSha256": source_sha,
            "nativeBackendSha256": native_sha,
            "configuredSimulations": SIMULATIONS,
            "valueScale": value_scale,
            "policyTransform": "identity",
            "rootNoise": False,
            "temperature": 0.0,
            "mctsSeed": MCTS_SEED,
        },
        "runtimeSha256",
    )


def _identities(
    contract: Mapping[str, object]
) -> tuple[SelectorIdentity, SelectorIdentity]:
    source_sha = str(contract["source"]["sha256"])
    model_sha = str(contract["model"]["fileSha256"])
    return (
        SelectorIdentity(
            selector_id=str(contract["candidate"]["selectorId"]),
            artifact_hash=model_sha,
            config_hash=str(contract["candidate"]["runtime"]["runtimeSha256"]),
            source_hash=source_sha,
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id=str(contract["opponent"]["selectorId"]),
            artifact_hash=model_sha,
            config_hash=str(contract["opponent"]["runtime"]["runtimeSha256"]),
            source_hash=source_sha,
            engine="native-mcts",
        ),
    )


def arena_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f62ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["suiteHash"],
                "candidateRuntimeSha256": contract["candidate"]["runtime"][
                    "runtimeSha256"
                ],
                "opponentRuntimeSha256": contract["opponent"]["runtime"][
                    "runtimeSha256"
                ],
            }
        )
    )


def outcome_band(win_rate: float, lower_bound: float) -> str:
    if win_rate > 0.50 and lower_bound > MINIMUM_LOWER_BOUND:
        return "value-scale-confirmation-passed"
    if win_rate > 0.50:
        return "value-scale-confirmation-inconclusive"
    return "value-scale-confirmation-failed"


def prepare_phase4f62(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / PREPARATION_REPORT).exists():
        return {**verify_preparation(workspace), "reused": True}
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F62Error("a running GPU ledger entry blocks F62 preparation")
    if ledger.total_hours + CONFIRMATION_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F62Error("insufficient global GPU budget for F62")
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256:
        raise Phase4F62Error("F21 config bytes changed")
    _load_f21_state(workspace)
    evidence = _f61_evidence(workspace)
    resources = _resources(workspace)
    suite, audit = _fresh_confirmation_suite(workspace)
    _write_immutable(
        workspace / SUITE_PATH,
        opening_suite_to_payload(suite),
        "F62 confirmation suite",
    )
    _write_immutable(workspace / ISOLATION_AUDIT, audit, "F62 isolation audit")
    source = fingerprint_sources(workspace, SOURCE_PATHS).to_payload()
    native_sha = loaded_native_backend_hash()
    workload = _self_hashed(
        {
            "suite": SUITE_NAME,
            "suiteHash": suite.content_hash,
            "openingPairs": OPENING_PAIRS,
            "games": GAMES,
            "candidateSimulations": SIMULATIONS,
            "opponentSimulations": SIMULATIONS,
            "actors": ACTORS,
            "mctsSeed": MCTS_SEED,
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapSamples": BOOTSTRAP_SAMPLES,
            "confidence": CONFIDENCE,
        },
        "workloadSha256",
    )
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f62-f21-value-scale-confirmation-contract",
            "status": "frozen",
            "hypothesis": "F21-value-scale-0.5-is-stronger-than-1.0",
            "source": source,
            "nativeBackendSha256": native_sha,
            "model": {
                "path": F21_MODEL.as_posix(),
                "fileSha256": EXPECTED_F21_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "configPath": F21_CONFIG.as_posix(),
                "configFileSha256": EXPECTED_F21_CONFIG_SHA256,
            },
            "candidate": {
                "selectorId": "phase4f21-value-scale-0.5-confirmation",
                "runtime": _runtime_binding(
                    CANDIDATE_VALUE_SCALE, str(source["sha256"]), native_sha
                ),
            },
            "opponent": {
                "selectorId": "phase4f21-value-scale-1.0-confirmation",
                "runtime": _runtime_binding(
                    OPPONENT_VALUE_SCALE, str(source["sha256"]), native_sha
                ),
            },
            "f61Evidence": evidence,
            "openingSuite": {
                "path": SUITE_PATH.as_posix(),
                "fileSha256": sha256_file(workspace / SUITE_PATH),
                "suiteHash": suite.content_hash,
                "openingPairs": OPENING_PAIRS,
                "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
                "isolationAuditSha256": audit["auditSha256"],
                "freshOverlapCount": 0,
                "f60AndF61TrajectoryDisjoint": True,
            },
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger.total_hours,
                "confirmationLimitHours": CONFIRMATION_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "minimumWinRateExclusive": 0.50,
                "minimumOneSided95LowerExclusive": MINIMUM_LOWER_BOUND,
                "confidenceEquivalentTwoSided": CONFIDENCE,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "authorizations": {
                "inferenceOnlyConfirmationAuthorized": True,
                "optimizerStepsAuthorized": 0,
                "trainingAuthorized": False,
                "promotionAuthorized": False,
                "arenaTrajectoriesTrainingUseAllowed": False,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F62 contract")
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f62-value-scale-confirmation-preparation",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingStarted": False,
            "optimizerSteps": 0,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "openingSuitePath": SUITE_PATH.as_posix(),
            "openingSuiteSha256": sha256_file(workspace / SUITE_PATH),
            "openingSuiteHash": suite.content_hash,
            "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
            "isolationAuditSha256": audit["auditSha256"],
            "baselineGlobalGpuHours": ledger.total_hours,
            "inferenceOnlyConfirmationAuthorized": True,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "nextStage": {
                "name": "run frozen 200-pair F21 value-scale confirmation",
                "requiresSeparateUserConfirmation": False,
            },
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, report, "F62 preparation")
    verify_preparation(workspace)
    return {**report, "reused": False}


def _contract(workspace: Path) -> dict[str, object]:
    contract = _mapping(workspace / CONTRACT, "F62 contract")
    _verify_self_hash(contract, "contractSha256", "F62 contract")
    authorizations = contract.get("authorizations", {})
    if (
        contract.get("kind") != "phase4f62-f21-value-scale-confirmation-contract"
        or contract.get("status") != "frozen"
        or authorizations.get("inferenceOnlyConfirmationAuthorized") is not True
        or authorizations.get("optimizerStepsAuthorized") != 0
        or authorizations.get("trainingAuthorized") is not False
        or authorizations.get("promotionAuthorized") is not False
        or authorizations.get("arenaTrajectoriesTrainingUseAllowed") is not False
    ):
        raise Phase4F62Error("F62 contract authorization is invalid")
    return contract


def _storage_amendment(
    workspace: Path,
    contract: Mapping[str, object],
    *,
    require_current_source: bool,
) -> dict[str, object] | None:
    path = workspace / STORAGE_AMENDMENT
    if not path.exists():
        return None
    amendment = _mapping(path, "F62 storage amendment")
    _verify_self_hash(amendment, "amendmentSha256", "F62 storage amendment")
    budget = contract.get("budget")
    progress_path = workspace / ARENA_ROOT / "progress.json"
    progress = _mapping(progress_path, "F62 arena progress")
    _verify_self_hash(progress, "progressSha256", "F62 arena progress")
    frozen_progress = amendment.get("progressAtAmendment", {})
    frozen_pairs = frozen_progress.get("completedPairs")
    if (
        not isinstance(budget, Mapping)
        or amendment.get("kind") != "phase4f62-storage-only-amendment"
        or amendment.get("status") != "frozen"
        or amendment.get("cpuOnly") is not True
        or amendment.get("cudaAccessed") is not False
        or amendment.get("originalContractPath") != CONTRACT.as_posix()
        or amendment.get("originalContractSha256") != contract.get("contractSha256")
        or amendment.get("arenaContractSha256") != arena_contract_sha256(contract)
        or amendment.get("originalSource") != contract.get("source")
        or (
            require_current_source
            and amendment.get("runtimeSource")
            != fingerprint_sources(workspace, SOURCE_PATHS).to_payload()
        )
        or amendment.get("changeScope")
        != "incremental-known-pair-integrity-refresh-v1"
        or amendment.get("selectorSemanticsChanged") is not False
        or amendment.get("openingSuiteChanged") is not False
        or amendment.get("arenaIdentityChanged") is not False
        or amendment.get("originalLimitHours") != CONFIRMATION_GPU_HOURS
        or amendment.get("extendedLimitHours") != AMENDED_CONFIRMATION_GPU_HOURS
        or amendment.get("baselineGlobalHours") != budget.get("baselineGlobalHours")
        or amendment.get("globalLimitHours") != GPU_HOUR_BUDGET
        or amendment.get("artifactLimitBytes") != EXTENDED_ARTIFACT_LIMIT_BYTES
        or frozen_progress.get("path")
        != _relative(progress_path, workspace)
        or not isinstance(frozen_progress.get("fileSha256"), str)
        or not isinstance(frozen_progress.get("progressSha256"), str)
        or type(frozen_pairs) is not int
        or not 0 < frozen_pairs < OPENING_PAIRS
        or frozen_progress.get("completedGames") != frozen_pairs * 2
        or type(progress.get("completedPairs")) is not int
        or progress.get("completedPairs") < frozen_pairs
        or progress.get("completedGames") != progress.get("completedPairs") * 2
        or progress.get("contractSha256") != arena_contract_sha256(contract)
        or amendment.get("trainingAuthorized") is not False
        or amendment.get("promotionAuthorized") is not False
    ):
        raise Phase4F62Error("F62 storage amendment provenance changed")
    return amendment


def _confirmation_gpu_hours(
    workspace: Path, contract: Mapping[str, object]
) -> float:
    amendment = _storage_amendment(
        workspace, contract, require_current_source=True
    )
    return (
        float(amendment["extendedLimitHours"])
        if amendment is not None
        else CONFIRMATION_GPU_HOURS
    )


def amend_phase4f62(workspace_root: str | Path) -> dict[str, object]:
    """Freeze a storage-only source and budget amendment for the resumed arena."""

    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    existing = _storage_amendment(
        workspace, contract, require_current_source=True
    )
    if existing is not None:
        return {**existing, "reused": True}
    if (workspace / GATE).exists() or (workspace / ARENA_ROOT / "result.json").exists():
        raise Phase4F62Error("completed F62 evidence cannot be amended")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F62Error("a running GPU ledger entry blocks the F62 amendment")
    budget = contract.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("confirmationLimitHours") != CONFIRMATION_GPU_HOURS
        or budget.get("globalLimitHours") != GPU_HOUR_BUDGET
        or budget.get("artifactLimitBytes") != EXTENDED_ARTIFACT_LIMIT_BYTES
    ):
        raise Phase4F62Error("F62 contract is not eligible for amendment")
    progress_path = workspace / ARENA_ROOT / "progress.json"
    progress = _mapping(progress_path, "F62 arena progress")
    progress_sha = _verify_self_hash(
        progress, "progressSha256", "F62 arena progress"
    )
    completed_pairs = progress.get("completedPairs")
    completed_games = progress.get("completedGames")
    if (
        type(completed_pairs) is not int
        or not 0 < completed_pairs < OPENING_PAIRS
        or completed_games != completed_pairs * 2
        or progress.get("contractSha256") != arena_contract_sha256(contract)
    ):
        raise Phase4F62Error("F62 progress is not eligible for amendment")
    baseline = float(budget["baselineGlobalHours"])
    if not (
        baseline + CONFIRMATION_GPU_HOURS
        <= ledger.total_hours
        < baseline + AMENDED_CONFIRMATION_GPU_HOURS
    ):
        raise Phase4F62Error("F62 ledger is outside the amendment window")
    resources = _resources(workspace)
    amendment = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f62-storage-only-amendment",
            "status": "frozen",
            "cpuOnly": True,
            "cudaAccessed": False,
            "originalContractPath": CONTRACT.as_posix(),
            "originalContractSha256": contract["contractSha256"],
            "arenaContractSha256": arena_contract_sha256(contract),
            "originalSource": contract["source"],
            "runtimeSource": fingerprint_sources(workspace, SOURCE_PATHS).to_payload(),
            "changeScope": "incremental-known-pair-integrity-refresh-v1",
            "selectorSemanticsChanged": False,
            "openingSuiteChanged": False,
            "arenaIdentityChanged": False,
            "originalLimitHours": CONFIRMATION_GPU_HOURS,
            "extendedLimitHours": AMENDED_CONFIRMATION_GPU_HOURS,
            "baselineGlobalHours": baseline,
            "globalGpuHoursAtAmendment": ledger.total_hours,
            "globalLimitHours": GPU_HOUR_BUDGET,
            "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
            "progressAtAmendment": {
                "path": _relative(progress_path, workspace),
                "fileSha256": sha256_file(progress_path),
                "progressSha256": progress_sha,
                "completedPairs": completed_pairs,
                "completedGames": completed_games,
            },
            "resourcesAtAmendment": resources,
            "trainingAuthorized": False,
            "promotionAuthorized": False,
            "arenaTrajectoriesTrainingUseAllowed": False,
        },
        "amendmentSha256",
    )
    _write_immutable(
        workspace / STORAGE_AMENDMENT, amendment, "F62 storage amendment"
    )
    return {**amendment, "reused": False}


def verify_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    report = _mapping(workspace / PREPARATION_REPORT, "F62 preparation")
    report_sha = _verify_self_hash(report, "reportSha256", "F62 preparation")
    suite, audit = _stored_suite(workspace)
    source = fingerprint_sources(workspace, SOURCE_PATHS).to_payload()
    amendment = _storage_amendment(
        workspace, contract, require_current_source=True
    )
    source_verified = contract.get("source") == source or (
        amendment is not None and amendment.get("runtimeSource") == source
    )
    if (
        not source_verified
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or sha256_file(workspace / F21_MODEL) != EXPECTED_F21_MODEL_SHA256
        or sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256
        or contract.get("f61Evidence") != _f61_evidence(workspace)
        or contract.get("openingSuite", {}).get("fileSha256")
        != sha256_file(workspace / SUITE_PATH)
        or contract.get("openingSuite", {}).get("suiteHash") != suite.content_hash
        or contract.get("openingSuite", {}).get("isolationAuditSha256")
        != audit.get("auditSha256")
        or report.get("contractSha256") != contract.get("contractSha256")
        or report.get("isolationAuditSha256") != audit.get("auditSha256")
    ):
        raise Phase4F62Error("F62 frozen preparation changed")
    _load_f21_state(workspace)
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f62-value-scale-confirmation-preparation-verification",
        "status": "verified",
        "contractPath": CONTRACT.as_posix(),
        "contractSha256": contract["contractSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report_sha,
        "openingSuitePath": SUITE_PATH.as_posix(),
        "openingSuiteHash": suite.content_hash,
        "openingPairs": len(suite.openings),
        "candidateValueScale": CANDIDATE_VALUE_SCALE,
        "opponentValueScale": OPPONENT_VALUE_SCALE,
        "inferenceOnlyConfirmationAuthorized": True,
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
    }


def _selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    identity: SelectorIdentity,
    value_scale: float,
) -> BoundMoveSelector:
    config = load_experiment_config(workspace / F21_CONFIG)
    model = PolicyValueNetwork(
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
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4F62Error("F21 state does not fit the frozen network") from error
    evaluator = ValueScaledEvaluator(
        TorchEvaluator(model.to(device).eval(), device=device, use_amp=True),
        value_scale,
    )
    search = MCTSConfig(
        simulations=SIMULATIONS,
        c_base=config.mcts.c_base,
        c_init=config.mcts.c_init,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=0.0,
        add_root_noise=False,
        seed=MCTS_SEED,
    )
    return BoundMoveSelector(
        MCTSSelector(evaluator, search, search_factory=NativeMCTS, engine="native-mcts"),
        identity,
    )


def gate_from_result(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    candidate_identity, opponent_identity = _identities(contract)
    statistics = result.get("statistics", {})
    interval = statistics.get("winRateCI", {})
    checks = {
        "contract": result.get("contractSha256") == arena_contract_sha256(contract),
        "suite": result.get("suite") == SUITE_NAME,
        "suiteHash": result.get("suiteHash")
        == contract["openingSuite"]["suiteHash"],
        "openingPairs": result.get("openingPairs") == OPENING_PAIRS,
        "games": result.get("games") == GAMES,
        "candidateIdentity": result.get("candidateIdentity")
        == candidate_identity.to_dict(),
        "opponentIdentity": result.get("opponentIdentity")
        == opponent_identity.to_dict(),
        "candidateRuntime": (
            result.get("candidateRuntime", {}).get("kind") == "native-mcts"
            and result.get("candidateRuntime", {}).get("trusted") is True
            and result.get("candidateRuntime", {}).get("configuredSimulations")
            == SIMULATIONS
            and result.get("candidateRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "opponentRuntime": (
            result.get("opponentRuntime", {}).get("kind") == "native-mcts"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("configuredSimulations")
            == SIMULATIONS
            and result.get("opponentRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "sameModelArtifact": (
            candidate_identity.artifact_hash == opponent_identity.artifact_hash
            == EXPECTED_F21_MODEL_SHA256
        ),
        "differentRuntimeBindings": (
            candidate_identity.config_hash != opponent_identity.config_hash
        ),
        "draws": statistics.get("draws") == 0,
        "winRateAboveHalf": float(statistics.get("winRate", -1.0)) > 0.50,
        "oneSided95LowerAboveHalf": float(interval.get("lower", -1.0))
        > MINIMUM_LOWER_BOUND,
    }
    decision_keys = {"winRateAboveHalf", "oneSided95LowerAboveHalf"}
    if not all(value for key, value in checks.items() if key not in decision_keys):
        raise Phase4F62Error(f"completed F62 integrity checks failed: {checks}")
    band = outcome_band(
        float(statistics["winRate"]), float(statistics["winRateCI"]["lower"])
    )
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f62-f21-value-scale-confirmation-gate",
            "status": "completed",
            "integrityPassed": True,
            "confirmationPassed": band == "value-scale-confirmation-passed",
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "candidateValueScale": CANDIDATE_VALUE_SCALE,
            "opponentValueScale": OPPONENT_VALUE_SCALE,
            "statistics": statistics,
            "outcomeBand": band,
            "valueCalibrationTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "arenaTrajectoriesTrainingUseAllowed": False,
            "nextStage": {
                "name": "Phase 4F63 value-head calibration design review",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )


def verify_phase4f62(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    verify_preparation(workspace)
    contract = _contract(workspace)
    gate = _mapping(workspace / GATE, "F62 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F62 gate")
    if gate != gate_from_result(workspace, contract):
        raise Phase4F62Error("stored F62 gate differs from arena result")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    ceiling = min(
        baseline + _confirmation_gpu_hours(workspace, contract),
        float(contract["budget"]["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours > ceiling:
        raise Phase4F62Error("F62 completed outside its frozen GPU budget")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f62-value-scale-confirmation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "statistics": gate["statistics"],
        "outcomeBand": gate["outcomeBand"],
        "confirmationPassed": gate["confirmationPassed"],
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
        "globalGpuHours": ledger.total_hours,
        "nextStage": gate["nextStage"],
    }


def run_phase4f62(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f62(workspace), "reused": True}
    verify_preparation(workspace)
    contract = _contract(workspace)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F62Error("F62 confirmation requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    ceiling = min(
        baseline + _confirmation_gpu_hours(workspace, contract),
        float(contract["budget"]["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours >= ceiling:
        raise Phase4F62Error("F62 GPU budget is unavailable or exhausted")
    resources = _resources(workspace)
    state = _load_f21_state(workspace)
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f62-confirmation-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
    started = time.monotonic()
    last_ledger = float("-inf")
    last_progress = float("-inf")
    candidate_selector: BoundMoveSelector | None = None
    opponent_selector: BoundMoveSelector | None = None

    def heartbeat(moves: int, pairs: int) -> None:
        nonlocal last_ledger, last_progress
        now = time.monotonic()
        if now - last_ledger >= 5.0:
            total = ledger.update(run_id, elapsed_seconds=now - started, status="running")
            if total >= ceiling:
                raise Phase4F62Error("F62 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                canonical_json_bytes(
                    {
                        "kind": "phase4f62-value-scale-confirmation-progress",
                        "status": "running",
                        "openingPairs": pairs,
                        "openingPairsTotal": OPENING_PAIRS,
                        "moves": moves,
                        "globalGpuHours": ledger.total_hours,
                    }
                ).decode("ascii"),
                end="",
                flush=True,
            )
            last_progress = now

    try:
        with RunLock(
            workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id
        ):
            ledger.begin(run_id, external_hours=0.0)
            print(
                canonical_json_bytes(
                    {
                        "kind": "phase4f62-value-scale-confirmation-progress",
                        "status": "started",
                        "candidate": candidate_identity.selector_id,
                        "candidateValueScale": CANDIDATE_VALUE_SCALE,
                        "opponent": opponent_identity.selector_id,
                        "opponentValueScale": OPPONENT_VALUE_SCALE,
                        "openingPairs": len(suite.openings),
                        "games": GAMES,
                        "simulations": SIMULATIONS,
                        "resources": resources,
                    }
                ).decode("ascii"),
                end="",
                flush=True,
            )
            candidate_selector = _selector(
                workspace,
                state,
                device,
                candidate_identity,
                CANDIDATE_VALUE_SCALE,
            )
            opponent_selector = _selector(
                workspace,
                state,
                device,
                opponent_identity,
                OPPONENT_VALUE_SCALE,
            )
            arena = run_resumable_paired_arena(
                suite,
                candidate_selector,
                opponent_selector,
                store_root=workspace / ARENA_ROOT,
                contract_sha256=arena_contract_sha256(contract),
                actors=ACTORS,
                heartbeat=heartbeat,
            )
            arena.finalize(
                bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                confidence=CONFIDENCE,
            )
            gate = gate_from_result(workspace, contract)
            total = ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="completed"
            )
            if total > ceiling:
                raise Phase4F62Error("F62 completed beyond its frozen GPU budget")
            _write_immutable(workspace / GATE, gate, "F62 gate")
            return {**gate, "globalGpuHours": total, "reused": False}
    except BaseException:
        try:
            ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="interrupted"
            )
        except Exception:
            pass
        raise
    finally:
        if candidate_selector is not None:
            candidate_selector.close()
        if opponent_selector is not None:
            opponent_selector.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "amend", "run", "verify"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f62(args.workspace_root)
    elif args.command == "amend":
        payload = amend_phase4f62(args.workspace_root)
    elif args.command == "run":
        payload = run_phase4f62(args.workspace_root, args.device)
    else:
        payload = verify_phase4f62(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
