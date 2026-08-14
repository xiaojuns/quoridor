"""Freeze and run the F31 generation-5 fresh-opening confirmation arena."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .classic_bridge import CLASSIC_ENGINE, HARD_MAX_DEPTH, ClassicSearchBudget
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import (
    OpeningSuite,
    load_opening_suite,
    opening_suite_to_payload,
    opening_position_orbit_digest,
    validate_suite_collection,
)
from .paired_arena import (
    BoundMoveSelector,
    SelectorIdentity,
    canonical_classic_backend_hash,
    loaded_native_backend_hash,
)
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _relative,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f32_fresh_screen import (
    FROZEN_SUITES,
    F31_CONFIG,
    FULL_SUITE as F32_FULL_SUITE,
    REPLAY_ROOTS,
    _candidate_payload,
    _verify_audit_bindings,
)
from .phase4f29_fresh_screen import _arena_references, _load_generator, _replay_orbits
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .selectors import ClassicHardSelector, MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
F32_CONTRACT = Path("training/configs/phase4f32-fresh-screen-frozen-v1.json")
F32_GATE = Path(
    "training/runs/phase4/phase4f/f31-fresh-development-screen-v1/screen-gate.json"
)
EXPECTED_F32_CONTRACT_SHA256 = (
    "74a27742fcaff0f047eb80b959c187f50f2968d80c3fb14522b25e626ee3c18d"
)
EXPECTED_F32_GATE_SHA256 = (
    "acfbe2f85c76ac40ecf464e7c41f2c5403eca8c182a8831bb0c246269e7cd9dd"
)
EXPECTED_F32_RESULT_SHA256 = (
    "b6bb1b225e5b8cb0b30df2f3f9011d692700536f55518690925b78cfef439bd4"
)

ROOT = Path("training/runs/phase4/phase4f/f31-fresh-confirmation-v2")
CONFIRMATION_SUITE = ROOT / "recovery-fresh-confirmation-200.json"
ARENA_ROOT = ROOT / "confirmation-arena"
GATE = ROOT / "confirmation-gate.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
ISOLATION_AUDIT = ROOT / "isolation-audit.json"
CONTRACT = Path("training/configs/phase4f33-confirmation-frozen-v2.json")
BUDGET_EXTENSION = ROOT / "budget-extension-v1.json"
BUDGET_EXTENSION_V2 = ROOT / "budget-extension-v2.json"

OPENING_PAIRS = 200
GAMES = 400
MCTS_SEED = 20260818
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
CANDIDATE_SIMULATIONS = 800
CLASSIC_NODES = 587
ACTORS = 32
CONFIRMATION_GPU_HOURS = 5.0
EXTENDED_CONFIRMATION_GPU_HOURS = 8.0
FINAL_CONFIRMATION_GPU_HOURS = 11.0
ARENA_WRITE_RESERVE_BYTES = 128 * 1024**2
MINIMUM_WIN_RATE = 0.55
MINIMUM_LOWER_BOUND = 0.50


class Phase4F33ConfirmationError(RuntimeError):
    """Raised when F33 preparation, lineage, isolation, or arena checks fail."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F33ConfirmationError(
            "workspace does not contain training/quoridor"
        )
    return workspace


def _resource_snapshot(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F33ConfirmationError(
            "12 GiB cap cannot reserve the F33 confirmation output"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F33ConfirmationError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _f32_evidence(workspace: Path) -> dict[str, object]:
    contract = _mapping(workspace / F32_CONTRACT, "F32 contract")
    contract_sha256 = _verify_self_hash(contract, "contractSha256", "F32 contract")
    gate = _mapping(workspace / F32_GATE, "F32 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F32 gate")
    result_path = workspace / str(gate.get("resultPath", ""))
    result = verify_completed_arena_result(result_path)
    if (
        contract_sha256 != EXPECTED_F32_CONTRACT_SHA256
        or gate_sha256 != EXPECTED_F32_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "f31-fresh-confirmation-candidate"
        or gate.get("confirmationArenaCandidate") is not True
        or gate.get("formalPromotionAuthorized") is not False
        or gate.get("resultSha256") != EXPECTED_F32_RESULT_SHA256
        or result.get("resultSha256") != EXPECTED_F32_RESULT_SHA256
        or result.get("statistics") != gate.get("statistics")
        or not all(gate.get("checks", {}).values())
    ):
        raise Phase4F33ConfirmationError("F32 gate is not eligible for confirmation")
    return {
        "contractPath": F32_CONTRACT.as_posix(),
        "contractSha256": contract_sha256,
        "gatePath": F32_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": str(gate["resultPath"]),
        "resultFileSha256": sha256_file(result_path),
        "resultSha256": EXPECTED_F32_RESULT_SHA256,
        "outcomeBand": gate["outcomeBand"],
        "statistics": gate["statistics"],
    }


def _fresh_confirmation_suite(
    workspace: Path,
) -> tuple[OpeningSuite, dict[str, object]]:
    suites: list[OpeningSuite] = []
    generator_suites: list[OpeningSuite] = []
    suite_refs: list[dict[str, object]] = []
    for relative, name in FROZEN_SUITES:
        path = workspace / relative
        suite = load_opening_suite(path, expected_suite=name)
        suites.append(suite)
        if name != "recovery-fresh-dev-v1":
            generator_suites.append(suite)
        suite_refs.append(
            {
                "path": relative.as_posix(),
                "name": name,
                "fileSha256": sha256_file(path),
                "suiteHash": suite.content_hash,
                "openingPairs": len(suite.openings),
            }
        )
    validate_suite_collection(*suites)
    f32_full = load_opening_suite(
        workspace / F32_FULL_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    suite_refs.append(
        {
            "path": F32_FULL_SUITE.as_posix(),
            "name": "phase4f32-full-fresh-pool",
            "fileSha256": sha256_file(workspace / F32_FULL_SUITE),
            "suiteHash": f32_full.content_hash,
            "openingPairs": len(f32_full.openings),
        }
    )
    forbidden = {
        opening_position_orbit_digest(opening)
        for suite in (*suites, f32_full)
        for opening in suite.openings
    }
    replay_refs: list[dict[str, object]] = []
    for relative in REPLAY_ROOTS:
        print(
            json.dumps(
                {
                    "kind": "phase4f33-preparation-progress",
                    "stage": "replay-isolation",
                    "path": relative.as_posix(),
                    "status": "started",
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            orbits, reference = _replay_orbits(workspace, relative)
        except Exception as error:
            raise Phase4F33ConfirmationError(
                f"cannot verify excluded replay {relative}"
            ) from error
        forbidden.update(orbits)
        replay_refs.append({**reference, "uniquePositionOrbits": len(orbits)})
    arena_refs = _arena_references(workspace)
    from .challenge_replay import arena_position_orbits

    arena_orbits: set[str] = set()
    for reference in arena_refs:
        arena_orbits.update(arena_position_orbits(workspace, reference))
    forbidden.update(arena_orbits)
    fresh = _load_generator(workspace).generate_fresh_development_suite(
        tuple(generator_suites), forbidden_orbits=forbidden
    )
    fresh_orbits = {
        opening_position_orbit_digest(opening) for opening in fresh.openings
    }
    overlap = len(fresh_orbits & forbidden)
    if overlap:
        raise Phase4F33ConfirmationError(
            "fresh confirmation suite overlaps excluded position orbits"
        )
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f33-fresh-confirmation-isolation-audit",
            "status": "completed",
            "frozenSuites": suite_refs,
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
            "f32FullPoolExcluded": True,
            "f32Evidence": _f32_evidence(workspace),
        },
        "auditSha256",
    )
    if len(fresh.openings) != OPENING_PAIRS:
        raise Phase4F33ConfirmationError("fresh confirmation suite is incomplete")
    f32_result = audit["f32Evidence"]["resultPath"]
    historical = audit.get("historicalScreenResults")
    if not isinstance(historical, list) or not any(
        isinstance(item, Mapping) and item.get("path") == f32_result
        for item in historical
    ):
        raise Phase4F33ConfirmationError(
            "F32 trajectories were not included in the confirmation exclusion set"
        )
    return fresh, audit


def _stored_suite(
    workspace: Path,
) -> tuple[OpeningSuite, dict[str, object]]:
    suite = load_opening_suite(
        workspace / CONFIRMATION_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    audit = _mapping(workspace / ISOLATION_AUDIT, "F33 isolation audit")
    _verify_self_hash(audit, "auditSha256", "F33 isolation audit")
    if (
        len(suite.openings) != OPENING_PAIRS
        or audit.get("kind") != "phase4f33-fresh-confirmation-isolation-audit"
        or audit.get("freshOverlapCount") != 0
        or audit.get("freshOpeningPositionOrbits") != OPENING_PAIRS
        or audit.get("confirmationOpeningPairs") != OPENING_PAIRS
        or audit.get("f32FullPoolExcluded") is not True
    ):
        raise Phase4F33ConfirmationError("stored F33 opening evidence is invalid")
    _verify_audit_bindings(workspace, audit)
    current_orbits = {
        opening_position_orbit_digest(opening) for opening in suite.openings
    }
    if len(current_orbits) != OPENING_PAIRS:
        raise Phase4F33ConfirmationError("F33 suite contains duplicate position orbits")
    return suite, audit


def _suite_reference(
    workspace: Path, suite: OpeningSuite, audit: Mapping[str, object]
) -> dict[str, object]:
    return {
        "path": CONFIRMATION_SUITE.as_posix(),
        "fileSha256": sha256_file(workspace / CONFIRMATION_SUITE),
        "suiteHash": suite.content_hash,
        "openingPairs": len(suite.openings),
        "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
        "isolationAuditSha256": audit["auditSha256"],
        "freshOverlapCount": audit["freshOverlapCount"],
        "contentPositionOrbitAndTrajectoryDisjoint": True,
    }


def _identities(
    contract: Mapping[str, object],
) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    workload = contract["workload"]
    return (
        SelectorIdentity(
            selector_id=(
                "phase4f31-stage-b-extended-g00000005-s000000000576-"
                "fresh-confirmation-v1"
            ),
            artifact_hash=str(candidate["candidateModelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id="classic-hard-phase4f33-fresh-confirmation-v2",
            artifact_hash=str(contract["classicBackendSha256"]),
            config_hash=str(workload["workloadSha256"]),
            source_hash=str(contract["source"]["sha256"]),
            engine=CLASSIC_ENGINE,
        ),
    )


def _arena_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f33ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["suiteHash"],
            }
        )
    )


def _budget_extension_v1(
    workspace: Path,
    contract: Mapping[str, object],
    *,
    require_current_source: bool,
) -> dict[str, object] | None:
    path = workspace / BUDGET_EXTENSION
    if not path.exists():
        return None
    extension = _mapping(path, "F33 budget extension v1")
    _verify_self_hash(extension, "extensionSha256", "F33 budget extension v1")
    budget = contract.get("budget")
    if (
        not isinstance(budget, Mapping)
        or extension.get("kind") != "phase4f33-confirmation-budget-extension"
        or extension.get("status") != "frozen"
        or extension.get("cpuOnly") is not True
        or extension.get("cudaAccessed") is not False
        or extension.get("originalContractPath") != CONTRACT.as_posix()
        or extension.get("originalContractSha256") != contract.get("contractSha256")
        or extension.get("arenaContractSha256") != _arena_contract_sha256(contract)
        or extension.get("originalSourceSha256")
        != contract.get("source", {}).get("sha256")
        or (
            require_current_source
            and extension.get("runtimeSource")
            != fingerprint_sources(workspace).to_payload()
        )
        or extension.get("originalLimitHours") != CONFIRMATION_GPU_HOURS
        or extension.get("extendedLimitHours")
        != EXTENDED_CONFIRMATION_GPU_HOURS
        or extension.get("baselineGlobalHours")
        != budget.get("baselineGlobalHours")
        or extension.get("globalLimitHours") != GPU_HOUR_BUDGET
        or extension.get("artifactLimitBytes") != MAXIMUM_ARTIFACT_LIMIT_BYTES
        or extension.get("minimumFreeDiskBytes")
        != DEFAULT_MINIMUM_FREE_DISK_BYTES
        or extension.get("semanticChangesAllowed") is not False
        or extension.get("completedResultAtExtension") is not False
    ):
        raise Phase4F33ConfirmationError(
            "F33 budget extension v1 provenance changed"
        )
    return extension


def _budget_extension(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, object] | None:
    v2_path = workspace / BUDGET_EXTENSION_V2
    if not v2_path.exists():
        return _budget_extension_v1(
            workspace, contract, require_current_source=True
        )
    previous = _budget_extension_v1(
        workspace, contract, require_current_source=False
    )
    if previous is None:
        raise Phase4F33ConfirmationError(
            "F33 budget extension v2 is missing its v1 predecessor"
        )
    extension = _mapping(v2_path, "F33 budget extension v2")
    _verify_self_hash(extension, "extensionSha256", "F33 budget extension v2")
    budget = contract.get("budget")
    if (
        not isinstance(budget, Mapping)
        or extension.get("kind") != "phase4f33-confirmation-budget-extension"
        or extension.get("revision") != 2
        or extension.get("status") != "frozen"
        or extension.get("cpuOnly") is not True
        or extension.get("cudaAccessed") is not False
        or extension.get("originalContractPath") != CONTRACT.as_posix()
        or extension.get("originalContractSha256") != contract.get("contractSha256")
        or extension.get("arenaContractSha256") != _arena_contract_sha256(contract)
        or extension.get("previousExtensionPath") != BUDGET_EXTENSION.as_posix()
        or extension.get("previousExtensionSha256")
        != previous.get("extensionSha256")
        or extension.get("originalSourceSha256")
        != contract.get("source", {}).get("sha256")
        or extension.get("previousRuntimeSourceSha256")
        != previous.get("runtimeSource", {}).get("sha256")
        or extension.get("runtimeSource")
        != fingerprint_sources(workspace).to_payload()
        or extension.get("originalLimitHours") != CONFIRMATION_GPU_HOURS
        or extension.get("previousLimitHours")
        != EXTENDED_CONFIRMATION_GPU_HOURS
        or extension.get("extendedLimitHours") != FINAL_CONFIRMATION_GPU_HOURS
        or extension.get("baselineGlobalHours")
        != budget.get("baselineGlobalHours")
        or extension.get("globalLimitHours") != GPU_HOUR_BUDGET
        or extension.get("artifactLimitBytes") != MAXIMUM_ARTIFACT_LIMIT_BYTES
        or extension.get("minimumFreeDiskBytes")
        != DEFAULT_MINIMUM_FREE_DISK_BYTES
        or extension.get("semanticChangesAllowed") is not False
        or extension.get("completedResultAtExtension") is not False
    ):
        raise Phase4F33ConfirmationError(
            "F33 budget extension v2 provenance changed"
        )
    return extension


def _confirmation_gpu_hours(
    workspace: Path, contract: Mapping[str, object]
) -> float:
    extension = _budget_extension(workspace, contract)
    return (
        float(extension["extendedLimitHours"])
        if extension is not None
        else CONFIRMATION_GPU_HOURS
    )


def extend_phase4f33_budget(workspace_root: str | Path) -> dict[str, object]:
    """Freeze a budget-only amendment while preserving the arena identity."""

    workspace = _workspace(workspace_root)
    if (workspace / BUDGET_EXTENSION_V2).exists():
        contract = _mapping(workspace / CONTRACT, "F33 contract")
        _verify_self_hash(contract, "contractSha256", "F33 contract")
        return dict(_budget_extension(workspace, contract) or {})
    contract = _mapping(workspace / CONTRACT, "F33 contract")
    _verify_self_hash(contract, "contractSha256", "F33 contract")
    budget = contract.get("budget")
    previous = _budget_extension_v1(
        workspace, contract, require_current_source=False
    )
    previous_limit = (
        EXTENDED_CONFIRMATION_GPU_HOURS
        if previous is not None
        else CONFIRMATION_GPU_HOURS
    )
    next_limit = (
        FINAL_CONFIRMATION_GPU_HOURS
        if previous is not None
        else EXTENDED_CONFIRMATION_GPU_HOURS
    )
    if (
        not isinstance(budget, Mapping)
        or budget.get("confirmationLimitHours") != CONFIRMATION_GPU_HOURS
        or budget.get("globalLimitHours") != GPU_HOUR_BUDGET
        or budget.get("artifactLimitBytes") != MAXIMUM_ARTIFACT_LIMIT_BYTES
        or contract.get("formalPromotion") is not False
        or contract.get("additionalTrainingAuthorized") is not False
        or contract.get("arenaTrajectoriesTrainingUseAllowed") is not False
    ):
        raise Phase4F33ConfirmationError(
            "the frozen F33 contract is not eligible for a budget-only extension"
        )
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F33ConfirmationError(
            "a running GPU ledger entry blocks the F33 budget extension"
        )
    baseline = float(budget["baselineGlobalHours"])
    if ledger.total_hours < baseline + previous_limit:
        raise Phase4F33ConfirmationError(
            "the current F33 budget has not been exhausted"
        )
    if (
        ledger.total_hours >= baseline + next_limit
        or baseline + next_limit > GPU_HOUR_BUDGET
    ):
        raise Phase4F33ConfirmationError(
            "the extended F33 or global GPU budget is unavailable"
        )
    if (workspace / GATE).exists() or (workspace / ARENA_ROOT / "result.json").exists():
        raise Phase4F33ConfirmationError(
            "a completed F33 result cannot receive a budget extension"
        )
    progress_path = workspace / ARENA_ROOT / "progress.json"
    progress = _mapping(progress_path, "F33 arena progress")
    _verify_self_hash(progress, "progressSha256", "F33 arena progress")
    completed_pairs = progress.get("completedPairs")
    completed_games = progress.get("completedGames")
    pair_records = progress.get("pairs")
    if (
        progress.get("status") != "in-progress"
        or progress.get("contractSha256") != _arena_contract_sha256(contract)
        or not isinstance(completed_pairs, int)
        or not 0 < completed_pairs < OPENING_PAIRS
        or completed_games != completed_pairs * 2
        or not isinstance(pair_records, list)
        or len(pair_records) != completed_pairs
    ):
        raise Phase4F33ConfirmationError(
            "F33 arena progress is not eligible for resumption"
        )
    for record in pair_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("file"), str):
            raise Phase4F33ConfirmationError("F33 arena pair record is invalid")
        pair_path = workspace / ARENA_ROOT / str(record["file"])
        if (
            pair_path.is_symlink()
            or not pair_path.is_file()
            or pair_path.stat().st_size != record.get("bytes")
            or sha256_file(pair_path) != record.get("sha256")
        ):
            raise Phase4F33ConfirmationError("F33 arena pair evidence changed")
    resources = _resource_snapshot(workspace)
    runtime_source = fingerprint_sources(workspace).to_payload()
    extension_core: dict[str, object] = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f33-confirmation-budget-extension",
            "revision": 2 if previous is not None else 1,
            "status": "frozen",
            "cpuOnly": True,
            "cudaAccessed": False,
            "reason": "measured 400-game arena duration exceeded the frozen estimate",
            "originalContractPath": CONTRACT.as_posix(),
            "originalContractSha256": contract["contractSha256"],
            "arenaContractSha256": _arena_contract_sha256(contract),
            "originalSourceSha256": contract["source"]["sha256"],
            "runtimeSource": runtime_source,
            "originalLimitHours": CONFIRMATION_GPU_HOURS,
            "extendedLimitHours": next_limit,
            "baselineGlobalHours": baseline,
            "globalGpuHoursAtExtension": ledger.total_hours,
            "globalLimitHours": GPU_HOUR_BUDGET,
            "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
            "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            "progressAtExtension": {
                "path": _relative(progress_path, workspace),
                "fileSha256": sha256_file(progress_path),
                "progressSha256": progress["progressSha256"],
                "completedPairs": completed_pairs,
                "completedGames": completed_games,
            },
            "resourcesAtExtension": resources,
            "completedResultAtExtension": False,
            "semanticChangesAllowed": False,
            "formalPromotionAuthorized": False,
            "additionalTrainingAuthorized": False,
    }
    if previous is not None:
        extension_core.update(
            {
                "previousExtensionPath": BUDGET_EXTENSION.as_posix(),
                "previousExtensionSha256": previous["extensionSha256"],
                "previousRuntimeSourceSha256": previous["runtimeSource"]["sha256"],
                "previousLimitHours": previous_limit,
            }
        )
    extension = _self_hashed(
        extension_core,
        "extensionSha256",
    )
    _write_immutable(
        workspace / (
            BUDGET_EXTENSION_V2 if previous is not None else BUDGET_EXTENSION
        ),
        extension,
        "F33 budget extension",
    )
    return extension


def prepare_phase4f33(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / CONTRACT).exists():
        contract = verify_phase4f33_preparation(workspace)
        return {
            "status": "prepared",
            "reused": True,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
        }
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F33ConfirmationError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + CONFIRMATION_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F33ConfirmationError("insufficient global GPU budget for F33")
    resources = _resource_snapshot(workspace)
    candidate, _ = _candidate_payload(workspace)
    f32_evidence = _f32_evidence(workspace)
    suite, audit = _fresh_confirmation_suite(workspace)
    _write_immutable(
        workspace / CONFIRMATION_SUITE,
        opening_suite_to_payload(suite),
        "F33 confirmation suite",
    )
    _write_immutable(workspace / ISOLATION_AUDIT, audit, "F33 isolation audit")
    suite_reference = _suite_reference(workspace, suite, audit)
    workload = _self_hashed(
        {
            "suite": "recovery-fresh-dev-v1",
            "suiteHash": suite.content_hash,
            "openingPairs": OPENING_PAIRS,
            "games": GAMES,
            "candidateSimulations": CANDIDATE_SIMULATIONS,
            "classicBudgetKind": "nodes",
            "classicBudgetLimit": CLASSIC_NODES,
            "classicMaxDepth": HARD_MAX_DEPTH,
            "actors": ACTORS,
            "mctsSeed": MCTS_SEED,
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapSamples": BOOTSTRAP_SAMPLES,
            "confidence": CONFIDENCE,
        },
        "workloadSha256",
    )
    source = fingerprint_sources(workspace).to_payload()
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f33-f31-fresh-classic-hard-confirmation",
            "status": "frozen",
            "formalPromotion": False,
            "confirmationArenaAuthorized": True,
            "additionalTrainingAuthorized": False,
            "arenaTrajectoriesTrainingUseAllowed": False,
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "classicBackendSha256": canonical_classic_backend_hash(),
            "candidate": candidate,
            "f32Evidence": f32_evidence,
            "openingSuite": suite_reference,
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger.total_hours,
                "confirmationLimitHours": CONFIRMATION_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "minimumWinRate": MINIMUM_WIN_RATE,
                "minimumOneSided95Lower": MINIMUM_LOWER_BOUND,
                "confidenceEquivalentTwoSided": CONFIDENCE,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F33 confirmation contract")
    preparation = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f33-confirmation-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "confirmationArenaAuthorized": True,
            "formalPromotionAuthorized": False,
            "additionalTrainingAuthorized": False,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "source": source,
            "candidate": candidate,
            "f32Evidence": f32_evidence,
            "openingSuite": suite_reference,
            "globalGpuHours": ledger.total_hours,
            "resources": resources,
        },
        "reportSha256",
    )
    _write_immutable(
        workspace / PREPARATION_REPORT, preparation, "F33 preparation report"
    )
    verify_phase4f33_preparation(workspace)
    return preparation


def verify_phase4f33_preparation(
    workspace_root: str | Path,
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F33 contract")
    _verify_self_hash(contract, "contractSha256", "F33 contract")
    preparation = _mapping(workspace / PREPARATION_REPORT, "F33 preparation")
    _verify_self_hash(preparation, "reportSha256", "F33 preparation")
    candidate, _ = _candidate_payload(workspace)
    f32_evidence = _f32_evidence(workspace)
    suite, audit = _stored_suite(workspace)
    suite_reference = _suite_reference(workspace, suite, audit)
    workload = contract.get("workload")
    if not isinstance(workload, Mapping):
        raise Phase4F33ConfirmationError("F33 workload is invalid")
    _verify_self_hash(workload, "workloadSha256", "F33 workload")
    current_source = fingerprint_sources(workspace).to_payload()
    extension = _budget_extension(workspace, contract)
    source_verified = contract.get("source") == current_source or (
        extension is not None and extension.get("runtimeSource") == current_source
    )
    if (
        contract.get("kind") != "phase4f33-f31-fresh-classic-hard-confirmation"
        or contract.get("status") != "frozen"
        or contract.get("formalPromotion") is not False
        or contract.get("confirmationArenaAuthorized") is not True
        or contract.get("additionalTrainingAuthorized") is not False
        or contract.get("arenaTrajectoriesTrainingUseAllowed") is not False
        or not source_verified
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("classicBackendSha256") != canonical_classic_backend_hash()
        or contract.get("candidate") != candidate
        or contract.get("f32Evidence") != f32_evidence
        or contract.get("openingSuite") != suite_reference
        or workload.get("openingPairs") != OPENING_PAIRS
        or workload.get("games") != GAMES
        or workload.get("suiteHash") != suite.content_hash
        or preparation.get("contractSha256") != contract.get("contractSha256")
        or preparation.get("source") != contract.get("source")
        or preparation.get("confirmationArenaAuthorized") is not True
        or preparation.get("formalPromotionAuthorized") is not False
        or preparation.get("additionalTrainingAuthorized") is not False
    ):
        raise Phase4F33ConfirmationError("F33 preparation provenance changed")
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    confirmation_gpu_hours = _confirmation_gpu_hours(workspace, contract)
    if (
        not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours
        > float(budget["baselineGlobalHours"]) + confirmation_gpu_hours
    ):
        raise Phase4F33ConfirmationError("F33 confirmation GPU budget is unavailable")
    _resource_snapshot(workspace)
    return contract


def _model_selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    identity: SelectorIdentity,
) -> BoundMoveSelector:
    config = load_experiment_config(workspace / F31_CONFIG)
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
        raise Phase4F33ConfirmationError(
            "candidate state does not match frozen network"
        ) from error
    model = model.to(device).eval()
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
    evaluator.enable_native_encoder(load_native_module())
    search = MCTSConfig(
        simulations=CANDIDATE_SIMULATIONS,
        c_base=config.mcts.c_base,
        c_init=config.mcts.c_init,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=0.0,
        add_root_noise=False,
        seed=MCTS_SEED,
    )
    return BoundMoveSelector(
        MCTSSelector(
            evaluator, search, search_factory=NativeMCTS, engine="native-mcts"
        ),
        identity,
    )


def _gate_from_result(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    candidate_identity, opponent_identity = _identities(contract)
    workload = contract["workload"]
    statistics = result.get("statistics", {})
    interval = statistics.get("winRateCI", {})
    checks = {
        "contract": result.get("contractSha256") == _arena_contract_sha256(contract),
        "suite": result.get("suite") == "recovery-fresh-dev-v1",
        "suiteHash": result.get("suiteHash") == workload["suiteHash"],
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
            == CANDIDATE_SIMULATIONS
            and result.get("candidateRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "opponentRuntime": (
            result.get("opponentRuntime", {}).get("kind") == "classic-hard"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("budgetKind") == "nodes"
            and result.get("opponentRuntime", {}).get("budgetLimit") == CLASSIC_NODES
            and result.get("opponentRuntime", {}).get("maxDepth") == HARD_MAX_DEPTH
            and result.get("opponentRuntime", {}).get("backendHash")
            == contract["classicBackendSha256"]
        ),
        "draws": statistics.get("draws") == 0,
        "winRate": float(statistics.get("winRate", -1.0)) >= MINIMUM_WIN_RATE,
        "oneSided95Lower": float(interval.get("lower", -1.0))
        > MINIMUM_LOWER_BOUND,
    }
    integrity_keys = tuple(key for key in checks if key not in {"winRate", "oneSided95Lower"})
    if not all(checks[key] for key in integrity_keys):
        raise Phase4F33ConfirmationError(
            f"completed F33 confirmation integrity checks failed: {checks}"
        )
    confirmation_passed = checks["winRate"] and checks["oneSided95Lower"]
    if confirmation_passed:
        band = "f31-fresh-confirmation-passed"
    elif float(statistics["winRate"]) >= 0.50:
        band = "f31-fresh-confirmation-inconclusive"
    else:
        band = "f31-fresh-confirmation-failed"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f33-f31-fresh-confirmation-gate",
            "status": "completed",
            "integrityPassed": True,
            "confirmationPassed": confirmation_passed,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "formalPromotionCandidate": confirmation_passed,
            "formalPromotionAuthorized": False,
            "additionalTrainingAuthorized": False,
            "arenaTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f33(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = verify_phase4f33_preparation(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F33 confirmation gate")
        _verify_self_hash(stored, "gateSha256", "F33 confirmation gate")
        if stored != _gate_from_result(workspace, contract):
            raise Phase4F33ConfirmationError(
                "stored F33 gate differs from arena result"
            )
        return {
            "status": "completed",
            "reused": True,
            "gatePath": GATE.as_posix(),
            "gateSha256": stored["gateSha256"],
            "resultPath": stored["resultPath"],
            "resultSha256": stored["resultSha256"],
            "statistics": stored["statistics"],
            "outcomeBand": stored["outcomeBand"],
        }
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F33ConfirmationError("F33 confirmation requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    confirmation_gpu_hours = _confirmation_gpu_hours(workspace, contract)
    if ledger.total_hours >= baseline + confirmation_gpu_hours:
        raise Phase4F33ConfirmationError("F33 confirmation GPU budget is exhausted")
    _resource_snapshot(workspace)
    candidate, state = _candidate_payload(workspace)
    if candidate != contract["candidate"]:
        raise Phase4F33ConfirmationError("candidate changed before F33 execution")
    suite = load_opening_suite(
        workspace / CONFIRMATION_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f33-confirmation-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
    started = time.monotonic()
    last_ledger = float("-inf")
    last_progress = float("-inf")
    candidate_selector: BoundMoveSelector | None = None
    opponent_selector: BoundMoveSelector | None = None

    def heartbeat(moves: int, pairs: int) -> None:
        nonlocal last_ledger, last_progress
        now = time.monotonic()
        if now - last_ledger >= 5.0:
            total = ledger.update(
                run_id, elapsed_seconds=now - started, status="running"
            )
            if total >= baseline + confirmation_gpu_hours:
                raise Phase4F33ConfirmationError(
                    "F33 confirmation GPU budget was exhausted"
                )
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f33-confirmation-progress",
                        "status": "running",
                        "openingPairs": pairs,
                        "openingPairsTotal": OPENING_PAIRS,
                        "moves": moves,
                        "globalGpuHours": ledger.total_hours,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            last_progress = now

    try:
        with RunLock(
            workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id
        ):
            ledger.begin(run_id, external_hours=0.0)
            candidate_selector = _model_selector(
                workspace, state, device, candidate_identity
            )
            opponent_selector = BoundMoveSelector(
                ClassicHardSelector(
                    budget=ClassicSearchBudget.nodes(CLASSIC_NODES),
                    max_depth=HARD_MAX_DEPTH,
                    request_timeout=max(5.0, CLASSIC_NODES / 1000.0 + 2.0),
                ),
                opponent_identity,
            )
            arena = run_resumable_paired_arena(
                suite,
                candidate_selector,
                opponent_selector,
                store_root=workspace / ARENA_ROOT,
                contract_sha256=_arena_contract_sha256(contract),
                actors=ACTORS,
                heartbeat=heartbeat,
            )
            arena.finalize(
                bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                confidence=CONFIDENCE,
            )
            gate = _gate_from_result(workspace, contract)
            _write_immutable(gate_path, gate, "F33 confirmation gate")
            total = ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="completed"
            )
            return {
                "status": "completed",
                "reused": False,
                "gatePath": GATE.as_posix(),
                "gateSha256": gate["gateSha256"],
                "resultPath": gate["resultPath"],
                "resultSha256": gate["resultSha256"],
                "statistics": gate["statistics"],
                "outcomeBand": gate["outcomeBand"],
                "globalGpuHours": total,
            }
    except BaseException:
        try:
            ledger.update(
                run_id,
                elapsed_seconds=time.monotonic() - started,
                status="interrupted",
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
        "command", choices=("prepare", "extend-budget", "verify", "run")
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f33(args.workspace_root)
    elif args.command == "extend-budget":
        payload = extend_phase4f33_budget(args.workspace_root)
    elif args.command == "verify":
        contract = verify_phase4f33_preparation(args.workspace_root)
        payload = {
            "status": "verified",
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "resultStatus": (
                "completed"
                if (Path(args.workspace_root) / GATE).is_file()
                else "not-started"
            ),
        }
    else:
        payload = run_phase4f33(args.workspace_root, args.device)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
