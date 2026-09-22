"""Prepare, run, and verify scaled F21 versus classic-hard confirmation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import arena_position_orbits
from .classic_bridge import CLASSIC_ENGINE, HARD_MAX_DEPTH, ClassicSearchBudget
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import (
    OpeningSuite,
    load_opening_suite,
    opening_position_orbit_digest,
    opening_suite_to_payload,
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
from .phase4f63_value_calibration_review import (
    F64_DESIGN,
    GATE as F63_GATE,
    REPORT as F63_REPORT,
    verify_phase4f63,
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
from .selectors import ClassicHardSelector, MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f64-scaled-f21-classic-confirmation-v1")
SUITE_PATH = ROOT / "fresh-scaled-classic-confirmation-200.json"
ISOLATION_AUDIT = ROOT / "isolation-audit.json"
CONTRACT = ROOT / "execution-contract.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
ARENA_ROOT = ROOT / "confirmation-arena"
GATE = ROOT / "confirmation-gate.json"

F59_POOL = Path(
    "training/runs/phase4/phase4f/f59-kl-trust-region-review-v1/"
    "f60-fresh-lineage-pool-200.json"
)
F22_RESULT = Path(
    "training/runs/phase4/phase4f/stage-a-screen-v2/screen-arena/result.json"
)
F61_RESULT = Path(
    "training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-arena/result.json"
)
F62_RESULT = Path(
    "training/runs/phase4/phase4f/f62-f21-value-scale-confirmation-v1/"
    "confirmation-arena/result.json"
)

EXPECTED_F63_REPORT_SHA256 = (
    "4a643a452255b2b0e1d71287cb554d4f09a787b74e46b68d1f9ca7a27fba21fa"
)
EXPECTED_F63_GATE_SHA256 = (
    "129764fa5371cf78ae7fe665d4a3ca44870ff801949539beb28eae0d2cb9eb6f"
)
EXPECTED_F64_DESIGN_SHA256 = (
    "2f64d76d24639faa4e86675c17806af0142e8abe164805db735c925317b6a60a"
)

SUITE_NAME = "recovery-fresh-dev-v1"
OPENING_PAIRS = 200
GAMES = 400
SIMULATIONS = 800
VALUE_SCALE = 0.5
CLASSIC_NODES = 587
ACTORS = 32
MCTS_SEED = 20260922
BOOTSTRAP_SEED = 20260922
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
MINIMUM_WIN_RATE = 0.55
MINIMUM_LOWER_BOUND = 0.50
CONFIRMATION_GPU_HOURS = 8.0
GLOBAL_GPU_HOURS_LIMIT = 120.0
WRITE_RESERVE_BYTES = 128 * 1024**2

SOURCE_PATHS = (
    "scripts/generate_phase4_openings.py",
    "training/quoridor/challenge_replay.py",
    "training/quoridor/classic_bridge.py",
    "training/quoridor/evaluator.py",
    "training/quoridor/native_mcts.py",
    "training/quoridor/network.py",
    "training/quoridor/openings.py",
    "training/quoridor/paired_arena.py",
    "training/quoridor/phase4_arena.py",
    "training/quoridor/phase4f29_fresh_screen.py",
    "training/quoridor/phase4f61_value_scale_screen.py",
    "training/quoridor/phase4f63_value_calibration_review.py",
    "training/quoridor/phase4f64_scaled_classic_confirmation.py",
    "training/quoridor/selectors.py",
)


class Phase4F64Error(RuntimeError):
    """Raised when F64 provenance, resources, or arena evidence is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F64Error("workspace does not contain training/quoridor")
    return workspace


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F64Error("F64 reserve exceeds the frozen 16 GiB artifact cap")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F64Error("F64 reserve violates the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _f63_evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f63(workspace)
    report = _mapping(workspace / F63_REPORT, "F63 report")
    gate = _mapping(workspace / F63_GATE, "F63 gate")
    design = _mapping(workspace / F64_DESIGN, "F64 design")
    report_sha = _verify_self_hash(report, "reportSha256", "F63 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F63 gate")
    design_sha = _verify_self_hash(design, "designSha256", "F64 design")
    if (
        verification.get("status") != "verified"
        or report_sha != EXPECTED_F63_REPORT_SHA256
        or gate_sha != EXPECTED_F63_GATE_SHA256
        or design_sha != EXPECTED_F64_DESIGN_SHA256
        or gate.get("f64ConfirmationPreparationAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
        or gate.get("valueHeadTrainingAuthorized") is not False
        or design.get("confirmationPreparationAuthorized") is not True
        or design.get("candidate", {}).get("valueScale") != VALUE_SCALE
        or design.get("openingIsolation", {}).get("openingPairs") != OPENING_PAIRS
        or design.get("openingIsolation", {}).get("games") != GAMES
    ):
        raise Phase4F64Error("F63 does not authorize the frozen F64 preparation")
    return {
        "reportPath": F63_REPORT.as_posix(),
        "reportSha256": report_sha,
        "gatePath": F63_GATE.as_posix(),
        "gateSha256": gate_sha,
        "designPath": F64_DESIGN.as_posix(),
        "designSha256": design_sha,
    }


def _fresh_suite(workspace: Path) -> tuple[OpeningSuite, dict[str, object]]:
    frozen: list[OpeningSuite] = []
    frozen_refs: list[dict[str, object]] = []
    for relative, name in FROZEN_SUITES:
        suite = load_opening_suite(workspace / relative, expected_suite=name)
        frozen.append(suite)
        frozen_refs.append(
            {
                "path": relative.as_posix(),
                "name": name,
                "fileSha256": sha256_file(workspace / relative),
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
    replay_refs: list[dict[str, object]] = []
    for relative in REPLAY_ROOTS:
        print(
            canonical_json_bytes(
                {
                    "kind": "phase4f64-preparation-progress",
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
            raise Phase4F64Error(f"cannot verify excluded replay {relative}") from error
        forbidden.update(orbits)
        replay_refs.append({**reference, "uniquePositionOrbits": len(orbits)})
    arena_refs = _arena_references(workspace)
    arena_orbits: set[str] = set()
    for reference in arena_refs:
        arena_orbits.update(arena_position_orbits(workspace, reference))
    forbidden.update(arena_orbits)
    historical = {str(reference["path"]) for reference in arena_refs}
    for required in (F22_RESULT, F61_RESULT, F62_RESULT):
        if required.as_posix() not in historical:
            raise Phase4F64Error(f"required excluded arena is absent: {required}")
    fresh = _load_generator(workspace).generate_fresh_development_suite(
        tuple(frozen), forbidden_orbits=forbidden
    )
    fresh_orbits = {
        opening_position_orbit_digest(opening) for opening in fresh.openings
    }
    overlap = len(fresh_orbits & forbidden)
    if len(fresh.openings) != OPENING_PAIRS or len(fresh_orbits) != OPENING_PAIRS:
        raise Phase4F64Error("fresh F64 suite is incomplete")
    if overlap:
        raise Phase4F64Error("fresh F64 suite overlaps excluded position orbits")
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f64-scaled-classic-confirmation-isolation-audit",
            "status": "completed",
            "frozenSuites": frozen_refs,
            "f59FullPool": {
                "path": F59_POOL.as_posix(),
                "fileSha256": sha256_file(workspace / F59_POOL),
                "suiteHash": pool.content_hash,
                "openingPairs": len(pool.openings),
            },
            "replayStores": replay_refs,
            "historicalArenaResults": arena_refs,
            "historicalArenaPositionOrbits": len(arena_orbits),
            "totalExcludedPositionOrbits": len(forbidden),
            "excludedPositionOrbitsSha256": sha256_bytes(
                canonical_json_bytes(sorted(forbidden))
            ),
            "freshOpeningPositionOrbits": len(fresh_orbits),
            "freshOverlapCount": overlap,
            "confirmationOpeningPairs": len(fresh.openings),
            "f22TrajectoriesExcluded": True,
            "f61TrajectoriesExcluded": True,
            "f62TrajectoriesExcluded": True,
            "allTrainingReplayExcluded": True,
        },
        "auditSha256",
    )
    return fresh, audit


def _stored_suite(workspace: Path) -> tuple[OpeningSuite, dict[str, object]]:
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    audit = _mapping(workspace / ISOLATION_AUDIT, "F64 isolation audit")
    _verify_self_hash(audit, "auditSha256", "F64 isolation audit")
    orbits = {opening_position_orbit_digest(opening) for opening in suite.openings}
    if (
        len(suite.openings) != OPENING_PAIRS
        or len(orbits) != OPENING_PAIRS
        or audit.get("kind")
        != "phase4f64-scaled-classic-confirmation-isolation-audit"
        or audit.get("freshOverlapCount") != 0
        or audit.get("confirmationOpeningPairs") != OPENING_PAIRS
        or audit.get("f22TrajectoriesExcluded") is not True
        or audit.get("f61TrajectoriesExcluded") is not True
        or audit.get("f62TrajectoriesExcluded") is not True
        or audit.get("allTrainingReplayExcluded") is not True
    ):
        raise Phase4F64Error("stored F64 opening evidence is invalid")
    return suite, audit


def _candidate_runtime(source_sha: str, native_sha: str) -> dict[str, object]:
    return _self_hashed(
        {
            "kind": "native-mcts",
            "sourceSha256": source_sha,
            "nativeBackendSha256": native_sha,
            "configuredSimulations": SIMULATIONS,
            "valueScale": VALUE_SCALE,
            "policyTransform": "identity",
            "rootNoise": False,
            "temperature": 0.0,
            "mctsSeed": MCTS_SEED,
        },
        "runtimeSha256",
    )


def _identities(
    contract: Mapping[str, object],
) -> tuple[SelectorIdentity, SelectorIdentity]:
    return (
        SelectorIdentity(
            selector_id=str(contract["candidate"]["selectorId"]),
            artifact_hash=EXPECTED_F21_MODEL_SHA256,
            config_hash=str(contract["candidate"]["runtime"]["runtimeSha256"]),
            source_hash=str(contract["source"]["sha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id=str(contract["opponent"]["selectorId"]),
            artifact_hash=str(contract["classicBackendSha256"]),
            config_hash=str(contract["workload"]["workloadSha256"]),
            source_hash=str(contract["source"]["sha256"]),
            engine=CLASSIC_ENGINE,
        ),
    )


def arena_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f64ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["suiteHash"],
                "candidateRuntimeSha256": contract["candidate"]["runtime"][
                    "runtimeSha256"
                ],
            }
        )
    )


def outcome_band(win_rate: float, lower_bound: float) -> str:
    if win_rate >= MINIMUM_WIN_RATE and lower_bound > MINIMUM_LOWER_BOUND:
        return "scaled-f21-runtime-promotion-candidate"
    return "value-scale-external-transfer-not-confirmed"


def prepare_phase4f64(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / PREPARATION_REPORT).exists():
        return {**verify_preparation(workspace), "reused": True}
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F64Error("a running GPU ledger entry blocks F64 preparation")
    if ledger.total_hours + CONFIRMATION_GPU_HOURS > GLOBAL_GPU_HOURS_LIMIT:
        raise Phase4F64Error("insufficient global GPU budget for F64")
    evidence = _f63_evidence(workspace)
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256:
        raise Phase4F64Error("F21 config bytes changed")
    _load_f21_state(workspace)
    resources = _resources(workspace)
    suite, audit = _fresh_suite(workspace)
    _write_immutable(workspace / SUITE_PATH, opening_suite_to_payload(suite), "F64 suite")
    _write_immutable(workspace / ISOLATION_AUDIT, audit, "F64 isolation audit")
    source = fingerprint_sources(workspace, paths=SOURCE_PATHS).to_payload()
    native_sha = loaded_native_backend_hash()
    classic_sha = canonical_classic_backend_hash()
    workload = _self_hashed(
        {
            "suite": SUITE_NAME,
            "suiteHash": suite.content_hash,
            "openingPairs": OPENING_PAIRS,
            "games": GAMES,
            "candidateSimulations": SIMULATIONS,
            "candidateValueScale": VALUE_SCALE,
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
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f64-scaled-f21-classic-hard-confirmation-contract",
            "status": "frozen",
            "hypothesis": "F21-value-scale-0.5-transfers-to-classic-hard",
            "source": source,
            "nativeBackendSha256": native_sha,
            "classicBackendSha256": classic_sha,
            "model": {
                "path": F21_MODEL.as_posix(),
                "fileSha256": EXPECTED_F21_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "configPath": F21_CONFIG.as_posix(),
                "configFileSha256": EXPECTED_F21_CONFIG_SHA256,
            },
            "candidate": {
                "selectorId": "phase4f21-value-scale-0.5-classic-confirmation",
                "runtime": _candidate_runtime(str(source["sha256"]), native_sha),
            },
            "opponent": {
                "selectorId": "classic-hard-phase4f64-fresh-confirmation",
                "engine": CLASSIC_ENGINE,
            },
            "f63Evidence": evidence,
            "openingSuite": {
                "path": SUITE_PATH.as_posix(),
                "fileSha256": sha256_file(workspace / SUITE_PATH),
                "suiteHash": suite.content_hash,
                "openingPairs": OPENING_PAIRS,
                "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
                "isolationAuditSha256": audit["auditSha256"],
                "freshOverlapCount": 0,
                "contentPositionOrbitAndTrajectoryDisjoint": True,
            },
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger.total_hours,
                "confirmationLimitHours": CONFIRMATION_GPU_HOURS,
                "globalLimitHours": GLOBAL_GPU_HOURS_LIMIT,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "minimumWinRate": MINIMUM_WIN_RATE,
                "minimumOneSided95Lower": MINIMUM_LOWER_BOUND,
                "confidenceEquivalentTwoSided": CONFIDENCE,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "authorizations": {
                "inferenceOnlyConfirmationAuthorized": True,
                "optimizerStepsAuthorized": 0,
                "trainingAuthorized": False,
                "valueHeadTrainingAuthorized": False,
                "promotionAuthorized": False,
                "arenaTrajectoriesTrainingUseAllowed": False,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F64 contract")
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f64-scaled-f21-classic-confirmation-preparation",
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
            "valueHeadTrainingAuthorized": False,
            "promotionAuthorized": False,
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, report, "F64 preparation")
    verify_preparation(workspace)
    return {**report, "reused": False}


def _contract(workspace: Path) -> dict[str, object]:
    contract = _mapping(workspace / CONTRACT, "F64 contract")
    _verify_self_hash(contract, "contractSha256", "F64 contract")
    auth = contract.get("authorizations", {})
    if (
        contract.get("kind")
        != "phase4f64-scaled-f21-classic-hard-confirmation-contract"
        or contract.get("status") != "frozen"
        or auth.get("inferenceOnlyConfirmationAuthorized") is not True
        or auth.get("optimizerStepsAuthorized") != 0
        or auth.get("trainingAuthorized") is not False
        or auth.get("valueHeadTrainingAuthorized") is not False
        or auth.get("promotionAuthorized") is not False
        or auth.get("arenaTrajectoriesTrainingUseAllowed") is not False
    ):
        raise Phase4F64Error("F64 authorization contract is invalid")
    return contract


def verify_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    report = _mapping(workspace / PREPARATION_REPORT, "F64 preparation")
    report_sha = _verify_self_hash(report, "reportSha256", "F64 preparation")
    suite, audit = _stored_suite(workspace)
    if (
        contract.get("source")
        != fingerprint_sources(workspace, paths=SOURCE_PATHS).to_payload()
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("classicBackendSha256") != canonical_classic_backend_hash()
        or contract.get("f63Evidence") != _f63_evidence(workspace)
        or contract.get("openingSuite", {}).get("fileSha256")
        != sha256_file(workspace / SUITE_PATH)
        or contract.get("openingSuite", {}).get("suiteHash") != suite.content_hash
        or contract.get("openingSuite", {}).get("isolationAuditSha256")
        != audit.get("auditSha256")
        or report.get("contractSha256") != contract.get("contractSha256")
        or report.get("isolationAuditSha256") != audit.get("auditSha256")
    ):
        raise Phase4F64Error("F64 frozen preparation changed")
    _load_f21_state(workspace)
    budget = contract.get("budget", {})
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours
        > float(budget["baselineGlobalHours"]) + CONFIRMATION_GPU_HOURS
    ):
        raise Phase4F64Error("F64 GPU budget is unavailable")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f64-scaled-classic-confirmation-preparation-verification",
        "status": "verified",
        "contractPath": CONTRACT.as_posix(),
        "contractSha256": contract["contractSha256"],
        "reportPath": PREPARATION_REPORT.as_posix(),
        "reportSha256": report_sha,
        "openingSuitePath": SUITE_PATH.as_posix(),
        "openingSuiteHash": suite.content_hash,
        "openingPairs": len(suite.openings),
        "candidateValueScale": VALUE_SCALE,
        "inferenceOnlyConfirmationAuthorized": True,
        "formalTrainingAuthorized": False,
        "valueHeadTrainingAuthorized": False,
        "promotionAuthorized": False,
    }


def _candidate_selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    identity: SelectorIdentity,
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
        raise Phase4F64Error("F21 state does not fit the frozen network") from error
    evaluator = ValueScaledEvaluator(
        TorchEvaluator(model.to(device).eval(), device=device, use_amp=True),
        VALUE_SCALE,
    )
    evaluator.enable_native_encoder(load_native_module())
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
        "suiteHash": result.get("suiteHash") == contract["openingSuite"]["suiteHash"],
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
            result.get("opponentRuntime", {}).get("kind") == "classic-hard"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("budgetKind") == "nodes"
            and result.get("opponentRuntime", {}).get("budgetLimit") == CLASSIC_NODES
            and result.get("opponentRuntime", {}).get("maxDepth") == HARD_MAX_DEPTH
            and result.get("opponentRuntime", {}).get("backendHash")
            == contract["classicBackendSha256"]
        ),
        "draws": statistics.get("draws") == 0,
        "minimumWinRate": float(statistics.get("winRate", -1.0))
        >= MINIMUM_WIN_RATE,
        "oneSided95Lower": float(interval.get("lower", -1.0))
        > MINIMUM_LOWER_BOUND,
    }
    decision = {"minimumWinRate", "oneSided95Lower"}
    if not all(value for key, value in checks.items() if key not in decision):
        raise Phase4F64Error(f"completed F64 integrity checks failed: {checks}")
    band = outcome_band(
        float(statistics["winRate"]), float(statistics["winRateCI"]["lower"])
    )
    passed = band == "scaled-f21-runtime-promotion-candidate"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f64-scaled-f21-classic-confirmation-gate",
            "status": "completed",
            "integrityPassed": True,
            "confirmationPassed": passed,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "candidateValueScale": VALUE_SCALE,
            "statistics": statistics,
            "outcomeBand": band,
            "runtimePromotionCandidate": passed,
            "runtimePromotionAuthorized": False,
            "formalTrainingAuthorized": False,
            "valueHeadTrainingAuthorized": False,
            "arenaTrajectoriesTrainingUseAllowed": False,
            "nextStage": {
                "name": "Phase 4F65 runtime calibration release review",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )


def run_phase4f64(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    verify_preparation(workspace)
    contract = _contract(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        gate = _mapping(gate_path, "F64 gate")
        _verify_self_hash(gate, "gateSha256", "F64 gate")
        if gate != gate_from_result(workspace, contract):
            raise Phase4F64Error("stored F64 gate differs from arena result")
        return {**gate, "reused": True}
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F64Error("F64 confirmation requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + CONFIRMATION_GPU_HOURS:
        raise Phase4F64Error("F64 confirmation GPU budget is exhausted")
    _resources(workspace)
    state = _load_f21_state(workspace)
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f64-confirmation-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
            if total >= baseline + CONFIRMATION_GPU_HOURS:
                raise Phase4F64Error("F64 confirmation GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f64-scaled-classic-confirmation-progress",
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
        with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
            ledger.begin(run_id, external_hours=0.0)
            candidate_selector = _candidate_selector(
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
            _write_immutable(gate_path, gate, "F64 gate")
            total = ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="completed"
            )
            return {**gate, "reused": False, "globalGpuHours": total}
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


def verify_phase4f64(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    verify_preparation(workspace)
    contract = _contract(workspace)
    gate = _mapping(workspace / GATE, "F64 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F64 gate")
    if gate != gate_from_result(workspace, contract):
        raise Phase4F64Error("stored F64 gate differs from arena result")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f64-scaled-classic-confirmation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "statistics": gate["statistics"],
        "outcomeBand": gate["outcomeBand"],
        "confirmationPassed": gate["confirmationPassed"],
        "runtimePromotionAuthorized": False,
        "formalTrainingAuthorized": False,
        "valueHeadTrainingAuthorized": False,
        "globalGpuHours": ledger.total_hours,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f64(args.workspace_root)
    elif args.command == "run":
        payload = run_phase4f64(args.workspace_root, args.device)
    else:
        payload = verify_phase4f64(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
