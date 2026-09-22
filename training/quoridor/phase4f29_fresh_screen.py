"""Freeze and run F27 on fresh development openings against classic-hard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
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
    build_opening_suite,
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
from .challenge_replay import arena_position_orbits, position_orbit_digest
from .phase4f28_conservative_screen import (
    Phase4F28ConservativeScreenError,
    _f27_candidate,
)
from .replay_store import ReplayStore
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .selectors import ClassicHardSelector, MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
F28_GATE = Path("training/runs/phase4/phase4f/stage-b-conservative-screen-v1/screen-gate.json")
EXPECTED_F28_GATE_SHA256 = "974106e2ae34caa6ff95a5d216509859014fb2a19cefc69dfb5e9d94a41c7611"
FROZEN_SUITES = (
    (Path("training/fixtures/openings/dev.json"), "dev"),
    (Path("training/fixtures/openings/final.json"), "final"),
    (Path("training/fixtures/openings/recovery-v1/recovery-dev-v1.json"), "recovery-dev-v1"),
    (Path("training/fixtures/openings/recovery-v1/recovery-final-v1.json"), "recovery-final-v1"),
)
REPLAY_ROOTS = (
    Path("training/runs/phase4/teacher-data/train"),
    Path("training/runs/phase4/teacher-data/holdout"),
    Path("training/runs/phase4/phase4f/recovery-challenge-v2-data/train"),
    Path("training/runs/phase4/phase4f/recovery-challenge-v2-data/holdout"),
)
ROOT = Path("training/runs/phase4/phase4f/fresh-development-screen-v1")
TRAIN_CONFIG = Path("training/configs/phase4f27-stage-b-block-01-v1.json")
FULL_SUITE = ROOT / "recovery-fresh-dev-v1.json"
SCREEN_SUITE = ROOT / "recovery-fresh-dev-screen-32.json"
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
ISOLATION_AUDIT = ROOT / "isolation-audit.json"
CONTRACT = Path("training/configs/phase4f29-fresh-development-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260815
MCTS_SEED = 20260815
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
CANDIDATE_SIMULATIONS = 800
CLASSIC_NODES = 587
ACTORS = 32
SCREEN_GPU_HOURS = 1.0
ARENA_WRITE_RESERVE_BYTES = 32 * 1024**2


class Phase4F29FreshScreenError(RuntimeError):
    """Raised when F29 screen evidence or execution is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F29FreshScreenError("workspace does not contain training/quoridor")
    return workspace


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F29FreshScreenError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F29FreshScreenError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F29FreshScreenError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F29FreshScreenError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F29FreshScreenError(f"existing immutable {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(path, encoded)


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F29FreshScreenError("artifact path escapes workspace") from error


def _resource_snapshot(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F29FreshScreenError("12 GiB cap cannot reserve the F29 arena output")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F29FreshScreenError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _candidate_payload(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    try:
        return _f27_candidate(workspace)
    except Phase4F28ConservativeScreenError as error:
        raise Phase4F29FreshScreenError("F27 candidate lineage failed") from error


def _f28_evidence(workspace: Path, candidate: Mapping[str, object]) -> dict[str, object]:
    gate = _mapping(workspace / F28_GATE, "F28 conservative screen gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F28 conservative screen gate")
    if (
        gate_sha256 != EXPECTED_F28_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("outcomeBand") != "conservative-stage-b-retained"
        or gate.get("candidateRetainedCandidate") is not True
        or gate.get("additionalStageBTrainingAuthorized") is not False
    ):
        raise Phase4F29FreshScreenError("F28 gate does not retain F27")
    return {
        "gatePath": F28_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "candidateModelSha256": candidate["candidateModelSha256"],
        "outcomeBand": gate["outcomeBand"],
    }


def _replay_orbits(workspace: Path, relative: Path) -> tuple[set[str], dict[str, object]]:
    manifest_path = workspace / relative / "manifest.json"
    manifest = _mapping(manifest_path, f"{relative} replay manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F29FreshScreenError("replay capacity is invalid")
    store = ReplayStore(
        workspace / relative,
        max_games=int(capacity["games"]),
        max_plies=int(capacity["plies"]),
        max_bytes=int(capacity["bytes"]),
    )
    orbits: set[str] = set()
    verification = store.verify(
        deep=True,
        game_validator=lambda _game, states: orbits.update(
            position_orbit_digest(state) for state in states[:-1]
        ),
    )
    return orbits, {
        "path": relative.as_posix(),
        "manifestSha256": sha256_file(manifest_path),
        "games": verification.game_count,
        "positions": verification.ply_count,
        "deepVerified": verification.deep,
    }


def _arena_references(workspace: Path) -> list[dict[str, object]]:
    phase4f = workspace / "training/runs/phase4/phase4f"
    references: list[dict[str, object]] = []
    result_paths = {
        *phase4f.glob("*/screen-arena/result.json"),
        *phase4f.glob("*/confirmation-arena/result.json"),
    }
    for path in sorted(result_paths):
        result = _mapping(path, "historical arena result")
        if result.get("status") != "completed" or not isinstance(result.get("resultSha256"), str):
            continue
        references.append(
            {
                "path": _relative(path, workspace),
                "fileSha256": sha256_file(path),
                "resultSha256": result["resultSha256"],
                "games": result.get("games"),
            }
        )
    if not references:
        raise Phase4F29FreshScreenError("no historical Phase 4F arenas found")
    return references


def _load_generator(workspace: Path) -> object:
    path = workspace / "scripts/generate_phase4_openings.py"
    spec = importlib.util.spec_from_file_location("phase4_opening_generator", path)
    if spec is None or spec.loader is None:
        raise Phase4F29FreshScreenError("opening generator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generate_and_select_suite(
    workspace: Path,
) -> tuple[OpeningSuite, OpeningSuite, dict[str, object]]:
    suites: list[OpeningSuite] = []
    suite_refs: list[dict[str, object]] = []
    for relative, name in FROZEN_SUITES:
        path = workspace / relative
        suite = load_opening_suite(path, expected_suite=name)
        suites.append(suite)
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

    forbidden: set[str] = {
        opening_position_orbit_digest(opening)
        for suite in suites
        for opening in suite.openings
    }
    replay_refs: list[dict[str, object]] = []
    for relative in REPLAY_ROOTS:
        print(json.dumps({"kind":"phase4f29-preparation-progress","stage":"replay-isolation","path":relative.as_posix(),"status":"started"}, separators=(",", ":"), sort_keys=True), flush=True)
        orbits, reference = _replay_orbits(workspace, relative)
        forbidden.update(orbits)
        replay_refs.append({**reference, "uniquePositionOrbits": len(orbits)})
    arena_refs = _arena_references(workspace)
    arena_orbits: set[str] = set()
    for reference in arena_refs:
        arena_orbits.update(arena_position_orbits(workspace, reference))
    forbidden.update(arena_orbits)

    generator = _load_generator(workspace)
    fresh = generator.generate_fresh_development_suite(
        tuple(suites), forbidden_orbits=forbidden
    )
    fresh_orbits = {
        opening_position_orbit_digest(opening) for opening in fresh.openings
    }
    overlap = len(fresh_orbits & forbidden)
    if overlap:
        raise Phase4F29FreshScreenError("fresh suite overlaps excluded position orbits")

    def score(opening: object) -> str:
        content_hash = getattr(opening, "content_hash")
        return hashlib.sha256(
            f"{SELECTION_SEED}\0{content_hash}".encode("ascii")
        ).hexdigest()

    selected = tuple(sorted(fresh.openings, key=lambda opening: (score(opening), opening.id))[:OPENING_PAIRS])
    screen = build_opening_suite(selected, suite="recovery-fresh-dev-v1")
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f29-fresh-development-isolation-audit",
            "status": "completed",
            "frozenSuites": suite_refs,
            "replayStores": replay_refs,
            "historicalScreenResults": arena_refs,
            "historicalScreenPositionOrbits": len(arena_orbits),
            "totalExcludedPositionOrbits": len(forbidden),
            "excludedPositionOrbitsSha256": sha256_bytes(canonical_json_bytes(sorted(forbidden))),
            "freshOpeningPositionOrbits": len(fresh_orbits),
            "freshOverlapCount": overlap,
        },
        "auditSha256",
    )
    return fresh, screen, audit


def _stored_suites(workspace: Path) -> tuple[OpeningSuite, OpeningSuite, dict[str, object]]:
    audit = _mapping(workspace / ISOLATION_AUDIT, "F29 isolation audit")
    _verify_self_hash(audit, "auditSha256", "F29 isolation audit")
    full = load_opening_suite(workspace / FULL_SUITE, expected_suite="recovery-fresh-dev-v1")
    screen = load_opening_suite(workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1")
    selected_ids = {opening.id for opening in screen.openings}
    if len(full.openings) != 200 or len(screen.openings) != OPENING_PAIRS or not selected_ids <= {opening.id for opening in full.openings}:
        raise Phase4F29FreshScreenError("stored fresh opening suites are invalid")
    return full, screen, audit


def _suite_reference(workspace: Path, full: OpeningSuite, screen: OpeningSuite, audit: Mapping[str, object]) -> dict[str, object]:
    return {
        "fullPath": FULL_SUITE.as_posix(),
        "fullFileSha256": sha256_file(workspace / FULL_SUITE),
        "fullSuiteHash": full.content_hash,
        "fullOpeningPairs": len(full.openings),
        "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
        "isolationAuditSha256": audit["auditSha256"],
        "freshOverlapCount": audit["freshOverlapCount"],
        "selectionAlgorithm": "sha256(seed-ascii+nul+opening-content-hash)-lowest-v1",
        "selectionSeed": SELECTION_SEED,
        "selectedOpeningIds": [opening.id for opening in screen.openings],
        "selectedSuiteHash": screen.content_hash,
        "selectedOpeningPairs": len(screen.openings),
    }


def _verify_audit_bindings(workspace: Path, audit: Mapping[str, object]) -> None:
    if (
        audit.get("status") != "completed"
        or audit.get("freshOverlapCount") != 0
        or audit.get("freshOpeningPositionOrbits") != 200
    ):
        raise Phase4F29FreshScreenError("fresh isolation audit did not pass")
    for key, hash_field in (("frozenSuites", "fileSha256"), ("replayStores", "manifestSha256"), ("historicalScreenResults", "fileSha256")):
        references = audit.get(key)
        if not isinstance(references, list) or not references:
            raise Phase4F29FreshScreenError(f"isolation audit {key} is empty")
        for reference in references:
            if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
                raise Phase4F29FreshScreenError(f"isolation audit {key} reference is invalid")
            path = workspace / str(reference["path"])
            if key == "replayStores":
                path = path / "manifest.json"
            if path.is_symlink() or not path.is_file() or sha256_file(path) != reference.get(hash_field):
                raise Phase4F29FreshScreenError(f"isolation audit {key} binding changed")


def _identities(contract: Mapping[str, object]) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    workload = contract["workload"]
    return (
        SelectorIdentity(
            selector_id="phase4f-stage-b-conservative-g00000001-s000000000192-fresh-dev",
            artifact_hash=str(candidate["candidateModelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id="classic-hard-phase4f29-unseen-screen-v1",
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
                "phase4f29ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["selectedSuiteHash"],
            }
        )
    )


def prepare_phase4f29(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / CONTRACT).exists():
        contract = verify_phase4f29_preparation(workspace)
        return {
            "status": "prepared",
            "reused": True,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
        }
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F29FreshScreenError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F29FreshScreenError("insufficient global GPU budget for F29 screen")
    resources = _resource_snapshot(workspace)
    candidate, _ = _candidate_payload(workspace)
    evidence = _f28_evidence(workspace, candidate)
    full, screen, audit = _generate_and_select_suite(workspace)
    _write_immutable(workspace / FULL_SUITE, opening_suite_to_payload(full), "F29 full fresh suite")
    _write_immutable(
        workspace / SCREEN_SUITE,
        opening_suite_to_payload(screen),
        "F29 screen opening suite",
    )
    _write_immutable(workspace / ISOLATION_AUDIT, audit, "F29 isolation audit")
    suite_reference = _suite_reference(workspace, full, screen, audit)
    workload = _self_hashed(
        {
            "suite": "recovery-fresh-dev-v1",
            "suiteHash": screen.content_hash,
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
            "kind": "phase4f29-fresh-development-classic-hard-screen",
            "status": "frozen",
            "formalPromotion": False,
            "additionalStageBTrainingAuthorized": False,
            "stageBTrainingAuthorized": False,
            "confirmationArenaAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "classicBackendSha256": canonical_classic_backend_hash(),
            "candidate": candidate,
            "f28Evidence": evidence,
            "openingSuite": {
                **suite_reference,
                "selectedPath": SCREEN_SUITE.as_posix(),
                "selectedFileSha256": sha256_file(workspace / SCREEN_SUITE),
                "contentPositionOrbitAndTrajectoryDisjoint": True,
            },
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger.total_hours,
                "screenLimitHours": SCREEN_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "insufficientBelowWinRate": 0.35,
                "belowConfirmationBelowWinRate": 0.50,
                "confirmationCandidateAtOrAboveWinRate": 0.50,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F29 screen contract")
    preparation = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f29-fresh-development-screen-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "screenAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "confirmationArenaAuthorized": False,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "source": source,
            "candidate": candidate,
            "openingSuite": contract["openingSuite"],
            "globalGpuHours": ledger.total_hours,
            "resources": resources,
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, preparation, "F29 preparation")
    verify_phase4f29_preparation(workspace)
    return preparation


def verify_phase4f29_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F29 screen contract")
    _verify_self_hash(contract, "contractSha256", "F29 screen contract")
    preparation = _mapping(workspace / PREPARATION_REPORT, "F29 preparation")
    _verify_self_hash(preparation, "reportSha256", "F29 preparation")
    candidate, _ = _candidate_payload(workspace)
    evidence = _f28_evidence(workspace, candidate)
    full, screen, audit = _stored_suites(workspace)
    _verify_audit_bindings(workspace, audit)
    reference = _suite_reference(workspace, full, screen, audit)
    selected_path = workspace / SCREEN_SUITE
    workload = contract.get("workload")
    if not isinstance(workload, Mapping):
        raise Phase4F29FreshScreenError("F29 workload is invalid")
    _verify_self_hash(workload, "workloadSha256", "F29 workload")
    if (
        contract.get("kind") != "phase4f29-fresh-development-classic-hard-screen"
        or contract.get("status") != "frozen"
        or contract.get("formalPromotion") is not False
        or contract.get("additionalStageBTrainingAuthorized") is not False
        or contract.get("stageBTrainingAuthorized") is not False
        or contract.get("confirmationArenaAuthorized") is not False
        or contract.get("screenTrajectoriesTrainingUseAllowed") is not False
        or contract.get("source") != fingerprint_sources(workspace).to_payload()
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("classicBackendSha256") != canonical_classic_backend_hash()
        or contract.get("candidate") != candidate
        or contract.get("f28Evidence") != evidence
        or contract.get("openingSuite")
        != {
            **reference,
            "selectedPath": SCREEN_SUITE.as_posix(),
            "selectedFileSha256": sha256_file(selected_path),
            "contentPositionOrbitAndTrajectoryDisjoint": True,
        }
        or load_opening_suite(selected_path, expected_suite="recovery-fresh-dev-v1") != screen
        or preparation.get("contractSha256") != contract.get("contractSha256")
        or preparation.get("source") != contract.get("source")
        or preparation.get("screenAuthorized") is not True
        or preparation.get("additionalStageBTrainingAuthorized") is not False
        or preparation.get("confirmationArenaAuthorized") is not False
    ):
        raise Phase4F29FreshScreenError("F29 preparation provenance changed")
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F29FreshScreenError("F29 screen GPU budget is unavailable")
    _resource_snapshot(workspace)
    return contract


def _model_selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    identity: SelectorIdentity,
) -> BoundMoveSelector:
    config = load_experiment_config(workspace / TRAIN_CONFIG)
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
        raise Phase4F29FreshScreenError("candidate state does not match frozen network") from error
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
        MCTSSelector(evaluator, search, search_factory=NativeMCTS, engine="native-mcts"),
        identity,
    )


def _gate_from_result(workspace: Path, contract: Mapping[str, object]) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    candidate_identity, opponent_identity = _identities(contract)
    workload = contract["workload"]
    checks = {
        "contract": result.get("contractSha256") == _arena_contract_sha256(contract),
        "suite": result.get("suite") == "recovery-fresh-dev-v1",
        "suiteHash": result.get("suiteHash") == workload["suiteHash"],
        "openingPairs": result.get("openingPairs") == OPENING_PAIRS,
        "games": result.get("games") == GAMES,
        "candidateIdentity": result.get("candidateIdentity") == candidate_identity.to_dict(),
        "opponentIdentity": result.get("opponentIdentity") == opponent_identity.to_dict(),
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
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F29FreshScreenError(f"completed F29 screen checks failed: {checks}")
    statistics = result["statistics"]
    win_rate = float(statistics["winRate"])
    if win_rate < 0.35:
        band = "fresh-development-insufficient"
    elif win_rate < 0.50:
        band = "fresh-development-below-confirmation"
    else:
        band = "final-confirmation-candidate"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f29-fresh-development-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "additionalStageBTrainingAuthorized": False,
            "stageBTrainingAuthorized": False,
            "confirmationArenaCandidate": win_rate >= 0.50,
            "formalPromotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f29(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = verify_phase4f29_preparation(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F29 screen gate")
        _verify_self_hash(stored, "gateSha256", "F29 screen gate")
        if stored != _gate_from_result(workspace, contract):
            raise Phase4F29FreshScreenError("stored F29 gate differs from arena result")
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
        raise Phase4F29FreshScreenError("F29 screen requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + SCREEN_GPU_HOURS:
        raise Phase4F29FreshScreenError("F29 screen GPU budget is exhausted")
    _resource_snapshot(workspace)
    candidate, state = _candidate_payload(workspace)
    if candidate != contract["candidate"]:
        raise Phase4F29FreshScreenError("candidate changed before F29 CUDA execution")
    suite = load_opening_suite(workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1")
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f29-screen-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
            if total >= baseline + SCREEN_GPU_HOURS:
                raise Phase4F29FreshScreenError("F29 screen GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f29-screen-progress",
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
            candidate_selector = _model_selector(workspace, state, device, candidate_identity)
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
            _write_immutable(gate_path, gate, "F29 screen gate")
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
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f29(args.workspace_root)
    elif args.command == "verify":
        contract = verify_phase4f29_preparation(args.workspace_root)
        payload = {
            "status": "verified",
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "resultStatus": (
                "completed" if (Path(args.workspace_root) / GATE).is_file() else "not-started"
            ),
        }
    else:
        payload = run_phase4f29(args.workspace_root, args.device)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
