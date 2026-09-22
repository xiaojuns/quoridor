"""Prepare, run, and verify the frozen F21 value-scale causal screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import arena_position_orbits
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import (
    Opening,
    OpeningSuite,
    build_opening_suite,
    load_opening_suite,
    opening_position_orbit_digest,
    opening_suite_to_payload,
)
from .paired_arena import BoundMoveSelector, SelectorIdentity, loaded_native_backend_hash
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
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
ROOT = Path("training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1")
SUITE_PATH = ROOT / "fresh-value-scale-screen-32.json"
SELECTION_AUDIT = ROOT / "opening-selection-audit.json"
CONTRACT = ROOT / "execution-contract.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"

F21_MODEL = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/checkpoints/"
    "candidate-model-checkpoint-g00000001-s000000000192.pt"
)
F21_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
FRESH_POOL = Path(
    "training/runs/phase4/phase4f/f59-kl-trust-region-review-v1/"
    "f60-fresh-lineage-pool-200.json"
)
F60_SUITE = Path(
    "training/runs/phase4/phase4f/f59-kl-trust-region-review-v1/"
    "f60-fresh-lineage-screen-32.json"
)
F60_GATE = Path(
    "training/runs/phase4/phase4f/f60-kl-lineage-screen-v1/screen-gate.json"
)
F60_RESULT = Path(
    "training/runs/phase4/phase4f/f60-kl-lineage-screen-v1/"
    "screen-arena/result.json"
)

EXPECTED_F21_MODEL_SHA256 = (
    "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
)
EXPECTED_F21_STATE_SHA256 = (
    "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
)
EXPECTED_F21_CONFIG_SHA256 = (
    "bdaa011c52a3a3f0925dcf603d52bd19efe4b7cee6039d6e03140b14d007a14e"
)
EXPECTED_FRESH_POOL_SHA256 = (
    "f0da226069d6697aa9e65c40b8209a70d580e2f6346d0e2ffa202149292902d0"
)
EXPECTED_FRESH_POOL_SUITE_HASH = (
    "sha256:5507cfc5693201df742ab1157ba733c5478e3fffdf48e566b73744194c7d461c"
)
EXPECTED_F60_SUITE_SHA256 = (
    "93f889d4b48b31e8823a270da337d9e33af3eebf0075a173221371c5f75f52d4"
)
EXPECTED_F60_GATE_SHA256 = (
    "31549fa1af040cf6fb74997b896b207579bc68937d3bf7bb9f7669236aa1d0b1"
)
EXPECTED_F60_RESULT_SHA256 = (
    "c4f40b6625f9a746841d2a72529633a88ce94eae4942d4f941bc2b801a537c68"
)

SUITE_NAME = "recovery-fresh-dev-v1"
OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260917
MCTS_SEED = 20260917
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
CANDIDATE_VALUE_SCALE = 0.5
OPPONENT_VALUE_SCALE = 1.0
SCREEN_GPU_HOURS = 2.0
WRITE_RESERVE_BYTES = 64 * 1024**2

SOURCE_PATHS = (
    "training/quoridor/evaluator.py",
    "training/quoridor/native_mcts.py",
    "training/quoridor/network.py",
    "training/quoridor/openings.py",
    "training/quoridor/paired_arena.py",
    "training/quoridor/phase4_arena.py",
    "training/quoridor/phase4f61_value_scale_screen.py",
    "training/quoridor/selectors.py",
)


class Phase4F61Error(RuntimeError):
    """Raised when the frozen value-scale causal screen is invalid."""


@dataclass
class ValueScaledEvaluator:
    """Preserve policy logits and scale only non-terminal network values."""

    evaluator: TorchEvaluator
    value_scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.value_scale, bool)
            or not isinstance(self.value_scale, (int, float))
            or not math.isfinite(float(self.value_scale))
            or not 0.0 <= float(self.value_scale) <= 1.0
        ):
            raise ValueError("value_scale must be finite and in [0, 1]")
        self.value_scale = float(self.value_scale)

    @property
    def device(self) -> torch.device:
        return torch.device(self.evaluator.device)

    def enable_native_encoder(self, module: object) -> None:
        self.evaluator.enable_native_encoder(module)

    def evaluate(self, state: object) -> tuple[Sequence[float], float]:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self, states: Iterable[object]
    ) -> list[tuple[Sequence[float], float]]:
        return [
            (policy, float(value) * self.value_scale)
            for policy, value in self.evaluator.evaluate_batch(states)
        ]


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F61Error("workspace does not contain training/quoridor")
    return workspace


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F61Error("F61 artifact escaped the workspace") from error


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F61Error("F61 reserve exceeds the frozen 16 GiB artifact cap")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F61Error("F61 reserve violates the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _load_f21_state(workspace: Path) -> Mapping[str, torch.Tensor]:
    path = workspace / F21_MODEL
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != EXPECTED_F21_MODEL_SHA256
    ):
        raise Phase4F61Error("F21 model file failed validation")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F61Error("F21 model payload is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != EXPECTED_F21_STATE_SHA256
    ):
        raise Phase4F61Error("F21 model state failed validation")
    return state


def _selection_rank(opening: Opening) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{SELECTION_SEED}\0{opening.content_hash}".encode("ascii")
    ).hexdigest()
    return digest, opening.id


def select_fresh_openings(
    pool: OpeningSuite,
    used: OpeningSuite,
    excluded_position_orbits: set[str],
) -> tuple[OpeningSuite, dict[str, object]]:
    used_ids = {opening.id for opening in used.openings}
    eligible = tuple(
        opening
        for opening in pool.openings
        if opening.id not in used_ids
        and opening_position_orbit_digest(opening) not in excluded_position_orbits
    )
    if len(eligible) < OPENING_PAIRS:
        raise Phase4F61Error("F61 does not have 32 fresh isolated openings")
    selected = tuple(sorted(eligible, key=_selection_rank)[:OPENING_PAIRS])
    suite = build_opening_suite(selected, suite=SUITE_NAME)
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f61-fresh-opening-selection-audit",
            "status": "completed",
            "selectionSeed": SELECTION_SEED,
            "selectionAlgorithm": "sha256(seed-ascii+nul+opening-content-hash)-lowest-v1",
            "poolOpenings": len(pool.openings),
            "f60UsedOpenings": len(used.openings),
            "f60ArenaPositionOrbits": len(excluded_position_orbits),
            "eligibleOpenings": len(eligible),
            "selectedOpeningIds": [opening.id for opening in suite.openings],
            "selectedOpeningOrbits": [
                opening_position_orbit_digest(opening) for opening in suite.openings
            ],
            "f60OpeningIdOverlap": len(
                {opening.id for opening in suite.openings} & used_ids
            ),
            "f60ArenaInitialOrbitOverlap": len(
                {
                    opening_position_orbit_digest(opening)
                    for opening in suite.openings
                }
                & excluded_position_orbits
            ),
        },
        "auditSha256",
    )
    if (
        audit["f60OpeningIdOverlap"] != 0
        or audit["f60ArenaInitialOrbitOverlap"] != 0
    ):
        raise Phase4F61Error("F61 selected openings overlap F60 evidence")
    return suite, audit


def _runtime_binding(
    value_scale: float, source_sha: str, native_sha: str
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f61-value-scaled-native-mcts-runtime",
            "modelFileSha256": EXPECTED_F21_MODEL_SHA256,
            "modelStateSha256": EXPECTED_F21_STATE_SHA256,
            "modelConfigFileSha256": EXPECTED_F21_CONFIG_SHA256,
            "sourceSha256": source_sha,
            "nativeBackendSha256": native_sha,
            "simulations": SIMULATIONS,
            "valueScale": value_scale,
            "policyTransform": "identity",
            "rootNoise": False,
            "temperature": 0.0,
            "mctsSeed": MCTS_SEED,
        },
        "runtimeSha256",
    )


def _contract_payload(
    *,
    baseline_hours: float,
    source: Mapping[str, object],
    resources: Mapping[str, object],
    suite: OpeningSuite,
    suite_file_sha: str,
    audit_sha: str,
    native_sha: str,
) -> dict[str, object]:
    candidate_runtime = _runtime_binding(
        CANDIDATE_VALUE_SCALE, str(source["sha256"]), native_sha
    )
    opponent_runtime = _runtime_binding(
        OPPONENT_VALUE_SCALE, str(source["sha256"]), native_sha
    )
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
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f61-f21-value-scale-causal-screen-contract",
            "status": "frozen",
            "hypothesis": "reducing-F21-value-influence-improves-paired-arena-strength",
            "source": dict(source),
            "nativeBackendSha256": native_sha,
            "model": {
                "path": F21_MODEL.as_posix(),
                "fileSha256": EXPECTED_F21_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "configPath": F21_CONFIG.as_posix(),
                "configFileSha256": EXPECTED_F21_CONFIG_SHA256,
            },
            "candidate": {
                "selectorId": "phase4f21-value-scale-0.5",
                "runtime": candidate_runtime,
            },
            "opponent": {
                "selectorId": "phase4f21-value-scale-1.0",
                "runtime": opponent_runtime,
            },
            "openingSuite": {
                "path": SUITE_PATH.as_posix(),
                "fileSha256": suite_file_sha,
                "suiteHash": suite.content_hash,
                "selectionAuditPath": SELECTION_AUDIT.as_posix(),
                "selectionAuditSha256": audit_sha,
                "f60OpeningIdOverlap": 0,
                "f60ArenaInitialOrbitOverlap": 0,
            },
            "workload": workload,
            "budget": {
                "baselineGlobalHours": baseline_hours,
                "screenLimitHours": SCREEN_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "hypothesisRejectedAtOrBelowWinRate": 0.50,
                "confirmationCandidateAboveWinRate": 0.50,
                "drawsRequired": 0,
                "confirmationOpeningPairs": 200,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "authorizations": {
                "inferenceOnlyScreenAuthorized": True,
                "optimizerStepsAuthorized": 0,
                "trainingAuthorized": False,
                "promotionAuthorized": False,
                "screenTrajectoriesTrainingUseAllowed": False,
            },
            "resourcesAtFreeze": dict(resources),
        },
        "contractSha256",
    )


def prepare_phase4f61(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / PREPARATION_REPORT).exists():
        return {**verify_preparation(workspace), "reused": True}
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256:
        raise Phase4F61Error("F21 config bytes changed")
    _load_f21_state(workspace)
    gate60 = _mapping(workspace / F60_GATE, "F60 gate")
    if (
        _verify_self_hash(gate60, "gateSha256", "F60 gate")
        != EXPECTED_F60_GATE_SHA256
        or gate60.get("resultSha256") != EXPECTED_F60_RESULT_SHA256
    ):
        raise Phase4F61Error("F60 completed evidence changed")
    if (
        sha256_file(workspace / FRESH_POOL) != EXPECTED_FRESH_POOL_SHA256
        or sha256_file(workspace / F60_SUITE) != EXPECTED_F60_SUITE_SHA256
    ):
        raise Phase4F61Error("F59/F60 opening evidence changed")
    pool = load_opening_suite(workspace / FRESH_POOL, expected_suite=SUITE_NAME)
    used = load_opening_suite(workspace / F60_SUITE, expected_suite=SUITE_NAME)
    if pool.content_hash != EXPECTED_FRESH_POOL_SUITE_HASH:
        raise Phase4F61Error("F59 fresh opening pool hash changed")
    excluded_orbits = arena_position_orbits(
        workspace,
        {
            "path": F60_RESULT.as_posix(),
            "fileSha256": sha256_file(workspace / F60_RESULT),
            "resultSha256": EXPECTED_F60_RESULT_SHA256,
        },
    )
    suite, audit = select_fresh_openings(pool, used, excluded_orbits)
    _write_immutable(workspace / SELECTION_AUDIT, audit, "F61 selection audit")
    _write_immutable(
        workspace / SUITE_PATH,
        opening_suite_to_payload(suite),
        "F61 opening suite",
    )
    source = fingerprint_sources(workspace, SOURCE_PATHS).to_payload()
    resources = _resources(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    contract = _contract_payload(
        baseline_hours=ledger.total_hours,
        source=source,
        resources=resources,
        suite=suite,
        suite_file_sha=sha256_file(workspace / SUITE_PATH),
        audit_sha=str(audit["auditSha256"]),
        native_sha=loaded_native_backend_hash(),
    )
    _write_immutable(workspace / CONTRACT, contract, "F61 contract")
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f61-value-scale-screen-preparation",
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
            "selectionAuditPath": SELECTION_AUDIT.as_posix(),
            "selectionAuditSha256": audit["auditSha256"],
            "baselineGlobalGpuHours": ledger.total_hours,
            "inferenceOnlyScreenAuthorized": True,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "nextStage": {
                "name": "run frozen F21 value-scale 0.5 versus 1.0 screen",
                "requiresSeparateUserConfirmation": False,
            },
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, report, "F61 preparation")
    return {**report, "reused": False}


def _contract(workspace: Path) -> dict[str, object]:
    contract = _mapping(workspace / CONTRACT, "F61 contract")
    _verify_self_hash(contract, "contractSha256", "F61 contract")
    if (
        contract.get("kind") != "phase4f61-f21-value-scale-causal-screen-contract"
        or contract.get("status") != "frozen"
        or contract.get("authorizations", {}).get("inferenceOnlyScreenAuthorized")
        is not True
        or contract.get("authorizations", {}).get("optimizerStepsAuthorized") != 0
        or contract.get("authorizations", {}).get("trainingAuthorized") is not False
        or contract.get("authorizations", {}).get("promotionAuthorized") is not False
    ):
        raise Phase4F61Error("F61 contract authorization is invalid")
    return contract


def verify_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    report = _mapping(workspace / PREPARATION_REPORT, "F61 preparation")
    report_sha = _verify_self_hash(report, "reportSha256", "F61 preparation")
    source = fingerprint_sources(workspace, SOURCE_PATHS).to_payload()
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    audit = _mapping(workspace / SELECTION_AUDIT, "F61 selection audit")
    audit_sha = _verify_self_hash(audit, "auditSha256", "F61 selection audit")
    if (
        source != contract.get("source")
        or sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_SHA256
        or sha256_file(workspace / F21_MODEL) != EXPECTED_F21_MODEL_SHA256
        or loaded_native_backend_hash() != contract.get("nativeBackendSha256")
        or sha256_file(workspace / SUITE_PATH)
        != contract.get("openingSuite", {}).get("fileSha256")
        or suite.content_hash != contract.get("openingSuite", {}).get("suiteHash")
        or audit_sha != contract.get("openingSuite", {}).get("selectionAuditSha256")
        or report.get("contractSha256") != contract.get("contractSha256")
        or report.get("selectionAuditSha256") != audit_sha
    ):
        raise Phase4F61Error("F61 frozen preparation changed")
    _load_f21_state(workspace)
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f61-value-scale-screen-preparation-verification",
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
        "inferenceOnlyScreenAuthorized": True,
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
    }


def _identities(
    contract: Mapping[str, object]
) -> tuple[SelectorIdentity, SelectorIdentity]:
    source_sha = str(contract["source"]["sha256"])
    model_sha = str(contract["model"]["fileSha256"])
    candidate = contract["candidate"]
    opponent = contract["opponent"]
    return (
        SelectorIdentity(
            selector_id=str(candidate["selectorId"]),
            artifact_hash=model_sha,
            config_hash=str(candidate["runtime"]["runtimeSha256"]),
            source_hash=source_sha,
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id=str(opponent["selectorId"]),
            artifact_hash=model_sha,
            config_hash=str(opponent["runtime"]["runtimeSha256"]),
            source_hash=source_sha,
            engine="native-mcts",
        ),
    )


def arena_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f61ContractSha256": contract["contractSha256"],
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


def outcome_band(win_rate: float) -> str:
    return (
        "value-scale-screen-positive"
        if win_rate > 0.50
        else "value-scale-not-supported"
    )


def _selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    contract: Mapping[str, object],
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
        raise Phase4F61Error("F21 state does not fit the frozen network") from error
    base = TorchEvaluator(model.to(device).eval(), device=device, use_amp=True)
    evaluator = ValueScaledEvaluator(base, value_scale)
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
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F61Error(f"completed F61 checks failed: {checks}")
    statistics = result["statistics"]
    band = outcome_band(float(statistics["winRate"]))
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f61-f21-value-scale-causal-screen-gate",
            "status": "completed",
            "integrityPassed": True,
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
            "confirmationArenaCandidate": band == "value-scale-screen-positive",
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStage": {
                "name": (
                    "200-pair fresh value-scale confirmation preparation"
                    if band == "value-scale-screen-positive"
                    else "stop value-calibration hypothesis and redesign"
                ),
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )


def verify_phase4f61(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    verify_preparation(workspace)
    contract = _contract(workspace)
    gate = _mapping(workspace / GATE, "F61 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F61 gate")
    if gate != gate_from_result(workspace, contract):
        raise Phase4F61Error("stored F61 gate differs from arena result")
    budget = contract["budget"]
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(budget["baselineGlobalHours"])
    ceiling = min(
        baseline + float(budget["screenLimitHours"]),
        float(budget["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours > ceiling:
        raise Phase4F61Error("F61 completed outside its frozen GPU budget")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f61-f21-value-scale-screen-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "candidateValueScale": gate["candidateValueScale"],
        "opponentValueScale": gate["opponentValueScale"],
        "statistics": gate["statistics"],
        "outcomeBand": gate["outcomeBand"],
        "confirmationArenaCandidate": gate["confirmationArenaCandidate"],
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
        "globalGpuHours": ledger.total_hours,
        "nextStage": gate["nextStage"],
    }


def run_phase4f61(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f61(workspace), "reused": True}
    verify_preparation(workspace)
    contract = _contract(workspace)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F61Error("F61 screen requires CUDA")
    budget = contract["budget"]
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(budget["baselineGlobalHours"])
    ceiling = min(
        baseline + float(budget["screenLimitHours"]),
        float(budget["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours >= ceiling:
        raise Phase4F61Error("F61 GPU budget is unavailable or exhausted")
    resources = _resources(workspace)
    state = _load_f21_state(workspace)
    suite = load_opening_suite(workspace / SUITE_PATH, expected_suite=SUITE_NAME)
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f61-value-scale-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F61Error("F61 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                canonical_json_bytes(
                    {
                        "kind": "phase4f61-value-scale-screen-progress",
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
                        "kind": "phase4f61-value-scale-screen-progress",
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
                contract,
                device,
                candidate_identity,
                CANDIDATE_VALUE_SCALE,
            )
            opponent_selector = _selector(
                workspace,
                state,
                contract,
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
                raise Phase4F61Error("F61 completed beyond its frozen GPU budget")
            _write_immutable(workspace / GATE, gate, "F61 gate")
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
        "command", choices=("prepare", "run", "verify"), nargs="?", default="run"
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f61(args.workspace_root)
    elif args.command == "run":
        payload = run_phase4f61(args.workspace_root, args.device)
    else:
        payload = verify_phase4f61(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
