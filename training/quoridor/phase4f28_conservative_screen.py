"""Freeze and run the conservative Stage B candidate-versus-parent screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
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
)
from .paired_arena import BoundMoveSelector, SelectorIdentity, loaded_native_backend_hash
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f26_lineage_screen import (
    Phase4F26LineageScreenError,
    _validated_candidate as _validated_prior_candidate,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .selectors import MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
F27_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f27-stage-b-review.json"
)
EXPECTED_F27_REVIEW_SHA256 = (
    "b1d70538c6756e8df192159f41a9fdd88d88064b0ccaf5331b1e68cc3800d8e7"
)
F27_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-b-block-01-v1"
)
F27_CONFIG = Path("training/configs/phase4f27-stage-b-block-01-v1.json")
F24_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f24-stage-b-review.json"
)
EXPECTED_F24_REVIEW_SHA256 = (
    "40810a17527d8db4c698d574f4cfa4add3cda82f3fcf051859e115e95aa1281a"
)
F24_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-b-block-00-v1"
)
F24_CONFIG = Path("training/configs/phase4f24-stage-b-block-00-v1.json")

FULL_SUITE = Path("training/fixtures/openings/recovery-v1/recovery-dev-v1.json")
PRIOR_SCREEN_SUITES = (
    Path("training/runs/phase4/phase4f/recovery-evaluation-v1/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v2/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v3/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v4/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-a-screen-v1/recovery-dev-unseen-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-a-screen-v2/recovery-dev-unseen-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-b-screen-v1/recovery-dev-unseen-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-b-lineage-screen-v1/recovery-dev-unseen-lineage-32.json"),
)
ROOT = Path("training/runs/phase4/phase4f/stage-b-conservative-screen-v1")
SCREEN_SUITE = ROOT / "recovery-dev-unseen-conservative-32.json"
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
CONTRACT = Path("training/configs/phase4f28-stage-b-conservative-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260814
MCTS_SEED = 20260814
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
SCREEN_GPU_HOURS = 2.0
ARENA_WRITE_RESERVE_BYTES = 32 * 1024**2


class Phase4F28ConservativeScreenError(RuntimeError):
    """Raised when F28 screen evidence or execution is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F28ConservativeScreenError(
            "workspace does not contain training/quoridor"
        )
    return workspace


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F28ConservativeScreenError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F28ConservativeScreenError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F28ConservativeScreenError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F28ConservativeScreenError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F28ConservativeScreenError(
                f"existing immutable {label} differs"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(path, encoded)


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F28ConservativeScreenError(
            "artifact path escapes workspace"
        ) from error


def _resource_snapshot(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F28ConservativeScreenError(
            "12 GiB cap cannot reserve the F28 arena output"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F28ConservativeScreenError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _candidate(
    workspace: Path,
    *,
    stage: str,
    review: Path,
    review_sha256: str,
    run: Path,
    config: Path,
    candidate_sha256: str,
    checkpoint_sha256: str,
    state_sha256: str,
) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    try:
        return _validated_prior_candidate(
            workspace,
            stage=stage,
            review_relative=review,
            expected_review_sha256=review_sha256,
            review_field="latestCandidate",
            run_relative=run,
            config_relative=config,
            expected_candidate_sha256=candidate_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_state_sha256=state_sha256,
        )
    except Phase4F26LineageScreenError as error:
        raise Phase4F28ConservativeScreenError(
            f"{stage} candidate failed frozen lineage validation"
        ) from error


def _f27_candidate(
    workspace: Path,
) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    return _candidate(
        workspace,
        stage="stage-b-conservative",
        review=F27_REVIEW,
        review_sha256=EXPECTED_F27_REVIEW_SHA256,
        run=F27_RUN,
        config=F27_CONFIG,
        candidate_sha256="6c7756474c05e5ebe2a6a8acfbac6a2b35f693a297bd112898589141f337a996",
        checkpoint_sha256="61df04e21c499b558b20a57de2daa240671b72bae75d9fd6292714c042c6e0dd",
        state_sha256="5f5091c982b1be28faf508ccec6f4b106d7a248c3efa213d62d36701b432a2ba",
    )


def _f24_parent(
    workspace: Path,
) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    return _candidate(
        workspace,
        stage="stage-b-parent",
        review=F24_REVIEW,
        review_sha256=EXPECTED_F24_REVIEW_SHA256,
        run=F24_RUN,
        config=F24_CONFIG,
        candidate_sha256="c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f",
        checkpoint_sha256="48b6717bd802071256a3cd3509105bea7e38eb497db1436d92e3d5f4f676b9a1",
        state_sha256="6a8ccf68cdb0a1e234eb66ecd9a7550ab19dc6791a1f00fed1f462e3b394682e",
    )


def _f27_evidence(
    workspace: Path, candidate: Mapping[str, object]
) -> dict[str, object]:
    review = _mapping(workspace / F27_REVIEW, "F27 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "F27 review")
    latest = review.get("latestCandidate")
    generations = review.get("generations")
    if (
        review_sha256 != EXPECTED_F27_REVIEW_SHA256
        or review.get("status") != "completed"
        or review.get("conservativeStageBPassed") is not True
        or review.get("f28ScreenAuthorized") is not True
        or review.get("additionalStageBTrainingAuthorized") is not False
        or not isinstance(latest, Mapping)
        or latest.get("candidateModelSha256")
        != candidate.get("candidateModelSha256")
        or not isinstance(generations, list)
        or len(generations) != 2
        or [item.get("generation") for item in generations] != [0, 1]
    ):
        raise Phase4F28ConservativeScreenError(
            "F27 review does not authorize the conservative screen"
        )
    return {
        "reviewPath": F27_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "runManifestPath": review["runManifestPath"],
        "runManifestSha256": review["runManifestSha256"],
        "candidateModelSha256": latest["candidateModelSha256"],
        "conservativeStageBPassed": True,
        "f28ScreenAuthorized": True,
    }


def _selected_suite(workspace: Path) -> tuple[OpeningSuite, dict[str, object]]:
    full_path = workspace / FULL_SUITE
    full = load_opening_suite(full_path, expected_suite="recovery-dev-v1")
    excluded_ids: set[str] = set()
    prior: list[dict[str, object]] = []
    for relative in PRIOR_SCREEN_SUITES:
        path = workspace / relative
        suite = load_opening_suite(path, expected_suite="recovery-dev-v1")
        ids = {opening.id for opening in suite.openings}
        excluded_ids.update(ids)
        prior.append(
            {
                "path": relative.as_posix(),
                "fileSha256": sha256_file(path),
                "suiteHash": suite.content_hash,
                "openingPairs": len(suite.openings),
            }
        )
    eligible = tuple(opening for opening in full.openings if opening.id not in excluded_ids)
    if len(eligible) < OPENING_PAIRS:
        raise Phase4F28ConservativeScreenError(
            "not enough unseen conservative screen openings"
        )

    def score(opening: object) -> str:
        return hashlib.sha256(
            f"{SELECTION_SEED}\0{getattr(opening, 'content_hash')}".encode("ascii")
        ).hexdigest()

    selected = tuple(
        sorted(eligible, key=lambda opening: (score(opening), opening.id))[
            :OPENING_PAIRS
        ]
    )
    screen = build_opening_suite(selected, suite="recovery-dev-v1")
    return screen, {
        "fullPath": FULL_SUITE.as_posix(),
        "fullFileSha256": sha256_file(full_path),
        "fullSuiteHash": full.content_hash,
        "fullOpeningPairs": len(full.openings),
        "priorScreens": prior,
        "excludedOpeningIds": len(excluded_ids),
        "eligibleOpeningPairs": len(eligible),
        "selectionAlgorithm": "sha256(seed-ascii+nul+opening-content-hash)-lowest-v1",
        "selectionSeed": SELECTION_SEED,
        "selectedOpeningIds": [opening.id for opening in screen.openings],
        "selectedSuiteHash": screen.content_hash,
        "selectedOpeningPairs": len(screen.openings),
    }


def _identities(contract: Mapping[str, object]) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    opponent = contract["opponent"]
    return (
        SelectorIdentity(
            selector_id="phase4f-stage-b-conservative-g00000001-s000000000192",
            artifact_hash=str(candidate["candidateModelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id="phase4f-stage-b-parent-g00000001-s000000000192",
            artifact_hash=str(opponent["candidateModelSha256"]),
            config_hash=str(opponent["configSha256"]),
            source_hash=str(opponent["trainingSourceSha256"]),
            engine="native-mcts",
        ),
    )


def _arena_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f28ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["selectedSuiteHash"],
            }
        )
    )


def prepare_phase4f28(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / CONTRACT).exists():
        contract = verify_phase4f28_preparation(workspace)
        return {
            "status": "prepared",
            "reused": True,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
        }
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F28ConservativeScreenError(
            "a running GPU ledger entry blocks preparation"
        )
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F28ConservativeScreenError("insufficient global GPU budget for F28")
    resources = _resource_snapshot(workspace)
    candidate, _ = _f27_candidate(workspace)
    opponent, _ = _f24_parent(workspace)
    evidence = _f27_evidence(workspace, candidate)
    screen, suite_reference = _selected_suite(workspace)
    _write_immutable(
        workspace / SCREEN_SUITE,
        opening_suite_to_payload(screen),
        "F28 conservative opening suite",
    )
    workload = _self_hashed(
        {
            "suite": "recovery-dev-v1",
            "suiteHash": screen.content_hash,
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
    source = fingerprint_sources(workspace).to_payload()
    contract = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f28-conservative-stage-b-versus-parent-screen",
            "status": "frozen",
            "formalPromotion": False,
            "additionalStageBTrainingAuthorized": False,
            "parentRollbackAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "f27Evidence": evidence,
            "candidate": candidate,
            "opponent": opponent,
            "openingSuite": {
                **suite_reference,
                "selectedPath": SCREEN_SUITE.as_posix(),
                "selectedFileSha256": sha256_file(workspace / SCREEN_SUITE),
                "contentAndPriorScreenDisjoint": True,
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
                "regressionBelowWinRate": 0.45,
                "inconclusiveBelowWinRate": 0.50,
                "retainedAtOrAboveWinRate": 0.50,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F28 conservative contract")
    preparation = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f28-conservative-stage-b-screen-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "conservativeScreenAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "parentRollbackAuthorized": False,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "source": source,
            "candidate": candidate,
            "opponent": opponent,
            "openingSuite": contract["openingSuite"],
            "globalGpuHours": ledger.total_hours,
            "resources": resources,
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, preparation, "F28 preparation")
    verify_phase4f28_preparation(workspace)
    return preparation


def verify_phase4f28_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F28 conservative contract")
    _verify_self_hash(contract, "contractSha256", "F28 conservative contract")
    preparation = _mapping(workspace / PREPARATION_REPORT, "F28 preparation")
    _verify_self_hash(preparation, "reportSha256", "F28 preparation")
    candidate, _ = _f27_candidate(workspace)
    opponent, _ = _f24_parent(workspace)
    evidence = _f27_evidence(workspace, candidate)
    screen, reference = _selected_suite(workspace)
    workload = contract.get("workload")
    if not isinstance(workload, Mapping):
        raise Phase4F28ConservativeScreenError("F28 workload is invalid")
    _verify_self_hash(workload, "workloadSha256", "F28 workload")
    selected_path = workspace / SCREEN_SUITE
    if (
        contract.get("kind")
        != "phase4f28-conservative-stage-b-versus-parent-screen"
        or contract.get("status") != "frozen"
        or contract.get("formalPromotion") is not False
        or contract.get("additionalStageBTrainingAuthorized") is not False
        or contract.get("parentRollbackAuthorized") is not False
        or contract.get("screenTrajectoriesTrainingUseAllowed") is not False
        or contract.get("source") != fingerprint_sources(workspace).to_payload()
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("f27Evidence") != evidence
        or contract.get("candidate") != candidate
        or contract.get("opponent") != opponent
        or contract.get("openingSuite")
        != {
            **reference,
            "selectedPath": SCREEN_SUITE.as_posix(),
            "selectedFileSha256": sha256_file(selected_path),
            "contentAndPriorScreenDisjoint": True,
        }
        or load_opening_suite(selected_path, expected_suite="recovery-dev-v1")
        != screen
        or preparation.get("contractSha256") != contract.get("contractSha256")
        or preparation.get("source") != contract.get("source")
        or preparation.get("conservativeScreenAuthorized") is not True
        or preparation.get("additionalStageBTrainingAuthorized") is not False
        or preparation.get("parentRollbackAuthorized") is not False
    ):
        raise Phase4F28ConservativeScreenError("F28 preparation provenance changed")
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F28ConservativeScreenError("F28 GPU budget is unavailable")
    _resource_snapshot(workspace)
    return contract


def _model_selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    config_path: str,
    device: torch.device,
    identity: SelectorIdentity,
) -> BoundMoveSelector:
    config = load_experiment_config(workspace / config_path)
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
        raise Phase4F28ConservativeScreenError(
            "model state does not match frozen network"
        ) from error
    model = model.to(device).eval()
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
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


def _gate_from_result(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    candidate_identity, opponent_identity = _identities(contract)
    workload = contract["workload"]
    checks = {
        "contract": result.get("contractSha256") == _arena_contract_sha256(contract),
        "suite": result.get("suite") == "recovery-dev-v1",
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
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F28ConservativeScreenError(
            f"completed F28 checks failed: {checks}"
        )
    statistics = result["statistics"]
    win_rate = float(statistics["winRate"])
    if win_rate < 0.45:
        band = "conservative-stage-b-regression"
    elif win_rate < 0.50:
        band = "conservative-stage-b-inconclusive"
    else:
        band = "conservative-stage-b-retained"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f28-conservative-stage-b-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "candidateRetainedCandidate": win_rate >= 0.50,
            "parentRollbackCandidate": win_rate < 0.45,
            "additionalStageBTrainingAuthorized": False,
            "parentRollbackAuthorized": False,
            "formalPromotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f28(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = verify_phase4f28_preparation(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F28 conservative gate")
        _verify_self_hash(stored, "gateSha256", "F28 conservative gate")
        if stored != _gate_from_result(workspace, contract):
            raise Phase4F28ConservativeScreenError(
                "stored F28 gate differs from arena result"
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
        raise Phase4F28ConservativeScreenError("F28 screen requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + SCREEN_GPU_HOURS:
        raise Phase4F28ConservativeScreenError("F28 GPU budget is exhausted")
    _resource_snapshot(workspace)
    candidate, candidate_state = _f27_candidate(workspace)
    opponent, opponent_state = _f24_parent(workspace)
    if candidate != contract["candidate"] or opponent != contract["opponent"]:
        raise Phase4F28ConservativeScreenError(
            "conservative screen model changed before CUDA execution"
        )
    suite = load_opening_suite(workspace / SCREEN_SUITE, expected_suite="recovery-dev-v1")
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f28-conservative-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F28ConservativeScreenError("F28 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f28-conservative-screen-progress",
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
            candidate_selector = _model_selector(
                workspace,
                candidate_state,
                str(candidate["configPath"]),
                device,
                candidate_identity,
            )
            opponent_selector = _model_selector(
                workspace,
                opponent_state,
                str(opponent["configPath"]),
                device,
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
            _write_immutable(gate_path, gate, "F28 conservative gate")
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
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f28(args.workspace_root)
    elif args.command == "verify":
        contract = verify_phase4f28_preparation(args.workspace_root)
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
        payload = run_phase4f28(args.workspace_root, args.device)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
