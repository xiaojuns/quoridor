"""Freeze and run the Stage B versus Stage A lineage screen."""

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
from .phase4_checkpoint import (
    PHASE4_CANDIDATE_MODEL_FORMAT,
    PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
    canonical_model_state_sha256,
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
    validate_sha256,
)
from .selectors import MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
F25_GATE = Path("training/runs/phase4/phase4f/stage-b-screen-v1/screen-gate.json")
EXPECTED_F25_GATE_SHA256 = (
    "7519f60e917c3435596e9116b2151a9e23f7bf9870a4a84569920e2613651cac"
)
STAGE_B_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f24-stage-b-review.json"
)
EXPECTED_STAGE_B_REVIEW_SHA256 = (
    "40810a17527d8db4c698d574f4cfa4add3cda82f3fcf051859e115e95aa1281a"
)
STAGE_B_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-b-block-00-v1"
)
STAGE_B_CONFIG = Path("training/configs/phase4f24-stage-b-block-00-v1.json")
STAGE_A_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f21-stage-a-review.json"
)
EXPECTED_STAGE_A_REVIEW_SHA256 = (
    "1a97d8dd760ffbadb7581d7d055b1bb47731ad7d76f9c7491fe5dfe9488a97a6"
)
STAGE_A_RUN = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1"
)
STAGE_A_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")

FULL_SUITE = Path("training/fixtures/openings/recovery-v1/recovery-dev-v1.json")
PRIOR_SCREEN_SUITES = (
    Path("training/runs/phase4/phase4f/recovery-evaluation-v1/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v2/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v3/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/recovery-evaluation-v4/recovery-dev-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-a-screen-v1/recovery-dev-unseen-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-a-screen-v2/recovery-dev-unseen-screen-32.json"),
    Path("training/runs/phase4/phase4f/stage-b-screen-v1/recovery-dev-unseen-screen-32.json"),
)
ROOT = Path("training/runs/phase4/phase4f/stage-b-lineage-screen-v1")
SCREEN_SUITE = ROOT / "recovery-dev-unseen-lineage-32.json"
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
CONTRACT = Path("training/configs/phase4f26-stage-b-lineage-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260812
MCTS_SEED = 20260812
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
SCREEN_GPU_HOURS = 2.0
ARENA_WRITE_RESERVE_BYTES = 32 * 1024**2


class Phase4F26LineageScreenError(RuntimeError):
    """Raised when F26 lineage evidence or execution is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F26LineageScreenError("workspace does not contain training/quoridor")
    return workspace


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F26LineageScreenError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F26LineageScreenError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F26LineageScreenError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F26LineageScreenError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F26LineageScreenError(f"existing immutable {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(path, encoded)


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F26LineageScreenError("artifact path escapes workspace") from error


def _resource_snapshot(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F26LineageScreenError("12 GiB cap cannot reserve the F26 arena output")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F26LineageScreenError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _validated_candidate(
    workspace: Path,
    *,
    stage: str,
    review_relative: Path,
    expected_review_sha256: str,
    review_field: str,
    run_relative: Path,
    config_relative: Path,
    expected_candidate_sha256: str,
    expected_checkpoint_sha256: str,
    expected_state_sha256: str,
) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    review = _mapping(workspace / review_relative, f"{stage} review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", f"{stage} review")
    candidate = review.get(review_field)
    if (
        review_sha256 != expected_review_sha256
        or review.get("status") != "completed"
        or not isinstance(candidate, Mapping)
        or candidate.get("generation") != 1
        or candidate.get("step") != 192
        or candidate.get("candidateModelSha256") != expected_candidate_sha256
        or candidate.get("checkpointSha256") != expected_checkpoint_sha256
        or candidate.get("modelStateSha256") != expected_state_sha256
    ):
        raise Phase4F26LineageScreenError(f"{stage} review is not lineage eligible")
    marker = _mapping(workspace / run_relative / "best-candidate.json", f"{stage} best index")
    if marker != candidate:
        raise Phase4F26LineageScreenError(f"{stage} best index differs from its review")
    candidate_name = candidate.get("candidateModelPath")
    checkpoint_name = candidate.get("checkpointPath")
    if (
        not isinstance(candidate_name, str)
        or Path(candidate_name).name != candidate_name
        or not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise Phase4F26LineageScreenError(f"{stage} candidate filenames are invalid")
    candidate_path = workspace / run_relative / "checkpoints" / candidate_name
    checkpoint_path = workspace / run_relative / "checkpoints" / checkpoint_name
    if (
        candidate_path.is_symlink()
        or not candidate_path.is_file()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or sha256_file(candidate_path) != expected_candidate_sha256
        or sha256_file(checkpoint_path) != expected_checkpoint_sha256
    ):
        raise Phase4F26LineageScreenError(f"{stage} candidate files failed SHA-256 validation")
    try:
        payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F26LineageScreenError(f"{stage} candidate payload is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION
        or payload.get("format") != PHASE4_CANDIDATE_MODEL_FORMAT
        or payload.get("generation") != 1
        or payload.get("step") != 192
        or payload.get("checkpointSha256") != expected_checkpoint_sha256
        or payload.get("modelStateSha256") != expected_state_sha256
        or not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != expected_state_sha256
        or not isinstance(identity, Mapping)
        or identity.get("runManifestSha256") != review.get("runManifestSha256")
    ):
        raise Phase4F26LineageScreenError(f"{stage} candidate payload contract failed")
    result = {
        "stage": stage,
        "generation": 1,
        "step": 192,
        "candidateModelPath": _relative(candidate_path, workspace),
        "candidateModelBytes": candidate_path.stat().st_size,
        "candidateModelSha256": expected_candidate_sha256,
        "checkpointPath": _relative(checkpoint_path, workspace),
        "checkpointSha256": expected_checkpoint_sha256,
        "modelStateSha256": expected_state_sha256,
        "runManifestSha256": review["runManifestSha256"],
        "configPath": config_relative.as_posix(),
        "configSha256": identity.get("configSha256"),
        "trainingSourceSha256": identity.get("sourceSha256"),
        "reviewPath": review_relative.as_posix(),
        "reviewSha256": review_sha256,
    }
    for field in (
        "candidateModelSha256",
        "checkpointSha256",
        "modelStateSha256",
        "runManifestSha256",
        "configSha256",
        "trainingSourceSha256",
    ):
        validate_sha256(result[field], field)
    return result, state


def _stage_b_candidate(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    return _validated_candidate(
        workspace,
        stage="stage-b",
        review_relative=STAGE_B_REVIEW,
        expected_review_sha256=EXPECTED_STAGE_B_REVIEW_SHA256,
        review_field="latestCandidate",
        run_relative=STAGE_B_RUN,
        config_relative=STAGE_B_CONFIG,
        expected_candidate_sha256="c5277f1a3b64747d97ab6c29e57589d2772f641c174b1a4b1b9fa18b6867aa5f",
        expected_checkpoint_sha256="48b6717bd802071256a3cd3509105bea7e38eb497db1436d92e3d5f4f676b9a1",
        expected_state_sha256="6a8ccf68cdb0a1e234eb66ecd9a7550ab19dc6791a1f00fed1f462e3b394682e",
    )


def _stage_a_candidate(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    return _validated_candidate(
        workspace,
        stage="stage-a",
        review_relative=STAGE_A_REVIEW,
        expected_review_sha256=EXPECTED_STAGE_A_REVIEW_SHA256,
        review_field="bestCandidate",
        run_relative=STAGE_A_RUN,
        config_relative=STAGE_A_CONFIG,
        expected_candidate_sha256="1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01",
        expected_checkpoint_sha256="dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b",
        expected_state_sha256="5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527",
    )


def _f25_evidence(workspace: Path, stage_b: Mapping[str, object]) -> dict[str, object]:
    gate = _mapping(workspace / F25_GATE, "F25 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F25 gate")
    result_path = workspace / str(gate.get("resultPath"))
    result = verify_completed_arena_result(result_path)
    if (
        gate_sha256 != EXPECTED_F25_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "stage-b-below-confirmation"
        or gate.get("additionalStageBTrainingAuthorized") is not False
        or gate.get("stageBTrainingAuthorized") is not False
        or gate.get("nextStageRequiresConfirmation") is not True
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("resultSha256") != result.get("resultSha256")
        or result.get("candidateIdentity", {}).get("artifactHash")
        != stage_b.get("candidateModelSha256")
    ):
        raise Phase4F26LineageScreenError("F25 gate does not authorize lineage diagnosis")
    return {
        "gatePath": F25_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "outcomeBand": gate["outcomeBand"],
        "candidateWins": gate["statistics"]["candidateWins"],
        "candidateLosses": gate["statistics"]["candidateLosses"],
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
        raise Phase4F26LineageScreenError("not enough unseen lineage openings")

    def score(opening: object) -> str:
        return hashlib.sha256(
            f"{SELECTION_SEED}\0{getattr(opening, 'content_hash')}".encode("ascii")
        ).hexdigest()

    selected = tuple(
        sorted(eligible, key=lambda opening: (score(opening), opening.id))[:OPENING_PAIRS]
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
            selector_id="phase4f-stage-b-g00000001-s000000000192",
            artifact_hash=str(candidate["candidateModelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id="phase4f-stage-a-best-g00000001-s000000000192",
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
                "phase4f26ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["selectedSuiteHash"],
            }
        )
    )


def prepare_phase4f26(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / CONTRACT).exists():
        contract = verify_phase4f26_preparation(workspace)
        return {
            "status": "prepared",
            "reused": True,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
        }
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F26LineageScreenError("a running GPU ledger entry blocks preparation")
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F26LineageScreenError("insufficient global GPU budget for F26")
    resources = _resource_snapshot(workspace)
    candidate, _ = _stage_b_candidate(workspace)
    opponent, _ = _stage_a_candidate(workspace)
    f25 = _f25_evidence(workspace, candidate)
    screen, suite_reference = _selected_suite(workspace)
    _write_immutable(
        workspace / SCREEN_SUITE,
        opening_suite_to_payload(screen),
        "F26 lineage opening suite",
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
            "kind": "phase4f26-stage-b-versus-stage-a-lineage-screen",
            "status": "frozen",
            "formalPromotion": False,
            "additionalStageBTrainingAuthorized": False,
            "stageBRestartAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "f25Evidence": f25,
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
    _write_immutable(workspace / CONTRACT, contract, "F26 lineage contract")
    preparation = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f26-stage-b-lineage-screen-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "lineageScreenAuthorized": True,
            "additionalStageBTrainingAuthorized": False,
            "stageBRestartAuthorized": False,
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
    _write_immutable(workspace / PREPARATION_REPORT, preparation, "F26 preparation")
    verify_phase4f26_preparation(workspace)
    return preparation


def verify_phase4f26_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F26 lineage contract")
    _verify_self_hash(contract, "contractSha256", "F26 lineage contract")
    preparation = _mapping(workspace / PREPARATION_REPORT, "F26 preparation")
    _verify_self_hash(preparation, "reportSha256", "F26 preparation")
    candidate, _ = _stage_b_candidate(workspace)
    opponent, _ = _stage_a_candidate(workspace)
    f25 = _f25_evidence(workspace, candidate)
    screen, reference = _selected_suite(workspace)
    workload = contract.get("workload")
    if not isinstance(workload, Mapping):
        raise Phase4F26LineageScreenError("F26 workload is invalid")
    _verify_self_hash(workload, "workloadSha256", "F26 workload")
    selected_path = workspace / SCREEN_SUITE
    if (
        contract.get("kind") != "phase4f26-stage-b-versus-stage-a-lineage-screen"
        or contract.get("status") != "frozen"
        or contract.get("formalPromotion") is not False
        or contract.get("additionalStageBTrainingAuthorized") is not False
        or contract.get("stageBRestartAuthorized") is not False
        or contract.get("screenTrajectoriesTrainingUseAllowed") is not False
        or contract.get("source") != fingerprint_sources(workspace).to_payload()
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("f25Evidence") != f25
        or contract.get("candidate") != candidate
        or contract.get("opponent") != opponent
        or contract.get("openingSuite")
        != {
            **reference,
            "selectedPath": SCREEN_SUITE.as_posix(),
            "selectedFileSha256": sha256_file(selected_path),
            "contentAndPriorScreenDisjoint": True,
        }
        or load_opening_suite(selected_path, expected_suite="recovery-dev-v1") != screen
        or preparation.get("contractSha256") != contract.get("contractSha256")
        or preparation.get("source") != contract.get("source")
        or preparation.get("lineageScreenAuthorized") is not True
        or preparation.get("additionalStageBTrainingAuthorized") is not False
        or preparation.get("stageBRestartAuthorized") is not False
    ):
        raise Phase4F26LineageScreenError("F26 preparation provenance changed")
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F26LineageScreenError("F26 GPU budget is unavailable")
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
        raise Phase4F26LineageScreenError("model state does not match frozen network") from error
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


def _gate_from_result(workspace: Path, contract: Mapping[str, object]) -> dict[str, object]:
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
        "candidateIdentity": result.get("candidateIdentity") == candidate_identity.to_dict(),
        "opponentIdentity": result.get("opponentIdentity") == opponent_identity.to_dict(),
        "candidateRuntime": (
            result.get("candidateRuntime", {}).get("kind") == "native-mcts"
            and result.get("candidateRuntime", {}).get("trusted") is True
            and result.get("candidateRuntime", {}).get("configuredSimulations") == SIMULATIONS
            and result.get("candidateRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "opponentRuntime": (
            result.get("opponentRuntime", {}).get("kind") == "native-mcts"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("configuredSimulations") == SIMULATIONS
            and result.get("opponentRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F26LineageScreenError(f"completed F26 checks failed: {checks}")
    statistics = result["statistics"]
    win_rate = float(statistics["winRate"])
    if win_rate < 0.45:
        band = "stage-b-lineage-regression"
    elif win_rate < 0.50:
        band = "stage-b-lineage-inconclusive"
    else:
        band = "stage-b-lineage-retained"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f26-stage-b-lineage-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "additionalStageBTrainingCandidate": win_rate >= 0.50,
            "stageBRestartCandidate": win_rate < 0.45,
            "additionalStageBTrainingAuthorized": False,
            "stageBRestartAuthorized": False,
            "formalPromotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f26(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = verify_phase4f26_preparation(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F26 lineage gate")
        _verify_self_hash(stored, "gateSha256", "F26 lineage gate")
        if stored != _gate_from_result(workspace, contract):
            raise Phase4F26LineageScreenError("stored F26 gate differs from arena result")
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
        raise Phase4F26LineageScreenError("F26 lineage screen requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + SCREEN_GPU_HOURS:
        raise Phase4F26LineageScreenError("F26 GPU budget is exhausted")
    _resource_snapshot(workspace)
    candidate, candidate_state = _stage_b_candidate(workspace)
    opponent, opponent_state = _stage_a_candidate(workspace)
    if candidate != contract["candidate"] or opponent != contract["opponent"]:
        raise Phase4F26LineageScreenError("lineage model changed before CUDA execution")
    suite = load_opening_suite(workspace / SCREEN_SUITE, expected_suite="recovery-dev-v1")
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f26-lineage-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F26LineageScreenError("F26 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f26-lineage-screen-progress",
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
            _write_immutable(gate_path, gate, "F26 lineage gate")
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
            ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status="interrupted")
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
        payload = prepare_phase4f26(args.workspace_root)
    elif args.command == "verify":
        contract = verify_phase4f26_preparation(args.workspace_root)
        payload = {
            "status": "verified",
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "resultStatus": (
                "completed" if (Path(args.workspace_root) / GATE).is_file() else "not-started"
            ),
        }
    else:
        payload = run_phase4f26(args.workspace_root, args.device)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
