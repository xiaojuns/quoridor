"""Freeze and run F21 Stage A versus F27 Stage B on F29 fresh openings."""

from __future__ import annotations

import argparse
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
from .openings import load_opening_suite
from .paired_arena import BoundMoveSelector, SelectorIdentity, loaded_native_backend_hash
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f26_lineage_screen import (
    Phase4F26LineageScreenError,
    _stage_a_candidate,
)
from .phase4f28_conservative_screen import (
    Phase4F28ConservativeScreenError,
    _f27_candidate,
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
from .storage import directory_size
from .selectors import MCTSSelector


SCHEMA_VERSION = 1
F29_GATE = Path(
    "training/runs/phase4/phase4f/fresh-development-screen-v1/screen-gate.json"
)
EXPECTED_F29_GATE_SHA256 = (
    "5c8ad75fee3a5774c17672540c874fee441362936a5efd53511c2cf81987c44d"
)
F29_ISOLATION_AUDIT = Path(
    "training/runs/phase4/phase4f/fresh-development-screen-v1/isolation-audit.json"
)
EXPECTED_F29_AUDIT_SHA256 = (
    "1e2375c5f89ed5e87b7902c64c35c6694d212183c6455a97ffed461fec47d24b"
)
SCREEN_SUITE = Path(
    "training/runs/phase4/phase4f/fresh-development-screen-v1/"
    "recovery-fresh-dev-screen-32.json"
)
ROOT = Path("training/runs/phase4/phase4f/fresh-lineage-screen-v1")
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"
PREPARATION_REPORT = ROOT / "preparation-report.json"
CONTRACT = Path("training/configs/phase4f30-fresh-lineage-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
MCTS_SEED = 20260816
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
SCREEN_GPU_HOURS = 2.0
ARENA_WRITE_RESERVE_BYTES = 32 * 1024**2


class Phase4F30FreshLineageScreenError(RuntimeError):
    """Raised when F30 evidence, preparation, or execution is invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F30FreshLineageScreenError(
            "workspace does not contain training/quoridor"
        )
    return workspace


def _mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Phase4F30FreshLineageScreenError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F30FreshLineageScreenError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise Phase4F30FreshLineageScreenError(f"{label} is not canonical JSON")
    return dict(payload)


def _verify_self_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    core = dict(payload)
    stored = core.pop(field, None)
    if not isinstance(stored, str) or stored != sha256_bytes(canonical_json_bytes(core)):
        raise Phase4F30FreshLineageScreenError(f"{label} self hash mismatch")
    return stored


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256_bytes(canonical_json_bytes(core))}


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F30FreshLineageScreenError(
                f"existing immutable {label} differs"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(path, encoded)


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F30FreshLineageScreenError("artifact path escapes workspace") from error


def _resource_snapshot(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARENA_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F30FreshLineageScreenError(
            "12 GiB cap cannot reserve the F30 arena output"
        )
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F30FreshLineageScreenError("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "arenaWriteReserveBytes": ARENA_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _stage_a(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    try:
        return _stage_a_candidate(workspace)
    except Phase4F26LineageScreenError as error:
        raise Phase4F30FreshLineageScreenError("F21 Stage A lineage failed") from error


def _stage_b(workspace: Path) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    try:
        return _f27_candidate(workspace)
    except Phase4F28ConservativeScreenError as error:
        raise Phase4F30FreshLineageScreenError("F27 Stage B lineage failed") from error


def _f29_evidence(
    workspace: Path, stage_b: Mapping[str, object]
) -> dict[str, object]:
    gate = _mapping(workspace / F29_GATE, "F29 gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F29 gate")
    result_path = workspace / str(gate.get("resultPath"))
    result = verify_completed_arena_result(result_path)
    audit = _mapping(workspace / F29_ISOLATION_AUDIT, "F29 isolation audit")
    audit_sha256 = _verify_self_hash(audit, "auditSha256", "F29 isolation audit")
    suite = load_opening_suite(
        workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    if (
        gate_sha256 != EXPECTED_F29_GATE_SHA256
        or gate.get("status") != "completed"
        or gate.get("integrityPassed") is not True
        or gate.get("outcomeBand") != "fresh-development-insufficient"
        or gate.get("confirmationArenaCandidate") is not False
        or gate.get("additionalStageBTrainingAuthorized") is not False
        or gate.get("stageBTrainingAuthorized") is not False
        or gate.get("nextStageRequiresConfirmation") is not True
        or gate.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("resultSha256") != result.get("resultSha256")
        or result.get("candidateIdentity", {}).get("artifactHash")
        != stage_b.get("candidateModelSha256")
        or audit_sha256 != EXPECTED_F29_AUDIT_SHA256
        or audit.get("freshOverlapCount") != 0
        or len(suite.openings) != OPENING_PAIRS
    ):
        raise Phase4F30FreshLineageScreenError(
            "F29 gate does not authorize fresh lineage diagnosis"
        )
    return {
        "gatePath": F29_GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "outcomeBand": gate["outcomeBand"],
        "stageBWins": gate["statistics"]["candidateWins"],
        "stageBLosses": gate["statistics"]["candidateLosses"],
        "isolationAuditPath": F29_ISOLATION_AUDIT.as_posix(),
        "isolationAuditSha256": audit_sha256,
        "freshOverlapCount": 0,
    }


def _suite_reference(workspace: Path) -> tuple[object, dict[str, object]]:
    path = workspace / SCREEN_SUITE
    suite = load_opening_suite(path, expected_suite="recovery-fresh-dev-v1")
    if len(suite.openings) != OPENING_PAIRS:
        raise Phase4F30FreshLineageScreenError("F29 fresh screen suite size changed")
    return suite, {
        "path": SCREEN_SUITE.as_posix(),
        "fileSha256": sha256_file(path),
        "suiteHash": suite.content_hash,
        "openingPairs": len(suite.openings),
        "reusedExactlyFromF29": True,
    }


def _identities(
    contract: Mapping[str, object]
) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    opponent = contract["opponent"]
    return (
        SelectorIdentity(
            selector_id="phase4f-stage-a-f21-best-fresh-lineage",
            artifact_hash=str(candidate["candidateModelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id="phase4f-stage-b-f27-conservative-fresh-lineage",
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
                "phase4f30ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["suiteHash"],
            }
        )
    )


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
        raise Phase4F30FreshLineageScreenError(
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


def prepare_phase4f30(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / CONTRACT).exists():
        contract = verify_phase4f30_preparation(workspace)
        return {
            "status": "prepared",
            "reused": True,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
        }
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(item.get("status") == "running" for item in ledger.runs.values()):
        raise Phase4F30FreshLineageScreenError(
            "a running GPU ledger entry blocks preparation"
        )
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F30FreshLineageScreenError("insufficient global GPU budget for F30")
    resources = _resource_snapshot(workspace)
    candidate, _ = _stage_a(workspace)
    opponent, _ = _stage_b(workspace)
    evidence = _f29_evidence(workspace, opponent)
    _, suite_reference = _suite_reference(workspace)
    workload = _self_hashed(
        {
            "suite": "recovery-fresh-dev-v1",
            "suiteHash": suite_reference["suiteHash"],
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
            "kind": "phase4f30-stage-a-versus-stage-b-fresh-lineage-screen",
            "status": "frozen",
            "formalPromotion": False,
            "trainingAuthorized": False,
            "finalConfirmationAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "source": source,
            "nativeBackendSha256": loaded_native_backend_hash(),
            "f29Evidence": evidence,
            "candidate": candidate,
            "opponent": opponent,
            "openingSuite": suite_reference,
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
                "stageBStrongerBelowStageAWinRate": 0.45,
                "inconclusiveBelowStageAWinRate": 0.55,
                "stageAStrongerAtOrAboveStageAWinRate": 0.55,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "resourcesAtFreeze": resources,
        },
        "contractSha256",
    )
    _write_immutable(workspace / CONTRACT, contract, "F30 lineage contract")
    preparation = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f30-fresh-lineage-screen-preparation",
            "status": "prepared",
            "cpuOnly": True,
            "cudaAccessed": False,
            "lineageScreenAuthorized": True,
            "trainingAuthorized": False,
            "finalConfirmationAuthorized": False,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "source": source,
            "candidate": candidate,
            "opponent": opponent,
            "openingSuite": suite_reference,
            "globalGpuHours": ledger.total_hours,
            "resources": resources,
        },
        "reportSha256",
    )
    _write_immutable(workspace / PREPARATION_REPORT, preparation, "F30 preparation")
    verify_phase4f30_preparation(workspace)
    return preparation


def verify_phase4f30_preparation(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / CONTRACT, "F30 lineage contract")
    _verify_self_hash(contract, "contractSha256", "F30 lineage contract")
    preparation = _mapping(workspace / PREPARATION_REPORT, "F30 preparation")
    _verify_self_hash(preparation, "reportSha256", "F30 preparation")
    candidate, _ = _stage_a(workspace)
    opponent, _ = _stage_b(workspace)
    evidence = _f29_evidence(workspace, opponent)
    suite, suite_reference = _suite_reference(workspace)
    workload = contract.get("workload")
    if not isinstance(workload, Mapping):
        raise Phase4F30FreshLineageScreenError("F30 workload is invalid")
    _verify_self_hash(workload, "workloadSha256", "F30 workload")
    if (
        contract.get("kind")
        != "phase4f30-stage-a-versus-stage-b-fresh-lineage-screen"
        or contract.get("status") != "frozen"
        or contract.get("formalPromotion") is not False
        or contract.get("trainingAuthorized") is not False
        or contract.get("finalConfirmationAuthorized") is not False
        or contract.get("screenTrajectoriesTrainingUseAllowed") is not False
        or contract.get("source") != fingerprint_sources(workspace).to_payload()
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or contract.get("f29Evidence") != evidence
        or contract.get("candidate") != candidate
        or contract.get("opponent") != opponent
        or contract.get("openingSuite") != suite_reference
        or suite.content_hash != workload.get("suiteHash")
        or preparation.get("contractSha256") != contract.get("contractSha256")
        or preparation.get("source") != contract.get("source")
        or preparation.get("lineageScreenAuthorized") is not True
        or preparation.get("trainingAuthorized") is not False
        or preparation.get("finalConfirmationAuthorized") is not False
    ):
        raise Phase4F30FreshLineageScreenError("F30 preparation provenance changed")
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F30FreshLineageScreenError("F30 GPU budget is unavailable")
    _resource_snapshot(workspace)
    return contract


def _gate_from_result(
    workspace: Path, contract: Mapping[str, object]
) -> dict[str, object]:
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
        raise Phase4F30FreshLineageScreenError(
            f"completed F30 checks failed: {checks}"
        )
    statistics = result["statistics"]
    stage_a_win_rate = float(statistics["winRate"])
    if stage_a_win_rate < 0.45:
        band = "stage-b-stronger-on-fresh-openings"
    elif stage_a_win_rate < 0.55:
        band = "fresh-lineage-inconclusive"
    else:
        band = "stage-a-stronger-on-fresh-openings"
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f30-fresh-lineage-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "stageARollbackCandidate": stage_a_win_rate >= 0.55,
            "stageBRetainedCandidate": stage_a_win_rate < 0.45,
            "trainingAuthorized": False,
            "finalConfirmationAuthorized": False,
            "formalPromotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f30(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = verify_phase4f30_preparation(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F30 lineage gate")
        _verify_self_hash(stored, "gateSha256", "F30 lineage gate")
        if stored != _gate_from_result(workspace, contract):
            raise Phase4F30FreshLineageScreenError(
                "stored F30 gate differs from arena result"
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
        raise Phase4F30FreshLineageScreenError("F30 lineage screen requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + SCREEN_GPU_HOURS:
        raise Phase4F30FreshLineageScreenError("F30 GPU budget is exhausted")
    _resource_snapshot(workspace)
    candidate, candidate_state = _stage_a(workspace)
    opponent, opponent_state = _stage_b(workspace)
    if candidate != contract["candidate"] or opponent != contract["opponent"]:
        raise Phase4F30FreshLineageScreenError(
            "lineage model changed before CUDA execution"
        )
    suite = load_opening_suite(
        workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f30-fresh-lineage-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F30FreshLineageScreenError(
                    "F30 GPU budget was exhausted"
                )
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f30-fresh-lineage-progress",
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
            _write_immutable(gate_path, gate, "F30 lineage gate")
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
        payload = prepare_phase4f30(args.workspace_root)
    elif args.command == "verify":
        contract = verify_phase4f30_preparation(args.workspace_root)
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
        payload = run_phase4f30(args.workspace_root, args.device)
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
