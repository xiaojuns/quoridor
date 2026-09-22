"""Run and verify the frozen F58-versus-F21 fresh-lineage screen."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import load_opening_suite
from .paired_arena import BoundMoveSelector, SelectorIdentity
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f59_kl_trust_region_review import (
    F60_CONTRACT,
    SCREEN_SUITE,
    _f58_and_f21_evidence,
    _relative,
    _workspace,
    verify_phase4f59,
)
from .run_manifest import RunLock, canonical_json_bytes, sha256_bytes, sha256_file
from .selectors import MCTSSelector
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f60-kl-lineage-screen-v1")
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"


class Phase4F60ScreenError(RuntimeError):
    """Raised when the frozen F60 lineage screen cannot run or verify."""


def _contract(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f59(workspace)
    contract = _mapping(workspace / F60_CONTRACT, "F60 contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F60 contract")
    authorizations = contract.get("authorizations")
    workload = contract.get("workload")
    if (
        verification.get("f60ContractSha256") != contract_sha
        or verification.get("f60LineageScreenAuthorized") is not True
        or contract.get("kind")
        != "phase4f60-f58-versus-f21-fresh-lineage-screen"
        or contract.get("status") != "frozen"
        or not isinstance(authorizations, Mapping)
        or authorizations.get("f60LineageScreenAuthorized") is not True
        or authorizations.get("classicHardScreenAuthorized") is not False
        or authorizations.get("formalTrainingAuthorized") is not False
        or authorizations.get("promotionAuthorized") is not False
        or authorizations.get("screenTrajectoriesTrainingUseAllowed") is not False
        or not isinstance(workload, Mapping)
        or _verify_self_hash(workload, "workloadSha256", "F60 workload")
        != workload.get("workloadSha256")
    ):
        raise Phase4F60ScreenError("F59 does not authorize this F60 screen")
    return contract


def _identities(contract: Mapping[str, object]) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    opponent = contract["opponent"]
    if not isinstance(candidate, Mapping) or not isinstance(opponent, Mapping):
        raise Phase4F60ScreenError("F60 selector identities are malformed")
    return (
        SelectorIdentity(
            selector_id=str(candidate["selectorId"]),
            artifact_hash=str(candidate["modelFileSha256"]),
            config_hash=str(candidate["trainingContractSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id=str(opponent["selectorId"]),
            artifact_hash=str(opponent["modelFileSha256"]),
            config_hash=str(opponent["configSha256"]),
            source_hash=str(opponent["sourceSha256"]),
            engine="native-mcts",
        ),
    )


def arena_contract_sha256(contract: Mapping[str, object]) -> str:
    workload = contract["workload"]
    opening = contract["openingSuite"]
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f60ContractSha256": contract["contractSha256"],
                "workloadSha256": workload["workloadSha256"],
                "suiteHash": opening["selectedSuiteHash"],
            }
        )
    )


def outcome_band(win_rate: float) -> str:
    if win_rate < 0.45:
        return "kl-policy-lineage-regression"
    if win_rate < 0.50:
        return "kl-policy-lineage-inconclusive"
    return "kl-policy-lineage-retained"


def _resources(workspace: Path, contract: Mapping[str, object]) -> dict[str, int]:
    budget = contract["budget"]
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    reserve = int(budget["arenaWriteReserveBytes"])
    artifact_limit = int(budget["artifactLimitBytes"])
    minimum_free = int(budget["minimumFreeDiskBytes"])
    if artifact_bytes + reserve >= artifact_limit:
        raise Phase4F60ScreenError("F60 arena reserve exceeds the frozen artifact cap")
    if free_bytes - reserve < minimum_free:
        raise Phase4F60ScreenError("F60 arena reserve violates the frozen disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": artifact_limit,
        "writeReserveBytes": reserve,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": minimum_free,
    }


def _selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    contract: Mapping[str, object],
    device: torch.device,
    identity: SelectorIdentity,
    simulations: int,
) -> BoundMoveSelector:
    config = load_experiment_config(workspace / str(contract["runtimeConfigPath"]))
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
        raise Phase4F60ScreenError(
            "model state does not fit the shared frozen network"
        ) from error
    model = model.to(device).eval()
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
    evaluator.enable_native_encoder(load_native_module())
    workload = contract["workload"]
    search = MCTSConfig(
        simulations=simulations,
        c_base=config.mcts.c_base,
        c_init=config.mcts.c_init,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=0.0,
        add_root_noise=False,
        seed=int(workload["mctsSeed"]),
    )
    return BoundMoveSelector(
        MCTSSelector(evaluator, search, search_factory=NativeMCTS, engine="native-mcts"),
        identity,
    )


def gate_from_result(workspace: Path, contract: Mapping[str, object]) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    workload = contract["workload"]
    candidate_identity, opponent_identity = _identities(contract)
    candidate_simulations = int(workload["candidateSimulations"])
    opponent_simulations = int(workload["opponentSimulations"])
    checks = {
        "contract": result.get("contractSha256") == arena_contract_sha256(contract),
        "suite": result.get("suite") == workload["suite"],
        "suiteHash": result.get("suiteHash") == workload["suiteHash"],
        "openingPairs": result.get("openingPairs") == workload["openingPairs"],
        "games": result.get("games") == workload["games"],
        "candidateIdentity": result.get("candidateIdentity")
        == candidate_identity.to_dict(),
        "opponentIdentity": result.get("opponentIdentity")
        == opponent_identity.to_dict(),
        "candidateRuntime": (
            result.get("candidateRuntime", {}).get("kind") == "native-mcts"
            and result.get("candidateRuntime", {}).get("trusted") is True
            and result.get("candidateRuntime", {}).get("configuredSimulations")
            == candidate_simulations
            and result.get("candidateRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "opponentRuntime": (
            result.get("opponentRuntime", {}).get("kind") == "native-mcts"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("configuredSimulations")
            == opponent_simulations
            and result.get("opponentRuntime", {}).get("backendHash")
            == contract["nativeBackendSha256"]
        ),
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F60ScreenError(f"completed F60 checks failed: {checks}")
    statistics = result["statistics"]
    band = outcome_band(float(statistics["winRate"]))
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f60-kl-policy-fresh-lineage-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "classicHardScreenCandidate": band == "kl-policy-lineage-retained",
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStage": {
                "name": "Phase 4F61 KL policy-head classic-hard screen preparation",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )


def verify_phase4f60(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    gate = _mapping(workspace / GATE, "F60 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F60 gate")
    if gate != gate_from_result(workspace, contract):
        raise Phase4F60ScreenError("stored F60 gate differs from the arena result")
    budget = contract["budget"]
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(budget["baselineGlobalHours"])
    ceiling = min(
        baseline + float(budget["screenLimitHours"]),
        float(budget["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours > ceiling:
        raise Phase4F60ScreenError("F60 completed outside its frozen GPU budget")
    _resources(workspace, contract)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f60-kl-policy-fresh-lineage-result-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "statistics": gate["statistics"],
        "outcomeBand": gate["outcomeBand"],
        "classicHardScreenCandidate": gate["classicHardScreenCandidate"],
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
        "globalGpuHours": ledger.total_hours,
        "nextStage": gate["nextStage"],
    }


def run_phase4f60(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f60(workspace), "reused": True}
    contract = _contract(workspace)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F60ScreenError("F60 screen requires CUDA")
    budget = contract["budget"]
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(budget["baselineGlobalHours"])
    ceiling = min(
        baseline + float(budget["screenLimitHours"]),
        float(budget["globalLimitHours"]),
    )
    if ledger.total_hours < baseline or ledger.total_hours >= ceiling:
        raise Phase4F60ScreenError("F60 GPU budget is unavailable or exhausted")
    resources = _resources(workspace, contract)
    _report58, _payload58, candidate_state, _payload21, opponent_state = (
        _f58_and_f21_evidence(workspace)
    )
    suite_path = workspace / str(contract["openingSuite"]["selectedPath"])
    if (
        suite_path != workspace / SCREEN_SUITE
        or sha256_file(suite_path)
        != contract["openingSuite"]["selectedFileSha256"]
    ):
        raise Phase4F60ScreenError("F60 opening suite bytes changed")
    suite = load_opening_suite(
        suite_path, expected_suite=str(contract["workload"]["suite"])
    )
    if (
        suite.content_hash != contract["openingSuite"]["selectedSuiteHash"]
        or len(suite.openings) != int(contract["workload"]["openingPairs"])
    ):
        raise Phase4F60ScreenError("F60 opening suite content changed")
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f60-lineage-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F60ScreenError("F60 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f60-lineage-screen-progress",
                        "status": "running",
                        "openingPairs": pairs,
                        "openingPairsTotal": int(contract["workload"]["openingPairs"]),
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
            print(
                canonical_json_bytes(
                    {
                        "kind": "phase4f60-lineage-screen-progress",
                        "status": "started",
                        "candidate": candidate_identity.selector_id,
                        "opponent": opponent_identity.selector_id,
                        "openingPairs": len(suite.openings),
                        "games": int(contract["workload"]["games"]),
                        "simulations": int(
                            contract["workload"]["candidateSimulations"]
                        ),
                        "resources": resources,
                    }
                ).decode("ascii"),
                end="",
                flush=True,
            )
            candidate_selector = _selector(
                workspace,
                candidate_state,
                contract,
                device,
                candidate_identity,
                int(contract["workload"]["candidateSimulations"]),
            )
            opponent_selector = _selector(
                workspace,
                opponent_state,
                contract,
                device,
                opponent_identity,
                int(contract["workload"]["opponentSimulations"]),
            )
            arena = run_resumable_paired_arena(
                suite,
                candidate_selector,
                opponent_selector,
                store_root=workspace / ARENA_ROOT,
                contract_sha256=arena_contract_sha256(contract),
                actors=int(contract["workload"]["actors"]),
                heartbeat=heartbeat,
            )
            arena.finalize(
                bootstrap_seed=int(contract["workload"]["bootstrapSeed"]),
                bootstrap_samples=int(contract["workload"]["bootstrapSamples"]),
                confidence=float(contract["workload"]["confidence"]),
            )
            gate = gate_from_result(workspace, contract)
            total = ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="completed"
            )
            if total > ceiling:
                raise Phase4F60ScreenError("F60 completed beyond its frozen GPU budget")
            _write_immutable(workspace / GATE, gate, "F60 gate")
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
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f60(args.workspace_root, args.device)
        if args.command == "run"
        else verify_phase4f60(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
