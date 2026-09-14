"""Run the frozen F44 curriculum versus F21 fresh-lineage screen."""

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
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f45_curriculum_pilot_review import (
    ACTORS,
    ARENA_WRITE_RESERVE_BYTES,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    F46_CONTRACT,
    GAMES,
    OPENING_PAIRS,
    SCREEN_GPU_HOURS,
    SCREEN_SUITE,
    SIMULATIONS,
    _f21_evidence,
    _f44_evidence,
    _relative,
    _resources,
    _workspace,
    verify_phase4f45,
)
from .run_manifest import RunLock, canonical_json_bytes, sha256_bytes, sha256_file
from .selectors import MCTSSelector


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f46-curriculum-lineage-screen-v1")
ARENA_ROOT = ROOT / "screen-arena"
GATE = ROOT / "screen-gate.json"


class Phase4F46ScreenError(RuntimeError):
    """Raised when the frozen F46 screen cannot execute or verify."""


def _contract(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f45(workspace)
    contract = _mapping(workspace / F46_CONTRACT, "F46 contract")
    _verify_self_hash(contract, "contractSha256", "F46 contract")
    if (
        verification.get("f46ContractSha256") != contract.get("contractSha256")
        or contract.get("status") != "frozen"
        or contract.get("authorizations", {}).get("f46LineageScreenAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F46ScreenError("F45 does not authorize this F46 screen")
    return contract


def _identities(contract: Mapping[str, object]) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    opponent = contract["opponent"]
    return (
        SelectorIdentity(
            selector_id=str(candidate["selectorId"]),
            artifact_hash=str(candidate["modelFileSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["sourceSha256"]),
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
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4f46ContractSha256": contract["contractSha256"],
                "workloadSha256": contract["workload"]["workloadSha256"],
                "suiteHash": contract["openingSuite"]["selectedSuiteHash"],
            }
        )
    )


def outcome_band(win_rate: float) -> str:
    if win_rate < 0.45:
        return "curriculum-lineage-regression"
    if win_rate < 0.50:
        return "curriculum-lineage-inconclusive"
    return "curriculum-lineage-retained"


def _selector(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    contract: Mapping[str, object],
    device: torch.device,
    identity: SelectorIdentity,
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
        raise Phase4F46ScreenError("model state does not fit the shared frozen network") from error
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
        seed=int(contract["workload"]["mctsSeed"]),
    )
    return BoundMoveSelector(
        MCTSSelector(evaluator, search, search_factory=NativeMCTS, engine="native-mcts"),
        identity,
    )


def gate_from_result(workspace: Path, contract: Mapping[str, object]) -> dict[str, object]:
    result_path = workspace / ARENA_ROOT / "result.json"
    result = verify_completed_arena_result(result_path)
    candidate_identity, opponent_identity = _identities(contract)
    checks = {
        "contract": result.get("contractSha256") == arena_contract_sha256(contract),
        "suite": result.get("suite") == "recovery-fresh-dev-v1",
        "suiteHash": result.get("suiteHash") == contract["workload"]["suiteHash"],
        "openingPairs": result.get("openingPairs") == OPENING_PAIRS,
        "games": result.get("games") == GAMES,
        "candidateIdentity": result.get("candidateIdentity") == candidate_identity.to_dict(),
        "opponentIdentity": result.get("opponentIdentity") == opponent_identity.to_dict(),
        "candidateRuntime": (
            result.get("candidateRuntime", {}).get("kind") == "native-mcts"
            and result.get("candidateRuntime", {}).get("trusted") is True
            and result.get("candidateRuntime", {}).get("configuredSimulations") == SIMULATIONS
            and result.get("candidateRuntime", {}).get("backendHash") == contract["nativeBackendSha256"]
        ),
        "opponentRuntime": (
            result.get("opponentRuntime", {}).get("kind") == "native-mcts"
            and result.get("opponentRuntime", {}).get("trusted") is True
            and result.get("opponentRuntime", {}).get("configuredSimulations") == SIMULATIONS
            and result.get("opponentRuntime", {}).get("backendHash") == contract["nativeBackendSha256"]
        ),
        "draws": result.get("statistics", {}).get("draws") == 0,
    }
    if not all(checks.values()):
        raise Phase4F46ScreenError(f"completed F46 checks failed: {checks}")
    statistics = result["statistics"]
    band = outcome_band(float(statistics["winRate"]))
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f46-curriculum-fresh-lineage-screen-gate",
            "status": "completed",
            "integrityPassed": True,
            "contractSha256": contract["contractSha256"],
            "arenaContractSha256": result["contractSha256"],
            "resultPath": _relative(result_path, workspace),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "statistics": statistics,
            "outcomeBand": band,
            "classicHardScreenCandidate": band == "curriculum-lineage-retained",
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "screenTrajectoriesTrainingUseAllowed": False,
            "nextStageRequiresConfirmation": True,
        },
        "gateSha256",
    )


def run_phase4f46(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    gate_path = workspace / GATE
    if gate_path.exists():
        stored = _mapping(gate_path, "F46 gate")
        _verify_self_hash(stored, "gateSha256", "F46 gate")
        if stored != gate_from_result(workspace, contract):
            raise Phase4F46ScreenError("stored F46 gate differs from arena result")
        return {**stored, "reused": True}
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F46ScreenError("F46 screen requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(contract["budget"]["baselineGlobalHours"])
    if ledger.total_hours >= baseline + SCREEN_GPU_HOURS:
        raise Phase4F46ScreenError("F46 GPU budget is exhausted")
    _resources(workspace)
    _f44_report, candidate_state = _f44_evidence(workspace)
    _f21_payload, opponent_state = _f21_evidence(workspace)
    suite_path = workspace / SCREEN_SUITE
    if sha256_file(suite_path) != contract["openingSuite"]["selectedFileSha256"]:
        raise Phase4F46ScreenError("F46 opening suite bytes changed")
    suite = load_opening_suite(suite_path, expected_suite="recovery-fresh-dev-v1")
    candidate_identity, opponent_identity = _identities(contract)
    run_id = f"phase4f46-lineage-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
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
                raise Phase4F46ScreenError("F46 GPU budget was exhausted")
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f46-lineage-screen-progress",
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
            candidate_selector = _selector(workspace, candidate_state, contract, device, candidate_identity)
            opponent_selector = _selector(workspace, opponent_state, contract, device, opponent_identity)
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
            _write_immutable(gate_path, gate, "F46 gate")
            total = ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status="completed")
            return {**gate, "globalGpuHours": total, "reused": False}
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
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = run_phase4f46(args.workspace_root, args.device)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
