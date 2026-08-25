"""Read-only policy/value/MCTS root-cause ablation for Phase 4.

This module deliberately does not create replay, checkpoints, optimizer state, or
GPU-ledger entries.  It evaluates one frozen model on a deterministic, balanced
sample of an existing holdout replay store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

import torch

from .classic_bridge import ClassicAIBridge, ClassicSearchBudget, HARD_MAX_DEPTH
from .codec import canonicalize_state, to_canonical_action_index
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .game import State, legal_action_indices, legal_pawn_moves, shortest_path_distance
from .mcts import ACTION_SIZE, MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .replay_store import ReplaySample, ReplayStore


SCHEMA_VERSION = 1
SELECTION_SEED = 20260817
CLASSIC_NODES = 587
DEFAULT_POSITIONS_PER_BAND = 6
DEFAULT_SEARCH_BUDGETS = (64, 256, 800, 2048)
DEFAULT_ABLATION_BUDGET = 800
DEFAULT_MODEL = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/"
    "checkpoints/candidate-model-checkpoint-g00000001-s000000000192.pt"
)
DEFAULT_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
DEFAULT_HOLDOUT = Path(
    "training/runs/phase4/phase4f/recovery-challenge-v2-data/holdout"
)
DEFAULT_OUTPUT = Path("training/runs/phase4/root-cause-ablation-v1/report.json")


class RootCauseAblationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FixedPosition:
    sample: ReplaySample
    band: str


class ComponentEvaluator:
    """Compose learned/uniform policy and learned/zero value components."""

    def __init__(
        self,
        learned: TorchEvaluator,
        *,
        learned_policy: bool,
        learned_value: bool,
    ) -> None:
        self.learned = learned
        self.learned_policy = learned_policy
        self.learned_value = learned_value
        self.positions = 0
        self.calls = 0
        self.batch_sizes: list[int] = []

    def enable_native_encoder(self, module: object) -> None:
        self.learned.enable_native_encoder(module)

    def evaluate(self, state: State) -> tuple[Sequence[float], float]:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self, states: Iterable[State]
    ) -> list[tuple[Sequence[float], float]]:
        state_list = list(states)
        if not state_list:
            return []
        self.calls += 1
        self.positions += len(state_list)
        self.batch_sizes.append(len(state_list))
        if self.learned_policy or self.learned_value:
            learned = self.learned.evaluate_batch(state_list)
        else:
            learned = [((0.0,) * ACTION_SIZE, 0.0) for _ in state_list]
        return [
            (
                policy if self.learned_policy else (0.0,) * ACTION_SIZE,
                value if self.learned_value else 0.0,
            )
            for policy, value in learned
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _load_replay(root: Path) -> ReplayStore:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        capacity = manifest["capacity"]
        return ReplayStore(
            root,
            max_games=int(capacity["games"]),
            max_plies=int(capacity["plies"]),
            max_bytes=int(capacity["bytes"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RootCauseAblationError(f"cannot open holdout replay: {root}") from error


def _band(ply_index: int, _ply_count: int) -> str:
    """Use the same absolute bands as the frozen fixed-health diagnostic."""

    if ply_index < 16:
        return "early"
    if ply_index < 40:
        return "middle"
    return "late"


def select_fixed_positions(
    store: ReplayStore,
    *,
    per_band: int = DEFAULT_POSITIONS_PER_BAND,
) -> tuple[FixedPosition, ...]:
    if type(per_band) is not int or per_band <= 0:
        raise ValueError("per_band must be a positive integer")
    candidates: dict[str, list[tuple[str, ReplaySample]]] = {
        "early": [],
        "middle": [],
        "late": [],
    }
    for game_id in store.game_ids:
        game, states = store.load_game_with_states(game_id)
        winner = states[-1].winner
        if winner not in (0, 1):
            raise RootCauseAblationError(f"holdout game has no winner: {game_id}")
        for ply_index, ply in enumerate(game.plies):
            state = states[ply_index]
            band = _band(ply_index, len(game.plies))
            sample = ReplaySample(
                game_id=game_id,
                ply_index=ply_index,
                state=state,
                action_index=ply.action_index,
                visit_counts=ply.visit_counts,
                root_value=ply.root_value,
                outcome=1.0 if state.current_player == winner else -1.0,
                model_generation=ply.model_generation,
                model_id=ply.model_id,
                game_metadata=game.metadata,
                ply_metadata=ply.metadata,
            )
            token = hashlib.sha256(
                f"{SELECTION_SEED}:{game_id}:{ply_index}".encode("ascii")
            ).hexdigest()
            candidates[band].append((token, sample))

    selected: list[FixedPosition] = []
    for band in ("early", "middle", "late"):
        ordered = sorted(candidates[band], key=lambda item: item[0])
        chosen: list[ReplaySample] = []
        for desired_player in (0, 1):
            target = per_band // 2
            chosen.extend(
                sample
                for _, sample in ordered
                if sample.state.current_player == desired_player
            )
            chosen = chosen[: target if desired_player == 0 else per_band]
        if len(chosen) < per_band:
            used = {(sample.game_id, sample.ply_index) for sample in chosen}
            chosen.extend(
                sample
                for _, sample in ordered
                if (sample.game_id, sample.ply_index) not in used
            )
            chosen = chosen[:per_band]
        if len(chosen) != per_band:
            raise RootCauseAblationError(f"not enough {band} holdout positions")
        selected.extend(FixedPosition(sample=sample, band=band) for sample in chosen)
    return tuple(selected)


def _network_config(path: Path) -> NetworkConfig:
    config = load_experiment_config(path)
    return NetworkConfig(
        input_channels=config.network.input_channels,
        channels=config.network.channels,
        blocks=config.network.residual_blocks,
        se_reduction=config.network.se_reduction,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden=config.network.value_hidden,
    )


def _load_model(model_path: Path, config_path: Path, device: torch.device) -> PolicyValueNetwork:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RootCauseAblationError(f"cannot load frozen model: {model_path}") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise RootCauseAblationError("frozen model does not contain modelState")
    model = PolicyValueNetwork(_network_config(config_path))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RootCauseAblationError("model state does not match network config") from error
    return model.to(device).eval()


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _classic_static_score(state: State) -> float:
    player = state.current_player
    opponent = 1 - player
    own_distance = shortest_path_distance(state, player)
    opponent_distance = shortest_path_distance(state, opponent)
    wall_balance = state.walls_remaining[player] - state.walls_remaining[opponent]
    mobility_balance = len(legal_pawn_moves(state, player)) - len(
        legal_pawn_moves(state, opponent)
    )
    return 120.0 * (opponent_distance - own_distance) + 7.0 * wall_balance + 3.0 * mobility_balance


def _direct_diagnostics(
    positions: Sequence[FixedPosition],
    learned: TorchEvaluator,
    classic_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    canonical = [canonicalize_state(position.sample.state) for position in positions]
    evaluated = learned.evaluate_batch(canonical)
    rows: list[dict[str, object]] = []
    for position, (logits, value), classic in zip(positions, evaluated, classic_rows):
        state = position.sample.state
        legal = tuple(legal_action_indices(state))
        canonical_legal = tuple(
            to_canonical_action_index(action, state.current_player) for action in legal
        )
        legal_logits = [float(logits[action]) for action in canonical_legal]
        probabilities = _softmax(legal_logits)
        ranked = sorted(
            zip(legal, probabilities), key=lambda item: (-item[1], item[0])
        )
        classic_action = int(classic["actionIndex"])
        rank = next(
            index + 1 for index, (action, _) in enumerate(ranked) if action == classic_action
        )
        rows.append(
            {
                "gameId": position.sample.game_id,
                "plyIndex": position.sample.ply_index,
                "band": position.band,
                "currentPlayer": state.current_player,
                "legalActions": len(legal),
                "outcome": position.sample.outcome,
                "learnedValue": float(value),
                "classicStaticScore": _classic_static_score(state),
                "classicSearchScore": float(classic["score"]),
                "classicActionIndex": classic_action,
                "recordedTeacherActionIndex": position.sample.action_index,
                "policyTop1ActionIndex": ranked[0][0],
                "classicActionRank": rank,
                "classicActionProbability": next(
                    probability for action, probability in ranked if action == classic_action
                ),
            }
        )
    values = [float(row["learnedValue"]) for row in rows]
    outcomes = [float(row["outcome"]) for row in rows]
    static = [float(row["classicStaticScore"]) for row in rows]
    search_scores = [float(row["classicSearchScore"]) for row in rows]
    return {
        "positions": len(rows),
        "policy": {
            "top1ClassicAgreement": mean(
                float(row["classicActionRank"] == 1) for row in rows
            ),
            "top5ClassicCoverage": mean(
                float(int(row["classicActionRank"]) <= 5) for row in rows
            ),
            "top10ClassicCoverage": mean(
                float(int(row["classicActionRank"]) <= 10) for row in rows
            ),
            "meanClassicActionRank": mean(
                int(row["classicActionRank"]) for row in rows
            ),
            "meanClassicActionProbability": mean(
                float(row["classicActionProbability"]) for row in rows
            ),
        },
        "value": {
            "mean": mean(values),
            "meanOutcome": mean(outcomes),
            "mseToOutcome": mean((value - outcome) ** 2 for value, outcome in zip(values, outcomes)),
            "signAccuracy": mean(float(value * outcome > 0) for value, outcome in zip(values, outcomes)),
            "pearsonToOutcome": _pearson(values, outcomes),
            "classicStaticSignAccuracy": mean(
                float(score * outcome > 0) for score, outcome in zip(static, outcomes)
            ),
            "classicSearchSignAccuracy": mean(
                float(score * outcome > 0) for score, outcome in zip(search_scores, outcomes)
            ),
            "pearsonToClassicStatic": _pearson(values, static),
            "pearsonToClassicSearch": _pearson(values, search_scores),
        },
        "rows": rows,
    }


def _classic_labels(positions: Sequence[FixedPosition]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ClassicAIBridge(request_timeout=10.0) as bridge:
        for ordinal, position in enumerate(positions):
            result = bridge.choose(
                position.sample.state,
                budget=ClassicSearchBudget.nodes(CLASSIC_NODES),
                max_depth=HARD_MAX_DEPTH,
                timeout=10.0,
            )
            rows.append(
                {
                    "ordinal": ordinal,
                    "actionIndex": result.action_index,
                    "score": result.score,
                    "depth": result.depth,
                    "nodes": result.nodes,
                    "elapsedMs": result.elapsed_ms,
                }
            )
    return rows


def _search_row(
    *,
    label: str,
    simulations: int,
    learned_policy: bool,
    learned_value: bool,
    positions: Sequence[FixedPosition],
    classic_rows: Sequence[Mapping[str, object]],
    learned: TorchEvaluator,
    native_module: object,
    c_base: float,
    c_init: float,
) -> dict[str, object]:
    evaluator = ComponentEvaluator(
        learned,
        learned_policy=learned_policy,
        learned_value=learned_value,
    )
    search = NativeMCTS(
        evaluator,
        MCTSConfig(
            simulations=simulations,
            c_base=c_base,
            c_init=c_init,
            dirichlet_epsilon=0.0,
            temperature=0.0,
            add_root_noise=False,
            seed=SELECTION_SEED,
        ),
        module=native_module,
    )
    states = [position.sample.state for position in positions]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    results = search.run_batch(
        states,
        temperatures=[0.0] * len(states),
        add_root_noise=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result_rows: list[dict[str, object]] = []
    for position, classic, result in zip(positions, classic_rows, results):
        visits = [count for count in result.visit_counts if count > 0]
        result_rows.append(
            {
                "gameId": position.sample.game_id,
                "plyIndex": position.sample.ply_index,
                "band": position.band,
                "actionIndex": result.action_index,
                "classicActionIndex": classic["actionIndex"],
                "classicAgreement": result.action_index == classic["actionIndex"],
                "rootValue": result.root_value,
                "visitedActions": len(visits),
                "maximumVisitFraction": max(visits) / sum(visits),
            }
        )
    return {
        "label": label,
        "simulations": simulations,
        "learnedPolicy": learned_policy,
        "learnedValue": learned_value,
        "positions": len(positions),
        "elapsedSeconds": elapsed,
        "positionsPerSecond": len(positions) / elapsed,
        "simulationsPerSecond": len(positions) * simulations / elapsed,
        "classicAgreement": mean(float(row["classicAgreement"]) for row in result_rows),
        "classicAgreementByBand": {
            band: mean(
                float(row["classicAgreement"])
                for row in result_rows
                if row["band"] == band
            )
            for band in ("early", "middle", "late")
        },
        "meanVisitedActions": mean(int(row["visitedActions"]) for row in result_rows),
        "meanMaximumVisitFraction": mean(
            float(row["maximumVisitFraction"]) for row in result_rows
        ),
        "evaluator": {
            "calls": evaluator.calls,
            "positions": evaluator.positions,
            "meanBatchSize": mean(evaluator.batch_sizes),
            "maximumBatchSize": max(evaluator.batch_sizes),
        },
        "peakCudaAllocatedBytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "rows": result_rows,
    }


def _nvidia_smi() -> dict[str, object] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5.0,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    first = output.splitlines()[0].split(", ")
    if len(first) != 3:
        return {"raw": output}
    return {"name": first[0], "driverVersion": first[1], "memoryMiB": int(first[2])}


def run_ablation(
    *,
    workspace: Path,
    model_relative: Path = DEFAULT_MODEL,
    config_relative: Path = DEFAULT_CONFIG,
    holdout_relative: Path = DEFAULT_HOLDOUT,
    output_relative: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda:0",
    positions_per_band: int = DEFAULT_POSITIONS_PER_BAND,
    search_budgets: Sequence[int] = DEFAULT_SEARCH_BUDGETS,
    ablation_budget: int = DEFAULT_ABLATION_BUDGET,
) -> dict[str, object]:
    workspace = workspace.resolve()
    model_path = workspace / model_relative
    config_path = workspace / config_relative
    holdout_root = workspace / holdout_relative
    output_path = workspace / output_relative
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RootCauseAblationError("CUDA was requested but torch.cuda.is_available() is false")
    if not search_budgets or any(type(value) is not int or value <= 0 for value in search_budgets):
        raise ValueError("search_budgets must contain positive integers")
    if type(ablation_budget) is not int or ablation_budget <= 0:
        raise ValueError("ablation_budget must be a positive integer")
    store = _load_replay(holdout_root)
    positions = select_fixed_positions(store, per_band=positions_per_band)
    model = _load_model(model_path, config_path, device)
    native_module = load_native_module()
    learned = TorchEvaluator(model, device=device, use_amp=device.type == "cuda")
    learned.enable_native_encoder(native_module)
    classic_rows = _classic_labels(positions)
    direct = _direct_diagnostics(positions, learned, classic_rows)

    matrix: list[dict[str, object]] = []
    config = load_experiment_config(config_path)
    for simulations in search_budgets:
        print(
            json.dumps({"stage": "search", "label": "learned-policy-learned-value", "simulations": simulations}),
            flush=True,
        )
        matrix.append(
            _search_row(
                label="learned-policy-learned-value",
                simulations=simulations,
                learned_policy=True,
                learned_value=True,
                positions=positions,
                classic_rows=classic_rows,
                learned=learned,
                native_module=native_module,
                c_base=config.mcts.c_base,
                c_init=config.mcts.c_init,
            )
        )
    for label, learned_policy, learned_value in (
        ("learned-policy-zero-value", True, False),
        ("uniform-policy-learned-value", False, True),
        ("uniform-policy-zero-value", False, False),
    ):
        print(
            json.dumps({"stage": "search", "label": label, "simulations": ablation_budget}),
            flush=True,
        )
        matrix.append(
            _search_row(
                label=label,
                simulations=ablation_budget,
                learned_policy=learned_policy,
                learned_value=learned_value,
                positions=positions,
                classic_rows=classic_rows,
                learned=learned,
                native_module=native_module,
                c_base=config.mcts.c_base,
                c_init=config.mcts.c_init,
            )
        )

    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4-root-cause-policy-value-search-ablation",
        "status": "completed",
        "readOnlyDiagnostic": True,
        "trainingStarted": False,
        "gpuLedgerUpdated": False,
        "selectionSeed": SELECTION_SEED,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "device": str(device),
            "cudaAvailable": torch.cuda.is_available(),
            "nvidiaSmi": _nvidia_smi(),
        },
        "model": {
            "path": model_relative.as_posix(),
            "sha256": _sha256_file(model_path),
            "configPath": config_relative.as_posix(),
            "configSha256": _sha256_file(config_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "channels": config.network.channels,
            "residualBlocks": config.network.residual_blocks,
        },
        "holdout": {
            "path": holdout_relative.as_posix(),
            "manifestSha256": _sha256_file(holdout_root / "manifest.json"),
            "storeGames": store.game_count,
            "storePlies": store.ply_count,
            "selectedPositions": len(positions),
            "positionsPerBand": positions_per_band,
            "selectionAlgorithm": "sha256(seed:gameId:plyIndex), balanced by ply band and side",
        },
        "classicReference": {
            "engine": "classic-hard-negamax",
            "nodes": CLASSIC_NODES,
            "maximumDepth": HARD_MAX_DEPTH,
            "meanElapsedMs": mean(float(row["elapsedMs"]) for row in classic_rows),
            "meanNodes": mean(int(row["nodes"]) for row in classic_rows),
        },
        "searchContract": {
            "scalingBudgets": list(search_budgets),
            "componentAblationBudget": ablation_budget,
            "cBase": config.mcts.c_base,
            "cInit": config.mcts.c_init,
            "rootNoise": False,
            "temperature": 0.0,
        },
        "direct": direct,
        "searchMatrix": matrix,
    }
    core_hash = hashlib.sha256(_canonical_json(report)).hexdigest()
    report["reportSha256"] = core_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(report))
    temporary.replace(output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positions-per-band", type=int, default=DEFAULT_POSITIONS_PER_BAND)
    parser.add_argument(
        "--search-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEARCH_BUDGETS),
    )
    parser.add_argument("--ablation-budget", type=int, default=DEFAULT_ABLATION_BUDGET)
    args = parser.parse_args(argv)
    report = run_ablation(
        workspace=args.workspace_root,
        model_relative=args.model,
        config_relative=args.config,
        holdout_relative=args.holdout,
        output_relative=args.output,
        device_name=args.device,
        positions_per_band=args.positions_per_band,
        search_budgets=args.search_budgets,
        ablation_budget=args.ablation_budget,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "reportPath": args.output.as_posix(),
                "reportSha256": report["reportSha256"],
                "positions": report["holdout"]["selectedPositions"],
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
