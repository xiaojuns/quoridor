from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from .arena import ArenaConfig, evaluate_candidate
from .checkpoint import CheckpointManager
from .config import (
    ExperimentConfig,
    load_experiment_config,
    resolve_device,
    seed_everything,
)
from .evaluator import TorchEvaluator
from .game import State
from .mcts import MCTSConfig
from .mcts import Evaluator, MCTS
from .network import NetworkConfig, PolicyValueNetwork
from .replay import ReplayBuffer
from .selfplay import SelfPlayConfig, play_self_game, play_self_games_scheduled
from .storage import atomic_json_save, directory_size
from .trainer import AlphaZeroTrainer


def _network_config(config: ExperimentConfig) -> NetworkConfig:
    return NetworkConfig(
        input_channels=config.network.input_channels,
        channels=config.network.channels,
        blocks=config.network.residual_blocks,
        se_reduction=config.network.se_reduction,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden=config.network.value_hidden,
    )


def _mcts_config(config: ExperimentConfig, *, simulations: int | None = None) -> MCTSConfig:
    return MCTSConfig(
        simulations=simulations or config.mcts.simulations,
        c_base=config.mcts.c_base,
        c_init=config.mcts.c_init,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=1.0,
        add_root_noise=True,
        seed=config.seed,
    )


def _trainer(config: ExperimentConfig, model: PolicyValueNetwork) -> AlphaZeroTrainer:
    return AlphaZeroTrainer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        use_amp=config.training.use_amp,
        schedule_steps=config.training.schedule_steps,
        minimum_learning_rate=config.training.minimum_learning_rate,
    )


def _quick_state() -> State:
    return State(
        pawns=((1, 0), (0, 8)),
        walls_remaining=(0, 0),
        current_player=0,
        move_number=32,
    )


def run_smoke(
    config: ExperimentConfig,
    *,
    device_override: str | None = None,
    full_game: bool = False,
    resume: bool = False,
    engine: str = "python",
) -> dict:
    started_at = time.monotonic()
    seed_everything(config.seed)
    device = resolve_device(device_override or config.device)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    architecture = _network_config(config)
    candidate_model = PolicyValueNetwork(architecture).to(device)
    trainer = _trainer(config, candidate_model)
    replay = ReplayBuffer(
        config.training.replay_capacity,
        seed=config.seed,
    )
    manager = CheckpointManager(
        run_dir,
        artifact_limit_bytes=config.artifact_limit_bytes,
        max_to_keep=config.training.max_checkpoints,
    )
    resumed_from = None
    resume_payload = None
    generation = 0
    if resume:
        resumed_from = manager.latest()
        if resumed_from is None:
            raise FileNotFoundError("--resume requested but no checkpoint exists")
        resume_payload = manager.load(
            candidate_model,
            trainer.optimizer,
            path=resumed_from,
            map_location=device,
            restore_rng=True,
        )
        trainer_runtime = resume_payload.get("extra", {}).get("trainerRuntime")
        if trainer_runtime is None:
            trainer.step = int(resume_payload["step"])
        else:
            trainer.load_runtime_state_dict(trainer_runtime)
            if trainer.step != int(resume_payload["step"]):
                raise ValueError("checkpoint step and trainer runtime step disagree")
        if resume_payload.get("replay") is not None:
            replay.load_state_dict(resume_payload["replay"])
        generation = int(resume_payload.get("extra", {}).get("generation", 0)) + 1

    champion_model = copy.deepcopy(candidate_model)
    if resume_payload is not None:
        champion_state = resume_payload.get("extra", {}).get("championModel")
        if champion_state is not None:
            champion_model.load_state_dict(champion_state)
    evaluator = TorchEvaluator(champion_model, device)
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS]
    if engine == "python":
        search_factory = MCTS
    elif engine == "native":
        from .native_mcts import NativeMCTS

        search_factory = NativeMCTS
    else:
        raise ValueError("engine must be python or native")
    search_config = _mcts_config(config)
    generation_seed = (config.seed + generation * 1_000_003) % (2**63)
    search_config = replace(search_config, seed=generation_seed)
    initial_state = None if full_game else _quick_state()
    games = []
    self_play_config = SelfPlayConfig(
        mcts=search_config,
        temperature_moves=config.self_play.temperature_moves,
        mirror_augmentation=True,
        seed=generation_seed,
        input_channels=architecture.input_channels,
    )
    if engine == "native":
        games.extend(
            play_self_games_scheduled(
                evaluator,
                self_play_config,
                config.self_play.games,
                actors=config.self_play.actors,
                initial_state=initial_state,
                search_factory=search_factory,
            )
        )
    else:
        for game_index in range(config.self_play.games):
            games.append(
                play_self_game(
                    evaluator,
                    replace(self_play_config, seed=generation_seed + game_index),
                    initial_state=initial_state,
                    search_factory=search_factory,
                )
            )
    for game in games:
        replay.extend(game.samples)

    metrics = []
    for _ in range(config.training.training_steps):
        batch = replay.sample(config.training.batch_size, device=device)
        metrics.append(asdict(trainer.train_step(batch)))

    arena_result = evaluate_candidate(
        TorchEvaluator(candidate_model, device),
        TorchEvaluator(champion_model, device),
        mcts_config=search_config,
        arena_config=ArenaConfig(
            games=config.arena.games,
            simulations=config.arena.simulations,
            acceptance_rate=0.55,
            seed=generation_seed + 10_000,
        ),
        initial_state=initial_state,
        search_factory=search_factory,
        batched=engine == "native",
    )
    if arena_result.accepted:
        champion_model.load_state_dict(candidate_model.state_dict())

    checkpoint_path = manager.save(
        step=trainer.step,
        model=candidate_model,
        optimizer=trainer.optimizer,
        config=config,
        replay_state=replay.state_dict(),
        extra={
            "generation": generation,
            "generationSeed": generation_seed,
            "selfPlayGames": len(games),
            "selfPlayMoves": sum(game.move_count for game in games),
            "terminationReasons": [game.termination_reason for game in games],
            "trainerRuntime": trainer.runtime_state_dict(),
            "championAccepted": arena_result.accepted,
            "championModel": (
                None if arena_result.accepted else champion_model.state_dict()
            ),
        },
    )

    resumed_model = PolicyValueNetwork(architecture).to(device)
    resumed_trainer = _trainer(config, resumed_model)
    payload = manager.load(
        resumed_model,
        resumed_trainer.optimizer,
        path=checkpoint_path,
        map_location=device,
        restore_rng=False,
    )
    resumed_runtime = payload.get("extra", {}).get("trainerRuntime")
    if resumed_runtime is None:
        resumed_trainer.step = int(payload["step"])
    else:
        resumed_trainer.load_runtime_state_dict(resumed_runtime)

    summary = {
        "version": 1,
        "experiment": config.name,
        "generation": generation,
        "generationSeed": generation_seed,
        "engine": engine,
        "device": str(device),
        "network": architecture.to_dict(),
        "parameterCount": sum(parameter.numel() for parameter in candidate_model.parameters()),
        "selfPlay": {
            "fullGame": full_game,
            "actors": config.self_play.actors if engine == "native" else 1,
            "games": len(games),
            "moves": sum(game.move_count for game in games),
            "samples": sum(len(game.samples) for game in games),
            "winners": {
                "0": sum(game.winner == 0 for game in games),
                "1": sum(game.winner == 1 for game in games),
            },
            "terminationReasons": {
                reason: sum(game.termination_reason == reason for game in games)
                for reason in ("goal", "repetition", "move-limit")
            },
            "illegalActions": 0,
        },
        "training": {
            "steps": trainer.step,
            "stepsThisRun": config.training.training_steps,
            "metrics": metrics,
            "replaySize": len(replay),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "resumedStep": resumed_trainer.step,
            "resumedFrom": str(resumed_from) if resumed_from is not None else None,
        },
        "arena": {
            "games": arena_result.games,
            "candidateWins": arena_result.candidate_wins,
            "championWins": arena_result.champion_wins,
            "candidateWinRate": arena_result.candidate_win_rate,
            "accepted": arena_result.accepted,
            "championUpdated": arena_result.accepted,
        },
        "artifacts": {
            "bytes": directory_size(run_dir),
            "limitBytes": config.artifact_limit_bytes,
        },
        "elapsedSeconds": time.monotonic() - started_at,
    }
    atomic_json_save(summary, run_dir / "smoke-summary.json")
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quoridor AlphaZero reference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run one self-play/train/checkpoint/arena cycle")
    smoke.add_argument(
        "--config",
        default="training/configs/smoke.json",
        help="experiment JSON path",
    )
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"))
    smoke.add_argument("--engine", choices=("python", "native"), default="python")
    smoke.add_argument(
        "--summary-only",
        action="store_true",
        help="print a compact result without the per-step metric list",
    )
    smoke.add_argument(
        "--full-game",
        action="store_true",
        help="start self-play and arena games from the standard initial state",
    )
    smoke.add_argument(
        "--resume",
        action="store_true",
        help="restore the latest checkpoint, optimizer, RNG, and replay first",
    )
    train = subparsers.add_parser(
        "train",
        help="run configured full-board self-play, updates, checkpoint, and arena",
    )
    train.add_argument(
        "--config",
        default="training/configs/reference.json",
        help="experiment JSON path",
    )
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    train.add_argument("--engine", choices=("python", "native"), default="python")
    train.add_argument(
        "--summary-only",
        action="store_true",
        help="print a compact result without the per-step metric list",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="restore the latest checkpoint, optimizer, RNG, and replay first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command in {"smoke", "train"}:
        config = load_experiment_config(args.config)
        if config.pipeline != "reference":
            raise RuntimeError(
                "Phase 4 configs are calibration-gated and cannot run through "
                "the legacy in-memory training pipeline"
            )
        summary = run_smoke(
            config,
            device_override=args.device,
            full_game=args.full_game if args.command == "smoke" else True,
            resume=args.resume,
            engine=args.engine,
        )
        output = summary
        if args.summary_only:
            output = {
                key: summary[key]
                for key in (
                    "version",
                    "experiment",
                    "engine",
                    "device",
                    "generation",
                    "generationSeed",
                    "selfPlay",
                    "checkpoint",
                    "arena",
                    "artifacts",
                    "elapsedSeconds",
                )
            }
            output["training"] = {
                key: summary["training"][key]
                for key in ("steps", "stepsThisRun", "replaySize")
            }
            metrics = summary["training"]["metrics"]
            output["training"]["finalMetric"] = metrics[-1] if metrics else None
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
