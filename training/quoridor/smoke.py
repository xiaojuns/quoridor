"""Small, real M4 self-play/update/checkpoint smoke workload.

The smoke path is deliberately a tactical one-game workload. It exercises the
new engine, MCTS, canonical replay, optimizer and resume artifact on CPU in a
bounded run; it is not a strength or throughput experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import engine
from .episode import start_episode
from .model_config import load_model_config
from .policy_value import NetworkConfig, PolicyValueNetwork
from .project_config import PROJECT_ROOT, ProjectConfig
from .project_runtime import RunSession
from .replay import ReplayBuffer
from .search_config import search_config_from_project
from .selfplay import NewSelfPlayConfig, play_new_selfplay_episode
from .trainer import AlphaZeroTrainer
from .training_checkpoint import (
    default_lineage,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .model_checkpoint import save_model_checkpoint


SMOKE_REPLAY_CAPACITY = 16
SMOKE_BATCH_SIZE = 1
SMOKE_OPTIMIZER_STEPS = 1


def _tactical_episode(max_game_plies: int) -> object:
    # Player zero has one immediate goal move. This is a valid v1 board and is
    # recorded in run metadata so the smoke does not imply opening strength.
    state = engine.BoardState(pawns=((1, 4), (4, 4)), to_play=0)
    return start_episode(state, max_game_plies=max_game_plies)


def _deterministic_smoke_model(config: NetworkConfig, winning_action: int) -> PolicyValueNetwork:
    torch.manual_seed(20261010)
    model = PolicyValueNetwork(config)
    # Keep the real requested architecture and inference path, while making
    # the one-game tactical fixture complete under a tiny CPU budget.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head[-1].bias[winning_action] = 20.0
    model.eval()
    return model


def run_smoke(config: ProjectConfig) -> dict[str, object]:
    """Run one complete game, one optimizer step, and a resume roundtrip."""

    if type(config) is not ProjectConfig or config.mode != "smoke":
        raise TypeError("M4 smoke requires a validated smoke ProjectConfig")
    search = search_config_from_project(config)
    # The smoke uses the repository's declared 64x4 architecture. The model
    # YAML is read and compared before work begins; no architecture override is
    # hidden in the actor.
    model_config = load_model_config(PROJECT_ROOT / "configs/model.yaml")
    if model_config != NetworkConfig():
        raise ValueError("smoke requires the declared 64-channel/4-block model config")
    tactical = _tactical_episode(config.document["budget"]["max_game_plies"])
    winning_action = engine.encode_move(0, 4)
    model = _deterministic_smoke_model(model_config, winning_action)
    replay = ReplayBuffer(capacity=SMOKE_REPLAY_CAPACITY, seed=config.document["seed"])
    lineage = default_lineage(config_sha256=config.sha256)
    lineage["smoke_fixture"] = "tactical-immediate-goal-v1"
    lineage["smoke_model_init"] = "zero-weights-winning-policy-bias"

    with RunSession(config) as run:
        run_manifest = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
        lineage["source_sha256"] = run_manifest["source_sha256"]
        lineage["seed"] = config.document["seed"]
        lineage["search_config"] = {
            "simulations": search.simulations,
            "c_puct": search.c_puct,
            "root_noise": search.root_noise,
            "temperature": search.temperature,
            "mode": search.mode,
        }
        lineage["model_architecture"] = model_config.to_dict()
        run.log(
            "m4_effective_config",
            model=model_config.to_dict(),
            search={
                "simulations": search.simulations,
                "c_puct": search.c_puct,
                "root_noise": search.root_noise,
                "temperature": search.temperature,
                "mode": search.mode,
            },
            replay_capacity=SMOKE_REPLAY_CAPACITY,
            batch_size=SMOKE_BATCH_SIZE,
            optimizer_steps=SMOKE_OPTIMIZER_STEPS,
            new_games=1,
            fixture=lineage["smoke_fixture"],
        )
        actor_config = NewSelfPlayConfig(
            search,
            max_game_plies=config.document["budget"]["max_game_plies"],
            seed=config.document["seed"],
            config_hash=config.sha256,
        )
        game = play_new_selfplay_episode(
            model,
            actor_config,
            deadline=run.deadline,
            initial_episode=tactical,
        )
        run.log(
            "selfplay_finished",
            status=game.status,
            reason=game.reason,
            plies=game.episode.plies,
            positions=len(game.positions),
        )
        if not game.replay_eligible:
            raise RuntimeError(f"smoke self-play was not replay eligible: {game.status}/{game.reason}")
        if not replay.add_game(game.replay_game()):
            raise RuntimeError("smoke replay rejected a completed game")
        run.log("replay_admitted", positions=len(replay), stats=replay.stats())

        trainer = AlphaZeroTrainer(
            model,
            learning_rate=1e-3,
            use_amp=False,
            require_legal_masks=True,
            schedule_steps=SMOKE_OPTIMIZER_STEPS,
        )
        metrics = None
        for _ in range(SMOKE_OPTIMIZER_STEPS):
            run.deadline.check()
            metrics = trainer.train_step(replay.sample(SMOKE_BATCH_SIZE))
        assert metrics is not None
        run.log("optimizer_step", step=trainer.step, metrics=metrics.__dict__)

        checkpoint_path = run.path / "training-resume.pt"
        run.deadline.check()
        artifact = save_training_checkpoint(
            checkpoint_path,
            trainer,
            replay=replay,
            lineage=lineage,
            config_sha256=config.sha256,
            counters={"games": 1, "accepted_games": replay.accepted_games},
        )
        run.deadline.check()
        run.log("checkpoint_saved", path=str(checkpoint_path), sha256=artifact.sha256)
        # Reload into fresh objects and verify the artifact identity/lineage.
        restored_model = PolicyValueNetwork(model_config)
        restored_trainer = AlphaZeroTrainer(
            restored_model,
            learning_rate=1e-3,
            use_amp=False,
            require_legal_masks=True,
            schedule_steps=SMOKE_OPTIMIZER_STEPS,
        )
        restored_replay = ReplayBuffer(capacity=SMOKE_REPLAY_CAPACITY, seed=config.document["seed"])
        run.deadline.check()
        loaded = load_training_checkpoint(
            checkpoint_path,
            restored_trainer,
            replay=restored_replay,
            expected_lineage=lineage,
            expected_config_sha256=config.sha256,
        )
        if loaded.artifact.sha256 != artifact.sha256 or len(restored_replay) != len(replay):
            raise RuntimeError("smoke checkpoint reload identity check failed")
        run.deadline.check()
        run.log("checkpoint_reloaded", sha256=loaded.artifact.sha256, step=restored_trainer.step)

        # Also publish a compatible inference-only artifact for future M5 input
        # validation; the training checkpoint remains the resume source.
        run.deadline.check()
        model_artifact = save_model_checkpoint(run.path / "model-weights.pt", restored_model)
        run.deadline.check()
        run.log("model_weights_saved", sha256=model_artifact.sha256)
        summary = {
            "status": "completed",
            "games": 1,
            "accepted_games": replay.accepted_games,
            "replay_positions": len(replay),
            "optimizer_steps": trainer.step,
            "loss": metrics.loss,
            "checkpoint_sha256": artifact.sha256,
            "model_sha256": model_artifact.sha256,
            "fixture": lineage["smoke_fixture"],
        }
        run.log("m4_smoke_completed", **summary)
    return {**summary, "run_dir": str(run.path.resolve())}


__all__ = ["run_smoke"]
