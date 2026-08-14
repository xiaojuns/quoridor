from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ARTIFACT_LIMIT_BYTES = 8 * 1024**3
MAXIMUM_ARTIFACT_LIMIT_BYTES = 12 * 1024**3
EXTENDED_ARTIFACT_LIMIT_BYTES = 16 * 1024**3


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    normalized = _non_negative_float(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest or null")
    return value


@dataclass(frozen=True)
class NetworkSettings:
    input_channels: int = 14
    channels: int = 192
    residual_blocks: int = 16
    se_reduction: int = 8
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 256


@dataclass(frozen=True)
class MCTSSettings:
    simulations: int = 256
    c_base: float = 19652.0
    c_init: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float | None = None


@dataclass(frozen=True)
class SelfPlaySettings:
    games: int = 20
    temperature_moves: int = 16
    actors: int = 1


@dataclass(frozen=True)
class ChallengeSamplingSettings:
    algorithm: str
    early_opening_depths: tuple[int, ...]
    minimum_early_fraction: float
    balance_side_to_move: bool
    balance_action_kind: tuple[str, ...]
    index_path: Path
    index_sha256: str

    def __post_init__(self) -> None:
        if self.algorithm != "stratified-opening-depth-and-action-kind-v1":
            raise ValueError("unsupported training.challengeReplay sampling algorithm")
        if (
            not self.early_opening_depths
            or any(type(value) is not int or value < 0 for value in self.early_opening_depths)
            or len(set(self.early_opening_depths)) != len(self.early_opening_depths)
        ):
            raise ValueError("challenge sampling earlyOpeningDepths are invalid")
        if not 0.0 < self.minimum_early_fraction <= 1.0:
            raise ValueError("challenge sampling minimumEarlyFraction must be in (0, 1]")
        if not self.balance_side_to_move:
            raise ValueError("challenge sampling must balance sideToMove")
        if self.balance_action_kind != ("move", "wall"):
            raise ValueError("challenge sampling action kinds must be move and wall")
        if self.index_path == Path("."):
            raise ValueError("challenge sampling indexPath must name an artifact")
        _optional_sha256(
            self.index_sha256,
            "training.challengeReplay.stratifiedSampling.indexSha256",
        )


@dataclass(frozen=True)
class ChallengeReplaySettings:
    root: Path
    contract_path: Path
    contract_sha256: str
    dataset_fingerprint: str
    dataset_manifest_sha256: str
    train_manifest_sha256: str
    fraction: float
    stratified_sampling: ChallengeSamplingSettings | None = None
    value_target_mode: str = "all-samples"

    def __post_init__(self) -> None:
        if self.root == Path(".") or self.contract_path == Path("."):
            raise ValueError("training.challengeReplay paths must name artifacts")
        for value, label in (
            (self.contract_sha256, "contractSha256"),
            (self.dataset_fingerprint, "datasetFingerprint"),
            (self.dataset_manifest_sha256, "datasetManifestSha256"),
            (self.train_manifest_sha256, "trainManifestSha256"),
        ):
            _optional_sha256(value, f"training.challengeReplay.{label}")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("training.challengeReplay.fraction must be in (0, 1)")
        if self.value_target_mode not in {"all-samples", "self-play-only"}:
            raise ValueError(
                "training.challengeReplay.valueTargetMode must be "
                "all-samples or self-play-only"
            )


@dataclass(frozen=True)
class HealthHoldoutSettings:
    root: Path
    contract_path: Path
    contract_sha256: str
    dataset_fingerprint: str
    dataset_manifest_sha256: str
    train_manifest_sha256: str
    holdout_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.root == Path(".") or self.contract_path == Path("."):
            raise ValueError("training.healthHoldout paths must name artifacts")
        for value, label in (
            (self.contract_sha256, "contractSha256"),
            (self.dataset_fingerprint, "datasetFingerprint"),
            (self.dataset_manifest_sha256, "datasetManifestSha256"),
            (self.train_manifest_sha256, "trainManifestSha256"),
            (self.holdout_manifest_sha256, "holdoutManifestSha256"),
        ):
            _optional_sha256(value, f"training.healthHoldout.{label}")


@dataclass(frozen=True)
class TrainingSettings:
    replay_capacity: int = 200_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    training_steps: int = 200
    max_checkpoints: int = 5
    max_grad_norm: float = 5.0
    use_amp: bool = True
    schedule_steps: int | None = None
    minimum_learning_rate: float = 0.0
    challenge_replay: ChallengeReplaySettings | None = None
    health_holdout: HealthHoldoutSettings | None = None
    learning_stage: str | None = None

    def __post_init__(self) -> None:
        if self.learning_stage not in {
            None,
            "classic-hard-guided",
            "pure-self-play",
        }:
            raise ValueError(
                "training.learningStage must be classic-hard-guided, "
                "pure-self-play, or null"
            )
        if (
            self.learning_stage == "classic-hard-guided"
            and (
                self.challenge_replay is None
                or self.health_holdout is None
            )
        ):
            raise ValueError(
                "classic-hard-guided training requires challengeReplay and healthHoldout"
            )
        if (
            self.learning_stage == "pure-self-play"
            and self.challenge_replay is not None
        ):
            raise ValueError(
                "pure-self-play training must not declare challengeReplay"
            )
        if self.learning_stage == "pure-self-play" and self.health_holdout is None:
            raise ValueError("pure-self-play training requires healthHoldout")
        if self.challenge_replay is not None and self.health_holdout is not None:
            challenge_identity = (
                self.challenge_replay.root,
                self.challenge_replay.contract_path,
                self.challenge_replay.contract_sha256,
                self.challenge_replay.dataset_fingerprint,
                self.challenge_replay.dataset_manifest_sha256,
                self.challenge_replay.train_manifest_sha256,
            )
            holdout_identity = (
                self.health_holdout.root,
                self.health_holdout.contract_path,
                self.health_holdout.contract_sha256,
                self.health_holdout.dataset_fingerprint,
                self.health_holdout.dataset_manifest_sha256,
                self.health_holdout.train_manifest_sha256,
            )
            if challenge_identity != holdout_identity:
                raise ValueError(
                    "challengeReplay and healthHoldout must bind one dataset"
                )


@dataclass(frozen=True)
class ArenaSettings:
    games: int = 40
    simulations: int = 256


@dataclass(frozen=True)
class InitializationSettings:
    mode: str = "random"
    expected_base_model_sha256: str | None = None
    artifact_path: Path | None = None
    artifact_sha256: str | None = None
    teacher_dataset_sha256: str | None = None
    pretraining_report_sha256: str | None = None
    champion_id: str | None = None
    champion_manifest_path: Path | None = None
    champion_manifest_sha256: str | None = None
    champion_manifest_file_sha256: str | None = None
    recovery_manifest_path: Path | None = None
    recovery_manifest_sha256: str | None = None
    recovery_manifest_file_sha256: str | None = None
    recovery_source_sha256: str | None = None
    recovery_config_sha256: str | None = None
    recovery_run_manifest_sha256: str | None = None
    recovery_checkpoint_sha256: str | None = None
    recovery_generation: int | None = None
    recovery_step: int | None = None
    model_state_sha256: str | None = None
    reset_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"random", "model", "champion", "recovery"}:
            raise ValueError(
                "initialization.mode must be random, model, champion, or recovery"
            )
        for value, label in (
            (self.expected_base_model_sha256, "expectedBaseModelSha256"),
            (self.artifact_sha256, "artifactSha256"),
            (self.teacher_dataset_sha256, "teacherDatasetSha256"),
            (self.pretraining_report_sha256, "pretrainingReportSha256"),
            (self.champion_manifest_sha256, "championManifestSha256"),
            (self.champion_manifest_file_sha256, "championManifestFileSha256"),
            (self.recovery_manifest_sha256, "recoveryManifestSha256"),
            (self.recovery_manifest_file_sha256, "recoveryManifestFileSha256"),
            (self.recovery_source_sha256, "recoverySourceSha256"),
            (self.recovery_config_sha256, "recoveryConfigSha256"),
            (self.recovery_run_manifest_sha256, "recoveryRunManifestSha256"),
            (self.recovery_checkpoint_sha256, "recoveryCheckpointSha256"),
            (self.model_state_sha256, "modelStateSha256"),
        ):
            _optional_sha256(value, f"initialization.{label}")
        for value, label in (
            (self.recovery_generation, "recoveryGeneration"),
            (self.recovery_step, "recoveryStep"),
        ):
            if value is not None and (
                type(value) is not int or value < 0
            ):
                raise ValueError(
                    f"initialization.{label} must be a non-negative integer or null"
                )
        if not isinstance(self.reset_optimizer, bool):
            raise ValueError("initialization.resetOptimizer must be boolean")
        for path, label in (
            (self.artifact_path, "artifactPath"),
            (self.champion_manifest_path, "championManifestPath"),
            (self.recovery_manifest_path, "recoveryManifestPath"),
        ):
            if path is not None and path == Path("."):
                raise ValueError(f"initialization.{label} must name a file")
        if self.champion_id is not None and (
            not isinstance(self.champion_id, str) or not self.champion_id.strip()
        ):
            raise ValueError("initialization.championId must be a non-empty string")
        if self.mode == "random":
            if any(
                value is not None
                for value in (
                    self.artifact_path,
                    self.artifact_sha256,
                    self.teacher_dataset_sha256,
                    self.pretraining_report_sha256,
                    self.champion_id,
                    self.champion_manifest_path,
                    self.champion_manifest_sha256,
                    self.champion_manifest_file_sha256,
                    self.recovery_manifest_path,
                    self.recovery_manifest_sha256,
                    self.recovery_manifest_file_sha256,
                    self.recovery_source_sha256,
                    self.recovery_config_sha256,
                    self.recovery_run_manifest_sha256,
                    self.recovery_checkpoint_sha256,
                    self.recovery_generation,
                    self.recovery_step,
                    self.model_state_sha256,
                )
            ):
                raise ValueError(
                    "random initialization must not declare a warm artifact"
                )
        elif self.mode == "model":
            if self.expected_base_model_sha256 is None or any(
                value is None
                for value in (
                    self.artifact_path,
                    self.artifact_sha256,
                    self.teacher_dataset_sha256,
                    self.pretraining_report_sha256,
                )
            ):
                raise ValueError(
                    "model initialization requires base/artifact/teacher/report hashes"
                )
            if any(
                value is not None
                for value in (
                    self.champion_id,
                    self.champion_manifest_path,
                    self.champion_manifest_sha256,
                    self.champion_manifest_file_sha256,
                    self.recovery_manifest_path,
                    self.recovery_manifest_sha256,
                    self.recovery_manifest_file_sha256,
                    self.recovery_source_sha256,
                    self.recovery_config_sha256,
                    self.recovery_run_manifest_sha256,
                    self.recovery_checkpoint_sha256,
                    self.recovery_generation,
                    self.recovery_step,
                    self.model_state_sha256,
                )
            ):
                raise ValueError("warm model initialization must not declare champion provenance")
            if not self.reset_optimizer:
                raise ValueError("warm initialization must reset the optimizer")
        elif self.mode == "champion":
            required = (
                self.expected_base_model_sha256,
                self.artifact_path,
                self.artifact_sha256,
                self.champion_id,
                self.champion_manifest_path,
                self.champion_manifest_sha256,
                self.champion_manifest_file_sha256,
                self.model_state_sha256,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "champion initialization requires base/artifact/model-state/manifest provenance"
                )
            if any(
                value is not None
                for value in (
                    self.teacher_dataset_sha256,
                    self.pretraining_report_sha256,
                    self.recovery_manifest_path,
                    self.recovery_manifest_sha256,
                    self.recovery_manifest_file_sha256,
                    self.recovery_source_sha256,
                    self.recovery_config_sha256,
                    self.recovery_run_manifest_sha256,
                    self.recovery_checkpoint_sha256,
                    self.recovery_generation,
                    self.recovery_step,
                )
            ):
                raise ValueError("champion initialization must not declare teacher provenance")
            if not self.reset_optimizer:
                raise ValueError("champion initialization must reset the optimizer")
        else:
            required = (
                self.expected_base_model_sha256,
                self.artifact_path,
                self.artifact_sha256,
                self.recovery_manifest_path,
                self.recovery_manifest_sha256,
                self.recovery_manifest_file_sha256,
                self.recovery_source_sha256,
                self.recovery_config_sha256,
                self.recovery_run_manifest_sha256,
                self.recovery_checkpoint_sha256,
                self.recovery_generation,
                self.recovery_step,
                self.model_state_sha256,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "recovery initialization requires artifact/archive/source/"
                    "config/generation/step/model-state provenance"
                )
            if any(
                value is not None
                for value in (
                    self.teacher_dataset_sha256,
                    self.pretraining_report_sha256,
                    self.champion_id,
                    self.champion_manifest_path,
                    self.champion_manifest_sha256,
                    self.champion_manifest_file_sha256,
                )
            ):
                raise ValueError(
                    "recovery initialization must not declare teacher or champion provenance"
                )
            # A frozen execution stage may opt into a full-checkpoint
            # continuation. The stage validator, not the generic config
            # parser, owns that narrower authorization boundary.


@dataclass(frozen=True)
class ExperimentConfig:
    pipeline: str
    name: str
    seed: int
    device: str
    run_dir: Path
    artifact_limit_bytes: int
    initialization: InitializationSettings
    network: NetworkSettings
    mcts: MCTSSettings
    self_play: SelfPlaySettings
    training: TrainingSettings
    arena: ArenaSettings

    def to_fingerprint_payload(self) -> dict[str, object]:
        """Preserve hashes for configs created before optional staged fields."""

        payload = asdict(self)
        training = payload["training"]
        initialization = payload["initialization"]
        if not isinstance(training, dict) or not isinstance(initialization, dict):
            raise TypeError("experiment config fingerprint payload is invalid")
        for key in ("health_holdout", "learning_stage"):
            if training.get(key) is None:
                training.pop(key)
        challenge_replay = training.get("challenge_replay")
        if isinstance(challenge_replay, dict) and challenge_replay.get(
            "stratified_sampling"
        ) is None:
            challenge_replay.pop("stratified_sampling", None)
        if (
            isinstance(challenge_replay, dict)
            and challenge_replay.get("value_target_mode") == "all-samples"
        ):
            challenge_replay.pop("value_target_mode", None)
        for key in (
            "recovery_manifest_path",
            "recovery_manifest_sha256",
            "recovery_manifest_file_sha256",
            "recovery_source_sha256",
            "recovery_config_sha256",
            "recovery_run_manifest_sha256",
            "recovery_checkpoint_sha256",
            "recovery_generation",
            "recovery_step",
        ):
            if initialization.get(key) is None:
                initialization.pop(key)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        network = data.get("network", {})
        mcts = data.get("mcts", {})
        self_play = data.get("selfPlay", {})
        training = data.get("training", {})
        arena = data.get("arena", {})
        initialization = data.get("initialization", {})
        if not isinstance(initialization, dict):
            raise ValueError("initialization must be a JSON object")
        challenge_replay = training.get("challengeReplay")
        if challenge_replay is not None and not isinstance(challenge_replay, dict):
            raise ValueError("training.challengeReplay must be a JSON object or null")
        challenge_sampling = (
            None if challenge_replay is None else challenge_replay.get("stratifiedSampling")
        )
        if challenge_sampling is not None and not isinstance(challenge_sampling, dict):
            raise ValueError(
                "training.challengeReplay.stratifiedSampling must be a JSON object or null"
            )
        health_holdout = training.get("healthHoldout")
        if health_holdout is not None and not isinstance(health_holdout, dict):
            raise ValueError("training.healthHoldout must be a JSON object or null")
        name = str(data.get("name", "experiment")).strip()
        if not name:
            raise ValueError("name must not be empty")
        pipeline = str(data.get("pipeline", "reference")).strip()
        if pipeline not in {"reference", "phase4"}:
            raise ValueError("pipeline must be reference or phase4")

        artifact_limit = _positive_int(
            data.get("artifactLimitBytes", DEFAULT_ARTIFACT_LIMIT_BYTES),
            "artifactLimitBytes",
        )
        if artifact_limit > EXTENDED_ARTIFACT_LIMIT_BYTES:
            raise ValueError("artifactLimitBytes must not exceed 16 GiB")
        device = str(data.get("device", "auto"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        seed = int(data.get("seed", 0))
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2^63)")

        return cls(
            pipeline=pipeline,
            name=name,
            seed=seed,
            device=device,
            run_dir=Path(data.get("runDir", f"training/runs/{name}")),
            artifact_limit_bytes=artifact_limit,
            initialization=InitializationSettings(
                mode=str(initialization.get("mode", "random")),
                expected_base_model_sha256=_optional_sha256(
                    initialization.get("expectedBaseModelSha256"),
                    "initialization.expectedBaseModelSha256",
                ),
                artifact_path=(
                    None
                    if initialization.get("artifactPath") is None
                    else Path(initialization["artifactPath"])
                ),
                artifact_sha256=_optional_sha256(
                    initialization.get("artifactSha256"),
                    "initialization.artifactSha256",
                ),
                teacher_dataset_sha256=_optional_sha256(
                    initialization.get("teacherDatasetSha256"),
                    "initialization.teacherDatasetSha256",
                ),
                pretraining_report_sha256=_optional_sha256(
                    initialization.get("pretrainingReportSha256"),
                    "initialization.pretrainingReportSha256",
                ),
                champion_id=(
                    None
                    if initialization.get("championId") is None
                    else str(initialization["championId"])
                ),
                champion_manifest_path=(
                    None
                    if initialization.get("championManifestPath") is None
                    else Path(initialization["championManifestPath"])
                ),
                champion_manifest_sha256=_optional_sha256(
                    initialization.get("championManifestSha256"),
                    "initialization.championManifestSha256",
                ),
                champion_manifest_file_sha256=_optional_sha256(
                    initialization.get("championManifestFileSha256"),
                    "initialization.championManifestFileSha256",
                ),
                recovery_manifest_path=(
                    None
                    if initialization.get("recoveryManifestPath") is None
                    else Path(initialization["recoveryManifestPath"])
                ),
                recovery_manifest_sha256=_optional_sha256(
                    initialization.get("recoveryManifestSha256"),
                    "initialization.recoveryManifestSha256",
                ),
                recovery_manifest_file_sha256=_optional_sha256(
                    initialization.get("recoveryManifestFileSha256"),
                    "initialization.recoveryManifestFileSha256",
                ),
                recovery_source_sha256=_optional_sha256(
                    initialization.get("recoverySourceSha256"),
                    "initialization.recoverySourceSha256",
                ),
                recovery_config_sha256=_optional_sha256(
                    initialization.get("recoveryConfigSha256"),
                    "initialization.recoveryConfigSha256",
                ),
                recovery_run_manifest_sha256=_optional_sha256(
                    initialization.get("recoveryRunManifestSha256"),
                    "initialization.recoveryRunManifestSha256",
                ),
                recovery_checkpoint_sha256=_optional_sha256(
                    initialization.get("recoveryCheckpointSha256"),
                    "initialization.recoveryCheckpointSha256",
                ),
                recovery_generation=(
                    None
                    if initialization.get("recoveryGeneration") is None
                    else initialization["recoveryGeneration"]
                ),
                recovery_step=(
                    None
                    if initialization.get("recoveryStep") is None
                    else initialization["recoveryStep"]
                ),
                model_state_sha256=_optional_sha256(
                    initialization.get("modelStateSha256"),
                    "initialization.modelStateSha256",
                ),
                reset_optimizer=_boolean(
                    initialization.get("resetOptimizer", True),
                    "initialization.resetOptimizer",
                ),
            ),
            network=NetworkSettings(
                input_channels=_positive_int(
                    network.get("inputChannels", 14),
                    "network.inputChannels",
                ),
                channels=_positive_int(network.get("channels", 192), "network.channels"),
                residual_blocks=_positive_int(
                    network.get("residualBlocks", 16),
                    "network.residualBlocks",
                ),
                se_reduction=_positive_int(
                    network.get("seReduction", 8),
                    "network.seReduction",
                ),
                policy_channels=_positive_int(
                    network.get("policyChannels", 32),
                    "network.policyChannels",
                ),
                value_channels=_positive_int(
                    network.get("valueChannels", 32),
                    "network.valueChannels",
                ),
                value_hidden=_positive_int(
                    network.get("valueHidden", 256),
                    "network.valueHidden",
                ),
            ),
            mcts=MCTSSettings(
                simulations=_positive_int(
                    mcts.get("simulations", 256),
                    "mcts.simulations",
                ),
                c_base=_non_negative_float(mcts.get("cBase", 19652), "mcts.cBase"),
                c_init=_non_negative_float(mcts.get("cInit", 1.25), "mcts.cInit"),
                dirichlet_epsilon=_non_negative_float(
                    mcts.get("dirichletEpsilon", 0.25),
                    "mcts.dirichletEpsilon",
                ),
                dirichlet_alpha=(
                    None
                    if mcts.get("dirichletAlpha") is None
                    else _non_negative_float(
                        mcts["dirichletAlpha"],
                        "mcts.dirichletAlpha",
                    )
                ),
            ),
            self_play=SelfPlaySettings(
                games=_positive_int(self_play.get("games", 20), "selfPlay.games"),
                temperature_moves=max(
                    0,
                    int(self_play.get("temperatureMoves", 16)),
                ),
                actors=_positive_int(
                    self_play.get("actors", 1),
                    "selfPlay.actors",
                ),
            ),
            training=TrainingSettings(
                replay_capacity=_positive_int(
                    training.get("replayCapacity", 200_000),
                    "training.replayCapacity",
                ),
                batch_size=_positive_int(
                    training.get("batchSize", 256),
                    "training.batchSize",
                ),
                learning_rate=_non_negative_float(
                    training.get("learningRate", 1e-3),
                    "training.learningRate",
                ),
                weight_decay=_non_negative_float(
                    training.get("weightDecay", 1e-4),
                    "training.weightDecay",
                ),
                training_steps=_positive_int(
                    training.get("trainingSteps", 200),
                    "training.trainingSteps",
                ),
                max_checkpoints=_positive_int(
                    training.get("maxCheckpoints", 5),
                    "training.maxCheckpoints",
                ),
                max_grad_norm=_positive_float(
                    training.get("maxGradNorm", 5.0),
                    "training.maxGradNorm",
                ),
                use_amp=_boolean(
                    training.get("useAmp", True),
                    "training.useAmp",
                ),
                schedule_steps=(
                    None
                    if training.get("scheduleSteps") is None
                    else _positive_int(
                        training["scheduleSteps"],
                        "training.scheduleSteps",
                    )
                ),
                minimum_learning_rate=_non_negative_float(
                    training.get("minimumLearningRate", 0.0),
                    "training.minimumLearningRate",
                ),
                challenge_replay=(
                    None
                    if challenge_replay is None
                    else ChallengeReplaySettings(
                        root=Path(challenge_replay["root"]),
                        contract_path=Path(challenge_replay["contractPath"]),
                        contract_sha256=_optional_sha256(
                            challenge_replay.get("contractSha256"),
                            "training.challengeReplay.contractSha256",
                        )
                        or "",
                        dataset_fingerprint=_optional_sha256(
                            challenge_replay.get("datasetFingerprint"),
                            "training.challengeReplay.datasetFingerprint",
                        )
                        or "",
                        dataset_manifest_sha256=_optional_sha256(
                            challenge_replay.get("datasetManifestSha256"),
                            "training.challengeReplay.datasetManifestSha256",
                        )
                        or "",
                        train_manifest_sha256=_optional_sha256(
                            challenge_replay.get("trainManifestSha256"),
                            "training.challengeReplay.trainManifestSha256",
                        )
                        or "",
                        fraction=_positive_float(
                            challenge_replay.get("fraction"),
                            "training.challengeReplay.fraction",
                        ),
                        value_target_mode=str(
                            challenge_replay.get(
                                "valueTargetMode", "all-samples"
                            )
                        ),
                        stratified_sampling=(
                            None
                            if challenge_sampling is None
                            else ChallengeSamplingSettings(
                                algorithm=str(challenge_sampling.get("algorithm", "")),
                                early_opening_depths=tuple(
                                    challenge_sampling.get("earlyOpeningDepths", ())
                                ),
                                minimum_early_fraction=_positive_float(
                                    challenge_sampling.get("minimumEarlyFraction"),
                                    "training.challengeReplay.stratifiedSampling.minimumEarlyFraction",
                                ),
                                balance_side_to_move=_boolean(
                                    challenge_sampling.get("balanceSideToMove"),
                                    "training.challengeReplay.stratifiedSampling.balanceSideToMove",
                                ),
                                balance_action_kind=tuple(
                                    challenge_sampling.get("balanceActionKind", ())
                                ),
                                index_path=Path(challenge_sampling.get("indexPath", ".")),
                                index_sha256=_optional_sha256(
                                    challenge_sampling.get("indexSha256"),
                                    "training.challengeReplay.stratifiedSampling.indexSha256",
                                )
                                or "",
                            )
                        ),
                    )
                ),
                health_holdout=(
                    None
                    if health_holdout is None
                    else HealthHoldoutSettings(
                        root=Path(health_holdout["root"]),
                        contract_path=Path(health_holdout["contractPath"]),
                        contract_sha256=_optional_sha256(
                            health_holdout.get("contractSha256"),
                            "training.healthHoldout.contractSha256",
                        )
                        or "",
                        dataset_fingerprint=_optional_sha256(
                            health_holdout.get("datasetFingerprint"),
                            "training.healthHoldout.datasetFingerprint",
                        )
                        or "",
                        dataset_manifest_sha256=_optional_sha256(
                            health_holdout.get("datasetManifestSha256"),
                            "training.healthHoldout.datasetManifestSha256",
                        )
                        or "",
                        train_manifest_sha256=_optional_sha256(
                            health_holdout.get("trainManifestSha256"),
                            "training.healthHoldout.trainManifestSha256",
                        )
                        or "",
                        holdout_manifest_sha256=_optional_sha256(
                            health_holdout.get("holdoutManifestSha256"),
                            "training.healthHoldout.holdoutManifestSha256",
                        )
                        or "",
                    )
                ),
                learning_stage=(
                    None
                    if training.get("learningStage") is None
                    else str(training["learningStage"])
                ),
            ),
            arena=ArenaSettings(
                games=_positive_int(arena.get("games", 40), "arena.games"),
                simulations=_positive_int(
                    arena.get("simulations", 256),
                    "arena.simulations",
                ),
            ),
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("experiment config must contain a JSON object")
    return ExperimentConfig.from_dict(data)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
