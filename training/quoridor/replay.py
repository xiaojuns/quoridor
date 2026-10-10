from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .storage import atomic_torch_save


ACTION_SIZE = 209
REPLAY_VERSION = "replay-v1"
SAMPLE_VERSION = "replay-sample-v1"
DEFAULT_REPLAY_CAPACITY = 100_000


@dataclass(frozen=True)
class TrainingSample:
    state: torch.Tensor
    policy: torch.Tensor
    value: float

    def normalized(self) -> "TrainingSample":
        if not isinstance(self.state, torch.Tensor):
            raise TypeError("sample state must be a tensor")
        if not isinstance(self.policy, torch.Tensor):
            raise TypeError("sample policy must be a tensor")
        state = (
            self.state.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        policy = (
            self.policy.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        if state.ndim != 3 or state.shape[-2:] != (9, 9):
            raise ValueError("sample state must have shape [channels, 9, 9]")
        if policy.shape != (ACTION_SIZE,):
            raise ValueError(f"sample policy must have shape [{ACTION_SIZE}]")
        if not bool(torch.isfinite(state).all().item()):
            raise ValueError("sample state must contain only finite values")
        if not bool(torch.isfinite(policy).all().item()):
            raise ValueError("sample policy must contain only finite values")
        if bool((policy < 0).any().item()):
            raise ValueError("sample policy must be non-negative")
        policy_sum = float(policy.sum().item())
        if policy_sum <= 0:
            raise ValueError("sample policy must contain positive probability mass")
        policy = policy / policy_sum
        value = float(self.value)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("sample value must be finite and in [-1, 1]")
        return TrainingSample(state=state, policy=policy, value=value)


@dataclass(frozen=True, slots=True)
class ReplayPosition:
    """One exact root position and its raw MCTS target.

    The complete :class:`~quoridor.episode.Episode` is retained here rather
    than reducing a position to a board key.  M4 replay therefore keeps the
    history needed by repetition adjudication and can re-encode the position
    without importing a model or a legacy game implementation.
    """

    episode: "Episode"
    action: int
    policy: tuple[float, ...]
    visits: tuple[int, ...] = ()
    root_value: float = 0.0

    def __post_init__(self) -> None:
        from .episode import Episode

        if not isinstance(self.episode, Episode) or self.episode.finished:
            raise ValueError("replay positions require a live Episode root")
        if type(self.action) is not int or not 0 <= self.action < ACTION_SIZE:
            raise ValueError("replay action is outside the action space")
        from . import engine

        if self.action not in engine.legal_actions(self.episode.state):
            raise ValueError("replay action must be legal at its root episode")
        policy = tuple(float(item) for item in self.policy)
        if len(policy) != ACTION_SIZE or any(
            not math.isfinite(item) or item < 0 for item in policy
        ):
            raise ValueError("replay policy must contain 209 finite non-negative values")
        total = math.fsum(policy)
        if total <= 0 or not math.isfinite(total):
            raise ValueError("replay policy must contain positive mass")
        if not 0 <= self.action < ACTION_SIZE or policy[self.action] <= 0:
            raise ValueError("selected replay action must have positive target mass")
        visits = tuple(self.visits)
        if visits and (
            len(visits) != ACTION_SIZE
            or any(type(item) is not int or item < 0 for item in visits)
            or visits[self.action] <= 0
        ):
            raise ValueError("replay visits must be 209 non-negative counts")
        value = float(self.root_value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("replay root value must be finite and in [-1,1]")
        object.__setattr__(self, "policy", tuple(item / total for item in policy))
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "root_value", value)

    @property
    def state(self):
        return self.episode.state

    @property
    def history(self) -> tuple[str, ...]:
        return self.episode.history

    @property
    def to_play(self) -> int:
        return self.episode.state.to_play

    @property
    def policy_target(self) -> tuple[float, ...]:
        return self.policy


@dataclass(frozen=True, slots=True)
class ReplayGame:
    """Immutable self-play game record submitted to :class:`ReplayBuffer`."""

    episode: "Episode"
    positions: tuple[ReplayPosition, ...]
    seed: int | None = None
    model_hash: str | None = None
    config_hash: str | None = None
    version: str = REPLAY_VERSION

    def __post_init__(self) -> None:
        from .episode import Episode

        if not isinstance(self.episode, Episode):
            raise TypeError("replay game requires a quoridor-v1 Episode")
        positions = tuple(self.positions)
        if any(not isinstance(item, ReplayPosition) for item in positions):
            raise TypeError("replay game positions must be ReplayPosition records")
        start_plies = positions[0].episode.plies if positions else self.episode.plies
        if tuple(item.episode.plies for item in positions) != tuple(range(start_plies, self.episode.plies)):
            raise ValueError("replay game must retain every actor root in ply order")
        for item in positions:
            if item.episode.max_game_plies != self.episode.max_game_plies:
                raise ValueError("position and final episode ply limits differ")
            if item.episode.plies >= self.episode.plies:
                raise ValueError("replay position must precede the final episode")
            if item.history != self.episode.history[: item.episode.plies + 1]:
                raise ValueError("replay position history must be an exact final-episode prefix")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("replay seed must be a non-negative integer")
        for name, value in (("model_hash", self.model_hash), ("config_hash", self.config_hash)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string")
        if self.version != REPLAY_VERSION:
            raise ValueError("unsupported replay version")
        object.__setattr__(self, "positions", positions)

    @property
    def eligible(self) -> bool:
        return bool(self.episode.outcome is not None and self.episode.outcome.replay_eligible)

    @property
    def final_episode(self):
        """Explicit alias used by actor/checkpoint integrations."""

        return self.episode


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """Encoded, immutable training record with current-player value target.

    ``features``, ``legal_mask``, ``policy`` and ``action`` use the encoder's
    canonical coordinates. ``raw_policy`` preserves the unmodified original
    root-visits target for lineage/audit; it is never temperature-smoothed.
    """

    features: torch.Tensor
    legal_mask: torch.Tensor
    policy: torch.Tensor
    to_play: int
    value: float
    action: int
    episode_plies: int
    history: tuple[str, ...]
    version: str = SAMPLE_VERSION
    raw_policy: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        features = self.features.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        legal = self.legal_mask.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()
        policy = self.policy.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if features.shape != (16, 9, 9) or not bool(torch.isfinite(features).all().item()):
            raise ValueError("replay features must be finite float32 [16,9,9]")
        if legal.shape != (ACTION_SIZE,):
            raise ValueError("replay legal_mask must be bool [209]")
        if policy.shape != (ACTION_SIZE,) or not bool(torch.isfinite(policy).all().item()):
            raise ValueError("replay policy must be finite float32 [209]")
        if bool((policy < 0).any().item()) or float(policy.sum()) <= 0:
            raise ValueError("replay policy must be non-negative with positive mass")
        if type(self.to_play) is not int or self.to_play not in (0, 1):
            raise ValueError("replay to_play must be 0 or 1")
        if type(self.action) is not int or not 0 <= self.action < ACTION_SIZE:
            raise ValueError("replay action is outside the action space")
        value = float(self.value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("replay value must be finite and in [-1,1]")
        if type(self.episode_plies) is not int or self.episode_plies < 0:
            raise ValueError("replay episode_plies must be non-negative")
        if type(self.history) is not tuple or not self.history or any(
            not isinstance(key, str) or not key for key in self.history
        ):
            raise ValueError("replay history must be an immutable non-empty tuple")
        if not bool(legal[self.action].item()) or bool((policy[~legal] > 0).any().item()):
            raise ValueError("replay policy must have support only on legal actions")
        if self.version != SAMPLE_VERSION:
            raise ValueError("unsupported replay sample version")
        raw_policy = self.raw_policy
        if raw_policy is not None:
            raw_policy = tuple(float(item) for item in raw_policy)
            if len(raw_policy) != ACTION_SIZE or any(
                not math.isfinite(item) or item < 0 for item in raw_policy
            ) or math.fsum(raw_policy) <= 0:
                raise ValueError("raw replay policy must contain 209 finite non-negative values")
            raw_policy = tuple(item / math.fsum(raw_policy) for item in raw_policy)
        policy = policy / policy.sum()
        features.requires_grad_(False)
        legal.requires_grad_(False)
        policy.requires_grad_(False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "legal_mask", legal)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "raw_policy", raw_policy)

    @property
    def state(self) -> torch.Tensor:
        """Compatibility alias used by historical trainer code."""

        return self.features

    @property
    def policy_target(self) -> torch.Tensor:
        return self.policy


def _clone_replay_sample(sample: ReplaySample) -> ReplaySample:
    """Copy tensor storage at the replay boundary so records stay immutable."""

    return ReplaySample(
        features=sample.features.clone(),
        legal_mask=sample.legal_mask.clone(),
        policy=sample.policy.clone(),
        to_play=sample.to_play,
        value=sample.value,
        action=sample.action,
        episode_plies=sample.episode_plies,
        history=sample.history,
        version=sample.version,
        raw_policy=sample.raw_policy,
    )


class ReplayBuffer:
    """Bounded oldest-first replay with whole-game admission.

    ``add`` retains the historical single-sample API. New M4 callers should
    submit a :class:`ReplayGame` using :meth:`add_game`; a truncated or failed
    game is counted and contributes zero positions. Admission is deliberately
    atomic, so a game is never partially represented in replay.
    """

    def __init__(self, capacity: int = DEFAULT_REPLAY_CAPACITY, *, seed: int = 0) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(seed) is not int:
            raise ValueError("replay seed must be an integer")
        self.capacity = capacity
        self.version = REPLAY_VERSION
        self._samples: deque[TrainingSample | ReplaySample] = deque(maxlen=self.capacity)
        self._random = random.Random(seed)
        self._seed = seed
        self._accepted_games = 0
        self._rejected_games = 0
        self._truncated_games = 0
        self._failed_games = 0
        self._rejected_positions = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[TrainingSample | ReplaySample, ...]:
        """Read-only insertion-order snapshot for audits and checkpoints."""

        return tuple(
            _clone_replay_sample(sample) if isinstance(sample, ReplaySample) else sample
            for sample in self._samples
        )

    @property
    def accepted_games(self) -> int:
        return self._accepted_games

    @property
    def rejected_games(self) -> int:
        return self._rejected_games

    @property
    def truncated_games(self) -> int:
        return self._truncated_games

    @property
    def failed_games(self) -> int:
        return self._failed_games

    @property
    def rejected_positions(self) -> int:
        return self._rejected_positions

    def add(self, sample: TrainingSample | ReplaySample) -> None:
        """Add one already encoded sample (legacy compatibility path)."""

        if isinstance(sample, TrainingSample):
            self._samples.append(sample.normalized())
        elif isinstance(sample, ReplaySample):
            self._samples.append(_clone_replay_sample(sample))
        else:
            raise TypeError("replay add expects TrainingSample or ReplaySample")

    def add_game(self, game: ReplayGame) -> bool:
        """Admit all positions from a complete game, or none from a stop.

        Returns ``True`` only for a terminal goal/repetition episode. Runtime
        truncation and failures update counters and return ``False``. Labels
        are generated from each recorded player's perspective, never from ply
        parity or a search-local approximation.
        """

        if not isinstance(game, ReplayGame) and callable(getattr(game, "replay_game", None)):
            game = game.replay_game()
        if not isinstance(game, ReplayGame):
            raise TypeError("replay add_game expects ReplayGame or a v1 self-play result")
        outcome = game.episode.outcome
        if outcome is None or not outcome.replay_eligible:
            self._rejected_games += 1
            if outcome is not None and outcome.truncated:
                self._truncated_games += 1
            elif outcome is not None and outcome.status == "failed":
                self._failed_games += 1
            self._rejected_positions += len(game.positions)
            return False
        if not game.positions:
            self._rejected_games += 1
            self._rejected_positions += 0
            return False
        from .encoder import encode_episode

        values = game.episode.value_targets(item.to_play for item in game.positions)
        if values is None or len(values) != len(game.positions):
            raise ValueError("terminal replay game did not produce complete value labels")
        pending: list[ReplaySample] = []
        for position, value in zip(game.positions, values):
            encoded = encode_episode(position.episode)
            # Features and legality are canonicalized by the v1 encoder.  Keep
            # the raw MCTS target on ReplayPosition, then apply the exact same
            # reversible action transform for tensor training coordinates.
            canonical_policy = encoded.transform.policy(position.policy)
            pending.append(ReplaySample(
                features=encoded.features,
                legal_mask=encoded.legal_mask,
                policy=torch.tensor(canonical_policy, dtype=torch.float32),
                to_play=position.to_play,
                value=value,
                action=encoded.transform.action(position.action),
                episode_plies=position.episode.plies,
                history=position.history,
                raw_policy=position.policy,
            ))
        for sample in pending:
            self._samples.append(sample)
        self._accepted_games += 1
        return True

    def extend_games(self, games: Iterable[ReplayGame]) -> int:
        accepted = 0
        for game in games:
            accepted += int(self.add_game(game))
        return accepted

    # Explicit aliases make the whole-game boundary hard to bypass in callers
    # that use ``append``/``add_episode`` terminology.
    add_episode = add_game
    append_game = add_game

    def extend(self, samples: Iterable[TrainingSample | ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not self._samples:
            raise ValueError("cannot sample an empty replay buffer")
        count = min(batch_size, len(self._samples))
        selected = self._random.sample(list(self._samples), count)
        states = torch.stack([sample.state for sample in selected]).to(device)
        policies = torch.stack([sample.policy for sample in selected]).to(device)
        values = torch.tensor([sample.value for sample in selected], dtype=torch.float32, device=device)
        # Legacy samples have no explicit legality/side metadata. Keep the
        # old three keys usable while exposing the strict M4 batch contract.
        legal_masks = torch.stack([
            sample.legal_mask if isinstance(sample, ReplaySample)
            else torch.ones(ACTION_SIZE, dtype=torch.bool)
            for sample in selected
        ]).to(device)
        to_play = torch.tensor([
            sample.to_play if isinstance(sample, ReplaySample) else 0 for sample in selected
        ], dtype=torch.int64, device=device)
        actions = torch.tensor([
            sample.action if isinstance(sample, ReplaySample) else -1 for sample in selected
        ], dtype=torch.int64, device=device)
        raw_policies = torch.stack([
            sample.policy if not isinstance(sample, ReplaySample) or sample.raw_policy is None
            else torch.tensor(sample.raw_policy, dtype=torch.float32)
            for sample in selected
        ]).to(device)
        return {
            "states": states,
            "features": states,
            "legal_masks": legal_masks,
            "policies": policies,
            "values": values,
            "to_play": to_play,
            "actions": actions,
            "raw_policies": raw_policies,
        }

    def stats(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "capacity": self.capacity,
            "positions": len(self),
            "accepted_games": self._accepted_games,
            "rejected_games": self._rejected_games,
            "truncated_games": self._truncated_games,
            "failed_games": self._failed_games,
            "rejected_positions": self._rejected_positions,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.version,
            "capacity": self.capacity,
            "seed": self._seed,
            "random_state": self._random.getstate(),
            "accepted_games": self._accepted_games,
            "rejected_games": self._rejected_games,
            "truncated_games": self._truncated_games,
            "failed_games": self._failed_games,
            "rejected_positions": self._rejected_positions,
            "samples": list(self._samples),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("replay state must be a mapping")
        if state.get("version", REPLAY_VERSION) != REPLAY_VERSION:
            raise ValueError("unsupported replay state version")
        if type(state.get("capacity")) is not int or state["capacity"] != self.capacity:
            raise ValueError("replay capacity does not match the resume destination")
        samples = state.get("samples", [])
        if not isinstance(samples, (list, tuple)):
            raise TypeError("replay samples must be a sequence")
        counters = ("accepted_games", "rejected_games", "truncated_games", "failed_games", "rejected_positions")
        for field in counters:
            if type(state.get(field, 0)) is not int or state.get(field, 0) < 0:
                raise ValueError(f"replay counter {field} must be non-negative")
        self._samples = deque(maxlen=self.capacity)
        for sample in samples[-self.capacity :]:
            if isinstance(sample, ReplaySample):
                self.add(sample)
            elif isinstance(sample, TrainingSample):
                self.add(sample)
            elif isinstance(sample, dict):
                self.add(TrainingSample(sample["state"], sample["policy"], sample["value"]))
            else:
                raise TypeError("invalid replay sample in state")
        if "random_state" in state:
            self._random.setstate(state["random_state"])
        for field in ("accepted_games", "rejected_games", "truncated_games", "failed_games", "rejected_positions"):
            setattr(self, f"_{field}", int(state.get(field, 0)))

    def save(self, path: str | Path, *, artifact_root: str | Path, limit_bytes: int) -> Path:
        return atomic_torch_save(self.state_dict(), path, artifact_root=artifact_root, limit_bytes=limit_bytes)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ReplayBuffer":
        state = torch.load(path, map_location=map_location, weights_only=False)
        replay = cls(int(state["capacity"]), seed=int(state.get("seed", 0)))
        replay.load_state_dict(state)
        return replay


__all__ = [
    "ACTION_SIZE", "REPLAY_VERSION", "SAMPLE_VERSION", "DEFAULT_REPLAY_CAPACITY",
    "TrainingSample", "ReplayPosition", "ReplayGame", "ReplaySample", "ReplayBuffer",
]
