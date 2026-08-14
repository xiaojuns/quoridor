"""Deterministic stratified sampling for classic-guidance replay."""

from __future__ import annotations

import math
import random
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .codec import HORIZONTAL_WALL_ACTION_OFFSET
from .config import ChallengeSamplingSettings
from .replay_store import ReplayStore
from .replay_tensor_cache import ReplayTensorCache, ReplayTensorSamplePlan
from .run_manifest import canonical_json_bytes, sha256_bytes


class ChallengeSamplerError(RuntimeError):
    """Raised when guidance data cannot satisfy the frozen sampling contract."""


@dataclass(frozen=True, slots=True)
class StratifiedChallengeSample:
    plan: ReplayTensorSamplePlan
    summary: Mapping[str, object]


def _balanced_quotas(total: int, labels: tuple[tuple[int, str], ...]) -> dict[tuple[int, str], int]:
    base, remainder = divmod(total, len(labels))
    return {
        label: base + (1 if ordinal < remainder else 0)
        for ordinal, label in enumerate(labels)
    }


class StratifiedChallengeSampler:
    """Index raw guidance metadata while loading tensors from the existing cache."""

    def __init__(
        self,
        replay: ReplayStore,
        cache: ReplayTensorCache,
        settings: ChallengeSamplingSettings,
        *,
        index_payload: Mapping[str, object] | None = None,
    ) -> None:
        if cache.replay is not replay:
            raise ValueError("challenge sampler cache belongs to another replay store")
        self.replay = replay
        self.cache = cache
        self.settings = settings
        self.labels = tuple(
            (side, action_kind)
            for side in (0, 1)
            for action_kind in settings.balance_action_kind
        )
        pools: dict[tuple[bool, int, str], list[int]] = {
            (early, side, kind): []
            for early in (False, True)
            for side, kind in self.labels
        }
        depth_counts: Counter[int] = Counter()
        action_counts: Counter[str] = Counter()
        side_counts: Counter[int] = Counter()
        global_index = 0
        early_depths = frozenset(settings.early_opening_depths)
        if index_payload is None:
            for game in replay.iter_games():
                for ply in game.plies:
                    depth = ply.metadata.get("openingDepth")
                    side = ply.metadata.get("sideToMove")
                    if type(depth) is not int or depth < 0 or side not in (0, 1):
                        raise ChallengeSamplerError(
                            "guidance ply lacks valid openingDepth/sideToMove metadata"
                        )
                    kind = (
                        "move"
                        if ply.action_index < HORIZONTAL_WALL_ACTION_OFFSET
                        else "wall"
                    )
                    if kind not in settings.balance_action_kind:
                        raise ChallengeSamplerError("guidance action kind is unsupported")
                    pools[(depth in early_depths, int(side), kind)].append(global_index)
                    depth_counts[depth] += 1
                    side_counts[int(side)] += 1
                    action_counts[kind] += 1
                    global_index += 1
        else:
            raw_pools = index_payload.get("pools")
            inventory = index_payload.get("inventory")
            if not isinstance(raw_pools, Mapping) or not isinstance(inventory, Mapping):
                raise ChallengeSamplerError("guidance sampling index schema is invalid")
            for early in (False, True):
                for side, kind in self.labels:
                    key = f"{'early' if early else 'other'}:{side}:{kind}"
                    values = raw_pools.get(key)
                    if not isinstance(values, list) or any(type(v) is not int for v in values):
                        raise ChallengeSamplerError("guidance sampling pool is invalid")
                    pools[(early, side, kind)] = values
            global_index = int(inventory.get("positions", -1))
            for key, value in dict(inventory.get("openingDepth", {})).items():
                depth_counts[int(key)] = int(value)
            for key, value in dict(inventory.get("sideToMove", {})).items():
                side_counts[int(key)] = int(value)
            for key, value in dict(inventory.get("actionKind", {})).items():
                action_counts[str(key)] = int(value)
        if global_index != replay.ply_count:
            raise ChallengeSamplerError("guidance index differs from replay ply count")
        if any(not pool for pool in pools.values()):
            raise ChallengeSamplerError("guidance data has an empty required stratum")
        self.pools = {key: tuple(values) for key, values in pools.items()}
        self.cache_snapshot = cache.snapshot(deep=False)
        content_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "algorithm": settings.algorithm,
                    "rawReplayManifestSha256": self.cache_snapshot[
                        "rawManifestSha256"
                    ],
                    "positions": global_index,
                    "pools": {
                        f"{'early' if early else 'other'}:{side}:{kind}": {
                            "count": len(values),
                            "indicesSha256": sha256_bytes(
                                canonical_json_bytes(list(values))
                            ),
                        }
                        for (early, side, kind), values in sorted(self.pools.items())
                    },
                }
            )
        )
        self.inventory = {
            "positions": global_index,
            "openingDepth": {
                str(key): value for key, value in sorted(depth_counts.items())
            },
            "sideToMove": {
                str(key): value for key, value in sorted(side_counts.items())
            },
            "actionKind": dict(sorted(action_counts.items())),
            "strata": {
                f"{'early' if early else 'other'}:{side}:{kind}": len(values)
                for (early, side, kind), values in sorted(self.pools.items())
            },
            "contentSha256": content_sha256,
        }
        self.index_payload = {
            "schemaVersion": 1,
            "kind": "phase4f35-stratified-guidance-index",
            "algorithm": settings.algorithm,
            "rawManifestSha256": self.cache_snapshot["rawManifestSha256"],
            "rawWindowSha256": self.cache_snapshot["rawWindowSha256"],
            "inventory": self.inventory,
            "pools": {
                f"{'early' if early else 'other'}:{side}:{kind}": list(values)
                for (early, side, kind), values in sorted(self.pools.items())
            },
        }
        self.index_sha256 = sha256_bytes(canonical_json_bytes(self.index_payload))
        self.index_payload = {**self.index_payload, "indexSha256": self.index_sha256}
        if index_payload is not None:
            supplied = dict(index_payload)
            if supplied != self.index_payload:
                raise ChallengeSamplerError("guidance sampling index content differs")

    def sample_plan(
        self,
        count: int,
        *,
        rng: random.Random,
        mirror_probability: float = 0.5,
    ) -> StratifiedChallengeSample:
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        if not 0.0 <= mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0, 1]")
        total_quotas = _balanced_quotas(count, self.labels)
        early_total = math.ceil(count * self.settings.minimum_early_fraction)
        early_quotas = _balanced_quotas(early_total, self.labels)
        for label in self.labels:
            early_quotas[label] = min(early_quotas[label], total_quotas[label])
        shortfall = early_total - sum(early_quotas.values())
        for label in self.labels:
            available = total_quotas[label] - early_quotas[label]
            take = min(shortfall, available)
            early_quotas[label] += take
            shortfall -= take
        if shortfall:
            raise ChallengeSamplerError("cannot allocate the required early quota")

        selected: list[int] = []
        strata_counts: dict[str, int] = {}
        for label in self.labels:
            side, kind = label
            for early, quota in (
                (True, early_quotas[label]),
                (False, total_quotas[label] - early_quotas[label]),
            ):
                pool = self.pools[(early, side, kind)]
                if quota > len(pool):
                    raise ChallengeSamplerError(
                        f"guidance stratum cannot satisfy quota: {early}:{side}:{kind}"
                    )
                selected.extend(rng.sample(pool, quota))
                strata_counts[
                    f"{'early' if early else 'other'}:{side}:{kind}"
                ] = quota
        rng.shuffle(selected)
        mirrored = tuple(rng.random() < mirror_probability for _ in selected)
        plan = ReplayTensorSamplePlan(
            manifest_sha256=str(self.cache_snapshot["manifestSha256"]),
            raw_window_sha256=str(self.cache_snapshot["rawWindowSha256"]),
            total_positions=int(self.cache_snapshot["plies"]),
            global_indices=tuple(selected),
            mirrored=mirrored,
        )
        side_totals = {
            str(side): sum(total_quotas[(side, kind)] for kind in ("move", "wall"))
            for side in (0, 1)
        }
        action_totals = {
            kind: sum(total_quotas[(side, kind)] for side in (0, 1))
            for kind in ("move", "wall")
        }
        return StratifiedChallengeSample(
            plan=plan,
            summary={
                "algorithm": self.settings.algorithm,
                "indexSha256": self.index_sha256,
                "positions": count,
                "earlyPositions": early_total,
                "earlyFraction": early_total / count,
                "sideToMove": side_totals,
                "actionKind": action_totals,
                "strata": strata_counts,
            },
        )

    def load_batch(
        self,
        count: int,
        *,
        rng: random.Random,
        mirror_probability: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], Mapping[str, object]]:
        sampled = self.sample_plan(
            count,
            rng=rng,
            mirror_probability=mirror_probability,
        )
        return self.cache.load_plan(sampled.plan, device=device), sampled.summary


def load_stratified_challenge_sampler(
    workspace: str | Path,
    replay: ReplayStore,
    cache: ReplayTensorCache,
    settings: ChallengeSamplingSettings,
) -> StratifiedChallengeSampler:
    root = Path(workspace).resolve()
    path = settings.index_path
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ChallengeSamplerError("guidance sampling index is outside workspace") from error
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChallengeSamplerError("guidance sampling index is unreadable") from error
    if (
        path.is_symlink()
        or not isinstance(payload, Mapping)
        or canonical_json_bytes(payload) != raw
    ):
        raise ChallengeSamplerError("guidance sampling index is not canonical")
    core = dict(payload)
    stored = core.pop("indexSha256", None)
    if stored != settings.index_sha256 or stored != sha256_bytes(canonical_json_bytes(core)):
        raise ChallengeSamplerError("guidance sampling index self hash mismatch")
    snapshot = cache.snapshot(deep=False)
    if (
        payload.get("algorithm") != settings.algorithm
        or payload.get("rawManifestSha256") != snapshot["rawManifestSha256"]
        or payload.get("rawWindowSha256") != snapshot["rawWindowSha256"]
    ):
        raise ChallengeSamplerError("guidance sampling index binds another replay")
    return StratifiedChallengeSampler(replay, cache, settings, index_payload=payload)
