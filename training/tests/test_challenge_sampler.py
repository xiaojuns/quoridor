from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.challenge_sampler import (
    ChallengeSamplerError,
    StratifiedChallengeSampler,
    load_stratified_challenge_sampler,
)
from quoridor.codec import ACTION_SPACE_SIZE, HORIZONTAL_WALL_ACTION_OFFSET, encode_move, position_hash
from quoridor.config import ChallengeSamplingSettings
from quoridor.game import State
from quoridor.replay_store import ReplayGame, ReplayPly
from quoridor.run_manifest import atomic_replace_bytes, canonical_json_bytes


class _Replay:
    def __init__(self, games: list[ReplayGame]) -> None:
        self.games = tuple(games)
        self.ply_count = sum(len(game.plies) for game in games)

    def iter_games(self):
        return iter(self.games)


class _Cache:
    def __init__(self, replay: _Replay) -> None:
        self.replay = replay

    def snapshot(self, *, deep: bool = False):
        return {
            "manifestSha256": "1" * 64,
            "rawManifestSha256": "2" * 64,
            "rawWindowSha256": "3" * 64,
            "plies": self.replay.ply_count,
        }


def _game(index: int, *, early: bool, side: int, wall: bool) -> ReplayGame:
    state = State(
        pawns=((8, 4), (0, 4)), walls_remaining=(10, 10),
        current_player=0, move_number=0, position_counts={},
    )
    state = replace(state, position_counts={position_hash(state): 1})
    action = HORIZONTAL_WALL_ACTION_OFFSET if wall else encode_move(7, 4)
    counts = [0] * ACTION_SPACE_SIZE
    counts[action] = 16
    return ReplayGame(
        game_id=f"g-{index}", initial_state=state,
        plies=(ReplayPly(action, tuple(counts), 0.0, 0, "teacher", {
            "openingDepth": 8 if early else 16,
            "sideToMove": side,
        }),),
    )


def _settings(path: Path, digest: str = "0" * 64) -> ChallengeSamplingSettings:
    return ChallengeSamplingSettings(
        algorithm="stratified-opening-depth-and-action-kind-v1",
        early_opening_depths=(8, 12), minimum_early_fraction=0.6,
        balance_side_to_move=True, balance_action_kind=("move", "wall"),
        index_path=path, index_sha256=digest,
    )


class StratifiedChallengeSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        games = []
        for early in (False, True):
            for side in (0, 1):
                for wall in (False, True):
                    for _ in range(20):
                        games.append(_game(len(games), early=early, side=side, wall=wall))
        self.replay = _Replay(games)
        self.cache = _Cache(self.replay)

    def test_plan_is_deterministic_balanced_and_without_replacement(self) -> None:
        sampler = StratifiedChallengeSampler(
            self.replay, self.cache, _settings(Path("index.json"))
        )
        first = sampler.sample_plan(64, rng=random.Random(41))
        second = sampler.sample_plan(64, rng=random.Random(41))
        self.assertEqual(first.plan, second.plan)
        self.assertEqual(len(set(first.plan.global_indices)), 64)
        self.assertEqual(first.summary["earlyPositions"], 39)
        self.assertGreaterEqual(first.summary["earlyFraction"], 0.6)
        self.assertEqual(first.summary["sideToMove"], {"0": 32, "1": 32})
        self.assertEqual(first.summary["actionKind"], {"move": 32, "wall": 32})

    def test_canonical_sidecar_loads_and_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "index.json"
            built = StratifiedChallengeSampler(
                self.replay, self.cache, _settings(path)
            )
            atomic_replace_bytes(path, canonical_json_bytes(built.index_payload))
            loaded = load_stratified_challenge_sampler(
                root, self.replay, self.cache,
                _settings(Path("index.json"), built.index_sha256),
            )
            self.assertEqual(loaded.inventory, built.inventory)
            damaged = dict(built.index_payload)
            damaged["rawWindowSha256"] = "4" * 64
            atomic_replace_bytes(path, canonical_json_bytes(damaged))
            with self.assertRaisesRegex(ChallengeSamplerError, "self hash"):
                load_stratified_challenge_sampler(
                    root, self.replay, self.cache,
                    _settings(Path("index.json"), built.index_sha256),
                )

    def test_empty_required_stratum_fails(self) -> None:
        replay = _Replay([_game(0, early=True, side=0, wall=False)])
        with self.assertRaisesRegex(ChallengeSamplerError, "empty required stratum"):
            StratifiedChallengeSampler(replay, _Cache(replay), _settings(Path("x")))


if __name__ == "__main__":
    unittest.main()
