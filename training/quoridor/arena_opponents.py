"""Fixed diagnostic opponents; never substitutes for the frozen neural arena.

The classic opponent deliberately retains the historical search heuristic and
internal loss-on-repetition convention. Only its action is consumed, validated
by the new engine; the caller adjudicates through Episode. This is a declared
baseline adapter, not a rules migration or a compatible neural champion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import random

import yaml

from . import engine
from .episode import Episode
from .project_config import PROJECT_ROOT
from .project_runtime import Deadline


CONFIG_PATH = PROJECT_ROOT / "configs/opponents-v1.yaml"
DIGEST_PATH = PROJECT_ROOT / "configs/opponents-v1.sha256"


def opponent_identity(kind: str) -> dict[str, object]:
    if kind not in {"random_legal", "shortest_path", "classic"}:
        raise ValueError("unknown fixed opponent")
    raw = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DIGEST_PATH.read_text(encoding="ascii").strip():
        raise ValueError("fixed opponent configuration digest mismatch")
    document = yaml.safe_load(raw)
    sources = ["training/quoridor/arena_opponents.py", "training/quoridor/engine.py"]
    if kind == "classic":
        sources += ["training/quoridor/classic_bridge.py", "training/quoridor/codec.py",
                    "training/quoridor/game.py", "scripts/classic-ai-bridge.mjs",
                    "src/ai.js", "src/game.js", "src/alphazero/state-codec.js"]
    return {
        "kind": kind, "protocol_id": document["protocol_id"],
        "formal_promotion": False, "config_sha256": digest,
        "settings": document[kind],
        "source_files": {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                         for path in sources},
    }


def _live(episode: Episode, deadline: Deadline) -> None:
    deadline.check()
    if type(episode) is not Episode or episode.finished:
        raise ValueError("baseline selection requires a live quoridor-v1 episode")


@dataclass(frozen=True)
class OpponentMove:
    action: int
    kind: str
    nodes: int = 0
    depth: int = 0


def random_legal(episode: Episode, *, deadline: Deadline) -> OpponentMove:
    """Uniform legal action, reproducible for the frozen seed and full history."""
    _live(episode, deadline)
    identity = opponent_identity("random_legal")
    payload = json.dumps([identity["settings"]["seed"], episode.history], separators=(",", ":"))
    rng = random.Random(int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big"))
    action = rng.choice(engine.legal_actions(episode.state))
    deadline.check()
    return OpponentMove(action, "random_legal")


def shortest_path(episode: Episode, *, deadline: Deadline) -> OpponentMove:
    """Move-only baseline; minimize remaining path distance, then action ID."""
    _live(episode, deadline)
    opponent_identity("shortest_path")
    player = episode.state.to_play
    candidates = []
    for position in engine.legal_pawn_moves(episode.state):
        deadline.check()
        action = engine.encode_move(*position)
        state = engine.apply_action(episode.state, action)
        candidates.append((engine.shortest_path_distance(state, player), action))
    deadline.check()
    return OpponentMove(min(candidates)[1], "shortest_path")


def _classic_state(episode: Episode):
    """Translate all history keys without importing legacy adjudication."""
    from .codec import position_hash
    from .game import State

    def translate(board):
        return State(pawns=board.pawns, walls_remaining=board.walls_remaining,
                     horizontal_walls=board.horizontal_walls, vertical_walls=board.vertical_walls,
                     current_player=board.to_play)

    counts = Counter(position_hash(translate(engine.board_from_position_key(key)))
                     for key in episode.history)
    return replace(translate(episode.state), move_number=episode.plies, position_counts=dict(counts))


class ClassicOpponent:
    """Explicit optional Node bridge with fixed node/depth limits and no fallback."""

    def __init__(self, *, deadline: Deadline):
        from .classic_bridge import ClassicAIBridge
        deadline.check()
        self._identity_json = json.dumps(opponent_identity("classic"), sort_keys=True)
        self._bridge = ClassicAIBridge(startup_timeout=min(5.0, deadline.remaining_seconds))
        try:
            deadline.check()
        except BaseException:
            self.close()
            raise

    @property
    def identity(self) -> dict[str, object]:
        return json.loads(self._identity_json)

    def choose(self, episode: Episode, *, deadline: Deadline) -> OpponentMove:
        from .classic_bridge import ClassicSearchBudget
        _live(episode, deadline)
        state = _classic_state(episode)
        deadline.check()
        settings = self.identity["settings"]
        result = self._bridge.choose(
            state, budget=ClassicSearchBudget.nodes(settings["node_limit"]),
            max_depth=settings["max_depth"], timeout=max(1e-9, deadline.remaining_seconds),
        )
        deadline.check()
        if result.action_index not in engine.legal_actions(episode.state):
            raise ValueError("classic opponent returned an illegal quoridor-v1 action")
        deadline.check()
        return OpponentMove(result.action_index, "classic", result.nodes, result.depth)

    def close(self) -> None:
        self._bridge.close()
