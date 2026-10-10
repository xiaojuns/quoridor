"""Reversible spatial/player transforms for the new rule contract.

These transforms do not adjudicate games and require no neural libraries.
Every key in an episode history receives the same global transform. In
particular, historical positions are never canonicalized independently by
their own side to play, which would destroy repetition identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence, TypeVar

from . import engine
from .episode import Episode


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Symmetry:
    """Horizontal reflection and/or 180-degree rotation with player swap.

    Rotation must swap players because their goal rows are fixed by player ID.
    The two operations commute and each of the four combinations is its own
    inverse. A current-player value is unchanged by all four transforms.
    """

    mirror: bool = False
    rotate_swap: bool = False

    def __post_init__(self) -> None:
        if type(self.mirror) is not bool or type(self.rotate_swap) is not bool:
            raise TypeError("symmetry flags must be bool")

    def inverse(self) -> Symmetry:
        return self

    def player(self, player: int) -> int:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be integer 0 or 1")
        return 1 - player if self.rotate_swap else player

    def _point(self, point: tuple[int, int], size: int) -> tuple[int, int]:
        if type(point) is not tuple or len(point) != 2 or any(type(x) is not int for x in point):
            raise TypeError("coordinate must be a tuple of two integers")
        if any(not 0 <= x < size for x in point):
            raise ValueError(f"coordinate must be inside a {size} by {size} grid")
        row, col = point
        if self.rotate_swap:
            row, col = size - 1 - row, size - 1 - col
        if self.mirror:
            col = size - 1 - col
        return row, col

    def square(self, point: engine.Position) -> engine.Position:
        return self._point(point, engine.BOARD_SIZE)

    def wall(self, point: engine.Wall) -> engine.Wall:
        return self._point(point, engine.WALL_GRID_SIZE)

    def state(self, state: engine.BoardState) -> engine.BoardState:
        if type(state) is not engine.BoardState:
            raise TypeError("transform requires a quoridor-v1 BoardState")
        players = (1, 0) if self.rotate_swap else (0, 1)
        return engine.BoardState(
            pawns=tuple(self.square(state.pawns[player]) for player in players),
            walls_remaining=tuple(state.walls_remaining[player] for player in players),
            horizontal_walls=frozenset(self.wall(wall) for wall in state.horizontal_walls),
            vertical_walls=frozenset(self.wall(wall) for wall in state.vertical_walls),
            to_play=self.player(state.to_play),
        )

    def history(self, history: tuple[str, ...]) -> tuple[str, ...]:
        if type(history) is not tuple or not history:
            raise ValueError("history must be a nonempty immutable tuple")
        return tuple(
            engine.position_key(self.state(engine.board_from_position_key(key)))
            for key in history
        )

    def episode(self, episode: Episode) -> Episode:
        if type(episode) is not Episode:
            raise TypeError("transform requires a quoridor-v1 Episode")
        outcome = episode.outcome
        if outcome is not None and outcome.winner is not None:
            outcome = replace(outcome, winner=self.player(outcome.winner))
        return Episode(
            state=self.state(episode.state),
            history=self.history(episode.history),
            max_game_plies=episode.max_game_plies,
            outcome=outcome,
        )

    def action(self, action: int) -> int:
        decoded = engine.decode_action(action)
        point = (decoded["row"], decoded["col"])
        if decoded["type"] == "move":
            return engine.encode_move(*self.square(point))
        return engine.encode_wall(decoded["orientation"], *self.wall(point))

    def policy(self, values: Sequence[T]) -> tuple[T, ...]:
        """Permute a length-209 policy or mask without changing its entries.

        The result is an immutable tuple; numeric sequences and one-dimensional
        tensor sequences are supported without importing tensor libraries here.
        ``output[action(i)] == input[i]``. No normalization or masking occurs.
        """

        if len(values) != engine.ACTION_SPACE_SIZE:
            raise ValueError("policy must have exactly 209 entries")
        if getattr(values, "ndim", 1) != 1:
            raise ValueError("policy must have one dimension")
        try:
            finite = all(math.isfinite(value) for value in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("policy entries must be finite numbers") from error
        if not finite:
            raise ValueError("policy entries must be finite numbers")
        return tuple(values[self.action(index)] for index in range(engine.ACTION_SPACE_SIZE))


IDENTITY = Symmetry()
MIRROR = Symmetry(mirror=True)
ROTATE_SWAP = Symmetry(rotate_swap=True)


def canonical_transform(state: engine.BoardState, *, mirror: bool = False) -> Symmetry:
    """Map the current player to player zero, optionally reflecting columns."""

    if type(state) is not engine.BoardState:
        raise TypeError("canonicalization requires a quoridor-v1 BoardState")
    return Symmetry(mirror=mirror, rotate_swap=state.to_play == 1)
