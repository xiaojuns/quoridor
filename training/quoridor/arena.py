from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .game import State, apply_action_index, create_initial_state, get_winner, is_terminal
from .mcts import Evaluator, MCTS, MCTSConfig


@dataclass(frozen=True)
class ArenaConfig:
    games: int = 40
    simulations: int = 256
    acceptance_rate: float = 0.55
    seed: int = 0

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if not 0 <= self.acceptance_rate <= 1:
            raise ValueError("acceptance_rate must be in [0, 1]")


@dataclass(frozen=True)
class ArenaResult:
    candidate_wins: int
    champion_wins: int
    games: int
    candidate_as_first_wins: int
    candidate_as_second_wins: int
    accepted: bool

    @property
    def candidate_win_rate(self) -> float:
        return self.candidate_wins / self.games


def evaluate_candidate(
    candidate: Evaluator,
    champion: Evaluator,
    *,
    mcts_config: MCTSConfig,
    arena_config: ArenaConfig,
    initial_state: State | None = None,
    search_factory: Callable[[Evaluator, MCTSConfig], MCTS] = MCTS,
    batched: bool = False,
) -> ArenaResult:
    candidate_wins = 0
    champion_wins = 0
    first_wins = 0
    second_wins = 0

    search_config = replace(
        mcts_config,
        simulations=arena_config.simulations,
        temperature=0.0,
        add_root_noise=False,
    )
    if batched:
        states = [initial_state or create_initial_state() for _ in range(arena_config.games)]
        if any(is_terminal(state) for state in states):
            raise ValueError("arena initial state must be non-terminal")
        candidate_search = search_factory(
            candidate,
            replace(search_config, seed=arena_config.seed),
        )
        champion_search = search_factory(
            champion,
            replace(search_config, seed=arena_config.seed + 1),
        )
        candidate_batch = getattr(candidate_search, "run_batch", None)
        champion_batch = getattr(champion_search, "run_batch", None)
        if not callable(candidate_batch) or not callable(champion_batch):
            raise TypeError("batched arena requires search engines with run_batch")

        while True:
            active = [
                index for index, state in enumerate(states) if not is_terminal(state)
            ]
            if not active:
                break
            candidate_indices = [
                index
                for index in active
                if states[index].current_player == index % 2
            ]
            champion_indices = [
                index
                for index in active
                if states[index].current_player != index % 2
            ]
            for indices, run_batch in (
                (candidate_indices, candidate_batch),
                (champion_indices, champion_batch),
            ):
                if not indices:
                    continue
                results = run_batch(
                    [states[index] for index in indices],
                    temperatures=[0.0] * len(indices),
                    add_root_noise=False,
                )
                if len(results) != len(indices):
                    raise RuntimeError("batched arena returned the wrong result count")
                for index, result in zip(indices, results):
                    if result.action_index is None:
                        raise RuntimeError("arena MCTS returned no legal action")
                    states[index] = apply_action_index(
                        states[index],
                        result.action_index,
                    )

        for game_index, state in enumerate(states):
            winner = get_winner(state)
            if winner is None:
                raise RuntimeError("arena game ended without a winner")
            candidate_player = game_index % 2
            if winner == candidate_player:
                candidate_wins += 1
                if candidate_player == 0:
                    first_wins += 1
                else:
                    second_wins += 1
            else:
                champion_wins += 1
        return ArenaResult(
            candidate_wins=candidate_wins,
            champion_wins=champion_wins,
            games=arena_config.games,
            candidate_as_first_wins=first_wins,
            candidate_as_second_wins=second_wins,
            accepted=(candidate_wins / arena_config.games)
            >= arena_config.acceptance_rate,
        )

    for game_index in range(arena_config.games):
        candidate_player = game_index % 2
        state = initial_state or create_initial_state()
        if is_terminal(state):
            raise ValueError("arena initial state must be non-terminal")
        candidate_search = search_factory(
            candidate,
            replace(search_config, seed=arena_config.seed + game_index * 2),
        )
        champion_search = search_factory(
            champion,
            replace(search_config, seed=arena_config.seed + game_index * 2 + 1),
        )

        while not is_terminal(state):
            search = (
                candidate_search
                if state.current_player == candidate_player
                else champion_search
            )
            result = search.run(state, temperature=0.0, add_root_noise=False)
            if result.action_index is None:
                raise RuntimeError("arena MCTS returned no legal action")
            state = apply_action_index(state, result.action_index)

        winner = get_winner(state)
        if winner is None:
            raise RuntimeError("arena game ended without a winner")
        if winner == candidate_player:
            candidate_wins += 1
            if candidate_player == 0:
                first_wins += 1
            else:
                second_wins += 1
        else:
            champion_wins += 1

    return ArenaResult(
        candidate_wins=candidate_wins,
        champion_wins=champion_wins,
        games=arena_config.games,
        candidate_as_first_wins=first_wins,
        candidate_as_second_wins=second_wins,
        accepted=(candidate_wins / arena_config.games) >= arena_config.acceptance_rate,
    )


__all__ = ["ArenaConfig", "ArenaResult", "evaluate_candidate"]
