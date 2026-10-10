"""Legacy Python adapter for the optional C++17 MCTS extension.

This backend still uses repetition/ply-cap losses; it is not supported by
quoridor-v1's ``rules_contract.load_engine``.
"""

from __future__ import annotations

import importlib
import random
from types import ModuleType

from .mcts import EdgeStats, Evaluator, MCTSConfig, SearchResult


class NativeMCTSUnavailable(RuntimeError):
    pass


def load_native_module() -> ModuleType:
    try:
        module = importlib.import_module("quoridor_native")
    except ImportError as error:
        raise NativeMCTSUnavailable(
            "quoridor_native is not importable; run "
            "./scripts/build-native-engine.sh and add native/build/python "
            "to PYTHONPATH"
        ) from error
    if not hasattr(module, "MCTS"):
        raise NativeMCTSUnavailable("quoridor_native does not expose MCTS")
    return module


class NativeMCTS:
    """Drop-in search adapter that keeps tree traversal and rules in C++."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        *,
        module: ModuleType | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.module = module or load_native_module()
        enable_native_encoder = getattr(evaluator, "enable_native_encoder", None)
        if callable(enable_native_encoder):
            enable_native_encoder(self.module)
        seed = self.config.seed
        if seed is None:
            seed = random.SystemRandom().randrange(2**64)
        self._search = self.module.MCTS(
            evaluator,
            simulations=self.config.simulations,
            c_base=self.config.c_base,
            c_init=self.config.c_init,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            dirichlet_alpha=self.config.dirichlet_alpha,
            temperature=self.config.temperature,
            add_root_noise=self.config.add_root_noise,
            seed=seed,
        )

    def run(
        self,
        state: object,
        *,
        temperature: float | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        native = self._search.run(
            state,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )
        return self._to_search_result(native)

    def run_batch(
        self,
        states: list[object] | tuple[object, ...],
        *,
        temperatures: list[float] | tuple[float, ...] | None = None,
        add_root_noise: bool | None = None,
    ) -> tuple[SearchResult, ...]:
        native_results = self._search.run_batch(
            states,
            temperatures=temperatures,
            add_root_noise=add_root_noise,
        )
        return tuple(self._to_search_result(result) for result in native_results)

    def create_session(self, state: object) -> int:
        return int(self._search.create_session(state))

    def sync_session(self, session_id: int, state: object) -> bool:
        return bool(self._search.sync_session(session_id, state))

    def run_session(
        self,
        session_id: int,
        state: object,
        *,
        temperature: float | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        native = self._search.run_session(
            session_id,
            state,
            temperature=temperature,
            add_root_noise=add_root_noise,
        )
        return self._to_search_result(native)

    def run_sessions(
        self,
        session_ids: list[int] | tuple[int, ...],
        states: list[object] | tuple[object, ...],
        *,
        temperatures: list[float] | tuple[float, ...] | None = None,
        add_root_noise: bool | None = None,
    ) -> tuple[SearchResult, ...]:
        native_results = self._search.run_sessions(
            session_ids,
            states,
            temperatures=temperatures,
            add_root_noise=add_root_noise,
        )
        return tuple(self._to_search_result(result) for result in native_results)

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: object,
    ) -> bool:
        return bool(
            self._search.advance_session(session_id, action_index, next_state)
        )

    def release_session(self, session_id: int) -> bool:
        return bool(self._search.release_session(session_id))

    def clear_sessions(self) -> None:
        self._search.clear_sessions()

    @property
    def session_count(self) -> int:
        return int(self._search.session_count)

    def session_node_count(self, session_id: int) -> int:
        return int(self._search.session_node_count(session_id))

    @staticmethod
    def _to_search_result(native: object) -> SearchResult:
        root_edges = {
            edge.action_index: EdgeStats(
                prior=float(edge.prior),
                N=int(edge.visits),
                W=float(edge.value_sum),
            )
            for edge in native.root_edges
        }
        return SearchResult(
            action_index=native.action_index,
            policy=tuple(float(value) for value in native.policy),
            visit_counts=tuple(int(value) for value in native.visit_counts),
            root_value=float(native.root_value),
            root_edges=root_edges,
        )


__all__ = ["NativeMCTSUnavailable", "NativeMCTS", "load_native_module"]
