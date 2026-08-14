"""Common action-selector boundary for neural and classic arena players."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol, Sequence

from .classic_bridge import (
    CLASSIC_ENGINE,
    HARD_MAX_DEPTH,
    ClassicAIBridge,
    ClassicSearchBudget,
)
from .game import State, legal_action_indices
from .mcts import Evaluator, MCTS, MCTSConfig


MCTS_ROOT_VISIT_SEMANTICS = "cumulative-root-visits-with-subtree-reuse-v1"


class SelectorError(RuntimeError):
    pass


class SelectorClosed(SelectorError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionResult:
    action_index: int
    engine: str
    value: float | None
    score: float | None
    simulations: int
    nodes: int
    elapsed_ms: float
    depth: int
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.action_index) is not int or not 0 <= self.action_index < 209:
            raise ValueError("action_index must be an integer in [0, 208]")
        if not isinstance(self.engine, str) or not self.engine:
            raise ValueError("engine must be a non-empty string")
        for value, label in ((self.value, "value"), (self.score, "score")):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{label} must be finite or None")
        for value, label in (
            (self.simulations, "simulations"),
            (self.nodes, "nodes"),
            (self.depth, "depth"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if (
            not isinstance(self.elapsed_ms, (int, float))
            or isinstance(self.elapsed_ms, bool)
            or not math.isfinite(float(self.elapsed_ms))
            or self.elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms must be finite and non-negative")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


class MoveSelector(Protocol):
    def choose(self, state: State) -> SelectionResult: ...

    def close(self) -> None: ...


class SessionMoveSelector(MoveSelector, Protocol):
    """Optional persistent-session API used by batched neural arenas."""

    @property
    def supports_sessions(self) -> bool: ...

    def create_session(self, state: State) -> int: ...

    def choose_sessions(
        self,
        session_ids: Sequence[int],
        states: Sequence[State],
    ) -> tuple[SelectionResult, ...]: ...

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: State,
    ) -> bool: ...

    def release_session(self, session_id: int) -> bool: ...


class ClassicHardSelector:
    """Selector backed by the exact browser hard AI implementation."""

    def __init__(
        self,
        *,
        budget: ClassicSearchBudget | None = None,
        max_depth: int = HARD_MAX_DEPTH,
        request_timeout: float | None = None,
        bridge: ClassicAIBridge | None = None,
    ) -> None:
        self.budget = budget or ClassicSearchBudget.time()
        if not isinstance(self.budget, ClassicSearchBudget):
            raise TypeError("budget must be a ClassicSearchBudget")
        if type(max_depth) is not int or not 1 <= max_depth <= 12:
            raise ValueError("max_depth must be an integer in [1, 12]")
        self.max_depth = max_depth
        self.request_timeout = request_timeout
        self._owns_bridge = bridge is None
        self._bridge = bridge or ClassicAIBridge()
        self._closed = False
        self._next_session_id = 0
        self._sessions: set[int] = set()

    def choose(self, state: State) -> SelectionResult:
        if self._closed:
            raise SelectorClosed("classic hard selector is closed")
        result = self._bridge.choose(
            state,
            budget=self.budget,
            max_depth=self.max_depth,
            timeout=self.request_timeout,
        )
        return SelectionResult(
            action_index=result.action_index,
            engine=CLASSIC_ENGINE,
            value=None,
            score=result.score,
            simulations=0,
            nodes=result.nodes,
            elapsed_ms=result.elapsed_ms,
            depth=result.depth,
            metadata={
                "budgetKind": result.budget.kind,
                "budgetLimit": result.budget.limit,
                "positionHash": result.position_hash,
                "historyEntries": result.history_entries,
                "historyOccurrences": result.history_occurrences,
            },
        )

    @property
    def supports_sessions(self) -> bool:
        """Expose a stateless session facade for mixed neural/classic batches."""

        return True

    def create_session(self, state: State) -> int:
        if self._closed:
            raise SelectorClosed("classic hard selector is closed")
        if not isinstance(state, State):
            raise TypeError("state must be a quoridor.game.State")
        session_id = self._next_session_id
        self._next_session_id += 1
        self._sessions.add(session_id)
        return session_id

    def choose_sessions(
        self,
        session_ids: Sequence[int],
        states: Sequence[State],
    ) -> tuple[SelectionResult, ...]:
        if self._closed:
            raise SelectorClosed("classic hard selector is closed")
        session_ids = tuple(session_ids)
        states = tuple(states)
        if not states or len(states) != len(session_ids):
            raise ValueError("session_ids and states must have the same non-zero length")
        if any(session_id not in self._sessions for session_id in session_ids):
            raise SelectorError("classic selector received an unknown session ID")
        if any(not isinstance(state, State) for state in states):
            raise TypeError("states must contain quoridor.game.State values")
        return tuple(self.choose(state) for state in states)

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: State,
    ) -> bool:
        if self._closed:
            raise SelectorClosed("classic hard selector is closed")
        if session_id not in self._sessions:
            raise SelectorError("classic selector received an unknown session ID")
        if type(action_index) is not int or not 0 <= action_index < 209:
            raise ValueError("action_index must be an integer in [0, 208]")
        if not isinstance(next_state, State):
            raise TypeError("next_state must be a quoridor.game.State")
        return True

    def release_session(self, session_id: int) -> bool:
        if self._closed:
            raise SelectorClosed("classic hard selector is closed")
        if session_id not in self._sessions:
            return False
        self._sessions.remove(session_id)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._sessions.clear()
        if self._owns_bridge:
            self._bridge.close()
        self._closed = True

    def __enter__(self) -> "ClassicHardSelector":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class MCTSSelector:
    """Deterministic evaluation wrapper around a policy/value MCTS."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig,
        *,
        search_factory: Callable[[Evaluator, MCTSConfig], MCTS] = MCTS,
        engine: str = "python-mcts",
    ) -> None:
        if not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must define evaluate(state)")
        if not isinstance(config, MCTSConfig):
            raise TypeError("config must be an MCTSConfig")
        if not callable(search_factory):
            raise TypeError("search_factory must be callable")
        if not isinstance(engine, str) or not engine:
            raise ValueError("engine must be a non-empty string")
        self.engine = engine
        self.config = replace(config, temperature=0.0, add_root_noise=False)
        self._search = search_factory(evaluator, self.config)
        self._closed = False

    def _assert_open(self) -> None:
        if self._closed:
            raise SelectorClosed("MCTS selector is closed")

    def _selection(
        self,
        state: State,
        result: object,
        *,
        elapsed_ms: float,
        batch_size: int,
    ) -> SelectionResult:
        legal = frozenset(legal_action_indices(state))
        if not legal:
            raise ValueError("selector cannot choose from a terminal state")
        action_index = getattr(result, "action_index", None)
        if action_index not in legal:
            raise SelectorError("MCTS returned an illegal or missing action")
        visit_counts = getattr(result, "visit_counts", ())
        if (
            not isinstance(visit_counts, Sequence)
            or len(visit_counts) != 209
            or any(type(visits) is not int or visits < 0 for visits in visit_counts)
        ):
            raise SelectorError(
                "MCTS visit counts must contain 209 non-negative integers"
            )
        root_visits = sum(visit_counts)
        if root_visits < self.config.simulations:
            raise SelectorError(
                "MCTS returned fewer root visits than the configured simulations"
            )
        return SelectionResult(
            action_index=action_index,
            engine=self.engine,
            value=float(getattr(result, "root_value")),
            score=None,
            simulations=self.config.simulations,
            nodes=root_visits,
            elapsed_ms=elapsed_ms,
            depth=0,
            metadata={
                "configuredSimulations": self.config.simulations,
                "batchSize": batch_size,
                "rootVisits": root_visits,
                "retainedRootVisits": root_visits - self.config.simulations,
                "visitCountSemantics": MCTS_ROOT_VISIT_SEMANTICS,
            },
        )

    def choose(self, state: State) -> SelectionResult:
        self._assert_open()
        if not isinstance(state, State):
            raise TypeError("state must be a quoridor.game.State")
        started_at = time.monotonic()
        result = self._search.run(
            state,
            temperature=0.0,
            add_root_noise=False,
        )
        elapsed_ms = (time.monotonic() - started_at) * 1_000.0
        return self._selection(
            state,
            result,
            elapsed_ms=elapsed_ms,
            batch_size=1,
        )

    @property
    def supports_sessions(self) -> bool:
        methods = (
            getattr(self._search, "create_session", None),
            getattr(self._search, "run_sessions", None),
            getattr(self._search, "advance_session", None),
            getattr(self._search, "release_session", None),
        )
        available = tuple(callable(method) for method in methods)
        if any(available) and not all(available):
            raise SelectorError(
                "MCTS backend exposes an incomplete persistent-session API"
            )
        return all(available)

    def create_session(self, state: State) -> int:
        self._assert_open()
        if not self.supports_sessions:
            raise SelectorError("MCTS backend does not support persistent sessions")
        if not isinstance(state, State):
            raise TypeError("state must be a quoridor.game.State")
        return int(self._search.create_session(state))

    def choose_sessions(
        self,
        session_ids: Sequence[int],
        states: Sequence[State],
    ) -> tuple[SelectionResult, ...]:
        self._assert_open()
        if not self.supports_sessions:
            raise SelectorError("MCTS backend does not support persistent sessions")
        session_ids = tuple(session_ids)
        states = tuple(states)
        if not states or len(states) != len(session_ids):
            raise ValueError("session_ids and states must have the same non-zero length")
        if any(type(session_id) is not int or session_id < 0 for session_id in session_ids):
            raise ValueError("session IDs must be non-negative integers")
        if any(not isinstance(state, State) for state in states):
            raise TypeError("states must contain quoridor.game.State values")
        started_at = time.monotonic()
        results = self._search.run_sessions(
            session_ids,
            states,
            temperatures=[0.0] * len(states),
            add_root_noise=False,
        )
        elapsed_ms = (time.monotonic() - started_at) * 1_000.0
        if len(results) != len(states):
            raise SelectorError("batched MCTS returned the wrong result count")
        return tuple(
            self._selection(
                state,
                result,
                elapsed_ms=elapsed_ms,
                batch_size=len(states),
            )
            for state, result in zip(states, results)
        )

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: State,
    ) -> bool:
        self._assert_open()
        if not self.supports_sessions:
            raise SelectorError("MCTS backend does not support persistent sessions")
        return bool(
            self._search.advance_session(session_id, action_index, next_state)
        )

    def release_session(self, session_id: int) -> bool:
        self._assert_open()
        if not self.supports_sessions:
            raise SelectorError("MCTS backend does not support persistent sessions")
        return bool(self._search.release_session(session_id))

    def close(self) -> None:
        if self._closed:
            return
        close_search = getattr(self._search, "close", None)
        if callable(close_search):
            close_search()
        self._closed = True

    def __enter__(self) -> "MCTSSelector":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "SelectorError",
    "SelectorClosed",
    "SelectionResult",
    "MoveSelector",
    "SessionMoveSelector",
    "ClassicHardSelector",
    "MCTSSelector",
]
