"""Cross-engine correctness and throughput validation for the native engine.

The Python rule engine remains the reference implementation.  This module keeps
the native binding behind a small adapter so the deterministic generators and
reporting code can be tested with Python on both sides before the extension is
built.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Protocol, Sequence

from .codec import position_hash, state_from_serializable, state_to_serializable
from .game import (
    MAX_HALF_MOVES,
    State,
    apply_action_index,
    create_initial_state,
    is_terminal,
    legal_action_indices,
)


REPORT_SCHEMA_VERSION = 2
DEFAULT_NATIVE_MODULE = "quoridor_native"
DEFAULT_STATE_COUNT = 10_000
DEFAULT_LONG_GAMES = 100
DEFAULT_SEED = 0x5EED1234
DEFAULT_BENCHMARK_SAMPLES = 128
DEFAULT_BENCHMARK_REPEATS = 3
MINIMUM_RULE_SPEEDUP = 20.0
MINIMUM_VALIDATION_STATES = 10_000
MINIMUM_LONG_GAMES = 100
MAX_FAILURE_DETAILS = 20
DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "alpha-zero-conformance-v1.json"
)


class NativeBindingUnavailable(RuntimeError):
    """Raised when the native extension cannot satisfy the binding contract."""


class EngineContractError(RuntimeError):
    """Raised when an engine returns a value outside the shared API contract."""


class EngineAdapter(Protocol):
    name: str
    preserves_history: bool

    def from_dict(self, dto: Mapping[str, object]) -> object: ...

    def from_reference(self, state: State) -> object: ...

    def to_dict(self, state: object) -> dict[str, object]: ...

    def to_reference(self, state: object) -> State: ...

    def legal_actions(self, state: object) -> tuple[int, ...]: ...

    def apply(self, state: object, action_index: int) -> object: ...

    def apply_legal(self, state: object, action_index: int) -> object: ...

    def apply_legal_hash(self, state: object, action_index: int) -> str: ...

    def hash(self, state: object) -> str: ...


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise EngineContractError(f"{name} must return a mapping")
    try:
        normalized = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise EngineContractError(f"{name} must return a JSON-safe mapping") from error
    if not isinstance(normalized, dict):
        raise EngineContractError(f"{name} must return a JSON object")
    return normalized


def _legal_tuple(values: object, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        try:
            values = tuple(values)  # type: ignore[arg-type]
        except TypeError as error:
            raise EngineContractError(f"{name} must return an iterable") from error
    actions = tuple(values)
    if any(type(action) is not int or not 0 <= action < 209 for action in actions):
        raise EngineContractError(f"{name} returned an action outside 0..208")
    if actions != tuple(sorted(set(actions))):
        raise EngineContractError(
            f"{name} must return unique action indices in ascending order"
        )
    return actions


@dataclass(frozen=True)
class PythonEngineAdapter:
    """Adapter for the reference engine, also used as the test fallback."""

    name: str = "python-reference"
    preserves_history: bool = True

    def from_dict(self, dto: Mapping[str, object]) -> State:
        return state_from_serializable(dto)

    def from_reference(self, state: State) -> State:
        return state

    def to_dict(self, state: object) -> dict[str, object]:
        if not isinstance(state, State):
            raise EngineContractError("Python adapter expected a State")
        return state_to_serializable(state)

    def to_reference(self, state: object) -> State:
        if not isinstance(state, State):
            raise EngineContractError("Python adapter expected a State")
        return state

    def legal_actions(self, state: object) -> tuple[int, ...]:
        if not isinstance(state, State):
            raise EngineContractError("Python adapter expected a State")
        return legal_action_indices(state)

    def apply(self, state: object, action_index: int) -> State:
        if not isinstance(state, State):
            raise EngineContractError("Python adapter expected a State")
        return apply_action_index(state, action_index)

    def apply_legal(self, state: object, action_index: int) -> State:
        return self.apply(state, action_index)

    def apply_legal_hash(self, state: object, action_index: int) -> str:
        return position_hash(self.apply(state, action_index))

    def hash(self, state: object) -> str:
        if not isinstance(state, State):
            raise EngineContractError("Python adapter expected a State")
        return position_hash(state)


class NativeEngineAdapter:
    """Adapter for the ``quoridor_native`` module-level binding API.

    Required callables are ``state_from_dict``, ``state_to_dict``,
    ``legal_action_indices``, ``apply_action_index``, and ``position_hash``.
    ``state_from_python`` and ``state_to_python`` are used when both exist so
    random-state validation includes the complete repetition history.
    """

    _REQUIRED_CALLABLES = (
        "state_from_dict",
        "state_to_dict",
        "legal_action_indices",
        "apply_action_index",
        "apply_legal_action_index",
        "apply_legal_action_position_hash",
        "position_hash",
    )

    def __init__(self, module: ModuleType, module_name: str | None = None) -> None:
        missing = [
            name
            for name in self._REQUIRED_CALLABLES
            if not callable(getattr(module, name, None))
        ]
        if missing:
            joined = ", ".join(missing)
            raise NativeBindingUnavailable(
                f"native module {module_name or module.__name__!r} is missing "
                f"required callable(s): {joined}"
            )
        has_from_python = callable(getattr(module, "state_from_python", None))
        has_to_python = callable(getattr(module, "state_to_python", None))
        if has_from_python != has_to_python:
            raise NativeBindingUnavailable(
                "native binding must expose state_from_python and state_to_python "
                "together"
            )
        self.module = module
        self.name = module_name or module.__name__
        self.preserves_history = has_from_python and has_to_python

    def from_dict(self, dto: Mapping[str, object]) -> object:
        return self.module.state_from_dict(dict(dto))

    def from_reference(self, state: State) -> object:
        if self.preserves_history:
            return self.module.state_from_python(state)
        return self.from_dict(state_to_serializable(state))

    def to_dict(self, state: object) -> dict[str, object]:
        return _json_dict(self.module.state_to_dict(state), "state_to_dict")

    def to_reference(self, state: object) -> State:
        if self.preserves_history:
            result = self.module.state_to_python(state)
            if not isinstance(result, State):
                raise EngineContractError("state_to_python must return quoridor.game.State")
            return result
        return state_from_serializable(self.to_dict(state))

    def legal_actions(self, state: object) -> tuple[int, ...]:
        return _legal_tuple(
            self.module.legal_action_indices(state),
            "legal_action_indices",
        )

    def apply(self, state: object, action_index: int) -> object:
        result = self.module.apply_action_index(state, action_index)
        if result is None:
            raise EngineContractError("apply_action_index must return the resulting state")
        return result

    def apply_legal(self, state: object, action_index: int) -> object:
        result = self.module.apply_legal_action_index(state, action_index)
        if result is None:
            raise EngineContractError(
                "apply_legal_action_index must return the resulting state"
            )
        return result

    def apply_legal_hash(self, state: object, action_index: int) -> str:
        result = self.module.apply_legal_action_position_hash(state, action_index)
        if not isinstance(result, str):
            raise EngineContractError(
                "apply_legal_action_position_hash must return a string"
            )
        return result

    def hash(self, state: object) -> str:
        result = self.module.position_hash(state)
        if not isinstance(result, str):
            raise EngineContractError("position_hash must return a string")
        return result


def load_native_engine(module_name: str = DEFAULT_NATIVE_MODULE) -> NativeEngineAdapter:
    """Import and validate the native extension, with an actionable error."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise NativeBindingUnavailable(
            f"could not import native module {module_name!r}; build the Phase 3 "
            "extension and ensure it is available on PYTHONPATH"
        ) from error
    return NativeEngineAdapter(module, module_name)


@dataclass(frozen=True)
class ReachableSample:
    index: int
    game_index: int
    state: State
    legal_actions: tuple[int, ...]
    action_index: int
    next_state: State


def generate_reachable_samples(
    state_count: int,
    seed: int = DEFAULT_SEED,
) -> Iterator[ReachableSample]:
    """Yield deterministic reachable states and one sampled legal transition."""

    state_count = _positive_int(state_count, "state_count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    random_source = random.Random(seed)
    game_index = 0
    state = create_initial_state(game_index % 2)
    for index in range(state_count):
        if is_terminal(state):
            game_index += 1
            state = create_initial_state(game_index % 2)

        legal_actions = legal_action_indices(state)
        if not legal_actions:
            raise RuntimeError(
                "reference engine produced a non-terminal state without actions"
            )
        action_index = legal_actions[random_source.randrange(len(legal_actions))]
        next_state = apply_action_index(state, action_index)
        yield ReachableSample(
            index=index,
            game_index=game_index,
            state=state,
            legal_actions=legal_actions,
            action_index=action_index,
            next_state=next_state,
        )
        state = next_state


def _summary_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 24:
            return {
                "count": len(value),
                "first": [_summary_value(item) for item in value[:24]],
            }
        return [_summary_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class _Failures:
    count: int = 0
    details: list[dict[str, object]] = field(default_factory=list)

    def add(
        self,
        *,
        stage: str,
        index: int | str,
        category: str,
        expected: object = None,
        actual: object = None,
        error: BaseException | None = None,
    ) -> None:
        self.count += 1
        if len(self.details) >= MAX_FAILURE_DETAILS:
            return
        detail: dict[str, object] = {
            "stage": stage,
            "index": index,
            "category": category,
        }
        if error is not None:
            detail["error"] = f"{type(error).__name__}: {error}"
        else:
            detail["expected"] = _summary_value(expected)
            detail["actual"] = _summary_value(actual)
        self.details.append(detail)


@dataclass
class _Timer:
    seconds: float = 0.0
    calls: int = 0

    def call(self, function: Any, *args: object) -> Any:
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1


def _compare_state(
    expected: State,
    engine: EngineAdapter,
    actual: object,
    failures: _Failures,
    *,
    stage: str,
    index: int | str,
    timer: _Timer | None = None,
) -> None:
    call = timer.call if timer is not None else lambda function, *args: function(*args)
    expected_hash = position_hash(expected)
    actual_hash = call(engine.hash, actual)
    if actual_hash != expected_hash:
        failures.add(
            stage=stage,
            index=index,
            category="position-hash",
            expected=expected_hash,
            actual=actual_hash,
        )

    expected_dto = state_to_serializable(expected)
    actual_dto = call(engine.to_dict, actual)
    if actual_dto != expected_dto:
        failures.add(
            stage=stage,
            index=index,
            category="state-dto",
            expected=expected_dto,
            actual=actual_dto,
        )

    if engine.preserves_history:
        actual_reference = call(engine.to_reference, actual)
        if dict(actual_reference.position_counts) != dict(expected.position_counts):
            failures.add(
                stage=stage,
                index=index,
                category="position-history",
                expected=dict(expected.position_counts),
                actual=dict(actual_reference.position_counts),
            )


def validate_fixtures(
    engine: EngineAdapter,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
) -> dict[str, object]:
    """Validate all committed Phase 0 positions through the DTO binding path."""

    path = Path(fixture_path)
    fixture = json.loads(path.read_text(encoding="utf-8"))
    positions = fixture.get("positions")
    if not isinstance(positions, list):
        raise ValueError("fixture must contain a positions array")

    failures = _Failures()
    illegal_actions = 0
    transition_count = 0
    started = time.perf_counter()
    for fixture_index, entry in enumerate(positions):
        fixture_id = entry.get("id", fixture_index)
        try:
            expected = state_from_serializable(entry["state"])
            actual = engine.from_dict(entry["state"])
            _compare_state(
                expected,
                engine,
                actual,
                failures,
                stage="fixtures-state",
                index=fixture_id,
            )
            expected_legal = tuple(entry["legalActionIndices"])
            actual_legal = engine.legal_actions(actual)
            if actual_legal != expected_legal:
                failures.add(
                    stage="fixtures-state",
                    index=fixture_id,
                    category="legal-actions",
                    expected=expected_legal,
                    actual=actual_legal,
                )

            action_index = entry["sample"]["actionIndex"]
            transition_count += 1
            if action_index not in actual_legal:
                illegal_actions += 1
                failures.add(
                    stage="fixtures-transition",
                    index=fixture_id,
                    category="sample-action-not-legal",
                    expected=action_index,
                    actual=actual_legal,
                )
                continue
            expected_next = apply_action_index(expected, action_index)
            try:
                actual_next = engine.apply(actual, action_index)
            except Exception as error:
                illegal_actions += 1
                failures.add(
                    stage="fixtures-transition",
                    index=fixture_id,
                    category="apply-exception",
                    error=error,
                )
                continue
            _compare_state(
                expected_next,
                engine,
                actual_next,
                failures,
                stage="fixtures-transition",
                index=fixture_id,
            )
            fixture_next_hash = entry["sample"]["nextPositionHash"]
            actual_next_hash = engine.hash(actual_next)
            if actual_next_hash != fixture_next_hash:
                failures.add(
                    stage="fixtures-transition",
                    index=fixture_id,
                    category="fixture-next-hash",
                    expected=fixture_next_hash,
                    actual=actual_next_hash,
                )
        except Exception as error:  # The report must survive a binding failure.
            if isinstance(error, ValueError) and "illegal" in str(error).lower():
                illegal_actions += 1
            failures.add(
                stage="fixtures",
                index=fixture_id,
                category="exception",
                error=error,
            )

    elapsed = time.perf_counter() - started
    expected_count = fixture.get("positionCount", len(positions))
    passed = (
        len(positions) == expected_count
        and transition_count == len(positions)
        and illegal_actions == 0
        and failures.count == 0
    )
    return {
        "path": str(path),
        "schemaVersion": fixture.get("schemaVersion"),
        "stateCount": len(positions),
        "transitionCount": transition_count,
        "illegalActions": illegal_actions,
        "failureCount": failures.count,
        "failures": failures.details,
        "seconds": elapsed,
        "passed": passed,
    }


def _run_differential(
    engine: EngineAdapter,
    state_count: int,
    seed: int,
    benchmark_sample_limit: int,
) -> tuple[dict[str, object], list[ReachableSample]]:
    failures = _Failures()
    python_timer = _Timer()
    native_timer = _Timer()
    illegal_actions = 0
    compared_state_count = 0
    transition_count = 0
    benchmark_samples: list[ReachableSample] = []

    random_source = random.Random(seed)
    benchmark_random = random.Random(seed ^ 0xB3C4D5E6)
    game_index = 0
    state = create_initial_state(game_index % 2)
    for index in range(state_count):
        if is_terminal(state):
            game_index += 1
            state = create_initial_state(game_index % 2)

        legal = python_timer.call(legal_action_indices, state)
        if not legal:
            failures.add(
                stage="differential",
                index=index,
                category="reference-no-legal-actions",
                actual=position_hash(state),
            )
            game_index += 1
            state = create_initial_state(game_index % 2)
            continue
        action_index = legal[random_source.randrange(len(legal))]
        next_state = python_timer.call(apply_action_index, state, action_index)
        sample = ReachableSample(
            index=index,
            game_index=game_index,
            state=state,
            legal_actions=legal,
            action_index=action_index,
            next_state=next_state,
        )
        if len(benchmark_samples) < benchmark_sample_limit:
            benchmark_samples.append(sample)
        else:
            replacement = benchmark_random.randrange(index + 1)
            if replacement < benchmark_sample_limit:
                benchmark_samples[replacement] = sample

        try:
            actual = native_timer.call(engine.from_reference, state)
            _compare_state(
                state,
                engine,
                actual,
                failures,
                stage="differential-state",
                index=index,
                timer=native_timer,
            )
            actual_legal = native_timer.call(engine.legal_actions, actual)
            compared_state_count += 1
            if actual_legal != legal:
                failures.add(
                    stage="differential-state",
                    index=index,
                    category="legal-actions",
                    expected=legal,
                    actual=actual_legal,
                )
            if action_index not in actual_legal:
                illegal_actions += 1
                failures.add(
                    stage="differential-transition",
                    index=index,
                    category="reference-action-not-native-legal",
                    expected=action_index,
                    actual=actual_legal,
                )
            else:
                try:
                    actual_next = native_timer.call(
                        engine.apply,
                        actual,
                        action_index,
                    )
                except Exception as error:
                    illegal_actions += 1
                    failures.add(
                        stage="differential-transition",
                        index=index,
                        category="apply-exception",
                        error=error,
                    )
                    state = next_state
                    continue
                _compare_state(
                    next_state,
                    engine,
                    actual_next,
                    failures,
                    stage="differential-transition",
                    index=index,
                    timer=native_timer,
                )
                transition_count += 1
        except Exception as error:
            if isinstance(error, ValueError) and "illegal" in str(error).lower():
                illegal_actions += 1
            failures.add(
                stage="differential",
                index=index,
                category="exception",
                error=error,
            )
        state = next_state

    passed = (
        transition_count == state_count
        and illegal_actions == 0
        and failures.count == 0
    )
    return (
        {
            "attemptedStateCount": state_count,
            "stateCount": compared_state_count,
            "attemptedTransitionCount": state_count,
            "transitionCount": transition_count,
            "gamesTouched": game_index + 1,
            "illegalActions": illegal_actions,
            "failureCount": failures.count,
            "failures": failures.details,
            "pythonSeconds": python_timer.seconds,
            "nativeSeconds": native_timer.seconds,
            "passed": passed,
        },
        benchmark_samples,
    )


def _run_long_games(
    engine: EngineAdapter,
    game_count: int,
    seed: int,
) -> dict[str, object]:
    failures = _Failures()
    random_source = random.Random(seed)
    illegal_actions = 0
    completed_games = 0
    total_half_moves = 0
    maximum_half_moves = 0
    termination_reasons: dict[str, int] = {}
    python_timer = _Timer()
    native_timer = _Timer()

    for game_index in range(game_count):
        expected = create_initial_state(game_index % 2)
        try:
            actual = native_timer.call(engine.from_reference, expected)
        except Exception as error:
            failures.add(
                stage="long-game",
                index=game_index,
                category="initialization-exception",
                error=error,
            )
            continue

        half_moves = 0
        game_failed = False
        while not is_terminal(expected):
            expected_legal = python_timer.call(legal_action_indices, expected)
            if not expected_legal:
                failures.add(
                    stage="long-game",
                    index=game_index,
                    category="reference-no-legal-actions",
                    actual=position_hash(expected),
                )
                game_failed = True
                break
            try:
                actual_legal = native_timer.call(engine.legal_actions, actual)
                if actual_legal != expected_legal:
                    failures.add(
                        stage="long-game",
                        index=f"{game_index}:{half_moves}",
                        category="legal-actions",
                        expected=expected_legal,
                        actual=actual_legal,
                    )
                action_index = expected_legal[
                    random_source.randrange(len(expected_legal))
                ]
                if action_index not in actual_legal:
                    illegal_actions += 1
                    failures.add(
                        stage="long-game",
                        index=f"{game_index}:{half_moves}",
                        category="reference-action-not-native-legal",
                        expected=action_index,
                        actual=actual_legal,
                    )
                    game_failed = True
                    break
                expected = python_timer.call(
                    apply_action_index,
                    expected,
                    action_index,
                )
                try:
                    actual = native_timer.call(engine.apply, actual, action_index)
                except Exception as error:
                    illegal_actions += 1
                    failures.add(
                        stage="long-game",
                        index=f"{game_index}:{half_moves}",
                        category="apply-exception",
                        error=error,
                    )
                    game_failed = True
                    break
                half_moves += 1
                _compare_state(
                    expected,
                    engine,
                    actual,
                    failures,
                    stage="long-game-state",
                    index=f"{game_index}:{half_moves}",
                    timer=native_timer,
                )
                if engine.hash(actual) != position_hash(expected):
                    game_failed = True
                    break
                if half_moves > MAX_HALF_MOVES:
                    failures.add(
                        stage="long-game",
                        index=game_index,
                        category="move-limit-not-enforced",
                        expected=MAX_HALF_MOVES,
                        actual=half_moves,
                    )
                    game_failed = True
                    break
            except Exception as error:
                if isinstance(error, ValueError) and "illegal" in str(error).lower():
                    illegal_actions += 1
                failures.add(
                    stage="long-game",
                    index=f"{game_index}:{half_moves}",
                    category="exception",
                    error=error,
                )
                game_failed = True
                break

        total_half_moves += half_moves
        maximum_half_moves = max(maximum_half_moves, half_moves)
        if not game_failed and is_terminal(expected):
            try:
                actual_reference = native_timer.call(engine.to_reference, actual)
                if not is_terminal(actual_reference):
                    failures.add(
                        stage="long-game",
                        index=game_index,
                        category="native-not-terminal",
                        expected=state_to_serializable(expected),
                        actual=engine.to_dict(actual),
                    )
                    continue
                completed_games += 1
                reason = expected.termination_reason or "unknown"
                termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
            except Exception as error:
                failures.add(
                    stage="long-game",
                    index=game_index,
                    category="terminal-check-exception",
                    error=error,
                )

    passed = (
        completed_games == game_count
        and illegal_actions == 0
        and failures.count == 0
    )
    return {
        "games": game_count,
        "completedGames": completed_games,
        "halfMoves": total_half_moves,
        "averageHalfMoves": total_half_moves / game_count,
        "maximumHalfMoves": maximum_half_moves,
        "terminationReasons": dict(sorted(termination_reasons.items())),
        "illegalActions": illegal_actions,
        "failureCount": failures.count,
        "failures": failures.details,
        "pythonSeconds": python_timer.seconds,
        "nativeSeconds": native_timer.seconds,
        "passed": passed,
    }


def _speedup(python_seconds: float, native_seconds: float) -> float | None:
    return python_seconds / native_seconds if native_seconds > 0 else None


def _benchmark_operation(
    operation_name: str,
    samples: Sequence[ReachableSample],
    engine: EngineAdapter,
    native_states: Sequence[object],
) -> dict[str, object]:
    def run_python() -> list[object]:
        if operation_name == "legalActions":
            return [list(legal_action_indices(sample.state)) for sample in samples]
        if operation_name in ("applyAction", "applyLegalAction"):
            return [
                position_hash(apply_action_index(sample.state, sample.action_index))
                for sample in samples
            ]
        if operation_name == "positionHash":
            return [position_hash(sample.state) for sample in samples]
        raise ValueError(f"unknown benchmark operation: {operation_name}")

    def run_native() -> list[object]:
        if operation_name == "legalActions":
            return [list(engine.legal_actions(state)) for state in native_states]
        if operation_name == "applyAction":
            return [
                engine.hash(engine.apply(state, sample.action_index))
                for state, sample in zip(native_states, samples)
            ]
        if operation_name == "applyLegalAction":
            return [
                engine.apply_legal_hash(state, sample.action_index)
                for state, sample in zip(native_states, samples)
            ]
        return [engine.hash(state) for state in native_states]

    # Warm both implementations, then use alternating-order median timings to
    # avoid a cold first call or transient scheduler noise deciding a hard gate.
    python_results = run_python()
    native_results = run_native()
    python_timings: list[float] = []
    native_timings: list[float] = []
    for repeat in range(DEFAULT_BENCHMARK_REPEATS):
        ordered = (("python", run_python), ("native", run_native))
        if repeat % 2:
            ordered = tuple(reversed(ordered))
        for name, operation in ordered:
            started = time.perf_counter()
            results = operation()
            elapsed = time.perf_counter() - started
            if name == "python":
                python_results = results
                python_timings.append(elapsed)
            else:
                native_results = results
                native_timings.append(elapsed)
    python_seconds = statistics.median(python_timings)
    native_seconds = statistics.median(native_timings)

    def checksum(results: object) -> str:
        serialized = json.dumps(
            results,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    python_checksum = checksum(python_results)
    native_checksum = checksum(native_results)

    return {
        "operations": len(samples),
        "repeats": DEFAULT_BENCHMARK_REPEATS,
        "pythonSeconds": python_seconds,
        "nativeSeconds": native_seconds,
        "pythonOperationsPerSecond": (
            len(samples) / python_seconds if python_seconds > 0 else None
        ),
        "nativeOperationsPerSecond": (
            len(samples) / native_seconds if native_seconds > 0 else None
        ),
        "speedup": _speedup(python_seconds, native_seconds),
        "pythonChecksum": python_checksum,
        "nativeChecksum": native_checksum,
        "checksumMatch": python_checksum == native_checksum,
    }


def benchmark_engines(
    engine: EngineAdapter,
    samples: Sequence[ReachableSample],
) -> dict[str, object]:
    """Benchmark legal generation, transition rules, and position hashing."""

    if not samples:
        raise ValueError("benchmark requires at least one sample")
    native_states = [engine.from_reference(sample.state) for sample in samples]
    sections = {
        operation: _benchmark_operation(
            operation,
            samples,
            engine,
            native_states,
        )
        for operation in (
            "legalActions",
            "applyAction",
            "applyLegalAction",
            "positionHash",
        )
    }
    python_seconds = sum(section["pythonSeconds"] for section in sections.values())
    native_seconds = sum(section["nativeSeconds"] for section in sections.values())
    checksum_match = all(section["checksumMatch"] for section in sections.values())
    return {
        "sampleCount": len(samples),
        **sections,
        "combined": {
            "operations": len(samples) * len(sections),
            "pythonSeconds": python_seconds,
            "nativeSeconds": native_seconds,
            "speedup": _speedup(python_seconds, native_seconds),
            "checksumMatch": checksum_match,
        },
        "passed": checksum_match,
    }


def run_validation(
    engine: EngineAdapter,
    *,
    state_count: int = DEFAULT_STATE_COUNT,
    games: int = DEFAULT_LONG_GAMES,
    seed: int = DEFAULT_SEED,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    benchmark_sample_limit: int = DEFAULT_BENCHMARK_SAMPLES,
    minimum_rule_speedup: float = MINIMUM_RULE_SPEEDUP,
    minimum_validation_states: int = MINIMUM_VALIDATION_STATES,
    minimum_long_games: int = MINIMUM_LONG_GAMES,
) -> dict[str, object]:
    """Run all correctness gates and return a JSON-safe summary."""

    state_count = _positive_int(state_count, "state_count")
    games = _positive_int(games, "games")
    benchmark_sample_limit = _positive_int(
        benchmark_sample_limit,
        "benchmark_sample_limit",
    )
    minimum_validation_states = _positive_int(
        minimum_validation_states,
        "minimum_validation_states",
    )
    minimum_long_games = _positive_int(minimum_long_games, "minimum_long_games")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        isinstance(minimum_rule_speedup, bool)
        or not isinstance(minimum_rule_speedup, (int, float))
        or minimum_rule_speedup < 0
    ):
        raise ValueError("minimum_rule_speedup must be non-negative")
    minimum_rule_speedup = float(minimum_rule_speedup)

    fixtures = validate_fixtures(engine, fixture_path)
    differential, samples = _run_differential(
        engine,
        state_count,
        seed,
        min(state_count, benchmark_sample_limit),
    )
    long_games = _run_long_games(engine, games, seed ^ 0xA5A5A5A5)
    try:
        benchmark = benchmark_engines(engine, samples)
    except Exception as error:
        benchmark = {
            "sampleCount": len(samples),
            "error": f"{type(error).__name__}: {error}",
            "combined": {
                "operations": 0,
                "pythonSeconds": 0.0,
                "nativeSeconds": 0.0,
                "speedup": None,
                "checksumMatch": False,
            },
            "passed": False,
        }
    combined_benchmark = benchmark["combined"]
    combined_speedup = combined_benchmark["speedup"]
    legal_speedup = benchmark.get("legalActions", {}).get("speedup")
    apply_speedup = benchmark.get("applyLegalAction", {}).get("speedup")

    illegal_actions = (
        fixtures["illegalActions"]
        + differential["illegalActions"]
        + long_games["illegalActions"]
    )
    gates = {
        "fixtures": bool(fixtures["passed"]),
        "minimumStateWorkload": state_count >= minimum_validation_states,
        "minimumLongGameWorkload": games >= minimum_long_games,
        "stateCount": differential["stateCount"] == state_count,
        "transitionCount": differential["transitionCount"] == state_count,
        "stateAgreement": bool(differential["passed"]),
        "longGamesCompleted": long_games["completedGames"] == games,
        "zeroIllegalActions": illegal_actions == 0,
        "benchmarkChecksums": bool(benchmark["passed"]),
        "legalActionSpeedup": (
            legal_speedup is not None and legal_speedup >= minimum_rule_speedup
        ),
        "applyLegalActionSpeedup": (
            apply_speedup is not None and apply_speedup >= minimum_rule_speedup
        ),
        "combinedRuleSpeedup": (
            combined_speedup is not None
            and combined_speedup >= minimum_rule_speedup
        ),
    }
    passed = all(gates.values())
    failures = [
        *fixtures["failures"],
        *differential["failures"],
        *long_games["failures"],
    ][:MAX_FAILURE_DETAILS]

    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "engine": engine.name,
        "historyTransfer": "full" if engine.preserves_history else "current-only",
        "seed": seed,
        "requested": {
            "states": state_count,
            "games": games,
            "benchmarkSamples": min(state_count, benchmark_sample_limit),
        },
        "fixtures": fixtures,
        "stateCount": differential["stateCount"],
        "transitionCount": differential["transitionCount"],
        "illegalActions": illegal_actions,
        "pythonSeconds": combined_benchmark["pythonSeconds"],
        "nativeSeconds": combined_benchmark["nativeSeconds"],
        "speedup": combined_benchmark["speedup"],
        "minimumSpeedup": minimum_rule_speedup,
        "minimumValidationStates": minimum_validation_states,
        "minimumLongGames": minimum_long_games,
        "differential": differential,
        "longGames": long_games,
        "benchmark": benchmark,
        "gates": gates,
        "failureCount": (
            fixtures["failureCount"]
            + differential["failureCount"]
            + long_games["failureCount"]
        ),
        "failures": failures,
        "passed": passed,
    }


def run_native_validation(
    *,
    state_count: int = DEFAULT_STATE_COUNT,
    games: int = DEFAULT_LONG_GAMES,
    seed: int = DEFAULT_SEED,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    benchmark_sample_limit: int = DEFAULT_BENCHMARK_SAMPLES,
    minimum_rule_speedup: float = MINIMUM_RULE_SPEEDUP,
    module_name: str = DEFAULT_NATIVE_MODULE,
) -> dict[str, object]:
    return run_validation(
        load_native_engine(module_name),
        state_count=state_count,
        games=games,
        seed=seed,
        fixture_path=fixture_path,
        benchmark_sample_limit=benchmark_sample_limit,
        minimum_rule_speedup=minimum_rule_speedup,
    )


def write_validation_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    """Write a stable, standards-compliant JSON report."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(report),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    destination.write_text(f"{serialized}\n", encoding="utf-8")
    return destination


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "DEFAULT_NATIVE_MODULE",
    "DEFAULT_STATE_COUNT",
    "DEFAULT_LONG_GAMES",
    "DEFAULT_SEED",
    "DEFAULT_BENCHMARK_SAMPLES",
    "DEFAULT_BENCHMARK_REPEATS",
    "MINIMUM_RULE_SPEEDUP",
    "MINIMUM_VALIDATION_STATES",
    "MINIMUM_LONG_GAMES",
    "DEFAULT_FIXTURE_PATH",
    "NativeBindingUnavailable",
    "EngineContractError",
    "EngineAdapter",
    "PythonEngineAdapter",
    "NativeEngineAdapter",
    "ReachableSample",
    "load_native_engine",
    "generate_reachable_samples",
    "validate_fixtures",
    "benchmark_engines",
    "run_validation",
    "run_native_validation",
    "write_validation_report",
]
