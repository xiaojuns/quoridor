"""Persistent JSONL bridge to the browser's classic hard Negamax engine."""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .codec import encode_action, position_hash
from .game import State, legal_action_indices
from .openings import (
    OpeningValidationError,
    state_from_opening_payload,
    state_to_opening_payload,
)


BRIDGE_PROTOCOL_VERSION = 1
CLASSIC_ENGINE = "classic-negamax"
HARD_TIME_LIMIT_MS = 850
HARD_MAX_DEPTH = 5


class ClassicBridgeError(RuntimeError):
    pass


class ClassicBridgeStartError(ClassicBridgeError):
    pass


class ClassicBridgeClosed(ClassicBridgeError):
    pass


class ClassicBridgeTimeout(ClassicBridgeError):
    pass


class ClassicBridgeCrashed(ClassicBridgeError):
    pass


class ClassicBridgeProtocolError(ClassicBridgeError):
    pass


class ClassicBridgeRemoteError(ClassicBridgeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"classic bridge rejected request ({code}): {message}")
        self.code = code
        self.remote_message = message


@dataclass(frozen=True, slots=True)
class ClassicSearchBudget:
    kind: str
    limit: int

    def __post_init__(self) -> None:
        if self.kind not in {"time", "nodes"}:
            raise ValueError('classic budget kind must be "time" or "nodes"')
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("classic budget limit must be a non-negative integer")

    @classmethod
    def time(cls, milliseconds: int = HARD_TIME_LIMIT_MS) -> "ClassicSearchBudget":
        return cls("time", milliseconds)

    @classmethod
    def nodes(cls, node_limit: int) -> "ClassicSearchBudget":
        return cls("nodes", node_limit)

    def to_payload(self) -> dict[str, object]:
        if self.kind == "time":
            return {"kind": "time", "timeLimitMs": self.limit}
        return {"kind": "nodes", "nodeLimit": self.limit}


@dataclass(frozen=True, slots=True)
class ClassicBridgeResult:
    action_index: int
    action: Mapping[str, object]
    score: float
    depth: int
    nodes: int
    elapsed_ms: float
    position_hash: str
    history_entries: int
    history_occurrences: int
    budget: ClassicSearchBudget
    engine: str = CLASSIC_ENGINE


@dataclass(frozen=True, slots=True)
class ClassicRootScore:
    action_index: int
    action: Mapping[str, object]
    searched: bool
    score: float | None


@dataclass(frozen=True, slots=True)
class ClassicAnalysisResult:
    action_index: int
    action: Mapping[str, object]
    score: float
    depth: int
    nodes: int
    elapsed_ms: float
    position_hash: str
    history_entries: int
    history_occurrences: int
    budget: ClassicSearchBudget
    root_scores: tuple[ClassicRootScore, ...]
    root_score_kind: str = "alpha-beta-return-v1"
    engine: str = CLASSIC_ENGINE


_EOF = object()


class ClassicAIBridge:
    """Own one serialized, persistent Node child process."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        startup_timeout: float = 5.0,
        request_timeout: float = 5.0,
        cwd: str | Path | None = None,
    ) -> None:
        self._startup_timeout = self._positive_timeout(
            startup_timeout, "startup_timeout"
        )
        self._request_timeout = self._positive_timeout(
            request_timeout, "request_timeout"
        )
        self._root = Path(__file__).resolve().parents[2]
        if command is None:
            node = shutil.which("node")
            if node is None:
                raise ClassicBridgeStartError("Node.js is not available on PATH")
            command = (node, str(self._root / "scripts" / "classic-ai-bridge.mjs"))
        if (
            isinstance(command, (str, bytes))
            or not isinstance(command, Sequence)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError("command must be a non-empty sequence of strings")

        self._lock = threading.Lock()
        self._stdout: queue.Queue[str | object] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=20)
        self._closed = False
        self._next_request_id = 1
        try:
            self._process = subprocess.Popen(
                tuple(command),
                cwd=str(Path(cwd) if cwd is not None else self._root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            raise ClassicBridgeStartError(
                f"cannot start classic AI bridge: {error}"
            ) from error
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            self._terminate_process()
            raise ClassicBridgeStartError("classic AI bridge pipes are unavailable")

        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="classic-ai-bridge-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="classic-ai-bridge-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._receive(self._startup_timeout)
            self._validate_ready(ready)
        except ClassicBridgeError:
            self._terminate_process()
            self._closed = True
            raise

    @staticmethod
    def _positive_timeout(value: object, label: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{label} must be finite and positive")
        return result

    def _read_stdout(self) -> None:
        try:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._stdout.put(line.rstrip("\r\n"))
        finally:
            self._stdout.put(_EOF)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip("\r\n"))

    def _diagnostics(self) -> str:
        detail = "\n".join(self._stderr)
        return f"; stderr: {detail}" if detail else ""

    def _receive(self, timeout: float) -> Mapping[str, object]:
        try:
            item = self._stdout.get(timeout=timeout)
        except queue.Empty as error:
            if self._process.poll() is not None:
                raise ClassicBridgeCrashed(
                    f"classic AI bridge exited with code {self._process.returncode}"
                    f"{self._diagnostics()}"
                ) from error
            raise ClassicBridgeTimeout(
                f"classic AI bridge did not respond within {timeout:.3f}s"
            ) from error
        if item is _EOF:
            return_code = self._process.poll()
            raise ClassicBridgeCrashed(
                f"classic AI bridge closed stdout unexpectedly"
                f" (exit code {return_code}){self._diagnostics()}"
            )
        try:
            payload = json.loads(str(item))
        except json.JSONDecodeError as error:
            raise ClassicBridgeProtocolError(
                f"classic AI bridge emitted invalid JSON: {item!r}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ClassicBridgeProtocolError(
                "classic AI bridge response must be a JSON object"
            )
        return payload

    @staticmethod
    def _validate_ready(payload: Mapping[str, object]) -> None:
        if (
            payload.get("version") != BRIDGE_PROTOCOL_VERSION
            or payload.get("type") != "ready"
            or payload.get("engine") != CLASSIC_ENGINE
        ):
            raise ClassicBridgeProtocolError(
                "classic AI bridge sent an invalid startup handshake"
            )
        defaults = payload.get("hardDefaults")
        if not isinstance(defaults, Mapping) or defaults.get(
            "timeLimitMs"
        ) != HARD_TIME_LIMIT_MS or defaults.get("maxDepth") != HARD_MAX_DEPTH:
            raise ClassicBridgeProtocolError(
                "classic AI bridge hard defaults do not match Python contract"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClassicBridgeClosed("classic AI bridge is closed")
        if self._process.poll() is not None:
            self._closed = True
            raise ClassicBridgeCrashed(
                f"classic AI bridge exited with code {self._process.returncode}"
                f"{self._diagnostics()}"
            )

    def _send(self, payload: Mapping[str, object]) -> None:
        self._ensure_open()
        assert self._process.stdin is not None
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            self._process.stdin.write(serialized + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ClassicBridgeCrashed(
                f"cannot write to classic AI bridge{self._diagnostics()}"
            ) from error

    def _next_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    @staticmethod
    def _response_for(
        payload: Mapping[str, object],
        request_id: int,
        expected_type: str,
    ) -> Mapping[str, object]:
        if payload.get("version") != BRIDGE_PROTOCOL_VERSION:
            raise ClassicBridgeProtocolError("classic bridge protocol version mismatch")
        if payload.get("id") != request_id:
            raise ClassicBridgeProtocolError("classic bridge response id mismatch")
        response_type = payload.get("type")
        if response_type == "error":
            error = payload.get("error")
            if not isinstance(error, Mapping):
                raise ClassicBridgeProtocolError("classic bridge error is malformed")
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise ClassicBridgeProtocolError("classic bridge error is malformed")
            raise ClassicBridgeRemoteError(code, message)
        if response_type != expected_type:
            raise ClassicBridgeProtocolError(
                f"expected classic bridge {expected_type!r}, received {response_type!r}"
            )
        return payload

    @staticmethod
    def _finite_number(value: object, label: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ClassicBridgeProtocolError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ClassicBridgeProtocolError(f"{label} must be finite")
        return result

    @staticmethod
    def _non_negative_integer(value: object, label: str) -> int:
        if type(value) is not int or value < 0:
            raise ClassicBridgeProtocolError(
                f"{label} must be a non-negative integer"
            )
        return value

    def choose(
        self,
        state: State,
        *,
        budget: ClassicSearchBudget | None = None,
        max_depth: int = HARD_MAX_DEPTH,
        timeout: float | None = None,
    ) -> ClassicBridgeResult:
        """Ask the classic engine for one move and revalidate it in Python."""

        if not isinstance(state, State):
            raise TypeError("state must be a quoridor.game.State")
        budget = budget or ClassicSearchBudget.time()
        if not isinstance(budget, ClassicSearchBudget):
            raise TypeError("budget must be a ClassicSearchBudget")
        if type(max_depth) is not int or not 1 <= max_depth <= 12:
            raise ValueError("max_depth must be an integer in [1, 12]")
        selected_timeout = (
            self._request_timeout
            if timeout is None
            else self._positive_timeout(timeout, "timeout")
        )
        try:
            state_payload = state_to_opening_payload(state)
            restored = state_from_opening_payload(state_payload)
        except OpeningValidationError as error:
            raise ValueError(
                f"classic bridge requires a complete state/history: {error}"
            ) from error
        if state_to_opening_payload(restored) != state_payload:
            raise ValueError("classic bridge state did not round-trip with full history")

        expected_hash = position_hash(state)
        legal_actions = frozenset(legal_action_indices(state))
        if not legal_actions:
            raise ValueError("classic bridge cannot choose from a terminal state")
        history_entries = len(state.position_counts)
        history_occurrences = sum(state.position_counts.values())

        with self._lock:
            try:
                self._ensure_open()
                request_id = self._next_id()
                request = {
                    "version": BRIDGE_PROTOCOL_VERSION,
                    "type": "choose",
                    "id": request_id,
                    "positionHash": expected_hash,
                    "state": state_payload,
                    "budget": budget.to_payload(),
                    "maxDepth": max_depth,
                }
                self._send(request)
                response = self._response_for(
                    self._receive(selected_timeout), request_id, "result"
                )
                result = self._parse_result(
                    response,
                    state=state,
                    expected_hash=expected_hash,
                    legal_actions=legal_actions,
                    expected_history_entries=history_entries,
                    expected_history_occurrences=history_occurrences,
                    budget=budget,
                )
            except (
                ClassicBridgeTimeout,
                ClassicBridgeCrashed,
                ClassicBridgeProtocolError,
            ):
                self._terminate_process()
                self._closed = True
                raise
            return result

    def analyze(
        self,
        state: State,
        *,
        budget: ClassicSearchBudget | None = None,
        max_depth: int = HARD_MAX_DEPTH,
        timeout: float | None = None,
    ) -> ClassicAnalysisResult:
        """Return the classic engine's deepest complete root-score vector."""

        if not isinstance(state, State):
            raise TypeError("state must be a quoridor.game.State")
        budget = budget or ClassicSearchBudget.time()
        if not isinstance(budget, ClassicSearchBudget):
            raise TypeError("budget must be a ClassicSearchBudget")
        if type(max_depth) is not int or not 1 <= max_depth <= 12:
            raise ValueError("max_depth must be an integer in [1, 12]")
        selected_timeout = self._request_timeout if timeout is None else self._positive_timeout(timeout, "timeout")
        try:
            state_payload = state_to_opening_payload(state)
            restored = state_from_opening_payload(state_payload)
        except OpeningValidationError as error:
            raise ValueError(f"classic bridge requires a complete state/history: {error}") from error
        if state_to_opening_payload(restored) != state_payload:
            raise ValueError("classic bridge state did not round-trip with full history")
        expected_hash = position_hash(state)
        legal_actions = frozenset(legal_action_indices(state))
        if not legal_actions:
            raise ValueError("classic bridge cannot analyze a terminal state")
        history_entries = len(state.position_counts)
        history_occurrences = sum(state.position_counts.values())

        with self._lock:
            try:
                self._ensure_open()
                request_id = self._next_id()
                self._send(
                    {
                        "version": BRIDGE_PROTOCOL_VERSION,
                        "type": "analyze",
                        "id": request_id,
                        "positionHash": expected_hash,
                        "state": state_payload,
                        "budget": budget.to_payload(),
                        "maxDepth": max_depth,
                    }
                )
                response = self._response_for(self._receive(selected_timeout), request_id, "analysis")
                choice = self._parse_result(
                    response,
                    state=state,
                    expected_hash=expected_hash,
                    legal_actions=legal_actions,
                    expected_history_entries=history_entries,
                    expected_history_occurrences=history_occurrences,
                    budget=budget,
                )
                if response.get("rootScoreKind") != "alpha-beta-return-v1":
                    raise ClassicBridgeProtocolError("classic root score kind mismatch")
                raw_scores = response.get("rootScores")
                if not isinstance(raw_scores, list) or len(raw_scores) != len(legal_actions):
                    raise ClassicBridgeProtocolError("classic root scores must cover every legal action")
                parsed: list[ClassicRootScore] = []
                seen: set[int] = set()
                for offset, entry in enumerate(raw_scores):
                    if not isinstance(entry, Mapping) or not isinstance(entry.get("action"), Mapping):
                        raise ClassicBridgeProtocolError(f"rootScores[{offset}] is malformed")
                    try:
                        action_index = encode_action(entry["action"])
                    except (TypeError, ValueError) as error:
                        raise ClassicBridgeProtocolError(f"rootScores[{offset}] action is invalid") from error
                    if action_index not in legal_actions or action_index in seen:
                        raise ClassicBridgeProtocolError("classic root scores contain illegal or duplicate actions")
                    seen.add(action_index)
                    searched = entry.get("searched")
                    if type(searched) is not bool:
                        raise ClassicBridgeProtocolError("classic root score searched flag is invalid")
                    raw_score = entry.get("score")
                    score = self._finite_number(raw_score, "root score") if searched else None
                    if not searched and raw_score is not None:
                        raise ClassicBridgeProtocolError("unsearched classic root action has a score")
                    parsed.append(ClassicRootScore(action_index, dict(entry["action"]), searched, score))
                if seen != set(legal_actions):
                    raise ClassicBridgeProtocolError("classic root scores omit legal actions")
            except (ClassicBridgeTimeout, ClassicBridgeCrashed, ClassicBridgeProtocolError):
                self._terminate_process()
                self._closed = True
                raise
        return ClassicAnalysisResult(
            action_index=choice.action_index,
            action=choice.action,
            score=choice.score,
            depth=choice.depth,
            nodes=choice.nodes,
            elapsed_ms=choice.elapsed_ms,
            position_hash=choice.position_hash,
            history_entries=choice.history_entries,
            history_occurrences=choice.history_occurrences,
            budget=choice.budget,
            root_scores=tuple(parsed),
        )

    def _parse_result(
        self,
        response: Mapping[str, object],
        *,
        state: State,
        expected_hash: str,
        legal_actions: frozenset[int],
        expected_history_entries: int,
        expected_history_occurrences: int,
        budget: ClassicSearchBudget,
    ) -> ClassicBridgeResult:
        if response.get("engine") != CLASSIC_ENGINE:
            raise ClassicBridgeProtocolError("classic bridge engine mismatch")
        if response.get("positionHash") != expected_hash:
            raise ClassicBridgeProtocolError(
                "JavaScript and Python position hashes do not match"
            )
        if response.get("historyEntries") != expected_history_entries or response.get(
            "historyOccurrences"
        ) != expected_history_occurrences:
            raise ClassicBridgeProtocolError(
                "classic bridge did not preserve the complete position history"
            )
        response_budget = response.get("budget")
        if not isinstance(response_budget, Mapping) or response_budget.get(
            "kind"
        ) != budget.kind or response_budget.get("limit") != budget.limit:
            raise ClassicBridgeProtocolError("classic bridge budget mismatch")

        action = response.get("action")
        if not isinstance(action, Mapping):
            raise ClassicBridgeProtocolError("classic bridge action must be an object")
        try:
            action_index = encode_action(action)
        except (TypeError, ValueError) as error:
            raise ClassicBridgeProtocolError(
                f"classic bridge returned an invalid action: {error}"
            ) from error
        if action_index not in legal_actions:
            raise ClassicBridgeProtocolError(
                "classic bridge returned an action that is illegal in Python"
            )
        # Recompute after response validation so the caller's State remains the
        # authoritative object throughout the request.
        if position_hash(state) != expected_hash:
            raise ClassicBridgeProtocolError("Python state mutated during bridge request")

        depth = self._non_negative_integer(response.get("depth"), "depth")
        nodes = self._non_negative_integer(response.get("nodes"), "nodes")
        if budget.kind == "nodes" and nodes > budget.limit:
            raise ClassicBridgeProtocolError("classic bridge exceeded its node budget")
        elapsed_ms = self._finite_number(response.get("elapsedMs"), "elapsedMs")
        if elapsed_ms < 0:
            raise ClassicBridgeProtocolError("elapsedMs must be non-negative")
        return ClassicBridgeResult(
            action_index=action_index,
            action=dict(action),
            score=self._finite_number(response.get("score"), "score"),
            depth=depth,
            nodes=nodes,
            elapsed_ms=elapsed_ms,
            position_hash=expected_hash,
            history_entries=expected_history_entries,
            history_occurrences=expected_history_occurrences,
            budget=budget,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pid(self) -> int:
        return self._process.pid

    def _terminate_process(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        stdin = process.stdin
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for thread_name in ("_stdout_thread", "_stderr_thread"):
            thread = getattr(self, thread_name, None)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=0.5)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def close(self) -> None:
        """Close the child process. Repeated calls are intentionally harmless."""

        with self._lock:
            if self._closed:
                return
            try:
                if self._process.poll() is None:
                    request_id = self._next_id()
                    self._send(
                        {
                            "version": BRIDGE_PROTOCOL_VERSION,
                            "type": "close",
                            "id": request_id,
                        }
                    )
                    response = self._receive(min(self._request_timeout, 1.0))
                    self._response_for(response, request_id, "closed")
            except ClassicBridgeError:
                pass
            finally:
                self._terminate_process()
                self._closed = True

    def __enter__(self) -> "ClassicAIBridge":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "CLASSIC_ENGINE",
    "HARD_TIME_LIMIT_MS",
    "HARD_MAX_DEPTH",
    "ClassicBridgeError",
    "ClassicBridgeStartError",
    "ClassicBridgeClosed",
    "ClassicBridgeTimeout",
    "ClassicBridgeCrashed",
    "ClassicBridgeProtocolError",
    "ClassicBridgeRemoteError",
    "ClassicSearchBudget",
    "ClassicBridgeResult",
    "ClassicRootScore",
    "ClassicAnalysisResult",
    "ClassicAIBridge",
]
