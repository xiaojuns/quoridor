from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path

from quoridor.classic_bridge import (
    CLASSIC_ENGINE,
    ClassicAIBridge,
    ClassicBridgeClosed,
    ClassicBridgeCrashed,
    ClassicBridgeProtocolError,
    ClassicBridgeResult,
    ClassicBridgeTimeout,
    ClassicSearchBudget,
)
from quoridor.codec import decode_action, encode_move, position_hash
from quoridor.game import (
    State,
    apply_action_index,
    create_initial_state,
    legal_action_indices,
)
from quoridor.mcts import MCTSConfig
from quoridor.selectors import (
    ClassicHardSelector,
    MCTSSelector,
    SelectorClosed,
    SelectorError,
)


READY = {
    "version": 1,
    "type": "ready",
    "engine": "classic-negamax",
    "hardDefaults": {"timeLimitMs": 850, "maxDepth": 5},
}


def repeated_state() -> State:
    state = create_initial_state()
    for action in (
        encode_move(8, 3),
        encode_move(0, 3),
        encode_move(8, 4),
        encode_move(0, 4),
    ):
        state = apply_action_index(state, action)
    return state


def fake_command(directory: str, body: str) -> tuple[str, ...]:
    script = Path(directory) / "fake_bridge.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return (sys.executable, "-u", str(script))


@unittest.skipUnless(shutil.which("node"), "Node.js is required for bridge integration")
class ClassicBridgeIntegrationTests(unittest.TestCase):
    def test_persistent_node_bridge_supports_node_and_default_time_budgets(self) -> None:
        state = repeated_state()
        with ClassicAIBridge() as bridge:
            pid = bridge.pid
            first = bridge.choose(state, budget=ClassicSearchBudget.nodes(48))
            second = bridge.choose(state, budget=ClassicSearchBudget.nodes(48))

            self.assertEqual(bridge.pid, pid)
            self.assertEqual(first.action_index, second.action_index)
            self.assertEqual(first.score, second.score)
            self.assertEqual(first.depth, second.depth)
            self.assertEqual(first.nodes, 48)
            self.assertEqual(second.nodes, 48)
            self.assertIn(first.action_index, legal_action_indices(state))
            self.assertEqual(first.position_hash, position_hash(state))
            self.assertEqual(first.history_entries, len(state.position_counts))
            self.assertEqual(
                first.history_occurrences,
                sum(state.position_counts.values()),
            )

            near_win = State(pawns=((1, 4), (0, 0)), current_player=0)
            near_win = replace(
                near_win,
                position_counts={position_hash(near_win): 1},
            )
            timed = bridge.choose(near_win)
            self.assertEqual(timed.budget, ClassicSearchBudget.time(850))
            self.assertEqual(timed.action_index, encode_move(0, 4))
            self.assertEqual(timed.engine, CLASSIC_ENGINE)

        bridge.close()
        self.assertTrue(bridge.closed)
        with self.assertRaises(ClassicBridgeClosed):
            bridge.choose(state, budget=ClassicSearchBudget.nodes(1))

    def test_truncated_history_is_rejected_before_sending(self) -> None:
        bridge = ClassicAIBridge()
        try:
            truncated = replace(create_initial_state(), position_counts={})
            with self.assertRaisesRegex(ValueError, "complete state/history"):
                bridge.choose(truncated, budget=ClassicSearchBudget.nodes(1))
            self.assertFalse(bridge.closed)
        finally:
            bridge.close()


class ClassicBridgeFailureTests(unittest.TestCase):
    def test_timeout_terminates_desynchronized_child_and_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = fake_command(
                directory,
                f"""
                import json
                import sys
                import time
                print(json.dumps({READY!r}), flush=True)
                sys.stdin.readline()
                time.sleep(10)
                """,
            )
            bridge = ClassicAIBridge(command=command, request_timeout=0.05)
            with self.assertRaises(ClassicBridgeTimeout):
                bridge.choose(create_initial_state())
            self.assertTrue(bridge.closed)
            bridge.close()
            bridge.close()

    def test_crash_and_protocol_mismatch_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            crash = fake_command(
                directory,
                f"""
                import json
                import sys
                print(json.dumps({READY!r}), flush=True)
                sys.stdin.readline()
                raise SystemExit(7)
                """,
            )
            crashed = ClassicAIBridge(command=crash)
            with self.assertRaises(ClassicBridgeCrashed):
                crashed.choose(create_initial_state())
            crashed.close()

        with tempfile.TemporaryDirectory() as directory:
            wrong_id = fake_command(
                directory,
                f"""
                import json
                import sys
                print(json.dumps({READY!r}), flush=True)
                request = json.loads(sys.stdin.readline())
                print(json.dumps({{
                    "version": 1,
                    "type": "result",
                    "id": request["id"] + 1,
                }}), flush=True)
                """,
            )
            malformed = ClassicAIBridge(command=wrong_id)
            with self.assertRaisesRegex(ClassicBridgeProtocolError, "id mismatch"):
                malformed.choose(create_initial_state())
            self.assertTrue(malformed.closed)
            malformed.close()

    def test_illegal_remote_action_fails_python_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            illegal_action = fake_command(
                directory,
                f"""
                import json
                import sys
                print(json.dumps({READY!r}), flush=True)
                request = json.loads(sys.stdin.readline())
                counts = request["state"]["positionCounts"]
                budget = request["budget"]
                limit = budget.get("timeLimitMs", budget.get("nodeLimit"))
                print(json.dumps({{
                    "version": 1,
                    "type": "result",
                    "id": request["id"],
                    "engine": "classic-negamax",
                    "positionHash": request["positionHash"],
                    "historyEntries": len(counts),
                    "historyOccurrences": sum(item["count"] for item in counts),
                    "budget": {{"kind": budget["kind"], "limit": limit}},
                    "action": {{"type": "move", "row": 0, "col": 0}},
                    "score": 0,
                    "depth": 0,
                    "nodes": 0,
                    "elapsedMs": 0,
                }}), flush=True)
                """,
            )
            bridge = ClassicAIBridge(command=illegal_action)
            with self.assertRaisesRegex(ClassicBridgeProtocolError, "illegal in Python"):
                bridge.choose(create_initial_state())
            self.assertTrue(bridge.closed)
            bridge.close()


class UniformEvaluator:
    def evaluate(self, _state: State):
        return [0.0] * 209, 0.0


class FakeClassicBridge:
    def __init__(self) -> None:
        self.calls = []

    def choose(self, state, *, budget, max_depth, timeout):
        self.calls.append((state, budget, max_depth, timeout))
        action_index = legal_action_indices(state)[0]
        return ClassicBridgeResult(
            action_index=action_index,
            action=decode_action(action_index),
            score=12.0,
            depth=3,
            nodes=99,
            elapsed_ms=4.5,
            position_hash=position_hash(state),
            history_entries=len(state.position_counts),
            history_occurrences=sum(state.position_counts.values()),
            budget=budget,
        )

    def close(self) -> None:
        raise AssertionError("injected bridge ownership belongs to the caller")


class SelectorTests(unittest.TestCase):
    def test_classic_selector_normalizes_result_and_is_idempotently_closed(self) -> None:
        bridge = FakeClassicBridge()
        budget = ClassicSearchBudget.nodes(99)
        selector = ClassicHardSelector(
            bridge=bridge,  # type: ignore[arg-type]
            budget=budget,
            max_depth=4,
            request_timeout=2.0,
        )
        state = create_initial_state()

        result = selector.choose(state)
        self.assertIn(result.action_index, legal_action_indices(state))
        self.assertEqual(result.engine, CLASSIC_ENGINE)
        self.assertEqual(result.nodes, 99)
        self.assertEqual(result.metadata["budgetKind"], "nodes")
        self.assertEqual(bridge.calls[0][1:], (budget, 4, 2.0))

        selector.close()
        selector.close()
        with self.assertRaises(SelectorClosed):
            selector.choose(state)

    def test_mcts_selector_forces_evaluation_mode_and_validates_action(self) -> None:
        selector = MCTSSelector(
            UniformEvaluator(),
            MCTSConfig(
                simulations=2,
                add_root_noise=True,
                temperature=1.0,
                seed=5,
            ),
        )
        state = create_initial_state()
        result = selector.choose(state)

        self.assertIn(result.action_index, legal_action_indices(state))
        self.assertEqual(result.simulations, 2)
        self.assertEqual(result.nodes, 2)
        self.assertEqual(result.metadata["configuredSimulations"], 2)
        self.assertEqual(selector.config.temperature, 0.0)
        self.assertFalse(selector.config.add_root_noise)

        selector.close()
        selector.close()
        with self.assertRaises(SelectorClosed):
            selector.choose(state)

    def test_classic_selector_session_facade_tracks_lifecycle(self) -> None:
        selector = ClassicHardSelector(
            bridge=FakeClassicBridge(),  # type: ignore[arg-type]
            budget=ClassicSearchBudget.nodes(99),
        )
        state = create_initial_state()
        first = selector.create_session(state)
        second = selector.create_session(state)
        results = selector.choose_sessions((first, second), (state, state))

        self.assertEqual(len(results), 2)
        self.assertTrue(selector.advance_session(first, results[0].action_index, state))
        self.assertTrue(selector.release_session(first))
        self.assertFalse(selector.release_session(first))
        with self.assertRaises(SelectorError):
            selector.choose_sessions((first,), (state,))
        selector.close()
        with self.assertRaises(SelectorClosed):
            selector.release_session(second)


if __name__ == "__main__":
    unittest.main()
