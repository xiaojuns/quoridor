from __future__ import annotations

import hashlib
import importlib.machinery
import io
import json
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from quoridor.classic_bridge import (
    CLASSIC_ENGINE,
    ClassicAIBridge,
    ClassicSearchBudget,
)
from quoridor.codec import decode_action, encode_move
from quoridor.game import State, legal_action_indices
from quoridor.openings import (
    Opening,
    OpeningLeg,
    build_opening_suite,
    generate_opening_suite,
)
from quoridor.paired_arena import (
    GATE_BOOTSTRAP_SAMPLES,
    ArenaGatePolicy,
    BoundMoveSelector,
    IllegalSelectionError,
    SelectorContractError,
    SelectorIdentity,
    canonical_classic_backend_hash,
    TerminalResultError,
    evaluate_confirmation_gate,
    evaluate_final_hard_gate,
    evaluate_noninferiority_gate,
    evaluate_screen_gate,
    inspect_selector_runtime,
    loaded_native_backend_hash,
    run_paired_arena,
)
from quoridor.mcts import MCTSConfig
from quoridor.native_mcts import NativeMCTS, NativeMCTSUnavailable
from quoridor.selectors import ClassicHardSelector, MCTSSelector, SelectionResult


class GoalSelector:
    def __init__(self, engine: str) -> None:
        self.engine = engine
        self.calls: list[tuple[int, int]] = []
        self.close_calls = 0

    def choose(self, state: State) -> SelectionResult:
        self.calls.append((state.current_player, state.move_number))
        moves = []
        for action_index in legal_action_indices(state):
            action = decode_action(action_index)
            if action["type"] == "move":
                row = int(action["row"])
                col = int(action["col"])
                progress = row if state.current_player == 0 else -row
                moves.append((progress, abs(col - 4), action_index))
        action_index = min(moves)[2]
        return SelectionResult(
            action_index=action_index,
            engine=self.engine,
            value=0.25,
            score=-3.0,
            simulations=16,
            nodes=23,
            elapsed_ms=1.5,
            depth=2,
            metadata={"nested": {"path": [1, 2]}, "player": state.current_player},
        )

    def close(self) -> None:
        self.close_calls += 1


class IllegalSelector:
    def __init__(self) -> None:
        self.calls = 0
        self.close_calls = 0

    def choose(self, _state: State) -> SelectionResult:
        self.calls += 1
        return SelectionResult(
            action_index=encode_move(0, 0),
            engine="illegal",
            value=None,
            score=None,
            simulations=0,
            nodes=0,
            elapsed_ms=0.0,
            depth=0,
            metadata={},
        )

    def close(self) -> None:
        self.close_calls += 1


class SelectorFailure(RuntimeError):
    pass


class ExplodingSelector:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.close_calls = 0

    def choose(self, _state: State) -> SelectionResult:
        raise self.error

    def close(self) -> None:
        self.close_calls += 1


class InvalidMetadataSelector(GoalSelector):
    def choose(self, state: State) -> SelectionResult:
        result = super().choose(state)
        return SelectionResult(
            action_index=result.action_index,
            engine=result.engine,
            value=result.value,
            score=result.score,
            simulations=result.simulations,
            nodes=result.nodes,
            elapsed_ms=result.elapsed_ms,
            depth=result.depth,
            metadata={"bad": object()},
        )


class MutatingSelector(GoalSelector):
    def __init__(self, engine: str) -> None:
        super().__init__(engine)
        self.clear_was_blocked = False

    def choose(self, state: State) -> SelectionResult:
        result = super().choose(state)
        try:
            state.position_counts.clear()  # type: ignore[attr-defined]
        except AttributeError:
            self.clear_was_blocked = True
        object.__setattr__(state, "position_counts", {})
        return result


class BudgetGoalSelector(GoalSelector):
    def __init__(self, engine: str, *, kind: str, limit: int) -> None:
        super().__init__(engine)
        self.kind = kind
        self.limit = limit

    def choose(self, state: State) -> SelectionResult:
        result = super().choose(state)
        return SelectionResult(
            action_index=result.action_index,
            engine=result.engine,
            value=result.value,
            score=result.score,
            simulations=0,
            nodes=result.nodes,
            elapsed_ms=result.elapsed_ms,
            depth=result.depth,
            metadata={"budgetKind": self.kind, "budgetLimit": self.limit},
        )


class BadPairedOpening(Opening):
    def paired_legs(self) -> tuple[OpeningLeg, OpeningLeg]:
        self.state.position_counts.clear()  # type: ignore[attr-defined]
        player = self.state.current_player
        return (
            OpeningLeg(self.id, 0, player, 1 - player, True),
            OpeningLeg(self.id, 1, player, 1 - player, True),
        )


class ShortWriteSink:
    def write(self, text: str) -> int:
        return len(text) - 1


class UniformEvaluator:
    def evaluate(self, _state: State) -> tuple[list[float], float]:
        return [0.0] * 209, 0.0


class FakeNativeSearch:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


class FakeNativeModule:
    MCTS = FakeNativeSearch


class PairedArenaTests(unittest.TestCase):
    @staticmethod
    def identity(selector_id: str, engine: str) -> SelectorIdentity:
        return SelectorIdentity(
            selector_id=selector_id,
            artifact_hash=hashlib.sha256(selector_id.encode("ascii")).hexdigest(),
            config_hash=hashlib.sha256(
                f"config:{selector_id}".encode("ascii")
            ).hexdigest(),
            source_hash=hashlib.sha256(b"phase4-test-source").hexdigest(),
            engine=engine,
        )

    def test_runs_both_swapped_legs_and_streams_auditable_games(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        candidate = GoalSelector("candidate-test")
        opponent = GoalSelector("opponent-test")
        sink = io.StringIO()

        result = run_paired_arena(
            suite,
            candidate,
            opponent,
            jsonl_sink=sink,
            bootstrap_seed=7,
            bootstrap_samples=100,
        )

        self.assertEqual(len(result.games), 2)
        self.assertEqual([game.leg for game in result.games], [0, 1])
        self.assertEqual([game.candidate_player for game in result.games], [0, 1])
        self.assertEqual([game.candidate_to_move for game in result.games], [True, False])
        self.assertEqual(
            result.games[0].initial_position_hash,
            result.games[1].initial_position_hash,
        )
        self.assertTrue(all(game.winner in (0, 1) for game in result.games))
        self.assertEqual({game.termination_reason for game in result.games}, {"goal"})
        self.assertEqual(result.statistics.games, 2)
        self.assertEqual(result.statistics.opening_pairs, 1)
        self.assertEqual(
            result.statistics.candidate_wins + result.statistics.candidate_losses,
            2,
        )
        self.assertEqual(result.candidate_to_move.games, 1)
        self.assertEqual(result.candidate_not_to_move.games, 1)
        self.assertEqual(result.candidate_as_player_zero.games, 1)
        self.assertEqual(result.candidate_as_player_one.games, 1)
        self.assertEqual(
            result.termination_counts,
            (("goal", 2), ("move-limit", 0), ("repetition", 0)),
        )

        for game in result.games:
            self.assertGreater(len(game.plies), 0)
            self.assertEqual(
                game.final_move_number - game.opening_move_number,
                len(game.plies),
            )
            for ply in game.plies:
                expected_role = (
                    "candidate" if ply.player == game.candidate_player else "opponent"
                )
                self.assertEqual(ply.selector_role, expected_role)
                self.assertEqual(ply.simulations, 16)
                self.assertEqual(ply.nodes, 23)
                self.assertGreaterEqual(ply.wall_time_ms, 0.0)
                self.assertEqual(json.loads(ply.metadata_json)["nested"]["path"], [1, 2])

        self.assertEqual(
            result.candidate_selection.moves + result.opponent_selection.moves,
            sum(len(game.plies) for game in result.games),
        )
        self.assertEqual(
            result.candidate_selection.engines,
            (("candidate-test", result.candidate_selection.moves),),
        )
        self.assertEqual(
            result.opponent_selection.engines,
            (("opponent-test", result.opponent_selection.moves),),
        )
        self.assertEqual(result.candidate_selection.simulations_per_move, 16.0)
        self.assertEqual(result.opponent_selection.nodes_per_move, 23.0)

        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        streamed = [json.loads(line) for line in lines]
        self.assertEqual([entry["leg"] for entry in streamed], [0, 1])
        self.assertEqual(
            streamed[0]["plies"][0]["positionHashBefore"],
            result.games[0].initial_position_hash,
        )
        self.assertIn("wallTimeMs", streamed[0]["plies"][0]["selection"])
        self.assertIn("selectionStats", result.to_dict()["summary"])
        self.assertEqual(candidate.close_calls, 0)
        self.assertEqual(opponent.close_calls, 0)
        self.assertFalse(sink.closed)

        with self.assertRaises(FrozenInstanceError):
            result.games[0].winner = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.suite_name = "final"  # type: ignore[misc]
        changed_game = replace(result.games[0], opening_hash="different-opening")
        with self.assertRaisesRegex(ValueError, "opening manifest"):
            replace(result, games=(changed_game, result.games[1]))

    def test_opening_side_to_move_drives_first_candidate_assignment(self) -> None:
        suite = generate_opening_suite(
            ((encode_move(7, 4),),),
            suite="dev",
        )
        result = run_paired_arena(
            suite,
            GoalSelector("candidate"),
            GoalSelector("opponent"),
            bootstrap_samples=50,
        )

        self.assertEqual(suite.openings[0].state.current_player, 1)
        self.assertEqual([game.candidate_player for game in result.games], [1, 0])
        self.assertEqual([game.candidate_to_move for game in result.games], [True, False])

    def test_illegal_action_fails_before_a_game_is_written_or_closed(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        candidate = IllegalSelector()
        opponent = GoalSelector("opponent")
        sink = io.StringIO()

        with self.assertRaisesRegex(IllegalSelectionError, "illegal action"):
            run_paired_arena(
                suite,
                candidate,
                opponent,
                jsonl_sink=sink,
                bootstrap_samples=10,
            )

        self.assertEqual(candidate.calls, 1)
        self.assertEqual(candidate.close_calls, 0)
        self.assertEqual(opponent.close_calls, 0)
        self.assertEqual(sink.getvalue(), "")

    def test_selector_exception_propagates_unchanged_and_is_not_closed(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        failure = SelectorFailure("injected selector failure")
        candidate = ExplodingSelector(failure)

        with self.assertRaises(SelectorFailure) as caught:
            run_paired_arena(
                suite,
                candidate,
                GoalSelector("opponent"),
                bootstrap_samples=10,
            )

        self.assertIs(caught.exception, failure)
        self.assertEqual(candidate.close_calls, 0)

    def test_verified_participant_identity_is_recorded_and_engine_bound(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        candidate_identity = self.identity("candidate-model", "candidate")
        opponent_identity = self.identity("champion-model", "opponent")
        result = run_paired_arena(
            suite,
            BoundMoveSelector(GoalSelector("candidate"), candidate_identity),
            BoundMoveSelector(GoalSelector("opponent"), opponent_identity),
            bootstrap_samples=10,
        )
        policy = ArenaGatePolicy(
            suite_hash=suite.content_hash,
            candidate_identity=candidate_identity,
            opponent_identity=opponent_identity,
            candidate_backend_hash="a" * 64,
            opponent_backend_hash="b" * 64,
        )

        self.assertEqual(result.candidate_identity, candidate_identity)
        self.assertTrue(result.games[0].candidate_identity.verified)
        self.assertTrue(
            all(
                check.passed
                for check in evaluate_screen_gate(result, policy=policy).checks
                if check.name
                in {"gatePolicy", "suiteHash", "candidateIdentity", "opponentIdentity"}
            )
        )
        self.assertEqual(
            result.to_dict()["candidateIdentity"]["artifactHash"],
            hashlib.sha256(b"candidate-model").hexdigest(),
        )

        with self.assertRaisesRegex(SelectorContractError, "engine mismatch"):
            run_paired_arena(
                suite,
                BoundMoveSelector(
                    GoalSelector("candidate"),
                    self.identity("candidate-model", "wrong-engine"),
                ),
                BoundMoveSelector(GoalSelector("opponent"), opponent_identity),
                bootstrap_samples=10,
            )

    def test_non_json_selector_metadata_is_a_contract_failure(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")

        with self.assertRaisesRegex(SelectorContractError, "non-JSON"):
            run_paired_arena(
                suite,
                InvalidMetadataSelector("candidate"),
                GoalSelector("opponent"),
                bootstrap_samples=10,
            )

    def test_selector_cannot_mutate_referee_or_opening_history(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        opening_counts = dict(suite.openings[0].state.position_counts)
        candidate = MutatingSelector("candidate")

        result = run_paired_arena(
            suite,
            candidate,
            GoalSelector("opponent"),
            bootstrap_samples=10,
        )

        self.assertEqual(len(result.games), 2)
        self.assertTrue(candidate.clear_was_blocked)
        self.assertEqual(suite.openings[0].state.position_counts, opening_counts)
        self.assertEqual(
            result.games[0].initial_position_hash,
            result.games[1].initial_position_hash,
        )

    def test_overridden_pairing_callback_cannot_mutate_validated_opening(self) -> None:
        valid = generate_opening_suite(((),), suite="dev").openings[0]
        bad = BadPairedOpening(
            id=valid.id,
            content_hash=valid.content_hash,
            suite=valid.suite,
            starting_player=valid.starting_player,
            actions=valid.actions,
            state=valid.state,
        )
        suite = build_opening_suite((bad,), suite="dev")
        candidate = GoalSelector("candidate")
        sink = io.StringIO()
        opening_counts = dict(suite.openings[0].state.position_counts)

        result = run_paired_arena(
            suite,
            candidate,
            GoalSelector("opponent"),
            jsonl_sink=sink,
            bootstrap_samples=10,
        )

        self.assertEqual(len(result.games), 2)
        self.assertGreater(len(candidate.calls), 0)
        self.assertEqual(suite.openings[0].state.position_counts, opening_counts)
        self.assertEqual(len(sink.getvalue().splitlines()), 2)

    def test_jsonl_short_write_fails_instead_of_silently_truncating(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")

        with self.assertRaisesRegex(OSError, "short write"):
            run_paired_arena(
                suite,
                GoalSelector("candidate"),
                GoalSelector("opponent"),
                jsonl_sink=ShortWriteSink(),
                bootstrap_samples=10,
            )

    def test_terminal_null_winner_is_rejected(self) -> None:
        suite = generate_opening_suite(((),), suite="dev")
        sink = io.StringIO()

        with patch("quoridor.paired_arena.get_winner", return_value=None):
            with self.assertRaisesRegex(TerminalResultError, "without a winner"):
                run_paired_arena(
                    suite,
                    GoalSelector("candidate"),
                    GoalSelector("opponent"),
                    jsonl_sink=sink,
                    bootstrap_samples=10,
                )

        self.assertEqual(sink.getvalue(), "")

    def test_phase4_gate_functions_apply_workload_and_track_contracts(self) -> None:
        result = run_paired_arena(
            generate_opening_suite(((),), suite="dev"),
            GoalSelector("candidate"),
            GoalSelector("opponent"),
            bootstrap_samples=20,
        )

        self.assertFalse(evaluate_screen_gate(result).passed)
        self.assertIn(
            "simulationsPerMove",
            {check.name for check in evaluate_screen_gate(result).checks},
        )
        confirmation = evaluate_confirmation_gate(result)
        self.assertFalse(confirmation.passed)
        self.assertEqual(
            confirmation.interval_kind,
            "one-sided-95-percent-lower",
        )
        self.assertEqual(confirmation.statistics.win_rate_ci.confidence, 0.90)
        self.assertEqual(
            confirmation.statistics.bootstrap_samples,
            GATE_BOOTSTRAP_SAMPLES,
        )
        self.assertFalse(
            next(check for check in confirmation.checks if check.name == "gatePolicy").passed
        )
        self.assertFalse(evaluate_noninferiority_gate(result).passed)
        with self.assertRaisesRegex(ValueError, "track"):
            evaluate_final_hard_gate(result, track="unknown")
        final = evaluate_final_hard_gate(result, track="wall-clock")
        self.assertFalse(final.passed)
        self.assertIn("maxMoveMs", {check.name for check in final.checks})

    def test_fixed_hard_gate_rejects_self_reported_classic_metadata(self) -> None:
        suite = generate_opening_suite(((),), suite="final")
        candidate_identity = self.identity("candidate-model", "candidate")
        hard_identity = self.identity("classic-hard", CLASSIC_ENGINE)
        result = run_paired_arena(
            suite,
            BoundMoveSelector(GoalSelector("candidate"), candidate_identity),
            BoundMoveSelector(
                BudgetGoalSelector(CLASSIC_ENGINE, kind="nodes", limit=1_000),
                hard_identity,
            ),
            bootstrap_samples=10,
        )
        policy = ArenaGatePolicy(
            suite_hash=suite.content_hash,
            candidate_identity=candidate_identity,
            opponent_identity=hard_identity,
            candidate_backend_hash="a" * 64,
            opponent_backend_hash="b" * 64,
            candidate_simulations=16,
            opponent_budget_kind="nodes",
            opponent_budget_limit=1_000,
        )

        decision = evaluate_final_hard_gate(result, track="fixed", policy=policy)
        checks = {check.name: check for check in decision.checks}
        for name in (
            "gatePolicy",
            "suiteHash",
            "candidateIdentity",
            "opponentIdentity",
            "candidateIdentityBinding",
            "opponentIdentityBinding",
            "opponentEngine",
            "trackWorkloadPolicy",
            "candidateSimulationsPerMove",
            "opponentSimulationsPerMove",
            "opponentBudget",
            "opponentNodeCount",
        ):
            self.assertTrue(checks[name].passed, name)
        self.assertFalse(checks["candidateRuntime"].passed)
        self.assertFalse(checks["opponentRuntime"].passed)
        self.assertFalse(decision.passed)

    def test_injected_native_module_and_classic_command_are_untrusted(self) -> None:
        neural = MCTSSelector(
            UniformEvaluator(),
            MCTSConfig(simulations=16, temperature=0.0, add_root_noise=False),
            search_factory=lambda evaluator, config: NativeMCTS(
                evaluator,
                config,
                module=FakeNativeModule(),  # type: ignore[arg-type]
            ),
        )
        self.assertEqual(inspect_selector_runtime(neural).kind, "untrusted")
        neural.close()

        ready = {
            "version": 1,
            "type": "ready",
            "engine": CLASSIC_ENGINE,
            "hardDefaults": {"timeLimitMs": 850, "maxDepth": 5},
        }
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_bridge.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import sys
                    print(json.dumps({ready!r}), flush=True)
                    for _line in sys.stdin:
                        pass
                    """
                ),
                encoding="utf-8",
            )
            bridge = ClassicAIBridge(command=(sys.executable, "-u", str(script)))
            selector = ClassicHardSelector(
                bridge=bridge,
                budget=ClassicSearchBudget.nodes(1_000),
            )
            self.assertEqual(inspect_selector_runtime(selector).kind, "untrusted")
            selector.close()
            bridge.close()

    def test_python_module_cannot_impersonate_native_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quoridor_native.py"
            path.write_text("# fake native backend\n", encoding="ascii")
            module = types.ModuleType("quoridor_native")
            loader = importlib.machinery.SourceFileLoader(
                "quoridor_native",
                str(path),
            )
            module.__file__ = str(path)
            module.__loader__ = loader
            module.__spec__ = importlib.machinery.ModuleSpec(
                "quoridor_native",
                loader,
                origin=str(path),
            )

            with patch(
                "quoridor.paired_arena.load_native_module",
                return_value=module,
            ):
                with self.assertRaisesRegex(
                    NativeMCTSUnavailable,
                    "native extension",
                ):
                    loaded_native_backend_hash()

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_canonical_classic_process_has_a_bound_backend_digest(self) -> None:
        selector = ClassicHardSelector(budget=ClassicSearchBudget.nodes(1_000))
        try:
            runtime = inspect_selector_runtime(selector)
            self.assertEqual(runtime.kind, "classic-hard")
            self.assertTrue(runtime.trusted)
            self.assertEqual(runtime.backend_hash, canonical_classic_backend_hash())
        finally:
            selector.close()


if __name__ == "__main__":
    unittest.main()
