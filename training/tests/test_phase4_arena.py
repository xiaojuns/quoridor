from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from quoridor.codec import ACTION_SPACE_SIZE, decode_action, position_hash
from quoridor.classic_bridge import ClassicBridgeResult, ClassicSearchBudget
from quoridor.game import State, apply_action_index, legal_action_indices
from quoridor.mcts import MCTSConfig, SearchResult
from quoridor.openings import generate_opening_suite
from quoridor.phase4_arena import (
    OpeningPairStore,
    Phase4ArenaCorruptionError,
    run_resumable_paired_arena,
    verify_completed_arena_result,
)
from quoridor.paired_arena import (
    _play_leg,
    _resolve_participant,
    _snapshot_opening,
    _validated_legs,
)
from quoridor.run_manifest import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from quoridor.selectors import (
    ClassicHardSelector,
    MCTS_ROOT_VISIT_SEMANTICS,
    MCTSSelector,
    SelectionResult,
)


class GoalClassicBridge:
    def choose(self, state, *, budget, max_depth, timeout):
        del timeout
        action_index = _goal_action(state)
        return ClassicBridgeResult(
            action_index=action_index,
            action=decode_action(action_index),
            score=0.0,
            depth=max_depth,
            nodes=min(32, budget.limit),
            elapsed_ms=0.1,
            position_hash=position_hash(state),
            history_entries=len(state.position_counts),
            history_occurrences=sum(state.position_counts.values()),
            budget=budget,
        )

    def close(self) -> None:
        pass


def _goal_action(state: State) -> int:
    moves = []
    for action_index in legal_action_indices(state):
        action = decode_action(action_index)
        if action["type"] != "move":
            continue
        row = int(action["row"])
        col = int(action["col"])
        progress = row if state.current_player == 0 else -row
        moves.append((progress, abs(col - 4), action_index))
    return min(moves)[2]


class GoalSelector:
    def __init__(self, engine: str) -> None:
        self.engine = engine
        self.calls = 0

    def choose(self, state: State) -> SelectionResult:
        self.calls += 1
        return SelectionResult(
            action_index=_goal_action(state),
            engine=self.engine,
            value=0.0,
            score=None,
            simulations=2,
            nodes=2,
            elapsed_ms=0.1,
            depth=0,
            metadata={},
        )

    def close(self) -> None:
        pass


class FailingSelector(GoalSelector):
    def choose(self, state: State) -> SelectionResult:
        del state
        raise RuntimeError("injected interruption")


def _result(action_index: int, simulations: int) -> SearchResult:
    policy = [0.0] * ACTION_SPACE_SIZE
    visits = [0] * ACTION_SPACE_SIZE
    policy[action_index] = 1.0
    visits[action_index] = simulations
    return SearchResult(
        action_index=action_index,
        policy=tuple(policy),
        visit_counts=tuple(visits),
        root_value=0.0,
        root_edges={},
    )


class SessionSearch:
    instances: list["SessionSearch"] = []

    def __init__(self, _evaluator: object, config: MCTSConfig) -> None:
        self.simulations = config.simulations
        self.next_id = 0
        self.states: dict[int, State] = {}
        self.decisions: dict[int, int] = {}
        self.batch_sizes: list[int] = []
        self.released: list[int] = []
        type(self).instances.append(self)

    def create_session(self, state: State) -> int:
        session_id = self.next_id
        self.next_id += 1
        self.states[session_id] = state
        self.decisions[session_id] = 0
        return session_id

    def run_sessions(
        self,
        session_ids,
        states,
        *,
        temperatures,
        add_root_noise,
    ):
        self.batch_sizes.append(len(states))
        self.assertEqualCompat(temperatures, [0.0] * len(states))
        self.assertEqualCompat(add_root_noise, False)
        output = []
        for session_id, state in zip(session_ids, states):
            self.assertEqualCompat(self.states[session_id], state)
            retained_visits = self.decisions[session_id]
            output.append(
                _result(
                    _goal_action(state),
                    self.simulations + retained_visits,
                )
            )
            self.decisions[session_id] += 1
        return tuple(output)

    @staticmethod
    def assertEqualCompat(left: object, right: object) -> None:
        if left != right:
            raise AssertionError(f"{left!r} != {right!r}")

    def advance_session(
        self,
        session_id: int,
        action_index: int,
        next_state: State,
    ) -> bool:
        self.assertEqualCompat(
            apply_action_index(self.states[session_id], action_index),
            next_state,
        )
        self.states[session_id] = next_state
        return True

    def release_session(self, session_id: int) -> bool:
        self.released.append(session_id)
        self.decisions.pop(session_id, None)
        return self.states.pop(session_id, None) is not None


class UniformEvaluator:
    def evaluate(self, _state: State):
        return [0.0] * ACTION_SPACE_SIZE, 0.0


class Phase4ArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = generate_opening_suite(
            [[], [_goal_action(generate_opening_suite([[]], suite="dev").openings[0].state)]],
            suite="dev",
        )
        self.contract_sha256 = hashlib.sha256(b"phase4e-arena").hexdigest()

    def _finalized_single_pair(self, root: Path) -> Path:
        suite = generate_opening_suite([[]], suite="dev")
        store = run_resumable_paired_arena(
            suite,
            GoalSelector("candidate"),
            GoalSelector("opponent"),
            store_root=root,
            contract_sha256=self.contract_sha256,
        )
        store.finalize(bootstrap_seed=7, bootstrap_samples=100)
        return root / "result.json"

    @staticmethod
    def _rewrite_first_pair(
        result_path: Path,
        mutate,
    ) -> None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reference = result["pairArtifacts"][0]
        old_path = result_path.parent / reference["file"]
        pair = json.loads(old_path.read_text(encoding="utf-8"))
        mutate(pair, result)
        pair_core = dict(pair)
        pair_core.pop("pairSha256", None)
        pair_sha256 = sha256_bytes(canonical_json_bytes(pair_core))
        new_path = (
            result_path.parent
            / (
                f"pair-{reference['ordinal']:06d}-{reference['openingId']}-"
                f"{pair_sha256[:12]}.json"
            )
        )
        new_path.write_bytes(
            canonical_json_bytes(
                {**pair_core, "pairSha256": pair_sha256}
            )
        )
        if new_path != old_path:
            old_path.unlink()
        reference.update(
            {
                "file": new_path.name,
                "sha256": sha256_file(new_path),
                "bytes": new_path.stat().st_size,
            }
        )
        result_core = dict(result)
        result_core.pop("resultSha256", None)
        result_path.write_bytes(
            canonical_json_bytes(
                {
                    **result_core,
                    "resultSha256": sha256_bytes(
                        canonical_json_bytes(result_core)
                    ),
                }
            )
        )

    def test_serial_pairs_are_atomic_resumable_and_self_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = GoalSelector("candidate")
            opponent = GoalSelector("opponent")
            store = run_resumable_paired_arena(
                self.suite,
                candidate,
                opponent,
                store_root=temporary,
                contract_sha256=self.contract_sha256,
            )
            self.assertEqual(store.completed_pairs, 2)
            result = store.finalize(bootstrap_seed=7, bootstrap_samples=100)
            self.assertEqual(result["games"], 4)
            self.assertEqual(result["openingPairs"], 2)
            self.assertEqual(
                verify_completed_arena_result(
                    Path(temporary) / "result.json"
                ),
                result,
            )
            suite_snapshot = Path(temporary) / "opening-suite.json"
            self.assertTrue(suite_snapshot.is_file())
            self.assertEqual(
                result["openingSuite"]["sha256"],
                sha256_file(suite_snapshot),
            )
            calls = candidate.calls + opponent.calls

            resumed_candidate = GoalSelector("candidate")
            resumed_opponent = GoalSelector("opponent")
            resumed = run_resumable_paired_arena(
                self.suite,
                resumed_candidate,
                resumed_opponent,
                store_root=temporary,
                contract_sha256=self.contract_sha256,
            )
            self.assertEqual(resumed.completed_pairs, 2)
            self.assertEqual(resumed_candidate.calls + resumed_opponent.calls, 0)
            self.assertGreater(calls, 0)

    def test_interrupted_leg_is_not_a_committed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                run_resumable_paired_arena(
                    self.suite,
                    FailingSelector("candidate"),
                    GoalSelector("opponent"),
                    store_root=temporary,
                    contract_sha256=self.contract_sha256,
                )
            progress = json.loads(
                (Path(temporary) / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["completedPairs"], 0)
            self.assertEqual(list(Path(temporary).glob("pair-*.json")), [])

    def test_pair_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = run_resumable_paired_arena(
                self.suite,
                GoalSelector("candidate"),
                GoalSelector("opponent"),
                store_root=temporary,
                contract_sha256=self.contract_sha256,
            )
            pair = next(Path(temporary).glob("pair-*.json"))
            payload = json.loads(pair.read_text(encoding="utf-8"))
            payload["games"][0]["winner"] = 1 - payload["games"][0]["winner"]
            pair.write_text(
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Phase4ArenaCorruptionError):
                OpeningPairStore(
                    temporary,
                    contract_sha256=self.contract_sha256,
                    suite=self.suite,
                    candidate_identity=store.candidate_identity,
                    opponent_identity=store.opponent_identity,
                    candidate_runtime=store.candidate_runtime,
                    opponent_runtime=store.opponent_runtime,
                )

    def test_session_selectors_use_cross_game_batches_and_release_sessions(self) -> None:
        SessionSearch.instances.clear()
        config = MCTSConfig(
            simulations=2,
            temperature=0.0,
            add_root_noise=False,
            seed=11,
        )
        candidate = MCTSSelector(
            UniformEvaluator(),
            config,
            search_factory=SessionSearch,
            engine="candidate-mcts",
        )
        opponent = MCTSSelector(
            UniformEvaluator(),
            config,
            search_factory=SessionSearch,
            engine="opponent-mcts",
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = run_resumable_paired_arena(
                self.suite,
                candidate,
                opponent,
                store_root=temporary,
                contract_sha256=self.contract_sha256,
                actors=4,
            )
            self.assertEqual(store.completed_pairs, 2)
            selections = [
                ply["selection"]
                for ordinal in store.completed_ordinals
                for game in store.pair_payload(ordinal)["games"]
                for ply in game["plies"]
            ]
            self.assertTrue(any(selection["nodes"] > 2 for selection in selections))
            for selection in selections:
                self.assertEqual(selection["simulations"], 2)
                self.assertEqual(
                    selection["nodes"],
                    selection["metadata"]["rootVisits"],
                )
                self.assertEqual(
                    selection["metadata"]["retainedRootVisits"],
                    selection["nodes"] - selection["simulations"],
                )
                self.assertEqual(
                    selection["metadata"]["visitCountSemantics"],
                    MCTS_ROOT_VISIT_SEMANTICS,
                )
        self.assertEqual(len(SessionSearch.instances), 2)
        self.assertTrue(
            any(
                batch_size > 1
                for search in SessionSearch.instances
                for batch_size in search.batch_sizes
            )
        )
        self.assertTrue(all(not search.states for search in SessionSearch.instances))
        self.assertTrue(all(not search.decisions for search in SessionSearch.instances))

    def test_mixed_native_style_and_classic_sessions_keep_neural_batches(self) -> None:
        SessionSearch.instances.clear()
        candidate = MCTSSelector(
            UniformEvaluator(),
            MCTSConfig(simulations=2, temperature=0.0, add_root_noise=False),
            search_factory=SessionSearch,
            engine="candidate-mcts",
        )
        opponent = ClassicHardSelector(
            bridge=GoalClassicBridge(),  # type: ignore[arg-type]
            budget=ClassicSearchBudget.nodes(32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = run_resumable_paired_arena(
                self.suite,
                candidate,
                opponent,
                store_root=temporary,
                contract_sha256=self.contract_sha256,
                actors=4,
            )
            self.assertEqual(store.completed_pairs, 2)
        self.assertTrue(
            any(size > 1 for size in SessionSearch.instances[0].batch_sizes)
        )
        self.assertFalse(SessionSearch.instances[0].states)

    def test_invalid_pair_is_fully_validated_before_durable_write(self) -> None:
        suite = generate_opening_suite([[]], suite="dev")
        config = MCTSConfig(
            simulations=2,
            temperature=0.0,
            add_root_noise=False,
            seed=17,
        )
        candidate = MCTSSelector(
            UniformEvaluator(),
            config,
            engine="candidate-mcts",
        )
        opponent = MCTSSelector(
            UniformEvaluator(),
            config,
            engine="opponent-mcts",
        )
        candidate_identity, candidate_runtime = _resolve_participant(
            candidate, None, "candidate"
        )
        opponent_identity, opponent_runtime = _resolve_participant(
            opponent, None, "opponent"
        )
        opening = _snapshot_opening(suite.openings[0])
        games = tuple(
            _play_leg(
                suite,
                opening,
                leg,
                candidate,
                opponent,
                candidate_identity,
                opponent_identity,
                candidate_runtime,
                opponent_runtime,
            )
            for leg in _validated_legs(opening)
        )
        bad_ply = replace(games[0].plies[0], metadata_json="{}")
        invalid_games = (
            replace(games[0], plies=(bad_ply, *games[0].plies[1:])),
            games[1],
        )

        try:
            with tempfile.TemporaryDirectory() as temporary:
                store = OpeningPairStore(
                    temporary,
                    contract_sha256=self.contract_sha256,
                    suite=suite,
                    candidate_identity=candidate_identity,
                    opponent_identity=opponent_identity,
                    candidate_runtime=candidate_runtime,
                    opponent_runtime=opponent_runtime,
                )
                with self.assertRaisesRegex(
                    Phase4ArenaCorruptionError,
                    "MCTS metadata fields are invalid",
                ):
                    store.commit_pair(0, invalid_games)
                self.assertEqual(
                    list(Path(temporary).glob("pair-*.json")),
                    [],
                )
                self.assertEqual(store.completed_pairs, 0)
        finally:
            candidate.close()
            opponent.close()

    def test_durable_pair_is_reread_and_uncommitted_corruption_is_removed(
        self,
    ) -> None:
        suite = generate_opening_suite([[]], suite="dev")
        candidate = GoalSelector("candidate")
        opponent = GoalSelector("opponent")
        candidate_identity, candidate_runtime = _resolve_participant(
            candidate, None, "candidate"
        )
        opponent_identity, opponent_runtime = _resolve_participant(
            opponent, None, "opponent"
        )
        opening = _snapshot_opening(suite.openings[0])
        games = tuple(
            _play_leg(
                suite,
                opening,
                leg,
                candidate,
                opponent,
                candidate_identity,
                opponent_identity,
                candidate_runtime,
                opponent_runtime,
            )
            for leg in _validated_legs(opening)
        )

        with tempfile.TemporaryDirectory() as temporary:
            store = OpeningPairStore(
                temporary,
                contract_sha256=self.contract_sha256,
                suite=suite,
                candidate_identity=candidate_identity,
                opponent_identity=opponent_identity,
                candidate_runtime=candidate_runtime,
                opponent_runtime=opponent_runtime,
            )

            def corrupt_create(path: Path, payload: bytes) -> None:
                path.write_bytes(payload + b"x")

            with patch(
                "quoridor.phase4_arena.atomic_create_bytes",
                side_effect=corrupt_create,
            ):
                with self.assertRaises(Phase4ArenaCorruptionError):
                    store.commit_pair(0, games)

            self.assertEqual(list(Path(temporary).glob("pair-*.json")), [])
            self.assertEqual(store.completed_pairs, 0)

    def test_stale_store_cannot_commit_a_conflicting_pair(self) -> None:
        suite = generate_opening_suite([[]], suite="dev")
        candidate = GoalSelector("candidate")
        opponent = GoalSelector("opponent")
        candidate_identity, candidate_runtime = _resolve_participant(
            candidate, None, "candidate"
        )
        opponent_identity, opponent_runtime = _resolve_participant(
            opponent, None, "opponent"
        )
        arguments = {
            "contract_sha256": self.contract_sha256,
            "suite": suite,
            "candidate_identity": candidate_identity,
            "opponent_identity": opponent_identity,
            "candidate_runtime": candidate_runtime,
            "opponent_runtime": opponent_runtime,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first = OpeningPairStore(temporary, **arguments)
            stale = OpeningPairStore(temporary, **arguments)
            opening = _snapshot_opening(suite.openings[0])
            games = tuple(
                _play_leg(
                    suite,
                    opening,
                    leg,
                    candidate,
                    opponent,
                    candidate_identity,
                    opponent_identity,
                    candidate_runtime,
                    opponent_runtime,
                )
                for leg in _validated_legs(opening)
            )
            changed_first_ply = replace(
                games[0].plies[0],
                wall_time_ms=games[0].plies[0].wall_time_ms + 1.0,
            )
            conflicting = (
                replace(
                    games[0],
                    plies=(changed_first_ply, *games[0].plies[1:]),
                ),
                games[1],
            )

            first.commit_pair(0, games)
            with self.assertRaisesRegex(
                Phase4ArenaCorruptionError,
                "claimed by another runner",
            ):
                stale.commit_pair(0, conflicting)
            self.assertEqual(
                len(list(Path(temporary).glob("pair-*.json"))),
                1,
            )
            recovered = OpeningPairStore(temporary, **arguments)
            self.assertEqual(recovered.completed_pairs, 1)

    def test_incremental_commit_does_not_replay_known_pair(self) -> None:
        candidate = GoalSelector("candidate")
        opponent = GoalSelector("opponent")
        candidate_identity, candidate_runtime = _resolve_participant(
            candidate, None, "candidate"
        )
        opponent_identity, opponent_runtime = _resolve_participant(
            opponent, None, "opponent"
        )
        arguments = {
            "contract_sha256": self.contract_sha256,
            "suite": self.suite,
            "candidate_identity": candidate_identity,
            "opponent_identity": opponent_identity,
            "candidate_runtime": candidate_runtime,
            "opponent_runtime": opponent_runtime,
        }
        games = []
        for source in self.suite.openings:
            opening = _snapshot_opening(source)
            games.append(
                tuple(
                    _play_leg(
                        self.suite,
                        opening,
                        leg,
                        candidate,
                        opponent,
                        candidate_identity,
                        opponent_identity,
                        candidate_runtime,
                        opponent_runtime,
                    )
                    for leg in _validated_legs(opening)
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            store = OpeningPairStore(temporary, **arguments)
            store.commit_pair(0, games[0])
            with patch.object(
                store,
                "_validate_pair_payload",
                wraps=store._validate_pair_payload,
            ) as validate:
                store.commit_pair(1, games[1])
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(store.completed_pairs, 2)

    def test_incremental_commit_rejects_changed_known_pair(self) -> None:
        candidate = GoalSelector("candidate")
        opponent = GoalSelector("opponent")
        candidate_identity, candidate_runtime = _resolve_participant(
            candidate, None, "candidate"
        )
        opponent_identity, opponent_runtime = _resolve_participant(
            opponent, None, "opponent"
        )
        arguments = {
            "contract_sha256": self.contract_sha256,
            "suite": self.suite,
            "candidate_identity": candidate_identity,
            "opponent_identity": opponent_identity,
            "candidate_runtime": candidate_runtime,
            "opponent_runtime": opponent_runtime,
        }
        games = []
        for source in self.suite.openings:
            opening = _snapshot_opening(source)
            games.append(
                tuple(
                    _play_leg(
                        self.suite,
                        opening,
                        leg,
                        candidate,
                        opponent,
                        candidate_identity,
                        opponent_identity,
                        candidate_runtime,
                        opponent_runtime,
                    )
                    for leg in _validated_legs(opening)
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            store = OpeningPairStore(temporary, **arguments)
            artifact = store.commit_pair(0, games[0])
            path = Path(temporary) / artifact.file_name
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(
                Phase4ArenaCorruptionError,
                "changed after validation",
            ):
                store.commit_pair(1, games[1])

    def test_completed_verifier_rejects_rehashed_missing_plies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self._finalized_single_pair(Path(temporary))

            def remove_plies(pair, _result) -> None:
                pair["games"][0]["plies"] = []

            self._rewrite_first_pair(result_path, remove_plies)
            with self.assertRaisesRegex(
                Phase4ArenaCorruptionError,
                "no plies",
            ):
                verify_completed_arena_result(result_path)

    def test_completed_verifier_rejects_rehashed_simulation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self._finalized_single_pair(Path(temporary))

            def change_simulations(pair, result) -> None:
                runtime = {
                    "implementation": "quoridor.selectors.MCTSSelector",
                    "kind": "python-mcts",
                    "trusted": False,
                    "configuredSimulations": 2,
                    "budgetKind": None,
                    "budgetLimit": None,
                    "maxDepth": None,
                    "identityBound": False,
                    "backendHash": None,
                }
                result["candidateRuntime"] = runtime
                result["opponentRuntime"] = runtime
                for game in pair["games"]:
                    game["candidateRuntime"] = runtime
                    game["opponentRuntime"] = runtime
                pair["games"][0]["plies"][0]["selection"]["simulations"] = 1

            self._rewrite_first_pair(result_path, change_simulations)
            with self.assertRaisesRegex(
                Phase4ArenaCorruptionError,
                "simulations differ",
            ):
                verify_completed_arena_result(result_path)

    def test_completed_verifier_rejects_suite_snapshot_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self._finalized_single_pair(Path(temporary))
            suite_path = Path(temporary) / "opening-suite.json"
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            suite["suiteHash"] = f"sha256:{'0' * 64}"
            suite_path.write_bytes(canonical_json_bytes(suite))
            with self.assertRaisesRegex(
                Phase4ArenaCorruptionError,
                "snapshot reference",
            ):
                verify_completed_arena_result(result_path)


if __name__ == "__main__":
    unittest.main()
