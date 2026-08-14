from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from quoridor import (
    apply_action_index,
    legal_action_indices,
    position_hash,
    state_from_serializable,
    state_to_serializable,
)
from quoridor.native_validation import (
    NativeBindingUnavailable,
    NativeEngineAdapter,
    PythonEngineAdapter,
    generate_reachable_samples,
    load_native_engine,
    run_validation,
    write_validation_report,
)


def _python_binding(*, include_history: bool = True) -> ModuleType:
    module = ModuleType("test_quoridor_native")
    module.state_from_dict = state_from_serializable
    module.state_to_dict = state_to_serializable
    module.legal_action_indices = legal_action_indices
    module.apply_action_index = apply_action_index
    module.apply_legal_action_index = apply_action_index
    module.apply_legal_action_position_hash = lambda state, action: position_hash(
        apply_action_index(state, action)
    )
    module.position_hash = position_hash
    if include_history:
        module.state_from_python = lambda state: state
        module.state_to_python = lambda state: state
    return module


class GeneratorTests(unittest.TestCase):
    def test_reachable_samples_are_deterministic_and_legal(self) -> None:
        first = list(generate_reachable_samples(20, seed=12345))
        second = list(generate_reachable_samples(20, seed=12345))

        def signature(samples):
            return [
                (
                    sample.game_index,
                    position_hash(sample.state),
                    sample.action_index,
                    position_hash(sample.next_state),
                )
                for sample in samples
            ]

        self.assertEqual(signature(first), signature(second))
        self.assertEqual([sample.index for sample in first], list(range(20)))
        for sample in first:
            self.assertIn(sample.action_index, sample.legal_actions)
            self.assertEqual(
                apply_action_index(sample.state, sample.action_index),
                sample.next_state,
            )

    def test_generator_rejects_invalid_counts_and_seed(self) -> None:
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    list(generate_reachable_samples(invalid))
        with self.assertRaises(ValueError):
            list(generate_reachable_samples(1, seed=True))


class NativeAdapterTests(unittest.TestCase):
    def test_binding_contract_and_full_history_transfer(self) -> None:
        adapter = NativeEngineAdapter(_python_binding(), "test-native")
        sample = next(generate_reachable_samples(1, seed=9))
        native_state = adapter.from_reference(sample.state)

        self.assertEqual(adapter.name, "test-native")
        self.assertTrue(adapter.preserves_history)
        self.assertEqual(adapter.hash(native_state), position_hash(sample.state))
        self.assertEqual(adapter.legal_actions(native_state), sample.legal_actions)
        self.assertEqual(adapter.to_reference(native_state), sample.state)
        self.assertEqual(
            adapter.to_dict(adapter.apply(native_state, sample.action_index)),
            state_to_serializable(sample.next_state),
        )

    def test_dto_only_binding_is_supported_but_reported(self) -> None:
        adapter = NativeEngineAdapter(
            _python_binding(include_history=False),
            "dto-native",
        )
        state = next(generate_reachable_samples(1)).state
        native_state = adapter.from_reference(state)
        self.assertFalse(adapter.preserves_history)
        self.assertEqual(adapter.to_reference(native_state), state)

    def test_missing_module_and_incomplete_binding_have_clear_errors(self) -> None:
        with self.assertRaisesRegex(
            NativeBindingUnavailable,
            "build the Phase 3 extension",
        ):
            load_native_engine("module_that_cannot_exist_for_quoridor_tests")

        incomplete = ModuleType("incomplete_native")
        incomplete.state_from_dict = lambda dto: dto
        with self.assertRaisesRegex(
            NativeBindingUnavailable,
            "state_to_dict.*legal_action_indices.*apply_action_index.*position_hash",
        ):
            NativeEngineAdapter(incomplete)

        half_history = _python_binding(include_history=False)
        half_history.state_from_python = lambda state: state
        with self.assertRaisesRegex(
            NativeBindingUnavailable,
            "state_from_python and state_to_python together",
        ):
            NativeEngineAdapter(half_history)


class ValidationReportTests(unittest.TestCase):
    def test_python_fallback_exercises_all_gates_and_writes_json(self) -> None:
        report = run_validation(
            PythonEngineAdapter(name="python-candidate"),
            state_count=12,
            games=1,
            seed=123,
            benchmark_sample_limit=4,
            minimum_rule_speedup=0,
            minimum_validation_states=1,
            minimum_long_games=1,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["engine"], "python-candidate")
        self.assertEqual(report["historyTransfer"], "full")
        self.assertEqual(report["fixtures"]["stateCount"], 48)
        self.assertEqual(report["fixtures"]["transitionCount"], 48)
        self.assertTrue(report["fixtures"]["passed"])
        self.assertEqual(report["stateCount"], 12)
        self.assertEqual(report["transitionCount"], 12)
        self.assertEqual(report["illegalActions"], 0)
        self.assertEqual(report["longGames"]["completedGames"], 1)
        self.assertGreater(report["longGames"]["halfMoves"], 0)
        self.assertEqual(report["failureCount"], 0)
        self.assertTrue(all(report["gates"].values()))
        self.assertGreater(report["pythonSeconds"], 0)
        self.assertGreater(report["nativeSeconds"], 0)
        self.assertGreater(report["speedup"], 0)
        self.assertEqual(report["benchmark"]["sampleCount"], 4)
        self.assertTrue(report["benchmark"]["combined"]["checksumMatch"])

        with tempfile.TemporaryDirectory() as directory:
            destination = write_validation_report(
                report,
                Path(directory) / "nested" / "native-validation.json",
            )
            self.assertTrue(destination.exists())
            restored = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(restored, report)

    def test_validation_rejects_non_positive_workloads(self) -> None:
        engine = PythonEngineAdapter()
        for keyword in ("state_count", "games", "benchmark_sample_limit"):
            with self.subTest(keyword=keyword):
                arguments = {
                    "state_count": 1,
                    "games": 1,
                    "benchmark_sample_limit": 1,
                }
                arguments[keyword] = 0
                with self.assertRaises(ValueError):
                    run_validation(engine, **arguments)


if __name__ == "__main__":
    unittest.main()
