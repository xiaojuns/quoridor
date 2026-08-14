from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from quoridor.calibration import (
    CalibrationPreflightError,
    CalibrationScenario,
    GpuBudgetLedger,
    build_calibration_models,
    build_scenarios,
    collect_preflight,
    decision_eligibility,
    run_scenario,
    write_report,
)
from quoridor.game import legal_action_indices
from quoridor.mcts import SearchResult
from quoridor.run_manifest import DEFAULT_MINIMUM_FREE_DISK_BYTES


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int):
        if index != 0:
            raise AssertionError("unexpected device")
        return SimpleNamespace(name="Fake GPU", total_memory=24 * 1024**3)

    def mem_get_info(self, index: int):
        if index != 0:
            raise AssertionError("unexpected device")
        return 20 * 1024**3, 24 * 1024**3


class FakeEvaluator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, states):
        states = list(states)
        self.batch_sizes.append(len(states))
        return [([0.0] * 209, 0.0) for _ in states]

    def enable_native_encoder(self, module: object) -> None:
        pass


class FakeBatchedSearch:
    def __init__(self, evaluator, config) -> None:
        self.evaluator = evaluator
        self.config = config

    @staticmethod
    def _pawn_action(state) -> int:
        pawn_actions = [action for action in legal_action_indices(state) if action < 81]
        if state.current_player == 0:
            return min(pawn_actions, key=lambda action: (action // 9, action % 9))
        return max(pawn_actions, key=lambda action: (action // 9, action % 9))

    def run_batch(self, states, *, temperatures=None, add_root_noise=None):
        states = list(states)
        self.evaluator.evaluate_batch(states)
        results = []
        for state in states:
            action = self._pawn_action(state)
            policy = [0.0] * 209
            policy[action] = 1.0
            visits = [0] * 209
            visits[action] = self.config.simulations
            results.append(
                SearchResult(
                    action_index=action,
                    policy=tuple(policy),
                    visit_counts=tuple(visits),
                    root_value=0.0,
                    root_edges={},
                )
            )
        return tuple(results)


def _completed_result(scenario: CalibrationScenario) -> dict[str, object]:
    moves = scenario.games * 12
    scheduled = moves * scenario.simulations
    return {
        **scenario.to_payload(),
        "status": "completed",
        "elapsedSeconds": 10.0,
        "gamesPerHour": scenario.games * 360.0,
        "moves": moves,
        "movesPerSecond": moves / 10.0,
        "scheduledSimulations": scheduled,
        "scheduledSimulationsPerSecond": scheduled / 10.0,
        "evaluation": {
            "calls": 10,
            "positions": 100,
            "batchSize": {
                "p50": float(scenario.actors),
                "p95": float(scenario.actors),
                "maximum": scenario.actors,
            },
        },
        "gpuUtilization": {
            "available": True,
            "sampleCount": 5,
            "error": None,
            "gpuPercent": {"p50": 80.0, "p95": 95.0},
        },
        "cudaMemory": {
            "peakAllocatedBytes": 1_000_000,
            "peakReservedBytes": 2_000_000,
        },
        "winners": {"player0": scenario.games, "player1": 0},
        "terminationReasons": {"goal": scenario.games},
        "replay": {
            "enabled": True,
            "deepVerified": True,
            "gameCount": scenario.games,
            "plyCount": moves,
            "committedBytes": 1000,
        },
        "rawReplayPliesCollected": moves,
        "trainingSamplesCollected": 0,
        "checkpointsWritten": 0,
    }


class CalibrationProfileTests(unittest.TestCase):
    def test_quick_profile_builds_only_its_declared_network(self) -> None:
        models = build_calibration_models(
            build_scenarios("quick", seed=1234),
            seed=1234,
        )

        self.assertEqual(set(models), {"128x10"})
        self.assertEqual(models["128x10"].config.channels, 128)
        self.assertEqual(models["128x10"].config.blocks, 10)

    def test_full_profile_has_the_required_sixteen_scenarios(self) -> None:
        scenarios = build_scenarios(
            "full",
            seed=17,
            full_games_per_actor=2,
        )
        self.assertEqual(len(scenarios), 16)
        self.assertEqual(
            {
                (scenario.network_name, scenario.actors, scenario.simulations)
                for scenario in scenarios
            },
            {
                (network, actors, simulations)
                for network in ("128x10", "192x16")
                for actors in (8, 16, 32, 64)
                for simulations in (128, 256)
            },
        )
        self.assertTrue(
            all(scenario.games == scenario.actors * 2 for scenario in scenarios)
        )
        for actors in (8, 16, 32, 64):
            for simulations in (128, 256):
                pair = [
                    scenario
                    for scenario in scenarios
                    if scenario.actors == actors
                    and scenario.simulations == simulations
                ]
                self.assertEqual(len(pair), 2)
                self.assertEqual(len({scenario.seed for scenario in pair}), 1)

    def test_only_a_completed_full_matrix_is_decision_eligible(self) -> None:
        full = build_scenarios("full", seed=23)
        results = [_completed_result(scenario) for scenario in full]
        self.assertTrue(
            decision_eligibility(
                "full", full, results, status="completed"
            )["eligible"]
        )
        self.assertFalse(
            decision_eligibility(
                "full", full, results[:-1], status="completed"
            )["eligible"]
        )
        missing_telemetry = [dict(result) for result in results]
        missing_telemetry[0].pop("gpuUtilization")
        self.assertFalse(
            decision_eligibility(
                "full", full, missing_telemetry, status="completed"
            )["eligible"]
        )

        quick = build_scenarios("quick", seed=23)
        quick_result = [_completed_result(quick[0])]
        eligibility = decision_eligibility(
            "quick", quick, quick_result, status="completed"
        )
        self.assertFalse(eligibility["eligible"])
        self.assertIn("quick profile is smoke-only", eligibility["reasons"])

    def test_fake_cpu_scenario_collects_metrics_but_no_training_data(self) -> None:
        scenario = CalibrationScenario(
            network_name="fake",
            channels=8,
            blocks=1,
            actors=2,
            simulations=3,
            games=3,
            seed=29,
        )
        evaluator = FakeEvaluator()
        with tempfile.TemporaryDirectory() as directory:
            result = run_scenario(
                scenario,
                evaluator,
                search_factory=FakeBatchedSearch,
                warmup_batches=1,
                replay_root=Path(directory) / "replay",
                temperature_moves=24,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["temperatureMoves"], 24)
        self.assertEqual(sum(result["winners"].values()), 3)
        self.assertEqual(
            result["scheduledSimulations"],
            result["moves"] * scenario.simulations,
        )
        self.assertGreater(result["scheduledSimulationsPerSecond"], 0)
        self.assertGreater(result["evaluation"]["calls"], 0)
        self.assertGreater(result["evaluation"]["batchSize"]["p50"], 0)
        self.assertGreater(result["evaluation"]["batchSize"]["p95"], 0)
        self.assertEqual(result["trainingSamplesCollected"], 0)
        self.assertEqual(result["checkpointsWritten"], 0)
        self.assertIsNone(result["cudaMemory"]["peakAllocatedBytes"])
        self.assertTrue(result["replay"]["deepVerified"])
        self.assertEqual(result["replay"]["gameCount"], scenario.games)
        self.assertEqual(result["replay"]["plyCount"], result["moves"])


class CalibrationResourceTests(unittest.TestCase):
    def test_gpu_ledger_persists_cumulative_usage_and_external_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = GpuBudgetLedger(path)
            self.assertEqual(ledger.record_external_hours(1.5), 1.5)
            self.assertFalse(ledger.runs)
            self.assertEqual(ledger.begin("run-1", external_hours=1.5), 1.5)
            self.assertEqual(
                ledger.update("run-1", elapsed_seconds=3600, status="completed"),
                2.5,
            )
            reopened = GpuBudgetLedger(path)
            self.assertEqual(reopened.total_hours, 2.5)
            self.assertEqual(reopened.begin("run-2", external_hours=0.5), 2.5)
            self.assertEqual(reopened.external_hours_floor, 1.5)

            payload = json.loads(path.read_text())
            payload["externalHoursFloor"] = 0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationPreflightError, "checksum"):
                GpuBudgetLedger(path)

    def test_preflight_records_fixed_budget_and_fake_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "calibration"
            result = collect_preflight(
                output_root,
                device_index=0,
                gpu_hours_used=1.5,
                cuda_api=FakeCuda(),
                disk_usage=lambda _: SimpleNamespace(
                    free=DEFAULT_MINIMUM_FREE_DISK_BYTES + 1
                ),
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["cuda"]["name"], "Fake GPU")
        self.assertEqual(result["gpuHours"]["limit"], 120.0)
        self.assertEqual(
            result["disk"]["minimumFreeBytes"],
            DEFAULT_MINIMUM_FREE_DISK_BYTES,
        )
        self.assertEqual(result["artifacts"]["limitBytes"], 8 * 1024**3)

    def test_preflight_rejects_disk_below_stop_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CalibrationPreflightError, "20 GiB"):
                collect_preflight(
                    directory,
                    device_index=0,
                    gpu_hours_used=0,
                    cuda_api=FakeCuda(),
                    disk_usage=lambda _: SimpleNamespace(
                        free=DEFAULT_MINIMUM_FREE_DISK_BYTES - 1
                    ),
                )

    def test_report_replace_is_atomic_and_json_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, {"status": "running", "scenarios": []})
            write_report(path, {"status": "completed", "scenarios": [1]})
            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(len(payload["reportSha256"]), 64)
            self.assertFalse(tuple(Path(directory).glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
