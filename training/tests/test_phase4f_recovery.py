from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quoridor.challenge_replay import arena_position_orbits
from quoridor.openings import load_opening_suite
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes, sha256_file
from quoridor.teacher_data import (
    TeacherDataConfig,
    TeacherDataContractError,
    evaluation_position_orbit_digest,
    exclusion_set_sha256,
    verify_recovery_preparation_gate,
)


ROOT = Path(__file__).resolve().parents[2]


class Phase4FRecoveryPreparationTests(unittest.TestCase):
    def _recovery_config(self) -> TeacherDataConfig:
        contract = json.loads(
            (
                ROOT
                / "training/configs/phase4f-recovery-challenge-prepared-v1.json"
            ).read_bytes()
        )
        orbits: set[str] = set()
        for reference in contract["excludedSuites"]:
            suite = load_opening_suite(ROOT / reference["path"])
            orbits.update(
                evaluation_position_orbit_digest(opening.state)
                for opening in suite.openings
            )
        for reference in contract["excludedTrajectoryEvidence"]:
            orbits.update(arena_position_orbits(ROOT, reference))
        generator = contract["generator"]
        return TeacherDataConfig(
            target_positions=generator["targetPositions"],
            seed=generator["seed"],
            node_budget=generator["nodeBudget"],
            max_depth=generator["maxDepth"],
            exploration_probability=generator["explorationProbability"],
            opening_depths=tuple(generator["openingDepths"]),
            maximum_attempts=250_000,
            exclusion_set_sha256=exclusion_set_sha256(tuple(sorted(orbits))),
            exclusion_orbit_count=len(orbits),
        )

    def test_prepared_contract_excludes_all_historical_fixed_trajectories(self) -> None:
        path = ROOT / "training/configs/phase4f-recovery-challenge-prepared-v1.json"
        raw = path.read_bytes()
        contract = json.loads(raw)
        core = dict(contract)
        stored = core.pop("contractSha256")
        self.assertEqual(raw, canonical_json_bytes(contract))
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        self.assertFalse(contract["trainingUseAllowed"])

        excluded: set[str] = set()
        for reference in contract["excludedTrajectoryEvidence"]:
            excluded.update(arena_position_orbits(ROOT, reference))
        self.assertEqual(len(excluded), 20_849)
        self.assertEqual(
            contract["exclusionSet"]["orbits"],
            21_449,
        )

    def test_recovery_report_freezes_its_cpu_only_gpu_ledger_snapshot(self) -> None:
        report_path = ROOT / "training/runs/phase4/phase4f/recovery-preparation/report.json"
        report = json.loads(report_path.read_bytes())
        core = dict(report)
        stored = core.pop("reportSha256")
        self.assertEqual(stored, sha256_bytes(canonical_json_bytes(core)))
        self.assertTrue(report["cpuOnly"])
        self.assertFalse(report["gpuAccessed"])
        self.assertFalse(report["trainingAuthorized"])
        self.assertEqual(
            report["resources"]["gpuLedgerFileSha256"],
            "703592d0ccf70dbdf1fdaffeac70a446cff47778b5c365ca9709902e35085021",
        )
        self.assertEqual(
            report["resources"]["globalGpuHoursBeforeAndAfter"],
            43.14977626422679,
        )

    def test_cpu_recovery_gate_replaces_only_the_irrelevant_cuda_smoke(self) -> None:
        report = "training/runs/phase4/phase4f/recovery-preparation/report.json"
        output = ROOT / "training/runs/phase4/phase4f/recovery-challenge-data"
        config = self._recovery_config()
        prepared = json.loads((ROOT / report).read_bytes())
        frozen_source = SimpleNamespace(to_payload=lambda: prepared["source"])
        ledger_path = (ROOT / "training/runs/phase4/gpu-budget-ledger.json").resolve()

        def frozen_ledger_sha256(path: str | Path) -> str:
            if Path(path).resolve() == ledger_path:
                return prepared["resources"]["gpuLedgerFileSha256"]
            return sha256_file(path)

        with patch(
            "quoridor.teacher_data.fingerprint_sources",
            return_value=frozen_source,
        ), patch(
            "quoridor.teacher_data.sha256_file",
            side_effect=frozen_ledger_sha256,
        ):
            evidence = verify_recovery_preparation_gate(
                workspace_root=ROOT,
                control_root=ROOT / "training/runs/phase4",
                report_path=report,
                output_root=output,
                config=config,
            )
        self.assertEqual(evidence["kind"], "phase4f2-cpu-preparation-gate-v1")
        with patch(
            "quoridor.teacher_data.fingerprint_sources",
            return_value=frozen_source,
        ), patch(
            "quoridor.teacher_data.sha256_file",
            side_effect=frozen_ledger_sha256,
        ):
            with self.assertRaisesRegex(
                TeacherDataContractError,
                "differs from the requested dataset",
            ):
                verify_recovery_preparation_gate(
                    workspace_root=ROOT,
                    control_root=ROOT / "training/runs/phase4",
                    report_path=report,
                    output_root=output,
                    config=replace(config, node_budget=config.node_budget + 1),
                )

    def test_completed_dataset_retains_the_frozen_generation_source(self) -> None:
        with self.assertRaisesRegex(
            TeacherDataContractError,
            "stale or does not authorize",
        ):
            verify_recovery_preparation_gate(
                workspace_root=ROOT,
                control_root=ROOT / "training/runs/phase4",
                report_path=(
                    "training/runs/phase4/phase4f/recovery-preparation/report.json"
                ),
                output_root=(
                    ROOT
                    / "training/runs/phase4/phase4f/recovery-challenge-data"
                ),
                config=self._recovery_config(),
            )


if __name__ == "__main__":
    unittest.main()
