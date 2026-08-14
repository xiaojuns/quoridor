from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quoridor.phase4_budget import (
    PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
    PHASE4D_BASELINE_REPORT_SHA256,
    PHASE4D_BASELINE_RUN_ID,
    PHASE4D_GPU_HOUR_LIMIT,
    PHASE4D_STAGE,
)
from quoridor.phase4_execution import (
    PHASE4D_EXECUTION_CONTRACT,
    PHASE4_BUDGET_STATUS_FROZEN,
    PHASE4_BUDGET_STATUS_PENDING,
    PHASE4_EXECUTION_SCHEMA_VERSION,
    PHASE4_EXECUTION_STATUS_FROZEN,
    PHASE4_EXECUTION_STATUS_PREPARED,
    PHASE4_GLOBAL_GPU_HOUR_LIMIT,
    PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT,
    Phase4ExecutionBudget,
    Phase4ExecutionContract,
    Phase4ExecutionContractError,
    load_phase4_execution_contract,
    phase4d_execution_contract,
)
from quoridor.phase4e1_contract import (
    FREEZE_MAXIMUM_GENERATIONS,
    FREEZE_MINIMUM_STAGE_GPU_HOURS,
    FREEZE_RESERVE_GPU_HOURS,
)
from quoridor.run_manifest import canonical_json_bytes, sha256_bytes


def _prepared_payload() -> dict[str, object]:
    return {
        "schemaVersion": PHASE4_EXECUTION_SCHEMA_VERSION,
        "stage": "phase4e",
        "status": PHASE4_EXECUTION_STATUS_PREPARED,
        "purpose": "phase4e0-preparation",
        "formalPromotion": False,
        "maximumGenerations": 12,
        "stageBudget": {
            "status": PHASE4_BUDGET_STATUS_PENDING,
            "baselineGlobalHours": None,
            "limitHours": None,
            "baselineRunId": None,
            "baselineReportSha256": None,
        },
    }


class Phase4ExecutionContractTests(unittest.TestCase):
    def test_phase4d_contract_preserves_the_frozen_pilot_semantics(self) -> None:
        contract = phase4d_execution_contract()
        self.assertEqual(contract, PHASE4D_EXECUTION_CONTRACT)
        self.assertEqual(contract.stage, PHASE4D_STAGE)
        self.assertEqual(contract.status, PHASE4_EXECUTION_STATUS_FROZEN)
        self.assertEqual(contract.maximum_generations, 5)
        self.assertFalse(contract.formal_promotion)
        budget = contract.require_frozen_budget()
        self.assertEqual(
            budget.baseline_global_hours,
            PHASE4D_BASELINE_GLOBAL_GPU_HOURS,
        )
        self.assertEqual(budget.limit_hours, PHASE4D_GPU_HOUR_LIMIT)
        self.assertEqual(budget.baseline_run_id, PHASE4D_BASELINE_RUN_ID)
        self.assertEqual(
            budget.baseline_report_sha256,
            PHASE4D_BASELINE_REPORT_SHA256,
        )
        self.assertEqual(
            contract.canonical_sha256,
            "155490ce957c909886ef72cc895b2c32eca7345614d189b591bf87c57fc1307d",
        )

    def test_prepared_contract_has_canonical_sha_and_cannot_authorize_gpu_work(
        self,
    ) -> None:
        payload = _prepared_payload()
        contract = Phase4ExecutionContract.from_dict(payload)
        self.assertEqual(contract.to_payload(), payload)
        self.assertEqual(
            contract.canonical_sha256,
            sha256_bytes(canonical_json_bytes(payload)),
        )
        self.assertEqual(contract.sha256, contract.canonical_sha256)
        self.assertFalse(contract.budget_frozen)
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "pending",
        ):
            contract.require_frozen_budget()

        reordered = {
            "stageBudget": payload["stageBudget"],
            "maximumGenerations": 12,
            "formalPromotion": False,
            "purpose": "phase4e0-preparation",
            "status": PHASE4_EXECUTION_STATUS_PREPARED,
            "stage": "phase4e",
            "schemaVersion": PHASE4_EXECUTION_SCHEMA_VERSION,
        }
        self.assertEqual(
            Phase4ExecutionContract.from_dict(reordered).canonical_sha256,
            contract.canonical_sha256,
        )

    def test_load_round_trips_a_prepared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase4e-execution.json"
            path.write_text(
                json.dumps(_prepared_payload(), indent=2),
                encoding="utf-8",
            )
            loaded = load_phase4_execution_contract(path)
            self.assertEqual(loaded.to_payload(), _prepared_payload())

            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(
                Phase4ExecutionContractError,
                "must be an object",
            ):
                load_phase4_execution_contract(path)

    def test_maximum_generations_is_a_count_and_bounds_absolute_targets(
        self,
    ) -> None:
        contract = Phase4ExecutionContract.from_dict(_prepared_payload())
        self.assertEqual(contract.validate_target_generations(0), 0)
        self.assertEqual(contract.validate_target_generations(12), 12)
        self.assertEqual(contract.remaining_generations(0), 12)
        self.assertEqual(contract.remaining_generations(5), 7)
        with self.assertRaisesRegex(Phase4ExecutionContractError, r"\[0, 12\]"):
            contract.validate_target_generations(13)
        with self.assertRaisesRegex(Phase4ExecutionContractError, r"\[0, 12\]"):
            contract.validate_target_generations(-1)
        with self.assertRaisesRegex(Phase4ExecutionContractError, r"\[0, 12\]"):
            contract.validate_target_generations(True)

        invalid = _prepared_payload()
        invalid["maximumGenerations"] = 0
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "positive integer",
        ):
            Phase4ExecutionContract.from_dict(invalid)

        invalid_schema = _prepared_payload()
        invalid_schema["schemaVersion"] = True
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "unsupported",
        ):
            Phase4ExecutionContract.from_dict(invalid_schema)

    def test_frozen_phase4e_contract_requires_reviewed_policy_and_provenance(
        self,
    ) -> None:
        baseline_global_hours = 9.25
        limit_hours = (
            PHASE4_LEGACY_GLOBAL_GPU_HOUR_LIMIT
            - baseline_global_hours
            - FREEZE_RESERVE_GPU_HOURS
        )
        budget = Phase4ExecutionBudget(
            status=PHASE4_BUDGET_STATUS_FROZEN,
            baseline_global_hours=baseline_global_hours,
            limit_hours=limit_hours,
            baseline_run_id="phase4e-calibration-1",
            baseline_report_sha256="a" * 64,
        )
        contract = Phase4ExecutionContract(
            schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
            stage="phase4e",
            status=PHASE4_EXECUTION_STATUS_FROZEN,
            purpose="phase4e-bootstrap-and-formal-promotion",
            formal_promotion=True,
            maximum_generations=FREEZE_MAXIMUM_GENERATIONS,
            stage_budget=budget,
        )
        self.assertEqual(
            Phase4ExecutionContract.from_dict(contract.to_payload()),
            contract,
        )
        self.assertGreaterEqual(
            contract.stage_budget.limit_hours,
            FREEZE_MINIMUM_STAGE_GPU_HOURS,
        )

        invalid_policies = (
            {"purpose": "phase4e-bootstrap-training"},
            {"formal_promotion": False},
            {"maximum_generations": FREEZE_MAXIMUM_GENERATIONS - 1},
            {
                "stage_budget": Phase4ExecutionBudget(
                    status=PHASE4_BUDGET_STATUS_FROZEN,
                    baseline_global_hours=baseline_global_hours,
                    limit_hours=limit_hours - 1.0,
                    baseline_run_id="phase4e-calibration-1",
                    baseline_report_sha256="a" * 64,
                )
            },
        )
        for mutation in invalid_policies:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                Phase4ExecutionContractError,
                "reviewed E1 policy",
            ):
                Phase4ExecutionContract(
                    schema_version=PHASE4_EXECUTION_SCHEMA_VERSION,
                    stage="phase4e",
                    status=PHASE4_EXECUTION_STATUS_FROZEN,
                    purpose=mutation.get(
                        "purpose",
                        "phase4e-bootstrap-and-formal-promotion",
                    ),
                    formal_promotion=mutation.get("formal_promotion", True),
                    maximum_generations=mutation.get(
                        "maximum_generations",
                        FREEZE_MAXIMUM_GENERATIONS,
                    ),
                    stage_budget=mutation.get("stage_budget", budget),
                )

        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "provenance must be complete",
        ):
            Phase4ExecutionBudget(
                status=PHASE4_BUDGET_STATUS_FROZEN,
                baseline_global_hours=7.5,
                limit_hours=20.0,
                baseline_run_id="phase4e-calibration-1",
            )

    def test_frozen_budget_cannot_exceed_global_gpu_limit(self) -> None:
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "global GPU-hour limit",
        ):
            Phase4ExecutionBudget(
                status=PHASE4_BUDGET_STATUS_FROZEN,
                baseline_global_hours=PHASE4_GLOBAL_GPU_HOUR_LIMIT - 1.0,
                limit_hours=2.0,
                baseline_run_id="phase4e-calibration-1",
                baseline_report_sha256="a" * 64,
            )

    def test_prepared_contract_rejects_provisional_gpu_numbers(self) -> None:
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "must not freeze GPU values",
        ):
            Phase4ExecutionBudget(
                status=PHASE4_BUDGET_STATUS_PENDING,
                baseline_global_hours=7.5,
                limit_hours=20.0,
            )

        payload = _prepared_payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(
            Phase4ExecutionContractError,
            "fields are invalid",
        ):
            Phase4ExecutionContract.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
