from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quoridor.phase4f_final import (
    PHASE4F_CANDIDATE_SIMULATIONS,
    _arena_contract_sha256,
    _gate_from_result,
    _self_hashed,
    _track_payload,
    load_phase4f_contract,
)
from quoridor.run_manifest import canonical_json_bytes


def contract_payload() -> dict[str, object]:
    tracks = {
        "fixed": _track_payload("fixed", 1_000),
        "wallClock": _track_payload("wall-clock", 800),
    }
    core = {
        "schemaVersion": 1,
        "kind": "phase4f-final-evaluation-contract",
        "stage": "phase4f",
        "status": "frozen",
        "purpose": "test",
        "productReleaseAuthorized": False,
        "candidate": {
            "id": "candidate",
            "modelSha256": "1" * 64,
            "configSha256": "2" * 64,
            "trainingSourceSha256": "3" * 64,
        },
        "evaluationSourceSha256": "4" * 64,
        "nativeBackendSha256": "5" * 64,
        "classicBackendSha256": "6" * 64,
        "modelConfig": {},
        "phase4eExecutionContract": {},
        "openingSuites": {
            "standardInitialSuiteHash": "sha256:" + "b" * 64,
        },
        "calibration": {},
        "workloadSha256": "7" * 64,
        "tracks": tracks,
        "resources": {},
        "postEvaluation": {},
    }
    return _self_hashed(core, "contractSha256")


def arena_result(contract: dict[str, object], pair_file: str) -> dict[str, object]:
    candidate = contract["candidate"]
    return {
        "contractSha256": "8" * 64,
        "suite": "final",
        "openingPairs": 200,
        "games": 400,
        "candidateIdentity": {
            "selectorId": "phase4f-champion-candidate",
            "artifactHash": candidate["modelSha256"],
            "configHash": candidate["configSha256"],
            "sourceHash": candidate["trainingSourceSha256"],
            "engine": "native-mcts",
            "verified": True,
        },
        "opponentIdentity": {
            "selectorId": "classic-hard-fixed",
            "artifactHash": contract["classicBackendSha256"],
            "configHash": contract["workloadSha256"],
            "sourceHash": contract["evaluationSourceSha256"],
            "engine": "classic-negamax",
            "verified": True,
        },
        "candidateRuntime": {
            "kind": "native-mcts",
            "trusted": True,
            "configuredSimulations": PHASE4F_CANDIDATE_SIMULATIONS,
            "backendHash": contract["nativeBackendSha256"],
        },
        "opponentRuntime": {
            "kind": "classic-hard",
            "trusted": True,
            "budgetKind": "nodes",
            "budgetLimit": 1_000,
            "maxDepth": 5,
            "backendHash": contract["classicBackendSha256"],
        },
        "statistics": {
            "winRate": 0.60,
            "winRateCI": {"lower": 0.55, "upper": 0.65, "confidence": 0.95},
        },
        "pairArtifacts": [{"file": pair_file}],
        "resultSha256": "9" * 64,
    }


def standard_result(contract: dict[str, object]) -> dict[str, object]:
    result = arena_result(contract, "unused.json")
    result.update(
        {
            "openingPairs": 1,
            "games": 2,
            "suiteHash": contract["openingSuites"]["standardInitialSuiteHash"],
            "statistics": {"winRate": 0.5},
            "resultSha256": "a" * 64,
        }
    )
    return result


class Phase4FFinalTests(unittest.TestCase):
    def test_completed_gate_comparison_uses_canonical_json_representation(self) -> None:
        expected = {"budget": [("nodes", 587)]}
        stored = {"budget": [["nodes", 587]]}
        self.assertNotEqual(expected, stored)
        self.assertEqual(canonical_json_bytes(expected), canonical_json_bytes(stored))

    def test_contract_loader_rejects_tampering(self) -> None:
        payload = contract_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_bytes(canonical_json_bytes(payload))
            self.assertEqual(load_phase4f_contract(path), payload)
            payload["status"] = "prepared"
            path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(Exception, "self hash"):
                load_phase4f_contract(path)

    def test_track_contracts_freeze_distinct_classic_budgets(self) -> None:
        fixed = _track_payload("fixed", 1234)
        wall = _track_payload("wall-clock", 800)
        self.assertEqual(fixed["opponentBudget"], {"kind": "nodes", "limit": 1234, "maxDepth": 5})
        self.assertEqual(wall["opponentBudget"], {"kind": "time", "limit": 800, "maxDepth": 5})
        contract = contract_payload()
        self.assertNotEqual(
            _arena_contract_sha256(contract, "fixed", "sha256:" + "a" * 64),
            _arena_contract_sha256(contract, "wall-clock", "sha256:" + "a" * 64),
        )

    def test_fixed_gate_checks_mixed_runtime_and_budget_telemetry(self) -> None:
        contract = contract_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = {
                "games": [
                    {
                        "plies": [
                            {
                                "selectorRole": "candidate",
                                "selection": {
                                    "simulations": 800,
                                    "nodes": 800,
                                    "wallTimeMs": 10.0,
                                    "metadata": {},
                                },
                            },
                            {
                                "selectorRole": "opponent",
                                "selection": {
                                    "simulations": 0,
                                    "nodes": 999,
                                    "wallTimeMs": 20.0,
                                    "metadata": {
                                        "budgetKind": "nodes",
                                        "budgetLimit": 1_000,
                                    },
                                },
                            },
                        ]
                    }
                ]
            }
            pair_path = root / "pair.json"
            pair_path.write_bytes(canonical_json_bytes(pair))
            result = arena_result(contract, pair_path.name)
            with patch(
                "quoridor.phase4f_final.verify_completed_arena_result",
                side_effect=(result, standard_result(contract)),
            ):
                gate = _gate_from_result(
                    root / "result.json",
                    standard_result_path=root / "standard-result.json",
                    contract=contract,
                    track_name="fixed",
                )
            self.assertTrue(gate["passed"])
            self.assertTrue(all(check["passed"] for check in gate["checks"]))


if __name__ == "__main__":
    unittest.main()
