from __future__ import annotations

from quoridor.phase4f60_kl_lineage_screen import (
    _identities,
    arena_contract_sha256,
    outcome_band,
)


def _contract() -> dict[str, object]:
    return {
        "contractSha256": "a" * 64,
        "candidate": {
            "selectorId": "candidate",
            "modelFileSha256": "b" * 64,
            "trainingContractSha256": "c" * 64,
            "trainingSourceSha256": "d" * 64,
        },
        "opponent": {
            "selectorId": "opponent",
            "modelFileSha256": "e" * 64,
            "configSha256": "f" * 64,
            "sourceSha256": "1" * 64,
        },
        "workload": {"workloadSha256": "2" * 64},
        "openingSuite": {"selectedSuiteHash": "sha256:" + "3" * 64},
    }


def test_outcome_bands_are_conservative() -> None:
    assert outcome_band(0.449) == "kl-policy-lineage-regression"
    assert outcome_band(0.45) == "kl-policy-lineage-inconclusive"
    assert outcome_band(0.499) == "kl-policy-lineage-inconclusive"
    assert outcome_band(0.50) == "kl-policy-lineage-retained"


def test_candidate_identity_binds_training_contract() -> None:
    candidate, opponent = _identities(_contract())
    assert candidate.config_hash == "c" * 64
    assert candidate.source_hash == "d" * 64
    assert opponent.config_hash == "f" * 64
    assert candidate.engine == opponent.engine == "native-mcts"


def test_arena_contract_binds_frozen_workload_and_suite() -> None:
    contract = _contract()
    original = arena_contract_sha256(contract)
    contract["workload"]["workloadSha256"] = "4" * 64
    assert arena_contract_sha256(contract) != original
