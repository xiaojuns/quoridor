from __future__ import annotations

from quoridor.phase4f62_value_scale_confirmation import (
    AMENDED_CONFIRMATION_GPU_HOURS,
    CONFIRMATION_GPU_HOURS,
    _identities,
    arena_contract_sha256,
    outcome_band,
)


def _contract() -> dict[str, object]:
    return {
        "contractSha256": "a" * 64,
        "source": {"sha256": "b" * 64},
        "model": {"fileSha256": "c" * 64},
        "candidate": {
            "selectorId": "scaled",
            "runtime": {"runtimeSha256": "d" * 64},
        },
        "opponent": {
            "selectorId": "baseline",
            "runtime": {"runtimeSha256": "e" * 64},
        },
        "workload": {"workloadSha256": "f" * 64},
        "openingSuite": {"suiteHash": "sha256:" + "1" * 64},
    }


def test_identities_share_model_and_bind_distinct_value_runtimes() -> None:
    candidate, opponent = _identities(_contract())
    assert candidate.artifact_hash == opponent.artifact_hash == "c" * 64
    assert candidate.config_hash == "d" * 64
    assert opponent.config_hash == "e" * 64
    assert candidate.config_hash != opponent.config_hash


def test_arena_contract_binds_both_runtime_hashes() -> None:
    contract = _contract()
    original = arena_contract_sha256(contract)
    contract["opponent"]["runtime"]["runtimeSha256"] = "2" * 64
    assert arena_contract_sha256(contract) != original


def test_confirmation_requires_win_rate_and_lower_bound_above_half() -> None:
    assert outcome_band(0.60, 0.51) == "value-scale-confirmation-passed"
    assert outcome_band(0.60, 0.50) == "value-scale-confirmation-inconclusive"
    assert outcome_band(0.50, 0.51) == "value-scale-confirmation-failed"


def test_storage_amendment_only_expands_the_stage_budget() -> None:
    assert CONFIRMATION_GPU_HOURS == 6.0
    assert AMENDED_CONFIRMATION_GPU_HOURS == 7.5
