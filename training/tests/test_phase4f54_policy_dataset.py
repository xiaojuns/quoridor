from __future__ import annotations

import pytest

from quoridor.challenge_replay import position_orbit_digest
from quoridor.codec import position_hash
from quoridor.game import create_initial_state, legal_action_indices
from quoridor.phase4f54_policy_dataset import (
    EXPECTED_CONTRACT_SHA256,
    Phase4F54Error,
    _percentile,
    build_sample,
    validate_sample,
)


def _analysis(state, limit: int = 16_384):
    actions = legal_action_indices(state)
    return {
        "format": "legal-action-root-scores-v1",
        "budget": {"kind": "nodes", "limit": limit, "maxDepth": 5},
        "actionIndex": actions[0],
        "score": 10,
        "depth": 5,
        "nodes": limit,
        "elapsedMs": 1,
        "rootScoreKind": "alpha-beta-return-v1",
        "legalActions": len(actions),
        "searchedActions": len(actions),
        "rootScores": [
            {"actionIndex": action, "searched": True, "score": 10 - index}
            for index, action in enumerate(actions)
        ],
    }


def _metadata(state):
    return {
        "ordinal": 7,
        "split": "train",
        "band": "early",
        "gameId": "game-1",
        "plyIndex": 0,
        "positionHash": position_hash(state),
        "positionOrbit": position_orbit_digest(state),
    }


def test_policy_sample_is_normalized_and_has_no_value_target() -> None:
    state = create_initial_state()
    sample = build_sample(
        state, _metadata(state), _analysis(state), EXPECTED_CONTRACT_SHA256
    )
    validate_sample(sample, EXPECTED_CONTRACT_SHA256)
    assert "value" not in sample
    assert sum(probability for _action, probability in sample["policy"]) == pytest.approx(1.0)
    assert {action for action, _probability in sample["policy"]} == set(
        legal_action_indices(state)
    )


def test_policy_sample_rejects_wrong_teacher_budget() -> None:
    state = create_initial_state()
    with pytest.raises(Phase4F54Error, match="budget"):
        build_sample(
            state,
            _metadata(state),
            _analysis(state, limit=4_096),
            EXPECTED_CONTRACT_SHA256,
        )


def test_percentile_interpolates() -> None:
    assert _percentile([0.0, 1.0], 0.5) == 0.5
