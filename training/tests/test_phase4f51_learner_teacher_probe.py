from __future__ import annotations

import pytest

from quoridor.phase4f51_learner_teacher_probe import (
    _replay_from_selfplay,
    ply_band,
    summarize_stability,
    summarize_targets,
    target_checks,
    tie_preserving_rank_target,
)
from quoridor.game import create_initial_state
from quoridor.selfplay import SelfPlayGame, SelfPlayPly


def _analysis(scores, action=0):
    return {
        "actionIndex": action,
        "rootScores": [
            {"actionIndex": index, "searched": score is not None, "score": score}
            for index, score in enumerate(scores)
        ],
    }


def test_ply_band_uses_absolute_boundaries():
    assert [ply_band(value) for value in (0, 15, 16, 39, 40, 127)] == [
        "early", "early", "middle", "middle", "late", "late"
    ]


def test_rank_target_preserves_ties_and_legal_smoothing():
    policy = tie_preserving_rank_target(_analysis([9, 9, 8, 7, None]), 1.0, top_k=3)
    assert policy[0] == pytest.approx(policy[1])
    assert policy[2] < policy[0]
    assert policy[3] == pytest.approx(0.015)
    assert policy[4] == pytest.approx(0.015)
    assert sum(policy.values()) == pytest.approx(1.0)


def test_rank_target_keeps_boundary_ties():
    scores = [12 - index for index in range(11)] + [1, 1, 0]
    policy = tie_preserving_rank_target(_analysis(scores), 2.0, top_k=12)
    assert policy[11] == pytest.approx(policy[12])
    assert policy[13] == pytest.approx(0.03)


def test_target_summary_and_checks_are_explicit():
    records = [_analysis([5, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6])] * 8
    summary = summarize_targets(records, 2.0)
    checks = target_checks(summary, {
        "meanEntropyMinimum": 0.0,
        "meanEntropyMaximum": 10.0,
        "top1ProbabilityP95Maximum": 1.0,
        "classicMaximumScoreGroupMassMinimum": 0.0,
        "classicTop5MassMinimum": 0.0,
        "policyMassMaximumAbsoluteError": 1e-9,
        "mirrorMaximumAbsoluteError": 1e-12,
    })
    assert all(checks.values())
    assert summary["mirrorMaximumAbsoluteError"] == 0.0


def test_stability_uses_action_and_top12_jaccard():
    low = {0: _analysis(list(range(20, 0, -1)), action=0)}
    high = {0: _analysis(list(range(20, 0, -1)), action=0)}
    assert summarize_stability(low, high) == {
        "positions": 1,
        "actionAgreement": 1.0,
        "meanTop12Jaccard": 1.0,
    }


def test_rule_limit_game_retains_only_128_probe_eligible_plies():
    counts = (1,) + (0,) * 208
    ply = SelfPlayPly(action_index=0, visit_counts=counts, root_value=0.0)
    game = SelfPlayGame(
        samples=(),
        initial_state=create_initial_state(),
        raw_plies=(ply,) * 256,
        winner=0,
        move_count=256,
        termination_reason="move-limit",
    )
    replay = _replay_from_selfplay(game, "train", 7, 3)
    assert len(replay.plies) == 256
    assert replay.metadata["probeEligiblePlies"] == 128
    assert replay.metadata["probeExcludedTailPlies"] == 128
