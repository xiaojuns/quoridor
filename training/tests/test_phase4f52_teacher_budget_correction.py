from __future__ import annotations

import pytest

from quoridor.phase4f52_teacher_budget_correction import (
    budget_for_band,
    stability_checks,
    summarize_banded_stability,
)


def _analysis(action: int, ranked: list[int], *, depth: int = 5):
    scores = {value: len(ranked) - index for index, value in enumerate(ranked)}
    return {
        "actionIndex": action,
        "depth": depth,
        "searchedActions": len(ranked),
        "legalActions": len(ranked),
        "rootScores": [
            {"actionIndex": value, "searched": True, "score": scores[value]}
            for value in sorted(ranked)
        ],
    }


def test_banded_budgets_match_correction_contract():
    assert [budget_for_band(band) for band in ("early", "middle", "late")] == [
        16_384,
        16_384,
        4_096,
    ]
    assert [
        budget_for_band(band, reference=True)
        for band in ("early", "middle", "late")
    ] == [65_536, 65_536, 16_384]
    with pytest.raises(ValueError):
        budget_for_band("unknown")


def test_stability_is_reported_and_gated_per_band():
    metadata = {
        0: {"band": "early"},
        1: {"band": "middle"},
        2: {"band": "late"},
    }
    primary = {
        0: _analysis(0, list(range(12))),
        1: _analysis(0, list(range(12))),
        2: _analysis(0, list(range(12))),
    }
    reference = {
        0: _analysis(1, list(range(1, 13))),
        1: _analysis(0, list(range(12))),
        2: _analysis(0, list(range(12))),
    }
    summary = summarize_banded_stability(primary, reference, metadata)
    checks = stability_checks(summary)
    assert summary["all"]["actionAgreement"] == pytest.approx(2 / 3)
    assert summary["early"]["meanTop12Jaccard"] == pytest.approx(11 / 13)
    assert checks["earlyActionAgreement"] is False
    assert checks["earlyTop12Jaccard"] is False
    assert checks["middleActionAgreement"] is True
    assert checks["lateTop12Jaccard"] is True


def test_stability_rejects_mismatched_ordinals():
    analysis = _analysis(0, list(range(12)))
    with pytest.raises(ValueError, match="ordinals"):
        summarize_banded_stability({0: analysis}, {1: analysis}, {0: {"band": "early"}})
