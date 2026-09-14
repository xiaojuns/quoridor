from __future__ import annotations

from quoridor.phase4f53_policy_head_preparation import (
    TRAIN_COUNTS,
    select_train_rows,
    selection_audit,
    selection_checks,
)


def _rows(split: str, count_per_stratum: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for band in ("early", "middle", "late"):
        for side in (0, 1):
            for index in range(count_per_stratum):
                rows.append(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "band": band,
                        "currentPlayer": side,
                        "rank": f"{index:064x}",
                        "positionOrbit": f"{split}-{band}-{side}-{index}",
                        "gameId": f"{split}-game-{index}",
                    }
                )
                ordinal += 1
    return rows


def test_train_selection_is_balanced_and_development_disjoint() -> None:
    counts = {"early": 4, "middle": 4, "late": 2}
    train = _rows("train", 5)
    development = _rows("development", 2)
    selected = select_train_rows(train + development, counts)
    audit = selection_audit(selected, development)
    assert audit["positions"] == 10
    assert audit["uniqueOrbits"] == 10
    assert audit["developmentOrbitOverlap"] == 0
    assert audit["trainDevelopmentGameOverlap"] == 0
    assert audit["byBand"] == counts
    assert audit["byBandAndSide"] == {
        "early/side-0": 2,
        "early/side-1": 2,
        "middle/side-0": 2,
        "middle/side-1": 2,
        "late/side-0": 1,
        "late/side-1": 1,
    }


def test_selection_checks_reject_overlap() -> None:
    expected_side = {
        f"{band}/side-{side}": total // 2
        for band, total in TRAIN_COUNTS.items()
        for side in (0, 1)
    }
    audit = {
        "positions": sum(TRAIN_COUNTS.values()),
        "uniqueOrbits": sum(TRAIN_COUNTS.values()),
        "developmentOrbitOverlap": 1,
        "trainDevelopmentGameOverlap": 0,
        "byBand": dict(TRAIN_COUNTS),
        "byBandAndSide": expected_side,
    }
    checks = selection_checks(audit)
    assert checks["developmentOrbitDisjoint"] is False
    assert all(value for key, value in checks.items() if key != "developmentOrbitDisjoint")
