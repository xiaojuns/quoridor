"""CPU-only diagnosis of a completed Phase 4F fixed-budget failure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .phase4_arena import verify_completed_arena_result
from .phase4f_final import (
    Phase4FError,
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
    load_phase4f_contract,
)
from .run_manifest import canonical_json_bytes, sha256_file


ANALYSIS_SCHEMA_VERSION = 1


def _rate(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    wins = sum(game.get("candidateWon") is True for game in games)
    count = len(games)
    return {
        "games": count,
        "wins": wins,
        "losses": count - wins,
        "winRate": wins / count if count else None,
    }


def _numeric(values: Iterable[float]) -> dict[str, object]:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "median": statistics.median(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _position_payload(value: object) -> list[object]:
    if not isinstance(value, str):
        raise Phase4FError("game position hash is not a string")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise Phase4FError("game position hash is invalid JSON") from error
    if not isinstance(payload, list) or len(payload) != 10:
        raise Phase4FError("game position hash has an invalid schema")
    return payload


def _group_rates(
    games: Sequence[Mapping[str, object]], key
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for game in games:
        groups.setdefault(str(key(game)), []).append(game)
    return {label: _rate(groups[label]) for label in sorted(groups)}


def _candidate_plies(game: Mapping[str, object]) -> list[Mapping[str, object]]:
    plies = game.get("plies")
    if not isinstance(plies, list):
        raise Phase4FError("game plies are invalid")
    selected = [
        ply
        for ply in plies
        if isinstance(ply, Mapping) and ply.get("selectorRole") == "candidate"
    ]
    if not selected:
        raise Phase4FError("game has no candidate decisions")
    return selected


def _selection_value(ply: Mapping[str, object]) -> float:
    selection = ply.get("selection")
    value = selection.get("value") if isinstance(selection, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4FError("candidate selection value is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise Phase4FError("candidate selection value is not finite")
    return result


def _action_kind(ply: Mapping[str, object]) -> str:
    action = ply.get("actionIndex")
    if type(action) is not int or not 0 <= action < 209:
        raise Phase4FError("candidate action index is invalid")
    return "move" if action < 81 else "wall"


def analyze_games(
    games: Sequence[Mapping[str, object]],
    *,
    pairs: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    if not games or any(len(pair) != 2 for pair in pairs):
        raise Phase4FError("failure analysis requires complete games and opening pairs")

    def opening_walls_placed(game: Mapping[str, object]) -> int:
        position = _position_payload(game.get("initialPositionHash"))
        remaining = position[6:8]
        if any(type(value) is not int for value in remaining):
            raise Phase4FError("opening wall counts are invalid")
        return 20 - int(remaining[0]) - int(remaining[1])

    action_rows: list[tuple[bool, str, float]] = []
    first_values: list[tuple[bool, float]] = []
    quartiles: list[list[tuple[bool, str, float]]] = [[], [], [], []]
    for game in games:
        won = game.get("candidateWon") is True
        candidate = _candidate_plies(game)
        first_values.append((won, _selection_value(candidate[0])))
        for index, ply in enumerate(candidate):
            row = (won, _action_kind(ply), _selection_value(ply))
            action_rows.append(row)
            quartiles[min(3, (4 * index) // len(candidate))].append(row)

    def decision_summary(rows: Sequence[tuple[bool, str, float]]) -> dict[str, object]:
        values = [value for _, _, value in rows]
        walls = sum(kind == "wall" for _, kind, _ in rows)
        losing = [value for won, _, value in rows if not won]
        return {
            "decisions": len(rows),
            "wallDecisions": walls,
            "wallRate": walls / len(rows) if rows else None,
            "values": _numeric(values),
            "losingDecisions": len(losing),
            "losingValueAtLeastZero": sum(value >= 0.0 for value in losing),
            "losingValueAtLeastHalf": sum(value >= 0.5 for value in losing),
            "losingValueAtLeastHalfRate": (
                sum(value >= 0.5 for value in losing) / len(losing) if losing else None
            ),
        }

    pair_outcomes = Counter(
        sum(game.get("candidateWon") is True for game in pair) for pair in pairs
    )
    overall = _rate(games)
    player_rates = _group_rates(games, lambda game: f"player-{game['candidatePlayer']}")
    to_move_rates = _group_rates(
        games, lambda game: "candidate-to-move" if game["candidateToMove"] else "opponent-to-move"
    )
    exact_opening_rates = _group_rates(games, lambda game: game["openingMoveNumber"])
    wall_bands = _group_rates(
        games,
        lambda game: (
            "00-04"
            if opening_walls_placed(game) <= 4
            else "05-09"
            if opening_walls_placed(game) <= 9
            else "10-14"
            if opening_walls_placed(game) <= 14
            else "15-20"
        ),
    )

    first_loss_values = [value for won, value in first_values if not won]
    first_sign_correct = sum((value >= 0.0) == won for won, value in first_values)
    losing_rows = [row for row in action_rows if not row[0]]
    winning_rows = [row for row in action_rows if row[0]]
    loss_lengths = [
        int(game["finalMoveNumber"]) - int(game["openingMoveNumber"])
        for game in games
        if game.get("candidateWon") is False
    ]
    win_lengths = [
        int(game["finalMoveNumber"]) - int(game["openingMoveNumber"])
        for game in games
        if game.get("candidateWon") is True
    ]

    no_win_pair_rate = pair_outcomes[0] / len(pairs)
    losing_overconfidence = decision_summary(losing_rows)["losingValueAtLeastHalfRate"]
    player_gap = abs(
        float(player_rates["player-0"]["winRate"])
        - float(player_rates["player-1"]["winRate"])
    )
    signals = [
        {
            "id": "large-classic-strength-gap",
            "severity": "critical",
            "observed": overall["winRate"],
            "threshold": 0.60,
            "interpretation": "The candidate is not close to the product hard-AI gate.",
        },
        {
            "id": "opening-pair-wide-failure",
            "severity": "critical" if no_win_pair_rate >= 0.5 else "high",
            "observed": no_win_pair_rate,
            "interpretation": "Most paired openings produced no candidate win after swapping sides.",
        },
        {
            "id": "losing-value-overconfidence",
            "severity": "high" if float(losing_overconfidence) >= 0.4 else "medium",
            "observed": losing_overconfidence,
            "interpretation": "The search value remains strongly positive on many decisions in games eventually lost.",
        },
        {
            "id": "player-orientation-asymmetry",
            "severity": "high" if player_gap >= 0.10 else "medium",
            "observed": player_gap,
            "interpretation": "Canonical player symmetry does not yield comparable outcomes across original player identities.",
        },
    ]

    return {
        "overall": overall,
        "outcomesByCandidatePlayer": player_rates,
        "outcomesBySideToMove": to_move_rates,
        "outcomesByOpeningMoveNumber": exact_opening_rates,
        "outcomesByOpeningWallsPlaced": wall_bands,
        "openingPairs": {
            "pairs": len(pairs),
            "zeroCandidateWins": pair_outcomes[0],
            "oneCandidateWin": pair_outcomes[1],
            "twoCandidateWins": pair_outcomes[2],
            "zeroCandidateWinRate": no_win_pair_rate,
        },
        "terminationReasons": dict(sorted(Counter(str(game["terminationReason"]) for game in games).items())),
        "gameLengthPlies": {"wins": _numeric(win_lengths), "losses": _numeric(loss_lengths)},
        "candidateDecisions": {
            "all": decision_summary(action_rows),
            "wins": decision_summary(winning_rows),
            "losses": decision_summary(losing_rows),
            "byGameQuartile": {
                str(index + 1): decision_summary(rows)
                for index, rows in enumerate(quartiles)
            },
        },
        "firstCandidateValue": {
            "all": _numeric(value for _, value in first_values),
            "losses": _numeric(first_loss_values),
            "signAccuracy": first_sign_correct / len(first_values),
            "lossesAtLeastZero": sum(value >= 0.0 for value in first_loss_values),
            "lossesAtLeastHalf": sum(value >= 0.5 for value in first_loss_values),
        },
        "signals": signals,
    }


def _read_pairs(result_path: Path, result: Mapping[str, object]) -> tuple[list[Mapping[str, object]], list[list[Mapping[str, object]]]]:
    games: list[Mapping[str, object]] = []
    pairs: list[list[Mapping[str, object]]] = []
    references = result.get("pairArtifacts")
    if not isinstance(references, list):
        raise Phase4FError("arena result pair references are invalid")
    for reference in references:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("file"), str):
            raise Phase4FError("arena result pair reference is invalid")
        pair = _mapping(result_path.parent / reference["file"], "Phase 4F pair")
        raw_games = pair.get("games")
        if not isinstance(raw_games, list) or len(raw_games) != 2 or any(not isinstance(game, Mapping) for game in raw_games):
            raise Phase4FError("Phase 4F pair games are invalid")
        materialized = [dict(game) for game in raw_games]
        pairs.append(materialized)
        games.extend(materialized)
    return games, pairs


def _recovery_plan(analysis: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "prepared",
        "decision": "stop-final-evaluation-and-recover-strength",
        "trainingAuthorized": False,
        "wallClockArenaAuthorized": False,
        "dataPolicy": {
            "forbidTrainingOnPhase4fFinalEvidence": True,
            "forbidTuningOnExactFinalPositions": True,
            "allowedSources": [
                "new content-disjoint generated positions",
                "dev-suite aggregate gates without trajectory ingestion",
                "self-play from the published champion",
                "new classic-hard teacher trajectories disjoint from every evaluation suite",
            ],
            "nextProductGateRequiresFreshUnseenFinalSuite": True,
        },
        "engineeringBeforeGpu": [
            "Add a provenance-bound champion model initialization path without optimizer or replay import.",
            "Add a configurable classic-hard teacher/challenge-data mixture to training.",
            "Add player-orientation and value-calibration diagnostics to generation health reports.",
            "Generate and freeze content-disjoint recovery dev and unseen final suites.",
            "Run native/Python differential and replay-cache gates after the source change.",
        ],
        "recommendedWorkload": {
            "network": "192x16 unchanged for attribution",
            "selfPlaySimulations": 512,
            "evaluationSimulations": 800,
            "initialModel": "published generation-14 champion model state",
            "optimizer": "reset",
            "targetedData": [
                "early positions with 0-4 walls already placed",
                "balanced original player identities",
                "classic-hard tactical wall responses",
                "positions where the prior model is positive but classic search refutes the line",
            ],
        },
        "budgetEnvelopeGpuHours": {
            "remainingPhase4eAtAnalysisStart": 26.850223,
            "recoveryTrainingMaximum": 14.0,
            "developmentScreensMaximum": 3.0,
            "reservedForFreshFinalGate": 9.5,
        },
        "promotionSequence": [
            "CPU correctness and provenance gates",
            "64-game recovery-dev screen against classic hard",
            "400-game recovery-dev confirmation against classic hard",
            "historical champion non-regression gate",
            "fresh unseen 400-game fixed-budget product gate",
            "wall-clock gate only after the fresh fixed-budget gate passes",
        ],
        "evidenceSummary": {
            "fixedWinRate": analysis["overall"]["winRate"],
            "zeroWinOpeningPairRate": analysis["openingPairs"]["zeroCandidateWinRate"],
            "losingValueAtLeastHalfRate": analysis["candidateDecisions"]["losses"]["losingValueAtLeastHalfRate"],
        },
    }


def run_failure_analysis(
    *,
    contract_path: Path,
    result_path: Path,
    standard_result_path: Path,
    gate_path: Path,
    output_path: Path,
) -> dict[str, object]:
    contract = load_phase4f_contract(contract_path)
    result = verify_completed_arena_result(result_path)
    standard = verify_completed_arena_result(standard_result_path)
    gate = _mapping(gate_path, "Phase 4F gate")
    _verify_self_hash(gate, field="gateSha256", label="Phase 4F gate")
    if gate.get("passed") is not False or gate.get("arenaResultSha256") != result.get("resultSha256"):
        raise Phase4FError("analysis input is not the failed fixed Phase 4F gate")
    games, pairs = _read_pairs(result_path, result)
    standard_games, standard_pairs = _read_pairs(standard_result_path, standard)
    analysis = analyze_games(games, pairs=pairs)
    standard_analysis = analyze_games(standard_games, pairs=standard_pairs)
    core = {
        "schemaVersion": ANALYSIS_SCHEMA_VERSION,
        "kind": "phase4f-fixed-failure-analysis",
        "status": "completed",
        "cpuOnly": True,
        "gpuAccessed": False,
        "inputs": {
            "contractPath": contract_path.as_posix(),
            "contractSha256": contract["contractSha256"],
            "contractFileSha256": sha256_file(contract_path),
            "gatePath": gate_path.as_posix(),
            "gateSha256": gate["gateSha256"],
            "resultPath": result_path.as_posix(),
            "resultSha256": result["resultSha256"],
            "standardResultPath": standard_result_path.as_posix(),
            "standardResultSha256": standard["resultSha256"],
        },
        "fixed": analysis,
        "standardInitial": standard_analysis,
        "recoveryPlan": _recovery_plan(analysis),
        "verificationDebt": {
            "completedGateCanonicalComparison": "fixed",
            "impactOnStoredResult": "none",
            "detail": "Reopened gates now compare canonical JSON bytes so tuple/list normalization is idempotent.",
        },
    }
    report = _self_hashed(core, "analysisSha256")
    _write_immutable(output_path, report, "Phase 4F failure analysis")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("training/configs/phase4f-final-frozen-v1.json"))
    parser.add_argument("--result", type=Path, default=Path("training/runs/phase4/phase4f/fixed/result.json"))
    parser.add_argument("--standard-result", type=Path, default=Path("training/runs/phase4/phase4f/fixed-standard-initial/result.json"))
    parser.add_argument("--gate", type=Path, default=Path("training/runs/phase4/phase4f/fixed/gate.json"))
    parser.add_argument("--output", type=Path, default=Path("training/runs/phase4/phase4f/failure-analysis/report.json"))
    args = parser.parse_args(argv)
    report = run_failure_analysis(
        contract_path=args.contract,
        result_path=args.result,
        standard_result_path=args.standard_result,
        gate_path=args.gate,
        output_path=args.output,
    )
    print(canonical_json_bytes({
        "status": report["status"],
        "analysisSha256": report["analysisSha256"],
        "outputPath": args.output.as_posix(),
        "fixedWinRate": report["fixed"]["overall"]["winRate"],
        "decision": report["recoveryPlan"]["decision"],
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
