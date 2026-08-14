#!/usr/bin/env python3
"""Run Phase 3 native-engine correctness gates and throughput benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from quoridor.native_validation import (  # noqa: E402
    DEFAULT_LONG_GAMES,
    DEFAULT_SEED,
    DEFAULT_STATE_COUNT,
    NativeBindingUnavailable,
    run_native_validation,
    write_validation_report,
)


def _parse_integer(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, received {value!r}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare quoridor_native against the Python reference engine, run "
            "complete random games, and benchmark rule operations."
        )
    )
    parser.add_argument(
        "--states",
        type=_parse_integer,
        default=DEFAULT_STATE_COUNT,
        help=f"reachable states to compare (default: {DEFAULT_STATE_COUNT})",
    )
    parser.add_argument(
        "--games",
        type=_parse_integer,
        default=DEFAULT_LONG_GAMES,
        help=f"complete random games to run (default: {DEFAULT_LONG_GAMES})",
    )
    parser.add_argument(
        "--seed",
        type=_parse_integer,
        default=DEFAULT_SEED,
        help=f"deterministic decimal or 0x-prefixed seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print gate and throughput summaries while retaining the full report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.states <= 0:
        parser.error("--states must be positive")
    if arguments.games <= 0:
        parser.error("--games must be positive")

    try:
        report = run_native_validation(
            state_count=arguments.states,
            games=arguments.games,
            seed=arguments.seed,
        )
    except NativeBindingUnavailable as error:
        parser.exit(2, f"native validation unavailable: {error}\n")

    if arguments.output is not None:
        write_validation_report(report, arguments.output)
    output = report
    if arguments.summary_only:
        output = {
            "schemaVersion": report["schemaVersion"],
            "passed": report["passed"],
            "engine": report["engine"],
            "requested": report["requested"],
            "stateCount": report["stateCount"],
            "transitionCount": report["transitionCount"],
            "illegalActions": report["illegalActions"],
            "failureCount": report["failureCount"],
            "gates": report["gates"],
            "longGames": {
                key: report["longGames"][key]
                for key in (
                    "completedGames",
                    "halfMoves",
                    "maximumHalfMoves",
                    "terminationReasons",
                    "illegalActions",
                )
            },
            "speedup": {
                "minimum": report["minimumSpeedup"],
                "legalActions": report["benchmark"]["legalActions"]["speedup"],
                "applyLegalAction": report["benchmark"]["applyLegalAction"]["speedup"],
                "combined": report["speedup"],
            },
            "failures": report["failures"],
            "reportPath": str(arguments.output) if arguments.output else None,
        }
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
