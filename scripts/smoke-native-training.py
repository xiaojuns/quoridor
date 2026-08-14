#!/usr/bin/env python3
"""Run a disposable batched native-MCTS training smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from quoridor.cli import run_smoke  # noqa: E402
from quoridor.config import load_experiment_config  # noqa: E402
from quoridor.storage import atomic_json_save  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(ROOT / "training/configs/smoke.json")
    with tempfile.TemporaryDirectory(prefix="quoridor-native-smoke-") as directory:
        config = replace(
            config,
            run_dir=Path(directory),
            self_play=replace(config.self_play, games=2, actors=2),
        )
        first = run_smoke(
            config,
            device_override=args.device,
            engine="native",
        )
        resumed = run_smoke(
            config,
            device_override=args.device,
            engine="native",
            resume=True,
        )
    for summary in (first, resumed):
        summary["checkpoint"]["ephemeral"] = True
    gates = {
        "nativeEngine": first["engine"] == resumed["engine"] == "native",
        "batchedActors": first["selfPlay"]["actors"] == 2,
        "zeroIllegalActions": (
            first["selfPlay"]["illegalActions"]
            + resumed["selfPlay"]["illegalActions"]
            == 0
        ),
        "checkpointResumed": resumed["checkpoint"]["resumedFrom"] is not None,
        "stepAdvanced": resumed["training"]["steps"] > first["training"]["steps"],
        "generationAdvanced": resumed["generation"] == first["generation"] + 1,
        "seedAdvanced": resumed["generationSeed"] != first["generationSeed"],
        "replayGrew": resumed["training"]["replaySize"] > first["training"]["replaySize"],
    }
    report = {
        "schemaVersion": 1,
        "firstRun": first,
        "resumedRun": resumed,
        "gates": gates,
        "passed": all(gates.values()),
    }
    if args.output is not None:
        atomic_json_save(report, args.output)
    output = report
    if args.summary_only:
        def compact(summary: dict) -> dict:
            return {
                "device": summary["device"],
                "generation": summary["generation"],
                "generationSeed": summary["generationSeed"],
                "steps": summary["training"]["steps"],
                "replaySize": summary["training"]["replaySize"],
                "selfPlayGames": summary["selfPlay"]["games"],
                "selfPlayMoves": summary["selfPlay"]["moves"],
                "illegalActions": summary["selfPlay"]["illegalActions"],
                "arenaAccepted": summary["arena"]["accepted"],
                "artifactBytes": summary["artifacts"]["bytes"],
                "elapsedSeconds": summary["elapsedSeconds"],
            }

        output = {
            "schemaVersion": report["schemaVersion"],
            "passed": report["passed"],
            "gates": report["gates"],
            "firstRun": compact(first),
            "resumedRun": compact(resumed),
            "reportPath": str(args.output) if args.output else None,
        }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
