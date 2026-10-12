"""M0 entry points. Legacy experiments are never invoked implicitly."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .project_config import ConfigError, load_project_config


# ``smoke`` remains the historical M0 direct-call placeholder so the old
# contract test cannot accidentally launch work. The supported M4 Make target
# invokes the explicit ``smoke-run`` command below.  M5 arena execution is
# enabled only when both immutable model inputs are supplied; retaining the
# no-input response keeps an accidental legacy invocation fail-closed.
NOT_IMPLEMENTED_MILESTONES = {"smoke": "M4", "arena": "M5", "experiment": "M6"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quoridor M0 project interfaces")
    parser.add_argument("command", choices=["check-config", "smoke-run", *NOT_IMPLEMENTED_MILESTONES])
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--candidate", help="M5 model-only candidate checkpoint")
    parser.add_argument("--champion", help="M5 model-only champion checkpoint")
    parser.add_argument("--candidate-sha256", help="optional exact candidate SHA-256")
    parser.add_argument("--champion-sha256", help="optional exact champion SHA-256")
    args = parser.parse_args(argv)
    model_arguments = (args.candidate, args.champion, args.candidate_sha256, args.champion_sha256)
    if any(value is not None for value in model_arguments) and (
        args.command != "arena" or not args.candidate or not args.champion
    ):
        print(json.dumps({
            "schema_version": 1, "event": "input_error",
            "message": "arena requires explicit nonempty --candidate and --champion; model inputs belong only to arena",
        }, sort_keys=True), file=sys.stderr)
        return 2
    try:
        config = load_project_config(
            args.config,
            expected_mode=None if args.command == "check-config" else ("smoke" if args.command == "smoke-run" else args.command),
        )
    except ConfigError as error:
        print(json.dumps({"schema_version": 1, "event": "config_error", "message": str(error)}), file=sys.stderr)
        return 2

    if args.command == "check-config":
        print(json.dumps({
            "schema_version": 1, "event": "config_validated", "mode": config.mode,
            "config_sha256": config.sha256, "frozen": config.mode == "arena",
        }, sort_keys=True))
        return 0

    if args.command == "smoke-run":
        from .smoke import run_smoke

        try:
            result = run_smoke(config)
        except Exception as error:
            print(json.dumps({
                "schema_version": 1, "event": "smoke_failed", "message": str(error),
            }, sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps({"schema_version": 1, "event": "smoke_completed", **result}, sort_keys=True))
        return 0

    # Keep the argument-less historical command an explicit stub.  The Make
    # target and new callers provide both model identities and take the M5 path.
    if args.command == "arena" and args.candidate and args.champion:
        from .arena_v1 import ArenaError, ArenaExecutionError, run_arena
        from .project_runtime import BudgetExpired

        try:
            result = run_arena(
                config,
                args.candidate,
                args.champion,
                candidate_sha256=args.candidate_sha256,
                champion_sha256=args.champion_sha256,
            )
        except KeyboardInterrupt as error:
            print(json.dumps({
                "schema_version": 1, "event": "arena_interrupted", "message": str(error),
            }, sort_keys=True), file=sys.stderr)
            return 130
        except (BudgetExpired, ArenaExecutionError) as error:
            print(json.dumps({
                "schema_version": 1, "event": "arena_incomplete", "message": str(error),
            }, sort_keys=True), file=sys.stderr)
            return 1
        except (ArenaError, OSError, ValueError, TypeError) as error:
            print(json.dumps({
                "schema_version": 1, "event": "arena_failed", "message": str(error),
            }, sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps({"schema_version": 1, "event": "arena_completed", **result}, sort_keys=True))
        return 0

    milestone = NOT_IMPLEMENTED_MILESTONES[args.command]
    print(json.dumps({
        "schema_version": 1, "event": "not_implemented", "command": args.command,
        "required_milestone": milestone,
        "message": f"{args.command} is NOT IMPLEMENTED; scheduled for {milestone}",
    }, sort_keys=True), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
