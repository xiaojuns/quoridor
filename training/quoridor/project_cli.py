"""M0 entry points. Legacy experiments are never invoked implicitly."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .project_config import ConfigError, load_project_config


NOT_IMPLEMENTED_MILESTONES = {"smoke": "M4", "arena": "M5", "experiment": "M6"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quoridor M0 project interfaces")
    parser.add_argument("command", choices=["check-config", *NOT_IMPLEMENTED_MILESTONES])
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args(argv)
    try:
        config = load_project_config(
            args.config,
            expected_mode=None if args.command == "check-config" else args.command,
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

    milestone = NOT_IMPLEMENTED_MILESTONES[args.command]
    print(json.dumps({
        "schema_version": 1, "event": "not_implemented", "command": args.command,
        "required_milestone": milestone,
        "message": f"{args.command} is NOT IMPLEMENTED; scheduled for {milestone}",
    }, sort_keys=True), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
