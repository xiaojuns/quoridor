#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/training${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -u -m quoridor.phase4f_failure_analysis "$@"
