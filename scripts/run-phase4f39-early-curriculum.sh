#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/training:${ROOT}/native/build/python${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND="${1:-prepare}"
case "${COMMAND}" in
  prepare|verify) ;;
  *)
    echo "usage: $0 [prepare|verify]" >&2
    exit 2
    ;;
esac

python3 -u -m quoridor.phase4f39_early_curriculum "${COMMAND}" --workspace-root "${ROOT}"
