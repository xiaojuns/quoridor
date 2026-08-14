#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"

usage() {
  cat <<'EOF'
Usage: ./scripts/rebaseline-phase4e.sh COMMAND [options]

Commands:
  supersede-e1   Verify and quarantine the superseded completed E1 evidence
  freeze-rebase  Freeze replacement E1 evidence and rebase the Phase 4E ledger

The command is CPU-only and defaults to a verification-only dry run. Pass
--execute explicitly to commit the requested operation. Run with COMMAND
--help for the full command-specific arguments.
EOF
}

if (($# == 0)); then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$1" != "supersede-e1" && "$1" != "freeze-rebase" ]]; then
  echo "Unsupported command: $1" >&2
  usage >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training${PYTHONPATH:+:${PYTHONPATH}}"

echo "[phase4e-rebaseline] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
python3 -u -m quoridor.phase4e_rebaseline "$@"
