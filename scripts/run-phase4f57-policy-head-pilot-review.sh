#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f57-policy-head-pilot-review-$(date +%Y%m%d-%H%M%S).log"

if (($# > 0)); then
  case "$1" in
    -h|--help)
      echo "Usage: $0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
fi

mkdir -p "${LOG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -u -m quoridor.phase4f56_policy_head_pilot verify \
  --workspace-root "${ROOT}"
python3 -u -m quoridor.phase4f57_policy_head_pilot_review run \
  --workspace-root "${ROOT}" 2>&1 | tee "${LOG_PATH}"
echo "[phase4f57] log: ${LOG_PATH}"
