#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f48-regression-safe-redesign-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -u -m quoridor.phase4f48_regression_safe_redesign run --workspace-root "${ROOT}" 2>&1 | tee "${LOG_PATH}"
echo "[phase4f48] log: ${LOG_PATH}"
