#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f63-value-calibration-review-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "[phase4f63] running CPU-only value-calibration design review"
python3 -u -m quoridor.phase4f63_value_calibration_review run \
  --workspace-root "${ROOT}" 2>&1 | tee "${LOG_PATH}"

echo "[phase4f63] running focused verification"
python3 -m pytest -q "${ROOT}/training/tests/test_phase4f63_value_calibration_review.py"

echo "[phase4f63] log: ${LOG_PATH}"
