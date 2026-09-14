#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f47-lineage-regression-diagnosis-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -u -m quoridor.phase4f46_result_verify --workspace-root "${ROOT}"
python3 -u -m quoridor.phase4f47_lineage_regression_diagnosis run --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f47] log: ${LOG_PATH}"
