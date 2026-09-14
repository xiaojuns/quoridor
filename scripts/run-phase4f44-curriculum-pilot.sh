#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f44-curriculum-pilot-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -u -m quoridor.phase4f43_curriculum_pilot verify --workspace-root "${ROOT}"
python3 -u -m quoridor.phase4f44_curriculum_pilot run --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f44] log: ${LOG_PATH}"
