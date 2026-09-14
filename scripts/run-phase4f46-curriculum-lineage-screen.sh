#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f46-curriculum-lineage-screen-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT}/training/runs/phase4/phase4f/f46-curriculum-lineage-screen-v1/screen-gate.json" ]]; then
  python3 -u -m quoridor.phase4f46_result_verify --workspace-root "${ROOT}" 2>&1 | tee "${LOG_PATH}"
  echo "[phase4f46] log: ${LOG_PATH}"
  exit 0
fi

python3 -u -m quoridor.phase4f45_curriculum_pilot_review verify --workspace-root "${ROOT}"
python3 -u -m quoridor.phase4f46_curriculum_lineage_screen run --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f46] log: ${LOG_PATH}"
