#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
REPORT_DIR="${ROOT_DIR}/training/runs/native-validation"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
VALIDATION_REPORT="${REPORT_DIR}/rules-${TIMESTAMP}.json"
SMOKE_REPORT="${REPORT_DIR}/training-smoke-${TIMESTAMP}.json"

if (($# != 0)); then
  echo "run-phase3-validation.sh takes no arguments; the acceptance workload is fixed" >&2
  exit 2
fi

mkdir -p "${REPORT_DIR}"
"${ROOT_DIR}/scripts/build-native-engine.sh"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "[phase3] Python and integration tests"
python3 -m unittest discover -s "${ROOT_DIR}/training/tests"

echo "[phase3] 10,000-state differential, long games, and throughput gates"
python3 "${ROOT_DIR}/scripts/validate-native-engine.py" \
  --states 10000 \
  --games 100 \
  --seed 0x5eed1234 \
  --output "${VALIDATION_REPORT}" \
  --summary-only

echo "[phase3] Batched native-MCTS training smoke"
python3 "${ROOT_DIR}/scripts/smoke-native-training.py" \
  --device auto \
  --output "${SMOKE_REPORT}" \
  --summary-only

echo "[phase3] validation report: ${VALIDATION_REPORT}"
echo "[phase3] training smoke report: ${SMOKE_REPORT}"
echo "[phase3] PASS"
