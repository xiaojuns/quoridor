#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
OUTPUT_ROOT="${ROOT_DIR}/training/runs/phase4/phase4f/recovery-challenge-data"

usage() {
  cat <<'EOF'
Usage: ./scripts/generate-phase4f-challenge-data.sh [--verify-only]

Generates 30,000 CPU-only classic-hard teacher positions for Phase 4F
recovery. Every playable position and horizontal mirror from the legacy dev,
legacy final, recovery dev, and recovery final suites is excluded. The run is
resumable; Ctrl+C commits the current replay state and rerunning continues it.
EOF
}

VERIFY_ARGS=()
while (($# > 0)); do
  case "$1" in
    --verify-only) VERIFY_ARGS+=(--verify-only); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f2-challenge] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

exec python3 -u -m quoridor.teacher_data \
  --output-root "${OUTPUT_ROOT}" \
  --artifact-root "${ROOT_DIR}/training/runs/phase4" \
  --target-positions 30000 \
  --seed 20260802 \
  --node-budget 587 \
  --max-depth 5 \
  --exploration-probability 0.10 \
  --opening-depths 0 4 8 12 16 \
  --maximum-attempts 250000 \
  --exclude-suite training/fixtures/openings/dev.json \
  --exclude-suite training/fixtures/openings/final.json \
  --exclude-suite training/fixtures/openings/recovery-v1/recovery-dev-v1.json \
  --exclude-suite training/fixtures/openings/recovery-v1/recovery-final-v1.json \
  --exclude-arena-result training/runs/phase4/phase4f/fixed/result.json \
  --exclude-arena-result training/runs/phase4/phase4f/fixed-standard-initial/result.json \
  --recovery-preparation-report training/runs/phase4/phase4f/recovery-preparation/report.json \
  "${VERIFY_ARGS[@]}"
