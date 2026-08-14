#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
PILOT_ROOT="${CONTROL_ROOT}/phase4f5/rebaseline/pilot-v3"
DEVICE_INDEX=0
GPU_HOURS_USED=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f5-rebaseline.sh [options]

Options:
  --device-index INDEX     CUDA device index (default: 0)
  --gpu-hours-used HOURS   Monotonic GPU hours used outside the shared ledger
  -h, --help               Show this help

Runs the confirmed Phase 4F5 CUDA memory smoke and one-generation recovery
pilot. Complete self-play games, the pilot checkpoint, and GPU accounting are
durable. Ctrl+C stops safely; rerun the same command to resume. This script
freezes a formal recovery contract but does not start formal recovery training.
EOF
}

while (($# > 0)); do
  case "$1" in
    --device-index)
      (($# >= 2)) || { echo "--device-index requires a value" >&2; exit 2; }
      DEVICE_INDEX="$2"
      shift 2
      ;;
    --gpu-hours-used)
      (($# >= 2)) || { echo "--gpu-hours-used requires a value" >&2; exit 2; }
      GPU_HOURS_USED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--device-index must be a non-negative integer" >&2
  exit 2
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] \
  || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f5] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi
if [[ ! "${ARTIFACT_BYTES}" =~ ^[0-9]+$ ]] \
  || ((ARTIFACT_BYTES >= 8 * 1024 * 1024 * 1024)); then
  echo "[phase4f5] Phase 4 artifacts reached the 8 GiB cap" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f5] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f5] verifying the frozen CPU preparation"
python3 -u -m quoridor.phase4f5_rebaseline verify-preparation \
  --workspace-root "${ROOT_DIR}"

echo "[phase4f5] verifying the current-source CPU replay input gate"
python3 -u -m quoridor.phase4_training --replay-input-gate

mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4f5-rebaseline-$(date +%Y%m%d-%H%M%S).log"
echo "[phase4f5] log: ${LOG_PATH}"

echo "[phase4f5] running or reusing the CUDA AMP memory smoke"
python3 -u -m quoridor.phase4f5_rebaseline memory-smoke \
  --workspace-root "${ROOT_DIR}" \
  --device-index "${DEVICE_INDEX}" \
  --gpu-hours-used "${GPU_HOURS_USED}" 2>&1 | tee -a "${LOG_PATH}"

RESUME_ARGS=()
if [[ -f "${PILOT_ROOT}/run-manifest.json" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "[phase4f5] running or resuming the isolated one-generation pilot"
python3 -u -m quoridor.phase4_training \
  --config training/configs/phase4f-recovery-pilot-v3.json \
  --execution-contract training/configs/phase4f5-calibration-execution-frozen-v1.json \
  --device cuda \
  --target-generations 1 \
  --gpu-hours-used "${GPU_HOURS_USED}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "[phase4f5] verifying the pilot and freezing the formal recovery contract"
python3 -u -m quoridor.phase4f5_rebaseline finalize \
  --workspace-root "${ROOT_DIR}" 2>&1 | tee -a "${LOG_PATH}"
