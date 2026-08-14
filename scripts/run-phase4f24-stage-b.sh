#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
CONFIG="${ROOT_DIR}/training/configs/phase4f24-stage-b-block-00-v1.json"
EXECUTION="${ROOT_DIR}/training/configs/phase4f24-stage-b-execution-frozen-v1.json"
RUN_DIR="${ROOT_DIR}/training/runs/phase4/phase4f/recovery-v3-stage-b-block-00-v1"
DEVICE_INDEX=0
GPU_HOURS_USED=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f24-stage-b.sh [options]

Options:
  --device-index INDEX     CUDA device index (default: 0)
  --gpu-hours-used HOURS   Monotonic unledgered GPU hours (default: 0)
  -h, --help               Show this help

Runs the independent two-generation pure-self-play Stage B block. It starts
from the F21 Stage A best model with empty optimizer and replay state. Ctrl+C
preserves completed work; rerun the identical command to resume. The review
does not start the subsequent external screen.
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
if [[ ! -f "${CONFIG}" || ! -f "${EXECUTION}" ]]; then
  echo "Phase 4F24 preparation artifacts are missing" >&2
  exit 1
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] \
  || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f24] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi
if [[ ! "${ARTIFACT_BYTES}" =~ ^[0-9]+$ ]] \
  || ((ARTIFACT_BYTES + 2 * 1024 * 1024 * 1024 >= 12 * 1024 * 1024 * 1024)); then
  echo "[phase4f24] 12 GiB cap cannot preserve the 2 GiB write reserve" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
cd "${ROOT_DIR}"

echo "[phase4f24] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f24] verifying frozen CPU preparation"
python3 -u -m quoridor.phase4f24_stage_b verify-preparation \
  --workspace-root "${ROOT_DIR}"

mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4f24-stage-b-$(date +%Y%m%d-%H%M%S).log"
echo "[phase4f24] log: ${LOG_PATH}"

echo "[phase4f24] running current-source CUDA AMP memory smoke"
python3 -u -m quoridor.training_memory_smoke \
  --device-index 0 \
  --gpu-hours-used "${GPU_HOURS_USED}" \
  --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"

echo "[phase4f24] running current-source CPU replay input gate"
python3 -u -m quoridor.phase4_training \
  --replay-input-gate \
  --config "${CONFIG}" \
  --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/run-manifest.json" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "[phase4f24] running or resuming formal Stage B to generation 2"
python3 -u -m quoridor.phase4_training \
  --config "${CONFIG}" \
  --execution-contract "${EXECUTION}" \
  --device cuda \
  --target-generations 2 \
  --gpu-hours-used "${GPU_HOURS_USED}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "[phase4f24] reviewing formal Stage B gates"
python3 -u -m quoridor.phase4f24_stage_b review \
  --workspace-root "${ROOT_DIR}" 2>&1 | tee -a "${LOG_PATH}"

echo "[phase4f24] log: ${LOG_PATH}"
