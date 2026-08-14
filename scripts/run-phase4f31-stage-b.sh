#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
CONFIG="${ROOT_DIR}/training/configs/phase4f31-stage-b-extended-v1.json"
EXECUTION="${ROOT_DIR}/training/configs/phase4f31-stage-b-execution-frozen-v1.json"
RUN_DIR="${ROOT_DIR}/training/runs/phase4/phase4f/recovery-v3-stage-b-extended-v1"
DEVICE_INDEX=0
GPU_HOURS_USED=0

while (($# > 0)); do
  case "$1" in
    --device-index) DEVICE_INDEX="$2"; shift 2 ;;
    --gpu-hours-used) GPU_HOURS_USED="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./scripts/run-phase4f31-stage-b.sh [--device-index N] [--gpu-hours-used HOURS]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${DEVICE_INDEX}" =~ ^[0-9]+$ ]] || { echo "invalid CUDA device index" >&2; exit 2; }
[[ -f "${CONFIG}" && -f "${EXECUTION}" ]] || { echo "Phase 4F31 preparation is missing" >&2; exit 1; }
AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
((AVAILABLE_KIB >= 20 * 1024 * 1024)) || { echo "[phase4f31] below 20 GiB disk stop line" >&2; exit 1; }
((ARTIFACT_BYTES + 2 * 1024 * 1024 * 1024 < 12 * 1024 * 1024 * 1024)) || { echo "[phase4f31] 12 GiB cap lacks 2 GiB reserve" >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
cd "${ROOT_DIR}"

echo "[phase4f31] building and testing native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
python3 -u -m quoridor.phase4f31_stage_b verify-preparation --workspace-root "${ROOT_DIR}"
mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4f31-stage-b-$(date +%Y%m%d-%H%M%S).log"
echo "[phase4f31] log: ${LOG_PATH}"

echo "[phase4f31] running current-source CUDA AMP memory smoke"
python3 -u -m quoridor.training_memory_smoke --device-index 0 --gpu-hours-used "${GPU_HOURS_USED}" --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"
echo "[phase4f31] running current-source CPU replay input gate"
python3 -u -m quoridor.phase4_training --replay-input-gate --config "${CONFIG}" --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"

RESUME_ARGS=()
[[ -f "${RUN_DIR}/run-manifest.json" ]] && RESUME_ARGS+=(--resume)
echo "[phase4f31] continuing exact F27 state through generation 5"
python3 -u -m quoridor.phase4_training \
  --config "${CONFIG}" \
  --execution-contract "${EXECUTION}" \
  --device cuda \
  --target-generations 6 \
  --gpu-hours-used "${GPU_HOURS_USED}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "[phase4f31] reviewing extended Stage B gates"
python3 -u -m quoridor.phase4f31_stage_b review --workspace-root "${ROOT_DIR}" 2>&1 | tee -a "${LOG_PATH}"
echo "[phase4f31] log: ${LOG_PATH}"
