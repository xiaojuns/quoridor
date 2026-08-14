#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
CONFIG="${ROOT_DIR}/training/configs/phase4f38-stage-a2-policy-only-pilot-v1.json"
EXECUTION="${ROOT_DIR}/training/configs/phase4f38-stage-a2-policy-only-pilot-execution-frozen-v1.json"
RUN_DIR="${ROOT_DIR}/training/runs/phase4/phase4f/stage-a2-policy-only-pilot-v1"
DEVICE_INDEX=0
GPU_HOURS_USED=0

while (($# > 0)); do
  case "$1" in
    --device-index) DEVICE_INDEX="$2"; shift 2 ;;
    --gpu-hours-used) GPU_HOURS_USED="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--device-index INDEX] [--gpu-hours-used HOURS]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${DEVICE_INDEX}" =~ ^[0-9]+$ ]] || { echo "invalid device index" >&2; exit 2; }
[[ -f "${CONFIG}" && -f "${EXECUTION}" ]] || { echo "F38 artifacts are missing" >&2; exit 1; }
AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
((AVAILABLE_KIB >= 20 * 1024 * 1024)) || { echo "[phase4f38] below 20 GiB disk stop line" >&2; exit 1; }
((ARTIFACT_BYTES + 1250 * 1024 * 1024 < 16 * 1024 * 1024 * 1024)) || { echo "[phase4f38] 16 GiB cap lacks the frozen reserve" >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
cd "${ROOT_DIR}"

"${ROOT_DIR}/scripts/build-native-engine.sh"
python3 -u -m quoridor.phase4f37_policy_only verify-preparation --workspace-root "${ROOT_DIR}"
mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4f38-stage-a2-policy-only-pilot-$(date +%Y%m%d-%H%M%S).log"
echo "[phase4f38] log: ${LOG_PATH}"
echo "[phase4f38] running current-source CUDA AMP memory smoke"
python3 -u -m quoridor.training_memory_smoke \
  --device-index 0 --gpu-hours-used "${GPU_HOURS_USED}" \
  --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"
echo "[phase4f38] running 16 GiB-bound CPU replay input gate"
python3 -u -m quoridor.phase4_training \
  --replay-input-gate --config "${CONFIG}" \
  --execution-contract "${EXECUTION}" 2>&1 | tee -a "${LOG_PATH}"

RESUME_ARGS=()
[[ -f "${RUN_DIR}/run-manifest.json" ]] && RESUME_ARGS+=(--resume)
echo "[phase4f38] training isolated policy-only Stage A2 pilot to generation 1"
python3 -u -m quoridor.phase4_training \
  --config "${CONFIG}" --execution-contract "${EXECUTION}" \
  --device cuda --target-generations 1 --gpu-hours-used "${GPU_HOURS_USED}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"
echo "[phase4f38] log: ${LOG_PATH}"
