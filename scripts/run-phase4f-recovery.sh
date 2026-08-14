#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
PREPARATION="${ROOT_DIR}/training/configs/phase4f6-recovery-prepared-v1.json"
CONFIG="${ROOT_DIR}/training/configs/phase4f-recovery-v2.json"
EXECUTION_CONTRACT="${ROOT_DIR}/training/configs/phase4f-recovery-execution-frozen-v1.json"
RUN_DIR="${ROOT_DIR}/training/runs/phase4/phase4f/recovery-v2"
DEVICE_INDEX=0
GPU_HOURS_USED=0
TARGET_GENERATIONS=4
RESUME_MODE="auto"

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f-recovery.sh [options]

Options:
  --target-generations N   Absolute committed-generation target, 0..16 (default: 4)
  --device-index INDEX     CUDA device index (default: 0)
  --gpu-hours-used HOURS   Monotonic GPU hours used outside the shared ledger
  --new                    Refuse an existing formal run manifest
  --resume                 Require an existing formal run manifest
  -h, --help               Show this help

The first invocation creates the approved recovery-v2 lineage. Ctrl+C keeps
complete replay games and the last committed checkpoint. Re-run the same
target to resume idempotently, or raise --target-generations after review.
EOF
}

while (($# > 0)); do
  case "$1" in
    --target-generations)
      (($# >= 2)) || { echo "--target-generations requires a value" >&2; exit 2; }
      TARGET_GENERATIONS="$2"
      shift 2
      ;;
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
    --new)
      [[ "${RESUME_MODE}" == "auto" ]] || { echo "--new and --resume are mutually exclusive" >&2; exit 2; }
      RESUME_MODE="new"
      shift
      ;;
    --resume)
      [[ "${RESUME_MODE}" == "auto" ]] || { echo "--new and --resume are mutually exclusive" >&2; exit 2; }
      RESUME_MODE="resume"
      shift
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

if [[ ! "${TARGET_GENERATIONS}" =~ ^[0-9]+$ ]] \
  || ((10#${TARGET_GENERATIONS} > 16)); then
  echo "--target-generations must be an integer in [0, 16]" >&2
  exit 2
fi
if [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--device-index must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -f "${PREPARATION}" || ! -f "${CONFIG}" || ! -f "${EXECUTION_CONTRACT}" ]]; then
  echo "Phase 4F6 preparation, recovery config, and execution contract must exist" >&2
  exit 1
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] \
  || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f-recovery] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi
if [[ ! "${ARTIFACT_BYTES}" =~ ^[0-9]+$ ]] \
  || ((ARTIFACT_BYTES >= 8 * 1024 * 1024 * 1024)); then
  echo "[phase4f-recovery] Phase 4 artifacts reached the 8 GiB cap" >&2
  exit 1
fi

MANIFEST="${RUN_DIR}/run-manifest.json"
RESUME_ARGS=()
if [[ "${RESUME_MODE}" == "resume" ]]; then
  [[ -f "${MANIFEST}" ]] || { echo "--resume requested but run manifest is missing: ${MANIFEST}" >&2; exit 1; }
  RESUME_ARGS+=(--resume)
elif [[ "${RESUME_MODE}" == "new" ]]; then
  [[ ! -e "${MANIFEST}" ]] || { echo "--new requested but run manifest exists: ${MANIFEST}" >&2; exit 1; }
elif [[ -f "${MANIFEST}" ]]; then
  RESUME_ARGS+=(--resume)
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f-recovery] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f-recovery] verifying Phase 4F6 launch preparation"
python3 -u -m quoridor.phase4f6_recovery verify --workspace-root "${ROOT_DIR}"

echo "[phase4f-recovery] verifying the current-source CPU replay input gate"
python3 -u -m quoridor.phase4_training --replay-input-gate

mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4f-recovery-$(date +%Y%m%d-%H%M%S).log"
COMMAND=(
  python3 -u -m quoridor.phase4_training
  --config "${CONFIG}"
  --execution-contract "${EXECUTION_CONTRACT}"
  --device cuda
  --target-generations "${TARGET_GENERATIONS}"
  --gpu-hours-used "${GPU_HOURS_USED}"
  "${RESUME_ARGS[@]}"
)

export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
echo "[phase4f-recovery] mode: ${RESUME_MODE}"
echo "[phase4f-recovery] target generations: ${TARGET_GENERATIONS}"
echo "[phase4f-recovery] command: ${COMMAND[*]}"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e
echo "[phase4f-recovery] log: ${LOG_PATH}"
exit "${STATUS}"
