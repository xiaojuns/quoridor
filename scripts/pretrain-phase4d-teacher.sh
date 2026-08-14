#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
RUN_ROOT="${CONTROL_ROOT}/teacher-pretraining-v2"
TEACHER_ROOT="${CONTROL_ROOT}/teacher-data"
GPU_HOURS_USED="0"
DEVICE_INDEX="0"
RESUME_MODE="auto"

usage() {
  cat <<'EOF'
Usage: ./scripts/pretrain-phase4d-teacher.sh [options]

Options:
  --device-index INDEX    CUDA device index (default: 0)
  --gpu-hours-used HOURS  Monotonic Phase 4 GPU hours outside the shared ledger
  --run-root PATH         Pretraining output root
  --new                   Refuse an existing pretraining lineage
  --resume                Require and strictly resume an existing lineage
  -h, --help              Show this help

The workload is frozen at 192x16, seed 20260714, batch 512, CUDA AMP,
512 optimizer steps, and a full fixed-holdout evaluation every 64 steps.
During training, Ctrl+C commits a checksummed completed-step checkpoint.
During dataset validation or CPU cache construction, rerun this launcher;
those stages are rebuilt before GPU accounting begins.
Progress, evaluation, checkpoint, and final interrupt records are flushed to
the timestamped log under training/runs/phase4/logs.
EOF
}

while (($# > 0)); do
  case "$1" in
    --device-index)
      DEVICE_INDEX="$2"
      shift 2
      ;;
    --gpu-hours-used)
      GPU_HOURS_USED="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --new)
      RESUME_MODE="new"
      shift
      ;;
    --resume)
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

if [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--device-index must be a non-negative integer" >&2
  exit 2
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${ROOT_DIR}/${RUN_ROOT}"
fi
if [[ ! -f "${TEACHER_ROOT}/teacher-manifest.json" ]]; then
  echo "Completed teacher manifest is missing: ${TEACHER_ROOT}/teacher-manifest.json" >&2
  exit 1
fi

mkdir -p "${CONTROL_ROOT}/logs"
AVAILABLE_KIB="$(df -Pk "${CONTROL_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "At least 20 GiB must remain free" >&2
  exit 1
fi

MANIFEST="${RUN_ROOT}/run-manifest.json"
RESUME_ARGS=()
if [[ "${RESUME_MODE}" == "resume" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "--resume requested but run manifest is missing: ${MANIFEST}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume)
elif [[ "${RESUME_MODE}" == "new" ]]; then
  if [[ -e "${MANIFEST}" ]]; then
    echo "--new requested but run manifest exists: ${MANIFEST}" >&2
    exit 1
  fi
elif [[ -f "${MANIFEST}" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

LOG_PATH="${CONTROL_ROOT}/logs/phase4d-teacher-pretrain-$(date +%Y%m%d-%H%M%S).log"
COMMAND=(
  python3 -u -m quoridor.teacher_pretraining
  --run-root "${RUN_ROOT}"
  --device-index "${DEVICE_INDEX}"
  --gpu-hours-used "${GPU_HOURS_USED}"
  "${RESUME_ARGS[@]}"
)

echo "[phase4d-teacher-pretrain] mode: ${RESUME_MODE}"
echo "[phase4d-teacher-pretrain] command: ${COMMAND[*]}"
set +e
"${COMMAND[@]}" 2>&1 | tee -i -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e
echo "[phase4d-teacher-pretrain] log: ${LOG_PATH}"
exit "${STATUS}"
