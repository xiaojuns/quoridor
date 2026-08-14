#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONFIG="training/configs/phase4-bootstrap.json"
DEVICE="cuda"
GENERATIONS=1
TARGET_GENERATIONS=""
GENERATIONS_EXPLICIT=0
GPU_HOURS_USED=0
RESUME_MODE="auto"

usage() {
  cat <<'EOF'
Usage: ./scripts/train-phase4.sh [options]

Options:
  --config PATH          Phase 4 JSON (default: phase4-bootstrap.json)
  --device DEVICE        auto, cpu, or cuda (default: cuda)
  --generations COUNT    Generations to commit, up to pilot total 5 (default: 1)
  --target-generations N Idempotently train to cumulative generation count 0..5
  --gpu-hours-used HOURS Monotonic GPU hours outside the shared ledger
  --new                  Refuse an existing run instead of auto-resuming
  --resume               Require an existing run manifest
  -h, --help             Show this help

The default mode creates a new lineage when no run manifest exists and resumes
strictly otherwise. Ctrl+C preserves complete raw replay games and the last
checksummed checkpoint. Run the same command again to continue.
EOF
}

while (($# > 0)); do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --generations)
      GENERATIONS="$2"
      GENERATIONS_EXPLICIT=1
      shift 2
      ;;
    --target-generations)
      TARGET_GENERATIONS="$2"
      shift 2
      ;;
    --gpu-hours-used)
      GPU_HOURS_USED="$2"
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

if [[ -n "${TARGET_GENERATIONS}" ]]; then
  if ((GENERATIONS_EXPLICIT)); then
    echo "--generations and --target-generations are mutually exclusive" >&2
    exit 2
  fi
  if [[ ! "${TARGET_GENERATIONS}" =~ ^[0-5]$ ]]; then
    echo "--target-generations must be an integer in [0, 5]" >&2
    exit 2
  fi
  GENERATION_ARGS=(--target-generations "${TARGET_GENERATIONS}")
else
  if [[ ! "${GENERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--generations must be a positive integer" >&2
    exit 2
  fi
  GENERATION_ARGS=(--generations "${GENERATIONS}")
fi
if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be auto, cpu, or cuda" >&2
  exit 2
fi
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 2
fi

CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
mkdir -p "${CONTROL_ROOT}/logs"
AVAILABLE_KIB="$(df -Pk "${CONTROL_ROOT}" | awk 'NR == 2 {print $4}')"
if ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "At least 20 GiB must remain free; available KiB: ${AVAILABLE_KIB}" >&2
  exit 1
fi

RUN_DIR="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d.get("runDir", "training/runs/" + str(d.get("name", "experiment"))))' "${CONFIG}")"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT_DIR}/${RUN_DIR}"
fi
MANIFEST="${RUN_DIR}/run-manifest.json"
RESUME_ARGS=()
if [[ "${RESUME_MODE}" == "resume" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "--resume requested but run manifest is missing: ${MANIFEST}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume)
elif [[ "${RESUME_MODE}" == "new" ]]; then
  if [[ -e "${MANIFEST}" ]]; then
    echo "--new requested but run manifest already exists: ${MANIFEST}" >&2
    exit 1
  fi
elif [[ -f "${MANIFEST}" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

LOG_PATH="${CONTROL_ROOT}/logs/phase4-training-$(date +%Y%m%d-%H%M%S).log"
COMMAND=(
  python3 -m quoridor.phase4_training
  --config "${CONFIG}"
  --device "${DEVICE}"
  "${GENERATION_ARGS[@]}"
  --gpu-hours-used "${GPU_HOURS_USED}"
  "${RESUME_ARGS[@]}"
)

echo "[phase4-training] mode: ${RESUME_MODE}"
echo "[phase4-training] command: ${COMMAND[*]}"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e
echo "[phase4-training] log: ${LOG_PATH}"
exit "${STATUS}"
