#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
CONTRACT="training/configs/phase4e1-calibration-v1.json"
DEVICE_INDEX=0
GPU_HOURS_USED=0

usage() {
  cat <<'EOF'
Usage: ./scripts/calibrate-phase4e1.sh [options]

Options:
  --contract PATH          Confirmed Phase 4E1 calibration contract
  --device-index INDEX     CUDA device index (default: 0)
  --gpu-hours-used HOURS   Monotonic GPU hours outside the shared ledger
  -h, --help               Show this help

Runs the confirmed Phase 4E1 Standard-shaped self-play calibration and the
batched 80-game arena throughput smoke. Complete arena opening pairs and the
global GPU ledger are durable. Press Ctrl+C to stop, then rerun the same command
to resume. This command never trains a model, writes a training checkpoint,
freezes the Phase 4E budget, or authorizes training.
EOF
}

while (($# > 0)); do
  case "$1" in
    --contract)
      if (($# < 2)); then
        echo "--contract requires a path" >&2
        usage >&2
        exit 2
      fi
      CONTRACT="$2"
      shift 2
      ;;
    --device-index)
      if (($# < 2)); then
        echo "--device-index requires an index" >&2
        usage >&2
        exit 2
      fi
      DEVICE_INDEX="$2"
      shift 2
      ;;
    --gpu-hours-used)
      if (($# < 2)); then
        echo "--gpu-hours-used requires an hours value" >&2
        usage >&2
        exit 2
      fi
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

if [[ "${CONTRACT}" != /* ]]; then
  CONTRACT="${ROOT_DIR}/${CONTRACT}"
fi
if [[ ! -f "${CONTRACT}" ]]; then
  echo "Phase 4E1 calibration contract does not exist: ${CONTRACT}" >&2
  exit 2
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] \
  || ((AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "[phase4e1-calibration] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

ARTIFACT_LIMIT_BYTES=$((8 * 1024 * 1024 * 1024))
ARTIFACT_BYTES=0
if [[ -d "${CONTROL_ROOT}" ]]; then
  ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
fi
if [[ ! "${ARTIFACT_BYTES}" =~ ^[0-9]+$ ]] \
  || ((ARTIFACT_BYTES >= ARTIFACT_LIMIT_BYTES)); then
  echo "[phase4e1-calibration] Phase 4 artifacts reached the 8 GiB stop line" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

python3 - "${CONTRACT}" <<'PY'
import json
import sys

from quoridor.phase4e1_contract import load_phase4e1_contract

contract = load_phase4e1_contract(sys.argv[1])
print(
    json.dumps(
        {
            "calibrationContractSha256": contract.sha256,
            "calibrationGpuHourLimit": (
                contract.resources.calibration_gpu_hour_limit
            ),
            "formalPromotion": contract.formal_promotion,
            "stage": contract.stage,
            "status": contract.status,
            "trainingAllowed": contract.training_allowed,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY

echo "[phase4e1-calibration] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "[phase4e1-calibration] verifying the current-source replay input gate"
python3 -m quoridor.phase4_training --replay-input-gate

mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4e1-calibration-$(date +%Y%m%d-%H%M%S).log"
COMMAND=(
  python3 -u -m quoridor.phase4e1_calibration run
  --workspace-root "${ROOT_DIR}"
  --contract "${CONTRACT}"
  --device-index "${DEVICE_INDEX}"
  --gpu-hours-used "${GPU_HOURS_USED}"
)

echo "[phase4e1-calibration] command: ${COMMAND[*]}"
echo "[phase4e1-calibration] Ctrl+C stops safely; rerun this command to resume"
echo "[phase4e1-calibration] log: ${LOG_PATH}"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e
exit "${STATUS}"
