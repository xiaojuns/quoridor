#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
BASE_CONFIG="training/configs/phase4-bootstrap.json"
OUTPUT_ROOT="training/runs/phase4/phase4d-pilot"
PRETRAINING_SUMMARY=""
GPU_HOURS_USED="0"

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4d-warm-cold-pilot.sh --pretraining-summary PATH [options]

Options:
  --pretraining-summary PATH  Completed Phase 4D pretraining summary v1
  --base-config PATH          Exact canonical bootstrap config (or identical copy)
  --output-root PATH          Pilot artifact root
  --gpu-hours-used HOURS      Monotonic GPU hours outside the shared ledger
  -h, --help                  Show this help

The script is interruption-safe. It alternates cold gN then warm gN, invoking
the durable trainer with the next absolute cumulative target. Re-run this
exact command to resume. A partially written arena JSONL is never committed
as a result; the 64-game screen restarts if interrupted.
EOF
}

while (($# > 0)); do
  case "$1" in
    --pretraining-summary)
      PRETRAINING_SUMMARY="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --gpu-hours-used)
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

if [[ -z "${PRETRAINING_SUMMARY}" ]]; then
  echo "--pretraining-summary is required" >&2
  exit 2
fi
if [[ "${PRETRAINING_SUMMARY}" != /* ]]; then
  PRETRAINING_SUMMARY="${ROOT_DIR}/${PRETRAINING_SUMMARY}"
fi
if [[ "${BASE_CONFIG}" != /* ]]; then
  BASE_CONFIG="${ROOT_DIR}/${BASE_CONFIG}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}"
fi
if [[ ! -f "${PRETRAINING_SUMMARY}" ]]; then
  echo "Pretraining summary does not exist: ${PRETRAINING_SUMMARY}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config does not exist: ${BASE_CONFIG}" >&2
  exit 2
fi

CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
mkdir -p "${CONTROL_ROOT}/logs"
AVAILABLE_KIB="$(df -Pk "${CONTROL_ROOT}" | awk 'NR == 2 {print $4}')"
if ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "At least 20 GiB must remain free; available KiB: ${AVAILABLE_KIB}" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "[phase4d-pilot] verifying current-source replay input gate"
python3 -m quoridor.phase4_training --replay-input-gate

CONTRACT="${OUTPUT_ROOT}/pilot-contract.json"
python3 -m quoridor.phase4_pilot prepare \
  --pretraining-summary "${PRETRAINING_SUMMARY}" \
  --base-config "${BASE_CONFIG}" \
  --output-root "${OUTPUT_ROOT}"

while true; do
  read -r NEXT_LANE TARGET_GENERATIONS < <(
    python3 -m quoridor.phase4_pilot next-step --contract "${CONTRACT}"
  )
  if [[ "${NEXT_LANE}" == "complete" ]]; then
    break
  fi
  if [[ "${NEXT_LANE}" != "cold" && "${NEXT_LANE}" != "warm" ]]; then
    echo "Pilot returned an invalid next lane: ${NEXT_LANE}" >&2
    exit 1
  fi
  CONFIG="${OUTPUT_ROOT}/effective-configs/${NEXT_LANE}.json"
  if [[ ! "${TARGET_GENERATIONS}" =~ ^[1-5]$ ]]; then
    echo "Pilot returned an invalid absolute target: ${TARGET_GENERATIONS}" >&2
    exit 1
  fi
  echo "[phase4d-pilot] training ${NEXT_LANE} to ${TARGET_GENERATIONS}/5"
  "${ROOT_DIR}/scripts/train-phase4.sh" \
    --config "${CONFIG}" \
    --device cuda \
    --target-generations "${TARGET_GENERATIONS}" \
    --gpu-hours-used "${GPU_HOURS_USED}"
  python3 -m quoridor.phase4_pilot validate-lane \
    --contract "${CONTRACT}" \
    --lane "${NEXT_LANE}"
done

python3 -m quoridor.phase4_pilot validate-lane \
  --contract "${CONTRACT}" \
  --lane cold
python3 -m quoridor.phase4_pilot validate-lane \
  --contract "${CONTRACT}" \
  --lane warm

echo "[phase4d-pilot] starting 64-game paired screen; partial arena files are not results"
python3 -m quoridor.phase4_pilot arena \
  --contract "${CONTRACT}" \
  --gpu-hours-used "${GPU_HOURS_USED}"
