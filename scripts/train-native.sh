#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONFIG="training/configs/reference.json"
DEVICE="cuda"
ITERATIONS=1

usage() {
  cat <<'EOF'
Usage: ./scripts/train-native.sh [options]

Options:
  --config PATH       Experiment JSON (default: training/configs/reference.json)
  --device DEVICE     auto, cpu, or cuda (default: cuda)
  --iterations COUNT  Completed checkpoint iterations to run (default: 1)
  -h, --help          Show this help

The script resumes automatically when the configured run directory contains a
checkpoint. Ctrl+C stops the current iteration; the last completed checkpoint
remains available for the next invocation.
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
    --iterations)
      ITERATIONS="$2"
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

if [[ ! "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--iterations must be a positive integer" >&2
  exit 2
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

RUN_DIR="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d.get("runDir", "training/runs/" + str(d.get("name", "experiment"))))' "${CONFIG}")"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT_DIR}/${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}/logs"
AVAILABLE_KIB="$(df -Pk "${RUN_DIR}" | awk 'NR == 2 {print $4}')"
if ((AVAILABLE_KIB < 12 * 1024 * 1024)); then
  echo "At least 12 GiB must be free on the run directory filesystem; available KiB: ${AVAILABLE_KIB}" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

LOG_PATH="${RUN_DIR}/logs/native-$(date +%Y%m%d-%H%M%S).log"

trap 'echo "[train-native] interrupted; the last completed checkpoint is unchanged"; exit 130' INT TERM

for ((iteration = 1; iteration <= ITERATIONS; ++iteration)); do
  COMMAND=(python3 -m quoridor.cli train --engine native --summary-only --device "${DEVICE}" --config "${CONFIG}")
  shopt -s nullglob
  CHECKPOINTS=("${RUN_DIR}"/checkpoints/checkpoint-*.pt)
  shopt -u nullglob
  if ((${#CHECKPOINTS[@]} > 0)); then
    COMMAND+=(--resume)
    if [[ ! -f "${RUN_DIR}/checkpoints/latest.json" ]]; then
      echo "[train-native] latest.json is missing; recovering from the newest checkpoint file"
    fi
  elif [[ -f "${RUN_DIR}/checkpoints/latest.json" ]]; then
    echo "latest.json exists but no checkpoint file is available" >&2
    exit 1
  fi
  echo "[train-native] iteration ${iteration}/${ITERATIONS}"
  echo "[train-native] command: ${COMMAND[*]}"
  "${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
done

echo "[train-native] log: ${LOG_PATH}"
echo "[train-native] completed ${ITERATIONS} iteration(s)"
