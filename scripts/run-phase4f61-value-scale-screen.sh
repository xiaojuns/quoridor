#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f61-value-scale-screen-$(date +%Y%m%d-%H%M%S).log"
DEVICE_INDEX=0

while (($# > 0)); do
  case "$1" in
    --device-index)
      (($# >= 2)) || { echo "--device-index requires a value" >&2; exit 2; }
      DEVICE_INDEX="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--device-index INDEX]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "${DEVICE_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "--device-index must be a non-negative integer" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
cd "${ROOT}"

echo "[phase4f61] building and testing the native engine"
"${ROOT}/scripts/build-native-engine.sh"

echo "[phase4f61] running CPU-only unit tests"
python3 -m pytest -q \
  "${ROOT}/training/tests/test_phase4f61_value_scale_screen.py"

echo "[phase4f61] freezing the inference-only value-scale screen"
python3 -u -m quoridor.phase4f61_value_scale_screen prepare \
  --workspace-root "${ROOT}"

echo "[phase4f61] running or resuming F21 value-scale 0.5 versus 1.0"
python3 -u -m quoridor.phase4f61_value_scale_screen run \
  --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f61] log: ${LOG_PATH}"
