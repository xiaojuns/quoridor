#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f64-scaled-classic-confirmation-$(date +%Y%m%d-%H%M%S).log"
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

echo "[phase4f64] building and testing the native engine"
"${ROOT}/scripts/build-native-engine.sh"

echo "[phase4f64] running CPU-only focused tests"
python3 -m pytest -q \
  "${ROOT}/training/tests/test_phase4_arena.py" \
  "${ROOT}/training/tests/test_phase4f61_value_scale_screen.py" \
  "${ROOT}/training/tests/test_phase4f63_value_calibration_review.py" \
  "${ROOT}/training/tests/test_phase4f64_scaled_classic_confirmation.py"

echo "[phase4f64] preparing or verifying the frozen 200-pair confirmation"
python3 -u -m quoridor.phase4f64_scaled_classic_confirmation prepare \
  --workspace-root "${ROOT}"

echo "[phase4f64] running or resuming scaled F21 versus classic-hard"
python3 -u -m quoridor.phase4f64_scaled_classic_confirmation run \
  --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"

echo "[phase4f64] log: ${LOG_PATH}"
