#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f60-kl-lineage-screen-$(date +%Y%m%d-%H%M%S).log"
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

echo "[phase4f60] building and testing the native engine"
"${ROOT}/scripts/build-native-engine.sh"

echo "[phase4f60] verifying frozen F59/F60 evidence"
python3 -u -m quoridor.phase4f59_kl_trust_region_review verify \
  --workspace-root "${ROOT}"
python3 -m pytest -q \
  "${ROOT}/training/tests/test_phase4f60_kl_lineage_screen.py"

echo "[phase4f60] running or resuming the 64-game F58-versus-F21 screen"
python3 -u -m quoridor.phase4f60_kl_lineage_screen run \
  --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f60] log: ${LOG_PATH}"
