#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTRACT="training/configs/phase4f-final-frozen-v1.json"
DEVICE="cuda"
TRACK=""

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f-final.sh --track fixed|wall-clock [options]

Options:
  --contract PATH  Frozen Phase 4F0 contract
  --device DEVICE  CUDA device name (default: cuda)

Runs or resumes one 400-game locked-final classic-hard evaluation track.
Complete opening pairs are committed atomically. Ctrl+C stops safely; rerun the
same command to resume. This command never sets best/product champion pointers.
EOF
}

while (($# > 0)); do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${TRACK}" != "fixed" && "${TRACK}" != "wall-clock" ]]; then
  echo "--track must be fixed or wall-clock" >&2
  exit 2
fi
if [[ "${CONTRACT}" != /* ]]; then
  CONTRACT="${ROOT_DIR}/${CONTRACT}"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
echo "[phase4f] verifying frozen preparation before CUDA access"
python3 -u -m quoridor.phase4f_final verify \
  --workspace-root "${ROOT_DIR}" \
  --contract "${CONTRACT}"
echo "[phase4f] running ${TRACK}; Ctrl+C stops safely and the same command resumes"
python3 -u -m quoridor.phase4f_final run \
  --workspace-root "${ROOT_DIR}" \
  --contract "${CONTRACT}" \
  --track "${TRACK}" \
  --device "${DEVICE}"

