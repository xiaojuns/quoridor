#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
DEVICE_INDEX=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f22-screen.sh [--device-index INDEX]

Runs or resumes the 64-game paired final Stage A candidate versus
classic-hard screen. Screen trajectories never enter training replay.
EOF
}

while (($# > 0)); do
  case "$1" in
    --device-index)
      (($# >= 2)) || { echo "--device-index requires a value" >&2; exit 2; }
      DEVICE_INDEX="$2"
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

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"
cd "${ROOT_DIR}"

echo "[phase4f22] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f22] verifying frozen screen contract"
python3 -u -m quoridor.phase4f22_screen verify \
  --workspace-root "${ROOT_DIR}"

echo "[phase4f22] running or resuming 64-game paired screen"
exec python3 -u -m quoridor.phase4f22_screen run \
  --workspace-root "${ROOT_DIR}" \
  --device cuda:0
