#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/native/build"
DEVICE_INDEX=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run-phase4f32-screen.sh [--device-index INDEX]

Runs or resumes the 64-game paired screen between the F31 generation-5
candidate (800-simulation native MCTS) and classic-hard on newly isolated
fresh openings. Arena trajectories never enter training replay.
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

echo "[phase4f32] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f32] verifying frozen F31 fresh-opening screen contract"
python3 -u -m quoridor.phase4f32_fresh_screen verify \
  --workspace-root "${ROOT_DIR}"

echo "[phase4f32] running or resuming 64-game fresh-opening screen"
exec python3 -u -m quoridor.phase4f32_fresh_screen run \
  --workspace-root "${ROOT_DIR}" \
  --device cuda:0
