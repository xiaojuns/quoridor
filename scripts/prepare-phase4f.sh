#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTRACT="training/configs/phase4f-final-frozen-v1.json"
VERIFY_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/prepare-phase4f.sh [--verify-only] [--contract PATH]

Builds and tests the native engine, deeply verifies the formal champion, and
freezes the Phase 4F classic-hard final-evaluation contract. Preparation is
CPU-only: it calibrates the fixed classic node budget on eight dev positions,
does not run locked-final games, and does not authorize product release.
EOF
}

while (($# > 0)); do
  case "$1" in
    --verify-only) VERIFY_ONLY=1; shift ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${CONTRACT}" != /* ]]; then
  CONTRACT="${ROOT_DIR}/${CONTRACT}"
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "[phase4f0] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f0] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

if ((VERIFY_ONLY)); then
  echo "[phase4f0] verifying frozen CPU-only preparation"
  python3 -u -m quoridor.phase4f_final verify \
    --workspace-root "${ROOT_DIR}" \
    --contract "${CONTRACT}"
else
  echo "[phase4f0] calibrating classic node budget on dev openings and freezing contract"
  python3 -u -m quoridor.phase4f_final prepare \
    --workspace-root "${ROOT_DIR}" \
    --contract "${CONTRACT}"
fi

