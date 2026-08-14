#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"

usage() {
  cat <<'EOF'
Usage: ./scripts/freeze-phase4f-recovery.sh [--verify-only]

CPU-only Phase 4F4 challenge replay freeze and recovery rebaseline
preparation. This command never starts CUDA training.
EOF
}

VERIFY_ARGS=()
while (($# > 0)); do
  case "$1" in
    --verify-only) VERIFY_ARGS+=(--verify-only); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f4] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f4] verifying native rules"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f4] freezing challenge replay and preparing CUDA rebaseline"
exec python3 -u -m quoridor.phase4f_recovery_freeze \
  --workspace-root "${ROOT_DIR}" \
  "${VERIFY_ARGS[@]}"
