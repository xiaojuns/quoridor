#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"

if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]]; then
  echo "[phase4-calibration] could not read free disk space" >&2
  exit 1
fi
if ((AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "[phase4-calibration] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

echo "[phase4-calibration] preflight disk check passed"
"${ROOT_DIR}/scripts/build-native-engine.sh"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
exec python3 -m quoridor.calibration "$@"
