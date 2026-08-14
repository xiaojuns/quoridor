#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "[phase4f17] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

exec python3 -u -m quoridor.phase4f17_staged --workspace-root "${ROOT_DIR}"
