#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f35] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
echo "[phase4f35] building the guidance strata index and freezing the pilot"
python3 -u -m quoridor.phase4f35_stage_a2 prepare --workspace-root "${ROOT_DIR}"
