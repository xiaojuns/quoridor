#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f18] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f18] freezing CPU-only Stage A and pilot preparation"
exec python3 -u -m quoridor.phase4f18_rebaseline prepare \
  --workspace-root "${ROOT_DIR}"
