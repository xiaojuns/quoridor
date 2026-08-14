#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/native/build"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f34] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"

echo "[phase4f34] freezing CPU-only F33 failure review and Stage A2 protocols"
exec python3 -u -m quoridor.phase4f34_failure_review prepare \
  --workspace-root "${ROOT_DIR}"
