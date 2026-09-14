#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"

"${ROOT}/scripts/build-native-engine.sh"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
python3 -u -m quoridor.phase4f39_early_curriculum verify --workspace-root "${ROOT}"
exec python3 -u -m quoridor.phase4f40_target_probe run \
  --workspace-root "${ROOT}" \
  --device cuda
