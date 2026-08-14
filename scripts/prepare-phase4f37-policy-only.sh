#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f37] building and testing the native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
echo "[phase4f37] reviewing F36 and freezing the policy-only pilot"
python3 -u -m quoridor.phase4f37_policy_only prepare --workspace-root "${ROOT_DIR}"
