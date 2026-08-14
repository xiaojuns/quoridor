#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4f2] verifying native rules and recovery opening partition"
"${ROOT_DIR}/scripts/build-native-engine.sh"
python3 -u scripts/generate-phase4f-recovery-openings.py --check

echo "[phase4f2] preparing CPU-only recovery contracts"
exec python3 -u -m quoridor.phase4f_recovery --workspace-root "${ROOT_DIR}"
