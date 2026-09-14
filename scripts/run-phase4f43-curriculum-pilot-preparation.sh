#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"

export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
python3 -u -m quoridor.phase4f42_curriculum_dataset verify --workspace-root "${ROOT}"
exec python3 -u -m quoridor.phase4f43_curriculum_pilot prepare --workspace-root "${ROOT}"
