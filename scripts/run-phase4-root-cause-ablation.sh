#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/build-native-engine.sh"

export PYTHONPATH="${ROOT}/native/build/python:${ROOT}/training${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -u -m quoridor.root_cause_ablation \
  --workspace-root "${ROOT}" \
  "$@"
