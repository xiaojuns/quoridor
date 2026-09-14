#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

./scripts/build-native-engine.sh
export PYTHONPATH="$ROOT/training:$ROOT/native/build/python${PYTHONPATH:+:$PYTHONPATH}"

python3 -u -m quoridor.phase4f49_teacher_search_audit run --workspace-root "$ROOT" --device cuda:0
