#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"

if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]]; then
  echo "[phase4d-memory-smoke] could not read free disk space" >&2
  exit 1
fi
if ((AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "[phase4d-memory-smoke] less than the 20 GiB free-disk stop line is available" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
exec python3 -m quoridor.training_memory_smoke "$@"
