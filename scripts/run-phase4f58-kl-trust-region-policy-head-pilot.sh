#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT}/native/build}"
LOG_DIR="${ROOT}/training/runs/phase4/logs"
LOG_PATH="${LOG_DIR}/phase4f58-kl-trust-region-policy-head-pilot-$(date +%Y%m%d-%H%M%S).log"
DEVICE_INDEX=0

while (($# > 0)); do
  case "$1" in
    --device-index) DEVICE_INDEX="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--device-index INDEX]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${DEVICE_INDEX}" =~ ^[0-9]+$ ]] || { echo "invalid device index" >&2; exit 2; }
mkdir -p "${LOG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_INDEX}"

python3 -u -m quoridor.phase4f57_policy_head_pilot_review verify \
  --workspace-root "${ROOT}"
python3 -m pytest -q "${ROOT}/training/tests/test_phase4f58_kl_trust_region_pilot.py"
python3 -u -m quoridor.phase4f58_kl_trust_region_pilot run \
  --workspace-root "${ROOT}" --device cuda:0 2>&1 | tee "${LOG_PATH}"
echo "[phase4f58] log: ${LOG_PATH}"
