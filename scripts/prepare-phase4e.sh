#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
EXECUTION_CONTRACT="training/configs/phase4e-execution-prepared-v1.json"
BOOTSTRAP_CONFIG="training/configs/phase4e-bootstrap-v1.json"
STANDARD_CONFIG="training/configs/phase4e-standard-v1.json"
PILOT_ROOT="training/runs/phase4/phase4d-pilot-v2"
BASELINE_ROOT="training/runs/phase4/baselines/bootstrap-warm-v2"
REPLAY_SOURCE="training/runs/phase4/phase4d-pilot-v2/lanes/warm/replay"
REPLAY_TARGET="training/runs/phase4/phase4e0/replay-migration-v1"
TARGET_MAX_GAMES=100000
TARGET_MAX_PLIES=100000
TARGET_MAX_BYTES=6442450944
TRANSFER_MODE="copy"
VERIFY_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/prepare-phase4e.sh [options]

Options:
  --verify-only              Verify existing baseline and replay migration
  --execution-contract PATH Prepared, non-authorizing Phase 4E contract
  --bootstrap-config PATH    Versioned Bootstrap config
  --standard-config PATH     Versioned Standard calibration candidate
  --pilot-root PATH          Completed Phase 4D pilot-v2 root
  --baseline-root PATH       Model-only bootstrap baseline root
  --replay-source PATH       Verified warm replay source
  --replay-target PATH       Audit-only migrated replay root
  --transfer-mode MODE       copy, hardlink, or auto (default: copy)
  -h, --help                 Show this help

This launcher is CPU-only. It builds/tests the native engine, verifies the
current-source replay-input gate, extracts or verifies the warm generation-4
model-only baseline, and audits the 320-game replay migration candidate. It
never freezes a GPU budget and never authorizes Phase 4E training.
EOF
}

while (($# > 0)); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --execution-contract)
      EXECUTION_CONTRACT="$2"
      shift 2
      ;;
    --bootstrap-config)
      BOOTSTRAP_CONFIG="$2"
      shift 2
      ;;
    --standard-config)
      STANDARD_CONFIG="$2"
      shift 2
      ;;
    --pilot-root)
      PILOT_ROOT="$2"
      shift 2
      ;;
    --baseline-root)
      BASELINE_ROOT="$2"
      shift 2
      ;;
    --replay-source)
      REPLAY_SOURCE="$2"
      shift 2
      ;;
    --replay-target)
      REPLAY_TARGET="$2"
      shift 2
      ;;
    --transfer-mode)
      TRANSFER_MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${TRANSFER_MODE}" in
  copy|hardlink|auto) ;;
  *)
    echo "--transfer-mode must be copy, hardlink, or auto" >&2
    exit 2
    ;;
esac

absolute_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "$1"
  fi
}

EXECUTION_CONTRACT="$(absolute_path "${EXECUTION_CONTRACT}")"
BOOTSTRAP_CONFIG="$(absolute_path "${BOOTSTRAP_CONFIG}")"
STANDARD_CONFIG="$(absolute_path "${STANDARD_CONFIG}")"
PILOT_ROOT="$(absolute_path "${PILOT_ROOT}")"
BASELINE_ROOT="$(absolute_path "${BASELINE_ROOT}")"
REPLAY_SOURCE="$(absolute_path "${REPLAY_SOURCE}")"
REPLAY_TARGET="$(absolute_path "${REPLAY_TARGET}")"

for required_file in \
  "${EXECUTION_CONTRACT}" \
  "${BOOTSTRAP_CONFIG}" \
  "${STANDARD_CONFIG}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required Phase 4E input does not exist: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${PILOT_ROOT}" || ! -d "${REPLAY_SOURCE}" ]]; then
  echo "The completed pilot-v2 root and warm replay source are required" >&2
  exit 2
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "At least 20 GiB must remain free before Phase 4E preparation" >&2
  exit 1
fi

ARTIFACT_LIMIT_BYTES=$((8 * 1024 * 1024 * 1024))
PREPARATION_RESERVE_BYTES=$((512 * 1024 * 1024))
CURRENT_BYTES=0
if [[ -d "${CONTROL_ROOT}" ]]; then
  CURRENT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
fi
if [[ ! "${CURRENT_BYTES}" =~ ^[0-9]+$ ]] || ((CURRENT_BYTES > ARTIFACT_LIMIT_BYTES)); then
  echo "Phase 4 artifacts already exceed the 8 GiB limit" >&2
  exit 1
fi
if ((VERIFY_ONLY == 0 && CURRENT_BYTES + PREPARATION_RESERVE_BYTES > ARTIFACT_LIMIT_BYTES)); then
  echo "Phase 4E preparation cannot reserve 512 MiB below the 8 GiB limit" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

echo "[phase4e-prepare] validating prepared contract and versioned configs"
python3 - "${EXECUTION_CONTRACT}" "${BOOTSTRAP_CONFIG}" "${STANDARD_CONFIG}" <<'PY'
import json
import sys

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4_training import validate_phase4_config

contract = load_phase4_execution_contract(sys.argv[1])
if (
    contract.stage != "phase4e"
    or contract.status != "prepared"
    or contract.budget_frozen
    or not contract.formal_promotion
):
    raise SystemExit(
        "Phase 4E0 requires a prepared, non-authorizing Phase 4E contract"
    )
for path in sys.argv[2:]:
    validate_phase4_config(load_experiment_config(path), contract)
print(
    json.dumps(
        {
            "executionContractSha256": contract.sha256,
            "formalPromotion": contract.formal_promotion,
            "gpuAuthorized": False,
            "maximumGenerations": contract.maximum_generations,
            "stage": contract.stage,
            "status": contract.status,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY

echo "[phase4e-prepare] building and testing the CPU native engine"
"${ROOT_DIR}/scripts/build-native-engine.sh"
echo "[phase4e-prepare] verifying current-source replay input gate"
python3 -m quoridor.phase4_training --replay-input-gate

if ((VERIFY_ONLY)); then
  echo "[phase4e-prepare] deeply verifying bootstrap baseline"
  python3 -m quoridor.phase4_baseline verify \
    --workspace-root "${ROOT_DIR}" \
    --baseline-root "${BASELINE_ROOT}"
  echo "[phase4e-prepare] deeply verifying replay migration; this CPU pass may take several minutes"
  python3 -m quoridor.phase4_replay_migration verify \
    --workspace-root "${ROOT_DIR}" \
    --baseline-root "${BASELINE_ROOT}" \
    --source-root "${REPLAY_SOURCE}" \
    --target-root "${REPLAY_TARGET}" \
    --target-max-games "${TARGET_MAX_GAMES}" \
    --target-max-plies "${TARGET_MAX_PLIES}" \
    --target-max-bytes "${TARGET_MAX_BYTES}"
else
  echo "[phase4e-prepare] preparing bootstrap baseline"
  python3 -m quoridor.phase4_baseline prepare \
    --workspace-root "${ROOT_DIR}" \
    --pilot-root "${PILOT_ROOT}" \
    --output-root "${BASELINE_ROOT}"
  echo "[phase4e-prepare] preparing replay migration; repeated CPU deep verification may take several minutes"
  python3 -m quoridor.phase4_replay_migration prepare \
    --workspace-root "${ROOT_DIR}" \
    --baseline-root "${BASELINE_ROOT}" \
    --source-root "${REPLAY_SOURCE}" \
    --target-root "${REPLAY_TARGET}" \
    --target-max-games "${TARGET_MAX_GAMES}" \
    --target-max-plies "${TARGET_MAX_PLIES}" \
    --target-max-bytes "${TARGET_MAX_BYTES}" \
    --transfer-mode "${TRANSFER_MODE}"
fi

FINAL_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
FINAL_AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if ((FINAL_BYTES > ARTIFACT_LIMIT_BYTES)); then
  echo "Phase 4 artifacts exceed the 8 GiB limit after verification" >&2
  exit 1
fi
if ((FINAL_AVAILABLE_KIB < MINIMUM_FREE_KIB)); then
  echo "Less than 20 GiB remains after Phase 4E verification" >&2
  exit 1
fi

printf \
  '{"artifactBytes":%s,"gpuAuthorized":false,"minimumFreeDiskBytes":%s,"mode":"%s","status":"completed"}\n' \
  "${FINAL_BYTES}" \
  "$((20 * 1024 * 1024 * 1024))" \
  "$([[ "${VERIFY_ONLY}" == "1" ]] && printf verify || printf prepare)"
