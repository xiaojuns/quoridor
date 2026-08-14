#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${QUORIDOR_NATIVE_BUILD_DIR:-${ROOT_DIR}/native/build}"
CONTROL_ROOT="${ROOT_DIR}/training/runs/phase4"
EXECUTION_CONTRACT="training/configs/phase4e-execution-prepared-v1.json"
CONFIG="training/configs/phase4e-bootstrap-v1.json"
BASELINE_ROOT="training/runs/phase4/baselines/bootstrap-warm-v2"
DEVICE="cuda"
GENERATIONS=1
TARGET_GENERATIONS=""
GENERATIONS_EXPLICIT=0
GPU_HOURS_USED=0
RESUME_MODE="auto"
NEW_EXPLICIT=0
RESUME_EXPLICIT=0

usage() {
  cat <<'EOF'
Usage: ./scripts/train-phase4e.sh [options]

Options:
  --execution-contract PATH Frozen Phase 4E execution/budget contract
  --config PATH              Versioned Phase 4E config
  --bootstrap-baseline PATH  Verified warm-g4 model-only baseline
  --device DEVICE            auto, cpu, or cuda (default: cuda)
  --generations COUNT        Additional generations to commit (default: 1)
  --target-generations N     Idempotent absolute committed-generation target
  --gpu-hours-used HOURS     Monotonic GPU hours outside the shared ledger
  --new                      Refuse an existing run manifest
  --resume                   Require an existing run manifest
  -h, --help                 Show this help

The checked-in prepared contract has a pending GPU budget and intentionally
fails before native build or CUDA access. Training is available only after a
separately reviewed, explicitly frozen execution contract is supplied.

Ctrl+C preserves complete replay games and the last valid checkpoint. Re-run
the exact same command to resume; prefer --target-generations for automation.
EOF
}

while (($# > 0)); do
  case "$1" in
    --execution-contract)
      EXECUTION_CONTRACT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --bootstrap-baseline)
      BASELINE_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --generations)
      GENERATIONS="$2"
      GENERATIONS_EXPLICIT=1
      shift 2
      ;;
    --target-generations)
      TARGET_GENERATIONS="$2"
      shift 2
      ;;
    --gpu-hours-used)
      GPU_HOURS_USED="$2"
      shift 2
      ;;
    --new)
      RESUME_MODE="new"
      NEW_EXPLICIT=1
      shift
      ;;
    --resume)
      RESUME_MODE="resume"
      RESUME_EXPLICIT=1
      shift
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

if ((NEW_EXPLICIT && RESUME_EXPLICIT)); then
  echo "--new and --resume are mutually exclusive" >&2
  exit 2
fi
if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be auto, cpu, or cuda" >&2
  exit 2
fi
if [[ -n "${TARGET_GENERATIONS}" && "${GENERATIONS_EXPLICIT}" == "1" ]]; then
  echo "--generations and --target-generations are mutually exclusive" >&2
  exit 2
fi
if [[ -z "${TARGET_GENERATIONS}" && ! "${GENERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--generations must be a positive integer" >&2
  exit 2
fi

absolute_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "$1"
  fi
}

EXECUTION_CONTRACT="$(absolute_path "${EXECUTION_CONTRACT}")"
CONFIG="$(absolute_path "${CONFIG}")"
BASELINE_ROOT="$(absolute_path "${BASELINE_ROOT}")"
if [[ ! -f "${EXECUTION_CONTRACT}" || ! -f "${CONFIG}" ]]; then
  echo "The execution contract and Phase 4E config must exist" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/training${PYTHONPATH:+:${PYTHONPATH}}"

set +e
CONTRACT_INFO="$(
  python3 - "${EXECUTION_CONTRACT}" <<'PY'
import sys

from quoridor.phase4_execution import (
    Phase4ExecutionContractError,
    load_phase4_execution_contract,
)

try:
    contract = load_phase4_execution_contract(sys.argv[1])
    if contract.stage != "phase4e" or not contract.formal_promotion:
        raise Phase4ExecutionContractError(
            "contract is not the formal Phase 4E execution stage"
        )
    contract.require_frozen_budget()
except (OSError, ValueError) as error:
    print(
        "[phase4e-training] execution contract does not authorize training: "
        f"{error}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(contract.maximum_generations, contract.sha256)
PY
)"
CONTRACT_STATUS=$?
set -e
if ((CONTRACT_STATUS != 0)); then
  exit "${CONTRACT_STATUS}"
fi
read -r MAXIMUM_GENERATIONS CONTRACT_SHA256 <<<"${CONTRACT_INFO}"

if [[ -n "${TARGET_GENERATIONS}" ]]; then
  if [[ ! "${TARGET_GENERATIONS}" =~ ^[0-9]+$ ]]; then
    echo "--target-generations must be in [0, ${MAXIMUM_GENERATIONS}]" >&2
    exit 2
  fi
  TARGET_VALUE=$((10#${TARGET_GENERATIONS}))
  if ((TARGET_VALUE > MAXIMUM_GENERATIONS)); then
    echo "--target-generations must be in [0, ${MAXIMUM_GENERATIONS}]" >&2
    exit 2
  fi
  GENERATION_ARGS=(--target-generations "${TARGET_GENERATIONS}")
else
  GENERATION_ARGS=(--generations "${GENERATIONS}")
fi

if [[ ! -d "${BASELINE_ROOT}" ]]; then
  echo "Bootstrap baseline does not exist: ${BASELINE_ROOT}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
echo "[phase4e-training] deeply verifying bootstrap baseline before build/CUDA"
python3 -m quoridor.phase4_baseline verify \
  --workspace-root "${ROOT_DIR}" \
  --baseline-root "${BASELINE_ROOT}"
RUN_DIR="$(
  python3 - "${CONFIG}" "${EXECUTION_CONTRACT}" <<'PY'
import sys

from quoridor.config import load_experiment_config
from quoridor.phase4_execution import load_phase4_execution_contract
from quoridor.phase4_training import validate_phase4_config

config = load_experiment_config(sys.argv[1])
contract = load_phase4_execution_contract(sys.argv[2])
validate_phase4_config(config, contract)
print(config.run_dir)
PY
)"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT_DIR}/${RUN_DIR}"
fi

AVAILABLE_KIB="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || ((AVAILABLE_KIB < 20 * 1024 * 1024)); then
  echo "At least 20 GiB must remain free before Phase 4E training" >&2
  exit 1
fi
ARTIFACT_BYTES=0
if [[ -d "${CONTROL_ROOT}" ]]; then
  ARTIFACT_BYTES="$(du -sb "${CONTROL_ROOT}" | awk '{print $1}')"
fi
if [[ ! "${ARTIFACT_BYTES}" =~ ^[0-9]+$ ]] \
  || ((ARTIFACT_BYTES >= 8 * 1024 * 1024 * 1024)); then
  echo "The Phase 4 artifact root has reached the 8 GiB stop line" >&2
  exit 1
fi

MANIFEST="${RUN_DIR}/run-manifest.json"
RESUME_ARGS=()
if [[ "${RESUME_MODE}" == "resume" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "--resume requested but run manifest is missing: ${MANIFEST}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume)
elif [[ "${RESUME_MODE}" == "new" ]]; then
  if [[ -e "${MANIFEST}" ]]; then
    echo "--new requested but run manifest already exists: ${MANIFEST}" >&2
    exit 1
  fi
elif [[ -f "${MANIFEST}" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${ROOT_DIR}/scripts/build-native-engine.sh"
export PYTHONPATH="${ROOT_DIR}/training:${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${CONTROL_ROOT}/logs"
LOG_PATH="${CONTROL_ROOT}/logs/phase4e-training-$(date +%Y%m%d-%H%M%S).log"
COMMAND=(
  python3 -m quoridor.phase4_training
  --config "${CONFIG}"
  --execution-contract "${EXECUTION_CONTRACT}"
  --bootstrap-baseline "${BASELINE_ROOT}"
  --device "${DEVICE}"
  "${GENERATION_ARGS[@]}"
  --gpu-hours-used "${GPU_HOURS_USED}"
  "${RESUME_ARGS[@]}"
)

echo "[phase4e-training] contract: ${CONTRACT_SHA256}"
echo "[phase4e-training] mode: ${RESUME_MODE}"
echo "[phase4e-training] command: ${COMMAND[*]}"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e
echo "[phase4e-training] log: ${LOG_PATH}"
exit "${STATUS}"
