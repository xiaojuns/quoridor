"""Prepare the corrected early-curriculum CUDA pilot without training it.

Phase 4F42 intentionally stores soft policy and value targets outside the
legacy replay format.  This module supplies the tensor adapter for that
format, proves that the targets survive canonicalization and mirroring, and
freezes the separately-confirmed F44 CUDA pilot contract.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .codec import (
    mirror_action_index,
    mirror_state,
    state_from_serializable,
    to_canonical_action_index,
)
from .network import ACTION_SIZE, PHASE4_INPUT_CHANNELS, NetworkConfig, PolicyValueNetwork, encode_state
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f39_early_curriculum import F21_CONFIG, F21_RUN
from .phase4f42_curriculum_dataset import (
    GATE as F42_GATE,
    MANIFEST as F42_MANIFEST,
    _validate_sample,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size
from .trainer import AlphaZeroTrainer


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f43-curriculum-pilot-preparation-v1")
REPORT = ROOT / "preparation-report.json"
GATE = ROOT / "preparation-gate.json"
CONTRACT = Path("training/configs/phase4f44-early-curriculum-pilot-frozen-v1.json")
RUN_ID = "phase4f43-curriculum-pilot-preparation-v1"

EXPECTED_F42_GATE_SHA256 = "94ad13fcb818bf4be2643ee2d77ac385f4258abfc7f4305dc8c98da763c378b3"
EXPECTED_F42_MANIFEST_SHA256 = "54e3c3a6c7df9a912d8ac6414815fc182a50876f2368cf073a6435fd338f3a45"
EXPECTED_DATASET_FINGERPRINT = "e0789c4b93dbd5367d3c3eff2aa6225c91db884a061b7f707d010a66157b5354"
EXPECTED_F21_CANDIDATE_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
EXPECTED_F21_MODEL_STATE_SHA256 = "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
F21_CANDIDATE = F21_RUN / "checkpoints/candidate-model-checkpoint-g00000001-s000000000192.pt"

WRITE_RESERVE_BYTES = 64 * 1024**2
SMOKE_POSITIONS = 64


class Phase4F43Error(RuntimeError):
    """Raised when the corrected curriculum pilot cannot be frozen safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f43-curriculum-pilot-preparation-progress",
                "schemaVersion": SCHEMA_VERSION,
                "stage": stage,
                "status": status,
                **fields,
            }
        ).decode("ascii"),
        end="",
        flush=True,
    )


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F43Error("workspace does not contain training/quoridor")
    return workspace


def _implementation_sha256(workspace: Path) -> str:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f43_curriculum_pilot.py",
            "training/quoridor/phase4f42_curriculum_dataset.py",
            "training/quoridor/codec.py",
            "training/quoridor/network.py",
            "training/quoridor/trainer.py",
        ),
    ).sha256


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F43Error("16 GiB artifact cap cannot reserve F43 evidence")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F43Error("insufficient free disk for F43 evidence")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "reservedWriteBytes": WRITE_RESERVE_BYTES,
    }


def _load_f42_evidence(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    gate = _mapping(workspace / F42_GATE, "F42 dataset gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F42 dataset gate")
    manifest = _mapping(workspace / F42_MANIFEST, "F42 dataset manifest")
    manifest_sha = _verify_self_hash(manifest, "manifestSha256", "F42 dataset manifest")
    if (
        gate_sha != EXPECTED_F42_GATE_SHA256
        or manifest_sha != EXPECTED_F42_MANIFEST_SHA256
        or gate.get("manifestSha256") != manifest_sha
        or gate.get("datasetFingerprint") != EXPECTED_DATASET_FINGERPRINT
        or manifest.get("datasetFingerprint") != EXPECTED_DATASET_FINGERPRINT
        or gate.get("pilotPreparationAuthorized") is not True
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
        or manifest.get("trainingStarted") is not False
        or manifest.get("optimizerSteps") != 0
    ):
        raise Phase4F43Error("F42 evidence does not authorize F43 preparation")
    return gate, manifest


def _load_initialization(workspace: Path) -> dict[str, object]:
    candidate = workspace / F21_CANDIDATE
    if sha256_file(candidate) != EXPECTED_F21_CANDIDATE_SHA256:
        raise Phase4F43Error("F21 initialization candidate bytes changed")
    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F43Error("F21 initialization candidate is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise Phase4F43Error("F21 initialization candidate lacks modelState")
    if canonical_model_state_sha256(state) != EXPECTED_F21_MODEL_STATE_SHA256:
        raise Phase4F43Error("F21 initialization model state changed")
    return {
        "mode": "candidate-model",
        "path": F21_CANDIDATE.as_posix(),
        "fileSha256": EXPECTED_F21_CANDIDATE_SHA256,
        "modelStateSha256": EXPECTED_F21_MODEL_STATE_SHA256,
        "networkConfigPath": F21_CONFIG.as_posix(),
        "networkConfigSha256": sha256_file(workspace / F21_CONFIG),
    }


def load_curriculum_samples(
    workspace: Path,
    manifest: Mapping[str, object],
    split: str,
) -> list[dict[str, object]]:
    if split not in {"train", "development"}:
        raise ValueError("curriculum split must be train or development")
    contract_sha256 = str(manifest["contractSha256"])
    references = manifest.get("chunks")
    if not isinstance(references, list):
        raise Phase4F43Error("F42 manifest chunk inventory is malformed")
    samples: list[dict[str, object]] = []
    for reference in references:
        if not isinstance(reference, Mapping) or reference.get("split") != split:
            continue
        chunk = _mapping(workspace / str(reference["path"]), "F42 dataset chunk")
        if _verify_self_hash(chunk, "chunkSha256", "F42 dataset chunk") != reference.get("sha256"):
            raise Phase4F43Error("F42 dataset chunk reference changed")
        rows = chunk.get("samples")
        if not isinstance(rows, list) or len(rows) != reference.get("positions"):
            raise Phase4F43Error("F42 dataset chunk position count changed")
        for sample in rows:
            if not isinstance(sample, Mapping):
                raise Phase4F43Error("F42 dataset sample is malformed")
            _validate_sample(sample, contract_sha256)
            samples.append(dict(sample))
    expected = 4096 if split == "train" else 512
    if len(samples) != expected:
        raise Phase4F43Error(f"F42 {split} split must contain {expected} positions")
    return samples


def tensorize_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    mirrored: Sequence[bool] | None = None,
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("curriculum batch cannot be empty")
    transforms = tuple(False for _ in samples) if mirrored is None else tuple(mirrored)
    if len(transforms) != len(samples) or any(type(value) is not bool for value in transforms):
        raise ValueError("mirror decisions must contain one bool per sample")
    states: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    values: list[float] = []
    for sample, use_mirror in zip(samples, transforms, strict=True):
        state = state_from_serializable(sample["state"])
        transformed_state = mirror_state(state) if use_mirror else state
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        for action, probability in sample["policy"]:
            raw_action = mirror_action_index(int(action)) if use_mirror else int(action)
            canonical_action = to_canonical_action_index(
                raw_action, transformed_state.current_player
            )
            policy[canonical_action] += float(probability)
        total = policy.sum()
        if not bool(torch.isfinite(total).item()) or not math.isclose(
            float(total.item()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise Phase4F43Error("tensor adapter changed policy target mass")
        states.append(encode_state(transformed_state, input_channels=PHASE4_INPUT_CHANNELS))
        policies.append(policy / total)
        values.append(float(sample["value"]))
    return {
        "states": torch.stack(states),
        "policies": torch.stack(policies),
        "values": torch.tensor(values, dtype=torch.float32),
    }


def _restore_mirrored_policy(policy: torch.Tensor) -> torch.Tensor:
    restored = torch.zeros_like(policy)
    for canonical_action in range(ACTION_SIZE):
        restored[mirror_action_index(canonical_action)] = policy[canonical_action]
    return restored


def _representative_samples(samples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int], list[dict[str, object]]] = {}
    for sample in samples:
        key = (str(sample["band"]), int(sample["state"]["currentPlayer"]))
        buckets.setdefault(key, []).append(dict(sample))
    chosen: list[dict[str, object]] = []
    cursor = Counter()
    keys = [(band, side) for band in ("early", "middle", "late") for side in (0, 1)]
    while len(chosen) < min(SMOKE_POSITIONS, len(samples)):
        made_progress = False
        for key in keys:
            rows = buckets.get(key, [])
            index = cursor[key]
            if index < len(rows):
                chosen.append(rows[index])
                cursor[key] += 1
                made_progress = True
                if len(chosen) == min(SMOKE_POSITIONS, len(samples)):
                    break
        if not made_progress:
            break
    return chosen


def run_adapter_smoke(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = _representative_samples(samples)
    identity = tensorize_samples(selected)
    mirrored = tensorize_samples(selected, mirrored=[True] * len(selected))
    policy_errors = [
        float((left - _restore_mirrored_policy(right)).abs().max().item())
        for left, right in zip(identity["policies"], mirrored["policies"], strict=True)
    ]
    value_error = float((identity["values"] - mirrored["values"]).abs().max().item())
    if max(policy_errors) > 1e-6 or value_error != 0.0:
        raise Phase4F43Error("curriculum tensor adapter breaks mirror target consistency")
    soft_policy_rows = sum(int(torch.count_nonzero(row).item()) > 1 for row in identity["policies"])
    soft_value_rows = int((identity["values"].abs() < 0.999).sum().item())
    if soft_policy_rows == 0 or soft_value_rows == 0:
        raise Phase4F43Error("curriculum tensor adapter did not preserve soft targets")

    smoke_batch = {key: value[:8].clone() for key, value in identity.items()}
    config = NetworkConfig(
        input_channels=PHASE4_INPUT_CHANNELS,
        channels=32,
        blocks=2,
        se_reduction=8,
        policy_channels=8,
        value_channels=8,
        value_hidden=64,
    )
    torch.manual_seed(20260825)
    model = PolicyValueNetwork(config)
    before = model.policy_head[-1].weight.detach().clone()
    trainer = AlphaZeroTrainer(
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_grad_norm=5.0,
        use_amp=False,
        schedule_steps=1,
        minimum_learning_rate=1e-5,
    )
    metrics = trainer.train_step(smoke_batch)
    parameter_delta = float((model.policy_head[-1].weight.detach() - before).abs().max().item())
    if parameter_delta <= 0 or not all(
        math.isfinite(value)
        for value in (metrics.loss, metrics.policy_loss, metrics.value_loss, metrics.grad_norm)
    ):
        raise Phase4F43Error("CPU optimizer smoke did not complete a finite update")
    return {
        "positions": len(selected),
        "shape": list(identity["states"].shape),
        "dtype": str(identity["states"].dtype),
        "policyMaximumAbsoluteMirrorError": max(policy_errors),
        "valueMaximumAbsoluteMirrorError": value_error,
        "softPolicyRows": soft_policy_rows,
        "softValueRows": soft_value_rows,
        "policyMassMaximumAbsoluteError": float(
            (identity["policies"].sum(dim=1) - 1.0).abs().max().item()
        ),
        "optimizerStep": {
            "loss": metrics.loss,
            "policyLoss": metrics.policy_loss,
            "valueLoss": metrics.value_loss,
            "gradientNorm": metrics.grad_norm,
            "parameterMaximumAbsoluteDelta": parameter_delta,
        },
    }


def _frozen_contract(
    implementation_sha256: str,
    manifest: Mapping[str, object],
    initialization: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f44-early-curriculum-pilot-contract",
            "status": "frozen",
            "objective": "test whether corrected soft early-curriculum targets improve F21 without self-play confounding",
            "implementationSha256": implementation_sha256,
            "inputs": {
                "datasetManifestPath": F42_MANIFEST.as_posix(),
                "datasetManifestSha256": EXPECTED_F42_MANIFEST_SHA256,
                "datasetFingerprint": EXPECTED_DATASET_FINGERPRINT,
                "trainPositions": 4096,
                "developmentPositions": 512,
                "format": manifest["format"],
            },
            "initialization": dict(initialization),
            "network": {
                "inputChannels": 16,
                "channels": 192,
                "residualBlocks": 16,
                "seReduction": 8,
                "policyChannels": 32,
                "valueChannels": 32,
                "valueHidden": 256,
            },
            "training": {
                "device": "cuda",
                "selfPlayGames": 0,
                "replayStoreUsage": False,
                "optimizer": "AdamW",
                "batchSize": 512,
                "optimizerSteps": 32,
                "evaluationSteps": [0, 8, 16, 24, 32],
                "learningRate": 2e-5,
                "minimumLearningRate": 5e-6,
                "schedule": "cosine-annealing",
                "weightDecay": 1e-4,
                "maximumGradientNorm": 5.0,
                "cudaAmp": True,
                "seed": 20260825,
                "batchMix": {"early": 0.60, "middle": 0.20, "late": 0.20},
                "shuffle": "sha256-seeded-epoch-permutation-v1",
                "mirrorProbability": 0.5,
                "mirrorRng": "one-seeded-draw-per-selected-position-v1",
                "loss": {"policy": "soft-cross-entropy", "value": "mean-squared-error", "weights": [1.0, 1.0]},
                "frozenParameters": ["input_conv", "input_bn", "trunk.0-7"],
                "trainableParameters": ["trunk.8-15", "policy_head", "value_features", "value_head"],
                "checkpointSelection": "minimum-development-total-loss-among-eligible-evaluations",
            },
            "evaluation": {
                "split": "development",
                "usesMirrorAverage": True,
                "compareAgainstUnmodifiedInitialization": True,
                "finalSplitUsage": False,
                "acceptance": {
                    "finiteTrainingAndEvaluation": True,
                    "earlyTotalLossRelativeImprovementMinimum": 0.01,
                    "overallTotalLossRelativeImprovementMinimum": 0.005,
                    "earlyPolicyLossRelativeRegressionMaximum": 0.01,
                    "earlyValueLossRelativeRegressionMaximum": 0.01,
                    "middleTotalLossRelativeRegressionMaximum": 0.02,
                    "lateTotalLossRelativeRegressionMaximum": 0.02,
                    "policyMassMaximumAbsoluteError": 1e-6,
                    "mirrorTargetMaximumAbsoluteError": 1e-6,
                },
            },
            "resources": {
                "gpuHoursLimit": 1.0,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
                "maximumOutputBytes": 512 * 1024**2,
            },
            "authorizations": {
                "cudaPilotTrainingAuthorized": True,
                "formalTrainingAuthorized": False,
                "selfPlayAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F44 corrected early-curriculum CUDA pilot",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "contractSha256",
    )


def prepare_phase4f43(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f43(workspace), "reused": True}
    resources = _resources(workspace)
    gate42, manifest = _load_f42_evidence(workspace)
    initialization = _load_initialization(workspace)
    implementation_sha256 = _implementation_sha256(workspace)
    _progress("preflight", "verified", datasetFingerprint=EXPECTED_DATASET_FINGERPRINT)
    train_samples = load_curriculum_samples(workspace, manifest, "train")
    development_samples = load_curriculum_samples(workspace, manifest, "development")
    _progress("adapter-smoke", "started", positions=SMOKE_POSITIONS)
    smoke = run_adapter_smoke(development_samples)
    contract = _frozen_contract(implementation_sha256, manifest, initialization)
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=RUN_ID):
        _write_immutable(workspace / CONTRACT, contract, "F44 pilot contract")
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f43-curriculum-pilot-preparation",
                "status": "completed",
                "implementationSha256": implementation_sha256,
                "f42GateSha256": gate42["gateSha256"],
                "datasetManifestSha256": manifest["manifestSha256"],
                "datasetFingerprint": EXPECTED_DATASET_FINGERPRINT,
                "positions": {"train": len(train_samples), "development": len(development_samples)},
                "initialization": initialization,
                "adapterSmoke": smoke,
                "executionContractPath": CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "resources": resources,
                "gpuHoursUsed": 0.0,
                "trainingStarted": False,
                "optimizerSteps": 0,
            },
            "reportSha256",
        )
        _write_immutable(workspace / REPORT, report, "F43 preparation report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f43-curriculum-pilot-preparation-gate",
                "status": "completed",
                "outcomeBand": "corrected-curriculum-cuda-pilot-ready",
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "executionContractPath": CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "datasetFingerprint": EXPECTED_DATASET_FINGERPRINT,
                "adapterSmokePassed": True,
                "cudaPilotTrainingAuthorized": True,
                "formalTrainingAuthorized": False,
                "trainingStarted": False,
                "nextStage": contract["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F43 preparation gate")
    _progress("preparation", "completed", contractSha256=contract["contractSha256"])
    return {**gate, "reused": False}


def verify_phase4f43(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    _load_f42_evidence(workspace)
    initialization = _load_initialization(workspace)
    contract = _mapping(workspace / CONTRACT, "F44 pilot contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F44 pilot contract")
    report = _mapping(workspace / REPORT, "F43 preparation report")
    report_sha = _verify_self_hash(report, "reportSha256", "F43 preparation report")
    gate = _mapping(workspace / GATE, "F43 preparation gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F43 preparation gate")
    if (
        contract.get("status") != "frozen"
        or contract.get("inputs", {}).get("datasetFingerprint") != EXPECTED_DATASET_FINGERPRINT
        or contract.get("initialization") != initialization
        or contract.get("authorizations", {}).get("cudaPilotTrainingAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
        or report.get("executionContractSha256") != contract_sha
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or gate.get("reportSha256") != report_sha
        or gate.get("executionContractSha256") != contract_sha
        or gate.get("cudaPilotTrainingAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F43Error("F43 preparation artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f43-curriculum-pilot-preparation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "executionContractPath": CONTRACT.as_posix(),
        "executionContractSha256": contract_sha,
        "datasetFingerprint": EXPECTED_DATASET_FINGERPRINT,
        "cudaPilotTrainingAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = prepare_phase4f43(args.workspace_root) if args.command == "prepare" else verify_phase4f43(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
