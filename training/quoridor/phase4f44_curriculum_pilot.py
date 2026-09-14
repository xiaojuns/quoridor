"""Run the frozen Phase 4F44 corrected early-curriculum CUDA pilot."""

from __future__ import annotations

import argparse
import io
import math
import random
import shutil
import subprocess
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .codec import (
    mirror_action_index,
    mirror_state,
    state_from_serializable,
    to_canonical_action_index,
)
from .native_mcts import load_native_module
from .network import ACTION_SIZE, PHASE4_INPUT_CHANNELS, NetworkConfig, PolicyValueNetwork
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f43_curriculum_pilot import (
    CONTRACT,
    EXPECTED_DATASET_FINGERPRINT,
    EXPECTED_F21_MODEL_STATE_SHA256,
    GATE as F43_GATE,
    _implementation_sha256 as _adapter_implementation_sha256,
    _load_f42_evidence,
    _load_initialization,
    load_curriculum_samples,
    tensorize_samples,
    verify_phase4f43,
)
from .run_manifest import (
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size
from .trainer import AlphaZeroTrainer


SCHEMA_VERSION = 1
EXPECTED_CONTRACT_SHA256 = "05c5a2e63da5fe5a42cfd708d35cbfbf14c9f42916f750717d21e0d3d187913c"
EXPECTED_F43_GATE_SHA256 = "af926a051248cf7870cba96babe4d3842ac2818f02182d4bbbaff175954851e9"
ROOT = Path("training/runs/phase4/phase4f/f44-early-curriculum-pilot-v1")
CHECKPOINT = ROOT / "checkpoints/best-model.pt"
REPORT = ROOT / "pilot-report.json"
GATE = ROOT / "pilot-gate.json"


class Phase4F44Error(RuntimeError):
    """Raised when the frozen F44 pilot cannot execute or verify safely."""


@dataclass(frozen=True)
class EncodedCurriculum:
    identity_states: torch.Tensor
    identity_policies: torch.Tensor
    mirrored_states: torch.Tensor
    mirrored_policies: torch.Tensor
    values: torch.Tensor
    bands: tuple[str, ...]
    ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.bands)
        if size == 0 or len(self.ordinals) != size:
            raise ValueError("encoded curriculum metadata is incomplete")
        expected = {
            "identity states": (size, 16, 9, 9),
            "identity policies": (size, 209),
            "mirrored states": (size, 16, 9, 9),
            "mirrored policies": (size, 209),
            "values": (size,),
        }
        tensors = {
            "identity states": self.identity_states,
            "identity policies": self.identity_policies,
            "mirrored states": self.mirrored_states,
            "mirrored policies": self.mirrored_policies,
            "values": self.values,
        }
        for label, tensor in tensors.items():
            if tuple(tensor.shape) != expected[label] or tensor.dtype != torch.float32:
                raise ValueError(f"{label} has the wrong shape or dtype")


class StratifiedBatchSampler:
    """Deterministic, independently reshuffled streams for each ply band."""

    COUNTS = {"early": 308, "middle": 102, "late": 102}

    def __init__(self, bands: Sequence[str], *, seed: int) -> None:
        self.seed = int(seed)
        self.groups = {
            band: [index for index, value in enumerate(bands) if value == band]
            for band in self.COUNTS
        }
        if any(
            len(self.groups[band]) < count
            for band, count in self.COUNTS.items()
        ):
            raise ValueError("each curriculum band must fill one batch allocation")
        self.cycles = Counter()
        self.cursors = Counter()
        self.permutations = {band: self._permutation(band, 0) for band in self.COUNTS}

    def _permutation(self, band: str, cycle: int) -> list[int]:
        digest = sha256(f"{self.seed}:{band}:{cycle}".encode("ascii")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        values = list(self.groups[band])
        rng.shuffle(values)
        return values

    def _take(self, band: str, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            permutation = self.permutations[band]
            cursor = self.cursors[band]
            available = min(count - len(selected), len(permutation) - cursor)
            selected.extend(permutation[cursor : cursor + available])
            self.cursors[band] += available
            if self.cursors[band] == len(permutation):
                self.cycles[band] += 1
                self.permutations[band] = self._permutation(band, self.cycles[band])
                self.cursors[band] = 0
        return selected

    def batch(self, step: int) -> tuple[list[int], list[bool]]:
        if type(step) is not int or step < 0:
            raise ValueError("training step must be a non-negative integer")
        indices: list[int] = []
        for band, count in self.COUNTS.items():
            indices.extend(self._take(band, count))
        order_seed = int.from_bytes(
            sha256(f"{self.seed}:batch:{step}".encode("ascii")).digest()[:8], "big"
        )
        order_rng = random.Random(order_seed)
        order_rng.shuffle(indices)
        mirror_rng = random.Random(
            int.from_bytes(
                sha256(f"{self.seed}:mirror:{step}".encode("ascii")).digest()[:8],
                "big",
            )
        )
        mirrored = [mirror_rng.random() < 0.5 for _ in indices]
        return indices, mirrored


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F44Error("workspace does not contain training/quoridor")
    return workspace


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f44-curriculum-pilot-progress",
                "schemaVersion": SCHEMA_VERSION,
                "stage": stage,
                "status": status,
                **fields,
            }
        ).decode("ascii"),
        end="",
        flush=True,
    )


def _runner_sha256(workspace: Path) -> str:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f44_curriculum_pilot.py",
            "training/quoridor/phase4f43_curriculum_pilot.py",
            "training/quoridor/network.py",
            "training/quoridor/trainer.py",
        ),
    ).sha256


def _contract(workspace: Path) -> dict[str, object]:
    verified43 = verify_phase4f43(workspace)
    if verified43["gateSha256"] != EXPECTED_F43_GATE_SHA256:
        raise Phase4F44Error("F43 gate is not the frozen F44 authorization")
    contract = _mapping(workspace / CONTRACT, "F44 contract")
    if _verify_self_hash(contract, "contractSha256", "F44 contract") != EXPECTED_CONTRACT_SHA256:
        raise Phase4F44Error("F44 contract hash differs")
    if _adapter_implementation_sha256(workspace) != contract.get("implementationSha256"):
        raise Phase4F44Error("F44 tensor adapter source changed after contract freeze")
    if (
        contract.get("status") != "frozen"
        or contract.get("inputs", {}).get("datasetFingerprint") != EXPECTED_DATASET_FINGERPRINT
        or contract.get("authorizations", {}).get("cudaPilotTrainingAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
        or contract.get("training", {}).get("selfPlayGames") != 0
        or contract.get("training", {}).get("replayStoreUsage") is not False
    ):
        raise Phase4F44Error("F44 contract does not authorize this pilot")
    return contract


def _network_config(contract: Mapping[str, object]) -> NetworkConfig:
    values = contract["network"]
    return NetworkConfig(
        input_channels=int(values["inputChannels"]),
        channels=int(values["channels"]),
        blocks=int(values["residualBlocks"]),
        se_reduction=int(values["seReduction"]),
        policy_channels=int(values["policyChannels"]),
        value_channels=int(values["valueChannels"]),
        value_hidden=int(values["valueHidden"]),
    )


def _load_model(workspace: Path, contract: Mapping[str, object], device: torch.device) -> PolicyValueNetwork:
    initialization = _load_initialization(workspace)
    if initialization != contract.get("initialization"):
        raise Phase4F44Error("F44 initialization differs from the frozen contract")
    try:
        payload = torch.load(workspace / str(initialization["path"]), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F44Error("F44 initialization is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise Phase4F44Error("F44 initialization lacks modelState")
    model = PolicyValueNetwork(_network_config(contract))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4F44Error("F44 initialization does not fit the frozen network") from error
    if canonical_model_state_sha256(model) != EXPECTED_F21_MODEL_STATE_SHA256:
        raise Phase4F44Error("F44 initialized model state differs")
    return model.to(device)


def freeze_early_trunk(model: PolicyValueNetwork) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.trunk[8:]:
        block.requires_grad_(True)
    model.policy_head.requires_grad_(True)
    model.value_features.requires_grad_(True)
    model.value_head.requires_grad_(True)
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if not trainable or any(
        name.startswith(("input_conv.", "input_bn."))
        or (name.startswith("trunk.") and int(name.split(".")[1]) < 8)
        for name in trainable
    ):
        raise Phase4F44Error("F44 trainable parameter boundary is incorrect")
    return trainable


def keep_frozen_batch_norm_eval(model: PolicyValueNetwork) -> None:
    model.input_bn.eval()
    for block in model.trunk[:8]:
        block.eval()


def _encode_curriculum(
    samples: Sequence[Mapping[str, object]],
    *,
    split: str,
    native_encoder: object | None = None,
) -> EncodedCurriculum:
    identity_chunks: list[dict[str, torch.Tensor]] = []
    mirror_chunks: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(samples), 64):
        rows = samples[start : start + 64]
        if native_encoder is None:
            identity_chunks.append(tensorize_samples(rows))
            mirror_chunks.append(tensorize_samples(rows, mirrored=[True] * len(rows)))
        else:
            identity_chunks.append(_native_tensorize(rows, native_encoder, use_mirror=False))
            mirror_chunks.append(_native_tensorize(rows, native_encoder, use_mirror=True))
        if (start + len(rows)) % 512 == 0 or start + len(rows) == len(samples):
            _progress(
                "tensor-cache",
                "progress",
                split=split,
                positions=start + len(rows),
                positionsTotal=len(samples),
            )
    identity = {key: torch.cat([chunk[key] for chunk in identity_chunks]) for key in identity_chunks[0]}
    mirrored = {key: torch.cat([chunk[key] for chunk in mirror_chunks]) for key in mirror_chunks[0]}
    if not torch.equal(identity["values"], mirrored["values"]):
        raise Phase4F44Error("F44 tensor cache changed mirrored value targets")
    return EncodedCurriculum(
        identity_states=identity["states"],
        identity_policies=identity["policies"],
        mirrored_states=mirrored["states"],
        mirrored_policies=mirrored["policies"],
        values=identity["values"],
        bands=tuple(str(sample["band"]) for sample in samples),
        ordinals=tuple(int(sample["ordinal"]) for sample in samples),
    )


def _native_tensorize(
    samples: Sequence[Mapping[str, object]],
    native_encoder: object,
    *,
    use_mirror: bool,
) -> dict[str, torch.Tensor]:
    if not callable(native_encoder):
        raise TypeError("native encoder must be callable")
    states = [state_from_serializable(sample["state"]) for sample in samples]
    transformed = [mirror_state(state) for state in states] if use_mirror else states
    encoded = np.asarray(native_encoder(transformed, input_channels=PHASE4_INPUT_CHANNELS))
    expected = (len(samples), PHASE4_INPUT_CHANNELS, 9, 9)
    if encoded.shape != expected or encoded.dtype != np.float32:
        raise Phase4F44Error("native encoder returned the wrong F44 tensor contract")
    policies = torch.zeros((len(samples), ACTION_SIZE), dtype=torch.float32)
    for row, (sample, state) in enumerate(zip(samples, transformed, strict=True)):
        for action, probability in sample["policy"]:
            raw_action = mirror_action_index(int(action)) if use_mirror else int(action)
            canonical = to_canonical_action_index(raw_action, state.current_player)
            policies[row, canonical] += float(probability)
    mass = policies.sum(dim=1)
    if not bool(torch.allclose(mass, torch.ones_like(mass), rtol=0.0, atol=1e-6)):
        raise Phase4F44Error("native F44 adapter changed policy target mass")
    return {
        "states": torch.from_numpy(np.ascontiguousarray(encoded)),
        "policies": policies / mass.unsqueeze(1),
        "values": torch.tensor([float(sample["value"]) for sample in samples], dtype=torch.float32),
    }


def _verify_native_adapter(
    samples: Sequence[Mapping[str, object]], native_encoder: object
) -> dict[str, object]:
    rows = samples[:8]
    maximum_state_error = 0.0
    maximum_policy_error = 0.0
    maximum_value_error = 0.0
    for use_mirror in (False, True):
        reference = tensorize_samples(rows, mirrored=[use_mirror] * len(rows))
        native = _native_tensorize(rows, native_encoder, use_mirror=use_mirror)
        maximum_state_error = max(
            maximum_state_error,
            float((reference["states"] - native["states"]).abs().max().item()),
        )
        maximum_policy_error = max(
            maximum_policy_error,
            float((reference["policies"] - native["policies"]).abs().max().item()),
        )
        maximum_value_error = max(
            maximum_value_error,
            float((reference["values"] - native["values"]).abs().max().item()),
        )
    if maximum_state_error != 0.0 or maximum_policy_error > 1e-6 or maximum_value_error != 0.0:
        raise Phase4F44Error("native F44 adapter differs from the frozen Python adapter")
    return {
        "positions": len(rows),
        "augmentations": len(rows) * 2,
        "stateMaximumAbsoluteError": maximum_state_error,
        "policyMaximumAbsoluteError": maximum_policy_error,
        "valueMaximumAbsoluteError": maximum_value_error,
    }


def _training_batch(
    cache: EncodedCurriculum,
    indices: Sequence[int],
    mirrored: Sequence[bool],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(indices) != len(mirrored) or not indices:
        raise ValueError("F44 training selection is malformed")
    index = torch.tensor(indices, dtype=torch.long)
    use_mirror = torch.tensor(mirrored, dtype=torch.bool)
    states = cache.identity_states[index].clone()
    policies = cache.identity_policies[index].clone()
    states[use_mirror] = cache.mirrored_states[index[use_mirror]]
    policies[use_mirror] = cache.mirrored_policies[index[use_mirror]]
    return {
        "states": states.to(device, non_blocking=True),
        "policies": policies.to(device, non_blocking=True),
        "values": cache.values[index].to(device, non_blocking=True),
    }


@torch.inference_mode()
def evaluate_model(
    model: PolicyValueNetwork,
    cache: EncodedCurriculum,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> dict[str, object]:
    model.eval()
    totals = {band: {"policy": 0.0, "value": 0.0, "positions": 0} for band in ("early", "middle", "late")}
    totals["all"] = {"policy": 0.0, "value": 0.0, "positions": 0}
    for mirror in (False, True):
        source_states = cache.mirrored_states if mirror else cache.identity_states
        source_policies = cache.mirrored_policies if mirror else cache.identity_policies
        for start in range(0, len(cache.bands), batch_size):
            end = min(start + batch_size, len(cache.bands))
            states = source_states[start:end].to(device, non_blocking=True)
            targets = source_policies[start:end].to(device, non_blocking=True)
            values = cache.values[start:end].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, predictions = model(states)
                policy_losses = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
                value_losses = F.mse_loss(predictions, values, reduction="none")
            policy_cpu = policy_losses.float().cpu()
            value_cpu = value_losses.float().cpu()
            for offset, band in enumerate(cache.bands[start:end]):
                for key in (band, "all"):
                    totals[key]["policy"] += float(policy_cpu[offset].item())
                    totals[key]["value"] += float(value_cpu[offset].item())
                    totals[key]["positions"] += 1
    result: dict[str, object] = {}
    for band, values in totals.items():
        positions = int(values["positions"])
        policy = float(values["policy"]) / positions
        value = float(values["value"]) / positions
        result[band] = {
            "positions": positions // 2,
            "augmentations": positions,
            "policyLoss": policy,
            "valueLoss": value,
            "totalLoss": policy + value,
        }
    return result


def acceptance_checks(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    acceptance: Mapping[str, object],
) -> dict[str, bool]:
    def improvement(band: str, metric: str) -> float:
        before = float(baseline[band][metric])
        return (before - float(candidate[band][metric])) / before

    def regression(band: str, metric: str) -> float:
        return -improvement(band, metric)

    finite = all(
        math.isfinite(float(row[metric]))
        for payload in (baseline, candidate)
        for row in payload.values()
        for metric in ("policyLoss", "valueLoss", "totalLoss")
    )
    return {
        "finiteTrainingAndEvaluation": finite,
        "earlyTotalLoss": improvement("early", "totalLoss")
        >= float(acceptance["earlyTotalLossRelativeImprovementMinimum"]),
        "overallTotalLoss": improvement("all", "totalLoss")
        >= float(acceptance["overallTotalLossRelativeImprovementMinimum"]),
        "earlyPolicyLoss": regression("early", "policyLoss")
        <= float(acceptance["earlyPolicyLossRelativeRegressionMaximum"]),
        "earlyValueLoss": regression("early", "valueLoss")
        <= float(acceptance["earlyValueLossRelativeRegressionMaximum"]),
        "middleTotalLoss": regression("middle", "totalLoss")
        <= float(acceptance["middleTotalLossRelativeRegressionMaximum"]),
        "lateTotalLoss": regression("late", "totalLoss")
        <= float(acceptance["lateTotalLossRelativeRegressionMaximum"]),
    }


def _cpu_state(model: PolicyValueNetwork) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().contiguous().clone())
        for name, tensor in sorted(model.state_dict().items())
    )


def _write_checkpoint(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    *,
    step: int,
    contract_sha256: str,
    runner_sha256: str,
) -> dict[str, object]:
    model_state_sha256 = canonical_model_state_sha256(state)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "format": "quoridor-phase4f44-curriculum-pilot-model-only",
        "step": step,
        "contractSha256": contract_sha256,
        "runnerSha256": runner_sha256,
        "modelStateSha256": model_state_sha256,
        "modelState": OrderedDict((name, tensor.clone()) for name, tensor in state.items()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path = workspace / CHECKPOINT
    encoded = buffer.getvalue()
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4F44Error("existing immutable F44 checkpoint differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_create_bytes(path, encoded)
    return {
        "path": CHECKPOINT.as_posix(),
        "fileSha256": sha256_file(path),
        "modelStateSha256": model_state_sha256,
        "step": step,
        "bytes": path.stat().st_size,
    }


def _cuda_preflight(device: torch.device) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F44Error("F44 pilot requires CUDA")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise Phase4F44Error("requested CUDA device does not exist")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5.0,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise Phase4F44Error("nvidia-smi preflight failed") from error
    if not output:
        raise Phase4F44Error("nvidia-smi returned no GPU")
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "totalMemoryBytes": int(properties.total_memory),
        "nvidiaSmi": output[device.index or 0],
        "torchCudaVersion": torch.version.cuda,
    }


def _resource_preflight(workspace: Path, contract: Mapping[str, object]) -> dict[str, int]:
    limits = contract["resources"]
    artifacts = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free = shutil.disk_usage(workspace).free
    reserve = int(limits["maximumOutputBytes"])
    if artifacts + reserve >= int(limits["artifactLimitBytes"]):
        raise Phase4F44Error("F44 output reserve exceeds the frozen artifact cap")
    if free - reserve < int(limits["minimumFreeDiskBytes"]):
        raise Phase4F44Error("F44 output reserve violates minimum free disk")
    return {
        "artifactBytes": artifacts,
        "artifactLimitBytes": int(limits["artifactLimitBytes"]),
        "freeDiskBytes": free,
        "reservedWriteBytes": reserve,
    }


def run_phase4f44(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f44(workspace), "reused": True}
    contract = _contract(workspace)
    device = torch.device(device_name)
    cuda = _cuda_preflight(device)
    resources = _resource_preflight(workspace, contract)
    runner_sha256 = _runner_sha256(workspace)
    _gate42, manifest = _load_f42_evidence(workspace)
    _progress("dataset", "loading")
    train_samples = load_curriculum_samples(workspace, manifest, "train")
    development_samples = load_curriculum_samples(workspace, manifest, "development")
    native_module = load_native_module()
    native_audit = _verify_native_adapter(development_samples, native_module.encode_states)
    native_backend_sha256 = loaded_native_backend_hash()
    _progress(
        "native-adapter",
        "verified",
        backendSha256=native_backend_sha256,
        positions=native_audit["positions"],
    )
    train_cache = _encode_curriculum(
        train_samples,
        split="train",
        native_encoder=native_module.encode_states,
    )
    development_cache = _encode_curriculum(
        development_samples,
        split="development",
        native_encoder=native_module.encode_states,
    )

    settings = contract["training"]
    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = _load_model(workspace, contract, device)
    trainable = freeze_early_trunk(model)
    frozen_bn_hook = model.register_forward_pre_hook(lambda _model, _inputs: keep_frozen_batch_norm_eval(model))
    trainer = AlphaZeroTrainer(
        model,
        learning_rate=float(settings["learningRate"]),
        weight_decay=float(settings["weightDecay"]),
        max_grad_norm=float(settings["maximumGradientNorm"]),
        use_amp=bool(settings["cudaAmp"]),
        schedule_steps=int(settings["optimizerSteps"]),
        minimum_learning_rate=float(settings["minimumLearningRate"]),
    )
    sampler = StratifiedBatchSampler(train_cache.bands, seed=seed)
    evaluation_steps = set(int(value) for value in settings["evaluationSteps"])
    acceptance = contract["evaluation"]["acceptance"]
    evaluations: list[dict[str, object]] = []
    baseline: dict[str, object] | None = None
    best_eligible: tuple[float, int, OrderedDict[str, torch.Tensor]] | None = None
    best_observed: tuple[float, int, OrderedDict[str, torch.Tensor]] | None = None
    timing: list[float] = []

    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline_global_hours = ledger.total_hours
    run_id = f"phase4f44-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    status = "interrupted"
    try:
        with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
            ledger.begin(run_id, external_hours=0.0)
            for step in range(int(settings["optimizerSteps"]) + 1):
                if step in evaluation_steps:
                    metrics = evaluate_model(model, development_cache, device)
                    if baseline is None:
                        baseline = metrics
                        checks = {"baseline": True}
                        eligible = False
                    else:
                        checks = acceptance_checks(baseline, metrics, acceptance)
                        eligible = all(checks.values())
                    evaluations.append({"step": step, "metrics": metrics, "checks": checks, "eligible": eligible})
                    score = float(metrics["all"]["totalLoss"])
                    state = _cpu_state(model)
                    if best_observed is None or score < best_observed[0]:
                        best_observed = (score, step, state)
                    if eligible and (best_eligible is None or score < best_eligible[0]):
                        best_eligible = (score, step, state)
                    total = ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status="running")
                    _progress(
                        "evaluation",
                        "completed",
                        step=step,
                        eligible=eligible,
                        totalLoss=score,
                        earlyTotalLoss=metrics["early"]["totalLoss"],
                        globalGpuHours=total,
                    )
                if step == int(settings["optimizerSteps"]):
                    break
                batch_started = time.monotonic()
                indices, mirrored = sampler.batch(step)
                batch = _training_batch(train_cache, indices, mirrored, device)
                metrics = trainer.train_step(batch)
                timing.append(time.monotonic() - batch_started)
                if time.monotonic() - started >= float(contract["resources"]["gpuHoursLimit"]) * 3600:
                    raise Phase4F44Error("F44 pilot GPU budget was exhausted")
                if (step + 1) % 4 == 0:
                    total = ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status="running")
                    _progress(
                        "training",
                        "progress",
                        step=step + 1,
                        optimizerSteps=int(settings["optimizerSteps"]),
                        loss=metrics.loss,
                        policyLoss=metrics.policy_loss,
                        valueLoss=metrics.value_loss,
                        globalGpuHours=total,
                    )

            if baseline is None or best_observed is None:
                raise Phase4F44Error("F44 evaluation schedule did not execute")
            selected = best_eligible if best_eligible is not None else best_observed
            checkpoint = _write_checkpoint(
                workspace,
                selected[2],
                step=selected[1],
                contract_sha256=str(contract["contractSha256"]),
                runner_sha256=runner_sha256,
            )
            pilot_passed = best_eligible is not None
            outcome = "corrected-curriculum-pilot-passed" if pilot_passed else "corrected-curriculum-pilot-failed"
            elapsed = time.monotonic() - started
            total = ledger.update(run_id, elapsed_seconds=elapsed, status="completed")
            status = "completed"
            report = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f44-corrected-early-curriculum-pilot",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": pilot_passed,
                    "contractPath": CONTRACT.as_posix(),
                    "contractSha256": contract["contractSha256"],
                    "runnerSha256": runner_sha256,
                    "nativeBackendSha256": native_backend_sha256,
                    "nativeAdapterAudit": native_audit,
                    "datasetFingerprint": EXPECTED_DATASET_FINGERPRINT,
                    "initialModelStateSha256": EXPECTED_F21_MODEL_STATE_SHA256,
                    "selectedCheckpoint": checkpoint,
                    "selectedEligible": best_eligible is not None,
                    "evaluations": evaluations,
                    "training": {
                        "optimizerSteps": trainer.step,
                        "trainableParameterNames": list(trainable),
                        "meanStepSeconds": sum(timing) / len(timing),
                        "maximumStepSeconds": max(timing),
                    },
                    "cuda": {
                        **cuda,
                        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
                    },
                    "resources": resources,
                    "elapsedSeconds": elapsed,
                    "baselineGlobalGpuHours": baseline_global_hours,
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "reportSha256",
            )
            _write_immutable(workspace / REPORT, report, "F44 pilot report")
            gate = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f44-curriculum-pilot-gate",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": pilot_passed,
                    "reportPath": REPORT.as_posix(),
                    "reportSha256": report["reportSha256"],
                    "selectedCheckpoint": checkpoint,
                    "optimizerSteps": trainer.step,
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "nextStage": {
                        "name": "Phase 4F45 corrected early-curriculum pilot review",
                        "requiresSeparateUserConfirmation": True,
                    },
                },
                "gateSha256",
            )
            _write_immutable(workspace / GATE, gate, "F44 pilot gate")
            return {**gate, "reused": False}
    except BaseException:
        try:
            ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status=status)
        except Exception:
            pass
        raise
    finally:
        frozen_bn_hook.remove()


def verify_phase4f44(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    report = _mapping(workspace / REPORT, "F44 pilot report")
    report_sha = _verify_self_hash(report, "reportSha256", "F44 pilot report")
    gate = _mapping(workspace / GATE, "F44 pilot gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F44 pilot gate")
    checkpoint = gate.get("selectedCheckpoint")
    if not isinstance(checkpoint, Mapping):
        raise Phase4F44Error("F44 selected checkpoint metadata is malformed")
    path = workspace / str(checkpoint["path"])
    if sha256_file(path) != checkpoint.get("fileSha256"):
        raise Phase4F44Error("F44 selected checkpoint bytes changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F44Error("F44 selected checkpoint is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or canonical_model_state_sha256(state) != checkpoint.get("modelStateSha256"):
        raise Phase4F44Error("F44 selected checkpoint model state changed")
    if (
        report.get("contractSha256") != contract["contractSha256"]
        or report.get("datasetFingerprint") != EXPECTED_DATASET_FINGERPRINT
        or report.get("selectedCheckpoint") != checkpoint
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or gate.get("reportSha256") != report_sha
        or gate.get("optimizerSteps") != 32
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F44Error("F44 pilot artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f44-curriculum-pilot-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "selectedCheckpoint": checkpoint,
        "pilotPassed": gate["pilotPassed"],
        "outcomeBand": gate["outcomeBand"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_phase4f44(args.workspace_root, args.device) if args.command == "run" else verify_phase4f44(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
