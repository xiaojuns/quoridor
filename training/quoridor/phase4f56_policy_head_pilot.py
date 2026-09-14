"""Run the frozen F56 isolated policy-head CUDA pilot."""

from __future__ import annotations

import argparse
import io
import math
import random
import shutil
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .codec import (
    mirror_action_index,
    mirror_state,
    state_from_serializable,
    to_canonical_action_index,
)
from .game import legal_action_indices
from .network import ACTION_SIZE, PHASE4_INPUT_CHANNELS, PolicyValueNetwork, encode_state
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f55_policy_head_pilot_preparation import (
    ANCHOR_COUNTS,
    ANCHOR_SET,
    EXPECTED_F21_STATE_SHA256,
    EXPECTED_F54_DATASET_FINGERPRINT,
    F54_MANIFEST,
    F56_CONTRACT,
    _load_initial_model,
    _masked_log_softmax,
    freeze_policy_head,
    load_policy_samples,
    tensorize_policy_samples,
    verify_phase4f55,
)
from .run_manifest import (
    RunLock,
    atomic_create_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_CONTRACT_SHA256 = "02f59197e7a671ceef69b72d0b5ad95d06d1f9b48642573f8899f8b41d0d6f51"
ROOT = Path("training/runs/phase4/phase4f/f56-isolated-policy-head-pilot-v1")
CHECKPOINT = ROOT / "selected-policy-head-model.pt"
REPORT = ROOT / "pilot-report.json"
GATE = ROOT / "pilot-gate.json"
RUN_ID_PREFIX = "phase4f56-isolated-policy-head-pilot"


class Phase4F56Error(RuntimeError):
    """Raised when the frozen F56 pilot cannot run or verify safely."""


@dataclass(frozen=True)
class TensorCache:
    identity_states: torch.Tensor
    identity_policies: torch.Tensor | None
    identity_masks: torch.Tensor
    mirrored_states: torch.Tensor
    mirrored_policies: torch.Tensor | None
    mirrored_masks: torch.Tensor
    bands: tuple[str, ...]

    @property
    def positions(self) -> int:
        return int(self.identity_states.shape[0])


class StratifiedSampler:
    def __init__(
        self,
        bands: Sequence[str],
        *,
        batch_size: int,
        mix: Mapping[str, float],
        seed: int,
    ) -> None:
        self._rng = random.Random(seed)
        self._indices = {
            band: [index for index, value in enumerate(bands) if value == band]
            for band in ("early", "middle", "late")
        }
        self._counts = {
            "early": round(batch_size * float(mix["early"])),
            "middle": round(batch_size * float(mix["middle"])),
        }
        self._counts["late"] = batch_size - sum(self._counts.values())
        if any(not values for values in self._indices.values()) or any(
            count <= 0 for count in self._counts.values()
        ):
            raise Phase4F56Error("stratified sampler has an empty band")
        self._queues: dict[str, list[int]] = {band: [] for band in self._indices}

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def batch(self) -> tuple[list[int], list[bool]]:
        selected: list[int] = []
        for band in ("early", "middle", "late"):
            required = self._counts[band]
            while len(self._queues[band]) < required:
                refill = list(self._indices[band])
                self._rng.shuffle(refill)
                self._queues[band].extend(refill)
            selected.extend(self._queues[band][:required])
            del self._queues[band][:required]
        order = list(range(len(selected)))
        self._rng.shuffle(order)
        selected = [selected[index] for index in order]
        mirrored = [self._rng.random() < 0.5 for _ in selected]
        return selected, mirrored


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f56-policy-head-pilot-progress",
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
        raise Phase4F56Error("workspace does not contain training/quoridor")
    return workspace


def _runner_source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f56_policy_head_pilot.py",
            "training/quoridor/phase4f55_policy_head_pilot_preparation.py",
            "training/quoridor/codec.py",
            "training/quoridor/network.py",
        ),
    ).to_payload()


def _contract(workspace: Path) -> dict[str, object]:
    payload = _mapping(workspace / F56_CONTRACT, "F56 contract")
    if (
        _verify_self_hash(payload, "contractSha256", "F56 contract")
        != EXPECTED_CONTRACT_SHA256
        or payload.get("status") != "frozen"
        or payload.get("authorizations", {}).get("cudaPilotTrainingAuthorized")
        is not True
        or payload.get("authorizations", {}).get("formalTrainingAuthorized")
        is not False
    ):
        raise Phase4F56Error("F56 contract is not the accepted frozen contract")
    return payload


def _cuda_preflight(device: torch.device) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F56Error("F56 requires CUDA")
    index = device.index or 0
    if index >= torch.cuda.device_count():
        raise Phase4F56Error("requested CUDA device does not exist")
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5.0,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise Phase4F56Error("nvidia-smi preflight failed") from error
    if len(rows) <= index:
        raise Phase4F56Error("nvidia-smi did not report the requested GPU")
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "totalMemoryBytes": int(properties.total_memory),
        "nvidiaSmi": rows[index],
        "torchCudaVersion": torch.version.cuda,
    }


def _resource_preflight(workspace: Path, contract: Mapping[str, object]) -> dict[str, int]:
    limits = contract["resources"]
    artifacts = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    output = directory_size(workspace / ROOT) if (workspace / ROOT).exists() else 0
    reserve = int(limits["maximumOutputBytes"]) - output
    free = shutil.disk_usage(workspace).free
    if reserve < 0:
        raise Phase4F56Error("F56 output already exceeds its frozen limit")
    if artifacts + reserve >= int(limits["artifactLimitBytes"]):
        raise Phase4F56Error("F56 write reserve exceeds the frozen artifact cap")
    if free - reserve < int(limits["minimumFreeDiskBytes"]):
        raise Phase4F56Error("F56 reserve violates the free-disk stop line")
    return {
        "artifactBytes": artifacts,
        "artifactLimitBytes": int(limits["artifactLimitBytes"]),
        "f56OutputBytes": output,
        "remainingWriteReserveBytes": reserve,
        "freeDiskBytes": free,
        "minimumFreeDiskBytes": int(limits["minimumFreeDiskBytes"]),
    }


def _policy_cache(samples: Sequence[Mapping[str, object]]) -> TensorCache:
    identity = tensorize_policy_samples(samples)
    mirrored = tensorize_policy_samples(samples, mirrored=True)
    return TensorCache(
        identity_states=identity["states"],
        identity_policies=identity["policies"],
        identity_masks=identity["legalMasks"],
        mirrored_states=mirrored["states"],
        mirrored_policies=mirrored["policies"],
        mirrored_masks=mirrored["legalMasks"],
        bands=tuple(str(sample["band"]) for sample in samples),
    )


def tensorize_anchor_samples(
    samples: Sequence[Mapping[str, object]], *, mirrored: bool = False
) -> dict[str, torch.Tensor]:
    states: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for sample in samples:
        original = state_from_serializable(sample["state"])
        state = mirror_state(original) if mirrored else original
        mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        for action in legal_action_indices(original):
            raw_action = mirror_action_index(action) if mirrored else action
            mask[to_canonical_action_index(raw_action, state.current_player)] = True
        if not bool(mask.any().item()):
            raise Phase4F56Error("F56 anchor state has no legal actions")
        states.append(encode_state(state, input_channels=PHASE4_INPUT_CHANNELS))
        masks.append(mask)
    if not states:
        raise Phase4F56Error("F56 anchor set is empty")
    return {"states": torch.stack(states), "legalMasks": torch.stack(masks)}


def _anchor_cache(samples: Sequence[Mapping[str, object]]) -> TensorCache:
    identity = tensorize_anchor_samples(samples)
    mirrored = tensorize_anchor_samples(samples, mirrored=True)
    return TensorCache(
        identity_states=identity["states"],
        identity_policies=None,
        identity_masks=identity["legalMasks"],
        mirrored_states=mirrored["states"],
        mirrored_policies=None,
        mirrored_masks=mirrored["legalMasks"],
        bands=tuple(str(sample["band"]) for sample in samples),
    )


def _load_inputs(
    workspace: Path, contract: Mapping[str, object]
) -> tuple[TensorCache, TensorCache, TensorCache]:
    manifest = _mapping(workspace / F54_MANIFEST, "F54 manifest")
    if (
        _verify_self_hash(manifest, "manifestSha256", "F54 manifest")
        != contract["inputs"]["datasetManifestSha256"]
        or manifest.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
    ):
        raise Phase4F56Error("F54 manifest differs from the F56 contract")
    train = load_policy_samples(workspace, manifest, "train")
    development = load_policy_samples(workspace, manifest, "development")
    anchor = _mapping(workspace / ANCHOR_SET, "F55 anchor set")
    if (
        _verify_self_hash(anchor, "anchorSetSha256", "F55 anchor set")
        != contract["inputs"]["anchorSetSha256"]
        or len(anchor.get("samples", [])) != contract["inputs"]["anchorPositions"]
    ):
        raise Phase4F56Error("F55 anchor set differs from the F56 contract")
    _progress("tensor-cache", "started")
    caches = (_policy_cache(train), _policy_cache(development), _anchor_cache(anchor["samples"]))
    _progress(
        "tensor-cache",
        "completed",
        train=caches[0].positions,
        development=caches[1].positions,
        anchor=caches[2].positions,
    )
    return caches


def _select_batch(
    cache: TensorCache,
    indices: Sequence[int],
    mirrored: Sequence[bool],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    choice = torch.tensor(mirrored, dtype=torch.bool)
    states = torch.where(
        choice[:, None, None, None],
        cache.mirrored_states[index],
        cache.identity_states[index],
    )
    masks = torch.where(
        choice[:, None], cache.mirrored_masks[index], cache.identity_masks[index]
    )
    result = {
        "states": states.to(device, non_blocking=True),
        "legalMasks": masks.to(device, non_blocking=True),
    }
    if cache.identity_policies is not None and cache.mirrored_policies is not None:
        result["policies"] = torch.where(
            choice[:, None],
            cache.mirrored_policies[index],
            cache.identity_policies[index],
        ).to(device, non_blocking=True)
    return result


def _legal_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)


def _teacher_loss(logits: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return -(
        batch["policies"] * _masked_log_softmax(logits, batch["legalMasks"])
    ).sum(dim=1).mean()


def _forward_kl(
    reference: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    log_current = _masked_log_softmax(logits, mask)
    log_reference = torch.where(
        mask,
        torch.log(reference.clamp_min(torch.finfo(reference.dtype).tiny)),
        torch.zeros_like(reference),
    )
    return (reference * (log_reference - log_current)).sum(dim=1).mean()


def _reference_anchor_policies(
    model: PolicyValueNetwork, cache: TensorCache, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for states, masks in (
            (cache.identity_states, cache.identity_masks),
            (cache.mirrored_states, cache.mirrored_masks),
        ):
            chunks: list[torch.Tensor] = []
            for start in range(0, cache.positions, 128):
                batch_states = states[start : start + 128].to(device)
                batch_masks = masks[start : start + 128].to(device)
                logits, _ = model(batch_states)
                chunks.append(_legal_probabilities(logits.float(), batch_masks).cpu())
            outputs.append(torch.cat(chunks))
    return outputs[0], outputs[1]


def _cpu_state(model: PolicyValueNetwork) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().contiguous().clone())
        for name, tensor in sorted(model.state_dict().items())
    )


def _frozen_audit(
    model: PolicyValueNetwork, initial: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    changed: list[str] = []
    maximum = 0.0
    for name, tensor in model.state_dict().items():
        if name.startswith("policy_head."):
            continue
        current = tensor.detach().cpu()
        reference = initial[name]
        if not torch.equal(current, reference):
            changed.append(name)
            if current.is_floating_point():
                maximum = max(maximum, float((current - reference).abs().max().item()))
            else:
                maximum = math.inf
    return {
        "changedTensorNames": changed,
        "maximumAbsoluteError": maximum,
        "passed": not changed,
    }


def evaluate_model(
    model: PolicyValueNetwork,
    development: TensorCache,
    anchor: TensorCache,
    anchor_reference: tuple[torch.Tensor, torch.Tensor],
    baseline_values: tuple[torch.Tensor, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    losses: dict[str, list[float]] = {band: [] for band in ("early", "middle", "late")}
    anchor_kls: list[float] = []
    value_error = 0.0
    model.eval()
    with torch.no_grad():
        for augmentation, (states, policies, masks) in enumerate(
            (
                (development.identity_states, development.identity_policies, development.identity_masks),
                (development.mirrored_states, development.mirrored_policies, development.mirrored_masks),
            )
        ):
            assert policies is not None
            for start in range(0, development.positions, 128):
                stop = min(start + 128, development.positions)
                batch_states = states[start:stop].to(device)
                batch_masks = masks[start:stop].to(device)
                logits, values = model(batch_states)
                rows = -(
                    policies[start:stop].to(device)
                    * _masked_log_softmax(logits.float(), batch_masks)
                ).sum(dim=1)
                value_error = max(
                    value_error,
                    float(
                        (
                            values.float().cpu()
                            - baseline_values[augmentation][start:stop]
                        ).abs().max().item()
                    ),
                )
                for offset, loss in enumerate(rows.cpu().tolist(), start=start):
                    losses[development.bands[offset]].append(float(loss))
        for augmentation, (states, masks, reference) in enumerate(
            (
                (anchor.identity_states, anchor.identity_masks, anchor_reference[0]),
                (anchor.mirrored_states, anchor.mirrored_masks, anchor_reference[1]),
            )
        ):
            del augmentation
            for start in range(0, anchor.positions, 128):
                stop = min(start + 128, anchor.positions)
                batch_states = states[start:stop].to(device)
                batch_masks = masks[start:stop].to(device)
                logits, _ = model(batch_states)
                row_kl = (
                    reference[start:stop].to(device)
                    * (
                        torch.where(
                            batch_masks,
                            torch.log(
                                reference[start:stop]
                                .to(device)
                                .clamp_min(torch.finfo(torch.float32).tiny)
                            ),
                            torch.zeros_like(logits.float()),
                        )
                        - _masked_log_softmax(logits.float(), batch_masks)
                    )
                ).sum(dim=1)
                anchor_kls.extend(float(value) for value in row_kl.cpu().tolist())
    all_losses = [value for values in losses.values() for value in values]
    frozen = _frozen_audit(model, initial_state)
    head_changed = any(
        not torch.equal(tensor.detach().cpu(), initial_state[name])
        for name, tensor in model.state_dict().items()
        if name.startswith("policy_head.")
    )
    return {
        "policyLoss": {
            "all": mean(all_losses),
            **{band: mean(values) for band, values in losses.items()},
        },
        "anchorForwardKl": mean(anchor_kls),
        "valueOutputMaximumAbsoluteError": value_error,
        "frozenTensors": frozen,
        "policyHeadChanged": head_changed,
    }


def acceptance_checks(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    limits: Mapping[str, object],
) -> dict[str, bool]:
    before = baseline["policyLoss"]
    current = candidate["policyLoss"]

    def improvement(band: str) -> float:
        return (float(before[band]) - float(current[band])) / float(before[band])

    def regression(band: str) -> float:
        return -improvement(band)

    finite = all(
        math.isfinite(float(value))
        for value in (
            *current.values(),
            candidate["anchorForwardKl"],
            candidate["valueOutputMaximumAbsoluteError"],
            candidate["frozenTensors"]["maximumAbsoluteError"],
        )
    )
    return {
        "finiteTrainingAndEvaluation": finite,
        "overallPolicyLoss": improvement("all")
        >= float(limits["overallPolicyLossRelativeImprovementMinimum"]),
        "earlyPolicyLoss": improvement("early")
        >= float(limits["earlyPolicyLossRelativeImprovementMinimum"]),
        "middlePolicyLoss": regression("middle")
        <= float(limits["middlePolicyLossRelativeRegressionMaximum"]),
        "latePolicyLoss": regression("late")
        <= float(limits["latePolicyLossRelativeRegressionMaximum"]),
        "anchorForwardKl": float(candidate["anchorForwardKl"])
        <= float(limits["anchorForwardKlMaximum"]),
        "frozenTensors": candidate["frozenTensors"]["passed"] is True
        and float(candidate["frozenTensors"]["maximumAbsoluteError"])
        <= float(limits["frozenTensorMaximumAbsoluteError"]),
        "valueOutput": float(candidate["valueOutputMaximumAbsoluteError"])
        <= float(limits["valueOutputMaximumAbsoluteError"]),
        "policyHeadChanged": candidate["policyHeadChanged"] is True,
    }


def _mixed_anchor_reference(
    identity: torch.Tensor,
    mirrored_reference: torch.Tensor,
    indices: Sequence[int],
    mirrored: Sequence[bool],
    device: torch.device,
) -> torch.Tensor:
    index = torch.tensor(indices, dtype=torch.long)
    choice = torch.tensor(mirrored, dtype=torch.bool)
    return torch.where(
        choice[:, None], mirrored_reference[index], identity[index]
    ).to(device, non_blocking=True)


def _amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def _memory_smoke(
    workspace: Path,
    contract: Mapping[str, object],
    teacher: TensorCache,
    anchor: TensorCache,
    device: torch.device,
) -> dict[str, object]:
    settings = contract["training"]
    model = _load_initial_model(workspace).to(device)
    names = freeze_policy_head(model)
    model.eval()
    initial = _cpu_state(model)
    teacher_sampler = StratifiedSampler(
        teacher.bands,
        batch_size=int(settings["batchSize"]),
        mix=settings["batchMix"],
        seed=int(settings["seed"]),
    )
    anchor_sampler = StratifiedSampler(
        anchor.bands,
        batch_size=int(settings["anchorBatchSize"]),
        mix=settings["batchMix"],
        seed=int(settings["seed"]) + 1,
    )
    anchor_reference = _reference_anchor_policies(model, anchor, device)
    teacher_indices, teacher_mirror = teacher_sampler.batch()
    anchor_indices, anchor_mirror = anchor_sampler.batch()
    teacher_batch = _select_batch(teacher, teacher_indices, teacher_mirror, device)
    anchor_batch = _select_batch(anchor, anchor_indices, anchor_mirror, device)
    reference = _mixed_anchor_reference(
        anchor_reference[0], anchor_reference[1], anchor_indices, anchor_mirror, device
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learningRate"]),
        weight_decay=float(settings["weightDecay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats(device)
    with _amp_context(True):
        teacher_logits, teacher_values = model(teacher_batch["states"])
        anchor_logits, anchor_values = model(anchor_batch["states"])
        teacher_loss = _teacher_loss(teacher_logits.float(), teacher_batch)
        anchor_loss = _forward_kl(reference, anchor_logits.float(), anchor_batch["legalMasks"])
        loss = teacher_loss + float(settings["loss"]["f21PolicyPreservationWeight"]) * anchor_loss
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(settings["maximumGradientNorm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    frozen = _frozen_audit(model, initial)
    result = {
        "passed": bool(
            teacher_logits.dtype == torch.float16
            and anchor_logits.dtype == torch.float16
            and teacher_values.dtype == torch.float16
            and anchor_values.dtype == torch.float16
            and math.isfinite(float(loss.item()))
            and math.isfinite(float(gradient_norm.item()))
            and frozen["passed"]
        ),
        "teacherBatchSize": int(teacher_batch["states"].shape[0]),
        "anchorBatchSize": int(anchor_batch["states"].shape[0]),
        "ampEnabled": True,
        "outputDtypes": {
            "teacherPolicy": str(teacher_logits.dtype),
            "teacherValue": str(teacher_values.dtype),
            "anchorPolicy": str(anchor_logits.dtype),
            "anchorValue": str(anchor_values.dtype),
        },
        "loss": float(loss.item()),
        "gradientNorm": float(gradient_norm.item()),
        "trainableParameterNames": list(names),
        "frozenTensors": frozen,
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
    }
    if not result["passed"]:
        raise Phase4F56Error("F56 current-source CUDA AMP memory smoke failed")
    del model, optimizer, scaler, teacher_batch, anchor_batch, anchor_reference
    torch.cuda.empty_cache()
    return result


def _baseline_values(
    model: PolicyValueNetwork, cache: TensorCache, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for states in (cache.identity_states, cache.mirrored_states):
            chunks = []
            for start in range(0, cache.positions, 128):
                _, values = model(states[start : start + 128].to(device))
                chunks.append(values.float().cpu())
            outputs.append(torch.cat(chunks))
    return outputs[0], outputs[1]


def _write_checkpoint(
    workspace: Path,
    state: Mapping[str, torch.Tensor],
    *,
    step: int,
    contract_sha: str,
    runner_sha: str,
) -> dict[str, object]:
    model_state_sha = canonical_model_state_sha256(state)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "format": "quoridor-phase4f56-isolated-policy-head-pilot-model-only",
        "step": step,
        "contractSha256": contract_sha,
        "runnerSha256": runner_sha,
        "initialModelStateSha256": EXPECTED_F21_STATE_SHA256,
        "modelStateSha256": model_state_sha,
        "modelState": OrderedDict((name, tensor.clone()) for name, tensor in state.items()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path = workspace / CHECKPOINT
    encoded = buffer.getvalue()
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise Phase4F56Error("existing F56 checkpoint differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_create_bytes(path, encoded)
    return {
        "path": CHECKPOINT.as_posix(),
        "fileSha256": sha256_file(path),
        "modelStateSha256": model_state_sha,
        "step": step,
        "bytes": path.stat().st_size,
    }


def run_phase4f56(
    workspace_root: str | Path, device_name: str = "cuda:0"
) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f56(workspace), "reused": True}
    preparation = verify_phase4f55(workspace)
    contract = _contract(workspace)
    if preparation.get("executionContractSha256") != EXPECTED_CONTRACT_SHA256:
        raise Phase4F56Error("F55 preparation does not authorize this F56 contract")
    device = torch.device(device_name)
    cuda = _cuda_preflight(device)
    resources = _resource_preflight(workspace, contract)
    source = _runner_source(workspace)
    train, development, anchor = _load_inputs(workspace, contract)
    settings = contract["training"]
    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline_hours = float(contract["resources"]["baselineGlobalGpuHours"])
    maximum_hours = float(contract["resources"]["maximumGlobalGpuHours"])
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F56Error("a running GPU ledger entry blocks F56")
    if not baseline_hours <= ledger.total_hours <= maximum_hours:
        raise Phase4F56Error("F56 GPU budget envelope is unavailable")
    run_baseline_hours = ledger.total_hours
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    started = time.monotonic()
    status = "interrupted"
    try:
        with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
            ledger.begin(run_id, external_hours=0.0)
            smoke = _memory_smoke(workspace, contract, train, anchor, device)
            _progress(
                "memory-smoke",
                "completed",
                peakAllocatedBytes=smoke["peakAllocatedBytes"],
                peakReservedBytes=smoke["peakReservedBytes"],
            )
            model = _load_initial_model(workspace).to(device)
            trainable = freeze_policy_head(model)
            initial_state = _cpu_state(model)
            anchor_reference = _reference_anchor_policies(model, anchor, device)
            baseline_values = _baseline_values(model, development, device)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(settings["learningRate"]),
                weight_decay=float(settings["weightDecay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(settings["optimizerSteps"]),
                eta_min=float(settings["minimumLearningRate"]),
            )
            scaler = torch.amp.GradScaler("cuda", enabled=bool(settings["cudaAmp"]))
            teacher_sampler = StratifiedSampler(
                train.bands,
                batch_size=int(settings["batchSize"]),
                mix=settings["batchMix"],
                seed=seed,
            )
            anchor_sampler = StratifiedSampler(
                anchor.bands,
                batch_size=int(settings["anchorBatchSize"]),
                mix=settings["batchMix"],
                seed=seed + 1,
            )
            evaluation_steps = set(int(value) for value in settings["evaluationSteps"])
            evaluations: list[dict[str, object]] = []
            baseline: dict[str, object] | None = None
            best: tuple[float, int, OrderedDict[str, torch.Tensor]] | None = None
            best_observed: tuple[
                float, int, OrderedDict[str, torch.Tensor]
            ] | None = None
            timing: list[float] = []
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(int(settings["optimizerSteps"]) + 1):
                if step in evaluation_steps:
                    metrics = evaluate_model(
                        model,
                        development,
                        anchor,
                        anchor_reference,
                        baseline_values,
                        initial_state,
                        device,
                    )
                    if baseline is None:
                        baseline = metrics
                        checks = {"baseline": True}
                        eligible = False
                    else:
                        checks = acceptance_checks(
                            baseline, metrics, contract["evaluation"]["acceptance"]
                        )
                        eligible = all(checks.values())
                    evaluations.append(
                        {"step": step, "metrics": metrics, "checks": checks, "eligible": eligible}
                    )
                    score = float(metrics["policyLoss"]["all"])
                    if step > 0 and (
                        best_observed is None or score < best_observed[0]
                    ):
                        best_observed = (score, step, _cpu_state(model))
                    if eligible and (
                        best is None
                        or score < best[0]
                    ):
                        best = (score, step, _cpu_state(model))
                    total = ledger.update(
                        run_id,
                        elapsed_seconds=time.monotonic() - started,
                        status="running",
                    )
                    _progress(
                        "evaluation",
                        "completed",
                        step=step,
                        eligible=eligible,
                        policyLoss=metrics["policyLoss"]["all"],
                        earlyPolicyLoss=metrics["policyLoss"]["early"],
                        anchorForwardKl=metrics["anchorForwardKl"],
                        globalGpuHours=total,
                    )
                if step == int(settings["optimizerSteps"]):
                    break
                step_started = time.monotonic()
                teacher_indices, teacher_mirror = teacher_sampler.batch()
                anchor_indices, anchor_mirror = anchor_sampler.batch()
                teacher_batch = _select_batch(
                    train, teacher_indices, teacher_mirror, device
                )
                anchor_batch = _select_batch(
                    anchor, anchor_indices, anchor_mirror, device
                )
                reference = _mixed_anchor_reference(
                    anchor_reference[0],
                    anchor_reference[1],
                    anchor_indices,
                    anchor_mirror,
                    device,
                )
                model.eval()
                with _amp_context(bool(settings["cudaAmp"])):
                    teacher_logits, _ = model(teacher_batch["states"])
                    anchor_logits, _ = model(anchor_batch["states"])
                    teacher_loss = _teacher_loss(teacher_logits.float(), teacher_batch)
                    anchor_loss = _forward_kl(
                        reference, anchor_logits.float(), anchor_batch["legalMasks"]
                    )
                    loss = (
                        float(settings["loss"]["teacherPolicyWeight"]) * teacher_loss
                        + float(settings["loss"]["f21PolicyPreservationWeight"])
                        * anchor_loss
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(settings["maximumGradientNorm"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                timing.append(time.monotonic() - step_started)
                elapsed = time.monotonic() - started
                if run_baseline_hours + elapsed / 3600.0 > maximum_hours:
                    raise Phase4F56Error("F56 pilot GPU budget was exhausted")
                if (step + 1) % 4 == 0:
                    total = ledger.update(run_id, elapsed_seconds=elapsed, status="running")
                    _progress(
                        "training",
                        "progress",
                        step=step + 1,
                        optimizerSteps=int(settings["optimizerSteps"]),
                        loss=float(loss.item()),
                        teacherPolicyLoss=float(teacher_loss.item()),
                        anchorForwardKl=float(anchor_loss.item()),
                        gradientNorm=float(gradient_norm.item()),
                        learningRate=float(optimizer.param_groups[0]["lr"]),
                        globalGpuHours=total,
                    )
            if baseline is None:
                raise Phase4F56Error("F56 baseline evaluation did not execute")
            if best_observed is None:
                raise Phase4F56Error("F56 post-update evaluation did not execute")
            selected = best if best is not None else best_observed
            checkpoint = _write_checkpoint(
                workspace,
                selected[2],
                step=selected[1],
                contract_sha=EXPECTED_CONTRACT_SHA256,
                runner_sha=str(source["sha256"]),
            )
            passed = best is not None
            outcome = (
                "isolated-policy-head-pilot-passed"
                if passed
                else "isolated-policy-head-pilot-failed"
            )
            elapsed = time.monotonic() - started
            total = ledger.update(run_id, elapsed_seconds=elapsed, status="completed")
            status = "completed"
            report = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f56-isolated-policy-head-pilot",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": passed,
                    "contractPath": F56_CONTRACT.as_posix(),
                    "contractSha256": EXPECTED_CONTRACT_SHA256,
                    "runnerSource": source,
                    "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
                    "initialModelStateSha256": EXPECTED_F21_STATE_SHA256,
                    "memorySmoke": smoke,
                    "training": {
                        "optimizerSteps": int(settings["optimizerSteps"]),
                        "trainableParameterNames": list(trainable),
                        "selfPlayGames": 0,
                        "replayStoreUsage": False,
                        "valueTarget": None,
                        "meanStepSeconds": mean(timing),
                        "maximumStepSeconds": max(timing),
                    },
                    "evaluations": evaluations,
                    "selectedCheckpoint": checkpoint,
                    "selectedEligible": passed,
                    "cuda": {
                        **cuda,
                        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
                    },
                    "resources": resources,
                    "elapsedSeconds": elapsed,
                    "baselineGlobalGpuHours": baseline_hours,
                    "runBaselineGlobalGpuHours": run_baseline_hours,
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "reportSha256",
            )
            _write_immutable(workspace / REPORT, report, "F56 pilot report")
            gate = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f56-isolated-policy-head-pilot-gate",
                    "status": "completed",
                    "outcomeBand": outcome,
                    "pilotPassed": passed,
                    "reportPath": REPORT.as_posix(),
                    "reportSha256": report["reportSha256"],
                    "selectedCheckpoint": checkpoint,
                    "optimizerSteps": int(settings["optimizerSteps"]),
                    "globalGpuHours": total,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "nextStage": {
                        "name": "Phase 4F57 isolated policy-head pilot review",
                        "requiresSeparateUserConfirmation": True,
                    },
                },
                "gateSha256",
            )
            _write_immutable(workspace / GATE, gate, "F56 pilot gate")
            return {**gate, "reused": False}
    except BaseException:
        try:
            ledger.update(
                run_id,
                elapsed_seconds=time.monotonic() - started,
                status=status,
            )
        except Exception:
            pass
        raise


def verify_phase4f56(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    _contract(workspace)
    report = _mapping(workspace / REPORT, "F56 report")
    report_sha = _verify_self_hash(report, "reportSha256", "F56 report")
    gate = _mapping(workspace / GATE, "F56 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F56 gate")
    checkpoint = gate.get("selectedCheckpoint")
    if not isinstance(checkpoint, Mapping):
        raise Phase4F56Error("F56 selected checkpoint metadata is malformed")
    path = workspace / str(checkpoint["path"])
    if sha256_file(path) != checkpoint.get("fileSha256"):
        raise Phase4F56Error("F56 selected checkpoint bytes changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F56Error("F56 selected checkpoint is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != checkpoint.get("modelStateSha256")
        or payload.get("contractSha256") != EXPECTED_CONTRACT_SHA256
    ):
        raise Phase4F56Error("F56 selected checkpoint model state changed")
    if (
        report.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or report.get("runnerSource") != _runner_source(workspace)
        or report.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
        or report.get("selectedCheckpoint") != checkpoint
        or report.get("memorySmoke", {}).get("passed") is not True
        or report.get("training", {}).get("trainableParameterNames")
        != [
            "policy_head.0.weight",
            "policy_head.1.weight",
            "policy_head.1.bias",
            "policy_head.4.weight",
            "policy_head.4.bias",
        ]
        or report.get("training", {}).get("selfPlayGames") != 0
        or report.get("training", {}).get("replayStoreUsage") is not False
        or report.get("training", {}).get("valueTarget") is not None
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or gate.get("reportSha256") != report_sha
        or gate.get("optimizerSteps") != 32
        or gate.get("formalTrainingAuthorized") is not False
        or gate.get("promotionAuthorized") is not False
    ):
        raise Phase4F56Error("F56 pilot artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f56-isolated-policy-head-pilot-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "selectedCheckpoint": checkpoint,
        "pilotPassed": gate["pilotPassed"],
        "outcomeBand": gate["outcomeBand"],
        "globalGpuHours": gate["globalGpuHours"],
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
    payload = (
        run_phase4f56(args.workspace_root, args.device)
        if args.command == "run"
        else verify_phase4f56(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
