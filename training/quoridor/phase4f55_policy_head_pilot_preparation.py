"""Freeze the isolated F56 policy-head pilot after validating the F54 dataset."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_orbit_digest
from .codec import (
    mirror_action_index,
    mirror_state,
    position_hash,
    state_from_serializable,
    state_to_serializable,
    to_canonical_action_index,
)
from .network import (
    ACTION_SIZE,
    PHASE4_INPUT_CHANNELS,
    NetworkConfig,
    PolicyValueNetwork,
    encode_state,
)
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f45_curriculum_pilot_review import (
    EXPECTED_F21_CONFIG_FILE_SHA256,
    EXPECTED_F21_MODEL_SHA256,
    EXPECTED_F21_STATE_SHA256,
    F21_CONFIG,
    F21_MODEL,
)
from .phase4f51_learner_teacher_probe import SELECTION as F51_SELECTION
from .phase4f53_policy_head_preparation import TRAIN_SELECTION
from .phase4f54_policy_dataset import (
    DATASET_MANIFEST as F54_MANIFEST,
    EXPECTED_CONTRACT_SHA256 as EXPECTED_F54_CONTRACT_SHA256,
    GATE as F54_GATE,
    REPORT as F54_REPORT,
    _materialize_split,
    validate_sample,
    verify_phase4f54,
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


SCHEMA_VERSION = 1
EXPECTED_F54_GATE_SHA256 = "26324301f03a0440034f380e4c6cc168a81e40c38f6e45d4f969af9a70fe3901"
EXPECTED_F54_REPORT_SHA256 = "5c66c1d6290ac428c4198415cf0ed811d7ba99a9d8e93aac9a83bf37361f25e7"
EXPECTED_F54_MANIFEST_SHA256 = "a87a7493e2434b944bb3adda944da32ef0e4f53f91dc8ff36a427a8b3f3e26da"
EXPECTED_F54_DATASET_FINGERPRINT = "6f7d652e9be6b89afc0f4252cd497bd495888eabe09452845f210f58bcbc124c"
EXPECTED_F51_SELECTION_SHA256 = "57b3c0b41c766ca6db6a0e89300f5a29f6a7c0166b3db8177d7029f1b2ae8f0b"
EXPECTED_F53_SELECTION_SHA256 = "95da56ca2244096ab9ce2ecee53aa6397de918b68eaafbe3a38a7cbc26d516a4"

ROOT = Path("training/runs/phase4/phase4f/f55-policy-head-pilot-preparation-v1")
ANCHOR_SET = ROOT / "f21-policy-anchor-set.json"
REPORT = ROOT / "preparation-report.json"
GATE = ROOT / "preparation-gate.json"
F56_CONTRACT = Path("training/configs/phase4f56-isolated-policy-head-pilot-frozen-v1.json")
RUN_ID_PREFIX = "phase4f55-policy-head-pilot-preparation"

ANCHOR_COUNTS = {"early": 256, "middle": 192, "late": 64}
SMOKE_POSITIONS = 8
F55_WRITE_RESERVE_BYTES = 64 * 1024**2
F56_OUTPUT_RESERVE_BYTES = 384 * 1024**2
F56_GPU_HOURS = 1.0


class Phase4F55Error(RuntimeError):
    """Raised when the isolated policy-head pilot cannot be frozen safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f55-policy-head-pilot-preparation-progress",
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
        raise Phase4F55Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f55_policy_head_pilot_preparation.py",
            "training/quoridor/phase4f54_policy_dataset.py",
            "training/quoridor/codec.py",
            "training/quoridor/network.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    total_reserve = F55_WRITE_RESERVE_BYTES + F56_OUTPUT_RESERVE_BYTES
    if artifact_bytes + total_reserve >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F55Error("16 GiB artifact cap cannot reserve F55 and F56 evidence")
    if free_bytes - total_reserve < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F55Error("F56 reserve would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "f55WriteReserveBytes": F55_WRITE_RESERVE_BYTES,
        "f56OutputReserveBytes": F56_OUTPUT_RESERVE_BYTES,
    }


def _ledger_snapshot(workspace: Path) -> dict[str, object]:
    relative = PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME
    ledger = GpuBudgetLedger(workspace / relative)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F55Error("a running GPU ledger entry blocks F55 preparation")
    if ledger.total_hours + F56_GPU_HOURS > 120.0:
        raise Phase4F55Error("the global 120 GPU-hour budget cannot fit F56")
    return {
        "path": relative.as_posix(),
        "fileSha256": sha256_file(workspace / relative),
        "totalGpuHours": ledger.total_hours,
        "maximumAfterF56": ledger.total_hours + F56_GPU_HOURS,
    }


def _f54_evidence(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f54(workspace)
    gate = _mapping(workspace / F54_GATE, "F54 gate")
    report = _mapping(workspace / F54_REPORT, "F54 report")
    manifest = _mapping(workspace / F54_MANIFEST, "F54 dataset manifest")
    hashes = {
        "gate": _verify_self_hash(gate, "gateSha256", "F54 gate"),
        "report": _verify_self_hash(report, "reportSha256", "F54 report"),
        "manifest": _verify_self_hash(
            manifest, "manifestSha256", "F54 dataset manifest"
        ),
    }
    expected = {
        "gate": EXPECTED_F54_GATE_SHA256,
        "report": EXPECTED_F54_REPORT_SHA256,
        "manifest": EXPECTED_F54_MANIFEST_SHA256,
    }
    if (
        verification.get("status") != "verified"
        or hashes != expected
        or verification.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
        or manifest.get("datasetFingerprint") != EXPECTED_F54_DATASET_FINGERPRINT
        or manifest.get("contractSha256") != EXPECTED_F54_CONTRACT_SHA256
        or manifest.get("trainingStarted") is not False
        or manifest.get("optimizerSteps") != 0
        or gate.get("datasetPassed") is not True
        or gate.get("f55PolicyHeadPilotPreparationAuthorized") is not True
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F55Error("F54 is not the accepted isolated policy dataset")
    return {"gate": gate, "report": report, "manifest": manifest, "hashes": hashes}


def load_policy_samples(
    workspace: Path, manifest: Mapping[str, object], split: str
) -> list[dict[str, object]]:
    if split not in {"train", "development"}:
        raise ValueError("policy split must be train or development")
    references = manifest.get("chunks")
    if not isinstance(references, list):
        raise Phase4F55Error("F54 dataset chunk inventory is malformed")
    samples: list[dict[str, object]] = []
    for reference in references:
        if not isinstance(reference, Mapping) or reference.get("split") != split:
            continue
        chunk = _mapping(workspace / str(reference["path"]), "F54 dataset chunk")
        if (
            _verify_self_hash(chunk, "chunkSha256", "F54 dataset chunk")
            != reference.get("sha256")
        ):
            raise Phase4F55Error("F54 dataset chunk reference changed")
        rows = chunk.get("samples")
        if not isinstance(rows, list) or len(rows) != reference.get("positions"):
            raise Phase4F55Error("F54 dataset chunk position count changed")
        for sample in rows:
            if not isinstance(sample, Mapping):
                raise Phase4F55Error("F54 dataset sample is malformed")
            validate_sample(sample, str(manifest["contractSha256"]))
            samples.append(dict(sample))
    expected = 1024 if split == "train" else 512
    if len(samples) != expected:
        raise Phase4F55Error(f"F54 {split} split must contain {expected} positions")
    return samples


def tensorize_policy_samples(
    samples: Sequence[Mapping[str, object]], *, mirrored: bool = False
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("policy batch cannot be empty")
    states: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    legal_masks: list[torch.Tensor] = []
    for sample in samples:
        original = state_from_serializable(sample["state"])
        state = mirror_state(original) if mirrored else original
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        for action, probability in sample["policy"]:
            raw_action = mirror_action_index(int(action)) if mirrored else int(action)
            canonical = to_canonical_action_index(raw_action, state.current_player)
            policy[canonical] += float(probability)
            mask[canonical] = True
        if (
            not bool(torch.isfinite(policy).all().item())
            or not math.isclose(float(policy.sum().item()), 1.0, abs_tol=1e-6)
            or not bool(mask.any().item())
        ):
            raise Phase4F55Error("F55 tensor adapter changed the policy target")
        states.append(encode_state(state, input_channels=PHASE4_INPUT_CHANNELS))
        policies.append(policy / policy.sum())
        legal_masks.append(mask)
    return {
        "states": torch.stack(states),
        "policies": torch.stack(policies),
        "legalMasks": torch.stack(legal_masks),
    }


def freeze_policy_head(model: PolicyValueNetwork) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.policy_head.requires_grad_(True)
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if not names or any(not name.startswith("policy_head.") for name in names):
        raise Phase4F55Error("policy-head trainable boundary is incorrect")
    model.eval()
    return names


def _network_config(workspace: Path) -> NetworkConfig:
    payload = _mapping(workspace / F21_CONFIG, "F21 network config")
    values = payload.get("network")
    if not isinstance(values, Mapping):
        raise Phase4F55Error("F21 network config is missing")
    return NetworkConfig(
        input_channels=int(values["inputChannels"]),
        channels=int(values["channels"]),
        blocks=int(values["residualBlocks"]),
        se_reduction=int(values["seReduction"]),
        policy_channels=int(values["policyChannels"]),
        value_channels=int(values["valueChannels"]),
        value_hidden=int(values["valueHidden"]),
    )


def _load_initial_model(workspace: Path) -> PolicyValueNetwork:
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_FILE_SHA256:
        raise Phase4F55Error("F21 network config changed")
    if sha256_file(workspace / F21_MODEL) != EXPECTED_F21_MODEL_SHA256:
        raise Phase4F55Error("F21 initialization model changed")
    try:
        payload = torch.load(workspace / F21_MODEL, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F55Error("F21 initialization model is unreadable") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or canonical_model_state_sha256(state) != EXPECTED_F21_STATE_SHA256:
        raise Phase4F55Error("F21 initialization model state changed")
    model = PolicyValueNetwork(_network_config(workspace))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4F55Error("F21 initialization does not fit its network") from error
    if canonical_model_state_sha256(model) != EXPECTED_F21_STATE_SHA256:
        raise Phase4F55Error("loaded F21 model state differs")
    return model


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    return torch.where(mask, values, torch.zeros_like(values))


def run_isolation_smoke(
    model: PolicyValueNetwork, batch: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    trainable = freeze_policy_head(model)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        before_policy, before_value = model(batch["states"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    logits, _value = model(batch["states"])
    loss = -(
        batch["policies"]
        * _masked_log_softmax(logits, batch["legalMasks"])
    ).sum(dim=1).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
    )
    optimizer.step()
    with torch.no_grad():
        after_policy, after_value = model(batch["states"])
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before_state[name])
    ]
    frozen_changed = [name for name in changed if not name.startswith("policy_head.")]
    frozen_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("policy_head.") and parameter.grad is not None
    ]
    policy_delta = float((after_policy - before_policy).abs().max().item())
    value_delta = float((after_value - before_value).abs().max().item())
    if (
        not math.isfinite(float(loss.item()))
        or not math.isfinite(float(gradient_norm.item()))
        or policy_delta <= 0.0
        or value_delta != 0.0
        or frozen_changed
        or frozen_gradients
        or any(not name.startswith("policy_head.") for name in changed)
    ):
        raise Phase4F55Error("policy-head isolation optimizer smoke failed")
    return {
        "positions": int(batch["states"].shape[0]),
        "modelMode": "eval",
        "loss": float(loss.item()),
        "gradientNorm": float(gradient_norm.item()),
        "trainableParameterNames": list(trainable),
        "changedTensorNames": changed,
        "frozenChangedTensorNames": frozen_changed,
        "frozenGradientNames": frozen_gradients,
        "policyLogitMaximumAbsoluteDelta": policy_delta,
        "valueOutputMaximumAbsoluteDelta": value_delta,
        "passed": True,
    }


def select_anchor_rows(
    rows: Sequence[Mapping[str, object]],
    excluded_orbits: set[str],
    counts: Mapping[str, int] = ANCHOR_COUNTS,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for band in ("early", "middle", "late"):
        total = int(counts[band])
        if total <= 0 or total % 2:
            raise ValueError("anchor band counts must be positive and even")
        for side in (0, 1):
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row.get("split") == "train"
                    and row.get("band") == band
                    and row.get("currentPlayer") == side
                    and str(row.get("positionOrbit")) not in excluded_orbits
                ),
                key=lambda row: (str(row["rank"]), str(row["positionOrbit"])),
            )
            required = total // 2
            if len(candidates) < required:
                raise Phase4F55Error(f"F51 cannot fill anchor {band}/side-{side}")
            selected.extend(dict(row) for row in candidates[:required])
    return sorted(selected, key=lambda row: int(row["ordinal"]))


def anchor_audit(
    rows: Sequence[Mapping[str, object]],
    supervised_orbits: set[str],
    development_orbits: set[str],
) -> dict[str, object]:
    orbits = [str(row["positionOrbit"]) for row in rows]
    strata = Counter((str(row["band"]), int(row["currentPlayer"])) for row in rows)
    return {
        "positions": len(rows),
        "uniqueMirrorOrbits": len(set(orbits)),
        "supervisedOrbitOverlap": len(set(orbits) & supervised_orbits),
        "developmentOrbitOverlap": len(set(orbits) & development_orbits),
        "byPlyBand": {
            band: sum(1 for row in rows if row["band"] == band)
            for band in ("early", "middle", "late")
        },
        "byPlyBandAndSide": {
            f"{band}/side-{side}": strata[(band, side)]
            for band in ("early", "middle", "late")
            for side in (0, 1)
        },
    }


def _anchor_checks(audit: Mapping[str, object]) -> dict[str, bool]:
    expected_strata = {
        f"{band}/side-{side}": count // 2
        for band, count in ANCHOR_COUNTS.items()
        for side in (0, 1)
    }
    return {
        "positionCount": audit.get("positions") == sum(ANCHOR_COUNTS.values()),
        "uniqueMirrorOrbits": audit.get("uniqueMirrorOrbits") == audit.get("positions"),
        "supervisedOrbitDisjoint": audit.get("supervisedOrbitOverlap") == 0,
        "developmentOrbitDisjoint": audit.get("developmentOrbitOverlap") == 0,
        "bandCounts": audit.get("byPlyBand") == ANCHOR_COUNTS,
        "bandSideBalance": audit.get("byPlyBandAndSide") == expected_strata,
    }


def _anchor_set(
    workspace: Path,
    source_sha: str,
    supervised: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    f51 = _mapping(workspace / F51_SELECTION, "F51 selection")
    f53 = _mapping(workspace / TRAIN_SELECTION, "F53 train selection")
    if (
        _verify_self_hash(f51, "selectionSha256", "F51 selection")
        != EXPECTED_F51_SELECTION_SHA256
        or _verify_self_hash(f53, "selectionSha256", "F53 train selection")
        != EXPECTED_F53_SELECTION_SHA256
    ):
        raise Phase4F55Error("F51 or F53 selection changed")
    supervised_orbits = {
        str(sample["source"]["positionOrbit"]) for sample in supervised
    }
    development_orbits = {
        str(sample["source"]["positionOrbit"]) for sample in development
    }
    rows = select_anchor_rows(f51["rows"], supervised_orbits | development_orbits)
    audit = anchor_audit(rows, supervised_orbits, development_orbits)
    checks = _anchor_checks(audit)
    if not all(checks.values()):
        raise Phase4F55Error("F21 policy anchor isolation gate failed")
    states, metadata = _materialize_split(workspace, "train", rows, 256)
    samples: list[dict[str, object]] = []
    for ordinal in sorted(metadata):
        state = states[ordinal]
        row = metadata[ordinal]
        if (
            position_hash(state) != row["positionHash"]
            or position_orbit_digest(state) != row["positionOrbit"]
        ):
            raise Phase4F55Error("materialized F21 anchor state changed")
        samples.append(
            {
                "ordinal": ordinal,
                "band": row["band"],
                "currentPlayer": row["currentPlayer"],
                "gameId": row["gameId"],
                "plyIndex": row["plyIndex"],
                "positionHash": row["positionHash"],
                "positionOrbit": row["positionOrbit"],
                "state": state_to_serializable(state),
            }
        )
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f55-f21-policy-preservation-anchor-set",
            "status": "completed",
            "sourceSha256": source_sha,
            "selectionAlgorithm": "lowest-sha256-rank-excluded-f54-orbits-by-band-and-side-v1",
            "targetAlgorithm": "frozen-F21-legal-policy-distribution-at-F56-start-v1",
            "f51SelectionSha256": EXPECTED_F51_SELECTION_SHA256,
            "f53SelectionSha256": EXPECTED_F53_SELECTION_SHA256,
            "f54DatasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
            "audit": audit,
            "checks": checks,
            "samples": samples,
        },
        "anchorSetSha256",
    )


def _initialization() -> dict[str, object]:
    return {
        "mode": "candidate-model",
        "modelPath": F21_MODEL.as_posix(),
        "modelFileSha256": EXPECTED_F21_MODEL_SHA256,
        "modelStateSha256": EXPECTED_F21_STATE_SHA256,
        "networkConfigPath": F21_CONFIG.as_posix(),
        "networkConfigFileSha256": EXPECTED_F21_CONFIG_FILE_SHA256,
        "resetOptimizer": True,
        "emptyReplay": True,
    }


def _contract(
    source_sha: str,
    evidence: Mapping[str, object],
    anchor: Mapping[str, object],
    ledger: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f56-isolated-policy-head-pilot-contract",
            "status": "frozen",
            "objective": "measure isolated classic-policy transfer without changing F21 representation or value behavior",
            "f55SourceSha256": source_sha,
            "inputs": {
                "datasetManifestPath": F54_MANIFEST.as_posix(),
                "datasetManifestSha256": evidence["hashes"]["manifest"],
                "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
                "trainPositions": 1024,
                "developmentPositions": 512,
                "format": evidence["manifest"]["format"],
                "anchorSetPath": ANCHOR_SET.as_posix(),
                "anchorSetSha256": anchor["anchorSetSha256"],
                "anchorPositions": sum(ANCHOR_COUNTS.values()),
            },
            "initialization": _initialization(),
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
                "modelMode": "eval",
                "selfPlayGames": 0,
                "replayStoreUsage": False,
                "valueTarget": None,
                "optimizer": "AdamW",
                "batchSize": 256,
                "anchorBatchSize": 128,
                "optimizerSteps": 32,
                "evaluationSteps": [0, 8, 16, 24, 32],
                "learningRate": 5e-5,
                "minimumLearningRate": 1e-5,
                "schedule": "cosine-annealing",
                "weightDecay": 1e-4,
                "maximumGradientNorm": 5.0,
                "cudaAmp": True,
                "seed": 20260911,
                "batchMix": {"early": 0.5, "middle": 0.375, "late": 0.125},
                "mirrorProbability": 0.5,
                "loss": {
                    "teacherPolicy": "legal-masked-soft-cross-entropy",
                    "teacherPolicyWeight": 1.0,
                    "f21PolicyPreservation": "legal-masked-forward-kl",
                    "f21PolicyPreservationWeight": 0.25,
                    "value": None,
                },
                "trainableParameters": ["policy_head"],
                "frozenParameters": [
                    "input_conv",
                    "input_bn",
                    "trunk",
                    "value_features",
                    "value_head",
                ],
                "frozenBuffers": "all-model-buffers-v1",
                "checkpointSelection": "minimum-development-policy-loss-among-eligible-evaluations",
            },
            "evaluation": {
                "split": "development",
                "usesMirrorAverage": True,
                "compareAgainstUnmodifiedInitialization": True,
                "finalSplitUsage": False,
                "acceptance": {
                    "finiteTrainingAndEvaluation": True,
                    "overallPolicyLossRelativeImprovementMinimum": 0.005,
                    "earlyPolicyLossRelativeImprovementMinimum": 0.0,
                    "middlePolicyLossRelativeRegressionMaximum": 0.01,
                    "latePolicyLossRelativeRegressionMaximum": 0.02,
                    "anchorForwardKlMaximum": 0.02,
                    "frozenTensorMaximumAbsoluteError": 0.0,
                    "valueOutputMaximumAbsoluteError": 0.0,
                    "policyHeadMustChange": True,
                },
            },
            "resources": {
                "baselineGlobalGpuHours": ledger["totalGpuHours"],
                "gpuHoursLimit": F56_GPU_HOURS,
                "maximumGlobalGpuHours": ledger["maximumAfterF56"],
                "globalGpuHoursLimit": 120.0,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "maximumOutputBytes": F56_OUTPUT_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "preflight": {
                "currentSourceCudaAmpMemorySmokeRequired": True,
                "datasetDeepVerificationRequired": True,
                "policyHeadIsolationSmokeRequired": True,
            },
            "authorizations": {
                "cudaPilotTrainingAuthorized": True,
                "selfPlayAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "nextStage": {
                "name": "Phase 4F56 run isolated policy-head CUDA pilot",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "contractSha256",
    )


def prepare_phase4f55(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f55(workspace), "reused": True}
    evidence = _f54_evidence(workspace)
    source = _source(workspace)
    resources = _resources(workspace)
    ledger = _ledger_snapshot(workspace)
    _progress("preflight", "verified", globalGpuHours=ledger["totalGpuHours"])
    supervised = load_policy_samples(workspace, evidence["manifest"], "train")
    development = load_policy_samples(workspace, evidence["manifest"], "development")
    anchor = _anchor_set(
        workspace, str(source["sha256"]), supervised, development
    )
    _progress("anchor-selection", "verified", positions=len(anchor["samples"]))
    smoke_rows = sorted(
        development,
        key=lambda sample: (str(sample["band"]), int(sample["ordinal"])),
    )[:SMOKE_POSITIONS]
    smoke = run_isolation_smoke(
        _load_initial_model(workspace), tensorize_policy_samples(smoke_rows)
    )
    _progress("isolation-smoke", "completed", positions=SMOKE_POSITIONS)
    contract = _contract(str(source["sha256"]), evidence, anchor, ledger)
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        _write_immutable(workspace / ANCHOR_SET, anchor, "F55 anchor set")
        _write_immutable(workspace / F56_CONTRACT, contract, "F56 pilot contract")
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f55-isolated-policy-head-pilot-preparation",
                "status": "completed",
                "cpuOnly": True,
                "cudaAccessed": False,
                "trainingStarted": False,
                "optimizerSteps": 0,
                "source": source,
                "f54Evidence": evidence["hashes"],
                "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
                "positions": {"train": len(supervised), "development": len(development)},
                "anchorSetPath": ANCHOR_SET.as_posix(),
                "anchorSetSha256": anchor["anchorSetSha256"],
                "anchorAudit": anchor["audit"],
                "anchorChecks": anchor["checks"],
                "initialization": _initialization(),
                "isolationSmoke": smoke,
                "executionContractPath": F56_CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "gpuLedger": ledger,
                "resources": resources,
                "globalGpuHours": ledger["totalGpuHours"],
                "authorizations": contract["authorizations"],
                "nextStage": contract["nextStage"],
            },
            "reportSha256",
        )
        _write_immutable(workspace / REPORT, report, "F55 preparation report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f55-policy-head-pilot-preparation-gate",
                "status": "completed",
                "outcomeBand": "isolated-policy-head-cuda-pilot-ready",
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "executionContractPath": F56_CONTRACT.as_posix(),
                "executionContractSha256": contract["contractSha256"],
                "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
                "anchorSetSha256": anchor["anchorSetSha256"],
                "isolationSmokePassed": True,
                "cudaPilotTrainingAuthorized": True,
                "formalTrainingAuthorized": False,
                "trainingStarted": False,
                "globalGpuHours": ledger["totalGpuHours"],
                "nextStage": contract["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F55 preparation gate")
    _progress("preparation", "completed", contractSha256=contract["contractSha256"])
    return {**gate, "reused": False}


def _verify_anchor_set(
    anchor: Mapping[str, object], supervised_orbits: set[str], development_orbits: set[str]
) -> None:
    samples = anchor.get("samples")
    if not isinstance(samples, list) or len(samples) != sum(ANCHOR_COUNTS.values()):
        raise Phase4F55Error("F55 anchor set size changed")
    rows: list[dict[str, object]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise Phase4F55Error("F55 anchor sample is malformed")
        state = state_from_serializable(sample["state"])
        if (
            position_hash(state) != sample.get("positionHash")
            or position_orbit_digest(state) != sample.get("positionOrbit")
            or int(state.current_player) != sample.get("currentPlayer")
        ):
            raise Phase4F55Error("F55 anchor state identity changed")
        rows.append(dict(sample))
    audit = anchor_audit(rows, supervised_orbits, development_orbits)
    if audit != anchor.get("audit") or not all(_anchor_checks(audit).values()):
        raise Phase4F55Error("F55 anchor audit changed")


def verify_phase4f55(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    evidence = _f54_evidence(workspace)
    supervised = load_policy_samples(workspace, evidence["manifest"], "train")
    development = load_policy_samples(workspace, evidence["manifest"], "development")
    anchor = _mapping(workspace / ANCHOR_SET, "F55 anchor set")
    anchor_sha = _verify_self_hash(anchor, "anchorSetSha256", "F55 anchor set")
    _verify_anchor_set(
        anchor,
        {str(sample["source"]["positionOrbit"]) for sample in supervised},
        {str(sample["source"]["positionOrbit"]) for sample in development},
    )
    contract = _mapping(workspace / F56_CONTRACT, "F56 pilot contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F56 pilot contract")
    report = _mapping(workspace / REPORT, "F55 preparation report")
    report_sha = _verify_self_hash(report, "reportSha256", "F55 preparation report")
    gate = _mapping(workspace / GATE, "F55 preparation gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F55 preparation gate")
    source = _source(workspace)
    if (
        report.get("source") != source
        or report.get("status") != "completed"
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or report.get("anchorSetSha256") != anchor_sha
        or report.get("executionContractSha256") != contract_sha
        or report.get("isolationSmoke", {}).get("passed") is not True
        or contract.get("status") != "frozen"
        or contract.get("f55SourceSha256") != source["sha256"]
        or contract.get("inputs", {}).get("datasetFingerprint")
        != EXPECTED_F54_DATASET_FINGERPRINT
        or contract.get("inputs", {}).get("anchorSetSha256") != anchor_sha
        or contract.get("training", {}).get("trainableParameters") != ["policy_head"]
        or contract.get("training", {}).get("valueTarget") is not None
        or contract.get("training", {}).get("selfPlayGames") != 0
        or contract.get("training", {}).get("replayStoreUsage") is not False
        or contract.get("authorizations", {}).get("cudaPilotTrainingAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
        or gate.get("reportSha256") != report_sha
        or gate.get("executionContractSha256") != contract_sha
        or gate.get("cudaPilotTrainingAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F55Error("F55 preparation artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f55-policy-head-pilot-preparation-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "executionContractPath": F56_CONTRACT.as_posix(),
        "executionContractSha256": contract_sha,
        "datasetFingerprint": EXPECTED_F54_DATASET_FINGERPRINT,
        "anchorSetSha256": anchor_sha,
        "positions": {"train": len(supervised), "development": len(development), "anchor": len(anchor["samples"])},
        "cudaPilotTrainingAuthorized": True,
        "formalTrainingAuthorized": False,
        "globalGpuHours": report["globalGpuHours"],
        "nextStage": gate["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = (
        prepare_phase4f55(args.workspace_root)
        if args.command == "prepare"
        else verify_phase4f55(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
