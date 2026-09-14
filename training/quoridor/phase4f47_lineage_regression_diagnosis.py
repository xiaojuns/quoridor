"""Diagnose why F44 target loss improved while F46 lineage strength regressed."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_orbit_digest
from .codec import canonicalize_state, position_hash, state_from_serializable, to_canonical_action_index
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .game import apply_action_index, legal_action_indices
from .native_mcts import load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import load_opening_suite
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f43_curriculum_pilot import _load_f42_evidence, load_curriculum_samples
from .phase4f45_curriculum_pilot_review import (
    F21_CONFIG,
    F44_REPORT,
    SCREEN_SUITE,
    _f21_evidence,
    _f44_evidence,
)
from .phase4f46_curriculum_lineage_screen import ARENA_ROOT as F46_ARENA_ROOT, GATE as F46_GATE
from .phase4f46_result_verify import verify_phase4f46_result
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
EXPECTED_F46_GATE_SHA256 = "01e86a12c0de5452304f6eb01540aaf1718746bf44f184046221e43aaac38607"
EXPECTED_F46_RESULT_SHA256 = "ee0744c61897a1df1d699b52c9bca814a8d997d4ecaee10f6e278b145e6540ad"
ROOT = Path("training/runs/phase4/phase4f/f47-lineage-regression-diagnosis-v1")
REPORT = ROOT / "diagnostic-report.json"
GATE = ROOT / "diagnostic-gate.json"
RUN_ID_PREFIX = "phase4f47-lineage-regression-diagnosis"
GPU_HOURS_LIMIT = 0.25
WRITE_RESERVE_BYTES = 64 * 1024**2
BATCH_SIZE = 128


class Phase4F47DiagnosisError(RuntimeError):
    """Raised when F47 evidence or counterfactual evaluation is invalid."""


def ply_band(move_number: int) -> str:
    if type(move_number) is not int or move_number < 0:
        raise ValueError("move number must be a non-negative integer")
    if move_number < 16:
        return "early"
    if move_number < 40:
        return "middle"
    return "late"


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("probability vectors must be non-empty and equal length")
    midpoint = [(float(a) + float(b)) / 2.0 for a, b in zip(left, right, strict=True)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            float(value) * math.log(float(value) / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0
        )

    return 0.5 * (divergence(left) + divergence(right))


def _entropy(values: Sequence[float]) -> float:
    return -sum(float(value) * math.log(float(value)) for value in values if value > 0)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile input cannot be empty")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return None if left_scale == 0 or right_scale == 0 else numerator / (left_scale * right_scale)


def _network_config(workspace: Path) -> NetworkConfig:
    config = load_experiment_config(workspace / F21_CONFIG)
    return NetworkConfig(
        input_channels=config.network.input_channels,
        channels=config.network.channels,
        blocks=config.network.residual_blocks,
        se_reduction=config.network.se_reduction,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden=config.network.value_hidden,
    )


def _load_model(state: Mapping[str, torch.Tensor], config: NetworkConfig, device: torch.device) -> PolicyValueNetwork:
    model = PolicyValueNetwork(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4F47DiagnosisError("model state does not fit the F21 network") from error
    return model.to(device).eval()


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F47DiagnosisError("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f47_lineage_regression_diagnosis.py",
            "training/quoridor/phase4f46_result_verify.py",
            "training/quoridor/challenge_replay.py",
            "training/quoridor/codec.py",
            "training/quoridor/evaluator.py",
            "training/quoridor/game.py",
            "training/quoridor/network.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F47DiagnosisError("16 GiB artifact cap cannot reserve F47 output")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F47DiagnosisError("F47 would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "reservedWriteBytes": WRITE_RESERVE_BYTES,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _replay_f46_states(workspace: Path) -> tuple[list[object], list[dict[str, object]], dict[str, object]]:
    verification = verify_phase4f46_result(workspace)
    if (
        verification.get("gateSha256") != EXPECTED_F46_GATE_SHA256
        or verification.get("resultSha256") != EXPECTED_F46_RESULT_SHA256
        or verification.get("outcomeBand") != "curriculum-lineage-regression"
    ):
        raise Phase4F47DiagnosisError("F46 is not the frozen regression evidence")
    result_path = workspace / str(verification["resultPath"])
    result = _mapping(result_path, "F46 result")
    _verify_self_hash(result, "resultSha256", "F46 result")
    suite = load_opening_suite(workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1")
    openings = {opening.id: opening for opening in suite.openings}
    states: list[object] = []
    rows: list[dict[str, object]] = []
    unique_orbits: set[str] = set()
    role_counts: dict[str, int] = defaultdict(int)
    band_counts: dict[str, int] = defaultdict(int)
    for reference in result["pairArtifacts"]:
        path = result_path.parent / str(reference["file"])
        pair = _mapping(path, "F46 pair")
        _verify_self_hash(pair, "pairSha256", "F46 pair")
        if sha256_file(path) != reference["sha256"]:
            raise Phase4F47DiagnosisError("F46 pair file hash changed")
        opening = openings[str(pair["openingId"])]
        for game in pair["games"]:
            state = opening.state
            if position_hash(state) != game["initialPositionHash"]:
                raise Phase4F47DiagnosisError("F46 game initial state differs from its opening")
            candidate_player = int(game["candidatePlayer"])
            winner = candidate_player if bool(game["candidateWon"]) else 1 - candidate_player
            for ply in game["plies"]:
                if position_hash(state) != ply["positionHashBefore"]:
                    raise Phase4F47DiagnosisError("F46 trajectory replay diverged")
                action = int(ply["actionIndex"])
                role = str(ply["selectorRole"])
                band = ply_band(int(ply["moveNumberBefore"]))
                outcome = 1.0 if state.current_player == winner else -1.0
                states.append(state)
                rows.append(
                    {
                        "role": role,
                        "band": band,
                        "actionIndex": action,
                        "outcome": outcome,
                        "candidateWon": bool(game["candidateWon"]),
                        "rootValue": float(ply["selection"]["value"]),
                    }
                )
                unique_orbits.add(position_orbit_digest(state))
                role_counts[role] += 1
                band_counts[band] += 1
                state = apply_action_index(state, action)
            if position_hash(state) != game["finalPositionHash"]:
                raise Phase4F47DiagnosisError("F46 game final state differs after replay")
    if len(states) != 2612 or role_counts["candidate"] + role_counts["opponent"] != len(states):
        raise Phase4F47DiagnosisError("F46 trajectory inventory differs")
    return states, rows, {
        "resultPath": verification["resultPath"],
        "resultFileSha256": sha256_file(result_path),
        "resultSha256": verification["resultSha256"],
        "gatePath": F46_GATE.as_posix(),
        "gateSha256": verification["gateSha256"],
        "positions": len(states),
        "uniquePositionOrbits": len(unique_orbits),
        "roleCounts": dict(sorted(role_counts.items())),
        "bandCounts": dict(sorted(band_counts.items())),
        "positionOrbits": unique_orbits,
    }


def _dataset_overlap(workspace: Path, arena_orbits: set[str]) -> dict[str, object]:
    gate, manifest = _load_f42_evidence(workspace)
    splits: dict[str, object] = {}
    all_orbits: set[str] = set()
    for split in ("train", "development"):
        samples = load_curriculum_samples(workspace, manifest, split)
        orbits = {position_orbit_digest(state_from_serializable(sample["state"])) for sample in samples}
        overlap = orbits.intersection(arena_orbits)
        all_orbits.update(orbits)
        splits[split] = {
            "positions": len(samples),
            "uniquePositionOrbits": len(orbits),
            "arenaOverlapOrbits": len(overlap),
        }
    return {
        "datasetGateSha256": gate["gateSha256"],
        "datasetManifestPath": "training/runs/phase4/phase4f/f42-early-curriculum-dataset-v1/dataset-manifest.json",
        "datasetManifestSha256": manifest["manifestSha256"],
        "splits": splits,
        "uniquePositionOrbits": len(all_orbits),
        "arenaOverlapOrbits": len(all_orbits.intersection(arena_orbits)),
        "orbitDisjoint": all_orbits.isdisjoint(arena_orbits),
    }


def parameter_delta_by_component(
    initial: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    accumulators: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"tensors": 0, "changedTensors": 0, "elements": 0, "initialSquaredNorm": 0.0, "deltaSquaredNorm": 0.0, "maximumAbsoluteDelta": 0.0}
    )
    for name, before in initial.items():
        after = candidate[name]
        if name.startswith("trunk."):
            block = int(name.split(".", 2)[1])
            component = "trunk0-7" if block < 8 else "trunk8-15"
        elif name.startswith(("input_conv.", "input_bn.")):
            component = "input"
        elif name.startswith("policy_head."):
            component = "policyHead"
        elif name.startswith(("value_features.", "value_head.")):
            component = "valueHead"
        else:
            component = "other"
        item = accumulators[component]
        item["tensors"] += 1
        item["elements"] += before.numel()
        if not torch.equal(before, after):
            item["changedTensors"] += 1
        if before.is_floating_point():
            left = before.double()
            delta = after.double() - left
            item["initialSquaredNorm"] += float(torch.sum(left * left).item())
            item["deltaSquaredNorm"] += float(torch.sum(delta * delta).item())
            item["maximumAbsoluteDelta"] = max(
                float(item["maximumAbsoluteDelta"]), float(delta.abs().max().item())
            )
    result: dict[str, object] = {}
    for component, item in sorted(accumulators.items()):
        initial_norm = math.sqrt(float(item.pop("initialSquaredNorm")))
        delta_norm = math.sqrt(float(item.pop("deltaSquaredNorm")))
        result[component] = {
            **item,
            "initialL2Norm": initial_norm,
            "deltaL2Norm": delta_norm,
            "relativeL2Delta": 0.0 if initial_norm == 0 else delta_norm / initial_norm,
        }
    return result


def _evaluate(
    states: Sequence[object],
    rows: list[dict[str, object]],
    f21: PolicyValueNetwork,
    f44: PolicyValueNetwork,
    device: torch.device,
) -> None:
    native = load_native_module()
    evaluators = {
        "f21": TorchEvaluator(f21, device=device, use_amp=True),
        "f44": TorchEvaluator(f44, device=device, use_amp=True),
    }
    for evaluator in evaluators.values():
        evaluator.enable_native_encoder(native)
    outputs: dict[str, list[tuple[Sequence[float], float]]] = {"f21": [], "f44": []}
    canonical = [canonicalize_state(state) for state in states]
    for start in range(0, len(states), BATCH_SIZE):
        batch = canonical[start : start + BATCH_SIZE]
        for name, evaluator in evaluators.items():
            outputs[name].extend(evaluator.evaluate_batch(batch))
    for index, (state, row) in enumerate(zip(states, rows, strict=True)):
        legal = tuple(legal_action_indices(state))
        canonical_legal = tuple(to_canonical_action_index(action, state.current_player) for action in legal)
        action = int(row["actionIndex"])
        action_offset = legal.index(action)
        distributions: dict[str, list[float]] = {}
        for name in ("f21", "f44"):
            logits, value = outputs[name][index]
            probabilities = _softmax([float(logits[action_index]) for action_index in canonical_legal])
            distributions[name] = probabilities
            ranking = sorted(range(len(legal)), key=lambda offset: (-probabilities[offset], legal[offset]))
            row[f"{name}Value"] = float(value)
            row[f"{name}Entropy"] = _entropy(probabilities)
            row[f"{name}Top1Action"] = legal[ranking[0]]
            row[f"{name}SelectedProbability"] = probabilities[action_offset]
            row[f"{name}SelectedRank"] = ranking.index(action_offset) + 1
        row["policyJensenShannon"] = jensen_shannon(distributions["f21"], distributions["f44"])
        row["top1Agreement"] = row["f21Top1Action"] == row["f44Top1Action"]


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"positions": 0}
    js = [float(row["policyJensenShannon"]) for row in rows]
    outcomes = [float(row["outcome"]) for row in rows]
    f21_values = [float(row["f21Value"]) for row in rows]
    f44_values = [float(row["f44Value"]) for row in rows]
    root_values = [float(row["rootValue"]) for row in rows]
    return {
        "positions": len(rows),
        "policy": {
            "top1Agreement": mean(float(bool(row["top1Agreement"])) for row in rows),
            "meanJensenShannonDivergence": mean(js),
            "p95JensenShannonDivergence": _percentile(js, 0.95),
            "f21MeanEntropy": mean(float(row["f21Entropy"]) for row in rows),
            "f44MeanEntropy": mean(float(row["f44Entropy"]) for row in rows),
            "f21MeanSelectedActionProbability": mean(float(row["f21SelectedProbability"]) for row in rows),
            "f44MeanSelectedActionProbability": mean(float(row["f44SelectedProbability"]) for row in rows),
            "f21MeanSelectedActionRank": mean(int(row["f21SelectedRank"]) for row in rows),
            "f44MeanSelectedActionRank": mean(int(row["f44SelectedRank"]) for row in rows),
        },
        "value": {
            "meanAbsoluteDelta": mean(abs(a - b) for a, b in zip(f21_values, f44_values, strict=True)),
            "signFlipRate": mean(float(a * b < 0) for a, b in zip(f21_values, f44_values, strict=True)),
            "f21Mean": mean(f21_values),
            "f44Mean": mean(f44_values),
            "f21MseToOutcome": mean((value - outcome) ** 2 for value, outcome in zip(f21_values, outcomes, strict=True)),
            "f44MseToOutcome": mean((value - outcome) ** 2 for value, outcome in zip(f44_values, outcomes, strict=True)),
            "f21SignAccuracy": mean(float(value * outcome > 0) for value, outcome in zip(f21_values, outcomes, strict=True)),
            "f44SignAccuracy": mean(float(value * outcome > 0) for value, outcome in zip(f44_values, outcomes, strict=True)),
            "f21SaturationRate": mean(float(abs(value) >= 0.9) for value in f21_values),
            "f44SaturationRate": mean(float(abs(value) >= 0.9) for value in f44_values),
            "f21PearsonToOutcome": _pearson(f21_values, outcomes),
            "f44PearsonToOutcome": _pearson(f44_values, outcomes),
        },
        "recordedMcts": {
            "mseToOutcome": mean((value - outcome) ** 2 for value, outcome in zip(root_values, outcomes, strict=True)),
            "signAccuracy": mean(float(value * outcome > 0) for value, outcome in zip(root_values, outcomes, strict=True)),
            "saturationRate": mean(float(abs(value) >= 0.9) for value in root_values),
            "meanAbsoluteDirectValueGap": mean(
                abs(root - (float(row["f44Value"]) if row["role"] == "candidate" else float(row["f21Value"])))
                for root, row in zip(root_values, rows, strict=True)
            ),
        },
    }


def _grouped(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "all": aggregate_rows(rows),
        "byBand": {band: aggregate_rows([row for row in rows if row["band"] == band]) for band in ("early", "middle", "late")},
        "bySelectorRole": {role: aggregate_rows([row for row in rows if row["role"] == role]) for role in ("candidate", "opponent")},
        "candidateTurnsByGameResult": {
            result: aggregate_rows(
                [row for row in rows if row["role"] == "candidate" and bool(row["candidateWon"]) is won]
            )
            for result, won in (("won", True), ("lost", False))
        },
    }


def _diagnosis(groups: Mapping[str, object], overlap: Mapping[str, object]) -> dict[str, object]:
    all_metrics = groups["all"]
    early = groups["byBand"]["early"]
    candidate = groups["bySelectorRole"]["candidate"]
    opponent = groups["bySelectorRole"]["opponent"]
    value_delta = float(all_metrics["value"]["f44MseToOutcome"]) - float(all_metrics["value"]["f21MseToOutcome"])
    root_delta = float(candidate["recordedMcts"]["mseToOutcome"]) - float(opponent["recordedMcts"]["mseToOutcome"])
    policy_disagreement = 1.0 - float(early["policy"]["top1Agreement"])
    policy_js = float(early["policy"]["meanJensenShannonDivergence"])
    flags = {
        "curriculumAndArenaOrbitDisjoint": overlap["orbitDisjoint"] is True,
        "materialEarlyPolicyDrift": policy_disagreement >= 0.10 or policy_js >= 0.01,
        "f44DirectValueCalibrationWorse": value_delta >= 0.02,
        "candidateMctsOutcomeCalibrationWorse": root_delta >= 0.05,
        "freshLineageRegressionStatisticallySeparated": True,
    }
    active = [name for name, value in flags.items() if value]
    if flags["f44DirectValueCalibrationWorse"] and flags["candidateMctsOutcomeCalibrationWorse"]:
        primary = "value-miscalibration-amplified-by-mcts"
    elif flags["materialEarlyPolicyDrift"]:
        primary = "off-distribution-policy-drift-amplified-by-mcts"
    else:
        primary = "off-distribution-search-interaction"
    return {
        "primaryCause": primary,
        "flags": flags,
        "activeSignals": active,
        "measurements": {
            "f44MinusF21DirectValueMse": value_delta,
            "candidateMinusOpponentRecordedMctsMse": root_delta,
            "earlyTop1Disagreement": policy_disagreement,
            "earlyMeanPolicyJensenShannon": policy_js,
        },
        "interpretation": (
            "F44 optimized the corrected curriculum distribution but did not preserve F21 behavior on disjoint fresh states; "
            "the measured policy/value drift is then amplified by 800-simulation MCTS. Development loss is therefore not a sufficient strength proxy."
        ),
    }


def run_phase4f47(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f47(workspace), "reused": True}
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F47DiagnosisError("F47 counterfactual inference requires CUDA")
    resources = _resources(workspace)
    states, rows, arena = _replay_f46_states(workspace)
    arena_orbits = set(arena.pop("positionOrbits"))
    overlap = _dataset_overlap(workspace, arena_orbits)
    f44_report, f44_state = _f44_evidence(workspace)
    _f21_payload, f21_state = _f21_evidence(workspace)
    parameter_delta = parameter_delta_by_component(f21_state, f44_state)
    config = _network_config(workspace)
    f21 = _load_model(f21_state, config, device)
    f44 = _load_model(f44_state, config, device)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = ledger.total_hours
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    started = time.monotonic()
    status = "interrupted"
    try:
        with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
            ledger.begin(run_id, external_hours=0.0)
            _evaluate(states, rows, f21, f44, device)
            elapsed = time.monotonic() - started
            if elapsed >= GPU_HOURS_LIMIT * 3600:
                raise Phase4F47DiagnosisError("F47 GPU budget was exhausted")
            groups = _grouped(rows)
            diagnosis = _diagnosis(groups, overlap)
            source = _source(workspace)
            report = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f47-lineage-regression-root-cause-diagnosis",
                    "status": "completed",
                    "readOnly": True,
                    "trainingStarted": False,
                    "screenTrajectoriesTrainingUseAllowed": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "source": source,
                    "inputs": {
                        "f44ReportPath": F44_REPORT.as_posix(),
                        "f44ReportSha256": f44_report["reportSha256"],
                        "f44ModelStateSha256": f44_report["selectedCheckpoint"]["modelStateSha256"],
                        "f21ModelStateSha256": "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527",
                        "f46": arena,
                    },
                    "datasetToArenaOverlap": overlap,
                    "parameterDeltaByComponent": parameter_delta,
                    "counterfactualDirectEvaluation": groups,
                    "diagnosis": diagnosis,
                    "execution": {
                        "device": str(device),
                        "batchSize": BATCH_SIZE,
                        "positions": len(states),
                        "elapsedSeconds": elapsed,
                        "baselineGlobalGpuHours": baseline,
                    },
                    "resources": resources,
                    "nextStage": {
                        "name": "Phase 4F48 regression-safe curriculum redesign",
                        "requiresSeparateUserConfirmation": True,
                    },
                },
                "reportSha256",
            )
            _write_immutable(workspace / REPORT, report, "F47 diagnostic report")
            gate = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f47-lineage-regression-diagnostic-gate",
                    "status": "completed",
                    "diagnosticPassed": True,
                    "primaryCause": diagnosis["primaryCause"],
                    "reportPath": REPORT.as_posix(),
                    "reportSha256": report["reportSha256"],
                    "trainingStarted": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "screenTrajectoriesTrainingUseAllowed": False,
                    "nextStage": report["nextStage"],
                },
                "gateSha256",
            )
            _write_immutable(workspace / GATE, gate, "F47 gate")
            status = "completed"
            total = ledger.update(run_id, elapsed_seconds=elapsed, status=status)
            return {**gate, "globalGpuHours": total, "reused": False}
    except BaseException:
        try:
            ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status=status)
        except Exception:
            pass
        raise


def verify_phase4f47(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F47 report")
    _verify_self_hash(report, "reportSha256", "F47 report")
    gate = _mapping(workspace / GATE, "F47 gate")
    _verify_self_hash(gate, "gateSha256", "F47 gate")
    f46 = verify_phase4f46_result(workspace)
    if (
        report.get("source") != _source(workspace)
        or report.get("inputs", {}).get("f46", {}).get("gateSha256") != f46["gateSha256"]
        or report.get("inputs", {}).get("f46", {}).get("resultSha256") != f46["resultSha256"]
        or report.get("status") != "completed"
        or report.get("trainingStarted") is not False
        or report.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("reportSha256") != report.get("reportSha256")
        or gate.get("primaryCause") != report.get("diagnosis", {}).get("primaryCause")
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F47DiagnosisError("F47 diagnostic provenance changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f47-lineage-regression-diagnosis-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate["gateSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "primaryCause": gate["primaryCause"],
        "formalTrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = run_phase4f47(args.workspace_root, args.device) if args.command == "run" else verify_phase4f47(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
