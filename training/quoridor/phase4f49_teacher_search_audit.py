"""Audit actual classic root scores and F21/F44 same-state MCTS coupling."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_orbit_digest
from .classic_bridge import ClassicAIBridge, ClassicSearchBudget
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f45_curriculum_pilot_review import F21_CONFIG, _f21_evidence, _f44_evidence
from .phase4f47_lineage_regression_diagnosis import (
    _load_model,
    _network_config,
    _replay_f46_states,
)
from .phase4f48_regression_safe_redesign import verify_phase4f48
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
CONTRACT = Path("training/configs/phase4f49-teacher-search-audit-frozen-v1.json")
EXPECTED_CONTRACT_SHA256 = "197a5e8247e346e7701b2d7a2c5bf95e9e28e53a74dd648677c6c6ad553147e3"
ROOT = Path("training/runs/phase4/phase4f/f49-teacher-search-audit-v1")
CHECKPOINT = ROOT / "audit-checkpoint.json"
CLASSIC_RESULT = ROOT / "classic-root-audit.json"
MCTS_RESULT = ROOT / "same-state-mcts-audit.json"
REPORT = ROOT / "audit-report.json"
GATE = ROOT / "audit-gate.json"
CLASSIC_SELECTION_SEED = 20260831
MCTS_SELECTION_SEED = 20260832
MCTS_SEED = 20260833
CLASSIC_ACTORS = min(8, max(1, (os.cpu_count() or 2) // 2))
CLASSIC_CHUNK = 8
MCTS_BATCH = 32
WRITE_RESERVE_BYTES = 128 * 1024**2
GPU_HOURS_LIMIT = 4.0


class Phase4F49AuditError(RuntimeError):
    """Raised when the frozen F49 diagnostic contract cannot be satisfied."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F49AuditError("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "src/ai.js",
            "scripts/classic-ai-bridge.mjs",
            "training/quoridor/classic_bridge.py",
            "training/quoridor/evaluator.py",
            "training/quoridor/mcts.py",
            "training/quoridor/native_mcts.py",
            "training/quoridor/phase4f49_teacher_search_audit.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F49AuditError("16 GiB artifact cap cannot reserve F49 output")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F49AuditError("F49 would violate the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "reservedWriteBytes": WRITE_RESERVE_BYTES,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _contract(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f48(workspace)
    contract = _mapping(workspace / CONTRACT, "F49 contract")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F49 contract")
    authorizations = contract.get("authorizations", {})
    if (
        verification.get("f49ContractSha256") != EXPECTED_CONTRACT_SHA256
        or contract_sha != EXPECTED_CONTRACT_SHA256
        or authorizations.get("classicRootAnalysisAuthorized") is not True
        or authorizations.get("diagnosticCudaInferenceAuthorized") is not True
        or authorizations.get("datasetGenerationAuthorized") is not False
        or authorizations.get("cudaTrainingAuthorized") is not False
        or authorizations.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F49AuditError("F48 does not authorize the frozen F49 diagnostics")
    return contract


def select_unique_positions(
    states: Sequence[object], rows: Sequence[Mapping[str, object]], count: int, seed: int
) -> list[tuple[object, Mapping[str, object], str, str]]:
    if len(states) != len(rows) or count <= 0:
        raise ValueError("state/row inventory is invalid")
    unique: dict[str, tuple[object, Mapping[str, object], str, str]] = {}
    for state, row in zip(states, rows, strict=True):
        orbit = position_orbit_digest(state)
        if orbit in unique:
            continue
        from .codec import position_hash

        state_hash = position_hash(state)
        rank = sha256_bytes(f"{seed}\0{orbit}\0{state_hash}".encode("ascii"))
        unique[orbit] = (state, row, orbit, rank)
    if len(unique) < count:
        raise Phase4F49AuditError("F46 does not contain enough unique diagnostic positions")
    return sorted(unique.values(), key=lambda item: (item[3], item[2]))[:count]


def _selection_payload(
    classic: Sequence[tuple[object, Mapping[str, object], str, str]],
    mcts_orbits: set[str],
) -> dict[str, object]:
    from .codec import position_hash

    rows = [
        {
            "ordinal": ordinal,
            "positionHash": position_hash(state),
            "positionOrbit": orbit,
            "absoluteMoveNumber": int(state.move_number),
            "selectorRole": str(row["role"]),
            "recordedActionIndex": int(row["actionIndex"]),
            "outcome": float(row["outcome"]),
            "selectedForMcts": orbit in mcts_orbits,
        }
        for ordinal, (state, row, orbit, _rank) in enumerate(classic)
    ]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f49-f46-diagnostic-position-selection",
        "algorithm": "sha256(seed-nul-orbit-nul-position-hash)-lowest-unique-orbits-v1",
        "classicSeed": CLASSIC_SELECTION_SEED,
        "mctsSeed": MCTS_SELECTION_SEED,
        "classicPositions": len(rows),
        "mctsPositions": sum(bool(row["selectedForMcts"]) for row in rows),
        "rows": rows,
    }


def _checkpoint_payload(
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
    classic_records: Mapping[str, object],
    mcts_records: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f49-resumable-audit-checkpoint",
            "contractSha256": contract_sha,
            "sourceSha256": source_sha,
            "selectionSha256": selection_sha,
            "classicRecords": dict(classic_records),
            "mctsRecords": dict(mcts_records),
        },
        "checkpointSha256",
    )


def _save_checkpoint(workspace: Path, payload: Mapping[str, object]) -> None:
    atomic_replace_bytes(workspace / CHECKPOINT, canonical_json_bytes(payload))


def _load_checkpoint(
    workspace: Path, contract_sha: str, source_sha: str, selection_sha: str
) -> tuple[dict[str, object], dict[str, object]]:
    path = workspace / CHECKPOINT
    if not path.exists():
        return {}, {}
    payload = _mapping(path, "F49 checkpoint")
    _verify_self_hash(payload, "checkpointSha256", "F49 checkpoint")
    if (
        payload.get("contractSha256") != contract_sha
        or payload.get("sourceSha256") != source_sha
        or payload.get("selectionSha256") != selection_sha
    ):
        raise Phase4F49AuditError("F49 checkpoint belongs to a different source or selection")
    classic = payload.get("classicRecords")
    mcts = payload.get("mctsRecords")
    if not isinstance(classic, Mapping) or not isinstance(mcts, Mapping):
        raise Phase4F49AuditError("F49 checkpoint records are malformed")
    return dict(classic), dict(mcts)


def _classic_record(ordinal: int, state: object) -> dict[str, object]:
    budgets: dict[str, object] = {}
    with ClassicAIBridge(request_timeout=180.0) as bridge:
        for limit in (587, 4096):
            result = bridge.analyze(
                state,
                budget=ClassicSearchBudget.nodes(limit),
                max_depth=5,
                timeout=180.0,
            )
            scores = [
                {
                    "actionIndex": item.action_index,
                    "searched": item.searched,
                    "score": item.score,
                }
                for item in result.root_scores
            ]
            budgets[str(limit)] = {
                "format": "legal-action-root-scores-v1",
                "actionIndex": result.action_index,
                "score": result.score,
                "depth": result.depth,
                "nodes": result.nodes,
                "elapsedMs": result.elapsed_ms,
                "rootScoreKind": result.root_score_kind,
                "legalActions": len(scores),
                "searchedActions": sum(bool(item["searched"]) for item in scores),
                "rootScores": scores,
            }
    return {"ordinal": ordinal, "budgets": budgets}


def _run_classic(
    workspace: Path,
    selected: Sequence[tuple[object, Mapping[str, object], str, str]],
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
    classic_records: dict[str, object],
    mcts_records: dict[str, object],
) -> dict[str, object]:
    missing = [index for index in range(len(selected)) if str(index) not in classic_records]
    chunks = [missing[start : start + CLASSIC_CHUNK] for start in range(0, len(missing), CLASSIC_CHUNK)]

    def run_chunk(indices: Sequence[int]) -> list[dict[str, object]]:
        return [_classic_record(index, selected[index][0]) for index in indices]

    if chunks:
        with ThreadPoolExecutor(max_workers=CLASSIC_ACTORS) as pool:
            futures = {pool.submit(run_chunk, chunk): tuple(chunk) for chunk in chunks}
            for future in as_completed(futures):
                for record in future.result():
                    classic_records[str(record["ordinal"])] = record
                checkpoint = _checkpoint_payload(
                    contract_sha, source_sha, selection_sha, classic_records, mcts_records
                )
                _save_checkpoint(workspace, checkpoint)
                print(
                    canonical_json_bytes(
                        {
                            "kind": "phase4f49-progress",
                            "stage": "classic-root-audit",
                            "status": "running",
                            "positions": len(classic_records),
                            "positionsTotal": len(selected),
                        }
                    ).decode("ascii"),
                    flush=True,
                )
    if len(classic_records) != len(selected):
        raise Phase4F49AuditError("classic root audit did not cover all selected positions")
    records = [classic_records[str(index)] for index in range(len(selected))]
    summary = summarize_classic(records)
    payload = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f49-actual-classic-root-audit",
            "status": "completed",
            "engine": "classic-hard-negamax",
            "rootOutputFormat": "legal-action-root-scores-v1",
            "positions": len(records),
            "actors": CLASSIC_ACTORS,
            "records": records,
            "summary": summary,
        },
        "resultSha256",
    )
    if not (workspace / CLASSIC_RESULT).exists():
        _write_immutable(workspace / CLASSIC_RESULT, payload, "F49 classic root result")
    return payload


def _searched_ranking(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    return sorted(
        (item for item in record["rootScores"] if item["searched"]),
        key=lambda item: (-float(item["score"]), int(item["actionIndex"])),
    )


def summarize_classic(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    agreements: list[float] = []
    top12_overlap: list[float] = []
    coverage: dict[str, list[float]] = {"587": [], "4096": []}
    depths: dict[str, Counter[int]] = {"587": Counter(), "4096": Counter()}
    for record in records:
        low = record["budgets"]["587"]
        high = record["budgets"]["4096"]
        agreements.append(float(low["actionIndex"] == high["actionIndex"]))
        low_top = {int(item["actionIndex"]) for item in _searched_ranking(low)[:12]}
        high_top = {int(item["actionIndex"]) for item in _searched_ranking(high)[:12]}
        top12_overlap.append(len(low_top & high_top) / max(1, len(low_top | high_top)))
        for key, item in (("587", low), ("4096", high)):
            coverage[key].append(float(item["searchedActions"]) / float(item["legalActions"]))
            depths[key][int(item["depth"])] += 1
    return {
        "positions": len(records),
        "budgetActionAgreement": mean(agreements),
        "meanTop12Jaccard": mean(top12_overlap),
        "meanSearchedLegalActionCoverage": {key: mean(values) for key, values in coverage.items()},
        "completedDepthCounts": {
            key: {str(depth): count for depth, count in sorted(counts.items())}
            for key, counts in depths.items()
        },
        "actualNegamaxRootScoresAvailable": all(
            item["budgets"][key]["rootScoreKind"] == "alpha-beta-return-v1"
            and item["budgets"][key]["format"] == "legal-action-root-scores-v1"
            for item in records
            for key in ("587", "4096")
        ),
    }


def _sparse_visits(result: object) -> list[list[int]]:
    return [[index, int(value)] for index, value in enumerate(result.visit_counts) if value]


def _run_mcts(
    workspace: Path,
    selected: Sequence[tuple[object, Mapping[str, object], str, str]],
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
    classic_records: dict[str, object],
    mcts_records: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, object], float]:
    native = load_native_module()
    config = load_experiment_config(workspace / F21_CONFIG)
    network_config = _network_config(workspace)
    _f21_payload, f21_state = _f21_evidence(workspace)
    _f44_report, f44_state = _f44_evidence(workspace)
    models = {
        "F21": _load_model(f21_state, network_config, device),
        "F44": _load_model(f44_state, network_config, device),
    }
    evaluators = {
        name: TorchEvaluator(model, device=device, use_amp=True) for name, model in models.items()
    }
    for evaluator in evaluators.values():
        evaluator.enable_native_encoder(native)
    started = time.monotonic()
    for simulations in (100, 400, 800, 3200):
        for start in range(0, len(selected), MCTS_BATCH):
            end = min(len(selected), start + MCTS_BATCH)
            states = [item[0] for item in selected[start:end]]
            chunk_seed = MCTS_SEED + start
            for name in ("F21", "F44"):
                key = f"{name}:{simulations}:{start}"
                if key in mcts_records:
                    continue
                search = NativeMCTS(
                    evaluators[name],
                    MCTSConfig(
                        simulations=simulations,
                        c_base=config.mcts.c_base,
                        c_init=config.mcts.c_init,
                        dirichlet_epsilon=0.0,
                        dirichlet_alpha=config.mcts.dirichlet_alpha,
                        temperature=0.0,
                        add_root_noise=False,
                        seed=chunk_seed,
                    ),
                    module=native,
                )
                results = search.run_batch(states, temperatures=[0.0] * len(states), add_root_noise=False)
                mcts_records[key] = {
                    "model": name,
                    "simulations": simulations,
                    "start": start,
                    "end": end,
                    "seed": chunk_seed,
                    "rows": [
                        {
                            "ordinal": start + offset,
                            "actionIndex": result.action_index,
                            "rootValue": result.root_value,
                            "visits": _sparse_visits(result),
                        }
                        for offset, result in enumerate(results)
                    ],
                }
                checkpoint = _checkpoint_payload(
                    contract_sha, source_sha, selection_sha, classic_records, mcts_records
                )
                _save_checkpoint(workspace, checkpoint)
                print(
                    canonical_json_bytes(
                        {
                            "kind": "phase4f49-progress",
                            "stage": "same-state-mcts-audit",
                            "status": "running",
                            "model": name,
                            "simulations": simulations,
                            "positions": end,
                            "positionsTotal": len(selected),
                        }
                    ).decode("ascii"),
                    flush=True,
                )
    elapsed = time.monotonic() - started
    expected = 2 * 4 * math.ceil(len(selected) / MCTS_BATCH)
    if len(mcts_records) != expected:
        raise Phase4F49AuditError("same-state MCTS audit is incomplete")
    rows_by_key: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for name in ("F21", "F44"):
        for simulations in (100, 400, 800, 3200):
            chunks = [
                value
                for value in mcts_records.values()
                if value["model"] == name and value["simulations"] == simulations
            ]
            rows_by_key[(name, simulations)] = [
                row for chunk in sorted(chunks, key=lambda item: item["start"]) for row in chunk["rows"]
            ]
    summary = summarize_mcts(rows_by_key, [item[1] for item in selected])
    payload = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f49-f21-f44-same-state-mcts-audit",
            "status": "completed",
            "positions": len(selected),
            "sameStatesAndSeeds": True,
            "rootNoise": False,
            "screenTrajectoryUse": "diagnostic-only-never-training",
            "simulations": [100, 400, 800, 3200],
            "batchSize": MCTS_BATCH,
            "records": [mcts_records[key] for key in sorted(mcts_records)],
            "summary": summary,
            "elapsedSeconds": elapsed,
        },
        "resultSha256",
    )
    if not (workspace / MCTS_RESULT).exists():
        _write_immutable(workspace / MCTS_RESULT, payload, "F49 MCTS result")
    return payload, elapsed


def _visit_distribution(row: Mapping[str, object]) -> dict[int, float]:
    total = sum(int(item[1]) for item in row["visits"])
    return {int(item[0]): int(item[1]) / total for item in row["visits"]} if total else {}


def _js_sparse(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    total = 0.0
    for action in set(left) | set(right):
        a, b = left.get(action, 0.0), right.get(action, 0.0)
        middle = (a + b) / 2.0
        if a:
            total += 0.5 * a * math.log(a / middle)
        if b:
            total += 0.5 * b * math.log(b / middle)
    return total


def summarize_mcts(
    rows: Mapping[tuple[str, int], Sequence[Mapping[str, object]]],
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_simulation: dict[str, object] = {}
    for simulations in (100, 400, 800, 3200):
        f21, f44 = rows[("F21", simulations)], rows[("F44", simulations)]
        outcomes = [float(label["outcome"]) for label in labels]
        by_simulation[str(simulations)] = {
            "actionAgreement": mean(float(a["actionIndex"] == b["actionIndex"]) for a, b in zip(f21, f44, strict=True)),
            "meanVisitJensenShannon": mean(
                _js_sparse(_visit_distribution(a), _visit_distribution(b))
                for a, b in zip(f21, f44, strict=True)
            ),
            "meanAbsoluteRootValueGap": mean(abs(float(a["rootValue"]) - float(b["rootValue"])) for a, b in zip(f21, f44, strict=True)),
            "outcomeMse": {
                "F21": mean((float(row["rootValue"]) - outcome) ** 2 for row, outcome in zip(f21, outcomes, strict=True)),
                "F44": mean((float(row["rootValue"]) - outcome) ** 2 for row, outcome in zip(f44, outcomes, strict=True)),
            },
        }
    stability: dict[str, object] = {}
    for name in ("F21", "F44"):
        reference = rows[(name, 3200)]
        stability[name] = {
            str(simulations): {
                "actionAgreementWith3200": mean(
                    float(item["actionIndex"] == target["actionIndex"])
                    for item, target in zip(rows[(name, simulations)], reference, strict=True)
                ),
                "meanAbsoluteRootValueGapTo3200": mean(
                    abs(float(item["rootValue"]) - float(target["rootValue"]))
                    for item, target in zip(rows[(name, simulations)], reference, strict=True)
                ),
            }
            for simulations in (100, 400, 800)
        }
    return {"bySimulation": by_simulation, "withinModelConvergence": stability}


def diagnose(classic: Mapping[str, object], mcts: Mapping[str, object]) -> dict[str, object]:
    by_sim = mcts["bySimulation"]
    convergence = mcts["withinModelConvergence"]
    agreement_3200 = float(by_sim["3200"]["actionAgreement"])
    f44_stability = float(convergence["F44"]["800"]["actionAgreementWith3200"])
    f21_stability = float(convergence["F21"]["800"]["actionAgreementWith3200"])
    f44_mse_delta = float(by_sim["3200"]["outcomeMse"]["F44"]) - float(by_sim["3200"]["outcomeMse"]["F21"])
    if min(f21_stability, f44_stability) < 0.80:
        primary = "search-budget-coupling"
    elif agreement_3200 < 0.80:
        primary = "persistent-network-policy-shift"
    else:
        primary = "mostly-converged-search-with-calibration-shift"
    return {
        "primaryCause": primary,
        "classicTeacherFinding": "actual-negamax-root-interface-required-surrogate-rejected",
        "measurements": {
            "classic587Vs4096ActionAgreement": classic["budgetActionAgreement"],
            "f21F44ActionAgreementAt3200": agreement_3200,
            "f21ActionStability800To3200": f21_stability,
            "f44ActionStability800To3200": f44_stability,
            "f44MinusF21OutcomeMseAt3200": f44_mse_delta,
        },
        "nextProtocol": {
            "teacherPolicySource": "actual-classic-root-search-only",
            "stateSource": "learner-occupancy-rollouts-with-separate-seeds",
            "firstTrainableParameters": ["policy_head"],
            "frozenParameters": ["input", "trunk", "value_features", "value_head"],
            "requiresIndependentF21AnchorReplay": True,
            "requiresPolicyPreservationKl": True,
            "valueTrainingSeparated": True,
        },
    }


def run_phase4f49(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f49(workspace), "reused": True}
    contract = _contract(workspace)
    resources = _resources(workspace)
    source = _source(workspace)
    states, rows, f46 = _replay_f46_states(workspace)
    classic_selected = select_unique_positions(states, rows, 512, CLASSIC_SELECTION_SEED)
    mcts_ranked = select_unique_positions(
        [item[0] for item in classic_selected],
        [item[1] for item in classic_selected],
        128,
        MCTS_SELECTION_SEED,
    )
    mcts_orbits = {item[2] for item in mcts_ranked}
    selection = _selection_payload(classic_selected, mcts_orbits)
    selection_sha = sha256_bytes(canonical_json_bytes(selection))
    classic_records, mcts_records = _load_checkpoint(
        workspace, contract["contractSha256"], source["sha256"], selection_sha
    )
    run_id = f"phase4f49-teacher-search-audit-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        classic_result = _run_classic(
            workspace,
            classic_selected,
            contract["contractSha256"],
            source["sha256"],
            selection_sha,
            classic_records,
            mcts_records,
        )
        device = torch.device(device_name)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise Phase4F49AuditError(
                "classic audit checkpoint completed; same-state MCTS audit requires visible CUDA"
            )
        selected_by_orbit = {item[2]: item for item in classic_selected}
        mcts_selected = [selected_by_orbit[item[2]] for item in mcts_ranked]
        ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
        baseline = ledger.total_hours
        ledger.begin(run_id, external_hours=0.0)
        status = "interrupted"
        gpu_started = time.monotonic()
        try:
            mcts_result, elapsed = _run_mcts(
                workspace,
                mcts_selected,
                contract["contractSha256"],
                source["sha256"],
                selection_sha,
                classic_records,
                mcts_records,
                device,
            )
            if elapsed >= GPU_HOURS_LIMIT * 3600:
                raise Phase4F49AuditError("F49 diagnostic GPU budget was exhausted")
            diagnosis = diagnose(classic_result["summary"], mcts_result["summary"])
            report = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f49-teacher-root-and-search-coupling-audit",
                    "status": "completed",
                    "readOnly": True,
                    "trainingStarted": False,
                    "datasetGenerationStarted": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "screenTrajectoriesTrainingUseAllowed": False,
                    "source": source,
                    "contractPath": CONTRACT.as_posix(),
                    "contractSha256": contract["contractSha256"],
                    "selection": selection,
                    "selectionSha256": selection_sha,
                    "f46Evidence": {key: value for key, value in f46.items() if key != "positionOrbits"},
                    "classicRootAudit": {
                        "path": CLASSIC_RESULT.as_posix(),
                        "resultSha256": classic_result["resultSha256"],
                        "summary": classic_result["summary"],
                    },
                    "sameStateMctsAudit": {
                        "path": MCTS_RESULT.as_posix(),
                        "resultSha256": mcts_result["resultSha256"],
                        "summary": mcts_result["summary"],
                    },
                    "diagnosis": diagnosis,
                    "resources": resources,
                    "execution": {
                        "device": str(device),
                        "classicActors": CLASSIC_ACTORS,
                        "mctsBatchSize": MCTS_BATCH,
                        "baselineGlobalGpuHours": baseline,
                        "diagnosticGpuSeconds": elapsed,
                    },
                    "nextStage": {
                        "name": "Phase 4F50 learner-occupancy teacher-root dataset preparation",
                        "requiresSeparateUserConfirmation": True,
                    },
                },
                "reportSha256",
            )
            _write_immutable(workspace / REPORT, report, "F49 audit report")
            gate = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f49-teacher-search-audit-gate",
                    "status": "completed",
                    "auditPassed": True,
                    "outcomeBand": diagnosis["primaryCause"],
                    "reportPath": REPORT.as_posix(),
                    "reportSha256": report["reportSha256"],
                    "classicResultSha256": classic_result["resultSha256"],
                    "mctsResultSha256": mcts_result["resultSha256"],
                    "trainingStarted": False,
                    "datasetGenerationStarted": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                    "screenTrajectoriesTrainingUseAllowed": False,
                    "nextStage": report["nextStage"],
                },
                "gateSha256",
            )
            _write_immutable(workspace / GATE, gate, "F49 audit gate")
            status = "completed"
            total = ledger.update(run_id, elapsed_seconds=elapsed, status=status)
            return {**gate, "globalGpuHours": total, "reused": False}
        except BaseException:
            try:
                ledger.update(run_id, elapsed_seconds=time.monotonic() - gpu_started, status=status)
            except Exception:
                pass
            raise


def verify_phase4f49(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F49 report")
    gate = _mapping(workspace / GATE, "F49 gate")
    classic = _mapping(workspace / CLASSIC_RESULT, "F49 classic result")
    mcts = _mapping(workspace / MCTS_RESULT, "F49 MCTS result")
    _verify_self_hash(report, "reportSha256", "F49 report")
    _verify_self_hash(gate, "gateSha256", "F49 gate")
    _verify_self_hash(classic, "resultSha256", "F49 classic result")
    _verify_self_hash(mcts, "resultSha256", "F49 MCTS result")
    if (
        report.get("source") != _source(workspace)
        or report.get("contractSha256") != EXPECTED_CONTRACT_SHA256
        or report.get("status") != "completed"
        or report.get("trainingStarted") is not False
        or report.get("datasetGenerationStarted") is not False
        or report.get("screenTrajectoriesTrainingUseAllowed") is not False
        or gate.get("reportSha256") != report.get("reportSha256")
        or gate.get("classicResultSha256") != classic.get("resultSha256")
        or gate.get("mctsResultSha256") != mcts.get("resultSha256")
        or gate.get("formalTrainingAuthorized") is not False
    ):
        raise Phase4F49AuditError("F49 immutable evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f49-teacher-search-audit-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate["gateSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "outcomeBand": gate["outcomeBand"],
        "formalTrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f49(args.workspace_root, args.device)
        if args.command == "run"
        else verify_phase4f49(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
