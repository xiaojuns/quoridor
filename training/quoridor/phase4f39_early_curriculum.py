"""CPU-only Phase 4F39 early-curriculum data and target preparation.

F39 audits the existing classic-guidance replay and freezes the next data and
target contract.  It does not generate trajectories, reanalyse positions, or
start training.  The next stage is deliberately limited to a small target
probe so that policy entropy and value calibration can be measured before a
large dataset is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .replay_store import ReplayStore
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f39-early-curriculum-preparation-v1")
AUDIT = ROOT / "seed-dataset-audit.json"
REPORT = ROOT / "preparation-report.json"
CONTRACT = Path("training/configs/phase4f39-early-curriculum-frozen-v1.json")

ROOT_CAUSE_REPORT = Path("training/runs/phase4/root-cause-ablation-v1/report.json")
EXPECTED_ROOT_CAUSE_REPORT_SHA256 = (
    "b96a4e21af161b63dadd8ec2bb8901b33b9805eddefb1bd153f1b0ca0e1a3eca"
)
F21_REVIEW = Path(
    "training/runs/phase4/phase4f/recovery-v3-preparation-v1/"
    "phase4f21-stage-a-review.json"
)
F21_RUN = Path("training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1")
F21_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
EXPECTED_F21_REVIEW_SHA256 = (
    "1a97d8dd760ffbadb7581d7d055b1bb47731ad7d76f9c7491fe5dfe9488a97a6"
)
EXPECTED_F21_CANDIDATE_SHA256 = (
    "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
)
EXPECTED_F21_CHECKPOINT_SHA256 = (
    "dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b"
)
EXPECTED_F21_MODEL_STATE_SHA256 = (
    "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
)

GUIDANCE_ROOT = Path("training/runs/phase4/phase4f/recovery-challenge-v2-data")
GUIDANCE_MANIFEST = GUIDANCE_ROOT / "teacher-manifest.json"
GUIDANCE_CONTRACT = Path(
    "training/configs/phase4f17-classic-guidance-dataset-frozen-v1.json"
)
EXPECTED_GUIDANCE_CONTRACT_SHA256 = (
    "bbf62d4a84eb032c24e64314670587e0fea0a73a3f473c908f0ca57007e4ef36"
)
EXPECTED_GUIDANCE_DATASET_FINGERPRINT = (
    "fa3342b1e4775fb2e8af8b03ec3b53e14bcdf9d9f7b67230cc587999aa1bd581"
)
EXPECTED_HOLDOUT_MANIFEST_SHA256 = (
    "61f8ec9d9a5927ee40adc330748c305f9fa0b4a091f296b4d7e88d7a532c4a83"
)

TARGET_POSITIONS = {"train": 100_000, "development": 5_000, "final": 5_000}
TARGET_MIX = {"early": 0.60, "middle": 0.20, "late": 0.20}
WRITE_RESERVE_BYTES = 512 * 1024**2


class Phase4F39Error(RuntimeError):
    """Raised when F39 input evidence or an immutable output is invalid."""


def _progress(stage: str, status: str, **fields: object) -> None:
    payload = {
        "kind": "phase4f39-preparation-progress",
        "schemaVersion": SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        **fields,
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


def _legacy_root_cause_mapping(path: Path) -> dict[str, object]:
    """Load the ablation artifact, whose canonical encoding predates newline JSON."""

    if path.is_symlink() or not path.is_file():
        raise Phase4F39Error("root-cause ablation report is missing or is a symlink")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4F39Error("root-cause ablation report is unreadable") from error
    if not isinstance(payload, Mapping):
        raise Phase4F39Error("root-cause ablation report is not a mapping")
    core = dict(payload)
    stored = core.pop("reportSha256", None)
    encoded = json.dumps(
        core, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    if stored != hashlib.sha256(encoded).hexdigest():
        raise Phase4F39Error("root-cause ablation report self hash mismatch")
    return dict(payload)


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F39Error("workspace does not contain training/quoridor")
    return workspace


def ply_band(ply_index: int) -> str:
    if type(ply_index) is not int or ply_index < 0:
        raise ValueError("ply_index must be a non-negative integer")
    if ply_index < 16:
        return "early"
    if ply_index < 40:
        return "middle"
    return "late"


def legal_action_bucket(count: int) -> str:
    if type(count) is not int or count <= 0:
        raise ValueError("legal action count must be positive")
    if count <= 4:
        return "01-04"
    if count <= 32:
        return "05-32"
    if count <= 64:
        return "33-64"
    return "65-plus"


def split_bucket(game_key: str, seed: int = 20260825) -> str:
    """Assign a whole game to an immutable 90/5/5 split."""

    if not isinstance(game_key, str) or not game_key:
        raise ValueError("game_key must be non-empty")
    bucket = int(sha256_bytes(f"{seed}:{game_key}".encode("ascii")), 16) % 20
    if bucket == 0:
        return "final"
    if bucket == 1:
        return "development"
    return "train"


def _open_store(root: Path) -> ReplayStore:
    manifest = _mapping(root / "manifest.json", f"{root} replay manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F39Error("replay capacity is invalid")
    return ReplayStore(
        root,
        max_games=int(capacity["games"]),
        max_plies=int(capacity["plies"]),
        max_bytes=int(capacity["bytes"]),
    )


def _inventory_store(workspace: Path, split: str) -> tuple[dict[str, object], set[str]]:
    relative = GUIDANCE_ROOT / split
    root = workspace / relative
    store = _open_store(root)
    bands: Counter[str] = Counter()
    sides: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    legal_buckets: Counter[str] = Counter()
    orbits: set[str] = set()
    missing_orbits = 0
    _progress(
        "seed-dataset-audit",
        "split-started",
        split=split,
        gamesTotal=store.game_count,
        positionsTotal=store.ply_count,
    )
    for game_number, game_id in enumerate(store.game_ids, start=1):
        game, states = store.load_game_with_states(game_id)
        winner = states[-1].winner
        if winner not in (0, 1):
            raise Phase4F39Error(f"guidance game has no winner: {game_id}")
        for ply_index, ply in enumerate(game.plies):
            state = states[ply_index]
            bands[ply_band(ply_index)] += 1
            sides[str(state.current_player)] += 1
            outcomes["win" if state.current_player == winner else "loss"] += 1
            teacher_action = ply.metadata.get("teacherActionIndex", ply.action_index)
            if type(teacher_action) is not int or not 0 <= teacher_action < 209:
                raise Phase4F39Error("teacher action metadata is invalid")
            actions["pawn" if teacher_action < 81 else "wall"] += 1
            legal_count = ply.metadata.get("legalActionCount")
            if type(legal_count) is not int:
                legal_count = sum(count > 0 for count in ply.visit_counts)
            legal_buckets[legal_action_bucket(legal_count)] += 1
            orbit = ply.metadata.get("teacherOrbitSha256")
            if isinstance(orbit, str) and len(orbit) == 64:
                if orbit in orbits:
                    raise Phase4F39Error(f"duplicate orbit within {split} guidance data")
                orbits.add(orbit)
            else:
                missing_orbits += 1
        if game_number % 100 == 0 or game_number == store.game_count:
            _progress(
                "seed-dataset-audit",
                "progress",
                split=split,
                games=game_number,
                gamesTotal=store.game_count,
                positions=sum(bands.values()),
                positionsTotal=store.ply_count,
            )
    if missing_orbits:
        raise Phase4F39Error(f"{split} guidance positions are missing orbit provenance")
    manifest_path = root / "manifest.json"
    _progress(
        "seed-dataset-audit",
        "split-completed",
        split=split,
        games=store.game_count,
        positions=store.ply_count,
    )
    return (
        {
            "path": relative.as_posix(),
            "manifestSha256": sha256_file(manifest_path),
            "games": store.game_count,
            "positions": store.ply_count,
            "bytes": store.committed_bytes,
            "deepVerified": True,
            "uniqueOrbits": len(orbits),
            "byPlyBand": dict(sorted(bands.items())),
            "bySideToMove": dict(sorted(sides.items())),
            "byOutcome": dict(sorted(outcomes.items())),
            "byTeacherActionKind": dict(sorted(actions.items())),
            "byLegalActionCount": dict(sorted(legal_buckets.items())),
        },
        orbits,
    )


def _root_cause_evidence(workspace: Path) -> dict[str, object]:
    report = _legacy_root_cause_mapping(workspace / ROOT_CAUSE_REPORT)
    report_sha256 = str(report["reportSha256"])
    direct = report.get("direct")
    policy = direct.get("policy") if isinstance(direct, Mapping) else None
    value = direct.get("value") if isinstance(direct, Mapping) else None
    search = report.get("searchMatrix")
    model = report.get("model")
    holdout = report.get("holdout")
    if (
        report_sha256 != EXPECTED_ROOT_CAUSE_REPORT_SHA256
        or report.get("status") != "completed"
        or report.get("readOnlyDiagnostic") is not True
        or report.get("trainingStarted") is not False
        or not isinstance(policy, Mapping)
        or not isinstance(value, Mapping)
        or not isinstance(search, list)
        or not isinstance(model, Mapping)
        or not isinstance(holdout, Mapping)
        or direct.get("positions") != 60
        or model.get("sha256") != EXPECTED_F21_CANDIDATE_SHA256
        or holdout.get("manifestSha256") != EXPECTED_HOLDOUT_MANIFEST_SHA256
    ):
        raise Phase4F39Error("root-cause ablation evidence changed")
    learned_800 = next(
        (
            row
            for row in search
            if isinstance(row, Mapping)
            and row.get("label") == "learned-policy-learned-value"
            and row.get("simulations") == 800
        ),
        None,
    )
    policy_zero_value = next(
        (
            row
            for row in search
            if isinstance(row, Mapping)
            and row.get("label") == "learned-policy-zero-value"
            and row.get("simulations") == 800
        ),
        None,
    )
    if not isinstance(learned_800, Mapping) or not isinstance(
        policy_zero_value, Mapping
    ):
        raise Phase4F39Error("root-cause search matrix is incomplete")
    return {
        "path": ROOT_CAUSE_REPORT.as_posix(),
        "reportSha256": report_sha256,
        "positions": direct["positions"],
        "directPolicyTop1": policy["top1ClassicAgreement"],
        "directPolicyTop5": policy["top5ClassicCoverage"],
        "directValueSignAccuracy": value["signAccuracy"],
        "directValueMse": value["mseToOutcome"],
        "search800ClassicAgreement": learned_800["classicAgreement"],
        "search800EarlyAgreement": learned_800["classicAgreementByBand"]["early"],
        "policyZeroValue800EarlyAgreement": policy_zero_value[
            "classicAgreementByBand"
        ]["early"],
        "diagnosis": "early policy and value errors jointly limit search",
    }


def _f21_evidence(workspace: Path) -> dict[str, object]:
    review = _mapping(workspace / F21_REVIEW, "F21 review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "F21 review")
    candidate = review.get("bestCandidate")
    if (
        review_sha256 != EXPECTED_F21_REVIEW_SHA256
        or review.get("status") != "completed"
        or not isinstance(candidate, Mapping)
        or candidate.get("candidateModelSha256") != EXPECTED_F21_CANDIDATE_SHA256
        or candidate.get("checkpointSha256") != EXPECTED_F21_CHECKPOINT_SHA256
        or candidate.get("modelStateSha256") != EXPECTED_F21_MODEL_STATE_SHA256
        or candidate.get("generation") != 1
    ):
        raise Phase4F39Error("F21 candidate lineage changed")
    model = workspace / F21_RUN / "checkpoints" / str(candidate["candidateModelPath"])
    checkpoint = workspace / F21_RUN / "checkpoints" / str(candidate["checkpointPath"])
    if sha256_file(model) != EXPECTED_F21_CANDIDATE_SHA256:
        raise Phase4F39Error("F21 candidate model changed")
    if sha256_file(checkpoint) != EXPECTED_F21_CHECKPOINT_SHA256:
        raise Phase4F39Error("F21 checkpoint changed")
    return {
        "reviewPath": F21_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "configPath": F21_CONFIG.as_posix(),
        "configSha256": sha256_file(workspace / F21_CONFIG),
        "candidateModelPath": model.relative_to(workspace).as_posix(),
        "candidateModelSha256": EXPECTED_F21_CANDIDATE_SHA256,
        "checkpointPath": checkpoint.relative_to(workspace).as_posix(),
        "checkpointSha256": EXPECTED_F21_CHECKPOINT_SHA256,
        "modelStateSha256": EXPECTED_F21_MODEL_STATE_SHA256,
        "generation": 1,
        "step": 192,
    }


def _guidance_evidence(workspace: Path) -> tuple[dict[str, object], dict[str, object]]:
    contract = _mapping(workspace / GUIDANCE_CONTRACT, "guidance contract")
    contract_sha256 = _verify_self_hash(
        contract, "contractSha256", "guidance contract"
    )
    manifest = _mapping(workspace / GUIDANCE_MANIFEST, "guidance dataset manifest")
    manifest_sha256 = _verify_self_hash(
        manifest, "manifestSha256", "guidance dataset manifest"
    )
    if (
        contract_sha256 != EXPECTED_GUIDANCE_CONTRACT_SHA256
        or manifest.get("datasetFingerprint") != EXPECTED_GUIDANCE_DATASET_FINGERPRINT
        or manifest.get("actual", {}).get("totalPositions") != 29_381
    ):
        raise Phase4F39Error("classic guidance evidence changed")
    train, train_orbits = _inventory_store(workspace, "train")
    holdout, holdout_orbits = _inventory_store(workspace, "holdout")
    overlap = train_orbits.intersection(holdout_orbits)
    if overlap:
        raise Phase4F39Error("guidance train and holdout orbit sets overlap")
    if holdout["manifestSha256"] != EXPECTED_HOLDOUT_MANIFEST_SHA256:
        raise Phase4F39Error("guidance holdout manifest changed")
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f39-seed-dataset-audit",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "datasetManifestPath": GUIDANCE_MANIFEST.as_posix(),
            "datasetManifestSha256": manifest_sha256,
            "datasetFingerprint": EXPECTED_GUIDANCE_DATASET_FINGERPRINT,
            "guidanceContractPath": GUIDANCE_CONTRACT.as_posix(),
            "guidanceContractSha256": contract_sha256,
            "splits": {"train": train, "sealedHoldout": holdout},
            "isolation": {
                "wholeGameSplit": True,
                "trainHoldoutGameOverlap": 0,
                "trainHoldoutOrbitOverlap": 0,
                "sealedHoldoutMayTrain": False,
                "rootCauseDiagnosticPositionsMayTrain": False,
                "arenaTrajectoriesMayTrain": False,
            },
            "capacityGap": {
                "existingTrainPositions": train["positions"],
                "targetTrainPositions": TARGET_POSITIONS["train"],
                "minimumAdditionalTrainPositions": max(
                    0, TARGET_POSITIONS["train"] - int(train["positions"])
                ),
            },
        },
        "auditSha256",
    )
    return audit, {"manifest": manifest, "contract": contract}


def build_contract(
    *,
    root_cause: Mapping[str, object],
    f21: Mapping[str, object],
    audit: Mapping[str, object],
    source_sha256: str,
) -> dict[str, object]:
    """Build the immutable curriculum contract from verified evidence."""

    core = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f39-early-curriculum-frozen-contract",
        "status": "frozen",
        "objective": "repair early policy ranking and value calibration before more self-play",
        "sourceSha256": source_sha256,
        "inputs": {
            "rootCauseAblation": dict(root_cause),
            "initialization": dict(f21),
            "seedDatasetAuditPath": AUDIT.as_posix(),
            "seedDatasetAuditSha256": audit["auditSha256"],
        },
        "dataset": {
            "version": "phase4f39-early-curriculum-v1",
            "minimumPositions": dict(TARGET_POSITIONS),
            "split": {
                "unit": "whole-game",
                "algorithm": "sha256(seed:game-key)-mod-20-v1",
                "seed": 20260825,
                "buckets": {"train": list(range(2, 20)), "development": [1], "final": [0]},
                "finalUsage": "single-read only after model and thresholds are frozen",
            },
            "absolutePlyBands": {
                "early": {"minimumInclusive": 0, "maximumExclusive": 16},
                "middle": {"minimumInclusive": 16, "maximumExclusive": 40},
                "late": {"minimumInclusive": 40, "maximumExclusive": None},
            },
            "trainingBatchMix": dict(TARGET_MIX),
            "balanceAxes": [
                "side-to-move",
                "terminal-outcome",
                "teacher-action-kind",
                "legal-action-count",
                "opening-depth",
            ],
            "seedTrainReuseAllowed": True,
            "seedHoldoutReuseAllowed": False,
            "forbiddenSources": [
                "root-cause diagnostic selected positions",
                "all historical arena trajectories",
                "development and final split positions",
            ],
            "deduplication": "canonical mirror-orbit across all splits and forbidden sources",
        },
        "policyTarget": {
            "algorithm": "classic-static-topk-plus-f21-mcts-visits-v1",
            "actionSpace": 209,
            "topK": 12,
            "classicWeight": 0.50,
            "mctsWeight": 0.50,
            "classicComponent": {
                "candidateSet": "all legal pawn actions plus top static wall actions; always include classic-hard action",
                "score": "classic evaluateState after action from acting-player perspective",
                "normalization": "legal-action masked softmax fitted on train only",
                "minimumEntropy": 0.35,
                "maximumTop1Probability": 0.85,
            },
            "mctsComponent": {
                "model": EXPECTED_F21_CANDIDATE_SHA256,
                "simulations": 800,
                "temperature": 1.0,
                "rootNoise": False,
                "rootSymmetry": "canonical plus mirrored consistency check",
            },
            "legality": "zero probability outside the legal action mask and exact renormalization",
        },
        "valueTarget": {
            "algorithm": "terminal-search-path-calibrated-v1",
            "weights": {"terminalOutcome": 0.60, "calibratedSearch": 0.25, "pathTempo": 0.15},
            "terminalOutcome": "signed from side-to-move",
            "calibratedSearch": {
                "input": "F21 800-simulation root value",
                "calibration": "isotonic fit on train only",
                "developmentUsage": "calibration selection only",
                "finalUsage": "never fit",
            },
            "pathTempo": {
                "formula": "tanh((opponentPath-ownPath+0.35*wallDifference)/4)",
                "clip": [-0.75, 0.75],
            },
            "finalClip": [-1.0, 1.0],
            "environmentRewardShaping": False,
        },
        "auxiliaryTargets": [
            "own-shortest-path",
            "opponent-shortest-path",
            "wall-count-difference",
            "legal-wall-count",
            "one-ply-winning-action-present",
            "one-ply-opponent-threat-present",
        ],
        "targetProbe": {
            "authorized": True,
            "requiresSeparateUserConfirmation": True,
            "trainPositions": 4096,
            "developmentPositions": 512,
            "finalPositions": 0,
            "trainingSteps": 0,
            "acceptance": {
                "policyTargetMeanEntropyMinimum": 0.55,
                "policyTargetTop1ProbabilityP95Maximum": 0.90,
                "classicActionTop5CoverageMinimum": 0.90,
                "mirrorPolicyMaximumAbsoluteError": 1e-6,
                "valueTargetMeanAbsoluteMaximum": 0.55,
                "valueTargetPositiveRateMinimum": 0.35,
                "valueTargetPositiveRateMaximum": 0.65,
                "mirrorValueMaximumAbsoluteError": 1e-6,
            },
        },
        "futureTrainingGates": {
            "trainingAuthorized": False,
            "stageA1": {
                "freezeResidualBlocks": 8,
                "train": "last 8 residual blocks plus policy/value/auxiliary heads",
                "maximumSteps": 2000,
            },
            "stageA2": {"fullUnfreeze": True, "maximumSteps": 1000, "lowerLearningRate": True},
            "batchMix": dict(TARGET_MIX),
            "distillationProtection": {"middleLateMaximumRegressionPoints": 3.0},
            "acceptance": {
                "earlyPolicyTop1Minimum": 0.60,
                "earlyPolicyTop5Minimum": 0.90,
                "earlyValueSignAccuracyMinimum": 0.60,
                "earlyValueMseMaximum": 1.30,
                "earlyConfidentWrongMaximumPer20": 2,
                "earlySearchClassicAgreementMinimum": 0.50,
                "searchScalingMustBeNonDecreasing": True,
            },
        },
        "authorizations": {
            "datasetGenerationAuthorized": False,
            "targetProbeAuthorized": True,
            "cudaTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
        },
        "nextStage": {
            "name": "Phase 4F40 target-generation conformance probe",
            "requiresSeparateUserConfirmation": True,
        },
    }
    return _self_hashed(core, "contractSha256")


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F39Error("16 GiB artifact cap cannot reserve F39/F40 evidence")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F39Error("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def prepare_phase4f39(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    _progress("preflight", "started")
    root_cause = _root_cause_evidence(workspace)
    f21 = _f21_evidence(workspace)
    _progress("preflight", "evidence-verified")
    audit, _guidance = _guidance_evidence(workspace)
    source = fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f39_early_curriculum.py",
            "training/quoridor/root_cause_ablation.py",
            "training/quoridor/teacher_data.py",
            "training/quoridor/challenge_sampler.py",
            "training/quoridor/replay_store.py",
            "training/quoridor/classic_bridge.py",
            "scripts/classic-ai-bridge.mjs",
            "src/ai.js",
        ),
    )
    contract = build_contract(
        root_cause=root_cause,
        f21=f21,
        audit=audit,
        source_sha256=source.sha256,
    )
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f39-early-curriculum-preparation",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingStarted": False,
            "sourceSha256": source.sha256,
            "rootCauseAblation": root_cause,
            "initialization": f21,
            "seedDatasetAuditPath": AUDIT.as_posix(),
            "seedDatasetAuditSha256": audit["auditSha256"],
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "resources": _resources(workspace),
            "targetProbeAuthorized": True,
            "datasetGenerationAuthorized": False,
            "formalTrainingAuthorized": False,
            "nextStage": contract["nextStage"],
        },
        "reportSha256",
    )
    _write_immutable(workspace / AUDIT, audit, "F39 seed dataset audit")
    _write_immutable(workspace / CONTRACT, contract, "F39 frozen contract")
    _write_immutable(workspace / REPORT, report, "F39 preparation report")
    _progress(
        "preparation",
        "completed",
        reportSha256=report["reportSha256"],
        contractSha256=contract["contractSha256"],
    )
    return report


def verify_phase4f39(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F39 preparation report")
    report_sha256 = _verify_self_hash(report, "reportSha256", "F39 preparation report")
    audit = _mapping(workspace / AUDIT, "F39 seed dataset audit")
    audit_sha256 = _verify_self_hash(audit, "auditSha256", "F39 seed dataset audit")
    contract = _mapping(workspace / CONTRACT, "F39 frozen contract")
    contract_sha256 = _verify_self_hash(contract, "contractSha256", "F39 frozen contract")
    if (
        report.get("status") != "completed"
        or report.get("trainingStarted") is not False
        or report.get("formalTrainingAuthorized") is not False
        or report.get("seedDatasetAuditSha256") != audit_sha256
        or report.get("contractSha256") != contract_sha256
        or contract.get("authorizations", {}).get("targetProbeAuthorized") is not True
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
    ):
        raise Phase4F39Error("F39 preparation is not internally consistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f39-early-curriculum-verification",
        "status": "verified",
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha256,
        "contractPath": CONTRACT.as_posix(),
        "contractSha256": contract_sha256,
        "auditPath": AUDIT.as_posix(),
        "auditSha256": audit_sha256,
        "targetProbeAuthorized": True,
        "formalTrainingAuthorized": False,
        "nextStage": report["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    parser.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = (
        prepare_phase4f39(args.workspace_root)
        if args.command == "prepare"
        else verify_phase4f39(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
