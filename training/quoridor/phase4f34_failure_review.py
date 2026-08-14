"""CPU-only F33 failure review and Stage A2 protocol freeze."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import position_hash_orbit_digest
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
    MAXIMUM_ARTIFACT_LIMIT_BYTES,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f34-failure-review-v1")
ANALYSIS = ROOT / "f33-failure-analysis.json"
ISOLATION_AUDIT = ROOT / "classic-guidance-isolation-audit.json"
REPORT = ROOT / "preparation-report.json"
TRAINING_PROTOCOL = Path("training/configs/phase4f34-stage-a2-protocol-frozen-v1.json")
EVALUATION_PROTOCOL = Path("training/configs/phase4f34-evaluation-protocol-frozen-v1.json")

F33_GATE = Path("training/runs/phase4/phase4f/f31-fresh-confirmation-v2/confirmation-gate.json")
F33_RESULT = Path("training/runs/phase4/phase4f/f31-fresh-confirmation-v2/confirmation-arena/result.json")
EXPECTED_F33_GATE_SHA256 = "a65a310dc633b5772ca28a5224a48d00545c7300b65a656ab935967abe301d44"
EXPECTED_F33_RESULT_SHA256 = "a0b39c230f0a1ae4a18fd28ff6e5b9df54aea7d52fa736988867a26521cb28fb"

F21_REVIEW = Path("training/runs/phase4/phase4f/recovery-v3-preparation-v1/phase4f21-stage-a-review.json")
F21_RUN = Path("training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1")
F21_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
F22_GATE = Path("training/runs/phase4/phase4f/stage-a-screen-v2/screen-gate.json")
EXPECTED_F21_REVIEW_SHA256 = "1a97d8dd760ffbadb7581d7d055b1bb47731ad7d76f9c7491fe5dfe9488a97a6"
EXPECTED_F21_CANDIDATE_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
EXPECTED_F21_CHECKPOINT_SHA256 = "dcccfd164c7737b2e9e16592aacd4b8194e5ac7eca73ef646d796a53c978880b"
EXPECTED_F21_MODEL_STATE_SHA256 = "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"
EXPECTED_F22_GATE_SHA256 = "850b023a86d752d939a34ef0fb373e718d261f05ec0303883bd0852db3fad14b"

GUIDANCE_CONTRACT = Path("training/configs/phase4f17-classic-guidance-dataset-frozen-v1.json")
GUIDANCE_ROOT = Path("training/runs/phase4/phase4f/recovery-challenge-v2-data")
EXPECTED_GUIDANCE_CONTRACT_SHA256 = "bbf62d4a84eb032c24e64314670587e0fea0a73a3f473c908f0ca57007e4ef36"
ARTIFACT_WRITE_RESERVE_BYTES = 1280 * 1024**2


class Phase4F34Error(RuntimeError):
    """Raised when F34 evidence or a frozen preparation changes."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F34Error("workspace does not contain training/quoridor")
    return workspace


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + ARTIFACT_WRITE_RESERVE_BYTES >= MAXIMUM_ARTIFACT_LIMIT_BYTES:
        raise Phase4F34Error("12 GiB cap cannot preserve the Stage A2 write reserve")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F34Error("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
        "artifactWriteReserveBytes": ARTIFACT_WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _f33_evidence(workspace: Path) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    gate = _mapping(workspace / F33_GATE, "F33 confirmation gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F33 confirmation gate")
    result = _mapping(workspace / F33_RESULT, "F33 confirmation result")
    result_sha256 = _verify_self_hash(result, "resultSha256", "F33 confirmation result")
    statistics = gate.get("statistics")
    references = result.get("pairArtifacts")
    if (
        gate_sha256 != EXPECTED_F33_GATE_SHA256
        or result_sha256 != EXPECTED_F33_RESULT_SHA256
        or gate.get("status") != "completed"
        or gate.get("outcomeBand") != "f31-fresh-confirmation-failed"
        or gate.get("formalPromotionCandidate") is not False
        or gate.get("additionalTrainingAuthorized") is not False
        or gate.get("arenaTrajectoriesTrainingUseAllowed") is not False
        or gate.get("resultSha256") != result_sha256
        or not isinstance(statistics, Mapping)
        or statistics.get("games") != 400
        or statistics.get("candidateWins") != 181
        or statistics.get("candidateLosses") != 219
        or not isinstance(references, list)
        or len(references) != 200
    ):
        raise Phase4F34Error("F33 evidence is not the frozen failed confirmation")
    games: list[dict[str, object]] = []
    orbits: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("file"), str):
            raise Phase4F34Error("F33 pair reference is invalid")
        path = workspace / F33_RESULT.parent / str(reference["file"])
        if path.stat().st_size != reference.get("bytes") or sha256_file(path) != reference.get("sha256"):
            raise Phase4F34Error("F33 pair artifact changed")
        pair = _mapping(path, "F33 pair artifact")
        _verify_self_hash(pair, "pairSha256", "F33 pair artifact")
        raw_games = pair.get("games")
        if not isinstance(raw_games, list) or len(raw_games) != 2:
            raise Phase4F34Error("F33 pair is incomplete")
        for game in raw_games:
            if not isinstance(game, Mapping):
                raise Phase4F34Error("F33 game is invalid")
            materialized = dict(game)
            games.append(materialized)
            initial = materialized.get("initialPositionHash")
            if not isinstance(initial, str):
                raise Phase4F34Error("F33 initial position hash is invalid")
            orbits.add(position_hash_orbit_digest(initial))
            plies = materialized.get("plies")
            if not isinstance(plies, list):
                raise Phase4F34Error("F33 plies are invalid")
            for ply in plies:
                if not isinstance(ply, Mapping):
                    raise Phase4F34Error("F33 ply is invalid")
                for field in ("positionHashBefore", "positionHashAfter"):
                    value = ply.get(field)
                    if not isinstance(value, str):
                        raise Phase4F34Error("F33 ply position hash is invalid")
                    orbits.add(position_hash_orbit_digest(value))
    return gate, games, orbits


def _rate(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    wins = sum(game.get("candidateWon") is True for game in games)
    return {"games": len(games), "wins": wins, "losses": len(games) - wins, "winRate": wins / len(games)}


def _failure_analysis(gate: Mapping[str, object], games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_depth: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    by_player: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    pair_wins: Counter[int] = Counter()
    decisions = {"wins": [0, 0], "losses": [0, 0]}
    first_values: dict[str, list[float]] = {"wins": [], "losses": []}
    pairs: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for game in games:
        by_depth[int(game["openingMoveNumber"])].append(game)
        by_player[int(game["candidatePlayer"])].append(game)
        pairs[str(game["openingId"])].append(game)
        outcome = "wins" if game.get("candidateWon") is True else "losses"
        candidate_plies = [ply for ply in game["plies"] if ply.get("selectorRole") == "candidate"]
        if not candidate_plies:
            raise Phase4F34Error("F33 game has no candidate decision")
        first_values[outcome].append(float(candidate_plies[0]["selection"]["value"]))
        decisions[outcome][0] += sum(int(ply["actionIndex"]) >= 81 for ply in candidate_plies)
        decisions[outcome][1] += len(candidate_plies)
    for pair in pairs.values():
        pair_wins[sum(game.get("candidateWon") is True for game in pair)] += 1
    depth_rates = {str(depth): _rate(items) for depth, items in sorted(by_depth.items())}
    early = [game for game in games if int(game["openingMoveNumber"]) <= 12]
    middle = [game for game in games if int(game["openingMoveNumber"]) >= 16]
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f34-f33-failure-analysis",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "inputGateSha256": gate["gateSha256"],
            "inputResultSha256": gate["resultSha256"],
            "overall": dict(gate["statistics"]),
            "byCandidatePlayer": {str(player): _rate(items) for player, items in sorted(by_player.items())},
            "byOpeningDepth": depth_rates,
            "openingBands": {"early-08-12": _rate(early), "middle-16-24": _rate(middle)},
            "openingPairs": {
                "pairs": len(pairs),
                "zeroCandidateWins": pair_wins[0],
                "oneCandidateWin": pair_wins[1],
                "twoCandidateWins": pair_wins[2],
            },
            "candidateWallRate": {
                key: values[0] / values[1] for key, values in decisions.items()
            },
            "firstCandidateValue": {
                key: {
                    "mean": sum(values) / len(values),
                    "positiveRate": sum(value >= 0 for value in values) / len(values),
                    "atLeastHalfRate": sum(value >= 0.5 for value in values) / len(values),
                }
                for key, values in first_values.items()
            },
            "diagnosis": {
                "statisticallyBelowClassicHard": True,
                "orientationFailure": False,
                "primaryWeakness": "early-opening policy with excess losing-game wall use",
                "smallScreenWasFalsePositive": True,
                "failedCandidateMayNotBePromoted": True,
            },
            "dataUse": {
                "aggregateDiagnosticsAllowed": True,
                "exactF33PositionsTrainingUseAllowed": False,
                "f33TrajectoriesReplayImportAllowed": False,
            },
        },
        "analysisSha256",
    )


def _rollback(workspace: Path) -> dict[str, object]:
    review = _mapping(workspace / F21_REVIEW, "F21 Stage A review")
    review_sha256 = _verify_self_hash(review, "reviewSha256", "F21 Stage A review")
    gate = _mapping(workspace / F22_GATE, "F22 Stage A screen gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F22 Stage A screen gate")
    candidate = review.get("bestCandidate")
    if (
        review_sha256 != EXPECTED_F21_REVIEW_SHA256
        or not isinstance(candidate, Mapping)
        or candidate.get("candidateModelSha256") != EXPECTED_F21_CANDIDATE_SHA256
        or candidate.get("checkpointSha256") != EXPECTED_F21_CHECKPOINT_SHA256
        or candidate.get("modelStateSha256") != EXPECTED_F21_MODEL_STATE_SHA256
        or gate_sha256 != EXPECTED_F22_GATE_SHA256
        or gate.get("outcomeBand") != "stage-b-transition-candidate"
        or gate.get("resultSha256") != "4cf946cd840e094102ee56f096581e7ef5c315da62e542bde365bd8d1a4710f5"
    ):
        raise Phase4F34Error("F21 rollback lineage changed")
    model = workspace / F21_RUN / "checkpoints" / str(candidate["candidateModelPath"])
    checkpoint = workspace / F21_RUN / "checkpoints" / str(candidate["checkpointPath"])
    if sha256_file(model) != EXPECTED_F21_CANDIDATE_SHA256 or sha256_file(checkpoint) != EXPECTED_F21_CHECKPOINT_SHA256:
        raise Phase4F34Error("F21 rollback artifact changed")
    return {
        "selectionReason": "last classic-guided checkpoint that passed a classic-hard screen",
        "reviewPath": F21_REVIEW.as_posix(),
        "reviewSha256": review_sha256,
        "screenGatePath": F22_GATE.as_posix(),
        "screenGateSha256": gate_sha256,
        "configPath": F21_CONFIG.as_posix(),
        "configFileSha256": sha256_file(workspace / F21_CONFIG),
        "runManifestPath": (F21_RUN / "run-manifest.json").as_posix(),
        "runManifestFileSha256": sha256_file(workspace / F21_RUN / "run-manifest.json"),
        "candidateModelPath": model.relative_to(workspace).as_posix(),
        "candidateModelSha256": EXPECTED_F21_CANDIDATE_SHA256,
        "checkpointPath": checkpoint.relative_to(workspace).as_posix(),
        "checkpointSha256": EXPECTED_F21_CHECKPOINT_SHA256,
        "modelStateSha256": EXPECTED_F21_MODEL_STATE_SHA256,
        "generation": candidate["generation"],
        "step": candidate["step"],
    }


def _open_store(path: Path) -> ReplayStore:
    manifest = _mapping(path / "manifest.json", "classic-guidance replay manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F34Error("classic-guidance replay capacity is invalid")
    return ReplayStore(path, max_games=int(capacity["games"]), max_plies=int(capacity["plies"]), max_bytes=int(capacity["bytes"]))


def _guidance_audit(workspace: Path, f33_orbits: set[str]) -> dict[str, object]:
    contract = _mapping(workspace / GUIDANCE_CONTRACT, "classic-guidance contract")
    contract_sha256 = _verify_self_hash(contract, "contractSha256", "classic-guidance contract")
    if contract_sha256 != EXPECTED_GUIDANCE_CONTRACT_SHA256 or contract.get("trainingUseAllowed") is not True:
        raise Phase4F34Error("classic-guidance contract changed")
    splits: dict[str, object] = {}
    for split in ("train", "holdout"):
        path = workspace / GUIDANCE_ROOT / split
        store = _open_store(path)
        verification = store.verify(deep=False)
        orbits: set[str] = set()
        missing = 0
        depths: Counter[str] = Counter()
        for game in store.iter_games():
            for ply in game.plies:
                value = ply.metadata.get("positionHash")
                if not isinstance(value, str):
                    missing += 1
                    continue
                orbits.add(position_hash_orbit_digest(value))
                depths[str(ply.metadata.get("openingDepth"))] += 1
        overlap = orbits & f33_orbits
        if missing or overlap:
            raise Phase4F34Error("classic-guidance data overlaps F33 evidence")
        splits[split] = {
            "manifestPath": (GUIDANCE_ROOT / split / "manifest.json").as_posix(),
            "manifestSha256": sha256_file(path / "manifest.json"),
            "games": verification.game_count,
            "positions": verification.ply_count,
            "uniquePositionOrbits": len(orbits),
            "openingDepthPositions": dict(sorted(depths.items())),
            "f33PositionOrbitOverlap": 0,
        }
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f34-classic-guidance-isolation-audit",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "algorithm": "stored-position-hash-horizontal-orbit-v1",
            "guidanceContractPath": GUIDANCE_CONTRACT.as_posix(),
            "guidanceContractSha256": contract_sha256,
            "datasetFingerprint": contract["datasetFingerprint"],
            "f33ResultPath": F33_RESULT.as_posix(),
            "f33ResultSha256": EXPECTED_F33_RESULT_SHA256,
            "f33UniquePositionOrbits": len(f33_orbits),
            "splits": splits,
            "existingGuidanceReusable": True,
            "f33TrajectoriesTrainingUseAllowed": False,
        },
        "auditSha256",
    )


def _training_protocol(rollback: Mapping[str, object], audit: Mapping[str, object]) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f34-stage-a2-training-protocol",
            "status": "frozen-preparation",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingAuthorized": False,
            "pilotAuthorized": False,
            "formalPromotionAuthorized": False,
            "initialization": {
                "source": "phase4f21-stage-a-best",
                "candidateModelPath": rollback["candidateModelPath"],
                "candidateModelSha256": rollback["candidateModelSha256"],
                "checkpointSha256": rollback["checkpointSha256"],
                "modelStateSha256": rollback["modelStateSha256"],
                "optimizer": "empty",
                "selfPlayReplay": "empty",
                "stageBReplayImportAllowed": False,
            },
            "rewardContract": {
                "terminalWin": 1.0,
                "terminalLoss": -1.0,
                "drawReward": None,
                "classicHardRewardShaping": False,
                "classicHardUse": "supervised-policy-and-value-replay-only",
            },
            "classicGuidance": {
                "contractPath": GUIDANCE_CONTRACT.as_posix(),
                "contractSha256": audit["guidanceContractSha256"],
                "datasetFingerprint": audit["datasetFingerprint"],
                "baseDataReusable": True,
                "challengeReplayFraction": 0.5,
                "samplingPolicy": {
                    "algorithm": "stratified-opening-depth-and-action-kind-v1",
                    "earlyOpeningDepths": [8, 12],
                    "minimumEarlyFractionWithinChallenge": 0.6,
                    "balanceSideToMove": True,
                    "balanceTeacherActionKind": ["move", "wall"],
                },
            },
            "stageA2": {
                "network": "192x16-unchanged",
                "selfPlayGamesPerGeneration": 128,
                "selfPlaySimulations": 512,
                "trainingStepsPerGeneration": 96,
                "batchSize": 512,
                "learningRate": 0.00005,
                "minimumLearningRate": 0.000005,
                "maximumGenerationsPerMicroBlock": 2,
                "selection": "strict-minimum-fixed-guidance-holdout-total-loss-with-external-screen",
            },
            "stageB": {
                "transitionRequiresSeparateConfirmation": True,
                "resetOptimizerAndReplay": True,
                "bridgeChallengeReplayFraction": 0.15,
                "bridgeMaximumGenerations": 1,
                "pureSelfPlayChallengeReplayFraction": 0.0,
                "pureSelfPlayMaximumGenerationsPerMicroBlock": 2,
                "everyMicroBlockRequiresFreshExternalScreen": True,
            },
            "stopConditions": {
                "fixedGuidanceHoldoutRegressionAllowed": False,
                "nonFiniteLossAllowed": False,
                "illegalActionsAllowed": False,
                "artifactLimitBytes": MAXIMUM_ARTIFACT_LIMIT_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "dataPolicy": {
                "f32AndF33TrajectoriesTrainingUseAllowed": False,
                "aggregateF33DiagnosticsUseAllowed": True,
                "exactF33PositionTuningAllowed": False,
                "newGuidanceMustExcludeEveryHistoricalEvaluationOrbit": True,
            },
            "nextStage": {
                "name": "Phase 4F35 Stage A2 data/sampler engineering and CUDA-pilot preparation",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "protocolSha256",
    )


def _evaluation_protocol() -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f34-fresh-evaluation-protocol",
            "status": "frozen-preparation",
            "arenaAuthorized": False,
            "formalPromotionAuthorized": False,
            "opponent": "classic-hard-587-nodes-depth-5",
            "candidateRuntime": "native-mcts-800-simulations",
            "developmentScreen": {
                "openingPairs": 32,
                "games": 64,
                "minimumWinRateForConfirmation": 0.5,
                "mustUseNewPosition-Orbit-DisjointSuite": True,
            },
            "formalConfirmation": {
                "openingPairs": 200,
                "games": 400,
                "minimumWinRate": 0.55,
                "minimumOneSided95Lower": 0.5,
                "confidenceEquivalentTwoSided": 0.9,
                "mustBeDisjointFromDevelopmentScreen": True,
            },
            "lineageGate": {
                "opponent": "phase4f21-stage-a-best",
                "openingPairs": 32,
                "games": 64,
                "purpose": "detect improvement relative to rollback baseline",
            },
            "isolation": {
                "excludeAllFrozenSuites": True,
                "excludeAllTrainingReplayPositionOrbits": True,
                "excludeAllHistoricalArenaTrajectoriesIncludingF32AndF33": True,
                "arenaTrajectoriesTrainingUseAllowed": False,
                "screenSelectionMayNotReuseConfirmationPool": True,
            },
            "nextStageAlwaysRequiresConfirmation": True,
        },
        "protocolSha256",
    )


def prepare_phase4f34(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / REPORT).exists():
        return verify_phase4f34(workspace)
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(run.get("status") == "running" for run in ledger.runs.values()):
        raise Phase4F34Error("a running GPU ledger entry blocks F34 preparation")
    before_hours = ledger.total_hours
    resources = _resources(workspace)
    gate, games, f33_orbits = _f33_evidence(workspace)
    analysis = _failure_analysis(gate, games)
    rollback = _rollback(workspace)
    audit = _guidance_audit(workspace, f33_orbits)
    protocol = _training_protocol(rollback, audit)
    evaluation = _evaluation_protocol()
    source = fingerprint_sources(workspace).to_payload()
    _write_immutable(workspace / ANALYSIS, analysis, "F34 failure analysis")
    _write_immutable(workspace / ISOLATION_AUDIT, audit, "F34 guidance audit")
    _write_immutable(workspace / TRAINING_PROTOCOL, protocol, "F34 training protocol")
    _write_immutable(workspace / EVALUATION_PROTOCOL, evaluation, "F34 evaluation protocol")
    if GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME).total_hours != before_hours:
        raise Phase4F34Error("CPU-only F34 changed the GPU ledger")
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f34-f33-failure-review-preparation",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingAuthorized": False,
            "pilotAuthorized": False,
            "formalPromotionAuthorized": False,
            "source": source,
            "f33GateSha256": gate["gateSha256"],
            "f33ResultSha256": gate["resultSha256"],
            "analysis": {"path": ANALYSIS.as_posix(), "analysisSha256": analysis["analysisSha256"]},
            "rollback": rollback,
            "guidanceAudit": {"path": ISOLATION_AUDIT.as_posix(), "auditSha256": audit["auditSha256"]},
            "trainingProtocol": {"path": TRAINING_PROTOCOL.as_posix(), "protocolSha256": protocol["protocolSha256"]},
            "evaluationProtocol": {"path": EVALUATION_PROTOCOL.as_posix(), "protocolSha256": evaluation["protocolSha256"]},
            "globalGpuHours": before_hours,
            "resources": resources,
            "nextStage": protocol["nextStage"],
        },
        "reportSha256",
    )
    _write_immutable(workspace / REPORT, report, "F34 preparation report")
    return verify_phase4f34(workspace)


def verify_phase4f34(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F34 preparation report")
    _verify_self_hash(report, "reportSha256", "F34 preparation report")
    analysis = _mapping(workspace / ANALYSIS, "F34 failure analysis")
    audit = _mapping(workspace / ISOLATION_AUDIT, "F34 guidance audit")
    protocol = _mapping(workspace / TRAINING_PROTOCOL, "F34 training protocol")
    evaluation = _mapping(workspace / EVALUATION_PROTOCOL, "F34 evaluation protocol")
    _verify_self_hash(analysis, "analysisSha256", "F34 failure analysis")
    _verify_self_hash(audit, "auditSha256", "F34 guidance audit")
    _verify_self_hash(protocol, "protocolSha256", "F34 training protocol")
    _verify_self_hash(evaluation, "protocolSha256", "F34 evaluation protocol")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if (
        report.get("status") != "completed"
        or report.get("cpuOnly") is not True
        or report.get("cudaAccessed") is not False
        or report.get("trainingAuthorized") is not False
        or report.get("pilotAuthorized") is not False
        or report.get("formalPromotionAuthorized") is not False
        or report.get("source") != fingerprint_sources(workspace).to_payload()
        or report.get("globalGpuHours") != ledger.total_hours
        or report.get("analysis", {}).get("analysisSha256") != analysis.get("analysisSha256")
        or report.get("guidanceAudit", {}).get("auditSha256") != audit.get("auditSha256")
        or report.get("trainingProtocol", {}).get("protocolSha256") != protocol.get("protocolSha256")
        or report.get("evaluationProtocol", {}).get("protocolSha256") != evaluation.get("protocolSha256")
        or protocol.get("trainingAuthorized") is not False
        or protocol.get("pilotAuthorized") is not False
        or protocol.get("dataPolicy", {}).get("f32AndF33TrajectoriesTrainingUseAllowed") is not False
        or evaluation.get("arenaAuthorized") is not False
        or audit.get("splits", {}).get("train", {}).get("f33PositionOrbitOverlap") != 0
        or audit.get("splits", {}).get("holdout", {}).get("f33PositionOrbitOverlap") != 0
    ):
        raise Phase4F34Error("F34 frozen preparation changed")
    _resources(workspace)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    payload = prepare_phase4f34(args.workspace_root) if args.command == "prepare" else verify_phase4f34(args.workspace_root)
    print(canonical_json_bytes({
        "status": payload["status"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": payload["reportSha256"],
        "f33WinRate": _mapping(Path(args.workspace_root) / ANALYSIS, "F34 analysis")["overall"]["winRate"],
        "rollbackCandidateSha256": payload["rollback"]["candidateModelSha256"],
        "trainingAuthorized": payload["trainingAuthorized"],
        "globalGpuHours": payload["globalGpuHours"],
        "nextStage": payload["nextStage"],
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
