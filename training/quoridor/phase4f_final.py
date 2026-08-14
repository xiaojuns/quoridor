"""Phase 4F final classic-hard evaluation preparation and execution."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GpuBudgetLedger
from .champion_store import ChampionRecord, ChampionStore
from .classic_bridge import (
    CLASSIC_ENGINE,
    HARD_MAX_DEPTH,
    ClassicSearchBudget,
)
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .openings import (
    OpeningSuite,
    build_opening_suite,
    generate_opening,
    load_opening_suite,
    validate_suite_partition,
)
from .paired_arena import (
    FINAL_WALL_CLOCK_LIMIT_MS,
    ArenaGatePolicy,
    BoundMoveSelector,
    SelectorIdentity,
    canonical_classic_backend_hash,
    loaded_native_backend_hash,
)
from .phase4_arena import run_resumable_paired_arena, verify_completed_arena_result
from .phase4_budget import PHASE4E_STAGE, Phase4StageBudgetLedger
from .phase4_execution import load_phase4_execution_contract
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    RunLock,
    atomic_replace_bytes,
    canonical_json_bytes,
    fingerprint_config,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .selectors import ClassicHardSelector, MCTSSelector
from .storage import directory_size


PHASE4F_SCHEMA_VERSION = 1
PHASE4F_STAGE = "phase4f"
PHASE4F_CANDIDATE_SIMULATIONS = 800
PHASE4F_OPENING_PAIRS = 200
PHASE4F_ACTORS = 32
PHASE4F_CALIBRATION_POSITIONS = 8
PHASE4F_WALL_BUDGET_MS = 800
PHASE4F_BOOTSTRAP_SEED = 20260714
PHASE4F_BOOTSTRAP_SAMPLES = 10_000
PHASE4F_CONFIDENCE = 0.95
PHASE4F_ROOT = Path("training/runs/phase4/phase4f0")
PHASE4F_CONTRACT = Path("training/configs/phase4f-final-frozen-v1.json")
PHASE4E_CONTRACT = Path("training/configs/phase4e-execution-frozen-v2.json")
PHASE4E_MODEL_CONFIG = Path("training/configs/phase4e-bootstrap-v1.json")
DEV_OPENINGS = Path("training/fixtures/openings/dev.json")
FINAL_OPENINGS = Path("training/fixtures/openings/final.json")


class Phase4FError(RuntimeError):
    """Raised when final-evaluation evidence or resources are invalid."""


def _mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4FError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4FError(f"{label} is not canonical JSON")
    return dict(payload)


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(core)
    payload[field] = sha256_bytes(canonical_json_bytes(core))
    return payload


def _verify_self_hash(
    payload: Mapping[str, object], *, field: str, label: str
) -> str:
    try:
        digest = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise Phase4FError(f"{label} has an invalid self hash") from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4FError(f"{label} self hash mismatch")
    return digest


def _write_immutable(path: Path, payload: Mapping[str, object], label: str) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Phase4FError(f"existing {label} differs")
        return
    atomic_replace_bytes(path, encoded)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise Phase4FError(f"artifact is outside workspace: {path}") from error


def _effective_config_sha256(path: Path, workspace: Path) -> str:
    config = load_experiment_config(path)
    if not config.run_dir.is_absolute():
        config = replace(config, run_dir=(workspace / config.run_dir).resolve())
    return fingerprint_config(config).sha256


def _network_config(path: Path) -> NetworkConfig:
    settings = load_experiment_config(path).network
    return NetworkConfig(
        input_channels=settings.input_channels,
        channels=settings.channels,
        blocks=settings.residual_blocks,
        se_reduction=settings.se_reduction,
        policy_channels=settings.policy_channels,
        value_channels=settings.value_channels,
        value_hidden=settings.value_hidden,
    )


def _standard_initial_suite() -> OpeningSuite:
    return build_opening_suite((generate_opening((), suite="final"),), suite="final")


def _champion_payload(record: ChampionRecord, manifest_path: Path) -> dict[str, object]:
    if not record.retained or record.file_name is None:
        raise Phase4FError("latest champion model is not retained")
    if record.metadata.get("formalPromotion") is not True:
        raise Phase4FError("latest champion is not a formal promotion")
    return {
        "id": record.champion_id,
        "generation": record.generation,
        "modelPath": f"champions/{record.file_name}",
        "modelSha256": record.model_sha256,
        "modelStateSha256": record.metadata.get("candidateModelStateSha256"),
        "configSha256": record.config_sha256,
        "trainingSourceSha256": record.source_sha256,
        "manifestPath": "champions/manifest.json",
        "manifestFileSha256": sha256_file(manifest_path),
    }


def _calibration_core(
    *,
    workspace: Path,
    dev: OpeningSuite,
    evaluation_source_sha256: str,
    classic_backend_sha256: str,
    selector_factory: Callable[[], ClassicHardSelector],
    positions: int,
) -> dict[str, object]:
    if type(positions) is not int or not 1 <= positions <= len(dev.openings):
        raise ValueError("positions must select a non-empty dev prefix")
    measurements: list[dict[str, object]] = []
    selector = selector_factory()
    try:
        for opening in dev.openings[:positions]:
            result = selector.choose(opening.state)
            if result.nodes <= 0:
                raise Phase4FError("classic calibration returned no searched nodes")
            measurements.append(
                {
                    "openingId": opening.id,
                    "openingHash": opening.content_hash,
                    "actionIndex": result.action_index,
                    "nodes": result.nodes,
                    "elapsedMs": result.elapsed_ms,
                    "depth": result.depth,
                }
            )
    finally:
        selector.close()
    nodes = [int(item["nodes"]) for item in measurements]
    elapsed = [float(item["elapsedMs"]) for item in measurements]
    node_budget = int(statistics.median_low(nodes))
    return {
        "schemaVersion": PHASE4F_SCHEMA_VERSION,
        "kind": "phase4f-classic-node-calibration",
        "status": "completed",
        "evaluationSourceSha256": evaluation_source_sha256,
        "classicBackendSha256": classic_backend_sha256,
        "suite": "dev",
        "suiteHash": dev.content_hash,
        "timeBudgetMs": FINAL_WALL_CLOCK_LIMIT_MS,
        "maxDepth": HARD_MAX_DEPTH,
        "positions": positions,
        "selectionPolicy": "ordered-dev-prefix-v1",
        "nodeBudgetPolicy": "median-low-positive-nodes-v1",
        "nodeBudget": node_budget,
        "nodeCounts": {
            "minimum": min(nodes),
            "medianLow": node_budget,
            "maximum": max(nodes),
        },
        "elapsedMs": {
            "minimum": min(elapsed),
            "median": statistics.median(elapsed),
            "maximum": max(elapsed),
        },
        "measurements": measurements,
    }


def _track_payload(kind: str, opponent_limit: int) -> dict[str, object]:
    if kind not in {"fixed", "wall-clock"}:
        raise ValueError("invalid Phase 4F track")
    return {
        "track": kind,
        "suite": "final",
        "openingPairs": PHASE4F_OPENING_PAIRS,
        "games": PHASE4F_OPENING_PAIRS * 2,
        "actors": PHASE4F_ACTORS,
        "candidateSimulations": PHASE4F_CANDIDATE_SIMULATIONS,
        "opponentBudget": {
            "kind": "nodes" if kind == "fixed" else "time",
            "limit": opponent_limit,
            "maxDepth": HARD_MAX_DEPTH,
        },
        "bootstrap": {
            "seed": PHASE4F_BOOTSTRAP_SEED,
            "samples": PHASE4F_BOOTSTRAP_SAMPLES,
            "confidence": PHASE4F_CONFIDENCE,
        },
        "thresholds": {
            "winRateMinimum": 0.60,
            "twoSided95LowerMinimum": 0.55,
            "illegalActions": 0,
            "draws": 0,
        },
    }


def prepare_phase4f(
    workspace_root: str | Path,
    *,
    contract_path: str | Path = PHASE4F_CONTRACT,
    output_root: str | Path = PHASE4F_ROOT,
    calibration_positions: int = PHASE4F_CALIBRATION_POSITIONS,
    selector_factory: Callable[[], ClassicHardSelector] | None = None,
) -> dict[str, object]:
    """Calibrate on dev positions and freeze the CPU-only final gate contract."""

    workspace = Path(workspace_root).resolve()
    artifact_root = workspace / "training/runs/phase4"
    output = (workspace / output_root).resolve()
    contract_file = (workspace / contract_path).resolve()
    evidence_path = output / "classic-node-calibration.json"
    report_path = output / "report.json"
    if directory_size(artifact_root) >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4FError("Phase 4 artifact limit is already reached")
    if shutil.disk_usage(artifact_root).free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4FError("less than 20 GiB free disk remains")

    store = ChampionStore(artifact_root / "champions", artifact_root=artifact_root)
    store.verify(deep=True)
    champion = store.resolve(role="latest")
    champion_data = _champion_payload(champion, store.manifest_path)
    dev = load_opening_suite(workspace / DEV_OPENINGS, expected_suite="dev")
    final = load_opening_suite(workspace / FINAL_OPENINGS, expected_suite="final")
    validate_suite_partition(dev, final)
    if len(dev.openings) != 200 or len(final.openings) != PHASE4F_OPENING_PAIRS:
        raise Phase4FError("Phase 4F requires exactly 200 dev and final openings")
    source = fingerprint_sources(workspace)
    classic_hash = canonical_classic_backend_hash()
    native_hash = loaded_native_backend_hash()
    model_config = workspace / PHASE4E_MODEL_CONFIG
    if _effective_config_sha256(model_config, workspace) != champion.config_sha256:
        raise Phase4FError("champion model config no longer matches its provenance")

    if evidence_path.exists():
        evidence = _mapping(evidence_path, "Phase 4F calibration evidence")
        _verify_self_hash(
            evidence,
            field="calibrationSha256",
            label="Phase 4F calibration evidence",
        )
    else:
        factory = selector_factory or (
            lambda: ClassicHardSelector(
                budget=ClassicSearchBudget.time(FINAL_WALL_CLOCK_LIMIT_MS),
                max_depth=HARD_MAX_DEPTH,
                request_timeout=5.0,
            )
        )
        evidence = _self_hashed(
            _calibration_core(
                workspace=workspace,
                dev=dev,
                evaluation_source_sha256=source.sha256,
                classic_backend_sha256=classic_hash,
                selector_factory=factory,
                positions=calibration_positions,
            ),
            "calibrationSha256",
        )
        _write_immutable(evidence_path, evidence, "Phase 4F calibration evidence")
    if (
        evidence.get("evaluationSourceSha256") != source.sha256
        or evidence.get("classicBackendSha256") != classic_hash
        or evidence.get("suiteHash") != dev.content_hash
        or evidence.get("positions") != calibration_positions
    ):
        raise Phase4FError("Phase 4F calibration evidence is stale")
    node_budget = evidence.get("nodeBudget")
    if type(node_budget) is not int or node_budget <= 0:
        raise Phase4FError("Phase 4F calibrated node budget is invalid")

    tracks = {
        "fixed": _track_payload("fixed", node_budget),
        "wallClock": _track_payload("wall-clock", PHASE4F_WALL_BUDGET_MS),
    }
    workload_sha256 = sha256_bytes(canonical_json_bytes(tracks))
    standard = _standard_initial_suite()
    phase4e_contract = workspace / PHASE4E_CONTRACT
    contract_core = {
        "schemaVersion": PHASE4F_SCHEMA_VERSION,
        "kind": "phase4f-final-evaluation-contract",
        "stage": PHASE4F_STAGE,
        "status": "frozen",
        "purpose": "locked-final-classic-hard-evaluation",
        "productReleaseAuthorized": False,
        "candidate": champion_data,
        "evaluationSourceSha256": source.sha256,
        "nativeBackendSha256": native_hash,
        "classicBackendSha256": classic_hash,
        "modelConfig": {
            "path": PHASE4E_MODEL_CONFIG.as_posix(),
            "fileSha256": sha256_file(model_config),
            "effectiveSha256": champion.config_sha256,
        },
        "phase4eExecutionContract": {
            "path": PHASE4E_CONTRACT.as_posix(),
            "fileSha256": sha256_file(phase4e_contract),
        },
        "openingSuites": {
            "dev": {
                "path": DEV_OPENINGS.as_posix(),
                "fileSha256": sha256_file(workspace / DEV_OPENINGS),
                "suiteHash": dev.content_hash,
            },
            "final": {
                "path": FINAL_OPENINGS.as_posix(),
                "fileSha256": sha256_file(workspace / FINAL_OPENINGS),
                "suiteHash": final.content_hash,
            },
            "standardInitialSuiteHash": standard.content_hash,
        },
        "calibration": {
            "path": _relative(evidence_path, workspace),
            "fileSha256": sha256_file(evidence_path),
            "calibrationSha256": evidence["calibrationSha256"],
            "positions": calibration_positions,
            "nodeBudget": node_budget,
        },
        "workloadSha256": workload_sha256,
        "tracks": tracks,
        "resources": {
            "globalGpuHourLimit": GPU_HOUR_BUDGET,
            "artifactLimitBytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
            "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            "gpuStage": PHASE4E_STAGE,
        },
        "postEvaluation": {
            "standardInitialGames": 2,
            "tacticalGateRequired": True,
            "tacticalFixtureStatus": "not-prepared",
            "setBestPointer": False,
            "setProductPointer": False,
        },
    }
    contract = _self_hashed(contract_core, "contractSha256")
    _write_immutable(contract_file, contract, "Phase 4F contract")
    report_core = {
        "schemaVersion": PHASE4F_SCHEMA_VERSION,
        "kind": "phase4f0-preparation-report",
        "status": "completed",
        "decisionEligible": True,
        "contractPath": _relative(contract_file, workspace),
        "contractFileSha256": sha256_file(contract_file),
        "contractSha256": contract["contractSha256"],
        "calibrationPath": _relative(evidence_path, workspace),
        "calibrationFileSha256": sha256_file(evidence_path),
        "championId": champion.champion_id,
        "generation": champion.generation,
        "nodeBudget": node_budget,
        "evaluationSourceSha256": source.sha256,
        "nativeBackendSha256": native_hash,
        "classicBackendSha256": classic_hash,
        "finalSuiteHash": final.content_hash,
        "tracks": ["fixed", "wall-clock"],
        "cudaAccessed": False,
        "lockedFinalGamesRun": 0,
        "productReleaseAuthorized": False,
    }
    report = _self_hashed(report_core, "reportSha256")
    _write_immutable(report_path, report, "Phase 4F preparation report")
    return verify_phase4f_preparation(workspace, contract_path=contract_file)


def load_phase4f_contract(path: str | Path) -> dict[str, object]:
    payload = _mapping(Path(path), "Phase 4F contract")
    _verify_self_hash(payload, field="contractSha256", label="Phase 4F contract")
    expected = {
        "schemaVersion",
        "kind",
        "stage",
        "status",
        "purpose",
        "productReleaseAuthorized",
        "candidate",
        "evaluationSourceSha256",
        "nativeBackendSha256",
        "classicBackendSha256",
        "modelConfig",
        "phase4eExecutionContract",
        "openingSuites",
        "calibration",
        "workloadSha256",
        "tracks",
        "resources",
        "postEvaluation",
        "contractSha256",
    }
    if (
        set(payload) != expected
        or payload.get("schemaVersion") != PHASE4F_SCHEMA_VERSION
        or payload.get("kind") != "phase4f-final-evaluation-contract"
        or payload.get("stage") != PHASE4F_STAGE
        or payload.get("status") != "frozen"
        or payload.get("productReleaseAuthorized") is not False
    ):
        raise Phase4FError("Phase 4F contract schema or status is invalid")
    return payload


def verify_phase4f_preparation(
    workspace_root: str | Path,
    *,
    contract_path: str | Path = PHASE4F_CONTRACT,
) -> dict[str, object]:
    workspace = Path(workspace_root).resolve()
    artifact_root = workspace / "training/runs/phase4"
    path = Path(contract_path)
    if not path.is_absolute():
        path = workspace / path
    contract = load_phase4f_contract(path)
    source_sha256 = fingerprint_sources(workspace).sha256
    if contract.get("evaluationSourceSha256") != source_sha256:
        raise Phase4FError("current evaluation source differs from Phase 4F contract")
    if contract.get("nativeBackendSha256") != loaded_native_backend_hash():
        raise Phase4FError("native backend differs from Phase 4F contract")
    if contract.get("classicBackendSha256") != canonical_classic_backend_hash():
        raise Phase4FError("classic backend differs from Phase 4F contract")

    store = ChampionStore(artifact_root / "champions", artifact_root=artifact_root)
    store.verify(deep=True)
    latest = store.resolve(role="latest")
    expected_champion = _champion_payload(latest, store.manifest_path)
    if contract.get("candidate") != expected_champion:
        raise Phase4FError("Phase 4F candidate is no longer the latest champion")
    model_config = contract.get("modelConfig")
    execution = contract.get("phase4eExecutionContract")
    suites = contract.get("openingSuites")
    calibration_ref = contract.get("calibration")
    tracks = contract.get("tracks")
    if not all(
        isinstance(value, Mapping)
        for value in (model_config, execution, suites, calibration_ref, tracks)
    ):
        raise Phase4FError("Phase 4F contract references are invalid")
    model_path = workspace / str(model_config["path"])
    execution_path = workspace / str(execution["path"])
    if (
        sha256_file(model_path) != model_config.get("fileSha256")
        or _effective_config_sha256(model_path, workspace)
        != model_config.get("effectiveSha256")
        or sha256_file(execution_path) != execution.get("fileSha256")
    ):
        raise Phase4FError("Phase 4F config or execution provenance changed")
    phase4e = load_phase4_execution_contract(execution_path)
    if phase4e.stage != PHASE4E_STAGE or not phase4e.formal_promotion:
        raise Phase4FError("Phase 4F requires the frozen formal Phase 4E contract")
    dev = load_opening_suite(workspace / DEV_OPENINGS, expected_suite="dev")
    final = load_opening_suite(workspace / FINAL_OPENINGS, expected_suite="final")
    validate_suite_partition(dev, final)
    for name, suite, relative in (
        ("dev", dev, DEV_OPENINGS),
        ("final", final, FINAL_OPENINGS),
    ):
        reference = suites.get(name)
        if not isinstance(reference, Mapping) or reference != {
            "path": relative.as_posix(),
            "fileSha256": sha256_file(workspace / relative),
            "suiteHash": suite.content_hash,
        }:
            raise Phase4FError(f"Phase 4F {name} suite provenance changed")
    if suites.get("standardInitialSuiteHash") != _standard_initial_suite().content_hash:
        raise Phase4FError("Phase 4F standard initial suite changed")
    evidence_path = workspace / str(calibration_ref["path"])
    evidence = _mapping(evidence_path, "Phase 4F calibration evidence")
    evidence_sha = _verify_self_hash(
        evidence,
        field="calibrationSha256",
        label="Phase 4F calibration evidence",
    )
    if (
        sha256_file(evidence_path) != calibration_ref.get("fileSha256")
        or evidence_sha != calibration_ref.get("calibrationSha256")
        or evidence.get("evaluationSourceSha256") != source_sha256
        or evidence.get("classicBackendSha256") != contract.get("classicBackendSha256")
        or evidence.get("suiteHash") != dev.content_hash
        or evidence.get("nodeBudget") != calibration_ref.get("nodeBudget")
    ):
        raise Phase4FError("Phase 4F calibration evidence changed")
    if sha256_bytes(canonical_json_bytes(tracks)) != contract.get("workloadSha256"):
        raise Phase4FError("Phase 4F workload hash mismatch")
    return {
        "status": "completed",
        "decisionEligible": True,
        "contractPath": _relative(path, workspace),
        "contractFileSha256": sha256_file(path),
        "contractSha256": contract["contractSha256"],
        "championId": latest.champion_id,
        "generation": latest.generation,
        "nodeBudget": evidence["nodeBudget"],
        "evaluationSourceSha256": source_sha256,
        "nativeBackendSha256": contract["nativeBackendSha256"],
        "classicBackendSha256": contract["classicBackendSha256"],
        "finalSuiteHash": final.content_hash,
        "cudaAccessed": False,
        "lockedFinalGamesRun": 0,
        "productReleaseAuthorized": False,
    }


def _track(contract: Mapping[str, object], name: str) -> Mapping[str, object]:
    key = "fixed" if name == "fixed" else "wallClock"
    tracks = contract.get("tracks")
    if name not in {"fixed", "wall-clock"} or not isinstance(tracks, Mapping):
        raise Phase4FError("invalid Phase 4F track")
    value = tracks.get(key)
    if not isinstance(value, Mapping) or value.get("track") != name:
        raise Phase4FError("Phase 4F track contract is invalid")
    return value


def _arena_contract_sha256(
    contract: Mapping[str, object], track: str, suite_hash: str
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "phase4fContractSha256": contract["contractSha256"],
                "track": track,
                "suiteHash": suite_hash,
                "workloadSha256": contract["workloadSha256"],
            }
        )
    )


def _identities(
    contract: Mapping[str, object], track: Mapping[str, object]
) -> tuple[SelectorIdentity, SelectorIdentity]:
    candidate = contract["candidate"]
    if not isinstance(candidate, Mapping):
        raise Phase4FError("Phase 4F candidate contract is invalid")
    workload_sha = str(contract["workloadSha256"])
    return (
        SelectorIdentity(
            selector_id=f"phase4f-champion-{candidate['id']}",
            artifact_hash=str(candidate["modelSha256"]),
            config_hash=str(candidate["configSha256"]),
            source_hash=str(candidate["trainingSourceSha256"]),
            engine="native-mcts",
        ),
        SelectorIdentity(
            selector_id=f"classic-hard-{track['track']}",
            artifact_hash=str(contract["classicBackendSha256"]),
            config_hash=workload_sha,
            source_hash=str(contract["evaluationSourceSha256"]),
            engine=CLASSIC_ENGINE,
        ),
    )


def _gate_from_result(
    result_path: Path,
    *,
    standard_result_path: Path,
    contract: Mapping[str, object],
    track_name: str,
) -> dict[str, object]:
    result = verify_completed_arena_result(result_path)
    standard_result = verify_completed_arena_result(standard_result_path)
    track = _track(contract, track_name)
    candidate_identity, opponent_identity = _identities(contract, track)
    budget = track["opponentBudget"]
    if not isinstance(budget, Mapping):
        raise Phase4FError("Phase 4F opponent budget is invalid")
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object, expected: object) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "actual": actual, "expected": expected}
        )

    statistics_payload = result.get("statistics")
    interval = (
        statistics_payload.get("winRateCI")
        if isinstance(statistics_payload, Mapping)
        else None
    )
    if not isinstance(statistics_payload, Mapping) or not isinstance(interval, Mapping):
        raise Phase4FError("Phase 4F arena statistics are invalid")
    check("suite", result.get("suite") == "final", result.get("suite"), "final")
    check(
        "openingPairs",
        result.get("openingPairs") == PHASE4F_OPENING_PAIRS,
        result.get("openingPairs"),
        PHASE4F_OPENING_PAIRS,
    )
    check("games", result.get("games") == 400, result.get("games"), 400)
    opening_suites = contract.get("openingSuites")
    expected_standard_hash = (
        opening_suites.get("standardInitialSuiteHash")
        if isinstance(opening_suites, Mapping)
        else None
    )
    check(
        "standardInitialGames",
        standard_result.get("openingPairs") == 1
        and standard_result.get("games") == 2
        and standard_result.get("suiteHash") == expected_standard_hash,
        {
            "openingPairs": standard_result.get("openingPairs"),
            "games": standard_result.get("games"),
            "suiteHash": standard_result.get("suiteHash"),
        },
        {"openingPairs": 1, "games": 2, "suiteHash": expected_standard_hash},
    )
    check(
        "candidateIdentity",
        result.get("candidateIdentity") == candidate_identity.to_dict(),
        result.get("candidateIdentity"),
        candidate_identity.to_dict(),
    )
    check(
        "opponentIdentity",
        result.get("opponentIdentity") == opponent_identity.to_dict(),
        result.get("opponentIdentity"),
        opponent_identity.to_dict(),
    )
    check(
        "standardInitialIdentities",
        standard_result.get("candidateIdentity") == candidate_identity.to_dict()
        and standard_result.get("opponentIdentity") == opponent_identity.to_dict(),
        {
            "candidate": standard_result.get("candidateIdentity"),
            "opponent": standard_result.get("opponentIdentity"),
        },
        "same frozen candidate and opponent identities",
    )
    candidate_runtime = result.get("candidateRuntime")
    opponent_runtime = result.get("opponentRuntime")
    check(
        "candidateRuntime",
        isinstance(candidate_runtime, Mapping)
        and candidate_runtime.get("kind") == "native-mcts"
        and candidate_runtime.get("trusted") is True
        and candidate_runtime.get("configuredSimulations")
        == PHASE4F_CANDIDATE_SIMULATIONS
        and candidate_runtime.get("backendHash")
        == contract.get("nativeBackendSha256"),
        candidate_runtime,
        f"trusted native-mcts:{PHASE4F_CANDIDATE_SIMULATIONS}",
    )
    check(
        "opponentRuntime",
        isinstance(opponent_runtime, Mapping)
        and opponent_runtime.get("kind") == "classic-hard"
        and opponent_runtime.get("trusted") is True
        and opponent_runtime.get("budgetKind") == budget.get("kind")
        and opponent_runtime.get("budgetLimit") == budget.get("limit")
        and opponent_runtime.get("maxDepth") == HARD_MAX_DEPTH
        and opponent_runtime.get("backendHash")
        == contract.get("classicBackendSha256"),
        opponent_runtime,
        "trusted classic-hard with frozen budget",
    )
    check(
        "winRate",
        float(statistics_payload.get("winRate", -1.0)) >= 0.60,
        statistics_payload.get("winRate"),
        ">= 0.60",
    )
    check(
        "twoSided95Lower",
        float(interval.get("lower", -1.0)) >= 0.55
        and interval.get("confidence") == PHASE4F_CONFIDENCE,
        interval,
        {"lower": ">= 0.55", "confidence": PHASE4F_CONFIDENCE},
    )
    candidate_simulations: list[int] = []
    opponent_simulations: list[int] = []
    opponent_nodes: list[int] = []
    opponent_budgets: list[tuple[object, object]] = []
    opponent_wall_ms: list[float] = []
    for reference in result["pairArtifacts"]:
        pair = _mapping(result_path.parent / str(reference["file"]), "arena pair")
        for game in pair["games"]:
            for ply in game["plies"]:
                selection = ply["selection"]
                if ply["selectorRole"] == "candidate":
                    candidate_simulations.append(int(selection["simulations"]))
                else:
                    opponent_simulations.append(int(selection["simulations"]))
                    opponent_nodes.append(int(selection["nodes"]))
                    opponent_wall_ms.append(float(selection["wallTimeMs"]))
                    metadata = selection["metadata"]
                    opponent_budgets.append(
                        (metadata.get("budgetKind"), metadata.get("budgetLimit"))
                    )
    check(
        "candidateSimulationsPerMove",
        bool(candidate_simulations)
        and set(candidate_simulations) == {PHASE4F_CANDIDATE_SIMULATIONS},
        sorted(set(candidate_simulations)),
        [PHASE4F_CANDIDATE_SIMULATIONS],
    )
    check(
        "opponentSimulationsPerMove",
        bool(opponent_simulations) and set(opponent_simulations) == {0},
        sorted(set(opponent_simulations)),
        [0],
    )
    expected_budget = (budget.get("kind"), budget.get("limit"))
    check(
        "opponentBudget",
        bool(opponent_budgets)
        and all(value == expected_budget for value in opponent_budgets),
        sorted(set(opponent_budgets)),
        [expected_budget],
    )
    if track_name == "fixed":
        limit = int(budget["limit"])
        check(
            "opponentNodeCount",
            bool(opponent_nodes) and all(0 < value <= limit for value in opponent_nodes),
            {"minimum": min(opponent_nodes, default=0), "maximum": max(opponent_nodes, default=0)},
            f"every move in [1, {limit}]",
        )
    else:
        check(
            "maximumOpponentMoveMs",
            bool(opponent_wall_ms)
            and max(opponent_wall_ms) <= FINAL_WALL_CLOCK_LIMIT_MS,
            max(opponent_wall_ms, default=0.0),
            f"<= {FINAL_WALL_CLOCK_LIMIT_MS}",
        )
    core = {
        "schemaVersion": PHASE4F_SCHEMA_VERSION,
        "kind": "phase4f-final-hard-gate",
        "status": "completed",
        "track": track_name,
        "passed": all(item["passed"] for item in checks),
        "contractSha256": contract["contractSha256"],
        "arenaContractSha256": result["contractSha256"],
        "arenaResultPath": result_path.as_posix(),
        "arenaResultSha256": result["resultSha256"],
        "standardInitial": {
            "resultPath": standard_result_path.as_posix(),
            "resultSha256": standard_result["resultSha256"],
            "statistics": standard_result["statistics"],
        },
        "statistics": dict(statistics_payload),
        "checks": checks,
        "productReleaseAuthorized": False,
    }
    return _self_hashed(core, "gateSha256")


def _model_selector(
    *,
    workspace: Path,
    artifact_root: Path,
    contract: Mapping[str, object],
    device: torch.device,
    identity: SelectorIdentity,
) -> BoundMoveSelector:
    candidate = contract["candidate"]
    model_config = contract["modelConfig"]
    if not isinstance(candidate, Mapping) or not isinstance(model_config, Mapping):
        raise Phase4FError("Phase 4F model references are invalid")
    store = ChampionStore(artifact_root / "champions", artifact_root=artifact_root)
    state = store.load_state_dict(str(candidate["id"]), map_location="cpu")
    model = PolicyValueNetwork(_network_config(workspace / str(model_config["path"])))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Phase4FError("champion state does not match the frozen network") from error
    model = model.to(device).eval()
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
    evaluator.enable_native_encoder(load_native_module())
    config = load_experiment_config(workspace / str(model_config["path"]))
    search = MCTSConfig(
        simulations=PHASE4F_CANDIDATE_SIMULATIONS,
        c_base=config.mcts.c_base,
        c_init=config.mcts.c_init,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=0.0,
        add_root_noise=False,
        seed=PHASE4F_BOOTSTRAP_SEED,
    )
    return BoundMoveSelector(
        MCTSSelector(
            evaluator,
            search,
            search_factory=NativeMCTS,
            engine="native-mcts",
        ),
        identity,
    )


def run_phase4f_track(
    workspace_root: str | Path,
    *,
    track_name: str,
    contract_path: str | Path = PHASE4F_CONTRACT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Run or resume one locked-final track and emit a deeply verified gate."""

    workspace = Path(workspace_root).resolve()
    artifact_root = workspace / "training/runs/phase4"
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = workspace / contract_file
    verify_phase4f_preparation(workspace, contract_path=contract_file)
    contract = load_phase4f_contract(contract_file)
    track = _track(contract, track_name)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4FError("Phase 4F final arena requires CUDA")
    if directory_size(artifact_root) >= DEFAULT_ARTIFACT_LIMIT_BYTES:
        raise Phase4FError("Phase 4 artifact limit is already reached")
    if shutil.disk_usage(artifact_root).free < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4FError("less than 20 GiB free disk remains")
    final = load_opening_suite(workspace / FINAL_OPENINGS, expected_suite="final")
    arena_contract = _arena_contract_sha256(contract, track_name, final.content_hash)
    result_root = artifact_root / "phase4f" / track_name
    standard_root = artifact_root / "phase4f" / f"{track_name}-standard-initial"
    gate_path = result_root / "gate.json"
    if gate_path.exists():
        gate = _mapping(gate_path, "Phase 4F gate")
        _verify_self_hash(gate, field="gateSha256", label="Phase 4F gate")
        recomputed = _gate_from_result(
            result_root / "result.json",
            standard_result_path=standard_root / "result.json",
            contract=contract,
            track_name=track_name,
        )
        # JSON round-tripping turns telemetry tuples into lists.  Compare the
        # canonical evidence bytes so reopening a valid completed gate remains
        # idempotent without weakening any value or type validation upstream.
        if canonical_json_bytes(recomputed) != canonical_json_bytes(gate):
            raise Phase4FError("stored Phase 4F gate differs from verified evidence")
        return gate

    candidate_identity, opponent_identity = _identities(contract, track)
    budget = track["opponentBudget"]
    if not isinstance(budget, Mapping):
        raise Phase4FError("Phase 4F opponent budget is invalid")
    ledger = GpuBudgetLedger(artifact_root / "gpu-budget-ledger.json")
    execution_ref = contract["phase4eExecutionContract"]
    if not isinstance(execution_ref, Mapping):
        raise Phase4FError("Phase 4F execution reference is invalid")
    execution = load_phase4_execution_contract(workspace / str(execution_ref["path"]))
    stage_budget = execution.require_frozen_budget()
    stage_ledger = Phase4StageBudgetLedger(artifact_root / "stage-budget-ledger.json")
    stage_ledger.ensure_stage(
        PHASE4E_STAGE,
        float(stage_budget.baseline_global_hours),
        float(stage_budget.limit_hours),
        baseline_run_id=str(stage_budget.baseline_run_id),
        baseline_report_sha256=str(stage_budget.baseline_report_sha256),
    )
    stage_ledger.assert_available(PHASE4E_STAGE, ledger.total_hours)
    if ledger.total_hours >= GPU_HOUR_BUDGET:
        raise Phase4FError("72 GPU-hour global budget is exhausted")

    run_id = f"phase4f-{track_name}-{str(contract['contractSha256'])[:12]}-{time.time_ns()}"
    started = time.monotonic()
    last_ledger = float("-inf")
    last_progress = float("-inf")
    candidate_selector: BoundMoveSelector | None = None
    opponent_selector: BoundMoveSelector | None = None

    def heartbeat(moves: int, pairs: int) -> None:
        nonlocal last_ledger, last_progress
        now = time.monotonic()
        if now - last_ledger >= 5.0:
            total = ledger.update(run_id, elapsed_seconds=now - started, status="running")
            stage_ledger.assert_available(PHASE4E_STAGE, total)
            last_ledger = now
        if now - last_progress >= 30.0:
            print(
                json.dumps(
                    {
                        "kind": "phase4f-progress",
                        "track": track_name,
                        "status": "running",
                        "openingPairs": pairs,
                        "openingPairsTotal": PHASE4F_OPENING_PAIRS,
                        "moves": moves,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            last_progress = now

    try:
        with RunLock(artifact_root / ".phase4.lock", run_id=run_id):
            ledger.begin(run_id, external_hours=0.0)
            candidate_selector = _model_selector(
                workspace=workspace,
                artifact_root=artifact_root,
                contract=contract,
                device=device,
                identity=candidate_identity,
            )
            classic = ClassicHardSelector(
                budget=ClassicSearchBudget(
                    str(budget["kind"]), int(budget["limit"])
                ),
                max_depth=HARD_MAX_DEPTH,
                request_timeout=max(5.0, int(budget["limit"]) / 1_000.0 + 2.0),
            )
            opponent_selector = BoundMoveSelector(classic, opponent_identity)
            arena = run_resumable_paired_arena(
                final,
                candidate_selector,
                opponent_selector,
                store_root=result_root,
                contract_sha256=arena_contract,
                actors=int(track["actors"]),
                heartbeat=heartbeat,
            )
            arena.finalize(
                bootstrap_seed=PHASE4F_BOOTSTRAP_SEED,
                bootstrap_samples=PHASE4F_BOOTSTRAP_SAMPLES,
                confidence=PHASE4F_CONFIDENCE,
            )
            standard = _standard_initial_suite()
            standard_arena = run_resumable_paired_arena(
                standard,
                candidate_selector,
                opponent_selector,
                store_root=standard_root,
                contract_sha256=_arena_contract_sha256(
                    contract,
                    f"{track_name}-standard-initial",
                    standard.content_hash,
                ),
                actors=2,
                heartbeat=heartbeat,
            )
            standard_arena.finalize(
                bootstrap_seed=PHASE4F_BOOTSTRAP_SEED,
                bootstrap_samples=PHASE4F_BOOTSTRAP_SAMPLES,
                confidence=PHASE4F_CONFIDENCE,
            )
            gate = _gate_from_result(
                result_root / "result.json",
                standard_result_path=standard_root / "result.json",
                contract=contract,
                track_name=track_name,
            )
            _write_immutable(gate_path, gate, "Phase 4F gate")
            total = ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="completed"
            )
            stage = stage_ledger.assert_available(PHASE4E_STAGE, total)
            return {
                "status": "completed",
                "track": track_name,
                "passed": gate["passed"],
                "gatePath": _relative(gate_path, artifact_root),
                "gateSha256": gate["gateSha256"],
                "statistics": gate["statistics"],
                "globalGpuHours": total,
                "stageGpu": stage,
                "productReleaseAuthorized": False,
            }
    except BaseException:
        try:
            ledger.update(
                run_id, elapsed_seconds=time.monotonic() - started, status="interrupted"
            )
        except Exception:
            pass
        raise
    finally:
        if candidate_selector is not None:
            candidate_selector.close()
        if opponent_selector is not None:
            opponent_selector.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workspace-root", type=Path, default=Path("."))
    prepare.add_argument("--contract", type=Path, default=PHASE4F_CONTRACT)
    prepare.add_argument("--output-root", type=Path, default=PHASE4F_ROOT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--workspace-root", type=Path, default=Path("."))
    verify.add_argument("--contract", type=Path, default=PHASE4F_CONTRACT)
    run = subparsers.add_parser("run")
    run.add_argument("--workspace-root", type=Path, default=Path("."))
    run.add_argument("--contract", type=Path, default=PHASE4F_CONTRACT)
    run.add_argument("--track", choices=("fixed", "wall-clock"), required=True)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_phase4f(
            args.workspace_root,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    elif args.command == "verify":
        payload = verify_phase4f_preparation(
            args.workspace_root,
            contract_path=args.contract,
        )
    else:
        payload = run_phase4f_track(
            args.workspace_root,
            track_name=args.track,
            contract_path=args.contract,
            device_name=args.device,
        )
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
