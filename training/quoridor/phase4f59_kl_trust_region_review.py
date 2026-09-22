"""Review F58 and freeze an isolated F58-versus-F21 lineage screen."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import arena_position_orbits
from .openings import (
    OpeningSuite,
    build_opening_suite,
    load_opening_suite,
    opening_position_orbit_digest,
    opening_suite_to_payload,
    validate_suite_collection,
)
from .paired_arena import loaded_native_backend_hash
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import (
    FROZEN_SUITES,
    REPLAY_ROOTS,
    _load_generator,
    _mapping,
    _replay_orbits,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f55_policy_head_pilot_preparation import (
    ANCHOR_SET,
    EXPECTED_F21_STATE_SHA256,
    F54_MANIFEST,
)
from .phase4f58_kl_trust_region_pilot import (
    CHECKPOINT as F58_CHECKPOINT,
    GATE as F58_GATE,
    REPORT as F58_REPORT,
    audit_training_evidence,
    verify_phase4f58,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_F58_GATE_SHA256 = "78afdcd38d9504838bc37508b5575251c647514e9d1c85c543d877558486f94a"
EXPECTED_F58_REPORT_SHA256 = "47bba59bb2fa43771d73bf610bcded3458a81d8d493e4886075f64f28c95109e"
EXPECTED_F58_MODEL_SHA256 = "4796535bbd66f210674a33c1c7f841c93a7f12ac939f4a8dc67693bb27e65da6"
EXPECTED_F58_STATE_SHA256 = "02a79834889f0b8b93c47e878007d81f9913c2d1e6a536b9fba51769042c970b"

F21_MODEL = Path(
    "training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/checkpoints/"
    "candidate-model-checkpoint-g00000001-s000000000192.pt"
)
F21_CONFIG = Path("training/configs/phase4f21-stage-a-block-01-v1.json")
EXPECTED_F21_MODEL_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
EXPECTED_F21_CONFIG_FILE_SHA256 = "bdaa011c52a3a3f0925dcf603d52bd19efe4b7cee6039d6e03140b14d007a14e"
EXPECTED_F54_MANIFEST_SHA256 = "a87a7493e2434b944bb3adda944da32ef0e4f53f91dc8ff36a427a8b3f3e26da"
EXPECTED_ANCHOR_SHA256 = "78e8cc8cd027dc2dfa591e781c501a0b7dd4d830cafd1c16237735e5ccab873b"

ROOT = Path("training/runs/phase4/phase4f/f59-kl-trust-region-review-v1")
DELTA_AUDIT = ROOT / "model-delta-audit.json"
ISOLATION_AUDIT = ROOT / "opening-isolation-audit.json"
FULL_SUITE = ROOT / "f60-fresh-lineage-pool-200.json"
SCREEN_SUITE = ROOT / "f60-fresh-lineage-screen-32.json"
REPORT = ROOT / "review-report.json"
GATE = ROOT / "review-gate.json"
F60_CONTRACT = Path("training/configs/phase4f60-kl-lineage-screen-frozen-v1.json")

OPENING_PAIRS = 32
GAMES = 64
SELECTION_SEED = 20260916
MCTS_SEED = 20260916
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.90
SIMULATIONS = 800
ACTORS = 32
SCREEN_GPU_HOURS = 2.0
WRITE_RESERVE_BYTES = 64 * 1024**2


class Phase4F59Error(RuntimeError):
    """Raised when F58 review or F60 freeze evidence is invalid."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        canonical_json_bytes(
            {
                "kind": "phase4f59-kl-trust-region-review-progress",
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
        raise Phase4F59Error("workspace does not contain training/quoridor")
    return workspace


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as error:
        raise Phase4F59Error("artifact path escapes workspace") from error


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f59_kl_trust_region_review.py",
            "training/quoridor/phase4f58_kl_trust_region_pilot.py",
            "training/quoridor/phase4_arena.py",
            "training/quoridor/challenge_replay.py",
            "training/quoridor/evaluator.py",
            "training/quoridor/mcts.py",
            "training/quoridor/native_mcts.py",
            "training/quoridor/network.py",
            "training/quoridor/openings.py",
            "training/quoridor/paired_arena.py",
            "training/quoridor/selectors.py",
            "scripts/generate_phase4_openings.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F59Error("16 GiB artifact cap cannot reserve F59/F60 evidence")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F59Error("F59/F60 reserve would violate the 20 GiB stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _load_model_state(
    path: Path, expected_file_sha: str, expected_state_sha: str
) -> tuple[dict[str, object], Mapping[str, torch.Tensor]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_file_sha:
        raise Phase4F59Error(f"model file failed validation: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Phase4F59Error(f"model payload is unreadable: {path}") from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or not isinstance(state, Mapping)
        or canonical_model_state_sha256(state) != expected_state_sha
    ):
        raise Phase4F59Error(f"model state failed validation: {path}")
    return dict(payload), state


def policy_head_delta_audit(
    baseline: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    if tuple(baseline) != tuple(candidate):
        raise Phase4F59Error("F21 and F58 tensor keys differ")
    changed: list[str] = []
    non_policy_changed: list[str] = []
    maximum_delta = 0.0
    changed_elements = 0
    for name, before in baseline.items():
        after = candidate[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise Phase4F59Error(f"tensor contract differs for {name}")
        if torch.equal(before, after):
            continue
        changed.append(name)
        if not name.startswith("policy_head."):
            non_policy_changed.append(name)
        changed_elements += int(torch.count_nonzero(before != after).item())
        if before.is_floating_point():
            maximum_delta = max(
                maximum_delta, float((before - after).abs().max().item())
            )
    checks = {
        "sameTensorContract": True,
        "modelStateChanged": bool(changed),
        "policyHeadChanged": any(name.startswith("policy_head.") for name in changed),
        "nonPolicyTensorsExact": not non_policy_changed,
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f59-policy-head-delta-audit",
        "checks": checks,
        "changedTensorNames": changed,
        "nonPolicyChangedTensorNames": non_policy_changed,
        "changedElements": changed_elements,
        "maximumAbsoluteDelta": maximum_delta,
        "passed": all(checks.values()),
    }


def review_f58(report: Mapping[str, object], audit: Mapping[str, object]) -> dict[str, object]:
    evaluations = report.get("evaluations")
    expected_steps = [0, 1, 2, 4, 8, 16, 24, 32]
    if (
        not isinstance(evaluations, list)
        or [entry.get("step") for entry in evaluations] != expected_steps
    ):
        raise Phase4F59Error("F58 evaluation schedule differs")
    baseline = evaluations[0]["metrics"]["policyLoss"]
    selected = evaluations[-1]["metrics"]["policyLoss"]
    bands = ("all", "early", "middle", "late")
    improvements = {
        band: (float(baseline[band]) - float(selected[band]))
        / float(baseline[band])
        for band in bands
    }
    sequences = {
        band: [float(entry["metrics"]["policyLoss"][band]) for entry in evaluations]
        for band in bands
    }
    exact_invariants = all(
        entry["metrics"]["frozenTensors"]["passed"] is True
        and float(entry["metrics"]["frozenTensors"]["maximumAbsoluteError"]) == 0.0
        and not entry["metrics"]["frozenTensors"]["changedTensorNames"]
        and float(entry["metrics"]["valueOutputMaximumAbsoluteError"]) == 0.0
        for entry in evaluations
    )
    checks = {
        "pilotPassed": report.get("pilotPassed") is True,
        "selectedStep32": report.get("selectedCheckpoint", {}).get("step") == 32,
        "trustRegionConverged": audit.get("stopReason") == "trust-region-converged",
        "hardKlInvariant": float(audit.get("maximumAcceptedMetric", math.inf)) <= 0.018,
        "allUpdatedCheckpointsFromStep2Eligible": all(
            entry.get("eligible") is True for entry in evaluations[2:]
        ),
        "allBandLossesStrictlyDecrease": all(
            all(before > after for before, after in zip(values, values[1:]))
            for values in sequences.values()
        ),
        "overallImprovementAtLeastFourPercent": improvements["all"] >= 0.04,
        "earlyImprovementAtLeastFourPercent": improvements["early"] >= 0.04,
        "middleAndLateImprove": improvements["middle"] > 0.0
        and improvements["late"] > 0.0,
        "frozenAndValueInvariantsExact": exact_invariants,
    }
    return {
        "baselineStep": 0,
        "selectedStep": 32,
        "relativePolicyLossImprovements": improvements,
        "policyLossSequences": sequences,
        "eligibleEvaluationSteps": audit.get("eligibleEvaluationSteps"),
        "trainingAudit": dict(audit),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _f58_and_f21_evidence(
    workspace: Path,
) -> tuple[dict[str, object], dict[str, object], Mapping[str, torch.Tensor], dict[str, object], Mapping[str, torch.Tensor]]:
    verification = verify_phase4f58(workspace)
    gate = _mapping(workspace / F58_GATE, "F58 gate")
    report = _mapping(workspace / F58_REPORT, "F58 report")
    if (
        verification.get("gateSha256") != EXPECTED_F58_GATE_SHA256
        or verification.get("reportSha256") != EXPECTED_F58_REPORT_SHA256
        or verification.get("pilotPassed") is not True
        or _verify_self_hash(gate, "gateSha256", "F58 gate")
        != EXPECTED_F58_GATE_SHA256
        or _verify_self_hash(report, "reportSha256", "F58 report")
        != EXPECTED_F58_REPORT_SHA256
        or report.get("selectedCheckpoint", {}).get("fileSha256")
        != EXPECTED_F58_MODEL_SHA256
        or report.get("selectedCheckpoint", {}).get("modelStateSha256")
        != EXPECTED_F58_STATE_SHA256
    ):
        raise Phase4F59Error("F58 is not the accepted pilot evidence")
    f58_payload, f58_state = _load_model_state(
        workspace / F58_CHECKPOINT,
        EXPECTED_F58_MODEL_SHA256,
        EXPECTED_F58_STATE_SHA256,
    )
    if sha256_file(workspace / F21_CONFIG) != EXPECTED_F21_CONFIG_FILE_SHA256:
        raise Phase4F59Error("F21 config file changed")
    f21_payload, f21_state = _load_model_state(
        workspace / F21_MODEL,
        EXPECTED_F21_MODEL_SHA256,
        EXPECTED_F21_STATE_SHA256,
    )
    return report, f58_payload, f58_state, f21_payload, f21_state


def _historical_arena_orbits(
    workspace: Path,
) -> tuple[set[str], list[dict[str, object]]]:
    orbits: set[str] = set()
    references: list[dict[str, object]] = []
    root = workspace / "training/runs/phase4/phase4f"
    for path in sorted(root.rglob("result.json")):
        result = _mapping(path, "historical arena result")
        if (
            result.get("status") != "completed"
            or not isinstance(result.get("pairArtifacts"), list)
            or not isinstance(result.get("resultSha256"), str)
        ):
            continue
        reference = {
            "path": _relative(path, workspace),
            "fileSha256": sha256_file(path),
            "resultSha256": result["resultSha256"],
            "games": result.get("games"),
        }
        orbits.update(arena_position_orbits(workspace, reference))
        references.append(reference)
    if not references:
        raise Phase4F59Error("no historical arena results were found")
    return orbits, references


def _supervised_orbits(workspace: Path) -> tuple[set[str], dict[str, object]]:
    manifest = _mapping(workspace / F54_MANIFEST, "F54 dataset manifest")
    if (
        _verify_self_hash(manifest, "manifestSha256", "F54 dataset manifest")
        != EXPECTED_F54_MANIFEST_SHA256
    ):
        raise Phase4F59Error("F54 dataset manifest changed")
    orbits: set[str] = set()
    for reference in manifest.get("chunks", []):
        path = workspace / str(reference["path"])
        chunk = _mapping(path, "F54 dataset chunk")
        if (
            _verify_self_hash(chunk, "chunkSha256", "F54 dataset chunk")
            != reference.get("sha256")
        ):
            raise Phase4F59Error("F54 dataset chunk changed")
        for sample in chunk.get("samples", []):
            orbits.add(str(sample["source"]["positionOrbit"]))
    anchor = _mapping(workspace / ANCHOR_SET, "F55 anchor set")
    if (
        _verify_self_hash(anchor, "anchorSetSha256", "F55 anchor set")
        != EXPECTED_ANCHOR_SHA256
    ):
        raise Phase4F59Error("F55 anchor set changed")
    anchor_orbits = {str(sample["positionOrbit"]) for sample in anchor["samples"]}
    return orbits | anchor_orbits, {
        "datasetManifestPath": F54_MANIFEST.as_posix(),
        "datasetManifestSha256": EXPECTED_F54_MANIFEST_SHA256,
        "datasetPositionOrbits": len(orbits),
        "anchorSetPath": ANCHOR_SET.as_posix(),
        "anchorSetSha256": EXPECTED_ANCHOR_SHA256,
        "anchorPositionOrbits": len(anchor_orbits),
    }


def _generate_suites(
    workspace: Path,
) -> tuple[OpeningSuite, OpeningSuite, dict[str, object]]:
    frozen: list[OpeningSuite] = []
    frozen_refs: list[dict[str, object]] = []
    for relative, name in FROZEN_SUITES:
        suite = load_opening_suite(workspace / relative, expected_suite=name)
        frozen.append(suite)
        frozen_refs.append(
            {
                "path": relative.as_posix(),
                "name": name,
                "fileSha256": sha256_file(workspace / relative),
                "suiteHash": suite.content_hash,
                "openingPairs": len(suite.openings),
            }
        )
    validate_suite_collection(*frozen)
    forbidden = {
        opening_position_orbit_digest(opening)
        for suite in frozen
        for opening in suite.openings
    }
    replay_refs: list[dict[str, object]] = []
    for relative in REPLAY_ROOTS:
        _progress("replay-isolation", "started", path=relative.as_posix())
        orbits, reference = _replay_orbits(workspace, relative)
        forbidden.update(orbits)
        replay_refs.append({**reference, "uniquePositionOrbits": len(orbits)})
        _progress(
            "replay-isolation",
            "completed",
            path=relative.as_posix(),
            positions=reference["positions"],
        )
    arena_orbits, arena_refs = _historical_arena_orbits(workspace)
    forbidden.update(arena_orbits)
    supervised, supervised_ref = _supervised_orbits(workspace)
    forbidden.update(supervised)
    generator = _load_generator(workspace)
    fresh = generator.generate_fresh_development_suite(
        tuple(frozen), forbidden_orbits=forbidden
    )
    fresh_orbits = {
        opening_position_orbit_digest(opening) for opening in fresh.openings
    }
    overlap = len(fresh_orbits & forbidden)
    if overlap or len(fresh.openings) != 200:
        raise Phase4F59Error("F60 fresh opening pool is not isolated")

    def rank(opening: object) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{SELECTION_SEED}\0{getattr(opening, 'content_hash')}".encode("ascii")
        ).hexdigest()
        return digest, str(getattr(opening, "id"))

    selected = tuple(sorted(fresh.openings, key=rank)[:OPENING_PAIRS])
    screen = build_opening_suite(selected, suite="recovery-fresh-dev-v1")
    audit = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f59-f60-opening-isolation-audit",
            "status": "completed",
            "frozenSuites": frozen_refs,
            "replayStores": replay_refs,
            "historicalArenaResults": arena_refs,
            "historicalArenaPositionOrbits": len(arena_orbits),
            "supervisedInputs": supervised_ref,
            "supervisedAndAnchorPositionOrbits": len(supervised),
            "totalExcludedPositionOrbits": len(forbidden),
            "excludedPositionOrbitsSha256": sha256_bytes(
                canonical_json_bytes(sorted(forbidden))
            ),
            "freshOpeningPositionOrbits": len(fresh_orbits),
            "freshOverlapCount": overlap,
            "selectionSeed": SELECTION_SEED,
            "selectedOpeningIds": [opening.id for opening in screen.openings],
        },
        "auditSha256",
    )
    return fresh, screen, audit


def _suite_reference(
    workspace: Path,
    full: OpeningSuite,
    screen: OpeningSuite,
    audit: Mapping[str, object],
) -> dict[str, object]:
    return {
        "fullPath": FULL_SUITE.as_posix(),
        "fullFileSha256": sha256_file(workspace / FULL_SUITE),
        "fullSuiteHash": full.content_hash,
        "fullOpeningPairs": len(full.openings),
        "selectedPath": SCREEN_SUITE.as_posix(),
        "selectedFileSha256": sha256_file(workspace / SCREEN_SUITE),
        "selectedSuiteHash": screen.content_hash,
        "selectedOpeningPairs": len(screen.openings),
        "selectedOpeningIds": [opening.id for opening in screen.openings],
        "selectionAlgorithm": "sha256(seed-ascii+nul+opening-content-hash)-lowest-v1",
        "selectionSeed": SELECTION_SEED,
        "isolationAuditPath": ISOLATION_AUDIT.as_posix(),
        "isolationAuditSha256": audit["auditSha256"],
        "freshOverlapCount": audit["freshOverlapCount"],
        "contentPositionOrbitAndTrajectoryDisjoint": True,
    }


def _f60_contract(
    *,
    source: Mapping[str, object],
    review_sha: str,
    delta_sha: str,
    suite: Mapping[str, object],
    ledger_hours: float,
    resources: Mapping[str, object],
    f21_payload: Mapping[str, object],
    f58_report: Mapping[str, object],
) -> dict[str, object]:
    identity = f21_payload.get("identity")
    if not isinstance(identity, Mapping):
        raise Phase4F59Error("F21 checkpoint identity is missing")
    runner_source = f58_report.get("runnerSource")
    if not isinstance(runner_source, Mapping):
        raise Phase4F59Error("F58 runner identity is missing")
    workload = _self_hashed(
        {
            "suite": "recovery-fresh-dev-v1",
            "suiteHash": suite["selectedSuiteHash"],
            "openingPairs": OPENING_PAIRS,
            "games": GAMES,
            "candidateSimulations": SIMULATIONS,
            "opponentSimulations": SIMULATIONS,
            "actors": ACTORS,
            "mctsSeed": MCTS_SEED,
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapSamples": BOOTSTRAP_SAMPLES,
            "confidence": CONFIDENCE,
        },
        "workloadSha256",
    )
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f60-f58-versus-f21-fresh-lineage-screen",
            "status": "frozen",
            "f59ReviewSha256": review_sha,
            "f59DeltaAuditSha256": delta_sha,
            "freezeSource": dict(source),
            "nativeBackendSha256": loaded_native_backend_hash(),
            "candidate": {
                "selectorId": "phase4f58-kl-policy-head-step-00000032",
                "modelPath": F58_CHECKPOINT.as_posix(),
                "modelFileSha256": EXPECTED_F58_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F58_STATE_SHA256,
                "trainingContractSha256": "67bb8d52329934b5c9a3b66038a13318ae5e3df9b896a2d423dfa8c2408d20ea",
                "trainingGateSha256": EXPECTED_F58_GATE_SHA256,
                "trainingSourceSha256": runner_source["sha256"],
            },
            "opponent": {
                "selectorId": "phase4f21-stage-a-best-g00000001-s000000000192",
                "modelPath": F21_MODEL.as_posix(),
                "modelFileSha256": EXPECTED_F21_MODEL_SHA256,
                "modelStateSha256": EXPECTED_F21_STATE_SHA256,
                "configPath": F21_CONFIG.as_posix(),
                "configFileSha256": EXPECTED_F21_CONFIG_FILE_SHA256,
                "configSha256": identity["configSha256"],
                "sourceSha256": identity["sourceSha256"],
            },
            "runtimeConfigPath": F21_CONFIG.as_posix(),
            "openingSuite": dict(suite),
            "workload": workload,
            "budget": {
                "baselineGlobalHours": ledger_hours,
                "screenLimitHours": SCREEN_GPU_HOURS,
                "globalLimitHours": GPU_HOUR_BUDGET,
                "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
                "arenaWriteReserveBytes": WRITE_RESERVE_BYTES,
                "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
            },
            "decisionPolicy": {
                "regressionBelowWinRate": 0.45,
                "inconclusiveBelowWinRate": 0.50,
                "lineageRetainedAtOrAboveWinRate": 0.50,
                "drawsRequired": 0,
                "nextStageAlwaysRequiresConfirmation": True,
            },
            "authorizations": {
                "f60LineageScreenAuthorized": True,
                "classicHardScreenAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "screenTrajectoriesTrainingUseAllowed": False,
            },
            "resourcesAtFreeze": dict(resources),
        },
        "contractSha256",
    )


def run_phase4f59(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f59(workspace), "reused": True}
    _progress("preflight", "started")
    report58, _payload58, state58, payload21, state21 = _f58_and_f21_evidence(
        workspace
    )
    contract58 = _mapping(
        workspace / "training/configs/phase4f58-kl-trust-region-policy-head-pilot-frozen-v1.json",
        "F58 contract",
    )
    training_audit = audit_training_evidence(report58, contract58)
    review = review_f58(report58, training_audit)
    delta = _self_hashed(policy_head_delta_audit(state21, state58), "auditSha256")
    if not review["passed"] or not delta["passed"]:
        raise Phase4F59Error("F58 evidence does not authorize a lineage screen")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    if any(record.get("status") == "running" for record in ledger.runs.values()):
        raise Phase4F59Error("a running GPU ledger entry blocks F59")
    if ledger.total_hours + SCREEN_GPU_HOURS > GPU_HOUR_BUDGET:
        raise Phase4F59Error("global GPU budget cannot reserve F60")
    resources = _resources(workspace)
    source = _source(workspace)
    run_id = f"phase4f59-kl-trust-region-review-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        full, screen, isolation = _generate_suites(workspace)
        _write_immutable(workspace / DELTA_AUDIT, delta, "F59 delta audit")
        _write_immutable(workspace / ISOLATION_AUDIT, isolation, "F59 isolation audit")
        _write_immutable(
            workspace / FULL_SUITE,
            opening_suite_to_payload(full),
            "F60 full opening pool",
        )
        _write_immutable(
            workspace / SCREEN_SUITE,
            opening_suite_to_payload(screen),
            "F60 screen opening suite",
        )
        suite = _suite_reference(workspace, full, screen, isolation)
        report_core = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f59-kl-trust-region-pilot-review",
            "status": "completed",
            "cpuOnly": True,
            "cudaAccessed": False,
            "trainingStarted": False,
            "optimizerSteps": 0,
            "f58GateSha256": EXPECTED_F58_GATE_SHA256,
            "f58ReportSha256": EXPECTED_F58_REPORT_SHA256,
            "f58ModelSha256": EXPECTED_F58_MODEL_SHA256,
            "f58ModelStateSha256": EXPECTED_F58_STATE_SHA256,
            "review": review,
            "modelDeltaAuditPath": DELTA_AUDIT.as_posix(),
            "modelDeltaAuditSha256": delta["auditSha256"],
            "openingIsolationAuditPath": ISOLATION_AUDIT.as_posix(),
            "openingIsolationAuditSha256": isolation["auditSha256"],
            "f60OpeningSuite": suite,
            "reviewPassed": True,
            "source": source,
            "globalGpuHours": ledger.total_hours,
            "resources": resources,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
        }
        review_content = _self_hashed(report_core, "reviewContentSha256")
        contract = _f60_contract(
            source=source,
            review_sha=str(review_content["reviewContentSha256"]),
            delta_sha=str(delta["auditSha256"]),
            suite=suite,
            ledger_hours=ledger.total_hours,
            resources=resources,
            f21_payload=payload21,
            f58_report=report58,
        )
        report = _self_hashed(
            {
                **report_core,
                "reviewContentSha256": review_content["reviewContentSha256"],
                "f60ContractPath": F60_CONTRACT.as_posix(),
                "f60ContractSha256": contract["contractSha256"],
                "f60LineageScreenAuthorized": True,
                "nextStage": {
                    "name": "Phase 4F60 F58-versus-F21 fresh-lineage screen",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "reportSha256",
        )
        _write_immutable(workspace / F60_CONTRACT, contract, "F60 contract")
        _write_immutable(workspace / REPORT, report, "F59 review report")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f59-kl-trust-region-pilot-review-gate",
                "status": "completed",
                "outcomeBand": "f60-fresh-lineage-screen-ready",
                "reviewPassed": True,
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f60ContractPath": F60_CONTRACT.as_posix(),
                "f60ContractSha256": contract["contractSha256"],
                "f60LineageScreenAuthorized": True,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "globalGpuHours": ledger.total_hours,
                "nextStage": report["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F59 review gate")
    return {**gate, "reused": False}


def _verify_reference_files(workspace: Path, audit: Mapping[str, object]) -> None:
    for collection, suffix in (
        ("frozenSuites", ""),
        ("replayStores", "/manifest.json"),
        ("historicalArenaResults", ""),
    ):
        references = audit.get(collection)
        if not isinstance(references, list) or not references:
            raise Phase4F59Error(f"F59 {collection} bindings are empty")
        for reference in references:
            path = workspace / f"{reference['path']}{suffix}"
            if path.is_symlink() or sha256_file(path) != reference.get(
                "manifestSha256" if collection == "replayStores" else "fileSha256"
            ):
                raise Phase4F59Error(f"F59 {collection} binding changed")


def verify_phase4f59(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report58, _payload58, state58, payload21, state21 = _f58_and_f21_evidence(
        workspace
    )
    contract58 = _mapping(
        workspace / "training/configs/phase4f58-kl-trust-region-policy-head-pilot-frozen-v1.json",
        "F58 contract",
    )
    expected_review = review_f58(
        report58, audit_training_evidence(report58, contract58)
    )
    expected_delta = _self_hashed(
        policy_head_delta_audit(state21, state58), "auditSha256"
    )
    delta = _mapping(workspace / DELTA_AUDIT, "F59 delta audit")
    isolation = _mapping(workspace / ISOLATION_AUDIT, "F59 isolation audit")
    report = _mapping(workspace / REPORT, "F59 report")
    gate = _mapping(workspace / GATE, "F59 gate")
    contract = _mapping(workspace / F60_CONTRACT, "F60 contract")
    delta_sha = _verify_self_hash(delta, "auditSha256", "F59 delta audit")
    isolation_sha = _verify_self_hash(
        isolation, "auditSha256", "F59 isolation audit"
    )
    report_sha = _verify_self_hash(report, "reportSha256", "F59 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F59 gate")
    contract_sha = _verify_self_hash(contract, "contractSha256", "F60 contract")
    _verify_reference_files(workspace, isolation)
    if (
        isolation.get("status") != "completed"
        or isolation.get("freshOverlapCount") != 0
        or isolation.get("supervisedInputs", {}).get("datasetManifestSha256")
        != EXPECTED_F54_MANIFEST_SHA256
        or isolation.get("supervisedInputs", {}).get("anchorSetSha256")
        != EXPECTED_ANCHOR_SHA256
    ):
        raise Phase4F59Error("F59 isolation audit changed")
    full = load_opening_suite(
        workspace / FULL_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    screen = load_opening_suite(
        workspace / SCREEN_SUITE, expected_suite="recovery-fresh-dev-v1"
    )
    selected_ids = {opening.id for opening in screen.openings}
    if (
        len(full.openings) != 200
        or len(screen.openings) != OPENING_PAIRS
        or not selected_ids <= {opening.id for opening in full.openings}
    ):
        raise Phase4F59Error("F60 opening suites changed")
    suite = _suite_reference(workspace, full, screen, isolation)
    source = _source(workspace)
    budget = contract.get("budget")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    content_core = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "reviewContentSha256",
            "f60ContractPath",
            "f60ContractSha256",
            "f60LineageScreenAuthorized",
            "nextStage",
            "reportSha256",
        }
    }
    content_sha = _self_hashed(content_core, "reviewContentSha256")[
        "reviewContentSha256"
    ]
    expected_contract = _f60_contract(
        source=source,
        review_sha=str(content_sha),
        delta_sha=str(delta_sha),
        suite=suite,
        ledger_hours=float(budget["baselineGlobalHours"]),
        resources=contract["resourcesAtFreeze"],
        f21_payload=payload21,
        f58_report=report58,
    ) if isinstance(budget, Mapping) else None
    if (
        delta != expected_delta
        or report.get("review") != expected_review
        or report.get("source") != source
        or report.get("reviewContentSha256") != content_sha
        or report.get("modelDeltaAuditSha256") != delta_sha
        or report.get("openingIsolationAuditSha256") != isolation_sha
        or report.get("f60OpeningSuite") != suite
        or report.get("f60ContractSha256") != contract_sha
        or report.get("reviewPassed") is not True
        or report.get("cpuOnly") is not True
        or report.get("cudaAccessed") is not False
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
        or report.get("formalTrainingAuthorized") is not False
        or report.get("promotionAuthorized") is not False
        or gate.get("reportSha256") != report_sha
        or gate.get("f60ContractSha256") != contract_sha
        or gate.get("f60LineageScreenAuthorized") is not True
        or gate.get("formalTrainingAuthorized") is not False
        or contract != expected_contract
        or contract.get("nativeBackendSha256") != loaded_native_backend_hash()
        or not isinstance(budget, Mapping)
        or ledger.total_hours < float(budget["baselineGlobalHours"])
        or ledger.total_hours
        > float(budget["baselineGlobalHours"]) + SCREEN_GPU_HOURS
    ):
        raise Phase4F59Error("F59/F60 frozen provenance changed")
    _resources(workspace)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f59-kl-trust-region-pilot-review-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "f60ContractPath": F60_CONTRACT.as_posix(),
        "f60ContractSha256": contract_sha,
        "reviewPassed": True,
        "f60LineageScreenAuthorized": True,
        "formalTrainingAuthorized": False,
        "globalGpuHours": ledger.total_hours,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = (
        run_phase4f59(args.workspace_root)
        if args.command == "run"
        else verify_phase4f59(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
