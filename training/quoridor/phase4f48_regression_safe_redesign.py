"""Audit the failed curriculum and freeze a regression-safe redesign.

F48 is CPU-only.  It traces every F42 target back to the immutable classic
guidance replay, audits the target and ply-band semantics, and freezes the
diagnostic-only F49 contract.  It never creates training data or an optimizer.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .codec import state_from_serializable, state_to_serializable
from .network import ACTION_SIZE
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f41_target_protocol import CONTRACT as F41_CONTRACT
from .phase4f42_curriculum_dataset import GATE as F42_GATE, MANIFEST as F42_MANIFEST
from .phase4f47_lineage_regression_diagnosis import (
    GATE as F47_GATE,
    REPORT as F47_REPORT,
    verify_phase4f47,
)
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
)
from .storage import directory_size


SCHEMA_VERSION = 1
ROOT = Path("training/runs/phase4/phase4f/f48-regression-safe-redesign-v1")
REPORT = ROOT / "redesign-report.json"
GATE = ROOT / "redesign-gate.json"
F49_CONTRACT = Path("training/configs/phase4f49-teacher-search-audit-frozen-v1.json")
RUN_ID = "phase4f48-regression-safe-redesign-v1"

EXPECTED_F47_GATE_SHA256 = "384fe0374ac40d1c256cd57f1c837ca162247dad1a744714978fb9c77e38998e"
EXPECTED_F47_REPORT_SHA256 = "e0d53e1bb73790b1325fc118fb3f8357622d1d0d08307d5e78428d841ded7953"
EXPECTED_F42_GATE_SHA256 = "94ad13fcb818bf4be2643ee2d77ac385f4258abfc7f4305dc8c98da763c378b3"
EXPECTED_F42_MANIFEST_SHA256 = "54e3c3a6c7df9a912d8ac6414815fc182a50876f2368cf073a6435fd338f3a45"
EXPECTED_F41_CONTRACT_SHA256 = "eaad63d8015682131d2f9ba76f447af916375055ca666b3e15c128eca7e430c1"
EXPECTED_F21_MODEL_SHA256 = "1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01"
EXPECTED_F21_STATE_SHA256 = "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527"

WRITE_RESERVE_BYTES = 32 * 1024**2
SURROGATE_TOP1_MINIMUM = 0.90


class Phase4F48Error(RuntimeError):
    """Raised when F48 evidence or immutable outputs are invalid."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F48Error("workspace does not contain training/quoridor")
    return workspace


def absolute_ply_band(move_number: int) -> str:
    if type(move_number) is not int or move_number < 0:
        raise ValueError("move_number must be a non-negative integer")
    if move_number < 16:
        return "early"
    if move_number < 40:
        return "middle"
    return "late"


def _rank(probabilities: Sequence[float], action: int) -> int:
    if not 0 <= action < len(probabilities):
        raise ValueError("action is outside the policy")
    ordered = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    return ordered.index(action) + 1


def jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("policies must have the same non-zero length")
    middle = [(float(a) + float(b)) / 2.0 for a, b in zip(left, right, strict=True)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / reference)
            for value, reference in zip(values, middle, strict=True)
            if value > 0.0
        )

    return 0.5 * divergence(left) + 0.5 * divergence(right)


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f48_regression_safe_redesign.py",
            "training/quoridor/phase4f47_lineage_regression_diagnosis.py",
            "training/quoridor/phase4f42_curriculum_dataset.py",
            "training/quoridor/phase4f41_target_protocol.py",
            "training/quoridor/phase4f40_target_probe.py",
        ),
    ).to_payload()


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F48Error("16 GiB artifact cap cannot reserve F48 evidence")
    if free_bytes - WRITE_RESERVE_BYTES < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F48Error("insufficient free disk for F48 evidence")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
        "reservedWriteBytes": WRITE_RESERVE_BYTES,
    }


def _load_inputs(workspace: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    f47 = verify_phase4f47(workspace)
    f47_report = _mapping(workspace / F47_REPORT, "F47 report")
    f42_gate = _mapping(workspace / F42_GATE, "F42 gate")
    f42_manifest = _mapping(workspace / F42_MANIFEST, "F42 manifest")
    f41 = _mapping(workspace / F41_CONTRACT, "F41 contract")
    if (
        f47.get("gateSha256") != EXPECTED_F47_GATE_SHA256
        or f47.get("reportSha256") != EXPECTED_F47_REPORT_SHA256
        or _verify_self_hash(f47_report, "reportSha256", "F47 report") != EXPECTED_F47_REPORT_SHA256
        or _verify_self_hash(f42_gate, "gateSha256", "F42 gate") != EXPECTED_F42_GATE_SHA256
        or _verify_self_hash(f42_manifest, "manifestSha256", "F42 manifest") != EXPECTED_F42_MANIFEST_SHA256
        or _verify_self_hash(f41, "contractSha256", "F41 contract") != EXPECTED_F41_CONTRACT_SHA256
        or f47_report.get("formalTrainingAuthorized") is not False
        or f47_report.get("screenTrajectoriesTrainingUseAllowed") is not False
    ):
        raise Phase4F48Error("F41/F42/F47 evidence differs from the reviewed lineage")
    return f47_report, f42_gate, f42_manifest, f41


def _load_f42_samples(workspace: Path, manifest: Mapping[str, object]) -> list[dict[str, object]]:
    references = manifest.get("chunks")
    if not isinstance(references, list):
        raise Phase4F48Error("F42 manifest chunk inventory is malformed")
    samples: list[dict[str, object]] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            raise Phase4F48Error("F42 manifest chunk reference is malformed")
        chunk = _mapping(workspace / str(reference["path"]), "F42 dataset chunk")
        if _verify_self_hash(chunk, "chunkSha256", "F42 dataset chunk") != reference.get("sha256"):
            raise Phase4F48Error("F42 dataset chunk reference changed")
        rows = chunk.get("samples")
        if not isinstance(rows, list) or len(rows) != reference.get("positions"):
            raise Phase4F48Error("F42 dataset chunk count changed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise Phase4F48Error("F42 sample is malformed")
            _verify_self_hash(row, "sampleSha256", "F42 sample")
            if row.get("contractSha256") != manifest.get("contractSha256"):
                raise Phase4F48Error("F42 sample contract changed")
            samples.append(dict(row))
    if len(samples) != 4608:
        raise Phase4F48Error("F42 dataset must contain exactly 4608 samples")
    return samples


def _policy_audit(workspace: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    samples = _load_f42_samples(workspace, manifest)
    source_games: set[str] = set()
    nominal = Counter()
    absolute = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    surrogate_ranks: list[int] = []
    surrogate_probabilities: list[float] = []
    divergences: list[float] = []
    mismatches = 0

    for sample in samples:
        source = sample["source"]
        teacher = int(source["teacherActionIndex"])
        state = state_from_serializable(sample["state"])
        if state_to_serializable(state) != sample["state"]:
            raise Phase4F48Error("F42 state serialization is not canonical")
        if not 0 <= teacher < ACTION_SIZE:
            raise Phase4F48Error("recorded classic teacher action is outside the action space")
        source_games.add(str(source["gameId"]))

        surrogate = [0.0] * ACTION_SIZE
        for action, probability in sample["policy"]:
            surrogate[int(action)] = float(probability)
        hard_teacher = [0.0] * ACTION_SIZE
        hard_teacher[teacher] = 1.0
        surrogate_ranks.append(_rank(surrogate, teacher))
        surrogate_probabilities.append(surrogate[teacher])
        divergences.append(jensen_shannon(surrogate, hard_teacher))

        nominal_band = str(sample["band"])
        absolute_band = absolute_ply_band(state.move_number)
        nominal[nominal_band] += 1
        absolute[absolute_band] += 1
        confusion[(nominal_band, absolute_band)] += 1
        mismatches += nominal_band != absolute_band

    positions = len(samples)
    return {
        "positions": positions,
        "sourceGames": len(source_games),
        "nominalBandDefinition": "trajectory-ply-index-after-opening-v1",
        "requiredBandDefinition": "absolute-state-move-number-v1",
        "nominalBands": dict(sorted(nominal.items())),
        "absoluteBands": dict(sorted(absolute.items())),
        "bandConfusion": {
            left: {right: confusion[(left, right)] for right in ("early", "middle", "late")}
            for left in ("early", "middle", "late")
        },
        "bandMismatchPositions": mismatches,
        "bandMismatchRate": mismatches / positions,
        "surrogatePolicy": {
            "teacherActionTop1Coverage": mean(rank == 1 for rank in surrogate_ranks),
            "teacherActionTop5Coverage": mean(rank <= 5 for rank in surrogate_ranks),
            "teacherActionMeanProbability": mean(surrogate_probabilities),
            "meanJensenShannonToHardTeacher": mean(divergences),
        },
        "recordedClassicTeacher": {
            "identity": "immutable-F42-source.teacherActionIndex-bound-through-F40-to-classic-replay",
            "hardTargetTop1Coverage": 1.0,
            "hardTargetTeacherActionProbability": 1.0,
        },
    }


def redesign_checks(policy_audit: Mapping[str, object], f47: Mapping[str, object], explicit_value_weight: float) -> dict[str, bool]:
    surrogate = policy_audit["surrogatePolicy"]
    diagnosis = f47["diagnosis"]
    return {
        "surrogateTeacherTop1BelowMinimum": float(surrogate["teacherActionTop1Coverage"]) < SURROGATE_TOP1_MINIMUM,
        "absolutePlyBandMismatchDetected": int(policy_audit["bandMismatchPositions"]) > 0,
        "valueTargetHasImplicitNeutralResidual": explicit_value_weight < 1.0,
        "lineageRegressionConfirmed": diagnosis.get("primaryCause") == "off-distribution-policy-drift-amplified-by-mcts",
        "screenTrajectoriesRemainExcludedFromTraining": f47.get("screenTrajectoriesTrainingUseAllowed") is False,
    }


def _f49_contract(
    report_sha256: str,
    source_sha256: str,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    return _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f49-teacher-root-and-search-coupling-audit-contract",
            "status": "frozen",
            "f48ReportSha256": report_sha256,
            "sourceSha256": source_sha256,
            "inputs": dict(inputs),
            "authorizations": {
                "diagnosticCudaInferenceAuthorized": True,
                "classicRootAnalysisAuthorized": True,
                "datasetGenerationAuthorized": False,
                "cudaTrainingAuthorized": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
            },
            "classicTeacherAudit": {
                "engine": "classic-hard-negamax",
                "budgets": [
                    {"kind": "nodes", "limit": 587, "maxDepth": 5},
                    {"kind": "nodes", "limit": 4096, "maxDepth": 5},
                ],
                "positions": 512,
                "requiredRootOutput": "legal-action-root-scores-v1",
                "policyTopK": 12,
                "forbiddenTargetSources": [
                    "one-ply-static-heuristic",
                    "student-mcts-policy-as-teacher-component",
                ],
            },
            "counterfactualMctsAudit": {
                "models": ["F21", "F44"],
                "positions": 128,
                "simulations": [100, 400, 800, 3200],
                "sameStatesAndSeeds": True,
                "rootNoise": False,
                "screenTrajectoryUse": "diagnostic-only-never-training",
            },
            "futureDatasetInvariants": {
                "stateSource": "learner-occupancy-rollouts-with-separate-seeds",
                "minimumUniqueTrainOrbits": 50000,
                "targetUniqueTrainOrbits": 100000,
                "plyBandDefinition": "absolute-state-move-number-v1",
                "teacherPolicySource": "actual-classic-root-search-only",
                "maximumLegalActionLabelSmoothing": 0.05,
                "screenTrajectoryTrainingUseAllowed": False,
            },
            "futureTrainingInvariants": {
                "firstStageTrainableParameters": ["policy_head"],
                "frozenParameters": ["input", "trunk", "value_features", "value_head"],
                "requiresIndependentF21AnchorReplay": True,
                "requiresPolicyPreservationKl": True,
                "valueTrainingSeparated": True,
                "checkpointSelectionRequiresExternalRegressionGate": True,
            },
            "nextStage": {
                "name": "Phase 4F49 teacher-root and same-state MCTS audit",
                "requiresSeparateUserConfirmation": True,
            },
        },
        "contractSha256",
    )


def run_phase4f48(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / REPORT).exists() or (workspace / GATE).exists() or (workspace / F49_CONTRACT).exists():
        return verify_phase4f48(workspace)
    resources = _resources(workspace)
    source = _source(workspace)
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=RUN_ID):
        f47, f42_gate, f42_manifest, f41 = _load_inputs(workspace)
        policy_audit = _policy_audit(workspace, f42_manifest)
        value_target = f41["valueTarget"]
        explicit_weight = sum(
            float(value_target[key])
            for key in ("terminalOutcomeWeight", "regularizedSearchWeight", "pathTempoWeight")
        )
        checks = redesign_checks(policy_audit, f47, explicit_weight)
        if not all(checks.values()):
            raise Phase4F48Error("F48 evidence does not support the frozen redesign")
        inputs = {
            "f47GateSha256": EXPECTED_F47_GATE_SHA256,
            "f47ReportSha256": EXPECTED_F47_REPORT_SHA256,
            "f42GateSha256": f42_gate["gateSha256"],
            "f42ManifestSha256": f42_manifest["manifestSha256"],
            "f41ContractSha256": f41["contractSha256"],
            "f21ModelSha256": EXPECTED_F21_MODEL_SHA256,
            "f21ModelStateSha256": EXPECTED_F21_STATE_SHA256,
        }
        report = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f48-regression-safe-curriculum-redesign",
                "status": "completed",
                "cpuOnly": True,
                "trainingStarted": False,
                "datasetGenerationStarted": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "source": source,
                "inputs": inputs,
                "audits": {
                    "policyAndPlyBand": policy_audit,
                    "valueTarget": {
                        "explicitWeightSum": explicit_weight,
                        "implicitNeutralResidualWeight": 1.0 - explicit_weight,
                        "components": dict(value_target),
                    },
                    "lineageRegression": {
                        "primaryCause": f47["diagnosis"]["primaryCause"],
                        "measurements": f47["diagnosis"]["measurements"],
                        "parameterDeltaByComponent": f47["parameterDeltaByComponent"],
                    },
                },
                "checks": checks,
                "redesignPassed": True,
                "resources": resources,
                "nextStage": {
                    "name": "Phase 4F49 teacher-root and same-state MCTS audit",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "reportSha256",
        )
        contract = _f49_contract(report["reportSha256"], str(source["sha256"]), inputs)
        _write_immutable(workspace / REPORT, report, "F48 redesign report")
        _write_immutable(workspace / F49_CONTRACT, contract, "F49 frozen contract")
        gate = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f48-regression-safe-curriculum-redesign-gate",
                "status": "completed",
                "redesignPassed": True,
                "reportPath": REPORT.as_posix(),
                "reportSha256": report["reportSha256"],
                "f49ContractPath": F49_CONTRACT.as_posix(),
                "f49ContractSha256": contract["contractSha256"],
                "trainingStarted": False,
                "datasetGenerationStarted": False,
                "formalTrainingAuthorized": False,
                "promotionAuthorized": False,
                "nextStage": report["nextStage"],
            },
            "gateSha256",
        )
        _write_immutable(workspace / GATE, gate, "F48 redesign gate")
        return {**gate, "reused": False}


def verify_phase4f48(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    report = _mapping(workspace / REPORT, "F48 report")
    gate = _mapping(workspace / GATE, "F48 gate")
    contract = _mapping(workspace / F49_CONTRACT, "F49 contract")
    _verify_self_hash(report, "reportSha256", "F48 report")
    _verify_self_hash(gate, "gateSha256", "F48 gate")
    _verify_self_hash(contract, "contractSha256", "F49 contract")
    if (
        report.get("source") != _source(workspace)
        or report.get("status") != "completed"
        or report.get("redesignPassed") is not True
        or report.get("trainingStarted") is not False
        or gate.get("reportSha256") != report.get("reportSha256")
        or gate.get("f49ContractSha256") != contract.get("contractSha256")
        or contract.get("f48ReportSha256") != report.get("reportSha256")
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
        or contract.get("authorizations", {}).get("datasetGenerationAuthorized") is not False
    ):
        raise Phase4F48Error("F48/F49 immutable evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f48-regression-safe-curriculum-redesign-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate["gateSha256"],
        "reportPath": REPORT.as_posix(),
        "reportSha256": report["reportSha256"],
        "f49ContractPath": F49_CONTRACT.as_posix(),
        "f49ContractSha256": contract["contractSha256"],
        "formalTrainingAuthorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    payload = run_phase4f48(args.workspace_root) if args.command == "run" else verify_phase4f48(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
