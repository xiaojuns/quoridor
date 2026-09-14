"""Phase 4F41 CPU-only correction of the F40 target protocol.

The correction consumes the immutable F40 search chunks.  It does not run
MCTS, construct an optimizer, generate a training dataset, or spend GPU time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f39_early_curriculum import CONTRACT as F39_CONTRACT
from .phase4f40_target_probe import (
    CALIBRATION as F40_CALIBRATION,
    CHUNKS as F40_CHUNKS,
    GATE as F40_GATE,
    RESULT as F40_RESULT,
    _entropy,
    _mcts_policy,
    _mirror_policy_back,
    _percentile,
    _softmax_scores,
)
from .run_manifest import canonical_json_bytes, fingerprint_sources


SCHEMA_VERSION = 1
EXPECTED_F39_CONTRACT_SHA256 = (
    "b4e9ce03cb8bc84253a834c05eea049c3518c8c4426709dc81161aca36372dc5"
)
EXPECTED_F40_GATE_SHA256 = (
    "ed0f45fcfb6f489feb99d4c84029546d846e9291414021f83d2446711d99e548"
)
EXPECTED_F40_RESULT_SHA256 = (
    "51215cab27cf88c9f72d621d90794ddffe6c81ebc20b3d732f87cad71ab55e16"
)

ROOT = Path("training/runs/phase4/phase4f/f41-target-protocol-correction-v1")
REPORT = ROOT / "correction-report.json"
GATE = ROOT / "correction-gate.json"
CONTRACT = Path("training/configs/phase4f41-target-protocol-frozen-v1.json")

POLICY_CLASSIC_WEIGHT = 0.50
POLICY_SEARCH_WEIGHT = 0.50
POLICY_TEMPERATURE_FIT_TOP1_P95_MAXIMUM = 0.875
POLICY_TEMPERATURE_MULTIPLIER = 1.08
VALUE_OUTCOME_WEIGHT = 0.30
VALUE_SEARCH_WEIGHT = 0.15
VALUE_PATH_TEMPO_WEIGHT = 0.10
VALUE_NEUTRAL_WEIGHT = 0.45
ISOTONIC_QUANTILE_BINS = 32
ISOTONIC_NEUTRAL_PRIOR_WEIGHT = 32.0


class Phase4F41Error(RuntimeError):
    """Raised when F40 evidence cannot authorize a corrected protocol."""


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F41Error("workspace does not contain training/quoridor")
    return workspace


def _implementation_sha256(workspace: Path) -> str:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f41_target_protocol.py",
            "training/quoridor/phase4f40_target_probe.py",
            "training/quoridor/phase4f39_early_curriculum.py",
        ),
    ).sha256


def _load_inputs(workspace: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    f39 = _mapping(workspace / F39_CONTRACT, "F39 contract")
    f39_sha = _verify_self_hash(f39, "contractSha256", "F39 contract")
    f40_gate = _mapping(workspace / F40_GATE, "F40 gate")
    gate_sha = _verify_self_hash(f40_gate, "gateSha256", "F40 gate")
    f40_result = _mapping(workspace / F40_RESULT, "F40 result")
    result_sha = _verify_self_hash(f40_result, "resultSha256", "F40 result")
    if (
        f39_sha != EXPECTED_F39_CONTRACT_SHA256
        or gate_sha != EXPECTED_F40_GATE_SHA256
        or result_sha != EXPECTED_F40_RESULT_SHA256
        or f40_gate.get("resultSha256") != result_sha
        or f40_gate.get("outcomeBand") != "target-probe-failed"
        or f40_gate.get("datasetGenerationAuthorized") is not False
        or f40_gate.get("formalTrainingAuthorized") is not False
        or f40_result.get("trainingStarted") is not False
        or f40_result.get("optimizerSteps") != 0
    ):
        raise Phase4F41Error("F39/F40 evidence does not authorize F41 correction")
    expected_failures = {
        "policyTargetTop1ProbabilityP95",
        "valueTargetMeanAbsolute",
    }
    checks = f40_gate.get("checks")
    if not isinstance(checks, Mapping) or {
        name for name, passed in checks.items() if passed is not True
    } != expected_failures:
        raise Phase4F41Error("F40 failure set differs from the frozen F41 correction scope")

    rows: list[dict[str, object]] = []
    chunks = f40_result.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 72:
        raise Phase4F41Error("F40 result does not reference 72 immutable chunks")
    for reference in chunks:
        if not isinstance(reference, Mapping):
            raise Phase4F41Error("F40 chunk reference is malformed")
        path = workspace / str(reference["path"])
        chunk = _mapping(path, "F40 chunk")
        chunk_sha = _verify_self_hash(chunk, "chunkSha256", "F40 chunk")
        if chunk_sha != reference.get("sha256"):
            raise Phase4F41Error(f"F40 chunk digest changed: {path}")
        rows.extend(dict(row) for row in chunk["rows"])
    if len(rows) != 4608 or [int(row["ordinal"]) for row in rows] != list(range(4608)):
        raise Phase4F41Error("F40 rows are incomplete or reordered")
    return f39, f40_result, rows


def _blend_policy(row: Mapping[str, object], temperature: float) -> dict[int, float]:
    classic = _softmax_scores(row["classicCandidates"], temperature)
    search = _mcts_policy(row)
    actions = set(classic).union(search)
    policy = {
        action: POLICY_CLASSIC_WEIGHT * classic.get(action, 0.0)
        + POLICY_SEARCH_WEIGHT * search.get(action, 0.0)
        for action in actions
    }
    total = sum(policy.values())
    return {action: value / total for action, value in policy.items()}


def _policy_metrics(rows: Sequence[Mapping[str, object]], split: str, temperature: float) -> dict[str, float]:
    policies: list[dict[int, float]] = []
    ranks: list[int] = []
    for row in rows:
        if row.get("split") != split:
            continue
        policy = _blend_policy(row, temperature)
        policies.append(policy)
        ranked = sorted(policy, key=lambda action: (-policy[action], action))
        ranks.append(ranked.index(int(row["teacherActionIndex"])) + 1)
    return {
        "meanEntropy": mean(_entropy(policy) for policy in policies),
        "top1ProbabilityP95": _percentile([max(policy.values()) for policy in policies], 0.95),
        "classicActionTop1Coverage": mean(rank == 1 for rank in ranks),
        "classicActionTop5Coverage": mean(rank <= 5 for rank in ranks),
        "meanSupport": mean(len(policy) for policy in policies),
    }


def fit_policy_temperature(rows: Sequence[Mapping[str, object]]) -> tuple[float, dict[str, float]]:
    temperature = 1.0
    while temperature < 1_000_000.0:
        metrics = _policy_metrics(rows, "train", temperature)
        if (
            metrics["meanEntropy"] >= 0.55
            and metrics["top1ProbabilityP95"] <= POLICY_TEMPERATURE_FIT_TOP1_P95_MAXIMUM
        ):
            return temperature, metrics
        temperature *= POLICY_TEMPERATURE_MULTIPLIER
    raise Phase4F41Error("corrected policy temperature cannot satisfy the train fit gate")


def fit_regularized_isotonic(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            grouped[float(row["mcts"]["rootValue"])].append(float(row["outcome"]))
    groups = sorted(grouped.items())
    observations = sum(len(values) for _, values in groups)
    target_weight = observations / ISOTONIC_QUANTILE_BINS
    bins: list[dict[str, float]] = []
    current: list[tuple[float, list[float]]] = []
    current_weight = 0
    for index, (x, values) in enumerate(groups):
        current.append((x, values))
        current_weight += len(values)
        if current_weight >= target_weight and index < len(groups) - 1:
            outcomes = [value for _, items in current for value in items]
            bins.append(
                {
                    "minimum": current[0][0],
                    "maximum": current[-1][0],
                    "observationWeight": float(len(outcomes)),
                    "priorWeight": ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
                    "weight": float(len(outcomes)) + ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
                    "value": sum(outcomes) / (len(outcomes) + ISOTONIC_NEUTRAL_PRIOR_WEIGHT),
                }
            )
            current = []
            current_weight = 0
    if current:
        outcomes = [value for _, items in current for value in items]
        bins.append(
            {
                "minimum": current[0][0],
                "maximum": current[-1][0],
                "observationWeight": float(len(outcomes)),
                "priorWeight": ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
                "weight": float(len(outcomes)) + ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
                "value": sum(outcomes) / (len(outcomes) + ISOTONIC_NEUTRAL_PRIOR_WEIGHT),
            }
        )
    blocks: list[dict[str, float]] = []
    for item in bins:
        blocks.append(item)
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append(
                {
                    "minimum": left["minimum"],
                    "maximum": right["maximum"],
                    "observationWeight": left["observationWeight"] + right["observationWeight"],
                    "priorWeight": left["priorWeight"] + right["priorWeight"],
                    "weight": weight,
                    "value": (left["value"] * left["weight"] + right["value"] * right["weight"]) / weight,
                }
            )
    return blocks


def isotonic_predict(blocks: Sequence[Mapping[str, float]], value: float) -> float:
    for block in blocks:
        if value <= float(block["maximum"]):
            return float(block["value"])
    return float(blocks[-1]["value"])


def corrected_target(
    row: Mapping[str, object], temperature: float, blocks: Sequence[Mapping[str, float]]
) -> tuple[dict[int, float], float, dict[str, float]]:
    policy = _blend_policy(row, temperature)
    outcome = float(row["outcome"])
    search = isotonic_predict(blocks, float(row["mcts"]["rootValue"]))
    tempo = float(row["pathTempo"])
    value = VALUE_OUTCOME_WEIGHT * outcome + VALUE_SEARCH_WEIGHT * search + VALUE_PATH_TEMPO_WEIGHT * tempo
    return policy, value, {
        "terminalOutcome": outcome,
        "regularizedSearch": search,
        "pathTempo": tempo,
    }


def _summaries(
    rows: Sequence[Mapping[str, object]], temperature: float, blocks: Sequence[Mapping[str, float]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for split in ("train", "development"):
        selected = [row for row in rows if row.get("split") == split]
        policies: list[dict[int, float]] = []
        values: list[float] = []
        components: list[dict[str, float]] = []
        ranks: list[int] = []
        for row in selected:
            policy, value, component = corrected_target(row, temperature, blocks)
            policies.append(policy)
            values.append(value)
            components.append(component)
            ranked = sorted(policy, key=lambda action: (-policy[action], action))
            ranks.append(ranked.index(int(row["teacherActionIndex"])) + 1)
        output[split] = {
            "positions": len(selected),
            "policy": {
                "meanEntropy": mean(_entropy(policy) for policy in policies),
                "top1ProbabilityP95": _percentile([max(policy.values()) for policy in policies], 0.95),
                "classicActionTop1Coverage": mean(rank == 1 for rank in ranks),
                "classicActionTop5Coverage": mean(rank <= 5 for rank in ranks),
                "meanSupport": mean(len(policy) for policy in policies),
            },
            "value": {
                "mean": mean(values),
                "meanAbsolute": mean(abs(value) for value in values),
                "positiveRate": mean(value > 0 for value in values),
                "minimum": min(values),
                "maximum": max(values),
                "components": {
                    name: {
                        "mean": mean(component[name] for component in components),
                        "meanAbsolute": mean(abs(component[name]) for component in components),
                    }
                    for name in ("terminalOutcome", "regularizedSearch", "pathTempo")
                },
            },
            "byPlyBand": {
                band: sum(row.get("band") == band for row in selected)
                for band in ("early", "middle", "late")
            },
        }
    return output


def _mirror_audit(
    rows: Sequence[Mapping[str, object]], temperature: float, blocks: Sequence[Mapping[str, float]]
) -> dict[str, object]:
    policy_errors: list[float] = []
    value_errors: list[float] = []
    for row in rows:
        if "mirrorAudit" not in row:
            continue
        policy, value, _ = corrected_target(row, temperature, blocks)
        mirror_policy, mirror_value, _ = corrected_target(row["mirrorAudit"], temperature, blocks)
        restored = _mirror_policy_back(mirror_policy)
        actions = set(policy).union(restored)
        policy_errors.append(max(abs(policy.get(a, 0.0) - restored.get(a, 0.0)) for a in actions))
        value_errors.append(abs(value - mirror_value))
    if len(policy_errors) != 64:
        raise Phase4F41Error("F40 mirror audit rows are incomplete")
    return {
        "positions": len(policy_errors),
        "policyMaximumAbsoluteError": max(policy_errors),
        "valueMaximumAbsoluteError": max(value_errors),
    }


def _checks(
    development: Mapping[str, object], mirror: Mapping[str, object], acceptance: Mapping[str, float]
) -> dict[str, bool]:
    policy = development["policy"]
    value = development["value"]
    return {
        "policyTargetMeanEntropy": policy["meanEntropy"] >= acceptance["policyTargetMeanEntropyMinimum"],
        "policyTargetTop1ProbabilityP95": policy["top1ProbabilityP95"] <= acceptance["policyTargetTop1ProbabilityP95Maximum"],
        "classicActionTop5Coverage": policy["classicActionTop5Coverage"] >= acceptance["classicActionTop5CoverageMinimum"],
        "valueTargetMeanAbsolute": value["meanAbsolute"] <= acceptance["valueTargetMeanAbsoluteMaximum"],
        "valueTargetPositiveRateMinimum": value["positiveRate"] >= acceptance["valueTargetPositiveRateMinimum"],
        "valueTargetPositiveRateMaximum": value["positiveRate"] <= acceptance["valueTargetPositiveRateMaximum"],
        "mirrorPolicy": mirror["policyMaximumAbsoluteError"] <= acceptance["mirrorPolicyMaximumAbsoluteError"],
        "mirrorValue": mirror["valueMaximumAbsoluteError"] <= acceptance["mirrorValueMaximumAbsoluteError"],
    }


def run_phase4f41(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f41(workspace), "reused": True}
    f39, f40_result, rows = _load_inputs(workspace)
    implementation_sha256 = _implementation_sha256(workspace)
    temperature, temperature_fit = fit_policy_temperature(rows)
    blocks = fit_regularized_isotonic(rows)
    summaries = _summaries(rows, temperature, blocks)
    mirror = _mirror_audit(rows, temperature, blocks)
    acceptance = f39["targetProbe"]["acceptance"]
    checks = _checks(summaries["development"], mirror, acceptance)
    passed = all(checks.values())
    report = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f41-target-protocol-correction-report",
            "status": "completed",
            "implementationSha256": implementation_sha256,
            "f39ContractSha256": EXPECTED_F39_CONTRACT_SHA256,
            "f40GateSha256": EXPECTED_F40_GATE_SHA256,
            "f40ResultSha256": EXPECTED_F40_RESULT_SHA256,
            "f40GlobalGpuHours": f40_result["globalGpuHours"],
            "gpuHoursUsed": 0.0,
            "trainingStarted": False,
            "optimizerSteps": 0,
            "fitSplit": "train",
            "policyCalibration": {
                "classicTemperature": temperature,
                "fitTargetTop1ProbabilityP95Maximum": POLICY_TEMPERATURE_FIT_TOP1_P95_MAXIMUM,
                "fitMetrics": temperature_fit,
            },
            "valueCalibration": {
                "algorithm": "tie-preserving-quantile-pava-neutral-prior-v1",
                "requestedBins": ISOTONIC_QUANTILE_BINS,
                "blocks": blocks,
                "neutralPriorWeightPerBin": ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
            },
            "summaries": summaries,
            "mirrorAudit": mirror,
            "checks": checks,
            "targetProtocolPassed": passed,
        },
        "reportSha256",
    )
    _write_immutable(workspace / REPORT, report, "F41 correction report")

    contract_sha: str | None = None
    if passed:
        contract = _self_hashed(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": "phase4f41-target-protocol-frozen-contract",
                "status": "frozen",
                "implementationSha256": implementation_sha256,
                "f39ContractSha256": EXPECTED_F39_CONTRACT_SHA256,
                "f40GateSha256": EXPECTED_F40_GATE_SHA256,
                "f40ResultSha256": EXPECTED_F40_RESULT_SHA256,
                "correctionReportPath": REPORT.as_posix(),
                "correctionReportSha256": report["reportSha256"],
                "dataset": {
                    "trainPositions": 4096,
                    "developmentPositions": 512,
                    "finalPositions": 0,
                    "source": "immutable-phase4f40-search-chunks",
                },
                "policyTarget": {
                    "classicWeight": POLICY_CLASSIC_WEIGHT,
                    "searchWeight": POLICY_SEARCH_WEIGHT,
                    "classicTemperature": temperature,
                    "fitSplit": "train",
                    "fitTop1ProbabilityP95Maximum": POLICY_TEMPERATURE_FIT_TOP1_P95_MAXIMUM,
                },
                "valueTarget": {
                    "terminalOutcomeWeight": VALUE_OUTCOME_WEIGHT,
                    "regularizedSearchWeight": VALUE_SEARCH_WEIGHT,
                    "pathTempoWeight": VALUE_PATH_TEMPO_WEIGHT,
                    "neutralResidualWeight": VALUE_NEUTRAL_WEIGHT,
                    "calibrationAlgorithm": "tie-preserving-quantile-pava-neutral-prior-v1",
                    "quantileBins": ISOTONIC_QUANTILE_BINS,
                    "neutralPriorWeightPerBin": ISOTONIC_NEUTRAL_PRIOR_WEIGHT,
                },
                "acceptance": acceptance,
                "authorizations": {
                    "datasetGenerationAuthorized": True,
                    "cudaTrainingAuthorized": False,
                    "formalTrainingAuthorized": False,
                    "promotionAuthorized": False,
                },
                "nextStage": {
                    "name": "Phase 4F42 corrected early-curriculum dataset generation",
                    "requiresSeparateUserConfirmation": True,
                },
            },
            "contractSha256",
        )
        _write_immutable(workspace / CONTRACT, contract, "F41 frozen target protocol")
        contract_sha = str(contract["contractSha256"])

    gate = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f41-target-protocol-correction-gate",
            "status": "completed",
            "outcomeBand": "target-protocol-corrected" if passed else "target-protocol-correction-failed",
            "reportPath": REPORT.as_posix(),
            "reportSha256": report["reportSha256"],
            "contractPath": CONTRACT.as_posix() if passed else None,
            "contractSha256": contract_sha,
            "checks": checks,
            "developmentSummary": summaries["development"],
            "mirrorAudit": mirror,
            "datasetGenerationAuthorized": passed,
            "cudaTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "nextStage": {
                "name": (
                    "Phase 4F42 corrected early-curriculum dataset generation"
                    if passed
                    else "Phase 4F41 target protocol correction v2"
                ),
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )
    _write_immutable(workspace / GATE, gate, "F41 correction gate")
    return {**gate, "reused": False}


def verify_phase4f41(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    gate = _mapping(workspace / GATE, "F41 gate")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F41 gate")
    report = _mapping(workspace / REPORT, "F41 report")
    report_sha = _verify_self_hash(report, "reportSha256", "F41 report")
    if gate.get("reportSha256") != report_sha or report.get("trainingStarted") is not False:
        raise Phase4F41Error("F41 report and gate are inconsistent")
    contract_sha = gate.get("contractSha256")
    if gate.get("datasetGenerationAuthorized") is True:
        contract = _mapping(workspace / CONTRACT, "F41 contract")
        actual = _verify_self_hash(contract, "contractSha256", "F41 contract")
        if actual != contract_sha or contract.get("status") != "frozen":
            raise Phase4F41Error("F41 frozen contract is inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f41-target-protocol-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "contractPath": gate.get("contractPath"),
        "contractSha256": contract_sha,
        "outcomeBand": gate["outcomeBand"],
        "datasetGenerationAuthorized": gate["datasetGenerationAuthorized"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_phase4f41(args.workspace_root) if args.command == "run" else verify_phase4f41(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
