"""Verify a completed F46 screen without rebuilding its pre-run opening pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .openings import load_opening_suite
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import _mapping, _verify_self_hash
from .phase4f45_curriculum_pilot_review import (
    DELTA_AUDIT,
    F46_CONTRACT,
    GATE as F45_GATE,
    REPORT as F45_REPORT,
    SCREEN_GPU_HOURS,
    SCREEN_SUITE,
    _f21_evidence,
    _f44_evidence,
    _source,
    _workspace,
    loss_review,
    model_delta_audit,
)
from .phase4f46_curriculum_lineage_screen import GATE, gate_from_result
from .run_manifest import canonical_json_bytes, sha256_file


SCHEMA_VERSION = 1


class Phase4F46ResultVerifyError(RuntimeError):
    """Raised when completed F46 evidence fails post-run verification."""


def verify_phase4f46_result(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _mapping(workspace / F46_CONTRACT, "F46 contract")
    _verify_self_hash(contract, "contractSha256", "F46 contract")
    f45_report = _mapping(workspace / F45_REPORT, "F45 review")
    _verify_self_hash(f45_report, "reportSha256", "F45 review")
    f45_gate = _mapping(workspace / F45_GATE, "F45 gate")
    _verify_self_hash(f45_gate, "gateSha256", "F45 gate")
    delta = _mapping(workspace / DELTA_AUDIT, "F45 delta audit")
    _verify_self_hash(delta, "auditSha256", "F45 delta audit")

    source = _source(workspace)
    f44_report, f44_state = _f44_evidence(workspace)
    _f21_payload, f21_state = _f21_evidence(workspace)
    recomputed_delta = model_delta_audit(f21_state, f44_state)
    recomputed_delta.pop("auditSha256", None)
    stored_delta = dict(delta)
    stored_delta.pop("auditSha256", None)
    suite_path = workspace / SCREEN_SUITE
    suite = load_opening_suite(suite_path, expected_suite="recovery-fresh-dev-v1")
    opening = contract.get("openingSuite")
    budget = contract.get("budget")
    if (
        contract.get("status") != "frozen"
        or contract.get("source") != source
        or contract.get("authorizations", {}).get("f46LineageScreenAuthorized") is not True
        or contract.get("authorizations", {}).get("formalTrainingAuthorized") is not False
        or not isinstance(opening, Mapping)
        or opening.get("selectedPath") != SCREEN_SUITE.as_posix()
        or opening.get("selectedFileSha256") != sha256_file(suite_path)
        or opening.get("selectedSuiteHash") != suite.content_hash
        or opening.get("selectedOpeningPairs") != len(suite.openings)
        or len(suite.openings) != 32
        or stored_delta != recomputed_delta
        or loss_review(f44_report) != f45_report.get("lossReview")
        or f45_report.get("source") != source
        or f45_report.get("f46ContractSha256") != contract.get("contractSha256")
        or f45_report.get("reviewPassed") is not True
        or f45_gate.get("reportSha256") != f45_report.get("reportSha256")
        or f45_gate.get("f46ContractSha256") != contract.get("contractSha256")
        or f45_gate.get("f46LineageScreenAuthorized") is not True
        or not isinstance(budget, Mapping)
    ):
        raise Phase4F46ResultVerifyError("F45/F46 frozen evidence changed")

    gate = _mapping(workspace / GATE, "F46 gate")
    _verify_self_hash(gate, "gateSha256", "F46 gate")
    if gate != gate_from_result(workspace, contract):
        raise Phase4F46ResultVerifyError("F46 gate differs from the verified arena result")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    baseline = float(budget["baselineGlobalHours"])
    if ledger.total_hours < baseline or ledger.total_hours > baseline + SCREEN_GPU_HOURS:
        raise Phase4F46ResultVerifyError("F46 completed outside its frozen GPU budget")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f46-curriculum-lineage-result-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate["gateSha256"],
        "resultPath": gate["resultPath"],
        "resultSha256": gate["resultSha256"],
        "statistics": gate["statistics"],
        "outcomeBand": gate["outcomeBand"],
        "formalTrainingAuthorized": False,
        "promotionAuthorized": False,
        "globalGpuHours": ledger.total_hours,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    print(canonical_json_bytes(verify_phase4f46_result(args.workspace_root)).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
