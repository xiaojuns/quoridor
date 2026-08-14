"""Reusable, fail-closed prerequisite gates for Phase 4D workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .run_manifest import fingerprint_sources, load_run_manifest, sha256_file
from .training_memory_smoke import (
    RUN_KIND as MEMORY_SMOKE_KIND,
    load_immutable_report as load_memory_smoke_report,
)


class Phase4GateError(RuntimeError):
    """Raised when no artifact satisfies a required Phase 4 gate."""


def _validate_memory_smoke_path(
    report_path: Path,
    *,
    workspace: Path,
    current_source_sha256: str,
) -> dict[str, object]:
    if report_path.is_symlink():
        raise Phase4GateError("memory-smoke report is a symlink")
    report = load_memory_smoke_report(report_path)
    gates = report.get("gates")
    contract = report.get("contract")
    if (
        report.get("kind") != MEMORY_SMOKE_KIND
        or report.get("status") != "completed"
        or report.get("passed") is not True
        or not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
        or not isinstance(contract, Mapping)
    ):
        raise Phase4GateError("memory-smoke gates did not pass")
    network = contract.get("network")
    precision = contract.get("precision")
    if (
        not isinstance(network, Mapping)
        or network.get("inputChannels") != 16
        or network.get("boardSize") != 9
        or network.get("actionSize") != 209
        or network.get("channels") != 192
        or network.get("residualBlocks") != 16
        or contract.get("batchSize") != 512
        or not isinstance(precision, Mapping)
        or precision.get("ampRequired") is not True
        or precision.get("autocastDtype") != "float16"
        or precision.get("parameterDtype") != "float32"
    ):
        raise Phase4GateError(
            "memory-smoke workload is not the frozen 192x16/batch512 AMP contract"
        )
    manifest_path = report_path.parent / "run-manifest.json"
    manifest = load_run_manifest(manifest_path)
    provenance = manifest.payload.get("provenance")
    source = provenance.get("source") if isinstance(provenance, Mapping) else None
    if (
        report.get("runManifestSha256") != manifest.sha256
        or not isinstance(source, Mapping)
        or source.get("sha256") != current_source_sha256
    ):
        raise Phase4GateError(
            "memory-smoke does not bind the current training source"
        )
    completed_at = report.get("completedAt")
    if not isinstance(completed_at, str) or not completed_at:
        raise Phase4GateError("memory-smoke completion time is missing")
    return {
        "path": report_path.relative_to(workspace).as_posix(),
        "fileSha256": sha256_file(report_path),
        "reportCoreSha256": report["reportSha256"],
        "runManifestSha256": manifest.sha256,
        "sourceSha256": current_source_sha256,
        "completedAt": completed_at,
        "gates": dict(gates),
    }


def verify_memory_smoke_evidence(
    evidence: Mapping[str, object],
    control_root: str | Path,
    workspace_root: str | Path,
) -> dict[str, object]:
    """Revalidate one lineage-bound memory-smoke evidence record exactly."""

    if not isinstance(evidence, Mapping):
        raise Phase4GateError("memory-smoke evidence must be a mapping")
    root = Path(control_root).resolve()
    workspace = Path(workspace_root).resolve()
    relative = evidence.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise Phase4GateError("memory-smoke evidence path is invalid")
    report_path = (workspace / relative).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as error:
        raise Phase4GateError("memory-smoke evidence path is outside Phase 4") from error
    actual = _validate_memory_smoke_path(
        report_path,
        workspace=workspace,
        current_source_sha256=fingerprint_sources(workspace).sha256,
    )
    if dict(evidence) != actual:
        raise Phase4GateError("memory-smoke evidence does not match its artifacts")
    return actual


def find_valid_memory_smoke(
    control_root: str | Path, workspace_root: str | Path
) -> dict[str, object]:
    """Return the newest current-source 192x16/batch-512 AMP smoke.

    Every candidate is self-hash validated by ``load_immutable_report`` and is
    then bound to its immutable run manifest and the current default training
    source fingerprint.  Corrupt or stale candidates never satisfy the gate.
    """

    root = Path(control_root).resolve()
    workspace = Path(workspace_root).resolve()
    current_source_sha256 = fingerprint_sources(workspace).sha256
    valid: list[dict[str, object]] = []
    for report_path in sorted((root / "training-memory-smoke").glob("*/report.json")):
        try:
            valid.append(
                _validate_memory_smoke_path(
                    report_path,
                    workspace=workspace,
                    current_source_sha256=current_source_sha256,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    if not valid:
        raise Phase4GateError(
            "no completed current-source 192x16/batch512 CUDA AMP memory-smoke "
            "report passed all gates"
        )
    return max(valid, key=lambda item: (str(item["completedAt"]), str(item["path"])))


__all__ = [
    "Phase4GateError",
    "find_valid_memory_smoke",
    "verify_memory_smoke_evidence",
]
