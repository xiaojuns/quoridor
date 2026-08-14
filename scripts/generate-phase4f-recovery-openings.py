#!/usr/bin/env python3
"""Generate or verify the Phase 4F recovery evaluation suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_phase4_openings import generate_recovery_suites
from quoridor.openings import (
    load_opening_suite,
    opening_suite_to_payload,
    save_opening_suite,
    validate_suite_collection,
)
from quoridor.run_manifest import (
    atomic_replace_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


DEFAULT_ROOT = ROOT / "training/fixtures/openings/recovery-v1"


def _manifest(suites, paths, legacy) -> dict[str, object]:
    core = {
        "schemaVersion": 1,
        "kind": "phase4f-recovery-opening-partition",
        "status": "frozen",
        "trainingUseAllowed": False,
        "oldFinalContamination": "diagnosed-and-historical-only",
        "legacySuites": [
            {
                "name": suite.name,
                "suiteHash": suite.content_hash,
                "fileSha256": sha256_file(path),
            }
            for suite, path in legacy
        ],
        "recoverySuites": [
            {
                "name": suite.name,
                "suiteHash": suite.content_hash,
                "file": path.name,
                "fileSha256": sha256_file(path),
                "openings": len(suite.openings),
            }
            for suite, path in zip(suites, paths, strict=True)
        ],
        "partitionContract": "content-and-horizontal-mirror-orbit-disjoint-v1",
    }
    return {**core, "manifestSha256": sha256_bytes(canonical_json_bytes(core))}


def run(output_root: Path, *, check: bool) -> dict[str, object]:
    legacy_paths = [
        ROOT / "training/fixtures/openings/dev.json",
        ROOT / "training/fixtures/openings/final.json",
    ]
    legacy = [
        load_opening_suite(path, expected_suite=name)
        for path, name in zip(legacy_paths, ("dev", "final"), strict=True)
    ]
    generated = generate_recovery_suites(legacy)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [output_root / f"{suite.name}.json" for suite in generated]
    manifest_path = output_root / "manifest.json"
    if check:
        loaded = [
            load_opening_suite(path, expected_suite=suite.name)
            for path, suite in zip(paths, generated, strict=True)
        ]
        for actual, expected in zip(loaded, generated, strict=True):
            if opening_suite_to_payload(actual) != opening_suite_to_payload(expected):
                raise RuntimeError(f"frozen {actual.name} differs from generation")
    else:
        loaded = []
        for path, suite in zip(paths, generated, strict=True):
            save_opening_suite(suite, path)
            path.chmod(0o644)
            loaded.append(load_opening_suite(path, expected_suite=suite.name))
    validate_suite_collection(*legacy, *loaded)
    expected_manifest = _manifest(
        loaded,
        paths,
        tuple(zip(legacy, legacy_paths, strict=True)),
    )
    if check:
        if json.loads(manifest_path.read_bytes()) != expected_manifest or manifest_path.read_bytes() != canonical_json_bytes(expected_manifest):
            raise RuntimeError("recovery opening manifest differs from generation")
    else:
        atomic_replace_bytes(manifest_path, canonical_json_bytes(expected_manifest))
        manifest_path.chmod(0o644)
    return expected_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run(args.output_root, check=args.check)
    print(canonical_json_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
