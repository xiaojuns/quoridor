from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


class ArtifactBudgetError(RuntimeError):
    pass


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def atomic_torch_save(
    payload: Any,
    destination: str | Path,
    *,
    artifact_root: str | Path,
    limit_bytes: int,
) -> Path:
    destination_path = Path(destination)
    root = Path(artifact_root)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    previous_size = destination_path.stat().st_size if destination_path.exists() else 0
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(payload, temporary_path)
        new_size = temporary_path.stat().st_size
        current_size = directory_size(root) - new_size - previous_size
        projected_size = current_size + new_size
        if projected_size > limit_bytes:
            raise ArtifactBudgetError(
                f"artifact budget exceeded: {projected_size} > {limit_bytes} bytes",
            )
        os.replace(temporary_path, destination_path)
        temporary_path = None
        return destination_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_json_save(payload: Any, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=True, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_path, destination_path)
        return destination_path
    finally:
        temporary_path.unlink(missing_ok=True)
