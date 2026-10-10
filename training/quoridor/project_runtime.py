"""Run provenance, JSONL events and a cooperative wall-clock budget for M0."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .project_config import PROJECT_ROOT, ProjectConfig


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"


def _write_once(path: Path, payload: dict) -> None:
    encoded = _json(payload)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


class BudgetExpired(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic):
        if type(seconds) not in (int, float):
            raise ValueError("budget seconds must be finite and positive")
        try:
            self.seconds = float(seconds)
        except OverflowError as error:
            raise ValueError("budget seconds must be finite and positive") from error
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("budget seconds must be finite and positive")
        self._clock = clock
        self._started = clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.seconds - self.elapsed_seconds)

    @property
    def expired(self) -> bool:
        return self.elapsed_seconds >= self.seconds

    def check(self) -> None:
        if self.expired:
            raise BudgetExpired("wall-clock budget exhausted")


def _provenance() -> dict:
    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout

    files = []
    names = set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    for name in sorted(names - {b""}):
        relative = name.decode("utf-8")
        path = PROJECT_ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        files.append({"path": relative, "sha256": digest})
    packages = {}
    for package in ("torch", "numpy", "PyYAML", "pytest", "hypothesis"):
        packages[package] = importlib.metadata.version(package)
    return {
        "git_commit": git("rev-parse", "HEAD").decode("ascii").strip(),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": hashlib.sha256(_json({"files": files}).encode("utf-8")).hexdigest(),
        "source_files": files,
        "dependency_lock_sha256": hashlib.sha256((PROJECT_ROOT / "training/uv.lock").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


class RunSession:
    """Single-writer run directory. Workers must call deadline.check themselves."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        run_root: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.deadline = Deadline(config.document["budget"]["wall_time_seconds"], clock=clock)
        root = run_root if run_root is not None else PROJECT_ROOT / config.document["output"]["run_root"]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
        self.path = root / config.mode / run_id
        self._sequence = 0
        self._started = False
        self._finished = False

    def __enter__(self) -> RunSession:
        if self._started:
            raise RuntimeError("a run session cannot be entered twice")
        provenance = _provenance()
        self.path.mkdir(parents=True, exist_ok=False)
        with (self.path / "config.yaml").open("xb") as stream:
            stream.write(self.config.yaml_bytes)
        _write_once(self.path / "run.json", {
            "schema_version": 1, "run_id": self.path.name,
            "started_at": _utc(), "run_dir": str(self.path.resolve()),
            "config_path": str(self.config.source_path), "config_sha256": self.config.sha256,
            "config": self.config.document, **provenance,
        })
        self._started = True
        self.log("run_started")
        return self

    def log(self, event: str, **fields: object) -> None:
        if not self._started or self._finished:
            raise RuntimeError("events require an active run session")
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a nonempty string")
        reserved = {"schema_version", "sequence", "timestamp", "elapsed_seconds", "event"}
        if reserved.intersection(fields):
            raise ValueError("event fields cannot override reserved metadata")
        encoded = _json({
            "schema_version": 1, "sequence": self._sequence, "timestamp": _utc(),
            "elapsed_seconds": self.deadline.elapsed_seconds, "event": event, **fields,
        })
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded)
        self._sequence += 1

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if isinstance(exception, BudgetExpired) or (exception is None and self.deadline.expired):
            status = "budget_exhausted"
        elif isinstance(exception, KeyboardInterrupt):
            status = "interrupted"
        elif exception is not None:
            status = "failed"
        else:
            status = "completed"
        self.log("run_finished", status=status, error=None if exception is None else str(exception))
        _write_once(self.path / "summary.json", {
            "schema_version": 1, "status": status, "finished_at": _utc(),
            "elapsed_seconds": self.deadline.elapsed_seconds,
        })
        self._finished = True
        return False
