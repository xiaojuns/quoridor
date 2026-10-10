"""Reject unsupported or drifted M0 environments before running checks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys


def main() -> int:
    expected = {
        "torch": "2.7.1+cpu",
        "numpy": "2.2.6",
        "PyYAML": "6.0.3",
        "pytest": "8.4.2",
        "hypothesis": "6.142.4",
    }
    errors: list[str] = []
    if sys.version_info[:2] != (3, 10):
        errors.append("Python 3.10 is required")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        errors.append("M0 supports Linux x86_64 CPU only")
    installed: dict[str, str | None] = {}
    for package, version in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[package] = actual
        if actual != version:
            errors.append(f"{package}: expected {version}, found {actual}")

    result = {
        "status": "error" if errors else "ok",
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "dependencies": installed,
        "errors": errors,
    }
    print(json.dumps(result, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
