#!/usr/bin/env python3
"""Repository-local entry point for Phase 4D teacher replay generation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from quoridor.teacher_data import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
