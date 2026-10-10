"""Explicit rule-version boundary for the new player and historical assets.

This gate verifies rule/codec/value identity only. Checkpoint architecture,
feature versions, content hashes and replay lineage still require their owning
loaders' checks in M2/M4; matching these three strings is not full validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType


RULES_VERSION = "quoridor-v1"
CODEC_VERSION = "az209-v1"
VALUE_PERSPECTIVE = "current-player-v1"
LEGACY_RULES_VERSION = "legacy-repetition-loss-v1"
SUPPORTED_BACKENDS = ("python",)


class IncompatibleRulesError(ValueError):
    """An asset is unversioned or belongs to a different rules contract."""


class UnsupportedBackendError(ValueError):
    """A backend has not implemented and passed the new rules contract."""


def contract_metadata() -> dict[str, str]:
    return {
        "rules_version": RULES_VERSION,
        "codec_version": CODEC_VERSION,
        "value_perspective": VALUE_PERSPECTIVE,
    }


def require_compatible_artifact(metadata: Mapping[str, object]) -> None:
    """Reject legacy/unversioned metadata; never infer compatibility from 209 IDs."""
    if not isinstance(metadata, Mapping):
        raise IncompatibleRulesError("artifact metadata must be a mapping")
    for field, expected in contract_metadata().items():
        actual = metadata.get(field)
        if type(actual) is not str or actual != expected:
            raise IncompatibleRulesError(
                f"{field} must be {expected!r}, got {actual!r}; "
                "legacy data needs an explicitly validated migration"
            )


def load_engine(
    backend: str = "python", *, rules_version: str = RULES_VERSION,
) -> ModuleType:
    """Return only a backend certified for quoridor-v1; no legacy fallback."""
    if type(rules_version) is not str or rules_version != RULES_VERSION:
        raise IncompatibleRulesError(f"unsupported rules version: {rules_version!r}")
    if type(backend) is not str or backend not in SUPPORTED_BACKENDS:
        raise UnsupportedBackendError(
            f"backend {backend!r} is not supported for {RULES_VERSION}; "
            "only python is implemented; native and browser remain legacy"
        )
    from . import engine

    return engine
