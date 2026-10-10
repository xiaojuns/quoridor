"""Strict YAML entry for the M2 architecture; this does not start a run."""

from __future__ import annotations

from pathlib import Path

import yaml

from .feature_spec import ENCODER_VERSION
from .policy_value import NetworkConfig
from .project_config import ConfigError, _UniqueKeyLoader


class ModelConfigError(ConfigError):
    """Unreadable, ambiguous, or incompatible model configuration."""


def load_model_config(path: str | Path) -> NetworkConfig:
    """Load explicit architecture fields without any default-filled metadata."""
    try:
        source = Path(path).resolve()
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, ValueError, RuntimeError, yaml.YAMLError) as error:
        raise ModelConfigError(f"cannot read model YAML: {error}") from error

    expected = {"schema_version", "encoder_version", "architecture"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ModelConfigError("model config requires exactly schema_version, encoder_version and architecture")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ModelConfigError("schema_version must be integer 1")
    if type(document["encoder_version"]) is not str or document["encoder_version"] != ENCODER_VERSION:
        raise ModelConfigError(f"encoder_version must be {ENCODER_VERSION!r}")
    try:
        return NetworkConfig.from_dict(document["architecture"])
    except (TypeError, ValueError) as error:
        raise ModelConfigError(f"invalid architecture: {error}") from error


__all__ = ["ModelConfigError", "load_model_config"]
