from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quoridor.feature_spec import ENCODER_VERSION
from quoridor.model_config import ModelConfigError, load_model_config
from quoridor.policy_value import NetworkConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def document():
    return {
        "schema_version": 1,
        "encoder_version": ENCODER_VERSION,
        "architecture": NetworkConfig().to_dict(),
    }


def write_config(tmp_path, value):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_repository_model_yaml_explicitly_declares_the_baseline_architecture():
    assert load_model_config(PROJECT_ROOT / "configs/model.yaml") == NetworkConfig()


def test_yaml_selects_a_small_valid_architecture_without_starting_a_job(tmp_path):
    value = document()
    value["architecture"] = NetworkConfig.smoke().to_dict()
    assert load_model_config(write_config(tmp_path, value)) == NetworkConfig.smoke()


@pytest.mark.parametrize("field,bad", [
    ("schema_version", True), ("schema_version", 1.0), ("schema_version", 2),
    ("schema_version", "1"), ("encoder_version", "legacy-16"),
    ("encoder_version", 16), ("encoder_version", None),
    ("architecture", None), ("architecture", []),
])
def test_yaml_rejects_invalid_schema_encoder_and_architecture_types(tmp_path, field, bad):
    value = document()
    value[field] = bad
    with pytest.raises(ModelConfigError, match=field):
        load_model_config(write_config(tmp_path, value))


@pytest.mark.parametrize("field,bad", [
    ("channels", True), ("channels", 8.0), ("channels", "8"),
    ("blocks", 0), ("input_channels", 14), ("value_hidden", float("nan")),
    ("policy_channels", -1), ("model_version", "old-resnet"),
])
def test_yaml_does_not_coerce_architecture_fields(tmp_path, field, bad):
    value = document()
    value["architecture"][field] = bad
    with pytest.raises(ModelConfigError, match=field):
        load_model_config(write_config(tmp_path, value))


@pytest.mark.parametrize("level", ["document", "architecture"])
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_yaml_requires_all_and_only_declared_fields(tmp_path, level, mutation):
    value = document()
    target = value if level == "document" else value["architecture"]
    if mutation == "missing":
        del target[next(iter(target))]
    else:
        target["typo"] = 1
    with pytest.raises(ModelConfigError):
        load_model_config(write_config(tmp_path, value))


@pytest.mark.parametrize("content,match", [
    ("schema_version: 1\nschema_version: 1\n", "duplicate"),
    ("architecture:\n  channels: 8\n  channels: 64\n", "duplicate"),
    ("1: value\n", "keys must be strings"),
    ("!!python/object:builtins.object {}", "cannot read"),
    ("[invalid", "cannot read"),
    ("- list\n- config", "exactly"),
    ("", "exactly"),
])
def test_yaml_rejects_ambiguous_unsafe_and_malformed_documents(tmp_path, content, match):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ModelConfigError, match=match):
        load_model_config(path)


def test_file_errors_are_model_configuration_errors(tmp_path):
    with pytest.raises(ModelConfigError, match="cannot read"):
        load_model_config(tmp_path / "missing.yaml")
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ModelConfigError, match="cannot read"):
        load_model_config(path)
    loop = tmp_path / "loop.yaml"
    loop.symlink_to(loop.name)
    with pytest.raises(ModelConfigError, match="cannot read"):
        load_model_config(loop)
    with pytest.raises(ModelConfigError, match="cannot read"):
        load_model_config("bad\x00path.yaml")
