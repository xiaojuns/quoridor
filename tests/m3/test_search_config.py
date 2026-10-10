"""The new search consumes validated YAML snapshots and frozen arena settings."""

from pathlib import Path

import pytest
import yaml

from quoridor.project_config import ConfigError, load_project_config
from quoridor.search_config import search_config_from_project


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("project_mode,search_mode", (("smoke", "selfplay"), ("experiment", "selfplay"), ("arena", "evaluation")))
def test_repository_yaml_maps_all_search_fields_and_seed(project_mode, search_mode):
    snapshot = load_project_config(PROJECT_ROOT / f"configs/{project_mode}.yaml")
    document = snapshot.document
    config = search_config_from_project(snapshot)
    assert config.mode == search_mode
    assert config.seed == document["seed"]
    for name in ("simulations", "c_puct", "root_noise", "temperature"):
        assert getattr(config, name) == document["search"][name]
    assert config.dirichlet_alpha == 0.3 and config.dirichlet_fraction == 0.25
    if project_mode == "arena":
        assert config.root_noise is False and config.temperature == 0


def test_search_uses_immutable_snapshot_even_if_source_file_or_document_copy_changes(tmp_path):
    path = tmp_path / "smoke.yaml"
    path.write_bytes((PROJECT_ROOT / "configs/smoke.yaml").read_bytes())
    snapshot = load_project_config(path)
    before = search_config_from_project(snapshot)
    edited = snapshot.document
    edited["search"]["simulations"] = 123
    edited["seed"] = 42
    path.write_text(yaml.safe_dump(edited), encoding="utf-8")
    assert search_config_from_project(snapshot) == before
    refreshed = search_config_from_project(load_project_config(path))
    assert refreshed.simulations == 123 and refreshed.seed == 42


def test_selfplay_can_explicitly_disable_noise_and_temperature_in_valid_yaml(tmp_path):
    document = load_project_config(PROJECT_ROOT / "configs/smoke.yaml").document
    document["search"].update(root_noise=False, temperature=0.0)
    path = tmp_path / "deterministic-selfplay.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = search_config_from_project(load_project_config(path))
    assert config.mode == "selfplay"
    assert config.root_noise is False and config.temperature == 0.0


def test_frozen_arena_yaml_cannot_be_tuned_before_search_bridge(tmp_path):
    source = PROJECT_ROOT / "configs/arena.yaml"
    snapshot = load_project_config(source)
    path = tmp_path / "arena.yaml"
    path.write_bytes(source.read_bytes().replace(b"simulations: 256", b"simulations: 257"))
    with pytest.raises(ConfigError, match="reviewed new protocol"):
        load_project_config(path)
    assert search_config_from_project(snapshot).simulations == 256


@pytest.mark.parametrize("bad", (None, {}, "configs/smoke.yaml"))
def test_bridge_requires_validated_snapshot(bad):
    with pytest.raises(TypeError, match="ProjectConfig"):
        search_config_from_project(bad)
