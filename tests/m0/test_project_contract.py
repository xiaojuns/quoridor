from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings, strategies as st

from quoridor.codec import decode_action, encode_action, mirror_action_index
from quoridor.project_cli import main
from quoridor.project_config import ConfigError, PROJECT_ROOT, load_project_config
from quoridor.project_runtime import BudgetExpired, Deadline, RunSession


def modified_config(tmp_path: Path, section: str, key: str, value: object) -> Path:
    document = load_project_config(PROJECT_ROOT / "configs/smoke.yaml").document
    document[section][key] = value
    path = tmp_path / "modified.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["smoke", "arena", "experiment"])
def test_example_config_and_cli_validate(mode, capsys):
    path = PROJECT_ROOT / f"configs/{mode}.yaml"
    config = load_project_config(path, expected_mode=mode)
    assert config.document["rules"]["action_size"] == 209
    assert config.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert main(["check-config", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "config_validated"


@pytest.mark.parametrize("mode,milestone", [("smoke", "M4"), ("arena", "M5"), ("experiment", "M6")])
def test_unimplemented_entry_does_not_invoke_a_legacy_job(mode, milestone, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a stub must not create a run")

    monkeypatch.setattr(RunSession, "__enter__", forbidden)
    assert main([mode, "--config", str(PROJECT_ROOT / f"configs/{mode}.yaml")]) == 3
    output = capsys.readouterr()
    assert output.out == ""
    event = json.loads(output.err)
    assert event["event"] == "not_implemented"
    assert event["required_milestone"] == milestone


@pytest.mark.parametrize("section,key,value", [
    ("rules", "board_size", 7), ("rules", "value_perspective", "player-zero"),
    ("budget", "wall_time_seconds", 0), ("budget", "wall_time_seconds", float("nan")),
    ("budget", "wall_time_seconds", float("inf")), ("budget", "wall_time_seconds", True),
    ("budget", "wall_time_seconds", 10**400),
    ("budget", "max_game_plies", 1.5), ("budget", "max_game_plies", False),
    ("output", "run_root", "../escape"), ("output", "run_root", "/tmp/escape"),
    ("output", "run_root", "training/runs"), ("search", "simulations", True),
    ("output", "run_root", "runs\x00"),
    ("search", "temperature", -1), ("search", "root_noise", "false"),
    ("search", "typo", 1),
])
def test_config_rejects_invalid_or_ambiguous_values(tmp_path, section, key, value):
    with pytest.raises(ConfigError):
        load_project_config(modified_config(tmp_path, section, key, value))


def test_config_rejects_duplicate_keys_and_unsafe_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: smoke\nmode: arena\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_project_config(path)
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read YAML"):
        load_project_config(path)


def test_config_has_no_mutable_snapshot_or_cross_mode_override():
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    config.document["rules"]["board_size"] = 7
    assert config.document["rules"]["board_size"] == 9
    with pytest.raises(ConfigError, match="expected mode"):
        load_project_config(config.source_path, expected_mode="arena")


def test_arena_requires_exact_frozen_bytes_and_opening_digest(tmp_path, monkeypatch):
    import quoridor.project_config as config_module

    original = PROJECT_ROOT / "configs/arena.yaml"
    copied = tmp_path / "arena.yaml"
    copied.write_bytes(original.read_bytes())
    load_project_config(copied)
    copied.write_bytes(original.read_bytes() + b"\n# changed protocol\n")
    with pytest.raises(ConfigError, match="differs"):
        load_project_config(copied)
    copied.write_bytes(original.read_bytes().replace(b"48f45b3d", b"00000000"))
    digest_file = tmp_path / "arena.sha256"
    digest_file.write_text(hashlib.sha256(copied.read_bytes()).hexdigest(), encoding="ascii")
    monkeypatch.setattr(config_module, "ARENA_DIGEST_PATH", digest_file)
    with pytest.raises(ConfigError, match="opening suite SHA-256"):
        load_project_config(copied)


def test_invalid_cli_config_is_nonzero_jsonl(tmp_path, capsys):
    assert main(["smoke", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert json.loads(capsys.readouterr().err)["event"] == "config_error"


def test_symlink_loop_config_path_is_a_json_error(tmp_path, capsys):
    path = tmp_path / "loop.yaml"
    path.symlink_to(path.name)
    assert main(["check-config", "--config", str(path)]) == 2
    assert json.loads(capsys.readouterr().err)["event"] == "config_error"


@given(st.integers(min_value=0, max_value=208))
def test_action_contract_roundtrip_and_mirror_are_bijections(action):
    assert encode_action(decode_action(action)) == action
    assert mirror_action_index(mirror_action_index(action)) == action


@given(st.floats(min_value=0.01, max_value=3600, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_deadline_uses_elapsed_time_and_exact_boundary(seconds):
    now = [0.0]
    deadline = Deadline(seconds, clock=lambda: now[0])
    now[0] = seconds / 2
    deadline.check()
    assert deadline.remaining_seconds == pytest.approx(seconds / 2)
    now[0] = seconds
    with pytest.raises(BudgetExpired):
        deadline.check()
    assert deadline.remaining_seconds == 0


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf"), 10**400])
def test_deadline_requires_a_real_budget(seconds):
    with pytest.raises(ValueError):
        Deadline(seconds)


@pytest.mark.parametrize("ending", ["completed", "failed", "interrupted", "budget_exhausted"])
def test_runtime_has_provenance_jsonl_and_truthful_finish(tmp_path, ending):
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    now = [0.0]
    run = RunSession(config, run_root=tmp_path, clock=lambda: now[0])
    exceptions = {"failed": ValueError("unit-test failure"), "interrupted": KeyboardInterrupt()}

    def exercise():
        with run:
            run.log("test_operation", samples=2)
            if ending in exceptions:
                raise exceptions[ending]
            if ending == "budget_exhausted":
                now[0] = 60
                run.deadline.check()

    if ending in exceptions:
        with pytest.raises(type(exceptions[ending])):
            exercise()
    elif ending == "budget_exhausted":
        with pytest.raises(BudgetExpired):
            exercise()
    else:
        exercise()

    manifest = json.loads((run.path / "run.json").read_text())
    assert manifest["config_sha256"] == config.sha256
    assert len(manifest["git_commit"]) == 40
    assert len(manifest["dependency_lock_sha256"]) == 64
    assert manifest["packages"]["hypothesis"]
    assert manifest["source_files"]
    assert (run.path / "config.yaml").read_bytes() == config.yaml_bytes
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == [0, 1, 2]
    assert events[-1]["status"] == ending
    assert json.loads((run.path / "summary.json").read_text())["status"] == ending
    with pytest.raises(RuntimeError, match="active"):
        run.log("after_finish")
    with pytest.raises(RuntimeError, match="twice"):
        run.__enter__()


def test_runtime_rejects_nonfinite_events_and_does_not_reuse_directory(tmp_path):
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    first = RunSession(config, run_root=tmp_path)
    second = RunSession(config, run_root=tmp_path)
    assert first.path != second.path
    with first:
        with pytest.raises(ValueError):
            first.log("bad_metric", loss=float("nan"))
        with pytest.raises(ValueError, match="reserved"):
            first.log("bad_field", sequence=999)
    events = [json.loads(line) for line in (first.path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2


def test_expired_context_cannot_report_completion_without_a_deadline_check(tmp_path):
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    now = [0.0]
    run = RunSession(config, run_root=tmp_path, clock=lambda: now[0])
    with run:
        now[0] = 61
    assert json.loads((run.path / "summary.json").read_text())["status"] == "budget_exhausted"


def test_engine_imports_and_initial_legality_work_without_ml_dependencies():
    script = """
import sys
class NoML:
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'} or fullname == 'quoridor.network':
            raise ImportError('ML module imported by engine: ' + fullname)
sys.meta_path.insert(0, NoML())
from quoridor.game import create_initial_state, legal_action_indices
actions = legal_action_indices(create_initial_state())
assert len(actions) == 131 and len(set(actions)) == 131
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
