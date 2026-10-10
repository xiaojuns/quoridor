from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from quoridor import codec as legacy_codec
from quoridor import game as legacy_game
from quoridor.rules_contract import (
    IncompatibleRulesError,
    UnsupportedBackendError,
    contract_metadata,
    load_engine,
    require_compatible_artifact,
)


ROOT = Path(__file__).resolve().parents[2]


def test_new_engine_backend_is_explicit_and_legacy_state_is_rejected():
    from quoridor import engine

    assert load_engine() is engine
    legacy = legacy_game.create_initial_state()
    with pytest.raises((TypeError, ValueError)):
        engine.apply_action(legacy, 67)
    with pytest.raises((TypeError, ValueError)):
        engine.validate_state(legacy)


@pytest.mark.parametrize("backend", ["native", "browser", "auto", "legacy", None, True])
def test_uncertified_backend_never_falls_back(backend):
    with pytest.raises(UnsupportedBackendError, match="not supported"):
        load_engine(backend)


@pytest.mark.parametrize("version", [None, 1, "legacy-repetition-loss-v1", "quoridor-v2"])
def test_engine_rejects_other_rules_versions(version):
    with pytest.raises(IncompatibleRulesError):
        load_engine(rules_version=version)


@pytest.mark.parametrize("metadata", [
    {}, {"codecVersion": 1}, None,
    {**contract_metadata(), "rules_version": "legacy-repetition-loss-v1"},
    {**contract_metadata(), "codec_version": 1},
    {**contract_metadata(), "value_perspective": "player-zero"},
])
def test_legacy_or_unversioned_artifact_rejected(metadata):
    with pytest.raises(IncompatibleRulesError):
        require_compatible_artifact(metadata)


@pytest.mark.parametrize("field", ["rules_version", "codec_version", "value_perspective"])
@pytest.mark.parametrize("bad_value", [None, True, 1, "wrong"])
def test_each_contract_identity_is_required_and_type_checked(field, bad_value):
    metadata = contract_metadata()
    del metadata[field]
    with pytest.raises(IncompatibleRulesError, match=field):
        require_compatible_artifact(metadata)
    metadata[field] = bad_value
    with pytest.raises(IncompatibleRulesError, match=field):
        require_compatible_artifact(metadata)


def test_matching_contract_does_not_mutate_metadata():
    metadata = contract_metadata()
    require_compatible_artifact(metadata)
    metadata["rules_version"] = "changed"
    assert contract_metadata()["rules_version"] == "quoridor-v1"


def test_legacy_action_numbers_remain_compatible_without_reusing_legacy_outcomes():
    engine = load_engine()
    for action in range(209):
        assert engine.decode_action(action) == legacy_codec.decode_action(action)
        assert engine.encode_action(legacy_codec.decode_action(action)) == action


def test_new_rule_modules_play_without_ml_native_or_legacy_rules():
    # Import package once because its public root exports intentionally stay legacy.
    # Block new imports and remove old rule modules to detect accidental reuse.
    code = """
import sys
import quoridor
for name in ('quoridor.game', 'quoridor.codec'):
    sys.modules.pop(name, None)
    delattr(quoridor, name.split('.')[-1])
class NoLegacyOrML:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'quoridor_native'} or fullname in {
            'quoridor.game', 'quoridor.codec', 'quoridor.network', 'quoridor.native_mcts',
        }:
            raise ImportError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, NoLegacyOrML())
from quoridor.rules_contract import load_engine
import quoridor.episode
engine = load_engine()
state = engine.initial_state()
assert len(engine.legal_actions(state)) == 131
child = engine.apply_action(state, 67)
assert child.to_play == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_engine_has_no_clock_dependency():
    source = ROOT / "training/quoridor/engine.py"
    tree = ast.parse(source.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split('.')[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or '').split('.')[0])
    assert not imports.intersection({'time', 'datetime', 'project_runtime'})
