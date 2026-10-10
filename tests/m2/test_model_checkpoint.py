from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from pathlib import Path

import pytest
import torch

from quoridor.feature_spec import encoder_metadata
from quoridor.model_checkpoint import (
    IncompatibleCheckpointError,
    load_model_checkpoint,
    save_model_checkpoint,
)
from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
from quoridor.rules_contract import contract_metadata


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(207)
        return PolicyValueNetwork(NetworkConfig.smoke())


@pytest.fixture
def artifact(tmp_path, model):
    return save_model_checkpoint(tmp_path / "model.pt", model)


def _payload(artifact):
    return torch.load(artifact.path, weights_only=True, map_location="cpu")


def _write_payload(tmp_path, payload):
    path = tmp_path / "modified.pt"
    torch.save(payload, path)
    return path


def test_roundtrip_exact_outputs_and_weights_including_batchnorm(tmp_path, model):
    # A synthetic forward updates BN buffers; this is not a training experiment.
    inputs = torch.linspace(-1, 1, 2 * 16 * 9 * 9).reshape(2, 16, 9, 9)
    model.train()
    with torch.no_grad():
        model(inputs)
    model.eval()
    with torch.no_grad():
        before = model(inputs)
    artifact = save_model_checkpoint(tmp_path / "model.pt", model)
    loaded = load_model_checkpoint(
        artifact.path, expected_sha256=artifact.sha256, expected_config=model.config,
    )
    assert loaded.artifact == artifact
    assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    assert loaded.model is not model
    assert loaded.model.config == model.config
    assert not loaded.model.training
    assert all(value.device.type == "cpu" for value in loaded.model.state_dict().values())
    assert loaded.metadata["encoder"] == encoder_metadata()
    assert loaded.metadata["rules_contract"] == contract_metadata()
    assert "state_dict" not in loaded.metadata
    with torch.no_grad():
        after = loaded.model(inputs)
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
    assert loaded.model.input_bn.num_batches_tracked == 1


def test_save_preserves_mode_weights_device_and_rng_load_preserves_rng(tmp_path, model):
    model.train()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    artifact = save_model_checkpoint(tmp_path / "model.pt", model)
    assert model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        assert value.device == weights[key].device
        assert torch.equal(value, weights[key])
    load_model_checkpoint(artifact.path)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_metadata_returns_independent_copies(artifact):
    loaded = load_model_checkpoint(artifact.path)
    loaded.metadata["encoder"]["planes"][0] = "wrong"
    assert loaded.metadata["encoder"] == encoder_metadata()


@pytest.mark.parametrize("field", [
    "schema_version", "kind", "rules_contract", "encoder", "model_version",
    "architecture", "state_dict",
])
def test_missing_required_field_rejected(tmp_path, artifact, field):
    payload = _payload(artifact)
    del payload[field]
    with pytest.raises(IncompatibleCheckpointError, match="fields mismatch"):
        load_model_checkpoint(_write_payload(tmp_path, payload))


def test_unknown_top_level_metadata_rejected(tmp_path, artifact):
    payload = _payload(artifact)
    payload["optimizer"] = {}
    with pytest.raises(IncompatibleCheckpointError, match="fields mismatch"):
        load_model_checkpoint(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(("location", "value"), [
    (("schema_version",), True),
    (("schema_version",), 1.0),
    (("kind",), "legacy-model"),
    (("model_version",), "az-resnet-v0"),
    (("rules_contract", "rules_version"), "legacy-repetition-loss-v1"),
    (("rules_contract", "codec_version"), 1),
    (("rules_contract", "value_perspective"), "absolute-player-zero"),
    (("encoder", "version"), "legacy-16"),
    (("encoder", "input_channels"), 14),
    (("encoder", "input_channels"), 16.0),
    (("encoder", "history_observation"), "full-history"),
    (("encoder", "planes"), ["wrong"] * 16),
    (("encoder", "normalization", "distance_divisor"), 9),
    (("architecture", "input_channels"), 14),
    (("architecture", "action_size"), 210),
    (("architecture", "channels"), True),
    (("architecture", "channels"), 8.0),
    (("architecture", "blocks"), 0),
    (("architecture", "model_version"), "other"),
])
def test_incompatible_metadata_rejected(tmp_path, artifact, location, value):
    payload = _payload(artifact)
    target = payload
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(IncompatibleCheckpointError):
        load_model_checkpoint(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("planes", [14, 16])
def test_legacy_checkpoint_not_inferred_compatible_from_plane_count(tmp_path, model, planes):
    path = _write_payload(tmp_path, {
        "model_state": model.state_dict(),
        "config": {"input_planes": planes, "action_size": 209},
    })
    with pytest.raises(IncompatibleCheckpointError, match="legacy"):
        load_model_checkpoint(path)


def test_explicit_expected_architecture_must_match(artifact, model):
    with pytest.raises(IncompatibleCheckpointError, match="expected architecture"):
        load_model_checkpoint(artifact.path, expected_config=replace(model.config, channels=16))


@pytest.mark.parametrize("case", [
    "missing", "extra", "shape", "dtype", "not_tensor", "sparse", "nan_weight",
    "infinite_bn_mean", "nan_bn_var", "negative_bn_var", "negative_bn_count",
    "wrong_bn_count_dtype", "not_mapping", "non_string_key",
])
def test_invalid_weights_and_all_buffers_rejected(tmp_path, artifact, case):
    payload = _payload(artifact)
    weights = payload["state_dict"]
    first = "input_conv.weight"
    if case == "missing":
        del weights[first]
    elif case == "extra":
        weights["extra"] = torch.zeros(1)
    elif case == "shape":
        weights[first] = weights[first].flatten()
    elif case == "dtype":
        weights[first] = weights[first].double()
    elif case == "not_tensor":
        weights[first] = [1.0]
    elif case == "sparse":
        weights[first] = weights[first].to_sparse()
    elif case == "nan_weight":
        weights[first].flatten()[0] = float("nan")
    elif case == "infinite_bn_mean":
        weights["input_bn.running_mean"][0] = float("inf")
    elif case == "nan_bn_var":
        weights["input_bn.running_var"][0] = float("nan")
    elif case == "negative_bn_var":
        weights["input_bn.running_var"][0] = -1
    elif case == "negative_bn_count":
        weights["input_bn.num_batches_tracked"].fill_(-1)
    elif case == "wrong_bn_count_dtype":
        weights["input_bn.num_batches_tracked"] = torch.tensor(0.0)
    elif case == "not_mapping":
        payload["state_dict"] = []
    elif case == "non_string_key":
        weights[42] = weights.pop(first)
    with pytest.raises(IncompatibleCheckpointError, match="state_dict"):
        load_model_checkpoint(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("field", ["channels", "blocks"])
def test_architecture_shape_fraud_rejected_before_cpu_allocation(tmp_path, artifact, monkeypatch, field):
    payload = _payload(artifact)
    payload["architecture"][field] = 10**9
    from quoridor import model_checkpoint

    def forbidden(*args, **kwargs):
        pytest.fail("a mismatched checkpoint must not construct a CPU model")

    monkeypatch.setattr(model_checkpoint, "_new_cpu_model", forbidden)
    with pytest.raises(IncompatibleCheckpointError):
        load_model_checkpoint(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("bad_hash", [True, 12, "", "deadbeef", "G" * 64, "A" * 64])
def test_expected_hash_has_strict_format(artifact, bad_hash):
    with pytest.raises(ValueError, match="expected_sha256"):
        load_model_checkpoint(artifact.path, expected_sha256=bad_hash)


def test_hash_mismatch_fails_before_deserializing(artifact, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("hash mismatch must reject before torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(IncompatibleCheckpointError, match="SHA-256 mismatch"):
        load_model_checkpoint(artifact.path, expected_sha256="0" * 64)


def test_load_hash_and_deserialization_use_same_bytes(artifact, monkeypatch):
    original = torch.load

    def replace_during_load(stream, **kwargs):
        assert isinstance(stream, io.BytesIO)
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        artifact.path.write_bytes(b"replaced after read")
        return original(stream, **kwargs)

    monkeypatch.setattr(torch, "load", replace_during_load)
    loaded = load_model_checkpoint(artifact.path, expected_sha256=artifact.sha256)
    assert loaded.artifact.sha256 == artifact.sha256
    assert artifact.path.read_bytes() == b"replaced after read"


def test_invalid_serialization_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(IncompatibleCheckpointError, match="weights-only"):
        load_model_checkpoint(path)


class _UnallowlistedObject:
    pass


def test_no_unsafe_pickle_fallback(tmp_path):
    path = _write_payload(tmp_path, _UnallowlistedObject())
    with pytest.raises(IncompatibleCheckpointError, match="weights-only"):
        load_model_checkpoint(path)


@pytest.mark.parametrize("symlink", [False, True])
def test_save_never_overwrites_existing_file_or_symlink(tmp_path, model, symlink):
    target = tmp_path / "model.pt"
    original = tmp_path / "original.pt"
    if symlink:
        original.write_bytes(b"preserve me")
        target.symlink_to(original)
    else:
        target.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        save_model_checkpoint(target, model)
    assert target.read_bytes() == b"preserve me"
    assert target.is_symlink() == symlink
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_serialization_does_not_publish_partial_artifact(tmp_path, model, monkeypatch):
    def broken_save(payload, stream):
        stream.write(b"partial bytes")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="simulated write failure"):
        save_model_checkpoint(tmp_path / "model.pt", model)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ["float64", "nonfinite"])
def test_save_rejects_invalid_model_before_publication(tmp_path, model, fault):
    if fault == "float64":
        model.double()
    else:
        with torch.no_grad():
            model.input_bn.running_mean[0] = float("nan")
    with pytest.raises(IncompatibleCheckpointError):
        save_model_checkpoint(tmp_path / "model.pt", model)
    assert list(tmp_path.iterdir()) == []
    if fault == "float64":
        assert model.input_conv.weight.dtype == torch.float64


def test_missing_path_is_explicit_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_does_not_create_run_or_parent_directory(tmp_path, model):
    with pytest.raises(FileNotFoundError):
        save_model_checkpoint(tmp_path / "nonexistent-run" / "model.pt", model)
    assert list(tmp_path.iterdir()) == []
