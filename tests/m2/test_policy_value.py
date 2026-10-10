from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch
from torch.nn import functional as F

from quoridor.feature_spec import MODEL_VERSION
from quoridor.policy_value import NetworkConfig, PolicyValueNetwork


def test_complete_config_roundtrips_json_and_is_immutable():
    config = NetworkConfig()
    assert config.model_version == MODEL_VERSION
    assert (config.channels, config.blocks) == (64, 4)
    document = config.to_dict()
    assert NetworkConfig.from_dict(json.loads(json.dumps(document))) == config
    document["channels"] = 123
    assert config.channels == 64
    with pytest.raises(FrozenInstanceError):
        config.channels = 123


@pytest.mark.parametrize("field", [field.name for field in fields(NetworkConfig) if field.name != "model_version"])
@pytest.mark.parametrize("bad", [True, 1.0, "1", None, 0, -1])
def test_config_rejects_invalid_integer_fields(field, bad):
    with pytest.raises((ValueError, TypeError), match=field):
        replace(NetworkConfig(), **{field: bad})


@pytest.mark.parametrize("field,bad", [
    ("model_version", "legacy"), ("model_version", None), ("model_version", 1),
    ("input_channels", 14), ("board_size", 7), ("action_size", 208),
])
def test_config_rejects_incompatible_contract(field, bad):
    with pytest.raises(ValueError, match=field):
        replace(NetworkConfig(), **{field: bad})


def test_deserialization_cannot_fill_missing_metadata_with_defaults():
    document = NetworkConfig().to_dict()
    del document["model_version"]
    with pytest.raises(ValueError, match="exactly"):
        NetworkConfig.from_dict(document)
    with pytest.raises(ValueError, match="exactly"):
        NetworkConfig.from_dict(NetworkConfig().to_dict() | {"legacy": True})
    with pytest.raises(TypeError, match="mapping"):
        NetworkConfig.from_dict([])
    with pytest.raises(TypeError, match="NetworkConfig"):
        PolicyValueNetwork({})


@pytest.mark.parametrize("batch", [1, 4])
def test_synthetic_forward_backward_has_finite_bounded_outputs_and_gradients(batch):
    torch.manual_seed(120)
    model = PolicyValueNetwork(NetworkConfig.smoke())
    inputs = torch.randn(batch, 16, 9, 9, requires_grad=True)
    original = inputs.detach().clone()
    logits, values = model(inputs)
    assert logits.shape == (batch, 209)
    assert values.shape == (batch,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert (values.abs() <= 1).all()
    assert torch.equal(inputs.detach(), original)
    targets = torch.arange(batch) % 209
    value_targets = torch.linspace(-1, 1, batch)
    loss = F.cross_entropy(logits, targets) + F.mse_loss(values, value_targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for head in (model.policy_head, model.value_head, model.trunk):
        assert sum(float(parameter.grad.abs().sum()) for parameter in head.parameters()) > 0


def test_heads_are_raw_logits_and_tanh_values_without_a_legal_mask():
    model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
    with torch.no_grad():
        model.policy_head[-1].weight.zero_()
        model.policy_head[-1].bias.copy_(torch.arange(209))
        model.value_head[-2].weight.zero_()
        model.value_head[-2].bias.fill_(2)
        logits, values = model(torch.zeros(1, 16, 9, 9))
    assert torch.equal(logits[0], torch.arange(209).float())
    assert values.item() == pytest.approx(torch.tanh(torch.tensor(2.)).item())


def test_evaluation_accepts_converted_floating_dtype_and_is_batch_independent():
    torch.manual_seed(4)
    model = PolicyValueNetwork(NetworkConfig.smoke()).double().eval()
    inputs = torch.randn(3, 16, 9, 9, dtype=torch.float64)
    with torch.inference_mode():
        logits, values = model(inputs)
        alone_logits, alone_values = model(inputs[:1])
    assert logits.dtype == values.dtype == torch.float64
    torch.testing.assert_close(logits[:1], alone_logits)
    torch.testing.assert_close(values[:1], alone_values)


@pytest.mark.parametrize("shape", [(16, 9, 9), (1, 14, 9, 9), (1, 16, 8, 9), (0, 16, 9, 9), (1, 16, 9, 9, 1)])
def test_invalid_input_shape_fails_before_forward(shape):
    with pytest.raises(ValueError, match="shape"):
        PolicyValueNetwork(NetworkConfig.smoke())(torch.zeros(shape))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_fails_before_mutating_batchnorm(bad):
    model = PolicyValueNetwork(NetworkConfig.smoke())
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    inputs = torch.zeros(1, 16, 9, 9)
    inputs[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        model(inputs)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_non_tensor_integer_sparse_and_dtype_mismatch_are_explicit_errors():
    model = PolicyValueNetwork(NetworkConfig.smoke())
    for inputs in (None, [], object()):
        with pytest.raises(TypeError, match="Tensor"):
            model(inputs)
    for dtype in (torch.int64, torch.bool, torch.complex64):
        with pytest.raises(TypeError, match="floating"):
            model(torch.zeros(1, 16, 9, 9, dtype=dtype))
    with pytest.raises(ValueError, match="dense"):
        model(torch.zeros(1, 16, 9, 9).to_sparse())
    with pytest.raises(ValueError, match="dtype"):
        model(torch.zeros(1, 16, 9, 9, dtype=torch.float64))


def test_nonfinite_network_outputs_are_rejected():
    model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
    with torch.no_grad():
        model.policy_head[-1].bias[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        model(torch.zeros(1, 16, 9, 9))


def test_network_import_and_inference_do_not_load_rules_or_encoder():
    script = """
import sys
import torch
forbidden = {
    'quoridor.game', 'quoridor.codec', 'quoridor.network',
    'quoridor.engine', 'quoridor.episode', 'quoridor.encoder',
}
class NoRules:
    def find_spec(self, fullname, path, target=None):
        if fullname in forbidden:
            raise ImportError('tensor network imported rules: ' + fullname)
sys.meta_path.insert(0, NoRules())
from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
with torch.inference_mode():
    logits, values = model(torch.zeros(1, 16, 9, 9))
assert logits.shape == (1, 209) and values.shape == (1,)
assert not forbidden.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
