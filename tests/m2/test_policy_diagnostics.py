import json

import pytest
import torch

from quoridor.policy import mask_policy_logits, masked_policy
from quoridor.value_diagnostics import value_diagnostics


def test_mask_preserves_only_legal_support_and_has_finite_gradients():
    logits = torch.zeros((2, 209), requires_grad=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, [2, 7, 200]] = True
    masked = mask_policy_logits(logits, mask)
    assert torch.isneginf(masked[~mask]).all()
    assert torch.equal(masked[mask], logits[mask])
    policy = masked_policy(logits, mask)
    assert torch.allclose(policy[0, [2, 7, 200]], torch.full((3,), 1 / 3))
    assert torch.equal(policy[~mask], torch.zeros_like(policy[~mask]))
    assert policy[0].sum().item() == pytest.approx(1)
    assert policy[1].sum() == 0
    (-policy[0, 2].log()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[~mask].count_nonzero() == 0
    assert logits.grad[0, 2] < 0


def test_single_legal_action_and_extreme_logits():
    logits = torch.full((209,), 1e30)
    logits[7] = -1e30
    mask = torch.zeros(209, dtype=torch.bool)
    mask[7] = True
    policy = masked_policy(logits, mask)
    assert policy[7] == 1 and policy.sum() == 1


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_logits_cannot_hide_in_illegal_actions(bad):
    logits = torch.zeros(209)
    logits[0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        masked_policy(logits, torch.zeros(209, dtype=torch.bool))


def test_mask_rejects_broadcasting_and_ambiguous_types():
    with pytest.raises(ValueError, match="exactly"):
        masked_policy(torch.zeros(2, 209), torch.ones(209, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        masked_policy(torch.zeros(209), torch.ones(209))
    with pytest.raises(TypeError, match="floating"):
        masked_policy(torch.zeros(209, dtype=torch.long), torch.ones(209, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        masked_policy(torch.zeros(208), torch.ones(208, dtype=torch.bool))


def test_sparse_policy_inputs_are_rejected_explicitly():
    with pytest.raises(ValueError, match="dense"):
        masked_policy(torch.ones(209).to_sparse(), torch.ones(209, dtype=torch.bool))
    with pytest.raises(ValueError, match="dense"):
        masked_policy(torch.ones(209), torch.ones(209, dtype=torch.bool).to_sparse())


def test_calibration_diagnostics_have_independently_known_values_and_json():
    metrics = value_diagnostics(
        torch.tensor([-1., -.5, 0., .5, 1.]),
        torch.tensor([-1., 1., 0., -1., 1.]), bins=2,
    )
    assert metrics['samples'] == 5
    assert metrics['value_mse'] == pytest.approx(.9)
    assert metrics['value_mae'] == pytest.approx(.6)
    assert metrics['value_bias'] == pytest.approx(0)
    assert metrics['decisive_sign_accuracy'] == .5
    assert metrics['draw_mean_abs_value'] == 0
    assert metrics['value_calibration_mae'] == pytest.approx(.6)
    assert [b['count'] for b in metrics['bins']] == [2, 3]
    assert metrics['bins'][0]['mean_prediction'] == -.75
    assert metrics['bins'][0]['mean_target'] == 0
    json.dumps(metrics, allow_nan=False)


def test_empty_bins_and_no_decisive_samples_are_missing_not_nan():
    metrics = value_diagnostics(torch.zeros(3), torch.zeros(3))
    assert metrics['decisive_sign_accuracy'] is None
    assert metrics['value_calibration_mae'] == 0
    assert metrics['bins'][0]['mean_prediction'] is None
    json.dumps(metrics, allow_nan=False)


def test_sparse_diagnostic_inputs_are_rejected_explicitly():
    with pytest.raises(ValueError, match="dense"):
        value_diagnostics(torch.ones(2).to_sparse(), torch.ones(2))
    with pytest.raises(ValueError, match="dense"):
        value_diagnostics(torch.ones(2), torch.ones(2).to_sparse())


@pytest.mark.parametrize("values,labels", [
    ([], []), ([1.1], [1]), ([0], [.5]), ([0], [float('nan')]),
    ([float('inf')], [1]), ([0, 1], [1]),
])
def test_invalid_or_unlabelled_samples_are_rejected(values, labels):
    with pytest.raises(ValueError):
        value_diagnostics(torch.tensor(values, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))


def test_network_encoder_mask_integration_and_finished_episode():
    from quoridor.encoder import encode_episode
    from quoridor.episode import advance_episode, start_episode
    from quoridor.policy_value import NetworkConfig, PolicyValueNetwork

    live = encode_episode(start_episode())
    ended = encode_episode(advance_episode(start_episode(max_game_plies=1), 67))
    model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
    with torch.no_grad():
        logits, values = model(torch.stack([live.features, ended.features]))
        policy = masked_policy(logits, torch.stack([live.legal_mask, ended.legal_mask]))
    assert policy[0].sum() == pytest.approx(1)
    assert torch.all(policy[0, ~live.legal_mask] == 0)
    assert policy[1].count_nonzero() == 0
    assert values.shape == (2,)
