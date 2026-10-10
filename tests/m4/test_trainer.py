from __future__ import annotations

import torch
import pytest

from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
from quoridor.trainer import AlphaZeroTrainer, NonFiniteTrainingError


def _batch(*, finite: bool = True) -> dict[str, torch.Tensor]:
    states = torch.zeros(2, 16, 9, 9)
    policies = torch.zeros(2, 209)
    policies[:, 3] = 0.25
    policies[:, 7] = 0.75
    masks = torch.zeros(2, 209, dtype=torch.bool)
    masks[:, 3] = True
    masks[:, 7] = True
    values = torch.tensor([1.0, -1.0])
    if not finite:
        states[0, 0, 0, 0] = float("nan")
    return {"states": states, "policies": policies, "values": values, "legal_masks": masks}


def test_policy_loss_masks_illegal_logits_and_updates_finitely() -> None:
    torch.manual_seed(4)
    model = PolicyValueNetwork(NetworkConfig.smoke())
    trainer = AlphaZeroTrainer(model, use_amp=False, require_legal_masks=True)
    metrics = trainer.train_step(_batch())
    assert metrics.loss == pytest.approx(metrics.policy_loss + metrics.value_loss)
    assert torch.isfinite(torch.tensor(metrics.grad_norm))
    assert trainer.step == 1


def test_missing_mask_is_rejected_for_new_trainer() -> None:
    model = PolicyValueNetwork(NetworkConfig.smoke())
    trainer = AlphaZeroTrainer(model, use_amp=False, require_legal_masks=True)
    batch = _batch()
    del batch["legal_masks"]
    with pytest.raises(ValueError, match="legal_masks"):
        trainer.train_step(batch)


def test_target_mass_outside_legal_support_is_rejected() -> None:
    model = PolicyValueNetwork(NetworkConfig.smoke())
    trainer = AlphaZeroTrainer(model, use_amp=False, require_legal_masks=True)
    batch = _batch()
    batch["policies"][0, 8] = 0.1
    batch["policies"][0, 7] = 0.65
    with pytest.raises(ValueError, match="illegal"):
        trainer.train_step(batch)


def test_nonfinite_input_fails_before_optimizer_update() -> None:
    model = PolicyValueNetwork(NetworkConfig.smoke())
    trainer = AlphaZeroTrainer(model, use_amp=False, require_legal_masks=True)
    with pytest.raises(NonFiniteTrainingError):
        trainer.train_step(_batch(finite=False))
    assert trainer.step == 0
