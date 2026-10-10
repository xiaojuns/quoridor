"""Tensor masking supplied with legal masks by the external encoder.

The policy/value model always returns unmasked logits. This module owns no
Quoridor rules and never silently substitutes priors for numerical failures.
"""

import torch
from torch import Tensor

from .feature_spec import ACTION_SIZE


def _validate(logits: Tensor, legal_mask: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating tensor")
    if logits.layout != torch.strided:
        raise ValueError("logits must be a dense strided tensor")
    if logits.ndim not in (1, 2) or logits.shape[-1] != ACTION_SIZE or logits.numel() == 0:
        raise ValueError("logits must have shape [209] or [batch,209] with a nonempty batch")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite policy logits")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must be a boolean tensor from the encoder")
    if legal_mask.layout != torch.strided:
        raise ValueError("legal_mask must be a dense strided tensor")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must have exactly the logits shape")
    return legal_mask.to(device=logits.device)


def mask_policy_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Return -inf at illegal actions; an all-false row stays entirely -inf."""
    mask = _validate(logits, legal_mask)
    return logits.masked_fill(~mask, -torch.inf)


def masked_policy(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Normalize over legal support; a finished episode has an all-zero policy."""
    mask = _validate(logits, legal_mask)
    has_legal = mask.any(dim=-1, keepdim=True)
    safe_mask = mask | ~has_legal
    probabilities = torch.softmax(logits.masked_fill(~safe_mask, -torch.inf), dim=-1)
    probabilities = probabilities * mask.to(dtype=logits.dtype)
    if not torch.isfinite(probabilities).all():
        raise ValueError("nonfinite masked policy probabilities")
    return probabilities
