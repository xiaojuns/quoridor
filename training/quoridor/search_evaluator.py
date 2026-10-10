"""Bridge exact episodes to tensor inference without moving rules into a model."""

from __future__ import annotations

import torch

from .encoder import encode_episode
from .episode import Episode
from .feature_spec import ACTION_SIZE
from .policy_value import PolicyValueNetwork
from .project_runtime import Deadline
from .puct import Evaluation


class NeuralEvaluator:
    """Return original-coordinate logits and the current player's value.

    The caller supplies an eval-mode model and keeps its weights/buffers immutable
    throughout each search. This adapter does not change model mode, choose a
    device, cache history-sensitive results, mask actions, or adjudicate games.
    ``mirror`` is an explicit fixed input transform; its inverse also transforms
    the predicted actions. It is not randomly sampled augmentation.

    Deadline checks are cooperative: an in-flight tensor operation cannot be
    preempted. A result completed after its deadline is discarded, including
    device-to-CPU synchronization before returning plain Python numbers.
    """

    def __init__(self, model: PolicyValueNetwork, *, mirror: bool = False) -> None:
        if not isinstance(model, PolicyValueNetwork):
            raise TypeError("model must be a PolicyValueNetwork")
        if type(mirror) is not bool:
            raise TypeError("mirror must be a bool")
        self.model = model
        self.mirror = mirror
        self._require_eval_mode()

    def _require_eval_mode(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise ValueError("search requires every model module in eval mode")

    def __call__(self, episode: Episode, *, deadline: Deadline) -> Evaluation:
        if not isinstance(deadline, Deadline):
            raise TypeError("deadline must be a Deadline")
        deadline.check()
        if type(episode) is not Episode:
            raise TypeError("evaluator requires a quoridor-v1 Episode")
        if episode.finished:
            raise ValueError("finished episodes must not be evaluated by the network")
        self._require_eval_mode()
        with torch.inference_mode():
            encoded = encode_episode(episode, mirror=self.mirror)
            deadline.check()
            reference = self.model.input_conv.weight
            inputs = encoded.features.unsqueeze(0).to(
                device=reference.device, dtype=reference.dtype
            )
            deadline.check()
            logits, values = self.model(inputs)
            deadline.check()
            if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise TypeError("model outputs must be tensors")
            if logits.shape != (1, ACTION_SIZE) or values.shape != (1,):
                raise ValueError("model outputs must have shapes [1,209] and [1]")
            if (
                not logits.is_floating_point() or not values.is_floating_point()
                or logits.layout != torch.strided or values.layout != torch.strided
            ):
                raise ValueError("model outputs must be dense floating-point tensors")
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                raise ValueError("model outputs must contain only finite values")
            value = float(values[0].cpu())
            if not -1.0 <= value <= 1.0:
                raise ValueError("model value must be in [-1,1]")
            canonical_logits = tuple(float(logit) for logit in logits[0].cpu().tolist())
            original_logits = encoded.transform.inverse().policy(canonical_logits)
            deadline.check()
            return Evaluation(logits=original_logits, value=value)


__all__ = ["NeuralEvaluator"]
