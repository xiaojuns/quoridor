from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    grad_norm: float
    learning_rate: float


class NonFiniteTrainingError(RuntimeError):
    """Raised before a numerically invalid update can be checkpointed."""


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 5.0,
        use_amp: bool = True,
        schedule_steps: int | None = None,
        minimum_learning_rate: float = 0.0,
    ) -> None:
        self.model = model
        self.max_grad_norm = float(max_grad_norm)
        if not math.isfinite(self.max_grad_norm) or not self.max_grad_norm > 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        parameters = tuple(model.parameters())
        self.device_type = parameters[0].device.type if parameters else "cpu"
        self.amp_enabled = bool(use_amp and self.device_type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        if schedule_steps is not None and schedule_steps <= 0:
            raise ValueError("schedule_steps must be positive when provided")
        if (
            not math.isfinite(float(minimum_learning_rate))
            or minimum_learning_rate < 0
            or minimum_learning_rate > learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate must be finite and between zero and learning_rate"
            )
        self.schedule_steps = schedule_steps
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(schedule_steps),
                eta_min=float(minimum_learning_rate),
            )
            if schedule_steps is not None
            else None
        )
        self.step = 0

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        value_weights: torch.Tensor | None = None,
    ) -> TrainMetrics:
        self._validate_batch(batch)
        self._validate_value_weights(value_weights, batch["states"].shape[0])
        self.model.train()
        target_policies = batch["policies"]
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled
            else nullcontext()
        )
        with autocast_context:
            policy_logits, predicted_values = self.model(batch["states"])
            if policy_logits.shape != target_policies.shape:
                raise ValueError("model policy shape does not match policy targets")
            if predicted_values.numel() != batch["values"].numel():
                raise ValueError("model value shape does not match value targets")
            target_values = batch["values"].reshape_as(predicted_values)

            self._require_finite(policy_logits, "model policy logits")
            self._require_finite(predicted_values, "model values")
            policy_loss = -(
                target_policies * F.log_softmax(policy_logits, dim=1)
            ).sum(dim=1).mean()
            value_losses = F.mse_loss(
                predicted_values,
                target_values,
                reduction="none",
            ).reshape(-1)
            if value_weights is None:
                value_loss = value_losses.mean()
            else:
                weights = value_weights.to(
                    device=value_losses.device,
                    dtype=value_losses.dtype,
                ).reshape(-1)
                value_loss = (value_losses * weights).sum() / weights.sum()
            loss = policy_loss + value_loss
            self._require_finite(policy_loss, "policy loss")
            self._require_finite(value_loss, "value loss")
            self._require_finite(loss, "total loss")

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        try:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError("model gradients are non-finite") from error
        self._require_finite(grad_norm, "gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        for parameter in self.model.parameters():
            self._require_finite(parameter, "model parameters after optimizer step")
        if self.scheduler is not None and self.step < int(self.schedule_steps):
            self.scheduler.step()
        self.step += 1

        return TrainMetrics(
            loss=float(loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            grad_norm=float(grad_norm.detach().item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    def runtime_state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "step": self.step,
            "ampEnabled": self.amp_enabled,
            "scaler": self.scaler.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }

    def load_runtime_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported trainer runtime state version")
        step = state.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("trainer runtime step must be a non-negative integer")
        saved_scheduler = state.get("scheduler")
        if (saved_scheduler is None) != (self.scheduler is None):
            raise ValueError("checkpoint scheduler does not match current config")
        saved_amp = state.get("ampEnabled")
        if not isinstance(saved_amp, bool):
            raise ValueError("checkpoint ampEnabled must be boolean")
        if saved_amp != self.amp_enabled:
            raise ValueError("checkpoint AMP mode does not match the current trainer")
        scaler_state = state.get("scaler")
        if not isinstance(scaler_state, dict):
            raise ValueError("checkpoint scaler state must be a mapping")
        self.scaler.load_state_dict(scaler_state)
        if self.scheduler is not None:
            if not isinstance(saved_scheduler, dict):
                raise ValueError("checkpoint scheduler state must be a mapping")
            self.scheduler.load_state_dict(saved_scheduler)
        self.step = step

    @staticmethod
    def _require_finite(tensor: torch.Tensor, label: str) -> None:
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise NonFiniteTrainingError(f"{label} contain NaN or Inf")

    @classmethod
    def _validate_batch(cls, batch: dict[str, torch.Tensor]) -> None:
        if set(batch) != {"states", "policies", "values"}:
            raise ValueError("training batch must contain states, policies, and values")
        states = batch["states"]
        policies = batch["policies"]
        values = batch["values"]
        if not all(isinstance(tensor, torch.Tensor) for tensor in batch.values()):
            raise TypeError("training batch entries must be tensors")
        if states.ndim != 4 or policies.ndim != 2 or values.ndim not in (1, 2):
            raise ValueError("training batch tensors have invalid dimensions")
        if states.shape[0] == 0 or policies.shape[0] != states.shape[0]:
            raise ValueError("training batch dimensions do not agree")
        if values.numel() != states.shape[0]:
            raise ValueError("training values must contain one value per state")
        cls._require_finite(states, "training states")
        cls._require_finite(policies, "training policies")
        cls._require_finite(values, "training values")
        if bool((policies < 0).any().item()):
            raise ValueError("training policies must be non-negative")
        policy_sums = policies.sum(dim=1)
        if not bool(torch.allclose(
            policy_sums,
            torch.ones_like(policy_sums),
            rtol=1e-4,
            atol=1e-5,
        )):
            raise ValueError("each training policy must sum to one")
        if bool(((values < -1) | (values > 1)).any().item()):
            raise ValueError("training values must be in [-1, 1]")

    @classmethod
    def _validate_value_weights(
        cls,
        value_weights: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if value_weights is None:
            return
        if not isinstance(value_weights, torch.Tensor):
            raise TypeError("value_weights must be a tensor or None")
        if value_weights.ndim not in (1, 2) or value_weights.numel() != batch_size:
            raise ValueError("value_weights must contain one weight per state")
        cls._require_finite(value_weights, "value weights")
        if bool(((value_weights < 0) | (value_weights > 1)).any().item()):
            raise ValueError("value_weights must be in [0, 1]")
        if not bool((value_weights > 0).any().item()):
            raise ValueError("value_weights must retain at least one value target")


__all__ = ["AlphaZeroTrainer", "NonFiniteTrainingError", "TrainMetrics"]
