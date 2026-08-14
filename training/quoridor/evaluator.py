from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from .network import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, encode_state


@dataclass
class TorchEvaluator:
    model: nn.Module
    device: torch.device | str
    use_amp: bool = True
    _native_encoder: Callable[..., object] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def enable_native_encoder(self, module: object) -> None:
        encoder = getattr(module, "encode_states", None)
        if not callable(encoder):
            raise ValueError("native module does not expose encode_states")
        self._native_encoder = encoder

    def evaluate(self, state: object) -> tuple[Sequence[float], float]:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self,
        states: Iterable[object],
    ) -> list[tuple[Sequence[float], float]]:
        state_list = list(states)
        if not state_list:
            return []
        self.model.eval()
        network_config = getattr(self.model, "config", None)
        input_channels = int(
            getattr(network_config, "input_channels", INPUT_CHANNELS)
        )
        if self._native_encoder is None:
            inputs = torch.stack(
                [
                    encode_state(
                        state,
                        device=self.device,
                        input_channels=input_channels,
                    )
                    for state in state_list
                ]
            )
        else:
            encoded = self._native_encoder(
                state_list,
                input_channels=input_channels,
            )
            inputs = torch.from_numpy(encoded)
            expected_shape = (
                len(state_list),
                input_channels,
                BOARD_SIZE,
                BOARD_SIZE,
            )
            if tuple(inputs.shape) != expected_shape:
                raise ValueError(
                    f"native encoder must return shape {expected_shape}"
                )
            if inputs.dtype != torch.float32:
                raise ValueError("native encoder must return float32 inputs")
            inputs = inputs.to(self.device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp and self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            policy_logits, value = self.model(inputs)
        if not bool(torch.isfinite(policy_logits).all().item()):
            raise ValueError("evaluator policy logits contain NaN or Inf")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("evaluator values contain NaN or Inf")
        logits = policy_logits.float().cpu()
        values = value.float().cpu().reshape(-1)
        if logits.shape != (len(state_list), ACTION_SIZE):
            raise ValueError(
                f"evaluator policy must have shape [{len(state_list)}, {ACTION_SIZE}]"
            )
        if values.shape != (len(state_list),):
            raise ValueError(f"evaluator value must have {len(state_list)} entries")
        return [
            (
                logits[index].tolist(),
                max(-1.0, min(1.0, float(values[index].item()))),
            )
            for index in range(len(state_list))
        ]


__all__ = ["TorchEvaluator"]
