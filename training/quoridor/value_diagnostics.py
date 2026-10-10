"""Current-player value diagnostics for completed, labelled samples only.

These descriptive errors measure value fit/calibration, never playing strength.
Missing targets from truncations must not be replaced with zero before calling.
"""

import torch
from torch import Tensor


def value_diagnostics(predictions: Tensor, targets: Tensor, *, bins: int = 10) -> dict:
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if not isinstance(predictions, Tensor) or not predictions.is_floating_point():
        raise TypeError("predictions must be a floating tensor")
    if not isinstance(targets, Tensor) or not targets.is_floating_point():
        raise TypeError("targets must be a floating tensor with completed-game labels")
    if predictions.layout != torch.strided or targets.layout != torch.strided:
        raise ValueError("predictions and targets must be dense strided tensors")
    if predictions.ndim != 1 or predictions.numel() == 0 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must have matching nonempty [samples] shapes")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("diagnostics require finite predictions and labels")
    if torch.any(predictions.abs() > 1) or not torch.all((targets == -1) | (targets == 0) | (targets == 1)):
        raise ValueError("values must be in [-1,1]; targets must be completed -1/0/+1 labels")
    values = predictions.detach().to(device="cpu", dtype=torch.float64)
    labels = targets.detach().to(device="cpu", dtype=torch.float64)
    errors = values - labels
    decisive = labels != 0
    drawn = labels == 0
    indices = (((values + 1) / 2) * bins).floor().long().clamp(max=bins - 1)
    buckets = []
    calibration_error = 0.0
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        predicted = float(values[selected].mean()) if count else None
        observed = float(labels[selected].mean()) if count else None
        if count:
            calibration_error += count * abs(predicted - observed)
        buckets.append({
            "lower": -1 + 2 * index / bins,
            "upper": -1 + 2 * (index + 1) / bins,
            "count": count, "mean_prediction": predicted, "mean_target": observed,
        })
    return {
        "samples": values.numel(),
        "value_mse": float(errors.square().mean()),
        "value_mae": float(errors.abs().mean()),
        "value_bias": float(errors.mean()),
        "mean_prediction": float(values.mean()),
        "mean_target": float(labels.mean()),
        "decisive_sign_accuracy": float((values[decisive].sign() == labels[decisive]).double().mean()) if decisive.any() else None,
        "draw_mean_abs_value": float(values[drawn].abs().mean()) if drawn.any() else None,
        "value_calibration_mae": calibration_error / values.numel(),
        "bins": buckets,
    }
