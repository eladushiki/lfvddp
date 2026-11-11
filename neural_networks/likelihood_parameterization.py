"""Shared parameterizations for log-safe likelihood shifts."""

import torch

from train.neural_output_config import NeuralOutputMapping

LIKELIHOOD_SHIFT_BOUND = 1.0 - 1e-6


def smoothly_bounded_likelihood_shift(
    unbounded_shift: torch.Tensor,
    output_mapping: NeuralOutputMapping | str = NeuralOutputMapping.TANH,
) -> torch.Tensor:
    """Map a neural output smoothly into the open log-safe shift interval."""
    mapping = NeuralOutputMapping.parse(output_mapping)
    if mapping is NeuralOutputMapping.ALGEBRAIC:
        # hypot avoids overflowing the square for large finite raw outputs.
        bounded = unbounded_shift / torch.hypot(
            unbounded_shift, torch.ones_like(unbounded_shift)
        )
    elif mapping is NeuralOutputMapping.TANH:
        bounded = torch.tanh(unbounded_shift)
    else:
        raise ValueError(f"Unsupported output_mapping: {mapping}")
    return LIKELIHOOD_SHIFT_BOUND * bounded
