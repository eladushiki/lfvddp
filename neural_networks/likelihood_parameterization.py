"""Shared parameterizations for log-safe likelihood shifts."""

import torch

from train.neural_output_config import LEGACY_NEURAL_OUTPUT_MAPPING

LIKELIHOOD_SHIFT_BOUND = 1.0 - 1e-6


def smoothly_bounded_likelihood_shift(
    unbounded_shift: torch.Tensor,
    output_mapping: str = LEGACY_NEURAL_OUTPUT_MAPPING,
) -> torch.Tensor:
    """Map a neural output smoothly into the open log-safe shift interval."""
    if output_mapping == "algebraic":
        # hypot avoids overflowing the square for large finite raw outputs.
        bounded = unbounded_shift / torch.hypot(
            unbounded_shift, torch.ones_like(unbounded_shift)
        )
    elif output_mapping == LEGACY_NEURAL_OUTPUT_MAPPING:
        bounded = torch.tanh(unbounded_shift)
    else:
        raise ValueError("output_mapping must be 'algebraic' or 'tanh'.")
    return LIKELIHOOD_SHIFT_BOUND * bounded
