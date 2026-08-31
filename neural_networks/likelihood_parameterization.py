"""Shared parameterizations for log-safe likelihood shifts."""

import torch


LIKELIHOOD_SHIFT_BOUND = 1.0 - 1e-6


def smoothly_bounded_likelihood_shift(
    unbounded_shift: torch.Tensor,
) -> torch.Tensor:
    """Map a neural output smoothly into the open log-safe shift interval."""
    return LIKELIHOOD_SHIFT_BOUND * torch.tanh(unbounded_shift)
