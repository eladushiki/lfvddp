"""Shared parameterizations for log-safe likelihood shifts."""

from math import log, log1p

import torch
from torch.nn import functional as F

from train.neural_output_config import NeuralOutputMapping

LIKELIHOOD_SHIFT_BOUND = 1.0 - 1e-6
LOG_LIKELIHOOD_FLOOR = log1p(-LIKELIHOOD_SHIFT_BOUND)
LOG_TWICE_LIKELIHOOD_BOUND = log(2 * LIKELIHOOD_SHIFT_BOUND)


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


def bounded_likelihood_shift_and_logs(
    unbounded_shift: torch.Tensor,
    output_mapping: NeuralOutputMapping | str = NeuralOutputMapping.TANH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the shift and both log factors before boundary rounding.

    For bound B and raw output z, 1 +/- B*tanh(z) equals
    (1-B) + 2*B*sigmoid(+/-2*z). Work in log space so the tiny
    sigmoid contribution and its derivative survive near the boundary.
    The algebraic map equals B*tanh(asinh(z)), so the same log-domain
    calculation applies after transforming its raw output with asinh.
    """

    mapping = NeuralOutputMapping.parse(output_mapping)
    shift = smoothly_bounded_likelihood_shift(unbounded_shift, mapping)
    tanh_argument = (
        torch.asinh(unbounded_shift)
        if mapping is NeuralOutputMapping.ALGEBRAIC
        else unbounded_shift
    )
    floor = unbounded_shift.new_tensor(LOG_LIKELIHOOD_FLOOR)
    doubled_argument = 2 * tanh_argument
    log_plus = torch.logaddexp(
        floor, LOG_TWICE_LIKELIHOOD_BOUND + F.logsigmoid(doubled_argument)
    )
    log_minus = torch.logaddexp(
        floor, LOG_TWICE_LIKELIHOOD_BOUND + F.logsigmoid(-doubled_argument)
    )
    return shift, log_plus, log_minus
