"""Adaptive neural function-space family."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from neural_networks.function_spaces.base import FunctionSpaceMetadata, immutable_options
from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift
from train.function_space_config import FunctionSpaceFamily


class AdaptiveNeuralFunction(nn.Module):
    """One-hidden-layer bounded sigmoid network used by both learned roles."""

    family = FunctionSpaceFamily.ADAPTIVE_NEURAL
    metadata = FunctionSpaceMetadata(
        regularity="adaptive",
        coefficient_topology="dense_two_layer",
    )

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.options = immutable_options(options or {})
        self.input_dimension = input_dimension
        self.hidden_size = hidden_size
        self.output_dimension = output_dimension
        self.hidden = nn.Linear(input_dimension, hidden_size, dtype=dtype, device=device)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_dimension, dtype=dtype, device=device)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return smoothly_bounded_likelihood_shift(
            self.output(self.activation(self.hidden(events)))
        )

    def evaluate(self, events: torch.Tensor) -> torch.Tensor:
        return self.forward(events)

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.hidden.weight, gain=gain)
        nn.init.uniform_(self.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(self.output.weight, gain=gain)
        nn.init.uniform_(self.output.bias, a=-0.3, b=0.3)
