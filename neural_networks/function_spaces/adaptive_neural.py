"""Adaptive neural function-space family."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from neural_networks.function_spaces.base import (
    PerEventFunctionSpace,
    immutable_options,
    scalar_output_dimension,
    unexpected_construction_options,
)
from neural_networks.likelihood_parameterization import (
    smoothly_bounded_likelihood_shift,
)
from train.function_space_config import adaptive_hidden_layer_sizes


class AdaptiveNeuralFunction(PerEventFunctionSpace):
    """Bounded sigmoid network with configurable hidden layers for either role."""

    family = "adaptive_neural"

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int | list[int] | tuple[int, ...],
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
        widths = adaptive_hidden_layer_sizes(hidden_size)
        self.hidden = nn.Linear(input_dimension, widths[0], dtype=dtype, device=device)
        self.additional_hidden = nn.ModuleList(
            nn.Linear(previous, following, dtype=dtype, device=device)
            for previous, following in zip(widths, widths[1:])
        )
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(
            widths[-1], output_dimension, dtype=dtype, device=device
        )

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        input_dimension = options.get("input_dimension")
        hidden_size = options.get("hidden_layer_nodes")
        if not isinstance(input_dimension, int) or input_dimension <= 0:
            raise ValueError("adaptive_neural requires a positive input_dimension.")
        adaptive_hidden_layer_sizes(hidden_size)
        scalar_output_dimension(options, cls.family)

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "AdaptiveNeuralFunction":
        """Construct the adaptive family from its configuration envelope."""

        input_dimension = options["input_dimension"]
        hidden_size = options["hidden_layer_nodes"]
        output_dimension = scalar_output_dimension(options, cls.family)
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        unexpected_construction_options(cls.family, construction)
        return cls(
            input_dimension=input_dimension,
            hidden_size=hidden_size,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=options,
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        values = self.activation(self.hidden(events))
        for layer in self.additional_hidden:
            values = self.activation(layer(values))
        return smoothly_bounded_likelihood_shift(self.output(values))

    def initialize_parameters(self, gain: float) -> None:
        for layer in (self.hidden, *self.additional_hidden, self.output):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.uniform_(layer.bias, a=-0.3, b=0.3)
