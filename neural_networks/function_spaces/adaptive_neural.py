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
from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift


class AdaptiveNeuralFunction(PerEventFunctionSpace):
    """One-hidden-layer bounded sigmoid network used by both learned roles."""

    family = "adaptive_neural"

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

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        input_dimension = options.get("input_dimension")
        hidden_size = options.get("hidden_layer_nodes")
        if not isinstance(input_dimension, int) or input_dimension <= 0:
            raise ValueError("adaptive_neural requires a positive input_dimension.")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                "adaptive_neural requires a positive hidden_layer_nodes."
            )
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
        return smoothly_bounded_likelihood_shift(
            self.output(self.activation(self.hidden(events)))
        )

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.hidden.weight, gain=gain)
        nn.init.uniform_(self.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(self.output.weight, gain=gain)
        nn.init.uniform_(self.output.bias, a=-0.3, b=0.3)
