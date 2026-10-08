"""Adaptive neural function-space family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from neural_networks.function_spaces.base import (
    SCALAR_OUTPUT_DIMENSION,
    PerEventFunctionSpace,
    immutable_options,
    scalar_output_dimension,
    unexpected_construction_options,
)
from neural_networks.function_spaces.neural_architecture import (
    hidden_layer_widths,
    neural_architecture,
    neural_parameter_count,
)
from neural_networks.likelihood_parameterization import (
    smoothly_bounded_likelihood_shift,
)


class AdaptiveNeuralFunction(PerEventFunctionSpace):
    """Fully connected bounded sigmoid network used by both learned roles."""

    family = "adaptive_neural"

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int | list[int] | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.architecture = neural_architecture(input_dimension, hidden_size)
        self.options = immutable_options(options or {})
        self.input_dimension = input_dimension
        self.hidden_size = (
            hidden_size if type(hidden_size) is int else tuple(hidden_size)
        )
        self.output_dimension = SCALAR_OUTPUT_DIMENSION
        # Retain historical state_dict keys and random initialization order.
        self.hidden = (
            nn.Linear(input_dimension, self.architecture[1], dtype=dtype, device=device)
            if len(self.architecture) > 2
            else None
        )
        self.activation = nn.Sigmoid()
        self.additional_hidden = nn.ModuleList(
            nn.Linear(source, destination, dtype=dtype, device=device)
            for source, destination in zip(
                self.architecture[1:-2], self.architecture[2:-1]
            )
        )
        self.output = nn.Linear(
            self.architecture[-2], SCALAR_OUTPUT_DIMENSION, dtype=dtype, device=device
        )

    @classmethod
    def architecture_from_options(
        cls, options: Mapping[str, Any], *, observable_count: int
    ) -> tuple[int, ...]:
        """Resolve the same validated architecture for construction and counting."""

        return neural_architecture(
            observable_count,
            options.get("hidden_layer_nodes"),
        )

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        if "input_dimension" in options:
            raise ValueError(
                "adaptive_neural input_dimension is inferred from observables and is not configurable."
            )
        scalar_output_dimension(options, cls.family)
        hidden_layer_widths(options.get("hidden_layer_nodes"))

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> AdaptiveNeuralFunction:
        """Construct the adaptive family from its configuration envelope."""

        hidden_size = options["hidden_layer_nodes"]
        observable_names = construction.pop("observable_names", None)
        if observable_names is None:
            raise ValueError(
                "adaptive_neural requires observable_names to infer its input width."
            )
        input_dimension = len(tuple(observable_names))
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        unexpected_construction_options(cls.family, construction)
        return cls(
            input_dimension=input_dimension,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            options=options,
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            events = self.activation(layer(events))
        return smoothly_bounded_likelihood_shift(self.output(events))

    @property
    def hidden_layers(self) -> tuple[nn.Linear, ...]:
        """Return hidden layers in evaluation order without registering aliases."""

        first = () if self.hidden is None else (self.hidden,)
        return (*first, *self.additional_hidden)

    def initialize_parameters(self, gain: float) -> None:
        for layer in (*self.hidden_layers, self.output):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.uniform_(layer.bias, a=-0.3, b=0.3)

    def statistical_degrees_of_freedom(self) -> int:
        """Return the raw parameter count used for diagnostic chi-square references."""

        return neural_parameter_count(self.architecture)

    @classmethod
    def analytic_degrees_of_freedom(
        cls, options: Mapping[str, Any], *, observable_count: int
    ) -> int:
        """Count configured parameters without allocating a network."""

        return neural_parameter_count(
            cls.architecture_from_options(options, observable_count=observable_count)
        )
