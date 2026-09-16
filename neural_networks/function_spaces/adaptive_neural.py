"""Adaptive neural function-space family."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    PerEventFunctionSpace,
    immutable_options,
    unexpected_construction_options,
)
from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift
from train.function_space_config import FunctionSpaceFamily


class AdaptiveNeuralFunction(PerEventFunctionSpace):
    """One-hidden-layer bounded sigmoid network used by both learned roles."""

    family = FunctionSpaceFamily.ADAPTIVE_NEURAL
    metadata = FunctionSpaceMetadata(
        regularity=FunctionSpaceRegularity.ADAPTIVE,
        coefficient_topology=CoefficientTopology.DENSE_TWO_LAYER,
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

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        # Input width and hidden width may be derived from the current run by
        # TrainConfig, unlike fixed-family geometry.
        return None

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "AdaptiveNeuralFunction":
        """Construct the adaptive family from its configuration envelope."""

        input_dimension = construction.pop(
            "input_dimension",
            options.get("input_dimension"),
        )
        hidden_size = construction.pop(
            "hidden_size",
            options.get("hidden_size", options.get("hidden_layer_nodes")),
        )
        output_dimension = construction.pop(
            "output_dimension",
            options.get("output_dimension", 1),
        )
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        if input_dimension is None or hidden_size is None:
            raise ValueError(
                "adaptive_neural requires input_dimension and hidden_size or hidden_layer_nodes."
            )
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

    def evaluate(self, events: torch.Tensor) -> torch.Tensor:
        return self.forward(events)

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.hidden.weight, gain=gain)
        nn.init.uniform_(self.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(self.output.weight, gain=gain)
        nn.init.uniform_(self.output.bias, a=-0.3, b=0.3)
