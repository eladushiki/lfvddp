"""Nuisance parameter calculations used by differentiating models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from data_tools.data_utils import DataSet
from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    BinIndicatorFunction,
    create_function_space,
    initialize_function_space_parameters,
)
from neural_networks.likelihood_parameterization import LIKELIHOOD_SHIFT_BOUND
from train.function_space_config import FunctionSpaceFamily, RoleState


@dataclass(frozen=True)
class WeightedNuisanceValues:
    """Nuisance values with optional multiplicities for compact reductions."""

    values: torch.Tensor
    weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class NuisanceEvaluation:
    """Nuisance values and multiplicities used to assemble the training loss.

    Neural control-region values remain in their contiguous A/B event groups
    and therefore need no weights. Scalar values are shared occupied-bin
    evaluations whose weights preserve each category's event multiplicities.
    The differentiating model owns the loss formula.
    """

    nuisance_sr_values: torch.Tensor
    nuisance_cr_a: WeightedNuisanceValues
    nuisance_cr_b: WeightedNuisanceValues


@dataclass(frozen=True)
class PreparedNuisanceData:
    """Static nuisance inputs prepared once for full-batch training."""

    sr_inputs: torch.Tensor


class NuisanceCalculation(nn.Module, ABC):
    """Mode-specific nuisance preparation, evaluation, and CR reduction."""

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

    @abstractmethod
    def prepare(
        self,
        raw_sr: DataSet,
        raw_a_cr: DataSet,
        raw_b_cr: DataSet,
        normalized_sr: DataSet,
        normalized_a_cr: DataSet,
        normalized_b_cr: DataSet,
    ) -> PreparedNuisanceData:
        """Prepare mode-specific nuisance inputs."""

    @abstractmethod
    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        """Evaluate nuisance values and loss-assembly weights for both regions."""

    def initialize_parameters(self, gain: float) -> None:
        """Initialize trainable nuisance parameters, when present."""

    def clamp_parameters(self) -> None:
        """Clamp trainable nuisance parameters, when needed."""


def build_nuisance_calculation(
    config: Any,
    dtype: torch.dtype,
    device: torch.device,
    resolved_config: Any = None,
) -> NuisanceCalculation:
    """Build the nuisance role from the shared resolved function-space config."""
    if resolved_config is None:
        resolver = getattr(config, "resolve_function_space_config", None)
        if not callable(resolver):
            raise TypeError("Nuisance construction requires a TrainConfig resolver.")
        resolved_config = resolver()
    spec = resolved_config.nuisance
    if spec.state is RoleState.DISABLED:
        return BlankNuisanceEstimator(dtype=dtype, device=device)

    function_space = create_function_space(
        "nuisance",
        spec,
        dtype=dtype,
        device=device,
    )
    if spec.family is FunctionSpaceFamily.BIN_INDICATORS:
        return ScalarBinnedNuisanceEstimator(
            dtype=dtype,
            device=device,
            bin_lookup=function_space,
        )
    if not isinstance(function_space, nn.Module):
        raise TypeError(
            f"Nuisance family {spec.family.value!r} did not produce a trainable module."
        )
    return NeuralPerEventNuisanceEstimator(
        input_dimension=getattr(function_space, "input_dimension", config.train__nn_input_dimension),
        hidden_size=getattr(function_space, "hidden", None).out_features
        if hasattr(function_space, "hidden")
        else 0,
        output_dimension=getattr(function_space, "output_dimension", config.train__nn_output_dimension),
        dtype=dtype,
        device=device,
        network=function_space,
    )


class BlankNuisanceEstimator(NuisanceCalculation):
    """A zero-nuisance representation for runs without nuisance training."""

    def prepare(
        self,
        raw_sr: DataSet,
        raw_a_cr: DataSet,
        raw_b_cr: DataSet,
        normalized_sr: DataSet,
        normalized_a_cr: DataSet,
        normalized_b_cr: DataSet,
    ) -> PreparedNuisanceData:
        return PreparedNuisanceData(
            torch.empty(
                normalized_sr.n_samples,
                dtype=self._dtype,
                device=self._device,
            )
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        empty_control_region = torch.empty(
            0, dtype=self._dtype, device=self._device
        )
        return NuisanceEvaluation(
            nuisance_sr_values=torch.zeros(
                data.sr_inputs.shape[0],
                dtype=self._dtype,
                device=self._device,
            ),
            nuisance_cr_a=WeightedNuisanceValues(empty_control_region),
            nuisance_cr_b=WeightedNuisanceValues(empty_control_region),
        )


class ScalarBinnedNuisanceEstimator(NuisanceCalculation):
    """A bounded scalar nuisance value for every detector-bin combination."""

    @dataclass(frozen=True)
    class _PreparedData(PreparedNuisanceData):
        nuisance_cr_bin_indices: torch.Tensor
        nuisance_cr_a_multiplicities: torch.Tensor
        nuisance_cr_b_multiplicities: torch.Tensor

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        bin_lookup: BinIndicatorFunction,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self._bin_lookup = bin_lookup
        number_of_bins = self._bin_lookup.geometry.number_of_bins
        self._nuisance_deltas = nn.ParameterDict(
            {
                f"dimension_{index}": nn.Parameter(
                    torch.empty(nbins, dtype=dtype, device=device)
                )
                for index, nbins in enumerate(number_of_bins)
            }
        )

    def _bin_indices(self, data: DataSet) -> torch.Tensor:
        return torch.tensor(
            self._bin_lookup.evaluate(data),
            dtype=torch.long,
            device=self._device,
        )

    def _values(self, bin_indices: torch.Tensor) -> torch.Tensor:
        values: Optional[torch.Tensor] = None
        for dimension in range(len(self._bin_lookup.geometry.number_of_bins)):
            value = torch.index_select(
                self._nuisance_deltas[f"dimension_{dimension}"],
                0,
                bin_indices[:, dimension],
            )
            values = value if values is None else values * value
        if values is None:
            raise RuntimeError("Nuisance bin geometry has no dimensions.")
        return values.clamp(
            min=-LIKELIHOOD_SHIFT_BOUND,
            max=LIKELIHOOD_SHIFT_BOUND,
        )

    def prepare(
        self,
        raw_sr: DataSet,
        raw_a_cr: DataSet,
        raw_b_cr: DataSet,
        normalized_sr: DataSet,
        normalized_a_cr: DataSet,
        normalized_b_cr: DataSet,
    ) -> PreparedNuisanceData:
        cr_indices = torch.cat(
            (self._bin_indices(raw_a_cr), self._bin_indices(raw_b_cr))
        )
        unique_indices, inverse_indices = torch.unique(
            cr_indices,
            dim=0,
            return_inverse=True,
        )
        number_of_cr_bins = unique_indices.shape[0]
        return self._PreparedData(
            sr_inputs=self._bin_indices(raw_sr),
            nuisance_cr_bin_indices=unique_indices,
            nuisance_cr_a_multiplicities=torch.bincount(
                inverse_indices[: raw_a_cr.n_samples],
                minlength=number_of_cr_bins,
            ).to(self._dtype),
            nuisance_cr_b_multiplicities=torch.bincount(
                inverse_indices[raw_a_cr.n_samples :],
                minlength=number_of_cr_bins,
            ).to(self._dtype),
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        if not isinstance(data, self._PreparedData):
            raise TypeError("Scalar nuisance data was not prepared by this calculation.")

        nuisance_cr_values = self._values(data.nuisance_cr_bin_indices)
        return NuisanceEvaluation(
            nuisance_sr_values=self._values(data.sr_inputs),
            nuisance_cr_a=WeightedNuisanceValues(
                nuisance_cr_values,
                data.nuisance_cr_a_multiplicities,
            ),
            nuisance_cr_b=WeightedNuisanceValues(
                nuisance_cr_values,
                data.nuisance_cr_b_multiplicities,
            ),
        )

    def initialize_parameters(self, gain: float) -> None:
        for parameter in self._nuisance_deltas.values():
            nn.init.normal_(parameter, mean=0.0, std=1e-3)

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self._nuisance_deltas.values():
                parameter.clamp_(
                    min=-LIKELIHOOD_SHIFT_BOUND,
                    max=LIKELIHOOD_SHIFT_BOUND,
                )


class _ThetaEstimator(AdaptiveNeuralFunction):
    """Wrapper retaining the nuisance output shape."""

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return super().forward(events).squeeze(-1)


class NeuralPerEventNuisanceEstimator(NuisanceCalculation):
    """A neural nuisance function evaluated independently for each event."""

    @dataclass(frozen=True)
    class _PreparedData(PreparedNuisanceData):
        cr_inputs: torch.Tensor
        number_of_a_cr_events: int

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
        device: torch.device,
        network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self.network = network or create_function_space(
            "nuisance",
            "adaptive_neural",
            {
                "input_dimension": input_dimension,
                "hidden_layer_nodes": hidden_size,
                "output_dimension": output_dimension,
            },
            dtype=dtype,
            device=device,
        )

    def prepare(
        self,
        raw_sr: DataSet,
        raw_a_cr: DataSet,
        raw_b_cr: DataSet,
        normalized_sr: DataSet,
        normalized_a_cr: DataSet,
        normalized_b_cr: DataSet,
    ) -> PreparedNuisanceData:
        number_of_a_cr_events = normalized_a_cr.n_samples
        return self._PreparedData(
            sr_inputs=torch.tensor(
                normalized_sr.events,
                dtype=self._dtype,
                device=self._device,
            ),
            cr_inputs=torch.tensor(
                np.concatenate((normalized_a_cr.events, normalized_b_cr.events)),
                dtype=self._dtype,
                device=self._device,
            ),
            number_of_a_cr_events=number_of_a_cr_events,
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        if not isinstance(data, self._PreparedData):
            raise TypeError("Neural nuisance data was not prepared by this calculation.")

        def values(inputs: torch.Tensor) -> torch.Tensor:
            result = self.network(inputs)
            return result.squeeze(-1) if result.ndim == 2 else result

        nuisance_cr_values = values(data.cr_inputs)
        return NuisanceEvaluation(
            nuisance_sr_values=values(data.sr_inputs),
            nuisance_cr_a=WeightedNuisanceValues(
                nuisance_cr_values[: data.number_of_a_cr_events]
            ),
            nuisance_cr_b=WeightedNuisanceValues(
                nuisance_cr_values[data.number_of_a_cr_events :]
            ),
        )

    def initialize_parameters(self, gain: float) -> None:
        initialize_function_space_parameters(self.network, gain)
