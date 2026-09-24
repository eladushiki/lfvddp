"""Nuisance parameter calculations used by differentiating models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from data_tools.data_utils import DataSet, ShiftAndNormalizationFactor
from neural_networks.function_spaces import create_function_space
from neural_networks.function_spaces.base import PerEventFunctionSpace
from neural_networks.function_spaces.bin_indicators import BinIndicatorFunction
from neural_networks.nuisance_contract import (
    NuisanceCalculation,
    NuisanceEvaluation,
    PreparedNuisanceData,
    WeightedNuisanceValues,
)
from train.function_space_config import (
    FunctionSpaceRole,
    ResolvedFunctionSpaceConfig,
    RoleState,
)
from train.train_config import TrainConfig


def build_nuisance_calculation(
    config: TrainConfig,
    dtype: torch.dtype,
    device: torch.device,
    resolved_config: Optional[ResolvedFunctionSpaceConfig] = None,
    normalization_factor: Optional[ShiftAndNormalizationFactor] = None,
    observable_names: Optional[Iterable[str]] = None,
) -> NuisanceCalculation:
    """Build the nuisance role from the same resolved shape used for ``f``."""
    if resolved_config is None:
        resolved_config = config.resolve_function_space_config()
    spec = resolved_config.nuisance
    if spec.state is RoleState.DISABLED:
        return NullNuisanceCalculation(dtype=dtype, device=device)

    if (normalization_factor is None) != (observable_names is None):
        raise ValueError(
            "Nuisance construction requires both normalization and observables."
        )

    function_space = create_function_space(
        FunctionSpaceRole.NUISANCE,
        spec,
        dtype=dtype,
        device=device,
    )
    return function_space.build_nuisance_calculation(
        dtype=dtype,
        device=device,
        normalization_factor=normalization_factor,
        observable_names=observable_names,
    )


class NullNuisanceCalculation(NuisanceCalculation):
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

    def prediction_values(
        self,
        raw_data: DataSet,
        normalized_data: DataSet,
    ) -> torch.Tensor:
        return torch.zeros(raw_data.n_samples, dtype=self._dtype, device=self._device)


class BinnedNuisanceCalculation(NuisanceCalculation):
    """Control-region reduction for a shared bin-indicator function space."""

    @dataclass(frozen=True)
    class _PreparedData(PreparedNuisanceData):
        nuisance_cr_bin_indices: torch.Tensor
        nuisance_cr_a_multiplicities: torch.Tensor
        nuisance_cr_b_multiplicities: torch.Tensor

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        function_space: BinIndicatorFunction,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self.function_space = function_space

    def _bin_indices(self, data: DataSet) -> torch.Tensor:
        return torch.tensor(
            self.function_space.bin_indices(data.events),
            dtype=torch.long,
            device=self._device,
        )

    def _values(self, bin_indices: torch.Tensor) -> torch.Tensor:
        return self.function_space.values_from_indices(bin_indices)

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
            raise TypeError("Binned nuisance data was not prepared by this calculation.")

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
        self.function_space.initialize_parameters(gain)

    def statistical_design_matrix(
        self,
        data: PreparedNuisanceData,
    ) -> torch.Tensor:
        """Use the occupied signal-region bins as the nuisance tangent design."""

        if not isinstance(data, self._PreparedData):
            raise TypeError("Binned nuisance data was not prepared by this calculation.")
        return self.function_space.statistical_design_matrix_from_indices(
            data.sr_inputs
        )

    def clamp_parameters(self) -> None:
        self.function_space.clamp_parameters()

    def prediction_values(
        self,
        raw_data: DataSet,
        normalized_data: DataSet,
    ) -> torch.Tensor:
        return self._values(self._bin_indices(raw_data))


class PerEventNuisanceEstimator(NuisanceCalculation):
    """A nuisance function evaluated independently for each event."""

    @dataclass(frozen=True)
    class _PreparedData(PreparedNuisanceData):
        cr_inputs: torch.Tensor
        number_of_a_cr_events: int

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        network: PerEventFunctionSpace,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self.network = network

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
            raise TypeError("Per-event nuisance data was not prepared by this calculation.")

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
        self.network.initialize_parameters(gain)

    def statistical_design_matrix(
        self,
        data: PreparedNuisanceData,
    ) -> Optional[torch.Tensor]:
        """Delegate fixed-family rank diagnostics to the wrapped network."""

        if not isinstance(data, self._PreparedData):
            raise TypeError("Per-event nuisance data was not prepared by this calculation.")
        return self.network.statistical_design_matrix(data.sr_inputs)

    def prediction_values(
        self,
        raw_data: DataSet,
        normalized_data: DataSet,
    ) -> torch.Tensor:
        values = self.network(
            torch.tensor(
                normalized_data.events,
                dtype=self._dtype,
                device=self._device,
            )
        )
        return values.squeeze(-1) if values.ndim == 2 else values
