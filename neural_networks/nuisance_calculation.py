"""Nuisance parameter calculations used by differentiating models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from data_tools.data_utils import DataSet
from data_tools.detector.detector_effect import DetectorEffect


_NUISANCE_BOUND = 1.0 - 1e-6


@dataclass(frozen=True)
class NuisanceEvaluation:
    """Nuisance values and region weights used to assemble the training loss.

    Control-region values may be per event (neural nuisance) or per occupied
    detector bin (scalar nuisance); their weights preserve the corresponding
    event multiplicities.  The differentiating model owns the loss formula.
    """

    nuisance_sr_values: torch.Tensor
    nuisance_cr_values: torch.Tensor
    nuisance_cr_a_weights: torch.Tensor
    nuisance_cr_b_weights: torch.Tensor


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
            nuisance_cr_values=empty_control_region,
            nuisance_cr_a_weights=empty_control_region,
            nuisance_cr_b_weights=empty_control_region,
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
        detector_effect: DetectorEffect,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self._detector_effect = detector_effect
        self._observable_names = detector_effect._observable_names
        self._detector_deltas = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(nbins, dtype=dtype, device=device))
                for name, nbins in zip(
                    self._observable_names,
                    detector_effect._numbers_of_bins,
                )
            }
        )

    def _bin_indices(self, data: DataSet) -> torch.Tensor:
        return torch.tensor(
            self._detector_effect.get_event_bin_centers(data, indexed=True),
            dtype=torch.long,
            device=self._device,
        )

    def _values(self, bin_indices: torch.Tensor) -> torch.Tensor:
        values: Optional[torch.Tensor] = None
        for dimension, name in enumerate(self._observable_names):
            value = torch.index_select(
                self._detector_deltas[name],
                0,
                bin_indices[:, dimension],
            )
            values = value if values is None else values * value
        if values is None:
            raise RuntimeError("Detector nuisance configuration has no observables.")
        return values.clamp(min=-_NUISANCE_BOUND, max=_NUISANCE_BOUND)

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

        return NuisanceEvaluation(
            nuisance_sr_values=self._values(data.sr_inputs),
            nuisance_cr_values=self._values(data.nuisance_cr_bin_indices),
            nuisance_cr_a_weights=data.nuisance_cr_a_multiplicities,
            nuisance_cr_b_weights=data.nuisance_cr_b_multiplicities,
        )

    def initialize_parameters(self, gain: float) -> None:
        for parameter in self._detector_deltas.values():
            nn.init.normal_(parameter, mean=0.0, std=1e-3)

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self._detector_deltas.values():
                parameter.clamp_(min=-_NUISANCE_BOUND, max=_NUISANCE_BOUND)


class _ThetaEstimator(nn.Module):
    """Neural implementation of the nuisance function theta(x)."""

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_size, dtype=dtype)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_dimension, dtype=dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(events))).squeeze(-1)


class NeuralPerEventNuisanceEstimator(NuisanceCalculation):
    """A neural nuisance function evaluated independently for each event."""

    @dataclass(frozen=True)
    class _PreparedData(PreparedNuisanceData):
        cr_inputs: torch.Tensor
        a_cr_mask: torch.Tensor
        b_cr_mask: torch.Tensor

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self.network = _ThetaEstimator(
            input_dimension=input_dimension,
            hidden_size=hidden_size,
            output_dimension=output_dimension,
            dtype=dtype,
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
        number_of_b_cr_events = normalized_b_cr.n_samples
        a_cr_mask = torch.cat(
            (
                torch.ones(
                    number_of_a_cr_events,
                    dtype=torch.bool,
                    device=self._device,
                ),
                torch.zeros(
                    number_of_b_cr_events,
                    dtype=torch.bool,
                    device=self._device,
                ),
            )
        )
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
            a_cr_mask=a_cr_mask,
            b_cr_mask=~a_cr_mask,
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        if not isinstance(data, self._PreparedData):
            raise TypeError("Neural nuisance data was not prepared by this calculation.")

        return NuisanceEvaluation(
            nuisance_sr_values=self.network(data.sr_inputs).clamp(
                min=-_NUISANCE_BOUND,
                max=_NUISANCE_BOUND,
            ),
            nuisance_cr_values=self.network(data.cr_inputs).clamp(
                min=-_NUISANCE_BOUND,
                max=_NUISANCE_BOUND,
            ),
            nuisance_cr_a_weights=data.a_cr_mask.to(self._dtype),
            nuisance_cr_b_weights=data.b_cr_mask.to(self._dtype),
        )

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.network.hidden.weight, gain=gain)
        nn.init.uniform_(self.network.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(self.network.output.weight, gain=gain)
        nn.init.uniform_(self.network.output.bias, a=-0.3, b=0.3)
