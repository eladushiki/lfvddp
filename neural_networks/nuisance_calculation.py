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
    """Nuisance values and CR multiplicities used by the common loss assembly."""

    sr_values: torch.Tensor
    cr_values: torch.Tensor
    a_cr_weights: torch.Tensor
    b_cr_weights: torch.Tensor


@dataclass(frozen=True)
class PreparedNuisanceData:
    """Static nuisance inputs prepared once for full-batch training."""

    sr_inputs: torch.Tensor
    cr_inputs: Optional[torch.Tensor]
    a_cr_weights: torch.Tensor
    b_cr_weights: torch.Tensor


class NuisanceCalculation(nn.Module, ABC):
    """Mode-specific nuisance preparation and evaluation."""

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
        """Prepare mode-specific inputs and CR weights."""

    @abstractmethod
    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        """Evaluate current parameters into the uniform loss representation."""

    def initialize_parameters(self, gain: float) -> None:
        """Initialize trainable nuisance parameters, when present."""

    def clamp_parameters(self) -> None:
        """Keep nuisance parameters in their valid domain, when needed."""


class NoNuisanceCalculation(NuisanceCalculation):
    """A zero-nuisance representation for runs without nuisance training."""

    def prepare(self, raw_sr: DataSet, raw_a_cr: DataSet, raw_b_cr: DataSet, normalized_sr: DataSet, normalized_a_cr: DataSet, normalized_b_cr: DataSet) -> PreparedNuisanceData:
        empty = torch.empty(0, dtype=self._dtype, device=self._device)
        return PreparedNuisanceData(
            torch.empty(normalized_sr.n_samples, dtype=self._dtype, device=self._device),
            None,
            empty,
            empty,
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        return NuisanceEvaluation(
            torch.zeros(data.sr_inputs.shape[0], dtype=self._dtype, device=self._device),
            data.a_cr_weights,
            data.a_cr_weights,
            data.b_cr_weights,
        )


class ScalarBinnedNuisanceCalculation(NuisanceCalculation):
    """A bounded scalar nuisance value for every detector-bin combination."""

    def __init__(self, detector_effect: DetectorEffect, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(dtype, device)
        self._detector_effect = detector_effect
        self._observable_names = detector_effect._observable_names
        self._detector_deltas = nn.ParameterDict({
            name: nn.Parameter(torch.empty(nbins, dtype=dtype, device=device))
            for name, nbins in zip(self._observable_names, detector_effect._numbers_of_bins)
        })

    def _bin_indices(self, data: DataSet) -> torch.Tensor:
        return torch.tensor(
            self._detector_effect.get_event_bin_centers(data, indexed=True),
            dtype=torch.long,
            device=self._device,
        )

    def _values(self, bin_indices: torch.Tensor) -> torch.Tensor:
        values: Optional[torch.Tensor] = None
        for dimension, name in enumerate(self._observable_names):
            value = torch.index_select(self._detector_deltas[name], 0, bin_indices[:, dimension])
            values = value if values is None else values * value
        if values is None:
            raise RuntimeError("Detector nuisance configuration has no observables.")
        return values.clamp(min=-_NUISANCE_BOUND, max=_NUISANCE_BOUND)

    def prepare(self, raw_sr: DataSet, raw_a_cr: DataSet, raw_b_cr: DataSet, normalized_sr: DataSet, normalized_a_cr: DataSet, normalized_b_cr: DataSet) -> PreparedNuisanceData:
        cr_indices = torch.cat((self._bin_indices(raw_a_cr), self._bin_indices(raw_b_cr)))
        unique_indices, inverse_indices = torch.unique(cr_indices, dim=0, return_inverse=True)
        number_of_bins = unique_indices.shape[0]
        a_weights = torch.bincount(inverse_indices[:raw_a_cr.n_samples], minlength=number_of_bins).to(self._dtype)
        b_weights = torch.bincount(inverse_indices[raw_a_cr.n_samples:], minlength=number_of_bins).to(self._dtype)
        return PreparedNuisanceData(self._bin_indices(raw_sr), unique_indices, a_weights, b_weights)

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        if data.cr_inputs is None:
            raise RuntimeError("Scalar nuisance control-region bins were not prepared.")
        return NuisanceEvaluation(
            self._values(data.sr_inputs),
            self._values(data.cr_inputs),
            data.a_cr_weights,
            data.b_cr_weights,
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

    def __init__(self, input_dimension: int, hidden_size: int, output_dimension: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_size, dtype=dtype)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_dimension, dtype=dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(events))).squeeze(-1)


class NeuralPerEventNuisanceCalculation(NuisanceCalculation):
    """A neural nuisance function evaluated independently for each event."""

    def __init__(self, input_dimension: int, hidden_size: int, output_dimension: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(dtype, device)
        self.network = _ThetaEstimator(input_dimension, hidden_size, output_dimension, dtype)

    def prepare(self, raw_sr: DataSet, raw_a_cr: DataSet, raw_b_cr: DataSet, normalized_sr: DataSet, normalized_a_cr: DataSet, normalized_b_cr: DataSet) -> PreparedNuisanceData:
        return PreparedNuisanceData(
            torch.tensor(normalized_sr.events, dtype=self._dtype, device=self._device),
            torch.tensor(np.concatenate((normalized_a_cr.events, normalized_b_cr.events)), dtype=self._dtype, device=self._device),
            torch.cat((
                torch.ones(normalized_a_cr.n_samples, dtype=self._dtype, device=self._device),
                torch.zeros(normalized_b_cr.n_samples, dtype=self._dtype, device=self._device),
            )),
            torch.cat((
                torch.zeros(normalized_a_cr.n_samples, dtype=self._dtype, device=self._device),
                torch.ones(normalized_b_cr.n_samples, dtype=self._dtype, device=self._device),
            )),
        )

    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        if data.cr_inputs is None:
            raise RuntimeError("Neural nuisance control-region events were not prepared.")
        return NuisanceEvaluation(
            self.network(data.sr_inputs).clamp(min=-_NUISANCE_BOUND, max=_NUISANCE_BOUND),
            self.network(data.cr_inputs).clamp(min=-_NUISANCE_BOUND, max=_NUISANCE_BOUND),
            data.a_cr_weights,
            data.b_cr_weights,
        )

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.network.hidden.weight, gain=gain)
        nn.init.uniform_(self.network.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(self.network.output.weight, gain=gain)
        nn.init.uniform_(self.network.output.bias, a=-0.3, b=0.3)
