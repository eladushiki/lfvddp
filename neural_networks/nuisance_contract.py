"""Dependency-free contracts shared by nuisance implementations and function spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from data_tools.data_utils import DataSet


@dataclass(frozen=True)
class WeightedNuisanceValues:
    """Nuisance values with optional multiplicities for compact reductions."""

    values: torch.Tensor
    weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class NuisanceEvaluation:
    """Nuisance values for each A/B signal/control category.

    Signal-region values are kept per event because they multiply the
    per-event signal hypothesis. Control-region values may instead be reduced
    to occupied bins, with weights retaining their multiplicities.
    """

    a_sr: WeightedNuisanceValues
    b_sr: WeightedNuisanceValues
    a_cr: WeightedNuisanceValues
    b_cr: WeightedNuisanceValues


@dataclass(frozen=True)
class PreparedNuisanceData:
    """Static category inputs prepared once for full-batch training."""

    a_sr_inputs: torch.Tensor
    b_sr_inputs: torch.Tensor


class NuisanceCalculation(nn.Module, ABC):
    """Mode-specific nuisance preparation, evaluation, and CR reduction."""

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

    @abstractmethod
    def prepare(
        self,
        raw_a_sr: DataSet,
        raw_b_sr: DataSet,
        raw_a_cr: DataSet,
        raw_b_cr: DataSet,
        normalized_a_sr: DataSet,
        normalized_b_sr: DataSet,
        normalized_a_cr: DataSet,
        normalized_b_cr: DataSet,
    ) -> PreparedNuisanceData:
        """Prepare mode-specific nuisance inputs."""

    @abstractmethod
    def evaluate(self, data: PreparedNuisanceData) -> NuisanceEvaluation:
        """Evaluate nuisance values and loss-assembly weights for both regions."""

    @abstractmethod
    def prediction_values(
        self,
        raw_data: DataSet,
        normalized_data: DataSet,
    ) -> torch.Tensor:
        """Evaluate the nuisance shift for prediction inputs."""

    def initialize_parameters(self, gain: float) -> None:
        """Initialize trainable nuisance parameters, when present."""

    def clamp_parameters(self) -> None:
        """Clamp trainable nuisance parameters, when needed."""
