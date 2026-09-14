"""Gaussian radial-basis function-space family."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    EventInput,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
)
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction
from train.function_space_config import FunctionSpaceFamily


class GaussianRadialBasisFunction(CenteredFeatureFunction):
    """Fixed-centre Gaussian radial features with trainable output coefficients only."""

    family = FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS
    metadata = FunctionSpaceMetadata(
        FunctionSpaceRegularity.SMOOTH,
        CoefficientTopology.LINEAR_COEFFICIENTS,
    )

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "GaussianRadialBasisFunction":
        return cls(
            CenterGeometry.from_options(options, "gaussian_radial_basis"),
            options=options,
            **cls.construction_kwargs(options, construction),
        )

    def features(self, events: EventInput) -> torch.Tensor:
        return torch.exp(-0.5 * self.centered_values(events).square().sum(dim=2))
