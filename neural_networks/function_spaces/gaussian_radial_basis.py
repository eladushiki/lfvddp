"""Gaussian radial-basis function-space family."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from neural_networks.function_spaces.base import EventInput, FunctionSpaceMetadata
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction
from train.function_space_config import FunctionSpaceFamily


class GaussianRadialBasisFunction(CenteredFeatureFunction):
    """Fixed-centre Gaussian radial features with trainable output coefficients only."""

    family = FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS
    metadata = FunctionSpaceMetadata("smooth", "linear_coefficients")

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "GaussianRadialBasisFunction":
        return cls(
            CenterGeometry.from_options(options, "gaussian_radial_basis"),
            options=options,
            **construction,
        )

    def features(self, events: EventInput) -> torch.Tensor:
        return torch.exp(-0.5 * self.centered_values(events).square().sum(dim=2))
