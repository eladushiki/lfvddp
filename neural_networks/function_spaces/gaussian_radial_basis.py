"""Gaussian radial-basis function-space family."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from neural_networks.function_spaces.base import EventInput
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction


class GaussianRadialBasisFunction(CenteredFeatureFunction):
    """Fixed-centre Gaussian radial features with trainable output coefficients only."""

    family = "gaussian_radial_basis"

    @classmethod
    def geometry_from_options(cls, options: Mapping[str, Any]) -> CenterGeometry:
        return CenterGeometry.from_options(options, cls.family)

    def features(self, events: EventInput) -> torch.Tensor:
        return torch.exp(-0.5 * self.centered_values(events).square().sum(dim=2))
