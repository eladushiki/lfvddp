"""Fixed-sigmoid function-space family."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from neural_networks.function_spaces.base import EventInput, tensor_product_basis_enabled
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction


class FixedSigmoidFunction(CenteredFeatureFunction):
    """Fixed-centre sigmoid features with trainable output coefficients only."""

    family = "fixed_sigmoid"

    @classmethod
    def geometry_from_options(cls, options: Mapping[str, Any]) -> CenterGeometry:
        if tensor_product_basis_enabled(options, cls.family):
            return CenterGeometry.tensor_product_from_options(options, cls.family)
        return CenterGeometry.from_options(options, cls.family)

    def features(self, events: EventInput) -> torch.Tensor:
        return torch.sigmoid(self.centered_values(events)).prod(dim=2)
