"""Shared option values for function-space unit tests."""

from pathlib import Path

from test.environment import ConfigType

FUNCTION_SPACE_OPTIONS = {
    "adaptive_neural": {"input_dimension": 1, "hidden_layer_nodes": 2},
    "bin_indicators": {"minima": [0.0], "maxima": [3.0], "number_of_bins": [3]},
    "cubic_bspline": {"knots": [0.0, 1.0, 2.0, 3.0]},
    "orthogonal_polynomial": {
        "basis": "legendre",
        "maximum_degree": 2,
        "domain": [0.0, 3.0],
    },
    "fixed_sigmoid": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
    "gaussian_radial_basis": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
}


# File-backed V7 architectures, with unchanged single-layer nuisance widths.
ADAPTIVE_DIMENSION_CASES = [
    {
        ConfigType.DATASET: Path(
            f"test/configs/dataset/disjoint_{dimension}D_generated_dataset_config.json"
        ),
        ConfigType.DETECTOR: Path(
            f"test/configs/detector/basic_{dimension}D_detector_config.json"
        ),
        ConfigType.TRAIN: Path("test/configs/train")
        / (
            "adaptive_neural_nuisance.json"
            if dimension == 1
            else f"adaptive_neural_{dimension}D_multilayer_nuisance.json"
        ),
    }
    for dimension in (1, 2, 4)
]
