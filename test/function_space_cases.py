"""Shared option values for function-space unit tests."""

from pathlib import Path

from test.environment import ConfigType

# File-backed architecture cases shared by model, checkpoint, and count coverage.
NEURAL_DEPTH_CASES = [
    ("adaptive_neural_nuisance_continuation", 1, (1, 4, 1), 13),
    ("neural_single_list_1D", 1, (1, 4, 1), 13),
    ("neural_deep_1D", 1, (1, 4, 3, 1), 27),
    ("neural_no_hidden_1D", 1, (1, 1), 2),
    ("neural_deep_2D", 2, (2, 4, 3, 2, 1), 38),
    ("neural_no_hidden_2D", 2, (2, 1), 3),
]

NEURAL_DEPTH_CONFIGS = [
    {
        ConfigType.DATASET: Path(
            f"test/configs/dataset/disjoint_{dimension}D_generated_dataset_config.json"
        ),
        ConfigType.DETECTOR: Path(
            f"test/configs/detector/basic_{dimension}D_detector_config.json"
        ),
        ConfigType.TRAIN: Path(f"test/configs/train/{name}.json"),
    }
    for name, dimension, _, _ in NEURAL_DEPTH_CASES
]

FUNCTION_SPACE_OPTIONS = {
    "adaptive_neural": {"hidden_layer_nodes": 2},
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
