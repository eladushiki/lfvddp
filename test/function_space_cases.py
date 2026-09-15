"""Shared transient training-config fixtures for function-space integration tests."""

from test.environment import TrainConfigFixture

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


def function_space_train_config(*, dimension: int = 1, f, nuisance, epochs: int = 1):
    """Materialize one small training configuration for a function-space case."""

    return TrainConfigFixture(
        {
            "random_seed": 18018,
            "train__epochs": epochs,
            "train__number_of_epochs_for_checkpoint": 1,
            "train__enable_progress_bar": False,
            "train__nn_input_dimension": dimension,
            "train__nn_inner_layer_nodes": 4,
            "train__learning_rate": 0.01,
            "train__final_learning_rate": 0.01,
            "train__f": f,
            "train__nuisance": nuisance,
        }
    )
