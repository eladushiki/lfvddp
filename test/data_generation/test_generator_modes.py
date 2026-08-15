from pathlib import Path

import numpy as np
import pytest

from data_tools.data_utils import DataSet
from data_tools.dataset_config import GeneratedDatasetParameters
from data_tools.event_generation.background import GaussianBackground
from data_tools.event_generation.distribution import (
    normalize_generator_selection,
    validate_generated_dataset,
    validate_generated_dataset_within_integration_domain,
)
from data_tools.event_generation.signal import MultivariateGaussianSignal
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path(
            "test/data_generation/configs/dataset/generator_modes_dataset_config.json"
        ),
        ConfigType.DETECTOR.value: Path(
            "test/configs/detector/basic_2D_detector_config.json"
        ),
    }],
    indirect=True,
)
def test_joint_repeated_and_per_dimension_generators(
    tmp_path,
    monkeypatch,
    function_execution_context,
    data_generation,
):
    np.save(tmp_path / "events.npy", np.arange(6, dtype=float).reshape(3, 2))

    config = function_execution_context.config
    joint = config.get_parameters(DataSet.DataSetCategory.A_SR)
    assert config.get_parameters(DataSet.DataSetCategory.A_SR) is joint
    assert joint.dataset__background_source_type == "generated"
    assert "exponential_background" in joint.dataset__background_source
    assert "multivariate_gaussian_signal" in joint.dataset__signal_source
    assert "multivariate_gaussian_signal" in joint.dataset__signal_description
    joint_background, correlated_signal = joint.dataset__data
    assert joint_background.events.shape == (8, 2)
    assert correlated_signal.events.shape == (4000, 2)
    assert np.corrcoef(correlated_signal.events, rowvar=False)[0, 1] > 0.8
    np.testing.assert_allclose(
        joint.dataset_generated__signal_integration_upper_limits,
        np.array([7.0, 8.0]),
    )

    repeated = config.get_parameters(DataSet.DataSetCategory.B_SR)
    repeated_background, _ = repeated.dataset__data
    assert repeated_background.events.shape == (100, 2)
    assert np.all((repeated_background.events >= 1.0) & (repeated_background.events <= 2.0))
    marginal = GaussianBackground(
        1,
        domain_min=1.0,
        domain_max=2.0,
        mean=1.5,
    )
    np.testing.assert_allclose(
        repeated.dataset_generated__background_pdf(np.array([1.5, 1.5])),
        marginal.pdf(1.5) ** 2,
    )

    per_dimension = config.get_parameters(DataSet.DataSetCategory.A_CR)
    per_dimension_background, _ = per_dimension.dataset__data
    assert np.all((per_dimension_background.events[:, 0] >= 0.0))
    assert np.all((per_dimension_background.events[:, 0] <= 1.0))
    assert np.all((per_dimension_background.events[:, 1] >= 9.0))
    assert np.all((per_dimension_background.events[:, 1] <= 10.0))

    with monkeypatch.context() as temporary_working_directory:
        temporary_working_directory.chdir(tmp_path)
        loaded = config.get_parameters(DataSet.DataSetCategory.B_CR)
        loaded_background, repeated_signal = loaded.dataset__data
        loaded_with_signal, _ = data_generation[DataSet.DataSetCategory.B_CR]
    assert loaded.dataset__number_of_dimensions == 2
    assert loaded.dataset__background_source_type == "loaded"
    assert loaded.dataset__background_source == "events.npy"
    assert loaded_background.events.shape == (3, 2)
    assert repeated_signal.events.shape == (20, 2)
    assert loaded_with_signal.events.shape == (23, 2)
    np.testing.assert_allclose(
        loaded.dataset_generated__signal_integration_upper_limits,
        np.array([4.7, 4.7]),
    )


def test_generator_configuration_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_generator_selection([])
    with pytest.raises(ValueError, match="non-empty string"):
        normalize_generator_selection({"function": ""})
    with pytest.raises(TypeError, match="arguments.*object"):
        normalize_generator_selection({"function": "x", "arguments": []})
    with pytest.raises(ValueError, match="Unexpected generator"):
        normalize_generator_selection({"function": "x", "parameters": {}})

    parameters = GeneratedDatasetParameters(
        name="invalid-list",
        type="generated",
        category="a_sr",
        dataset_generated__number_of_dimensions=3,
        dataset_generated__background_generator=[
            {"function": "exponential_background"},
            {"function": "exponential_background"},
        ],
        dataset__number_of_background_events=1,
        dataset__number_of_signal_events=0,
    )
    with pytest.raises(ValueError, match="either one generator.*exactly 3"):
        _ = parameters.dataset__data

    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        validate_generated_dataset(DataSet(np.ones((2, 1))), 2, 2, "broken")


def test_generated_signal_must_fit_inside_integration_domain():
    dataset = DataSet(np.array([[0.5, 1.0], [1.5, 3.0]]))

    validate_generated_dataset_within_integration_domain(
        dataset,
        np.array([2.0, 3.0]),
        "signal",
    )
    with pytest.raises(ValueError, match="domain_max is too tight"):
        validate_generated_dataset_within_integration_domain(
            dataset,
            np.array([2.0, 2.5]),
            "signal",
        )


@pytest.mark.parametrize(
    ("mean", "covariance", "error"),
    [
        ([0.0], [[1.0, 0.0], [0.0, 1.0]], "mean must have shape"),
        ([0.0, 0.0], [[1.0]], "covariance must have shape"),
        ([0.0, np.inf], [[1.0, 0.0], [0.0, 1.0]], "must be finite"),
        ([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]], "must be symmetric"),
        ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ],
)
def test_multivariate_gaussian_validates_parameters(mean, covariance, error):
    with pytest.raises(ValueError, match=error):
        MultivariateGaussianSignal(2, mean=mean, covariance=covariance)


def test_signal_events_require_a_generator():
    with pytest.raises(ValueError, match="signal_generator"):
        GeneratedDatasetParameters(
            name="missing-signal",
            type="generated",
            category="a_sr",
            dataset_generated__number_of_dimensions=2,
            dataset_generated__background_generator={
                "function": "exponential_background"
            },
            dataset__number_of_background_events=1,
            dataset__number_of_signal_events=1,
        )
