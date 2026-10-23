from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.stats import halfnorm

from data_tools.data_utils import DataSet
from data_tools.event_generation.positive_signal import PositiveSignalDistribution
from data_tools.event_generation.signal import (
    GaussianSignal,
    MultivariateGaussianSignal,
    NoSignal,
)
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [{ConfigType.DATASET: Path("test/configs/dataset/positive_signal.json")}],
    indirect=True,
)
def test_positive_signals_preserve_counts_and_conditional_statistics(
    function_execution_context, tmp_path, monkeypatch
):
    np.save(tmp_path / "events.npy", np.ones((10, 2)))
    monkeypatch.chdir(tmp_path)
    for category in ("A_SR", "B_SR", "A_CR", "B_CR"):
        parameters = function_execution_context.config.get_parameters(
            DataSet.DataSetCategory[category]
        )
        assert parameters.dataset__signal_positive_only is True
        _, signal = parameters.dataset__data
        assert signal.events.shape == (4000, 2)
        assert np.all(signal.events >= 0)
        assert np.all(signal.events != 0)  # No clipping/pileup at the boundary.
        pdf = parameters.dataset_generated__signal_pdf
        assert pdf(np.array([-0.5, 0.5])) == 0
        integral = dblquad(lambda y, x, pdf=pdf: pdf(np.array([x, y])), 0, 12, 0, 12)[0]
        assert integral == pytest.approx(1, abs=2e-6)
        if category == "B_SR":
            assert np.corrcoef(signal.events, rowvar=False)[0, 1] > 0.4
        else:
            scales = np.array([1, 2]) if category == "A_CR" else np.ones(2)
            np.testing.assert_allclose(
                signal.events.mean(axis=0), halfnorm.mean() * scales, atol=0.07
            )
            np.testing.assert_allclose(
                signal.events.var(axis=0), halfnorm.var() * scales**2, atol=0.15
            )


@pytest.mark.parametrize(
    "function_execution_context",
    [{ConfigType.DATASET: Path("test/configs/dataset/unrestricted_signal.json")}],
    indirect=True,
)
def test_positive_signal_toggle_restores_unrestricted_distribution(
    function_execution_context,
):
    parameters = function_execution_context.config.get_parameters(
        DataSet.DataSetCategory.A_SR
    )
    assert parameters.dataset__signal_positive_only is False
    _, signal = parameters.dataset__data
    assert signal.events.shape == (4000, 2)
    assert np.any(signal.events < 0)
    np.testing.assert_allclose(signal.events.mean(axis=0), 0, atol=0.08)
    np.testing.assert_allclose(signal.events.var(axis=0), 1, atol=0.12)
    assert parameters.dataset_generated__signal_pdf(np.array([-0.5, 0.5])) > 0


def test_positive_signal_one_dimension_and_empty_sample():
    distribution = PositiveSignalDistribution(GaussianSignal(1, 0, 1), 1)
    assert distribution.generate_amount(0).events.shape == (0, 1)
    assert distribution.generate_amount(100).events.shape == (100, 1)
    assert distribution.pdf(-1) == 0
    np.testing.assert_allclose(distribution.pdf(np.array([0, 1])), halfnorm.pdf([0, 1]))
    assert quad(distribution.pdf, 0, np.inf)[0] == pytest.approx(1)


def test_positive_signal_impossible_region_fails_without_hanging():
    distribution = PositiveSignalDistribution(
        MultivariateGaussianSignal(2, [-1, -1], [[1, -1], [-1, 1]]), 2
    )
    with pytest.raises(ValueError, match="positive-region probability"):
        distribution.generate_amount(1)


def test_positive_signal_rare_region_has_bounded_sampling():
    distribution = PositiveSignalDistribution(
        GaussianSignal(1, -1, 0.1, domain_max=1), 1
    )
    with pytest.raises(ValueError, match="draw budget"):
        distribution.generate_amount(1)

    with pytest.raises(ValueError, match="positive-region probability"):
        PositiveSignalDistribution(NoSignal(1), 1).generate_amount(1)
