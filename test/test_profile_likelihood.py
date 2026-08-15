import numpy as np
import pytest
from scipy.stats import norm

from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)


def test_continuous_injected_significance_handles_pdf_underflow():
    n_background_events = 10_000
    n_signal_events = 100
    upper_limit = 782.0632583774859
    expected_integral = (
        (n_signal_events + n_background_events)
        * np.log1p(n_signal_events / n_background_events)
        * (1 - np.exp(-upper_limit))
    )
    expected_significance = np.sqrt(
        2 * (-n_signal_events + expected_integral)
    )

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda x: np.exp(-x),
        signal_pdf=lambda x: np.exp(-x),
        n_background_events=n_background_events,
        n_signal_events=n_signal_events,
        upper_limit=upper_limit,
    )

    np.testing.assert_allclose(significance, expected_significance)


def test_continuous_injected_significance_resolves_narrow_signal_on_wide_domain():
    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda x: np.exp(-x),
        signal_pdf=lambda x: norm.pdf(x, loc=6.4, scale=0.16),
        n_background_events=10_000,
        n_signal_events=100,
        upper_limit=782.0632583774859,
    )

    np.testing.assert_allclose(significance, 17.96065413689458)


def test_gaussian_tail_limit_preserves_significance_within_point_one_percent():
    calculation_arguments = {
        "background_pdf": lambda x: np.exp(-x),
        "signal_pdf": lambda x: norm.pdf(x, loc=6.4, scale=0.16),
        "n_background_events": 10_000,
        "n_signal_events": 100,
    }
    unbounded_significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        **calculation_arguments,
    )
    finite_significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        **calculation_arguments,
        upper_limit=6.4 + 4.5 * 0.16,
    )

    relative_difference = abs(
        finite_significance / unbounded_significance - 1
    )
    assert relative_difference < 0.001


def test_continuous_injected_significance_matches_1d_for_uniform_2d_pdf():
    n_background_events = 10_000
    n_signal_events = 100
    expected_significance = np.sqrt(
        2 * (
            -n_signal_events
            + (n_background_events + n_signal_events)
            * np.log1p(n_signal_events / n_background_events)
        )
    )

    one_dimensional_significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda coordinate: 1.0,
        signal_pdf=lambda coordinate: 1.0,
        n_background_events=n_background_events,
        n_signal_events=n_signal_events,
        upper_limit=1.0,
    )

    def two_dimensional_uniform_pdf(coordinates):
        assert coordinates.shape == (2,)
        return 1.0

    two_dimensional_significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=two_dimensional_uniform_pdf,
        signal_pdf=two_dimensional_uniform_pdf,
        n_background_events=n_background_events,
        n_signal_events=n_signal_events,
        upper_limit=np.array([1.0, 1.0]),
    )

    assert np.isfinite(one_dimensional_significance)
    assert np.isfinite(two_dimensional_significance)
    np.testing.assert_allclose(one_dimensional_significance, expected_significance)
    np.testing.assert_allclose(two_dimensional_significance, expected_significance)


@pytest.mark.parametrize("invalid_density", [np.nan, np.inf, -np.inf])
def test_continuous_injected_significance_rejects_nonfinite_pdf_density(invalid_density):
    with pytest.raises(ValueError, match="finite scalar density"):
        calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=lambda coordinate: invalid_density,
            signal_pdf=lambda coordinate: 1.0,
            n_background_events=10,
            n_signal_events=1,
            upper_limit=1.0,
        )


def test_continuous_injected_significance_rejects_nonfinite_upper_limit():
    with pytest.raises(ValueError, match="positive numbers or infinity"):
        calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=lambda coordinate: 1.0,
            signal_pdf=lambda coordinate: 1.0,
            n_background_events=10,
            n_signal_events=1,
            upper_limit=np.array([1.0, np.nan]),
        )
