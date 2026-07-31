import numpy as np
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
