import json

import numpy as np
import pytest
from scipy.stats import norm

import data_tools.profile_likelihood as profile_likelihood
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_binned,
    calc_injected_t_significance_by_sqrt_q0_continuous,
    calc_median_t_significance_relative_to_background,
    calc_t_significance_relative_to_background,
)
from data_tools.signal_calibration import (
    calc_n_signal_events_for_generated_signal,
    calc_n_signal_events_for_target_injected_t_significance,
    main as calibrate_signal_events,
)


def test_median_t_significance_uses_the_signal_distribution_median():
    background_t_values = np.arange(100, dtype=float)
    signal_t_values = np.array([0.0, 30.0, 90.0])

    significance = calc_median_t_significance_relative_to_background(
        background_t_values, signal_t_values
    )

    assert significance == pytest.approx(
        calc_t_significance_relative_to_background(
            np.median(signal_t_values), background_t_values
        )
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
    expected_significance = np.sqrt(2 * (-n_signal_events + expected_integral))

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda x: np.exp(-x),
        signal_pdf=lambda x: np.exp(-x),
        n_background_events=n_background_events,
        n_signal_events=n_signal_events,
        upper_limit=upper_limit,
    )

    np.testing.assert_allclose(significance, expected_significance)


def test_binned_injected_significance_matches_evident_formula():
    background_bin_counts = np.asarray([50.0, 50.0])
    signal_bin_counts = np.asarray([0.0, 25.0])
    expected_significance = np.sqrt(2 * (-25.0 + 75.0 * np.log(75.0 / 50.0)))

    significance = calc_injected_t_significance_by_sqrt_q0_binned(
        background_bin_counts,
        signal_bin_counts,
    )

    np.testing.assert_allclose(significance, expected_significance)


def test_binned_injected_significance_rejects_signal_without_background():
    with pytest.raises(ValueError, match="zero expected background"):
        calc_injected_t_significance_by_sqrt_q0_binned(
            background_bin_counts=np.asarray([0.0]),
            signal_bin_counts=np.asarray([1.0]),
        )


def test_signal_event_calibration_inverts_continuous_significance():
    target_significance = 1.5
    n_signal_events = calc_n_signal_events_for_target_injected_t_significance(
        background_pdf=lambda x: 1.0,
        signal_pdf=lambda x: 1.0,
        n_background_events=10_000,
        target_significance=target_significance,
        upper_limit=1.0,
    )

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda x: 1.0,
        signal_pdf=lambda x: 1.0,
        n_background_events=10_000,
        n_signal_events=n_signal_events,
        upper_limit=1.0,
    )

    assert n_signal_events > 0
    np.testing.assert_allclose(significance, target_significance, rtol=1e-6)


def test_signal_event_calibration_reuses_signal_generator_parameters():
    n_signal_events = calc_n_signal_events_for_generated_signal(
        background_generator={"function": "exponential_background"},
        signal_generator={
            "function": "gaussian_signal",
            "arguments": {
                "location": 3.0,
                "gaussian_signal_sigma": 0.16,
            },
        },
        number_of_dimensions=1,
        n_background_events=10_000,
        target_significance=1.0,
    )

    assert n_signal_events > 0


def test_signal_calibration_entry_point_outputs_json(capsys):
    calibrate_signal_events(
        [
            "--background-generator",
            '{"function":"exponential_background"}',
            "--signal-generator",
            '{"function":"gaussian_signal","arguments":{"location":3.0,"gaussian_signal_sigma":0.16}}',
            "--number-of-dimensions",
            "1",
            "--background-events",
            "10000",
            "--target-significance",
            "1",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result[0]["target_significance"] == 1
    assert result[0]["mean_signal_events"] > 0


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
        upper_limit=6.4 + 6.0 * 0.16,
    )

    relative_difference = abs(finite_significance / unbounded_significance - 1)
    assert relative_difference < 0.001


def test_continuous_injected_significance_matches_1d_for_uniform_2d_pdf():
    n_background_events = 10_000
    n_signal_events = 100
    expected_significance = np.sqrt(
        2
        * (
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


def test_multidimensional_significance_resolves_narrow_signal_peak():
    def signal(coordinates):
        return np.prod(
            norm.pdf(coordinates, loc=4.0, scale=0.004),
            axis=-1,
        )

    def background(coordinates):
        return np.exp(-np.sum(coordinates, axis=-1))

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=background,
        signal_pdf=signal,
        n_background_events=50_000,
        n_signal_events=100,
        upper_limit=np.array([4.60517019, 6.4]),
    )

    assert significance > 1


def test_two_dimensional_significance_uses_bounded_regioned_tensor_rule(monkeypatch):
    def forbidden_cubature(*args, **kwargs):
        raise AssertionError("2D significance should not use adaptive cubature")

    monkeypatch.setattr(profile_likelihood, "cubature", forbidden_cubature)

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda coordinates: 1.0,
        signal_pdf=lambda coordinates: 1.0,
        n_background_events=10_000,
        n_signal_events=100,
        upper_limit=np.ones(2),
    )
    expected = np.sqrt(2 * (10_100 * np.log1p(100 / 10_000) - 100))

    np.testing.assert_allclose(significance, expected, rtol=1e-5)


def test_four_dimensional_significance_vectorizes_large_event_count_pdf_calls():
    batch_shapes = []

    def uniform_pdf(coordinates):
        values = np.asarray(coordinates)
        if values.ndim == 2:
            batch_shapes.append(values.shape)
            return np.ones(values.shape[0])
        return 1.0

    n_background_events = 10_000_000
    n_signal_events = 100
    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=uniform_pdf,
        signal_pdf=uniform_pdf,
        n_background_events=n_background_events,
        n_signal_events=n_signal_events,
        upper_limit=np.ones(4),
    )
    expected = np.sqrt(
        2
        * (
            (n_background_events + n_signal_events)
            * np.log1p(n_signal_events / n_background_events)
            - n_signal_events
        )
    )

    np.testing.assert_allclose(significance, expected, rtol=1e-5)
    assert batch_shapes
    assert all(shape[1] == 4 for shape in batch_shapes)
    assert max(shape[0] for shape in batch_shapes) > 1


def test_four_dimensional_significance_uses_bounded_tensor_rule(monkeypatch):
    def forbidden_cubature(*args, **kwargs):
        raise AssertionError("4D significance should not use adaptive cubature")

    monkeypatch.setattr(profile_likelihood, "cubature", forbidden_cubature)

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda coordinates: 1.0,
        signal_pdf=lambda coordinates: 1.0,
        n_background_events=10_000,
        n_signal_events=100,
        upper_limit=np.ones(4),
    )
    expected = np.sqrt(2 * (10_100 * np.log1p(100 / 10_000) - 100))

    np.testing.assert_allclose(significance, expected, rtol=1e-5)


def test_multidimensional_significance_resolves_localized_integrand():
    signal_batch_sizes = []

    def localized_signal_pdf(coordinates):
        values = np.asarray(coordinates)
        if values.ndim == 2:
            signal_batch_sizes.append(values.shape[0])
        return np.prod(norm.pdf(values, loc=3.0, scale=0.3), axis=-1)

    domain_width = 6.0
    domain_volume = domain_width**4
    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=lambda coordinates: 1 / domain_volume,
        signal_pdf=localized_signal_pdf,
        n_background_events=10_000_000,
        n_signal_events=100_000,
        upper_limit=np.full(4, domain_width),
    )

    np.testing.assert_allclose(significance, 526.19, rtol=0.02)
    assert signal_batch_sizes


def test_multidimensional_significance_supports_scalar_only_pdfs():
    def scalar_only_uniform_pdf(coordinates):
        values = np.asarray(coordinates)
        if values.ndim != 1:
            raise TypeError("one point at a time")
        return 1.0

    significance = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=scalar_only_uniform_pdf,
        signal_pdf=scalar_only_uniform_pdf,
        n_background_events=10_000,
        n_signal_events=100,
        upper_limit=np.ones(2),
    )
    expected = np.sqrt(2 * (10_100 * np.log1p(100 / 10_000) - 100))

    np.testing.assert_allclose(significance, expected)


@pytest.mark.parametrize("invalid_density", [np.nan, np.inf, -np.inf])
def test_continuous_injected_significance_rejects_nonfinite_pdf_density(
    invalid_density,
):
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


def test_continuous_injected_significance_rejects_infinite_multidimensional_limit():
    with pytest.raises(ValueError, match="upper limits must be finite"):
        calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=lambda coordinate: 1.0,
            signal_pdf=lambda coordinate: 1.0,
            n_background_events=10,
            n_signal_events=1,
            upper_limit=np.array([1.0, np.inf]),
        )
