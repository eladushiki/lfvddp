"""Unbinned goodness-of-fit scores for the plotted chi-square reference."""

import numpy as np
from scipy.stats import chi2, goodness_of_fit, kstest


def chi_square_goodness_of_fit(
    t_values: np.ndarray, degrees_of_freedom: int
) -> tuple[float, float]:
    """Return built-in KS D and Anderson-Darling A² for a fixed reference.

    The caller supplies the retained histogram sample. All chi-square parameters
    remain fixed. Only statistics are reported for this potentially selected
    sample; the Monte Carlo p-value from goodness_of_fit is discarded.
    """
    reference = chi2(df=degrees_of_freedom)
    ks_score = kstest(t_values, reference.cdf).statistic
    # An infinite AD score can cause invalid subtraction in the discarded
    # Monte Carlo p-value calculation; preserve the score without that warning.
    with np.errstate(invalid="ignore"):
        ad_result = goodness_of_fit(
            chi2,
            t_values,
            known_params={"df": degrees_of_freedom, "loc": 0, "scale": 1},
            statistic="ad",
            # SciPy requires at least one null sample even when only the observed
            # statistic is needed. Keep this unused calibration minimal and isolated
            # from the application's random state.
            n_mc_samples=1,
            rng=np.random.default_rng(0),
        )
    return float(ks_score), float(ad_result.statistic)
