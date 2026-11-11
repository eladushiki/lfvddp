"""Unbinned goodness-of-fit scores for the plotted chi-square reference."""

import numpy as np
from scipy.stats import chi2, kstest


def chi_square_goodness_of_fit(
    t_values: np.ndarray, degrees_of_freedom: int
) -> tuple[float, float]:
    """Return KS D and one-sample Anderson-Darling A² for a fixed reference.

    The caller supplies the retained histogram sample. No reference parameters
    are fitted, and no p-values are assigned to this potentially selected sample.
    Log-tail probabilities preserve precision when the CDF rounds to one;
    zero-probability tails correctly give an infinite AD statistic.
    """
    reference = chi2(df=degrees_of_freedom)
    ordered = np.sort(t_values)
    sample_size = ordered.size
    weights = 2 * np.arange(1, sample_size + 1) - 1
    ks_score = kstest(ordered, reference.cdf).statistic
    ad_score = (
        -sample_size
        - np.sum(weights * (reference.logcdf(ordered) + reference.logsf(ordered[::-1])))
        / sample_size
    )
    return float(ks_score), float(ad_score)
