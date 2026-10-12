# Issue 020: Correct the misleading ensemble-mean significance annotation

**Status:** open

## Problem

The background-only test-statistic histogram annotates

```text
Z(mean t)
```

using `calc_t_significance_by_chi2_percentile`. That helper first takes the
sample mean and then evaluates it as though the mean were one individual draw
from the configured chi-square distribution:

```python
norm.ppf(chi2.cdf(np.mean(t_distribution), df=degrees_of_freedom))
```

This answers the single-draw question "where would one value equal to the
observed mean lie in the reference distribution?" It does not quantify how
strongly an ensemble of many null toys establishes that its mean differs from
the reference expectation.

For independent toys with sample standard deviation `s` and count `n`, the
Monte Carlo standard error of the estimated mean is `s / sqrt(n)`, not `s`.
For example, the analyzed background-only ensembles have `s` near 5.6 and
`n` near 1000, so their mean uncertainty is about 0.18. Displayed values near
`Z(mean t) = 0.7` therefore substantially understate the ensemble-level
disagreement with the nominal chi-square mean and are liable to be interpreted
as a goodness-of-fit or calibration significance when they are neither.

The current helper signature also encourages this category error by accepting
an entire distribution while silently reducing it to its mean.

## Required solution

- Remove `Z(mean t)` from the background-only distribution annotation.
- Keep the descriptive sample mean, standard deviation, retained toy count,
  and excluded/non-finite counts.
- Add the Monte Carlo standard error of the mean, `SEM = s / sqrt(n)`, clearly
  labeled as uncertainty on the estimated ensemble mean.
- If the plot reports a reference-mean discrepancy, label it explicitly and
  compute it using an appropriate sampling distribution:
  - for a predeclared central chi-square reference with fixed `df` and raw
    independent toys, use expected mean `df` and standard error
    `sqrt(2 * df / n)`; or
  - use a bootstrap or other documented empirical procedure when filtering,
    fitting, weights, or dependence invalidate that expression.
- Do not label an ensemble goodness-of-fit diagnostic as discovery
  significance.
- Separate the single-observation conversion from ensemble diagnostics. A
  single-observation helper must accept a scalar `t_value` and compute the
  Gaussian-equivalent quantile from the chi-square survival probability without
  taking a hidden mean.
- Audit callers of `calc_t_significance_by_chi2_percentile` and rename or
  replace the helper so its input and statistical meaning are unambiguous.
- Document whether displayed ensemble diagnostics use raw or filtered toys.

## Acceptance criteria

- Background-only histograms no longer display `Z(mean t)`.
- The annotation distinguishes the width of individual `t` values from the
  uncertainty on their estimated mean.
- The retained sample count and `SEM` are displayed or otherwise included in
  plot metadata.
- Any reference-mean discrepancy is labeled as such and is not presented as a
  single-event or discovery significance.
- The single-observation chi-square-to-Gaussian conversion accepts a scalar and
  is tested against known chi-square survival probabilities.
- Ensemble-mean diagnostics are implemented in a separately named statistical
  helper and tested for at least one analytic chi-square case and one empirical
  or filtered-data case where applicable.
- Existing performance plots that intentionally compare a signal statistic
  with a background distribution retain their intended meaning after the
  helper audit.
- User-facing plot documentation explains the reported quantities.

## Scope notes

- Do not infer a formal chi-square goodness-of-fit p-value from a degree of
  freedom fitted to the same displayed sample without accounting for that fit.
- Do not treat data-dependent outlier removal as irrelevant to the uncertainty
  calculation. Either use raw toys for the analytic diagnostic or calibrate the
  filtered estimator empirically.
- No image-regression test is required for the plot itself; test the underlying
  statistical helpers and verify the plot through the normal project plotting
  workflow.
