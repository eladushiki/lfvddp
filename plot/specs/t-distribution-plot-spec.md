# T Distribution Plot Specification

## Status

- **Document status:** Draft template
- **Implementation status:** Current behavior documented below
- **Primary implementation:** `plot/plots.py`: `t_distribution_plot`
- **Primary utilities:** `plot/plots.py`: `_filter_t_distribution_outliers`
- **Plot scope:** Single submission

## Purpose

Compare the submission's empirical test-statistic distribution with its target chi-square distribution. The plot exposes the observed median test statistic and the corresponding significance estimate, while making non-converged or overfitted runs visible in the legend.

A reader should be able to determine:

- Whether the empirical distribution is compatible with the target chi-square shape.
- The median test statistic and median significance estimate.
- How many runs were omitted because they did not converge or were classified as overfitted.

## Invocation and Inputs

| Input | Current behavior |
| --- | --- |
| Execution context | Must supply a merged `PlottingConfig`, `TrainConfig`, and `DetectorConfig`. |
| Training results | The result aggregator loads recorded test-statistic (`t`) values from the submission. |
| `number_of_bins` | Required instruction controlling empirical histogram bin count. |
| `cut_non_converged` | Optional; default `true`; controls removal of non-finite `t` values. |
| `cut_overfitted` | Optional; default `true`; controls removal of extreme finite `t` values. |

## Data Selection and Filtering

- Values are aggregated from the current submission's training results.
- Non-finite values are classified as non-converged.
- Finite outliers are classified from the tail of the finite distribution using the implementation's robust thresholding rule.
- Each category is removed only when its corresponding `cut_*` option is enabled.
- The legend reports the resulting sample count and omitted-category counts.

## Rendering

The figure contains one axes:

| Element | Rendering |
| --- | --- |
| Empirical test statistics | Normalized histogram with the configured number of bins. |
| Target distribution | Chi-square probability-density curve using the train statistic's degrees of freedom. |
| Median statistic | Marked and labelled on the distribution. |
| Significance | Derived from the median statistic and reported in the plot annotation. |

The plot uses the configured histogram, edge, and chi-square colors, line width, and alpha. It labels the horizontal axis as the test statistic and the vertical axis as probability density, with a legend identifying the empirical and reference distributions.

## Configuration Contract

| Key or instruction | Current default | Effect |
| --- | ---: | --- |
| `number_of_bins` | Required | Histogram resolution. |
| `cut_non_converged` | `true` | Omits non-finite runs from the displayed distribution. |
| `cut_overfitted` | `true` | Omits finite outliers classified as overfitted. |
| `plot__figure_size` | `[10, 9]` | Figure dimensions in inches. |
| `plot__figure_styling.plot.histogram_color` | `plum` | Empirical histogram color. |
| `plot__figure_styling.plot.edge_color` | `darkorchid` | Histogram edge color. |
| `plot__figure_styling.plot.chi2_color` | `grey` | Reference chi-square curve color. |

## Output Contract

- **Return type:** Matplotlib `Figure`.
- **Saving:** The plot factory/calling workflow persists the figure.
- **Configured output name:** The plot instruction's `name`, subject to the factory's normal output naming.

## Acceptance Criteria

- [ ] The input configuration type and required plot instruction are validated.
- [ ] The empirical histogram contains only the selected `t` values.
- [ ] The chi-square reference uses the configured statistic degrees of freedom.
- [ ] Median statistic, significance, and omitted-run information are readable.
- [ ] The figure is reproducible from the recorded submission results and configuration.
