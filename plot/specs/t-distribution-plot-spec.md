# T Distribution Plot Specification

## Status

- **Implementation status:** Ongoing - up to date behavior documented below
- **Primary implementation:** `plot/plots.py`: `t_distribution_plot`
- **Primary utilities:** `plot/plot_utils.py`: `_filter_t_distribution_outliers`
- **Plot scope:** Single submission

## Purpose

Compare the submission's empirical test-statistic distribution with its theoretical chi-square distribution, while making non-converged or overfitted runs visible in the legend.

A reader should be able to determine:

- Whether the empirical distribution is compatible with the target chi-square shape.
- The median test statistic and median-based significance estimate.
- How many runs were omitted because they did not converge or were classified as overfitted.

## Invocation and Inputs

| Input | Current behavior |
| --- | --- |
| Execution context | Must supply a merged `PlottingConfig`, `TrainConfig`, and `DetectorConfig`. |
| Training results | The result aggregator loads recorded test-statistic (`t`) values and the saved run contexts from a single submission. |
| `number_of_bins` | Required instruction controlling empirical histogram bin count. |
| `cut_non_converged` | Optional; default `true`; controls removal (and indication) of non-convergent (negative and apart from distribution bulk) `t` values. |
| `cut_overfitted` | Optional; default `true`; controls removal (and indication) of extreme finite (large and apart from distribution bulk) `t` values. |

## Data Selection and Filtering

- Values are aggregated from the current submission's training results.
- Non-finite values are classified as non-converged.
- Non-finite and negative values are invalid for the rendered distribution and are always removed.
- Finite outliers are classified against the central 5%-95% non-negative reference population, preventing either extreme from contaminating the opposite tail's threshold.
- Other lower- and upper-tail outliers are removed when their corresponding `cut_*` option is enabled.
- The legend reports the resulting sample count and omitted-category counts.

## Rendering

The figure contains one axes:

| Element | Rendering |
| --- | --- |
| Empirical test statistics | Normalized histogram with the configured number of bins. |
| Target distribution | Always show a chi-square probability-density curve using the shared diagnostic degree count reconstructed by the result aggregator from saved run contexts. |
| Median statistic | Marked and labelled on the distribution. |
| Significance | Derived from the median statistic and reported in the plot annotation. |

Adaptive LFVDDP networks use the raw count of all signal-network weights and
biases at any depth, including no hidden layers; a 1-4-1 network therefore uses 13.
The input width is inferred from the saved detector observables, hidden widths
come from the neural options, and the output width is the scalar constant one.
NPLM uses the raw signal-network parameter count, preserving its historical
reference. Fixed families use their family-owned constrained dimension. Shared
nuisance parameters do not contribute to this reference. A zero-dimensional
space raises an explicit error because it has no chi-square density curve.
Parameter counting alone does not establish Wilks validity.
These diagnostic references and annotations are independent of the empirical
background calibration used by the multi-run significance plot.

The plot uses the configured histogram, edge, and chi-square colors, line width, and alpha. It labels the horizontal axis as the test statistic and the vertical axis as probability density, with a legend identifying the empirical and reference distributions.

The histogram bins end at the largest retained `t` value. The displayed x-axis
extends 5% beyond that value, so every non-disqualified value and its marker
remain inside the plot frame without adding an empty tail to the histogram.

### Further Requirements

- The noramlization of the bins should be set such that given that the $chi^2$ distribution accurately describes their creation, the bin heights would match its plot in any point.
- Every Carpenter figure reserves the same 12% bottom row for hash stamping, so that row can be cropped without hiding plot content.
- This one-panel plot uses Carpenter's standard left, right, top, and bottom borders, shared with the percentile-progression and significance plots.

## Configuration Contract

| Key or instruction | Current default | Effect |
| --- | ---: | --- |
| `number_of_bins` | Required | Histogram resolution. |
| `cut_non_converged` | `true` | Omits runs that are too far off to the lower side from the displayed distribution. |
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
- [x] The chi-square reference uses the configured diagnostic degree count, including adaptive networks and NPLM.
- [x] The curve, legend entry, and median-based significance annotation are always shown for positive degree counts.
- [ ] Median statistic, significance, and omitted-run information are readable.
- [x] The x-axis includes all retained `t` values with 5% right-side headroom.
- [ ] The figure is reproducible from the recorded submission results and configuration.
