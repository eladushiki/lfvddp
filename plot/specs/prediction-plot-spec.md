# Prediction Plot Specification

## Status

- **Implementation status:** Ongoing - up to date behavior documented below
- **Primary implementation:** `plot/plots.py`: `plot_prediction_process_1d` and `plot_prediction_process_2d`
- **Primary utilities:** `plot/plot_utils.py` (1D and shared evaluation), `plot/prediction_heatmaps.py` (2+D histograms and rendering)
- **Plot scope:** Single train process results

## Purpose

Visualize the learned LFVDDP predictions alongside the observed signal-region (SR) and control-region (CR) distributions. The plot is intended to make the model's signal and null hypotheses, including nuisance terms, comparable across the selected observables.

From looking at the plot, a reader should be able to understand:
- How does the nuisance and signal fits look like?
- How good of a fit did the process provide?
- Did it look like the training converged to a reasonable result?
- How better of a fit do the signal hypothesis propose?
- How well can the signal hypothesis differentiate the two datasets better that the null hypothesis?

## Invocation and Inputs

### Required runtime inputs

| Input | Current behavior |
| --- | --- |
| Execution context | Supplies the merged dataset, detector, training, and plotting configuration. |
| Numerator training | Supplies the primary model, data batch, and numerator detector effect. |
| Denominator training | Supplies the denominator model and detector effect. |
| Observables | Uses the supplied `along_observables`; otherwise uses the configured detector observables. The 1D plot requires one observable; the 2D plot uses the first two supplied observables when more are available. |

### Required dataset categories

| Region | Dataset categories | Derived background |
| --- | --- | --- |
| SR | `A_SR`, `B_SR` | `A_SR + B_SR` |
| CR | `A_CR`, `B_CR` | `A_CR + B_CR` |

Display limits may omit tails explicitly configured for visualization. No event selection, training sample, normalization denominator, or saved significance histogram is changed; the plots remain faithful within their displayed range.

## Figure Layout

A 2 × 2 figure with four logical blocks. In 1D each block is one axes; in 2+D each block contains separate, aligned heatmaps:

| Position | Panel | 1D contents | 2+D contents |
| --- | --- | --- | --- |
| Top left | SR distribution | Histograms of A-SR, B-SR, their combined background, and weighted numerator-model predictions. | Six SR maps in a 2 × 3 grid: top row: pooled background, A data, B data; bottom row: A/B difference in counting-error units, fitted A, fitted B. Fits align beneath their corresponding data and the difference beneath the pooled background. Hidden observables are histogrammed out. |
| Top right | CR distribution | Histograms of A-CR, B-CR, their combined background, and weighted denominator-model predictions. | The same six maps for CR; the fitted densities use the denominator nuisance model. |
| Bottom left | SR prediction | Exactly the null terms $1+\theta(x)$ and $1-\theta(x)$ plus the signal products $(1+f(x))(1+\theta(x))$ and $(1-f(x))(1-\theta(x))$, evaluated over the SR. | Six SR maps in a 2 × 3 grid: signal terms above null terms; each row shows the A-minus-B factor difference, A factor, and B factor. |
| Bottom right | CR prediction | Exactly the signal nuisance terms $1+\theta(x)$ and $1-\theta(x)$ plus the null nuisance terms $1+\theta(x)$ and $1-\theta(x)$, evaluated over the CR. | Six CR maps in a 2 × 3 grid, with the same difference/A/B ordering as SR. |

### Plot axes

- The distribution panels share synchronized output-axis limits.
  - 1D: the x axes of all 4 subplots should match in name and limit values. Labels should be displayed once for each column. y axis should match values for top 2 and bottom 2 plots separately. It's label should be displayed once in each row.
  - 2+D: all maps share observable names, increasing axis direction, and display limits. All ten density maps share a color normalization; all eight model-factor maps use a separate shared linear normalization with limits symmetric around 1; all six difference maps share a symmetric diverging scale centered at 0 across the top and bottom blocks; density and model-factor maps use the same sequential viridis palette with separate numerical limits. No 3D perspective or overlaid surfaces are used.
- Spacing:
  - 1D: No spacing needed between plots.
  - 2+D: compact nested grids with constrained layout; small physical padding replaces fixed exterior margins and large inter-block gaps. Observable labels appear on the outside of each block. In distribution blocks, the difference colorbar sits beneath the background/difference column and the shared density colorbar spans the A/B columns. In model-factor blocks, the factor-difference colorbar sits beneath the left column and the shared model-factor colorbar spans the A/B columns. The canvas is 1.6 times the configured width and 1.4 times the configured height (16 × 12.6 inches by default).
- Minimal borders of figure to allow for all labels and titles to fit nicely without being cut or overlap.
- Legends:
  - 1D: top two plots: to the bottom left of each plot. Bottom two plots: to the upper left, with their top edge at 82% of panel height so they do not touch the title.
  - 2+D: no overlay legends. Every map has a short external content label and each block has an external region label. These content identifiers remain visible in non-debug exports; optional axes and figure titles still follow the debug-title policy.
- In debug mode, the suptitle is shown in sentence case as `<configured title> of <runtag>`; its default configured title is `Datasets along the process`. Generated non-debug figures omit all axes and figure titles.
- The 1D figure retains Carpenter's 12% bottom run-stamp row. The 2+D figure instead uses a compact footer sized to the configured run-stamp font, with the stamp 0.025 inches above the figure edge; constrained layout fills the remaining canvas. No fixed 12% or 16% footer is reserved for heatmaps. Debug titles receive space only when displayed.

## Current Distribution Rendering

### Resonant-signal zoom insets (2D only)

- When exactly two detector observables are configured and a nonzero signal
  injection is Gaussian in both coordinates, all 24 data, fit, difference, and
  model-factor maps include a small upper-right inset. SR and CR maps use the
  same signal window, including CR panels with no injected signal.
- Center the window on the configured injected mean and extend it three marginal
  standard deviations in each direction. For a multivariate Gaussian, take the
  square root of the covariance diagonal; correlations do not change these
  axis-aligned bounds. Respect the displayed observable order and explicit
  loaded-signal observable names. Joint, isotropic, and independently configured
  Gaussian generators are supported. Multiple Gaussian injections share their
  enclosing window; zero-width distributions omit the inset.
- Insets magnify each panel's existing mesh with identical color normalization,
  bins, and masks. Do not resample, refit, or independently normalize the inset.
  Show small coordinate ticks and a `Signal zoom` label; mark the corresponding
  window on the parent map with a thin outline. The inset may extend beyond
  the displayed data range, leaving unsupported areas blank.
- Background-only and nonresonant signals, 1D figures, and projections from
  higher-dimensional detector configurations retain their existing rendering.

- Display-bin edges are derived from finite unified A/B and SR/CR data for the configured detector observables.
- `plot__prediction_process_display_percentiles` defaults to `[0, 100]` (the full observed range). An explicit interval such as `[0, 99.5]` crops only the visualization. Compute each observable's percentiles before constructing its equally spaced display bins and continuous prediction grid.
- Observable ranges remain independent; each observable uses its own percentile interval across A/B and SR/CR.
- This option applies to both 1D and multidimensional views. Normalization continues to use full sample sizes, so cropped distributions are not renormalized to the visible area. The saved training-sample histogram edges used for injected significance retain their full observed range.
- The selected observable bins use `plot__prediction_process_number_of_bins`.
- The default configuration uses **30 bins**.
- The top panels draw A-region, B-region, and combined-background distributions.
- The model predictions are converted to weighted distributions using the corresponding combined background as the reference dataset.
- When `plot__prediction_process_normalize_each_prediction` is enabled, each of A, B and background should be normalized independently. That being said, all A datasets, predictions and hypothesis plots should be normalized by the same factor such that the top two pdfs would integrate to 1. Same goes for B. The current default is **enabled**.
- Each 1D distribution panel receives a prediction-process legend; 2+D content is identified by external map labels.

The logarithmic density scale uses the smallest strictly positive bin value
across all ten observed and fitted-density maps. Fitted bins contain weighted
expected counts, which may be much smaller than one even when the supporting
pooled bin contains a single event; these bins can extend the shared scale far
below the smallest observed count. No minimum-occupancy cut or color-range
percentile trimming is applied. Model-factor color limits are computed
independently so these sparse fitted bins do not set the bottom factor range.

## Prediction Rendering

In the factor labels, $x$ denotes the event's observable vector, $f(x)$ is the
learned signal shift, and $\theta(x)$ is the nuisance shift from the model
identified by the signal/null label. A and B denote the two compared samples.

### SR prediction panel

The bottom-left panel displays exactly four functions over the signal region:

- Null hypothesis: $1+\theta(x)$ and $1-\theta(x)$ from the denominator model.
- Signal hypothesis: $(1+f(x))(1+\theta(x))$ and $(1-f(x))(1-\theta(x))$ from the numerator model.

It does not display nuisance-removed $1+f(x)$ or $1-f(x)$ terms, numerator nuisance-only terms, or detector efficiency.

### CR prediction panel

The bottom-right panel displays exactly four nuisance functions over the control region:

- Signal hypothesis: $1+\theta(x)$ and $1-\theta(x)$ from the numerator model.
- Null hypothesis: $1+\theta(x)$ and $1-\theta(x)$ from the denominator model.

It does not display signal product terms, nuisance-removed terms, or detector efficiency.

### Common behavior

- A horizontal reference line is drawn at prediction value **1.0** in 1D prediction panels.
- Prediction values are evaluated over a spanning dataset. Every selected
  observable uses a uniform 1,000-point grid across its display range by
  default for one- and two-dimensional data, and a 100-point grid for data
  above two dimensions. Unselected projected observables use the configured
  nuisance-grid centers when available and display-bin centers otherwise. The
  Cartesian grid is evaluated in complete projected-coordinate chunks of up to
  100,000 events by default, so peak memory is bounded by a chunk rather than
  by the full multidimensional grid. Complete coordinates preserve the legacy
  floating-point summation order.
- The model output is projected onto the selected observable(s) before rendering when hidden observables exist. In 2+D heatmaps, projected sums are divided by the number of hidden-coordinate grid points, displaying a uniform grid average of each model factor. Nuisance terms are averaged before adding/subtracting unity, so a neutral model remains 1 in every dimension. This is a display normalization, not a density-weighted marginal or a change to model evaluation/training. The 1D projection is unchanged.
- In 1D null terms use dashed lines and signal terms use solid lines. In 2+D signal and null terms have separate labeled maps, using the same sequential viridis palette as distributions and one shared linear factor scale with limits symmetric around unity.
- 1D subplot titles remain at 90% of panel height. 2+D content labels sit above the map, outside the data. Prediction heatmaps retain the existing mask outside the convex hull of observed data and the origin. No smoothing, clipping, or downsampling of model-grid values is applied.

## Multidimensional Difference Map

The additional top-row diagnostic uses the same display bins and independent
Poisson counting variances to show a localized A/B discrepancy against noisy
backgrounds. It is not a fitted residual or calibrated discovery significance.

For each bin, the displayed difference is

$$d = \frac{c_A n_A-c_B n_B}{\sqrt{c_A^2 n_A+c_B^2 n_B}}.$$

Here $d$ is the difference in counting-error units; $n_A$ and $n_B$ are the
observed A and B counts in that bin; $c_A$ and $c_B$ are their display scaling
factors. With independent normalization enabled, $c_A=1/N_A$ and $c_B=1/N_B$,
where $N_A$ and $N_B$ are the total observed sample counts. Otherwise both
scaling factors are 1. For an empty sample the factor is set to 1, avoiding a
division by zero. Bins with no counting variance are masked. The error scale
uses plug-in Poisson variances; it does not propagate fitted-model uncertainty
or the covariance induced by normalizing a sample to its observed total.
Both regions and all four model-factor differences use one shared full finite
symmetric range, without significance cuts. The colorbar labels retain the
distinct meanings of counting-error units and model-factor differences.

### Model-factor differences

Each row of each bottom block includes the pointwise A factor minus the B
factor shown in that row, computed after the existing hidden-coordinate
projection. Signal-hypothesis differences occupy the first row and
null-hypothesis differences the second. The four maps share the top
distribution differences' linear diverging scale with blue for negative, white for zero, and red for positive differences,
using their full finite symmetric range. They retain the model factors' convex
hull mask and continuous prediction grid. These are dimensionless factor
differences, without counting-error normalization or a significance
interpretation; no count variance is assigned to model factors. The 1D view
keeps its overlaid factor curves, so the extra heatmap column applies only to
the multidimensional view.

## Current Visual Encoding

| Element | Current encoding |
| --- | --- |
| Combined background | Gray |
| Positive signal-shift / A component | Blue (`tab:blue`) |
| Negative signal-shift / B component | Orange (`tab:orange`) |
| Positive signal and nuisance variants | Cornflower blue / lightsky blue |
| Negative signal and nuisance variants | Sandybrown / moccasin |
| Reference prediction | Gray dotted horizontal line at 1.0 |

For 2+D maps, densities and model factors use `viridis` with separate numerical limits. Density colors are logarithmic when configured and factor colors remain linear, with limits symmetric around unity. All differences use `RdBu_r` with one shared symmetric zero-centered normalization. Nonpositive values on logarithmic scales and unsupported model coordinates are masked. Linear-density zeros remain visible. The table above describes the unchanged 1D encoding.

The figure uses the global plotting configuration. The baseline configuration sets a white figure face, classic Matplotlib style, serif font family, font size 24, and figure size 10 × 9 inches.

## Configuration Contract

| Key | Current default | Effect |
| --- | ---: | --- |
| `plot__prediction_process_number_of_bins` | `30` | Number of display bins for the prediction-process distributions. |
| `plot__prediction_process_continuous_axis_points` | `1000` | Number of uniformly spaced prediction points along each selected observable. |
| `plot__prediction_process_continuous_axis_points_above_2d` | `100` | Number of uniformly spaced prediction points along each selected observable when the data has more than two observables. |
| `plot__prediction_process_chunk_size` | `100000` | Maximum number of full-dimensional grid events evaluated at once. |
| `plot__prediction_process_normalize_each_prediction` | `true` | Normalizes each A/B/background distribution and its corresponding prediction with that component's shared sample-count factor. |
| `plot__prediction_process_log_scale` | `true` | Displays prediction-process distribution outputs on a logarithmic axis when enabled; applies to the 1D y-axis and the 2+D density color scale; difference and model-factor colors remain linear. |
| `plot__figure_size` | `[10, 9]` | Base figure dimensions in inches. |

## Output Contract

- **Return type:** Matplotlib `Figure`
- **Saving:** The plot function does not save the figure itself; the plot factory/calling workflow owns output persistence.
- **Current output name:** <by current defaults>

## Acceptance Criteria

- [ ] The selected-observable dimensionality is validated before plotting.
- [ ] SR and CR distribution panels show the intended data and weighted predictions.
- [ ] Prediction panels show the intended numerator and denominator terms with unambiguous legends (1D) or separate labeled maps (2+D).
- [ ] Axis labels, units, and normalization are scientifically correct and consistent as defined above.
- [ ] Colors, line styles, and markers remain distinguishable in grayscale and for color-vision deficiencies. All labels, titles and plotted data are visible fully.
- [ ] The artifact is reproducible from a recorded configuration and training output.
