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
| Bottom left | SR prediction | Exactly the null terms $1+\theta(x)$ and $1-\theta(x)$ plus the signal products $(1+f(x))(1+\theta(x))$ and $(1-f(x))(1-\theta(x))$, evaluated over the SR. | The same four SR factors in separate 2 × 2 heatmaps; signal terms above null terms, A to the left of B. |
| Bottom right | CR prediction | Exactly the signal nuisance terms $1+\theta(x)$ and $1-\theta(x)$ plus the null nuisance terms $1+\theta(x)$ and $1-\theta(x)$, evaluated over the CR. | The same four CR factors in separate 2 × 2 heatmaps, with the same ordering as SR. |

### Plot axes

- The distribution panels share synchronized output-axis limits.
  - 1D: the x axes of all 4 subplots should match in name and limit values. Labels should be displayed once for each column. y axis should match values for top 2 and bottom 2 plots separately. It's label should be displayed once in each row.
  - 2+D: all maps share observable names, increasing axis direction, and display limits. All ten density maps share a color normalization; both difference maps share a symmetric scale; all eight model-factor maps share a symmetric diverging scale centered at 1. No 3D perspective or overlaid surfaces are used.
- Spacing:
  - 1D: No spacing needed between plots.
  - 2+D: compact nested grids with constrained layout; small physical padding replaces fixed exterior margins and large inter-block gaps. Observable labels appear on the outside of each block. The difference colorbar sits beneath the background/difference column; the shared density colorbar spans the A/B columns. The canvas is 1.6 times the configured width and 1.4 times the configured height (16 × 12.6 inches by default).
- Minimal borders of figure to allow for all labels and titles to fit nicely without being cut or overlap.
- Legends:
  - 1D: top two plots: to the bottom left of each plot. Bottom two plots: to the upper left, with their top edge at 82% of panel height so they do not touch the title.
  - 2+D: no overlay legends. Every map has a short external content label and each block has an external region label. These content identifiers remain visible in non-debug exports; optional axes and figure titles still follow the debug-title policy.
- In debug mode, the suptitle is shown in sentence case as `<configured title> of <runtag>`; its default configured title is `Datasets along the process`. Generated non-debug figures omit all axes and figure titles.
- The 1D figure retains Carpenter's 12% bottom run-stamp row. The 2+D figure instead uses a compact footer sized to the configured run-stamp font, with the stamp 0.025 inches above the figure edge; constrained layout fills the remaining canvas. No fixed 12% or 16% footer is reserved for heatmaps. Debug titles receive space only when displayed.

## Current Distribution Rendering

- Display-bin edges are derived from finite unified A/B and SR/CR data for the configured detector observables.
- `plot__prediction_process_display_percentiles` defaults to `[0, 100]` (the full observed range). An explicit interval such as `[0, 99.5]` crops only the visualization. Compute each observable's percentiles before constructing its equally spaced display bins and continuous prediction grid.
- `plot__prediction_process_share_observable_range` defaults to `false`. When enabled, use the union of those percentile intervals as the same numerical range for every observable and every SR/CR/A/B map. Enable this only for commensurate observables such as electron and muon transverse momenta.
- These options apply to both 1D and multidimensional views. Normalization continues to use full sample sizes, so cropped distributions are not renormalized to the visible area. The saved training-sample histogram edges used for injected significance retain their full observed range.
- The selected observable bins use `plot__prediction_process_number_of_bins`.
- The default configuration uses **30 bins**.
- The top panels draw A-region, B-region, and combined-background distributions.
- The model predictions are converted to weighted distributions using the corresponding combined background as the reference dataset.
- When `plot__prediction_process_normalize_each_prediction` is enabled, each of A, B and background should be normalized independently. That being said, all A datasets, predictions and hypothesis plots should be normalized by the same factor such that the top two pdfs would integrate to 1. Same goes for B. The current default is **enabled**.
- Each 1D distribution panel receives a prediction-process legend; 2+D content is identified by external map labels.

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
- In 1D null terms use dashed lines and signal terms use solid lines. In 2+D signal and null terms have separate labeled maps, all using blue below unity, white at unity, and red above unity.
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
Both regions use the full finite symmetric range, without significance cuts.

## Current Visual Encoding

| Element | Current encoding |
| --- | --- |
| Combined background | Gray |
| Positive signal-shift / A component | Blue (`tab:blue`) |
| Negative signal-shift / B component | Orange (`tab:orange`) |
| Positive signal and nuisance variants | Cornflower blue / lightsky blue |
| Negative signal and nuisance variants | Sandybrown / moccasin |
| Reference prediction | Gray dotted horizontal line at 1.0 |

For 2+D maps, density uses `viridis` (logarithmic color when configured), and differences/model factors use `RdBu_r`. Empty logarithmic-density bins and unsupported model coordinates are masked. Linear-density zeros remain visible. The table above describes the unchanged 1D encoding.

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
