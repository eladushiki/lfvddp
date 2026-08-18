# Prediction Plot Specification

## Status

- **Document status:** Draft template
- **Implementation status:** Current behavior documented below
- **Primary implementation:** `plot/plots.py`: `plot_prediction_process_1d` and `plot_prediction_process_2d`
- **Primary utilities:** `plot/plot_utils.py`
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

No further selection or manipulation, other than normalization in the sake of display, should be done in the plotting procedure. The plots should be faithful to their source training procedure and the data it used.

## Figure Layout

A 2 × 2 figure with four panels:

| Position | Panel | 1D contents | 2+D contents |
| --- | --- | --- | --- |
| Top left | SR distribution | Histograms of A-SR, B-SR, their combined background, and weighted numerator-model predictions. | Same 2D SR histograms of the data. If more than 2D, its projection over the first two dimensions. |
| Top right | CR distribution | Histograms of A-CR, B-CR, their combined background, and weighted denominator-model predictions. | Same 2D CR histograms of the data. If more than 2D, its projection over the first two dimensions. |
| Bottom left | SR prediction | Curves for signal-hypothesis and null hypothesis predictions over the SR. | Same SR plots of the 2d functions. |
| Bottom right | CR prediction | Curves for signal-hypothesis and null hypothesis predictions over the CR. In addition, the corresponding detector effect to which they should fit. | Same CR plots of the 2d functions. |

### Plot axes

- The distribution panels share synchronized output-axis limits.
  - 1D: the x axes of all 4 subplots should match in name and limit values. Labels should be displayed once for each column. y axis should match values for top 2 and bottom 2 plots separately. It's label should be displayed once in each row.
  - 2D: the x and y axes of all 4 subplots should match in name, observed direction and limit values. z axis should match values for top 2 and bottom 2 plots separately.
- Spacing:
  - 1D: No spacing needed between plots.
  - 2D: Just enough spacing for all labels to show.
- Minimal borders of figure to allow for all labels and titles to fit nicely without being cut or overlap.
- Legends:
  - 1D: top two plots: to the bottom left of each plots. Bottom two plots: to the top left of each plot.
  - 2D: default / uninterrupting location.
- The figure has a configurable title; its current default is `Datasets Along the Process`.
- "run hash" stamping should be given it's height in bottom border, such that when displaying the image in the paper it could be cropped out.

## Current Distribution Rendering

- Display-bin edges are derived from the unified data for the configured detector observables.
- The selected observable bins use `plot__prediction_process_number_of_bins`.
- The default configuration uses **30 bins**.
- The top panels draw A-region, B-region, and combined-background distributions.
- The model predictions are converted to weighted distributions using the corresponding combined background as the reference dataset.
- When `plot__prediction_process_normalize_each_prediction` is enabled, each of A, B and background should be normalized independently. That being said, all A datasets, predictions and hypothesis plots should be normalized by the same factor such that the top two pdfs would integrate to 1. Same goes for B. The current default is **enabled**.
- Each distribution panel receives a prediction-process legend.
- Normalization shou

## Prediction Rendering

### SR panel: numerator model

The numerator model supplies:

| Quantity | Current label/meaning |
| --- | --- |
| `predict` | \(e^{f(x)}(1+\eta(x))\) signal-hypothesis component |
| `predict_secondary` | \(e^{g(x)}(1-\eta(x))\) signal-hypothesis component |
| `predict_eta` | Numerator nuisance factor \(\eta(x)\) |

The implementation also derives and displays the eta-removed signal terms \(e^{f(x)}\) and \(e^{g(x)}\), together with \(1+\eta(x)\) and \(1-\eta(x)\).

### CR panel: denominator model

The denominator model supplies `predict_eta`, which is displayed as the null-hypothesis nuisance factors \(1+\eta(x)\) and \(1-\eta(x)\).

### Common behavior

- A horizontal reference line is drawn at prediction value **1.0** in 1D prediction panels.
- Prediction values are evaluated over a spanning dataset built from detector-bin coordinates.
- The model output is projected onto the selected observable(s) before rendering, only if there are more then 2 observables in the data.
- Null hypothesis terms use dash-dot lines; signal hypothesis terms use solid lines; detector effect plots use double line.

## Current Visual Encoding

| Element | Current encoding |
| --- | --- |
| Combined background | Gray |
| \(f\)-family / A component | Blue (`tab:blue`) |
| \(g\)-family / B component | Orange (`tab:orange`) |
| Lighter \(f\)-family variants | Cornflower blue / lightsky blue |
| Lighter \(g\)-family variants | Sandybrown / moccasin |
| Reference prediction | Gray dotted horizontal line at 1.0 |

The figure uses the global plotting configuration. The baseline configuration sets a white figure face, classic Matplotlib style, serif font family, font size 24, and figure size 10 × 9 inches.

## Configuration Contract

| Key | Current default | Effect |
| --- | ---: | --- |
| `plot__prediction_process_number_of_bins` | `30` | Number of display bins for the prediction-process distributions. |
| `plot__prediction_process_normalize_each_prediction` | `true` | Normalizes each A/B/background distribution and its corresponding prediction with that component's shared sample-count factor. |
| `plot__figure_size` | `[10, 9]` | Base figure dimensions in inches. |

## Output Contract

- **Return type:** Matplotlib `Figure`
- **Saving:** The plot function does not save the figure itself; the plot factory/calling workflow owns output persistence.
- **Current output name:** <by current defaults>

## Acceptance Criteria

- [ ] The selected-observable dimensionality is validated before plotting.
- [ ] SR and CR distribution panels show the intended data and weighted predictions.
- [ ] Prediction panels show the intended numerator and denominator curves with unambiguous legends.
- [ ] Axis labels, units, and normalization are scientifically correct and consistent as defined above.
- [ ] Colors, line styles, and markers remain distinguishable in grayscale and for color-vision deficiencies. All labels, titles and plotted data are visible fully.
- [ ] The artifact is reproducible from a recorded configuration and training output.
