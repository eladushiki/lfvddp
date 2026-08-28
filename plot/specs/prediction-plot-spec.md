# Prediction Plot Specification

## Status

- **Implementation status:** Ongoing - up to date behavior documented below
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
| Bottom left | SR prediction | Exactly the null terms $1+\theta(x)$ and $1-\theta(x)$ plus the signal products $(1+f(x))(1+\theta(x))$ and $(1-f(x))(1-\theta(x))$, evaluated over the SR. | Same four SR functions as 2D surfaces. |
| Bottom right | CR prediction | Exactly the signal nuisance terms $1+\theta(x)$ and $1-\theta(x)$ plus the null nuisance terms $1+\theta(x)$ and $1-\theta(x)$, evaluated over the CR. | Same four CR functions as 2D surfaces. |

### Plot axes

- The distribution panels share synchronized output-axis limits.
  - 1D: the x axes of all 4 subplots should match in name and limit values. Labels should be displayed once for each column. y axis should match values for top 2 and bottom 2 plots separately. It's label should be displayed once in each row.
  - 2D: the x and y axes of all 4 subplots should match in name, observed direction and limit values. z axis should match values for top 2 and bottom 2 plots separately.
- Spacing:
  - 1D: No spacing needed between plots.
  - 2D: Just enough spacing for all labels to show.
- Minimal borders of figure to allow for all labels and titles to fit nicely without being cut or overlap.
- Legends:
  - 1D: top two plots: to the bottom left of each plot. Bottom two plots: to the upper left, with their top edge at 82% of panel height so they do not touch the title.
  - 2D: default / uninterrupting location, also below the title.
- The suptitle is `<configured title>: A prediction process of <runtag>`; its default configured title is `Datasets Along the Process`.
- Every Carpenter figure reserves the same 12% bottom row for the `run hash` stamp, so that row can be cropped for paper display without cutting plot content.

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
- Prediction values are evaluated over a spanning dataset built from detector-bin coordinates.
- The model output is projected onto the selected observable(s) before rendering, only if there are more then 2 observables in the data.
- Null hypothesis terms use dashed lines; signal hypothesis terms use solid lines.
- Every subplot title is positioned inside its own panel at 90% of panel height, avoiding the suptitle and adjacent plots.

## Current Visual Encoding

| Element | Current encoding |
| --- | --- |
| Combined background | Gray |
| Positive signal-shift / A component | Blue (`tab:blue`) |
| Negative signal-shift / B component | Orange (`tab:orange`) |
| Positive signal and nuisance variants | Cornflower blue / lightsky blue |
| Negative signal and nuisance variants | Sandybrown / moccasin |
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
