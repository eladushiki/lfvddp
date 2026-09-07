# Issue 018: Add configurable signal-hypothesis function spaces

**Status:** open

## Description

Make the signal-hypothesis function space selectable through configuration,
using the same kind of mode choice that already exists for nuisance fitting.

The adaptive neural network must remain the backward-compatible default. Its
ability to learn arbitrary locations and scales is central to the project's
model-independent discovery goal. The additional fixed-basis modes are not a
replacement for that search; they provide regular statistical baselines,
diagnostic comparisons, and optional function spaces whose null distribution
can be related to an identifiable design-matrix rank.

Support these top-level modes:

1. `adaptive_neural`: the current signal implementation, bitwise equivalent to
   existing behavior when no new option is configured.
2. `nplm`: the existing "train like NPLM" behavior, preserved as a selectable
   special case.
3. `fixed_basis`: a signal shift linear in trainable coefficients over
   predetermined basis functions.

The `fixed_basis` mode must initially support:

- `bin_indicators`: one trainable coefficient per configured detector bin,
  sharing bin setup and lookup machinery with the existing binwise nuisance
  implementation where appropriate.
- `cubic_bspline`: cubic B-spline basis functions with fixed knots.
- `orthogonal_polynomial`: Legendre or Chebyshev functions through a configured
  maximum degree on the normalized observable domain.
- `fixed_sigmoid`: sigmoid features with configured, non-trainable centers and
  widths; only their linear output coefficients are fitted.
- `radial_basis`: Gaussian radial basis functions with configured,
  non-trainable centers and widths.

Basis geometry must not be selected from the A/B labels being tested. It may be
specified directly, derived from physics or simulation, or learned from an
independent reference sample. The adaptive-neural mode remains available when
data-driven feature location and scale are required; its null distribution is
to be calibrated empirically rather than inferred from raw parameter count.

## Implementation instructions

- **RESEARCH:** map the current signal, NPLM, nuisance, configuration,
  checkpoint, prediction, and plotting responsibilities.
- **PLAN:** define a class hierarchy with one signal-function-space interface
  and no duplicated loss or evaluation logic. Treat bin indicators as one
  fixed-basis implementation and reuse equivalent binwise nuisance machinery.
- **CONFIGURE:** use one mode parameter plus one mode-specific dictionary.
  Validate required and forbidden keys for every mode through the normal
  configuration path.
- **IMPLEMENT:** preserve the existing adaptive-neural numerical path as the
  default and add the new implementations incrementally.
- **TEST:** run all relevant unit, integration, and system tests. Add focused
  coverage for basis construction, evaluation, rank, serialization, and
  end-to-end training or prediction.
- **DOCUMENT:** record the statistical distinction between adaptive feature
  search and fixed identifiable bases, including the need for empirical null
  calibration in adaptive mode.
- **UPDATE:** migrate tracked cluster config packs to the new configuration
  format without changing their behavior, and add background-only comparison
  packs for the new modes.

## Degrees-of-freedom requirements

- Do not infer the fixed-basis degrees of freedom from a generic neural raw
  parameter count.
- For a fixed basis, compute or validate the rank of the evaluated design
  matrix after applying the statistic's normalization constraints.
- Explicitly represent the observed-count constraint that removes the constant
  direction. For 13 independent basis functions including the constant, the
  expected regular rank is generically 12.
- Surface the resulting rank in plot metadata and use it for analytic
  chi-square overlays only in regular fixed-basis modes.
- Adaptive-neural and NPLM modes must not claim a Wilks degree count solely from
  the number of trainable weights.

## Acceptance criteria

- The function-space mode is configurable through the existing configuration
  composition and validation path.
- Existing configurations select `adaptive_neural` implicitly and retain their
  intended numerical behavior.
- The current NPLM behavior is available through the explicit `nplm` mode.
- All five fixed-basis families can be constructed and evaluated in supported
  observable dimensions with correct dtype and device behavior.
- Bin-indicator mode uses one trainable coefficient per configured bin and
  shares common setup/evaluation code with equivalent nuisance functionality.
- Cubic B-spline bases form the expected local overlapping partition over the
  configured domain and handle boundary knots correctly.
- Polynomial, fixed-sigmoid, and radial bases honor their configured order,
  centers, and widths without making those geometry values trainable.
- Signal evaluation, loss calculation, training, checkpoint save/restore,
  prediction, and plotting work for every supported mode.
- Invalid modes, missing parameters, incompatible keys, invalid knot sequences,
  nonpositive widths, and rank-deficient fixed-basis configurations fail with
  useful validation errors.
- Tests verify linearity in the fixed-basis coefficients and show that
  subtracting two coefficient vectors produces the same function as
  subtracting their evaluated functions.
- At least one background-only ensemble validates chi-square behavior for a
  regular 13-function basis after removal of the constant direction, and
  compares it with the adaptive-neural null distribution.
- User-facing configuration documentation and examples are updated.

## Scope notes

- Do not reduce the flexibility or change the defaults of the adaptive neural
  discovery model.
- Do not change nuisance function-space modes except where a shared abstraction
  is necessary to satisfy the one-definition rule.
- Choosing among several fixed bases after examining the same A/B labels is
  itself a trials factor. Either predeclare the basis or include that selection
  in the empirical null calibration.
- Add background-only cluster configuration packs for at least bin indicators,
  cubic B-splines, and fixed sigmoids, alongside the unchanged adaptive-neural
  reference pack.
