# Issue 003: Generalize continuous injected-t significance to N-dimensional PDFs

**Status:** Open

## Problem

`profile_likelihood.py::calc_injected_t_significance_by_sqrt_q0_continuous` fails when its PDFs describe more than one observable. The calculation currently makes one-dimensional assumptions about PDF evaluation, integration, or array shape that do not hold for an N-dimensional process.

## Investigation and implementation instructions

1. Reproduce the failure with a minimal two-dimensional PDF and record the failing shape operation or integration assumption.
2. Trace the function’s inputs and callers to establish the canonical representation of:
   - sampled event coordinates;
   - PDF values or callable PDF interfaces;
   - integration bins, domain limits, and weights.
3. Replace one-dimensional indexing, flattening, and integration logic with operations that preserve a final event/sample axis while accepting any number of observable dimensions.
4. Use one shared dimensionality-normalization helper if callers provide multiple supported PDF shapes. Do not add separate 1-D and 2-D code paths unless their mathematics differs.
5. Preserve numeric behavior for existing one-dimensional inputs, including normalization and zero-density/error handling.
6. Add focused tests using a normalized 1-D PDF and a normalized 2-D PDF. Assert finite significance values and, where analytically appropriate, expected equivalence under separable distributions.
7. Run the profile-likelihood tests and the full test suite. If invalid domains or non-finite PDF values are rejected, assert that the error identifies the offending condition without logging data values unnecessarily.

## Acceptance criteria

- The function accepts PDFs over two or more observables without shape or dimensionality errors.
- Existing one-dimensional behavior remains unchanged within numerical tolerance.
- Tests exercise one- and two-dimensional PDFs and at least one invalid/non-finite input path.
- The implementation has one clear source of truth for dimensional PDF handling.
