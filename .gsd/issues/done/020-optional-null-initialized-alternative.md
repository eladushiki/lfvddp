# Issue 020: Optional null-initialized alternative

**Status:** done

Independent alternative fits can miss the already feasible null solution and
produce negative final likelihood ratios because optimization failed. Provide
an explicit, optional initialization that embeds the fitted null while retaining
random signal hidden parameters and each hypothesis's original evaluation budget.

Implemented `train__initialize_from_null: true` (default false) for LFVDDP adaptive
neural signal functions. Sequential execution fits or restores the null first,
copies its nuisance tensors and normalization, and zeros only the signal output
head. A single forward evaluation without gradients per alternative fit/continuation
invocation retains the feasible null
outside the training budget and history, with best epoch -1. Learned, binned,
and absent nuisance functions are supported. NPLM and nonadaptive signal spaces
are rejected; explicitly using the parallel launcher is rejected as well.

Partial and completed continuation restore live alternative parameters and
optimizer state. An improved null candidate can replace a completed alternative's
best state without restarting optimization. Checkpoints reject changes to the
initialization flag; historical checkpoints imply the default false.

Verification covers exact likelihood embedding, independently owned tensors and
normalization, zero hidden gradients followed by output and hidden learning,
1D/2D and one-/two-hidden-layer signals, Adam and Adam/L-BFGS, full per-model
budgets, incomplete/completed resume, extended budgets and retained warmup,
updated completed checkpoints, public predictions, and prediction plotting.

No likelihood formulas, dataset generation, raw statistic definition, cluster
submission behavior, or plotting geometry/specifications are changed.
