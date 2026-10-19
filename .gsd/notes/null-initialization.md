# Optional null initialization

The initialization is an opt-in execution dependency, not a second training
loop or an optimizer strategy. `SequentialTrainLauncher` groups records by the
same in-memory `DataBatch` and requires exactly one null and one alternative per
batch. It runs the null first and always materializes both models on continuation;
the ordinary completed-checkpoint shortcut only has scalar/history information.
The existing per-name RNG fork preserves branch seeds despite the changed order.

`train/null_initialization.py` owns compatibility checks and tensor copying.
It copies the fitted null normalization before constructing the alternative's
function spaces, so fixed nuisance geometry is interpreted consistently. It
copies the entire retained-best nuisance state, including buffers, and independently copies
the normalization map. Model/source roles, nuisance specifications, dtype, fitted
learned-null status, and continuation normalization are validated.

Only adaptive signal output weights and bias are zeroed. Hidden weights retain
their seeded random initialization. This exactly embeds the null likelihood;
initial hidden gradients vanish at zero output, while output gradients can move
first and subsequently enable hidden learning. A single no-gradient forward
evaluation per alternative fit/continuation invocation records the feasible
state before any optimizer update. Its model name and actual candidate loss are
logged with an explicit unbudgeted/no-gradient marker. It consumes
no gradient-evaluation budget and creates no history row. Its sentinel best epoch
is -1. Each branch retains its complete configured budget, and raw statistics and
history definitions remain unchanged.

For resumed alternatives, candidate evaluation temporarily changes parameters,
then restores their live checkpoint values. The optimizer is never reset. An
extended null fit may supply a better candidate even if the alternative budget
is already complete. The shared checkpoint-save helper persists its updated
best state while preserving the live parameter/optimizer pairing. Returning or
plotting the model then restores that best state as usual. Source and destination
normalization maps must match on continuation; incompatible maps are rejected.

The initialization flag is recorded only when true, preserving default checkpoint
shape and legacy compatibility. Missing flags mean false. Both model loading and
the completed-checkpoint shortcut share its compatibility validator. Changing
flags during continuation is rejected rather than silently restarting a fit.

Caller inventory: TrainConfig validation; single_train launcher selection and
train_for_t; both concrete launchers; _follow_instructions_for_t; calc_min_LFVNN;
DifferentiatingModel.fit and checkpoint loading/saving; static-null calculation;
primary/secondary/nuisance prediction; prediction-process plotting; profiler
branch counts. Existing default-false paths remain covered by the regression
matrix. No 1D/ND plot geometry or specification change applies.
