# Issue 014: Config Organization

**Status:** Done

# Description
A collection of cleanup and organization items for cleaner config files
1. Messy nuisance definitions: merge nuisance type parameters (`train__data_is_train_for_nuisances`, `train__nuisance_is_neural_network`, etc.) by the following logic:
   1. If train_like_NPLM: nuisances could be just either on or off
   2. If train like lfvddp: nuisances could either be off, binned or a neural network
   3. Any nuisance type comes with a specialized set of parameters. They are all messy in the same file right now, they should be more readable from the code, especially which are needed at any time.
2. Detector should define efficiency function: Efficiency, efficiency uncertainty and error should be moved to detector config currently defined in either independent dataset. Also, should be defined once for a-type and b-type datasets. Hence, should be specified at most twice.
3. Have default values for parameters:
   1. cluster__qsub_io = 0.1
   2. cluster__qsub_mem = 2
   3. cluster__qsub_ncpus = 8
   4. cluster__qsub_ngpus_for_train = 0
   5. plot_config defaults from "configs/basic-generated/plot_config.json", except for `plot__plot_specifications` which should stay explicit
   6. find other communal styling magic values from plots and have them specified in plot_config, with default values as in the majority of plots found.
   7. train__nn_xavier_gain = 1
   8. train__nuisance_binning_minima = 0
   9. train__learning_rate = 0.03

# Anecdotes and Caveats
1. Update backed configuration packages such that all parameters stay the same, but the files themselves with updated structure
2. Use ssh to manipulate existing packs on the project clone on the cluster and update to new structure, omitting parameters that comply new defaults, without changing any value.

# Implementation
- Added canonical nuisance mode selection with compatibility fallback for legacy configuration fields.
- Moved detector effect selection to detector-level A/B effect groups, retaining legacy fallback during migration.
- Added requested cluster, training, plotting, and shared plot-layout defaults.
- Updated the basic generated configuration pack and added configuration organization tests.
