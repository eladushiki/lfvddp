# Issue 007: Add Nuisance Implementation as NN

**Status:** Done

## Feature Overview

An additional implementation of the nuisance parameter function that is a simple NN, over the current piecewise constant function.

## Description
- Add a boolean configuration option in TrainConfig such that when toggled, the inner implementation of the nuisance parameters (now, "eta") would be a simple neural network with:
  - Input and output dimension same as f, g. That is, taken from same configuration parameters and defaults.
  - Middle layer with a configurable but defaulting at N=2 neurons.
  - Architecture and activation function mimics f, g (= fully connected, same activation function and initialization)
- Such that, calling "eta" of a differentiating model would call the replacement implementation.
- This should be transparent from the outside, as in - use of any exported API of the model.

## Additional Requirements
- Existing implementation of nuisance parameters as binwise - constant should be available via configuration.
- Default, for backwards compatibility, should be the old implementation.
- Change naming convention for the nuisance parameters to "theta" from current "eta", in graphs and other products as well as inside-code naming.
- This removes the always-on need for the whole binning process. Try to estimate - if its runtime impact is more that a few tensor arithmetic operations per epoch, toggle that process along with the nuisance type configured.
- Plotting should stay the same, and display truthfully the state of the eta function over the domain at the end of training

## Completion

Implemented the opt-in neural theta nuisance implementation with a two-node default hidden layer. The legacy binwise implementation remains the default, and the exported nuisance API is `predict_theta`. Neural theta training bypasses detector-bin compression..
