# Issue 004: Fix the prediction display of the multiple dimension trainings

**Status**: Open

## Problem

Looking at the 3D graph representing the trained model's prediction over some range, at least for 4d, it seems against my intuition - we don't get an around 1 value for the whole prediction except for where there is a signal.

## TODO

- Read the paper in the branch
- Go over the train and prediction representation via prediction_process_plot in 2+ dimensions
- Verify that what is displayed is the projection of the model predicition to 2d. In case of more than 2d data - each point should use coordinates of the 2 appearing axes and sum over the other dimensions to represent truthfully the distribution and model predictions.