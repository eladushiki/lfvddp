# Issue 005: Add adaptive lr to trainings

**Status:** done

## Description

I experience that LR = 3e-2 is good enough for most of the learning progress and gives us convergence upon 500k epochs. Although, in 2+ dims we get sometimes convergence problems.

I want to be able to train with high lr for speedup for most of the time, but towards the end I want to slow down to 1e-3.

Add config optional parameter of a final lr, such that if filles the lr gradually descends from the initial to the final.

Backward compatibility: keep constant lr if only current parameter or no lr parameters are given.

Continuing a run: if epochs not changes - no problem. If --target-epochs specified - treat as if the new target epochs was stated from the beginning and continue with lr from the current point.