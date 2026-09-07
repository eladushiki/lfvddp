# GSD State

**Active Milestone:** M001: Training speedup validation
**Active Slice:** S01: Benchmark matrix and reproducibility
**Phase:** evaluating-gates
**Requirements Status:** 2 active · 0 validated · 0 deferred · 0 out of scope

## Milestone Registry
- 🔄 **M001:** Training speedup validation

## Recent Decisions
- D005 (PR #111 neural nuisance refactor): Encapsulate nuisance parameter modes behind a dedicated nuisance-calculation class hierarchy. -> Use a common nuisance calculation interface with scalar-bin and neural per-event implementations; have DifferentiatingModel delegate nuisance preparation, evaluation, and loss contribution to it.
- D006 (Requested loss-clarity refactor): Training-loss ownership -> Centralize the complete negative log-likelihood in DifferentiatingModel._assemble_loss; nuisance implementations return only nuisance values and control-region weights.
- D007 (Issue 011): Nuisance implementation configuration ownership -> Keep scalar nuisance binning under TrainConfig as train__nuisance_binning_* and make it mutually exclusive with neural nuisance hidden-layer configuration.
- D008 (M001 training speedup validation): Which workload matrix qualifies a training speedup -> Every candidate must be tested in nuisance and no-nuisance modes at both 100k and 1m events, with identical mathematical settings and seed controls; single-batch data-loading and batch-size hypotheses are excluded.
- D009 (M001 training speedup validation): How GPU capacity affects benchmark scheduling -> Record requested resources and scheduler concurrency for every run; CPU benchmarks may use the normal concurrency cap, while GPU benchmarks run with a reduced cap derived from available GPU capacity and are never oversubscribed.

## Blockers
- None

## Next Action
Evaluate 2 quality gate(s) for S01 before execution.
