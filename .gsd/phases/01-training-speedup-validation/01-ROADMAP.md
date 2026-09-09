# M001: Training speedup validation

**Vision:** Identify mathematically equivalent training speedups through controlled cluster experiments, then package only candidates that improve or preserve both nuisance modes at 100k and 1m events.

## Success Criteria

- No candidate is called successful unless it improves or preserves both nuisance and no-nuisance modes at 100k and 1m events.
- All measurements include paired seeds, resource requests, effective scheduler concurrency, PBS state, wall time, and training time.
- Single-batch-irrelevant hypotheses are excluded from implementation scope.
- Every successful mechanism is isolated into its own issue and dedicated PR under the repository workflow.

## Slices

- [ ] **S01: Benchmark matrix and reproducibility** `risk:high` `depends:[]`
  > After this: A committed benchmark plan generates paired baseline and candidate runs for nuisance and no-nuisance modes at 100k and 1m events, with resource-aware scheduler metadata.

- [ ] **S02: Invariant computation reduction** `risk:high` `depends:[S01]`
  > After this: The invariant-computation candidate runs through the full benchmark matrix and is accepted or rejected using paired timing and output checks.

- [ ] **S03: Tensor allocation and copy reduction** `risk:high` `depends:[S01]`
  > After this: The allocation/copy candidate completes the full matrix without regression and reports memory and operator timing evidence.

- [ ] **S04: Checkpoint serialization optimization** `risk:high` `depends:[S01]`
  > After this: A checkpoint mechanism with unchanged checkpoint contents and update order is compared against baseline under the full matrix and accepted only if checkpoint cost falls without training regression.

- [ ] **S05: Candidate acceptance and delivery** `risk:medium` `depends:[S02,S03,S04]`
  > After this: Each accepted mechanism has an evidence-backed issue and a dedicated implementation PR; rejected mechanisms have documented reasons and no code PR.

## Boundary Map

Not provided.
<!-- gsd:state-version=6:0 -->
