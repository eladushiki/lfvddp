# Training speed experiments

All experiments in this directory are run from a commit-stamped branch and must preserve training math bitwise. Each round records its hypothesis, configuration delta, job count, epoch limit, profiler use, and measured results.

## Round 01: checkpoint I/O

- **Source branch:** `experiment/training-speedups-v2`
- **Source commit:** `8107a9e`
- **Hypothesis:** reducing checkpoint frequency improves wall-clock training time by reducing checkpoint serialization and filesystem I/O, without changing model updates.
- **Common settings:** generated basic pack, one job per pack, 2,000 epochs, same model/data/optimizer settings, progress bar disabled.
- **Variants:**
  - `round-01-baseline`: checkpoint every 1,000 epochs; profiler disabled.
  - `round-01-checkpoint-sparse`: checkpoint every 2,000 epochs; profiler disabled.
  - `round-01-profile-baseline`: checkpoint every 1,000 epochs; profiler enabled for runtime attribution.
- **Concurrency:** 3 jobs maximum for this round.
- **Bitwise-equivalence requirement:** compare only runtime/resource behavior; do not accept a speedup if training outputs differ.

### Results

| Variant | Job ID | Wall time | Training time | Checkpoint time/count | Peak memory | Output equivalence | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| baseline | pending | pending | pending | pending | pending | pending | pending |
| checkpoint-sparse | pending | pending | pending | pending | pending | pending | pending |
| profile-baseline | pending | pending | pending | pending | pending | pending | pending |

**Status:** Config packs created and committed. Cluster submission is pending because scheduler commands are not available on the current login host; no performance result has been claimed.

## Round 01 results

Results were collected from source commit  after correcting the experiment packs to use the explicit no-op detector uncertainty modifier. Earlier submissions (, -, , ) were setup/configuration failures and are excluded from performance measurements.

| Variant | PBS job | Exit | PBS wall s | Reported training s (numerator + denominator) | Checkpoint/profile artifacts |
|---|---:|---:|---:|---:|---|
| baseline |  | 0 | 297 | 252.311 + 7.669 | 2 checkpoints (one per model), no profile |
| checkpoint sparse |  | 0 | 293 | 258.601 + 7.641 | 2 checkpoints (one per model), checkpoint interval 2,000 epochs |
| profile baseline |  | 0 | 286 | 250.163 + 8.124 | 2 checkpoints (one per model), 2 profiler reports |

The profiler reports are  and . Training outputs and histories were produced for all three successful runs. No automated bitwise-equivalence or checksum comparison was recorded in these jobs; equivalence remains unverified. The wall-time difference is therefore observational only and is not a validated speedup claim.
