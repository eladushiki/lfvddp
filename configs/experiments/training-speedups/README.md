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

Results were collected from source commit f28eace after correcting the experiment packs to use the explicit no-op detector uncertainty modifier. Earlier submissions (4931611, 4931638-4931640, 4931649, 4931655) were setup/configuration failures and are excluded from performance measurements.

| Variant | PBS job | Exit | PBS wall s | Reported training s (numerator + denominator) | Checkpoint/profile artifacts |
|---|---:|---:|---:|---:|---|
| baseline | 4931801 | 0 | 297 | 252.311 + 7.669 | 2 checkpoints (one per model), no profile |
| checkpoint sparse | 4931808 | 0 | 293 | 258.601 + 7.641 | 2 checkpoints (one per model), checkpoint interval 2,000 epochs |
| profile baseline | 4931809 | 0 | 286 | 250.163 + 8.124 | 2 checkpoints (one per model), 2 profiler reports |

The profiler reports are A_numerator.2D.profile.txt and A_denominator.2D.profile.txt. Training outputs and histories were produced for all three successful runs. No automated bitwise-equivalence or checksum comparison was recorded in these jobs; equivalence remains unverified. The wall-time difference is therefore observational only and is not a validated speedup claim.


### Repeat results and aggregate view

The two repeat jobs also completed successfully from the isolated experiment worktree:

| Variant | PBS job | Exit | PBS wall s | Reported training s (numerator + denominator) | Artifacts |
|---|---:|---:|---:|---:|---|
| baseline repeat | 4932270 | 0 | 329 | 292.451 + 5.285 | 2 checkpoints, 2 weights, 1 history |
| checkpoint sparse repeat | 4932271 | 0 | 318 | 264.015 + 6.889 | 2 checkpoints, 2 weights, 1 history |

Across the two baseline and two sparse runs, mean PBS wall time was 313.0 seconds for baseline versus 305.5 seconds for sparse checkpointing, a 2.4 percent reduction. Mean total reported training time was 278.86 seconds versus 268.57 seconds, a 3.7 percent reduction. The samples are not seed-paired and vary substantially, so this is a cautious observational result rather than proof of a speedup. No bitwise-equivalence comparison was executed; model-output equivalence remains unverified.

## Reproducible cluster environment

The experiment environment is the repository-locked environment, not the login host's default Python.

1. Use Python 3.11 or newer with the repository's CVMFS view mounted.
2. From the isolated worktree, run COMPILER=native CXX= bash scripts/setup_python_environment.sh. The explicit variables are required on this cluster because the CVMFS setup script is sourced under set -u; native avoids selecting an unavailable compiler toolchain and the empty CXX lets the view initialize compiler variables.
3. The script creates .venv with --system-site-packages, installs uv, and runs the locked uv sync from uv.lock. Do not install a separate requirements file or use the main checkout's environment.
4. Verify with .venv/bin/python -c import torch and python -m compileall -q train neural_networks data_tools frame.
5. For Singularity jobs, /app/.venv must exist in the bound worktree and the verified environment must be explicitly bound there. The experiment packs set SINGULARITY_BINDPATH to the verified environment and retain the worktree bind for source code.

The cluster run path was validated with torch 2.11.0+cu130. Environment setup failures and configuration failures are recorded separately from valid performance measurements.
