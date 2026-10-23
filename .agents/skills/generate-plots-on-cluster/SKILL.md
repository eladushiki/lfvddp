---
name: generate-plots-on-cluster
description: "Reconcile tracked ATLAS work and generate verified single-run/group plots and archival cleanup through the committed runner."
---

# Generate plots on cluster

Read [the shared state schema](../daily-atlas-routine/references/submission-state.schema.md) and
[the versioned routine procedure](../daily-atlas-routine/references/procedure.md). The user's instructions
override older skill wording. In particular retain one first worker example for
every array regardless of debug mode.

Use the existing shared SSH shell; never open another connection. Run:

```sh
python .agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py --phase plots
```

Inside a transaction that already holds the canonical lock, append `--held-lock`;
do not nest lock acquisition. Use the current skill scripts; no obsolete-checkout fallback is maintained.

The runner owns acceptance, complete-family membership, progression plots,
pruning, archive verification and immediate atomic state saves. It skips retired,
explicitly frozen entries and protected research output. It preserves failed-element
evidence and retains one worker directory in every archive cleanup.

Stop immediately on lock contention or access failure. Inspect the structured
report for scientific review or unsupported recovery; do not invent a workaround,
change a completion threshold, adopt jobs, or regenerate completed stages.
This phase authorizes no new qsub. Walltime continuations are recorded for the
submission phase. Run the full runner for the daily routine.
