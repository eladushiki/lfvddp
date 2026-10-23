---
name: submit-on-cluster
description: "Submit saved ATLAS requests and continuations in FIFO order using durable intents and PBS-enforced quota."
---

# Submit on cluster

Read [the shared state schema](../daily-atlas-routine/references/submission-state.schema.md) and
[the versioned routine procedure](../daily-atlas-routine/references/procedure.md).
Use the existing shared SSH shell. Never open another connection.

```sh
python .agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py --phase submit
```

Append `--held-lock` when the caller already holds the canonical lock. Use the current skill scripts. The full daily routine uses the runner
without a phase restriction.

Submit only saved requests. Preserve ordered configuration layers, branch
requirements/migrations, pre-submission actions, debug mode and only-train mode.
Continue whole walltime arrays first, then fresh requests in saved order. Let PBS
enforce quota; use the run-local strictly-smaller rejection threshold. Never split
arrays, add requests, adopt unrelated jobs or bypass a blocked submission.

The runner checkpoints a canonical intent before launching training and a durable
receipt around each qsub. Reconcile an interrupted intent before retrying. An
uncertain scheduler outcome blocks further submission; missing history is not
proof that qsub never happened. Each qsub call gets one attempt.

An unsatisfied checkout action is reported without changing priority. Use the
repository GitHub App for preparation, only while the scheduler is inactive.
Keep the user's interval unchanged and report verified IDs, counts, errors and
remaining work from the runner's structured result.
