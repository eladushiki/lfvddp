---
name: daily-atlas-routine
description: Run the shared ATLAS reconciliation, plotting, cleanup and saved submission routine; investigate operational exceptions without inventing work.
---

# ATLAS routine

Use `$ssh-to-cluster` to open exactly one configured SSH shell, then reuse it for
all cluster work. Stop on access failure or interactive authentication; do not
retry or switch hosts. The user chooses the schedule interval.

Read [the canonical state and locking rules](references/submission-state.schema.md)
before accessing state. Run from the configured project root:

```sh
python .agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py
```

The runner acquires the shared lock. If the caller already holds that lock,
append `--held-lock`. Stop immediately on exit 75. Never recreate missing state.
See [the procedure](references/procedure.md) for phase selection, recovery,
authorized history compaction, and deployment.

The scripts implement repeated mechanics: evidence reconciliation, atomic
checkpoints, explicit aggregate membership, verified archives and durable qsub
receipts. They preserve saved configuration order, flags and queue priority.
They contain no pack/version-specific policy. Put one-time scientific or runtime
directives in the conversation and the relevant saved requests, not Python cases.
Use `frozen: true` to exclude explicitly protected entries, regardless of naming.

Read the structured report and relevant stage logs. Use judgment to inspect a
scientific figure, interpret an unusual failure, or carry out a user-authorized
checkout preparation. Stop uncertain submissions for review; lack of scheduler
evidence is not permission to submit again. Do not use an ad-hoc replacement
workflow or silently change queue membership or acceptance rules.

The plotting and submission skills select phases of this same implementation.
Skill-owned tests are separate from the application's public suite. Run them
explicitly from the repository root:

```sh
python -m pytest .agents/skills/daily-atlas-routine/tests .agents/skills/generate-plots-on-cluster/tests --override-ini addopts=''
```
