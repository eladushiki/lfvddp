---
name: generate-plots-on-cluster
description: "Detect completed tracked ATLAS submissions and generate their single-run and grouped multi-run plots."
---

# Generate Plots on Cluster

Process only submissions recorded in `.agents/submission-state.yaml`. Ignore
untracked jobs when deciding what to plot, even though all jobs still count
toward scheduler limits.

Read [the submission-state schema](../../submission-state.schema.md) before
changing state. This skill assumes `ssh-to-cluster` has already opened one
shared shell at the remote project root. Do not run `ssh`, `scp`, or open a
second connection.

## Detect finished submissions

- Inspect active array jobs with `qstat -tu $USER` and history with
  `qstat -xu $USER`.
- Count scheduler states with
  `qstat -tu $USER | grep <state-letter> | wc -l`.
- Match only job IDs saved in submission `attempts`. Do not add pre-existing or
  otherwise unknown scheduler jobs to state.
- Mark a submission `finished` only when every saved array job has left active
  states and completed successfully. Record `finished_at`.
- Record failures and their scheduler evidence in `last_error`; do not plot a
  failed or partially completed array. Handle scheduler walltime kills with the
  continuation procedure below; other failures remain blocked.

## Continue walltime-killed submissions

When scheduler history proves that a tracked attempt was killed for exceeding
walltime:

1. Choose an additional walltime from the scheduler evidence. If it supplies no
   better estimate, use the killed attempt's configured total walltime so the
   recovered total doubles.
2. Before submitting, apply the same whole-array quota check as
   `submit-on-cluster`. A continuation has priority over new FIFO requests; if
   it does not fit, set the submission to `continuation_requested` and stop new
   submissions for this run.
3. Continue the saved run without debug mode:

   ```sh
   python train/submit_train.py \
     --continue <remote-submission-directory> \
     --extra-time <HH:MM:SS>
   ```

4. Verify that the saved context's total `cluster__qsub_walltime` increased by
   the added duration. Update the original `config_pack` to that same total so
   future fresh runs inherit the correction. Running jobs use staged configs,
   so this targeted source-pack edit is not a checkout update. Do not copy the
   corrected total into top-level submission state.
5. Append a `continuation` attempt with its job IDs, added time, submission
   timestamp, and source-config update evidence. Return the submission to
   `submitted` and reconcile all attempts on later checks.

## Single-submission plots

For each newly `finished` submission, use its saved timestamped
`remote_submission_directory`:

```sh
python plot/create_plots.py <remote-submission-directory>
```

Verify that the command succeeds and creates the configured single-submission
figures. Then set the submission to `analyzed` and record
`single_run_plot.completed_at`. A rerun must skip submissions already marked
`analyzed` unless the user explicitly requests regeneration.

## Multi-run plots

Use `plot_groups` as the single definition of background and signal membership.
A group is ready only when its background and every signal submission are
`analyzed`.

Use the group's saved `remote_multi_run_directory` as the signal tree and its
explicit `background_submission` timestamped directory as the reference.
Plot 02 is exceptional because that reference belongs to Plot 01. Do not infer
a different background at runtime.

Run:

```sh
python plot/create_plots.py <remote-multi-run-directory> \
  --multi-run-plots \
  --background-directory <background-submission-directory>
```

Verify the configured aggregate plots, set the group status to `analyzed`, and
record `completed_at`. Skip completed groups on later daily runs.

### Submission residue cleanup

Before running a multi-run plot, inspect the timestamped submission directories
under `remote_multi_run_directory`. A directory is a residue candidate when any
of these conditions is verified:

- submission setup created the directory but failed before `qsub`, so it has no
  scheduler job IDs;
- the submission failed; or
- no more than 90% of its expected array jobs succeeded.

An earlier residue can be selected instead of a later completed submission for
the same configuration and make an otherwise ready aggregate plot fail. Prefer
the completed submission only after the residue has been reviewed; do not hide
or silently ignore the residue.

The user has granted standing permission to delete one narrow class of residue:
a newly created pre-`qsub` directory when the submission error clearly says the
whole array could not fit the queued-element quota. Before deleting it, verify
from its saved context that `run_successful` is false and `qsub_submissions` is
empty; delete only that exact timestamped directory and verify that it is gone.
A pre-`qsub` residue is not adopted as a tracked attempt because no scheduler
job was submitted.

For every other residue, report its exact remote path, the evidence for its
classification, and whether a completed replacement exists, then obtain the
user's explicit permission before deletion. Until permission is given, leave
the directory unchanged, keep the group failed, and save the candidate and
reason in `last_error`.

After the underlying failure has been fixed, a failed-job residue becomes
eligible for the same permission-gated cleanup. After an approved deletion,
retry the multi-run command, verify its products, and clear the group's stale
`last_error` only when the retry succeeds.

## Failure handling

Save the failed stage and concise error without advancing its status. Continue
with independent ready submissions or groups, but do not bypass a dependency.
Include every failure in the daily summary.

## Safety

This skill authorizes plotting and the state updates needed to make plotting
idempotent. It does not authorize submission, remote Git updates, or discovery
of unrecorded jobs.
