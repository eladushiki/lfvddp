---
name: submit-on-cluster
description: "Submit explicitly requested ATLAS array jobs in saved priority order while letting PBS enforce its live queue quota."
---

# Submit on Cluster

Run this skill inside the shared lock transaction defined in
[the submission-state schema](../../submission-state.schema.md#shared-access-and-locking).
Read and save `$SUBMISSION_STATE_PATH` there; hold the lock through every
action and its state update. Never use a local YAML copy or nest lock acquisition.

Submit only explicit `requested` entries in the canonical cluster submission state, in
file order. Never submit `retired` entries or invent requests. Read
[the submission-state schema](../../submission-state.schema.md) before changing
state.

This skill assumes `ssh-to-cluster` has already opened one shared shell at the
remote project root. Do not run `ssh`, `scp`, or open a second connection.

## Queue and repository state

- Before selecting a checkout for `requested` or `continuation_requested`
  work, resolve `required_submission_branch` through the ordered
  `branch_migrations` list in submission state. The latest applicable matching
  migration wins. Do not rewrite a submitted attempt's recorded branch: it is
  provenance for the code that actually ran.
- Count queued elements with `qstat -tu $USER | grep Q | wc -l` and running
  elements with `qstat -tu $USER | grep R | wc -l`.
- Existing untracked jobs are not added to state, but their scheduler rows
  are included in scheduler reporting.
- Never pull, checkout, reset, merge, rebase, or replace the checkout while any
  jobs are queued or running. This is not a submission gate: record the current
  branch and commit, then submit more jobs from the same checkout when quota
  permits. The targeted walltime correction defined by
  `generate-plots-on-cluster` is allowed because active jobs use staged config
  copies.
- When queued and running counts are both zero, a clean `main` checkout may be
  fast-forwarded to `origin/main`. Record the observed checkout either way; a
  Git update is not required before submission.

## Ordered configuration layers

Resolve the ordered configuration arguments from the saved request. Legacy
entries use the single `config_pack`. Versioned plot requests use
`config_packs`, whose order is part of the request and must not be sorted or
inferred again at submission time. Always use:

1. `configs/plots-vN/generic`
2. `configs/plots-vN/mandatory-optional/<dimension>/<data-source>`
3. the plot-specific pack

Dimension defaults to `1d`; select `2d` or `4d` only when explicitly indicated
in the plot-pack name. Data source defaults to `generated`; select
`cms_open_data` only when the plot-pack name clearly identifies CMS Open Data.
Before submitting, verify that the saved three paths follow this version,
dimension, and source rule and that merging them in order succeeds.
Later packs override values from earlier packs. Read `cluster__qsub_n_jobs`
from the merged configuration, not from one directory in isolation.

## Priority submission

For the first `requested` entry:

1. If the entry has `required_submission_branch`, the remote checkout must be
   on that exact branch before it can submit. When scheduler jobs are active,
   leave the entry `requested` and stop: do not change the checkout and do not
   let lower-priority work overtake it. With no active jobs, first perform its
   saved `required_pre_submission_action`, then verify the branch and commit.
2. Read `cluster__qsub_n_jobs` from the request's merged ordered configuration
   packs; array size is not copied into state.
3. Recount queued elements immediately before submission for reporting.
4. Submit the whole array and let PBS enforce its current quota. Never split an
   array or reserve capacity locally.
5. Use the entry's `output_root`. Explicit pack values take precedence; seeded
   Plot 01-05 requests derive missing roots as
   `results/highlights/2026-09/plot-XX`.
6. Run the current submission entry point from the observed remote checkout.
   Append `--debug` exactly when the saved entry has `debug: true`; otherwise
   omit it. The normal saved `only_train: true` setting uses `--only-train`:

   ```sh
   python -m train.submit_train --configs <ordered-config-pack>... \
     --only-train [--debug] --out-dir <output-root>
   ```

7. Capture every returned parent job ID. Discover the newly created timestamped
   `*_run_of_submit_train.py_*` directory under `output_root`; do not predict its
   name. Save it as `remote_submission_directory`.
8. Verify every returned parent ID has active array elements in `qstat -tu
   $USER` before updating state. A returned `qsub` ID alone is not evidence
   that training is running. If an array is absent, inspect its PBS output and
   scheduler history; record it as failed or blocked rather than `submitted`.
   Only then update the same entry to `submitted` with an `initial` attempt for
   its first submission or a `rerun` attempt when prior attempts exist. Save its
   job IDs and timestamp, the new timestamped directory, and the observed remote
   commit.
9. Continue until no `requested` entries remain.

If PBS rejects a whole array because of its current queue-state quota, keep the
entry `requested`, record `last_error`, and defer it only for this routine run.
Do not retry the same deferred entry again during that run. Record the rejected
array size as the current run's size threshold. Apply the narrowly authorized
pre-`qsub` cleanup rule in `generate-plots-on-cluster`, then continue scanning
only for later requests whose whole array is strictly smaller than that
threshold. Skip equal-sized and larger requests without invoking PBS: under the
same queue state they cannot fit either. If a smaller request is also rejected,
lower the threshold to that size. Stop scanning when no smaller request remains.
This is a per-run optimization, not an inferred or saved internal quota; the
next routine starts without a threshold and lets PBS evaluate the first eligible
request again.

For other submission or verification failures, keep the entry in place, set it
`blocked` with `blocked_reason` and `last_error`, and stop processing so later
requests cannot overtake it.

## Summary

Report scheduler counts, observed checkout, every submitted or blocked request,
configured or inferred quota changes, job IDs, timestamped output directories,
and remaining priority-ordered work.

## Safety

The daily routine explicitly authorizes submissions already present as
`requested`. No other pack may be submitted without a new explicit user
request.
