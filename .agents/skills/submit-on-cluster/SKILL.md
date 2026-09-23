---
name: submit-on-cluster
description: "Submit explicitly requested ATLAS array jobs in saved priority order while letting PBS enforce its live queue quota."
---

# Submit on Cluster

Submit only explicit `requested` entries in `.agents/submission-state.yaml`, in
file order. Never submit `retired` entries or invent requests. Read
[the submission-state schema](../../submission-state.schema.md) before changing
state.

This skill assumes `ssh-to-cluster` has already opened one shared shell at the
remote project root. Do not run `ssh`, `scp`, or open a second connection.

## Queue and repository state

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
## Priority submission

For the first `requested` entry:

1. If the entry has `required_submission_branch`, the remote checkout must be
   on that exact branch before it can submit. When scheduler jobs are active,
   leave the entry `requested` and stop: do not change the checkout and do not
   let lower-priority work overtake it. With no active jobs, first perform its
   saved `required_pre_submission_action`, then verify the branch and commit.
2. Read `cluster__qsub_n_jobs` from its configuration pack; array size has one
   definition in the pack and is not copied into state.
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
   python -m train.submit_train --configs <config-pack> \
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
Do not retry the same deferred entry again during that run. Apply the narrowly
authorized pre-`qsub` cleanup rule in `generate-plots-on-cluster`, then continue
scanning later saved requests for arrays PBS will accept. Do not infer or save
an internal quota from the rejection.

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
