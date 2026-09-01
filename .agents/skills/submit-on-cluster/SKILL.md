---
name: submit-on-cluster
description: "Submit explicitly requested ATLAS array jobs in priority order, greedily filling the queued-element quota with whole arrays."
---

# Submit on Cluster

Submit only explicit `requested` entries and verified
`continuation_requested` entries in `.agents/submission-state.yaml`. List order
defines priority within each status class; continuations have priority over new
requests. Never invent requests. Read
[the submission-state schema](../../submission-state.schema.md) before changing
state.

This skill assumes `ssh-to-cluster` has already opened one shared shell at the
remote project root. Do not run `ssh`, `scp`, or open a second connection.

## Queue and repository gates

- Count queued elements with `qstat -tu $USER | grep Q | wc -l` and running
  elements with `qstat -tu $USER | grep R | wc -l`.
- The intended queued-element limit is read from state and starts at exactly
  1500, with no reserved capacity.
- Existing untracked jobs are not added to state, but their scheduler rows
  count toward quota calculations.
- Never pull, checkout, reset, merge, rebase, or replace the checkout while any
  jobs are queued or running. This is not a submission gate: record the current
  branch and commit, then submit more jobs from the same checkout when quota
  permits. The targeted walltime correction defined by
  `generate-plots-on-cluster` is allowed because active jobs use staged config
  copies.
- When queued and running counts are both zero, a clean `main` checkout may be
  fast-forwarded to `origin/main`. Record the observed checkout either way; a
  Git update is not required before submission.
- If the scheduler reports a changed administrator quota with an explicit
  numeric value, save it as the observed and enforced limit. If a whole-array
  submission is rejected for quota without an explicit limit, infer the upper
  bound `queued_before_submission + array_size - 1`, store it separately, and
  lower the enforced limit to the smallest known bound. Repeated quota
  rejections may tighten that bound further.

## Priority-ordered quota filling

Make complete passes over eligible entries, visiting all
`continuation_requested` entries first and then all `requested` entries. Within
each class, visit entries in file order. Do not reorder the saved list when a
lower-priority entry fits before a higher-priority one.

For each candidate:

1. Read `cluster__qsub_n_jobs` from its configuration pack; array size has one
   definition in the pack and is not copied into state.
2. Recount queued elements immediately before submission.
3. Submit only if `queued + cluster__qsub_n_jobs <= limit`. Do not split an
   array. If it does not fit, leave it in place and continue scanning for a
   smaller, lower-priority whole array.
4. Use the entry's `output_root`. Explicit pack values take precedence; seeded
   Plot 01-05 requests derive missing roots as
   `results/highlights/2026-09/plot-XX`.
5. Run the current submission entry point from the observed remote checkout:

   ```sh
   python train/submit_train.py --configs <config-pack> \
     --only-train --out-dir <output-root>
   ```

6. Capture every returned parent job ID. Discover the newly created timestamped
   `*_run_of_submit_train.py_*` directory under `output_root`; do not predict its
   name. Save it as `remote_submission_directory`.
7. Verify the job with `qstat -wu $USER`, then update the same entry to
   `submitted` with an initial `attempt` containing its job IDs and timestamp,
   the timestamped directory, and the observed remote commit.
8. After a successful submission, restart the scan at the highest-priority
   remaining candidate so newly available priority information is respected.

Stop only after a complete pass submits nothing. Unused capacity is acceptable
only when no remaining whole array fits. For example, with a 1500-element limit
and priority-ordered arrays of sizes 1000, 1000, and 100, submit the first 1000,
leave the second 1000 in place, and then submit the 100.

If submission or verification fails, keep the entry in place, set it `blocked`
with `blocked_reason` and `last_error`, and stop processing because the remote
submission state may be uncertain. Capacity-based deferral alone is not a
failure and does not block lower-priority candidates. Apply the inferred-limit
update above before recording a quota rejection.

## Summary

Report scheduler counts, observed checkout, every submitted or blocked request,
explicit or inferred quota changes, job IDs, timestamped output directories,
capacity-deferred requests, and remaining priority-ordered work.

## Safety

The daily routine explicitly authorizes submissions already present as
`requested`. No other pack may be submitted without a new explicit user
request.
