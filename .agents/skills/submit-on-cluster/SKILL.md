---
name: submit-on-cluster
description: "Submit explicitly requested ATLAS array jobs in FIFO order without exceeding the queued-element quota."
---

# Submit on Cluster

Submit only explicit `requested` entries in `.agents/submission-state.yaml`, in
file order. Never invent requests. Read
[the submission-state schema](../../submission-state.schema.md) before changing
state.

This skill assumes `ssh-to-cluster` has already opened one shared shell at the
remote project root. Do not run `ssh`, `scp`, or open a second connection.

## Queue and repository gates

- Count queued elements with `qstat -tu $USER | grep Q | wc -l` and running
  elements with `qstat -tu $USER | grep R | wc -l`.
- The intended queued-element limit is read from state and starts at exactly
  1000, with no reserved capacity.
- Existing untracked jobs are not added to state, but their scheduler rows
  count toward both the initial empty-queue gate and quota calculations.
- Do not update the remote checkout until both queued and running counts are
  zero. At the start of that empty-queue cycle, require branch `main`, a clean
  worktree, and then fast-forward to `origin/main`. Save the verified commit in
  `remote_main.commit` and mark it ready.
- Do not change the checkout under running jobs. Later submissions in the same
  cycle use the saved verified main commit; refresh again when the scheduler is
  empty.
- If the scheduler reports a changed administrator quota with an explicit
  numeric value, update the intended and observed limits in state. Never infer
  a new limit from a rejection alone.

## FIFO submission

For the first `requested` entry:

1. Read `cluster__qsub_n_jobs` from its configuration pack; array size has one
   definition in the pack and is not copied into state.
2. Recount queued elements immediately before submission.
3. Submit only if `queued + cluster__qsub_n_jobs <= limit`. Do not split an
   array. If it does not fit, leave it requested and stop FIFO processing for
   this run.
4. Use the entry's `output_root`. Explicit pack values take precedence; seeded
   Plot 01-05 requests derive missing roots as
   `results/highlights/2026-09/plot-XX`.
5. Run the current submission entry point from the verified remote checkout:

   ```sh
   python train/submit_train.py --configs <config-pack> \
     --only-train --out-dir <output-root>
   ```

6. Capture every returned parent job ID. Discover the newly created timestamped
   `*_run_of_submit_train.py_*` directory under `output_root`; do not predict its
   name. Save it as `remote_submission_directory`.
7. Verify the job with `qstat -wu $USER`, then update the same entry to
   `submitted` with `job_ids`, `submitted_at`, the timestamped directory, and
   `remote_main.commit`.
8. Continue with the next FIFO entry while its whole array fits.

If submission or verification fails, keep the entry in place, set it `blocked`
with `blocked_reason` and `last_error`, and stop FIFO processing so later
requests cannot overtake it.

## Summary

Report scheduler counts, remote-main readiness, every submitted or blocked
request, job IDs, timestamped output directories, and remaining FIFO work.

## Safety

The daily routine explicitly authorizes submissions already present as
`requested`. No other pack may be submitted without a new explicit user
request.
