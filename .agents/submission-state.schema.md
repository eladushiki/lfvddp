# Cluster submission state

`.agents/submission-state.yaml` is the ignored, local source of truth for the
daily cluster routine. List order is FIFO order. A routine may update existing
entries, but it must never add a new request unless the user explicitly asks.

## Top-level structure

```yaml
version: 3
last_checked_at: null

limits:
  max_queued_elements: 1000
  limit_source: configured
  observed_admin_max_queued_elements: null
  inferred_max_queued_elements: null
  updated_at: null

remote_checkout:
  branch: null
  commit: null
  observed_at: null
  latest_main_checked_at: null

plot_groups: []
submissions: []
```

- `last_checked_at` is the completion time of the most recent successful
  scheduler reconciliation. A failed SSH attempt does not advance it.
- `limits.max_queued_elements` is the enforced limit. It starts at 1000 with no
  reserve. `limit_source` is `configured`, `scheduler_message`, or
  `rejection_inference`.
- Save an explicit numeric scheduler limit in
  `observed_admin_max_queued_elements`. When a quota rejection provides no
  number, set `inferred_max_queued_elements` to
  `queued_before_submission + array_size - 1`. Enforce the smallest known bound
  and timestamp every change.
- `remote_checkout` records what the routine actually observed. Never replace
  or update the checkout while jobs are active. The targeted source-pack
  walltime correction is safe because active jobs use staged config copies. An
  empty queue permits a clean `main` fast-forward, but submission does not wait
  for a Git update.

## Submission entries

```yaml
submissions:
  - id: plot-02-reproduction-signals--nonlocal--significance-01
    status: requested
    config_pack: configs/plot-02-reproduction-signals/nonlocal/significance-01
    output_root: results/highlights/2026-09/plot-02
    purpose: Generate Plot 02 nonlocal significance outputs.
    requested_at: 2026-08-31T09:00:00+03:00
    plot_groups:
      - plot-02-reproduction-signals
```

Required initial fields are `id`, `status`, `config_pack`, `output_root`,
`purpose`, `requested_at`, and `plot_groups`. `plot_groups` may be empty.
Array size is deliberately absent: read `cluster__qsub_n_jobs` from the pack
immediately before the quota check.

Submission statuses and their additional fields are:

- `requested`: explicitly authorized and waiting in FIFO order.
- `blocked`: temporarily unable to submit; requires `blocked_reason` and
  `last_error`. A retry keeps the same list position.
- `submitted`: requires `attempts`, `remote_commit`, and the runtime-discovered
  `remote_submission_directory`.
- `continuation_requested`: a saved attempt was killed specifically for
  walltime and its whole continuation array is waiting for quota. Requires
  `pending_continuation.extra_time`, scheduler evidence, and source-pack update
  status.
- `finished`: every saved array job completed successfully; requires
  `finished_at`. Failed or partial arrays remain blocked with evidence.
- `analyzed`: the single-submission plot completed; requires
  `single_run_plot.completed_at`.

`last_error` may be retained on any non-successful stage for reporting, but it
must be cleared when that same stage later succeeds.

Each initial submission or continuation is saved once in `attempts`:

```yaml
attempts:
  - kind: initial
    job_ids: ["12345[]"]
    submitted_at: 2026-08-31T09:05:00+03:00
    scheduler_outcome: active
  - kind: continuation
    job_ids: ["12399[]"]
    submitted_at: 2026-09-01T09:07:00+03:00
    extra_time: "12:00:00"
    scheduler_outcome: active
    source_config_updated_at: 2026-09-01T09:06:00+03:00
```

Match scheduler history against the job IDs in all attempts. For a verified
walltime kill, choose an evidence-based `extra_time`, defaulting to the killed
attempt's configured total when the scheduler provides no better estimate. The
continuation command persists the increased total in the saved context; update
the original `config_pack` to that same total so later fresh runs use it too.
Walltime remains defined in configuration; only the per-attempt added duration
is retained as audit evidence in state.

## Plot groups

```yaml
plot_groups:
  - id: plot-02-reproduction-signals
    status: pending
    background_submission: plot-01-reproduction-bkg--1e5-events-bkg
    signal_submissions:
      - plot-02-reproduction-signals--nonlocal--significance-01
    remote_multi_run_directory: results/highlights/2026-09/plot-02
```

A group has exactly one `background_submission` and an ordered list of
`signal_submissions`. These IDs, rather than directory-name inference, define
membership. Plot 02 explicitly points to its Plot 01 background.

Group statuses are:

- `pending`: at least one member has not completed single-submission plotting.
- `ready`: every referenced submission is `analyzed`.
- `analyzed`: the multi-run command completed; requires `completed_at`.
- `failed`: the last aggregate attempt failed; requires `last_error` and may be
  retried without changing membership.

The signal tree is `remote_multi_run_directory`. The background submission's
saved timestamped directory is passed separately with
`--background-directory`, so a group cannot accidentally consume an unrelated
background.

## Update guarantees

- Save state immediately after each verified submission or plotting stage.
- Match scheduler output only against job IDs saved in `attempts`; pre-existing
  jobs affect counts but are never adopted.
- Never predict timestamped output directories. Discover and save the directory
  created by the submission command.
- Skip completed stages on retries. State transitions make the daily routine
  idempotent.
