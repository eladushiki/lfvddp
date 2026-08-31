# Cluster submission state

`.agents/submission-state.yaml` is the ignored, local source of truth for the
daily cluster routine. List order is FIFO order. A routine may update existing
entries, but it must never add a new request unless the user explicitly asks.

## Top-level structure

```yaml
version: 2
last_checked_at: null

limits:
  max_queued_elements: 1000
  observed_admin_max_queued_elements: null

remote_main:
  status: waiting_for_empty_queue
  commit: null
  updated_at: null

plot_groups: []
submissions: []
```

- `last_checked_at` is the completion time of the most recent successful
  scheduler reconciliation. A failed SSH attempt does not advance it.
- `limits.max_queued_elements` is the enforced limit. It starts at 1000 with no
  reserve. Update it only when the scheduler reports a new numeric administrator
  limit, and preserve that observed value separately.
- `remote_main.status` is `waiting_for_empty_queue`, `ready`, or `blocked`.
  `commit` records the latest-main commit verified at the start of the current
  empty-queue cycle.

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
- `submitted`: requires `job_ids`, `submitted_at`, `remote_commit`, and the
  runtime-discovered `remote_submission_directory`.
- `finished`: every saved array job completed successfully; requires
  `finished_at`. Failed or partial arrays remain blocked with evidence.
- `analyzed`: the single-submission plot completed; requires
  `single_run_plot.completed_at`.

`last_error` may be retained on any non-successful stage for reporting, but it
must be cleared when that same stage later succeeds.

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
- Match scheduler output only against saved `job_ids`; pre-existing jobs affect
  counts but are never adopted.
- Never predict timestamped output directories. Discover and save the directory
  created by the submission command.
- Skip completed stages on retries. State transitions make the daily routine
  idempotent.
