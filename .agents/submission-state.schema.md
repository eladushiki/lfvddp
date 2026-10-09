# Cluster submission state

`$WIS_CLUSTER_REMOTE_PROJECT_ROOT/.agents/submission-state.yaml` on the cluster
is the sole, ignored source of truth for all agents and the daily cluster routine.
The root is configured in local `.gsd/SECRETS.md` and exported by `ssh-to-cluster`.
Local copies and copies in other remote worktrees are never authoritative. List order is the saved priority order. A routine may
update existing entries, but it must never add a new request unless the user
explicitly asks.

## Shared access and locking

Every agent must read this section before using state. In the shared cluster
SSH shell, run a transaction using the committed helper from any checkout:

```sh
python /path/to/repo/.agents/scripts/with_submission_state_lock.py -- bash
# Inside this child shell:
cat "$SUBMISSION_STATE_PATH"
# Perform authorized actions and atomically save the updated state here.
exit
```

When the active cluster checkout predates this helper, its deployed copy is
`$WIS_CLUSTER_REMOTE_PROJECT_ROOT/.agents/submission-state-tools/with_submission_state_lock.py`.
This ignored runtime copy is installed from the committed helper without changing
the active checkout; use it until the checkout can obtain the committed version.

The helper resolves the configured canonical project root, changes to it,
exports `SUBMISSION_STATE_PATH`, and atomically creates the adjacent
`.agents/submission-state.lock/` directory. Directory creation coordinates
agents across cluster hosts on the shared filesystem. Lock contention returns
exit code 75 without running the command; defer work and reread state after a
later successful acquisition. No local lock or copied YAML can substitute.

- Acquire the lock **before reading state for any decision that can change
  state or produce side effects**. Hold it through submission, continuation,
  plotting, cleanup, and the immediate corresponding state save. This avoids
  duplicate actions as well as lost updates. A complete routine may hold one
  lock; individual transactions must reread state after each acquisition.
- Every state edit, including adding user-authorized requests, retirement,
  branch migrations, and scheduler observations, requires this same lock.
  Call downstream skills within the held transaction; do not acquire it again.
- Save via a temporary file in the canonical `.agents` directory, validate the
  YAML, then use `os.replace` to publish it atomically. Never replace state from
  a snapshot read before acquiring the lock. Read-only reporting may read the
  atomically published file without a lock, but cannot act on that snapshot.
- The helper records host, PID, acquisition time, and command in
  `submission-state.lock/owner.json`, and releases the lock when the command
  exits, including a nonzero exit. Keep all child work in the foreground; do
  not detach writers or side effects beyond the command lifetime.
- Transactions have a default two-hour lifetime (not an acquisition wait).
  Override it explicitly with `--timeout-seconds <positive-seconds>` before
  `-- <command>`. On timeout (exit 124), or HUP/INT/TERM, the helper terminates
  the child process group, waits up to ten seconds, then sends KILL if needed.
  It verifies that no live group members remain before releasing the lock;
  uncertain termination retains the lock. Even a normally exiting command has
  remaining group children stopped. This bounds abandoned SSH child shells
  without stealing another owner's lock by age. SIGKILL or host loss still
  requires the owner-liveness inspection below. Reconcile scheduler/output
  evidence after every interrupted transaction before repeating any action.
- An abrupt process/host death can leave a lock. Never steal it based on age.
  Inspect its owner on the recorded host and verify both the holder and its
  children have stopped before manually removing `owner.json` and the empty
  lock directory. If ownership or liveness is uncertain, stop and report it.
  Reconcile scheduler/output evidence before repeating an interrupted action:
  a crash between `qsub` and the save may have already submitted jobs.
- Missing state is an error. Do not recreate an empty queue or fall back to a
  local copy. Migration must acquire this lock, refuse to overwrite existing
  remote state, validate and atomically install the source, verify its checksum,
  then replace the old local file with a pointer to the canonical cluster path.

This is a cooperative protocol: all agents and scripts must use the helper;
filesystem permissions alone cannot enforce it for processes sharing one user.

## Top-level structure

```yaml
version: 5
last_checked_at: null

remote_checkout:
  branch: null
  commit: null
  observed_at: null
  latest_main_checked_at: null

branch_migrations: []

plot_groups: []
submissions: []
```

- `last_checked_at` is the completion time of the most recent successful
  scheduler reconciliation. A failed SSH attempt does not advance it.
- Queue counts are observational. Do not maintain or enforce an internal queue
  limit; submit whole arrays and let PBS enforce its live quota.
- `remote_checkout` records what the routine actually observed. Never replace
  or update the checkout while jobs are active. The targeted source-pack
  walltime correction is safe because active jobs use staged config copies. An
  empty queue permits a clean `main` fast-forward, but submission does not wait
  for a Git update.
- `branch_migrations` is an ordered list of queue-only branch replacements. A
  migration applies only to entries whose `status` is listed in
  `applies_to_statuses`; it does not rewrite submitted or analyzed attempts,
  whose recorded branch remains audit evidence. Before choosing a checkout for
  a pending entry, resolve its `required_submission_branch` through this list,
  using the latest matching migration. Each migration records `from`, `to`,
  `applies_to_statuses`, `enacted_at`, and `reason`.

## Submission entries

```yaml
submissions:
  - id: plot-02-reproduction-signals--nonlocal--significance-01
    status: requested
    config_packs:
      - configs/plots-v4/generic
      - configs/plots-v4/mandatory-optional/1d/generated
      - configs/plots-v4/plot-02-reproduction-signals/nonlocal/significance-01
    output_root: results/highlights/2026-09/plot-02
    purpose: Generate Plot 02 nonlocal significance outputs.
    requested_at: 2026-08-31T09:00:00+03:00
    plot_groups:
      - plot-02-reproduction-signals
```

Required initial fields are `id`, `status`, `output_root`, `purpose`,
`requested_at`, `plot_groups`, and exactly one of `config_pack` or
`config_packs`. `plot_groups` may be empty. `config_pack` is the legacy
single-directory form. `config_packs` is an ordered list and is required for
versioned plot requests from `plots-v4` onward.

For every versioned request, `config_packs` contains exactly three layers: the
version's `generic` pack, one
`mandatory-optional/<dimension>/<data-source>` choice pack, and the
plot-specific pack. Dimension defaults to `1d`; select `2d` or `4d` only when
explicitly indicated in the plot-pack name. Data source defaults to `generated`;
select `cms_open_data` only when the plot-pack name clearly states CMS Open
Data. Preserve list order because later layers override earlier values. Read
array size and all other effective values from the merged list rather than
duplicating them in state.
Array size is deliberately absent: read `cluster__qsub_n_jobs` from the merged
configuration immediately before submission.

Optional request controls are recorded with the request, rather than inferred
from a directory name:

- `required_submission_branch`: the exact remote branch required to run the
  pack. A mismatched active checkout leaves this priority entry waiting; it
  must not be bypassed.
- `required_pre_submission_action`: the Git preparation required before that
  branch is used, such as fetching and fast-forwarding it. It may run only
  after all scheduler jobs are inactive.
- `debug: true`: requires `train.submit_train --debug` for every attempt of
  this request.
- `only_train: true`: requires `--only-train`; this keeps a scheduled training
  request from unexpectedly generating plots during submission.

These fields are execution requirements, not audit-only annotations. The
submission procedure must verify them before `qsub`.

Submission statuses and their additional fields are:

- `requested`: explicitly authorized and waiting in FIFO order. A previously
  attempted entry may return to this status with `retry_requested_at` and
  `retry_reason`; it retains its attempt history and list position.
- `blocked`: temporarily unable to submit; requires `blocked_reason` and
  `last_error`. A retry keeps the same list position.
- `submitted`: requires `attempts`, `remote_commit`, and the runtime-discovered
  `remote_submission_directory`.
- `continuation_requested`: a saved attempt was killed specifically for
  walltime and its whole continuation array is waiting for submission. Requires
  `pending_continuation.extra_time`, scheduler evidence, and source-pack update
  status.
- `finished`: every saved array job left active states and either all elements
  succeeded or more than 90% of expected elements have verified exit status 0.
  Requires `finished_at`; accepted partial results also record
  `accepted_successful_elements`, `expected_elements`, and `completion_basis`.
  Results at or below 90% remain blocked with evidence.
- `analyzed`: the single-submission plot completed; requires
  `single_run_plot.completed_at`.
- `retired`: preserved audit history that is no longer eligible for submission,
  reconciliation, or plotting. Requires `original_id`, `retired_at`, and
  `retired_reason`; `plot_groups` must be empty. If it has saved results, its
  `remote_submission_directory` must be outside every active group's
  `remote_multi_run_directory`.

`last_error` may be retained on any non-successful stage for reporting, but it
must be cleared when that same stage later succeeds.

After a verified single-submission percentile-progression plot exists, a
submission may record `intermediate_prune` even while an aggregate group is
pending. It records `completed_at`, the removed HDF5, `training_outcomes`,
runtime-resource, and PBS-log counts, plus the successful worker-context and
PBS-exit-status evidence. This removes only regenerable intermediates; it
preserves contexts, configs, final statistics, and generated plots. After every
referencing plot group is analyzed, a submission may record `artifact_archive` with
`completed_at`, the archive path, verified dependent groups, and the scheduler,
`run_successful`, and PBS exit-status evidence that every tracked array job
succeeded and no `--extra-time` continuation is required. The archive contains
the remaining `single_train.py` outputs; histories already recorded in
`intermediate_prune` need not be recreated. A failed check must be retained in
`last_error` and reported rather than archived.

A `retired` submission may record `retired_artifact_archive` when it has no
live scheduler elements and no non-retired plot group depends on it. This is a
terminal audit archive for canceled, failed, or partial work and does not imply
successful completion. It records `completed_at`, `archive_path`,
`scheduler_inactive_evidence`, `non_retired_groups`, and the observed failure
or partial-completion evidence. Generate any plots that remain usable before
creating this archive.

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
  - kind: rerun
    job_ids: ["12420[]"]
    submitted_at: 2026-09-02T09:10:00+03:00
    scheduler_outcome: active
```

A fresh retry appends a `rerun` attempt and updates the top-level
`remote_submission_directory` to the newly discovered directory. Earlier
attempts and directories remain audit history.

Match scheduler history against the job IDs in all attempts. For a verified
walltime kill, choose an evidence-based `extra_time`, defaulting to the killed
attempt's configured total when the scheduler provides no better estimate. The
continuation command persists the increased total in the saved context; update
the last ordered `config_packs` layer that defines the walltime (or the legacy
`config_pack`) to that same total so later fresh runs use it too.
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

Active significance series contain exactly five points, numbered `01` through
`05`. The legacy ten-point migration retains old points `02, 04, 06, 08, 10`
and renames them `01, 02, 03, 04, 05`; old odd points remain as `retired`
audit entries and must not appear in `signal_submissions`.

Group statuses are:

- `pending`: at least one member has not completed single-submission plotting.
- `ready`: every referenced submission is `analyzed`.
- `analyzed`: the multi-run command completed; requires `completed_at`.
- `failed`: the last aggregate attempt failed; requires `last_error` and may be
  retried without changing membership.
- `retired`: an incomplete legacy group was superseded. Requires `retired_at`
  and `retired_reason`; it is not eligible for plotting or archival decisions.

An `analyzed` group may record `output_directory` for its generated products.
When an archive was restored before a previously blocked aggregate retry, it
may record `archive_restore` with `completed_at`, `restored_archives`, and
evidence. A group may also record `stale_residue_cleanup` with `completed_at`,
the number of deleted directories, and the scheduler/context evidence that
they were failed or superseded. These audit records do not alter membership or
permit unreviewed deletion.

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
- Skip `retired` submissions entirely.
