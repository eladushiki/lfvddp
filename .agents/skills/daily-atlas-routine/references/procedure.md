# ATLAS routine

Run the committed command from the one configured SSH shell:

```sh
python .agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py
```

The command checks Torch, acquires the shared canonical lock once, reconciles PBS,
plots eligible tracked results, archives completed work, and submits saved whole
arrays in priority order. It never opens SSH, changes the schedule, invents work,
or retries an unavailable connection. Exit 75 means another transaction owns the
lock: stop immediately. The lock helper bounds the foreground process lifetime
and releases only after its children terminate.

An already held transaction must use `--held-lock`. `--phase plots` excludes new
submissions; `--phase submit` excludes plotting/cleanup. `--dry-run` observes the
scheduler and state without changing either workflow or saved state. Missing
canonical state is an error, including on an idle queue.

The runner loads only `$WIS_CLUSTER_REMOTE_PROJECT_ROOT/.agents/submission-state.yaml`.
It uses the installed safe C YAML implementation when available, validates each
atomic checkpoint, fsyncs it, and refuses an unexpected outside writer. Runtime
logs, receipts and explicit plot plans live under canonical
`.agents/submission-state-tools/`. These are evidence, not alternate queue files.

## Submission recovery

Before launching the submission entrypoint, save `pending_submission_intent` in
canonical state and write its receipt. The committed adapter adds a short unique
PBS name and `LFVDDP_SUBMISSION_INTENT` token. It checkpoints each call immediately
before invoking qsub and stores the returned scheduler ID before returning to
`train.submit_train`. Each qsub invocation is attempted once.

On interruption, reconcile the receipt before any new qsub. A returned ID must
have active scheduler elements or accepted finished output. If qsub started but
its return was lost, match exactly one PBS parent by owner, token and name in
scheduler history. An absent, expired, conflicting or inaccessible match is
uncertain: stop for review, never interpret absence as permission to resubmit.
Receipt-confirmed jobs belong to an already saved request/intent; unrelated jobs
are never adopted. Recover the context's missing qsub record from the verified
receipt when necessary. Verified pre-qsub quota residues alone may be removed.

Before any new group member is submitted, compare the resolved planned packs
with the saved configurations of already submitted members. All `train__*` and
`detector__*` fields must match across the entire explicit group. Recheck saved
submission and available worker contexts before aggregate restoration, cleanup
or plotting. Missing evidence or a mismatch stops that stage with field/value
and path diagnostics; never bypass the guard or silently rewrite hyperparameters.
Injection parameters, seeds, array sizes and plotting controls may differ.

Continuations precede fresh requests. Quota rejections leave the request pending;
for that run only, later arrays must be strictly smaller than the rejected size.
Other submission failures stop the queue. An earlier blocked submission-stage
failure holds later requests. Preserve ordered packs, `debug`, `only_train`,
branch migrations and required checkout preparation.

A saved Git preparation action requires verified GitHub App preparation; the
runner reports an unsatisfied gate and leaves the request pending. It never uses
personal credentials or executes free-form instruction text as shell code. Never
update a checkout with active scheduler jobs. A pending scientific calibration
that requires figure inspection is reported for review, not guessed or repeated.
Scientific requirements remain request metadata and one-time operator instructions;
the runner never infers them from pack names.

## Completion, plots and cleanup

`atlas_routine.completion.accepted_size` implements the schema's shared
strict greater-than-90-percent criterion. Use the original expected array size,
unique worker indices and matching exit-zero PBS logs. Every tracked parent must
be inactive. Failed and missing evidence remains in state. Walltime kills request
whole-array continuation before acceptance. Partial arrays can be plotted and
archived, but their intermediates are not pruned first.

Use explicit saved group members and background paths for aggregate plotting.
Plot-only contexts and untracked/retired runs cannot enter an aggregate through
recursive discovery. Gate the entire group, not a subgroup. Save a plot intent so
an interrupted successful plot can be verified and adopted without regeneration.
Archive only after all referring groups are analyzed; standalone work has no
aggregate dependency. Retain the first lexicographic worker directory for EVERY
array regardless of debug mode, plus contexts, configs, generated plots and the
verified archive. Keep failure logs and protected/frozen artifacts unchanged.

On interruption after archive publication, compare remaining source bytes with
the archive before finishing removal. Do not recompress partially removed sources
or overwrite conflicting evidence. Already analyzed/archived stages are skipped.

## Explicit history reset

Only when the user authorizes it, and scheduler jobs, requests, continuations,
unfinished groups and unresolved intents are absent, and you have checked that
any one-time scientific follow-ups are complete:

```sh
python .agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py --archive-completed-state
```

This creates an immutable gzip snapshot of the original canonical YAML and
verifies its uncompressed SHA256. Only then does it atomically clear submissions
and plot groups. Preserve global policies, branch migrations, completed scientific
calibration and a `state_history` audit reference. Historical snapshots never drive
new work or replace the live canonical file. Results/config trees are untouched.
This flag is not part of ordinary scheduled runs.

## Reports and deployment

The runner emits verified stage events and a final JSON report with counts,
errors, required review, scheduler query count and timings for state I/O, plotting
and cleanup. Empty runs use one active/history snapshot and scan no result trees.
The agent reports meaningful changes or required intervention; an unchanged idle
run needs no new narrative. The user alone chooses the schedule interval.

The skill imports LFVDDP's existing training/config/plot entry points; it is not
an application package and does not alter `frame` import paths. Deployment is
an optional code-only bundle of these committed skill scripts, including the
lock helper. Use the matching current application interfaces; obsolete checkout
compatibility is not maintained. The launcher verifies a deployed manifest.

When the runner reports a saved checkout gate, perform the repository GitHub App
fetch/fast-forward in the same held transaction, rechecking that PBS is idle.
Then invoke `--held-lock --record-prepared-commit <verified-origin-main-SHA>`.
This validates clean `main`, equality with the fetched `origin/main`, and records
preparation for this lock acquisition before proceeding. It is an operator
attestation of the authenticated preparation, not a substitute for doing it.
Existing verification may be reused while jobs remain active on that same commit;
an idle transaction with a saved latest-main action requires fresh preparation.

Create the deployment bundle with
`python .agents/skills/daily-atlas-routine/scripts/package_atlas_routine.py /tmp/atlas-routine.tar.gz` after
committing. It reads Git objects, never uncommitted files or local secrets. Transfer
that code-only archive through the existing SSH shell and unpack it into the
ignored runtime directory. `manifest.json` records the revision and every source
checksum; the launcher verifies it on every invocation and reports its revision.


## Maintenance boundaries

The scripts, schema, procedure, fixtures and tests belong to this skill. Archive
mechanics and their tests belong to `generate-plots-on-cluster`. Public training
and plotting packages have no dependency on the routine. Update the explicit
skill test suite when changing the lock, launch/receipt boundary, completion,
FIFO/quota behavior or cleanup. Plot utilities need no new plot tests.

Operational stage failures are checkpointed at one shared boundary. Submission
failures stop scanning; recover receipts before launching anything again.
Unexpected programming errors escape instead of being converted into continued
workflow work. Investigation and one-time scientific instructions remain with
the agent; the script does not pretend to automate those decisions.


Refactor entry-point inventory: the full launcher, plots/submission phases and
held-lock workers share this package; packaging and the process lock remain
separate commands. The private tests exercise phases, live/idle PBS snapshots,
initial and continuation receipts, quota/uncertain recovery, archive and restore,
source verification, and enabled preparation gates. The SSH helper is exercised
with a fake SSH executable. Scientific model entry points are reused, unchanged;
this refactor introduces no model/configuration modes. Verify plotting's named
arguments by inspection rather than adding plot-utility tests.


Worker adapters update the existing `sys.argv` list before invoking application
entry points. Application modules import that list directly; replacing it leaves
context descriptors bound to the runner name. Submission/plot names must remain
`submit_train.py`/`create_plots.py` so discovery, recovery and cleanup agree.
The private adapter tests exercise real descriptor construction after prior
application imports, including both saved plotting debug modes. Existing tracked
runner-named submission directories remain valid; do not rename active work.

### Transient checkout failures

Submission checks are read-only; they do not run `git update-index` or change
assume-unchanged flags. Fresh and continuation guards confirm a dirty observation
once and require an unchanged HEAD. Persistent changes report their paths.
If the worker still fails specifically at the cleanliness guard, the routine may
retry that worker once only when the receipt contains zero qsub calls and the
same clean branch/commit is verified before and after the failure. It saves the
failed log and clean-check evidence on the intent before retrying. A started,
uncertain, accepted, or rejected qsub call is never retried by this path. Other
failures and a second cleanliness failure stop the queue normally.

Caller inventory: fresh and continuation execution contexts use the throwing
guard; `is_same_as_commit` uses the read-only boolean guard; container workers
retain their validated commit environment behavior. No model, training, prediction
or plotting algorithms/configuration modes change. The Git tests cover real
tracked/staged/untracked changes, ignored caches, transient changes, HEAD changes
and container metadata absence. Private routine tests cover initial/continuation
retry, durable evidence and all no-retry boundaries.
