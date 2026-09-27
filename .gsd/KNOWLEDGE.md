# Project Knowledge

Append-only register of project-specific rules, patterns, and lessons learned.
Agents read this before every unit. Add entries when you discover something worth remembering.
## Rules

| # | Scope | Rule | Why | Added |
|---|-------|------|-----|-------|
| 1 | All GitHub operations | Interpret “gh app” as the repository-configured GSD Develop GitHub App. For PRs, comments, and other GitHub writes, mint its installation token and use `gh` before considering a Codex plugin, browser login, or personal `gh` authentication. | This preserves the repository’s configured authorization path and makes “gh app” unambiguous. | 2026-09-24 |

## Patterns

| # | Pattern | Where | Notes |
|---|---------|-------|-------|
| 1 | Materialize regional datasets as A/B pairs | `data_tools/dataset_pair.py` | Keep source materialization separate from the shared regional finalization so pairwise invariants cannot be bypassed by a category-specific branch. |
| 2 | Define A/B and SR/CR topology once | `data_tools/data_utils.py:DATASET_REGIONS` | Use named `.sr`, `.cr`, `.a`, and `.b` fields rather than duplicate category tuples or positional indexing. |

## Lessons Learned

| # | What Happened | Root Cause | Fix | Scope |
|---|--------------|------------|-----|-------|
| L001 | A Codex cluster SSH attempt could not resolve the configured jump-host name, although SSH from the desktop could. | The command ran inside Codex's restricted network sandbox. | Run the shared SSH helper with the approved elevated network permission, then reuse that one session. | Codex-driven cluster work |
| L002 | GitHub operations were attempted through an unauthenticated personal `gh` login. | This project uses the GSD Develop GitHub App rather than a personal CLI session. | For every GitHub API operation, mint an installation token with the configured GSD Develop App (App ID `4558280`) and the private-key reference in `.gsd/SECRETS.md`; do not use device login or request another GitHub integration. | All project GitHub work |
| L003 | Aggregate plotting reported no finite t values after array artifacts had been archived. | Aggregate plotting reads the per-run final statistics, which archive cleanup had removed before the dependent aggregate plot ran. | Archive a submission only after every plot group that uses it has completed. If an archive was made early, restore it before plotting; do not misclassify a verified archive as a failed residue. | Cluster plot lifecycle |
| L004 | Single-submission cleanup removed `training_outcomes` before aggregate plotting. | Percentile-progression aggregate plots require the training histories stored there. | Preserve complete `single_train.py` outputs through every dependent aggregate plot, then archive their `training_outcomes` together with the rest of the run; never delete them separately. | Cluster plot lifecycle |
| L005 | Copied v2/v3 array packs inherited a 52-CPU request. | The Plot 04 background template contained an unsuitable historical resource value. | Use `cluster__qsub_ncpus: 4` for every v2/v3 pack and its saved run configuration. All v2/v3 queue entries use `debug: true` and must be submitted with `--debug`; never alter a scheduler allocation, debug mode, or resubmit an already submitted array solely to correct metadata. | v2/v3 cluster packs |
