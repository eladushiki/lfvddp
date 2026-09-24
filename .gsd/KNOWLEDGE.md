# Project Knowledge

Append-only register of project-specific rules, patterns, and lessons learned.
Agents read this before every unit. Add entries when you discover something worth remembering.
## Rules

| # | Scope | Rule | Why | Added |
|---|-------|------|-----|-------|

## Patterns

| # | Pattern | Where | Notes |
|---|---------|-------|-------|
| 1 | Materialize regional datasets as A/B pairs | `data_tools/dataset_pair.py` | Keep source materialization separate from the shared regional finalization so pairwise invariants cannot be bypassed by a category-specific branch. |
| 2 | Define A/B and SR/CR topology once | `data_tools/data_utils.py:DATASET_REGIONS` | Use named `.sr`, `.cr`, `.a`, and `.b` fields rather than duplicate category tuples or positional indexing. |

## Lessons Learned

| # | What Happened | Root Cause | Fix | Scope |
|---|--------------|------------|-----|-------|
| L001 | A Codex cluster SSH attempt could not resolve the configured jump-host name, although SSH from the desktop could. | The command ran inside Codex's restricted network sandbox. | Run the shared SSH helper with the approved elevated network permission, then reuse that one session. | Codex-driven cluster work |
