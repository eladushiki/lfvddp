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
