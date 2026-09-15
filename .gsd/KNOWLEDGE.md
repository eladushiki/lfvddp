# Project Knowledge

Append-only register of project-specific rules, patterns, and lessons learned.
Agents read this before every unit. Add entries when you discover something worth remembering.
## Rules

| # | Scope | Rule | Why | Added |
|---|-------|------|-----|-------|

## Patterns

| # | Pattern | Where | Notes |
|---|---------|-------|-------|

## Lessons Learned

| # | What Happened | Root Cause | Fix | Scope |
|---|--------------|------------|-----|-------|
| L001 | A Codex cluster SSH attempt could not resolve the configured jump-host name, although SSH from the desktop could. | The command ran inside Codex's restricted network sandbox. | Run the shared SSH helper with the approved elevated network permission, then reuse that one session. | Codex-driven cluster work |
