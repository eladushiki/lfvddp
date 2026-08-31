# LFVDDP

Implementing ML machinery to differentiate between similar and different pairs of physically equivalent datasets.

## Development Principles
- THE ONE DEFINITION RULE: Every variable value should be implemted once. Any logic should be implemented once and reused. Have a single source of truth.
- SINGLE PURPOSE RULE: Each function, class or file should have a single purpose. If there are many, split.
- TESTING PHILOSOPHY:
  - Any complex enough (not semantic) logic in the codebase should be tested at least once. Simple additions should not. Any configuration should be tested at least once with all its options, but not all possible combinations of them - just the interesting ones and the edge cases.
  - Testing logic should never be tested, its validation comes from succeeding to run the project.
  - NO TESTING NEEDED FOR PLOT OR PLOT UTILITIES!
  When adding a test, examine `conftest.py`, `test_train.py:test_learning` and it's use of fixtures. Mimic config file creation and fixture use rather than adding filename variables and/or editing configuration values in runtime.
  For every refactor, rename, or deletion, first inventory all callers and public entry points, then test each affected entry point with every meaningful configuration mode and at least one enabled-feature fixture. A no-feature fixture is not sufficient: for model changes this includes training, checkpoint restore, primary prediction, secondary prediction, nuisance prediction, and plotting integration where applicable. Do not declare the change verified until the inventory has corresponding tests or an explicit reason why a path is not applicable.
- DOCUMENTATION: Change README.md only for indicating any user facing change.
- DIMENSIONAL PLOT PARITY: When changing a 1D graph or its n-dimensional equivalent, locate the corresponding graph function and apply the equivalent change where appropriate, or explicitly verify that no corresponding change is needed.

## Development Workflow
- Do not add unrelated work to an existing feature branch or pull request.
- Use one branch per issue, created directly from the latest `main` branch.
- For independent issues, use separate isolated worktrees and parallel workers when doing so reduces elapsed time without coupling their changes.
- When issues that appear in `.gsd/issues/open` are done, use `git mv` to move to `.gsd/issues/done` directory, in the same pr that solves them, for traceability. Add brief explanation on what was done.
- Open exactly one pull request per issue, from that dedicated branch to `main`.
  - If asked, answer questions inside Github threads.
  - If you solved a concern raised in a thread on Github, comment on that thread what you did to this end. Never mark as resolved yourself.
- Finish every user-requested change by committing it, pushing its dedicated branch, and opening its pull request. The only exception is intentionally local secrets such as `.env`. Leave local worktree in a prunable state on push - important changed commited, temp files deleted, git head clean.
- `.gsd/SECRETS.md` contains local authentication data and must not be backed up
  in the Git repository. Codex has standing authorization to read and use it for
  in-scope GitHub operations without asking again. Never print, commit, or
  otherwise expose its credentials.
- Worktrees of merged branches should be pruned or removed.
- Document insignts on working norms we discussed and design decisions made during the development process.

## Skills and Specs
- Skills can be found in `.agents/skills` directory
- Plot are designed according to specs, that appear in `plots/specs`. If a spec is updated, the plots should too. If a plot is updated, its spec should too.
