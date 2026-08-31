# Issue 017: Revamp the README

**Status:** Done

## Statement

Reorganize the README around the end-user workflow, update setup instructions
to use the supported environment scripts, and remove stale anecdotes and
implementation-level design material.

## Acceptance criteria

- The README gives a clear path from setup through configuration, training,
  cluster submission, continuation, plotting, and outputs.
- Setup uses `scripts/setup_python_environment.sh` and documents later
  activation with `scripts/activate_python_environment.sh`.
- Commands and options agree with the current entry points and tracked config
  packs.
- Obsolete environment anecdotes, internal implementation details, and
  contributor-only design discussion are removed.
- User-facing caveats and troubleshooting guidance remain concise and current.

## Completion note

The README now follows the user journey from prerequisites and scripted setup
through configuration, execution, continuation, plotting, outputs, testing, and
troubleshooting. Stale platform anecdotes and implementation-oriented design
sections were removed, and all documented entry-point options were checked
against the current command-line interfaces.
