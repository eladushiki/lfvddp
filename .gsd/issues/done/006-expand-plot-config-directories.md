# Issue 006: Expand plotting configuration directories

**Status:** Done

## Problem

The plotting entry point passed a submission's staged `configs` directory to the
configuration file loader without expanding it. The loader then treated the
directory as an extensionless file and rejected it before any plots were created.

## Acceptance criteria

- Plotting expands staged and explicitly supplied configuration directories.
- Supported JSON and YAML files retain deterministic merge order.
- Run roots restrict recursive discovery to their staged `configs` directory.
- The execution context records the expanded configuration files.
- A regression test covers a staged `configs` directory.

## Delivered

Moved configuration path expansion into the shared textual-data utility and
applied it when plotting selects its configuration paths. Run-directory inputs
now search only their staged `configs` directory, excluding sibling run-context
files. Plot contexts receive the expanded JSON/YAML files, with regression
coverage in `test/plot/test_create_plots.py` and
`test/context/test_execution_context.py`.
