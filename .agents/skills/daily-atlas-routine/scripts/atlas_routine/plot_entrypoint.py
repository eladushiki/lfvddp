"""Explicit canonical group membership, never recursive plot-context guessing."""

import json
import sys
from pathlib import Path


def main(args):
    # Context descriptors and plot metadata must identify the plotting entrypoint.
    sys.argv[:] = ["create_plots.py", *args]
    from frame.context.execution_context import ExecutionContext
    from plot import plot_utils, plots
    from plot.create_plots import create_plots

    plan = json.loads(Path(args[0]).read_text())
    selected = [Path(path).resolve() for path in [plan["background"], *plan["members"]]]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate explicit plot membership")

    def discover(directory, *unused, **kwargs):
        root = Path(directory).resolve()
        return [
            (ExecutionContext.load_from_run_dir(path), path / "context.json")
            for path in selected
            if path == root or path.is_relative_to(root)
        ]

    plot_utils.utils__discover_performance_contexts = discover
    plots.utils__discover_performance_contexts = discover
    create_plots(
        multi_run_plots=True,
        submission_directory=Path(plan["directory"]),
        debug="--debug" in args,
        background_directory=Path(plan["background"]),
    )
