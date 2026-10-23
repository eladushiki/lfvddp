"""Skill worker names must reach application contexts through the shared argv."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frame.context import execution_context
from frame.context.run_descriptor import parse_run_descriptor
from atlas_routine import plot_entrypoint, submit_entrypoint

FIXTURE = Path(__file__).parent / "configs/atlas_plot_plan.json"


class EntrypointIdentityTest(unittest.TestCase):
    def setUp(self):
        self.argv = sys.argv
        self.original = self.argv[:]
        self.addCleanup(self.restore_arguments)

    def restore_arguments(self):
        self.argv[:] = self.original

    def descriptor(self):
        context = SimpleNamespace(
            time="fixture-stamp",
            config=SimpleNamespace(config__dirsafe_runtag="fixture-tag"),
        )
        return parse_run_descriptor(
            execution_context.ExecutionContext._make_unique_descriptor(context)
        )

    def test_submission_worker_preserves_imported_argv_identity(self):
        observed = []
        with (
            patch.object(submit_entrypoint, "install_receipts"),
            patch.object(
                submit_entrypoint.runpy,
                "run_module",
                side_effect=lambda *args, **kwargs: observed.append(self.descriptor()),
            ),
        ):
            submit_entrypoint.main(["--only-train"])
        self.assertIs(sys.argv, self.argv)
        self.assertIs(execution_context.argv, self.argv)
        self.assertEqual(observed[0].entrypoint, "submit_train.py")
        self.assertEqual(sys.argv[1:], ["--only-train"])

    def test_plot_worker_identity_with_saved_debug_modes(self):
        from plot import plot_utils, plots
        from plot import create_plots

        for debug in (False, True):
            with (
                self.subTest(debug=debug),
                patch.object(plot_utils, "utils__discover_performance_contexts"),
                patch.object(plots, "utils__discover_performance_contexts"),
                patch.object(create_plots, "create_plots") as invoke,
            ):
                names = []
                invoke.side_effect = lambda **kwargs: names.append(
                    self.descriptor().entrypoint
                )
                plot_entrypoint.main([str(FIXTURE), *(["--debug"] if debug else [])])
                self.assertIs(sys.argv, self.argv)
                self.assertEqual(names, ["create_plots.py"])
                self.assertEqual(invoke.call_args.kwargs["debug"], debug)
