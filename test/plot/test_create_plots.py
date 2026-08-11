from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import plot.create_plots as create_plots_module
from plot.create_plots import _config_paths_for_plots, _plot_options_from_args
from plot.plotting_config import PlotScope


def test_multi_run_plots_uses_explicit_config_paths(tmp_path):
    runs_directory = tmp_path / "runs"
    plot_config = tmp_path / "plot.json"
    runs_directory.mkdir()
    plot_config.touch()

    (
        submission_directory,
        additional_config_paths,
        multi_run_plots,
        debug,
    ) = _plot_options_from_args([
        str(runs_directory), str(plot_config), "--multi-run-plots", "--debug"
    ])

    assert submission_directory == runs_directory
    assert additional_config_paths == [plot_config]
    assert multi_run_plots is True
    assert debug is True
    assert _config_paths_for_plots(
        submission_directory,
        additional_config_paths,
        multi_run_plots,
    ) == [plot_config]


def test_single_submission_plots_include_staged_configs(tmp_path):
    submission_directory = tmp_path / "submission"
    (submission_directory / "configs").mkdir(parents=True)

    assert _config_paths_for_plots(
        submission_directory, [], multi_run_plots=False
    ) == [submission_directory / "configs"]


def test_create_plots_builds_context_without_reparsing_plot_options(
    monkeypatch, tmp_path
):
    submission_directory = tmp_path / "runs"
    plot_config = tmp_path / "plot.json"
    submission_directory.mkdir()
    plot_config.touch()
    config = object()
    context = object()
    observed = {}

    def fake_create_config_from_paths(config_paths, out_dir):
        observed["config_paths"] = config_paths
        observed["out_dir"] = out_dir
        return config

    @contextmanager
    def fake_context_manager(*args):
        observed["context_arguments"] = args
        yield context

    def fake_create_configured_plots(
        actual_context, scope, performance_directory=None
    ):
        observed["plot_arguments"] = (
            actual_context,
            scope,
            performance_directory,
        )

    monkeypatch.setattr(
        create_plots_module,
        "create_config_from_paths",
        fake_create_config_from_paths,
    )
    monkeypatch.setattr(
        create_plots_module,
        "version_controlled_execution_context",
        fake_context_manager,
    )
    monkeypatch.setattr(
        create_plots_module,
        "create_configured_plots",
        fake_create_configured_plots,
    )

    create_plots_module.create_plots(
        multi_run_plots=True,
        submission_directory=submission_directory,
        additional_config_paths=[plot_config],
        debug=True,
    )

    assert observed["config_paths"] == [plot_config]
    assert observed["out_dir"] == str(submission_directory)
    assert observed["context_arguments"][:3] == (
        config,
        [plot_config],
        create_plots_module.argv,
    )
    assert observed["context_arguments"][3] == Namespace(
        continue_from=None,
        debug=True,
        build_container=False,
        only_train=False,
    )
    assert observed["plot_arguments"] == (
        context,
        PlotScope.MULTI_RUN,
        submission_directory,
    )
