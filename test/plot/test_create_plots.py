from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pytest

import plot.create_plots as create_plots_module
from plot.create_plots import _config_paths_for_plots, _plot_options_from_args
from plot.plotting_config import PlotScope


def test_multi_run_plots_uses_discovered_background_configs(
    monkeypatch, tmp_path
):
    runs_directory = tmp_path / "runs"
    background_directory = runs_directory / "background"
    configs_directory = background_directory / "configs"
    plot_config = configs_directory / "plot.json"
    runs_directory.mkdir()
    configs_directory.mkdir(parents=True)
    plot_config.write_text("{}")
    monkeypatch.setattr(
        create_plots_module,
        "utils__discover_background_only_parent_directory",
        lambda _: background_directory,
    )

    (
        submission_directory,
        multi_run_plots,
        debug,
        explicit_background_directory,
    ) = _plot_options_from_args([
        str(runs_directory), "--multi-run-plots", "--debug"
    ])

    assert submission_directory == runs_directory
    assert multi_run_plots is True
    assert debug is True
    assert explicit_background_directory is None
    assert _config_paths_for_plots(
        submission_directory,
        multi_run_plots,
    ) == [plot_config]


def test_multi_run_plots_accepts_explicit_background_directory(tmp_path):
    runs_directory = tmp_path / "signals"
    background_directory = tmp_path / "background"
    background_config = background_directory / "configs" / "plot.json"
    runs_directory.mkdir()
    background_config.parent.mkdir(parents=True)
    background_config.write_text("{}")

    (
        submission_directory,
        multi_run_plots,
        debug,
        explicit_background_directory,
    ) = _plot_options_from_args([
        str(runs_directory),
        "--multi-run-plots",
        "--background-directory",
        str(background_directory),
    ])

    assert submission_directory == runs_directory
    assert multi_run_plots is True
    assert debug is False
    assert explicit_background_directory == background_directory
    assert _config_paths_for_plots(
        submission_directory,
        multi_run_plots,
        explicit_background_directory,
    ) == [background_config]


def test_plot_cli_rejects_additional_config_path(capsys, tmp_path):
    submission_directory = tmp_path / "submission"
    (submission_directory / "configs").mkdir(parents=True)

    with pytest.raises(SystemExit):
        _plot_options_from_args([
            str(submission_directory),
            str(submission_directory / "configs"),
        ])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_plot_cli_rejects_background_without_multi_run(capsys, tmp_path):
    submission_directory = tmp_path / "submission"
    background_directory = tmp_path / "background"
    (submission_directory / "configs").mkdir(parents=True)
    background_directory.mkdir()

    with pytest.raises(SystemExit):
        _plot_options_from_args([
            str(submission_directory),
            "--background-directory",
            str(background_directory),
        ])
    assert "requires --multi-run-plots" in capsys.readouterr().err


def test_single_submission_plots_include_staged_configs(tmp_path):
    submission_directory = tmp_path / "submission"
    configs_directory = submission_directory / "configs"
    nested_directory = configs_directory / "nested"
    nested_directory.mkdir(parents=True)
    first_config = configs_directory / "0000_cluster_config.json"
    second_config = nested_directory / "0001_plot_config.yaml"
    first_config.write_text("{}")
    second_config.write_text("{}")
    (configs_directory / "notes.txt").write_text("not a config")

    assert _config_paths_for_plots(
        submission_directory, multi_run_plots=False
    ) == [first_config, second_config]


def test_create_plots_builds_context_without_reparsing_plot_options(
    monkeypatch, tmp_path
):
    submission_directory = tmp_path / "runs"
    background_directory = submission_directory / "background"
    configs_directory = background_directory / "configs"
    plot_config = configs_directory / "plot.json"
    submission_directory.mkdir()
    configs_directory.mkdir(parents=True)
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
        actual_context,
        scope,
        performance_directory=None,
        background_directory=None,
    ):
        observed["plot_arguments"] = (
            actual_context,
            scope,
            performance_directory,
            background_directory,
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
    monkeypatch.setattr(
        create_plots_module,
        "utils__discover_background_only_parent_directory",
        lambda _: background_directory,
    )

    create_plots_module.create_plots(
        multi_run_plots=True,
        submission_directory=submission_directory,
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
        background_directory,
    )
