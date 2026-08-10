from pathlib import Path

from plot.create_plots import _context_arguments, _plot_options_from_args


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
    assert _context_arguments(
        submission_directory,
        additional_config_paths,
        multi_run_plots,
        debug,
    ) == [
        "--configs",
        str(plot_config),
        "--out-dir",
        str(runs_directory),
        "--debug",
    ]


def test_single_submission_plots_include_staged_configs(tmp_path):
    submission_directory = tmp_path / "submission"
    (submission_directory / "configs").mkdir(parents=True)

    assert _context_arguments(
        submission_directory, [], multi_run_plots=False, debug=False
    ) == [
        "--configs",
        str(submission_directory / "configs"),
        "--out-dir",
        str(submission_directory),
    ]
