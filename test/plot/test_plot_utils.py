from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from data_tools.data_utils import DataSet
from plot.plot_utils import (
    _integration_upper_limits_for_dimensions,
    utils__discover_background_only_parent_directory,
    utils__finalize_prediction_process_layout,
    utils__prediction_mesh_mask,
    utils__project_prediction_values_sliced,
)


def test_prediction_process_layout_shares_row_ranges_and_compacts_1d_labels():
    figure, axes = plt.subplots(2, 2)
    sr_distribution_ax, cr_distribution_ax = axes[0]
    sr_prediction_ax, cr_prediction_ax = axes[1]
    for axis, x_limits, y_limits in (
        (sr_distribution_ax, (0, 2), (1, 4)),
        (cr_distribution_ax, (-1, 3), (2, 6)),
        (sr_prediction_ax, (0, 4), (-2, 2)),
        (cr_prediction_ax, (-3, 2), (-4, 3)),
    ):
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_xlabel("observable")
        axis.set_ylabel("output")

    utils__finalize_prediction_process_layout(
        distribution_axes=[sr_distribution_ax, cr_distribution_ax],
        prediction_axes=[sr_prediction_ax, cr_prediction_ax],
        number_of_dimensions=1,
    )

    assert [axis.get_xlim() for axis in figure.axes] == [(-3, 4)] * 4
    assert sr_distribution_ax.get_ylim() == cr_distribution_ax.get_ylim()
    assert sr_prediction_ax.get_ylim() == cr_prediction_ax.get_ylim()
    assert sr_distribution_ax.get_xlabel() == ""
    assert cr_distribution_ax.get_xlabel() == ""
    assert cr_distribution_ax.get_ylabel() == ""
    assert cr_prediction_ax.get_ylabel() == ""
    plt.close(figure)


@pytest.mark.parametrize(
    ("number_of_dimensions", "expected_upper_limits"),
    [
        (1, np.inf),
        (4, np.full(4, np.inf)),
    ],
)
def test_integration_upper_limits_cover_generated_pdf_domain(
    number_of_dimensions,
    expected_upper_limits,
):
    upper_limits = _integration_upper_limits_for_dimensions(
        number_of_dimensions,
    )

    if number_of_dimensions == 1:
        assert upper_limits == expected_upper_limits
    else:
        np.testing.assert_array_equal(upper_limits, expected_upper_limits)


def test_prediction_mesh_mask_limits_points_to_origin_data_hull():
    mesh_points = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [-0.1, 0.1],
    ])
    data_points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    assert utils__prediction_mesh_mask(mesh_points, data_points).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_project_prediction_values_sliced_sums_unshown_dimensions():
    spanning_dataset = DataSet(
        data=np.array([
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 20.0],
            [1.0, 0.0, 10.0],
            [1.0, 0.0, 20.0],
        ]),
        observable_names=["x", "y", "unshown"],
    )

    coordinates, projected_values = utils__project_prediction_values_sliced(
        values=np.array([1.0, 2.0, 3.0, 4.0]),
        spanning_dataset=spanning_dataset,
        along_observables=["x", "y"],
    )

    np.testing.assert_array_equal(
        coordinates, np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    np.testing.assert_array_equal(projected_values, np.array([3.0, 7.0]))


def test_project_prediction_values_sliced_accumulates_across_hidden_grid_sizes():
    for hidden_grid_size in (1, 3):
        spanning_dataset = DataSet(
            data=np.array([
                [shown_value, hidden_value]
                for shown_value in (0.0, 1.0)
                for hidden_value in range(hidden_grid_size)
            ]),
            observable_names=["shown", "hidden"],
        )

        _, projected_values = utils__project_prediction_values_sliced(
            values=np.ones(spanning_dataset.n_samples),
            spanning_dataset=spanning_dataset,
            along_observables=["shown"],
        )

        np.testing.assert_array_equal(
            projected_values, np.full(2, hidden_grid_size)
        )


def test_discover_background_only_parent_directory_selects_outermost(
    monkeypatch, tmp_path
):
    background_directory = tmp_path / "background"
    signal_directory = tmp_path / "signal"
    (background_directory / "configs").mkdir(parents=True)
    (signal_directory / "configs").mkdir(parents=True)
    background_context = SimpleNamespace(
        config=SimpleNamespace(dataset__has_signal=False)
    )
    signal_context = SimpleNamespace(
        config=SimpleNamespace(dataset__has_signal=True)
    )
    context_paths = [
        (background_context, background_directory / "run" / "context.json"),
        (signal_context, signal_directory / "run" / "context.json"),
    ]

    monkeypatch.setattr(
        "plot.plot_utils.ExecutionContext.discover_run_contexts",
        lambda _: context_paths,
    )
    assert utils__discover_background_only_parent_directory(str(tmp_path)) == (
        background_directory
    )


def test_discover_background_only_parent_directory_requires_background(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "plot.plot_utils.ExecutionContext.discover_run_contexts",
        lambda _: [
            (
                SimpleNamespace(
                    config=SimpleNamespace(dataset__has_signal=True)
                ),
                tmp_path / "signal" / "context.json",
            )
        ],
    )

    with pytest.raises(ValueError, match="No background-only submission"):
        utils__discover_background_only_parent_directory(str(tmp_path))


def test_discover_background_only_parent_directory_ignores_plot_outputs(
    monkeypatch, tmp_path
):
    background_directory = tmp_path / "background"
    plot_output_directory = tmp_path / "run_from_create_plots"
    (background_directory / "configs").mkdir(parents=True)
    background_context = SimpleNamespace(
        config=SimpleNamespace(dataset__has_signal=False)
    )
    plot_context = SimpleNamespace(
        config=SimpleNamespace(dataset__has_signal=False)
    )
    monkeypatch.setattr(
        "plot.plot_utils.ExecutionContext.discover_run_contexts",
        lambda _: [
            (plot_context, plot_output_directory / "context.json"),
            (background_context, background_directory / "run" / "context.json"),
        ],
    )

    assert utils__discover_background_only_parent_directory(str(tmp_path)) == (
        background_directory
    )
