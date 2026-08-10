from pathlib import Path

import numpy as np
import pytest

from plot.plot_utils import (
    utils__discover_background_only_parent_directory,
    utils__prediction_mesh_mask,
)


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


def test_discover_background_only_parent_directory_selects_outermost(
    monkeypatch, tmp_path
):
    background_directory = tmp_path / "background"
    signal_directory = tmp_path / "signal"
    background_context = object()
    signal_context = object()
    context_paths = [
        (background_context, background_directory / "run" / "context.json"),
        (signal_context, signal_directory / "run" / "context.json"),
    ]

    monkeypatch.setattr(
        "plot.plot_utils.ExecutionContext.discover_run_contexts",
        lambda _: context_paths,
    )
    monkeypatch.setattr(
        "plot.plot_utils.utils__context_has_signal",
        lambda context: context is signal_context,
    )

    assert utils__discover_background_only_parent_directory(str(tmp_path)) == (
        background_directory
    )


def test_discover_background_only_parent_directory_requires_background(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "plot.plot_utils.ExecutionContext.discover_run_contexts",
        lambda _: [(object(), tmp_path / "signal" / "context.json")],
    )
    monkeypatch.setattr(
        "plot.plot_utils.utils__context_has_signal", lambda _: True
    )

    with pytest.raises(ValueError, match="No background-only"):
        utils__discover_background_only_parent_directory(str(tmp_path))
