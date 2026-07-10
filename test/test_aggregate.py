from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from frame.aggregate import ResultAggregator
from frame.file_system.training_history import HistoryKeys, save_training_history
from plot.plots import t_train_percentile_progression_plot
from plot.plotting_config import PlottingConfig


def _save_t_history(
    parent: Path,
    run_name: str,
    sample_name: str,
    run_hash: int,
    numerator,
    denominator,
    t,
) -> None:
    outcome_dir = parent / run_name / "training_outcomes"
    outcome_dir.mkdir(parents=True, exist_ok=True)
    save_training_history(
        {
            HistoryKeys.EPOCH.value: [4, 9],
            HistoryKeys.NUMERATOR.value: numerator,
            HistoryKeys.DENOMINATOR.value: denominator,
            HistoryKeys.T.value: t,
        },
        outcome_dir / f"{sample_name}_{run_hash}.history.h5",
        epochs=10,
    )


def test_result_aggregator_keeps_each_paired_history_and_sums_t(tmp_path):
    _save_t_history(tmp_path, "run_1", "A", 1, [4, 3], [4.5, 4], [1, 2])
    _save_t_history(tmp_path, "run_1", "B", 1, [5, 4], [6.5, 6], [3, 4])
    _save_t_history(tmp_path, "run_2", "A", 2, [3, 2], [4.5, 4], [3, 4])
    _save_t_history(tmp_path, "run_2", "B", 2, [2, 1], [4.5, 4], [5, 6])
    (tmp_path / "run_1" / "final_t_1.txt").write_text("6\n")
    (tmp_path / "run_2" / "final_t_2.txt").write_text("10\n")

    aggregator = ResultAggregator(tmp_path)

    np.testing.assert_array_equal(aggregator.all_epochs, [4, 9])
    np.testing.assert_allclose(
        aggregator.all_history_values["A"][HistoryKeys.NUMERATOR.value],
        [[4, 3], [3, 2]],
    )
    np.testing.assert_allclose(
        aggregator.all_history_values["B"][HistoryKeys.DENOMINATOR.value],
        [[6.5, 6], [4.5, 4]],
    )
    np.testing.assert_allclose(aggregator.all_test_statistics, [[4, 6], [8, 10]])
    np.testing.assert_allclose(np.sort(aggregator.all_t_values), [6, 10])


def test_progression_plot_has_numerator_denominator_and_t_for_each_sample(tmp_path):
    for run_index in (1, 2):
        _save_t_history(
            tmp_path, f"run_{run_index}", "A", run_index, [4, 3], [4.5, 4], [1, 2]
        )
        _save_t_history(
            tmp_path, f"run_{run_index}", "B", run_index, [5, 4], [6.5, 6], [3, 4]
        )

    config = PlottingConfig(
        plot__target_run_parent_directory=str(tmp_path),
        plot__pyplot_styling={"rcParams": {}, "style.use": "default"},
        plot__figure_styling={"patch_set_facecolor": "white"},
        plot__figure_size=(12, 8),
        plot__plot_specifications=[],
    )
    context = SimpleNamespace(config=config, run_hash="test")

    figure = t_train_percentile_progression_plot(context)
    try:
        assert len(figure.axes) == 6
        assert {axis.get_title() for axis in figure.axes} == {
            "A: numerator minimization",
            "A: denominator minimization",
            r"A: $t=-2\,N+2\,D$",
            "B: numerator minimization",
            "B: denominator minimization",
            r"B: $t=-2\,N+2\,D$",
        }
    finally:
        plt.close(figure)
