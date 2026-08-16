from pathlib import Path
from types import SimpleNamespace

import numpy as np

from frame.aggregate import ResultAggregator
from frame.file_system.training_history import HistoryKeys, save_training_history


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


def test_injected_significances_use_dataset_integration_limits(
    tmp_path,
    monkeypatch,
):
    integration_limits = np.array([4.0, 5.0])
    dataset_parameters = SimpleNamespace(
        dataset_generated__background_pdf=lambda coordinates: 1.0,
        dataset_generated__signal_pdf=lambda coordinates: 1.0,
        dataset__number_of_background_events=100,
        dataset__number_of_signal_events=10,
        dataset_generated__integration_upper_limits=integration_limits,
    )
    context = SimpleNamespace(config=SimpleNamespace())
    calculation_arguments = {}

    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda parent_directory: [(context, parent_directory)],
    )
    monkeypatch.setattr(
        "frame.aggregate.utils__get_signal_dataset_parameters",
        lambda signal_context: dataset_parameters,
    )
    monkeypatch.setattr(
        "frame.aggregate.calc_injected_t_significance_by_sqrt_q0_continuous",
        lambda **arguments: calculation_arguments.update(arguments) or 2.5,
    )

    significances = ResultAggregator(tmp_path).all_injected_significances

    np.testing.assert_allclose(significances, [2.5])
    np.testing.assert_array_equal(
        calculation_arguments["upper_limit"],
        integration_limits,
    )
