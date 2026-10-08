from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from frame.aggregate import ResultAggregator
from frame.file_system.training_history import HistoryKeys, save_training_history
from test.environment import ConfigType


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


def test_injected_significances_cache_duplicate_dataset_parameters(
    tmp_path,
    monkeypatch,
):
    dataset_parameters = SimpleNamespace(
        dataset_generated__background_pdf=lambda coordinates: 1.0,
        dataset_generated__signal_pdf=lambda coordinates: 1.0,
        dataset__number_of_background_events=100,
        dataset__number_of_signal_events=10,
        dataset_generated__integration_upper_limits=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    contexts = [SimpleNamespace(config=SimpleNamespace()) for _ in range(3)]
    calculation_count = 0

    def fake_calculation(**arguments):
        nonlocal calculation_count
        calculation_count += 1
        return 2.5

    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda parent_directory: [(context, parent_directory) for context in contexts],
    )
    monkeypatch.setattr(
        "frame.aggregate.utils__get_signal_dataset_parameters",
        lambda signal_context: dataset_parameters,
    )
    monkeypatch.setattr(
        "frame.aggregate.calc_injected_t_significance_by_sqrt_q0_continuous",
        fake_calculation,
    )

    significances = ResultAggregator(tmp_path).all_injected_significances

    np.testing.assert_allclose(significances, [2.5, 2.5, 2.5])
    assert calculation_count == 1


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_1D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(
                    "test/configs/train/orthogonal_legendre_binned.json"
                ),
            },
            id="orthogonal-polynomial",
        ),
    ],
    indirect=True,
)
def test_aggregate_derives_hypothesis_dof_from_run_context(
    tmp_path,
    monkeypatch,
    function_execution_context,
):
    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda parent_directory: [(function_execution_context, parent_directory)],
    )

    assert ResultAggregator(tmp_path).chi_square_degrees_of_freedom == 3


@pytest.mark.parametrize(
    ("function_execution_context", "expected_dof"),
    [
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    f"test/configs/dataset/disjoint_{dimension}D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    f"test/configs/detector/basic_{dimension}D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(f"test/configs/train/{fixture}.json"),
            },
            expected_dof,
            id=fixture,
        )
        for fixture, dimension, expected_dof in [
            ("short_1D_train_config_with_neural_nuisance", 1, 13),
            ("two_dimensional_adaptive_neural_binned", 2, 17),
        ]
    ],
    indirect=["function_execution_context"],
)
def test_aggregate_derives_adaptive_dof_from_run_context(
    tmp_path,
    monkeypatch,
    function_execution_context,
    expected_dof,
):
    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda parent_directory: [(function_execution_context, parent_directory)],
    )

    assert ResultAggregator(tmp_path).chi_square_degrees_of_freedom == expected_dof


def test_aggregate_keeps_nplm_raw_count_without_importing_backend(
    tmp_path, monkeypatch
):
    from frame.command_line.handle_args import create_config_from_paths
    from test.environment import DEFAULT_CONFIG_PATHS

    paths = {
        **DEFAULT_CONFIG_PATHS,
        ConfigType.TRAIN: Path(
            "test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json"
        ),
    }
    context = SimpleNamespace(config=create_config_from_paths(list(paths.values())))
    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda _: [(context, tmp_path)],
    )
    assert ResultAggregator(tmp_path).chi_square_degrees_of_freedom == 17
