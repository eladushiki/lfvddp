import time
from pathlib import Path

import pytest

from frame.cluster.walltime import parse_walltime
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    CONTEXT_FILE_NAME,
    LOCAL_PROJECT_ROOT,
    TRAINING_OUTCOMES_DIR_NAME,
)
from test.submission.submit_test_utils import (
    load_submit_context,
    require_server_prerequisites,
    run_submit,
    build_submit_command,
    wait_for_job_to_finish,
)
from test.environment import ConfigType
from train.checkpoints import TRAINING_CHECKPOINT_SUFFIX, _torch_load
from train.training_names import symmetric_training_names


def _array_indices(n_jobs: int) -> set[int]:
    return set(range(1, n_jobs + 1))


def _single_train_context_dirs(submit_run_dir: Path) -> list[tuple[Path, ExecutionContext]]:
    contexts = []
    for context_path in submit_run_dir.glob(f"*/{CONTEXT_FILE_NAME}"):
        context = ExecutionContext.load_from_run_dir(context_path.parent)
        if context.array_index is None:
            continue
        if "single_train.py" not in str(context.run_descriptor):
            continue
        contexts.append((context_path.parent, context))
    return contexts


def _checkpoint_epochs_by_array_and_training(submit_run_dir: Path) -> dict[int, dict[str, int]]:
    epochs_by_array = {}
    for run_dir, context in _single_train_context_dirs(submit_run_dir):
        checkpoint_dir = run_dir / TRAINING_OUTCOMES_DIR_NAME
        for checkpoint_path in checkpoint_dir.glob(f"*.{TRAINING_CHECKPOINT_SUFFIX}"):
            checkpoint = _torch_load(checkpoint_path)
            model_name = checkpoint["model_name"]
            epoch = int(checkpoint["epoch"])
            array_epochs = epochs_by_array.setdefault(context.array_index, {})
            array_epochs[model_name] = max(array_epochs.get(model_name, -1), epoch)
    return epochs_by_array


def _completed_epoch_count(
    training_epochs: dict[str, int],
    expected_training_names: list[str],
    target_epochs: int,
) -> int:
    return sum(
        max(0, min(training_epochs.get(training_name, -1) + 1, target_epochs))
        for training_name in expected_training_names
    )


def _next_unfinished_training_name(
    training_epochs: dict[str, int],
    expected_training_names: list[str],
    target_epochs: int,
) -> str:
    final_epoch = target_epochs - 1
    for training_name in expected_training_names:
        if training_epochs.get(training_name, -1) < final_epoch:
            return training_name
    raise AssertionError(f"All trainings are already complete: {training_epochs}")


def _progress_summary(progress: dict[int, dict[str, int]]) -> dict[int, dict[str, int]]:
    return {
        array_index: dict(sorted(training_epochs.items()))
        for array_index, training_epochs in sorted(progress.items())
    }


def _wait_for_all_completed_arrays_to_report_progress(
    submit_run_dir: Path,
    expected_training_names: list[str],
    target_epochs: int,
    submitted_walltime: str,
    n_jobs: int,
) -> dict[int, dict[str, int]]:
    deadline = time.monotonic() + parse_walltime(submitted_walltime)
    expected_array_indices = _array_indices(n_jobs)

    while time.monotonic() < deadline:
        progress = _checkpoint_epochs_by_array_and_training(submit_run_dir)
        if set(progress) == expected_array_indices and all(
            _completed_epoch_count(
                progress[array_index],
                expected_training_names,
                target_epochs,
            ) > 0
            for array_index in expected_array_indices
        ):
            return progress
        time.sleep(2)

    pytest.fail(
        f"Timed out waiting for all completed arrays to report checkpoint progress. "
        f"Current: {_progress_summary(_checkpoint_epochs_by_array_and_training(submit_run_dir))}"
    )


def _assert_continuation_advanced_next_training_for_each_array(
    previous_progress: dict[int, dict[str, int]],
    current_progress: dict[int, dict[str, int]],
    expected_training_names: list[str],
    target_epochs: int,
    n_jobs: int,
) -> None:
    expected_array_indices = _array_indices(n_jobs)
    expected_training_names_set = set(expected_training_names)

    assert set(current_progress) == expected_array_indices, _progress_summary(current_progress)
    for array_index in expected_array_indices:
        previous_training_epochs = previous_progress[array_index]
        current_training_epochs = current_progress[array_index]
        unexpected_training_names = (
            set(previous_training_epochs) | set(current_training_epochs)
        ) - expected_training_names_set

        assert not unexpected_training_names, (
            f"Unexpected training checkpoints for array {array_index}: "
            f"{sorted(unexpected_training_names)}"
        )
        previous_completed_epochs = _completed_epoch_count(
            previous_training_epochs,
            expected_training_names,
            target_epochs,
        )
        current_completed_epochs = _completed_epoch_count(
            current_training_epochs,
            expected_training_names,
            target_epochs,
        )
        assert current_completed_epochs > previous_completed_epochs, (
            f"Array {array_index} did not advance after continuation. "
            f"Previous: {previous_training_epochs}; current: {current_training_epochs}"
        )

        for training_name, previous_epoch in previous_training_epochs.items():
            current_epoch = current_training_epochs.get(training_name, -1)
            assert current_epoch >= previous_epoch, (
                f"{training_name} regressed for array {array_index}: "
                f"{previous_epoch} -> {current_epoch}"
            )

        next_training_name = _next_unfinished_training_name(
            previous_training_epochs,
            expected_training_names,
            target_epochs,
        )
        previous_epoch = previous_training_epochs.get(next_training_name, -1)
        current_epoch = current_training_epochs.get(next_training_name, -1)
        assert current_epoch > previous_epoch, (
            f"{next_training_name} did not resume for array {array_index}: "
            f"{previous_epoch} -> {current_epoch}"
        )


@pytest.mark.server
@pytest.mark.long
@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER: Path("test/submission/continuation/configs/continuation_cluster_config.json"),
            ConfigType.DATASET: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
            ConfigType.DETECTOR: Path("test/configs/detector/basic_1D_detector_config.json"),
            ConfigType.TRAIN: Path("test/configs/train/long_1D_train_config_with_nuisance.json"),
        },
        {
            ConfigType.CLUSTER: Path("test/submission/continuation/configs/continuation_cluster_config.json"),
            ConfigType.DATASET: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
            ConfigType.DETECTOR: Path("test/configs/detector/basic_1D_detector_config.json"),
            ConfigType.TRAIN: Path("test/configs/train/long_1D_train_config_without_nuisance.json"),
        },
    ],
    indirect=True,
)
def test_submit_continue_advances_all_array_jobs(
    function_execution_context,
):
    require_server_prerequisites()

    out_dir = Path("results") / f"pytest_server_continuation_{time.time_ns()}"
    config = function_execution_context.config

    run_submit(build_submit_command(function_execution_context, out_dir))
    submit_context = load_submit_context(out_dir, config.config__dirsafe_runtag)
    submit_run_dir = LOCAL_PROJECT_ROOT / submit_context.unique_out_dir
    first_job_id = submit_context.qsub_submissions[0]["job_id"]
    expected_training_names = symmetric_training_names(config.train__data_is_train_for_nuisances)
    target_epochs = config.train__epochs

    wait_for_job_to_finish(first_job_id)

    first_progress = _wait_for_all_completed_arrays_to_report_progress(
        submit_run_dir,
        expected_training_names,
        target_epochs,
        submitted_walltime=config.cluster__qsub_walltime_chunks[0],
        n_jobs=config.cluster__qsub_n_jobs,
    )

    run_submit(build_submit_command(function_execution_context, out_dir, continue_training=True))

    submit_context = ExecutionContext.load_from_run_dir(submit_run_dir)
    second_job_id = submit_context.qsub_submissions[1]["job_id"]

    wait_for_job_to_finish(second_job_id)

    second_progress = _wait_for_all_completed_arrays_to_report_progress(
        submit_run_dir,
        expected_training_names,
        target_epochs,
        submitted_walltime=config.cluster__qsub_walltime_chunks[1],
        n_jobs=config.cluster__qsub_n_jobs,
    )

    _assert_continuation_advanced_next_training_for_each_array(
        first_progress,
        second_progress,
        expected_training_names,
        target_epochs,
        n_jobs=config.cluster__qsub_n_jobs,
    )
