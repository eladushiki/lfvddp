import time
from pathlib import Path
from typing import Optional

import pytest

from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONTEXT_FILE_NAME, LOCAL_PROJECT_ROOT, TRAINING_OUTCOMES_DIR_NAME
from test.submission.submit_test_utils import (
    create_submit_config,
    load_submit_context,
    require_server_prerequisites,
    run_submit,
    submit_command,
    wait_for_job_to_finish,
)
from test.environment import ConfigType
from train.checkpoints import TRAINING_CHECKPOINT_SUFFIX, _torch_load


ARRAY_JOB_COUNT = 5
PROGRESS_WAIT_TIMEOUT_SECONDS = 2 * 60

CONFIG_PATHS = {
    ConfigType.CLUSTER: Path("test/configs/continuation/server_cluster_config.json"),
    ConfigType.DATASET: Path("test/configs/continuation/server_dataset_config.json"),
    ConfigType.DETECTOR: Path("test/configs/continuation/server_detector_config.json"),
    ConfigType.TRAIN: Path("test/configs/continuation/server_train_config.json"),
    ConfigType.USER: Path("test/configs/continuation/server_user_config.json"),
}


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


def _checkpoint_epochs_by_array(submit_run_dir: Path) -> dict[int, int]:
    epochs_by_array = {}
    for run_dir, context in _single_train_context_dirs(submit_run_dir):
        checkpoint_dir = run_dir / TRAINING_OUTCOMES_DIR_NAME
        epochs = []
        for checkpoint_path in checkpoint_dir.glob(f"*.{TRAINING_CHECKPOINT_SUFFIX}"):
            checkpoint = _torch_load(checkpoint_path)
            epochs.append(int(checkpoint["epoch"]))
        if epochs:
            current_epoch = epochs_by_array.get(context.array_index, -1)
            epochs_by_array[context.array_index] = max(current_epoch, max(epochs))
    return epochs_by_array


def _wait_for_all_arrays_to_progress(
    submit_run_dir: Path,
    previous_epochs: Optional[dict[int, int]] = None,
) -> dict[int, int]:
    deadline = time.monotonic() + PROGRESS_WAIT_TIMEOUT_SECONDS
    expected_arrays = set(range(1, ARRAY_JOB_COUNT + 1))

    while time.monotonic() < deadline:
        epochs = _checkpoint_epochs_by_array(submit_run_dir)
        if previous_epochs is None:
            if set(epochs) == expected_arrays:
                return epochs
        elif all(epochs.get(array_index, -1) > previous_epochs[array_index] for array_index in expected_arrays):
            return epochs
        time.sleep(2)

    pytest.fail(
        f"Timed out waiting for all arrays to progress. "
        f"Previous: {previous_epochs}; current: {_checkpoint_epochs_by_array(submit_run_dir)}"
    )


@pytest.mark.server
@pytest.mark.long
def test_submit_continue_advances_all_array_jobs():
    require_server_prerequisites()

    out_dir = Path("results") / f"pytest_server_continuation_{time.time_ns()}"
    config = create_submit_config(CONFIG_PATHS, out_dir)

    run_submit(submit_command(CONFIG_PATHS, out_dir))
    submit_context = load_submit_context(out_dir, config.config__dirsafe_runtag)
    submit_run_dir = LOCAL_PROJECT_ROOT / submit_context.unique_out_dir
    first_job_id = submit_context.qsub_submissions[0]["job_id"]

    wait_for_job_to_finish(first_job_id)
    first_epochs = _wait_for_all_arrays_to_progress(submit_run_dir)

    run_submit(submit_command(CONFIG_PATHS, out_dir, continue_training=True))
    submit_context = ExecutionContext.load_from_run_dir(submit_run_dir)
    second_job_id = submit_context.qsub_submissions[1]["job_id"]

    wait_for_job_to_finish(second_job_id)
    second_epochs = _wait_for_all_arrays_to_progress(submit_run_dir, previous_epochs=first_epochs)

    assert set(second_epochs) == set(range(1, ARRAY_JOB_COUNT + 1))
