from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from frame.command_line.execution import (
    CACHE_CONTENTION_EXIT_STATUS,
    CACHE_LOCK_TIMEOUT_SEC,
    format_qsub_execution_script,
)
from frame.file_system.textual_data import load_dict_from_json
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType
from train import cpu_runtime
from train.model_trainer import ParallelTrainLauncher, SequentialTrainLauncher
from train.runtime_resources import (
    ALLOCATED_CPUS_ENV,
    ALLOCATED_GPU_IDS_ENV,
    RuntimeAllocation,
    detect_runtime_allocation,
)
from train.single_train import select_train_launcher_class
from train.training_profiler import TrainingResourceProfiler


RESOURCE_CLUSTER_CONFIG = {
    ConfigType.CLUSTER.value: Path(
        "test/configs/cluster/resource_aware_cluster_config.json"
    )
}
ONE_DIMENSION_WITH_NUISANCE_CONFIG = {
    **RESOURCE_CLUSTER_CONFIG,
    ConfigType.DATASET.value: Path(
        "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
    ),
    ConfigType.DETECTOR.value: Path(
        "test/configs/detector/basic_1D_detector_config.json"
    ),
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_1D_train_config_with_nuisance.json"
    ),
}
ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG = {
    **RESOURCE_CLUSTER_CONFIG,
    ConfigType.DATASET.value: Path(
        "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
    ),
    ConfigType.DETECTOR.value: Path(
        "test/configs/detector/basic_1D_detector_config.json"
    ),
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_1D_train_config_without_nuisance.json"
    ),
}
ONE_DIMENSION_LIKE_NPLM_CONFIG = {
    **RESOURCE_CLUSTER_CONFIG,
    ConfigType.DATASET.value: Path(
        "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
    ),
    ConfigType.DETECTOR.value: Path(
        "test/configs/detector/basic_1D_detector_config.json"
    ),
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json"
    ),
}


def _allocation(cpus, gpus=0):
    return RuntimeAllocation(
        cpu_count=cpus,
        cpu_affinity=tuple(range(cpus)),
        assigned_gpu_ids=tuple(str(index) for index in range(gpus)),
        visible_gpu_count=gpus,
        gpu_names=tuple(f"GPU {index}" for index in range(gpus)),
        gpu_total_memory_bytes=tuple(1_000_000 for _ in range(gpus)),
        requested_cpus=cpus,
        requested_gpus=gpus,
    )


def test_cpu_runtime_configures_interop_threads_only_once(monkeypatch):
    interop_calls = []
    intraop_calls = []
    monkeypatch.setattr(cpu_runtime, "_INTEROP_THREADS_CONFIGURED", False)
    monkeypatch.setattr(torch, "set_num_interop_threads", interop_calls.append)
    monkeypatch.setattr(torch, "set_num_threads", intraop_calls.append)

    cpu_runtime.configure_cpu_runtime(8, log_metadata=False)
    cpu_runtime.configure_cpu_runtime(3, log_metadata=False)

    assert interop_calls == [1]
    assert intraop_calls == [8, 3]


def test_runtime_cpu_count_uses_export_and_affinity(monkeypatch):
    monkeypatch.setattr(
        "train.runtime_resources._affinity_cpu_ids", lambda: (2, 3, 4, 5)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    allocation = detect_runtime_allocation(
        requested_cpus=1,
        requested_gpus=0,
        environment={ALLOCATED_CPUS_ENV: "8"},
    )

    assert allocation.requested_cpus == 1
    assert allocation.cpu_count == 4
    assert allocation.cpu_affinity == (2, 3, 4, 5)


def test_runtime_gpu_count_uses_scheduler_assignment(monkeypatch):
    monkeypatch.setattr("train.runtime_resources._affinity_cpu_ids", lambda: (0,))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: f"assigned-{index}"
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: type("Properties", (), {"total_memory": 1234})(),
    )

    allocation = detect_runtime_allocation(
        requested_cpus=1,
        requested_gpus=1,
        environment={
            ALLOCATED_CPUS_ENV: "1",
            ALLOCATED_GPU_IDS_ENV: "GPU-a,GPU-b",
        },
    )

    assert allocation.usable_gpu_count == 2
    assert allocation.assigned_gpu_ids == ("GPU-a", "GPU-b")
    assert allocation.gpu_names == ("assigned-0", "assigned-1")


def test_requested_gpu_without_runtime_visibility_fails(monkeypatch):
    monkeypatch.setattr("train.runtime_resources._affinity_cpu_ids", lambda: (0,))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="exposed no CUDA devices"):
        detect_runtime_allocation(
            requested_cpus=1,
            requested_gpus=1,
            environment={ALLOCATED_CPUS_ENV: "1"},
        )


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_qsub_script_passes_observed_resources(function_execution_context):
    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )

    assert "#PBS -l ncpus=8" in script
    assert "#PBS -l ngpus=1" in script
    assert "THREADS_PER_PROCESS=$(detect_thread_count)" in script
    assert (
        'export_container_variable LFVDDP_ALLOCATED_CPUS "$THREADS_PER_PROCESS"'
        in script
    )
    assert (
        'export_container_variable LFVDDP_ALLOCATED_GPU_IDS "$ALLOCATED_GPU_IDS"'
        in script
    )
    assert (
        f'export_container_variable LFVDDP_COMMIT_HASH "'
        f'{function_execution_context.commit_hash}"'
    ) in script
    assert "export_container_variable PYTHONUNBUFFERED 1" in script
    assert "export_container_variable PYTHONFAULTHANDLER 1" in script
    assert "singularity exec --nv" in script
    assert "unset LD_PRELOAD" in script
    assert 'touch "${temporary_sandbox}/.ready"' in script
    assert script.index('touch "${temporary_sandbox}/.ready"') < script.index(
        'mv "$temporary_sandbox" "$SANDBOX_DIR"'
    )
    build_function = script.split("build_sandbox()", 1)[1].split(
        "acquire_cache_lock()", 1
    )[0]
    assert 'rm -rf "$SANDBOX_DIR"' not in build_function
    assert 'touch "$LEASE_FILE"' in script
    assert 'rm -rf "$SANDBOX_DIR" "$LEASES_DIR"' in script
    assert 'LOCK_FILE="${SANDBOX_DIR}.flock"' in script
    assert (
        'LOCK_TIMEOUT_SEC="${SINGULARITY_CACHE_LOCK_TIMEOUT_SEC:-'
        f'{CACHE_LOCK_TIMEOUT_SEC}}}"' in script
    )
    assert 'flock -w "$LOCK_TIMEOUT_SEC" "$CACHE_LOCK_FD"' in script
    assert 'flock -u "$CACHE_LOCK_FD"' in script
    assert "qrerun" not in script
    assert f"CACHE_CONTENTION_EXIT_STATUS={CACHE_CONTENTION_EXIT_STATUS}" in script
    assert 'exit "$CACHE_CONTENTION_EXIT_STATUS"' in script
    assert "SINGULARITY_SANDBOX_RETRY_MAX" not in script
    assert "declare -F release_sandbox" in script
    assert "trap 'exit 143' TERM" in script
    subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("function_execution_context", [{}], indirect=True)
def test_cluster_submission_defaults_to_32_requested_cpus(
    function_execution_context,
):
    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )

    assert function_execution_context.config.cluster__qsub_ncpus == 32
    assert "#PBS -l ncpus=32" in script
    assert "REQUESTED_CPUS=32" in script


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_runtime_resource_report_is_documented(function_execution_context):
    allocation = _allocation(3)
    profiler = TrainingResourceProfiler(
        function_execution_context,
        allocation,
        requested_memory_gib=16,
    )
    with profiler.stage("preparation"):
        pass
    profiler.record_branch(
        "numerator",
        device="cpu",
        cpu_threads=3,
        elapsed_seconds=2.0,
        epochs=10,
        static=False,
        peak_rss=123,
    )

    report_path = profiler.save()
    contents = load_dict_from_json(report_path)

    assert report_path.name.startswith("runtime_resources_")
    assert contents["requested"]["memory GiB"] == 16
    assert contents["observed"]["cpu_count"] == 3
    assert contents["assignments"]["numerator"]["epochs per second"] == 5.0
    assert contents["timing seconds"]["preparation"] >= 0


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG, ONE_DIMENSION_LIKE_NPLM_CONFIG],
    indirect=True,
)
def test_nonparallel_training_modes_select_sequential_launcher(
    function_execution_context,
):
    assert (
        select_train_launcher_class(
            function_execution_context.config, _allocation(8, 4)
        )
        is SequentialTrainLauncher
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
@pytest.mark.parametrize(
    "cpus,gpus,expected",
    [
        (8, 0, [("cpu", 7), ("cpu", 1)]),
        (8, 1, [("cuda:0", 7), ("cpu", 1)]),
        (8, 2, [("cuda:0", 7), ("cuda:1", 1)]),
        (8, 4, [("cuda:0", 7), ("cuda:1", 1)]),
    ],
)
def test_parallel_branch_placement(
    function_execution_context,
    data_generation,
    detector_effect,
    cpus,
    gpus,
    expected,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    allocation = _allocation(cpus, gpus)
    profiler = TrainingResourceProfiler(function_execution_context, allocation)
    launcher = ParallelTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=allocation,
        profiler=profiler,
    )
    indices = [
        launcher.add_training(
            detected_batch,
            detector_effect,
            is_numerator=is_numerator,
            name=f"branch_{is_numerator}",
        )
        for is_numerator in (True, False)
    ]

    assignments = launcher._parallel_assignments(indices)

    assert [(item.device, item.cpu_threads) for item in assignments] == expected
    assert bool(profiler.unused_resources) is (gpus > 2)


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_parallel_launcher_rejects_shared_only_capacity(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    launcher = ParallelTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=_allocation(1, 1),
    )
    indices = [
        launcher.add_training(
            detected_batch,
            detector_effect,
            is_numerator=is_numerator,
            name=f"invalid_parallel_{is_numerator}",
        )
        for is_numerator in (True, False)
    ]

    with pytest.raises(RuntimeError, match="requires independent capacity"):
        launcher._parallel_assignments(indices)


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_sequential_launcher_calculates_static_denominator_without_worker(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    launcher = SequentialTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=_allocation(1),
    )
    numerator_index = launcher.add_training(
        detected_batch, detector_effect, is_numerator=True, name="auto_numerator"
    )
    denominator_index = launcher.add_training(
        detected_batch, detector_effect, is_numerator=False, name="auto_denominator"
    )

    launcher.execute_trainings()

    assert launcher.get_training(numerator_index).model is not None
    denominator = launcher.get_training(denominator_index)
    assert denominator.model is not None
    assert denominator.model._epochs_executed == 0
    assert denominator.result == pytest.approx(
        float(detected_batch.unified_data.n_samples)
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_sequential_parent_path_applies_each_cpu_thread_assignment(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    monkeypatch,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    launcher = SequentialTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=_allocation(4),
    )
    launcher.add_training(
        detected_batch, detector_effect, is_numerator=True, name="cpu_numerator"
    )
    launcher.add_training(
        detected_batch, detector_effect, is_numerator=False, name="cpu_denominator"
    )
    configured_threads = []

    def record_cpu_runtime(cpu_threads, log_metadata):
        configured_threads.append(cpu_threads)

    def complete_without_model_work(training, device="cpu"):
        training.result = 0.0
        return None, 0.0

    monkeypatch.setattr(
        "train.model_trainer.configure_cpu_runtime", record_cpu_runtime
    )
    monkeypatch.setattr(
        launcher, "_follow_instructions_for_t", complete_without_model_work
    )

    launcher.execute_trainings()

    # The static denominator runs first with one thread.  The sole trainable
    # numerator then receives the complete observed CPU allocation.
    assert configured_threads == [1, 4]


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_parallel_launcher_runs_trainable_branches_in_cpu_workers(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    capsys,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    launcher = ParallelTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=_allocation(2),
    )
    indices = [
        launcher.add_training(
            detected_batch,
            detector_effect,
            is_numerator=is_numerator,
            name=f"parallel_{is_numerator}",
        )
        for is_numerator in (True, False)
    ]

    launcher.execute_trainings()

    for index in indices:
        training = launcher.get_training(index)
        assert training.model is not None
        assert training.history is not None
        assert training.result is not None
        assert training.model._epochs_executed == 100

    output = capsys.readouterr().out
    numerator_begin = output.index("BEGIN TRAINING OUTPUT: parallel_True")
    numerator_end = output.index("END TRAINING OUTPUT: parallel_True")
    denominator_begin = output.index("BEGIN TRAINING OUTPUT: parallel_False")
    denominator_end = output.index("END TRAINING OUTPUT: parallel_False")
    assert numerator_begin < numerator_end < denominator_begin < denominator_end
    for model_name in ("parallel_True", "parallel_False"):
        worker_output_path = (
            function_execution_context.training_outcomes_dir
            / f"{model_name}.worker_output.txt"
        )
        assert worker_output_path.is_file()
        assert "Training worker started" in worker_output_path.read_text()


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_parallel_placement_preserves_seeded_results(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )

    def histories_for(launcher_class, allocation):
        launcher = launcher_class(
            function_execution_context,
            detector_effect,
            allocation=allocation,
        )
        indices = [
            launcher.add_training(
                detected_batch,
                detector_effect,
                is_numerator=is_numerator,
                name=f"reproducible_{is_numerator}",
            )
            for is_numerator in (True, False)
        ]
        launcher.execute_trainings()
        return [launcher.get_training(index).history for index in indices]

    sequential_histories = histories_for(SequentialTrainLauncher, _allocation(1))
    parallel_histories = histories_for(ParallelTrainLauncher, _allocation(2))

    for sequential, parallel in zip(sequential_histories, parallel_histories):
        assert sequential is not None and parallel is not None
        torch.testing.assert_close(
            torch.as_tensor(sequential["loss"]),
            torch.as_tensor(parallel["loss"]),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_parallel_worker_failure_is_propagated(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    capsys,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    empty_sr_batch = DataBatch(
        (
            DataSet(
                np.empty((0, dataset.n_observables)),
                observable_names=dataset.observable_names,
                category=dataset.category,
            )
            if dataset.category
            in (DataSet.DataSetCategory.A_SR, DataSet.DataSetCategory.B_SR)
            else dataset,
            parameters,
        )
        for dataset, parameters in detected_batch
    )
    launcher = ParallelTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=_allocation(2),
    )
    launcher.add_training(
        empty_sr_batch,
        detector_effect,
        is_numerator=True,
        name="failing_numerator",
    )
    launcher.add_training(
        detected_batch,
        detector_effect,
        is_numerator=False,
        name="successful_denominator",
    )

    with pytest.raises(RuntimeError, match="failing_numerator failed"):
        launcher.execute_trainings()

    output = capsys.readouterr().out
    failing_begin = output.index("BEGIN TRAINING OUTPUT: failing_numerator")
    failing_traceback = output.index("Traceback (most recent call last)")
    failing_end = output.index("END TRAINING OUTPUT: failing_numerator (FAILED)")
    successful_begin = output.index(
        "BEGIN TRAINING OUTPUT: successful_denominator"
    )
    assert failing_begin < failing_traceback < failing_end < successful_begin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_lfvnn_prepared_data_and_optimizer_use_assigned_cuda_device(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="cuda_device_model",
        device="cuda:0",
    )
    prepared = model._prepare_training_data(detected_batch)
    optimizer = model.configure_optimizers()
    assert optimizer is not None

    loss = model(prepared)
    loss.backward()
    optimizer.step()

    assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}
    assert prepared.sr_events.device.type == "cuda"
    assert prepared.nuisance_bin_indices is not None
    assert prepared.nuisance_bin_indices.device.type == "cuda"
    assert prepared.a_cr_bin_counts is not None
    assert prepared.a_cr_bin_counts.device.type == "cuda"
    assert all(
        value.device.type == "cuda"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
