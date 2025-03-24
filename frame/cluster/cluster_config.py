from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass
class ClusterConfig:
    cluster__project_root_at_cluster_abspath: PurePosixPath
    cluster__environment_activation_script_at_cluster_abspath: PurePosixPath

    # qsub command parameters
    cluster__qsub_queue: str
    cluster__qsub_n_jobs: int
    cluster__qsub_job_name: str
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_ngpus_for_train: int
