from dataclasses import dataclass


@dataclass
class ClusterConfig:
    cluster__repo_url: str
    cluster__environment_activation_command: str
    cluster__singularity_executable: str

    # qsub command parameters
    cluster__qsub_queue: str
    cluster__qsub_n_jobs: int
    cluster__qsub_job_name: str
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_ngpus_for_train: int

    @property
    def repo_name(self) -> str:
        return self.cluster__repo_url.rstrip("/").split("/")[-1].replace(".git", "")
