from dataclasses import dataclass, field
from typing import Optional

from frame.cluster.walltime import parse_walltime, split_walltime


@dataclass
class ClusterConfig:
    cluster__repo_url: str
    cluster__environment_activation_command: str
    cluster__singularity_executable: str

    # qsub command parameters
    cluster__qsub_queue: str
    cluster__qsub_n_jobs: int
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_ngpus_for_train: int
    cluster__qsub_ncpus: Optional[int] = None
    cluster__qsub_walltime_limit: str = "72:00:00"
    cluster__qsub_total_walltime: Optional[str] = None
    cluster__qsub_walltime_chunks: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.cluster__qsub_total_walltime = self.cluster__qsub_total_walltime or self.cluster__qsub_walltime
        self.cluster__qsub_walltime_chunks = split_walltime(
            self.cluster__qsub_total_walltime,
            max_seconds=parse_walltime(self.cluster__qsub_walltime_limit),
        )
        self.cluster__qsub_walltime = self.cluster__qsub_walltime_chunks[0]

    @property
    def repo_name(self) -> str:
        return self.cluster__repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    @property
    def cluster__qsub_needs_continuation(self) -> bool:
        return len(self.cluster__qsub_walltime_chunks) > 1

    def next_walltime_chunk(self, submitted_chunk_count: int = 0) -> Optional[str]:
        if submitted_chunk_count < len(self.cluster__qsub_walltime_chunks):
            return self.cluster__qsub_walltime_chunks[submitted_chunk_count]
        return None

    def use_walltime_chunk(self, walltime: str) -> None:
        self.cluster__qsub_walltime = walltime
