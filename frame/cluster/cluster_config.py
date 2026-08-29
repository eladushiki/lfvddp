from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from frame.cluster.walltime import format_walltime, parse_walltime, split_walltime


DEFAULT_QSUB_IO = 0.1
DEFAULT_QSUB_MEM = 2
DEFAULT_QSUB_NCPUS = 8
DEFAULT_QSUB_NGPUS_FOR_TRAIN = 0


@dataclass
class ClusterConfig:
    cluster__repo_url: str
    cluster__environment_activation_command: str
    cluster__singularity_executable: str

    # qsub command parameters
    cluster__qsub_queue: str
    cluster__qsub_n_jobs: int
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: float = DEFAULT_QSUB_IO
    cluster__qsub_mem: int = DEFAULT_QSUB_MEM
    cluster__qsub_ngpus_for_train: int = DEFAULT_QSUB_NGPUS_FOR_TRAIN
    cluster__qsub_ncpus: int = DEFAULT_QSUB_NCPUS
    cluster__uv_cache_dir: Optional[Path] = None
    cluster__qsub_walltime_limit: str = "72:00:00"
    cluster__qsub_total_walltime: Optional[str] = None
    cluster__qsub_walltime_chunks: list[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.cluster__qsub_ncpus < 1:
            raise ValueError("cluster__qsub_ncpus must be positive.")
        if self.cluster__qsub_ngpus_for_train < 0:
            raise ValueError("cluster__qsub_ngpus_for_train cannot be negative.")
        self._set_total_walltime(
            self.cluster__qsub_total_walltime or self.cluster__qsub_walltime
        )

    def _set_total_walltime(self, total_walltime: str) -> None:
        """Set the total budget and derive every per-submission walltime value."""
        self.cluster__qsub_total_walltime = total_walltime
        self.cluster__qsub_walltime_chunks = split_walltime(
            self.cluster__qsub_total_walltime,
            max_seconds=parse_walltime(self.cluster__qsub_walltime_limit),
        )
        self.cluster__qsub_walltime = self.cluster__qsub_walltime_chunks[0]

    def add_walltime(self, extra_walltime: str) -> None:
        total_seconds = parse_walltime(self.cluster__qsub_total_walltime)
        extra_seconds = parse_walltime(extra_walltime)
        self._set_total_walltime(format_walltime(total_seconds + extra_seconds))

    @property
    def repo_name(self) -> str:
        return self.cluster__repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    @property
    def cluster__qsub_needs_continuation(self) -> bool:
        return len(self.cluster__qsub_walltime_chunks) > 1

    def next_walltime_chunk(
        self, submitted_walltime_seconds: int = 0
    ) -> Optional[str]:
        remaining_seconds = (
            parse_walltime(self.cluster__qsub_total_walltime)
            - submitted_walltime_seconds
        )
        if remaining_seconds <= 0:
            return None

        walltime_limit_seconds = parse_walltime(self.cluster__qsub_walltime_limit)
        return format_walltime(min(remaining_seconds, walltime_limit_seconds))

    def use_walltime_chunk(self, walltime: str) -> None:
        self.cluster__qsub_walltime = walltime
