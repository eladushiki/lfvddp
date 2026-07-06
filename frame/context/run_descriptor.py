from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from frame.file_structure import CONTEXT_FILE_NAME


RUN_OF_TOKEN = "_run_of_"
PID_TOKEN = "_pid_"
RUNTAG_SEPARATOR = "_"


@dataclass(frozen=True)
class RunDescriptorParts:
    stamp: str
    dirsafe_runtag: str
    entrypoint: str
    pid: int


def _entrypoint_name(entrypoint: Union[str, Path]) -> str:
    return Path(entrypoint).name


def build_run_descriptor(
    stamp: str,
    dirsafe_runtag: str,
    entrypoint: Union[str, Path],
    pid: int,
) -> str:
    entrypoint_name = _entrypoint_name(entrypoint)
    return (
        f"{stamp}{RUNTAG_SEPARATOR}{dirsafe_runtag}"
        f"{RUN_OF_TOKEN}{entrypoint_name}{PID_TOKEN}{pid}"
    )


def parse_run_descriptor(
    raw: Optional[str],
    dirsafe_runtag: Optional[str] = None,
) -> Optional[RunDescriptorParts]:
    if not raw:
        return None

    try:
        prefix, pid_text = raw.rsplit(PID_TOKEN, 1)
        prefix, entrypoint = prefix.rsplit(RUN_OF_TOKEN, 1)
    except ValueError:
        return None

    if not entrypoint or _entrypoint_name(entrypoint) != entrypoint:
        return None
    if not pid_text.isdigit():
        return None

    pid = int(pid_text)
    if str(pid) != pid_text:
        return None

    if dirsafe_runtag is not None:
        expected_suffix = f"{RUNTAG_SEPARATOR}{dirsafe_runtag}"
        if not prefix.endswith(expected_suffix):
            return None
        stamp = prefix[:-len(expected_suffix)]
    else:
        try:
            stamp, dirsafe_runtag = prefix.rsplit(RUNTAG_SEPARATOR, 1)
        except ValueError:
            return None

    if not stamp or not dirsafe_runtag:
        return None

    return RunDescriptorParts(
        stamp=stamp,
        dirsafe_runtag=dirsafe_runtag,
        entrypoint=entrypoint,
        pid=pid,
    )


def run_descriptor_matches(
    raw: Optional[str],
    entrypoint: Optional[Union[str, Path]] = None,
    dirsafe_runtag: Optional[str] = None,
) -> bool:
    parts = parse_run_descriptor(raw, dirsafe_runtag=dirsafe_runtag)
    if parts is None:
        return False
    if entrypoint is not None and parts.entrypoint != _entrypoint_name(entrypoint):
        return False
    return True


def context_glob_for_run(
    dirsafe_runtag: str,
    entrypoint: Optional[Union[str, Path]] = None,
) -> str:
    entrypoint_pattern = _entrypoint_name(entrypoint) if entrypoint is not None else "*"
    return (
        f"*{RUNTAG_SEPARATOR}{dirsafe_runtag}"
        f"{RUN_OF_TOKEN}{entrypoint_pattern}{PID_TOKEN}*/{CONTEXT_FILE_NAME}"
    )
