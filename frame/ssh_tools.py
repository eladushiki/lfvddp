from os.path import isfile
from pathlib import Path, PurePosixPath
from typing import Optional
from scp import SCPClient
from paramiko import AutoAddPolicy, SSHClient


def scp_connection(
        remote_host: str,
        user_name: str,
        password: str,
    ):
    """
    Copy a file using SCP client.
    Returns: the file path on success, None otherwise.
    """
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        remote_host,
        username=user_name,
        password=password
    )
    if not ssh:
        raise ConnectionError("Could not connect to remote host")
    
    # Use existing connection to call SCP
    return SCPClient(ssh.get_transport()) # type: ignore


def scp_get_remote_file(
        remote_host: str,
        user_name: str,
        password: str,
        remote_file: PurePosixPath,
        local_file: Path,
    ) -> Optional[Path]:
    connection = scp_connection(remote_host, user_name, password)
    connection.get(str(remote_file), str(local_file))

    if isfile(local_file):
        return local_file


def scp_put_file_to_remote(
        remote_host: str,
        user_name: str,
        password: str,
        remote_file: PurePosixPath,
        local_file: Path,
    ) -> None:
    connection = scp_connection(remote_host, user_name, password)
    connection.put(str(local_file), str(remote_file))
