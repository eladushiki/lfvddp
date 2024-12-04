from os.path import isfile
from pathlib import Path
from typing import Optional
from scp import SCPClient
from paramiko import SSHClient


def copy_remote_file(
        remote_host: str,
        user_name: str,
        password: str,
        remote_file: Path,
        local_file: Path,
    ) -> Optional[Path]:
    """
    Copy a file using SCP client.
    Returns: the file path on success, None otherwise.
    """
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(
        remote_host,
        username=user_name,
        password=password
    )

    # Use existing connection to call SCP
    scp = SCPClient(ssh.get_transport())
    scp.get(str(remote_file), str(local_file))

    if isfile(local_file):
        return local_file
