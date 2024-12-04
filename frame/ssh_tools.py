from contextlib import contextmanager
from os.path import isfile
from pathlib import Path, PurePosixPath
from typing import Optional
from scp import SCPClient
from paramiko import AutoAddPolicy, SSHClient

from train.train_config import ClusterConfig


@contextmanager
def open_ssh_client(config: ClusterConfig):
    remote_host = config.cluster__host_address
    user_name = config.cluster__user
    password = config.cluster__password
    try:
        if config.cluster__is_use_ssh_jump:  # Note: never succeeded, since direct SSH to sshproxy is blocked
            jump_client = SSHClient()
            jump_client.load_system_host_keys()
            jump_client.set_missing_host_key_policy(AutoAddPolicy())

            if not (jump_host := config.cluster__jump_host_address) or \
               not (jump_user := config.cluster__jump_user) or \
               not (jump_password := config.cluster__jump_password):
                raise ValueError("SSH jump missing parameters")
            jump_client.connect(
                hostname=config.cluster__jump_host_address,
                username=config.cluster__jump_user,
                password=config.cluster__jump_password,
            )
            if not jump_client:
                raise ConnectionError("SSH jump client could not connect to remote host")
        
        ssh_client = SSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())

        if config.cluster__is_use_ssh_jump:
            channel = jump_client.get_transport().open_channel('direct-tcpip', (remote_host, 22), (jump_host, 22))
            ssh_client.connect(
                hostname=remote_host,
                username=user_name,
                password=password,
                sock=channel,
            )
        else:
            ssh_client.connect(
                hostname=remote_host,
                username=user_name,
                password=password
            )
        if not ssh_client:
            raise ConnectionError("SSH client could not connect to remote host")

        yield ssh_client

    finally:
        if config.cluster__is_use_ssh_jump:
            jump_client.close()
        ssh_client.close()


@contextmanager
def open_scp_client(config: ClusterConfig):
    """
    Copy a file using SCP client.
    Returns: the file path on success, None otherwise.
    """
    with open_ssh_client(config) as ssh_client:
        try:
            # Use existing connection to call SCP
            scp_client = SCPClient(ssh_client.get_transport()) # type: ignore
            
            if not scp_client:
                raise ConnectionError("SCP client could not connect to remote host")
            
            yield scp_client

        finally:
            scp_client.close()
    

def scp_get_remote_file(
        config: ClusterConfig,
        remote_file: PurePosixPath,
        local_file: Path,
    ) -> Optional[Path]:

    with open_scp_client(config) as client:
        client.get(str(remote_file), str(local_file))

    if isfile(local_file):
        return local_file


def scp_put_file_to_remote(
        config: ClusterConfig,
        remote_file: PurePosixPath,
        local_file: Path,
    ) -> None:
    with open_scp_client(config) as client:
        client.put(str(local_file), str(remote_file))


def run_command_over_ssh(
        config: ClusterConfig,
        command: str,
    ):
    with open_ssh_client(config) as ssh_client:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        return stdin, stdout, stderr
