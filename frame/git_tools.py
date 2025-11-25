import subprocess


def get_commit_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


def is_git_head_clean() -> bool:
    # Run `git update-index --assume-unchanged` for the unignored NPLM pycache
    subprocess.run(['git', 'update-index', '--assume-unchanged', 'neural_networks/NPLM/src/NPLM/__pycache__/'], check=True)
    
    # Check if the Git working directory is clean
    return subprocess.check_output(['git', 'status', '--porcelain']).strip() == b''


def current_git_branch() -> str:
    """Get the current Git branch name.
    
    Returns:
        str: The name of the current Git branch
    """
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')


def default_git_branch() -> str:
    """Get the default Git branch name, usually 'main' or 'master'.
    
    Returns:
        str: The name of the default Git branch
    """
    remote_info = subprocess.check_output(['git', 'remote', 'show', 'origin']).decode('utf-8')
    for line in remote_info.splitlines():
        if 'HEAD branch' in line:
            return line.split(':')[-1].strip()
    return 'main'  # Fallback to 'main' if not found


def get_remote_commit_hash(branch: str = 'main') -> str:
    """Get the latest commit hash from a remote branch without updating local git state.
    
    Args:
        branch: The branch name to get the commit hash from
        
    Returns:
        str: The commit hash of the latest commit on the remote branch
    """
    # Use ls-remote to query the remote without fetching or updating local state
    output = subprocess.check_output(['git', 'ls-remote', 'origin', f'refs/heads/{branch}']).strip().decode('utf-8')
    # Output format: "commit_hash\trefs/heads/branch_name"
    return output.split()[0] if output else ''
