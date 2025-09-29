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
