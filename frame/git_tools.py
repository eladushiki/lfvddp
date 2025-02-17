import subprocess


def get_commit_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


def is_git_head_clean() -> bool:
    return subprocess.check_output(['git', 'status', '--porcelain']).strip() == b''