import subprocess
from os import environ

COMMIT_HASH_ENVIRONMENT_VARIABLE = "LFVDDP_COMMIT_HASH"


def get_commit_hash() -> str:
    if commit_hash := environ.get(COMMIT_HASH_ENVIRONMENT_VARIABLE):
        return commit_hash
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


class DirtyGitHeadError(RuntimeError):
    """A version-controlled entry point observed checkout changes."""


def git_head_changes() -> str:
    """Read checkout changes without refreshing or otherwise writing the index."""
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        text=True,
    ).strip()


def is_git_head_clean() -> bool:
    try:
        require_clean_git_head()
    except DirtyGitHeadError:
        return False
    return True


def require_clean_git_head() -> None:
    """Confirm an apparent change once, without accepting a concurrent HEAD move."""
    if COMMIT_HASH_ENVIRONMENT_VARIABLE in environ:
        return
    before = get_commit_hash()
    changes = git_head_changes()
    if changes:
        confirmed = git_head_changes()
        if confirmed:
            raise DirtyGitHeadError(
                "Commit changes before running the script.\n" + confirmed
            )
    if get_commit_hash() != before:
        raise DirtyGitHeadError("Git HEAD changed during the cleanliness check.")


def current_git_branch() -> str:
    """Get the current Git branch name.

    Returns:
        str: The name of the current Git branch
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .strip()
        .decode("utf-8")
    )


def default_git_branch() -> str:
    """Get the default Git branch name, usually 'main' or 'master'.

    Returns:
        str: The name of the default Git branch
    """
    remote_info = subprocess.check_output(["git", "remote", "show", "origin"]).decode(
        "utf-8"
    )
    for line in remote_info.splitlines():
        if "HEAD branch" in line:
            return line.split(":")[-1].strip()
    return "main"  # Fallback to 'main' if not found


def get_remote_commit_hash(branch: str = "main") -> str:
    """Get the latest commit hash from a remote branch without updating local git state.

    Args:
        branch: The branch name to get the commit hash from

    Returns:
        str: The commit hash of the latest commit on the remote branch
    """
    # Use ls-remote to query the remote without fetching or updating local state
    output = (
        subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{branch}"])
        .strip()
        .decode("utf-8")
    )
    # Output format: "commit_hash\trefs/heads/branch_name"
    return output.split()[0] if output else ""
