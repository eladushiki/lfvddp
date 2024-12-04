from argparse import ArgumentParser

from frame.git_tools import get_commit_hash, is_git_head_clean


def main() -> None:
    remote_commit_hash = parse_arg_commit_hash()

    if not is_git_head_clean():
        raise RuntimeError("Git head is not clean")

    local_commit_hash = get_commit_hash()

    # Return 0 if the commit hashes match, 1 otherwise
    return remote_commit_hash != local_commit_hash


def parse_arg_commit_hash():
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "--commit-hash", type=str, required=True,
        help="Commit hash to compare with",
        dest="commit_hash",
    )

    args = argument_parser.parse_args()

    return args.commit_hash


if __name__ == "__main__":
    main()
