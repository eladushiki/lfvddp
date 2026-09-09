"""Dedicated entry point for the opt-in six-case PBS thread-pool probe.

Usage::

    python -m tools.thread_pool_probe.submit \
        --configs tools/thread_pool_probe/configs --only-train

The normal ``train.submit_train`` entry point does not enable probe behavior.
This wrapper opts in before delegating to the ordinary submission workflow.
"""

import os


def main() -> None:
    os.environ["LFVDDP_THREAD_PROBE"] = "1"
    from train.submit_train import submit_process

    submit_process()


if __name__ == "__main__":
    main()
