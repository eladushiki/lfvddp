"""Versioned qsub adapter: receipt before invocation, durable ID before return."""

import inspect
import runpy
import subprocess
import sys
from frame.cluster.submission_intent import QUOTA_ERROR, SubmissionReceipt


def install_receipts(module):
    original = module.qsub_a_script
    signature = inspect.signature(original)

    def submit(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        receipt = SubmissionReceipt.from_environment(
            bound.arguments["context"], bound.arguments["job_name"]
        )
        if receipt is None:
            raise ValueError("Routine submission requires a saved canonical intent")
        bound.arguments.update(
            max_tries=1,
            job_name=receipt.call["pbs_name"],
            env_vars=dict(
                bound.arguments["env_vars"] or {},
                LFVDDP_SUBMISSION_INTENT=receipt.data["id"],
            ),
        )
        execute = module.check_output

        def recorded(*command_args, **command_kwargs):
            receipt.started()
            command_kwargs.setdefault("timeout", 40)
            try:
                output = execute(*command_args, **command_kwargs)
                receipt.accepted(output.decode().strip())
                return output
            except subprocess.CalledProcessError as error:
                receipt.failed(
                    error.output.decode(errors="replace"),
                    definite=bool(
                        QUOTA_ERROR.search(error.output.decode(errors="replace"))
                    ),
                )
                raise
            except BaseException as error:
                receipt.failed(str(error), definite=False)
                raise

        module.check_output = recorded
        try:
            return original(*bound.args, **bound.kwargs)
        finally:
            module.check_output = execute

    module.qsub_a_script = submit


def main(args):
    import frame.submit as submit

    install_receipts(submit)
    sys.argv = ["submit_train.py", *args]
    runpy.run_module("train.submit_train", run_name="__main__")
