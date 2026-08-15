"""Requeue marked LFVDDP jobs that yield a busy sandbox-cache allocation.

Install this file as a PBS ``execjob_epilogue`` site hook. It intentionally
depends only on PBS's embedded ``pbs`` module so it can run on execution hosts.
"""

import pbs


EXIT_STATUS_VARIABLE = "LFVDDP_CACHE_CONTENTION_EXIT_STATUS"

event = pbs.event()
job = event.job
expected_exit_status = job.Variable_List.get(EXIT_STATUS_VARIABLE)

if (
    job.in_ms_mom()
    and expected_exit_status is not None
    and int(job.Exit_status) == int(expected_exit_status)
):
    pbs.logjobmsg(
        job.id,
        "Requeueing LFVDDP job after sandbox cache contention.",
    )
    job.rerun()
    event.reject("LFVDDP sandbox cache busy; job requeued")

event.accept()
