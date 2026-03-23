"""Parallel dispatch utilities with global deadline timeout."""

import logging
import multiprocessing
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)

import numpy as np

logger = logging.getLogger(__name__)

_EXECUTORS = {
    "process": ProcessPoolExecutor,
    "thread": ThreadPoolExecutor,
}


def timed_parallel_dispatch(
    func,
    args_list,
    timeout=None,
    n_jobs=-1,
    default_value=np.nan,
    backend="process",
    kwargs_list=None,
):
    """Dispatch jobs in parallel with a global deadline timer.

    All jobs are submitted immediately.  A single wall-clock timer starts at
    submission time.  Jobs that complete before the deadline keep their return
    values; jobs still running when the deadline expires receive *default_value*
    (``np.nan`` by default).

    Parameters
    ----------
    func : callable
        The function to call for each job.
    args_list : list[tuple]
        Positional arguments for each invocation of *func*.  Length determines
        the number of jobs.
    timeout : float | None
        Global deadline in seconds.  ``None`` disables the timeout (wait
        forever).
    n_jobs : int
        Maximum number of parallel workers.  ``-1`` uses ``cpu_count()``.
    default_value : object
        Value inserted for jobs that time out or raise an exception.
    backend : ``'process'`` | ``'thread'``
        Executor backend.  ``'process'`` uses ``ProcessPoolExecutor`` (true
        parallelism, can force-terminate); ``'thread'`` uses
        ``ThreadPoolExecutor`` (lighter, works well with GIL-releasing code
        like NumPy / POT).
    kwargs_list : list[dict] | None
        Per-job keyword arguments, parallel to *args_list*.  ``None`` means
        no keyword arguments.

    Returns
    -------
    list
        Results in the same order as *args_list*.  Timed-out or failed entries
        contain *default_value*.
    """
    n = len(args_list)
    if n == 0:
        return []

    if kwargs_list is None:
        kwargs_list = [{}] * n
    if len(kwargs_list) != n:
        raise ValueError(
            f"kwargs_list length ({len(kwargs_list)}) must match "
            f"args_list length ({n})"
        )

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, n))

    executor_cls = _EXECUTORS.get(backend)
    if executor_cls is None:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'process' or 'thread'."
        )

    # Pre-fill with default so timed-out slots already have the right value.
    results = [default_value] * n
    timed_out = False

    executor = executor_cls(max_workers=n_jobs)
    try:
        # Submit all jobs and map each future back to its index.
        future_to_idx: dict[Future, int] = {}
        for idx, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
            future = executor.submit(func, *args, **kwargs)
            future_to_idx[future] = idx

        try:
            # as_completed yields futures as they finish.  If the global
            # deadline is hit, it raises TimeoutError.
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.debug(
                        "Job %d raised an exception – using default_value",
                        idx,
                        exc_info=True,
                    )
                    # results[idx] already == default_value

        except TimeoutError:
            timed_out = True
            pending = [
                idx
                for fut, idx in future_to_idx.items()
                if not fut.done()
            ]
            logger.info(
                "Global deadline reached after %.1fs – %d/%d jobs still "
                "pending; filling with default_value",
                timeout,
                len(pending),
                n,
            )
    finally:
        # On timeout: don't block waiting for slow jobs to finish.
        # cancel_futures=True cancels queued work; already-running threads
        # may linger but we don't wait for them.
        executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    return results
