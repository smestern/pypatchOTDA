"""Tests for patchOTDA.processUtils.timed_parallel_dispatch."""

import math
import time

import numpy as np
import pytest

from patchOTDA.processUtils import timed_parallel_dispatch


# ---------------------------------------------------------------------------
# Helper functions (module-level so they are picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _square(x):
    return x * x


def _sleep_and_return(duration, value):
    time.sleep(duration)
    return value


def _raise_error(msg):
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllComplete:
    """All jobs finish within the deadline."""

    def test_basic_results(self):
        results = timed_parallel_dispatch(
            _square,
            [(i,) for i in range(5)],
            timeout=10,
            backend="thread",
        )
        assert results == [0, 1, 4, 9, 16]

    def test_process_backend(self):
        results = timed_parallel_dispatch(
            _square,
            [(i,) for i in range(5)],
            timeout=10,
            backend="process",
        )
        assert results == [0, 1, 4, 9, 16]


class TestTimeout:
    """Some jobs exceed the global deadline."""

    def test_partial_timeout_thread(self):
        # Jobs: instant returns + one that sleeps 10s.
        args = [(0.0, 10), (0.0, 20), (10.0, 30)]
        results = timed_parallel_dispatch(
            _sleep_and_return,
            args,
            timeout=2,
            n_jobs=3,
            backend="thread",
        )
        assert results[0] == 10
        assert results[1] == 20
        # The slow job should have timed out → NaN
        assert np.isnan(results[2])

    def test_partial_timeout_process(self):
        args = [(0.0, 10), (0.0, 20), (10.0, 30)]
        results = timed_parallel_dispatch(
            _sleep_and_return,
            args,
            timeout=2,
            n_jobs=3,
            backend="process",
        )
        assert results[0] == 10
        assert results[1] == 20
        assert np.isnan(results[2])


class TestNoTimeout:
    """timeout=None waits for all jobs regardless of duration."""

    def test_no_timeout(self):
        args = [(0.5, 42)]
        results = timed_parallel_dispatch(
            _sleep_and_return,
            args,
            timeout=None,
            backend="thread",
        )
        assert results == [42]


class TestExceptionHandling:
    """Jobs that raise exceptions get default_value."""

    def test_exception_returns_default(self):
        results = timed_parallel_dispatch(
            _raise_error,
            [("boom",), ("bang",)],
            timeout=10,
            backend="thread",
        )
        assert all(np.isnan(r) for r in results)

    def test_custom_default_value(self):
        sentinel = -999.0
        results = timed_parallel_dispatch(
            _raise_error,
            [("oops",)],
            timeout=10,
            default_value=sentinel,
            backend="thread",
        )
        assert results == [sentinel]


class TestKwargsList:
    """kwargs_list is forwarded correctly."""

    def test_kwargs(self):
        def _add(a, b=0):
            return a + b

        results = timed_parallel_dispatch(
            _add,
            [(1,), (2,), (3,)],
            timeout=10,
            kwargs_list=[{"b": 10}, {"b": 20}, {"b": 30}],
            backend="thread",
        )
        assert results == [11, 22, 33]


class TestEdgeCases:
    """Empty input, single job, bad backend."""

    def test_empty_args(self):
        assert timed_parallel_dispatch(_square, [], timeout=1) == []

    def test_single_job(self):
        results = timed_parallel_dispatch(
            _square, [(7,)], timeout=10, backend="thread"
        )
        assert results == [49]

    def test_bad_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            timed_parallel_dispatch(_square, [(1,)], backend="gpu")

    def test_mismatched_kwargs_list(self):
        with pytest.raises(ValueError, match="kwargs_list length"):
            timed_parallel_dispatch(
                _square, [(1,), (2,)], kwargs_list=[{}], backend="thread"
            )


class TestOrderPreserved:
    """Results match input order regardless of completion order."""

    def test_order(self):
        # Stagger completion: job 0 finishes last among the fast ones.
        args = [(0.3, "a"), (0.1, "b"), (0.0, "c")]
        results = timed_parallel_dispatch(
            _sleep_and_return,
            args,
            timeout=5,
            n_jobs=3,
            backend="thread",
        )
        assert results == ["a", "b", "c"]
