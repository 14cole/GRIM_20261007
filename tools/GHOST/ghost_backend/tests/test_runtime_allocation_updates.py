"""CPU reservations, bounded tile work, and unchanged compressed products."""
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.compressed import operator as operators, tile_processes
from ghost_backend.execution import options
from ghost_backend.execution.thread_control import threadpool_info, threadpool_limits
from ghost_backend.hpc import scheduler


def make_operator():
    rng = np.random.default_rng(704)
    n = 129
    op = operators.StreamedOperator(SimpleNamespace(n=n), rng.normal(size=(n, 2)), tile=24, assemble=False)
    for j, cols in enumerate(op.groups):
        for i, rows in enumerate(op.groups):
            rank = len(cols) if i == j else 3
            left = rng.normal(size=(len(rows), rank)) + 1j*rng.normal(size=(len(rows), rank))
            right = None if i == j else rng.normal(size=(rank, len(cols))) + 1j*rng.normal(size=(rank, len(cols)))
            op.tiles[i, j] = left, right
    return op, rng


def reference_product(op, b, trans):
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    result = np.zeros(b.shape, complex)
    def adj(a):
        return a.T if trans == 1 else a.conj().T
    for (i, j), (left, right) in op.tiles.items():
        rows, cols = op.groups[i], op.groups[j]
        if trans == 0:
            result[rows] += left@b[cols] if right is None else left@(right@b[cols])
        else:
            result[cols] += adj(left)@b[rows] if right is None else adj(right)@(adj(left)@b[rows])
    return result[:, 0] if vector else result


def test_affinity_slurm_and_per_solve_reservations_bound_every_cpu_team():
    with mock.patch.object(options.os, 'cpu_count', return_value=128), \
            mock.patch.object(options.os, 'sched_getaffinity', return_value=set(range(8)), create=True), \
            mock.patch.object(options, 'physical_core_count', return_value=64), \
            mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '16', 'SLURM_MEM_PER_CPU': '4096', 'SLURM_MEM_PER_NODE': ''}):
        assert scheduler.detect_cores() == 8
        assert scheduler.detect_memory_gb() == 64.0
        for request in ('auto', 32):
            with options.execution_scope(dict(assembly_threads=request, blas_threads='auto')):
                assert options.allocated_cpu_budget() == 8
                assert options.effective_assembly_threads() == 8
                assert options.blas_core_budget() == 8
            with options.execution_scope(dict(assembly_threads=request, blas_threads=32), assembly_threads=2):
                assert options.effective_assembly_threads() == 2
                with options.linear_algebra_threads(5000) as used:
                    assert used == 2
                with options.execution_scope(dict(assembly_threads=request), assembly_threads=64):
                    assert options.allocated_cpu_budget() == 2
        assert options.effective_assembly_threads(32) == 8


@pytest.mark.parametrize('trans', [0, 1, 2])
@pytest.mark.parametrize('columns', [0, 7, 8, 17])
@pytest.mark.parametrize('dtype', [float, complex])
def test_spatial_rhs_layout_preserves_all_products_and_input(trans, columns, dtype):
    op, rng = make_operator()
    raw = rng.normal(size=(op.n, max(1, columns)*2)).astype(dtype)
    if dtype is complex:
        raw += 1j*rng.normal(size=raw.shape)
    b = raw[:, ::2] if columns else raw[:, 0]
    before = b.copy()
    with threadpool_limits(1):
        expected = reference_product(op, b, trans)
        for cpus in (1, 4):
            with options.execution_scope(options.automatic_options(), assembly_threads=cpus):
                actual = op.matmul(b, trans)
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(b, before)


def test_one_cpu_never_starts_compression_or_product_thread_pool():
    op, _ = make_operator()
    with options.execution_scope(options.automatic_options(), assembly_threads=1), \
            mock.patch.object(operators, 'ThreadPoolExecutor', side_effect=AssertionError('unreserved team')):
        op.matmul(np.ones((op.n, 8)))
        op.equilibrate()
        seen = []
        with operators.TileWriter() as writer:
            writer.submit(lambda x: 2*x, seen.append, 3)
        assert seen == [6]


def test_parallel_products_temporarily_use_one_blas_thread():
    op, _ = make_operator()
    counts = []
    op.checkpoint = lambda: counts.extend(pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas')
    with options.execution_scope(options.automatic_options(), assembly_threads=4), threadpool_limits(3):
        op.matmul(np.ones((op.n, 8), complex))
        assert counts and set(counts) == {1}
        assert all(pool['num_threads'] == 3 for pool in threadpool_info() if pool['user_api'] == 'blas')


def test_tile_writer_reserves_cpus_for_compression_beside_native_assembly():
    with options.execution_scope(options.automatic_options(), assembly_threads=4):
        with operators.TileWriter() as writer:
            assert writer.workers == 2
            assert options.allocated_cpu_budget() == 2
            assert options.effective_assembly_threads() == 2
        assert options.allocated_cpu_budget() == 4


def test_tile_process_override_is_capped_by_current_reservation():
    op, _ = make_operator()
    with mock.patch.dict(os.environ, {'GHOST_TILE_PROCESSES': '100'}), \
            mock.patch.object(tile_processes, 'MIN_TILES', 1), \
            mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=32), \
            mock.patch('ghost_backend.compressed.runtime.storage_budget', return_value=0):
        for cpus, expected in ((1, 0), (2, 2)):
            with options.execution_scope(options.automatic_options(), assembly_threads=cpus):
                count, _ = tile_processes.prepare(SimpleNamespace(process_tiles=True), [op])
                assert count == expected


class FakeOracle:
    calls = entries = dropped_routes = max_entries = 0
    def get_with_error(self, rows, cols):
        self.calls += 1
        self.entries += 1
        self.max_entries = 1
        return None, None


class FakeOperator:
    groups = [np.asarray([i]) for i in range(5)]
    def __init__(self):
        self.pilot_tiles = {}
    def compress_tile(self, i, j, raw, tail, proposed=None):
        return i, j


class FakeFuture:
    def __init__(self, owner, task):
        self.owner, self.task, self.cancelled = owner, task, False
    def result(self, timeout=None):
        if self.owner.fail == self.task[:2]:
            from concurrent.futures.process import BrokenProcessPool
            raise BrokenProcessPool('test native worker death')
        if self.owner.waiting:
            from concurrent.futures import TimeoutError
            self.owner.wait_attempts += 1
            raise TimeoutError()
        self.owner.waiting_count -= 1
        i, j, missing = self.task
        return {index: (i, j) for index in missing}, [([1, 1, 0], 1)]
    def cancel(self):
        self.cancelled = True


class FakeExecutor:
    def __init__(self, *, fail=None, waiting=False):
        self.fail, self.waiting = fail, waiting
        self.wait_attempts = 0
        self.futures, self.waiting_count, self.peak = [], 0, 0
        self._processes, self.closed = {}, False
    def submit(self, function, task):
        future = FakeFuture(self, task)
        self.futures.append(future)
        self.waiting_count += 1
        self.peak = max(self.peak, self.waiting_count)
        return future
    def shutdown(self, **kwargs):
        self.closed = True


@pytest.mark.parametrize('fail', [None, (2, 0)])
def test_process_tile_window_bounds_results_and_recovers_in_original_order(fail):
    executor = FakeExecutor(fail=fail)
    oracle = FakeOracle()
    with mock.patch('concurrent.futures.ProcessPoolExecutor', return_value=executor):
        result = list(tile_processes.compressed_tiles(oracle, [FakeOperator()], 2, b'', lambda: None))
    assert result == [[(i, j)] for j in range(5) for i in range(5)]
    assert executor.peak <= 4
    assert oracle.calls == 25
    assert executor.closed


def test_waiting_process_tiles_respond_to_cancellation_and_close_workers():
    executor = FakeExecutor(waiting=True)
    def checkpoint():
        if executor.wait_attempts:
            raise InterruptedError('cancelled')
    with mock.patch('concurrent.futures.ProcessPoolExecutor', return_value=executor):
        with pytest.raises(InterruptedError, match='cancelled'):
            list(tile_processes.compressed_tiles(FakeOracle(), [FakeOperator()], 2, b'', checkpoint))
    assert executor.closed
    assert executor.wait_attempts == 1
    assert all(future.cancelled for future in executor.futures)
