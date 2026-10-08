"""Worker near-moment reuse preserves coefficients and avoids quadrature work."""
import copy
import os
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.compressed import tile_processes as tp, worker_pool
from ghost_backend.execution.options import automatic_options, execution_scope
from ghost_backend.execution.runtime import single_thread_worker_environment
from ghost_backend.tests.general_fixtures import fixture
from ghost_backend.twod import polynomial_quadrature as pq
from test_polynomial_basis import mesh_for


class NearOracle:
    """A real singular-pair query with only one tile, to isolate cache work."""
    def __init__(self, degree, k=37.5, shift=0.):
        self.element = copy.deepcopy(mesh_for(fixture('rectangle', 16), degree).elements[0])
        self.element.p0 += shift
        self.element.p1 += shift
        self.element.center += shift
        self.k = k
        self.n = degree+1
        self.calls = self.entries = self.dropped_routes = self.max_entries = 0

    def get_with_error(self, rows, cols):
        blocks = pq.near_blocks([(self.element, self.element)], self.k, threads=1)[0]
        raw = (blocks[0]+1j*blocks[1])[np.ix_(rows, cols)]
        self.calls += 1
        self.entries += raw.size
        self.max_entries = max(self.max_entries, raw.size)
        return raw, np.zeros(raw.shape)


class RawCompressor:
    def __init__(self, n):
        self.groups = [np.arange(n)]

    def compress_tile(self, i, j, raw, tail, proposal=None):
        return raw.copy(), tail.copy()


def payload(oracle):
    return pickle.dumps((oracle, [RawCompressor(oracle.n)], automatic_options()))


class WorkerMomentCacheTests(unittest.TestCase):
    def test_spawned_worker_reuses_p2_moments_in_p3_generation(self):
        quadratic, cubic = NearOracle(2), NearOracle(3)
        reference = cubic.get_with_error(np.arange(cubic.n), np.arange(cubic.n))[0]
        with tp._without_main_module(), single_thread_worker_environment():
            pool = worker_pool.WorkerPool(1)
            try:
                first, first_counts = pool.executor.submit(worker_pool.run_tile,
                    pool.prepare(payload(quadratic)), (0, 0, [0])).result(timeout=30)
                second, second_counts = pool.executor.submit(worker_pool.run_tile,
                    pool.prepare(payload(cubic)), (0, 0, [0])).result(timeout=30)
                self.assertGreater(first_counts.moment_cache['stores'], 0)
                self.assertGreater(second_counts.moment_cache['hits'], 0)
                self.assertEqual(second_counts.moment_cache['stores'], 0)
                np.testing.assert_array_equal(second[0][0], reference)
                # Same degree but a changed wavenumber or physical geometry is not reusable.
                for changed in (NearOracle(3, k=38.5), NearOracle(3, shift=.01)):
                    expected = changed.get_with_error(np.arange(changed.n), np.arange(changed.n))[0]
                    actual, counters = pool.executor.submit(worker_pool.run_tile,
                        pool.prepare(payload(changed)), (0, 0, [0])).result(timeout=30)
                    self.assertEqual(counters.moment_cache['hits'], 0)
                    self.assertGreater(counters.moment_cache['stores'], 0)
                    np.testing.assert_array_equal(actual[0][0], expected)
            finally:
                pool.close()

    def test_repeated_tile_uses_no_new_quadrature(self):
        oracle = NearOracle(3)
        with mock.patch.object(tp, '_MOMENTS', None), mock.patch.object(tp, '_WORKER', None), \
                mock.patch('ghost_backend.execution.thread_control.threadpool_limits'), \
                mock.patch('ghost_backend.twod.assembly.native.far.SPLIT_WORKERS', 1, create=True), \
                mock.patch.object(pq, '_evaluate_chunk', wraps=pq._evaluate_chunk) as evaluate:
            tp._initialize(payload(oracle))
            first, _ = tp._tile((0, 0, [0]))
            work = evaluate.call_count
            self.assertGreater(work, 0)
            second, counters = tp._tile((0, 0, [0]))
            self.assertEqual(evaluate.call_count, work)
            self.assertGreater(counters.moment_cache['hits'], 0)
            self.assertEqual(counters.moment_cache['stores'], 0)
            np.testing.assert_array_equal(first[0][0], second[0][0])

    def test_cache_owns_entries_and_evicts_within_its_budget(self):
        with mock.patch.object(tp, '_MOMENTS', None), mock.patch.object(tp, 'WORKER_MOMENT_BYTES', 2048):
            cache = tp._moment_cache()
            batch = np.ones((50, 2, 4, 4), complex)
            cache.put(b'a', batch[0])
            self.assertFalse(np.shares_memory(cache.values[b'a'], batch))
            cache.put(b'b', batch[1])
            cache.put(b'c', batch[2])
            self.assertEqual(set(cache.values), {b'b', b'c'})
            self.assertEqual(cache.evictions, 1)
            self.assertLessEqual(len(cache.values)*cache.ENTRY_BYTES, cache.budget)
            table = SimpleNamespace(evidence={'bytes': 1})
            self.assertIs(cache.store_table('separately priced CPU tables', table), table)
            self.assertEqual(cache.table_bytes, 0)
            self.assertFalse(cache.tables)

    def test_process_admission_and_idle_forecast_price_each_cache(self):
        op = SimpleNamespace(groups=[np.arange(1)]*16)
        oracle = SimpleNamespace(process_tiles=True)
        with execution_scope(automatic_options(), assembly_threads=2), \
                mock.patch.dict(os.environ, {'GHOST_TILE_PROCESSES': '2'}), \
                mock.patch.object(tp, '_compressor', return_value=None), \
                mock.patch('ghost_backend.compressed.runtime.storage_budget', return_value=0), \
                mock.patch('ghost_backend.compressed.worker_pool.retained_bytes', return_value=0):
            # 0.8GiB admits two old384MiB workers, but not their additional caches.
            with mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=.8):
                self.assertEqual(tp.prepare(oracle, [op]), (0, None))
            with mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=1.):
                self.assertEqual(tp.prepare(oracle, [op])[0], 2)
            self.assertEqual(worker_pool.forecast_bytes(10000, 2),
                             2*(tp.WORKER_BASE_BYTES+tp.WORKER_MOMENT_BYTES))
            pool = worker_pool.WorkerPool.__new__(worker_pool.WorkerPool)
            pool.workers, pool.payload_bytes = 2, 123
            self.assertEqual(pool.reserved_bytes, 2*(tp.WORKER_BYTES+246))


if __name__ == '__main__':
    unittest.main()
