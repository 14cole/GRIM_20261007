"""Compressed tiles assembled in worker processes match the in-process operator."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ghost_backend.compressed.tile_processes as tile_processes
import ghost_backend.execution.cpu as execution
import ghost_backend.twod.formulations.regions as regions
from ghost_backend.compressed.polarization_cache import build_pair
from ghost_backend.compressed.regional_coefficients import PairedOracle
from ghost_backend.execution.options import automatic_options, execution_scope
from test_compact_multi_region import prepared


class CrashingPairedOracle(PairedOracle):
    """Ends its worker process once, on the first query that finds the flag file."""
    flag = None

    def get_with_error(self, rows, cols):
        if self.flag and os.path.exists(self.flag):
            try:
                os.remove(self.flag)
            except OSError:
                pass
            os._exit(3)
        return super().get_with_error(rows, cols)


def paired_operators(oracle_class, workers, flag=None, degree=1, count=160, details=False):
    options = dict(automatic_options(), basis_order=degree)
    with execution_scope(options):
        mesh, te, _ = prepared('mixed', 'TE', count)
        _, tm, _ = prepared('mixed', 'TM', count)
    with mock.patch.dict(os.environ, {'GHOST_TILE_PROCESSES': str(workers)}), \
            mock.patch.object(tile_processes, 'MIN_TILES', 1), \
            execution_scope(options), execution._STATE.override(execution.CPUState()), \
            tempfile.TemporaryDirectory() as directory:
        oracle = oracle_class(mesh, te, tm, cut=32)
        oracle.flag = flag
        coordinates = regions.dof_coordinates(mesh, oracle.oracles[0].layout)
        if flag is not None:
            Path(flag).write_text('crash')
        operators = build_pair(oracle, coordinates, tile=16, budget=2**30, spool_directory=directory)
        operators[1].load()
        identity = np.eye(operators[0].n, dtype=complex)
        result = ([op.matmul(identity) for op in operators] + [op.row_error.copy() for op in operators],
                  [(source.calls, source.entries) for source in oracle.oracles])
        return result + ([op.evidence for op in operators],) if details else result


class TileProcessTests(unittest.TestCase):
    def test_worker_processes_match_in_process_assembly(self):
        serial, serial_counts = paired_operators(PairedOracle, 0)
        parallel, parallel_counts = paired_operators(PairedOracle, 2)
        for expected, actual in zip(serial, parallel):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(parallel_counts, serial_counts)

    def test_lost_worker_hands_remaining_tiles_to_this_process(self):
        # Keep identical query semantics: custom oracle subclasses deliberately
        # retain their overridden query instead of the standard reciprocal API.
        serial, _ = paired_operators(CrashingPairedOracle, 0)
        with tempfile.TemporaryDirectory() as directory:
            flag = str(Path(directory) / 'crash-once')
            recovered, _ = paired_operators(CrashingPairedOracle, 2, flag)
            self.assertFalse(Path(flag).exists())
        for expected, actual in zip(serial, recovered):
            np.testing.assert_array_equal(actual, expected)

    def test_polynomial_workers_publish_shared_bounded_cache_evidence(self):
        serial, _ = paired_operators(PairedOracle, 0, degree=2, count=24)
        parallel, _, evidence = paired_operators(PairedOracle, 2, degree=2, count=24, details=True)
        for expected, actual in zip(serial, parallel):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)
        first, second = (item['worker_moment_cache'] for item in evidence)
        self.assertIs(first, second)  # Each polarization shares the same traversal, not twice the work.
        self.assertGreater(first['stores'], 0)
        self.assertLessEqual(first['peak_accounted_retained_bytes'],
                             first['workers']*first['budget_bytes_per_worker'])

    def test_small_daemon_or_unmarked_work_stays_in_process(self):
        class Operator:
            groups = [np.arange(4)]*4
        marked = type('Marked', (), {'process_tiles': True})()
        with execution_scope(automatic_options()):
            self.assertEqual(tile_processes.prepare(object(), [Operator()]), (0, None))
            self.assertEqual(tile_processes.prepare(marked, [Operator()]), (0, None))
            with mock.patch.object(tile_processes, 'MIN_TILES', 1), \
                    mock.patch.object(tile_processes.mp, 'current_process', return_value=mock.Mock(daemon=True)):
                self.assertEqual(tile_processes.prepare(marked, [Operator()]), (0, None))


if __name__ == '__main__':
    unittest.main()
