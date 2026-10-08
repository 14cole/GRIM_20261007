"""Independent review checks for the run-owned compressed resources."""
import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from ghost_backend.compressed.fast_far import FarProposal
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.twod.preparation import preparation_scope, run_resources


class CompressedReviewTests(unittest.TestCase):
    def test_small_thin_layer_tiles_price_the_actual_worker_grid(self):
        from ghost_backend.compressed.worker_pool import forecast_bytes
        from ghost_backend.compressed.tile_processes import WORKER_BYTES
        with patch.dict(os.environ, {'GHOST_TILE_PROCESSES': '2'}), \
                patch('ghost_backend.compressed.worker_pool.retained_bytes', return_value=0):
            # Thin-layer routing can require smaller tiles than the usual 512.
            self.assertEqual(forecast_bytes(3000, 2), 0)
            self.assertEqual(forecast_bytes(3000, 2, groups=32), 2*WORKER_BYTES)

    def test_pilot_identity_includes_quadrature_and_route_truncation(self):
        from ghost_backend.compressed.pilots import key, attach
        from ghost_backend.execution.options import execution_scope, validate_options
        with patch('ghost_backend.twod.assembly.session.system_key', return_value=('same geometry',)):
            def identity(options=None, **kwargs):
                with execution_scope(validate_options(options or {})):
                    return key(None, [], 'TE', 'multi_region', **kwargs)
            baseline = identity()
            for options in ({'assembly_tile': 64}, {'far_quadrature_order': 12}, {'far_grading': False}):
                self.assertNotEqual(baseline, identity(options))
            self.assertNotEqual(baseline, identity(far_order_floors={3+1j: 12}))
            self.assertNotEqual(baseline, identity(cut=None))
            self.assertEqual(baseline, identity(cut=32.))
            oracle = SimpleNamespace(cut=32., far_order_floors=None, n=1)
            with execution_scope(validate_options({})), \
                    patch('ghost_backend.twod.assembly.kernels.mesh_key', return_value=b'same mesh'):
                attach(oracle, None, [], 'TE', 'multi_region')
            self.assertEqual(baseline, oracle.pilot_identity)

    def test_forecast_cache_identity_includes_quadrature_settings(self):
        from ghost_backend.compressed.memory import geometry_storage
        from ghost_backend.execution.options import execution_scope, validate_options
        session = SimpleNamespace(pending=None, checkpoint=lambda: None)
        result = dict(method='test sample', operator_bytes=100, operator_allowance_bytes=120,
                      sampled=True, samples=4)
        with patch('ghost_backend.twod.assembly.session.current_session', return_value=session), \
                patch('ghost_backend.twod.assembly.session.system_key', return_value=('same geometry',)), \
                patch('ghost_backend.compressed.coefficients.NativeOracle', return_value=SimpleNamespace(n=9000)), \
                patch('ghost_backend.compressed.runtime.coordinates', return_value=None), \
                patch('ghost_backend.compressed.memory.sample_operator', return_value=result.copy()) as sample:
            for options in ({}, {}, {'far_quadrature_order': 12}, {'far_quadrature_order': 12}, {}):
                with execution_scope(validate_options(options)):
                    value = geometry_storage(None, [], 'TE', 'te_robin', 1., dofs=9000)
                self.assertEqual(value['operator_bytes'], 100)
            self.assertEqual(sample.call_count, 2)
            self.assertEqual(len(session.memory_storage), 2)

    def test_rejected_cur_retains_the_qualified_qr_compression(self):
        oracle = SimpleNamespace(n=16)
        operator = StreamedOperator(oracle, np.arange(16.)[:, None], tile=8, assemble=False)
        raw = np.ones((8, 8), complex)
        wrong = FarProposal(2*np.ones((8, 1)), np.ones((1, 8)))
        result = operator.compress_tile(0, 1, raw, np.zeros(raw.shape), wrong)
        self.assertFalse(wrong.evidence['accepted'])
        self.assertTrue(result[4])
        self.assertIsNotNone(result[3][1])
        np.testing.assert_allclose(result[3][0] @ result[3][1], raw, rtol=1e-14, atol=1e-14)

    def test_all_resources_close_when_one_close_fails(self):
        closed = []
        def failure():
            closed.append('first')
            raise OSError('cleanup failed')
        with self.assertRaisesRegex(OSError, 'cleanup failed'):
            with preparation_scope():
                run_resources().update(first=SimpleNamespace(close=failure),
                    second=SimpleNamespace(close=lambda: closed.append('second')))
        self.assertEqual(closed, ['first', 'second'])

    def test_cleanup_keeps_original_cancellation_error(self):
        closed = []
        def failure():
            closed.append('first')
            raise OSError('cleanup failed')
        with self.assertRaisesRegex(InterruptedError, 'requested cancellation'):
            with preparation_scope():
                run_resources().update(first=SimpleNamespace(close=failure),
                    second=SimpleNamespace(close=lambda: closed.append('second')))
                raise InterruptedError('requested cancellation')
        self.assertEqual(closed, ['first', 'second'])

    def test_reused_workers_refresh_physics_and_close_at_run_exit(self):
        from test_compact_multi_region import prepared
        from ghost_backend.compressed import tile_processes
        from ghost_backend.compressed.regional_coefficients import PairedOracle
        from ghost_backend.compressed.polarization_cache import build_pair
        from ghost_backend.execution.options import execution_scope, validate_options
        from ghost_backend.execution.cpu import CPUState, _STATE
        from ghost_backend.twod.formulations.regions import dof_coordinates
        options = validate_options(dict(assembly_threads=2, blas_threads=1, ram_budget_gib=4.))
        retained = None
        with preparation_scope(), execution_scope(options), _STATE.override(CPUState()), \
                patch.object(tile_processes, 'MIN_TILES', 1), tempfile.TemporaryDirectory() as directory:
            for multiplier in (1., 1.2):
                mesh, infos, _ = prepared('mixed', 'TE', 80)
                for info in infos:
                    info.k_plus *= multiplier
                    info.k_minus *= multiplier
                arrays = []
                for worker_count in (0, 2):
                    oracle = PairedOracle(mesh, infos, infos, cut=32)
                    xy = dof_coordinates(mesh, oracle.oracles[0].layout)
                    with patch.dict(os.environ, {'GHOST_TILE_PROCESSES': str(worker_count)}):
                        operators = build_pair(oracle, xy, tile=64, budget=2**28, spool_directory=directory)
                    operators[1].load()
                    identity = np.eye(oracle.n, dtype=complex)
                    arrays.append([o.matmul(identity) for o in operators] + [o.row_error for o in operators])
                for serial, parallel in zip(*arrays):
                    np.testing.assert_array_equal(serial, parallel)
                pool = run_resources()['compressed_tile_workers']
                if retained is not None:
                    self.assertIs(pool, retained)
                    self.assertEqual(set(pool.executor._processes), process_ids)
                retained = pool
                process_ids = set(pool.executor._processes)
                self.assertFalse(pool.closed)
            self.assertEqual(pool.generation, 2)
            worker_directory = Path(pool.directory.name)
            processes = list(pool.executor._processes.values())
        self.assertTrue(pool.closed)
        self.assertFalse(worker_directory.exists())
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_lost_persistent_worker_closes_and_next_assembly_recreates_pool(self):
        from test_tile_processes import paired_operators, CrashingPairedOracle
        from ghost_backend.compressed.regional_coefficients import PairedOracle
        with preparation_scope(), tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'GHOST_COMPRESSED_RECIPROCAL': 'off'}):
            recovered, _ = paired_operators(CrashingPairedOracle, 2,
                                            str(Path(directory)/'crash-once'))
            broken = run_resources()['compressed_tile_workers']
            self.assertTrue(broken.closed)
            self.assertFalse(Path(broken.directory.name).exists())
            regular, _ = paired_operators(PairedOracle, 2)
            replacement = run_resources()['compressed_tile_workers']
            self.assertIsNot(broken, replacement)
            self.assertFalse(replacement.closed)
            for left, right in zip(recovered, regular):
                np.testing.assert_array_equal(left, right)
        self.assertTrue(replacement.closed)


if __name__ == '__main__':
    unittest.main()
