"""BoR admission and compact caches preserve the signed complex operators."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import dispatch, solver as bor
from ghost_backend.bor.near_storage import mode_blocks
from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot
from ghost_backend.runs.bor_setup import resource_summary


CYLINDER = '''title: thin coated cylinder
segment: outer
properties: 3 -20 0 1 0
5 5 5.02 5
5.02 5 5.02 0
5.02 0 5 0
segment: covered
properties: 4 -20 0 1 0
5 5 5 0
segment: top
properties: 2 -20 0 0 0
0 5 5 5
segment: bottom
properties: 2 -20 0 0 0
5 0 0 0
dielectrics:
1 101.19 -11.88 3.45 -3.47
'''


class NearStorageTests(unittest.TestCase):
    def test_self_cache_preserves_independently_integrated_signed_modes(self):
        solver = bor.BorPecSolver(bor.sphere_generatrix(.025, 6), 1e9,
            near_depth=1, medium=(2.5-.3j, 1.2-.1j))
        pairs = [(1, 1), (1, 2), (2, 1)]
        mm = 3
        kinds = ('efie', 'mfie', 'ibc')
        for kind in kinds:
            solver._prepare_near_contractions(kind, pairs, mm, workers=2)
        for kind in kinds:
            originals = []
            for e, f in pairs:
                # The same rule the solver applies to this pair and kernel kind.
                points = bor._same_surface_points(solver.gen, e, f, (kind,), 1)
                original = bor._contract_near_points(solver.gen, e, solver.gen, f,
                    solver.k, mm, (kind,), points)
                originals.append(original[kind])
            record = solver._near_contractions[(kind, mm)]
            values = record['values']
            self.assertEqual(values.shape[:2], (4, mm+1))
            self.assertLess(values.shape[2], 4*len(pairs))
            weights = np.linspace(1., 2., solver.gen.n_elems) * (1.+.2j)
            for m in range(-mm, mm+1):
                actual = np.zeros((4, solver.Nn, solver.Nn), complex)
                expected = np.zeros_like(actual)
                weight = weights[record['source_elems']] if kind == 'ibc' else 1.
                for uv, component in enumerate(mode_blocks(values, m)):
                    np.add.at(actual[uv], (record['rows'], record['cols']), component*weight)
                for (e, f), block in zip(pairs, originals):
                    weight = weights[f] if kind == 'ibc' else 1.
                    for uv in range(4):
                        expected[uv][np.ix_([e,e+1],[f,f+1])] += block[uv,m+mm]*weight
                np.testing.assert_allclose(actual, expected,
                    rtol=2e-13, atol=2e-13*np.max(abs(expected)))

    def test_cross_cache_owns_only_half_and_preserves_both_signs(self):
        sp = bor.BorPecSolver(np.array([[.025, .01], [.025, 0.]]), 1e9,
            medium=(2.5-.3j, 1.2-.1j))
        sq = bor.BorPecSolver(np.array([[.024, .01], [.024, 0.]]), 1e9,
            medium=(2.5-.3j, 1.2-.1j))
        cross = bor.BorCrossOperators(sp, sq)
        mm = 3
        original, refinement = cross._integrate_near(0, 0, mm)
        cross._store_near(0, 0, mm, original, refinement)
        stored = cross._near_data(0, 0, mm)
        for kind, values in stored.items():
            self.assertTrue(values.flags.owndata)
            self.assertFalse(np.shares_memory(values, original[kind]))
            self.assertEqual(values.nbytes * (2*mm+1), original[kind].nbytes * (mm+1))
            for m in range(-mm, mm+1):
                np.testing.assert_allclose(mode_blocks(values, m), original[kind][:, m+mm],
                    rtol=2e-13, atol=2e-13*np.max(abs(original[kind])))


class MemoryPlanningTests(unittest.TestCase):
    def test_preparation_workers_are_independent_but_price_their_own_phase(self):
        with mock.patch('ghost_backend.execution.options.allocated_cpu_budget',return_value=4), \
                mock.patch.object(bor,'estimate_bor_dense_peak_gb',return_value=.2), \
                mock.patch('ghost_backend.bor.near_parallel.process_backend_possible',return_value=False):
            plan = bor.plan_bor_mode_workers(10,4,2,1,.2,memory_limit_gb=2.,preparation_workers=4)
        self.assertEqual(plan['workers'],1)
        self.assertEqual(plan['near_preparation']['workers'],4)
        self.assertEqual(plan['requested_preparation_workers'],4)
        expected = max(bor.estimate_bor_total_peak_gb(.2,.2),
            bor.estimate_bor_total_peak_gb(.2+4*bor._NEAR_TASK_SCRATCH_BYTES/1e9,0.))
        self.assertEqual(plan['estimated_peak_gb'],expected)

    def test_preparation_workers_obey_cpu_and_memory_even_with_one_modal_task(self):
        with mock.patch.object(bor,'estimate_bor_dense_peak_gb',return_value=.2), \
                mock.patch('ghost_backend.bor.near_parallel.process_backend_possible',return_value=False):
            for cpu,limit,expected in ((4,1.3,2),(2,2.,2)):
                with mock.patch('ghost_backend.execution.options.allocated_cpu_budget',return_value=cpu):
                    plan = bor.plan_bor_mode_workers(10,4,2,1,.2,memory_limit_gb=limit,preparation_workers=4)
                self.assertEqual(plan['workers'],1)
                self.assertEqual(plan['near_preparation']['workers'],expected)
                self.assertLessEqual(plan['estimated_peak_gb'],limit)
            with mock.patch('ghost_backend.execution.options.allocated_cpu_budget',return_value=4):
                rejected = bor.plan_bor_mode_workers(10,4,2,1,.2,memory_limit_gb=.9,preparation_workers=4)
            self.assertFalse(rejected['fits_memory'])

    def test_axial_streamed_solve_prepares_with_four_workers_but_solves_one_mode(self):
        points = bor.sphere_generatrix(.025,8)
        kwargs = dict(freq_hz=1e9,thetas_deg=[0.,180.],assembly='streaming',
                      bor_options=dict(factorization='dense',near_backend='threads'))
        reference = bor.solve_bor(points,workers=1,**kwargs)
        with mock.patch('ghost_backend.execution.options.allocated_cpu_budget',return_value=4), \
                mock.patch.object(bor,'_solve_memory_limit_gb',return_value=4.):
            result = bor.solve_bor(points,workers=4,**kwargs)
        plan = result['modal_execution']['worker_plan']
        self.assertEqual(plan['workers'],1)
        self.assertEqual(plan['near_preparation']['workers'],4)
        self.assertEqual(result['stream_mode_block'],2)
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(result[field],reference[field],rtol=3e-12,atol=1e-13)

    def test_runtime_reduces_concurrency_and_keeps_original_fields(self):
        # Controlled workspace prices exercise real admission and execution,
        # without allocating a large physical matrix just to force throttling.
        def linear_price(n, rhs, workers=1, mode_tasks=None):
            return .3 * min(workers, mode_tasks or workers)
        request = dict(n_dofs=2, thetas=[20., 90.], pols=['VV', 'HH'],
            m_max=3, mode_tol=1e-6, assembly_peak_gb=.2,
            assemble=lambda m: (np.eye(2, dtype=complex), None),
            rhs=lambda m, th, pol: np.ones(2, complex),
            farfield=lambda m, x, th, pol: complex(x[0])/(abs(m)+1))
        with mock.patch.object(bor, '_solve_memory_limit_gb', return_value=1.6), \
                mock.patch.object(bor, 'estimate_bor_dense_peak_gb', side_effect=linear_price):
            expected, _, _ = bor._mode_sweep(**request, workers=1)
            actual, _, stats = bor._mode_sweep(**request, workers=4)
        np.testing.assert_array_equal(actual, expected)
        plan = stats['modal_execution']['worker_plan']
        self.assertEqual(plan['requested_workers'], 4)
        # Mode phase: 0.5 + 1.2 * (0.2 + 0.3 * workers) <= 1.6 admits two; the
        # near scratch belongs to the earlier preparation phase, not on top.
        self.assertEqual(plan['workers'], 2)
        self.assertTrue(plan['fits_memory'])
        self.assertLessEqual(plan['estimated_peak_gb'], 1.6)

    def test_unfit_single_worker_is_rejected_before_preparation(self):
        prepare = mock.Mock()
        with mock.patch.object(bor, '_solve_memory_limit_gb', return_value=.6):
            with self.assertRaises(MemoryError):
                bor._mode_sweep(1, [90.], ['VV'], 0, 1e-6,
                    mock.Mock(), mock.Mock(), mock.Mock(), prepare=prepare, workers=15)
        prepare.assert_not_called()

    def test_throttled_waves_do_not_cross_streamed_cache_boundaries(self):
        import threading
        import time
        submitted, running, violations, peak = [], set(), [], [0]
        lock = threading.Lock()
        original_submit = bor.ThreadPoolExecutor.submit
        def record(executor, function, mode, *args):
            submitted.append(mode)
            def tracked(*call_args):
                with lock:
                    # A mode may only start while every running mode shares
                    # its four-mode stream range (the resident cache block).
                    violations.extend(
                        (other, mode) for other in running if other // 4 != mode // 4
                    )
                    running.add(mode)
                    peak[0] = max(peak[0], len(running))
                try:
                    return function(*call_args)
                finally:
                    with lock:
                        running.discard(mode)
            return original_submit(executor, tracked, mode, *args)
        def price(n, rhs, workers=1, mode_tasks=None):
            return .3 * min(workers, mode_tasks or workers)
        def assemble(m):
            # Slow first mode of each block: the sliding window then holds
            # later modes of the block in flight next to it.
            if m % 4 == 0:
                time.sleep(0.2)
            return np.ones((1, 1), complex), None
        # 0.5 + 1.2 * (0.2 + 0.3 * workers): three mode workers fit 2.0 GB, four
        # do not, so the four-mode stream ranges are straddled by the window.
        with mock.patch.object(bor, '_solve_memory_limit_gb', return_value=2.0), \
                mock.patch.object(bor, 'estimate_bor_dense_peak_gb', side_effect=price), \
                mock.patch.object(bor.ThreadPoolExecutor, 'submit', new=record):
            _, _, stats = bor._mode_sweep(1, [90.], ['VV'], 7, 1e-6,
                assemble,
                lambda *args: np.ones(1, complex), lambda *args: 1.+0j,
                workers=4, assembly_peak_gb=.2, stream_mode_block=4)
        self.assertEqual(stats['modal_execution']['worker_plan']['workers'], 3)
        self.assertEqual(submitted, list(range(8)))
        self.assertEqual(violations, [])
        self.assertGreaterEqual(peak[0], 2)
        self.assertLessEqual(peak[0], 3)

    def test_mode_workers_inherit_the_callers_execution_options(self):
        # Executor threads do not inherit context variables; each mode runs in
        # a copy of the caller's context (CPU allocation, residual storage...).
        from ghost_backend.execution.options import execution_scope, option
        import threading
        seen, threads = [], set()
        def assemble(m):
            seen.append(option('dense_residual_storage', 'unset'))
            threads.add(threading.get_ident())
            return np.eye(2, dtype=complex), None
        with execution_scope(dict(dense_residual_storage='memory')):
            bor._mode_sweep(2, [20., 90.], ['VV', 'HH'], 5, 1e-6, assemble,
                lambda m, th, pol: np.ones(2, complex),
                lambda m, x, th, pol: complex(x[0])/(abs(m)+1), workers=3)
        self.assertNotIn(threading.get_ident(), threads)
        self.assertEqual(set(seen), {'memory'})

    def test_user_cylinder_refined_5ghz_fits_by_compressing_and_throttling(self):
        snapshot = build_geometry_snapshot(*parse_geometry(CYLINDER))
        with mock.patch.object(bor, '_solve_memory_limit_gb', return_value=14.5), \
                mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=14.5), \
                mock.patch.object(bor.BorPecSolver, 'prepare_operators', side_effect=AssertionError('assembly')):
            estimate = dispatch.estimate_bor_resources(snapshot, 5., [0.,45.,90.,180.],
                workers=15, mesh_certification=True, fine_factor=1.5,
                bor_options={'factorization':'auto'})
        self.assertEqual(estimate['mesh_elements'], 5688)
        self.assertEqual(estimate['n_unknowns_estimate'], 14248)
        self.assertEqual(estimate['assembly_estimate'], 'compressed')
        self.assertEqual(estimate['active_mode_workers'], 1)
        self.assertLessEqual(estimate['estimated_peak_gb'], 14.5)
        self.assertTrue(estimate['mesh_certification'])

    def test_auto_preview_preserves_certification_and_fine_factor(self):
        with mock.patch.object(dispatch, 'estimate_bor_resources', return_value={'estimated_peak_gb':1.}) as price:
            dispatch.resolve_automatic_factorization(dict(geometry_snapshot={},
                frequency_ghz=5., aspects_deg=[90.], mesh_certification=True, fine_factor=2.))
        self.assertTrue(price.call_args.kwargs['mesh_certification'])
        self.assertEqual(price.call_args.kwargs['fine_factor'], 2.)

    def test_desktop_preview_uses_one_backend_for_the_whole_sweep(self):
        snapshot = build_geometry_snapshot(*parse_geometry(CYLINDER))
        value = dict(schema='grim.bor-run-setup', version=1, frequencies_ghz=[1.,5.],
            aspects_deg=[0.,90.], units='inches', mesh_certification=True,
            accuracy='standard', cfie_alpha=.5, bor_options={})
        real = dispatch.estimate_bor_resources
        with mock.patch.object(dispatch, 'estimate_bor_resources', wraps=real) as estimates:
            text = resource_summary(snapshot, None, value)
        self.assertIn('auto \u2192 compressed', text)
        self.assertIn('simultaneous mode workers', text)
        final_calls = estimates.call_args_list[-2:]
        self.assertEqual([call.kwargs['bor_options']['factorization'] for call in final_calls],
                         ['compressed', 'compressed'])



class ModalFactorResidualStorageTests(unittest.TestCase):
    def test_owned_system_is_factored_in_place_with_spooled_residuals(self):
        from ghost_backend.bor.factor import ModalFactor
        from ghost_backend.execution.options import execution_scope
        rng = np.random.default_rng(7)
        for order in ('C', 'F'):
            a = np.array(rng.normal(size=(120, 120)) + 1j*rng.normal(size=(120, 120))
                         + 25*np.eye(120), order=order)
            original = a.copy()
            rhs = rng.normal(size=(120, 3)) + 1j*rng.normal(size=(120, 3))
            with execution_scope(dict(dense_residual_storage='disk')):
                factor = ModalFactor(a, 2, True, owned=True)
                try:
                    # One dense matrix per mode: the LU overwrote the system.
                    self.assertTrue(np.shares_memory(factor.lu, a))
                    self.assertEqual(factor.event['residual_storage'], 'disk')
                    solution = factor.solve(rhs)
                finally:
                    factor.close()
            np.testing.assert_allclose(original @ solution, rhs, rtol=1e-12, atol=1e-12)
            exact = np.linalg.cond(original, 1)
            self.assertTrue(0.3*exact <= factor.condition <= 1.0000001*exact)
        # A borrowed matrix is never overwritten.
        b = np.array(original)
        with execution_scope(dict(dense_residual_storage='disk')):
            factor = ModalFactor(b, 2, False)
        self.assertFalse(np.shares_memory(factor.lu, b))
        np.testing.assert_array_equal(b, original)

    def test_spooled_residuals_reproduce_the_in_memory_solve(self):
        from ghost_backend.execution.options import execution_scope
        points = bor.sphere_generatrix(.05, 16)
        common = dict(formulation='cfie', workers=2, assembly='tables')
        results = {}
        for policy in ('memory', 'disk'):
            with execution_scope(dict(dense_residual_storage=policy)):
                results[policy] = bor.solve_bor(points, 1.5e9, [0., 60., 180.], **common)
        self.assertTrue(all(event.get('residual_storage') == 'disk'
                            for event in results['disk']['modal_execution']['systems']))
        for key in ('amp_vv', 'amp_hh'):
            np.testing.assert_allclose(results['disk'][key], results['memory'][key],
                                       rtol=1e-12, atol=1e-15)

if __name__ == '__main__':
    unittest.main()
