"""Analytic mode selection and retained preparation preserve physical fields."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor
from ghost_backend.bor import options
from ghost_backend.bor.tiled import Primitive
from ghost_backend.bor.preparation import numerical_preparation, prepared_surface


class AxialModeTests(unittest.TestCase):
    def test_only_exact_endpoints_select_the_finite_expansion(self):
        self.assertEqual(bor._bor_mode_limits(1000., 1., [0., 180.], None), (1, 1))
        self.assertEqual(bor._bor_mode_limits(1000., 1., [0.], 0), (0, 1))
        for angles in ([1e-12], [180.-1e-12], [0., 90.], []):
            self.assertFalse(bor._all_axial_aspects(angles))

    def test_axial_rhs_is_exactly_zero_outside_first_signed_orders(self):
        surface = bor.BorPecSolver(bor.sphere_generatrix(.05, 8), 1e9)
        for mode in (0, 2, -2, 8):
            np.testing.assert_array_equal(surface.rhs_vv_hh_batch(mode, [0., 180.]), 0.)

    def test_dense_and_compressed_axial_fields_match_full_modal_sweep(self):
        points = bor.sphere_generatrix(.05, 8)
        for backend, impedance in (('dense', None), ('compressed', None), ('dense', 55+8j)):
            common = dict(freq_hz=1e9, thetas_deg=[0., 180.], n_modes=10,
                          workers=1, zs=impedance,
                          bor_options=dict(factorization=backend, near_backend='threads'))
            with self.subTest(backend=backend, impedance=impedance):
                with mock.patch.object(bor, '_all_axial_aspects', return_value=False):
                    expected = bor.solve_bor(points, **common)
                actual = bor.solve_bor(points, **common)
                for field in ('amp_vv', 'amp_hh'):
                    np.testing.assert_allclose(actual[field], expected[field], rtol=3e-11, atol=1e-13)
                self.assertEqual(actual['modes_used'], 1)
                self.assertEqual(actual['mode_selection'], 'exact_axial_first_order')
                self.assertEqual(len(actual['modal_execution']['systems']), 1)
                self.assertTrue(actual['mode_converged'])

    def test_axial_zero_cap_still_fails(self):
        with self.assertRaises(options.ModalConvergenceError):
            bor.solve_bor(bor.sphere_generatrix(.03, 6), 1e9, [0.], n_modes=0,
                          bor_options=dict(factorization='dense'))


class RetainedPreparationTests(unittest.TestCase):
    def test_automatic_tiles_are_route_specific_and_explicit_sizes_are_preserved(self):
        auto = options.validate_options({})
        self.assertEqual(auto['compression_tile'],'auto')
        self.assertEqual(options.resolved_compression_tile(auto,False),32)
        self.assertEqual(options.resolved_compression_tile(auto,True),128)
        for tile in (8,24,32,64,128):
            requested = options.validate_options(dict(compression_tile=tile))
            self.assertEqual(options.resolved_compression_tile(requested,False),tile)
            self.assertEqual(options.resolved_compression_tile(requested,True),tile)

    def test_joint_near_families_preserve_coplanar_mfie_analytic_zero(self):
        gen = bor.Generatrix(np.column_stack((np.linspace(.01,.06,6),np.full(6,.1))))
        reference, order, error = bor._converged_disjoint_blocks(
            gen,0,gen,2,21.,4,('efie',),signed=False)
        with mock.patch.object(bor,'_contract_near_batch',wraps=bor._contract_near_batch) as contracted:
            actual, actual_order, actual_error = bor._converged_disjoint_blocks(
                gen,0,gen,2,21.,4,('efie','mfie'),signed=False)
        np.testing.assert_array_equal(actual['mfie'],0.)
        np.testing.assert_array_equal(actual['efie'],reference['efie'])
        self.assertEqual((actual_order,actual_error),(order,error))
        self.assertTrue(all(call.args[3] == ('efie',) for call in contracted.call_args_list))

    def test_automatic_cap_retry_reuses_context_and_solved_mode_fields(self):
        points = bor.sphere_generatrix(.035,8)
        common = dict(freq_hz=1e9,thetas_deg=[30.,90.],workers=1,
                      bor_options=dict(factorization='dense'))
        reference = bor.solve_bor(points,n_modes=16,**common)
        original = bor._bor_mode_limits
        def initial_small(k,radius,angles,n_modes):
            return (2,1) if n_modes is None else original(k,radius,angles,n_modes)
        with mock.patch.object(bor,'_bor_mode_limits',side_effect=initial_small):
            actual = bor.solve_bor(points,**common)
        self.assertGreater(actual['numerical_preparation']['reused_surface_contexts'],0)
        events = actual['modal_execution']['systems']
        self.assertEqual(len({event['mode'] for event in events}),len(events))
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[field],reference[field],rtol=2e-7,atol=1e-13)

    def test_far_cache_is_optional_when_one_mode_exceeds_available_memory(self):
        with options.option_scope(options.validate_options(dict(factorization='compressed'))), \
                mock.patch.object(bor, 'COMPRESSED_FAR_CACHE_MIN_NODES', 1), \
                mock.patch.object(bor, 'plan_bor_mode_workers', return_value=dict(fits_memory=False)):
            self.assertIsNone(bor.plan_compressed_far_cache(16, 8, 'cfie', False, .01, 2, 6, .1))

    def test_far_cache_reduces_mode_band_before_falling_back(self):
        def admission(n, rhs, workers, tasks, assembly, **kwargs):
            return dict(fits_memory=assembly < .10012+bor.COMPRESSED_FAR_WORK_GB, workers=1)
        with options.option_scope(options.validate_options(dict(factorization='compressed'))), \
                mock.patch.object(bor, 'COMPRESSED_FAR_CACHE_MIN_NODES', 1), \
                mock.patch.object(bor, 'plan_bor_mode_workers', side_effect=admission):
            plan = bor.plan_compressed_far_cache(16, 8, 'cfie', False, .01, 2, 6, .1)
            self.assertIsNotNone(plan)
            self.assertLess(plan[0], 9)
            self.assertLess(plan[1], .00012)

    def test_far_cache_keeps_priced_alignment_when_memory_throttles_workers(self):
        with options.option_scope(options.validate_options(dict(factorization='compressed'))), \
                mock.patch.object(bor, 'COMPRESSED_FAR_CACHE_MIN_NODES', 1), \
                mock.patch.object(bor, 'plan_bor_mode_workers', return_value=dict(fits_memory=True,workers=3)):
            plan = bor.plan_compressed_far_cache(16, 20, 'cfie', False, .00036, 4, 6, .1)
            from ghost_backend.bor.streaming import _aligned_stream_mode_block
            self.assertEqual(_aligned_stream_mode_block(20, plan[0], plan[2]), plan[0])

    def test_far_cache_has_a_separate_bounded_optional_allowance(self):
        with options.option_scope(options.validate_options(dict(factorization='compressed'))), \
                mock.patch.object(bor,'plan_bor_mode_workers',return_value=dict(fits_memory=True,workers=4)):
            plan = bor.plan_compressed_far_cache(1536,26,'cfie',False,12.,4,38,2.)
        self.assertEqual(plan[0],2)
        self.assertEqual(plan[2],2)
        self.assertLessEqual(plan[1],512*1024**2/1e9)
        self.assertEqual(bor.COMPRESSED_FAR_WORK_GB,256*1024**2/1e9)

    def test_frequency_reuse_is_modal_and_preserves_new_operator_checks(self):
        from ghost_backend.execution.options import execution_scope, validate_options
        from ghost_backend.twod.preparation import preparation_scope
        from ghost_backend.compressed import recycling
        points = bor.sphere_generatrix(.035, 8)
        kwargs = dict(thetas_deg=[0., 180.], workers=1, bor_options=dict(factorization='compressed'))
        reference = bor.solve_bor(points, 1.02e9, **kwargs)
        with execution_scope(validate_options(dict(frequency_preconditioner='reuse', ram_budget_gib=2.))), preparation_scope():
            bor.solve_bor(points, 1.e9, **kwargs)
            self.assertGreater(recycling.live_bytes(), 0)
            actual = bor.solve_bor(points, 1.02e9, **kwargs)
            events = actual['modal_execution']['systems']
            self.assertTrue(events[0]['frequency_preconditioner']['reused'])
            self.assertEqual(events[0]['factorizations'], 0)
            self.assertLess(events[0]['max_backward_error'], 1e-10)
        for field in ('amp_vv', 'amp_hh'):
            np.testing.assert_allclose(actual[field], reference[field], rtol=2e-10, atol=1e-13)

    def test_extended_near_bands_preserve_old_storage_and_match_full_cap(self):
        points = bor.sphere_generatrix(.035, 6)
        actual = bor.BorPecSolver(points, 1e9)
        actual._compressed = True
        actual.prepare_operators(1, mfie=True, ibc=True)
        old = {kind: actual._prepared_near(kind, 1)['values'] for kind in ('efie', 'mfie', 'ibc')}
        actual.prepare_operators(3, mfie=True, ibc=True)
        reference = bor.BorPecSolver(points, 1e9)
        reference._compressed = True
        reference.prepare_operators(3, mfie=True, ibc=True)
        self.assertEqual(actual.near_preparation_bands[-1]['first_mode'], 2)
        for kind in old:
            values = actual._prepared_near(kind, 3)['values']
            self.assertTrue(np.shares_memory(values[:, 1], old[kind]))
            for mode in range(4):
                np.testing.assert_allclose(values[:, mode], reference._prepared_near(kind, 3)['values'][:, mode],
                                           rtol=5e-7, atol=1e-13)

    def test_preparation_context_reuses_only_identical_surface_slots(self):
        points = bor.sphere_generatrix(.035, 6)
        with numerical_preparation() as owner:
            owner.begin_attempt()
            first = prepared_surface(points, 1e9)
            owner.begin_attempt()
            self.assertIs(first, prepared_surface(points.copy(), 1e9))
            owner.begin_attempt()
            self.assertIsNot(first, prepared_surface(points, 1.1e9))
            self.assertEqual(owner.evidence()['reused_surface_contexts'], 1)
        self.assertFalse(owner.surfaces)

    def test_exact_streamed_queries_match_direct_coefficients_without_far_sampling(self):
        points = bor.sphere_generatrix(.035, 8)
        with options.option_scope(options.validate_options(dict(factorization='compressed'))):
            direct = bor.BorPecSolver(points, 1e9)
            retained = bor.BorPecSolver(points, 1e9)
            direct.prepare_operators(3, mfie=True, ibc=True)
            retained.prepare_operators(3, mfie=True, ibc=True)
            zs = np.full(retained.g.rho.shape, 45+7j)
            retained.enable_streaming(3, mfie=True, ibc_zs_pt=zs, mode_block=4, workers=1)
            try:
                rows = np.array([0, 2, 4, 6, 8, 9, 11, 13, 15, 17])
                cols = rows[::-1]
                for kind in ('T', 'K', 'IBC'):
                    for mode in (0, 1, -1, 3, -3):
                        weight = zs if kind == 'IBC' else None
                        element_weight = np.full(8, 45+7j) if kind == 'IBC' else None
                        expected = Primitive(direct, kind, mode, 3, weight, element_weight).query(rows, cols)
                        with mock.patch('ghost_backend.bor.kernels.banded_modal_kernels',
                                        side_effect=AssertionError('retained query repeated far sampling')):
                            actual = Primitive(retained, kind, mode, 3, weight, element_weight).query(rows, cols)
                        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-12)
            finally:
                retained.close_streaming()

    def test_streamed_compressed_physical_fields_match_direct_queries(self):
        points = bor.sphere_generatrix(.035, 8)
        kwargs = dict(freq_hz=1e9, thetas_deg=[20., 90., 150.], n_modes=9, workers=1,
                      bor_options=dict(factorization='compressed'))
        expected = bor.solve_bor(points, **kwargs)
        with mock.patch.object(bor, 'COMPRESSED_FAR_CACHE_MIN_NODES', 1):
            actual = bor.solve_bor(points, **kwargs)
            explicit = bor.solve_bor(points, **dict(kwargs,
                bor_options=dict(factorization='compressed',compression_tile=32)))
        for event in expected['modal_execution']['systems']:
            self.assertEqual(event['compression_tile'],32)
        for event in actual['modal_execution']['systems']:
            self.assertEqual(event['compression_tile'],128)
            self.assertEqual(event['compression_tile_requested'],'auto')
        for event in explicit['modal_execution']['systems']:
            self.assertEqual(event['compression_tile'],32)
        for field in ('amp_vv', 'amp_hh'):
            np.testing.assert_allclose(actual[field], expected[field], rtol=3e-9, atol=1e-13)
            np.testing.assert_allclose(explicit[field], expected[field], rtol=3e-9, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
