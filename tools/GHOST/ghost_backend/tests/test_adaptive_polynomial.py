"""Production h/p acceptance, resource planning and explicit-override behavior."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.tests.general_fixtures import fixture
from ghost_backend.twod import solver as s
from ghost_backend.execution.options import validate_options, execution_scope, efficient_defaults
from ghost_backend.execution.selection import select_backend


def wavelength_fixture(case='rectangle'):
    counts = dict(rectangle=4, reentrant=8, acute=4, gap=8, dielectric=4, mixed=8, sheet=2)
    result = fixture(case, counts[case])
    for segment in result['segments']: segment['properties'][1] = '-100'
    return result


class AdaptivePolynomialTests(unittest.TestCase):
    def setUp(self):
        # Exercise the numerical controller cheaply; the production small-model
        # policy is checked separately below and in a real HPC worker.
        small = patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS', 0)
        small.start()
        self.addCleanup(small.stop)

    def test_small_model_keeps_reference_method_automatically(self):
        with patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS', 1024):
            result = s.solve_monostatic_rcs_2d_certified(wavelength_fixture(), [1.], [0., 90.],
                geometry_units='meters', execution_options=dict(mesh_strategy='adaptive',factorization='adaptive'))
        self.assertFalse(result['metadata']['adaptive_mesh']['used'])
        self.assertIn('Small reference mesh', result['metadata']['adaptive_mesh']['reason'])
        self.assertEqual(result['metadata']['polynomial_degree'], 1)

    def test_default_and_invalid_degree(self):
        self.assertEqual(efficient_defaults()['mesh_strategy'], 'adaptive')
        for degree in (0, 4, True, 2., '2'):
            with self.assertRaises(ValueError): validate_options(dict(basis_order=degree))

    def test_candidate_reduces_unknowns_and_preserves_input(self):
        snapshot = wavelength_fixture()
        before = copy.deepcopy(snapshot)
        args = dict(geometry_snapshot=snapshot, frequencies_ghz=[1.], elevations_deg=[0., 47., 91.],
                    geometry_units='meters')
        adaptive = s.solve_monostatic_rcs_2d_certified(**args,
            execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        reference = s.solve_monostatic_rcs_2d_certified(**args,
            execution_options=dict(mesh_strategy='global', factorization='dense'))
        self.assertEqual(snapshot, before)
        self.assertTrue(adaptive['metadata']['mesh_convergence_certified'])
        self.assertTrue(adaptive['metadata']['adaptive_mesh']['used'])
        self.assertEqual(adaptive['metadata']['polynomial_degree'], 3)
        self.assertEqual(adaptive['metadata']['execution_options']['basis_order'], 3)
        self.assertLessEqual(adaptive['metadata']['adaptive_mesh']['acceptance_policy']['complex_max_limit'], .002)
        self.assertLess(adaptive['metadata']['linear_node_count'], reference['metadata']['linear_node_count'])
        for pol in ('VV','HH'):
            def fields(result):
                return np.array([complex(row['rcs_amp_real'],row['rcs_amp_imag']) for row in result['co_solved_samples'][pol]])
            error = np.max(abs(fields(adaptive)-fields(reference)))/np.max(abs(fields(reference)))
            self.assertLess(error, .002)

    def test_planner_has_same_polynomial_mesh_as_execution(self):
        snapshot = wavelength_fixture('dielectric')
        args = dict(geometry_snapshot=snapshot, frequencies_ghz=[1.], elevations_deg=[0., 61.], geometry_units='meters')
        options = validate_options(dict(mesh_strategy='adaptive', factorization='adaptive'))
        plan = select_backend(args, options, certified=True)
        self.assertEqual({record['polynomial_degree'] for record in plan['meshes']}, {2,3})
        result = s.solve_monostatic_rcs_2d_certified(**args, execution_options=options)
        steps = result['metadata']['adaptive_mesh']['steps']
        self.assertTrue(steps)
        self.assertEqual(steps[0]['panels'], plan['meshes'][0]['panels'])
        self.assertTrue(all(step['backend_selection'] for step in steps))

    def test_admitted_whole_request_backend_preference_is_retained(self):
        def prefer_compressed(selection, key, batch=False, entries=None):
            self.assertIn('compressed', selection['candidates'])
            return dict(selection, selected='compressed', retry_order=['dense'])
        with patch('ghost_backend.execution.timing_history.adjust', prefer_compressed):
            result = s.solve_monostatic_rcs_2d_certified(wavelength_fixture(), [1.], [0.,47.],
                geometry_units='meters', execution_options=dict(mesh_strategy='adaptive',factorization='adaptive'))
        self.assertTrue(result['metadata']['adaptive_mesh']['used'])
        self.assertTrue(all(step['backend']=='compressed' for step in result['metadata']['adaptive_mesh']['steps']))

    def test_frequency_sweep_forecasts_each_side_of_size_crossover(self):
        args = dict(geometry_snapshot=wavelength_fixture(), frequencies_ghz=[.5,1.],
                    elevations_deg=[0.,47.], geometry_units='meters')
        options = validate_options(dict(mesh_strategy='adaptive', factorization='adaptive'))
        with patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS',100):
            plan = select_backend(args, options, certified=True)
            result = s.solve_monostatic_rcs_2d_certified(**args, execution_options=options)
        self.assertEqual({m['polynomial_degree'] for m in plan['meshes'] if m['frequency_ghz']==.5},{1})
        self.assertEqual({m['polynomial_degree'] for m in plan['meshes'] if m['frequency_ghz']==1.},{2,3})
        self.assertEqual([r['metadata']['polynomial_degree'] for r in result['metadata']['frequency_metadata']],[1,3])
        self.assertEqual(result['metadata']['polynomial_degree_min'],1)
        self.assertEqual(result['metadata']['polynomial_degree_max'],3)

    def test_failed_comparison_refines_and_then_uses_reference(self):
        snapshot = wavelength_fixture()
        real_finish = s._finish_certified_2d_pair
        seen = []
        def finish(base, fine, policy):
            from ghost_backend.execution.options import option
            if option('basis_order',1) > 1:
                seen.append(fine['metadata']['panel_count'])
                raise ValueError('Certified 2-D mesh convergence failed: injected unresolved field error')
            return real_finish(base, fine, policy)
        with patch.object(s, '_finish_certified_2d_pair', finish), \
                patch('ghost_backend.twod.adaptive_geometry.initial_coarsening', return_value=4.):
            result = s.solve_monostatic_rcs_2d_certified(snapshot, [1.], [0.,67.], geometry_units='meters',
                execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(b > a for a,b in zip(seen,seen[1:])))
        self.assertTrue(result['metadata']['adaptive_mesh']['fallback'])
        self.assertEqual(result['metadata']['polynomial_degree'], 1)
        self.assertTrue(result['metadata']['mesh_convergence_certified'])

    def test_fixed_panel_override_is_preserved(self):
        result = s.solve_monostatic_rcs_2d_certified(fixture('rectangle',24),[1.],[0.,90.], geometry_units='meters',
            execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        self.assertFalse(result['metadata']['adaptive_mesh']['used'])
        self.assertEqual(result['metadata']['polynomial_degree'],1)

    def test_aggressive_policy_requires_resolved_wavelength_requests(self):
        from ghost_backend.twod.adaptive_geometry import initial_coarsening, predicted_hp_size
        from ghost_backend.twod.constants import DEFAULT_PANELS_PER_WAVELENGTH
        shape = wavelength_fixture()
        for density in (0, -20, -100):
            shape['segments'][0]['properties'][1] = str(density)
            expected = 8. if density or DEFAULT_PANELS_PER_WAVELENGTH >= 20 else 4.
            self.assertEqual(initial_coarsening(shape), expected)
        for density in (-1, -4, -8, -19, 24):
            shape['segments'][0]['properties'][1] = str(density)
            self.assertEqual(initial_coarsening(shape), 4.)
        other = copy.deepcopy(shape['segments'][0])
        shape['segments'][0]['properties'][1] = '-20'
        shape['segments'].append(other)  # Explicit counts are left alone.
        self.assertEqual(initial_coarsening(shape), 8.)
        other['properties'][1] = '-8'
        self.assertEqual(initial_coarsening(shape), 4.)
        self.assertEqual(predicted_hp_size([(80,False),(9,True)],8.), (89,19))

    def test_aggressive_convergence_rejection_retries_conservative_controller_once(self):
        from ghost_backend.twod import adaptivity
        from ghost_backend.execution.options import option
        calls = []
        comparisons = []
        def low_level(**kwargs):
            calls.append((kwargs['geometry_snapshot']['_2d_hp_coarsening'], option('basis_order')))
            return {'metadata': {'panel_count': 16, 'linear_node_count': 48}}
        def finish(base, fine, policy):
            comparisons.append(policy)
            if len(comparisons) <= 3:
                raise ValueError('Certified 2-D mesh convergence failed: injected unresolved field error')
            return fine
        options = validate_options(dict(mesh_strategy='adaptive', factorization='dense'))
        with execution_scope(options), patch.object(s, '_finish_certified_2d_pair', finish):
            result = adaptivity.run_certified(low_level, wavelength_fixture(),
                dict(frequencies_ghz=[1.], elevations_deg=[0.], geometry_units='meters'), None, None)
        self.assertEqual([degree for _,degree in calls], [2,3,3,3,2,3])
        np.testing.assert_allclose([coarse for coarse,_ in calls], [8.,8.,8/1.5,8/1.5**2,4.,4.])
        evidence = result['metadata']['adaptive_mesh']
        self.assertEqual(len(evidence['steps']), 6)
        np.testing.assert_allclose([step['coarsening'] for step in evidence['steps']],
                                   [coarse for coarse,_ in calls])
        self.assertFalse(evidence['fallback'])
        self.assertEqual(evidence['conservative_retry']['completed_steps'], 4)
        self.assertGreaterEqual(evidence['elapsed_seconds'], evidence['conservative_retry']['seconds'])
        self.assertTrue(all(policy['complex_max_limit'] <= .002 for policy in comparisons))

    def test_ineligible_conservative_retry_readmits_the_reference_pair(self):
        from ghost_backend.twod import adaptivity
        from ghost_backend.execution.options import option
        options = validate_options(dict(mesh_strategy='adaptive', factorization='dense'))
        def reference(*args):
            self.assertEqual(option('factorization'), 'compressed')
            self.assertEqual(option('mesh_strategy'), 'global')
            return {'metadata': {}}
        low_level = unittest.mock.Mock(side_effect=ValueError('Quality gate failed: injected condition rejection'))
        with execution_scope(options), \
                patch('ghost_backend.twod.adaptive_geometry.eligible_snapshot',
                      side_effect=[(True,''),(False,'Drawn primitives limit coarsening')]), \
                patch.object(adaptivity, 'automatic_backend_requested', return_value=True), \
                patch('ghost_backend.execution.selection.select_backend',
                      side_effect=[{'selected':'dense','retry_order':[]},
                                   {'selected':'compressed','retry_order':[]}]) as select, \
                patch.object(s, '_run_certified_2d_pair_impl', side_effect=reference) as solve_reference:
            result = adaptivity.run_certified(low_level, wavelength_fixture(),
                dict(frequencies_ghz=[1.], elevations_deg=[0.], geometry_units='meters'), None, None)
        self.assertEqual(low_level.call_count, 1)
        self.assertEqual(solve_reference.call_count, 1)
        self.assertTrue(select.call_args.kwargs['certified'])
        evidence = result['metadata']['adaptive_mesh']
        self.assertFalse(evidence['used'])
        self.assertTrue(evidence['fallback'])
        self.assertEqual(evidence['final_backend'], 'compressed')
        self.assertEqual(evidence['conservative_retry']['to_coarsening'], 4.)

    def test_cancellation_and_unrelated_errors_do_not_trigger_conservative_retry(self):
        from ghost_backend.twod import adaptivity
        options = validate_options(dict(mesh_strategy='adaptive', factorization='dense'))
        for failure in (InterruptedError('cancelled'), ValueError('unrelated invalid input')):
            with execution_scope(options):
                with self.assertRaises(type(failure)), patch.object(s, '_finish_certified_2d_pair') as finish:
                    low_level = unittest.mock.Mock(side_effect=failure)
                    adaptivity.run_certified(low_level, wavelength_fixture(),
                        dict(frequencies_ghz=[1.], elevations_deg=[0.], geometry_units='meters'), None, None)
                self.assertEqual(low_level.call_count, 1)
                finish.assert_not_called()

    def test_known_numerical_failures_retry_only_the_aggressive_candidate(self):
        from ghost_backend.twod import adaptivity
        from ghost_backend.execution.errors import BackendNumericalError
        options = validate_options(dict(mesh_strategy='adaptive', factorization='dense'))
        for failure_type in (BackendNumericalError, np.linalg.LinAlgError):
            calls = []
            def low_level(**kwargs):
                coarsening = kwargs['geometry_snapshot']['_2d_hp_coarsening']
                calls.append(coarsening)
                if coarsening > 4.:
                    raise failure_type('injected numerical rejection')
                return {'metadata': {'panel_count': 16, 'linear_node_count': 48}}
            with execution_scope(options), \
                    patch.object(s, '_finish_certified_2d_pair', side_effect=lambda base,fine,policy: fine):
                result = adaptivity.run_certified(low_level, wavelength_fixture(),
                    dict(frequencies_ghz=[1.], elevations_deg=[0.], geometry_units='meters'), None, None)
            self.assertEqual(calls, [8.,4.,4.])
            self.assertIn('numerical failure', result['metadata']['adaptive_mesh']['conservative_retry']['reason'])
            with execution_scope(options), \
                    patch('ghost_backend.twod.adaptive_geometry.initial_coarsening', return_value=4.), \
                    patch.object(s, '_finish_certified_2d_pair') as finish:
                failing = unittest.mock.Mock(side_effect=failure_type('original failure'))
                with self.assertRaises(failure_type):
                    adaptivity.run_certified(failing, wavelength_fixture(),
                        dict(frequencies_ghz=[1.], elevations_deg=[0.], geometry_units='meters'), None, None)
                self.assertEqual(failing.call_count, 1)
                finish.assert_not_called()

    def test_mixed_high_frequency_keeps_previously_successful_conservative_path(self):
        shape = wavelength_fixture('mixed')
        for segment in shape['segments']: segment['properties'][1] = '-20'
        args = dict(geometry_snapshot=shape, frequencies_ghz=[30.],
            elevations_deg=np.arange(0.,360.,20.), geometry_units='meters', solver_method='experimental_cpu',
            execution_options=dict(mesh_strategy='adaptive', factorization='dense', assembly_threads=1, blas_threads=1))
        with patch('ghost_backend.twod.adaptive_geometry.initial_coarsening', return_value=4.):
            reference = s.solve_monostatic_rcs_2d_certified(**args)
        result = s.solve_monostatic_rcs_2d_certified(**args)
        evidence = result['metadata']['adaptive_mesh']
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertTrue(result['metadata']['quality_gate']['passed'])
        self.assertFalse(evidence['fallback'])
        self.assertEqual(evidence['conservative_retry']['to_coarsening'], 4.)
        self.assertIn('Quality gate failed:', evidence['conservative_retry']['reason'])
        for pol in ('VV','HH'):
            field = lambda r: np.array([complex(x['rcs_amp_real'],x['rcs_amp_imag']) for x in r['co_solved_samples'][pol]])
            np.testing.assert_allclose(field(result),field(reference),rtol=2e-10,atol=2e-12)


if __name__ == '__main__': unittest.main()
