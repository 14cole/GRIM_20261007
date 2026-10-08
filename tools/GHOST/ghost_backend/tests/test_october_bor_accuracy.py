"""Independent compression probes and optional same-mesh quadrature checks."""
import pickle
import unittest
import numpy as np
from ghost_backend.bor import compressed_far as cf
from ghost_backend.bor.options import validate_options, option_scope, configured, output_reserved_gb, estimate_output_gb
from ghost_backend.bor.quadrature import compare_fields, checked_solve
from ghost_backend.bor.near_parallel import NearTask


class AccuracyChecks(unittest.TestCase):
    def test_localized_row_is_not_dropped_by_cross_approximation(self):
        values = np.zeros((4, 1, 64, 64), complex)
        values[:, :, 0, :] = 1
        class Tiles:
            def block(self, family, I, J, near_free):
                return values[:, :, I[0]:I[1], J[0]:J[1]]
        u, v = cf._cross(Tiles(), 'efie', (0,64), (0,64), np.ones(1), 1e-10,
                         np.random.default_rng(0))
        expected = values.transpose(2,0,1,3).reshape(64,-1)
        np.testing.assert_allclose(u @ v, expected, atol=1e-12)

    def test_comparison_rejects_local_field_change_and_changed_grid(self):
        angles = np.arange(10.)
        original = {('VV',1.): (angles, np.ones(10, complex))}
        changed = np.ones(10, complex); changed[-1] += .01
        self.assertFalse(compare_fields(original, {('VV',1.): (angles, changed)})['passed'])
        with self.assertRaises(RuntimeError):
            compare_fields(original, {('VV',1.): (angles[::-1], changed)})

    def test_checked_solve_returns_refined_fields_and_does_not_recurse(self):
        calls = []
        def solve(*args, bor_options, **kwargs):
            calls.append(bor_options)
            value = 1 + 1e-4*bor_options['near_refinement']
            return dict(theta_deg=[0.,90.], amp_vv=[value,value], amp_hh=[value,value],
                        modes_used=3, near_quadrature={})
        result = checked_solve(solve, (), {}, validate_options({'quadrature_check':'refine'}))
        self.assertEqual([c['near_refinement'] for c in calls], [0,1])
        self.assertTrue(all(c['quadrature_check']=='off' for c in calls))
        self.assertEqual(result['amp_vv'][0],1.0001)
        self.assertTrue(result['quadrature_comparison']['passed'])
        self.assertTrue(result['near_quadrature']['self_and_junction_convergence_checked'])

    def test_process_task_serializes_junction_refinement(self):
        with option_scope(validate_options({'near_refinement':1})):
            task = NearTask(None,None,1.,2,('efie',),pair_kind={(0,0):'corner00'})
        self.assertEqual(pickle.loads(pickle.dumps(task)).junction_refinement,1)

    def test_options_reject_saturated_comparison(self):
        with self.assertRaises(ValueError):
            validate_options({'near_refinement':2,'quadrature_check':'refine'})

    def test_resource_preview_does_not_attempt_field_comparison(self):
        @configured
        def estimate_bor_resources(geometry_snapshot, frequency_ghz, aspects_deg):
            return {'estimated_peak_gb': .5}
        result = estimate_bor_resources({}, 1., [0., 90.],
            bor_options={'quadrature_check':'refine', 'factorization':'dense'})
        self.assertEqual(result['estimated_peak_gb'], .5)
        self.assertNotIn('quadrature_comparison', result)

    def test_public_resource_preview_accepts_refined_comparison(self):
        from pathlib import Path
        from ghost_backend import run_local_bor
        from ghost_backend.bor.dispatch import estimate_bor_resources
        path = Path(__file__).resolve().parents[1]/'geometry/geometries/body.geo'
        snapshot, directory = run_local_bor._load_snapshot(str(path))
        result = estimate_bor_resources(snapshot, 1., [0., 90.], geometry_units='meters',
            material_base_dir=directory, workers=1, mesh_certification=False,
            bor_options={'quadrature_check':'refine'})
        self.assertGreater(result['estimated_peak_gb'], 0.)
        self.assertNotIn('quadrature_comparison', result)

    def test_direct_comparison_adds_retained_fields_to_output_reservation(self):
        reservations = []
        @configured
        def solve_bor(points, freq_hz, thetas_deg):
            reservations.append(output_reserved_gb())
            return dict(theta_deg=thetas_deg, amp_vv=np.ones(len(thetas_deg)),
                        amp_hh=np.ones(len(thetas_deg)), modes_used=3)
        result = solve_bor(None, 1e9, [0., 90.],
            bor_options={'quadrature_check':'refine', 'factorization':'dense'})
        self.assertAlmostEqual(reservations[0], estimate_output_gb(1, 2))
        self.assertAlmostEqual(reservations[1] - reservations[0],
            result['quadrature_comparison']['comparison_field_bytes']/1e9)

    def test_checked_solve_preserves_actual_resolved_options(self):
        def solve(*args, bor_options, **kwargs):
            actual = dict(bor_options, factorization='dense')
            return dict(theta_deg=[0.], amp_vv=[1.], amp_hh=[1.],
                bor_execution_options=actual, metadata={'bor_execution_options': actual})
        result = checked_solve(solve, (), {}, validate_options({'quadrature_check':'refine'}))
        self.assertEqual(result['bor_execution_options']['factorization'], 'dense')
        self.assertEqual(result['metadata']['bor_execution_options']['factorization'], 'dense')
        self.assertEqual(result['requested_bor_execution_options']['factorization'], 'auto')
