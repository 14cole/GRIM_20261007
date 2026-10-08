"""Oversized meshes must be rejected before their coordinate arrays are allocated."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import dispatch, solver as bor
from ghost_backend.twod import geometry as td
from ghost_backend.twod import adaptive_geometry


def snapshot(points, count):
    return dict(segments=[dict(name='body', seg_type=2,
        properties=['2', str(count), '0', '0', '0'],
        point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                     for a, b in zip(points[:-1], points[1:])])], ibcs=[], dielectrics=[])


class MeshAllocationLimitTests(unittest.TestCase):
    def test_bor_preview_rejects_huge_counts_before_building_breakpoints(self):
        body = snapshot(bor.sphere_generatrix(.1, 12), 10**9)
        with patch.object(dispatch, '_primitive_breaks', side_effect=AssertionError('premature allocation')):
            with self.assertRaisesRegex(ValueError, 'max_elements|element limit'):
                dispatch.estimate_bor_resources(body, .1, [0.], geometry_units='meters',
                    max_elements=100, workers=1, mesh_certification=False,
                    bor_options=dict(factorization='dense'))

    def test_bor_mesher_checks_its_own_limit_before_allocation(self):
        chains = dispatch._chains_from_snapshot(snapshot(bor.sphere_generatrix(.1, 12), 10**9), 1.)
        chains[0].grade_start = True
        with patch.object(dispatch, '_primitive_breaks', side_effect=AssertionError('premature allocation')):
            with self.assertRaisesRegex(ValueError, 'max_elements|element limit|> max'):
                dispatch._mesh_generatrix(chains, 1., 100, 1e-12)

    def test_bor_count_includes_grading_when_a_single_panel_has_two_junctions(self):
        chain = dispatch._SegChain('middle', 2, 1, 1, 0, 0, np.array([[.1, .1], [.1, 0.]]))
        for count in (1, 2, 7):
            for start, end in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(count=count, start=start, end=end):
                    chain.n_prop, chain.grade_start, chain.grade_end = count, start, end
                    points, _, _ = dispatch._mesh_generatrix([chain], 1., 100, 1e-12)
                    self.assertEqual(dispatch._run_element_count([chain], 1.), len(points)-1)
                    self.assertTrue(np.all(np.diff(points[:, 1]) < 0))

    def test_2d_global_rejects_huge_counts_before_discretizing(self):
        angles = np.linspace(0., -2*np.pi, 13)
        body = snapshot(.1*np.column_stack((np.cos(angles), np.sin(angles))), 10**9)
        with patch.object(td, '_discretize_primitive', side_effect=AssertionError('premature allocation')):
            with self.assertRaisesRegex(ValueError, 'panel limit'):
                td._build_panels(body, 1., 1., max_panels=100)

    def test_2d_adaptive_rejects_huge_counts_before_linspace(self):
        angles = np.linspace(0., -2*np.pi, 13)
        body = snapshot(.1*np.column_stack((np.cos(angles), np.sin(angles))), 10**9)
        body['_2d_hp_coarsening'] = 4.
        with patch.object(adaptive_geometry.np, 'linspace', side_effect=AssertionError('premature allocation')):
            with self.assertRaisesRegex(ValueError, 'panel limit'):
                td._build_panels(body, 1., 1., max_panels=100)

    def test_adaptive_limit_applies_after_coarsening(self):
        a, b = np.array([0., 0.]), np.array([1., 0.])
        _, points = adaptive_geometry.panel_parameters({'_2d_hp_coarsening': 4.}, 0,
            a, b, 200, False, 1., set(), max_panels=60)
        self.assertEqual(len(points)-1, 50)
        with self.assertRaisesRegex(ValueError, 'panel limit'):
            adaptive_geometry.panel_parameters({'_2d_hp_coarsening': 4.}, 0,
                a, b, 200, True, 1., set(), max_panels=60)


if __name__ == '__main__':
    unittest.main()
