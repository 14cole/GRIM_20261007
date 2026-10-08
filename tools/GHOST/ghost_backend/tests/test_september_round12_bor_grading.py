"""Round 12: BoR conductor generatrices are graded toward impedance jumps, as the 2D mesher grades its junctions."""
from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch

RADIUS = .1
FREQUENCY = 3.3*299792458./(2*math.pi*RADIUS)/1e9
DENSE = dict(factorization='dense')


def _chain(name, points, ibc):
    return dict(name=name, seg_type=2, properties=['2', '1', str(ibc), '0', '0'],
                point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                             for a, b in zip(points[:-1], points[1:])])


def _halves(elements, first, second, laws):
    points = bor.sphere_generatrix(RADIUS, elements)
    half = elements//2
    return dict(segments=[_chain('upper', points[:half+1], first), _chain('lower', points[half:], second)],
                ibcs=list(laws), dielectrics=[])


def _law(flag, value):
    return [str(flag), 'constant', repr(float(np.real(value))), repr(float(np.imag(value))), '0', '0']


def _elements(snapshot, frequency=FREQUENCY, **kwargs):
    return dispatch.estimate_bor_resources(snapshot, frequency, [0., 90.], geometry_units='meters', workers=1,
                                           mesh_certification=False, bor_options=dict(DENSE), **kwargs)['mesh_elements']


class BorJunctionGradingTests(unittest.TestCase):
    def test_the_2d_rule_decides_which_chain_ends_are_graded(self):
        extra = 2*dispatch.BOR_JUNCTION_GRADING_LEVELS
        self.assertEqual(extra, 8)
        for first, second, laws, graded in (
                (0, 1, [_law(1, 100+50j)], True),                        # PEC | impedance
                (1, 1, [_law(1, 100+50j)], False),                       # one law on both chains
                (1, 2, [_law(1, 100+50j), _law(2, 100+50j)], False),     # the same law under two flags
                (1, 2, [_law(1, 100+50j), _law(2, 200+100j)], False),    # a factor of two
                (1, 2, [_law(1, 50), _law(2, 300+100j)], True),          # 6.3
                (0, 1, [_law(1, 0)], False),                             # a zero-ohm law is PEC
                (1, 2, [_law(1, 75-20j), _law(2, 77.62j)], True)):       # phase alone (105 degrees)
            with self.subTest(flags=(first, second), laws=laws):
                self.assertEqual(_elements(_halves(24, first, second, laws)), 24+(extra if graded else 0))

    def test_a_table_is_evaluated_at_the_frequency_of_each_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'zs.csv').write_text('frequency_hz,resistance_ohm,reactance_ohm\n1e9,0,0\n2e9,100,50\n')
            snapshot = _halves(24, 0, 1, [['1', 'zs.csv']])
            self.assertEqual(_elements(snapshot, 1., material_base_dir=folder), 24)       # PEC | PEC at 1 GHz
            self.assertEqual(_elements(snapshot, 2., material_base_dir=folder), 32)

    def test_graded_elements_halve_toward_the_junction_and_preview_counts_the_solved_mesh(self):
        snapshot = _halves(24, 0, 1, [_law(1, 100+50j)])
        meshes = []
        mesher = dispatch._mesh_generatrix
        def watched(*args, **kwargs):
            meshes.append(mesher(*args, **kwargs))
            return meshes[-1]
        with patch.object(dispatch, '_mesh_generatrix', watched):
            result = dispatch.solve_monostatic_rcs_bor(snapshot, [FREQUENCY], [0., 90.], geometry_units='meters', workers=1,
                                                       bor_options=dict(DENSE))
        record = result['metadata']['per_frequency'][0]
        self.assertEqual((record['mesh_elements_total'], record['graded_impedance_junctions']), (32, 1))
        self.assertEqual(_elements(snapshot), 32)
        points, chain_of_element, arc = meshes[-1]
        lengths = np.hypot(*np.diff(points, axis=0).T)
        base = lengths.max()
        np.testing.assert_allclose(sorted(lengths[lengths < .99*base]/base), sorted([1/16, 1/16, 1/8, 1/4, 1/2]*2), rtol=1e-3)
        ratios = lengths[1:]/lengths[:-1]
        self.assertLessEqual(max(ratios.max(), 1/ratios.min()), 2.+1e-6)
        self.assertEqual(list(chain_of_element), [0]*16+[1]*16)
        self.assertTrue(np.all(np.diff(arc[:16]) > 0) and np.all(np.diff(arc[16:]) > 0) and 0 < arc[15] < 1)

    def test_other_kinds_and_ungraded_conductors_keep_their_meshes(self):
        points = bor.sphere_generatrix(RADIUS, 24)
        pec = dict(segments=[_chain('pec', points, 0)], ibcs=[], dielectrics=[])
        self.assertEqual(_elements(pec), 24)
        chains = dispatch._chains_from_snapshot(_halves(24, 0, 1, [_law(1, 100+50j)]), 1.)
        self.assertFalse(any(chain.grade_start or chain.grade_end for chain in chains))   # unmarked until a frequency is known
        mesh = dispatch._mesh_generatrix(chains, 1., 1000, 1e-12)[0]
        self.assertEqual(len(mesh)-1, 24)

    def test_grading_restores_fast_convergence_at_an_impedance_jump(self):
        # PEC | 100+50j sphere, CFIE, ka = 3.3: ungraded 0.20 / 0.11 / 0.055 dB at 30 / 60 / 120 elements,
        # graded 0.058 / 0.015 / 0.004 dB (first order against second)
        def solve(elements, levels):
            with patch.object(dispatch, 'BOR_JUNCTION_GRADING_LEVELS', levels):
                result = dispatch.solve_monostatic_rcs_bor(_halves(elements, 0, 1, [_law(1, 100+50j)]), [FREQUENCY],
                    [0., 45., 90., 135., 180.], geometry_units='meters', workers=1, bor_options=dict(DENSE))
            return np.array([row['rcs_linear'] for row in result['samples']])
        error = lambda value, reference: float(np.max(np.abs(10*np.log10(value/reference))))
        reference = solve(80, 4)
        graded = [error(solve(count, 4), reference) for count in (20, 40)]
        uniform = error(solve(40, 0), reference)
        self.assertLess(graded[1], .04)
        self.assertGreater(graded[0]/graded[1], 3.)          # second order
        self.assertGreater(uniform, 3*graded[1])             # 0.16 dB against 0.03


if __name__ == '__main__':
    unittest.main()
