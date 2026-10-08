"""Round 12: the 2D junction rule measures the complex jump of the evaluated impedance, so phase counts."""
from pathlib import Path
import cmath
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND.parent), str(BACKEND/'tests')]
from ghost_backend.twod import solver as td
from ghost_backend.twod import geometry as td_geometry

RADIUS = .1
DENSE_2D = dict(factorization='dense', mesh_strategy='global')
PARTNER = 75-20j


def _arc(count, ibc, name, start, stop):
    theta = np.linspace(start, stop, count+1)
    pairs = [dict(x1=float(RADIUS*np.cos(a)), y1=float(RADIUS*np.sin(a)),
                  x2=float(RADIUS*np.cos(b)), y2=float(RADIUS*np.sin(b))) for a, b in zip(theta[:-1], theta[1:])]
    return dict(name=name, seg_type=2, properties=['2', '1', str(ibc), '0', '0'], point_pairs=pairs)


def _halves(count, law):
    rows = [['1', 'constant', repr(PARTNER.real), repr(PARTNER.imag), '0', '0'],
            ['2', 'constant', repr(float(np.real(law))), repr(float(np.imag(law))), '0', '0']]
    return dict(segments=[_arc(count//2, 2, 'a', 0., -np.pi), _arc(count//2, 1, 'b', -np.pi, -2*np.pi)], ibcs=rows, dielectrics=[])


class ComplexContrastTests(unittest.TestCase):
    WAVELENGTH = 2*np.pi*RADIUS/4.3

    def test_the_rule_is_the_factor_of_four_for_laws_in_phase(self):
        graded = td_geometry.impedance_jump_is_graded
        for first, second, expected in ((0j, 50+0j, True), (0j, 0j, False), (10., 39., False), (10., 41., True),
                                        (10j, 41j, True), (10-10j, 39-39j, False), (1e-9, 1., True)):
            with self.subTest(pair=(first, second)):
                self.assertEqual(graded(first, second), expected)
                self.assertEqual(graded(second, first), expected)

    def test_phase_alone_grades_beyond_about_44_degrees(self):
        magnitude, phase = abs(PARTNER), math.degrees(cmath.phase(PARTNER))
        for offset, extra in ((-30., 0), (30., 0), (40., 0), (50., 16), (75., 16), (105., 16), (-60., 16)):
            law = cmath.rect(magnitude, math.radians(phase+offset))
            if law.real < 0:
                continue
            with self.subTest(offset=offset):
                panels = td._build_panels(_halves(104, law), 1., self.WAVELENGTH)
                self.assertEqual(len(panels), 104+extra)

    def test_levels_follow_the_mesh_density(self):
        levels = td_geometry._junction_grading_levels
        self.assertEqual([levels(1./density, 1.) for density in (10, 32, 33, 64, 65, 200, 1000)], [4, 4, 5, 5, 6, 7, 9])
        self.assertEqual(levels(1e-3, None), 4)                       # no wavelength: the base levels
        self.assertEqual(levels(4e-7, 1.), 2)                         # never within two decades of the 1e-9 m node snap
        pec = dict(segments=[_arc(208, 0, 'a', 0., -np.pi), _arc(208, 1, 'b', -np.pi, -2*np.pi)],
                   ibcs=[['1', 'constant', '75', '-20', '0', '0']], dielectrics=[])
        panels = td._build_panels(pec, 1., self.WAVELENGTH)           # 97 panels per wavelength: six levels
        self.assertEqual(len(panels), 416+4*6)
        lengths = np.array([panel.length for panel in panels])
        self.assertAlmostEqual(lengths.min()/lengths.max(), 1/64, places=6)
        ratios = lengths[1:]/lengths[:-1]
        self.assertLessEqual(max(ratios.max(), 1/ratios.min()), 2.+1e-9)

    def test_the_tm_junction_stays_second_order_on_fine_meshes(self):
        # four fixed levels: 0.0168 / 0.0071 % at 416 / 832 panels (ratio 2.4); the density rule: 0.0148 / 0.0035 %
        frequency = 4.3*td.C0/(2*np.pi*RADIUS*1e9)
        def solve(count):
            shape = dict(segments=[_arc(count//2, 0, 'a', 0., -np.pi), _arc(count//2, 1, 'b', -np.pi, -2*np.pi)],
                         ibcs=[['1', 'constant', '75', '-20', '0', '0']], dielectrics=[])
            result = td.solve_monostatic_rcs_2d_single_polarization(shape, [frequency], [0., 60., 90., 180.], 'TM',
                geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE_2D))
            return np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']])
        with patch.object(td_geometry, 'JUNCTION_GRADING_LEVELS', 8):     # a reference that does not lean on the rule under test
            reference = solve(1664)
        errors = [100*np.max(np.abs(solve(count)-reference))/np.max(np.abs(reference)) for count in (416, 832)]
        self.assertLess(errors[1], .0045)
        self.assertGreater(errors[0]/errors[1], 3.5)

    def test_a_reactive_law_next_to_a_resistive_one_converges_like_a_graded_junction(self):
        # +77.6j | 75-20j (105 degrees apart, equal magnitudes), TM: uniform panels gave 0.59 / 0.15 %
        frequency = 4.3*td.C0/(2*np.pi*RADIUS*1e9)
        def solve(count):
            result = td.solve_monostatic_rcs_2d_single_polarization(_halves(count, 77.62j), [frequency], [0., 60., 90., 180.],
                'TM', geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE_2D))
            return (np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']]),
                    result['metadata']['panel_count'])
        reference, _ = solve(1664)
        errors = []
        for count, graded in ((104, 120), (208, 228)):       # 24 and 48 panels per wavelength: four and five levels
            values, panels = solve(count)
            self.assertEqual(panels, graded)
            errors.append(100*np.max(np.abs(values-reference))/np.max(np.abs(reference)))
        self.assertLess(errors[0], .35)
        self.assertLess(errors[1], .09)


if __name__ == '__main__':
    unittest.main()
