"""Round 11: 2D junction grading keyed on the PEC flag, not on the evaluated law (Codex's review of round 10, finding 2)."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver as td

RADIUS = .1
DENSE_2D = dict(factorization='dense', mesh_strategy='global')
PARTNER = ['1', 'constant', '75', '-20', '0', '0']


def _arc(count, ibc, name, start, stop):
    theta = np.linspace(start, stop, count+1)
    pairs = [dict(x1=float(RADIUS*np.cos(a)), y1=float(RADIUS*np.sin(a)),
                  x2=float(RADIUS*np.cos(b)), y2=float(RADIUS*np.sin(b))) for a, b in zip(theta[:-1], theta[1:])]
    return dict(name=name, seg_type=2, properties=['2', '1', str(ibc), '0', '0'], point_pairs=pairs)


def _halves(count, first_flag, laws):
    """Half circle with ``first_flag`` | half circle with 75-20j ohms (flag 1)."""
    return dict(segments=[_arc(count//2, first_flag, 'a', 0., -np.pi), _arc(count//2, 1, 'b', -np.pi, -2*np.pi)],
                ibcs=[PARTNER]+list(laws), dielectrics=[])


def _amplitudes(result):
    return np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']])


class JunctionLawTests(unittest.TestCase):
    """Grading keyed on the PEC flag; the field solve keys on the evaluated impedance."""

    WAVELENGTH = 2*np.pi*RADIUS/4.3
    FREQUENCY = 4.3*td.C0/(2*np.pi*RADIUS*1e9)

    def _solve(self, snapshot, pol, count=None):
        result = td.solve_monostatic_rcs_2d_single_polarization(snapshot, [self.FREQUENCY], [0., 60., 90., 180.], pol,
            geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE_2D))
        return _amplitudes(result), result['metadata']['panel_count']

    def test_a_zero_ohm_law_is_the_pec_flag(self):
        # Codex's reproduction: flag 2 carrying 0 ohms was left ungraded (104 panels,
        # 0.97 % TM error against 0.24 %, fields 1.04 % apart from the flag-0 input).
        zero = ['2', 'constant', '0', '0', '0', '0']
        flagged, panels_flagged = self._solve(_halves(104, 0, []), 'TM')
        by_law, panels_by_law = self._solve(_halves(104, 2, [zero]), 'TM')
        self.assertEqual((panels_flagged, panels_by_law), (120, 120))
        np.testing.assert_allclose(by_law, flagged, rtol=1e-12)
        # two PEC definitions meeting each other are not a junction, exact or within the solver's tolerance
        for law in (zero, ['2', 'constant', '1e-13', '0', '0', '0']):
            both = dict(segments=[_arc(52, 0, 'a', 0., -np.pi), _arc(52, 2, 'b', -np.pi, -2*np.pi)], ibcs=[law], dielectrics=[])
            self.assertEqual(len(td._build_panels(both, 1., self.WAVELENGTH)), 104)

    def test_contrast_decides_not_the_flag(self):
        law = lambda value: ['2', 'constant', repr(float(np.real(value))), repr(float(np.imag(value))), '0', '0']
        for value, extra in ((1e-13, 16),       # PEC within the solver's tolerance
                             (.026+.026j, 16),  # copper at 10 GHz next to 75-20j: Dirichlet-like at the panel scale
                             (1., 16), (15., 16),
                             (30., 0), (75-20j, 0), (200., 0),     # within a factor of four: a staircase step
                             (400., 16), (1e5, 16)):               # PMC-like: the TE analogue
            with self.subTest(law=value):
                snapshot = _halves(104, 2, [law(value)])
                self.assertEqual(len(td._build_panels(snapshot, 1., self.WAVELENGTH)), 104+extra)
                materials = td.MaterialLibrary.from_entries(snapshot['ibcs'], [], base_dir='.')
                self.assertEqual(len(td._build_panels(snapshot, 1., self.WAVELENGTH, materials=materials,
                                                      frequencies_ghz=[self.FREQUENCY])), 104+extra)

    def test_high_contrast_junctions_converge_like_the_pec_one(self):
        # uniform panels: 0.36 / 0.17 % (TM, 1 ohm) and 2.0 / 1.1 % (TE, 1e5 ohm) at 104 / 208 panels
        for pol, value, bounds in (('TM', 1., (.35, .09)), ('TE', 1e5, (.25, .06))):
            with self.subTest(pol=pol, law=value):
                laws = [['2', 'constant', repr(value), '0', '0', '0']]
                reference, _ = self._solve(_halves(1664, 2, laws), pol)
                errors = [100*np.max(np.abs(self._solve(_halves(count, 2, laws), pol)[0]-reference))/np.max(np.abs(reference))
                          for count in (104, 208)]
                self.assertLess(errors[0], bounds[0])
                self.assertLess(errors[1], bounds[1])
                self.assertGreater(errors[0]/errors[1], 3.)

    def test_a_taper_is_classified_by_its_value_at_the_junction(self):
        # flag 3 rises linearly from 0 ohms where segment b starts (meeting PEC: no jump)
        # to 100 ohms where it ends (meeting PEC: a jump)
        taper = ['3', 'linear', '0', '0', '100', '0']
        snapshot = dict(segments=[_arc(52, 0, 'a', 0., -np.pi), _arc(52, 3, 'b', -np.pi, -2*np.pi)], ibcs=[taper], dielectrics=[])
        panels = td._build_panels(snapshot, 1., self.WAVELENGTH)
        self.assertEqual(len(panels), 104+8)
        short = [panel for panel in panels if panel.length < .4*max(p.length for p in panels)]
        for panel in short:       # every short panel lies at the closing vertex (RADIUS, 0), none at (-RADIUS, 0)
            self.assertLess(np.hypot(panel.center[0]-RADIUS, panel.center[1]), .3*RADIUS)

    def test_a_table_is_evaluated_at_every_frequency_the_mesh_serves(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'zs.csv').write_text('frequency_hz,resistance_ohm,reactance_ohm\n1e9,0,0\n2e9,60,0\n3e9,60,0\n')
            snapshot = _halves(104, 2, [['2', 'zs.csv']])
            materials = td.MaterialLibrary.from_entries(snapshot['ibcs'], [], base_dir=folder)
            count = lambda frequencies: len(td._build_panels(snapshot, 1., self.WAVELENGTH, materials=materials,
                                                             frequencies_ghz=frequencies))
            self.assertEqual(count([2.]), 104)            # 60 ohms next to 75-20j
            self.assertEqual(count([1.]), 120)            # PEC at this frequency
            self.assertEqual(count({1., 2., 3.}), 120)    # a fixed mesh serving all three
            self.assertEqual(count([2., 3.]), 104)
            self.assertEqual(count([1.5]), 104)           # 30 ohms: within the contrast
            self.assertEqual(count([.5]), 104)            # outside the table: the field solve reports that
            # The solves follow: a mesh per frequency, or one reference mesh for the sweep.
            common = dict(geometry_units='meters', material_base_dir=folder, strict_quality_gate=False,
                          execution_options=dict(DENSE_2D))
            sweep = td.solve_monostatic_rcs_2d_single_polarization(snapshot, [1., 2.], [0.], 'TM', **common)
            # 1 GHz: 50 panels per wavelength, so five levels per side since round 12 (104 + 4*5); 2 GHz: no junction
            self.assertEqual((sweep['metadata']['panel_count_min'], sweep['metadata']['panel_count_max']), (104, 124))
            fixed = td.solve_monostatic_rcs_2d_single_polarization(snapshot, [1., 2.], [0.], 'TM',
                                                                   mesh_reference_ghz=2., **common)
            self.assertEqual((fixed['metadata']['panel_count_min'], fixed['metadata']['panel_count_max']), (120, 120))
        # without a library a table cannot be evaluated: graded only against PEC
        self.assertEqual(len(td._build_panels(snapshot, 1., self.WAVELENGTH)), 104)
        against_pec = dict(segments=[_arc(52, 0, 'a', 0., -np.pi), _arc(52, 2, 'b', -np.pi, -2*np.pi)],
                           ibcs=[['2', 'zs.csv']], dielectrics=[])
        self.assertEqual(len(td._build_panels(against_pec, 1., self.WAVELENGTH)), 120)

    def test_every_mesh_consumer_counts_the_same_panels(self):
        from ghost_backend.execution.options import validate_options
        from ghost_backend.execution.selection import select_backend
        from ghost_backend.hpc import scheduler
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'zs.csv').write_text('frequency_hz,resistance_ohm,reactance_ohm\n1e9,0,0\n9e9,0,0\n')
            for label, snapshot in (('zero law', _halves(104, 2, [['2', 'constant', '0', '0', '0', '0']])),
                                    ('zero table', _halves(104, 2, [['2', 'zs.csv']]))):
                with self.subTest(definition=label):
                    arguments = dict(geometry_snapshot=snapshot, frequencies_ghz=[self.FREQUENCY], elevations_deg=[0.],
                                     geometry_units='meters', material_base_dir=folder)
                    solved = td.solve_monostatic_rcs_2d_single_polarization(polarization='TM', strict_quality_gate=False,
                        execution_options=dict(DENSE_2D), **arguments)
                    self.assertEqual(solved['metadata']['panel_count'], 120)
                    planned = select_backend(dict(arguments, polarization='TM'), validate_options(dict(DENSE_2D)))
                    self.assertEqual({record['panels'] for record in planned['meshes']}, {120})
                    materials = td.MaterialLibrary.from_entries(snapshot['ibcs'], [], base_dir=folder)
                    records = scheduler._resource_records_for_frequency(td, snapshot, materials, self.FREQUENCY,
                                                                        [('TM', 'TM')], 1., 20000)
                    self.assertEqual(records['TM']['panels'], 120)

if __name__ == '__main__':
    unittest.main()
