"""Regressions for the round-10 work on the five open items of the September audit."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch
from ghost_backend.twod import solver as td
from ghost_backend.twod import geometry as td_geometry
from ghost_backend.twod.assembly import scatter
from ghost_backend.twod.formulations import regions
from ghost_backend.validation.sphere import (sigma_coated_pec_sphere, sigma_dielectric_sphere,
                                             sigma_impedance_sphere)

RADIUS = .1
DENSE_2D = dict(factorization='dense', mesh_strategy='global')
IBC_A = ['1', 'constant', '75', '-20', '0', '0']
IBC_B = ['2', 'constant', '200', '0', '0', '0']


def _both(out):
    return list(out['sigma_vv']) + list(out['sigma_hh'])


def _error_db(value, reference):
    return float(np.max(np.abs(10*np.log10(np.asarray(value, float)/reference))))


def _arc(count, seg_type, ibc=0, material=0, name='arc', start=0., stop=-2*np.pi, radius=RADIUS):
    theta = np.linspace(start, stop, count+1)
    pairs = [dict(x1=float(radius*np.cos(a)), y1=float(radius*np.sin(a)),
                  x2=float(radius*np.cos(b)), y2=float(radius*np.sin(b))) for a, b in zip(theta[:-1], theta[1:])]
    return dict(name=name, seg_type=seg_type, properties=[str(seg_type), '1', str(ibc), str(material), '0'],
                point_pairs=pairs)


def _halves(count, first, second):
    half = count//2
    return dict(segments=[_arc(half, 2, ibc=first, name='a', stop=-np.pi),
                          _arc(half, 2, ibc=second, name='b', start=-np.pi)], ibcs=[IBC_A, IBC_B], dielectrics=[])


def _amplitudes(result):
    return np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']])


class InterfaceWeightTests(unittest.TestCase):
    """PMCHWT's equal weights let a pole-localized reactive mode cancel at real frequencies."""

    def test_weights_rotate_the_denser_region_negatively_and_keep_unit_modulus(self):
        weights = bor._region_equation_weights([None, (3., 1.), (3.-.3j, 1.), (.5, 1.), (1., 1.), (2., 4.)])
        np.testing.assert_allclose(np.abs(weights), 1.)
        theta = math.radians(bor.BOR_INTERFACE_WEIGHT_PHASE_DEGREES)
        # ranks by Re(eps*mu): 0.5 < 1 (air, (1,1)) < 3 (both) < 8
        np.testing.assert_allclose(np.angle(weights), [0., -theta, -theta, theta, 0., -2*theta], atol=1e-14)
        self.assertEqual(bor._region_equation_weights([(4., 1.), None], exterior=1)[1], 1.)

    def test_coarse_lossless_bodies_have_no_spurious_real_frequency_singularity(self):
        # 20 elements, eps_r = 3: PMCHWT was singular at 0.95126650 and 1.09777450 GHz
        # (condition 2e8, +6.7 dB axially, 10 dB off axis); the coated sphere lost 8.4 dB.
        points = bor.sphere_generatrix(RADIUS, 20)
        for frequency in (.95126650e9, 1.09777450e9):
            with self.subTest(body='dielectric', frequency=frequency):
                out = bor.solve_bor_dielectric(points, frequency, [0., 90.], 3., 1., workers=1,
                                               bor_options=dict(factorization='dense'))
                reference = sigma_dielectric_sphere(RADIUS, 3., 1., frequency)
                self.assertLess(_error_db(_both(out), reference), .25)
                self.assertLess(out['max_cond'], 1e6)
        core = bor.sphere_generatrix(.07, 14)
        for frequency in (.95e9, 1.1e9):
            with self.subTest(body='coated', frequency=frequency):
                out = bor.solve_bor_coated_pec(points, core, frequency, [0., 90.], 3., 1., workers=1,
                                               bor_options=dict(factorization='dense'))
                self.assertLess(_error_db(_both(out),
                                          sigma_coated_pec_sphere(.07, RADIUS, 3., 1., frequency)), .5)
                self.assertLess(out['max_cond'], 1e6)

    def test_weighted_equations_keep_production_accuracy_and_every_backend(self):
        points, frequency = bor.sphere_generatrix(RADIUS, 40), 1.2e9
        for eps in (3., 3.-.3j, .5):
            with self.subTest(eps=eps):
                out = bor.solve_bor_dielectric(points, frequency, [0., 60., 120.], eps, 1., workers=1,
                                               bor_options=dict(factorization='dense'))
                self.assertLess(_error_db(_both(out),
                                          sigma_dielectric_sphere(RADIUS, eps, 1., frequency)), .1)
        small = bor.sphere_generatrix(RADIUS, 16)
        args = (small, .9e9, [0., 60.], 3., 1.)
        dense = bor.solve_bor_dielectric(*args, workers=1, bor_options=dict(factorization='dense'))
        for label, options in (('streaming', dict(assembly='streaming', bor_options=dict(factorization='dense'))),
                               ('compressed', dict(bor_options=dict(factorization='compressed')))):
            with self.subTest(path=label):
                other = bor.solve_bor_dielectric(*args, workers=1, **options)
                np.testing.assert_allclose(other['sigma_vv'], dense['sigma_vv'], rtol=1e-8)
                np.testing.assert_allclose(other['sigma_hh'], dense['sigma_hh'], rtol=1e-8)


class VaryingImpedanceBorTests(unittest.TestCase):
    """Closed bodies with a spatially varying impedance were rejected; the EFIE alone has resonances."""

    @staticmethod
    def _profile(kind, elements):
        theta = (np.arange(elements)+.5)*np.pi/elements
        if kind == 'zones':
            return np.where(theta < np.pi/2, 0., 100+50j)
        return 50+250*np.sin(theta)**2+40j*np.cos(theta)**2

    def _solve(self, kind, elements, ka, formulation):
        frequency = ka*td.C0/(2*np.pi*RADIUS)
        out = bor.solve_bor(bor.sphere_generatrix(RADIUS, elements), frequency, [0., 45., 90., 135., 180.],
                            formulation=formulation, zs=self._profile(kind, elements), workers=1,
                            bor_options=dict(factorization='dense'))
        return np.array(list(out['sigma_vv'])+list(out['sigma_hh'])), out

    def test_uniform_profile_given_per_element_is_the_scalar_law(self):
        frequency = 3.3*td.C0/(2*np.pi*RADIUS)
        common = dict(formulation='cfie', workers=1, bor_options=dict(factorization='dense'))
        scalar = bor.solve_bor(bor.sphere_generatrix(RADIUS, 40), frequency, [0., 90.], zs=100+50j, **common)
        array = bor.solve_bor(bor.sphere_generatrix(RADIUS, 40), frequency, [0., 90.], zs=np.full(40, 100+50j), **common)
        np.testing.assert_array_equal(array['sigma_vv'], scalar['sigma_vv'])
        self.assertLess(_error_db(scalar['sigma_vv'], sigma_impedance_sphere(RADIUS, frequency, 100+50j)), .02)

    def test_cfie_converges_to_the_efie_answer_away_from_its_resonances(self):
        for kind, tolerance in (('taper', .01), ('zones', .15)):
            with self.subTest(kind=kind):
                efie, _ = self._solve(kind, 120, 3.3, 'efie')
                cfie, out = self._solve(kind, 60, 3.3, 'cfie')
                self.assertEqual(out['formulation'].lower()[:4], 'cfie')
                self.assertLess(_error_db(cfie, efie), tolerance)

    def test_cfie_is_smooth_where_the_efie_resonates(self):
        # First PEC-cavity resonance region of the sphere (ka = 2.744): the tapered
        # EFIE departs 16 dB from its neighbours, the CFIE stays on the trend.
        levels = {}
        for formulation in ('efie', 'cfie'):
            values = [self._solve('taper', 40, ka, formulation)[0] for ka in (2.7425, 2.7450, 2.7475)]
            levels[formulation] = np.max(np.abs(10*np.log10(values[1]/np.sqrt(values[0]*values[2]))))
        self.assertGreater(levels['efie'], 1.)
        self.assertLess(levels['cfie'], .01)

    def test_dispatcher_accepts_a_closed_pec_impedance_body(self):
        points = bor.sphere_generatrix(.05, 16)
        chain = lambda name, pts, ibc: dict(name=name, seg_type=2, properties=['2', '1', str(ibc), '0', '0'],
            point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1])) for a, b in zip(pts[:-1], pts[1:])])
        snapshot = dict(segments=[chain('pec', points[:9], 0), chain('ibc', points[8:], 1)],
                        ibcs=[['1', 'constant', '100', '50', '0', '0']], dielectrics=[])
        self.assertEqual(dispatch._conductor_formulation(np.array([0., 100+50j])), 'cfie')
        result = dispatch.solve_monostatic_rcs_bor(snapshot, [1.], [0., 90.], geometry_units='meters', workers=1,
                                                   bor_options=dict(factorization='dense'))
        self.assertIn('IBC-CFIE', result['metadata']['formulation'])
        self.assertTrue(all(np.isfinite(row['rcs_linear']) and row['rcs_linear'] > 0 for row in result['samples']))


class BorEstimatorTests(unittest.TestCase):
    """The snapshot preview priced table plans at 3.5 x tables, two to three times the gate it predicts."""

    def test_preview_tracks_the_run_time_gate_for_table_and_streamed_plans(self):
        chain = lambda name, kind, pts, material=0: dict(name=name, seg_type=kind, properties=[str(kind), '1', '0', str(material), '0'],
            point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1])) for a, b in zip(pts[:-1], pts[1:])])
        dielectrics = [['1', '2.56', '-0.1', '1.0', '0.0']]
        bodies = dict(
            conductor=dict(segments=[chain('pec', 2, bor.sphere_generatrix(RADIUS, 60))], ibcs=[], dielectrics=[]),
            dielectric=dict(segments=[chain('d', 3, bor.sphere_generatrix(RADIUS, 40), 1)], ibcs=[], dielectrics=dielectrics),
            coated=dict(segments=[chain('o', 3, bor.sphere_generatrix(RADIUS, 40), 1),
                                  chain('c', 4, bor.sphere_generatrix(.07, 28), 1)], ibcs=[], dielectrics=dielectrics))
        frequency = 12.*td.C0/(2*np.pi*RADIUS*1e9)

        class Admitted(Exception):
            pass
        guard = bor._guard_bor_dense_memory
        for kind, snapshot in bodies.items():
            for assembly in ('tables', 'streaming'):
                with self.subTest(kind=kind, assembly=assembly):
                    gates = []
                    def gate(*args, **kwargs):
                        gates.append(guard(*args, **kwargs))
                        raise Admitted()
                    with patch.object(td, '_solve_memory_limit_gb', return_value=64.), \
                            patch.object(bor, '_solve_memory_limit_gb', return_value=64.):
                        preview = dispatch.estimate_bor_resources(snapshot, frequency, [0., 60.], geometry_units='meters',
                            n_modes=20, workers=1, assembly=assembly, mesh_certification=False,
                            bor_options=dict(factorization='dense'))
                        with patch.object(bor, '_guard_bor_dense_memory', gate), self.assertRaises(Admitted):
                            dispatch.solve_monostatic_rcs_bor(snapshot, [frequency], [0., 60.], geometry_units='meters',
                                n_modes=20, workers=1, assembly=assembly, bor_options=dict(factorization='dense'))
                    # never below (snapshot entries cannot fall back), never far above
                    self.assertGreaterEqual(preview['estimated_peak_gb'], gates[0])
                    self.assertLess(preview['estimated_peak_gb'], 1.25*gates[0])

    def test_table_plan_model_is_the_run_time_one(self):
        # 200-element CFIE sphere, cap 20: 1.085 GB of tables, measured process peak
        # 1.34 GB. The blanket factor asked 3.80 GB of assembly for it.
        tables = bor.estimate_bor_table_gb(200, 20, 'cfie', False, 4, False)
        model = dispatch._table_plan_peak_gb(tables, [(200, True)], 20)
        self.assertAlmostEqual(tables, 1.0854, places=3)
        self.assertGreater(model, 1.34)
        self.assertLess(model, 1.6)


class JunctionGradingTests(unittest.TestCase):
    """TM: u = 0 meets a Robin law at a PEC/impedance junction; uniform panels converged at first order."""

    def test_only_pec_impedance_junctions_of_conductors_are_graded(self):
        wavelength = 2*np.pi*RADIUS/4.3
        for first, second, extra in ((0, 1, 16), (1, 2, 0), (0, 0, 0)):
            with self.subTest(flags=(first, second)):
                panels = td._build_panels(_halves(104, first, second), 1., wavelength)
                self.assertEqual(len(panels), 104+extra)
        panels = td._build_panels(_halves(104, 0, 1), 1., wavelength)
        lengths = np.array([panel.length for panel in panels])
        base = lengths.max()
        np.testing.assert_allclose(sorted(lengths[lengths < .99*base]/base),
                                   sorted([1/16, 1/16, 1/8, 1/4, 1/2]*4), rtol=1e-6)
        ratios = lengths[1:]/lengths[:-1]
        self.assertLessEqual(max(ratios.max(), 1/ratios.min()), 2.+1e-9)
        # a dielectric interface next to a conductor is not a Dirichlet/Robin junction
        shape = dict(segments=[_arc(40, 3, material=1), _arc(24, 4, material=1, radius=.06, name='core')],
                     ibcs=[], dielectrics=[['1', '2.56', '-.1', '1', '0']])
        self.assertEqual(len(td._build_panels(shape, 1., wavelength)), 64)

    def test_tm_junction_error_falls_at_second_order(self):
        def solve(count, pol):
            frequency = 4.3*td.C0/(2*np.pi*RADIUS)
            result = td.solve_monostatic_rcs_2d_single_polarization(_halves(count, 0, 1), [frequency/1e9],
                [0., 60., 90., 180.], pol, geometry_units='meters', strict_quality_gate=False,
                execution_options=dict(DENSE_2D))
            return _amplitudes(result)
        reference = solve(1664, 'TM')
        errors = [100*np.max(np.abs(solve(count, 'TM')-reference))/np.max(np.abs(reference)) for count in (104, 208, 416)]
        # uniform panels gave 0.97, 0.49, 0.24 %
        self.assertLess(errors[0], .35)
        self.assertLess(errors[1], .09)
        self.assertLess(errors[2], .03)
        self.assertGreater(errors[0]/errors[1], 3.)


class PairedPolarizationAssemblyTests(unittest.TestCase):
    """A conductor's TM system cannot be derived from the TE one; both now share one kernel traversal."""

    @staticmethod
    def _prepared(shape, ka=6.):
        k0 = ka/RADIUS
        frequency = k0*td.C0/(2e9*np.pi)
        materials = td.MaterialLibrary.from_entries(shape['ibcs'], shape['dielectrics'], base_dir='.')
        panels = td._build_panels(shape, 1., 2*np.pi/k0)
        infos = td._build_coupled_panel_info(panels, materials, frequency, 'TE', k0)
        mesh, _ = td._build_linear_mesh_interface_aware(panels, infos, polarization='TE')
        return mesh, td._build_linear_coupled_infos(mesh, materials, frequency, 'TE', k0)

    def test_paired_matrices_are_the_separately_assembled_ones(self):
        from ghost_backend.execution import options
        shapes = dict(pec=dict(segments=[_arc(72, 2)], ibcs=[], dielectrics=[]),
                      mixed=_halves(72, 0, 1),
                      coated=dict(segments=[_arc(72, 3, material=1), _arc(40, 4, material=1, radius=.06, name='core')],
                                  ibcs=[], dielectrics=[['1', '2.56', '-.1', '1', '0']]))
        for label, shape in shapes.items():
            with self.subTest(body=label), options.execution_scope({}):
                mesh, infos = self._prepared(shape)
                pair = scatter.assemble_pair(mesh, infos, 8, 8)
                for (matrix, layout), pol in zip(pair, ('TE', 'TM')):
                    alone, _ = scatter.assemble_multi(mesh, infos, pol, 8, 8)
                    self.assertEqual(layout['polarization'], pol)
                    np.testing.assert_allclose(matrix, alone, rtol=0, atol=1e-13*np.max(np.abs(alone)))

    def test_co_polarized_solve_pairs_conductors_once_and_matches_single_polarization_solves(self):
        frequency = 6.*td.C0/(2*np.pi*RADIUS*1e9)
        engine = td._assemble_linear_operator_matrices_multi
        for label, shape, traversals in (('pec', dict(segments=[_arc(96, 2)], ibcs=[], dielectrics=[]), 1),
                                         ('mixed', _halves(96, 0, 1), 1),
                                         ('dielectric', dict(segments=[_arc(96, 3, material=1)], ibcs=[],
                                                             dielectrics=[['1', '2.56', '-.1', '1', '0']]), None)):
            with self.subTest(body=label):
                calls = []
                def counted(*args, **kwargs):
                    calls.append(len(args[3]))
                    return engine(*args, **kwargs)
                common = dict(geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE_2D))
                with patch.object(td, '_assemble_linear_operator_matrices_multi', counted):
                    paired = td.solve_monostatic_rcs_2d(shape, [frequency], [0., 60., 180.], **common)
                if traversals is not None:
                    self.assertEqual(len(calls), traversals)      # one traversal serves both polarizations
                for channel, pol in (('VV', 'TE'), ('HH', 'TM')):
                    single = td.solve_monostatic_rcs_2d_single_polarization(shape, [frequency], [0., 60., 180.], pol, **common)
                    rows = paired['co_solved_samples'][channel]
                    np.testing.assert_allclose([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in rows],
                                               _amplitudes(single), rtol=1e-10)

    def test_pairing_needs_a_co_polarized_session_and_room_for_the_second_matrix(self):
        from ghost_backend.twod.assembly import session as assembly_session
        frequency = 6.*td.C0/(2*np.pi*RADIUS*1e9)
        shape = dict(segments=[_arc(96, 2)], ibcs=[], dielectrics=[])
        common = dict(geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE_2D))
        pairs = []
        original = scatter.assemble_pair
        def watched(*args, **kwargs):
            pairs.append(True)
            return original(*args, **kwargs)
        with patch.object(scatter, 'assemble_pair', watched):
            # A session whose driver has not announced a TM solve (other drivers
            # open sessions too; the bare single-polarization entry opens none).
            with assembly_session._SESSION.override(assembly_session.AssemblySession()):
                td.solve_monostatic_rcs_2d_single_polarization(shape, [frequency], [0.], 'TE', **common)
            self.assertEqual(pairs, [])
            plan = assembly_session.plan_paired_assembly
            with patch.object(assembly_session, 'plan_paired_assembly',
                              lambda pol, kind, dofs, estimate, limit: plan(pol, kind, dofs, estimate, estimate)):
                td.solve_monostatic_rcs_2d(shape, [frequency], [0.], **common)
            self.assertEqual(pairs, [])                           # the second matrix does not fit
            td.solve_monostatic_rcs_2d(shape, [frequency], [0.], **common)
            self.assertEqual(pairs, [True])

if __name__ == '__main__':
    unittest.main()
