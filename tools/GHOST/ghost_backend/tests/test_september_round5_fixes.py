"""Regressions for the round-5 to round-9 corrections of the September audit implementation."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch, geometry as bor_geometry
from ghost_backend.bor.options import (BorAdmissionError, ModalConvergenceError, configured,
                                       current_options, next_mode_cap, option_scope)
from ghost_backend.twod import solver as td
from ghost_backend.twod import geometry as td_geometry
from ghost_backend.twod.formulations import combined_regions, regions
from ghost_backend.validation.sphere import sigma_dielectric_sphere

DENSE = dict(factorization='dense', mesh_strategy='global')
RADIUS = .1
# Discrete TE resonance of the legacy SLP equation on the 104-panel circle with
# a PEC half and a 75-20j ohm half (located by its condition-number peak).
MIXED_RESONANCE_KA = 5.1371836974


def _arc(count, seg_type, ibc=0, material=0, name='arc', start=0., stop=-2*np.pi, radius=RADIUS, panels=1):
    theta = np.linspace(start, stop, count+1)
    pairs = [dict(x1=float(radius*np.cos(a)), y1=float(radius*np.sin(a)),
                  x2=float(radius*np.cos(b)), y2=float(radius*np.sin(b))) for a, b in zip(theta[:-1], theta[1:])]
    return dict(name=name, seg_type=seg_type, properties=[str(seg_type), str(panels), str(ibc), str(material), '0'],
                point_pairs=pairs)


def _halves(count, first, second, ibcs):
    half = count//2
    return dict(segments=[_arc(half, 2, ibc=first, name='a', stop=-np.pi),
                          _arc(half, 2, ibc=second, name='b', start=-np.pi)], ibcs=ibcs, dielectrics=[])


IBC_A = ['1', 'constant', '75', '-20', '0', '0']
IBC_B = ['2', 'constant', '200', '0', '0', '0']


def _solve(shape, pol, ka, angles=(0., 60., 90., 180.), **kwargs):
    frequency = ka*td.C0/(2*np.pi*RADIUS)
    result = td.solve_monostatic_rcs_2d_single_polarization(shape, [frequency/1e9], list(angles), pol,
        geometry_units='meters', strict_quality_gate=False, execution_options=dict(DENSE), **kwargs)
    amplitude = np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']])
    return amplitude, np.array([s['rcs_linear'] for s in result['samples']]), result['metadata']


def _uniform_only_couplings(mesh, layout):
    """The former eligibility rule: closed and uniform law. Used as the legacy reference."""
    if 'combined_couplings' in layout:
        return layout['combined_couplings']
    result = dict(_CURRENT_COUPLINGS(mesh, dict(layout)))
    for (mi, rid) in list(result):
        alpha = layout['ifaces'][mi]['robin_alpha_elements'][layout['ifaces'][mi]['eids']]
        if not np.all(alpha == alpha[0]):
            for other in layout['region_ifaces'][rid]:
                result.pop((other, rid), None)
    layout['combined_couplings'] = result
    return result


_CURRENT_COUPLINGS = combined_regions.couplings


class MixedBoundaryTwoDTests(unittest.TestCase):
    def test_closed_mixed_contours_are_combined_and_have_no_resonance(self):
        for name, shape in (('pec|ibc', _halves(104, 0, 1, [IBC_A])), ('ibc|ibc', _halves(104, 1, 2, [IBC_A, IBC_B])),
                            ('taper', dict(segments=[_arc(104, 2, ibc=21)], dielectrics=[],
                                           ibcs=[['21', 'cosine', '20', '5', '376.73', '0']]))):
            with self.subTest(body=name):
                values = {}
                for offset in (-1e-3, 0., 1e-3):
                    _, sigma, meta = _solve(shape, 'TE', MIXED_RESONANCE_KA+offset, compute_condition_number=True)
                    values[offset] = 10*np.log10(sigma)
                self.assertIn('combined', meta['formulation'])
                self.assertLess(meta['condition_est_max'], 1e3)
                trend = .5*(values[-1e-3]+values[1e-3])
                self.assertLess(float(np.max(abs(values[0.]-trend))), .01)

    def test_legacy_equation_still_shows_the_defect_this_guards(self):
        # Keeps the regression honest: with the former eligibility the same
        # input is tens of dB off its neighbours while every gate passes.
        shape = _halves(104, 0, 1, [IBC_A])
        with patch.object(combined_regions, 'couplings', _uniform_only_couplings):
            values = [10*np.log10(_solve(shape, 'TE', MIXED_RESONANCE_KA+offset)[1]) for offset in (-1e-3, 0., 1e-3)]
        self.assertGreater(float(np.max(abs(values[1]-.5*(values[0]+values[2])))), 3.)

    def test_same_impedance_under_two_flags_matches_one_flag(self):
        single = dict(segments=[_arc(208, 2, ibc=1)], ibcs=[IBC_A], dielectrics=[])
        double = _halves(208, 1, 2, [IBC_A, ['2']+IBC_A[1:]])
        for pol in ('TE', 'TM'):
            with self.subTest(pol=pol):
                expected, actual = _solve(single, pol, 4.3)[0], _solve(double, pol, 4.3)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)

    def test_varying_law_agrees_with_the_legacy_equation_away_from_resonance(self):
        shapes = dict(taper=dict(segments=[_arc(208, 2, ibc=21)], dielectrics=[],
                                 ibcs=[['21', 'cosine', '20', '5', '376.73', '0']]),
                      zones=_halves(208, 1, 2, [IBC_A, IBC_B]))
        for name, shape in shapes.items():
            for pol in ('TE', 'TM'):
                with self.subTest(body=name, pol=pol):
                    combined, _, meta = _solve(shape, pol, 4.3)
                    self.assertIn('combined', meta['formulation'])
                    with patch.object(combined_regions, 'couplings', _uniform_only_couplings):
                        legacy, _, old = _solve(shape, pol, 4.3)
                    self.assertNotIn('combined', old['formulation'])
                    # 0.81% before the element-weighted trace jump (TM taper).
                    self.assertLess(float(np.max(abs(combined-legacy))/np.max(abs(legacy))), 3e-3)

    def test_tm_pec_impedance_junction_nodes_carry_the_dirichlet_row(self):
        # u = 0 on a PEC element, so a node touching one is a Dirichlet node, as in
        # the standalone Robin assembler. An averaged Robin row there left the
        # combined TM equation at 1.5% from the former one (208 panels); now 0.014%.
        shape = _halves(208, 0, 1, [IBC_A])
        materials = td.MaterialLibrary.from_entries(shape['ibcs'], shape['dielectrics'], base_dir='.')
        k0 = 4.3/RADIUS
        panels = td._build_panels(shape, 1., 2*np.pi/k0)
        frequency_ghz = k0*td.C0/(2e9*np.pi)
        for pol, expect_zero in (('TM', True), ('TE', False)):
            infos = td._build_coupled_panel_info(panels, materials, frequency_ghz, pol, k0)
            mesh, _ = td_geometry._build_linear_mesh_interface_aware(panels, infos, polarization=pol)
            coupled = td._build_linear_coupled_infos(mesh, materials, frequency_ghz, pol, k0)
            interface = regions.build_layout(mesh, coupled, pol)['ifaces'][0]
            nodal = dict(zip(interface['nodes'], interface['robin_alpha']))
            touching_pec = {n for ei in interface['eids'] if interface['robin_alpha_elements'][ei] == 0
                            for n in mesh.elements[ei].node_ids}
            touching_ibc = {n for ei in interface['eids'] if interface['robin_alpha_elements'][ei] != 0
                            for n in mesh.elements[ei].node_ids}
            junctions = touching_pec & touching_ibc
            self.assertEqual(len(junctions), 2)
            for node in junctions:
                self.assertEqual(nodal[node] == 0, expect_zero, (pol, node, nodal[node]))
        combined, _, meta = _solve(shape, 'TM', 4.3)
        self.assertIn('combined', meta['formulation'])
        with patch.object(combined_regions, 'couplings', _uniform_only_couplings):
            legacy = _solve(shape, 'TM', 4.3)[0]
        self.assertLess(float(np.max(abs(combined-legacy))/np.max(abs(legacy))), 1e-3)

    def test_only_closed_conductor_loops_share_nodes_across_impedance_flags(self):
        def split_nodes(shape):
            materials = td.MaterialLibrary.from_entries(shape['ibcs'], shape['dielectrics'], base_dir='.')
            k0 = 4.3/RADIUS
            panels = td._build_panels(shape, 1., 2*np.pi/k0)
            infos = td._build_coupled_panel_info(panels, materials, k0*td.C0/(2e9*np.pi), 'TM', k0)
            closed = td_geometry._closed_conductor_loop_panels(panels, infos, 1e-9)
            _, stats = td_geometry._build_linear_mesh_interface_aware(panels, infos, polarization='TM')
            return len(closed), len(panels), stats['linear_interface_split_nodes']
        closed, total, split = split_nodes(_halves(64, 0, 1, [IBC_A]))
        self.assertEqual((closed, split), (total, 0))
        open_strip = dict(segments=[_arc(16, 2, ibc=1, name='a', stop=-.5*np.pi),
                                    _arc(16, 2, ibc=2, name='b', start=-.5*np.pi, stop=-np.pi)],
                          ibcs=[IBC_A, IBC_B], dielectrics=[])
        closed, total, split = split_nodes(open_strip)
        self.assertEqual((closed, split), (0, 1))


class CombinedReuseTwoDTests(unittest.TestCase):
    def _co_polarized(self, shape, ka=4.3):
        frequency = ka*td.C0/(2*np.pi*RADIUS)
        return td.solve_monostatic_rcs_2d(shape, [frequency/1e9], [0., 70.], geometry_units='meters',
            strict_quality_gate=False, execution_options=dict(DENSE))

    def test_standalone_conductors_assemble_tm_fresh_instead_of_row_strips(self):
        from ghost_backend.compressed import regional_coefficients
        for name, shape in (('pec', dict(segments=[_arc(96, 2)], ibcs=[], dielectrics=[])),
                            ('ibc', dict(segments=[_arc(96, 2, ibc=1)], ibcs=[IBC_A], dielectrics=[]))):
            with self.subTest(body=name):
                with patch.object(regional_coefficients, 'PreparedOracle',
                                  side_effect=AssertionError('row-strip conductor rebuild started')):
                    result = self._co_polarized(shape)
                expected = _solve(shape, 'TM', 4.3, angles=(0., 70.))[0]
                actual = np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['co_solved_samples']['HH']])
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)

    def test_reuse_is_kept_while_conductor_rows_are_a_small_share(self):
        from ghost_backend.compressed import regional_coefficients
        shape = dict(segments=[_arc(64, 3, material=1), _arc(12, 4, material=1, name='core', radius=.2*RADIUS)],
                     ibcs=[], dielectrics=[['1', '3', '-0.12', '1', '0']])
        with patch.object(regional_coefficients, 'PreparedOracle', wraps=regional_coefficients.PreparedOracle) as oracle:
            self._co_polarized(shape)
        self.assertGreater(oracle.call_count, 0)
        self.assertEqual(regions.COMBINED_REUSE_CONDUCTOR_FRACTION_MAX_INVERSE, 8)


class BorPlanningTests(unittest.TestCase):
    def test_direct_automatic_backend_prices_the_plan_that_will_run(self):
        ka, core, eps = 12., .07, 2.56-.1j
        frequency = ka*td.C0/(2*np.pi*RADIUS)
        outer = int(math.ceil(np.pi*RADIUS/((td.C0/frequency)/abs(np.sqrt(eps))/20.)))
        supplied = dict(points_outer=bor.sphere_generatrix(RADIUS, outer),
            points_core=bor.sphere_generatrix(core, int(outer*core/RADIUS)), freq_hz=frequency,
            thetas_deg=[0., 60., 120.], eps_r=eps, mu_r=1., n_modes=None, gauss_order=4, workers=8)
        layout, radius = dispatch._direct_surface_layout(supplied)
        self.assertEqual([(conductor, closed) for _, conductor, closed in layout], [(False, True), (True, True)])
        self.assertAlmostEqual(radius, RADIUS, places=4)
        with patch.object(td, '_solve_memory_limit_gb', return_value=4.):
            # About 1,000 unknowns: a streamed dense plan fits easily. The former
            # single full-table bound (about 11 GB) selected compression here.
            self.assertEqual(dispatch._resolve_direct_factorization(dict(supplied, assembly='auto')), 'dense')
            self.assertEqual(dispatch._resolve_direct_factorization(dict(supplied, assembly='streaming')), 'dense')
        with patch.object(td, '_solve_memory_limit_gb', return_value=.6):
            self.assertEqual(dispatch._resolve_direct_factorization(dict(supplied, assembly='auto')), 'compressed')

    def test_automatic_plans_go_from_tables_to_streaming_before_compression(self):
        # Codex's round-5 reproduction: the far tables (1.46 GB; the core-to-outer
        # operators are derived from the outer-to-core ones, not stored) are below
        # the coated solver's streaming threshold, so it runs tables (3.01 GB gate)
        # whatever the limit is. Round 6 sent limits below that to compression;
        # Codex's round-6 review measured that at 18.2 minutes against 3.4 for
        # the dense streamed plan (2.30 GB gate), which fits them.
        frequency = 12.*td.C0/(2*np.pi*RADIUS)
        outer, core = bor.sphere_generatrix(RADIUS, 130), bor.sphere_generatrix(.07, 91)
        supplied = dict(points_outer=outer, points_core=core, freq_hz=frequency, thetas_deg=[0., 60., 120.],
                        eps_r=2.56-.1j, mu_r=1., n_modes=20, gauss_order=4, workers=1, assembly='auto')
        layout, radius = dispatch._direct_surface_layout(supplied)
        plan = dispatch._direct_dense_plan(supplied, layout, 20, 4)
        self.assertEqual(plan['assumed_assembly'], 'tables')  # the coated solver's own decision

        class Reached(Exception):
            pass
        started, gated = [], []
        guard = bor._guard_bor_dense_memory
        def gate(*args, **kwargs):
            gated.append((current_options()['factorization'], 'streaming' if kwargs.get('streaming') else 'tables'))
            return guard(*args, **kwargs)
        def stop(solver, *args, **kwargs):
            started.append((current_options()['factorization'], 'tables' if solver._stream is None else 'streaming'))
            raise Reached()
        # Round 10: the chooser prices tables with the run-time model (3.19 GB here,
        # gate 3.01 GB), so it never admits a table plan the gate rejects. The
        # streamed plan is priced at 2.33 GB (gate 2.30 GB), so the streamed window
        # runs from 2.33 GB to the 3.01 GB table gate: its bottom (2.35 GB) and
        # top (2.9 GB, where an explicit dense request for tables is rejected).
        # The run-time transition stays covered by letting the limit drop between
        # the two (last case: planned under 4 GB, admitted under 2.9 GB).
        cases = ((2., 2., ('compressed', None), [('compressed', 'tables')]),
                 (2.35, 2.35, ('dense', 'streaming'), [('dense', 'streaming')]),
                 (2.9, 2.9, ('dense', 'streaming'), [('dense', 'streaming')]),
                 (4., 4., ('dense', None), [('dense', 'tables')]),
                 (4., 2.9, ('dense', None), [('dense', 'tables'), ('dense', 'streaming')]))
        for planning_limit, limit, chosen, gates in cases:
            with self.subTest(planning_limit=planning_limit, limit=limit):
                del started[:], gated[:]
                with patch.object(td, '_solve_memory_limit_gb', return_value=planning_limit), \
                        patch.object(bor, '_solve_memory_limit_gb', return_value=limit), \
                        patch.object(bor, '_guard_bor_dense_memory', gate), \
                        patch.object(bor.BorPecSolver, 'prepare_operators', stop):
                    self.assertEqual(dispatch.resolve_automatic_plan(dict(supplied)), chosen)
                    self.assertEqual(dispatch._resolve_direct_factorization(dict(supplied)), chosen[0])
                    with self.assertRaises(Reached):
                        bor.solve_bor_coated_pec(outer, core, frequency, [0., 60., 120.], 2.56-.1j, 1.,
                                                 n_modes=20, gauss_order=4, workers=1, assembly='auto')
                    # An explicit factorization keeps its rejection.
                    if planning_limit == limit == 2.9:
                        # The near-preparation phase remains larger than this
                        # limit. Check the model's own reported requirement;
                        # removing unused dense bases changes its rounded GB.
                        with self.assertRaises(bor.BorAdmissionError) as rejected:
                            bor.solve_bor_coated_pec(outer, core, frequency, [0., 60., 120.], 2.56-.1j, 1.,
                                                     n_modes=20, gauss_order=4, workers=1, assembly='tables',
                                                     bor_options=dict(factorization='dense'))
                        self.assertGreater(rejected.exception.required_gb, limit)
                        self.assertIn(f'{rejected.exception.required_gb:.2f} GB', str(rejected.exception))
                self.assertEqual(gated[:len(gates)], gates)
                self.assertEqual(started[0], gates[-1])  # every rejection preceded preparation

        # An explicit assembly with the factorization left automatic (Codex's
        # round-7 review): it is never exchanged for the other dense assembly,
        # but 'auto' still authorizes compression, as it always has.
        del started[:], gated[:]
        with patch.object(td, '_solve_memory_limit_gb', return_value=2.9), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=2.9), \
                patch.object(bor, '_guard_bor_dense_memory', gate), \
                patch.object(bor.BorPecSolver, 'prepare_operators', stop):
            self.assertEqual(dispatch.resolve_automatic_plan(dict(supplied, assembly='tables')), ('compressed', None))
            self.assertEqual(dispatch.resolve_automatic_plan(dict(supplied, assembly='streaming')), ('dense', None))
        with patch.object(td, '_solve_memory_limit_gb', return_value=4.), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=2.9), \
                patch.object(bor, '_guard_bor_dense_memory', gate), \
                patch.object(bor.BorPecSolver, 'prepare_operators', stop):
            with self.assertRaises(Reached):
                bor.solve_bor_coated_pec(outer, core, frequency, [0., 60., 120.], 2.56-.1j, 1.,
                                         n_modes=20, gauss_order=4, workers=1, assembly='tables')
        self.assertEqual(gated, [('dense', 'tables'), ('compressed', 'tables')])  # no streamed attempt
        self.assertEqual(started, [('compressed', 'tables')])
        with patch.object(td, '_solve_memory_limit_gb', return_value=2.):
            self.assertEqual(dispatch.resolve_automatic_plan(dict(supplied, assembly='streaming')), ('compressed', None))

    def test_single_precision_keeps_its_memory_diagnostic(self):
        # Codex's round-6 reproduction. Compression needs double precision, so
        # the round-6 fallback turned this memory rejection into a ValueError.
        arguments = dict(points=bor.sphere_generatrix(RADIUS, 24), freq_hz=5e8, thetas_deg=[0., 60.],
                         n_modes=6, workers=1)
        # The solvers normalize the spelling, so the chooser must too (round-7 review).
        for spelling in ('single', ' Single '):
            with self.subTest(table_precision=spelling):
                raised = {}
                with patch.object(td, '_solve_memory_limit_gb', return_value=.5), \
                        patch.object(bor, '_solve_memory_limit_gb', return_value=.5):
                    self.assertEqual(dispatch.resolve_automatic_plan(dict(arguments, table_precision=spelling)),
                                     ('dense', None))
                    for factorization in ('auto', 'dense'):
                        with self.assertRaises(MemoryError) as caught:
                            bor.solve_bor(**arguments, table_precision=spelling,
                                          bor_options=dict(factorization=factorization))
                        raised[factorization] = caught.exception
                self.assertNotIsInstance(raised['auto'], ValueError)
                # October 2026: a conductor's automatic assembly is the streamed one,
                # so the single-precision request has exactly one dense plan; its
                # rejection is the diagnostic (no second plan to report beside it).
                self.assertEqual(raised['auto'].required_gb, raised['dense'].required_gb)
                self.assertEqual(raised['auto'].mode_cap, 6)
                self.assertTrue(raised['auto'].streaming)
                self.assertIsNone(raised['auto'].__cause__)
                self.assertIsNone(raised['dense'].__cause__)

    def test_only_admission_rejections_change_the_plan(self):
        # An allocation failure after preparation is not an admission rejection:
        # round 6 caught it too, discarded the prepared operators and started over.
        prepared = []
        prepare = bor.BorPecSolver.prepare_operators
        def monitor(solver, *args, **kwargs):
            prepared.append(current_options()['factorization'])
            return prepare(solver, *args, **kwargs)
        with patch.object(bor.BorPecSolver, 'prepare_operators', monitor), \
                patch.object(bor.BorPecSolver, 'assemble_mode', side_effect=MemoryError('allocation failed mid-solve')):
            with self.assertRaisesRegex(MemoryError, 'allocation failed mid-solve') as caught:
                bor.solve_bor(bor.sphere_generatrix(RADIUS, 24), 5e8, [0., 60.], n_modes=6, workers=1)
        self.assertNotIsInstance(caught.exception, BorAdmissionError)
        self.assertEqual(prepared, ['dense'])
        from ghost_backend.bor.streaming import StreamingMemoryError
        budget = StreamingMemoryError('below the one-mode minimum')  # raised while planning
        self.assertIsInstance(budget, BorAdmissionError)
        self.assertTrue(budget.streaming)

    def test_fallback_plans_keep_an_extended_mode_cap(self):
        # A rejection that only appears at an extended automatic cap must not
        # send the next plan back to the cap already known to be too small.
        calls = []
        @configured
        def calculation(freq_hz=1e9, n_modes=None, assembly='auto', table_precision='auto'):
            backend = current_options()['factorization']
            calls.append((backend, assembly, n_modes))
            if n_modes is None:
                raise ModalConvergenceError('tail unconverged', 12)
            if backend == 'dense':
                raise BorAdmissionError('too large', streaming=assembly == 'streaming', required_gb=9.)
            return {'mode_converged': True}
        result = calculation(bor_options={'factorization': 'auto'})
        self.assertEqual(calls, [('dense', 'auto', None), ('dense', 'auto', 24),
                                 ('dense', 'streaming', 24), ('compressed', 'tables', 24)])
        self.assertEqual(result['automatic_mode_cap_extensions'], [24])
        fallback = result['automatic_factorization_fallback']
        self.assertEqual((fallback['rejected'], fallback['used'], fallback['assembly']), ('dense', 'compressed', None))
        self.assertEqual([(item['factorization'], item['assembly']) for item in fallback['rejections']],
                         [('dense', 'tables'), ('dense', 'streaming')])
        self.assertEqual(result['bor_execution_options']['factorization'], 'compressed')
        del calls[:]
        with self.assertRaises(BorAdmissionError):  # single precision never reaches compression
            calculation(table_precision='single', bor_options={'factorization': 'auto'})
        self.assertEqual([call[:2] for call in calls[1:]], [('dense', 'auto'), ('dense', 'streaming')])
        # An explicit assembly is kept for the dense plan and never exchanged for
        # the other one; automatic factorization may still compress.
        for assembly in ('tables', 'streaming'):
            del calls[:]
            result = calculation(n_modes=24, assembly=assembly, bor_options={'factorization': 'auto'})
            self.assertEqual(calls, [('dense', assembly, 24), ('compressed', 'tables', 24)])
            self.assertNotIn('automatic_assembly', result)
            self.assertEqual(len(result['automatic_factorization_fallback']['rejections']), 1)

    def test_requirements_from_an_obsolete_mode_cap_are_not_reported(self):
        # Codex's round-7 scenario: tables are rejected at cap 24, the streamed
        # plan is admitted there but needs cap 36, where it and compression are
        # rejected. Cap 24 is known not to converge, so its 3.0 GB is no advice.
        calls = []
        @configured
        def calculation(freq_hz=1e9, n_modes=None, assembly='auto', table_precision='auto'):
            backend = current_options()['factorization']
            cap = 12 if n_modes is None else n_modes
            calls.append((backend, assembly, cap))
            if cap == 12 or (cap == 24 and assembly == 'streaming'):
                raise ModalConvergenceError('cap {} is insufficient'.format(cap), cap)
            need = (cap/8. if assembly == 'auto' else 5.) if backend == 'dense' else 4.2
            raise BorAdmissionError('needs {} GB'.format(need), streaming=assembly == 'streaming', required_gb=need)
        with self.assertRaises(BorAdmissionError) as caught:
            calculation(bor_options={'factorization': 'auto'})
        self.assertEqual(calls, [('dense', 'auto', 12), ('dense', 'auto', 24), ('dense', 'streaming', 24),
                                 ('dense', 'streaming', 36), ('compressed', 'tables', 36)])
        error = caught.exception
        self.assertEqual((error.required_gb, error.mode_cap, error.expanded_caps), (4.2, 36, [24, 36]))
        self.assertIn('this requirement is for compressed/tables at mode cap 36', str(error))
        self.assertIn('dense/tables at mode cap 24: needs 3.0 GB', str(error.__cause__))
        self.assertIn('dense/streaming at mode cap 36: needs 5.0 GB', str(error.__cause__))

    def test_started_process_pool_serves_later_small_calls(self):
        # Idle process workers stay resident until the preparation ends; a later
        # thread team sized without them would exceed the admitted plan
        # (6 thread scratches + 2 idle workers = 3.007 GB against 2.5 GB).
        from ghost_backend.bor import near_parallel
        from ghost_backend.bor.options import option_scope, validate_options
        created = []
        class Pool:
            def __init__(self, max_workers, **kwargs):
                created.append(max_workers)
            def shutdown(self, **kwargs):
                pass
        with option_scope(validate_options(dict(near_backend='auto'))), \
                patch.object(near_parallel, 'process_capable', return_value=True), \
                patch.object(near_parallel, 'ProcessPoolExecutor', Pool):
            # Headroom is (2.5 - (0.5 + 1.2 * 0.2)) / 1.2 GB: the dense peak of the
            # mode phase is not charged while the near preparation runs.
            plan = bor.plan_near_preparation(8, .2, .2, 2.5)
            self.assertEqual((plan['workers'], plan['process_workers']), (7, 3))
            with near_parallel.process_scope(plan['workers'], plan['process_workers']) as state:
                self.assertIsNone(near_parallel.executor_for(100, 24))      # small first: threads
                pool = near_parallel.executor_for(1000, 24)
                self.assertIsNotNone(pool)
                self.assertIs(near_parallel.executor_for(100, 24), pool)   # small later: same pool
                self.assertEqual((created, state['jobs']), ([3], 1100))

    def test_first_cap_extension_follows_the_measured_tail(self):
        error = ModalConvergenceError('unconverged', 72, tail=8.14e-6, tolerance=1e-6)
        self.assertEqual(next_mode_cap(error, 0), 84)
        self.assertEqual(next_mode_cap(error, 1), 108)
        self.assertEqual(next_mode_cap(ModalConvergenceError('unconverged', 72), 0), 108)
        slow = ModalConvergenceError('unconverged', 72, tail=.3, tolerance=1e-6)
        self.assertEqual(next_mode_cap(slow, 0), 72+27)
        self.assertEqual(next_mode_cap(ModalConvergenceError('unconverged', 2, tail=.5, tolerance=1e-6), 0), 14)


def _snapshot_chain(name, seg_type, points, material=0):
    return dict(name=name, seg_type=seg_type, properties=[str(seg_type), '1', '0', str(material), '0'],
                point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                             for a, b in zip(points[:-1], points[1:])])


class BorSnapshotPlanTests(unittest.TestCase):
    def test_snapshot_chooser_prices_streaming_before_compression(self):
        # The coated sphere of the direct-call tests as a geometry snapshot, with
        # its far Gauss points kept (173/121 elements at three points = 130/91 at
        # four). The preview prices its tables at 3.21 GB (3.19 GB for 130/91 at
        # four points; 8.59 GB before round 10 gave it the run-time model) and a
        # streamed plan at 2.48 GB; every limit in between used to mean
        # compression, which Codex and I measured at 18 minutes against 2 to 3
        # for streaming.
        from ghost_backend.runs.bor_setup import resource_summary
        def coated(outer, core):
            return dict(segments=[_snapshot_chain('outer', 3, bor.sphere_generatrix(RADIUS, outer), 1),
                                  _snapshot_chain('core', 4, bor.sphere_generatrix(.07, core), 1)],
                        ibcs=[], dielectrics=[['1', '2.56', '-0.1', '1.0', '0.0']])
        snapshot = coated(173, 121)
        frequency = 12.*td.C0/(2*np.pi*RADIUS*1e9)
        aspects = [0., 60., 120.]
        arguments = dict(geometry_snapshot=snapshot, frequencies_ghz=[frequency], elevations_deg=aspects,
                         geometry_units='meters', n_modes=20, workers=1)
        # Single precision has no compressed plan but both dense ones (round-8
        # review). Halved tables without the reverse cross made them the smaller
        # single-precision plan of the smaller body, so these cases use 200/140
        # elements, whose 1.95 GB of far tables still keep the solvers' own
        # tables decision: tables are priced at 2.59 GB, the streamed plan at
        # 2.39 GB.
        single = dict(arguments, geometry_snapshot=coated(200, 140))
        for limit, assembly, precision, chosen in (
                (10., 'auto', 'auto', ('dense', None)),
                (2.8, 'auto', 'auto', ('dense', 'streaming')),
                (2., 'auto', 'auto', ('compressed', None)),
                (2.8, 'tables', 'auto', ('compressed', None)),   # an explicit assembly is not exchanged
                (2.5, 'auto', ' single ', ('dense', 'streaming')),
                (2.4, 'tables', 'single', ('dense', None)),      # explicit: left to the solve's gate
                (9., 'auto', 'single', ('dense', None)),
                # Neither fits the preview: the smaller plan is left to the solve's gate.
                (2., 'auto', 'single', ('dense', 'streaming'))):
            body = single if precision.strip() == 'single' else arguments
            with self.subTest(limit=limit, assembly=assembly, precision=precision), \
                    patch.object(td, '_solve_memory_limit_gb', return_value=limit):
                plan = dispatch.resolve_automatic_plan(dict(body, assembly=assembly, table_precision=precision))
                self.assertEqual(plan, chosen)
                self.assertEqual(dispatch.resolve_automatic_factorization(
                    dict(body, assembly=assembly, table_precision=precision)), chosen[0])
        small = dict(segments=[_snapshot_chain('sphere', 2, bor.sphere_generatrix(.05, 10))], ibcs=[], dielectrics=[])
        with patch.object(td, '_solve_memory_limit_gb', return_value=.6):   # tables far below the streamed 2.09 GB
            self.assertEqual(dispatch.resolve_automatic_plan(dict(
                geometry_snapshot=small, frequencies_ghz=[1.], elevations_deg=aspects, geometry_units='meters',
                workers=1, table_precision='single')), ('dense', None))
        # Preview and solve resolve the same plan, so an automatic preview prices
        # what will run; the GUI summary resolves it itself and must agree. It
        # uses the automatic mode cap (23 here). The solvers stream far tables
        # above 2 GB by their own rule; 226 elements keep tables (about 1 GB of
        # far tables; the plan is priced at 2.57 GB, streamed 2.37 GB).
        snapshot = coated(133, 93)
        with patch.object(td, '_solve_memory_limit_gb', return_value=2.45), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=2.45):
            tables = dispatch.estimate_bor_resources(snapshot, frequency, aspects, geometry_units='meters', workers=1,
                                                     mesh_certification=False, bor_options=dict(factorization='dense'))
            self.assertEqual(tables['assembly_estimate'], 'tables')
            self.assertGreater(tables['estimated_peak_gb'], 2.45)
            preview = dispatch.estimate_bor_resources(snapshot, frequency, aspects, geometry_units='meters', workers=1,
                                                      mesh_certification=False, bor_options=dict(factorization='auto'))
            self.assertEqual((preview['assembly_estimate'], preview['automatic_assembly']), ('streaming', 'streaming'))
            self.assertLessEqual(preview['estimated_peak_gb'], 2.45)
            summary = resource_summary(snapshot, None, dict(
                schema='grim.bor-run-setup', version=1, frequencies_ghz=[frequency], aspects_deg=aspects,
                units='meters', mesh_certification=False, accuracy='tight', cfie_alpha=.5,
                bor_options=dict(factorization='auto')))
        self.assertIn('auto \u2192 dense (streamed far blocks) factorization', summary)
        forecast = float(summary.split('estimated peak ')[1].split(' GB')[0])
        self.assertLessEqual(forecast, 2.45)

    def test_single_precision_snapshot_streams_instead_of_failing_admission(self):
        # Codex's round-8 reproduction, single precision. The chooser returned
        # before pricing, the solve ran tables and its gate rejected them,
        # although the streamed plan is admitted.  Codex used 135/95 elements;
        # with the reverse cross operators derived instead of stored (and three
        # far Gauss points) their tables are the smaller plan, so this uses
        # 200/140 elements, whose 1.95 GB of far tables keep the solve's own
        # tables decision: gates 2.38 GB (tables) and 2.35 GB (streamed),
        # chooser prices 2.59 and 2.39 GB, and the limit sits between the gates
        # at 2.365 GB.
        snapshot = dict(segments=[_snapshot_chain('outer', 3, bor.sphere_generatrix(RADIUS, 200), 1),
                                  _snapshot_chain('core', 4, bor.sphere_generatrix(.07, 140), 1)],
                        ibcs=[], dielectrics=[['1', '2.56', '-0.1', '1.0', '0.0']])
        frequency = 12.*td.C0/(2*np.pi*RADIUS*1e9)

        class Admitted(Exception):
            pass
        gates = []
        guard = bor._guard_bor_dense_memory
        def gate(*args, **kwargs):
            gates.append((bool(kwargs.get('streaming')), guard(*args, **kwargs)))
            raise Admitted()
        with patch.object(td, '_solve_memory_limit_gb', return_value=2.365), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=2.365), \
                patch.object(bor, '_guard_bor_dense_memory', gate):
            with self.assertRaises(Admitted):
                dispatch.solve_monostatic_rcs_bor(snapshot, [frequency], [0., 60., 120.], geometry_units='meters',
                                                  n_modes=20, workers=1, table_precision='single')
            self.assertEqual([streaming for streaming, _ in gates], [True])
            self.assertLess(gates[0][1], 2.365)
            # An explicit request for tables is not exchanged and keeps its memory rejection.
            del gates[:]
            with self.assertRaisesRegex(MemoryError, '2.38 GB'):
                dispatch.solve_monostatic_rcs_bor(snapshot, [frequency], [0., 60., 120.], geometry_units='meters',
                                                  n_modes=20, workers=1, table_precision='single', assembly='tables')

    def test_run_setup_summary_resolves_its_plan_under_the_callers_options(self):
        # Codex's round-8 reproduction: 181 aspects in batches of 256 cost more
        # than in the default batches of 64. Resolved under the defaults, the
        # summary announced a streamed dense plan (forecast over the limit) for a
        # solve that chooses compression.
        import os
        from ghost_backend.bor.options import validate_options
        from ghost_backend.runs.bor_setup import resource_summary
        snapshot = dict(segments=[_snapshot_chain('outer', 3, bor.sphere_generatrix(RADIUS, 133), 1),
                                  _snapshot_chain('core', 4, bor.sphere_generatrix(.07, 93), 1)],
                        ibcs=[], dielectrics=[['1', '2.56', '-0.1', '1.0', '0.0']])
        frequency = 12.*td.C0/(2*np.pi*RADIUS*1e9)
        aspects = np.linspace(0., 180., 181).tolist()
        arguments = dict(geometry_snapshot=snapshot, frequencies_ghz=[frequency], elevations_deg=aspects,
                         geometry_units='meters', workers=7, mesh_certification=False)
        # Under the phase-wise memory model the near-preparation phase bounds
        # this small body at both batch sizes (the same streamed plan, 2.51 GB
        # under a 2.6 GB limit that the one-worker table plan's 2.67 GB fails),
        # so the batch no longer flips the plan; the summary must still resolve
        # it under the caller's batch, which the recorded option scope shows.
        seen = []
        real_resolve = dispatch.resolve_automatic_plan
        def resolving(plan_arguments, certified=False):
            seen.append(current_options()['angle_batch_size'])
            return real_resolve(plan_arguments, certified)
        with patch.object(os, 'cpu_count', return_value=8), \
                patch.object(td, '_solve_memory_limit_gb', return_value=2.6), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=2.6), \
                patch.object(dispatch, 'resolve_automatic_plan', side_effect=resolving):
            summaries = {}
            for batch in (64, 256):
                options = validate_options(dict(angle_batch_size=batch))
                with option_scope(options):
                    solve_plan = dispatch.resolve_automatic_plan(arguments)  # what a public call resolves
                summaries[batch] = solve_plan, resource_summary(snapshot, None, dict(
                    schema='grim.bor-run-setup', version=1, frequencies_ghz=[frequency], aspects_deg=aspects,
                    units='meters', mesh_certification=False, accuracy='standard', cfie_alpha=.5, bor_options=options))
        self.assertEqual(seen, [64, 64, 256, 256])
        for batch in (64, 256):
            self.assertEqual(summaries[batch][0], ('dense', 'streaming'))
            self.assertIn('auto \u2192 dense (streamed far blocks) factorization', summaries[batch][1])
            self.assertLessEqual(float(summaries[batch][1].split('estimated peak ')[1].split(' GB')[0]), 2.6)

    def test_imposed_streaming_reaches_the_snapshot_solvers(self):
        snapshot = dict(segments=[_snapshot_chain('sphere', 2, bor.sphere_generatrix(.05, 10))], ibcs=[], dielectrics=[])
        common = dict(geometry_units='meters', workers=1)
        # October 2026: conductors stream by default; tables remain an explicit choice.
        tables = dispatch.solve_monostatic_rcs_bor(snapshot, [1.], [0., 60.], bor_options=dict(factorization='dense'),
                                                   assembly='tables', **common)
        self.assertEqual(tables['metadata']['per_frequency'][0]['assembly'], 'tables')
        self.assertNotIn('automatic_assembly', tables)
        # No small body has a natural window (streaming is its larger plan), so
        # the chooser's answer is supplied; the entries must carry it through.
        with patch.object(dispatch, 'resolve_automatic_plan', return_value=('dense', 'streaming')):
            for entry in (dispatch.solve_monostatic_rcs_bor, dispatch.solve_monostatic_rcs_bor_survey):
                with self.subTest(entry=entry.__name__):
                    streamed = entry(snapshot, [1.], [0., 60.], bor_options=dict(factorization='auto'), **common)
                    metadata = streamed['metadata']
                    self.assertEqual(metadata['per_frequency'][0]['assembly'], 'streaming')
                    self.assertEqual((streamed['automatic_assembly'], metadata['automatic_assembly'],
                                      metadata['assembly_requested']), ('streaming', 'streaming', 'auto'))
                    np.testing.assert_allclose([row['rcs_linear'] for row in streamed['samples']],
                                               [row['rcs_linear'] for row in tables['samples']], rtol=1e-9)


class BorGeometryGuardTests(unittest.TestCase):
    @staticmethod
    def _rim(angle_degrees, count=6):
        height = RADIUS*math.tan(math.radians(angle_degrees)/2)
        up = np.column_stack([np.linspace(0, RADIUS, count+1), np.linspace(height, 0, count+1)])
        down = np.column_stack([np.linspace(RADIUS, 0, count+1), np.linspace(0, -height, count+1)])[1:]
        return np.vstack([up, down])

    def test_threshold_is_inclusive_within_rounding(self):
        bor_geometry.require_resolved_corners(self._rim(15.))
        bor_geometry.require_resolved_corners(self._rim(15.+1e-9))
        with self.assertRaisesRegex(ValueError, 'not validated below 15 degrees'):
            bor_geometry.require_resolved_corners(self._rim(14.9))

    def test_axial_tips_are_not_rim_corners(self):
        # A 5 degree half-angle cone: measured quadrature sensitivity 0.012 dB.
        tip = np.array([[0., .2], [.0175, 0.], [0., 0.]])
        bor_geometry.require_resolved_corners(tip)

    def test_partial_layouts_check_stitched_surfaces_but_not_junction_wedges(self):
        from ghost_backend.tests.test_bor_physics_regression import _partial_coating_snapshot
        # The repository fixture ends its coating in a 12.5 degree wedge; its
        # physics regressions pass and no refined-rule evidence exists for
        # junctions, so that angle is not a rejection criterion.
        shape = _partial_coating_snapshot()
        estimate = dispatch.estimate_bor_resources(shape, 1., [0., 90.], geometry_units='meters',
            workers=2, mesh_certification=False)
        self.assertEqual(estimate['geometry_kind'], 'partial')
        # An acute corner between two chains of the SAME bare conductor is a rim.
        bare = next(s for s in shape['segments'] if s['seg_type'] == 2)
        pairs = bare['point_pairs']
        start, end = (pairs[0]['x1'], pairs[0]['y1']), (pairs[-1]['x2'], pairs[-1]['y2'])
        spike = (start[0]+.03, start[1]-.002)  # about 7.6 degrees included
        def chain(name, points):
            return dict(name=name, seg_type=2, properties=list(bare['properties']),
                        point_pairs=[dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1]) for a, b in zip(points[:-1], points[1:])])
        shape['segments'] = [s for s in shape['segments'] if s is not bare] + [
            chain('out', [start, spike]), chain('back', [spike, (start[0]+.0005, start[1]-.004), end])]
        with self.assertRaisesRegex(ValueError, 'not validated below 15 degrees'):
            dispatch.estimate_bor_resources(shape, 1., [0., 90.], geometry_units='meters',
                workers=2, mesh_certification=False)


class KnownDefectTests(unittest.TestCase):
    def test_coarse_lossless_dielectric_pole_mode_does_not_corrupt_axial_backscatter(self):
        # Recorded as an expected failure until round 10: on a coarse lossless
        # PMCHWT mesh the |m| = 1 system had a node-to-node alternating current
        # localized at the two poles whose singular value crossed zero at
        # isolated frequencies. 20 elements, eps_r = 3, radius 0.1 m: condition
        # estimate about 1.8e8 and +6.7 dB here. The interior equations now carry
        # a unit-modulus weight (see test_september_round10_fixes.py).
        frequency = 1.0977737284e9
        result = bor.solve_bor_dielectric(bor.sphere_generatrix(RADIUS, 20), frequency, [0.], 3., 1.,
            n_modes=3, workers=1, bor_options=dict(factorization='dense'))
        error = 10*math.log10(result['sigma_vv'][0]/sigma_dielectric_sphere(RADIUS, 3., 1., frequency))
        self.assertLess(abs(error), .5)


if __name__ == '__main__':
    unittest.main()
