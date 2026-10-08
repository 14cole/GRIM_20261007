"""Regressions for independently reproduced September 2026 audit failures."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver as td
from ghost_backend.bor import solver as bor
from ghost_backend.bor.options import option_scope, validate_options
from ghost_backend.validation.cylinder import sigma_dielectric_cylinder


def segment(points, kind=1, flag=1, density=-20):
    return dict(name='part', seg_type=kind,
        properties=[str(kind), str(density), str(flag), '0', '0'],
        point_pairs=[dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1])
                     for a, b in zip(points, points[1:])])


def strip(split=False, reverse=False, distinct_names=True):
    parts = [segment([(-.15, 0), (0, 0), (.15, 0)])]
    if split:
        points = [(0, 0), (.15, 0)]
        parts = [segment([(-.15, 0), (0, 0)]),
                 segment(points[::-1] if reverse else points, flag=2)]
        if distinct_names:
            parts[0]['name'], parts[1]['name'] = 'left_sheet', 'right_sheet'
    return dict(segments=parts, ibcs=[[str(i), 'constant', '100', '0', '100', '0']
                                    for i in (1, 2)], dielectrics=[])


def circle(radius=.05, count=24, eps=None, density=0):
    angle = np.linspace(0, -2*np.pi, count+1)
    points = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
    part = segment(points, kind=2 if eps is None else 3, flag=0, density=density)
    part['properties'][3] = '0' if eps is None else '1'
    return dict(segments=[part], ibcs=[], dielectrics=[] if eps is None else
                [['1', str(complex(eps).real), str(complex(eps).imag), '1', '0']])


def solve(shape, frequency=1., pol='TE', **kwargs):
    options = kwargs.pop('execution_options', dict(factorization='dense', mesh_strategy='global'))
    return td.solve_monostatic_rcs_2d_single_polarization(shape, [frequency], [30., 60.], pol,
        geometry_units='meters', solver_method='experimental_cpu' if options['factorization']=='compressed' else 'direct',
        execution_options=options, **kwargs)


def field(result):
    return np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in result['samples']])


class SheetTopologyTests(unittest.TestCase):
    def test_impedance_flags_do_not_break_te_trace_or_change_phase(self):
        for degree in (1, 2, 3):
            options = dict(factorization='dense', mesh_strategy='global', basis_order=degree)
            reference = field(solve(strip(), execution_options=options))
            for distinct in (False, True):
                for reverse in (False, True):
                    with self.subTest(degree=degree, reverse=reverse, distinct_names=distinct):
                        result = solve(strip(True, reverse, distinct), execution_options=options)
                        np.testing.assert_allclose(field(result), reference, rtol=1e-10, atol=1e-12)

    def test_compressed_sheet_uses_same_continuous_topology(self):
        reference = field(solve(strip()))
        result = solve(strip(True), execution_options=dict(factorization='compressed',
                       mesh_strategy='global', compressed_storage_mib=64))
        np.testing.assert_allclose(field(result), reference, rtol=2e-7, atol=1e-10)

    def test_attached_te_sheet_is_rejected_before_returning_a_field(self):
        shape = strip()
        shape['segments'] = [segment([(-.05,-.05),(-.05,.05),(.05,.05),(.05,0),
                                      (.05,-.05),(-.05,-.05)], kind=2, flag=0),
                             segment([(.05,0),(.15,0)])]
        with self.assertRaisesRegex(ValueError, 'coupled junction condition'):
            solve(shape)

    def test_attachment_inside_a_pec_panel_is_also_rejected(self):
        shape = strip()
        shape['segments'] = [segment([(-.05,-.05),(-.05,.05),(.05,.05),
                                      (.05,-.05),(-.05,-.05)], kind=2, flag=0, density=1),
                             segment([(.05,0),(.15,0)], density=1)]
        # Geometry validation may reject the intersection before meshing.
        with self.assertRaisesRegex(ValueError, 'unsupported segment intersection|coupled junction condition'):
            solve(shape)


class ResonanceAndMeshTests(unittest.TestCase):
    def test_explicit_reference_cannot_coarsen_single_frequency(self):
        shape = circle()
        reference = solve(shape, 12.)
        result = solve(shape, 12., mesh_reference_ghz=1.)
        self.assertEqual(result['metadata']['panel_count'], reference['metadata']['panel_count'])
        np.testing.assert_allclose(field(result), field(reference), rtol=1e-12)

    def test_combined_potential_removes_original_neumann_resonances(self):
        for ka, eps, radius, count in ((1.8412,4-.2j,.05,256), (3.0542,4-.2j,.05,256),
                                      (10.515,2.56,.1,512), (10.52,2.56,.1,512),
                                      (10.525,2.56,.1,512)):
            frequency = ka*td.C0/(2*np.pi*radius)/1e9
            for pol in ('TE', 'TM'):
                with self.subTest(ka=ka, pol=pol):
                    result = solve(circle(radius,count,eps,density=1), frequency, pol,
                                   compute_condition_number=True)
                    truth = sigma_dielectric_cylinder(radius,eps,1,frequency*1e9,pol)
                    errors = [10*np.log10(r['rcs_linear']/truth) for r in result['samples']]
                    self.assertLess(max(abs(np.array(errors))), .01)
                    self.assertLess(result['metadata']['condition_est_max'], 1e5)

    def test_native_dielectric_coefficients_match_dense_combined_potential(self):
        from ghost_backend.twod.formulations.dielectric import assemble_system
        from ghost_backend.compressed.coefficients import NativeOracle
        shape = circle(.05, 24, 2.56, density=1)
        material = td.MaterialLibrary.from_entries([], shape['dielectrics'], '.')
        panels = td._build_panels(shape, 1., td.C0/1e9)
        k = 2*np.pi*1e9/td.C0
        for pol in ('TE', 'TM'):
            infos = td._build_coupled_panel_info(panels,material,1.,pol,k)
            mesh,_ = td._build_linear_mesh_interface_aware(panels,infos)
            oracle = NativeOracle(mesh,infos,pol,k,'dielectric')
            matrix = assemble_system(mesh,infos,pol,k)
            ids = np.arange(oracle.n)
            np.testing.assert_allclose(oracle.get(ids,ids), matrix, rtol=2e-9, atol=2e-10)


class BorStorageTests(unittest.TestCase):
    def test_automatic_storage_respects_a_small_job_allocation(self):
        from ghost_backend.compressed.runtime import automatic_storage_bytes
        with patch.object(td, '_solve_memory_limit_gb', return_value=.25):
            budget = automatic_storage_bytes()
        self.assertGreaterEqual(budget, 16*1024**2)
        self.assertLess(budget, .25*1024**3)

    def test_automatic_budget_is_priced_like_the_resolved_explicit_cap(self):
        from ghost_backend.compressed.runtime import automatic_storage_bytes
        with patch.object(td, '_solve_memory_limit_gb', return_value=16.):
            mib = int(automatic_storage_bytes()//1024**2)
            values = []
            for cap in (0, mib):
                with option_scope(validate_options(dict(factorization='compressed', compressed_storage_mib=cap))):
                    values.append(bor.estimate_bor_dense_peak_gb(10000,128,4,20))
            self.assertAlmostEqual(values[0], values[1], delta=.002)
            self.assertGreater(values[0], 10.)

    def test_near_iterator_does_not_eagerly_compute_every_pair(self):
        visited = []
        iterator = bor._iter_near_pairs(lambda p: visited.append(p) or p, list(range(100)), 1)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(visited, [0])
        iterator.close()

    def test_cross_surface_classification_is_translation_invariant(self):
        def pair(offset):
            solvers = []
            for radius in (.05, .0499):
                gen = bor.sphere_generatrix(radius, 12)
                gen[:,1] += offset
                solvers.append(bor.BorPecSolver(gen, 1e9))
            return bor.BorCrossOperators(*solvers)
        # Name is the public class used by coated/multiregion assembly.
        a, b = pair(0.), pair(1e6)
        self.assertEqual(a.near_pairs, b.near_pairs)
        self.assertEqual(a.pair_kind, b.pair_kind)


class BandedFftTests(unittest.TestCase):
    def test_complex_modal_coefficients_match_large_uniform_fft(self):
        from ghost_backend.bor import kernels
        # Close rings, disparate radii, and a nearly axial point; all are far
        # in the quadrature sense here and are compared to an 8192-point FFT.
        rp = np.array([.1,.1,.001,.03,.05])
        rq = np.array([.101,.2,.1,.031,.08])
        zp = np.array([0.,0.,0.,.2,-.2])
        zq = np.array([0.,.3,.1,.3,.4])
        args = (rp,zp,np.full(5,.6),np.full(5,.8),rq,zq,np.full(5,-.8),np.full(5,.6))
        for k in (20.,20.-3j):
            for kind in ('g','mfie','ibc'):
                with self.subTest(k=k,kind=kind):
                    coordinates = (rp,zp,rq,zq) if kind=='g' else args
                    got = kernels.banded_modal_kernels(kind,coordinates,k,12,np.zeros(5,bool),work_bytes=1e6)
                    if kind=='g':
                        truth = kernels.modal_kernels_fft(*coordinates,k,12,n_xi=8192)
                    elif kind=='mfie':
                        truth = kernels.mfie_kernels_fft(*coordinates,k,12,n_xi=8192)
                    else:
                        # Existing IBC routine computes the Cartesian product.
                        raw = kernels.ibc_kernels_fft(*coordinates,k,12,n_xi=8192)
                        truth = tuple(v[np.arange(5),np.arange(5)] for v in raw)
                    np.testing.assert_allclose(got,truth,rtol=2e-9,atol=2e-9)

    def test_streamed_complex_fields_match_uniform_grid(self):
        from ghost_backend.bor import kernels
        values = []
        for banded in (False,True):
            with patch.object(kernels,'BANDED_FFT',banded):
                values.append(bor.solve_bor_dielectric(bor.sphere_generatrix(.05,24),1e9,
                    [0.,30.,90.],2.56-.02j,1.,assembly='streaming',workers=1,
                    bor_options={'factorization':'dense'}))
        for key in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(values[0][key],values[1][key],rtol=2e-8,atol=1e-10)


class BorPlanningTests(unittest.TestCase):
    def test_large_junction_table_estimate_keeps_auto_precision_double(self):
        from ghost_backend.bor import dispatch
        from test_bor_physics_regression import _partial_coating_snapshot
        def table_bytes(layout, modes, single_tables=False):
            return 5. if single_tables else 10.
        estimates = []
        with patch.object(dispatch, '_estimate_multisurface_operator_gb', side_effect=table_bytes):
            for precision in ('auto', 'double', 'single'):
                estimates.append(dispatch.estimate_bor_resources(_partial_coating_snapshot(),
                    1., [0.,90.], geometry_units='meters', n_modes=8,
                    mesh_certification=False, assembly='tables', table_precision=precision,
                    bor_options={'factorization':'dense'}))
        self.assertEqual(estimates[0]['table_precision_estimate'], 'double')
        self.assertEqual(estimates[0]['estimated_peak_gb'], estimates[1]['estimated_peak_gb'])
        self.assertEqual(estimates[2]['table_precision_estimate'], 'single')
        self.assertGreater(estimates[0]['estimated_peak_gb'], estimates[2]['estimated_peak_gb'])

    def test_auto_does_not_hide_geometry_validation(self):
        from ghost_backend.bor.dispatch import resolve_automatic_factorization
        with patch('ghost_backend.bor.dispatch.estimate_bor_resources',side_effect=ValueError('bad geometry')):
            with self.assertRaisesRegex(ValueError,'bad geometry'):
                resolve_automatic_factorization(dict(geometry_snapshot={},frequencies_ghz=[1.],elevations_deg=[30.]))

    def test_direct_api_auto_accounts_for_geometry(self):
        from ghost_backend.bor.dispatch import resolve_automatic_factorization
        with patch.object(td,'_solve_memory_limit_gb',return_value=.1):
            mode = resolve_automatic_factorization(dict(points=bor.sphere_generatrix(1.,100),
                freq_hz=10e9,thetas_deg=[90.],workers=1,gauss_order=4))
        self.assertEqual(mode,'compressed')

    def test_auto_modes_expand_but_explicit_cap_is_preserved(self):
        from ghost_backend.bor.options import configured, ModalConvergenceError
        calls = []
        @configured
        def calculation(freq_hz=1e9,n_modes=None):
            calls.append(n_modes)
            if n_modes is None or n_modes<20:
                raise ModalConvergenceError('tail unconverged',12 if n_modes is None else n_modes)
            return {'mode_converged':True}
        result = calculation(bor_options={'factorization':'dense'})
        self.assertEqual(calls,[None,24])
        self.assertEqual(result['automatic_mode_cap_extensions'],[24])
        with self.assertRaises(ModalConvergenceError):
            calculation(n_modes=12,bor_options={'factorization':'dense'})


if __name__ == '__main__':
    unittest.main()
