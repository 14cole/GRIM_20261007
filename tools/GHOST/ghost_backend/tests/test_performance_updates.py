"""Accuracy, ownership and resource contracts for the reconciled performance updates."""
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import numpy as np
from scipy.special import hankel2

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.bor import kernels, streaming, solver as bor
import legacy_near_rules as legacy
from ghost_backend.execution import cpu, options
from ghost_backend.execution.thread_control import threadpool_info, threadpool_limits
from ghost_backend.twod import solver as td, operators as ops
from ghost_backend.twod.assembly import session
from ghost_backend.twod.formulations import dielectric, regions
from ghost_backend.twod.basis import values, derivative_matrix
from test_compact_multi_region import prepared
from test_experimental_cpu import fixture, fields


class BorSamplingTests(unittest.TestCase):
    def test_native_and_parity_match_independent_full_grid_for_complex_media(self):
        rng = np.random.default_rng(821)
        a, b = rng.uniform(-np.pi,np.pi,(2,19))
        points = [rng.uniform(.01,1,19), rng.normal(size=19), np.cos(a), np.sin(a),
                  rng.uniform(.01,1,19), rng.normal(size=19), np.cos(b), np.sin(b)]
        points[0][0] = 0.0  # Axis endpoint.
        xi = rng.uniform(.001,np.pi,(19,57)); weights = rng.uniform(0,.1,xi.shape)
        modes = np.array([-37,-4,-1,0,1,4,37])
        native = streaming._NATIVE
        for wave in (3., 3.-.4j, .02-4j):
            for family in ('mfie','ibc'):
                with self.subTest(wave=wave, family=family):
                    fn = kernels._mfie_brackets if family == 'mfie' else kernels._ibc_brackets_grid
                    # MFIE's public grid wrapper is nested in the near rule;
                    # evaluate its shared-grid reference one point at a time.
                    def numpy_grid(grid):
                        with mock.patch.object(streaming,'_NATIVE',None):
                            if family == 'ibc':
                                return fn(*points,wave,grid)
                            rows = [fn(*(p[i:i+1] for p in points),wave,grid[i]) for i in range(19)]
                            return [np.concatenate([r[j] for r in rows]) for j in range(4)]
                    positive, negative = numpy_grid(xi), numpy_grid(-xi)
                    projected = kernels._project_parity_brackets(positive,weights,xi,modes)
                    reference = legacy._project_pm_brackets(positive,negative,weights,xi,modes)
                    for actual, expected in zip(projected,reference):
                        np.testing.assert_allclose(actual,expected,rtol=5e-13,atol=2e-14)
                    if native is not None and hasattr(native,'near_brackets'):
                        result = kernels._native_brackets(points,wave,xi,True,family)
                        for actual, expected in zip(result,positive):
                            error=np.max(abs(actual-expected))/max(np.max(abs(expected)),1e-300)
                            self.assertLess(error,2e-13)
                    with mock.patch.object(streaming,'_NATIVE',object()):
                        self.assertIsNone(kernels._native_brackets(points,wave,xi,True,family))

    def test_workers_follow_cpu_and_memory_reservations_and_restore(self):
        with mock.patch.object(options.os,'cpu_count',return_value=32):
            roomy=bor.plan_near_preparation(15,1.,1.,16.)
            self.assertEqual(roomy['workers'],15)
            # Near scratch is priced against the retained operators alone: the
            # preparation pool is gone before the mode workspaces exist.
            base=bor.estimate_bor_total_peak_gb(1.,0.)
            limited=bor.plan_near_preparation(15,1.,1.,base+1.2*3.5*roomy['scratch_bytes_per_worker']/1e9)
            self.assertEqual(limited['workers'],3)
            with options.execution_scope({},assembly_threads=2):
                self.assertEqual(bor.plan_near_preparation(15,1.,1.,16.)['workers'],2)
        old=bor._near_preparation_workers(99)
        with self.assertRaisesRegex(RuntimeError,'abort'):
            with bor._NEAR_WORKER_LIMIT.override(11):
                self.assertEqual(bor._near_preparation_workers(99),11)
                raise RuntimeError('abort')
        self.assertEqual(bor._near_preparation_workers(99),old)
        seen=[]
        with mock.patch.object(bor,'_solve_memory_limit_gb',return_value=.6):
            with self.assertRaises(MemoryError):
                bor._mode_sweep(1,[90.],['VV'],0,1e-6,lambda m:None,lambda *a:None,
                               lambda *a:None,prepare=lambda m:seen.append(m))
        self.assertEqual(seen,[])

    def test_preview_and_auto_selection_include_near_scratch(self):
        from ghost_backend.bor import dispatch
        from test_bor_physics_regression import _pec_sphere_snapshot
        with mock.patch.object(bor,'_solve_memory_limit_gb',return_value=16.):
            preview=dispatch.estimate_bor_resources(_pec_sphere_snapshot(explicit_elements=12),
                1.,[90.],geometry_units='meters',workers=3,n_modes=8,
                mesh_certification=False,bor_options={'factorization':'dense'})
        # A parallel request prepares its near pairs on the CPU allocation's
        # physical cores, independently of the mode-worker count (October 2026).
        self.assertEqual(preview['near_preparation']['workers'],
            max(3,min(options.allocated_cpu_budget(),options.physical_core_count())))
        self.assertGreaterEqual(preview['estimated_peak_gb'],
            bor.estimate_bor_total_peak_gb(preview['near_preparation']['scratch_gb'],0.))
        # A tiny matrix can fit the old unpadded estimate but cannot admit the
        # fixed margin and even one near task; auto must not choose dense.
        with mock.patch.object(td,'_solve_memory_limit_gb',return_value=.6):
            selected=dispatch._resolve_direct_factorization(dict(points=bor.sphere_generatrix(.01,4),
                freq_hz=1e8,thetas_deg=[90.],n_modes=3,workers=1))
        self.assertEqual(selected,'compressed')


class TwoDPerformanceTests(unittest.TestCase):
    def test_combined_reuse_matches_fresh_tm_with_polynomial_and_robin_rows(self):
        for kind, degree in (('coated',1),('layered',2),('equal_k',3),('mixed',2)):
            with self.subTest(kind=kind,degree=degree), options.execution_scope({'basis_order':degree}):
                mesh,te,k=prepared(kind,'TE',12)
                _,tm,_=prepared(kind,'TM',12)
                expected,_=regions.assemble_system(mesh,tm,'TM')
                owner=session.AssemblySession()
                with session._SESSION.override(owner):
                    first,_=regions.assemble_system(mesh,te,'TE')
                    actual,_=regions.assemble_system(mesh,tm,'TM')
                self.assertIs(actual,first)
                np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-12)
        mesh,te,k=prepared('layered','TE',12); _,tm,_=prepared('layered','TM',12)
        expected,_=regions.assemble_system(mesh,tm,'TM',9,8)
        with session._SESSION.override(session.AssemblySession()):
            first,_=regions.assemble_system(mesh,te,'TE',9,8)
            actual,_=regions.assemble_system(mesh,tm,'TM',9,8)
        self.assertIsNot(first,actual)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-14)

    def test_fused_dielectric_matches_independent_combined_operators(self):
        from ghost_backend.twod.assembly.mass import add_mass
        for degree in (1,2,3):
            with options.execution_scope({'basis_order':degree}):
                mesh,infos,k=prepared('magnetic','TM',16)
                actual=dielectric.assemble_system(mesh,infos,'TM',k)
                n=len(mesh.nodes); eta=dielectric.coupling(k)
                s,d=td._assemble_linear_operator_matrices(mesh,k,False)
                _,kp=td._assemble_linear_operator_matrices(mesh,k,True)
                w=td._assemble_linear_hypersingular_matrix(mesh,k)
                top=d+eta*s; bottom=w-eta*kp
                add_mass(top,mesh,.5); add_mass(bottom,mesh,.5*eta)
                np.testing.assert_allclose(actual[:n,:n],top,rtol=3e-11,atol=3e-12)
                np.testing.assert_allclose(actual[n:,:n],bottom,rtol=3e-11,atol=3e-12)

    def test_bistatic_tables_preserve_fields_and_release_state_on_abort(self):
        args=(fixture('ibc',40),[.6],[0.,35.],[0.,71.,180.])
        kw=dict(geometry_units='meters',execution_options={'blas_threads':1})
        actual=td.solve_bistatic_rcs_2d_survey(*args,**kw)
        self.assertIn('cpu_kernel_execution',actual['metadata'])
        self.assertIsNone(cpu.current_state())
        from ghost_backend.twod.assembly import kernels as table_kernels
        with mock.patch.object(table_kernels,'select_far_kernels',side_effect=lambda mesh,k,g,h,**kw:(g,h)), \
                mock.patch.object(ops,'_NATIVE_FAR',False):
            reference=td.solve_bistatic_rcs_2d_survey(*args,**kw)
        for pol in ('VV','HH'):
            a,b=fields(actual,pol),fields(reference,pol)
            self.assertLess(np.max(abs(a-b))/np.max(abs(b)),1e-10)
        abort=threading.Event(); abort.set()
        with self.assertRaises(InterruptedError):
            td.solve_bistatic_rcs_2d_survey(*args,abort_event=abort,**kw)
        self.assertIsNone(cpu.current_state())

    def test_w_far_rule_against_independent_high_order_blocks(self):
        rng=np.random.default_rng(64)
        for degree in (1,2,3):
            for wave in (3.,3.-.001j,2.-2j,.01-2j):
                for angle in (0.,.37,1.5707,2.1):
                    length=float(rng.uniform(.001,1.)); tangent=np.array([np.cos(angle),np.sin(angle)])
                    order=ops._graded_w_far_floor(wave,[1.,length],3.,degree)
                    blocks=[]
                    for q in (order,40):
                        x,w=np.polynomial.legendre.leggauss(q); x=(x+1)/2; w=w/2
                        p=np.column_stack((x-.5,0*x)); s=np.array([3.,0.])+(x[:,None]-.5)*length*tangent
                        g=-.25j*hankel2(0,wave*np.linalg.norm(p[:,None,:]-s[None,:,:],axis=2))
                        phi=values(x,degree); mass=(phi.T*w)@g@(phi*w[:,None])
                        derivative=derivative_matrix(degree)
                        blocks.append(derivative.T@mass@derivative-wave**2*length*tangent[0]*mass)
                    self.assertLess(np.max(abs(blocks[0]-blocks[1]))/np.max(abs(blocks[1])),1e-12)
        self.assertEqual(ops._graded_w_far_floor(4.,[1.],3.,1),16)
        self.assertEqual(ops._graded_w_far_floor(1.,[1.],2.99,1),16)
        with options.execution_scope({'far_grading':False}):
            self.assertEqual(ops._graded_w_far_floor(1.,[1.],3.,1),16)


class BlasPolicyTests(unittest.TestCase):
    def test_caps_and_auto_are_effective_at_lu_and_restore(self):
        import scipy.linalg
        def counts():
            return [row['num_threads'] for row in threadpool_info() if row.get('user_api')=='blas']
        before=counts()
        with threadpool_limits(limits=8,user_api='blas'):
            for setting,size,allocation,expected in ((1,5000,8,1),(8,5000,3,3),
                                                     ('auto',128,8,1),('auto',4096,3,3)):
                with options.execution_scope({'blas_threads':setting},assembly_threads=allocation):
                    with options.linear_algebra_threads(size):
                        self.assertTrue(counts())
                        self.assertEqual(set(counts()),{expected})
                        scipy.linalg.lu_factor(np.eye(3,dtype=complex))
                self.assertEqual(set(counts()),{8})
        self.assertEqual(counts(),before)


if __name__=='__main__':unittest.main()
