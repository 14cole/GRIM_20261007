"""Adverse physical cases and resource contracts from the audit exchange."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch
from ghost_backend.bor import kernels, streaming
from ghost_backend.bor.options import configured, option_scope, validate_options
from ghost_backend.twod import solver as td, operators
from ghost_backend.validation.cylinder import sigma_impedance_cylinder, sigma_pec_cylinder
from ghost_backend.validation.sphere import sigma_impedance_sphere, sigma_coated_pec_sphere
from ghost_backend.tests.test_rcs_physics_regression import _circle_segment
from ghost_backend.tests.test_2d_capability_acceptance import _segment


class ResonanceTests(unittest.TestCase):
    def test_discrete_104_panel_te_resonance_pec_and_lossy_ibc(self):
        # The polygon's resonance, not the nearby smooth-circle eigenvalue.
        frequency = 5.13718385499639 * td.C0 / (2*np.pi*.1)
        for z in (0., 75-20j):
            shape = dict(segments=[_circle_segment(.1, 104, 2, ibc=int(bool(z)))],
                ibcs=[['1','constant',str(complex(z).real),str(complex(z).imag),'0','0']] if z else [],
                dielectrics=[])
            truth = (sigma_impedance_cylinder(.1,z,frequency,'TE') if z else
                     sigma_pec_cylinder(.1,frequency,'TE'))
            amplitudes = []
            for backend in ('dense', 'compressed'):
                result = td.solve_monostatic_rcs_2d_single_polarization(shape,
                    [frequency/1e9], [0.], 'TE', geometry_units='meters',compute_condition_number=True,
                    execution_options=dict(factorization=backend, mesh_strategy='global',
                        assembly_threads=1, blas_threads=1, rhs_compression='off'))
                row = result['samples'][0]
                self.assertLess(abs(10*np.log10(row['rcs_linear']/truth)), .02)
                self.assertLess(result['metadata']['condition_est_max'], 1000)
                amplitudes.append(complex(row['rcs_amp_real'],row['rcs_amp_imag']))
            np.testing.assert_allclose(amplitudes[1], amplitudes[0], rtol=2e-5)

    def test_lossy_uniform_ibc_production_at_efie_resonance(self):
        f = 7.145227364003155 * bor.C0 / (2*np.pi*.1)
        shape = dict(segments=[_segment('ibc',2,bor.sphere_generatrix(.1,44),ibc=1)],
            ibcs=[['1','constant','200','0','0','0']],dielectrics=[])
        result = dispatch.solve_monostatic_rcs_bor(shape,[f/1e9],[0.],geometry_units='meters',
            n_modes=3,workers=1,bor_options=dict(factorization='dense'))
        self.assertIn('CFIE',result['metadata']['formulation'])
        truth = sigma_impedance_sphere(.1,f,200.)
        self.assertLess(abs(10*np.log10(result['samples'][0]['rcs_linear']/truth)), .02)
        direct = bor.solve_bor(bor.sphere_generatrix(.1,44),f,[0.],zs=200.,n_modes=3,
            bor_options=dict(factorization='dense'))
        self.assertEqual(direct['formulation'],'cfie')
        self.assertLess(abs(10*np.log10(direct['sigma_vv'][0]/truth)),.02)

    def test_lossless_core_resonance_all_assembly_paths(self):
        f = 4.482362394479282 * bor.C0 / (2*np.pi*.06*np.sqrt(3.))
        outer, core = bor.sphere_generatrix(.1,32), bor.sphere_generatrix(.06,32)
        truth = sigma_coated_pec_sphere(.06,.1,3.,1.,f)
        fields = []
        for assembly, backend in (('tables','dense'), ('streaming','dense'), ('tables','compressed')):
            result = bor.solve_bor_coated_pec(outer,core,f,[0.],3.,n_modes=3,
                assembly=assembly,bor_options=dict(factorization=backend,rhs_compression='off'))
            self.assertLess(abs(10*np.log10(result['sigma_vv'][0]/truth)), .02)
            fields.append(result['amp_vv'][0])
        np.testing.assert_allclose(fields[1:],fields[:1]*2,rtol=2e-5,atol=1e-10)
        generic = bor.solve_bor_coated_n_pec([outer],core,f,[0.],[3.],[1.],n_modes=3,
            bor_options=dict(factorization='dense'))
        np.testing.assert_allclose(generic['amp_vv'][0],fields[0],rtol=2e-8)


class GeometryAndResourcesTests(unittest.TestCase):
    def test_invalid_coating_refused_by_preview_certified_and_direct(self):
        outer = bor.sphere_generatrix(.025,12)
        for core in (bor.sphere_generatrix(.04,12), bor.sphere_generatrix(.01,12)+[0.,.15]):
            shape = dict(segments=[_segment('outer',3,outer,pos=1),
                _segment('core',4,core,pos=1)],ibcs=[],dielectrics=[['1','3','0','1','0']])
            with self.assertRaisesRegex(ValueError,'inside|intersect'):
                dispatch.estimate_bor_resources(shape,1.,[0.],geometry_units='meters')
            with self.assertRaisesRegex(ValueError,'inside|intersect'):
                dispatch.solve_monostatic_rcs_bor_certified(shape,[1.],[0.],geometry_units='meters')
            with self.assertRaisesRegex(ValueError,'inside|intersect'):
                bor.solve_bor_coated_pec(outer,core,1e9,[0.],3.)

    def test_acute_corner_guard_precedes_quadrature(self):
        points = np.array([[0.,.002],[.1,0.],[0.,-.002]])
        with patch.object(bor.BorPecSolver,'prepare_operators',side_effect=AssertionError('assembly')):
            with self.assertRaisesRegex(ValueError,'included angle'):
                bor.solve_bor(points,1e9,[0.],formulation='cfie')

    def test_long_close_panels_route_to_converged_integration(self):
        points = np.array([[0.,.1],[.1,.1],[.1,0.],[.12,0.],
                           [.12,.102],[.1,.102],[0.,.102]])
        solver = bor.BorPecSolver(points,1e8)
        # Panels 0 and 5 overlap radially, 2 mm apart and far apart by index.
        self.assertIn(5,solver._near_sources_by_element[0])
        self.assertIn(0,solver._near_sources_by_element[5])

    def test_output_grid_reservation_can_refuse_before_assembly(self):
        shape = dict(segments=[_segment('sphere',2,bor.sphere_generatrix(.01,8))],
                     ibcs=[],dielectrics=[])
        small = dispatch.estimate_bor_resources(shape,1.,[0.],frequency_count=1,
            geometry_units='meters',mesh_certification=False,bor_options=dict(factorization='dense'))
        large = dispatch.estimate_bor_resources(shape,1.,np.linspace(0,180,2000),frequency_count=1000,
            geometry_units='meters',mesh_certification=False,bor_options=dict(factorization='dense'))
        self.assertGreater(large['estimated_peak_gb']-small['estimated_peak_gb'],8.)

    def test_closed_tapered_ibc_is_not_silently_sent_to_efie(self):
        # Rejected until round 10; the automatic choice is now the varying-impedance CFIE.
        result = bor.solve_bor(bor.sphere_generatrix(.1,12),1e9,[0.],zs=np.linspace(100,200,12),
                               bor_options=dict(factorization='dense'))
        self.assertIn('CFIE',result['formulation'].upper())


class ExecutionTests(unittest.TestCase):
    def test_failed_spool_restore_discards_pending_polarization_matrix(self):
        from ghost_backend.linalg.residual_spool import ResidualSpool
        from ghost_backend.twod.assembly.session import AssemblySession, _SESSION
        shape = dict(segments=[_circle_segment(.04,16,2,ibc=1)],
            ibcs=[['1','constant','75','-20','0','0']],dielectrics=[])
        session = AssemblySession()
        # Round 10: a conductor's TM system is normally assembled in the TE step's
        # traversal, which leaves no TE buffer to restore. This test is about the
        # retained-buffer path, so the pairing is declined here.
        import ghost_backend.twod.assembly.session as assembly_session
        with _SESSION.override(session), patch.object(assembly_session,'plan_paired_assembly',return_value=False), \
                patch.object(ResidualSpool,'restore_into',
                side_effect=InterruptedError('canceled during restore')):
            with self.assertRaisesRegex(InterruptedError,'during restore'):
                td.solve_monostatic_rcs_2d(shape,[1.],[0.],geometry_units='meters',
                    execution_options=dict(factorization='dense',mesh_strategy='global',
                        dense_residual_storage='disk'))
        self.assertIsNone(session.pending)

    def test_disk_residuals_preserve_two_polarization_matrix_reuse(self):
        from ghost_backend.linalg.residual_spool import ResidualSpool
        shape = dict(segments=[_circle_segment(.04,32,2,ibc=1)],
            ibcs=[['1','constant','75','-20','0','0']],dielectrics=[])
        common = dict(geometry_snapshot=shape,frequencies_ghz=[1.],elevations_deg=[0.,45.],
            geometry_units='meters',compute_condition_number=True)
        options = dict(factorization='dense',mesh_strategy='global')
        expected = td.solve_monostatic_rcs_2d(**common,
            execution_options=dict(options,dense_residual_storage='memory'))
        reads = []
        multiply = ResidualSpool.__matmul__
        def record(spool,rhs):
            reads.append(spool.shape[0])
            return multiply(spool,rhs)
        with patch.object(ResidualSpool,'__matmul__',record):
            actual = td.solve_monostatic_rcs_2d(**common,
                execution_options=dict(options,dense_residual_storage='disk'))
        self.assertTrue(reads)
        for channel in ('VV','HH'):
            values = lambda r: [complex(row['rcs_amp_real'],row['rcs_amp_imag'])
                                 for row in r['co_solved_samples'][channel]]
            np.testing.assert_allclose(values(actual),values(expected),rtol=1e-12,atol=1e-12)

    def test_spooled_original_preserves_full_residual_and_condition_evidence(self):
        from ghost_backend.linalg.dense import DenseFactor
        from ghost_backend.execution.options import execution_scope
        rng = np.random.default_rng(42)
        a = np.asarray(rng.normal(size=(80,80))+1j*rng.normal(size=(80,80)),order='F')
        a += 20*np.eye(80)
        rhs = rng.normal(size=(80,3))+1j*rng.normal(size=(80,3))
        original = a.copy()
        diagnostics = {}
        with execution_scope(dict(dense_residual_storage='disk')):
            factor = DenseFactor(a,diagnostics,owned_matrix=True)
            try:
                self.assertTrue(np.shares_memory(factor.lu,a))
                self.assertEqual(factor.event['residual_rows'],'all')
                solution = factor.solve(rhs)
                np.testing.assert_allclose(original@solution,rhs,rtol=1e-12,atol=1e-12)
                residual = np.linalg.norm(original@solution-rhs,axis=0)/np.linalg.norm(rhs,axis=0)
                np.testing.assert_allclose(factor.relative_residual,residual,atol=2e-16)
                self.assertTrue(np.isfinite(diagnostics['condition_est']))
                handle = factor._residual_spool.file
            finally:
                factor.close()
            self.assertTrue(handle.closed)

    def test_automatic_retry_retains_completed_modes_and_tail_history(self):
        seen = []
        @configured
        def solve(freq_hz, n_modes=None):
            cap = 2 if n_modes is None else n_modes
            def assemble(m):
                seen.append(m)
                return np.ones((1,1),complex),None
            field, used, stats = bor._mode_sweep(1,[20.],['VV'],cap,1e-6,
                assemble,lambda m,t,p: np.ones(1,complex),
                lambda m,x,t,p: complex(1 if abs(m)<=4 else 0),workers=1)
            bor._require_mode_convergence(stats,1e-6)
            return dict(field=field,modes_used=used,**stats)
        result = solve(1e9,bor_options=dict(factorization='dense'))
        self.assertEqual(result['automatic_mode_cap_extensions'],[14])
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(result['modes_used'],6)
        self.assertEqual(result['field'][0,0],9.)
        self.assertEqual(len(result['modal_execution']['systems']),13)

    def test_physical_mode_resume_matches_a_fresh_larger_cap(self):
        points = bor.sphere_generatrix(.025,18)
        common = dict(points=points,freq_hz=1e9,thetas_deg=[0.,40.,90.],
            workers=1,bor_options=dict(factorization='dense',rhs_compression='off'))
        reference = bor.solve_bor(**common,n_modes=13)
        limits = bor._bor_mode_limits
        assembled = []
        original = bor.BorPecSolver.assemble_mode
        def small_cap(k,rho,thetas,n_modes):
            cap,floor = limits(k,rho,thetas,n_modes)
            return (1 if n_modes is None else cap),floor
        def record(solver,m,cap,**kwargs):
            assembled.append(m)
            return original(solver,m,cap,**kwargs)
        with patch.object(bor,'_bor_mode_limits',small_cap), patch.object(bor.BorPecSolver,'assemble_mode',record):
            resumed = bor.solve_bor(**common)
        self.assertEqual(resumed['automatic_mode_cap_extensions'],[13])
        self.assertEqual(len(assembled),len(set(assembled)))
        for channel in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(resumed[channel],reference[channel],rtol=2e-5,atol=1e-9)

    def test_process_near_preparation_matches_threads_for_lossy_cross_and_self(self):
        # Spawned workers import the parent's entry module, so processes are
        # only offered under a guarded ``__main__``. Whether the module that
        # imported this test is guarded depends on the runner (pytest's is,
        # ``python -m unittest``'s is not), so exercise process preparation
        # from a controlled guarded child interpreter instead.
        import json, os, subprocess, tempfile, textwrap
        root = str(Path(__file__).resolve().parents[2])
        code = textwrap.dedent("""
            import json, sys
            sys.path.insert(0, ROOT)
            import numpy as np
            from ghost_backend.bor import solver as bor
            from ghost_backend.bor.near_parallel import process_scope, process_capable
            from ghost_backend.bor.options import option_scope, validate_options

            def prepare(backend):
                a = bor.BorPecSolver(bor.sphere_generatrix(.025,6),1e9,near_depth=1,medium=(2.5-.3j,1.))
                b = bor.BorPecSolver(bor.sphere_generatrix(.02,6),1e9,near_depth=1,medium=(2.5-.3j,1.))
                cross = bor.BorCrossOperators(a,b)
                with option_scope(validate_options(dict(near_backend=backend))), process_scope(2) as state:
                    for kind in ('efie','mfie','ibc'):
                        a._prepare_near_contractions(kind,[(1,1),(1,2),(1,4)],3,workers=2)
                    cross.prepare(3,workers=2)
                return a, cross, state['jobs']

            if __name__ == '__main__':
                a, cross, thread_jobs = prepare('threads')
                b, other, process_jobs = prepare('processes')
                worst = 0.
                for key, record in a._near_contractions.items():
                    expected, actual = record['values'], b._near_contractions[key]['values']
                    worst = max(worst, float(np.max(abs(actual-expected))/np.max(abs(expected))))
                for pair, blocks in cross._cache[3].items():
                    for kind, expected in blocks.items():
                        actual = other._cache[3][pair][kind]
                        worst = max(worst, float(np.max(abs(actual-expected))/np.max(abs(expected))))
                print(json.dumps(dict(capable=process_capable(), thread_jobs=thread_jobs,
                                      process_jobs=process_jobs, worst=worst)))
            """).replace('ROOT', repr(root))
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'near_process_probe.py'
            script.write_text(code, encoding='utf-8')
            done = subprocess.run([sys.executable, '-B', str(script)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=900,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertTrue(report['capable'])
        self.assertEqual(report['thread_jobs'], 0)
        self.assertGreater(report['process_jobs'], 0)
        self.assertLessEqual(report['worst'], 2e-13)

    def test_process_workers_require_a_guarded_entry_module(self):
        # An unguarded entry module would rerun its top-level work in every
        # spawned worker, so such hosts must stay on threads. This includes the
        # standard ``python -m unittest`` entry module.
        import tempfile
        from ghost_backend.bor import near_parallel
        # Located by path: importing it would rerun the test runner, which is
        # exactly what a spawned worker would do.
        unittest_main = Path(unittest.__file__).with_name('__main__.py')
        with tempfile.TemporaryDirectory() as folder:
            guarded, unguarded = Path(folder)/'guarded.py', Path(folder)/'unguarded.py'
            guarded.write_text("import os\nif __name__ == '__main__':\n    print(os.getcwd())\n")
            unguarded.write_text("import os\nprint(os.getcwd())\n")
            self.assertTrue(near_parallel._guarded_main(str(guarded)))
            self.assertFalse(near_parallel._guarded_main(str(unguarded)))
        self.assertFalse(near_parallel._guarded_main(str(unittest_main)))
        with option_scope(validate_options(dict(near_backend='processes'))):
            with patch.object(near_parallel,'process_capable',return_value=False):
                plan = bor.plan_near_preparation(8,1.,1.,64.)
                self.assertEqual((plan['process_workers'],plan['process_overhead_bytes_per_worker']),(0,0))
                with near_parallel.process_scope(8,plan['process_workers']):
                    self.assertIsNone(near_parallel.executor_for(10**6,50))

    def test_memory_plan_charges_process_workers_only_when_they_will_run(self):
        from ghost_backend.bor import near_parallel
        shape = dict(segments=[_segment('body',2,bor.sphere_generatrix(.1,48))],ibcs=[],dielectrics=[])
        common = dict(geometry_units='meters',workers=8,mesh_certification=False)
        with patch.object(near_parallel,'process_capable',return_value=True):
            small = dispatch.estimate_bor_resources(shape,1.5,[0.,90.],**common)
            threads = dispatch.estimate_bor_resources(shape,1.5,[0.,90.],**common,
                bor_options=dict(near_backend='threads'))
            forced = dispatch.estimate_bor_resources(shape,1.5,[0.,90.],**common,
                bor_options=dict(near_backend='processes'))
            large = dispatch.estimate_bor_resources(shape,12.,[0.,90.],**common)
        # Automatic selection keeps this small job on threads: no process charge.
        self.assertEqual(small['near_preparation']['process_overhead_bytes_per_worker'],0)
        self.assertAlmostEqual(small['estimated_peak_gb'],threads['estimated_peak_gb'],places=9)
        self.assertGreater(forced['estimated_peak_gb'],small['estimated_peak_gb']+1.)
        self.assertGreater(large['near_preparation']['process_overhead_bytes_per_worker'],0)
        # The executor obeys the same policy and the admitted pool size.
        pairs = small['near_preparation']
        with near_parallel.process_scope(8,pairs['process_workers']):
            self.assertIsNone(near_parallel.executor_for(268,21))

    def test_batched_touching_rule_matches_individual_duffy_rules(self):
        shape = dict(segments=[_circle_segment(.1,12,2)],ibcs=[],dielectrics=[])
        panels = td._build_panels(shape,1.,1.)
        mesh = td._build_linear_mesh(panels)
        pairs = [(mesh.elements[a],mesh.elements[b]) for a,b in ((0,1),(1,0),(0,11),(11,0))]
        shared = [operators._linear_shared_interval_endpoint_info(a,(0.,1.),b,(0.,1.)) for a,b in pairs]
        for k in (31.,31-2j):
            for obs_deriv in (False,True):
                batch = operators._integrate_linear_touching_pairs_sk_batched(pairs,shared,k,obs_deriv,12)
                for i, ((a,b), ends) in enumerate(zip(pairs,shared)):
                    reference = operators._integrate_linear_touching_duffy_sk_vectorized(
                        a,b,k,obs_deriv,(0.,1.),(0.,1.),*ends,order=12)
                    for actual,expected in zip(batch,reference):
                        np.testing.assert_allclose(actual[i],expected,rtol=3e-14,atol=1e-16)


if __name__ == '__main__':
    unittest.main()
