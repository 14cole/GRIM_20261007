"""Independent Galerkin assembly, ownership and work-count checks for HP reuse."""
import copy
import os
import tempfile
import weakref
import unittest
from unittest import mock
import numpy as np
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.twod import solver as s
from ghost_backend.twod.assembly import polynomial_pair as pp
from ghost_backend.twod.formulations import regions as mr
from ghost_backend.tests.general_fixtures import fixture
from ghost_backend.tests.test_polynomial_basis import mesh_for


def inputs(case='mixed', count=48, degree=2, freq=1.):
    if case == 'junction':
        from ghost_backend.tests.test_2d_capability_acceptance import TopologyAcceptanceTests
        snapshot = TopologyAcceptanceTests._partial_coating(8)
    elif case in ('magnetic', 'layered'):
        from ghost_backend.tests.test_experimental_cpu import fixture as material_fixture
        snapshot = material_fixture(case, count)
    else:
        snapshot = fixture(case, count)
    mesh = mesh_for(snapshot, degree, freq)
    from ghost_backend.twod.preparation import prepare_geometry
    _, _, materials, _ = prepare_geometry(snapshot, None, 'meters')
    infos = s._build_linear_coupled_infos(mesh, materials, freq, 'TE', 2*np.pi*freq*1e9/s.C0)
    return mesh, infos


class PolynomialPairTests(unittest.TestCase):
    def setUp(self):
        enabled = mock.patch.dict(os.environ, GHOST_POLYNOMIAL_PAIR='auto')
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_unset_pair_setting_keeps_the_ordinary_driver(self):
        with mock.patch.dict(os.environ):
            os.environ.pop('GHOST_POLYNOMIAL_PAIR', None)
            with pp.polynomial_pair_scope(3) as owner, \
                    mock.patch.object(pp, 'prepare', side_effect=AssertionError('Optional sharing was enabled')):
                self.assertIsNone(owner)
                result = s.solve_monostatic_rcs_2d(fixture('mixed', 48), [.6], [0., 73.],
                    geometry_units='meters', solver_method='experimental_cpu',
                    execution_options=dict(mesh_strategy='local', basis_order=2, factorization='dense',
                                           assembly_threads=1, blas_threads=1))
                self.assertEqual(result['metadata']['polynomial_degree'], 2)

    def test_nested_component_profile_captures_worker_thread_work(self):
        from concurrent.futures import ThreadPoolExecutor
        from ghost_backend.twod.assembly.profiling import profile_scope, assembly_component
        with profile_scope() as profile:
            @assembly_component('worker')
            def work(value): return value + 1
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(list(pool.map(work, [1, 2])), [2, 3])
            self.assertEqual(profile.calls['worker'], 2)
            self.assertGreaterEqual(profile.seconds['worker'], 0.)

    def test_prolongation_reproduces_quadratics_and_junctions(self):
        from ghost_backend.twod.basis import values, abscissae
        for case in ('rectangle', 'reentrant', 'acute', 'gap', 'dielectric', 'mixed', 'junction', 'magnetic', 'layered'):
            with self.subTest(case=case):
                coarse, infos = inputs(case)
                fine = pp.cubic_mesh(coarse)
                for pol in ('TE', 'TM'):
                    old, new = mr.build_layout(coarse, infos, pol), mr.build_layout(fine, infos, pol)
                    p = pp.prolongation(coarse, fine, old, new)
                    np.testing.assert_allclose(p @ np.ones(p.shape[1]), 1., atol=4e-15)
                    # Every density component has the same physical polynomial,
                    # but distinct interfaces never become connected by P.
                    cx, fx = mr.dof_coordinates(coarse, old), mr.dof_coordinates(fine, new)
                    for axis in (0, 1):
                        for degree in (1, 2):
                            np.testing.assert_allclose(p @ cx[:, axis]**degree,
                                                       fx[:, axis]**degree, atol=4e-15)
                    self.assertLessEqual(np.diff(p.indptr).max(), 3)

    def test_projected_matrix_matches_independent_galerkin_assembly(self):
        from ghost_backend.twod.assembly.scatter import assemble_multi
        options = validate_options(dict(factorization='dense', assembly_threads=1))
        with execution_scope(options):
            for case in ('rectangle', 'reentrant', 'acute', 'gap', 'dielectric', 'mixed', 'junction', 'magnetic', 'layered'):
                coarse, infos = inputs(case)
                fine = pp.cubic_mesh(coarse)
                for pol in ('TE', 'TM'):
                    with self.subTest(case=case, pol=pol):
                        a2, l2 = assemble_multi(coarse, infos, pol, 8, 8)
                        a3, l3 = assemble_multi(fine, infos, pol, 8, 8)
                        p = pp.prolongation(coarse, fine, l2, l3)
                        projected = pp.project_dense(a3, p)
                        relative = np.linalg.norm(projected-a2)/max(np.linalg.norm(a2), 1e-300)
                        self.assertLess(relative, 2e-10)

    def test_dense_pair_builds_only_cubic_kernel_and_reuses_owned_matrix(self):
        from ghost_backend.twod.assembly import scatter
        coarse, infos = inputs('mixed')
        fine = pp.cubic_mesh(coarse)
        options = validate_options(dict(factorization='dense', assembly_threads=1))
        with execution_scope(options), pp.polynomial_pair_scope(3) as pair:
            with mock.patch.object(scatter, 'assemble_multi', wraps=scatter.assemble_multi) as builds:
                a2, _ = mr.assemble_system(coarse, infos, 'TE')
                self.assertEqual(builds.call_count, 1)
                self.assertEqual(len(builds.call_args.args[0].elements[0].node_ids), 4)
                retained = next(iter(pair.pending.values()))[0]
                self.assertEqual(pp.retained_bytes(), retained.nbytes)
                a3, _ = mr.assemble_system(fine, infos, 'TE')
                self.assertIs(a3, retained)
                self.assertEqual(builds.call_count, 1)
                self.assertEqual(pp.retained_bytes(), 0)
                self.assertNotEqual(a2.shape, a3.shape)
            self.assertEqual([e['action'] for e in pair.evidence],
                             ['project_quadratic', 'reuse_cubic_operator'])

    def test_memory_decline_preserves_independent_path(self):
        coarse, infos = inputs()
        with execution_scope(validate_options(dict(factorization='dense'))), pp.polynomial_pair_scope(3) as pair:
            with mock.patch.object(s, '_solve_memory_limit_gb', return_value=.001):
                self.assertIsNone(pp.dense_system(coarse, infos, 'TE'))
            self.assertFalse(pair.pending)
            self.assertEqual(pair.evidence[-1]['action'], 'independent_assembly')
            self.assertGreater(pair.evidence[-1]['required_gib'], pair.evidence[-1]['budget_gib'])

    def test_dense_spool_releases_fine_before_projection_and_restores_exactly(self):
        from ghost_backend.twod.assembly import scatter, dense_pair_storage as ds
        coarse, infos = inputs('mixed')
        fine = pp.cubic_mesh(coarse)
        references, refs, spools = {}, [], []
        assemble = scatter.assemble_multi
        def tracked(*args, **kwargs):
            matrix, layout = assemble(*args, **kwargs)
            refs.append(weakref.ref(matrix))
            references['fine'] = matrix.copy()
            return matrix, layout
        project = ds.project_spooled
        def streamed(spool, p, checkpoint):
            self.assertIsNone(refs[0](), 'Cubic array remains resident during coarse projection')
            spools.append(spool)
            return project(spool, p, checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            options = validate_options(dict(factorization='dense', assembly_threads=1, temporary_directory=directory))
            with execution_scope(options), pp.polynomial_pair_scope(3) as pair, \
                    mock.patch.object(pp, 'DENSE_SPOOL_MIN_BYTES', 0), \
                    mock.patch.object(ds.DensePairSpool, 'block_bytes', 2048), \
                    mock.patch.object(scatter, 'assemble_multi', side_effect=tracked), \
                    mock.patch.object(ds, 'project_spooled', side_effect=streamed):
                a2, _ = mr.assemble_system(coarse, infos, 'TE')
                self.assertEqual(pp.retained_bytes(), 0)
                self.assertGreater(pair.evidence[0]['retained_fine_disk_bytes'], 0)
                self.assertEqual(pair.evidence[0]['fine_storage'], 'disk')
                plan = pp.prepare(coarse, infos, 'TE')
                np.testing.assert_allclose(a2, pp.project_dense(references['fine'], plan['prolongation']),
                                           rtol=2e-12, atol=1e-16)
                a3, _ = mr.assemble_system(fine, infos, 'TE')
                np.testing.assert_array_equal(a3, references['fine'])
                self.assertTrue(spools[0].file.closed)
                self.assertEqual(len(refs), 1)

    def test_dense_spool_cleanup_on_cancel_and_read_failure(self):
        from ghost_backend.twod.assembly.dense_pair_storage import DensePairSpool, project_spooled
        coarse, infos = inputs()
        fine = pp.cubic_mesh(coarse)
        layout = mr.build_layout(fine, infos, 'TE')
        matrix = np.eye(layout['n_dof'], dtype=complex)
        with tempfile.TemporaryDirectory() as directory:
            spool = DensePairSpool(matrix, directory, lambda: None)
            with self.assertRaises(InterruptedError):
                with pp.polynomial_pair_scope(3):
                    pp.remember(fine, infos, 'TE', spool, layout, 'dense')
                    raise InterruptedError('cancelled')
            self.assertTrue(spool.file.closed)
            spool = DensePairSpool(matrix, directory, lambda: None)
            spool.file.truncate(16)
            with pp.polynomial_pair_scope(3) as pair:
                pp.remember(fine, infos, 'TE', spool, layout, 'dense')
                self.assertIsNone(pp.take(fine, infos, 'TE', 'dense'))
                self.assertIn('storage unavailable', pair.evidence[-1]['reason'])
                from ghost_backend.execution.timing_history import _clean_success
                self.assertFalse(_clean_success(dict(adaptive_mesh=dict(polynomial_pair=pair.evidence)), 'dense'))
                self.assertTrue(spool.file.closed)

    def test_dense_spool_disk_admission_declines_without_kernel_work(self):
        from ghost_backend.twod.assembly import scatter
        coarse, infos = inputs()
        with execution_scope(validate_options(dict(factorization='dense'))), pp.polynomial_pair_scope(3) as pair, \
                mock.patch.object(pp, 'DENSE_SPOOL_MIN_BYTES', 0), \
                mock.patch('shutil.disk_usage', return_value=type('Usage', (), {'free': 1})()), \
                mock.patch.object(scatter, 'assemble_multi', side_effect=AssertionError('unadmitted work')):
            self.assertIsNone(pp.dense_system(coarse, infos, 'TE'))
            self.assertFalse(pair.pending)
            self.assertIn('temporary storage', pair.evidence[-1]['reason'])

    def test_spooled_pair_preserves_both_physical_solves_and_checks(self):
        snapshot = fixture('mixed', 48)
        def solve(degree):
            return s.solve_monostatic_rcs_2d(snapshot, [.6], [0., 47., 123.],
                geometry_units='meters', solver_method='experimental_cpu',
                strict_quality_gate=True, compute_condition_number=True,
                execution_options=dict(mesh_strategy='local', basis_order=degree,
                    factorization='dense', assembly_threads=1, blas_threads=1))
        independent = [solve(degree) for degree in (2, 3)]
        with mock.patch.object(pp, 'DENSE_SPOOL_MIN_BYTES', 0), pp.polynomial_pair_scope(3) as pair:
            actual = [solve(degree) for degree in (2, 3)]
            self.assertEqual(pair.evidence[0]['fine_storage'], 'disk')
            self.assertEqual(pair.evidence[-1]['action'], 'reuse_cubic_operator')
        for degree, a, b in zip((2, 3), actual, independent):
            self.assertEqual(a['metadata']['polynomial_degree'], degree)
            self.assertTrue(a['metadata']['condition_est_computed'])
            for pol in ('VV', 'HH'):
                fields = [np.asarray([complex(row['rcs_amp_real'], row['rcs_amp_imag'])
                    for row in result['co_solved_samples'][pol]]) for result in (a, b)]
                if degree == 3:
                    np.testing.assert_array_equal(*fields)
                else:
                    self.assertLess(np.max(abs(fields[0]-fields[1])) / np.max(abs(fields[1])), 2e-11)

    def test_unavailable_spool_directory_is_optional_and_write_failure_is_clean(self):
        from ghost_backend.twod.assembly import dense_pair_storage as ds
        coarse, infos = inputs()
        options = validate_options(dict(factorization='dense', assembly_threads=1))
        with execution_scope(options), pp.polynomial_pair_scope(3) as pair, \
                mock.patch.object(pp, 'DENSE_SPOOL_MIN_BYTES', 0):
            with mock.patch('ghost_backend.execution.options.temporary_directory', side_effect=ValueError('missing')):
                self.assertIsNone(pp.dense_system(coarse, infos, 'TE'))
            with mock.patch.object(ds, 'DensePairSpool', side_effect=OSError('disk quota')):
                self.assertIsNone(pp.dense_system(coarse, infos, 'TE'))
            self.assertFalse(pair.pending)
            self.assertFalse(pair.building)
            self.assertEqual(len(pair.evidence), 2)
            self.assertTrue(all(e['action'] == 'independent_assembly' for e in pair.evidence))
            from ghost_backend.execution.timing_history import _clean_success
            for event, clean in zip(pair.evidence, (True, False)):
                self.assertEqual(_clean_success(dict(adaptive_mesh=dict(polynomial_pair=[event])), 'dense'), clean)

    def test_identity_material_quadrature_and_backend_mismatch_do_not_reuse(self):
        coarse, infos = inputs()
        fine = pp.cubic_mesh(coarse)
        layout = mr.build_layout(fine, infos, 'TE')
        matrix = np.eye(layout['n_dof'], dtype=complex)
        with pp.polynomial_pair_scope(3):
            pp.remember(fine, infos, 'TE', matrix, layout, 'dense')
            changed = copy.deepcopy(infos)
            changed[0].k_plus += .01
            self.assertIsNone(pp.take(fine, changed, 'TE', 'dense'))
            self.assertIsNone(pp.take(fine, infos, 'TE', 'dense', 9, 8))
            self.assertIsNone(pp.take(fine, infos, 'TE', 'compressed'))
            self.assertIs(pp.take(fine, infos, 'TE', 'dense')[0], matrix)

    def test_varying_conductor_law_is_ineligible(self):
        coarse, infos = inputs('rectangle')
        infos = copy.deepcopy(infos)
        infos[0].robin_impedance = 75.+3j
        with pp.polynomial_pair_scope(3) as pair:
            self.assertIsNone(pp.prepare(coarse, infos, 'TE'))
            self.assertIn('uniform conductor law', pair.evidence[-1]['reason'])

    def test_context_releases_pending_on_cancellation(self):
        class Owned:
            nbytes = 1024
            closed = False
            def close(self): self.closed = True
        coarse, infos = inputs()
        fine = pp.cubic_mesh(coarse)
        owned = Owned()
        with self.assertRaises(InterruptedError):
            with pp.polynomial_pair_scope(3):
                pp.remember(fine, infos, 'TE', owned, {}, 'compressed')
                raise InterruptedError('cancelled')
        self.assertTrue(owned.closed)
        self.assertIsNone(pp.current_pair())

    def test_failed_backend_discards_retained_operator_before_retry(self):
        coarse, infos = inputs()
        fine = pp.cubic_mesh(coarse)
        with pp.polynomial_pair_scope(3) as pair:
            pp.remember(fine, infos, 'TE', np.eye(2, dtype=complex), {}, 'dense')
            self.assertGreater(pp.retained_bytes(), 0)
            pp.discard_backend('hierarchical')
            self.assertEqual(pp.retained_bytes(), 0)
            self.assertEqual(pair.evidence[-1]['action'], 'discard_after_backend_rejection')


if __name__ == '__main__':
    unittest.main()
