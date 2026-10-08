"""Numerical and lifecycle contracts of bounded 2-D reuse and near storage."""
import copy
import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.twod import solver, operators, polynomial_quadrature as polynomial
from ghost_backend.twod.assembly import kernels, near_store
from ghost_backend.twod.preparation import prepare_geometry
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.compressed.coefficients import NativeOracle, PairedNativeOracle
from ghost_backend.twod.formulations.dielectric import assemble_system
from test_experimental_cpu import fixture


def mesh_for(kind='lossy', degree=1):
    snapshot = fixture(kind, 24)
    with execution_scope(validate_options(dict(basis_order=degree, assembly_threads=1, blas_threads=1))):
        _, _, materials, scale = prepare_geometry(snapshot, None, 'meters')
        wave, _, _ = solver._mesh_wavelength_for_snapshot(snapshot, materials, .6)
        panels = solver._build_panels(snapshot, scale, wave)
        k = 2*np.pi*.6e9/solver.C0
        preview = solver._build_coupled_panel_info(panels, materials, .6, 'TE', k)
        mesh = solver._build_linear_mesh_interface_aware(panels, preview)[0]
        infos = solver._build_linear_coupled_infos(mesh, materials, .6, 'TE', k)
    return mesh, infos, k


class ProjectionTests(unittest.TestCase):
    def test_cache_matches_each_basis_potential_mask_and_respects_cap(self):
        rng = np.random.RandomState(235)
        obs = np.linspace(-178., 180., 37)
        for degree in (1, 2, 3):
            mesh, _, k = mesh_for('pec', degree)
            rho = rng.randn(len(mesh.nodes), 7)+1j*rng.randn(len(mesh.nodes), 7)
            for budget in (0, 500, 1024**2):
                with self.subTest(degree=degree, budget=budget):
                    plan = kernels.GridProjection(mesh, k, obs, budget_bytes=budget)
                    for mask in (None, np.arange(len(mesh.elements)) % 2 == 0):
                        for potential in ('SLP', 'DLP', 'SLP'):
                            expected = kernels.farfield(mesh, rho, k, obs, potential, element_mask=mask, projection='grid')
                            actual = kernels.farfield(mesh, rho, k, obs, potential, element_mask=mask,
                                projection='grid', prepared_projection=plan)
                            np.testing.assert_array_equal(actual, expected)
                    self.assertLessEqual(plan.bytes, budget)
                    if budget == 1024**2:
                        self.assertEqual(plan.builds, 2)
                        self.assertEqual(plan.hits, 4)
                    else:
                        self.assertEqual(plan.hits, 0)
                    with self.assertRaisesRegex(ValueError, 'another mesh'):
                        kernels.farfield(mesh, rho, k+1, obs, 'SLP', projection='grid', prepared_projection=plan)

    def test_bistatic_solver_reuses_observation_moments_with_same_fields(self):
        args = dict(geometry_snapshot=fixture('lossy', 24), frequencies_ghz=[.6],
            incidence_angles_deg=[0., 17., 43., 91., 121., 180.],
            observation_angles_deg=[0., 42., 80., 180.], geometry_units='meters',
            execution_options=dict(assembly_threads=1, blas_threads=1))
        real, plans = kernels.GridProjection, []
        def create(*args):
            plan = real(*args)
            plans.append(plan)
            return plan
        with mock.patch('ghost_backend.twod.fields.configured_batch_size', return_value=2):
            with mock.patch.object(kernels, 'GridProjection', side_effect=lambda *a: real(*a, budget_bytes=0)):
                reference = solver.solve_bistatic_rcs_2d(**args)
            with mock.patch.object(kernels, 'GridProjection', side_effect=create):
                actual = solver.solve_bistatic_rcs_2d(**args)
        for channel in ('VV', 'HH'):
            a, b = actual['co_solved_samples'][channel], reference['co_solved_samples'][channel]
            np.testing.assert_array_equal([[r['rcs_amp_real'], r['rcs_amp_imag']] for r in a],
                                          [[r['rcs_amp_real'], r['rcs_amp_imag']] for r in b])
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(p.builds == 1 and p.hits == 5 for p in plans))

    def test_compression_hint_lives_across_frequency_factors_but_not_runs(self):
        import ghost_backend.linalg.sweep as sweep
        seen, real = [], sweep.solve
        def observe(factor, rhs, basis, **kwargs):
            seen.append(kwargs.get('hint'))
            return real(factor, rhs, basis, **kwargs)
        args = dict(geometry_snapshot=fixture('pec', 24), frequencies_ghz=[.6, .7],
            elevations_deg=[0., 40.], geometry_units='meters',
            execution_options=dict(assembly_threads=1, blas_threads=1))
        with mock.patch.object(sweep, 'solve', side_effect=observe):
            solver.solve_monostatic_rcs_2d(**args)
            first = list(seen)
            seen.clear()
            solver.solve_monostatic_rcs_2d(**args)
        self.assertEqual(len(first), 4)
        self.assertIsNotNone(first[0])
        self.assertTrue(all(h is first[0] for h in first))
        self.assertTrue(all(h is seen[0] for h in seen))
        self.assertIsNot(first[0], seen[0])


class FusedDielectricTests(unittest.TestCase):
    def test_arbitrary_tiles_and_paired_queries_match_dense_both_polarizations(self):
        mesh, infos, k = mesh_for()
        n = len(mesh.nodes)
        first = NativeOracle(mesh, infos, 'TE', k, 'dielectric')
        second = NativeOracle(mesh, infos, 'TM', k, 'dielectric')
        reference = [assemble_system(mesh, infos, p, k) for p in ('TE', 'TM')]
        rows = np.array([n+4, 2, n+1, 8, n-1])
        columns = np.array([0, n+2, n+3, 13])
        pair = PairedNativeOracle(first, second)
        with mock.patch.object(operators, '_assemble_multi', wraps=operators._assemble_multi) as assemble:
            result = pair.get_with_error(rows, columns)
        self.assertEqual(assemble.call_count, 2)
        for (value, error), expected in zip(result, reference):
            np.testing.assert_allclose(value, expected[np.ix_(rows, columns)], rtol=2e-12, atol=2e-15)
            np.testing.assert_array_equal(error, 0.)
        self.assertIsNone(first.query_cache)
        self.assertIsNone(second.query_cache)
        # Empty and single equation-block queries must not create phantom rows.
        for rr, cc in (([], [1]), ([1], []), ([n+1, n+2], [n+4]), ([0, 1], [2, 4])):
            value = first.get(rr, cc)
            np.testing.assert_allclose(value, reference[0][np.ix_(rr, cc)], rtol=2e-12, atol=2e-15)

    def test_unequal_rules_keep_independent_primitive_path(self):
        mesh, infos, k = mesh_for()
        oracle = NativeOracle(mesh, infos, 'TE', k, 'dielectric', obs_order=5, src_order=8)
        ids = np.arange(oracle.n)
        with mock.patch.object(oracle, '_dielectric_tile', side_effect=AssertionError('unequal rules must not fuse')):
            actual = oracle.get(ids, ids)
        expected = assemble_system(mesh, infos, 'TE', k, obs_order=5, src_order=8)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-15)


class NearStorageTests(unittest.TestCase):
    def test_moment_cache_is_reserved_during_factor_and_solve_too(self):
        from ghost_backend.compressed.memory import forecast
        resources = dict(formulation='single_dielectric', basis_width=2)
        with execution_scope(validate_options(dict(factorization='dense', assembly_threads=1, blas_threads=1))):
            linear = solver._estimate_memory_gb(800, False, system_dofs=1600, dense_resources=resources)
            higher = solver._estimate_memory_gb(800, False, system_dofs=1600,
                dense_resources=dict(resources, basis_width=4))
        self.assertEqual((higher-linear)*1024**3, polynomial.MOMENT_CACHE_BYTES)
        linear = forecast(800, 1600, 64, 64, 1, 512*1024**2, resources)
        higher = forecast(800, 1600, 64, 64, 1, 512*1024**2, dict(resources, basis_width=4))
        for phase in ('assembly', 'factorization', 'solve'):
            self.assertEqual(higher['phase_bytes'][phase]-linear['phase_bytes'][phase], polynomial.MOMENT_CACHE_BYTES)
        self.assertEqual(higher['projection_cache_bytes'], kernels.PROJECTION_CACHE_BYTES)

    def test_disk_records_reverse_reads_truncation_and_cleanup(self):
        values = np.arange(12*16).reshape(12, 4, 4)+2j
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('ghost_backend.execution.options.temporary_directory', return_value=directory):
            with near_store.NearStore(12, 4, ['S', 'K'], budget_bytes=0) as store:
                for start in range(0, 12, 3):
                    store.write('S', start, values[start:start+3])
                    store.write('K', start, 2*values[start:start+3])
                file = store.file
                self.assertEqual(store.memory_bytes, 0)
                positions = [8, 2, 3, 4, 2, 11]
                np.testing.assert_array_equal(store.read('K', positions), 2*values[positions])
                np.testing.assert_array_equal(store.read('S', []), np.empty((0,4,4), complex))
                file.truncate(8)
                with self.assertRaisesRegex(OSError, 'incomplete'):
                    store.read('K', [0])
            self.assertTrue(file.closed)
            self.assertEqual(list(Path(directory).iterdir()), [])
            def cancel():
                raise InterruptedError('cancel')
            with self.assertRaises(InterruptedError):
                with near_store.NearStore(12, 4, ['S'], checkpoint=cancel, budget_bytes=0) as failed:
                    failed.write('S', 0, values)
            self.assertTrue(failed.file.closed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_forced_small_near_batches_and_disk_preserve_fused_coefficients(self):
        for degree in (1, 2, 3):
            with self.subTest(degree=degree):
                mesh, infos, k = mesh_for(degree=degree)
                reference = assemble_system(mesh, infos, 'TE', k)
                real, sizes, stores = operators._near_pair_blocks, [], []
                real_store = near_store.NearStore
                def integrate(*args, **kwargs):
                    sizes.append(len(args[1]))
                    return real(*args, **kwargs)
                def store(*args, **kwargs):
                    value = real_store(*args, **kwargs)
                    stores.append(value)
                    return value
                with mock.patch.object(near_store, 'NEAR_STORAGE_BYTES', 0), \
                        mock.patch.object(near_store, 'NEAR_INTEGRATION_PAIRS', 7), \
                        mock.patch.object(near_store, 'NearStore', side_effect=store), \
                        mock.patch.object(operators, '_near_pair_blocks', side_effect=integrate):
                    actual = assemble_system(mesh, infos, 'TE', k)
                np.testing.assert_array_equal(actual, reference)
                self.assertTrue(sizes and max(sizes) <= 7)
                self.assertTrue(stores and all(s.file.closed for s in stores))

    def test_assembly_failure_closes_partial_near_spool(self):
        mesh, infos, k = mesh_for()
        real_store, real_integrate = near_store.NearStore, operators._near_pair_blocks
        stores, calls = [], []
        def create(*args, **kwargs):
            value = real_store(*args, **kwargs)
            stores.append(value)
            return value
        def integrate(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise InterruptedError('cancel during near integration')
            return real_integrate(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('ghost_backend.execution.options.temporary_directory', return_value=directory), \
                mock.patch.object(near_store, 'NEAR_STORAGE_BYTES', 0), \
                mock.patch.object(near_store, 'NEAR_INTEGRATION_PAIRS', 7), \
                mock.patch.object(near_store, 'NearStore', side_effect=create), \
                mock.patch.object(operators, '_near_pair_blocks', side_effect=integrate):
            with self.assertRaises(InterruptedError):
                assemble_system(mesh, infos, 'TE', k)
            self.assertTrue(stores and all(s.file.closed for s in stores))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_adaptive_polynomial_stages_are_bounded_and_match_recursive_rule(self):
        mesh, _, _ = mesh_for('pec', 2)
        a, b = mesh.elements[0], copy.deepcopy(mesh.elements[0])
        b.p0 = b.p0 + .04*b.length*b.normal
        b.p1 = b.p1 + .04*b.length*b.normal
        b.center = b.center + .04*b.length*b.normal
        b.panel_index = -1
        pairs = [(a, b), (mesh.elements[1], b), (a, a)]*3
        expected = [polynomial.near_block(o, s, 37.5) for o, s in pairs]
        real, sizes, depths = polynomial._moments, [], []
        def moments(tasks, *args, **kwargs):
            sizes.append(len(tasks))
            depths.extend(t.depth for t in tasks)
            return real(tasks, *args, **kwargs)
        with mock.patch.object(polynomial, '_ACTIVE_TASKS', 2), \
                mock.patch.object(polynomial, '_moments', side_effect=moments):
            actual = polynomial.near_blocks(pairs, 37.5, threads=1)
        self.assertLessEqual(max(sizes), 2)
        self.assertGreater(max(depths), 0)
        for value, reference in zip(actual, expected):
            for x, y in zip(value, reference):
                np.testing.assert_allclose(x, y, rtol=2e-10, atol=2e-15)


if __name__ == '__main__':
    unittest.main()
