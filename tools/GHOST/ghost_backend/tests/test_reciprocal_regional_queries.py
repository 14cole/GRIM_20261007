"""Opposite tiles share physical kernels, never assume equation symmetry."""
import unittest
from unittest import mock
import numpy as np
from ghost_backend.compressed import regional_coefficients as rc
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.twod.assembly.native import far as native
from ghost_backend.twod.formulations.regions import dof_coordinates
from ghost_backend.tests.test_compact_multi_region import prepared


class ReciprocalRegionalQueriesTests(unittest.TestCase):
    def setUp(self):
        scope = execution_scope(validate_options(dict(assembly_threads=1, blas_threads=1, basis_order=3)))
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_opposite_material_routes_match_independent_queries(self):
        rng = np.random.default_rng(921)
        for kind in ('pec', 'ibc', 'lossy', 'magnetic', 'coated', 'layered', 'mixed', 'equal_k'):
            mesh, te, _ = prepared(kind, 'TE', 16)
            _, tm, _ = prepared(kind, 'TM', 16)
            oracle = rc.PairedOracle(mesh, te, tm, cut=32)
            n = oracle.n
            for missing in ([0, 1], [0], [1]):
                with self.subTest(kind=kind, missing=missing):
                    rows, cols = rng.permutation(n)[:n//3], rng.permutation(n)[:n//4]
                    requests = [(rows, cols, missing), (cols, rows, missing)]
                    reference = [{index: oracle.oracles[index].get_with_error(rr, cc)
                                  for index in requested} for rr, cc, requested in requests]
                    actual = rc.reciprocal_values(oracle, requests)
                    for result, expected in zip(actual, reference):
                        self.assertEqual(set(result), set(expected))
                        for index in result:
                            np.testing.assert_allclose(result[index][0], expected[index][0],
                                                       rtol=3e-11, atol=2e-17)
                            np.testing.assert_array_equal(result[index][1], expected[index][1])
                            self.assertIsNone(result[index][2])

    def test_native_far_kernel_point_pairs_are_reused(self):
        mesh, infos, _ = prepared('pec', 'TE', 64)
        oracle = rc.PreparedOracle(mesh, infos, 'TE', cut=32)
        xy = dof_coordinates(mesh, oracle.layout)
        span = np.ptp(xy[:, 0])
        rows = np.flatnonzero(xy[:, 0] < xy[:, 0].min()+.18*span)
        cols = np.flatnonzero(xy[:, 0] > xy[:, 0].max()-.18*span)
        counts = [0]
        original = native.far_block
        def counted(*args, **kwargs):
            value = original(*args, **kwargs)
            if value is not None:
                counts[0] += int(np.count_nonzero(args[8])) * len(args[4])**2
            return value
        from ghost_backend.execution import cpu
        options = validate_options(dict(assembly_threads=1, blas_threads=1, basis_order=3, assembly_tile=16))
        with execution_scope(options), cpu._STATE.override(cpu.CPUState()), mock.patch.object(native, 'far_block', side_effect=counted):
            reference = [oracle.get_with_error(rows, cols), oracle.get_with_error(cols, rows)]
            independent = counts[0]
            counts[0] = 0
            shared = rc.reciprocal_values(oracle, [(rows, cols, [0]), (cols, rows, [0])])
        if not independent:
            self.skipTest('Qualified native far library is unavailable.')
        self.assertLess(counts[0], .75*independent)
        for result, expected in zip(shared, reference):
            np.testing.assert_allclose(result[0][0], expected[0], rtol=3e-11, atol=2e-17)

    def test_custom_oracle_semantics_are_not_bypassed(self):
        class Custom(rc.PreparedOracle):
            pass
        with self.assertRaises(TypeError):
            rc.reciprocal_values(object.__new__(Custom), [])


if __name__ == '__main__':
    unittest.main()
