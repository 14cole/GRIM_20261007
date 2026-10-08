"""Numerical and bounded-work contracts for the large-system audit fixes."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import scipy.linalg as la

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver
from ghost_backend.twod.formulations import combined_regions


class SolverOptimizationTests(unittest.TestCase):
    def test_only_cubic_pass_collects_refinement_indicators(self):
        from ghost_backend.tests.general_fixtures import fixture
        from ghost_backend.twod import adaptivity
        shape = fixture('rectangle', 4)
        for segment in shape['segments']:
            segment['properties'][1] = '-100'
        degrees = []
        original = adaptivity.Indicators.observe
        def observe(owner, mesh, density):
            degrees.append(len(mesh.elements[0].node_ids)-1)
            return original(owner, mesh, density)
        with mock.patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS', 0), \
                mock.patch.object(adaptivity.Indicators, 'observe', observe):
            result = solver.solve_monostatic_rcs_2d_certified(shape, [1.], [0., 67.],
                geometry_units='meters', execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertIn(3, degrees)
        self.assertNotIn(2, degrees)
        self.assertFalse(result['metadata']['adaptive_mesh']['steps'][0]['indicators_computed'])

    def test_condition_block_probes_match_scaled_inverse_and_adjoint(self):
        rng = np.random.default_rng(318)
        a = rng.normal(size=(12, 12)) + 1j*rng.normal(size=(12, 12)) + 12*np.eye(12)
        a = np.logspace(-2, 2, 12)[:, None]*a*np.logspace(1, -1, 12)[None, :]
        rows, columns, norm = solver._equilibrated_scaling_and_norm_1(a)
        lu, piv = la.lu_factor(a)
        probes = rng.normal(size=(12, 3)) + 1j*rng.normal(size=(12, 3))
        expected = columns[:, None]*np.linalg.solve(a, np.diag(rows))
        calls = []

        def checked_solve(rhs, trans=0):
            calls.append((rhs.shape, trans))
            return la.lu_solve((lu, piv), rhs, trans=trans)

        def examine(operator):
            np.testing.assert_allclose(operator.matmat(probes), expected@probes, rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(operator.rmatmat(probes), expected.conj().T@probes, rtol=2e-13, atol=2e-13)
            return np.linalg.norm(expected, 1)

        with mock.patch.object(solver, '_deterministic_onenormest', side_effect=examine):
            actual = solver._equilibrated_condition_from_lu(a, None, None, checked_solve,
                                                            scaling=(rows, columns, norm))
        self.assertEqual(calls, [((12, 3), 0), ((12, 3), 2)])
        self.assertAlmostEqual(actual, norm*np.linalg.norm(expected, 1), places=12)

    def test_batched_condition_preserves_scalar_estimator_value(self):
        rng = np.random.default_rng(65)
        a = rng.normal(size=(20, 20)) + 1j*rng.normal(size=(20, 20)) + 20*np.eye(20)
        lu, piv = la.lu_factor(a)
        rows, columns, norm = solver._equilibrated_scaling_and_norm_1(a)
        from scipy.sparse.linalg import LinearOperator
        old = LinearOperator(a.shape,
            matvec=lambda x: columns*la.lu_solve((lu, piv), rows*np.asarray(x).reshape(-1)),
            rmatvec=lambda x: rows*la.lu_solve((lu, piv), columns*np.asarray(x).reshape(-1), trans=2),
            dtype=complex)
        expected = norm*solver._deterministic_onenormest(old)
        actual = solver._equilibrated_condition_from_lu(a, lu, piv, scaling=(rows, columns, norm))
        np.testing.assert_allclose(actual, expected, rtol=5e-14)

    def test_no_exterior_double_layer_returns_no_projection(self):
        mesh = SimpleNamespace(nodes=np.zeros((4, 2)))
        layout = dict(region_props={0: dict(has_incident=True), 1: dict(has_incident=False)})
        for couplings in ({}, {(0, 1): 2j}):
            with mock.patch.object(combined_regions, 'couplings', return_value=couplings):
                self.assertIsNone(combined_regions.exterior_double_density(mesh, layout))

    def test_exterior_double_layer_retains_incident_routes(self):
        mesh = SimpleNamespace(nodes=np.zeros((4, 2)))
        layout = dict(region_props={0: dict(has_incident=True), 1: dict(has_incident=False)},
            ifaces=[dict(nodes=np.array([0, 2]), r_m=0)], dof_map={(0, 'minus'): (1, 2)})
        solution = np.arange(12).reshape(4, 3).astype(complex)
        with mock.patch.object(combined_regions, 'couplings', return_value={(0, 0): 2j, (0, 1): 3j}):
            density = combined_regions.exterior_double_density(mesh, layout)(solution)
        expected = np.zeros((4, 3), complex)
        expected[[0, 2]] = 2j*solution[1:3]
        np.testing.assert_array_equal(density, expected)


if __name__ == '__main__':
    unittest.main()
