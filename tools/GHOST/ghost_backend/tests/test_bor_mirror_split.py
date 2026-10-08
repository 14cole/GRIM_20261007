"""Mirror-symmetric BoR modes: even/odd half factors against LU."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.bor import solver as bor_solver
from ghost_backend.bor.factor import MirrorSplit, ModalFactor

C0 = bor_solver.C0
FREQUENCY_HZ = 1.0e9


def _symmetric_system(n_nodes, rng, defect=0.0):
    """A reduced [t; phi] system of ``n_nodes`` nodes with A = R A R (+ ``defect``)."""
    n = 2 * n_nodes
    component = np.repeat([0, 1], n_nodes)
    node = np.tile(np.arange(n_nodes), 2)
    target = component * n_nodes + (n_nodes - 1 - node)
    sign = np.where(component == 0, -1.0, 1.0)
    base = np.eye(n) * 8 + (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(n)
    reflected = sign[:, None] * sign[None, :] * base[np.ix_(target, target)]
    a = 0.5 * (base + reflected)
    if defect:
        a = a + defect * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return a, target, sign


class MirrorSplitTests(unittest.TestCase):
    def test_exact_symmetry_gives_exact_solutions(self):
        rng = np.random.default_rng(7)
        for n_nodes in (40, 41):            # 41: the middle node maps onto itself
            with self.subTest(nodes=n_nodes):
                a, target, sign = _symmetric_system(n_nodes, rng)
                split = MirrorSplit(a, target, sign)
                b = rng.standard_normal((len(a), 3)) + 1j * rng.standard_normal((len(a), 3))
                np.testing.assert_allclose(split.solve(b), np.linalg.solve(a, b), rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(split.solve(b[:, 0], trans=2),
                                           np.linalg.solve(a.conj().T, b[:, 0]), rtol=1e-11, atol=1e-12)
                self.assertEqual(sum(len(lu) for lu, _ in split.factors), len(a))
        with self.assertRaises(ValueError):
            MirrorSplit(a, np.roll(target, 1), sign)

    def test_refinement_absorbs_quadrature_asymmetry_and_lu_replaces_a_broken_split(self):
        rng = np.random.default_rng(8)
        a, target, sign = _symmetric_system(60, rng, defect=1e-9)
        b = rng.standard_normal((len(a), 4)) + 0j
        factor = ModalFactor(a.copy(), 3, True, mirror=(target, sign))
        self.assertEqual(factor.event["backend"], "mirror")
        np.testing.assert_allclose(factor.solve(b), np.linalg.solve(a, b), rtol=1e-11, atol=1e-13)
        self.assertGreater(factor.event["refinement_steps"], 0)
        self.assertLessEqual(factor.event["max_backward_error"], bor_solver.BOR_LINEAR_BACKWARD_ERROR_MAX)
        # An asymmetry far beyond quadrature: the refined halves miss the gate, LU takes over.
        broken, target, sign = _symmetric_system(60, rng, defect=0.3)
        factor = ModalFactor(broken.copy(), 3, False, mirror=(target, sign))
        np.testing.assert_allclose(factor.solve(b), np.linalg.solve(broken, b), rtol=1e-10, atol=1e-12)
        self.assertEqual(factor.event["backend"], "lu")
        self.assertIn("mirror", factor.event["mirror_fallback"])

    def test_mirror_map_of_symmetric_and_asymmetric_generatrices(self):
        radius = 3.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        sphere = bor_solver.BorPecSolver(bor_solver.sphere_generatrix(radius, 60), FREQUENCY_HZ)
        mirror = bor_solver.mirror_map(sphere)
        self.assertIsNotNone(mirror)
        for m in (0, 1, 4):
            target, sign = mirror(m)
            np.testing.assert_array_equal(target[target], np.arange(len(target)))
            np.testing.assert_array_equal(sign[target], sign)
        zs = np.linspace(10.0, 50.0, sphere.gen.n_elems)
        self.assertIsNone(bor_solver.mirror_map(sphere, (zs,)))
        self.assertIsNotNone(bor_solver.mirror_map(sphere, (zs + zs[::-1], None)))
        points = np.asarray(bor_solver.sphere_generatrix(radius, 60))
        cone = points.copy()
        cone[:, 0] *= np.linspace(1.0, 0.6, len(points))
        cone[0, 0] = cone[-1, 0] = 0.0
        self.assertIsNone(bor_solver.mirror_map(bor_solver.BorPecSolver(cone, FREQUENCY_HZ)))
        self.assertTrue(bor_solver.mirror_split_pays(4318, 362))
        self.assertFalse(bor_solver.mirror_split_pays(4318, 7202))

    def test_sphere_solve_matches_lu(self):
        radius = 5.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        points = bor_solver.sphere_generatrix(radius, 160)
        thetas = np.linspace(0.0, 180.0, 11)
        mirrored = bor_solver.solve_bor(points, FREQUENCY_HZ, thetas, workers=2)
        with mock.patch.object(bor_solver, "mirror_map", return_value=None):
            plain = bor_solver.solve_bor(points, FREQUENCY_HZ, thetas, workers=2)
        systems = mirrored["modal_execution"]["systems"]
        self.assertTrue(all(system.get("backend") == "mirror" for system in systems))
        for key in ("amp_vv", "amp_hh"):
            expected = np.asarray(plain[key])
            np.testing.assert_allclose(mirrored[key], expected, rtol=0,
                                       atol=1e-11 * float(np.max(np.abs(expected))))


if __name__ == "__main__":
    unittest.main()
