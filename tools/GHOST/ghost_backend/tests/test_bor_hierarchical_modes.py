"""Hierarchical BoR mode factors: same solutions as LU, priced as the smaller factor."""
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.bor import solver as bor_solver
from ghost_backend.bor.factor import ModalFactor
from ghost_backend.linalg import hierarchical as hf

C0 = bor_solver.C0
FREQUENCY_HZ = 1.0e9


class HierarchicalModeTests(unittest.TestCase):
    def test_sphere_solve_matches_lu(self):
        radius = 6.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        points = bor_solver.sphere_generatrix(radius, 200)
        thetas = np.linspace(0.0, 180.0, 13)
        results = {}
        for label, threshold in (("lu", "0"), ("hodlr", "300")):
            # The sphere is mirror symmetric: keep its modes off the mirror split.
            with mock.patch.dict(os.environ, {"GHOST_HIERARCHICAL_MIN_UNKNOWNS": threshold}), \
                 mock.patch.object(bor_solver, "mirror_map", return_value=None):
                results[label] = bor_solver.solve_bor(points, FREQUENCY_HZ, thetas, workers=2)
        systems = results["hodlr"]["modal_execution"]["systems"]
        self.assertTrue(systems)
        self.assertTrue(all(system.get("backend") == "hodlr" for system in systems))
        self.assertFalse(any(system.get("backend") == "hodlr"
                             for system in results["lu"]["modal_execution"]["systems"]))
        for key in ("amp_vv", "amp_hh"):
            expected = np.asarray(results["lu"][key])
            np.testing.assert_allclose(results["hodlr"][key], expected, rtol=0,
                                       atol=1e-11 * float(np.max(np.abs(expected))))
        self.assertAlmostEqual(results["hodlr"]["max_cond"] / results["lu"]["max_cond"], 1.0, places=6)

    def test_rejected_factor_falls_back_to_lu(self):
        rng = np.random.default_rng(4)
        n = 600
        u = rng.standard_normal((n, 6)) + 1j * rng.standard_normal((n, 6))
        a = np.eye(n) * 20 + u @ u.conj().T / n     # compressible off-diagonal blocks
        coordinates = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        b = rng.standard_normal((n, 3)) + 0j
        with mock.patch.dict(os.environ, {"GHOST_HIERARCHICAL_MIN_UNKNOWNS": "100"}):
            factor = ModalFactor(a.copy(), 2, True, coordinates=coordinates)
            self.assertEqual(factor.event["backend"], "hodlr")
            np.testing.assert_allclose(factor.solve(b), np.linalg.solve(a, b), rtol=1e-11, atol=1e-13)
            with mock.patch.object(factor.hierarchical, "solve",
                                   side_effect=hf.HierarchicalRejected("stalled")):
                x = factor.solve(b)
            self.assertEqual(factor.event["backend"], "lu")
            self.assertIn("stalled", factor.event["hierarchical_fallback"])
            np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-11, atol=1e-13)
            with mock.patch.object(hf, "HierarchicalFactor", side_effect=hf.HierarchicalRejected("rank")):
                built = ModalFactor(a.copy(), 2, False, coordinates=coordinates)
            self.assertIsNone(built.hierarchical)
            self.assertIn("rank", built.event["hierarchical_fallback"])
        # Without coordinates (other formulations) a mode keeps LU.
        with mock.patch.dict(os.environ, {"GHOST_HIERARCHICAL_MIN_UNKNOWNS": "100"}):
            self.assertIsNone(ModalFactor(a.copy(), 2, False).hierarchical)

    def test_pricing_follows_the_factor(self):
        n = hf.HIERARCHICAL_MIN_UNKNOWNS
        lu = bor_solver.estimate_bor_dense_peak_gb(n, 64, 4, 10)
        hodlr = bor_solver.estimate_bor_dense_peak_gb(n, 64, 4, 10, hierarchical=True)
        small = bor_solver.estimate_bor_dense_peak_gb(n // 2, 64, 4, 10, hierarchical=True)
        self.assertEqual(small, bor_solver.estimate_bor_dense_peak_gb(n // 2, 64, 4, 10))
        ratio = bor_solver.BOR_HIERARCHICAL_MATRIX_EQUIVALENTS / bor_solver.BOR_DENSE_MATRIX_EQUIVALENTS
        self.assertLess(hodlr, lu)
        self.assertGreater(hodlr, ratio * lu * 0.9)


if __name__ == "__main__":
    unittest.main()
