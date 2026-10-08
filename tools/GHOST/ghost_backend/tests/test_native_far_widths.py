"""All native basis widths retain the independent Hankel quadrature result."""
import itertools
import unittest

import numpy as np

from ghost_backend.twod.assembly import kernels
from ghost_backend.twod.assembly.native import far


class NativeFarWidthTests(unittest.TestCase):
    def test_basis_widths_and_kernel_routes_against_direct_hankel(self):
        if far.library() is None:
            self.skipTest('Native far kernel is not installed.')
        rng = np.random.default_rng(937)
        q, mb, nb = 5, 2, 3
        observers = rng.uniform(0., .04, size=(mb, q, 2))
        sources = rng.uniform(.12, .16, size=(nb, q, 2))
        normals_o = np.array([[1., 0.], [0., 1.]])
        normals_s = np.array([[0., 1.], [1., 0.], [1., 1.]])/np.array([[1.], [1.], [np.sqrt(2)]])
        weights = rng.uniform(.1, 1., q)
        mask = np.array([[True, False, True], [True, True, False]])
        for k in (37., 37.-3j):
            table = kernels.KernelTable(k, .4, degree=16)
            for width in (1, 2, 3, 4, 5, 8):
                phi = rng.uniform(-1., 1., size=(q, width))
                weighted = weights[:, None]*phi
                for obs_derivative, want_s, want_k, mirrored in itertools.product((False, True), repeat=4):
                    if not want_s and not want_k:
                        continue
                    with self.subTest(k=k, width=width, obs_derivative=obs_derivative,
                                      want_s=want_s, want_k=want_k, mirrored=mirrored):
                        actual = far.far_block(table, k, observers, sources, weights, phi,
                            normals_o, normals_s, mask, obs_derivative, want_s, want_k, mirrored)
                        self.assertIsNotNone(actual)
                        reference = [np.zeros((width*width, mb, nb), complex)
                                     if enabled else None for enabled in (want_s, want_k, want_k and mirrored)]
                        for i, j in zip(*np.nonzero(mask)):
                            delta = observers[i, :, None, :]-sources[j, None, :, :]
                            distance = np.linalg.norm(delta, axis=-1)
                            value = kernels.values(k, distance)
                            if want_s:
                                reference[0][:, i, j] = (weighted.T @ value[:, :, 0] @ weighted).ravel()
                            if want_k:
                                normal = normals_o[i] if obs_derivative else normals_s[j]
                                derivative = value[:, :, 1]*(delta @ normal)/distance
                                if obs_derivative:
                                    derivative *= -1
                                reference[1][:, i, j] = (weighted.T @ derivative @ weighted).ravel()
                                if mirrored:
                                    reverse_normal = normals_s[j] if obs_derivative else normals_o[i]
                                    derivative = value[:, :, 1]*(delta @ reverse_normal)/distance
                                    if not obs_derivative:
                                        derivative *= -1
                                    reference[2][:, i, j] = (weighted.T @ derivative @ weighted).T.ravel()
                        for observed, expected in zip(actual, reference):
                            if expected is None:
                                self.assertIsNone(observed)
                            else:
                                np.testing.assert_allclose(observed, expected, rtol=3e-12, atol=3e-14)


if __name__ == '__main__':
    unittest.main()
