"""Equivalent half-grid sums across complex, packed-real and FFT projections."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import kernels


class ProjectionMethodTests(unittest.TestCase):
    def test_signed_reordered_and_aliased_modes_keep_the_same_quadrature(self):
        rng = np.random.default_rng(1287)
        for size in (64, 512, 2048):
            modes = np.array([0, 1, -1, 12, -7, size//2, size//2+3, size-1, size, size+2, -size-3, 1])
            samples = rng.standard_normal((9, size//2+1)) + 1j*rng.standard_normal((9, size//2+1))
            tables = kernels._half_grid_tables(size, modes, True)
            for method in ('real', 'fft'):
                alternative = kernels._half_grid_tables(size, modes, True, method)
                for odd in (False, True):
                    with self.subTest(size=size, method=method, odd=odd):
                        index = int(odd)
                        expected = samples @ tables[index]
                        actual = kernels._project_half_grid(samples, alternative[index], modes, size, method, odd)
                        exact_zero = (modes % (size//2) == 0) if odd and method == 'fft' else np.zeros(len(modes), bool)
                        # Explicit trig tables have sin(integer*pi) rounding at
                        # these exactly zero DST bins. The FFT path removes it.
                        np.testing.assert_array_equal(actual[:, exact_zero], 0.)
                        self.assertLess(np.max(abs(expected[:, exact_zero]), initial=0.), 2e-13)
                        np.testing.assert_allclose(actual[:, ~exact_zero], expected[:, ~exact_zero], rtol=3e-12, atol=1e-13)

    def test_cache_accounts_for_real_tables_and_avoids_tables_for_fft(self):
        modes = np.arange(129)
        ordinary = kernels._half_grid_tables(2048, modes, True)
        real = kernels._half_grid_tables(2048, modes, True, 'real')
        transformed = kernels._half_grid_tables(2048, modes, True, 'fft')
        self.assertEqual(sum(a.nbytes for a in ordinary[:2]), 2*sum(a.nbytes for a in real[:2]))
        self.assertEqual(transformed[:2], (None, None))
        self.assertIs(real, kernels._half_grid_tables(2048, modes, True, 'real'))
        self.assertTrue(all(not a.flags.writeable for a in real))
        self.assertLessEqual(kernels._FAR_TABLE_BYTES[0], kernels.FAR_TABLE_CACHE_BYTES)

    def test_shape_selection_retains_small_complex_products(self):
        self.assertEqual(kernels._half_grid_projection_method(64, 1025, 33), 'complex')
        self.assertEqual(kernels._half_grid_projection_method(256, 1025, 129), 'real')
        self.assertEqual(kernels._half_grid_projection_method(256, 1025, 513), 'fft')

    def test_all_far_kernel_families_and_fallback_preserve_complex_media_and_near_masks(self):
        rp = np.linspace(.08, .12, 8)[:, None]
        zp = np.linspace(-.1, .1, 8)[:, None]
        rq = np.linspace(.14, .16, 6)[None, :]
        zq = np.linspace(.14, .22, 6)[None, :]
        mask = np.zeros((8, 6), bool)
        mask[0, :3] = True
        modes = np.array([-32, -3, 0, 1, 8, 32, -1, 3])
        for native in (True, False):
            sampler = kernels._native_pair_sampler
            with mock.patch.object(kernels, '_native_pair_sampler', side_effect=sampler if native else lambda kind: None):
                for k in (100., 100.-8j):
                    for kind in ('g', 'mfie', 'ibc'):
                        coordinates = (rp, zp, rq, zq) if kind == 'g' else (
                            rp, zp, np.ones_like(rp)*.6, np.ones_like(rp)*.8,
                            rq, zq, np.ones_like(rq)*.8, np.ones_like(rq)*.6)
                        def evaluate(method, dtype=np.complex128):
                            with mock.patch.object(kernels, '_half_grid_projection_method', return_value=method):
                                return kernels.banded_modal_kernels(kind, coordinates, k, 32, mask, modes,
                                                                    work_bytes=100000, threads=1, out_dtype=dtype)
                        expected = evaluate('complex')
                        expected = (expected,) if kind == 'g' else expected
                        for method in ('real', 'fft'):
                            for dtype in (np.complex128, np.complex64):
                                with self.subTest(native=native, k=k, kind=kind, method=method, dtype=dtype):
                                    actual = evaluate(method, dtype)
                                    actual = (actual,) if kind == 'g' else actual
                                    for value, reference in zip(actual, expected):
                                        self.assertEqual(value.dtype, dtype)
                                        np.testing.assert_array_equal(value[mask], 0)
                                        np.testing.assert_allclose(value, reference,
                                            rtol=2e-7 if dtype == np.complex64 else 2e-12,
                                            atol=(2e-7 if dtype == np.complex64 else 2e-13)*max(np.max(abs(reference)), 1e-20))

    def test_full_conductor_and_lossy_dielectric_complex_fields_match(self):
        from ghost_backend.bor import solver
        points = solver.sphere_generatrix(.035, 12)
        common = dict(points=points, freq_hz=1e9, thetas_deg=[0., 37., 90., 180.],
                      workers=1, gauss_order=3, bor_options={'factorization': 'dense'})
        for solve, extra in ((solver.solve_bor, {'formulation': 'cfie', 'assembly': 'streaming'}),
                             (solver.solve_bor_dielectric, {'eps_r': 2.5-.05j})):
            with mock.patch.object(kernels, '_half_grid_projection_method', return_value='complex'):
                expected = solve(**common, **extra)
            for method in ('real', 'fft'):
                with self.subTest(solver=solve.__name__, method=method), \
                        mock.patch.object(kernels, '_half_grid_projection_method', return_value=method):
                    actual = solve(**common, **extra)
                    for key in ('amp_vv', 'amp_hh', 'sigma_vv', 'sigma_hh'):
                        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=2e-13)


if __name__ == '__main__':
    unittest.main()
