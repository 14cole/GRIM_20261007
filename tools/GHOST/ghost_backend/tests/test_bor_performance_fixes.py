"""BoR runtime fixes: BLAS thread bounds, parallel near preparation, sparse constraints."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import kernels
from ghost_backend.bor import solver as bor
import legacy_near_rules as legacy

C0 = bor.C0
FREQUENCY_HZ = 1.0e9


def _sphere(radius, elements):
    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))


def _hemisphere(radius, elements, upper=True):
    start, stop = (0.0, 0.5 * math.pi) if upper else (0.5 * math.pi, math.pi)
    theta = np.linspace(start, stop, int(elements) + 1)
    return np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))


class BoundedBlasThreadTests(unittest.TestCase):
    def limits_requested(self, cpus, workers, current_threads, physical=None):
        info = [dict(user_api="blas", num_threads=current_threads)]
        with mock.patch.object(bor.os, "cpu_count", return_value=cpus), \
                mock.patch("ghost_backend.execution.options.physical_core_count",
                           return_value=cpus if physical is None else physical), \
                mock.patch("ghost_backend.execution.thread_control.threadpool_info",
                           return_value=info), \
                mock.patch("ghost_backend.execution.thread_control.threadpool_limits") as limits:
            with bor._bounded_blas_threads(workers):
                pass
        return [call.kwargs["limits"] for call in limits.call_args_list]

    def test_concurrent_workers_share_the_cores(self):
        self.assertEqual(self.limits_requested(16, 4, 16), [4])
        self.assertEqual(self.limits_requested(16, 15, 16), [1])

    def test_single_worker_and_existing_lower_limits_are_left_alone(self):
        self.assertEqual(self.limits_requested(16, 1, 16), [])
        self.assertEqual(self.limits_requested(16, 4, 2), [])

    def test_blas_threads_never_exceed_the_physical_cores(self):
        # 16 SMT threads on 8 cores: one worker gets 8, four workers 2 each.
        self.assertEqual(self.limits_requested(16, 1, 16, physical=8), [8])
        self.assertEqual(self.limits_requested(16, 4, 16, physical=8), [2])

    def test_mode_sweep_applies_the_bound_for_its_worker_count(self):
        seen = []
        original = bor._bounded_blas_threads

        def spy(workers):
            seen.append(workers)
            return original(workers)

        with mock.patch.object(bor, "_bounded_blas_threads", side_effect=spy):
            bor._mode_sweep(
                1, [90.0], ("VV",), 0, 1e-6,
                lambda m: (np.ones((1, 1), complex), None),
                lambda m, th, pol: np.ones(1, complex),
                lambda m, x, th, pol: complex(x[0]),
                workers=3,
            )
        self.assertEqual(seen, [1])  # Only one mode task is admitted.


class NearKernelRefinementTests(unittest.TestCase):
    def test_only_unconverged_points_are_refined_and_values_are_per_point(self):
        k = 2.0 * math.pi * FREQUENCY_HZ / C0
        radius = 3.0 / k
        gen = kernels.Generatrix(_sphere(radius, 30))
        s, sp, _ = bor._cell_points("diag", depth=4)
        rp, zp, trp, tzp, *_ = bor._points_on_element(gen, 10, s)
        rq, zq, trq, tzq, *_ = bor._points_on_element(gen, 10, sp)
        n = len(rp)
        args = (rp, zp, np.full(n, trp), np.full(n, tzp),
                rq, zq, np.full(n, trq), np.full(n, tzq))
        sizes = []
        rule = legacy._mfie_kernels_near_rule

        def spy(*a, **kw):
            sizes.append(len(np.atleast_1d(a[0])))
            return rule(*a, **kw)

        spy.__name__ = rule.__name__
        values = kernels._checked_near_kernels(spy, args, k, 15, 48, 0, True)
        chunk_first = max(sizes)
        self.assertLess(min(sizes), chunk_first,
                        "later refinement rounds must evaluate a subset")

        # A point's kernel does not depend on which other points share its batch.
        for index in (0, n // 2, n - 1):
            alone = kernels._checked_near_kernels(
                rule, tuple(a[index:index + 1] for a in args), k, 15, 48, 0, True)
            for batch_value, single_value in zip(values, alone):
                scale = np.max(np.abs(single_value))
                np.testing.assert_allclose(batch_value[index], single_value[0],
                                           rtol=0, atol=1e-7 * scale)


class ParallelNearPreparationTests(unittest.TestCase):
    def test_parallel_preparation_matches_serial_bit_for_bit(self):
        radius = 2.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        serial = bor.BorPecSolver(_sphere(radius, 14), FREQUENCY_HZ)
        parallel = bor.BorPecSolver(_sphere(radius, 14), FREQUENCY_HZ)
        serial.prepare_operators(4, efie=True, mfie=True, workers=1)
        parallel.prepare_operators(4, efie=True, mfie=True, workers=4)
        for key, prepared in serial._near_contractions.items():
            other = parallel._near_contractions[key]
            for field in ("rows", "cols", "source_elems", "values"):
                np.testing.assert_array_equal(other[field], prepared[field])
        self.assertEqual(parallel.near_quadrature_order_max,
                         serial.near_quadrature_order_max)

    def test_map_preserves_order_and_propagates_failures(self):
        self.assertEqual(list(bor._iter_near_pairs(lambda p: p * 2, range(40), 4)),
                         [2 * p for p in range(40)])

        def failing(pair):
            if pair == 7:
                raise RuntimeError("abort requested")
            return pair

        with self.assertRaisesRegex(RuntimeError, "abort requested"):
            list(bor._iter_near_pairs(failing, range(40), 4))

    def test_near_worker_count_is_bounded_by_scratch_budget(self):
        self.assertEqual(bor._near_preparation_workers(1), 1)
        cap = bor.NEAR_PREPARATION_SCRATCH_BYTES // bor._NEAR_TASK_SCRATCH_BYTES
        self.assertEqual(bor._near_preparation_workers(1000), cap)


class MultiRegionConstraintTests(unittest.TestCase):
    def banded_system(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        outer = 1.12 * radius
        surfaces = [
            (_hemisphere(outer, 4, upper=True), False),
            (_hemisphere(outer, 4, upper=False), False),
            (np.asarray([[outer, 0.0], [radius, 0.0]]), False),
            (_hemisphere(radius, 4, upper=True), True),
            (_hemisphere(radius, 4, upper=False), True),
        ]
        regions = [
            {"medium": None, "bounds": [(0, +1), (1, +1)], "exterior": True},
            {"medium": (2.4 - 0.06j, 1.0 - 0.01j), "bounds": [(0, -1), (2, +1), (3, +1)]},
            {"medium": (3.1 - 0.10j, 1.0 - 0.02j), "bounds": [(1, -1), (2, -1), (4, +1)]},
        ]
        return bor._MultiRegionBor(surfaces, regions, FREQUENCY_HZ,
                                   near_factor=2.0, near_order=12)

    def test_sparse_constraints_are_built_once_and_match_dense(self):
        # The sweep projects excitations (Q^H b) and expands solutions (Q x)
        # with the transform ``assemble`` returns: one sparse matrix per
        # constraint category, and rhs/farfield work on the full vector.
        system = self.banded_system()
        m_max = 3
        system.prepare(m_max)
        _, transform = system.assemble(1, m_max)
        self.assertTrue(bor.issparse(transform))
        self.assertIs(system.build_Q(1), system.build_Q(1))
        self.assertIs(system.build_Q(2), system.build_Q(5))
        full = np.zeros(system.n_full, complex)
        for (si, _) in system.regions[system.ext_region]["bounds"]:
            solver = system.solv[(si, system.ext_region)]
            full[system.off_J[si]:system.off_J[si] + 2 * system.Nn[si]] =                 solver.rhs_mode(1, 37.0, "VV")
            if system.off_M[si] is not None:
                full[system.off_M[si]:system.off_M[si] + 2 * system.Nn[si]] =                     bor.ETA0 * solver.rhs_h_mode(1, 37.0, "VV")
        np.testing.assert_array_equal(system.rhs(1, 37.0, "VV"), full)
        dense_q = np.asarray(transform.toarray())
        np.testing.assert_allclose(transform.conj().T @ full, dense_q.conj().T @ full,
                                   rtol=1e-14, atol=0)

    def test_batched_excitation_and_far_field_match_per_aspect(self):
        system = self.banded_system()
        thetas = [0.0, 37.0, 90.0, 143.0]
        pols = ("VV", "HH")
        for m in (0, 1, 3):
            batch = system.rhs_batch(m, thetas, pols)
            single = np.column_stack([system.rhs(m, th, pol) for th in thetas for pol in pols])
            np.testing.assert_allclose(batch, single, rtol=1e-13, atol=1e-13 * np.max(abs(single)))
            rng = np.random.default_rng(m)
            x = rng.normal(size=batch.shape) + 1j * rng.normal(size=batch.shape)
            fields = system.farfield_batch(m, x, thetas, pols)
            expected = np.array([[system.farfield(m, x[:, 2 * it + ip], th, pol)
                                  for it, th in enumerate(thetas)] for ip, pol in enumerate(pols)])
            np.testing.assert_allclose(fields, expected, rtol=1e-12, atol=1e-12 * np.max(abs(expected)))

    def test_multiregion_signed_mode_symmetry_matches_full_signed_sweep(self):
        common = dict(freq_hz=FREQUENCY_HZ, thetas_deg=[0.0, 37.0, 90.0, 143.0],
                      n_modes=8, mode_tol=1.0e-6, workers=2, progress=None,
                      check_abort=None, formulation="banded", extra={},
                      table_precision="double", assembly="tables")
        reduced = bor._solve_multiregion(self.banded_system(), **common)
        self.assertTrue(reduced["signed_mode_symmetry_used"])

        impl = bor._mode_sweep_impl

        def without_symmetry(*args, **kwargs):
            kwargs["signed_mode_symmetry"] = False
            return impl(*args, **kwargs)

        with mock.patch.object(bor, "_mode_sweep_impl", side_effect=without_symmetry):
            full = bor._solve_multiregion(self.banded_system(), **common)
        self.assertFalse(full["signed_mode_symmetry_used"])
        for key in ("amp_vv", "amp_hh"):
            np.testing.assert_allclose(np.asarray(reduced[key]), np.asarray(full[key]),
                                       rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()


def _numpy_brackets():
    """Force the NumPy bracket path regardless of the native kernel."""
    return mock.patch.object(kernels, "_native_mfie_brackets", lambda *a, **k: None)


def _pair_points(count, seed=17):
    rng = np.random.default_rng(seed)
    rho_p = rng.uniform(0.5, 1.0, count)
    z_p = rng.uniform(-1.0, 1.0, count)
    return (rho_p, z_p, rng.uniform(-1.0, 1.0, count), rng.uniform(-1.0, 1.0, count),
            rho_p + rng.uniform(-0.02, 0.02, count), z_p + rng.uniform(-0.02, 0.02, count),
            rng.uniform(-1.0, 1.0, count), rng.uniform(-1.0, 1.0, count))


def _native_kernel():
    from ghost_backend.bor.streaming import _NATIVE
    return _NATIVE if (_NATIVE is not None and hasattr(_NATIVE, "near_mfie")) else None


class NativeNearBracketTests(unittest.TestCase):
    """The paired native sampler must mirror the NumPy brackets it replaces."""

    K = 83.8

    def setUp(self):
        # Without the kernel both branches are NumPy and the comparisons are vacuous.
        if _native_kernel() is None:
            self.skipTest("native BoR kernel with near_mfie is not built here")
        engaged = kernels._native_mfie_brackets(
            _pair_points(4), self.K, np.linspace(-np.pi, np.pi, 8), False
        )
        self.assertIsNotNone(engaged, "native sampler declined a paired call")

    def test_shared_grid_brackets_match_numpy(self):
        points = _pair_points(64)
        xi = 2.0 * np.pi * np.arange(128) / 128.0 - np.pi
        native = kernels._mfie_brackets(*points, self.K, xi)
        with _numpy_brackets():
            reference = kernels._mfie_brackets(*points, self.K, xi)
        for got, want in zip(native, reference):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)

    def test_per_pair_grid_near_rule_matches_numpy(self):
        points = _pair_points(48, seed=23)
        native = legacy._mfie_kernels_near_rule(*points, self.K, 6)
        with _numpy_brackets():
            reference = legacy._mfie_kernels_near_rule(*points, self.K, 6)
        for got, want in zip(native, reference):
            np.testing.assert_allclose(got, want, rtol=1e-11, atol=0.0)

    def test_complex_wavenumber_matches_numpy_and_legacy_library_falls_back(self):
        from ghost_backend.bor import streaming
        points = _pair_points(8)
        xi = np.linspace(-np.pi, np.pi, 32)
        native = kernels._mfie_brackets(*points, 83.8 - 4.0j, xi)
        with _numpy_brackets():
            reference = kernels._mfie_brackets(*points, 83.8 - 4.0j, xi)
        for got, want in zip(native, reference):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
        legacy = type('LegacyLibrary', (), {'near_mfie': _native_kernel().near_mfie})()
        with mock.patch.object(streaming, '_NATIVE', legacy):
            self.assertIsNone(kernels._native_mfie_brackets(points, 83.8 - 4.0j, xi, False))

    def test_non_paired_shapes_decline_the_native_path(self):
        rho_p = np.linspace(0.5, 1.0, 8)
        outer = (rho_p[:, None], rho_p[:, None], rho_p[:, None], rho_p[:, None],
                 rho_p[None, :], rho_p[None, :], rho_p[None, :], rho_p[None, :])
        self.assertIsNone(
            kernels._native_mfie_brackets(outer, 83.8, np.linspace(0.0, 1.0, 4), False)
        )
        mismatched = tuple([rho_p] * 7 + [rho_p[:4]])
        self.assertIsNone(
            kernels._native_mfie_brackets(mismatched, 83.8, np.linspace(0.0, 1.0, 4), False)
        )


class ModalProjectionTests(unittest.TestCase):
    """Half-range +-xi projection, built for |m| and mirrored by parity."""

    @staticmethod
    def _signed_reference(Fp, Fm, w_pos, xi_pos, m):
        """The direct signed-m form the |m| construction replaces."""
        S = np.stack([(a + b) * w_pos for a, b in zip(Fp, Fm)], axis=1)
        D = np.stack([(a - b) * w_pos for a, b in zip(Fp, Fm)], axis=1)
        out = np.empty((len(xi_pos), len(Fp), len(m)), complex)
        for m0 in range(0, len(m), 32):
            arg = xi_pos[:, :, None] * m[None, None, m0:m0 + 32]
            out[:, :, m0:m0 + 32] = (
                np.matmul(S, np.cos(arg)) - 1j * np.matmul(D, np.sin(arg))
            )
        return [out[:, i, :] for i in range(out.shape[1])]

    def test_matches_the_signed_form_across_mode_ranges(self):
        rng = np.random.default_rng(5)
        for pairs, samples, m_max in ((40, 60, 3), (25, 48, 10), (12, 32, 17)):
            with self.subTest(pairs=pairs, m_max=m_max):
                Fp = [rng.standard_normal((pairs, samples))
                      + 1j * rng.standard_normal((pairs, samples)) for _ in range(4)]
                Fm = [rng.standard_normal((pairs, samples))
                      + 1j * rng.standard_normal((pairs, samples)) for _ in range(4)]
                weights = rng.standard_normal((pairs, samples))
                xi = rng.standard_normal((pairs, samples))
                modes = np.arange(-m_max, m_max + 1)
                got = legacy._project_pm_brackets(Fp, Fm, weights, xi, modes)
                want = self._signed_reference(Fp, Fm, weights, xi, modes)
                self.assertEqual(len(got), len(want))
                for a, b in zip(got, want):
                    np.testing.assert_allclose(a, b, rtol=1e-11, atol=0.0)

    def test_symmetric_samples_make_the_mode_range_even(self):
        """Fp == Fm kills the sine term, so +m and -m must coincide exactly."""
        rng = np.random.default_rng(9)
        shared = [rng.standard_normal((6, 16)) + 1j * rng.standard_normal((6, 16))
                  for _ in range(4)]
        weights = rng.standard_normal((6, 16))
        xi = rng.standard_normal((6, 16))
        m_max = 3
        projected = legacy._project_pm_brackets(
            shared, list(shared), weights, xi, np.arange(-m_max, m_max + 1)
        )
        for bracket in projected:
            for order in range(1, m_max + 1):
                np.testing.assert_allclose(
                    bracket[:, m_max + order], bracket[:, m_max - order],
                    rtol=1e-12, atol=0.0,
                )

    def test_antisymmetric_samples_make_the_mode_range_odd(self):
        """Fm == -Fp kills the cosine term, so +m and -m must be negatives."""
        rng = np.random.default_rng(11)
        shared = [rng.standard_normal((5, 12)) + 1j * rng.standard_normal((5, 12))
                  for _ in range(4)]
        weights = rng.standard_normal((5, 12))
        xi = rng.standard_normal((5, 12))
        m_max = 3
        projected = legacy._project_pm_brackets(
            shared, [-value for value in shared], weights, xi,
            np.arange(-m_max, m_max + 1),
        )
        for bracket in projected:
            for order in range(1, m_max + 1):
                np.testing.assert_allclose(
                    bracket[:, m_max + order], -bracket[:, m_max - order],
                    rtol=1e-12, atol=0.0,
                )
