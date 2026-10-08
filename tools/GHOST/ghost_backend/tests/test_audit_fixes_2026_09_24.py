"""Audit fixes of 2026-09-24: single-thread spawned workers, residual storage
chosen by need, the fused pre-LU matrix pass, shared excitation records, the
bounded mode window and the bounded spill working set (see
AUDIT_FIXES_2026-09-24.md; the packed EFIE storage is tested with the other
streaming tests)."""
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.execution import runtime
from ghost_backend.linalg import residual_spool

GIB = 1024**3


class WorkerEnvironmentTests(unittest.TestCase):
    def test_pins_nest_and_the_last_release_restores_the_caller_values(self):
        names = [name for name, _ in runtime.SINGLE_THREAD_WORKER_ENVIRONMENT]
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "16"}, clear=False):
            for name in names[1:]:
                os.environ.pop(name, None)
            with runtime.single_thread_worker_environment():
                self.assertEqual([os.environ[name] for name in names], ["1"] * len(names))
                runtime.pin_worker_environment()
                runtime.release_worker_environment()
                self.assertEqual([os.environ[name] for name in names], ["1"] * len(names))
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "16")
            for name in names[1:]:
                self.assertNotIn(name, os.environ)
        # An unmatched release is ignored.
        runtime.release_worker_environment()

    def test_near_process_scope_pins_only_while_its_pool_exists(self):
        from ghost_backend.bor import near_parallel
        with mock.patch.object(near_parallel, "process_backend_possible", return_value=True), \
                mock.patch.object(near_parallel, "processes_selected", return_value=True), \
                mock.patch.object(near_parallel, "ProcessPoolExecutor") as pool, \
                mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "8"}, clear=False):
            with near_parallel.process_scope(4, 4):
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")
                near_parallel.executor_for(100, 5)
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")
        pool.return_value.shutdown.assert_called_once()


class ResidualStorageByNeedTests(unittest.TestCase):
    def available(self, gib):
        return mock.patch("ghost_backend.twod.solver._detect_available_gb", return_value=gib)

    def test_auto_keeps_a_large_matrix_in_memory_when_its_copy_fits(self):
        large = 8 * GIB
        with self.available(20.0):
            self.assertFalse(residual_spool.auto_spooled(large))
        # Short of the copy and its margin: spool.
        with self.available(8.5):
            self.assertTrue(residual_spool.auto_spooled(large))
        # Unknown availability keeps the former behavior for a large matrix.
        with self.available(0.0):
            self.assertTrue(residual_spool.auto_spooled(large))
        # Small matrices are never spooled automatically.
        with self.available(0.0):
            self.assertFalse(residual_spool.auto_spooled(residual_spool.AUTO_SPOOL_MIN_BYTES - 16))

    def test_both_solvers_route_auto_through_the_same_decision(self):
        from ghost_backend.bor.factor import _residual_spool_selected
        from ghost_backend.execution.options import execution_scope
        a = np.asfortranarray(np.eye(64, dtype=complex))
        with mock.patch.object(residual_spool, "AUTO_SPOOL_MIN_BYTES", 0):
            for fits, expected in ((True, False), (False, True)):
                with mock.patch.object(residual_spool, "copy_fits", return_value=fits):
                    with execution_scope(dict(dense_residual_storage="auto")):
                        self.assertEqual(residual_spool.selected(a, True, "dense"), expected)
                    self.assertEqual(_residual_spool_selected(a, True, "auto"), expected)
                    # Explicit policies are unchanged.
                    self.assertTrue(_residual_spool_selected(a, True, "disk"))
                    self.assertFalse(_residual_spool_selected(a, True, "memory"))
                    self.assertFalse(_residual_spool_selected(a, False, "disk"))

    def test_auto_factor_keeps_its_original_when_memory_allows(self):
        from ghost_backend.bor.factor import ModalFactor
        rng = np.random.default_rng(3)
        a = np.asfortranarray(rng.normal(size=(96, 96)) + 1j * rng.normal(size=(96, 96)) + 20 * np.eye(96))
        original = a.copy()
        rhs = rng.normal(size=(96, 2)) + 1j * rng.normal(size=(96, 2))
        with mock.patch.object(residual_spool, "AUTO_SPOOL_MIN_BYTES", 0), self.available(64.0):
            factor = ModalFactor(a, 1, False, owned=True, residual_storage=("auto", "."))
        self.assertEqual(factor.event["residual_storage"], "memory")
        self.assertFalse(np.shares_memory(factor.lu, a))
        np.testing.assert_array_equal(a, original)
        np.testing.assert_allclose(original @ factor.solve(rhs), rhs, rtol=1e-12, atol=1e-12)


class FusedMatrixPassTests(unittest.TestCase):
    def test_one_pass_reproduces_the_finite_check_norm_and_row_maxima(self):
        from ghost_backend.linalg.workspace import checked_row_norms, first_nonfinite, matrix_inf_norm
        rng = np.random.default_rng(11)
        base = rng.normal(size=(257, 257)) + 1j * rng.normal(size=(257, 257))
        for order in ("C", "F"):
            for block in (1024, 4096, 1024 * 1024):
                a = np.array(base, order=order)
                first, norm, row_max = checked_row_norms(a, block)
                self.assertIsNone(first)
                self.assertEqual(norm, matrix_inf_norm(a, block))          # bitwise
                np.testing.assert_array_equal(row_max, np.max(np.abs(a), axis=1))
                bad = a.copy()
                bad[200, 17] = complex(np.nan, 0.0)
                bad[31, 250] = complex(1.0, np.inf)
                self.assertEqual(checked_row_norms(bad, block)[0], first_nonfinite(bad, block))
                self.assertEqual(checked_row_norms(bad, block)[0], (31, 250))

    def test_dense_factor_condition_and_residual_evidence_are_unchanged(self):
        import ghost_backend.twod.solver as rcs
        from ghost_backend.linalg.dense import DenseFactor
        rng = np.random.default_rng(5)
        a = np.asfortranarray(rng.normal(size=(150, 150)) + 1j * rng.normal(size=(150, 150))
                              + 30 * np.eye(150))
        rhs = rng.normal(size=(150, 4)) + 1j * rng.normal(size=(150, 4))
        diagnostics = {}
        factor = DenseFactor(a, diagnostics)
        solution = factor.solve(rhs)
        rows, cols, norm = rcs._equilibrated_scaling_and_norm_1(a)
        fused_rows, fused_cols, fused_norm = factor._condition_scaling
        np.testing.assert_array_equal(fused_rows, rows)
        np.testing.assert_array_equal(fused_cols, cols)
        self.assertEqual(fused_norm, norm)
        from ghost_backend.linalg.workspace import matrix_inf_norm
        self.assertEqual(factor.matrix_inf, matrix_inf_norm(a))
        lu, piv = rcs._SCIPY_LINALG.lu_factor(a)
        self.assertEqual(diagnostics["condition_est"], rcs._equilibrated_condition_from_lu(a, lu, piv))
        np.testing.assert_allclose(a @ solution, rhs, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, r"NaN/Inf at index \(3, 4\)"):
            bad = a.copy()
            bad[3, 4] = np.nan
            DenseFactor(bad)


class SharedAngularRecordTests(unittest.TestCase):
    def test_recurrence_matches_direct_bessel_triplets(self):
        from ghost_backend.bor import solver as bor
        rng = np.random.default_rng(2)
        rho = np.concatenate([rng.uniform(0.0, 0.2, 400), [1e-12, 1e-9, 0.0]])
        z = rng.uniform(-1.0, 1.0, rho.size)
        # On-axis aspects (u = 0), a tiny angle, broadside and back.
        thetas = np.array([0.0, 1e-9, 0.5, 45.0, 90.0, 150.0, 180.0])
        with np.errstate(all="raise"):
            for k, top in ((20.0, 16), (400.0, 95), (400.0, 400)):
                record = bor._AngularChunk(k, rho, z, thetas, top)
                for m in list(range(0, top - 1, 7)) + [-1, -2, -(top - 1), top - 1]:
                    with np.errstate(all="ignore"):
                        expected = bor._bessel_triplet(m, record.u)
                    for actual, reference in zip(record.triplet(m), expected):
                        self.assertLess(float(np.max(np.abs(actual - reference))), 5e-13)
        with self.assertRaisesRegex(ValueError, "below the requested mode"):
            record.triplet(top)

    def test_sweep_excitation_matches_the_direct_evaluation(self):
        from ghost_backend.bor import solver as bor
        solver = bor.BorPecSolver(bor.sphere_generatrix(0.1, 30), 3.0e9)
        thetas = np.linspace(0.0, 180.0, 71)
        solutions = np.random.default_rng(4).normal(size=(2 * solver.Nn, 2 * thetas.size)) + 0j
        with mock.patch.object(bor, "_reserve_angular_bytes", return_value=False):
            direct = [(solver.rhs_vv_hh_batch(m, thetas, 0.5, 0.5 * bor.ETA0),
                       solver.farfield_vv_hh_batch(m, solutions, thetas)) for m in (0, 1, -3, 9)]
        self.assertEqual(solver._angular_shared, {})
        for (rhs, far), m in zip(direct, (0, 1, -3, 9)):
            np.testing.assert_allclose(solver.rhs_vv_hh_batch(m, thetas, 0.5, 0.5 * bor.ETA0), rhs,
                                       rtol=0, atol=1e-13 * np.max(np.abs(rhs)))
            np.testing.assert_allclose(solver.farfield_vv_hh_batch(m, solutions, thetas), far,
                                       rtol=0, atol=1e-13 * np.max(np.abs(far)))
        # One record per 64-aspect chunk, shared by every mode; mode 9 rebuilt
        # nothing (the minimum top order already covers it).
        self.assertEqual(len(solver._angular_shared), 2)
        self.assertTrue(all(record.top >= 16 for record in solver._angular_shared.values()))

    def test_records_respect_the_budget_and_release_it(self):
        import gc
        from ghost_backend.bor import solver as bor
        before = bor._ANGULAR_CACHE_USED[0]
        solver = bor.BorPecSolver(bor.sphere_generatrix(0.1, 30), 3.0e9)
        solver.rhs_vv_hh_batch(2, np.linspace(0.0, 180.0, 10))
        self.assertGreater(bor._ANGULAR_CACHE_USED[0], before)
        solver = None
        gc.collect()
        self.assertEqual(bor._ANGULAR_CACHE_USED[0], before)
        self.assertLessEqual(bor.angular_cache_bound_bytes(10**6, 10**4), bor.ANGULAR_CACHE_BUDGET_BYTES)
        self.assertEqual(bor.angular_cache_bound_bytes(100, 10),
                         bor.ANGULAR_CACHE_BYTES_PER_VALUE * 2 * 100 * 5)


class ModeWindowTests(unittest.TestCase):
    @staticmethod
    def _sweep(far_calls=None, slow_from=None, **kwargs):
        import time
        from ghost_backend.bor import solver as bor

        def assemble(m):
            if slow_from is not None and abs(m) >= slow_from:
                time.sleep(0.3)
            return np.eye(2, dtype=complex), None

        def rhs(m, th, pol):
            return np.ones(2, complex)

        def farfield(m, x, th, pol):
            if far_calls is not None:
                far_calls.append(abs(m))
            return complex(x[0]) * 10.0 ** (-2 * abs(m))     # a geometric tail
        return bor._mode_sweep(2, [20., 90.], ['VV', 'HH'], 30, 1e-6, assemble, rhs, farfield,
                               workers=16, min_mode_before_tail=3, **kwargs)

    def test_horizon_rules(self):
        from ghost_backend.bor import solver as bor
        self.assertEqual(bor._initial_mode_horizon(3, 30), 3 + 8 + bor.MODE_WINDOW_MARGIN)
        self.assertEqual(bor._initial_mode_horizon(14, 20), 20)
        # Before the tail the horizon is kept; a quiet increment needs one more mode.
        self.assertEqual(bor._predicted_mode_horizon(2, 3, 1e-2, 1.0, 1e-6, 30, 12), 12)
        self.assertEqual(bor._predicted_mode_horizon(5, 3, 1e-7, 1e-4, 1e-6, 30, 12),
                         5 + 1 + bor.MODE_WINDOW_MARGIN)
        # Ratio 0.1 from 1e-3: three more modes to the first quiet one, then its pair.
        self.assertEqual(bor._predicted_mode_horizon(5, 3, 1e-3, 1e-2, 1e-6, 30, 12),
                         5 + 3 + 1 + bor.MODE_WINDOW_MARGIN)
        # A tail that is not decaying lifts the limit; unknown history keeps it.
        self.assertEqual(bor._predicted_mode_horizon(5, 3, 1e-2, 1e-3, 1e-6, 30, 12), 30)
        self.assertEqual(bor._predicted_mode_horizon(3, 3, 1e-2, math.inf, 1e-6, 30, 12), 12)

    def test_window_stops_at_the_predicted_tail_with_identical_fields(self):
        from ghost_backend.bor import solver as bor
        F, used, stats = self._sweep()
        with mock.patch.object(bor, "_initial_mode_horizon", lambda tail, cap: cap), \
                mock.patch.object(bor, "_predicted_mode_horizon", lambda *args: args[5]):
            F0, used0, stats0 = self._sweep()
        np.testing.assert_array_equal(F, F0)
        # Signed-mode symmetry doubles each increment: quiet at 4 and 5.
        self.assertEqual((used, used0), (5, 5))
        self.assertTrue(stats["mode_converged"] and stats0["mode_converged"])
        self.assertEqual((stats["mode_tasks_started"], stats0["mode_tasks_started"]), (13, 21))

    def test_modes_running_past_convergence_stop_at_their_next_checkpoint(self):
        calls = []
        F, used, stats = self._sweep(far_calls=calls, slow_from=6)
        self.assertEqual(used, 5)
        self.assertTrue(stats["mode_converged"])
        self.assertGreater(stats["mode_tasks_started"], 6)       # started, then stopped
        self.assertEqual([m for m in calls if m >= 6], [])


class SpillReleaseTests(unittest.TestCase):
    def test_rows_are_released_once_every_block_reaching_them_is_done(self):
        from ghost_backend.bor import streaming
        released = []
        fake = mock.MagicMock()
        fake._mmap = object()
        with mock.patch.object(streaming, "_release_spilled",
                               lambda array, ranges, flush=True: released.append(ranges)), \
                mock.patch.object(streaming, "_spilled_rows",
                                  lambda array, layout, lo, hi, modes=None: (lo, hi)):
            tracker = streaming._SpilledRowRelease([(fake, ("full", 11))], elements=10, rows=3,
                                                   tiles_per_row=2, nodes=11)
            # Row blocks start at 0, 3, 6, 9; tiles complete out of order.
            for e0 in (3, 3, 0, 9, 9, 6):
                tracker.tile_done(e0)
            self.assertEqual(released, [])            # block 0 still has a tile
            tracker.tile_done(0)                      # blocks 0 and 3 complete: rows 0..5
            self.assertEqual(released, [(0, 6)])
            tracker.tile_done(6)                      # 6 and 9 complete: every row
            self.assertEqual(released, [(0, 6), (6, 11)])
            tracker.finish()
            self.assertEqual(released, [(0, 6), (6, 11)])

    def test_only_whole_pages_inside_a_range_are_released(self):
        from ghost_backend.bor import streaming
        calls = []
        array = np.zeros(8192, dtype=np.complex128)

        class Mapping:
            def flush(self, offset, size):
                calls.append(("flush", offset, size))

            def madvise(self, option, start, length):
                calls.append(("drop", start, length))

        holder = mock.MagicMock(wraps=array)
        holder._mmap = Mapping()
        holder.dtype = array.dtype
        holder.ctypes.data = 0
        page = streaming._SPILL_PAGE
        with mock.patch.object(streaming.os, "name", "posix"), \
                mock.patch.object(streaming.mmap, "MADV_DONTNEED", 4, create=True):
            streaming._release_spilled(holder, [(10, 10 + 3 * page // 16), (0, 1)])
        drops = [c for c in calls if c[0] == "drop"]
        self.assertEqual(len(drops), 1)
        _, start, length = drops[0]
        self.assertEqual(start % page, 0)
        self.assertGreaterEqual(start, 10 * 16)
        self.assertLessEqual(start + length, (10 + 3 * page // 16) * 16)
        flushes = [c for c in calls if c[0] == "flush"]
        self.assertTrue(all(offset % streaming._SPILL_GRANULARITY == 0 for _, offset, _ in flushes))

    def test_spilled_and_ram_streams_assemble_the_same_systems(self):
        import tempfile
        from ghost_backend.bor import solver as bor, streaming
        solver = bor.BorPecSolver(bor.sphere_generatrix(0.1, 24), 1.5e9)
        zs = np.full(solver.P, 60.0 + 20.0j)
        ram = streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, ibc_zs_pt=zs)
        with tempfile.TemporaryDirectory() as base:
            spilled = streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, ibc_zs_pt=zs,
                                                   spill=base)
            self.assertIsInstance(spilled.Z, np.memmap)
            Nn = solver.Nn
            for m in (0, 2, -3, 5):
                systems = []
                for stream in (ram, spilled):
                    Z = np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
                    quads = (Z[:Nn, :Nn], Z[:Nn, Nn:], Z[Nn:, :Nn], Z[Nn:, Nn:])
                    stream.write_efie_blocks(m, quads, 0.3 - 0.7j)
                    stream.add_blocks("mfie", m, quads, -1.25)
                    stream.add_blocks("ibc", m, quads, 0.5j)
                    systems.append(Z)
                np.testing.assert_array_equal(systems[0], systems[1])
            spilled.close()
            spilled = None
            sp = bor.BorPecSolver(bor.sphere_generatrix(.035, 14), 1e9, medium=(2.5 - .05j, 1.))
            sq = bor.BorPecSolver(bor.sphere_generatrix(.02, 10), 1e9, medium=(2.5 - .05j, 1.))
            cross = bor.BorCrossOperators(sp, sq)
            ram = streaming.StreamingCrossFarBlocks(cross, 4)
            spilled = streaming.StreamingCrossFarBlocks(cross, 4, spill=base)
            self.assertIsInstance(spilled.Z, np.memmap)
            for m in (0, 3, -2):
                for mine, theirs in zip(spilled.efie_blocks(m) + spilled.bracket_blocks(m),
                                        ram.efie_blocks(m) + ram.bracket_blocks(m)):
                    np.testing.assert_array_equal(mine, theirs)
            spilled.close()
            spilled = None


class NodalBandTests(unittest.TestCase):
    """The node-summed tile contractions equal the former per-element-pair
    assembly (kept here as the reference) to rounding."""

    @staticmethod
    def element_pair_efie(Gn, left, right_groups, modes, ord_lo, k, f0, f1, re, go_p):
        from ghost_backend.bor import streaming as st
        modes = np.asarray(modes)
        nm, fc = len(modes), f1 - f0
        c0 = int(modes[0]) - ord_lo
        lower_index = np.abs(modes - 1) - ord_lo
        coefficients = {None: None, "-1": np.full(nm, -1.0 / k ** 2, dtype=complex),
                        "-jm": -(1j * modes / k ** 2), "+jm": (1j * modes / k ** 2),
                        "-m2": -(modes.astype(float) ** 2 / k ** 2)}
        tested = st._contract_test_side(Gn, left, re, go_p)
        band = np.zeros((4, fc + 1, re + 1, nm), dtype=np.complex128)
        for a in range(2):
            for lx, rights in st._EFIE_SOURCE_GROUPS:
                product = st._contract_source_group(tested[:, 2 * st._LEFT_INDEX[lx] + a],
                                                    right_groups[lx][f0:f1])
                for ri, rx in enumerate(rights):
                    uv, kernel, coefficient = st._EFIE_TERMS_BY_PAIR[(lx, rx)]
                    for b in range(2):
                        S = product[:, :, 2 * ri + b]
                        lower, upper = S[:, :, lower_index], S[:, :, c0 + 1:c0 + nm + 1]
                        piece = {"cen": S[:, :, c0:c0 + nm], "cos": 0.5 * (lower + upper),
                                 "sin": (lower - upper) / 2j, "-sin": (upper - lower) / 2j}[kernel]
                        if coefficients[coefficient] is not None:
                            piece = piece * coefficients[coefficient]
                        band[uv, b:b + fc, a:a + re] += piece
        return band

    @staticmethod
    def element_pair_brackets(Fs, left, right, f0, f1, re, go_p):
        from ghost_backend.bor import streaming as st
        fc, nm = f1 - f0, Fs[0].shape[-1]
        band = np.zeros((4, fc + 1, re + 1, nm), dtype=np.complex128)
        for uv, kernel in enumerate(Fs):
            tested = st._contract_test_side(kernel, left, re, go_p)
            for a in range(2):
                product = st._contract_source_group(tested[:, a], right[f0:f1])
                for b in range(2):
                    band[uv, b:b + fc, a:a + re] += product[:, :, b]
        return band

    def test_efie_and_bracket_bands(self):
        from ghost_backend.bor import streaming as st
        rng = np.random.default_rng(20260925)
        re, go, fc, f0 = 4, 3, 7, 2
        ne = f0 + fc + 1

        def cplx(*shape):
            return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        right = {name: rng.standard_normal((ne, go, 2)) for name in st._LEFT_KINDS}
        groups = st._stacked_right_groups(right)
        left = rng.standard_normal((10, re, go))
        for k in (3.0, 3.0 - 0.2j):
            for first in (0, 1, 4):
                modes = np.arange(first, first + 6)
                ord_lo = max(0, first - 1)
                Gn = cplx(re * go, fc * go, modes[-1] + 2 - ord_lo)
                args = (Gn, left, groups, modes, ord_lo, k, f0, f0 + fc, re, go)
                expected = self.element_pair_efie(*args)
                np.testing.assert_allclose(st._efie_band(*args), expected,
                                           rtol=0, atol=1e-13 * np.max(np.abs(expected)))
        Fs = [cplx(re * go, fc * go, 5) for _ in range(4)]
        left2 = rng.standard_normal((2, re, go))
        stacked = st._stacked_right(rng.standard_normal((ne, go, 2)))
        expected = self.element_pair_brackets(Fs, left2, stacked, f0, f0 + fc, re, go)
        np.testing.assert_allclose(st._bracket_band(Fs, left2, stacked, f0, f0 + fc, re, go),
                                   expected, rtol=0, atol=1e-13 * np.max(np.abs(expected)))


def _blas_threads():
    from ghost_backend.execution.thread_control import threadpool_info
    return max(int(pool["num_threads"]) for pool in threadpool_info()
               if pool.get("user_api") == "blas")


class ThreadPoolBlasTests(unittest.TestCase):
    """Pools of Python threads make their BLAS calls on one thread (OpenBLAS
    faulted under concurrent multithreaded calls from far tiles)."""

    def setUp(self):
        import ghost_backend.bor.solver  # noqa: F401  (loads SciPy's BLAS before limiting)
        from ghost_backend.execution.thread_control import threadpool_limits
        self.outer = threadpool_limits(limits=3, user_api="blas")
        self.addCleanup(self.outer.restore_original_limits)
        if _blas_threads() != 3:
            self.skipTest("BLAS thread count cannot be controlled here")

    def test_sections_nest_across_threads_and_the_last_restores(self):
        import threading
        from ghost_backend.execution.options import single_thread_blas
        entered, release, seen = threading.Event(), threading.Event(), []

        def other():
            with single_thread_blas():
                entered.set()
                release.wait(10)
                seen.append(_blas_threads())

        thread = threading.Thread(target=other)
        thread.start()
        entered.wait(10)
        with single_thread_blas():
            seen.append(_blas_threads())
        seen.append(_blas_threads())      # the other section is still open
        release.set()
        thread.join(10)
        self.assertEqual(seen, [1, 1, 1])
        self.assertEqual(_blas_threads(), 3)

    def test_far_tiles_and_local_near_threads_run_on_one_blas_thread(self):
        from ghost_backend.bor import solver as bor, streaming
        seen = []
        streaming._run_tiles(range(4), lambda tile: seen.append(_blas_threads()), 4)
        self.assertEqual(seen, [1] * 4)
        streaming._run_tiles(range(4), lambda tile: seen.append(_blas_threads()), 1)
        self.assertEqual(seen[4:], [3] * 4)            # serial tiles keep the share
        with mock.patch.object(bor, "_near_preparation_workers", lambda workers: workers):
            results = list(bor._iter_near_pairs(lambda pair: (pair, _blas_threads()),
                                                range(6), 3))
        self.assertEqual(results, [(pair, 1) for pair in range(6)])
        self.assertEqual(_blas_threads(), 3)


def _cap(radius, elements, start, stop, bulge=0.0):
    theta = np.linspace(start, stop, int(elements) + 1)
    local = radius * (1.0 + bulge * np.sin(2.0 * theta))
    return np.column_stack((local * np.sin(theta), local * np.cos(theta)))


class MaterialSpillTests(unittest.TestCase):
    """Material and junction solves spill their far blocks as solve_bor does:
    a budget short of every mode builds every mode once into memory-mapped
    files, with the answer of the in-memory ranges."""

    FREQUENCY_HZ = 1.0e9

    def setUp(self):
        import tempfile
        import shutil
        self.base = tempfile.mkdtemp(prefix="ghost-spilltest-")
        self.addCleanup(shutil.rmtree, self.base, True)
        self.radius = 299792458.0 / (2.0 * math.pi * self.FREQUENCY_HZ)

    def spill_dirs(self):
        import gc
        import glob
        gc.collect()
        return glob.glob(os.path.join(self.base, "ghost-bor-*"))

    def three_mode_budget(self, solve, kwargs):
        """The one-range in-memory solve and a budget of three of its modes."""
        from ghost_backend.bor import streaming
        single = streaming.plan_streaming_mode_block
        combined = streaming.plan_combined_streaming_mode_block
        per_mode = []

        def spy_single(n_elems, m_max, formulation, has_ibc, single_blocks, budget, workers):
            # solve_bor_dielectric gives each of its two streams half the budget.
            per_mode.append(2.0 * streaming.estimate_streaming_block_gb(
                n_elems, m_max, 1, formulation, has_ibc, single_blocks))
            return single(n_elems, m_max, formulation, has_ibc, single_blocks, budget, workers)

        def spy_combined(m_max, requirements, budget, workers):
            requirements = tuple(requirements)
            per_mode.append(streaming.combined_stream_mode_gb(m_max, requirements))
            return combined(m_max, requirements, budget, workers)

        with mock.patch.object(streaming, "plan_streaming_mode_block", spy_single), \
                mock.patch.object(streaming, "plan_combined_streaming_mode_block", spy_combined):
            reference = solve(assembly="streaming", stream_budget_gb=8.0,
                              bor_options=dict(factorization="dense"), **kwargs)
        self.assertEqual(len(per_mode), 1)
        return reference, 3.0 * per_mode[0]

    def check(self, solve, **kwargs):
        from ghost_backend.execution.options import execution_scope
        kwargs = dict(freq_hz=self.FREQUENCY_HZ, thetas_deg=[0.0, 37.0, 90.0, 151.0],
                      n_modes=8, workers=1, table_precision="double", **kwargs)
        reference, budget = self.three_mode_budget(solve, kwargs)
        self.assertEqual(reference["stream_spill_gb"], 0.0)
        with execution_scope(dict(temporary_directory=self.base)):
            spilled = solve(assembly="streaming", stream_budget_gb=budget,
                            bor_options=dict(factorization="dense"), **kwargs)
            ranged = solve(assembly="streaming", stream_budget_gb=budget,
                           bor_options=dict(factorization="dense", stream_spill="off"), **kwargs)
        # One build per stream, as when the budget holds every mode; the
        # in-memory ranges rebuild every stream per range.
        self.assertGreater(spilled["stream_spill_gb"], 0.0)
        self.assertEqual(spilled["stream_sweeps"], reference["stream_sweeps"])
        self.assertEqual(spilled["stream_mode_block"], reference["stream_mode_block"])
        self.assertEqual(ranged["stream_spill_gb"], 0.0)
        self.assertGreater(ranged["stream_sweeps"], reference["stream_sweeps"])
        for key in ("amp_vv", "amp_hh"):
            np.testing.assert_allclose(spilled[key], reference[key], rtol=2e-10, atol=2e-12)
            np.testing.assert_allclose(ranged[key], reference[key], rtol=2e-10, atol=2e-12)
        self.assertEqual(self.spill_dirs(), [])

    def test_dielectric(self):
        from ghost_backend.bor.solver import solve_bor_dielectric
        self.check(solve_bor_dielectric, points=_cap(self.radius, 10, 0.0, math.pi),
                   eps_r=3.0 - 0.1j, mu_r=1.0 - 0.02j)

    def test_coated(self):
        from ghost_backend.bor.solver import solve_bor_coated_pec
        self.check(solve_bor_coated_pec,
                   points_outer=_cap(1.2 * self.radius, 12, 0.0, math.pi),
                   points_core=_cap(0.8 * self.radius, 10, 0.0, math.pi),
                   eps_r=3.0 - 0.1j, mu_r=1.0 - 0.02j)

    def test_partial_coating(self):
        from ghost_backend.bor.solver import solve_bor_partial_coating
        self.check(solve_bor_partial_coating,
                   points_interface=_cap(self.radius, 5, 0.0, 0.5 * math.pi, bulge=0.12),
                   points_covered=_cap(self.radius, 5, 0.0, 0.5 * math.pi),
                   bare_pieces=[_cap(self.radius, 5, 0.5 * math.pi, math.pi)],
                   eps_r=2.5 - 0.08j, mu_r=1.0 - 0.01j)

    def test_two_layer_coating(self):
        from ghost_backend.bor.solver import solve_bor_coated2_pec
        self.check(solve_bor_coated2_pec,
                   points_outer=_cap(1.3 * self.radius, 10, 0.0, math.pi),
                   points_mid=_cap(1.05 * self.radius, 9, 0.0, math.pi),
                   points_core=_cap(0.8 * self.radius, 8, 0.0, math.pi),
                   eps_inner=2.2 - 0.06j, mu_inner=1.0 - 0.01j,
                   eps_outer=3.0 - 0.1j, mu_outer=1.0 - 0.02j)


class PreviewSpillMirrorTests(unittest.TestCase):
    """estimate_bor_resources prices the spill each streamed solve decides."""

    def setUp(self):
        import tempfile
        import shutil
        self.base = tempfile.mkdtemp(prefix="ghost-spilltest-")
        self.addCleanup(shutil.rmtree, self.base, True)

    def room(self, available=True):
        from ghost_backend.bor import streaming
        return mock.patch.object(streaming, "spill_directory",
                                 lambda _gb: self.base if available else None)

    def test_mirror_regimes(self):
        from ghost_backend.bor.dispatch import _mirror_stream_spill as mirror
        plan = (4, 1.5, 3)                              # 4 of 11 modes, 3 aligned workers
        with self.room():
            self.assertEqual(mirror(plan, 8, 10, 0.25),
                             (11, 0.5, 8, 2.75, 2.75, self.base))
            # Approximate streams: priced only when the solve surely spills,
            # else the in-memory block with every requested worker.
            self.assertEqual(mirror(plan, 8, 10, 0.25, exact=False, surely_short=True),
                             (11, 0.5, 8, 2.75, 2.75, self.base))
            self.assertEqual(mirror(plan, 8, 10, 0.25, exact=False, surely_short=False),
                             (4, 1.5, 8, 0.0, 2.75, None))
            # A block of every mode never spills.
            self.assertEqual(mirror((11, 2.75, 8), 8, 10, 0.25), (11, 2.75, 8, 0.0, 0.0, None))
        with self.room(False):
            # No room: the exact solve keeps its ranges; an approximate one
            # may still spill on its own smaller streams.
            self.assertEqual(mirror(plan, 8, 10, 0.25), (4, 1.5, 3, 0.0, 2.75, None))
            self.assertEqual(mirror(plan, 8, 10, 0.25, exact=False, surely_short=True),
                             (4, 1.5, 8, 0.0, 2.75, None))
        from ghost_backend.bor.options import option_scope, validate_options
        with option_scope(validate_options(dict(stream_spill="off"))):
            self.assertEqual(mirror(plan, 8, 10, 0.25), (4, 1.5, 3, 0.0, 0.0, None))

    def preview(self, snapshot, budget, workers=2):
        from ghost_backend.bor import dispatch
        return dispatch.estimate_bor_resources(
            snapshot, 1.0, [0.0, 90.0], geometry_units="meters", n_modes=8,
            workers=workers, table_precision="double", assembly="streaming",
            stream_budget_gb=budget, mesh_certification=False,
            bor_options=dict(factorization="dense"))

    @staticmethod
    def snapshot(segments, dielectrics):
        def segment(name, seg_type, points, material):
            return {"name": name, "seg_type": seg_type,
                    "properties": [str(seg_type), "6", "0", str(material), "0"],
                    "point_pairs": [{"x1": float(a[0]), "y1": float(a[1]),
                                     "x2": float(b[0]), "y2": float(b[1])}
                                    for a, b in zip(points[:-1], points[1:])]}
        return {"title": "spill preview", "ibcs": [], "dielectrics": dielectrics,
                "segments": [segment(*spec) for spec in segments]}

    def test_dielectric_preview_spills_both_streams(self):
        from ghost_backend.bor import streaming
        snapshot = self.snapshot([("body", 3, _cap(0.04, 12, 0.0, math.pi), 1)],
                                 [["1", "3.0", "-0.05", "1", "0"]])
        probe = self.preview(snapshot, 8.0)
        cap = probe["mode_cap_estimate"]
        per_mode = 2.0 * streaming.estimate_streaming_block_gb(
            probe["mesh_elements"], cap, 1, "efie", True, False)
        with self.room():
            estimate = self.preview(snapshot, 3.0 * per_mode, workers=6)
        self.assertEqual(probe["stream_mode_block_estimate"], cap + 1)
        self.assertEqual(estimate["stream_mode_block_estimate"], cap + 1)
        self.assertAlmostEqual(estimate["held_assembly_gb"],
                               streaming.STREAM_SPILL_RESIDENT_MODES * per_mode)
        self.assertAlmostEqual(estimate["stream_spill_gb_estimate"], (cap + 1) * per_mode)
        self.assertEqual(estimate["stream_spill_directory"], self.base)
        self.assertEqual(estimate["worker_plan"]["requested_workers"], 6)

    def test_partial_preview_spills_only_when_the_solve_surely_does(self):
        from ghost_backend.bor import streaming
        snapshot = self.snapshot(
            [("coating interface", 3, _cap(0.04, 6, 0.0, 0.5 * math.pi, bulge=0.12), 1),
             ("covered core", 4, _cap(0.04, 6, 0.0, 0.5 * math.pi), 1),
             ("bare core", 2, _cap(0.04, 6, 0.5 * math.pi, math.pi), 0)],
            [["1", "2.5", "-0.08", "1.0", "-0.01"]])
        probe = self.preview(snapshot, 8.0)
        cap = probe["mode_cap_estimate"]
        self.assertEqual(probe["surface_count"], 3)
        n = probe["mesh_elements"] // 3
        efie = streaming.estimate_rectangular_streaming_block_gb(n, n, cap, 1, False, False)
        # The layout prices seven rotated streams (two interface sides, the two
        # conductors, three crosses); the solve always builds the two interface
        # sides and the conductors' EFIE families.
        layout_total, always_total = (cap + 1) * 14.0 * efie, (cap + 1) * 6.0 * efie
        between = 0.5 * (layout_total + always_total)
        with self.room():
            maybe = self.preview(snapshot, between, workers=8)
            sure = self.preview(snapshot, 0.8 * always_total, workers=8)
        self.assertLess(maybe["stream_mode_block_estimate"], cap + 1)
        self.assertEqual(maybe["stream_spill_gb_estimate"], 0.0)
        self.assertAlmostEqual(maybe["stream_spill_candidate_gb"], layout_total)
        self.assertEqual(maybe["worker_plan"]["requested_workers"], 8)
        self.assertEqual(sure["stream_mode_block_estimate"], cap + 1)
        self.assertAlmostEqual(sure["stream_spill_gb_estimate"], layout_total)
        self.assertEqual(sure["stream_spill_directory"], self.base)
        self.assertEqual(sure["worker_plan"]["requested_workers"], 8)


if __name__ == "__main__":
    unittest.main()
