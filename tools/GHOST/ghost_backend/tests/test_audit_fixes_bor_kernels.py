"""Audit fixes of the BoR modal kernels (bor/kernels.py, bor/native/bor_stream_kernel.c).

Independent references: composite Gauss-Legendre rules graded toward the
near-singularity (ratio-2 panels starting far below d/a, every panel cut to a
small phase) for the Green's function, and a 3-D vector form of the MFIE/IBC
brackets with the cancellation-free R components for skew pairs.
"""
import io
import math
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import kernels
from ghost_backend.bor import streaming
import legacy_near_rules as legacy


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------

def _graded_nodes(delta, phase_per_rad, n=32, top=math.pi, max_phase=1.5):
    x, w = roots_legendre(n)
    edges = [0.0, max(delta * 1e-4, 1e-300)]
    while edges[-1] < top:
        edges.append(min(top, 2.0 * edges[-1]))
    X, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        pieces = max(1, int(math.ceil((b - a) * phase_per_rad / max_phase)))
        cuts = np.linspace(a, b, pieces + 1)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            X.append(0.5 * (c1 - c0) * x + 0.5 * (c1 + c0))
            W.append(0.5 * (c1 - c0) * w)
    return np.concatenate(X), np.concatenate(W)


def _reference_green(rp, zp, rq, zq, k, m_max):
    d = math.hypot(rp - rq, zp - zq)
    a2 = 4.0 * rp * rq
    a = math.sqrt(a2)
    x, w = _graded_nodes(d / a, m_max + 2 + abs(k) * a / 2.0)
    s = np.sin(0.5 * x)
    R = np.sqrt(d * d + a2 * s * s)
    f = np.exp(-1j * complex(k) * R) / (4 * np.pi * R)
    return 2.0 * np.cos(np.outer(np.arange(m_max + 2), x)) @ (w * f)


def _reference_brackets(family, point, k, m_max):
    """Orders 0..m_max of tt, tf, ft, ff from the 3-D vector form."""
    rp, zp, trp, tzp, rq, zq, trq, tzq = point
    d = math.hypot(rp - rq, zp - zq)
    a = 2.0 * math.sqrt(rp * rq)
    x, w = _graded_nodes(d / a, m_max + 3 + abs(k) * a / 2.0)
    cx, sx = np.cos(x), np.sin(x)
    h = 2.0 * np.sin(0.5 * x) ** 2
    one, zero = np.ones_like(x), np.zeros_like(x)
    Rv = np.stack([(rp - rq) + rq * h, rq * sx, (zp - zq) * one])     # r_p - r_q(xi), stable
    R = np.sqrt(d * d + 2.0 * rp * rq * h)
    kk = complex(k)
    p = (1 + 1j * kk * R) * np.exp(-1j * kk * R) / (4 * np.pi * R ** 3)
    t_p = np.stack([trp * one, zero, tzp * one])
    f_p = np.stack([zero, one, zero])
    n_p = np.stack([-tzp * one, zero, trp * one])
    t_q = np.stack([trq * cx, -trq * sx, tzq * one])
    f_q = np.stack([sx, cx, zero])
    n_q = np.cross(t_q, f_q, axis=0)
    dot = lambda u, v: np.sum(u * v, axis=0)
    values = []
    for W_ in (t_p, f_p):
        for J in (t_q, f_q):
            if family == 'mfie':
                values.append(-p * (dot(W_, Rv) * dot(n_p, J) - dot(W_, J) * dot(n_p, Rv)))
            else:
                values.append(p * dot(W_, np.cross(Rv, np.cross(n_q, J, axis=0), axis=0)))
    m = np.arange(m_max + 1)
    C, S = np.cos(np.outer(m, x)), np.sin(np.outer(m, x))
    return (2 * C @ (w * values[0]), -2j * S @ (w * values[1]),
            -2j * S @ (w * values[2]), 2 * C @ (w * values[3]))


def _relative(got, ref):
    got = got if isinstance(got, (tuple, list)) else (got,)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    scale = max(float(np.max(np.abs(r))) for r in ref)
    return max(float(np.max(np.abs(g - r))) for g, r in zip(got, ref)) / scale


def _pair(delta, rho=1.0, skew=0.0, alpha=0.927):
    """Two points d = 2 rho delta apart on an element of tangent angle alpha;
    the source tangent is turned by `skew`."""
    tr, tz = math.cos(alpha), math.sin(alpha)
    d = 2.0 * rho * delta
    return (rho, 0.0, tr, tz, rho + d * tr, d * tz, math.cos(alpha + skew), math.sin(alpha + skew))


def _native_has(name):
    library = streaming._NATIVE
    return library is not None and hasattr(library, name)


# --------------------------------------------------------------------------
# fix 2: no acceptance of a comparison with a capped (shared) order
# --------------------------------------------------------------------------

class SpuriousConvergenceTests(unittest.TestCase):
    K, M_MAX = 30.0, 40

    def test_green_at_d_over_a_1e8_is_accurate(self):
        # audit t15: the old checked rule returned 1.1e-3 (d/a 1e-8) and 1.7e-5
        # (3e-8) relative error without raising.
        for delta in (3e-8, 1e-8, 1e-9):
            point = _pair(delta)
            got = kernels.modal_kernels_near(*[np.array([point[i]]) for i in (0, 1, 4, 5)], self.K, self.M_MAX)[0]
            reference = _reference_green(point[0], point[1], point[4], point[5], self.K, self.M_MAX)
            self.assertLess(_relative(got, reference), 1e-12, delta)

    def test_brackets_at_d_over_a_1e8_are_accurate(self):
        for family, near in (('mfie', kernels.mfie_kernels_near), ('ibc', kernels.ibc_kernels_near)):
            for k in (self.K, self.K - 6.0j):
                point = _pair(1e-8, skew=0.4)
                got = near(*[np.array([v]) for v in point], k, self.M_MAX, signed=False)
                reference = _reference_brackets(family, point, k, self.M_MAX)
                self.assertLess(_relative([g[0] for g in got], reference), 1e-10, (family, k))

    def test_legacy_driver_raises_instead_of_accepting_a_capped_tail(self):
        # d/a = 1e-8 saturates the legacy tail at NEAR_ANGULAR_MAX_ORDER while the
        # core still grows: coarse and fine would share the tail.
        point = _pair(1e-8)
        args = tuple(np.array([point[i]]) for i in (0, 1, 4, 5))
        try:
            got = kernels._checked_near_kernels(legacy._modal_kernels_near_rule, args,
                                                self.K, self.M_MAX, 48, 0, False)[0]
        except ValueError as exc:
            self.assertIn('did not converge', str(exc))
        else:
            reference = _reference_green(point[0], point[1], point[4], point[5], self.K, self.M_MAX)
            self.assertLess(_relative(got, reference), 1e-7)

    def test_every_compared_level_increases_every_order(self):
        seen = []
        original = kernels._near_rule_moments

        def spy(kind, stable, points, delta, k, layout, orders, count):
            seen.append((kind, layout, tuple(int(o) for o in orders)))
            return original(kind, stable, points, delta, k, layout, orders, count)

        point = _pair(1e-6)
        with mock.patch.object(kernels, '_near_rule_moments', spy), \
                mock.patch.object(kernels, 'NEAR_ANGULAR_RTOL', 1e-300), \
                mock.patch.object(kernels, 'NEAR_ANGULAR_MAX_ORDER', 400):
            with self.assertRaisesRegex(ValueError, 'did not converge'):
                kernels.modal_kernels_near(*[np.array([point[i]]) for i in (0, 1, 4, 5)], self.K, 8)
            with self.assertRaisesRegex(ValueError, 'did not converge'):
                kernels.mfie_kernels_near(*[np.array([v]) for v in point], self.K, 8)
        self.assertGreater(len(seen), 4)
        for (kind0, layout0, first), (kind1, layout1, second) in zip(seen[:-1], seen[1:]):
            if (kind0, layout0) == (kind1, layout1) and len(first) == len(second) and first != second:
                if all(b >= a for a, b in zip(first, second)):
                    self.assertTrue(all(b > a for a, b in zip(first, second)), (first, second))
        self.assertTrue(all(max(orders) <= 400 for _, _, orders in seen))

    def test_accuracy_limit_is_still_reported(self):
        with mock.patch.object(kernels, 'NEAR_ANGULAR_MAX_ORDER', 128):
            with self.assertRaisesRegex(ValueError, 'accuracy limit'):
                kernels.modal_kernels_near([1.], [0.], [1.], [.01], 500., 512)


# --------------------------------------------------------------------------
# fix 3/4: graded near rule accuracy and cost
# --------------------------------------------------------------------------

class GradedNearRuleTests(unittest.TestCase):
    def test_green_rule_matches_reference_over_separations(self):
        for k, m_max in ((30.0, 10), (30.0 - 6.0j, 80), (300.0, 150)):
            for delta in (0.3, 1e-2, 1e-4, 1e-7):
                point = _pair(delta)
                got = kernels.modal_kernels_near(*[np.array([point[i]]) for i in (0, 1, 4, 5)], k, m_max)[0]
                reference = _reference_green(point[0], point[1], point[4], point[5], k, m_max)
                tolerance = 5e-12 if abs(k) > 100 else 1e-12
                self.assertLess(_relative(got, reference), tolerance, (k, m_max, delta))

    def test_bracket_rules_match_vector_reference(self):
        for family, near in (('mfie', kernels.mfie_kernels_near), ('ibc', kernels.ibc_kernels_near)):
            for k in (33.0, 33.0 - 4.0j):
                for delta in (0.2, 1e-3, 1e-6):
                    point = _pair(delta, rho=0.1, skew=0.5)
                    got = near(*[np.array([v]) for v in point], k, 12, signed=False)
                    reference = _reference_brackets(family, point, k, 12)
                    self.assertLess(_relative([g[0] for g in got], reference), 1e-11, (family, k, delta))

    def test_signed_and_nonnegative_bracket_layouts_agree(self):
        point = _pair(1e-3, rho=0.1, skew=0.5)
        arrays = [np.array([v]) for v in point]
        signed = kernels.mfie_kernels_near(*arrays, 33.0, 6)
        half = kernels.mfie_kernels_near(*arrays, 33.0, 6, signed=False)
        for full, part, odd in zip(signed, half, (False, True, True, False)):
            np.testing.assert_array_equal(full[:, 6:], part)
            np.testing.assert_array_equal(full[:, :6][:, ::-1], -part[:, 1:] if odd else part[:, 1:])

    def test_axis_pairs(self):
        for k in (50.0, 50.0 - 20.0j):
            green = kernels.modal_kernels_near([0.0], [0.0], [0.01], [-0.004], k, 5)[0]
            R = math.hypot(0.01, 0.004)
            self.assertAlmostEqual(abs(green[0] - 2 * np.pi * np.exp(-1j * k * R) / (4 * np.pi * R)), 0.0, delta=1e-14)
            self.assertTrue(np.all(green[1:] == 0))
            for near in (kernels.mfie_kernels_near, kernels.ibc_kernels_near):
                values = near([0.0], [0.0], [0.6], [-0.8], [0.01], [-0.004], [0.6], [-0.8], k, 5)
                self.assertTrue(all(np.all(np.isfinite(v)) for v in values))
            point = (1e-5, 0.0, 0.6, -0.8, 2e-5, -0.004, 0.6, -0.8)
            got = kernels.mfie_kernels_near(*[np.array([v]) for v in point], k, 4, signed=False)
            self.assertLess(_relative([g[0] for g in got], _reference_brackets('mfie', point, k, 4)), 1e-11)

    def test_samples_per_point_stay_bounded_at_small_separation(self):
        # the single-tail rule needed ~26,000 samples per point at d/a = 1e-7
        counted = []
        original = kernels._near_rule_moments

        def spy(kind, stable, points, delta, k, layout, orders, count):
            counted.append(len(delta) * int(np.sum(orders)))
            return original(kind, stable, points, delta, k, layout, orders, count)

        with mock.patch.object(kernels, '_near_rule_moments', spy):
            for delta in (1e-3, 1e-7):
                counted.clear()
                point = _pair(delta)
                kernels.modal_kernels_near(*[np.array([point[i]]) for i in (0, 1, 4, 5)], 30.0, 10)
                self.assertLess(sum(counted), 1000, delta)

    def test_results_do_not_depend_on_the_batch(self):
        rng = np.random.default_rng(3)
        n = 40
        rho = rng.uniform(0.05, 0.2, n)
        gap = 10 ** rng.uniform(-7, -1, n)
        angle = rng.uniform(0, np.pi, n)
        args = (rho, np.zeros(n), np.cos(angle), np.sin(angle), rho + gap * np.cos(angle), gap * np.sin(angle),
                np.cos(angle + 0.3), np.sin(angle + 0.3))
        batch = kernels.ibc_kernels_near(*args, 40.0 - 2.0j, 9)
        for index in (0, 17, 39):
            alone = kernels.ibc_kernels_near(*[a[index:index + 1] for a in args], 40.0 - 2.0j, 9)
            for x, y in zip(batch, alone):
                np.testing.assert_array_equal(x[index], y[0])

    def test_numpy_fallback_agrees_with_native(self):
        if not _native_has('near_brackets_rule'):
            self.skipTest('native near rule kernels are not built here')
        rng = np.random.default_rng(8)
        n = 24
        rho = rng.uniform(0.05, 0.2, n)
        gap = 10 ** rng.uniform(-6, -1, n)
        angle = rng.uniform(0, np.pi, n)
        args = (rho, np.zeros(n), np.cos(angle), np.sin(angle), rho + gap * np.cos(angle), gap * np.sin(angle),
                np.cos(angle + 0.2), np.sin(angle + 0.2))
        native_g = kernels.modal_kernels_near(args[0], args[1], args[4], args[5], 30.0 - 1j, 7)
        native_b = kernels.mfie_kernels_near(*args, 30.0 - 1j, 7)
        with mock.patch.object(kernels, '_native_entry', lambda name: None):
            numpy_g = kernels.modal_kernels_near(args[0], args[1], args[4], args[5], 30.0 - 1j, 7)
            numpy_b = kernels.mfie_kernels_near(*args, 30.0 - 1j, 7)
        self.assertLess(_relative(numpy_g, native_g), 1e-12)
        self.assertLess(_relative(list(numpy_b), list(native_b)), 1e-10)


# --------------------------------------------------------------------------
# fix 1/5: native projections and samplers
# --------------------------------------------------------------------------

class NativeKernelTests(unittest.TestCase):
    def test_parity_moments_match_direct_projection(self):
        rng = np.random.default_rng(1)
        n, na, count = 9, 157, 37
        even = rng.standard_normal((n, 4, na))
        odd = rng.standard_normal((n, 4, na))
        xi = np.sort(rng.uniform(0, np.pi, (n, na)), axis=1)
        orders = np.arange(count)
        direct_c = np.einsum('nra,nam->nrm', even, np.cos(xi[:, :, None] * orders))
        direct_s = np.einsum('nra,nam->nrm', odd, np.sin(xi[:, :, None] * orders))
        scale = np.sum(np.abs(even), axis=2).max()
        cosines, sines = kernels._parity_moments(even, odd, xi, count)
        self.assertLess(np.max(np.abs(cosines - direct_c)) / scale, 1e-14)
        self.assertLess(np.max(np.abs(sines - direct_s)) / scale, 1e-14)
        if _native_has('parity_moments') and _native_has('trig_moments'):
            # one fused pass == the two single-parity calls, bitwise
            separate_c = kernels._native_trig_moments(even, xi, count, False)[0]
            separate_s = kernels._native_trig_moments(odd, xi, count, True)[1]
            np.testing.assert_array_equal(cosines, separate_c)
            np.testing.assert_array_equal(sines, separate_s)
        with mock.patch.object(kernels, '_native_entry', lambda name: None), \
                mock.patch.object(kernels, '_native_trig_moments', lambda *a, **k: None), \
                redirect_stderr(io.StringIO()):
            numpy_c, numpy_s = kernels._parity_moments(even, odd, xi, count)
        self.assertLess(np.max(np.abs(numpy_c - direct_c)) / scale, 1e-14)
        self.assertLess(np.max(np.abs(numpy_s - direct_s)) / scale, 1e-14)

    def test_trig_moments_high_orders(self):
        if not _native_has('trig_moments'):
            self.skipTest('native trig_moments is not built here')
        rng = np.random.default_rng(2)
        X = rng.standard_normal((2, 2, 1500))
        xi = np.sort(rng.uniform(0, np.pi, (2, 1500)), axis=1)
        cosine, sine = kernels._native_trig_moments(X, xi, 1200, True)
        orders = np.arange(1200)
        scale = np.sum(np.abs(X), axis=2)[:, :, None]
        self.assertLess(np.max(np.abs(cosine - np.einsum('nra,nam->nrm', X, np.cos(xi[:, :, None] * orders))) / scale), 1e-13)
        self.assertLess(np.max(np.abs(sine - np.einsum('nra,nam->nrm', X, np.sin(xi[:, :, None] * orders))) / scale), 1e-13)

    def test_near_green_sampler_matches_numpy_form(self):
        if not _native_has('near_green'):
            self.skipTest('native near_green is not built here')
        rng = np.random.default_rng(4)
        n, na = 20, 33
        rp = rng.uniform(0, 1, n); rq = rp + rng.uniform(-1e-3, 1e-3, n); rq[0] = 0.0
        zp = rng.uniform(-1, 1, n); zq = zp + rng.uniform(-1e-3, 1e-3, n)
        xi = np.sort(rng.uniform(0, np.pi, (n, na)), axis=1)
        for k in (30.0, 30.0 - 6.0j):
            native = kernels._green_samples(rp, zp, rq, zq, k, xi)
            with mock.patch.object(kernels, '_native_entry', lambda name: None):
                numpy_form = kernels._green_samples(rp, zp, rq, zq, k, xi)
            self.assertLess(np.max(np.abs(native - numpy_form) / np.abs(numpy_form)), 1e-15)

    def test_stable_bracket_sampler_matches_numpy_closed_forms(self):
        if not _native_has('near_brackets_stable'):
            self.skipTest('native near_brackets_stable is not built here')
        rng = np.random.default_rng(6)
        n, na = 16, 21
        rho = 0.1 + 0.01 * rng.random(n); gap = 10 ** rng.uniform(-9, -2, n); angle = rng.uniform(0, 2 * np.pi, n)
        points = (rho, np.zeros(n), np.cos(angle), np.sin(angle), rho + gap * np.cos(angle), gap * np.sin(angle),
                  np.cos(angle + 0.3), np.sin(angle + 0.3))
        xi = np.sort(rng.uniform(0, np.pi, (n, na)), axis=1)
        w = rng.uniform(0, 1, (n, na))
        for family in ('mfie', 'ibc'):
            reference = kernels._stable_brackets(points, 33.0 - 4j, xi, family)
            moments = kernels._bracket_node_moments(family, True, points, 33.0 - 4j, xi, w, 5)
            with mock.patch.object(kernels, '_native_entry', lambda name: None), redirect_stderr(io.StringIO()):
                numpy_moments = kernels._bracket_node_moments(family, True, points, 33.0 - 4j, xi, w, 5)
            for a, b in zip(moments, numpy_moments):
                self.assertLess(np.max(np.abs(a - b)) / np.max(np.abs(b)), 1e-14)
            self.assertTrue(all(np.all(np.isfinite(r)) for r in reference))


# --------------------------------------------------------------------------
# fix 6/8/10/13: far bandwidth rule, tables, dtype
# --------------------------------------------------------------------------

def _half_grid_reference(kind, coords, k, modes, size=4096):
    half = size // 2 + 1
    xi = 2 * np.pi * np.arange(half) / size - np.pi
    w = np.full(half, 2.0 * (2 * np.pi / size)); w[0] = w[-1] = 2 * np.pi / size
    C = np.cos(np.outer(xi, modes)); S = np.sin(np.outer(xi, modes))
    if kind == 'g':
        a, b, c, d = coords
        R = np.sqrt(((a - c) ** 2 + (b - d) ** 2)[:, None] + 4 * (a * c)[:, None] * np.sin(0.5 * xi) ** 2)
        return ((np.exp(-1j * complex(k) * R) / (4 * np.pi * R) * w) @ C,)
    with mock.patch.object(kernels, '_native_brackets', lambda *a, **kw: None), \
            mock.patch.object(kernels, '_native_mfie_brackets', lambda *a, **kw: None):
        F = (kernels._mfie_brackets(*coords, k, xi) if kind == 'mfie'
             else kernels._ibc_brackets_grid(*coords, k, np.broadcast_to(xi, (len(coords[0]), half))))
    return ((F[0] * w) @ C, (F[1] * w) @ (-1j * S), (F[2] * w) @ (-1j * S), (F[3] * w) @ C)


class FarRuleTests(unittest.TestCase):
    def setUp(self):
        radius, self.k = 1.0, 20.0
        theta = np.linspace(0, np.pi, 64)
        gen = kernels.Generatrix(np.column_stack([radius * np.sin(theta), radius * np.cos(theta)]))
        g = kernels.gauss_on_generatrix(gen, 4)
        rng = np.random.default_rng(12)
        ip = rng.integers(0, g.rho.size, 200); iq = rng.integers(0, g.rho.size, 200)
        keep = np.abs(g.elem[ip] - g.elem[iq]) > 2
        self.ip, self.iq, self.g = ip[keep], iq[keep], g

    def coords(self, kind):
        g, ip, iq = self.g, self.ip, self.iq
        if kind == 'g':
            return (g.rho[ip], g.z[ip], g.rho[iq], g.z[iq])
        return (g.rho[ip], g.z[ip], g.trho[ip], g.tz[ip], g.rho[iq], g.z[iq], g.trho[iq], g.tz[iq])

    def test_bandwidth_rule_matches_dense_half_grid(self):
        for k in (self.k, self.k - 3.0j):
            for kind in ('g', 'mfie', 'ibc'):
                m_max = 26
                modes = np.arange(m_max + 2) if kind == 'g' else np.arange(m_max + 1)
                coords = self.coords(kind)
                near = np.zeros(len(coords[0]), bool)
                got = kernels.banded_modal_kernels(kind, coords, k, m_max, near, modes, threads=2)
                got = got if kind != 'g' else (got,)
                reference = _half_grid_reference(kind, coords, k, modes)
                scale = np.maximum.reduce([np.max(np.abs(r), axis=1) for r in reference])
                error = np.maximum.reduce([np.max(np.abs(a - b), axis=1) for a, b in zip(got, reference)])
                live = scale > 0
                self.assertLess(float(np.max(error[live] / scale[live])), 1e-13, (k, kind))

    def test_sample_counts_are_lower_and_bounded_by_n_xi_for_pairs(self):
        g, ip, iq = self.g, self.ip, self.iq
        gap = np.hypot(g.rho[ip] - g.rho[iq], g.z[ip] - g.z[iq])
        for bracket in (False, True):
            m_max = 26
            counts = kernels._far_sample_counts(bracket, g.rho[ip], g.rho[iq], gap, self.k, m_max + (0 if bracket else 1))
            self.assertTrue(np.all(counts % 32 == 0))
            bound = kernels.n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max, float(gap.min()), bracket=bracket)
            self.assertGreaterEqual(bound, int(counts.max()))
            radius = np.maximum(g.rho[ip], g.rho[iq])
            old = np.exp2(np.ceil(np.log2(np.maximum(
                np.maximum(max(128, 6 * (m_max + 2)) if bracket else max(64, 4 * (m_max + 2)),
                           (8 if bracket else 6) * (2 * self.k * radius + 4)), 16 * np.pi * radius / gap))))
            self.assertLess(counts.sum(), 0.8 * old.sum())

    def test_safety_cap_still_rejects_close_pairs(self):
        with self.assertRaisesRegex(ValueError, 'safety cap'):
            kernels.banded_modal_kernels('g', ([1.0], [0.0], [1.0], [1e-4]), 1.0, 2, np.zeros(1, bool))
        with self.assertRaisesRegex(ValueError, 'safety cap'):
            kernels.n_xi_for_pairs(1.0, 1.0, 2, 1e-4)

    def test_single_precision_tables_are_the_rounded_double_tables(self):
        coords = self.coords('mfie')
        near = np.zeros(len(coords[0]), bool)
        double = kernels.nonnegative_bracket_tables('mfie', coords, self.k, 10, 0, near)
        single = kernels.nonnegative_bracket_tables('mfie', coords, self.k, 10, 0, near, out_dtype=np.complex64)
        for a, b in zip(double, single):
            self.assertEqual(b.dtype, np.complex64)
            np.testing.assert_array_equal(a.astype(np.complex64), b)
        green = kernels.modal_kernels_fft(*self.coords('g'), self.k, 10, near_mask=near, out_dtype=np.complex64)
        self.assertEqual(green.dtype, np.complex64)
        with self.assertRaises(ValueError):
            kernels.banded_modal_kernels('g', self.coords('g'), self.k, 10, near, out_dtype=np.float64)

    def test_transform_tables_are_cached_and_the_green_path_skips_sines(self):
        kernels._FAR_TABLES.clear()
        kernels._FAR_TABLE_BYTES[0] = 0
        coords = self.coords('g')
        near = np.zeros(len(coords[0]), bool)
        first = kernels.banded_modal_kernels('g', coords, self.k, 10, near)
        cached = dict(kernels._FAR_TABLES)
        self.assertTrue(cached)
        self.assertTrue(all(entry[1] is None for entry in cached.values()))
        second = kernels.banded_modal_kernels('g', coords, self.k, 10, near)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(cached), set(kernels._FAR_TABLES))


# --------------------------------------------------------------------------
# fix 7, 9, 11, 12
# --------------------------------------------------------------------------

class HousekeepingTests(unittest.TestCase):
    def test_thread_count_honours_every_allocation(self):
        with mock.patch.object(kernels, '_host_physical_cores', return_value=8), \
                mock.patch.object(kernels, '_affinity_cpu_count', return_value=16), \
                mock.patch.object(kernels, '_scheduler_cpu_budget', return_value=None), \
                mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '', 'SLURM_CPUS_ON_NODE': ''}):
            self.assertEqual(kernels.physical_cpu_count(), 8)
            with mock.patch.object(kernels, '_affinity_cpu_count', return_value=3):
                self.assertEqual(kernels.physical_cpu_count(), 3)
            with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}):
                self.assertEqual(kernels.physical_cpu_count(), 2)
            with mock.patch.dict(os.environ, {'SLURM_CPUS_ON_NODE': '5'}):
                self.assertEqual(kernels.physical_cpu_count(), 5)
            with mock.patch.object(kernels, '_scheduler_cpu_budget', return_value=0):
                self.assertEqual(kernels.physical_cpu_count(), 8)
        with mock.patch.object(kernels, '_host_physical_cores', return_value=None), \
                mock.patch.object(kernels, '_affinity_cpu_count', return_value=None), \
                mock.patch.object(kernels, '_scheduler_cpu_budget', return_value=None), \
                mock.patch.object(kernels.os, 'cpu_count', return_value=None):
            self.assertEqual(kernels.physical_cpu_count(), 1)

    def test_scheduler_allocation_bounds_native_teams(self):
        from ghost_backend.execution.options import DEFAULTS, execution_scope
        with execution_scope(dict(DEFAULTS), assembly_threads=2):
            self.assertLessEqual(kernels.physical_cpu_count(), 2)
        self.assertGreaterEqual(kernels.physical_cpu_count(), 1)

    def test_gauss_rule_cache_holds_data_dependent_orders(self):
        self.assertGreaterEqual(kernels.cached_leggauss.cache_info().maxsize, 1024)

    def test_mfie_for_mode_accepts_only_known_layouts(self):
        m_max = 3
        centered = np.arange(2 * m_max + 1, dtype=float)[None, :] * np.ones((2, 1))
        nonnegative = np.arange(m_max + 1, dtype=float)[None, :] * np.ones((2, 1))
        np.testing.assert_array_equal(kernels.mfie_for_mode(centered, -2, m_max), centered[:, 1])
        np.testing.assert_array_equal(kernels.mfie_for_mode(nonnegative, -2, m_max, odd=True), -nonnegative[:, 2])
        np.testing.assert_array_equal(kernels.mfie_for_mode(nonnegative, -2, m_max), nonnegative[:, 2])
        with self.assertRaises(ValueError):
            kernels.mfie_for_mode(np.zeros((2, 5)), 1, m_max)       # neither m_max+1 nor 2*m_max+1
        with self.assertRaises(ValueError):
            kernels.mfie_for_mode(nonnegative, 4, m_max)             # outside the table
        np.testing.assert_array_equal(kernels.mfie_for_mode(np.ones((2, 1)), 0, 0), np.ones(2))

    def test_numpy_fallback_of_the_table_path_is_announced(self):
        class Legacy:            # a library built before the paired samplers
            pass
        kernels._NOTICE_SHOWN.clear()
        stderr = io.StringIO()
        with mock.patch.object(streaming, '_NATIVE', Legacy()), redirect_stderr(stderr):
            values = kernels.banded_modal_kernels('g', ([1.0], [0.0], [1.0], [1.0]), 3.0, 2, np.zeros(1, bool))
            kernels.banded_modal_kernels('g', ([1.0], [0.0], [1.0], [1.0]), 3.0, 2, np.zeros(1, bool))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(stderr.getvalue().count('build_kernel.py'), 1)
        self.assertIn('sample_g_pairs', stderr.getvalue())
        kernels._NOTICE_SHOWN.clear()


if __name__ == '__main__':
    unittest.main()
