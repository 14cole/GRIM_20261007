"""September 2026 audit fixes of the 2-D boundary operators and their assembly.

Each test checks one fix against an independent reference or against the
arithmetic it must reproduce bit for bit.
"""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import hankel2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.operators as ops
import ghost_backend.twod.polynomial_quadrature as pq
from ghost_backend.twod.assembly import kernels, separation
from ghost_backend.twod.assembly.scatter import MatrixDestination, SystemScatter
from ghost_backend.twod.basis import values as basis_values, derivative_matrix
from ghost_backend.execution.cpu import CPUState, _STATE
from ghost_backend.execution.options import execution_scope, validate_options


def panels(points, closed=True):
    result = []
    count = len(points)
    for i in range(count if closed else count - 1):
        a = np.asarray(points[i], float)
        b = np.asarray(points[(i + 1) % count], float)
        length = float(np.linalg.norm(b - a))
        tangent = (b - a) / length
        result.append(rcs.Panel('p', 2, 0, 0, 0, a, b, (a + b) / 2, tangent,
                                np.array([-tangent[1], tangent[0]]), length))
    return result


def mesh_of(points, closed=True):
    return rcs._build_linear_mesh(panels(points, closed))


def circle(radius, count, center=(0., 0.)):
    theta = -2 * np.pi * np.arange(count) / count
    return [np.asarray(center) + radius * np.array([np.cos(t), np.sin(t)]) for t in theta]


def fused(mesh, k, mask=None):
    n = len(mesh.nodes)
    ids = np.arange(n)
    mats = [np.zeros((n, n), complex, order='F') for _ in range(4)]
    ops._assemble_linear_operator_matrices_multi(mesh, k, True, [mask],
        output_node_ids_many=[(ids, ids)], double_layer_output_node_ids_many=[(ids, ids)],
        operator_outputs=[(MatrixDestination(mats[0], n), MatrixDestination(mats[1], n))],
        additional_operator_outputs=[(MatrixDestination(mats[2], n), MatrixDestination(mats[3], n))])
    return mats


def composite_blocks(p0, p1, q0, q1, k, pieces=96, order=20):
    """Independent S and K' (observation normal) blocks of two straight panels."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x + 1) / 2, w / 2
    t = (np.arange(pieces)[:, None] + x[None, :]).ravel() / pieces
    wt = np.tile(w, pieces) / pieces
    lo, ls = np.linalg.norm(p1 - p0), np.linalg.norm(q1 - q0)
    diff = (p0 + t[:, None] * (p1 - p0))[:, None, :] - (q0 + t[:, None] * (q1 - q0))[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    to = (p1 - p0) / lo
    no = np.array([-to[1], to[0]])
    green = .25j * hankel2(0, k * r)
    derivative = -.25j * k * hankel2(1, k * r) * (diff @ no) / r
    phi = np.column_stack((1 - t, t))
    weights = np.outer(wt, wt)
    return (phi.T @ (weights * green) @ phi * lo * ls,
            phi.T @ (weights * derivative) @ phi * lo * ls)


class KernelTableTests(unittest.TestCase):
    """Item 1: real-k tables for large domains; no pointless degree retry."""

    def test_table_tolerance_follows_kernel_conditioning(self):
        k = 2 * np.pi
        upper = 300.0
        table = kernels.KernelTable(k, upper)
        self.assertLessEqual(table.evidence['max_check_to_tolerance'], 1.0)
        self.assertGreater(table.evidence['max_check_relative'], 2e-13)  # the former fixed limit
        rng = np.random.default_rng(7)
        r = np.sort(rng.uniform(20.0, upper, 4000))
        got = np.stack([p(r) for p in table.polys], axis=-1)
        reference = kernels.values(k, r)
        relative = np.abs(got - reference) / np.abs(reference)
        self.assertLess(float(np.max(relative / (8 * np.finfo(float).eps * k * r[:, None]))), 1.0)

    def test_large_domain_keeps_a_table_and_caps_it_at_the_interval_budget(self):
        reach = kernels.table_reach(2 * np.pi)
        self.assertGreater(2 * np.pi * reach, 2000.0)
        for radius, partial in ((60.0, False), (160.0, True)):
            mesh = mesh_of(circle(radius, 24))
            state = CPUState()
            with _STATE.override(state):
                green, _ = kernels.select_far_kernels(mesh, 2 * np.pi, ops._far_green_into, ops._far_hankel1_into)
            self.assertTrue(state.table_events[-1]['used'])
            self.assertIs(state.table_events[-1]['partial_domain'], partial)
            self.assertTrue(hasattr(green, 'table'))

    def test_domain_rejections_are_not_retried_at_a_higher_degree(self):
        mesh = mesh_of(circle(1.0, 12))
        degrees = []

        def rejected(k, upper, degree=12, tolerance=2e-13):
            degrees.append(degree)
            raise kernels.DomainRejected('Interval budget exceeded')
        state = CPUState()
        with _STATE.override(state), mock.patch.object(kernels, 'KernelTable', rejected):
            kernels.select_far_kernels(mesh, 5.0, ops._far_green_into, ops._far_hankel1_into)
        self.assertEqual(degrees, [12])
        self.assertFalse(state.table_events[-1]['used'])

    def test_native_block_beyond_a_real_table_matches_exact_kernels(self):
        from ghost_backend.twod.assembly.native import far
        if far.library() is None:
            self.skipTest('native far-block library unavailable')
        k = 2 * np.pi
        table = kernels.KernelTable(k, 128 / k)
        distances = np.geomspace(128 / k * 1.01, 900.0, 23)
        source = np.stack([distances, np.zeros_like(distances)], axis=-1)[:, None, :]
        s, d, _ = far.far_block(table, k, np.zeros((1, 1, 2)), source, np.ones(1), np.ones((1, 1)),
                                np.array([[0., 1.]]), np.tile([1., 0.], (len(distances), 1)),
                                np.ones((1, len(distances))), False, True, True, False)
        reference = kernels.values(k, distances)
        # Both sides round k*r once: agreement is limited to a few eps |k| r.
        limit = np.maximum(1e-14, 8 * np.finfo(float).eps * k * distances)
        self.assertTrue(np.all(abs(s[0, 0] - reference[:, 0]) <= limit * abs(reference[:, 0])))
        self.assertTrue(np.all(abs(d[0, 0] + reference[:, 1]) <= limit * abs(reference[:, 1])))


class HypersingularTests(unittest.TestCase):
    """Item 2: the separate W is the fused engine's W-only pass."""

    def test_separate_w_is_the_fused_w_with_masks_compact_and_destinations(self):
        mesh = mesh_of(circle(1.0, 60))
        e = len(mesh.elements)
        mask = np.arange(e) % 3 != 1
        for k in (7.0, 7.0 - 2.5j):
            for route in ('scipy', 'cpu'):
                with (_STATE.override(CPUState()) if route == 'cpu' else nullcontext()):
                    w_fused = fused(mesh, k, mask)[3]
                    w_alone = ops._assemble_linear_hypersingular_matrix(mesh, k, source_element_mask=mask)
                    np.testing.assert_array_equal(np.asarray(w_alone), w_fused)
                    n = len(mesh.nodes)
                    rows, cols = np.arange(0, n, 3), np.arange(1, n, 2)
                    compact = ops._assemble_linear_hypersingular_matrix(
                        mesh, k, source_element_mask=mask, output_node_ids=(rows, cols))
                    np.testing.assert_array_equal(compact.values, w_fused[np.ix_(rows, cols)])
                    storage = np.ones((2 * n, 2 * n), complex, order='F')
                    view = storage[n:, :n]
                    returned = ops._assemble_linear_hypersingular_matrix(mesh, k, source_element_mask=mask,
                                                                         destination=view)
                    self.assertTrue(np.shares_memory(returned, storage))
                    np.testing.assert_allclose(view - 1, w_fused, rtol=0, atol=4e-16 * np.max(abs(view)))

    def test_w_matches_maue_identity_of_independent_blocks(self):
        # Separated panels (graded far rule with the W floor) and a close pair.
        for q0, q1 in (((3.2, 0.4), (4.1, 1.0)), ((0.3, 0.6), (1.3, 0.6))):
            p0, p1 = np.array([0., 0.]), np.array([1., 0.])
            q0, q1 = np.asarray(q0), np.asarray(q1)
            mesh = rcs._build_linear_mesh(panels([p0, p1], False) + panels([q0, q1], False))
            k = 1.7 - .2j
            w = np.asarray(ops._assemble_linear_hypersingular_matrix(mesh, k))
            s, _ = composite_blocks(p0, p1, q0, q1, k)
            e0, e1 = mesh.elements
            expected = ops._hypersingular_block_from_s_block(s, k, e0.normal, e1.normal, e0.length, e1.length)
            np.testing.assert_allclose(w[:2, 2:], expected, rtol=1e-11, atol=1e-13 * np.max(abs(expected)))


class TouchingTests(unittest.TestCase):
    """Items 3 and 4: one touching tolerance, and the graded corner rule."""

    def test_snapped_node_with_offset_endpoints_is_a_touching_pair(self):
        results = []
        for gap in (0.0, 3e-10):
            points = [np.array([0., 0.]), np.array([.01, 0.]), np.array([.01 + gap, 0.]), np.array([.02, .003])]
            mesh = rcs._build_linear_mesh(panels(points[:2], False) + panels(points[2:], False))
            self.assertEqual(len(mesh.nodes), 3)
            s, kp = ops._assemble_linear_operator_matrices(mesh, 60.0, True)
            w = np.asarray(ops._assemble_linear_hypersingular_matrix(mesh, 60.0))
            results.append((s, kp, w))
        for a, b in zip(*results):
            self.assertLess(np.max(abs(a - b)) / np.max(abs(b)), 1e-6)
        e0, e1 = mesh.elements
        self.assertIsNotNone(ops._linear_shared_interval_endpoint_info(e0, (0., 1.), e1, (0., 1.)))

    def test_vectorized_endpoint_test_matches_the_pair_rule(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(-1, 1, (40, 2))
        points = np.concatenate([base, base + rng.choice([0., 4e-10, 3e-9], (40, 1)) * rng.standard_normal((40, 2))])
        segments = [(points[i], points[j]) for i, j in rng.integers(0, 80, (60, 2)) if i != j]
        pans = [p for a, b in segments for p in panels([a, b], False)]
        mesh = rcs._build_linear_mesh(pans)
        el = mesh.elements
        p0 = np.array([e.p0 for e in el]); p1 = np.array([e.p1 for e in el])
        ids = np.array([e.node_ids for e in el])
        obs, src = np.meshgrid(np.arange(len(el)), np.arange(len(el)), indexing='ij')
        obs, src = obs.ravel(), src.ravel()
        touching, obs_start, src_start = ops._shared_endpoints(p0, p1, ids, obs, src)
        for i, (o, s) in enumerate(zip(obs, src)):
            info = ops._linear_shared_interval_endpoint_info(el[o], (0., 1.), el[s], (0., 1.))
            self.assertEqual(info is not None, bool(touching[i]))
            if info is not None:
                self.assertEqual(info, (bool(obs_start[i]), bool(src_start[i])))

    def test_touching_blocks_converge_to_the_graded_reference(self):
        from ghost_backend.twod.polynomial_quadrature import block
        cases = {'collinear': ((0, 0), (1, 0), (1, 0), (2, 0)),
                 'corner 90': ((0, 0), (1, 0), (1, 0), (1, 1)),
                 'reflex 170': ((0, 0), (1, 0), (1, 0), (1 + np.cos(np.radians(170)), np.sin(np.radians(170)))),
                 'reversed': ((0, 0), (1, 0), (2, 0), (1, 0))}
        for name, (a, b, c, d) in cases.items():
            mesh = rcs._build_linear_mesh(panels([a, b], False) + panels([c, d], False))
            e0, e1 = mesh.elements
            shared = ops._linear_shared_interval_endpoint_info(e0, (0., 1.), e1, (0., 1.))
            for k in (3.0, 2.0 - 1.5j):
                for derivative in (True, False):
                    reference = block(e0, e1, k, derivative, order=144)
                    batch = ops._integrate_linear_touching_pairs_sk_batched([(e0, e1)], [shared], k, derivative, 9)
                    single = ops._integrate_linear_touching_duffy_sk_vectorized(
                        e0, e1, k, derivative, (0., 1.), (0., 1.), *shared, order=9)
                    for got, alone, expected in zip(batch, single, reference):
                        scale = np.max(abs(expected)) if np.max(abs(expected)) > 1e-14 else np.max(abs(reference[0]))
                        self.assertLess(np.max(abs(got[0] - expected)) / scale, 2e-9, (name, k, derivative))
                        np.testing.assert_allclose(got[0], alone, rtol=3e-14, atol=1e-16 * scale)


class FarRuleTests(unittest.TestCase):
    """Items 5 and 8: calibrated W floor and degree-aware graded far orders."""

    @staticmethod
    def blocks(q, wave, lo, ls, angle, phi_c, degree, distance):
        x, w = np.polynomial.legendre.leggauss(q)
        x, w = (x + 1) / 2, w / 2
        ts = np.array([np.cos(angle), np.sin(angle)])
        ns = np.array([-ts[1], ts[0]])
        c = distance * np.array([np.cos(phi_c), np.sin(phi_c)])
        p = np.column_stack(((x - .5) * lo, 0 * x))
        s = c + (x[:, None] - .5) * ls * ts
        d = p[:, None, :] - s[None, :, :]
        r = np.linalg.norm(d, axis=2)
        green = .25j * hankel2(0, wave * r)
        h = .25j * wave * hankel2(1, wave * r)
        phi = basis_values(x, degree)
        weight = phi.T * w
        mass = weight @ green @ weight.T
        dmat = derivative_matrix(degree)
        w_block = dmat.T @ mass @ dmat - wave ** 2 * lo * ls * ts[0] * mass
        k_block = weight @ (-h * d[:, :, 1] / r) @ weight.T * lo * ls
        d_block = weight @ (h * (d @ ns) / r) @ weight.T * lo * ls
        scale = np.max(weight @ np.abs(h) @ weight.T) * lo * ls
        return mass * lo * ls, k_block, d_block, w_block, scale

    def test_w_floor_is_calibrated_to_1e12(self):
        self.assertEqual([ops._graded_w_far_floor(3., [1.], 3., d) for d in (1, 2, 3)], [8, 9, 9])
        self.assertEqual(ops._graded_w_far_floor(3.01, [1.], 3., 1), 16)
        for degree in (1, 2, 3):
            order = ops._graded_w_far_floor(3., [1.], 3., degree)
            for wave in (3., 3. - .001j, 2. - 2j, .01 - 2j, 1e-4):
                for angle in (0., .37, 1.5707, 2.1, np.pi):
                    for ls in (1., .5, .01):
                        _, _, _, w, _ = self.blocks(order, wave, 1., ls, angle, 0., degree, 3.)
                        _, _, _, wr, _ = self.blocks(48, wave, 1., ls, angle, 0., degree, 3.)
                        self.assertLess(np.max(abs(w - wr)) / np.max(abs(wr)), 1e-12, (degree, wave, angle, ls))

    def test_graded_far_orders_hold_1e12_for_every_degree(self):
        for degree in (1, 2, 3):
            for kl in (.15, .5, 1.5, 3.):
                for ratio in (3., 5., 10.):
                    for arg in (0., -.7, -1.45):
                        wave = kl * np.exp(1j * arg)
                        order = ops._graded_far_order(kl, ratio, 16, degree, attenuating=arg < -np.pi / 4)
                        for angle, phi_c, ls in ((0., 0., 1.), (1.1, .5, .3), (2.9, 1.2, 1.), (np.pi / 2, 0., .5)):
                            s, kb, db, _, scale = self.blocks(order, wave, 1., ls, angle, phi_c, degree, ratio)
                            sr, kr, dr, _, _ = self.blocks(48, wave, 1., ls, angle, phi_c, degree, ratio)
                            label = (degree, kl, ratio, arg, angle)
                            self.assertLess(np.max(abs(s - sr)) / np.max(abs(sr)), 1e-12, label)
                            self.assertLess(max(np.max(abs(kb - kr)), np.max(abs(db - dr))) / scale, 1e-12, label)

    def test_orders_are_not_reduced_outside_the_calibrated_table(self):
        self.assertEqual(ops._graded_far_order(3.5, 4., 8, 1), 8)
        self.assertEqual(ops._graded_far_order(1., 2.5, 8, 1), 8)
        self.assertEqual(ops._graded_far_order(1., 4., 8, 3), 8)
        self.assertEqual(ops._graded_far_order(.1, 12., 8, 1), 5)


class NearPassTests(unittest.TestCase):
    """Items 6 and 9: vectorized near scatter/Maue; adaptive rule for hovering ends."""

    def test_batched_maue_blocks_equal_the_pair_routine(self):
        rng = np.random.default_rng(11)
        s = rng.standard_normal((50, 2, 2)) + 1j * rng.standard_normal((50, 2, 2))
        angles = rng.uniform(0, 2 * np.pi, (50, 2))
        no = np.stack((np.cos(angles[:, 0]), np.sin(angles[:, 0])), axis=1)
        ns = np.stack((np.cos(angles[:, 1]), np.sin(angles[:, 1])), axis=1)
        lo, ls = rng.uniform(.01, 2, 50), rng.uniform(.01, 2, 50)
        for k in (3.3, 2.1 - .7j):
            batch = ops._maue_blocks(s, k, no, ns, lo, ls)
            for i in range(50):
                np.testing.assert_array_equal(
                    batch[i], ops._hypersingular_block_from_s_block(s[i], k, no[i], ns[i], lo[i], ls[i]))

    def test_pair_scatter_equals_per_pair_scatter(self):
        rng = np.random.default_rng(12)
        nodes = 15
        routes = []
        # Routes of one destination write disjoint entries (here: disjoint rows).
        for offset, weighted in ((0, True), (6, False)):
            row_map = np.full(nodes, -1, np.int64)
            column_map = np.full(nodes, -1, np.int64)
            row_map[rng.choice(nodes, 5, replace=False)] = offset + rng.choice(6, 5, replace=False)
            column_map[rng.choice(nodes, 10, replace=False)] = rng.choice(10, 10, replace=False)
            routes.append((row_map, column_map, rng.standard_normal(nodes) + 1j * rng.standard_normal(nodes)
                           if weighted else None))
        rows = rng.integers(0, nodes, (40, 2))
        columns = rng.integers(0, nodes, (40, 2))
        blocks = rng.standard_normal((40, 2, 2)) + 1j * rng.standard_normal((40, 2, 2))
        for order in ('F', 'C'):
            a = np.zeros((12, 10), complex, order=order)
            b = np.zeros((12, 10), complex, order=order)
            SystemScatter(a, nodes, np.arange(nodes), np.arange(nodes), routes).scatter_pairs(rows, columns, blocks)
            reference = SystemScatter(b, nodes, np.arange(nodes), np.arange(nodes), routes)
            for p in range(40):
                reference.scatter_add(rows[p][:, None], columns[p][None, :], blocks[p])
            np.testing.assert_array_equal(a, b)

    def test_hovering_panel_ends_use_the_adaptive_rule(self):
        k = 2.0
        p0, p1 = np.array([0., 0.]), np.array([1., 0.])
        for q0, q1 in (((.5, .25), (.5, 1.25)), ((.8, .25), (.8, 1.25)), ((.8, .26), (1.5, .97)),
                       ((.9, .25), (1.9, .25))):
            q0, q1 = np.asarray(q0), np.asarray(q1)
            mesh = rcs._build_linear_mesh(panels([p0, p1], False) + panels([q0, q1], False))
            self.assertTrue(separation.requires_adaptive(*mesh.elements))
            s, kp = ops._assemble_linear_operator_matrices(mesh, k, True)
            sr, kr = composite_blocks(p0, p1, q0, q1, k)
            self.assertLess(np.max(abs(s[:2, 2:] - sr)) / np.max(abs(sr)), 1e-9)
            self.assertLess(np.max(abs(kp[:2, 2:] - kr)) / np.max(abs(kr)), 1e-9)

    def test_vector_classifier_matches_the_pair_predicate(self):
        rng = np.random.default_rng(5)
        pans = panels([(0., 0.), (1., 0.)], False)
        for _ in range(400):
            a = rng.uniform(-1.5, 2.5, 2)
            angle = rng.uniform(0, 2 * np.pi)
            b = a + rng.uniform(.3, 1.6) * np.array([np.cos(angle), np.sin(angle)])
            pans += panels([a, b], False)
        mesh = rcs._build_linear_mesh(pans)
        el = mesh.elements
        o = np.zeros(len(el) - 1, int)
        s = np.arange(1, len(el))
        p0 = np.array([e.p0 for e in el]); p1 = np.array([e.p1 for e in el])
        centers = np.array([e.center for e in el]); lengths = np.array([e.length for e in el])
        scale = np.maximum(lengths[o], lengths[s])
        distance = np.linalg.norm(centers[o] - centers[s], axis=1)
        vector = separation.adaptive_pairs(p0[o], p1[o], p0[s], p1[s], distance, scale, lengths[o], lengths[s])
        scalar = np.array([separation.requires_adaptive(el[0], el[j]) for j in s])
        np.testing.assert_array_equal(vector, scalar)
        tile = separation.close_pairs(p0[:1], p1[:1], p0[1:], p1[1:], distance[None, :], scale[None, :])
        np.testing.assert_array_equal(tile[0] | (distance < .75 * scale), scalar)


class ScatterTests(unittest.TestCase):
    """Item 7: MatrixDestination routes and the native tile scatter."""

    def test_tile_scatter_reproduces_numpy_sums(self):
        from ghost_backend.twod.assembly.native import far
        if far.tile_library() is None:
            self.skipTest('native tile scatter unavailable (older ghost_far library)')
        rng = np.random.default_rng(8)
        for trial in range(120):
            nodes = int(rng.integers(8, 40))
            width = int(rng.integers(1, 4))
            m, n = int(rng.integers(1, 15)), int(rng.integers(1, 15))
            rows = rng.integers(0, nodes, m) if trial % 2 else rng.permutation(max(nodes, m))[:m] % nodes
            columns = rng.integers(0, nodes, (n, width))
            nr, nc = int(rng.integers(nodes // 2, nodes + 2)), int(rng.integers(nodes // 2, nodes + 2))
            row_map = np.where(rng.random(nodes) < .8, rng.integers(0, nr, nodes), -1)
            column_map = np.where(rng.random(nodes) < .8, rng.integers(0, nc, nodes), -1)
            base = rng.standard_normal((2 * nr + 2, 2 * nc + 2)) + 1j * rng.standard_normal((2 * nr + 2, 2 * nc + 2))
            layout = trial % 3
            target = (np.asfortranarray(base[:nr, :nc]) if layout == 0 else np.ascontiguousarray(base[:nr, :nc])
                      if layout == 1 else np.asfortranarray(base)[1:nr + 1, ::2][:, :nc])
            expected = np.array(target)
            raw = rng.standard_normal((width, n, m)) + 1j * rng.standard_normal((width, n, m))
            values = raw.transpose(0, 2, 1)
            scale = [None, rng.standard_normal((m, n)), rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))][trial % 3]
            weights = None if trial % 4 == 0 else rng.standard_normal(nodes) + 1j * rng.standard_normal(nodes)
            routes = [(row_map, column_map, weights)]
            SystemScatter(target, nodes, [], [], routes).scatter_tile(rows, columns, values, scale)
            block = values if scale is None else values * scale[None]
            if weights is not None:
                block = block * weights[rows][None, :, None]
            for b in range(width):
                rr, cc = np.broadcast_arrays(row_map[rows][:, None], column_map[columns[:, b]][None, :])
                keep = (rr >= 0) & (cc >= 0)
                np.add.at(expected, (rr[keep], cc[keep]), block[b][keep])
            np.testing.assert_array_equal(target, expected)

    def test_matrix_destination_has_an_identity_route(self):
        n = 9
        matrix = np.zeros((n, n), complex, order='F')
        destination = MatrixDestination(matrix, n)
        self.assertEqual(len(destination.routes), 1)
        rows = np.array([0, 4, 4, 8])
        columns = np.array([[1, 2], [3, 3], [8, 0]])
        values = np.arange(24).reshape(2, 4, 3) * (1 + .5j)
        destination.scatter_add_columns(rows, columns, values)
        expected = np.zeros((n, n), complex)
        for b in range(2):
            np.add.at(expected, (rows[:, None], columns[None, :, b]), values[b])
        np.testing.assert_array_equal(matrix, expected)


class SelfTermTests(unittest.TestCase):
    """Item 10: long self elements no longer fall back to the plain Duffy rule."""

    def test_self_block_beyond_the_series_range(self):
        element = mesh_of([(0., 0.), (1., 0.), (5., 0.), (6., 0.)], closed=False).elements[0]
        for magnitude in (8.1, 20., 60.):
            for arg in (0., -.7):
                k = magnitude * np.exp(1j * arg)

                def integrand(u):
                    g = .25j * hankel2(0, k * max(u, 1e-300))
                    return np.array([g * (2 - 3 * u + u ** 3) / 3, g * (1 - u ** 3) / 3])
                points = sorted({min(1., j / magnitude) for j in (.5, 1, 2, 4, 8, 16, 32, 64)})
                ref = quad_vec(integrand, 0, 1, epsabs=0, epsrel=1e-14, limit=4000, points=points)[0]
                expected = np.array([[ref[0], ref[1]], [ref[1], ref[0]]])
                self.assertIsNone(ops._single_layer_self_block_exact(element, k))
                actual = ops._single_layer_block_linear(element, element, k)
                self.assertLess(np.max(abs(actual - expected)) / np.max(abs(expected)), 1e-12, (magnitude, arg))


class NearTableTests(unittest.TestCase):
    """Item 11: lossy near tables belong to the solve, within its table budget."""

    def test_near_tables_are_scoped(self):
        self.assertFalse(hasattr(pq._kernel_table, 'cache_info'))
        mesh = mesh_of(circle(.05, 24))
        pairs = [(mesh.elements[i], mesh.elements[(i + 1) % 24]) for i in range(6)]
        state = CPUState()
        with _STATE.override(state):
            for j in range(3):
                ops._integrate_linear_pairs_box_sk_batched(
                    mesh.elements, np.arange(3), np.arange(3) + 5, 60 * (1 - .1j * (j + 1)), True, 8)
        near = [key for key in state.tables if key[0] == 'near']
        self.assertEqual(len(near), 3)
        self.assertEqual(state.table_bytes, sum(t.evidence['bytes'] for t in state.tables.values() if t is not None))
        with pq.moment_cache_scope() as cache:
            pq.near_blocks(pairs, 60 - 9j)
            self.assertEqual(len(cache.tables), 1)
        pq.near_blocks(pairs, 60 - 7j)
        pq.near_blocks(pairs, 60 - 8j)
        self.assertEqual(pq._LAST_TABLE[0][2], -8.0)


class DeterminismTests(unittest.TestCase):
    """Item 12: far tiles are scattered in tile order, so threads change nothing."""

    def test_threaded_assembly_is_bitwise_serial(self):
        mesh = mesh_of(circle(1.0, 300))
        results = []
        for threads in (1, 4):
            with execution_scope(validate_options(dict(assembly_threads=threads, assembly_tile=24))):
                with _STATE.override(CPUState()):
                    results.append(fused(mesh, 40.0 - 1j))
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)


class LoadReuseTests(unittest.TestCase):
    """Item 13: monostatic polynomial far fields reuse the batch's RHS loads."""

    def test_reused_loads_give_the_same_fields_without_recomputing_moments(self):
        from ghost_backend.tests.general_fixtures import fixture
        calls = []
        original = kernels.plane_wave_moments

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        amplitudes = []
        for reuse in (True, False):
            calls.clear()
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(kernels, 'plane_wave_moments', counted))
                if not reuse:
                    stack.enter_context(mock.patch.object(kernels, '_load_cache', lambda: None))
                result = rcs.solve_monostatic_rcs_2d(fixture('dielectric', 48), [2.], [0., 33., 90., 147.],
                    geometry_units='meters', execution_options=dict(basis_order=2))
            amplitudes.append(np.array([complex(row['rcs_amp_real'], row['rcs_amp_imag'])
                                        for rows in result['co_solved_samples'].values() for row in rows]))
            if reuse:
                reused_calls = len(calls)
            else:
                fresh_calls = len(calls)
        self.assertLess(reused_calls, fresh_calls)
        np.testing.assert_allclose(amplitudes[0], amplitudes[1], rtol=1e-13, atol=1e-15 * np.max(abs(amplitudes[1])))


if __name__ == '__main__':
    unittest.main()
