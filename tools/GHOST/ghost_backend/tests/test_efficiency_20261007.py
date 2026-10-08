"""Numerical, memory and retry contracts for the October efficiency changes."""
from unittest import mock

import numpy as np
import pytest

from ghost_backend.linalg import hierarchical as h
from ghost_backend.linalg import residual_spool as spool
from ghost_backend.linalg.workspace import checked_matrix_norms, matrix_inf_norm, first_nonfinite


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_fused_norms_preserve_both_reduction_orders(layout):
    rng = np.random.default_rng(711)
    a = rng.normal(size=(149, 151)) + 1j*rng.normal(size=(149, 151))
    a = np.asfortranarray(a) if layout == "F" else a[:, ::2] if layout == "strided" else a
    sums = np.zeros(a.shape[1])
    for start in range(0, len(a), 64):
        sums += np.sum(abs(a[start:start+64]), axis=0)
    assert checked_matrix_norms(a, True) == (None, matrix_inf_norm(a), float(max(sums)))
    assert checked_matrix_norms(a, False) == (None, matrix_inf_norm(a), None)
    a[100, 2], a[3, 20] = np.inf, np.nan
    assert checked_matrix_norms(a, True) == (first_nonfinite(a), None, None)


def test_rank_capacity_does_not_repeat_build_but_numerical_failure_does():
    for rejection, count in [(h.HierarchicalCapacityRejected("rank limit"), 1),
                              (h.HierarchicalRejected("numerical check"), 2)]:
        with mock.patch.object(h, "ordered_copy", return_value=np.eye(8)), \
                mock.patch.object(h.HierarchicalFactor, "_rebuild", side_effect=rejection) as build:
            factor = h.HierarchicalFactor.__new__(h.HierarchicalFactor)
            with pytest.raises(h.HierarchicalRejected):
                factor.__init__(np.eye(8, dtype=complex))
            assert build.call_count == count
            assert factor.ordered is None
            assert spool._PENDING_WORKSPACE_BYTES == 0


def test_larger_fallback_leaf_keeps_exact_transposed_solutions_and_budget():
    rng = np.random.default_rng(727)
    a = rng.normal(size=(534, 534)).astype(complex)
    a += 1j*rng.normal(size=a.shape)
    a += 100*np.eye(len(a))
    rhs = rng.normal(size=(len(a), 3)) + 1j*rng.normal(size=(len(a), 3))
    with mock.patch.object(h, "factor_storage_budget", return_value=2*a.nbytes), \
            mock.patch.object(h, "ordered_copy", return_value=None):
        factor = h.HierarchicalFactor(a)
    assert factor.root.leaf
    assert factor.bytes <= factor.budget
    for trans, matrix in enumerate((a, a.T, a.conj().T)):
        x = factor.solve(rhs, trans=trans)
        np.testing.assert_allclose(matrix @ x, rhs, atol=8e-14, rtol=8e-14)
    with mock.patch.object(h, "factor_storage_budget", return_value=1), \
            mock.patch.object(h, "ordered_copy", return_value=None):
        with pytest.raises(h.HierarchicalCapacityRejected, match="storage"):
            h.HierarchicalFactor(a)


def test_ordered_copy_obeys_allocation_and_bounds_serial_gathers():
    class Tracked(np.ndarray):
        sizes = []
        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(key, tuple) and isinstance(key[0], np.ndarray):
                self.sizes.append(result.nbytes)
            return result
    a = np.arange(64**2, dtype=complex).reshape(64, 64).view(Tracked)
    perm = np.arange(63, -1, -1)
    with mock.patch.object(spool, "copy_fits", return_value=False):
        assert h.ordered_copy(a, perm, lambda: None) is None
    with mock.patch.object(spool, "copy_fits", return_value=True), \
            mock.patch.object(h, "PRODUCT_PANEL_BYTES", 2048):
        result = h.ordered_copy(a, perm, lambda: None)
    assert max(a.sizes) <= 2048
    np.testing.assert_array_equal(result, np.asarray(a)[np.ix_(perm, perm)])


def test_pending_factor_storage_is_in_copy_admission_and_released():
    mib = 1024**2
    with mock.patch("ghost_backend.twod.solver._process_rss_bytes", return_value=100*mib), \
            mock.patch("ghost_backend.twod.solver._solve_memory_limit_gb", return_value=.5), \
            mock.patch("ghost_backend.compressed.worker_pool.retained_bytes", return_value=0):
        assert spool.copy_fits(100*mib)
        with spool.reserve_factor_workspace(80*mib):
            assert not spool.copy_fits(100*mib)
            with spool.reserve_factor_workspace(20*mib):
                assert spool._PENDING_WORKSPACE_BYTES == 100*mib
        assert spool.copy_fits(100*mib)


@pytest.mark.parametrize("mode", [-5, -1, 0, 1, 5])
@pytest.mark.parametrize("scales", [(1., 0.), (0., 1.), (.37+.2j, .63-.1j), (0., 0.)])
def test_bor_shared_excitation_matches_independent_scalar_fields(mode, scales):
    from ghost_backend.bor.solver import BorPecSolver, ETA0
    theta = np.linspace(0., np.pi, 21)
    gen = np.column_stack((.05*np.sin(theta), .06*np.cos(theta)+.01*np.sin(theta)**2))
    solver = BorPecSolver(gen, 1e9)
    angles = np.array([0., 13., 77., 125., 180.])
    ef, mf = scales[0], scales[1]*ETA0
    expected = np.stack([ef*solver.rhs_mode(mode, t, pol) + mf*solver.rhs_mfie_mode(mode, t, pol)
                         for t in angles for pol in ("VV", "HH")], axis=1)
    actual = solver.rhs_vv_hh_batch(mode, angles, ef, mf, angle_chunk=2)
    np.testing.assert_allclose(actual, expected, rtol=8e-14,
                               atol=8e-14*max(1e-20, np.max(abs(expected))))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_matched_load_reuse_keeps_weighted_masked_derivative_and_owned_rhs(degree):
    from ghost_backend.twod.assembly import kernels
    from test_twod_remaining_performance import mesh_for
    mesh, _, k = mesh_for("pec", degree)
    mask = np.arange(len(mesh.elements)) % 2 == 0
    angles = np.array([17., 64., 136.])
    rng = np.random.default_rng(343)
    rho = rng.normal(size=(len(mesh.nodes), 3)) + 1j*rng.normal(size=(len(mesh.nodes), 3))
    coeff = np.linspace(1., 3., len(mesh.elements)) * (1+.3j)
    expected = [kernels.farfield(mesh, rho, k, angles, p, element_mask=mask, projection="matched")
                for p in ("SLP", "DLP")]
    cache = {}
    with mock.patch.object(kernels, "_load_cache", return_value=cache):
        bu, bd = kernels.incident_loads(mesh, k, angles, element_mask=mask,
                                        observation_coefficients=coeff)
        bu[:] = 0
        bd[:] = 0
        with mock.patch.object(kernels, "moments", side_effect=AssertionError("recomputed")), \
                mock.patch.object(kernels, "plane_wave_moments", side_effect=AssertionError("recomputed")):
            actual = [kernels.farfield(mesh, rho, k, angles, p, element_mask=mask, projection="matched")
                      for p in ("SLP", "DLP")]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14*max(np.max(np.abs(expected)), 1e-20))


def test_far_cache_local_references_do_not_retain_globally_evicted_tables():
    import gc
    import threading
    import weakref
    from collections import OrderedDict
    from ghost_backend.bor import kernels
    with mock.patch.object(kernels, "_FAR_TABLE_LOCAL", threading.local()), \
            mock.patch.object(kernels, "_FAR_TABLES", OrderedDict()), \
            mock.patch.object(kernels, "_FAR_TABLE_BYTES", [0]), \
            mock.patch.object(kernels, "FAR_TABLE_CACHE_BYTES", 16384):
        entry = kernels._half_grid_tables(64, np.array([0, 1]), True)
        reference = weakref.ref(entry[0])
        del entry
        for i in range(1, 140):
            kernels._half_grid_tables(64, np.array([i, i+1]), True)
        gc.collect()
        assert reference() is None
        assert kernels._FAR_TABLE_BYTES[0] <= 16384
        assert len(kernels._FAR_TABLE_LOCAL.tables) <= kernels._FAR_TABLE_LOCAL_ENTRIES
        expected = kernels._half_grid_tables(64, np.array([0, 1]), True)
        actual = kernels._half_grid_tables(64, np.array([0, 1]), True)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_prepared_near_geometry_preserves_real_and_lossy_pair_blocks():
    from ghost_backend.twod import operators
    from test_twod_remaining_performance import mesh_for
    mesh, _, k = mesh_for("pec", 1)
    elements = mesh.elements
    p0 = np.asarray([e.p0 for e in elements])
    p1 = np.asarray([e.p1 for e in elements])
    geometry = (p0, p1, p1-p0, np.asarray([e.length for e in elements]),
                np.asarray([e.normal for e in elements]))
    obs, src = np.arange(8), np.arange(8)+3
    for wave in (k, k*(1-.2j)):
        for normal in (False, True):
            args = (elements, obs, src, wave, normal, 16)
            expected = operators._integrate_linear_pairs_box_sk_batched(*args)
            actual = operators._integrate_linear_pairs_box_sk_batched(*args, prepared_geometry=geometry)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout,orders", [(-1, (32,)), (0, (24, 32)), (3, (24, 24, 24, 24, 32))])
@pytest.mark.parametrize("kind,stable", [("g", True), ("mfie", False), ("mfie", True), ("ibc", True)])
def test_native_shared_rule_nodes_match_explicit_sampling(layout, orders, kind, stable):
    from ghost_backend.bor import kernels
    symbol = "near_green_rule" if kind == "g" else "near_brackets_rule"
    if kernels._native_entry(symbol) is None:
        pytest.skip("optional native library unavailable")
    rho = np.linspace(.08, .13, 17)
    gap = np.linspace(.001, .01, 17)
    tr = np.linspace(.1, .4, 17)
    tz = np.sqrt(1-tr*tr)
    points = (rho, np.zeros(17), tr, tz, rho+gap, gap*.5, tr, tz)
    if kind == "g":
        points = (points[0], points[1], points[4], points[5])
    delta = gap*np.sqrt(1.25) / (2*np.sqrt(rho*(rho+gap)))
    for k in (30., 30.-2j):
        expected = kernels._near_rule_moments(kind, stable, points, delta, k, layout, orders, 37)
        original = kernels._native_entry
        # The independent path samples nodes then projects their moments.
        with mock.patch.object(kernels, "_native_entry",
                               side_effect=lambda name: None if name == symbol else original(name)):
            actual = kernels._near_rule_moments(kind, stable, points, delta, k, layout, orders, 37)
        np.testing.assert_allclose(actual, expected, rtol=5e-12,
                                   atol=5e-13*max(np.max(np.abs(expected)), 1e-30))
