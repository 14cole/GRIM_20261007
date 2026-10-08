"""2-D quadrature, boundary operators, and linear-mesh field evaluation."""
from ghost_backend.execution.options import option, current_options, execution_scope, effective_assembly_threads

import cmath
import math
import os
import threading
import numpy as np
from ghost_backend.twod.assembly.compact import CompactOperator, scatter_basis_columns, scatter_operator_add
from ghost_backend.linalg.workspace import first_nonfinite
from ghost_backend.twod.assembly.separation import requires_adaptive, close_pairs, segment_distance


from ghost_backend.twod.geometry import (
    ComplexTable,
    ImpedanceTaper,
    LinearElement,
    LinearMesh,
    LinearNode,
    MaterialLibrary,
    MediumTable,
    Panel,
    PanelCoupledInfo,
)
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.twod.assembly.profiling import assembly_component
from ghost_backend.execution.cpu import current_state, cached_operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from ghost_backend.twod.constants import EPS, EULER_GAMMA
from ghost_backend.twod.special import _SCIPY_SPECIAL, _hankel2_0, _hankel2_1
from ghost_backend.twod.geometry import _linear_shape_values, _surface_robin_alpha
from ghost_backend.twod.basis import values as _polynomial_values, mesh_degree, derivative_matrix


def _linear_param_to_point(elem: 'LinearElement', xi: 'float') -> 'np.ndarray':
    return elem.p0 + float(xi) * (elem.p1 - elem.p0)

def _linear_interval_point(elem: 'LinearElement', interval: 'Tuple[float, float]', use_start: 'bool') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, a if use_start else b)

def _linear_interval_length(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    return max(abs(b - a) * float(elem.length), 0.0)

def _linear_interval_midpoint(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, 0.5 * (a + b))

def _linear_map_local_to_parent(interval: 'Tuple[float, float]', local_xi: 'float', start_is_shared: 'bool') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    h = b - a
    x = float(local_xi)
    return (a + h * x) if start_is_shared else (b - h * x)

# The linear mesh merges endpoints on a 1e-9 m snap grid
# (geometry._linear_node_snap_key). Every endpoint-coincidence test uses this
# one tolerance, and endpoints that are one mesh node count as touching even
# where their stored coordinates differ by up to sqrt(2) snap widths.
NODE_SNAP_TOLERANCE = 1.0e-9


def _endpoint_node(elem: 'LinearElement', parameter: 'float') -> 'Optional[int]':
    if parameter == 0.0:
        return int(elem.node_ids[0])
    if parameter == 1.0:
        return int(elem.node_ids[1])
    return None


def _linear_shared_interval_endpoint_info(
    obs_elem: 'LinearElement',
    obs_interval: 'Tuple[float, float]',
    src_elem: 'LinearElement',
    src_interval: 'Tuple[float, float]',
    tol: 'float' = NODE_SNAP_TOLERANCE,
) -> 'Optional[Tuple[bool, bool]]':
    """(obs start shared, src start shared) for the first touching endpoint pair, or None.

    Interval ends touch within ``tol`` or when they are the same mesh node and
    lie within two snap widths of each other.
    """
    obs_ends = [float(obs_interval[0]), float(obs_interval[1])]
    src_ends = [float(src_interval[0]), float(src_interval[1])]
    obs_pts = [_linear_param_to_point(obs_elem, t) for t in obs_ends]
    src_pts = [_linear_param_to_point(src_elem, t) for t in src_ends]
    for obs_is_start, (op, ot) in enumerate(zip(obs_pts, obs_ends)):
        for src_is_start, (sp, st) in enumerate(zip(src_pts, src_ends)):
            distance = float(np.linalg.norm(op - sp))
            if distance <= float(tol):
                return bool(obs_is_start == 0), bool(src_is_start == 0)
            node = _endpoint_node(obs_elem, ot)
            if (node is not None and node == _endpoint_node(src_elem, st)
                    and distance <= 2.0 * NODE_SNAP_TOLERANCE):
                return bool(obs_is_start == 0), bool(src_is_start == 0)
    return None

def _integrate_linear_pair_box(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
) -> 'np.ndarray':
    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    obs_scale = max(float(obs_interval[1]) - float(obs_interval[0]), 0.0)
    src_scale = max(float(src_interval[1]) - float(src_interval[0]), 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for tobs, wobs in zip(qt_obs, qw_obs):
        xi_obs = float(obs_interval[0]) + obs_scale * float(tobs)
        phi_obs = _linear_shape_values(xi_obs)
        robs = _linear_param_to_point(obs_elem, xi_obs)
        for tsrc, wsrc in zip(qt_src, qw_src):
            xi_src = float(src_interval[0]) + src_scale * float(tsrc)
            phi_src = _linear_shape_values(xi_src)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            kval = complex(kernel_eval(robs, rsrc))
            block += (float(wobs) * float(wsrc) * kval) * np.outer(phi_obs, phi_src)

    return block * obs_len * src_len

def _integrate_linear_pair_box_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Vectorized tensor-Gauss 2x2 S and K block assembly for one element pair.

    Evaluates all quadrature point pairs at once using array Hankel functions,
    avoiding per-point Python-loop overhead.  Returns (S_block, K_block).
    """

    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    oa, ob = float(obs_interval[0]), float(obs_interval[1])
    sa, sb = float(src_interval[0]), float(src_interval[1])
    obs_scale = max(ob - oa, 0.0)
    src_scale = max(sb - sa, 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    nobs = len(qt_obs)
    nsrc = len(qt_src)


    xi_obs_all = oa + obs_scale * np.asarray(qt_obs, dtype=float)
    xi_src_all = sa + src_scale * np.asarray(qt_src, dtype=float)
    phi_obs_all = np.column_stack([1.0 - xi_obs_all, xi_obs_all])
    phi_src_all = np.column_stack([1.0 - xi_src_all, xi_src_all])

    obs_seg = obs_elem.p1 - obs_elem.p0
    src_seg = src_elem.p1 - src_elem.p0
    robs_all = obs_elem.p0[None, :] + xi_obs_all[:, None] * obs_seg[None, :]
    rsrc_all = src_elem.p0[None, :] + xi_src_all[:, None] * src_seg[None, :]


    diff = robs_all[:, None, :] - rsrc_all[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist_safe = np.maximum(dist, EPS)

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one element-pair operator must be requested.")

    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    kr[np.abs(kr) <= 1e-12] = 1e-12 + 0.0j
    if compute_single_layer:

        h0 = _hankel2_0_array(kr.ravel()).reshape(nobs, nsrc)
        g_vals = 0.25j * h0

    if compute_double_layer:
        h1 = _hankel2_1_array(kr.ravel()).reshape(nobs, nsrc)
        if obs_normal_deriv:

            proj = np.sum(diff * obs_elem.normal[None, None, :], axis=2) / dist_safe
            dk_vals = (-0.25j * complex(k0)) * h1 * proj
        else:

            proj = np.sum(src_elem.normal[None, None, :] * diff, axis=2) / dist_safe
            dk_vals = (0.25j * complex(k0)) * h1 * proj
        dk_vals[dist <= EPS] = 0.0


    w_outer = np.outer(np.asarray(qw_obs, dtype=float), np.asarray(qw_src, dtype=float))


    if compute_single_layer:
        weighted_g = w_outer * g_vals

        s_block = np.einsum(
            'ij,ia,jb->ab', weighted_g, phi_obs_all, phi_src_all
        )
    if compute_double_layer:
        weighted_k = w_outer * dk_vals
        k_block = np.einsum(
            'ij,ia,jb->ab', weighted_k, phi_obs_all, phi_src_all
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale


def _integrate_linear_pairs_box_sk_batched(
    elements: 'Sequence[LinearElement]',
    obs_indices: 'np.ndarray',
    src_indices: 'np.ndarray',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    prepared_geometry=None,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Tensor-Gauss S/K blocks for many full-interval element pairs.

    This is the same calculation as `_integrate_linear_pair_box_sk_vectorized`
    with an additional leading pair axis. It is used only for separated near
    pairs whose adaptive classifier selected a fixed tensor rule; singular,
    touching, and recursively adaptive pairs retain their dedicated paths.
    """

    obs_ids = np.asarray(obs_indices, dtype=np.int64).reshape(-1)
    src_ids = np.asarray(src_indices, dtype=np.int64).reshape(-1)
    if obs_ids.size != src_ids.size:
        raise ValueError("Batched near-pair index arrays must have equal length.")
    npairs = int(obs_ids.size)
    width = len(elements[0].node_ids) if elements else 2
    zero = np.zeros((npairs, width, width), dtype=np.complex128)
    if npairs == 0:
        return zero.copy(), zero.copy()
    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one batched near-pair operator is required.")

    if width > 2:
        # Polynomial pairs keep their kernel samples as monomial moments, so
        # the cubic accuracy candidate of a certified solve reuses the
        # quadratic candidate's samples (the same panels and wavenumber).
        return _box_blocks_polynomial(elements, obs_ids, src_ids, k0, obs_normal_deriv, order,
                                      bool(compute_single_layer), bool(compute_double_layer))
    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    weights = np.asarray(qw, dtype=float)
    phi = _polynomial_values(q, width - 1)
    if prepared_geometry is None:
        obs_elems = [elements[int(index)] for index in obs_ids]
        src_elems = [elements[int(index)] for index in src_ids]
        obs_p0 = np.asarray([e.p0 for e in obs_elems], float)
        src_p0 = np.asarray([e.p0 for e in src_elems], float)
        obs_p1 = np.asarray([e.p1 for e in obs_elems], float)
        src_p1 = np.asarray([e.p1 for e in src_elems], float)
        obs_seg, src_seg = obs_p1-obs_p0, src_p1-src_p0
        normals = np.asarray([e.normal for e in (obs_elems if obs_normal_deriv else src_elems)], float)
        pair_scales = np.asarray([o.length*s.length for o, s in zip(obs_elems, src_elems)])
    else:
        p0, p1, segments, lengths, normal_array = prepared_geometry
        obs_p0, src_p0 = p0[obs_ids], p0[src_ids]
        obs_p1, src_p1 = p1[obs_ids], p1[src_ids]
        obs_seg, src_seg = segments[obs_ids], segments[src_ids]
        normals = normal_array[obs_ids if obs_normal_deriv else src_ids]
        pair_scales = lengths[obs_ids] * lengths[src_ids]
    obs_pts = obs_p0[:, None, :] + q[None, :, None] * obs_seg[:, None, :]
    src_pts = src_p0[:, None, :] + q[None, :, None] * src_seg[:, None, :]
    diff = obs_pts[:, :, None, :] - src_pts[:, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3))
    dist_safe = np.maximum(dist, EPS)
    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    tiny = np.abs(kr) <= 1e-12
    w_outer = np.outer(weights, weights)
    green = derivative = None
    if not np.any(tiny):
        # Validated screened tables (lossy media) or real Bessel functions; the
        # scaled complex Hankel routine remains the reference fallback.
        from ghost_backend.twod.polynomial_quadrature import _kernels, _table_for_ends
        table = _table_for_ends(k0, obs_p0, obs_p1, src_p0, src_p1)
        green, derivative = _kernels(k0, dist_safe, bool(compute_double_layer), table)
    else:
        kr[tiny] = 1e-12 + 0.0j

    if compute_single_layer:
        if green is None:
            green = 0.25j * _hankel2_0_array(kr.reshape(-1)).reshape(dist.shape)
        weighted_g = w_outer[None, :, :] * green
        # Contract the two quadrature axes separately instead of visiting every
        # (i, j, a, b) combination in the generic einsum loop. Each pair remains
        # an independent matrix product, including in threaded batches.
        s_blocks = phi.T @ weighted_g @ phi
    else:
        s_blocks = zero.copy()

    if compute_double_layer:
        if derivative is None:
            derivative = (0.25j * complex(k0)) * _hankel2_1_array(kr.reshape(-1)).reshape(dist.shape)
        if obs_normal_deriv:
            proj = np.sum(
                diff * normals[:, None, None, :], axis=3
            ) / dist_safe
            dk_vals = -derivative * proj
        else:
            proj = np.sum(
                diff * normals[:, None, None, :], axis=3
            ) / dist_safe
            dk_vals = derivative * proj
        dk_vals[dist <= EPS] = 0.0
        k_blocks = phi.T @ (w_outer[None, :, :] * dk_vals) @ phi
    else:
        k_blocks = zero.copy()

    scales = pair_scales[:, None, None]
    return s_blocks * scales, k_blocks * scales

def _box_monomial_moments(obs_elems, src_elems, k0, obs_normal_deriv, order, want_s, want_k):
    """Scaled monomial S and K moments, shape (T, 2, D+1, D+1), of full-interval
    pairs by the tensor-Gauss rule of ``order`` points per axis (the same
    kernels as the nodal box rule; D = polynomial_quadrature._MOMENT_DEGREE)."""
    from ghost_backend.twod.polynomial_quadrature import _kernels, _table_for_ends, _MOMENT_DEGREE
    count = len(obs_elems)
    moments = np.zeros((count, 2, _MOMENT_DEGREE + 1, _MOMENT_DEGREE + 1), dtype=np.complex128)
    if not count:
        return moments
    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    weights = np.asarray(qw, dtype=float)
    monomials = np.vander(q, _MOMENT_DEGREE + 1, increasing=True)
    obs_p0 = np.asarray([elem.p0 for elem in obs_elems], dtype=float)
    src_p0 = np.asarray([elem.p0 for elem in src_elems], dtype=float)
    obs_seg = np.asarray([elem.p1 - elem.p0 for elem in obs_elems], dtype=float)
    src_seg = np.asarray([elem.p1 - elem.p0 for elem in src_elems], dtype=float)
    obs_pts = obs_p0[:, None, :] + q[None, :, None] * obs_seg[:, None, :]
    src_pts = src_p0[:, None, :] + q[None, :, None] * src_seg[:, None, :]
    diff = obs_pts[:, :, None, :] - src_pts[:, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3))
    dist_safe = np.maximum(dist, EPS)
    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    tiny = np.abs(kr) <= 1e-12
    w_outer = np.outer(weights, weights)
    green = derivative = None
    if not np.any(tiny):
        table = _table_for_ends(k0, obs_p0, obs_p0 + obs_seg, src_p0, src_p0 + src_seg)
        green, derivative = _kernels(k0, dist_safe, bool(want_k), table)
    else:
        kr[tiny] = 1e-12 + 0.0j
    if want_s:
        if green is None:
            green = 0.25j * _hankel2_0_array(kr.reshape(-1)).reshape(dist.shape)
        moments[:, 0] = monomials.T @ (w_outer[None, :, :] * green) @ monomials
    if want_k:
        if derivative is None:
            derivative = (0.25j * complex(k0)) * _hankel2_1_array(kr.reshape(-1)).reshape(dist.shape)
        normals = np.asarray([elem.normal for elem in (obs_elems if obs_normal_deriv else src_elems)], dtype=float)
        proj = np.sum(diff * normals[:, None, None, :], axis=3) / dist_safe
        dk_vals = (-derivative if obs_normal_deriv else derivative) * proj
        dk_vals[dist <= EPS] = 0.0
        moments[:, 1] = monomials.T @ (w_outer[None, :, :] * dk_vals) @ monomials
    scales = np.asarray([obs.length * src.length for obs, src in zip(obs_elems, src_elems)], dtype=float)
    moments *= scales[:, None, None, None]
    return moments


def _box_blocks_polynomial(elements, obs_ids, src_ids, k0, obs_normal_deriv, order, want_s, want_k):
    """Nodal S and K blocks of polynomial fixed-order pairs through their monomial moments.

    The moments depend on the pair geometry, the wavenumber and the rule, not
    on the polynomial degree; within a certified request's moment-cache scope
    they are stored, so the cubic candidate projects the quadratic candidate's
    samples instead of integrating them again (the nodal projection is the
    same linear map as the direct nodal rule, to rounding).
    """
    from ghost_backend.twod.polynomial_quadrature import _MOMENT_CACHE, box_moment_keys
    from ghost_backend.twod.basis import coefficients
    obs_elems = [elements[int(index)] for index in obs_ids]
    src_elems = [elements[int(index)] for index in src_ids]
    npairs = len(obs_elems)
    width = len(elements[0].node_ids)
    degree = width - 1
    cache = _MOMENT_CACHE.get()
    if cache is None:
        moments = _box_monomial_moments(obs_elems, src_elems, k0, obs_normal_deriv, order, want_s, want_k)
    else:
        keys = box_moment_keys(k0, obs_normal_deriv, order, want_s, want_k, obs_elems, src_elems)
        found = [cache.get(key) for key in keys]
        todo = [index for index, value in enumerate(found) if value is None]
        if todo:
            fresh = _box_monomial_moments([obs_elems[i] for i in todo], [src_elems[i] for i in todo],
                                          k0, obs_normal_deriv, order, want_s, want_k)
            for position, value in zip(todo, fresh):
                found[position] = value
                cache.put(keys[position], value)
        moments = (np.stack(found) if npairs else
                   np.zeros((0, 2, degree + 1, degree + 1), dtype=np.complex128))
    co = coefficients(degree)
    block = moments[:, :, :degree + 1, :degree + 1]
    zero = np.zeros((npairs, width, width), dtype=np.complex128)
    s_blocks = np.einsum('ia,tij,jb->tab', co, block[:, 0], co) if want_s else zero.copy()
    k_blocks = np.einsum('ia,tij,jb->tab', co, block[:, 1], co) if want_k else zero.copy()
    return s_blocks, k_blocks


def _single_layer_self_block_exact(
    elem: 'LinearElement',
    k0: 'Union[complex, float]',
    interval: 'Tuple[float, float]' = (0.0, 1.0),
) -> 'Optional[np.ndarray]':
    """
    Closed-form linear-Galerkin single-layer self block for a straight element.

    On a straight element the kernel depends only on u = |t - s|, so

        B_ij = l^2 * (j/4) * Int_0^1 H0^(2)(k*l*u) * C_ij(u) du

    with shape-pair weights (phi0 = 1-t, phi1 = t):

        C_diag(u) = (2 - 3u + u^3)/3      C_off(u) = (1 - u^3)/3

    (their sum reproduces the constant-basis weight 2(1-u) used by the
    exact `_single_layer_self_term`).  Substituting the small-argument
    series H0^(2)(x) = J0(x)[1 - j(2/pi)(ln(x/2)+gamma)] - j*R(x) turns the
    u-integral into exact moments:

        Int u^p du = 1/(p+1)        Int u^p ln(u) du = -1/(p+1)^2

    so the whole block is a rapidly convergent series -- machine precision,
    unlike the (u, uv) "Duffy" map, whose unresolved log singularity along
    the diagonal capped the self block at ~0.1-1% error.

    Returns None when |k*l| is too large for the series to be well
    conditioned (caller falls back to numeric quadrature).
    """

    a, b = float(interval[0]), float(interval[1])
    h = b - a
    ell = float(elem.length) * h
    if ell <= 0.0:
        return np.zeros((2, 2), dtype=np.complex128)
    z = complex(k0) * ell
    if abs(z) > 8.0:
        return None
    if abs(z) <= 1e-30:
        return None

    c_diag = (2.0 / 3.0, -1.0, 0.0, 1.0 / 3.0)
    c_off = (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0)

    def moment(p: 'int', coeffs) -> 'float':
        return sum(c / (p + q + 1) for q, c in enumerate(coeffs))

    def log_moment(p: 'int', coeffs) -> 'float':
        return -sum(c / (p + q + 1) ** 2 for q, c in enumerate(coeffs))

    two_over_pi = 2.0 / math.pi
    log_term = cmath.log(z / 2.0) + EULER_GAMMA
    z_quarter_sq = (z / 2.0) ** 2

    b_diag = 0.0 + 0.0j
    b_off = 0.0 + 0.0j
    alpha = 1.0 + 0.0j
    harmonic = 0.0
    m = 0
    while True:


        a_m = alpha * (1.0 - 1j * two_over_pi * (log_term - harmonic))
        p = 2 * m
        b_diag += a_m * moment(p, c_diag) - 1j * two_over_pi * alpha * log_moment(p, c_diag)
        b_off += a_m * moment(p, c_off) - 1j * two_over_pi * alpha * log_moment(p, c_off)
        m += 1
        alpha *= -z_quarter_sq / (m * m)
        harmonic += 1.0 / m
        if m > 60:
            return None
        if abs(alpha) < 1e-18 * max(1.0, abs(b_diag)):
            break

    block_local = (0.25j * ell * ell) * np.array(
        [[b_diag, b_off], [b_off, b_diag]], dtype=np.complex128,
    )
    if a == 0.0 and b == 1.0:
        return block_local


    t_mat = np.array([[1.0 - a, 1.0 - b], [a, b]], dtype=np.complex128)
    return t_mat @ block_local @ t_mat.T


def _integrate_linear_self_duffy(
    elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    interval: 'Tuple[float, float]',
    order: 'int' = 20,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(4, int(order)))
    a, b = float(interval[0]), float(interval[1])
    h = max(b - a, 0.0)
    elem_len = float(elem.length) * h
    block = np.zeros((2, 2), dtype=np.complex128)
    if elem_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        t_major = a + h * uu
        s_major = t_major
        robs_major = _linear_param_to_point(elem, t_major)
        rsrc_major = _linear_param_to_point(elem, s_major)
        phi_t_major = _linear_shape_values(t_major)
        phi_s_major = _linear_shape_values(s_major)
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)

            xi_t = a + h * uu
            xi_s = a + h * (uu * vv)
            phi_t = _linear_shape_values(xi_t)
            phi_s = _linear_shape_values(xi_s)
            robs = _linear_param_to_point(elem, xi_t)
            rsrc = _linear_param_to_point(elem, xi_s)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_t, phi_s)

            xi_t2 = a + h * (uu * vv)
            xi_s2 = a + h * uu
            phi_t2 = _linear_shape_values(xi_t2)
            phi_s2 = _linear_shape_values(xi_s2)
            robs2 = _linear_param_to_point(elem, xi_t2)
            rsrc2 = _linear_param_to_point(elem, xi_s2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_t2, phi_s2)

    return block * (elem_len * elem_len)

# Endpoint-touching pairs use a radially graded two-triangle rule: the corner
# distance u = t**4 (weight 4 t**7) instead of the plain Duffy u. The plain map
# leaves u*log(u) terms of the linear basis in the integrand, so its error only
# fell as n**-4 (9e-5..6e-4 per S block at the former 9 points, an O(h) term in
# the operator spectrum). The graded map is the one polynomial_quadrature uses:
# at 20 points collinear S blocks are converged to 3e-13, a right-angle corner to
# 1e-14 and a 170-degree reflex corner to 5e-13 (S) / 1.4e-10 (K').
_TOUCHING_RADIAL_ORDER = 20


def _touching_order(order: 'int') -> 'int':
    """Gauss points per direction of the touching rule; ``order`` is a lower bound."""
    return max(int(order), _TOUCHING_RADIAL_ORDER)


def _touching_rule(order: 'int') -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Local distances (x, y) from the shared corner and weights on the unit square."""
    from ghost_backend.twod.polynomial_quadrature import _nodes
    return _nodes(_touching_order(order), True)


def _touching_frame(elem, interval, start_is_shared):
    """Shared corner point, direction away from it, and parent coordinate map."""
    a, b = float(interval[0]), float(interval[1])
    seg = elem.p1 - elem.p0
    h = b - a
    if start_is_shared:
        corner = elem.p0 if a == 0.0 else elem.p0 + a * seg
        return corner, h * seg, a, h
    corner = elem.p1 if b == 1.0 else elem.p0 + b * seg
    return corner, -h * seg, b, -h


def _touching_geometry(obs_elem, src_elem, obs_interval, src_interval,
                       obs_start_is_shared, src_start_is_shared, x, y):
    """Parent coordinates and corner-stable separations of the touching rule points.

    Separations are formed as (corner offset) + x d_obs - y d_src, so points
    within 1e-10 element lengths of the corner keep their relative accuracy
    instead of cancelling two absolute positions.
    """
    corner_o, d_o, base_o, step_o = _touching_frame(obs_elem, obs_interval, obs_start_is_shared)
    corner_s, d_s, base_s, step_s = _touching_frame(src_elem, src_interval, src_start_is_shared)
    origin = corner_o - corner_s
    roundoff = 64.0 * np.finfo(float).eps * max(
        float(np.linalg.norm(corner_o)), float(np.linalg.norm(corner_s)),
        float(obs_elem.length), float(src_elem.length))
    if float(np.linalg.norm(origin)) <= roundoff:
        origin = np.zeros(2)
    diff = origin[None, :] + x[:, None] * d_o[None, :] - y[:, None] * d_s[None, :]
    return base_o + step_o * x, base_s + step_s * y, diff


def _integrate_linear_touching_duffy(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
) -> 'np.ndarray':
    """Touching-pair block for a scalar kernel callable (graded corner rule)."""
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block
    x, y, weights = _touching_rule(order)
    xi_obs, xi_src, _ = _touching_geometry(
        obs_elem, src_elem, obs_interval, src_interval, obs_start_is_shared, src_start_is_shared, x, y)
    for weight, xo, xs in zip(weights, xi_obs, xi_src):
        robs = _linear_param_to_point(obs_elem, float(xo))
        rsrc = _linear_param_to_point(src_elem, float(xs))
        block += float(weight) * complex(kernel_eval(robs, rsrc)) * np.outer(
            _linear_shape_values(float(xo)), _linear_shape_values(float(xs)))
    return block * (obs_len * src_len)


def _integrate_linear_touching_duffy_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Vectorized graded-corner rule for endpoint-touching panels (see _touching_rule)."""

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one touching-pair operator must be requested.")

    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    x, y, weights = _touching_rule(order)
    xi_obs, xi_src, diff = _touching_geometry(
        obs_elem, src_elem, obs_interval, src_interval, obs_start_is_shared, src_start_is_shared, x, y)
    phi_obs = np.column_stack((1.0 - xi_obs, xi_obs))
    phi_src = np.column_stack((1.0 - xi_src, xi_src))

    if compute_single_layer:
        g_vals = _green_2d_array(k0, np.linalg.norm(diff, axis=1))
        s_block = np.einsum(
            'q,q,qa,qb->ab', weights, g_vals, phi_obs, phi_src
        )

    if compute_double_layer:
        if obs_normal_deriv:
            dk_vals = _dgreen_dn_obs_array(k0, diff, obs_elem.normal)
        else:
            src_normals = np.broadcast_to(src_elem.normal, diff.shape)
            dk_vals = _dgreen_dn_src_array(k0, diff, src_normals)
        k_block = np.einsum(
            'q,q,qa,qb->ab', weights, dk_vals, phi_obs, phi_src
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale


def _near_kernel_values(k0, distance, with_green, with_derivative, table=None):
    """(j/4) H0(k r) and (j/4) k H1(k r) at near-pair sample distances.

    A validated kernel table (the solve's far table: native Horner, 2e-13
    relative) gives both channels in one pass; samples outside the table, and
    all samples without one, use the exact evaluators. Distances are floored at
    EPS as the exact routines floor them.
    """
    green = derivative = None
    if table is not None:
        values = table.evaluate(distance)
        green = values[..., 0].copy() if with_green else None
        derivative = values[..., 1].copy() if with_derivative else None
        bad = ~np.isfinite(values).all(axis=-1)
        if not np.any(bad):
            return green, derivative
        rest = distance[bad]
        if with_green:
            green[bad] = _green_2d_array(k0, rest)
        if with_derivative:
            kr = np.asarray(complex(k0) * rest, complex)
            kr[np.abs(kr) <= 1e-12] = 1e-12 + 0j
            derivative[bad] = (0.25j * complex(k0)) * _hankel2_1_array(kr)
        return green, derivative
    if with_green:
        green = _green_2d_array(k0, distance)
    if with_derivative:
        kr = np.asarray(complex(k0) * distance, complex)
        kr[np.abs(kr) <= 1e-12] = 1e-12 + 0j
        derivative = (0.25j * complex(k0)) * _hankel2_1_array(kr)
    return green, derivative


def _integrate_linear_touching_pairs_sk_batched(pairs, shared, k0,
        obs_normal_deriv, order, compute_single_layer=True, compute_double_layer=True, table=None):
    """The graded-corner touching rule with a bounded leading pair dimension.

    Full elements only; each pair equals `_integrate_linear_touching_duffy_sk_vectorized`
    on (0, 1) x (0, 1) to rounding, or to the table's 2e-13 when ``table`` (a
    validated kernel table) evaluates the kernels.
    """
    x0, y0, weights = _touching_rule(order)
    shared = np.asarray(shared, dtype=bool).reshape(-1, 2)
    obs, src = zip(*pairs)
    obs_p0 = np.asarray([e.p0 for e in obs], dtype=float)
    obs_p1 = np.asarray([e.p1 for e in obs], dtype=float)
    src_p0 = np.asarray([e.p0 for e in src], dtype=float)
    src_p1 = np.asarray([e.p1 for e in src], dtype=float)
    obs_start, src_start = shared[:, :1], shared[:, 1:]
    corner_o = np.where(obs_start, obs_p0, obs_p1)
    corner_s = np.where(src_start, src_p0, src_p1)
    d_o = np.where(obs_start, 1.0, -1.0) * (obs_p1 - obs_p0)
    d_s = np.where(src_start, 1.0, -1.0) * (src_p1 - src_p0)
    origin = corner_o - corner_s
    roundoff = 64.0 * np.finfo(float).eps * np.maximum.reduce([
        np.linalg.norm(corner_o, axis=1), np.linalg.norm(corner_s, axis=1),
        np.asarray([e.length for e in obs], float), np.asarray([e.length for e in src], float)])
    origin[np.linalg.norm(origin, axis=1) <= roundoff] = 0.0
    diff = (origin[:, None, :] + x0[None, :, None] * d_o[:, None, :]
            - y0[None, :, None] * d_s[:, None, :])
    x = np.where(obs_start, x0[None, :], 1.0 - x0[None, :])
    y = np.where(src_start, y0[None, :], 1.0 - y0[None, :])
    po, ps = np.stack((1-x, x), axis=-1), np.stack((1-y, y), axis=-1)
    dist = np.linalg.norm(diff, axis=-1)
    safe = np.maximum(dist, EPS)
    s = k = np.zeros((len(pairs), 2, 2), complex)
    green, deriv = _near_kernel_values(k0, safe, compute_single_layer, compute_double_layer, table)
    if compute_single_layer:
        s = po.swapaxes(1, 2) @ ((weights[None, :] * green)[:, :, None] * ps)
    if compute_double_layer:
        normals = np.asarray([e.normal for e in (obs if obs_normal_deriv else src)])
        deriv *= np.sum(diff * normals[:, None, :], axis=-1) / safe
        if obs_normal_deriv:
            deriv = -deriv
        deriv[dist <= EPS] = 0
        k = po.swapaxes(1, 2) @ ((weights[None, :] * deriv)[:, :, None] * ps)
    scale = np.asarray([a.length*b.length for a, b in pairs])[:, None, None]
    return s * scale, k * scale


def _integrate_linear_pair_recursive(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    depth: 'int' = 0,
    max_depth: 'int' = 3,
) -> 'np.ndarray':
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    same_elem_same_interval = (
        obs_elem.panel_index == src_elem.panel_index
        and abs(float(obs_interval[0]) - float(src_interval[0])) <= 1.0e-15
        and abs(float(obs_interval[1]) - float(src_interval[1])) <= 1.0e-15
    )
    if same_elem_same_interval:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_self_duffy(
            obs_elem,
            kernel_eval,
            interval=obs_interval,
            order=order,
        )

    shared = _linear_shared_interval_endpoint_info(obs_elem, obs_interval, src_elem, src_interval)
    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy(
            obs_elem,
            src_elem,
            kernel_eval,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
        )

    obs_mid = _linear_interval_midpoint(obs_elem, obs_interval)
    src_mid = _linear_interval_midpoint(src_elem, src_interval)
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_len, src_len, EPS)
    ratio = distance / scale


    if depth < max_depth and ratio < 0.95:
        oa, ob = float(obs_interval[0]), float(obs_interval[1])
        sa, sb = float(src_interval[0]), float(src_interval[1])
        if ratio < 0.16:
            om = 0.5 * (oa + ob)
            sm = 0.5 * (sa + sb)
            sub_obs = [(oa, om), (om, ob)]
            sub_src = [(sa, sm), (sm, sb)]
            for oi in sub_obs:
                for si in sub_src:
                    block += _integrate_linear_pair_recursive(
                        obs_elem,
                        src_elem,
                        kernel_eval,
                        oi,
                        si,
                        obs_order=obs_order,
                        src_order=src_order,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
            return block
        if obs_len >= src_len:
            om = 0.5 * (oa + ob)
            return (
                _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (oa, om), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
                + _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (om, ob), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
            )
        sm = 0.5 * (sa + sb)
        return (
            _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sa, sm),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
            + _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sm, sb),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
        )

    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))
    return _integrate_linear_pair_box(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=obs_interval,
        src_interval=src_interval,
        obs_order=tensor_order,
        src_order=tensor_order,
    )

def _integrate_linear_pair_generic(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_order: 'int' = 6,
    src_order: 'int' = 6,
) -> 'np.ndarray':
    """
    Assemble a 2x2 Galerkin block for one observation/source element pair.

    This upgraded implementation keeps the straight-element tensor-Gauss backbone but
    adds two accuracy-critical improvements for the experimental linear/Galerkin path:
    - Duffy-type quadrature for same-element and endpoint-touching singular pairs
    - adaptive recursive interval subdivision for near-singular pairs
    """

    return _integrate_linear_pair_recursive(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=(0.0, 1.0),
        src_interval=(0.0, 1.0),
        obs_order=obs_order,
        src_order=src_order,
        depth=0,
        max_depth=6,
    )

def _stable_hankel2_array(order: 'int', x: 'np.ndarray') -> 'np.ndarray':
    """Robust array Hankel evaluator for real and complex arguments.

    Uses scaled SciPy Hankel for complex arguments when available, then repairs
    any remaining non-finite entries with the existing scalar helpers.
    """

    z = np.asarray(x, dtype=np.complex128)
    out: 'Optional[np.ndarray]' = None
    if _SCIPY_SPECIAL is not None:
        try:

            if np.all(np.abs(z.imag) <= 1e-14) and np.all(z.real >= 0.0):
                xr = np.maximum(z.real.astype(float, copy=False), 1e-12)
                if order == 0:
                    out = np.asarray(_SCIPY_SPECIAL.j0(xr) - 1j * _SCIPY_SPECIAL.y0(xr), dtype=np.complex128)
                else:
                    out = np.asarray(_SCIPY_SPECIAL.j1(xr) - 1j * _SCIPY_SPECIAL.y1(xr), dtype=np.complex128)
            elif hasattr(_SCIPY_SPECIAL, 'hankel2e'):
                scaled = np.asarray(_SCIPY_SPECIAL.hankel2e(order, z), dtype=np.complex128)
                out = scaled * np.exp(-1j * z)
            else:
                out = np.asarray(_SCIPY_SPECIAL.hankel2(order, z), dtype=np.complex128)
        except Exception:
            out = None
    if out is None:
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        return np.asarray(vec(z), dtype=np.complex128)

    finite = np.isfinite(out.real) & np.isfinite(out.imag)
    if not np.all(finite):
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        repaired = np.asarray(vec(z[~finite]), dtype=np.complex128)
        out = np.asarray(out, dtype=np.complex128)
        out[~finite] = repaired
    return np.asarray(out, dtype=np.complex128)

def _hankel2_0_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(0, x)

def _hankel2_1_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(1, x)

def _green_2d_array(k0: 'Union[complex, float]', r: 'np.ndarray') -> 'np.ndarray':
    rr = np.maximum(np.asarray(r, dtype=float), EPS)
    x = np.asarray(complex(k0) * rr, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    return 0.25j * _hankel2_0_array(x)

def _dgreen_dn_obs_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_obs: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = (r_vec[mask] @ np.asarray(n_obs, dtype=float)) / rrm
    out[mask] = (-0.25j * complex(k0)) * h1 * projection
    return out

def _dgreen_dn_src_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_src: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = np.sum(np.asarray(n_src, dtype=float)[mask] * r_vec[mask], axis=1) / rrm
    out[mask] = (0.25j * complex(k0)) * h1 * projection
    return out


_SELF_SERIES_LIMIT = 8.0
_SELF_SUBINTERVAL_KL = 2.0


def _single_layer_self_block_composite(
    elem: 'LinearElement',
    k0: 'Union[complex, float]',
) -> 'np.ndarray':
    """Self single-layer block of an electrically long straight element.

    Beyond |k l| = 8 the exact series is ill conditioned, and a single graded
    rule cannot follow the kernel's oscillation or decay along the diagonal
    (the former two-triangle Duffy fallback was 4e-2 off at |k l| = 8.1 and
    O(1) beyond 40). The element is split into m equal sub-intervals with
    |k| l / m <= 2 (at most 8 once m exceeds 256): the exact series integrates
    each sub-interval's own block, the graded corner rule each pair of
    neighbours, and a 16-point tensor rule the separated pairs, all in the
    parent element's linear basis. The element is straight, so the kernel of
    every neighbour pair depends on x + y alone and that of every separated pair
    on the index offset alone: one kernel evaluation per offset serves all
    pairs. Separated pairs attenuated below exp(-45) by a lossy wavenumber are
    skipped.
    """

    k = complex(k0)
    ell = float(elem.length)
    kl = abs(k) * ell
    m = max(2, int(math.ceil(kl / _SELF_SUBINTERVAL_KL)))
    if m > 256:
        m = max(256, int(math.ceil(kl / _SELF_SERIES_LIMIT)))
    h = 1.0 / m
    block = np.zeros((2, 2), dtype=np.complex128)
    for i in range(m):
        sub = _single_layer_self_block_exact(elem, k, (i * h, (i + 1) * h))
        if sub is None:
            raise FloatingPointError(
                f"Self single-layer series failed on a sub-interval of panel {elem.panel_index} (|k l|={kl:.6g}).")
        block += sub

    def phi(x):
        return np.stack((1.0 - x, x), axis=-1)

    # Neighbour sub-intervals meet at s = (i + 1) h; local distances from it.
    x, y, weights = _touching_rule(_TOUCHING_RADIAL_ORDER)
    green = _green_2d_array(k, ell * h * (x + y)) * weights
    shared = (np.arange(m - 1) + 1.0) * h
    for sign in (-1.0, 1.0):
        xo = shared[:, None] + sign * h * x[None, :]
        xs = shared[:, None] - sign * h * y[None, :]
        block += np.einsum('q,iqa,iqb->ab', green, phi(xo), phi(xs)) * (ell * h) ** 2

    t, wt = _get_quadrature(16)
    t = np.asarray(t, dtype=float)
    wt = np.asarray(wt, dtype=float)
    decay = max(0.0, -k.imag) * ell * h
    for d in range(2, m):
        if decay * (d - 1) > 45.0:
            break
        # Obs sub-interval i, src sub-interval i + d: y - x = (d + t_q - t_p) h.
        kernel = _green_2d_array(k, ell * h * (d + t[None, :] - t[:, None])) * np.outer(wt, wt)
        start = np.arange(m - d)[:, None]
        lower = phi((start + t[None, :]) * h)
        upper = phi((start + d + t[None, :]) * h)
        block += (np.einsum('pq,ipa,iqb->ab', kernel, lower, upper)
                  + np.einsum('pq,iqa,ipb->ab', kernel, upper, lower)) * (ell * h) ** 2
    return block


def _single_layer_block_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
) -> 'np.ndarray':
    if len(obs_elem.node_ids) > 2 or len(src_elem.node_ids) > 2:
        from ghost_backend.twod.polynomial_quadrature import near_block
        return near_block(obs_elem, src_elem, k0)[0]
    if obs_elem.panel_index != src_elem.panel_index:
        shared = _linear_shared_interval_endpoint_info(obs_elem, (0., 1.), src_elem, (0., 1.))
        if shared is not None:
            return _integrate_linear_touching_duffy_sk_vectorized(
                obs_elem, src_elem, k0, False, (0., 1.), (0., 1.), shared[0], shared[1],
                order=max(6, max(int(obs_order), int(src_order)) + 1),
                compute_single_layer=True, compute_double_layer=False)[0]
    if obs_elem.panel_index == src_elem.panel_index:
        exact = _single_layer_self_block_exact(obs_elem, k0)
        if exact is not None:
            return exact
        if abs(complex(k0)) * float(obs_elem.length) > _SELF_SERIES_LIMIT:
            return _single_layer_self_block_composite(obs_elem, k0)
    elif requires_adaptive(obs_elem, src_elem):
        return _integrate_linear_pair_adaptive_sk(
            obs_elem, src_elem, k0, False, max(16, obs_order), max(16, src_order),
            compute_single_layer=True, compute_double_layer=False)[0]
    return _integrate_linear_pair_generic(
        obs_elem,
        src_elem,
        lambda robs, rsrc: _green_2d(k0, max(float(np.linalg.norm(robs - rsrc)), EPS)),
        obs_order=obs_order,
        src_order=src_order,
    )


def _sk_blocks_near_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Compute S and K 2x2 blocks for a near element pair.

    Uses Duffy transforms for self and touching pairs (via the existing recursive
    path), and the vectorized tensor-Gauss path for separated-near pairs.
    """
    if len(obs_elem.node_ids) > 2 or len(src_elem.node_ids) > 2:
        from ghost_backend.twod.polynomial_quadrature import near_block
        sb, kb = near_block(obs_elem, src_elem, k0, obs_normal_deriv)
        return (sb if compute_single_layer else np.zeros_like(sb),
                kb if compute_double_layer else np.zeros_like(kb))
    same_elem = obs_elem.panel_index == src_elem.panel_index


    shared = (
        None
        if same_elem
        else _linear_shared_interval_endpoint_info(
            obs_elem, (0.0, 1.0), src_elem, (0.0, 1.0), tol=1.0e-9
        )
    )

    zero = np.zeros((2, 2), dtype=np.complex128)
    if same_elem:


        s_blk = (
            _single_layer_block_linear(
                obs_elem, src_elem, k0, obs_order, src_order
            ) if compute_single_layer else zero
        )
        return s_blk, zero

    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=(0.0, 1.0),
            src_interval=(0.0, 1.0),
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )


    obs_mid = obs_elem.center
    src_mid = src_elem.center
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_elem.length, src_elem.length, EPS)
    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))

    if requires_adaptive(obs_elem, src_elem):
        return _integrate_linear_pair_adaptive_sk(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_order=tensor_order,
            src_order=tensor_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    return _integrate_linear_pair_box_sk_vectorized(
        obs_elem, src_elem, k0, obs_normal_deriv,
        obs_interval=(0.0, 1.0), src_interval=(0.0, 1.0),
        obs_order=tensor_order, src_order=tensor_order,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
    )


NEAR_PAIR_QUADRATURE_RTOL = 1.0e-9
NEAR_PAIR_QUADRATURE_MAX_DEPTH = 12


def _integrate_linear_pair_adaptive_sk(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    rtol: 'float' = NEAR_PAIR_QUADRATURE_RTOL,
    max_depth: 'int' = NEAR_PAIR_QUADRATURE_MAX_DEPTH,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Converged S/K quadrature for separated, nearly singular panel pairs.

    Each box is compared with its four-way bisection.  Only children whose
    parent comparison has not converged are refined further, so a narrow
    diagonal interaction costs O(2**depth), rather than uniformly applying a
    very high tensor rule to the whole pair.  A child block already computed
    for its parent's error estimate is reused as its own coarse estimate.

    The error denominator is the sum of child-block norms, rather than the
    norm of their possibly cancelling sum.  This prevents a physical null in
    one 2x2 block from making the convergence test spuriously permissive.
    Failure at ``max_depth`` is explicit: silently accepting an unresolved
    close-gap interaction can produce a small linear-system residual for the
    wrong discrete operator.
    """

    zero = np.zeros((2, 2), dtype=np.complex128)

    def evaluate(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        return _integrate_linear_pair_box_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_order=obs_order,
            src_order=src_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    def relative_error(
        coarse: 'np.ndarray',
        children: 'List[np.ndarray]',
    ) -> 'float':


        fine = sum(children, zero.copy())
        scale_norm = sum(float(np.linalg.norm(block)) for block in children)
        floor = np.finfo(float).eps * max(
            1.0,
            float(obs_elem.length) * float(src_elem.length),
        )
        return float(np.linalg.norm(fine - coarse)) / max(scale_norm, floor)

    def recurse(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
        depth: 'int',
        coarse: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None,
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        coarse_s, coarse_k = coarse if coarse is not None else evaluate(
            obs_interval, src_interval
        )
        oa, ob = map(float, obs_interval)
        sa, sb = map(float, src_interval)
        om = 0.5 * (oa + ob)
        sm = 0.5 * (sa + sb)
        child_intervals = [
            ((oa, om), (sa, sm)),
            ((oa, om), (sm, sb)),
            ((om, ob), (sa, sm)),
            ((om, ob), (sm, sb)),
        ]
        child_blocks = [evaluate(oi, si) for oi, si in child_intervals]
        s_children = [block[0] for block in child_blocks]
        k_children = [block[1] for block in child_blocks]
        err_s = (
            relative_error(coarse_s, s_children)
            if compute_single_layer else 0.0
        )
        err_k = (
            relative_error(coarse_k, k_children)
            if compute_double_layer else 0.0
        )
        error = max(err_s, err_k)
        if error <= float(rtol):
            return (
                sum(s_children, zero.copy()),
                sum(k_children, zero.copy()),
            )
        if depth >= int(max_depth):
            gap_ratio = float(np.linalg.norm(obs_elem.center - src_elem.center)) / max(
                float(obs_elem.length), float(src_elem.length), EPS
            )
            raise FloatingPointError(
                "Separated-near Galerkin quadrature did not converge: "
                f"panel pair ({obs_elem.panel_index}, {src_elem.panel_index}), "
                f"center-gap/length={gap_ratio:.6g}, estimated relative block "
                f"error={error:.3e} after depth {depth}. Refine the boundary "
                "mesh or increase NEAR_PAIR_QUADRATURE_MAX_DEPTH."
            )

        s_total = zero.copy()
        k_total = zero.copy()
        for (oi, si), child in zip(child_intervals, child_blocks):
            child_s, child_k = recurse(
                oi, si, depth + 1, coarse=child
            )
            s_total += child_s
            k_total += child_k
        return s_total, k_total

    return recurse((0.0, 1.0), (0.0, 1.0), depth=0)

_TANGENT_OUTER = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)

def _hypersingular_block_from_s_block(
    s_block: 'np.ndarray',
    k0: 'Union[complex, float]',
    n_obs: 'np.ndarray',
    n_src: 'np.ndarray',
    obs_length: 'float',
    src_length: 'float',
) -> 'np.ndarray':
    """
    Compute the 2x2 hypersingular D block from the single-layer S block via Maue identity.

    The Maue regularisation recasts the hypersingular kernel integral as:
        D_ij = -k^2 (n_obs . n_src) S_ij
             + (1/(L_obs*L_src)) * tangent_outer_ij * sum(S_block)

    where tangent_outer = [[1,-1],[-1,1]] encodes the linear shape-function
    tangential derivatives.  This avoids all hypersingular quadrature.
    """

    if s_block.shape != (2, 2):
        from ghost_backend.twod.polynomial_quadrature import hypersingular
        return hypersingular(s_block, k0, n_obs, n_src, obs_length, src_length)
    k2 = complex(k0) ** 2
    n_dot_n = float(np.dot(n_obs, n_src))
    raw_integral = complex(np.sum(s_block))
    denom = max(float(obs_length) * float(src_length), EPS * EPS)
    return -k2 * n_dot_n * s_block + _TANGENT_OUTER * (raw_integral / denom)


_ASSEMBLY_TILE_TARGET_BYTES = 24 * 1024 * 1024
_NEAR_BATCH_MAX_SAMPLES = 1_000_000
# Concurrent near batches keep the former single-batch working set in total.
_NEAR_BATCH_THREAD_SAMPLES = 125_000
# Bound the advanced-index copies and Maue blocks used by the final scatter.
# This does not alter pair order within any destination operator.
_NEAR_SCATTER_MAX_PAIRS = 16_384


def _env_positive_int(name: 'str', default: 'int') -> 'int':
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_ASSEMBLY_THREADS = _env_positive_int("GHOST_ASSEMBLY_THREADS", 1)
_ASSEMBLY_TILE = _env_positive_int("GHOST_ASSEMBLY_TILE", 0)


_FAR_QUAD_ORDER = _env_positive_int("GHOST_FAR_QUAD_ORDER", 0)


def _env_float(name: 'str', default: 'float') -> 'float':
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


# Far pairs of an attenuating medium whose kernels are negligible are skipped:
# with lower a lower bound of the point-pair distance (centre distance less
# half of each length), |i H0(kr)/4| <= exp(-a)/sqrt(8 pi a) for a = -Im(k)
# lower > 1 (DLMF 10.27.8, 10.32.9), below 1e-18 of the near-diagonal kernel
# scale at the cut of 40 (the compressed backend applies the same bound at 32
# and accounts the dropped routes in its error evidence).  The near/far
# classification and the graded orders are unchanged.  0 disables the cut.
FAR_ATTENUATION_CUT = _env_float("GHOST_FAR_ATTENUATION_CUT", 40.0)


_ASSEMBLY_COMPACT_BELOW = 0.5


def set_assembly_compaction(fraction: 'float') -> 'None':
    """Set rectangular/union packing threshold; zero selects full-width assembly."""

    global _ASSEMBLY_COMPACT_BELOW
    _ASSEMBLY_COMPACT_BELOW = float(fraction)


def set_far_quadrature_order(order: 'int') -> 'None':
    """Override the far-pair quadrature order (0 restores the default rule)."""

    global _FAR_QUAD_ORDER
    _FAR_QUAD_ORDER = max(0, int(order))


def set_assembly_threads(count: 'int') -> 'None':
    """
    Set how many threads tiled operator assembly may use (1 = serial).

    Assembly tiles are independent and the heavy numpy/SciPy ufuncs inside them
    release the GIL, so this scales usefully when a node has more cores than
    concurrent solves.  When a run has at least one unit per core, leave it at
    1 and let the process pool own the parallelism -- threads and processes
    competing for the same cores is strictly worse than either alone.
    """

    global _ASSEMBLY_THREADS
    _ASSEMBLY_THREADS = max(1, int(count))


def get_assembly_threads() -> 'int':
    """Current tiled-assembly thread count."""

    return int(effective_assembly_threads(_ASSEMBLY_THREADS))


def _assembly_tile_size(nelems: 'int', bytes_per_entry: 'int') -> 'int':
    """Pick an element-tile edge so one tile's working set stays cache-sized.

    When assembly threads are enabled the tile is also capped so there are
    several observation blocks per thread: blocks are the unit of parallelism,
    and the symmetric traversal makes the first block the most expensive (it
    pairs with every later one), so a handful of coarse blocks would both
    starve threads and hand them wildly unequal work.
    """

    if option('assembly_tile', _ASSEMBLY_TILE) > 0:
        return max(1, min(int(nelems), option('assembly_tile', _ASSEMBLY_TILE)))
    if nelems <= 192:
        return int(nelems)
    entries = float(_ASSEMBLY_TILE_TARGET_BYTES) / float(max(1, int(bytes_per_entry)))
    tile = int(math.sqrt(max(1.0, entries)))
    tile = max(128, min(1024, min(int(nelems), tile)))
    if get_assembly_threads() > 1:
        per_thread_blocks = 4
        tile = min(
            tile,
            max(64, int(math.ceil(nelems / (per_thread_blocks * get_assembly_threads())))),
        )
    return max(1, min(int(nelems), tile))


# Calibrated far-rule orders per polynomial degree: rows (|k| Lmax bound,
# (near, mid, far) for degree 1, 2, 3), where near, mid and far are the columns
# for a smallest centre-distance ratio below 5, below 10, and from 10 up. Each
# order is the smallest Gauss rule whose element-pair blocks stay within 1e-12
# of an independent 48-point rule over the bin: S relative to the block's
# largest entry, K' and D relative to the integral of the kernel magnitude (the
# block's scale before cancellation; a K' block of collinear panels is exactly
# zero, so its own size is no yardstick). Sampled at ratios 3..60, |k| L up to
# each bound, length ratios 1..0.1 and orientations 0..pi. The first table holds
# for arg k from 0 to -45 degrees; strongly attenuating media (arg k below -45
# degrees, sampled to -89) use the second. Outside the tables (ratio below 3,
# |k| L above 3) the order is not reduced. test_audit_fixes_2d_operators keeps
# the check.
_FAR_ORDER_TABLE = (
    (0.15, ((7, 6, 5), (7, 6, 5), (8, 7, 6))),
    (0.50, ((7, 6, 5), (7, 6, 6), (8, 7, 6))),
    (1.50, ((7, 6, 6), (8, 7, 7), (8, 7, 7))),
    (3.00, ((8, 8, 8), (8, 8, 8), (9, 9, 9))),
)
_FAR_ORDER_TABLE_ATTENUATING = (
    (0.15, ((7, 6, 5), (7, 6, 5), (8, 7, 6))),
    (0.50, ((7, 6, 5), (7, 6, 6), (8, 7, 6))),
    (1.50, ((7, 7, 7), (8, 7, 7), (8, 8, 8))),
    (3.00, ((8, 8, 8), (9, 8, 8), (9, 9, 9))),
)
_FAR_TABLE_MIN_RATIO = 3.0

_FAR_GRADED = _env_positive_int("GHOST_FAR_GRADED", 1) != 0
_NATIVE_FAR = True


def set_far_quadrature_grading(enabled: 'bool') -> 'None':
    """Enable/disable per-tile far-quadrature grading (default on).

    Grading only reduces the order below what the caller configured, and only
    where the calibrated table (``_FAR_ORDER_TABLE``) says the reduction keeps
    the element-pair block within 1e-12 of a converged rule, so turning it off
    should change nothing that matters.  The switch exists to make that testable.
    """

    global _FAR_GRADED
    _FAR_GRADED = bool(enabled)


def _graded_far_order(kl_max: 'float', ratio_min: 'float', cap: 'int', degree: 'int' = 1,
                      attenuating: 'bool' = False) -> 'int':
    """Quadrature order for a tile whose worst far pair has these parameters.

    ``degree`` is the basis polynomial degree (1..3): higher degrees need more
    points for the same block accuracy. ``attenuating`` selects the table for
    wavenumbers with arg k below -45 degrees.
    """

    if not option('far_grading', _FAR_GRADED):
        return int(cap)
    if not float(ratio_min) >= _FAR_TABLE_MIN_RATIO:
        return int(cap)
    row = min(max(int(degree), 1), 3) - 1
    table = _FAR_ORDER_TABLE_ATTENUATING if attenuating else _FAR_ORDER_TABLE
    for bound, orders in table:
        if kl_max <= bound:
            near, mid, far = orders[row]
            if ratio_min < 5.0:
                order = near
            elif ratio_min < 10.0:
                order = mid
            else:
                order = far
            return max(2, min(int(cap), int(order)))
    return int(cap)


# Calibrated W far rule (degree 1: 8 points, degrees 2-3: 9 points). The
# calibration compares the completed Maue block, including its cancellation,
# with independent 48-point blocks for passive k with 1e-4 <= |k| Lmax <= 3,
# arg k from 0 to -90 degrees, length ratios down to 0.001 and all orientations
# at a separation of three panel lengths: worst block errors 1.5e-14 (degree 1
# at 8 points), 3.7e-14 and 4.6e-14 (degrees 2 and 3 at 9 points), against 1e-12
# promised (test_performance_updates and test_audit_fixes_2d_operators).
_W_FAR_FLOOR = {1: 8, 2: 9, 3: 9}


def _graded_w_far_floor(k, lengths, separation, degree):
    """Calibrated W rule for separated, resolved panels; otherwise retain 16.

    The calibration compares the completed Maue block (including cancellation)
    against 40/48 point rules for degrees 1..3, complex passive k, orientations
    and length ratios. Use global geometry bounds so compact queries and worker
    tiles choose the same rule. Self/adjacent/near quadrature is unchanged.
    """
    wave = complex(k)
    longest = float(np.max(lengths))
    if (option('far_grading', _FAR_GRADED) and float(separation) >= 3.0
            and 1 <= degree <= 3 and wave.real >= 0 and wave.imag <= 0
            and 1.e-4 <= abs(wave)*longest <= 3.0
            and float(np.min(lengths)) >= .001*longest):
        return _W_FAR_FLOOR[int(degree)]
    return 16


def _wavenumber_is_real(k0: 'Union[complex, float]') -> 'bool':
    value = complex(k0)
    return value.imag == 0.0 and value.real > 0.0


def _axpy_into(acc: 'np.ndarray', src: 'np.ndarray', coeff: 'float',
               scratch: 'np.ndarray') -> 'None':
    """acc += coeff * src, in place, without allocating a temporary.

    Deliberately not scipy's BLAS axpy, though not for the reason an earlier
    comment here gave: its f2py overhead is microseconds, not milliseconds, and
    zaxpy writes into its y argument, so pinned to one thread it is about 1.95x
    faster for exactly equal results. It loses because BLAS threads itself
    while this assembly is already thread-parallel at the tile level -- swapping
    it in measured 13.65 -> 12.65 s at one assembly thread but 8.33 -> 13.98 s
    at four. Gating on the thread count would make threaded and serial runs
    differ in their last bits, which _expand_near_chunks exists to prevent.
    """

    np.multiply(src, coeff, out=scratch)
    np.add(acc, scratch, out=acc)


def _accumulate_first(acc: 'np.ndarray', src: 'np.ndarray', coeff: 'float',
                      index: 'int', scratch: 'np.ndarray') -> 'None':
    """acc = coeff*src on the first index, acc += coeff*src afterwards.

    Overwriting on the first source node is what lets the partial sums skip a
    zero fill per observation node, which would otherwise cost about as much as
    the accumulations it saves.
    """

    if index:
        _axpy_into(acc, src, coeff, scratch)
    else:
        np.multiply(src, coeff, out=acc)


def _far_kernel_argument(k0: 'Union[complex, float]', dist: 'np.ndarray',
                         out: 'np.ndarray') -> 'None':
    """kr = k0 * dist on the real fast path, floored where the scalar
    evaluators floor it so the two agree entry for entry."""

    np.multiply(dist, complex(k0).real, out=out)
    np.maximum(out, 1e-12, out=out)


def _far_green_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) H_0^(2)(k0 r) over a whole tile.

    For real k0 this is (1/4)(Y_0(kr) + j J_0(kr)), so the two real Bessel
    evaluations write straight into the halves of the complex output buffer.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        _SCIPY_SPECIAL.y0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.real)
        _SCIPY_SPECIAL.j0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_0_array(arg), 0.25j, out=out)


def _far_hankel1_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) k0 H_1^(2)(k0 r) over a whole tile.

    Kept separate from the projection so both orientations of an element pair
    can reuse one Bessel evaluation -- the normal-derivative kernels differ
    only by which normal the displacement is projected onto.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        coeff = 0.25 * complex(k0).real
        _SCIPY_SPECIAL.y1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.real)
        _SCIPY_SPECIAL.j1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_1_array(arg), 0.25j * complex(k0), out=out)


def _run_tiled_obs_blocks(
    nelems: 'int',
    tile: 'int',
    body: 'Callable[[int, int], None]',
) -> 'None':
    """Run ``body(i0, i1)`` over every observation tile, threaded when asked."""

    starts = list(range(0, nelems, tile))
    workers = min(get_assembly_threads(), len(starts))
    from ghost_backend.twod.assembly.session import current_session
    owner=current_state() or current_session()
    captured = current_options()
    allocation = get_assembly_threads()
    def run(i0):
        if captured is not None:
            with execution_scope(captured, assembly_threads=allocation):
                return run_body(i0)
        return run_body(i0)
    def run_body(i0):
        if owner is not None:owner.checkpoint()
        body(i0,min(i0+tile,nelems))
        if owner is not None:owner.checkpoint()
    if workers <= 1:
        for i0 in starts:
            run(i0)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, starts))


def _run_ordered_tiles(count: 'int', compute: 'Callable[[int], Optional[Callable[[], None]]]') -> 'None':
    """Run ``compute(i)`` for every tile, committing its result in tile order.

    ``compute`` integrates one tile and returns a ``commit`` callable (or None)
    that scatters it. Tiles are computed by up to the assembly thread count, but
    commits run on the calling thread strictly in index order, so every
    destination entry receives its far contributions in the serial order and a
    threaded assembly reproduces the serial one bit for bit. At most twice the
    worker count of finished tiles wait for their commit.
    """

    workers = min(get_assembly_threads(), count)
    from ghost_backend.twod.assembly.session import current_session
    owner = current_state() or current_session()
    captured = current_options()
    allocation = get_assembly_threads()

    def run(index):
        if owner is not None:
            owner.checkpoint()
        if captured is not None:
            with execution_scope(captured, assembly_threads=allocation):
                return compute(index)
        return compute(index)

    if workers <= 1:
        for index in range(count):
            commit = run(index)
            if commit is not None:
                commit()
            if owner is not None:
                owner.checkpoint()
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        pending = deque()
        submitted = 0
        while submitted < count and len(pending) < 2 * workers:
            pending.append(pool.submit(run, submitted))
            submitted += 1
        while pending:
            commit = pending.popleft().result()
            if submitted < count:
                pending.append(pool.submit(run, submitted))
                submitted += 1
            if commit is not None:
                commit()
            if owner is not None:
                owner.checkpoint()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


_NEAR_CLASSIFY_CHUNK = 1 << 19


def _near_fixed_order_positions(
    panel_index: 'np.ndarray',
    obs_idx: 'np.ndarray',
    src_idx: 'np.ndarray',
    p0_arr: 'np.ndarray',
    p1_arr: 'np.ndarray',
    centers: 'np.ndarray',
    lengths: 'np.ndarray',
    obs_order: 'int',
    src_order: 'int',
    node_ids: 'Optional[np.ndarray]' = None,
) -> 'Dict[int, np.ndarray]':
    """Bucket near pairs by tensor-quadrature order, for every pair at once.

    The per-pair route applies three exclusions -- a self panel, a pair sharing
    an endpoint, and a pair close enough to need the adaptive rule -- and orders
    the rest from the centre-distance ratio. Each is a geometric predicate, so
    all of them evaluate as arrays; doing it a pair at a time was the largest
    interpreter-locked stretch of the assembly and what kept it from using its
    threads.

    Positions stay ascending within each bucket, and every block is still
    integrated independently, so the assembled coefficients are unchanged.
    Scratch is bounded by processing the pair list in chunks.
    """

    from ghost_backend.twod.assembly.separation import adaptive_pairs
    buckets: 'Dict[int, List[np.ndarray]]' = {}
    floor = int(max(obs_order, src_order))
    for start in range(0, obs_idx.size, _NEAR_CLASSIFY_CHUNK):
        o = obs_idx[start:start + _NEAR_CLASSIFY_CHUNK]
        s = src_idx[start:start + _NEAR_CLASSIFY_CHUNK]
        keep = panel_index[o] != panel_index[s]
        # Shared endpoints: the per-pair check's snap tolerance or one mesh node.
        if node_ids is None:
            for a in (p0_arr, p1_arr):
                for b in (p0_arr, p1_arr):
                    keep &= np.linalg.norm(a[o] - b[s], axis=1) > NODE_SNAP_TOLERANCE
        else:
            touching, _, _ = _shared_endpoints(p0_arr, p1_arr, node_ids, o, s)
            keep &= ~touching
        # requires_adaptive, term for term.
        scale = np.maximum(lengths[o], lengths[s])
        distance = np.linalg.norm(centers[o] - centers[s], axis=1)
        adaptive = adaptive_pairs(p0_arr[o], p1_arr[o], p0_arr[s], p1_arr[s],
                                  distance, scale, lengths[o], lengths[s])
        keep &= ~adaptive
        # _near_singular_scheme's order ladder, then the caller's clamp.
        ratio = distance / np.maximum(scale, EPS)
        adapt = np.select(
            [ratio < 0.25, ratio < 0.60, ratio < 1.50, ratio < 3.00],
            [64, 56, 40, 28], default=16,
        )
        order = np.maximum(floor, np.minimum(16, np.maximum(5, adapt)))
        for value in np.unique(order[keep]):
            buckets.setdefault(int(value), []).append(
                start + np.flatnonzero(keep & (order == value))
            )
    return {order: np.concatenate(parts) for order, parts in buckets.items()}


def _expand_near_chunks(
    chunks: 'List[Tuple[int, np.ndarray, int, np.ndarray, bool]]',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Flatten recorded near-pair tiles into ascending (obs, src) index arrays.

    Each chunk carries the tile's global source-element ids, because a masked
    assembly compacts the source axis and the recorded column is an index into
    that compacted axis rather than into the mesh.

    Sorting here is what keeps the near-field accumulation order independent of
    how tiles were scheduled, so a threaded assembly reproduces a serial one
    exactly rather than only to within floating-point reassociation.
    """

    if not chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    obs_parts: 'List[np.ndarray]' = []
    src_parts: 'List[np.ndarray]' = []
    for row_base, src_global, ncols, flat, transposed in chunks:
        local_rows = flat // ncols
        rows = row_base + local_rows if np.isscalar(row_base) else row_base[local_rows]
        cols = src_global[flat % ncols]
        if transposed:
            obs_parts.append(cols)
            src_parts.append(rows)
        else:
            obs_parts.append(rows)
            src_parts.append(cols)
    obs_idx = np.concatenate(obs_parts)
    src_idx = np.concatenate(src_parts)
    order = np.lexsort((src_idx, obs_idx))
    return obs_idx[order], src_idx[order]


def _warn_far_quadrature_override(materials: 'MaterialLibrary') -> 'None':
    """Record a far-quadrature override in the solve's own warnings.

    The override changes computed values, so a field produced under it must
    say so wherever it travels; the warning list is copied into the .grim
    audit, which is the record that survives the run directory.
    """

    if option('far_quadrature_order', _FAR_QUAD_ORDER) <= 0:
        return
    materials.warn_once(
        f"Far-pair quadrature order overridden to {option('far_quadrature_order', _FAR_QUAD_ORDER)} "
        "via GHOST_FAR_QUAD_ORDER / set_far_quadrature_order. "
        "The default uses distance-graded orders (normally 5 to 10). "
        "Changing this rule means these values are NOT bit-comparable "
        "with default-rule results. The mesh-convergence certificate still "
        "applies to the discretization, not to this quadrature choice."
    )


@timed_stage("operators")
@cached_operator("SK")
def _assemble_linear_operator_matrices_multi(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    source_element_masks: 'Sequence[Optional[np.ndarray]]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    compute_double_layer_many: 'Optional[Sequence[bool]]' = None,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
    single_layer_observation_coefficients_many: 'Optional[Sequence[Optional[np.ndarray]]]' = None,
    output_node_ids_many=None,
    double_layer_output_node_ids_many=None,
    operator_outputs=None,
    prepared_geometry=None,
    additional_operator_outputs=None,
    minimum_far_order=0,
) -> 'List[Tuple[np.ndarray, np.ndarray]]':
    """
    Assemble S and K/K' for several source-element masks in ONE traversal.

    The masks select which source elements contribute, but they are applied to
    the finished tile accumulators -- the quadrature itself does not depend on
    them.  Assembling each mask separately therefore repeats every Hankel
    evaluation, which is most of the cost of a solve.  The multi-region
    formulation asks for exactly this: one operator per (region, interface
    side), where both sides of a region share a wavenumber and differ only in
    which elements are active.

    Returns one (S, K) pair per mask, in the order given.  A mask selecting no
    element gets zero matrices without costing anything.  When
    ``single_layer_observation_coefficients`` is supplied, every returned S
    matrix uses that piecewise-constant coefficient inside the observation
    integral. ``single_layer_observation_coefficients_many`` instead supplies
    one coefficient vector per source mask.  This permits several differently
    weighted S matrices to share exactly the same kernel/quadrature traversal.

        S_c[i,j] = sum_e integral_e phi_i(x) c_e (S phi_j)(x) ds,

    while K/K' remains unweighted.  This is required for spatially varying
    Robin and sheet coefficients: multiplying completed rows by a nodal
    average is not the same weak form.

    ``output_node_ids_many`` optionally supplies (observation nodes, source
    nodes) per output. Coefficients are assembled directly into rectangular
    CompactOperator storage, retaining global IDs for block lookup. Omitted
    observation/source nodes consume no dense storage or scattered entries.

    With explicit S/K' destinations, ``additional_operator_outputs`` supplies
    one (D, W) destination pair per mask. These unweighted operators reuse the
    same Green and normal-derivative quadrature. W uses its calibrated far
    rule, with the order-16 floor outside the calibrated range.

    Two passes, as before:
    1. Far interactions: batched quadrature over cache-sized element tiles,
       one kernel evaluation per unordered pair. Tiles may be integrated on
       several threads, but they are scattered strictly in tile order, so the
       result does not depend on the thread count.
    2. Near interactions: self, endpoint-touching, fixed-rule and adaptive
       element pairs over the union of all masks, followed by one ordered
       scatter per requested output. Overlapping weighted/unweighted masks
       therefore do not duplicate singular-kernel evaluation.
    """
    return _assemble_multi(
        mesh, k0, obs_normal_deriv, source_element_masks, obs_order=obs_order,
        src_order=src_order, far_ratio=far_ratio, compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer, compute_double_layer_many=compute_double_layer_many,
        single_layer_observation_coefficients=single_layer_observation_coefficients,
        single_layer_observation_coefficients_many=single_layer_observation_coefficients_many,
        output_node_ids_many=output_node_ids_many,
        double_layer_output_node_ids_many=double_layer_output_node_ids_many,
        operator_outputs=operator_outputs, prepared_geometry=prepared_geometry,
        additional_operator_outputs=additional_operator_outputs, minimum_far_order=minimum_far_order)


def _shared_endpoints(
    p0_arr: 'np.ndarray',
    p1_arr: 'np.ndarray',
    node_ids: 'np.ndarray',
    obs: 'np.ndarray',
    src: 'np.ndarray',
    tol: 'float' = None,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Endpoint-touching test for many element pairs at once.

    Returns (touching, obs_start_is_shared, src_start_is_shared) with the first
    matching endpoint pair in the order of ``_linear_shared_interval_endpoint_info``
    (obs start/src start, obs start/src end, obs end/src start, obs end/src end).
    Endpoints touch within the node-snap tolerance, or when they are one mesh node
    (snapped coordinates may sit up to sqrt(2) snap widths apart).
    """
    tol = NODE_SNAP_TOLERANCE if tol is None else float(tol)
    touching = np.zeros(obs.size, dtype=bool)
    obs_start = np.zeros(obs.size, dtype=bool)
    src_start = np.zeros(obs.size, dtype=bool)
    for a, pa in ((0, p0_arr), (1, p1_arr)):
        for b, pb in ((0, p0_arr), (1, p1_arr)):
            distance = np.linalg.norm(pa[obs] - pb[src], axis=1)
            match = (distance <= tol) | ((node_ids[obs, a] == node_ids[src, b]) & (distance <= 2.0 * tol))
            match &= ~touching
            obs_start[match] = a == 0
            src_start[match] = b == 0
            touching |= match
    return touching, obs_start, src_start


def _maue_blocks(s_blocks, k0, obs_normals, src_normals, obs_lengths, src_lengths):
    """``_hypersingular_block_from_s_block`` for a stack of element blocks.

    The linear case repeats the scalar routine's floating-point operations
    (np.dot's rounding of the normal product, np.sum's pairing of the four
    entries and a real division by the length product), so each block equals the
    per-pair result bit for bit.
    """
    if not len(s_blocks):
        return np.zeros_like(s_blocks)
    k2 = complex(k0) ** 2
    n_dot_n = np.vecdot(np.asarray(obs_normals, dtype=float), np.asarray(src_normals, dtype=float))
    if s_blocks.shape[1:] != (2, 2):
        width = s_blocks.shape[1]
        derivative = derivative_matrix(width - 1)
        return ((-k2 * n_dot_n)[:, None, None] * s_blocks
                + np.einsum('ia,pij,jb->pab', derivative, s_blocks, derivative)
                / (np.asarray(obs_lengths, float) * np.asarray(src_lengths, float))[:, None, None])
    raw = (s_blocks[:, 0, 0] + s_blocks[:, 0, 1]) + (s_blocks[:, 1, 0] + s_blocks[:, 1, 1])
    denom = np.maximum(np.asarray(obs_lengths, float) * np.asarray(src_lengths, float), EPS * EPS)
    quotient = np.empty(raw.shape, dtype=np.complex128)
    quotient.real = raw.real / denom
    quotient.imag = raw.imag / denom
    return (((-k2) * n_dot_n)[:, None, None] * s_blocks
            + _TANGENT_OUTER[None, :, :] * quotient[:, None, None])



@assembly_component('near_integration')
def _near_pair_blocks(elements, obs_idx, src_idx, k0, obs_normal_deriv, obs_order, src_order,
                      integrate_s, want_k, p0_arr, seg_arr, centers, lengths, node_ids, far_table,
                      single_layer_blocks=None, double_layer_blocks=None, prepared_geometry=None):
    """Integrate one bounded set of pairs, using the unchanged quadrature rules."""
    npairs, width = len(obs_idx), node_ids.shape[1]
    panel_index = np.asarray([e.panel_index for e in elements], dtype=np.int64)
    s_near, k_near = single_layer_blocks, double_layer_blocks
    if integrate_s and s_near is None:
        s_near = np.zeros((npairs, width, width), dtype=np.complex128)
    if want_k and k_near is None:
        k_near = np.zeros((npairs, width, width), dtype=np.complex128)
    done = np.zeros(npairs, dtype=bool)


    fixed_positions_by_order = _near_fixed_order_positions(
        panel_index, obs_idx, src_idx, p0_arr, p0_arr + seg_arr, centers, lengths,
        obs_order, src_order, node_ids=node_ids,
    )

    batches = []
    for tensor_order, positions in fixed_positions_by_order.items():
        # Batch boundaries do not depend on the thread count, so threaded and
        # serial assembly integrate identical batches.
        batch_pairs = max(1, _NEAR_BATCH_THREAD_SAMPLES // max(1, tensor_order * tensor_order))
        for start in range(0, len(positions), batch_pairs):
            batches.append((tensor_order, positions[start:start + batch_pairs]))

    def _fixed_batch(batch):
        tensor_order, selected = batch
        selected_arr = np.asarray(selected, dtype=np.int64)
        return _integrate_linear_pairs_box_sk_batched(
            elements,
            obs_idx[selected_arr],
            src_idx[selected_arr],
            k0,
            obs_normal_deriv,
            tensor_order,
            compute_single_layer=integrate_s,
            compute_double_layer=want_k,
            prepared_geometry=prepared_geometry,
        )

    from ghost_backend.twod.polynomial_quadrature import map_checked, solve_checkpoint
    near_workers = min(get_assembly_threads(), len(batches),
                       max(1, _NEAR_BATCH_MAX_SAMPLES // _NEAR_BATCH_THREAD_SAMPLES))
    batch_results = map_checked(_fixed_batch, batches, near_workers, solve_checkpoint())
    for (_, selected), (s_batch, k_batch) in zip(batches, batch_results):
        if s_near is not None:
            s_near[selected] = s_batch
        if k_near is not None:
            k_near[selected] = k_batch
        done[selected] = True
    batch_results = None

    p1_arr = p0_arr + seg_arr
    if width == 2:
        candidates = np.flatnonzero(~done & (panel_index[obs_idx] != panel_index[src_idx]))
        touching, obs_start, src_start = _shared_endpoints(
            p0_arr, p1_arr, node_ids, obs_idx[candidates], src_idx[candidates])
        touching_pos = candidates[touching]
        endpoints = np.column_stack((obs_start[touching], src_start[touching]))
        order = max(6, max(int(obs_order), int(src_order)) + 1)
        samples = 2 * _touching_order(order) ** 2
        per_batch = max(1, _NEAR_BATCH_THREAD_SAMPLES // samples)
        jobs = [(touching_pos[start:start+per_batch], endpoints[start:start+per_batch])
                for start in range(0, len(touching_pos), per_batch)]
        def integrate_touching(job):
            positions, shared = job
            pairs = [(elements[int(obs_idx[p])], elements[int(src_idx[p])]) for p in positions]
            return _integrate_linear_touching_pairs_sk_batched(pairs, shared, k0,
                obs_normal_deriv, order, integrate_s, want_k, table=far_table)
        near_workers = min(get_assembly_threads(), len(jobs),
                           max(1, _NEAR_BATCH_MAX_SAMPLES // _NEAR_BATCH_THREAD_SAMPLES))
        results = map_checked(integrate_touching, jobs, near_workers, solve_checkpoint())
        for (positions, _), (sb, kb) in zip(jobs, results):
            if s_near is not None:
                s_near[positions] = sb
            if k_near is not None:
                k_near[positions] = kb
            done[positions] = True
        results = None

    if width > 2:
        # Polynomial self, touching and adaptive pairs share batched stages.
        from ghost_backend.twod.polynomial_quadrature import near_blocks
        remaining = np.flatnonzero(~done)
        blocks = near_blocks([(elements[int(obs_idx[pos])], elements[int(src_idx[pos])]) for pos in remaining],
                             k0, obs_normal_deriv)
        for pos, (s_blk, k_blk) in zip(remaining, blocks):
            if integrate_s:
                s_near[pos] = s_blk
            if want_k:
                k_near[pos] = k_blk
        done[remaining] = True
        blocks = None

    # Self and adaptive linear pairs, one at a time (O(N) self terms plus the
    # few close pairs whose rule adapts).
    for pos in np.flatnonzero(~done):
        s_blk, k_blk = _sk_blocks_near_linear(
            obs_elem=elements[int(obs_idx[pos])],
            src_elem=elements[int(src_idx[pos])],
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_order=obs_order,
            src_order=src_order,
            compute_single_layer=integrate_s,
            compute_double_layer=want_k,
        )
        if s_near is not None:
            s_near[pos] = s_blk
        if k_near is not None:
            k_near[pos] = k_blk
    return s_near, k_near

def _assemble_multi(
    mesh, k0, obs_normal_deriv, source_element_masks, obs_order=8, src_order=8,
    far_ratio=3.0, compute_single_layer=True, compute_double_layer=True,
    compute_double_layer_many=None, single_layer_observation_coefficients=None,
    single_layer_observation_coefficients_many=None, output_node_ids_many=None,
    double_layer_output_node_ids_many=None, operator_outputs=None, prepared_geometry=None,
    additional_operator_outputs=None, minimum_far_order=0,
):
    """Implementation of `_assemble_linear_operator_matrices_multi` (no stage/cache wrappers)."""
    if prepared_geometry is not None:
        prepared_geometry.validate(mesh)
    far_green, far_hankel = _far_green_into, _far_hankel1_into
    if current_state() is not None:
        from ghost_backend.twod.assembly.kernels import select_far_kernels
        far_green, far_hankel = select_far_kernels(mesh, k0, far_green, far_hankel,
            domain_upper=None if prepared_geometry is None else prepared_geometry.domain_upper)
    # Validated kernel tables let the native block quadrature replace the numpy far loop.
    far_table = getattr(far_green, 'table', None)
    native_far = _NATIVE_FAR and far_table is not None

    width = mesh_degree(mesh) + 1
    nnodes = len(mesh.nodes)
    elements = list(mesh.elements) if prepared_geometry is None else prepared_geometry.elements
    nelems = len(elements)
    n_masks = len(source_element_masks)
    if n_masks == 0:
        raise ValueError("At least one source-element mask must be requested.")
    # Regional combined potentials consume source-normal D and Maue W in
    # addition to S/K'. Route these element blocks in the same traversal;
    # do not allocate global operators or re-query 64-row strips.
    if additional_operator_outputs is not None:
        if not obs_normal_deriv or operator_outputs is None or len(additional_operator_outputs) != n_masks:
            raise ValueError('Combined outputs require matching S/K-prime destinations.')
        d_mats = [pair[0] for pair in additional_operator_outputs]
        w_mats = [pair[1] for pair in additional_operator_outputs]
        want_d_masks = [bool(len(o.row_ids) and len(o.column_ids)) for o in d_mats]
        want_w_masks = [bool(len(o.row_ids) and len(o.column_ids)) for o in w_mats]
    else:
        d_mats = w_mats = [None] * n_masks
        want_d_masks = want_w_masks = [False] * n_masks
    want_d, want_w = any(want_d_masks), any(want_w_masks)
    integrate_s = bool(compute_single_layer) or want_w
    if compute_double_layer_many is None:
        want_k_masks = [bool(compute_double_layer)] * n_masks
    else:
        want_k_masks = [bool(value) for value in compute_double_layer_many]
        if len(want_k_masks) != n_masks:
            raise ValueError(
                "compute_double_layer_many length must match "
                "source_element_masks."
            )
    if not integrate_s and not (any(want_k_masks) or want_d):
        raise ValueError("At least one linear operator must be requested.")
    if output_node_ids_many is not None and len(output_node_ids_many) != n_masks:
        raise ValueError("output_node_ids_many length must match source_element_masks.")
    if double_layer_output_node_ids_many is not None and (
            output_node_ids_many is None or len(double_layer_output_node_ids_many) != n_masks):
        raise ValueError('Separate double-layer outputs require matching compact output lists.')


    zero_view = np.broadcast_to(
        np.zeros((), dtype=np.complex128), (nnodes, nnodes)
    )
    if operator_outputs is not None:
        if len(operator_outputs) != n_masks or output_node_ids_many is None:
            raise ValueError('Operator destinations require matching compact output lists.')
        s_mats = [pair[0] for pair in operator_outputs]
        k_mats = [pair[1] for pair in operator_outputs]
    elif output_node_ids_many is None:
        s_mats = [np.zeros((nnodes, nnodes), dtype=np.complex128)
                  if compute_single_layer else zero_view for _ in range(n_masks)]
        k_mats = [np.zeros((nnodes, nnodes), dtype=np.complex128)
                  if want else zero_view for want in want_k_masks]
    else:


        s_mats = [CompactOperator(nnodes, rows, cols, compute_single_layer)
                  for rows, cols in output_node_ids_many]
        k_outputs = output_node_ids_many if double_layer_output_node_ids_many is None else double_layer_output_node_ids_many
        k_mats = [CompactOperator(nnodes, rows, cols, want_k_masks[i])
                  for i, (rows, cols) in enumerate(k_outputs)]
    if not elements:
        return list(zip(s_mats, k_mats))

    src_masks = []
    for mask in source_element_masks:
        if mask is None:
            src_masks.append(np.ones(nelems, dtype=bool))
            continue
        resolved = np.asarray(mask, dtype=bool).reshape(-1)
        if resolved.size != nelems:
            raise ValueError("source_element_mask length must match mesh element count.")
        src_masks.append(resolved)

    if (
        single_layer_observation_coefficients is not None
        and single_layer_observation_coefficients_many is not None
    ):
        raise ValueError(
            "Supply either shared or per-mask single-layer observation "
            "coefficients, not both."
        )

    def _validated_slp_coeff(value):
        if value is None:
            return None
        resolved = np.asarray(value, dtype=np.complex128).reshape(-1)
        if resolved.size != nelems:
            raise ValueError(
                "single_layer_observation_coefficients length must match "
                "mesh element count."
            )
        if not np.all(
            np.isfinite(resolved.real) & np.isfinite(resolved.imag)
        ):
            raise ValueError(
                "single-layer observation coefficients must all be finite."
            )
        return resolved

    if single_layer_observation_coefficients_many is not None:
        supplied_coeffs = list(single_layer_observation_coefficients_many)
        if len(supplied_coeffs) != n_masks:
            raise ValueError(
                "single_layer_observation_coefficients_many length must "
                "match source_element_masks."
            )
        slp_obs_coeffs = [
            _validated_slp_coeff(value) for value in supplied_coeffs
        ]
    else:
        shared_coeff = _validated_slp_coeff(
            single_layer_observation_coefficients
        )
        slp_obs_coeffs = [shared_coeff] * n_masks


    active = [index for index, mask in enumerate(src_masks) if bool(np.any(mask))]
    if not active:
        return list(zip(s_mats, k_mats))

    if prepared_geometry is None:
        centers = np.stack([e.center for e in elements], axis=0)
        lengths = np.asarray([e.length for e in elements], dtype=float)
        node_ids = np.asarray([e.node_ids for e in elements], dtype=int)
    else:
        centers, lengths, node_ids = prepared_geometry.centers, prepared_geometry.lengths, prepared_geometry.node_ids
    obs_masks = [np.ones(nelems, dtype=bool) for _ in range(n_masks)]
    if output_node_ids_many is not None:
        for mi in range(n_masks):
            requested = [s_mats[mi], k_mats[mi]]
            if additional_operator_outputs is not None:
                requested.extend((d_mats[mi], w_mats[mi]))
            if prepared_geometry is not None:
                obs_masks[mi] = prepared_geometry.elements_touching(*(o.row_ids for o in requested))
                src_masks[mi] = src_masks[mi] & prepared_geometry.elements_touching(*(o.column_ids for o in requested))
                continue
            obs_masks[mi] = np.logical_or.reduce([np.any(o.row_map[node_ids] >= 0, axis=1) for o in requested])
            src_masks[mi] = src_masks[mi] & np.logical_or.reduce([np.any(o.column_map[node_ids] >= 0, axis=1) for o in requested])
        active = [mi for mi in active if np.any(src_masks[mi]) and np.any(obs_masks[mi])]
        if not active:
            return list(zip(s_mats, k_mats))
    if prepared_geometry is None:
        p0_arr = np.stack([e.p0 for e in elements], axis=0)
        seg_arr = np.stack([e.p1 - e.p0 for e in elements], axis=0)
        normals_arr = np.stack([e.normal for e in elements], axis=0)
    else:
        p0_arr, seg_arr, normals_arr = prepared_geometry.p0, prepared_geometry.segments, prepared_geometry.normals

    # Scatter interfaces of the requested outputs (system routes, compact or
    # dense storage all take the same tile and pair scatters).
    from ghost_backend.twod.assembly.scatter import scatter_target
    s_out = [scatter_target(s_mats[mi], nnodes) if (compute_single_layer and mi in active) else None
             for mi in range(n_masks)]
    k_out = [scatter_target(k_mats[mi], nnodes) if (want_k_masks[mi] and mi in active) else None
             for mi in range(n_masks)]
    d_out = [scatter_target(d_mats[mi], nnodes) if (want_d_masks[mi] and mi in active) else None
             for mi in range(n_masks)]
    w_out = [scatter_target(w_mats[mi], nnodes) if (want_w_masks[mi] and mi in active) else None
             for mi in range(n_masks)]

    want_s = integrate_s
    want_k = any(want_k_masks) or want_d


    far_obs_order = option('far_quadrature_order', _FAR_QUAD_ORDER) or int(obs_order)
    far_src_order = option('far_quadrature_order', _FAR_QUAD_ORDER) or int(src_order)


    _rule_cache: 'Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = {}
    _rule_lock = threading.Lock()

    def _rule(order: 'int') -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
        cached = _rule_cache.get(order)
        if cached is None:
            nodes, weights = _get_quadrature(max(2, int(order)))
            phi = _polynomial_values(nodes, width - 1)
            cached = (np.asarray(nodes, dtype=float),
                      np.asarray(weights, dtype=float), phi)
            with _rule_lock:
                cached = _rule_cache.setdefault(order, cached)
        return cached


    union_src = np.logical_or.reduce([src_masks[i] for i in active])
    union_obs = np.logical_or.reduce([obs_masks[i] for i in active])
    src_ids, obs_ids_active = np.flatnonzero(union_src), np.flatnonzero(union_obs)
    union_ids = np.flatnonzero(union_src | union_obs)
    matching_rules = int(max(2, far_obs_order)) == int(max(2, far_src_order))
    rectangle = len(src_ids)*len(obs_ids_active)
    symmetric_cost = len(union_ids)**2 * _ASSEMBLY_COMPACT_BELOW
    symmetric = matching_rules and rectangle >= symmetric_cost
    if _ASSEMBLY_COMPACT_BELOW <= 0:
        obs_sel = src_sel = np.arange(nelems)
        symmetric = matching_rules
    elif symmetric:
        obs_sel = src_sel = union_ids
    else:
        obs_sel, src_sel = obs_ids_active, src_ids
    n_obs, n_src = len(obs_sel), len(src_sel)
    src_centers = centers[src_sel]
    src_lengths = lengths[src_sel]
    src_node_ids = node_ids[src_sel]
    src_normals = normals_arr[src_sel]

    abs_k = abs(complex(k0))
    attenuating = -complex(k0).imag > complex(k0).real
    attenuation_rate = (-complex(k0).imag if FAR_ATTENUATION_CUT > 1.0 and complex(k0).imag < 0.0
                        and complex(k0).real > 0.0 else 0.0)
    w_far_floor = _graded_w_far_floor(k0, lengths, far_ratio, width-1) if want_w else 16

    n_acc = width**2 * (int(want_s) + (2 if want_k else 0))
    # Partial sums per source basis function, alongside the accumulators.
    n_acc += width * (int(want_s) + (2 if want_k else 0))
    if want_w:
        n_acc += 2 * width**2
    n_kernel = int(want_s) + (2 if want_k else 0)
    tile = _assembly_tile_size(nelems, 16 * (n_acc + n_kernel + 1) + 8 * 7)

    real_k = _wavenumber_is_real(k0)

    dgreen_sign = -1.0 if obs_normal_deriv else 1.0
    derivative = derivative_matrix(width - 1)

    tiles = [(i0, j0) for i0 in range(0, n_obs, tile)
             for j0 in range(i0 if symmetric else 0, n_src, tile)]
    near_records: 'List[Optional[List[Tuple[np.ndarray, np.ndarray, int, np.ndarray, bool]]]]' = [None] * len(tiles)

    @assembly_component('far_integration')
    def _far_tile(index: 'int'):
        i0, j0 = tiles[index]
        i1 = min(i0 + tile, n_obs)
        j1 = min(j0 + tile, n_src)
        mb, nb = i1 - i0, j1 - j0
        obs_slice = obs_sel[i0:i1]
        obs_nid = node_ids[obs_slice]
        obs_norm = normals_arr[obs_slice]
        obs_len = lengths[obs_slice]
        obs_p0 = p0_arr[obs_slice]
        obs_seg = seg_arr[obs_slice]
        obs_ctr = centers[obs_slice]
        mirrored = symmetric and j0 > i0
        src_slice = slice(j0, j1)
        src_global = src_sel[src_slice]
        src_nid = src_node_ids[src_slice]
        src_len = src_lengths[src_slice]
        src_norm = src_normals[src_slice]

        mdx = obs_ctr[:, 0][:, None] - src_centers[src_slice, 0][None, :]
        mdy = obs_ctr[:, 1][:, None] - src_centers[src_slice, 1][None, :]
        centre_dist = np.sqrt(mdx * mdx + mdy * mdy)
        scale = np.maximum(np.maximum(obs_len[:, None], src_len[None, :]), EPS)
        far_sym = (centre_dist / scale) >= float(far_ratio)
        far_sym &= ~(
            (obs_nid[:, 0][:, None] == src_nid[None, :, 0])
            | (obs_nid[:, 0][:, None] == src_nid[None, :, 1])
            | (obs_nid[:, 1][:, None] == src_nid[None, :, 0])
            | (obs_nid[:, 1][:, None] == src_nid[None, :, 1])
        )


        np.logical_and(
            far_sym,
            obs_slice[:, None] != src_global[None, :],
            out=far_sym,
        )
        # Far pairs whose kernels are negligible (FAR_ATTENUATION_CUT) are
        # neither evaluated nor scattered; they stay far for the near
        # classification and for the graded order below.
        far_eval = far_sym
        if attenuation_rate > 0.0:
            lower = centre_dist - 0.5 * (obs_len[:, None] + src_len[None, :])
            far_eval = far_sym & (attenuation_rate * lower < FAR_ATTENUATION_CUT)


        far_ij = {}
        far_ji = {}
        any_ij = False
        any_ji = False
        eligible_ij = np.zeros_like(far_sym)
        eligible_ji = np.zeros_like(far_sym)
        for mi in active:
            mask = src_masks[mi]
            src_msk = mask[src_global]
            pairs_ij = src_msk[None, :] & obs_masks[mi][obs_slice][:, None]
            eligible_ij |= pairs_ij
            fij = far_eval & pairs_ij
            far_ij[mi] = fij
            any_ij = any_ij or bool(fij.any())
            if mirrored:
                obs_msk = mask[obs_slice]
                pairs_ji = obs_msk[:, None] & obs_masks[mi][src_global][None, :]
                eligible_ji |= pairs_ji
                fji = far_eval & pairs_ji
                far_ji[mi] = fji
                any_ji = any_ji or bool(fji.any())

        local_near = []
        near_union = (~far_sym) & eligible_ij
        flat = np.flatnonzero(near_union.ravel())
        if flat.size:
            local_near.append((obs_slice, src_global, nb, flat, False))
        if mirrored:
            near_union_t = (~far_sym) & eligible_ji
            flat_t = np.flatnonzero(near_union_t.ravel())
            if flat_t.size:
                local_near.append((obs_slice, src_global, nb, flat_t, True))
        near_records[index] = local_near

        if not (any_ij or any_ji):
            return None


        any_far = far_sym & (eligible_ij | eligible_ji)
        ratio_min = float(np.min(
            np.where(any_far, centre_dist / scale, np.inf)
        )) if any_far.any() else float("inf")
        if far_eval is not far_sym:
            any_far = far_eval & (eligible_ij | eligible_ji)
        kl_max = abs_k * float(max(obs_len.max(), src_len.max()))
        rule_floor = max(int(minimum_far_order), width + 2 if width > 2 else 2)
        tile_obs_order = max(rule_floor,
            _graded_far_order(kl_max, ratio_min, far_obs_order, width - 1, attenuating))
        tile_src_order = max(rule_floor,
            _graded_far_order(kl_max, ratio_min, far_src_order, width - 1, attenuating))
        if want_w:
            # Preserve W's independently qualified rule on both axes.
            tile_obs_order = tile_src_order = max(tile_obs_order, tile_src_order,
                w_far_floor, int(obs_order), int(src_order))
        t_obs_f, qw_obs, phi_obs_arr = _rule(tile_obs_order)
        t_src_f, qw_src, phi_src_arr = _rule(tile_src_order)

        obs_pts = (obs_p0[:, None, :]
                   + t_obs_f[None, :, None] * obs_seg[:, None, :])
        src_p0 = p0_arr[src_global]
        src_seg = seg_arr[src_global]
        src_pts = src_p0[:, None, :] + t_src_f[None, :, None] * src_seg[:, None, :]
        native = None
        # The native ABI currently takes one quadrature rule for both axes.
        # Unequal rules use the exact same kernels in the two-rule NumPy path.
        if native_far and tile_obs_order == tile_src_order:
            from ghost_backend.twod.assembly.native.far import far_block
            native = far_block(far_table, k0, obs_pts, src_pts, qw_obs, phi_obs_arr,
                               obs_norm, src_norm, any_far, obs_normal_deriv,
                               want_s, want_k, mirrored or want_d)
        if native is not None:
            acc_s, acc_k, acc_kt = native
        else:
            acc_s, acc_k, acc_kt = _far_tile_numpy(
                far_green, far_hankel, k0, real_k, obs_pts, src_pts, qw_obs, qw_src,
                phi_obs_arr, phi_src_arr, obs_norm, src_norm, obs_normal_deriv, dgreen_sign,
                width, want_s, want_k, mirrored or want_d)

        len_prod = obs_len[:, None] * src_len[None, :]
        acc_w = None
        if want_w:
            blocks = acc_s.reshape(width, width, mb, nb)
            normal_dot = obs_norm[:, 0, None]*src_norm[None, :, 0] + obs_norm[:, 1, None]*src_norm[None, :, 1]
            acc_w = (np.einsum('ia,ijmn,jb->abmn', derivative, blocks, derivative)
                     - complex(k0)**2 * normal_dot * len_prod * blocks).reshape(width**2, mb, nb)
        # Per-mask scale factors are formed here, on the worker; the ordered
        # commit only multiplies them into the scatter entry by entry.
        scatters = []
        for mi in active:
            fij = far_ij[mi]
            if fij.any():
                scale_ij = len_prod * fij
                slp_obs_coeff = slp_obs_coeffs[mi]
                scale_s_ij = (scale_ij * slp_obs_coeff[obs_slice][:, None]
                              if slp_obs_coeff is not None else scale_ij)
                mask_ij = fij.astype(float) if w_out[mi] is not None else None
                for a in range(width):
                    rows = obs_nid[:, a]
                    if s_out[mi] is not None and acc_s is not None:
                        scatters.append((s_out[mi], rows, src_nid, acc_s[width * a:width * (a + 1)], scale_s_ij))
                    if k_out[mi] is not None and acc_k is not None:
                        scatters.append((k_out[mi], rows, src_nid, acc_k[width * a:width * (a + 1)], scale_ij))
                    if d_out[mi] is not None:
                        scatters.append((d_out[mi], rows, src_nid, acc_kt[a::width], scale_ij))
                    if w_out[mi] is not None:
                        scatters.append((w_out[mi], rows, src_nid, acc_w[width*a:width*(a+1)], mask_ij))
            fji = far_ji.get(mi)
            if fji is not None and fji.any():
                scale_ji = len_prod * fji
                slp_obs_coeff = slp_obs_coeffs[mi]
                scale_s_ji = (scale_ji * slp_obs_coeff[src_global][None, :]
                              if slp_obs_coeff is not None else scale_ji)
                mask_ji = fji.astype(float) if w_out[mi] is not None else None
                for a in range(width):
                    rows = src_nid[:, a]
                    if s_out[mi] is not None and acc_s is not None:
                        scatters.append((s_out[mi], rows, obs_nid,
                                         acc_s[a::width].transpose(0, 2, 1), scale_s_ji.T))
                    if k_out[mi] is not None and acc_kt is not None:
                        scatters.append((k_out[mi], rows, obs_nid,
                                         acc_kt[width * a:width * (a + 1)].transpose(0, 2, 1), scale_ji.T))
                    if d_out[mi] is not None:
                        scatters.append((d_out[mi], rows, obs_nid,
                                         acc_k[a::width].transpose(0, 2, 1), scale_ji.T))
                    if w_out[mi] is not None:
                        scatters.append((w_out[mi], rows, obs_nid,
                                         acc_w[a::width].transpose(0, 2, 1), mask_ji.T))
        if not scatters:
            return None

        @assembly_component('far_scatter')
        def commit():
            for target, rows, columns, values, factor in scatters:
                target.scatter_tile(rows, columns, values, factor)
        return commit

    _run_ordered_tiles(len(tiles), _far_tile)


    obs_idx, src_idx = _expand_near_chunks(
        [chunk for records in near_records if records for chunk in records])
    near_records = None
    npairs = int(obs_idx.size)
    if not npairs:
        return list(zip(s_mats, k_mats))
    from ghost_backend.twod.assembly.near_store import NearStore, NEAR_INTEGRATION_PAIRS
    from ghost_backend.twod.polynomial_quadrature import solve_checkpoint
    kinds = (['S'] if integrate_s else []) + (['K'] if want_k else []) + (['D'] if want_d else [])
    near_geometry = (p0_arr, np.asarray([e.p1 for e in elements], float), seg_arr, lengths,
                     np.asarray([e.normal for e in elements], float)) if width == 2 else None
    with NearStore(npairs, width, kinds, solve_checkpoint()) as near:
        for start in range(0, npairs, NEAR_INTEGRATION_PAIRS):
            stop = min(start + NEAR_INTEGRATION_PAIRS, npairs)
            sb, kb = _near_pair_blocks(elements, obs_idx[start:stop], src_idx[start:stop],
                k0, obs_normal_deriv, obs_order, src_order, integrate_s, want_k,
                p0_arr, seg_arr, centers, lengths, node_ids, far_table,
                single_layer_blocks=near.arrays['S'][start:stop] if 'S' in near.arrays else None,
                double_layer_blocks=near.arrays['K'][start:stop] if 'K' in near.arrays else None,
                prepared_geometry=near_geometry)
            if sb is not None:
                near.write('S', start, sb)
            if kb is not None:
                near.write('K', start, kb)
            sb = kb = None

        # D(x,y) = K'(y,x). Retain the same reverse-pair reuse, but read only
        # a bounded batch of records; missing reverse pairs keep their own rule.
        if want_d:
            keys = obs_idx.astype(np.int64) * nelems + src_idx
            for start in range(0, npairs, NEAR_INTEGRATION_PAIRS):
                stop = min(start + NEAR_INTEGRATION_PAIRS, npairs)
                oi, si = obs_idx[start:stop], src_idx[start:stop]
                needs = np.zeros(stop-start, bool)
                for mi in active:
                    if want_d_masks[mi]:
                        needs |= src_masks[mi][si] & obs_masks[mi][oi]
                reverse_keys = si.astype(np.int64)*nelems + oi
                location = np.minimum(np.searchsorted(keys, reverse_keys), npairs-1)
                present = needs & (keys[location] == reverse_keys)
                values = np.zeros((stop-start, width, width), complex)
                values[present] = near.read('K', location[present]).transpose(0, 2, 1)
                missing = np.flatnonzero(needs & ~present)
                if width > 2 and len(missing):
                    from ghost_backend.twod.polynomial_quadrature import near_blocks
                    for pos, (_, block) in zip(missing, near_blocks(
                            [(elements[int(oi[p])], elements[int(si[p])]) for p in missing], k0, False)):
                        values[pos] = block
                else:
                    for pos in missing:
                        values[pos] = _sk_blocks_near_linear(elements[int(oi[pos])], elements[int(si[pos])],
                            k0, False, obs_order, src_order, compute_single_layer=False, compute_double_layer=True)[1]
                near.write('D', start, values)
                values = None
            keys = None

        # Keep the previous destination -> S/K'/D/W -> ascending pair order.
        # Interleaving operators by chunk could change a fused matrix's sums.
        def scatter_chunks(pairs):
            for start in range(0, len(pairs), _NEAR_SCATTER_MAX_PAIRS):
                selected = pairs[start:start + _NEAR_SCATTER_MAX_PAIRS]
                yield selected, node_ids[obs_idx[selected]], node_ids[src_idx[selected]]

        for mi in active:
            pairs = np.flatnonzero(src_masks[mi][src_idx] & obs_masks[mi][obs_idx])
            if not len(pairs):
                continue
            if s_out[mi] is not None:
                slp_obs_coeff = slp_obs_coeffs[mi]
                for selected, rows, columns in scatter_chunks(pairs):
                    values = near.read('S', selected)
                    if slp_obs_coeff is not None:
                        np.multiply(slp_obs_coeff[obs_idx[selected]][:, None, None], values, out=values)
                    s_out[mi].scatter_pairs(rows, columns, values)
            if k_out[mi] is not None:
                for selected, rows, columns in scatter_chunks(pairs):
                    k_out[mi].scatter_pairs(rows, columns, near.read('K', selected))
            if d_out[mi] is not None:
                for selected, rows, columns in scatter_chunks(pairs):
                    d_out[mi].scatter_pairs(rows, columns, near.read('D', selected))
            if w_out[mi] is not None:
                for selected, rows, columns in scatter_chunks(pairs):
                    oi, si = obs_idx[selected], src_idx[selected]
                    values = _maue_blocks(near.read('S', selected), k0, normals_arr[oi], normals_arr[si],
                                         lengths[oi], lengths[si])
                    w_out[mi].scatter_pairs(rows, columns, values)
    return list(zip(s_mats, k_mats))

def _far_tile_numpy(far_green, far_hankel, k0, real_k, obs_pts, src_pts, qw_obs, qw_src,
                    phi_obs_arr, phi_src_arr, obs_norm, src_norm, obs_normal_deriv, dgreen_sign,
                    width, want_s, want_k, with_kt):
    """NumPy far-tile accumulators (width**2, mb, nb), the native block's reference."""
    mb, nb = obs_pts.shape[0], src_pts.shape[0]
    acc_s = (
        np.zeros((width**2, mb, nb), dtype=np.complex128)
        if want_s else None
    )
    acc_k = (
        np.zeros((width**2, mb, nb), dtype=np.complex128)
        if want_k else None
    )
    acc_kt = (
        np.zeros((width**2, mb, nb), dtype=np.complex128)
        if (want_k and with_kt) else None
    )
    # One partial sum per source basis function. The coefficient
    # w*phi_o[a]*phi_s[b] is separable, so the source-quadrature loop
    # can accumulate phi_s[b] alone and pay the phi_o[a] fold-in once
    # per observation node instead of once per quadrature pair. That
    # turns width**2 accumulations per pair into width.
    part_s = (
        [np.empty((mb, nb), dtype=np.complex128) for _ in range(width)]
        if want_s else None
    )
    part_k = (
        [np.empty((mb, nb), dtype=np.complex128) for _ in range(width)]
        if want_k else None
    )
    part_kt = (
        [np.empty((mb, nb), dtype=np.complex128) for _ in range(width)]
        if acc_kt is not None else None
    )
    g_buf = np.empty((mb, nb), dtype=np.complex128) if want_s else None
    h1_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
    dk_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
    cscratch = np.empty((mb, nb), dtype=np.complex128)
    dx = np.empty((mb, nb), dtype=float)
    dy = np.empty((mb, nb), dtype=float)
    dist = np.empty((mb, nb), dtype=float)
    krbuf = np.empty((mb, nb), dtype=float)
    work = np.empty((mb, nb), dtype=float)
    proj = np.empty((mb, nb), dtype=float) if want_k else None


    if obs_normal_deriv:
        n_ij = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])
        n_ji = (src_norm[None, :, 0], src_norm[None, :, 1])
    else:
        n_ij = (src_norm[None, :, 0], src_norm[None, :, 1])
        n_ji = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])

    for qi in range(len(qw_obs)):
        r_obs = obs_pts[:, qi, :]
        w_obs_qi = float(qw_obs[qi])
        phi_o = phi_obs_arr[qi]

        for qj in range(len(qw_src)):
            r_src = src_pts[:, qj, :]
            w_src_qj = float(qw_src[qj])
            phi_s = phi_src_arr[qj]

            np.subtract(r_obs[:, 0][:, None], r_src[None, :, 0], out=dx)
            np.subtract(r_obs[:, 1][:, None], r_src[None, :, 1], out=dy)
            np.multiply(dx, dx, out=dist)
            np.multiply(dy, dy, out=work)
            np.add(dist, work, out=dist)
            np.sqrt(dist, out=dist)
            np.maximum(dist, EPS, out=dist)
            if real_k:
                _far_kernel_argument(k0, dist, krbuf)

            if want_s and want_k and hasattr(far_green, 'pair'):
                far_green.pair(k0, real_k, dist, krbuf, work, g_buf, h1_buf)
            else:
                if want_s:
                    far_green(k0, real_k, dist, krbuf, work, g_buf)
                if want_k:
                    far_hankel(k0, real_k, dist, krbuf, work, h1_buf)

            if want_k:
                np.multiply(dx, n_ij[0], out=proj)
                np.multiply(dy, n_ij[1], out=work)
                np.add(proj, work, out=proj)
                np.divide(proj, dist, out=proj)
                np.multiply(h1_buf, proj, out=dk_buf)
                if dgreen_sign < 0.0:
                    np.negative(dk_buf, out=dk_buf)
            for b in range(width):
                coeff_b = w_src_qj * float(phi_s[b])
                if part_s is not None:
                    _accumulate_first(part_s[b], g_buf, coeff_b, qj, cscratch)
                if part_k is not None:
                    _accumulate_first(part_k[b], dk_buf, coeff_b, qj, cscratch)

            if acc_kt is not None:


                np.multiply(dx, n_ji[0], out=proj)
                np.multiply(dy, n_ji[1], out=work)
                np.add(proj, work, out=proj)
                np.divide(proj, dist, out=proj)
                np.multiply(h1_buf, proj, out=dk_buf)
                if dgreen_sign > 0.0:
                    np.negative(dk_buf, out=dk_buf)
                for a in range(width):
                    _accumulate_first(
                        part_kt[a], dk_buf, w_src_qj * float(phi_s[a]),
                        qj, cscratch,
                    )

        for a in range(width):
            coeff_a = w_obs_qi * float(phi_o[a])
            for b in range(width):
                if acc_s is not None:
                    _axpy_into(acc_s[width * a + b], part_s[b], coeff_a, cscratch)
                if acc_k is not None:
                    _axpy_into(acc_k[width * a + b], part_k[b], coeff_a, cscratch)
        if acc_kt is not None:
            for b in range(width):
                coeff_b = w_obs_qi * float(phi_o[b])
                for a in range(width):
                    _axpy_into(acc_kt[width * a + b], part_kt[a], coeff_b, cscratch)
    return acc_s, acc_k, acc_kt
def _assemble_linear_operator_matrices(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
    single_layer_destination=None,
    double_layer_destination=None,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Assemble dense linear-Galerkin S and K/K' matrices on global nodal DOFs.

    ``compute_single_layer`` and ``compute_double_layer`` let formulation
    callers skip an operator they do not consume.  A zero matrix is returned
    for a skipped operator so the long-standing two-array return contract is
    preserved.

    Single-mask front end for `_assemble_linear_operator_matrices_multi`; a
    caller wanting several masks at one wavenumber should use that directly
    so the quadrature is shared.
    """

    from ghost_backend.twod.assembly.scatter import MatrixDestination, SystemScatter
    n = len(mesh.nodes)
    ids, empty = np.arange(n), np.empty(0, int)
    matrices, destinations, node_lists = [], [], []
    for enabled, target in ((compute_single_layer, single_layer_destination),
                            (compute_double_layer, double_layer_destination)):
        if enabled:
            matrix = np.zeros((n, n), complex, order='F') if target is None else target
            destination = MatrixDestination(matrix, n)
        else:
            if target is not None:
                raise ValueError('A disabled operator cannot have a destination.')
            matrix = np.broadcast_to(np.zeros((), complex), (n, n))
            destination = SystemScatter(matrix, n, empty, ids, [])
        matrices.append(matrix)
        destinations.append(destination)
        node_lists.append((ids if enabled else empty, ids))
    _assemble_linear_operator_matrices_multi(
        mesh=mesh,
        k0=k0,
        obs_normal_deriv=obs_normal_deriv,
        source_element_masks=[source_element_mask],
        obs_order=obs_order,
        src_order=src_order,
        far_ratio=far_ratio,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
        single_layer_observation_coefficients=(
            single_layer_observation_coefficients
        ),
        output_node_ids_many=[node_lists[0]],
        double_layer_output_node_ids_many=[node_lists[1]],
        operator_outputs=[tuple(destinations)],
    )
    return tuple(matrices)

@timed_stage("near_and_hypersingular")
@cached_operator("D")
def _assemble_linear_hypersingular_matrix(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
    destination=None,
    output_node_ids=None,
    prepared_geometry=None,
) -> 'np.ndarray':
    """
    Assemble the hypersingular operator W via the Maue identity.

    W is computed element-by-element from single-layer S blocks:
        W_block = -k^2 (n_obs . n_src) S_block
                + tangent_outer / (L_obs * L_src) * sum(S_block)

    This avoids all hypersingular quadrature; the log singularity in S is handled
    by the self series and the graded touching rule.

    The blocks come from the fused engine in a W-only pass: its S far tiles
    (graded Gauss orders with the calibrated W floor, native accumulation when a
    validated kernel table is available, tile scatters in a fixed order) and its
    near pass (self, touching, fixed-rule and adaptive pairs). The fused S/K'/D/W
    assembly therefore produces the same W. ``destination`` receives W added to
    its contents; ``output_node_ids`` returns rectangular CompactOperator storage.
    """
    from ghost_backend.twod.assembly.scatter import MatrixDestination, SystemScatter
    nnodes = len(mesh.nodes)
    if output_node_ids is not None:
        if destination is not None:
            raise ValueError('Choose a compact hypersingular query or a dense destination.')
        d_mat = CompactOperator(nnodes, *output_node_ids)
        output = d_mat
    elif destination is None:
        d_mat = np.zeros((nnodes, nnodes), dtype=np.complex128, order='F')
        output = MatrixDestination(d_mat, nnodes)
    else:
        output = MatrixDestination(destination, nnodes)
        d_mat = output.matrix
    if prepared_geometry is not None:
        prepared_geometry.validate(mesh)
    if not mesh.elements:
        return d_mat
    if source_element_mask is not None:
        mask = np.asarray(source_element_mask, dtype=bool).reshape(-1)
        if mask.size != len(mesh.elements):
            raise ValueError("source_element_mask length must match mesh element count.")
        if not np.any(mask):
            return d_mat
    storage = np.zeros((0, 0), dtype=np.complex128)
    empty = np.empty(0, dtype=np.int64)
    unused = SystemScatter(storage, nnodes, empty, empty, [])
    _assemble_multi(
        mesh, k0, True, [source_element_mask], obs_order=obs_order, src_order=src_order,
        far_ratio=far_ratio, compute_single_layer=False, compute_double_layer=False,
        output_node_ids_many=[(empty, empty)], operator_outputs=[(unused, unused)],
        prepared_geometry=prepared_geometry, additional_operator_outputs=[(unused, output)])
    return d_mat


@timed_stage("excitation")
def _linear_element_incident_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    if len(elem.node_ids) > 2:
        from ghost_backend.twod.assembly.kernels import incident
        return incident(elem, k_air, elevations_deg, order)
    if current_state() is not None:
        from ghost_backend.twod.assembly.kernels import incident
        return incident(elem, k_air, elevations_deg, order)
    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * phase[None, :]
    return out * float(elem.length)

@timed_stage("excitation")
def _linear_element_incident_dn_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    """
    Galerkin-tested normal derivative of the incident plane wave on one element.

    du_inc/dn = j*k*(d_inc . n) * exp(j*k*d_inc . r)

    Used by TE sheet and impedance/flux right-hand sides.
    """
    if len(elem.node_ids) > 2:
        from ghost_backend.twod.assembly.kernels import incident_dn
        return incident_dn(elem, k_air, elevations_deg, order)
    if current_state() is not None:
        from ghost_backend.twod.assembly.kernels import incident_dn
        return incident_dn(elem, k_air, elevations_deg, order)

    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)

    d_dot_n = dirs @ np.asarray(elem.normal, dtype=float)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * (1j * k_air * d_dot_n * phase)[None, :]
    return out * float(elem.length)


@timed_stage("far_field")
def _farfield_linear_density_many(
    mesh: 'LinearMesh',
    density: 'np.ndarray',
    k_air: 'float',
    observation_angles_deg: 'np.ndarray',
    potential: 'str',
    order: 'int' = 8,
    element_mask: 'Optional[np.ndarray]' = None,
    projection: 'str' = "matched",
    prepared_projection=None,
) -> 'np.ndarray':
    """Vectorized SLP/DLP far field for matched or rectangular projections.

    ``density`` may contain one column (one incidence projected at every
    observation angle) or one column per observation angle (the monostatic
    batched-solve case). With ``projection='grid'``, every density column is
    projected at every observation angle and the result has shape
    ``(density_columns, observation_angles)``. Element tiling bounds the
    temporary phase matrix in either mode.
    """
    if mesh_degree(mesh) > 1:
        from ghost_backend.twod.assembly.kernels import farfield
        return farfield(mesh, density, k_air, observation_angles_deg, potential, order, element_mask, projection,
                        prepared_projection=prepared_projection)
    if current_state() is not None:
        from ghost_backend.twod.assembly.kernels import farfield
        return farfield(mesh, density, k_air, observation_angles_deg, potential, order, element_mask, projection,
                        prepared_projection=prepared_projection)

    obs = np.asarray(observation_angles_deg, dtype=float).reshape(-1)
    rho = np.asarray(density, dtype=np.complex128)
    if rho.ndim == 1:
        rho = rho[:, None]
    if rho.shape[0] != len(mesh.nodes):
        raise ValueError("Far-field density height must match mesh node count.")
    projection_mode = str(projection).strip().lower()
    if projection_mode not in {"matched", "grid"}:
        raise ValueError("Far-field projection must be 'matched' or 'grid'.")
    if projection_mode == "matched" and rho.shape[1] not in (1, obs.size):
        raise ValueError(
            "Far-field density must have one column or one per observation angle."
        )
    kind = str(potential).strip().upper()
    if kind not in {"SLP", "DLP"}:
        raise ValueError("Far-field potential must be 'SLP' or 'DLP'.")

    if element_mask is None:
        elements = list(mesh.elements)
    else:
        mask = np.asarray(element_mask, dtype=bool).reshape(-1)
        if mask.size != len(mesh.elements):
            raise ValueError("Far-field element mask must match mesh element count.")
        elements = [elem for elem, keep in zip(mesh.elements, mask) if keep]
    if not elements:
        shape = (rho.shape[1], obs.size) if projection_mode == "grid" else (obs.size,)
        return np.zeros(shape, dtype=np.complex128)
    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    wq = np.asarray(qw, dtype=float)
    phi_q = np.column_stack((1.0 - q, q))
    dirs = np.column_stack((
        np.cos(np.deg2rad(obs)), np.sin(np.deg2rad(obs))
    ))
    node_ids = np.asarray([elem.node_ids for elem in elements], dtype=int)
    p0 = np.asarray([elem.p0 for elem in elements], dtype=float)
    seg = np.asarray([elem.p1 - elem.p0 for elem in elements], dtype=float)
    lengths = np.asarray([elem.length for elem in elements], dtype=float)
    normals = np.asarray([elem.normal for elem in elements], dtype=float)

    amp = (
        np.zeros((rho.shape[1], obs.size), dtype=np.complex128)
        if projection_mode == "grid"
        else np.zeros(obs.size, dtype=np.complex128)
    )
    phase_entries = 2_000_000
    tile = max(1, min(
        len(elements), phase_entries // max(1, obs.size * q.size)
    ))
    for start in range(0, len(elements), tile):
        stop = min(start + tile, len(elements))
        pts = (
            p0[start:stop, None, :]
            + q[None, :, None] * seg[start:stop, None, :]
        )
        phase = np.exp(
            1j * float(k_air) * np.einsum('ad,eqd->aeq', dirs, pts)
        )
        local = rho[node_ids[start:stop], :]
        rho_q = np.einsum('qi,eic->eqc', phi_q, local)
        weights = lengths[start:stop, None] * wq[None, :]
        if kind == "DLP":
            dot_n = dirs @ normals[start:stop].T
            phase *= (1j * float(k_air)) * dot_n[:, :, None]
        if projection_mode == "grid":
            amp += np.einsum(
                'aeq,eqc,eq->ca', phase, rho_q, weights
            )
        elif rho.shape[1] == 1:
            amp += np.einsum(
                'aeq,eq,eq->a', phase, rho_q[:, :, 0], weights
            )
        else:
            amp += np.einsum(
                'aeq,eqa,eq->a', phase, rho_q, weights
            )
    return amp

def _linear_mass_block(elem: 'LinearElement') -> 'np.ndarray':
    """Consistent 2-node boundary mass matrix on one straight element."""

    if len(elem.node_ids) > 2:
        from ghost_backend.twod.basis import mass_block
        return mass_block(elem).astype(complex)
    l = float(elem.length)
    return l * np.asarray([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=np.complex128)

def _linear_coupled_interface_signature(elem: 'LinearElement', info: 'PanelCoupledInfo') -> 'Tuple[Any, ...]':
    return (
        int(elem.seg_type),
        int(elem.ibc_flag),
        int(elem.pos_mat),
        int(elem.neg_mat),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )

def _linear_coupled_node_report(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
) -> 'Dict[str, int]':
    """
    Summarize node configurations for the nodal coupled solve.

    The linear/Galerkin path handles shared geometric junctions by
    augmenting the nodal system with trace-continuity and region-wise flux-balance rows.
    Branching and mixed-interface node counts are reported for diagnostics; they are
    not automatic blockers by themselves.
    """

    incident: 'Dict[int, List[int]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        for nid in elem.node_ids:
            incident.setdefault(int(nid), []).append(int(eidx))

    branching_nodes = 0
    mixed_interface_nodes = 0
    for nid, elem_ids in incident.items():
        unique = sorted(set(int(v) for v in elem_ids))
        if len(unique) <= 1:
            continue
        sigs = {
            _linear_coupled_interface_signature(mesh.elements[eidx], infos[eidx])
            for eidx in unique
        }
        if len(unique) > 2:
            branching_nodes += 1
        if len(sigs) > 1:
            mixed_interface_nodes += 1

    return {
        "linear_node_count": int(len(mesh.nodes)),
        "linear_element_count": int(len(mesh.elements)),
        "linear_branching_nodes": int(branching_nodes),
        "linear_mixed_interface_nodes": int(mixed_interface_nodes),
        "linear_unsupported_nodes": 0,
    }

def _build_linear_junction_constraints(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    materialize: 'bool' = True,
) -> 'Tuple[np.ndarray, Dict[str, int]]':
    """
    Build nodal junction constraints for the linear/Galerkin coupled solve.

    The linear trace unknown is continuous only across explicitly shared nodes. When the
    interface-aware mesh intentionally splits nodes at the same geometric coordinate, we
    restore pointwise continuity at true shared geometric junctions with explicit trace
    constraints. We also add region-wise flux-balance constraints using the endpoint sign
    convention. When ``materialize`` is false, compute the same candidate counts and
    orientation diagnostics without allocating the dense trace/flux matrix.
    """

    nnodes = len(mesh.nodes)
    grouped: 'Dict[Tuple[int, int], List[Tuple[int, int, int]]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        n0, n1 = (int(v) for v in elem.node_ids[:2])
        grouped.setdefault(mesh.nodes[n0].key, []).append((int(eidx), 0, n0))
        grouped.setdefault(mesh.nodes[n1].key, []).append((int(eidx), 1, n1))

    rows: 'List[np.ndarray]' = []
    trace_count = 0
    flux_count = 0
    junction_nodes = 0
    orientation_conflict_nodes = 0
    constrained_nodes: 'Set[int]' = set()
    constrained_elems: 'Set[int]' = set()

    for entries in grouped.values():
        unique_elems = sorted({int(eidx) for eidx, _, _ in entries})
        unique_nodes = sorted({int(nid) for _, _, nid in entries})
        if len(unique_elems) < 2 and len(unique_nodes) < 2:
            continue

        by_elem_sign: 'Dict[int, int]' = {}
        seg_names: 'Set[str]' = set()
        region_set: 'Set[int]' = set()
        for eidx, local_end, nid in entries:
            endpoint_sign = +1 if int(local_end) == 0 else -1
            by_elem_sign[int(eidx)] = by_elem_sign.get(int(eidx), 0) + endpoint_sign
            seg_names.add(mesh.elements[int(eidx)].name)
            info = infos[int(eidx)]
            if info.minus_region >= 0:
                region_set.add(int(info.minus_region))
            if info.plus_region >= 0:
                region_set.add(int(info.plus_region))

        if len(seg_names) >= 2:
            signs = [int(np.sign(by_elem_sign.get(eidx, 0))) for eidx in unique_elems]
            has_pos = any(s > 0 for s in signs)
            has_neg = any(s < 0 for s in signs)
            if not (has_pos and has_neg):
                orientation_conflict_nodes += 1

        if len(unique_nodes) > 1:
            ref_nid = unique_nodes[0]
            for other_nid in unique_nodes[1:]:
                if materialize:
                    row = np.zeros(2 * nnodes, dtype=np.complex128)
                    row[ref_nid] = 1.0 + 0.0j
                    row[other_nid] = -1.0 + 0.0j
                    rows.append(row)
                trace_count += 1
                constrained_nodes.add(ref_nid)
                constrained_nodes.add(other_nid)

        for region in sorted(region_set):
            sparse_row: 'Dict[int, complex]' = {}
            terms = 0
            for eidx, local_end, nid in entries:
                endpoint_sign = +1 if int(local_end) == 0 else -1
                info = infos[int(eidx)]
                coeff_u = 0.0 + 0.0j
                coeff_q = 0.0 + 0.0j
                participates = False
                if info.minus_region == region:
                    coeff_q += 1.0 + 0.0j
                    participates = True
                if info.plus_region == region:
                    coeff_u += complex(info.q_plus_gamma)
                    coeff_q += complex(info.q_plus_beta)
                    participates = True
                if not participates:
                    continue

                w = complex(float(endpoint_sign), 0.0)
                nid_i = int(nid)
                sparse_row[nid_i] = sparse_row.get(nid_i, 0.0j) + w * coeff_u
                flux_index = nnodes + nid_i
                sparse_row[flux_index] = (
                    sparse_row.get(flux_index, 0.0j) + w * coeff_q
                )
                terms += 1
                constrained_nodes.add(nid_i)
                constrained_elems.add(int(eidx))

            row_norm_sq = sum(abs(value) ** 2 for value in sparse_row.values())
            if terms >= 2 and row_norm_sq > 0.0:
                if materialize:
                    row = np.zeros(2 * nnodes, dtype=np.complex128)
                    for index, value in sparse_row.items():
                        row[index] = value
                    rows.append(row)
                flux_count += 1

        junction_nodes += 1

    constraint_count = int(trace_count + flux_count)
    if constraint_count == 0:
        return np.zeros((0, 2 * nnodes), dtype=np.complex128), {
            "junction_nodes": 0,
            "junction_constraints": 0,
            "junction_panels": 0,
            "junction_trace_constraints": 0,
            "junction_flux_constraints": 0,
            "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
        }

    c_mat = (
        np.vstack(rows)
        if materialize
        else np.zeros((0, 0), dtype=np.complex128)
    )
    return c_mat, {
        "junction_nodes": int(junction_nodes),
        "junction_constraints": constraint_count,
        "junction_panels": int(len(constrained_elems)),
        "junction_trace_constraints": int(trace_count),
        "junction_flux_constraints": int(flux_count),
        "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
    }

def _ensure_finite_linear_system(a_mat: 'np.ndarray', rhs: 'Optional[np.ndarray]' = None, label: 'str' = "linear system") -> 'None':
    """Raise a clear error before calling LAPACK if the assembled system contains NaN/Inf."""

    a_eval = np.asarray(a_mat)
    first = first_nonfinite(a_eval)
    if first is not None:
        raise ValueError(f"{label}: system matrix contains NaN/Inf at index {first}.")
    if rhs is None:
        return
    b_eval = np.asarray(rhs)
    first = first_nonfinite(b_eval)
    if first is not None:
        raise ValueError(f"{label}: RHS contains NaN/Inf at index {first}.")

def _assemble_linear_mass_matrix(mesh: 'LinearMesh') -> 'np.ndarray':
    """Assemble the global consistent mass matrix for the linear boundary mesh."""

    nnodes = len(mesh.nodes)
    m_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        m_mat[np.ix_(ids, ids)] += _linear_mass_block(elem)
    return m_mat


def _assemble_linear_weighted_mass_matrix(
    mesh: 'LinearMesh',
    element_coefficients: 'np.ndarray',
) -> 'np.ndarray':
    """Assemble ``integral phi_i c_h phi_j ds`` for elementwise-constant c.

    Material tables and impedance tapers are sampled at element centers, so
    the discrete coefficient represented by ``PanelCoupledInfo`` is naturally
    piecewise constant. Keeping it inside each element weak integral is exact
    for that discrete material model and avoids unweighted node averaging on
    nonuniform meshes and at taper endpoints.
    """

    coeff = np.asarray(element_coefficients, dtype=np.complex128).reshape(-1)
    if coeff.size != len(mesh.elements):
        raise ValueError(
            "Weighted mass coefficient count must match mesh element count."
        )
    if not np.all(np.isfinite(coeff.real) & np.isfinite(coeff.imag)):
        raise ValueError("Weighted mass coefficients must all be finite.")
    nnodes = len(mesh.nodes)
    weighted = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        weighted[np.ix_(ids, ids)] += (
            complex(coeff[eidx]) * _linear_mass_block(elem)
        )
    return weighted


def _robin_alpha_elements(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Return per-element Robin alpha and the PEC-element mask."""

    if len(mesh.elements) != len(infos):
        raise ValueError(
            "Robin coefficient construction requires matching elements and infos."
        )
    alpha = np.zeros(len(mesh.elements), dtype=np.complex128)
    pec = np.zeros(len(mesh.elements), dtype=bool)
    for eidx, info in enumerate(infos):
        z_surf = complex(info.robin_impedance)
        if abs(z_surf) <= EPS:
            pec[eidx] = True
            continue
        eps_m = info.eps_minus if info.minus_region >= 0 else info.eps_plus
        mu_m = info.mu_minus if info.minus_region >= 0 else info.mu_plus
        k_m = info.k_minus if info.minus_region >= 0 else info.k_plus
        alpha[eidx] = _surface_robin_alpha(
            pol, eps_m, mu_m, k_m, z_surf
        )
    return alpha, pec


def _green_2d(k0: 'Union[complex, float]', r: 'float') -> 'complex':
    """2D scalar Green's function G = j/4 * H0^(2)(k r)."""

    x = complex(k0) * max(r, EPS)
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    return 0.25j * _hankel2_0(x)


def _quadrature_nodes(order: 'int' = 10) -> 'Tuple[np.ndarray, np.ndarray]':
    qx, qw = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    return t, w

_QUAD_CACHE: 'Dict[int, Tuple[np.ndarray, np.ndarray]]' = {}
_QUAD_LOCK = threading.Lock()

def _get_quadrature(order: 'int') -> 'Tuple[np.ndarray, np.ndarray]':
    o = int(order)
    result = _QUAD_CACHE.get(o)
    if result is not None:
        return result
    with _QUAD_LOCK:

        if o not in _QUAD_CACHE:
            _QUAD_CACHE[o] = _quadrature_nodes(o)
        return _QUAD_CACHE[o]

def _near_singular_scheme(distance: 'float', panel_length: 'float') -> 'Tuple[int, int]':
    """
    Choose quadrature order and source-panel subdivision count.

    This improves near-singular accuracy when observation points approach a panel.
    """

    ratio = float(distance) / max(float(panel_length), EPS)
    if ratio < 0.25:
        return 64, 16
    if ratio < 0.60:
        return 56, 10
    if ratio < 1.50:
        return 40, 6
    if ratio < 3.00:
        return 28, 3
    return 16, 1
