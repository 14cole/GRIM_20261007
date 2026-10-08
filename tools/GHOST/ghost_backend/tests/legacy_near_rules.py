"""Legacy two-piece BoR near rules (the pre-graded NumPy samplers), kept beside the tests as
independent references for the graded native rules of ``ghost_backend.bor.kernels``."""
from ghost_backend.bor.kernels import (
    _cosine_moments,
    _green_samples,
    _ibc_brackets_grid,
    _native_mfie_brackets,
    _project_parity_brackets,
    _stable_brackets,
    cached_leggauss,
    math,
    np,
)


def _modal_kernels_near_rule(rho_p, z_p, rho_q, z_q, k, m_max: 'int', order: 'int' = 48,
                       tail_order: 'int' = 0):
    """
    G_m for m = 0..m_max+1 at near-singular point pairs (legacy two-piece rule).

    Substitution xi = 2 asin(s), then s = s_scale * sinh(v): concentrates
    quadrature at xi = 0 where R -> d; a single Gauss-Legendre tail panel
    covers [2 asin(s0), pi].  Its tail must resolve the 1/R near-singularity
    beyond the core, so its order grows without bound as d/a falls; the
    production rule is the graded one of ``modal_kernels_near``.  This form
    remains for ``_checked_near_kernels`` and as an independent check.
    Inputs are 1-D arrays of pair coordinates (n_pairs,).
    Returns [n_pairs, m_max+2].
    """

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    n = rho_p.size
    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q

    out = np.zeros((n, m_max + 2), dtype=np.complex128)
    m = np.arange(m_max + 2)


    on_axis = rr4 <= 1e-30
    if np.any(on_axis):
        R0 = np.sqrt(d2[on_axis])
        g0 = np.exp(-1j * complex(k) * R0) / (4.0 * np.pi * np.maximum(R0, 1e-300))

        out[on_axis, 0] = 2.0 * np.pi * g0

    idx = np.flatnonzero(~on_axis)
    if idx.size == 0:
        return out

    d = np.sqrt(np.maximum(d2[idx], 1e-300))
    a = np.sqrt(rr4[idx])
    points = (rho_p[idx], z_p[idx], rho_q[idx], z_q[idx])


    s0 = np.minimum(0.25, 20.0 * d / a)
    xg, wg = cached_leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg


    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = (d / a)[:, None] * np.sinh(v)
    s = np.minimum(s, 1.0)
    xi = 2.0 * np.arcsin(s)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    dxi_dv = 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    g = _green_samples(*points, k, xi)
    w_all = wv * dxi_dv
    gw = g * w_all
    acc = _cosine_moments(gw, xi, len(m))


    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    required_tail = int(max(64, math.ceil(4.0 * osc)))
    n_tail = int(tail_order) if tail_order > 0 else required_tail
    xt, wt = cached_leggauss(n_tail)
    u01t = 0.5 * (xt + 1.0)
    w01t = 0.5 * wt
    xi0 = 2.0 * np.arcsin(s0)
    span = np.pi - xi0
    xi_t = xi0[:, None] + u01t[None, :] * span[:, None]
    w_t = w01t[None, :] * span[:, None]
    gt = _green_samples(*points, k, xi_t)
    gtw = gt * w_t
    acc += _cosine_moments(gtw, xi_t, len(m))

    out[idx, :] = acc
    return out


def _project_pm_brackets(Fp, Fm, w_pos, xi_pos, m) -> 'List[np.ndarray]':
    """Project half-range +-xi bracket samples onto modes:

        proj_m = int_0^pi [F(+xi) e^{-jm xi} + F(-xi) e^{+jm xi}] dxi
               = int_0^pi [S cos(m xi) - j D sin(m xi)] dxi,
        S = F(+xi) + F(-xi),   D = F(+xi) - F(-xi)   (weights folded in).

    Splitting into real cos/sin einsums costs ~4x fewer flops than the
    complex-exponential form and shares the trig tables across brackets.
    Returns one [n_pairs, len(m)] array per bracket.

    Building the trig tables, not the products, is the bulk of the cost, so
    they are built for |m| only: cos is even and sin is odd, which halves
    both the tables and the products for a symmetric -m_max..m_max range.
    Real and imaginary parts ride as extra GEMM rows so the tables stay
    float64 instead of being promoted to complex per chunk."""

    sums = [(Fpos + Fneg) * w_pos for Fpos, Fneg in zip(Fp, Fm)]
    diffs = [(Fpos - Fneg) * w_pos for Fpos, Fneg in zip(Fp, Fm)]
    count = len(sums)
    stacked = np.stack(
        [value.real for value in sums] + [value.imag for value in sums]
        + [value.real for value in diffs] + [value.imag for value in diffs],
        axis=1,
    )
    del sums, diffs

    m = np.asarray(m)
    magnitude = np.abs(m)
    orders = np.arange(int(magnitude.max()) + 1 if m.size else 0)
    pairs, rows = len(xi_pos), stacked.shape[1]
    cosines = np.empty((pairs, rows, len(orders)))
    sines = np.empty((pairs, rows, len(orders)))
    for m0 in range(0, len(orders), 32):
        arg = xi_pos[:, :, None] * orders[None, None, m0:m0 + 32]
        cosines[:, :, m0:m0 + 32] = np.matmul(stacked, np.cos(arg))
        sines[:, :, m0:m0 + 32] = np.matmul(stacked, np.sin(arg))

    s_cos = cosines[:, 0:count] + 1j * cosines[:, count:2 * count]
    d_sin = sines[:, 2 * count:3 * count] + 1j * sines[:, 3 * count:4 * count]
    # sin(m xi) = sign(m) sin(|m| xi); sign(0) and sin(0) agree at zero.
    signs = np.sign(m).astype(float)
    projected = s_cos[:, :, magnitude] - 1j * signs[None, None, :] * d_sin[:, :, magnitude]
    return [projected[:, i, :] for i in range(count)]


def _mfie_kernels_near_rule(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                      m_max: 'int', order: 'int' = 48,
                      tail_order: 'int' = 0, stable: 'bool' = False):
    """Modal MFIE kernels for near point pairs (1-D pair lists) via the
    same capped sinh core + oscillation tail as the legacy Green's rule, with
    the negative half integrated by exact bracket parity. Returns four arrays
    [n_pairs, 2*m_max+1].  Legacy two-piece rule (``_checked_near_kernels``);
    production uses the graded rule of ``mfie_kernels_near``."""

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    tr_p = np.broadcast_to(np.asarray(tr_p, dtype=float), rho_p.shape)
    tz_p = np.broadcast_to(np.asarray(tz_p, dtype=float), rho_p.shape)
    tr_q = np.broadcast_to(np.asarray(tr_q, dtype=float), rho_q.shape)
    tz_q = np.broadcast_to(np.asarray(tz_q, dtype=float), rho_q.shape)

    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q
    d = np.sqrt(np.maximum(d2, 1e-300))
    a = np.sqrt(np.maximum(rr4, 1e-300))
    s0 = np.minimum(0.25, 20.0 * d / np.maximum(a, 1e-300))
    s0 = np.where(rr4 <= 1e-30, 1.0, s0)

    xg, wg = cached_leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg

    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = np.minimum((d / a)[:, None] * np.sinh(v), 1.0)
    xi_c = 2.0 * np.arcsin(s)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    w_c = wv * 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    axis = rr4 <= 1e-30
    xi_c[axis] = np.pi * u01
    w_c[axis] = np.pi * w01

    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    required_tail = int(max(64, math.ceil(4.0 * osc)))
    n_tail = int(tail_order) if tail_order > 0 else required_tail
    xt, wt = cached_leggauss(n_tail)
    xi0 = 2.0 * np.arcsin(np.minimum(s0, 1.0))
    span = np.pi - xi0
    xi_t = xi0[:, None] + 0.5 * (xt + 1.0)[None, :] * span[:, None]
    w_t = 0.5 * wt[None, :] * span[:, None]

    xi_pos = np.concatenate([xi_c, xi_t], axis=1)
    w_pos = np.concatenate([w_c, w_t], axis=1)
    m = np.arange(-m_max, m_max + 1)
    outs = []


    def brackets_grid(xi):
        native = _native_mfie_brackets(
            (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q), k, xi, True
        )
        if native is not None:
            return native

        cx, sx = np.cos(xi), np.sin(xi)
        Rx = rho_p[:, None] - rho_q[:, None] * cx
        Ry = rho_q[:, None] * sx
        Rz = (z_p - z_q)[:, None] * np.ones_like(cx)
        R = np.maximum(np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2), 1e-300)
        p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)
        WtR = tr_p[:, None] * Rx + tz_p[:, None] * Rz
        WfR = Ry
        nR = -tz_p[:, None] * Rx + tr_p[:, None] * Rz
        n_tq = -(tz_p * tr_q)[:, None] * cx + (tr_p * tz_q)[:, None] * np.ones_like(cx)
        n_fq = -tz_p[:, None] * sx
        Wt_tq = (tr_p * tr_q)[:, None] * cx + (tz_p * tz_q)[:, None] * np.ones_like(cx)
        Wt_fq = tr_p[:, None] * sx
        Wf_tq = -tr_q[:, None] * sx
        Wf_fq = cx * np.ones_like(Rx)
        return (-p * (WtR * n_tq - Wt_tq * nR), -p * (WtR * n_fq - Wt_fq * nR),
                -p * (WfR * n_tq - Wf_tq * nR), -p * (WfR * n_fq - Wf_fq * nR))

    Fp = (_stable_brackets((rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q), k, xi_pos, 'mfie')
          if stable else brackets_grid(xi_pos))
    outs = _project_parity_brackets(Fp, w_pos, xi_pos, m)
    return tuple(outs)


def _ibc_kernels_near_rule(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                     m_max: 'int', order: 'int' = 48,
                     tail_order: 'int' = 0, stable: 'bool' = False):
    """Modal IBC kernels for near point-pair lists [n_pairs, 2*m_max+1],
    same two-piece grid as the legacy MFIE rule (``_checked_near_kernels``;
    production uses the graded rule of ``ibc_kernels_near``)."""

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    tr_p = np.broadcast_to(np.asarray(tr_p, dtype=float), rho_p.shape)
    tz_p = np.broadcast_to(np.asarray(tz_p, dtype=float), rho_p.shape)
    tr_q = np.broadcast_to(np.asarray(tr_q, dtype=float), rho_q.shape)
    tz_q = np.broadcast_to(np.asarray(tz_q, dtype=float), rho_q.shape)

    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q
    d = np.sqrt(np.maximum(d2, 1e-300))
    a = np.sqrt(np.maximum(rr4, 1e-300))
    s0 = np.minimum(0.25, 20.0 * d / np.maximum(a, 1e-300))
    s0 = np.where(rr4 <= 1e-30, 1.0, s0)

    xg, wg = cached_leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg
    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = np.minimum((d / a)[:, None] * np.sinh(v), 1.0)
    xi_c = 2.0 * np.arcsin(s)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    w_c = wv * 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    axis = rr4 <= 1e-30
    xi_c[axis] = np.pi * u01
    w_c[axis] = np.pi * w01
    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    required_tail = int(max(64, math.ceil(4.0 * osc)))
    n_tail = int(tail_order) if tail_order > 0 else required_tail
    xt, wt = cached_leggauss(n_tail)
    xi0 = 2.0 * np.arcsin(np.minimum(s0, 1.0))
    span = np.pi - xi0
    xi_t = xi0[:, None] + 0.5 * (xt + 1.0)[None, :] * span[:, None]
    w_t = 0.5 * wt[None, :] * span[:, None]
    xi_pos = np.concatenate([xi_c, xi_t], axis=1)
    w_pos = np.concatenate([w_c, w_t], axis=1)
    m = np.arange(-m_max, m_max + 1)

    Fp = (_stable_brackets((rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q), k, xi_pos, 'ibc')
          if stable else _ibc_brackets_grid(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi_pos))
    outs = _project_parity_brackets(Fp, w_pos, xi_pos, m)
    return tuple(outs)
