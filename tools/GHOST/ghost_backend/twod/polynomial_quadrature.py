"""Polynomial Galerkin near blocks with analytic logarithmic self terms."""
from functools import lru_cache
import threading
import numpy as np
from scipy.special import hankel2e
from ghost_backend.twod.basis import values, coefficients, derivative_matrix


@lru_cache(maxsize=4)
def log_moments(degree):
    # Integral x**n y**m log|x-y| on the unit square, split at y=x.
    result = np.zeros((degree + 1, degree + 1))
    harmonic = [0.] + [sum(1. / j for j in range(1, i + 1)) for i in range(1, degree + 2)]
    for n in range(degree + 1):
        for m in range(degree + 1):
            total = n + m + 2
            result[n, m] = (-1. / ((m + 1) * total**2) - harmonic[m + 1] / ((m + 1) * total)
                            -1. / ((n + 1) * total**2) - harmonic[n + 1] / ((n + 1) * total))
    c = coefficients(degree)
    return c.T @ result @ c


def hypersingular(s, k, normal_o, normal_s, length_o, length_s):
    do = derivative_matrix(s.shape[0] - 1)
    ds = derivative_matrix(s.shape[1] - 1)
    return -complex(k)**2 * np.dot(normal_o, normal_s) * s + do.T @ s @ ds / (length_o * length_s)


def _node_at(element, parameter):
    if parameter == 0.0:
        return int(element.node_ids[0])
    if parameter == 1.0:
        return int(element.node_ids[1])
    return None


def _shared_ends(obs, src, intervals):
    """Touching interval ends as (obs start, src start), or None (last match wins).

    Ends touch when they coincide to 1e-12, or when they are one mesh node: the
    linear mesh merges endpoints on a 1e-9 m snap grid, so such coordinates may
    differ by up to sqrt(2) snap widths (operators.NODE_SNAP_TOLERANCE).
    """
    from ghost_backend.twod.operators import NODE_SNAP_TOLERANCE
    (oa, ob), (sa, sb) = intervals
    shared = None
    for a in (0, 1):
        for b in (0, 1):
            to, ts = (oa if a == 0 else ob), (sa if b == 0 else sb)
            po = obs.p0 + to * (obs.p1 - obs.p0)
            ps = src.p0 + ts * (src.p1 - src.p0)
            distance = np.linalg.norm(po - ps)
            node = _node_at(obs, to)
            if distance <= 1e-12 or (node is not None and node == _node_at(src, ts)
                                      and distance <= 2.0 * NODE_SNAP_TOLERANCE):
                shared = (a == 0, b == 0)
    return shared


def block(obs, src, k, obs_derivative=True, order=24, intervals=((0., 1.), (0., 1.))):
    from ghost_backend.twod.operators import _get_quadrature
    qo, qs = len(obs.node_ids) - 1, len(src.node_ids) - 1
    oa, ob = intervals[0]; sa, sb = intervals[1]
    t, w = _get_quadrature(order)
    same = obs.panel_index == src.panel_index and intervals[0] == intervals[1]
    shared = None if same else _shared_ends(obs, src, intervals)
    stable_difference = None
    if same or shared is not None:
        # A fourth-power radial substitution regularizes endpoint log terms.
        u = t[:, None]**4
        v = t[None, :]
        weights = (4 * t[:, None]**7 * w[:, None] * w[None, :]).ravel()
        x = np.concatenate((np.broadcast_to(u, (order, order)).ravel(), (u*v).ravel()))
        y = np.concatenate(((u*v).ravel(), np.broadcast_to(u, (order, order)).ravel()))
        weight = np.tile(weights, 2)
        if same:
            stable_difference = ((ob-oa)*(x-y))[:, None]*(obs.p1-obs.p0)
        if shared is not None:
            eo = obs.p0 + (oa if shared[0] else ob)*(obs.p1-obs.p0)
            es = src.p0 + (sa if shared[1] else sb)*(src.p1-src.p0)
            origin = eo-es
            roundoff = 64*np.finfo(float).eps*max(np.linalg.norm(eo), np.linalg.norm(es), obs.length, src.length)
            if np.linalg.norm(origin) <= roundoff: origin = np.zeros(2)
            do = (ob-oa)*(obs.p1-obs.p0)*(1 if shared[0] else -1)
            ds = (sb-sa)*(src.p1-src.p0)*(1 if shared[1] else -1)
            stable_difference = origin + x[:, None]*do - y[:, None]*ds
            if not shared[0]: x = 1 - x
            if not shared[1]: y = 1 - y
    else:
        x, y = np.meshgrid(t, t, indexing='ij')
        x, y = x.ravel(), y.ravel()
        weight = np.outer(w, w).ravel()
    x = oa + (ob - oa) * x
    y = sa + (sb - sa) * y
    diff = ((obs.p0-src.p0) + x[:, None]*(obs.p1-obs.p0) - y[:, None]*(src.p1-src.p0)
            if stable_difference is None else stable_difference)
    distance = np.linalg.norm(diff, axis=1)
    phi_o, phi_s = values(x, qo), values(y, qs)
    if np.any(distance <= 0):
        raise ValueError('Polynomial quadrature encountered intersecting integration points.')
    argument = complex(k)*distance
    phase = np.exp(-1j*argument)
    # Duffy nodes approach a corner without reaching it. A fixed distance cutoff
    # would remove a changing portion of the integrable kernel as order rises.
    green = .25j*hankel2e(0, argument)*phase
    exact_log = same and intervals == ((0., 1.), (0., 1.))
    if exact_log:
        green -= np.log(abs(x - y)) / (2 * np.pi)
    sbk = phi_o.T @ ((weight * green)[:, None] * phi_s)
    if exact_log: sbk += log_moments(qo) / (2 * np.pi)
    if same:
        kbk = np.zeros_like(sbk)
    else:
        normal = obs.normal if obs_derivative else src.normal
        derivative = ((-.25j if obs_derivative else .25j)*complex(k)*hankel2e(1, argument)*phase
                      * (diff @ normal)/distance)
        kbk = phi_o.T @ ((weight * derivative)[:, None] * phi_s)
    scale = obs.length * src.length * (ob - oa) * (sb - sa)
    if not np.all(np.isfinite(sbk)) or not np.all(np.isfinite(kbk)):
        raise ValueError('Polynomial near quadrature did not converge: non-finite kernel block.')
    return sbk * scale, kbk * scale


def near_block(obs, src, k, obs_derivative=True, depth=0, intervals=((0., 1.), (0., 1.))):
    # The absolute scale prevents relative tests on symmetry-zero K blocks.
    from ghost_backend.twod.operators import NEAR_PAIR_QUADRATURE_RTOL, NEAR_PAIR_QUADRATURE_MAX_DEPTH
    from ghost_backend.execution.cpu import current_state
    state = current_state()
    if state is not None: state.checkpoint()
    low = block(obs, src, k, obs_derivative, 20, intervals)
    high = block(obs, src, k, obs_derivative, 36, intervals)
    scale = max(np.max(abs(high[0])), np.max(abs(high[1])), obs.length * src.length * 1e-10)
    error = max(np.max(abs(a-b)) for a, b in zip(low, high))
    if error <= NEAR_PAIR_QUADRATURE_RTOL * scale:
        return high
    if obs.panel_index == src.panel_index or depth >= 2*NEAR_PAIR_QUADRATURE_MAX_DEPTH:
        finest = block(obs, src, k, obs_derivative, 72, intervals)
        error = max(np.max(abs(a-b)) for a, b in zip(high, finest))
        if error > NEAR_PAIR_QUADRATURE_RTOL * scale:
            extra = block(obs, src, k, obs_derivative, 144, intervals)
            error = max(np.max(abs(a-b)) for a, b in zip(finest, extra))
            finest = extra
        if error > NEAR_PAIR_QUADRATURE_RTOL * scale:
            raise ValueError('Polynomial near quadrature did not converge; refine the mesh. '
                             'Panels {} / {}, k={}, intervals={}, relative change={:.3g}.'.format(
                                 obs.panel_index, src.panel_index, k, intervals, error/scale))
        return finest
    oi, si = intervals
    if (oi[1]-oi[0])*obs.length >= (si[1]-si[0])*src.length:
        mid = sum(oi) / 2
        children = [((oi[0], mid), si), ((mid, oi[1]), si)]
    else:
        mid = sum(si) / 2
        children = [(oi, (si[0], mid)), (oi, (mid, si[1]))]
    parts = [near_block(obs, src, k, obs_derivative, depth + 1, child) for child in children]
    return tuple(a + b for a, b in zip(*parts))


# ---------------------------------------------------------------------------
# Batched near blocks.
#
# near_block above is the reference: one pair at a time, recursively. The
# functions below run the identical adaptive rule (orders 20/36, then 72/144 on
# a panel's self term or at the depth limit, otherwise bisection of the longer
# interval) for many pairs at once. Each stage evaluates every outstanding
# (pair, interval, order) in fixed-size vectorized chunks spread over the
# assembly threads.
#
# Blocks are kept as scaled monomial moments up to cubic degree, so the same
# kernel samples project onto the quadratic and cubic nodal bases. Moments are
# cached for the lifetime of a certified request, so the cubic check on the
# quadratic candidate's panels reuses its Hankel evaluations. Row-wise sums keep
# every moment independent of chunk composition and thread count.
# ---------------------------------------------------------------------------
from contextlib import contextmanager
from ghost_backend.execution.runtime import ScopedValue

_MOMENT_DEGREE = 3
# Rows per moment GEMM (see _monomial_moments).
_MOMENT_GEMM_ROWS = 64
# About 200 bytes of working storage per sample. Concurrent chunks stay within
# the near-batch workspace that dense resource forecasts already reserve.
_CHUNK_SAMPLES = 1 << 17
_ACTIVE_TASKS = 512
MOMENT_CACHE_BYTES = 256 * 1024**2
_MOMENT_CACHE = ScopedValue('ghost_polynomial_near_moments', None)


class MomentCache:
    """Bounded first-in first-out store of scaled near-pair moments."""
    # Key bytes, two 4x4 complex moment blocks, and container overhead.
    ENTRY_BYTES = 1024

    # Near-batch kernel tables of this scope share the CPU table budget's size.
    TABLE_BUDGET = 32 * 1024**2

    def __init__(self, budget_bytes=MOMENT_CACHE_BYTES):
        self.budget = int(budget_bytes)
        self.values = {}
        self.hits = self.stores = self.evictions = 0
        self.tables = {}
        self.table_bytes = 0

    def store_table(self, key, table):
        """Keep a near kernel table for this scope (first in, first out); return the kept one."""
        if key in self.tables:
            return self.tables[key]
        size = 0 if table is None else int(table.evidence['bytes'])
        if size > self.TABLE_BUDGET:
            return table
        while self.tables and self.table_bytes + size > self.TABLE_BUDGET:
            old = self.tables.pop(next(iter(self.tables)))
            self.table_bytes -= 0 if old is None else int(old.evidence['bytes'])
        self.tables[key] = table
        self.table_bytes += size
        return table

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.hits += 1
        return value

    def put(self, key, value):
        if key in self.values or self.ENTRY_BYTES > self.budget:
            return
        while (len(self.values) + 1) * self.ENTRY_BYTES > self.budget:
            self.values.pop(next(iter(self.values)))
            self.evictions += 1
        self.values[key] = value
        self.stores += 1


@contextmanager
def moment_cache_scope(budget_bytes=MOMENT_CACHE_BYTES):
    """Share near-pair moments across nested solves; reuse an enclosing cache."""
    existing = _MOMENT_CACHE.get()
    if existing is not None:
        yield existing
        return
    cache = MomentCache(budget_bytes)
    with _MOMENT_CACHE.override(cache):
        yield cache


def solve_checkpoint():
    """The abort checkpoint of the active CPU state or assembly session, if any."""
    from ghost_backend.execution.cpu import current_state
    from ghost_backend.twod.assembly.session import current_session
    owner = current_state() or current_session()
    return owner.checkpoint if owner is not None else None


def map_checked(function, jobs, workers, checkpoint=None):
    """Yield ordered results with at most twice the worker count in flight.

    Consumers can copy each result into its final destination immediately;
    completed results do not accumulate until every job has finished. Queued
    jobs are cancelled on failure, cancellation, or explicit iterator close.

    Every job runs in a copy of the caller's context: the scoped values of the
    solve (the CPU kernel-table store, the moment cache, the execution
    options) are visible on the worker threads, so a near-pair batch finds the
    solve's kernel table instead of rebuilding one per thread and domain.
    """
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            if checkpoint is not None:
                checkpoint()
            result = function(job)
            if checkpoint is not None:
                checkpoint()
            yield result
            result = None
        return
    import contextvars
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    context = contextvars.copy_context()

    def run(job):
        return context.copy().run(function, job)

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        pending = deque()
        submitted = 0
        while submitted < len(jobs) and len(pending) < 2 * workers:
            pending.append(pool.submit(run, jobs[submitted]))
            submitted += 1
        while pending:
            result = pending.popleft().result()
            if checkpoint is not None:
                checkpoint()
            yield result
            result = None
            if submitted < len(jobs):
                pending.append(pool.submit(run, jobs[submitted]))
                submitted += 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@lru_cache(maxsize=1)
def _log_monomials():
    degree = _MOMENT_DEGREE
    result = np.zeros((degree + 1, degree + 1))
    harmonic = [0.] + [sum(1. / j for j in range(1, i + 1)) for i in range(1, degree + 2)]
    for n in range(degree + 1):
        for m in range(degree + 1):
            total = n + m + 2
            result[n, m] = (-1. / ((m + 1) * total**2) - harmonic[m + 1] / ((m + 1) * total)
                            -1. / ((n + 1) * total**2) - harmonic[n + 1] / ((n + 1) * total))
    result.flags.writeable = False
    return result


@lru_cache(maxsize=16)
def _nodes(order, singular):
    from ghost_backend.twod.operators import _get_quadrature
    t, w = _get_quadrature(order)
    if singular:
        u = t[:, None]**4
        v = t[None, :]
        weights = (4 * t[:, None]**7 * w[:, None] * w[None, :]).ravel()
        x = np.concatenate((np.broadcast_to(u, (order, order)).ravel(), (u*v).ravel()))
        y = np.concatenate(((u*v).ravel(), np.broadcast_to(u, (order, order)).ravel()))
        weight = np.tile(weights, 2)
    else:
        x, y = np.meshgrid(t, t, indexing='ij')
        x, y = x.ravel(), y.ravel()
        weight = np.outer(w, w).ravel()
    for array in (x, y, weight):
        array.flags.writeable = False
    return x, y, weight


class _Task:
    __slots__ = ('node', 'pair', 'depth', 'intervals', 'kind', 'shared')

    def __init__(self, node, pair, depth, intervals):
        self.node, self.pair, self.depth, self.intervals = node, pair, depth, intervals
        self.kind = self.shared = None


def _classify(task, obs, src):
    (oa, ob), (sa, sb) = task.intervals
    if obs.panel_index == src.panel_index and task.intervals[0] == task.intervals[1]:
        task.kind, task.shared = 'same', None
        return
    shared = _shared_ends(obs, src, task.intervals)
    task.kind, task.shared = ('shared', shared) if shared is not None else ('regular', None)


def _moment_key(k, obs_derivative, obs, src, task, order):
    # Endpoint welding can change across mesh generations even when this
    # pair's coordinates do not. Include the classified rule and its corner
    # orientation; polynomial degree is deliberately absent from the key.
    kind = {'regular': 0, 'same': 1, 'shared': 2}[task.kind]
    shared = (-1, -1) if task.shared is None else task.shared
    head = np.array([complex(k).real, complex(k).imag, float(obs_derivative),
                     float(obs.panel_index == src.panel_index),
                     task.intervals[0][0], task.intervals[0][1],
                     task.intervals[1][0], task.intervals[1][1], float(order),
                     float(kind), float(shared[0]), float(shared[1])])
    return b''.join(np.asarray(part, float).tobytes() for part in
                    (head, obs.p0, obs.p1, obs.normal, src.p0, src.p1, src.normal))


def box_moment_keys(k, obs_derivative, order, want_s, want_k, obs_elems, src_elems):
    """Moment-cache keys of fixed-order (tensor-Gauss) pairs: wavenumber, rule, channels and geometry.

    The byte layout differs from ``_moment_key`` (the polynomial Duffy rules),
    so the two families never share an entry."""
    k = complex(k)
    rows = np.empty((len(obs_elems), 18), dtype=float)
    rows[:, 0:6] = (k.real, k.imag, float(bool(obs_derivative)), float(order), float(bool(want_s)), float(bool(want_k)))
    rows[:, 6:8] = [e.p0 for e in obs_elems]
    rows[:, 8:10] = [e.p1 for e in obs_elems]
    rows[:, 10:12] = [e.normal for e in obs_elems]
    rows[:, 12:14] = [e.p0 for e in src_elems]
    rows[:, 14:16] = [e.p1 for e in src_elems]
    rows[:, 16:18] = [e.normal for e in src_elems]
    return [b'box' + row.tobytes() for row in rows]


def _build_kernel_table(k_real, k_imag, upper):
    from ghost_backend.twod.assembly.kernels import KernelTable, Rejected
    try:
        return KernelTable(complex(k_real, k_imag), upper)
    except Rejected:
        return None


# Outside any solve scope only the most recent near table is kept.
_LAST_TABLE = [None, None]
_TABLE_LOCK = threading.Lock()


def _kernel_table(k_real, k_imag, upper):
    """Validated piecewise kernel table for a lossy medium, or None if rejected.

    Tables live as long as the solve that asked for them: in the CPU state's
    table store (inside its 32 MiB table budget) when one is active, otherwise in
    the solve's moment-cache scope; a direct call outside both keeps only the
    latest table. A module-level cache of every wavenumber would outlive the
    solve and escape the table budget (tens of MB over a lossy sweep).
    """
    key = ('near', float(k_real), float(k_imag), float(upper))
    from ghost_backend.execution.cpu import current_state
    state = current_state()
    if state is not None:
        from ghost_backend.twod.assembly.kernels import cached_table
        return cached_table(state, key, lambda: _build_kernel_table(k_real, k_imag, upper))
    cache = _MOMENT_CACHE.get()
    if cache is not None:
        with _TABLE_LOCK:
            if key in cache.tables:
                return cache.tables[key]
        table = _build_kernel_table(k_real, k_imag, upper)
        with _TABLE_LOCK:
            return cache.store_table(key, table)
    with _TABLE_LOCK:
        if _LAST_TABLE[0] == key:
            return _LAST_TABLE[1]
    table = _build_kernel_table(k_real, k_imag, upper)
    with _TABLE_LOCK:
        _LAST_TABLE[:] = [key, table]
    return table


def _table_for(k, pairs):
    k = complex(k)
    if not (k.imag < 0 < k.real) or not pairs:
        return None
    return _table_for_ends(k, np.array([o.p0 for o, _ in pairs], float), np.array([o.p1 for o, _ in pairs], float),
                           np.array([s.p0 for _, s in pairs], float), np.array([s.p1 for _, s in pairs], float))


def _table_for_ends(k, obs_p0, obs_p1, src_p0, src_p1):
    """The near table of a lossy wavenumber reaching every endpoint pair of these elements.

    The reach is the largest endpoint-to-endpoint distance over the four
    endpoint combinations, taken as arrays (the per-pair loop of four norms
    was the one interpreter-bound stretch of a batch)."""
    k = complex(k)
    if not (k.imag < 0 < k.real) or not len(obs_p0):
        return None
    reach = 0.
    for a in (obs_p0, obs_p1):
        for b in (src_p0, src_p1):
            difference = a - b
            reach = max(reach, float(np.max(np.sqrt(difference[:, 0] ** 2 + difference[:, 1] ** 2))))
    if not np.isfinite(reach) or reach <= 0:
        return None
    # Power-of-two domains let later calls at this wavenumber reuse the table.
    return _kernel_table(k.real, k.imag, float(2.0 ** np.ceil(np.log2(reach * (1 + 1e-9)))))


def _kernels(k, distance, with_derivative, table):
    """(j/4) H0(kr) and, if requested, (j/4) k H1(kr) at every distance."""
    k = complex(k)
    if k.imag == 0 and k.real > 0:
        from scipy.special import j0, j1, y0, y1
        kr = k.real * distance
        green = np.empty(distance.shape, complex)
        green.real, green.imag = .25 * y0(kr), .25 * j0(kr)
        derivative = None
        if with_derivative:
            derivative = np.empty(distance.shape, complex)
            derivative.real, derivative.imag = .25 * k.real * y1(kr), .25 * k.real * j1(kr)
        return green, derivative
    if table is not None:
        if with_derivative:
            values = table.evaluate(distance)
            green, derivative = values[..., 0].copy(), values[..., 1].copy()
            bad = ~(np.isfinite(green) & np.isfinite(derivative))
        else:
            green, derivative = np.array(table.evaluate(distance, 0), complex), None
            bad = ~np.isfinite(green)
        if np.any(bad):
            exact_green, exact_derivative = _kernels(k, distance[bad], with_derivative, None)
            green[bad] = exact_green
            if with_derivative:
                derivative[bad] = exact_derivative
        return green, derivative
    argument = k * distance
    phase = np.exp(-1j * argument)
    green = .25j * hankel2e(0, argument) * phase
    derivative = .25j * k * hankel2e(1, argument) * phase if with_derivative else None
    return green, derivative


def _evaluate_chunk(tasks, pairs, k, obs_derivative, order, kind, shared, table=None):
    """Scaled monomial S and K moments, shape (T, 2, 4, 4), for one node set."""
    count = len(tasks)
    obs = [pairs[t.pair][0] for t in tasks]
    src = [pairs[t.pair][1] for t in tasks]
    p0o = np.array([e.p0 for e in obs], float)
    p0s = np.array([e.p0 for e in src], float)
    seg_o = np.array([e.p1 - e.p0 for e in obs], float)
    seg_s = np.array([e.p1 - e.p0 for e in src], float)
    iv = np.array([(t.intervals[0][0], t.intervals[0][1], t.intervals[1][0], t.intervals[1][1])
                   for t in tasks], float)
    oa, ob, sa, sb = iv[:, 0:1], iv[:, 1:2], iv[:, 2:3], iv[:, 3:4]
    x0, y0, weight = _nodes(order, kind != 'regular')
    if kind == 'same':
        diff = ((ob - oa) * (x0 - y0)[None, :])[:, :, None] * seg_o[:, None, :]
        x, y = x0, y0
    elif kind == 'shared':
        eo = p0o + (oa if shared[0] else ob) * seg_o
        es = p0s + (sa if shared[1] else sb) * seg_s
        origin = eo - es
        roundoff = 64 * np.finfo(float).eps * np.maximum.reduce(
            [np.linalg.norm(eo, axis=1), np.linalg.norm(es, axis=1),
             np.array([e.length for e in obs]), np.array([e.length for e in src])])
        origin[np.linalg.norm(origin, axis=1) <= roundoff] = 0.
        do = (ob - oa) * seg_o * (1 if shared[0] else -1)
        ds = (sb - sa) * seg_s * (1 if shared[1] else -1)
        diff = origin[:, None, :] + x0[None, :, None] * do[:, None, :] - y0[None, :, None] * ds[:, None, :]
        x = x0 if shared[0] else 1 - x0
        y = y0 if shared[1] else 1 - y0
    else:
        x, y = x0, y0
    base_x, base_y = x, y
    x = oa + (ob - oa) * x[None, :]
    y = sa + (sb - sa) * y[None, :]
    if kind == 'regular':
        diff = ((p0o - p0s)[:, None, :] + x[:, :, None] * seg_o[:, None, :]
                - y[:, :, None] * seg_s[:, None, :])
    distance = np.linalg.norm(diff, axis=2)
    if np.any(distance <= 0):
        raise ValueError('Polynomial quadrature encountered intersecting integration points.')
    with_derivative = kind != 'same'
    green, derivative = _kernels(k, distance, with_derivative, table)
    exact = np.all(iv == (0., 1., 0., 1.), axis=1) if kind == 'same' else np.zeros(count, bool)
    if np.any(exact):
        green[exact] -= np.log(abs(x0 - y0))[None, :] / (2 * np.pi)
    kernels = [weight[None, :] * green]
    if with_derivative:
        normal = np.array([(o if obs_derivative else s).normal for o, s in zip(obs, src)], float)
        projection = diff[:, :, 0] * normal[:, None, 0] + diff[:, :, 1] * normal[:, None, 1]
        if obs_derivative:
            np.negative(derivative, out=derivative)
        kernels.append(weight[None, :] * (derivative * projection / distance))
    moments = np.zeros((count, 2, _MOMENT_DEGREE + 1, _MOMENT_DEGREE + 1), complex)
    x = y = None
    # Every task of a chunk shares its quadrature nodes up to the affine maps
    # x = oa + (ob - oa) base_x and y = sa + (sb - sa) base_y, so the moments
    # are one matrix product against the base-node monomials followed by a
    # per-task binomial change of variable (the identity on unit intervals).
    moments[:, :len(kernels)] = _monomial_moments(kernels, base_x, base_y, oa[:, 0], (ob - oa)[:, 0],
                                                  sa[:, 0], (sb - sa)[:, 0])
    kernels = None
    if np.any(exact):
        moments[exact, 0] += _log_monomials() / (2 * np.pi)
    scale = (np.array([e.length for e in obs]) * np.array([e.length for e in src])
             * (ob - oa)[:, 0] * (sb - sa)[:, 0])
    moments *= scale[:, None, None, None]
    if not np.all(np.isfinite(moments)):
        raise ValueError('Polynomial near quadrature did not converge: non-finite kernel block.')
    return moments


@lru_cache(maxsize=1)
def _binomial_table():
    from math import comb
    degree = _MOMENT_DEGREE
    result = np.array([[float(comb(a, i)) if i <= a else 0. for i in range(degree + 1)]
                       for a in range(degree + 1)])
    result.flags.writeable = False
    return result


def _binomial_transform(offset, scale):
    """``B[t, a, i] = C(a, i) offset_t**(a-i) scale_t**i``, so ``(offset + scale u)**a = sum_i B[t, a, i] u**i``."""
    degree = _MOMENT_DEGREE
    binomial = _binomial_table()
    offsets = np.vander(np.asarray(offset, float), degree + 1, increasing=True)
    scales = np.vander(np.asarray(scale, float), degree + 1, increasing=True)
    result = np.zeros((len(offsets), degree + 1, degree + 1))
    for a in range(degree + 1):
        for i in range(a + 1):
            result[:, a, i] = binomial[a, i] * offsets[:, a - i] * scales[:, i]
    return result


def _monomial_moments(kernels, base_x, base_y, x_offset, x_scale, y_offset, y_scale):
    """``sum_q kernel[t, q] x[t, q]**a y[t, q]**b`` for ``a, b <= _MOMENT_DEGREE``, shape (T, C, D+1, D+1).

    ``x[t, q] = x_offset[t] + x_scale[t] base_x[q]`` (likewise y), so the
    moments of the base nodes are one GEMM of the (T, q) kernels against the
    (q, 16) base monomials, followed by the binomial change of variable of each
    task.  Row ``t`` of the product depends on that task's kernels alone, so a
    moment is independent of the chunk's composition and of the thread count.
    """
    degree = _MOMENT_DEGREE
    monomials_x = np.vander(np.asarray(base_x, float), degree + 1, increasing=True)
    monomials_y = np.vander(np.asarray(base_y, float), degree + 1, increasing=True)
    monomials = (monomials_x[:, :, None] * monomials_y[:, None, :]).reshape(len(monomials_x), -1).astype(complex)
    stacked = np.stack(kernels)                                   # (C, T, q)
    channels, count, samples = stacked.shape
    # Every product has the same shape (_MOMENT_GEMM_ROWS x q by q x 16; the
    # last block zero-padded), so the BLAS kernel and its summation order are
    # the same for every task whatever the chunk holds: a moment depends on
    # its task alone, as the former row-wise reductions did.
    base = np.empty((channels, count, monomials.shape[1]), dtype=np.complex128)
    block = np.zeros((channels, _MOMENT_GEMM_ROWS, samples), dtype=np.complex128)
    for start in range(0, count, _MOMENT_GEMM_ROWS):
        rows = min(_MOMENT_GEMM_ROWS, count - start)
        block[:, :rows] = stacked[:, start:start + rows]
        if rows < _MOMENT_GEMM_ROWS:
            block[:, rows:] = 0.0
        base[:, start:start + rows] = (block @ monomials)[:, :rows]
    base = np.ascontiguousarray(base.reshape(channels, count, degree + 1, degree + 1).transpose(1, 0, 2, 3))
    unit = (x_offset == 0.) & (x_scale == 1.) & (y_offset == 0.) & (y_scale == 1.)
    if np.all(unit):
        return base
    transform_x = _binomial_transform(x_offset, x_scale)
    transform_y = _binomial_transform(y_offset, y_scale)
    return np.einsum('tai,tcij,tbj->tcab', transform_x, base, transform_y)


def _moments(tasks, pairs, k, obs_derivative, order, cache, threads, checkpoint, table=None):
    result = [None] * len(tasks)
    groups = {}
    for index, task in enumerate(tasks):
        key = None
        if cache is not None:
            key = _moment_key(k, obs_derivative, *pairs[task.pair], task, order)
            cached = cache.get(key)
            if cached is not None:
                result[index] = cached
                continue
        groups.setdefault((task.kind, task.shared), []).append((index, key))
    jobs = []
    for (kind, shared), members in groups.items():
        samples = order * order * (1 if kind == 'regular' else 2)
        per_chunk = max(1, _CHUNK_SAMPLES // samples)
        for start in range(0, len(members), per_chunk):
            jobs.append((kind, shared, members[start:start + per_chunk]))

    def run(job):
        kind, shared, members = job
        return _evaluate_chunk([tasks[i] for i, _ in members], pairs, k, obs_derivative, order, kind, shared, table)
    from ghost_backend.twod.operators import _NEAR_BATCH_MAX_SAMPLES
    workers = min(threads, len(jobs), max(1, _NEAR_BATCH_MAX_SAMPLES // _CHUNK_SAMPLES))
    outputs = map_checked(run, jobs, workers, checkpoint)
    for (_, _, members), values in zip(jobs, outputs):
        for (index, key), value in zip(members, values):
            result[index] = value
            if cache is not None:
                cache.put(key, value)
    return result


def _project(moment, obs, src):
    co = coefficients(len(obs.node_ids) - 1)
    cs = coefficients(len(src.node_ids) - 1)
    qo, qs = co.shape[0], cs.shape[0]
    return tuple(co.T @ moment[channel, :qo, :qs] @ cs for channel in (0, 1))


def near_blocks(pairs, k, obs_derivative=True, threads=None):
    """near_block for many (obs, src) element pairs: the same rule in batched stages."""
    from ghost_backend.twod.operators import (NEAR_PAIR_QUADRATURE_RTOL,
                                              NEAR_PAIR_QUADRATURE_MAX_DEPTH, get_assembly_threads)
    pairs = list(pairs)
    threads = get_assembly_threads() if threads is None else max(1, int(threads))
    checkpoint = solve_checkpoint()
    cache = _MOMENT_CACHE.get()
    rtol = NEAR_PAIR_QUADRATURE_RTOL
    table = _table_for(k, pairs)
    values, parents = {}, {}
    pending = [_Task(i, i, 0, ((0., 1.), (0., 1.))) for i in range(len(pairs))]
    next_node = len(pairs)

    def accept(node, value):
        # Fold siblings as soon as both converge. Their left + right order is
        # exactly the former recursive total(), without retaining every leaf.
        values[node] = value
        while node in parents:
            parent, first, second = parents[node]
            if first not in values or second not in values:
                break
            a, b = values.pop(first), values.pop(second)
            del parents[first], parents[second]
            node = parent
            values[node] = tuple(x+y for x, y in zip(a, b))

    while pending:
        if checkpoint is not None:
            checkpoint()
        # Work depth first in bounded batches. Breadth-first refinement could
        # retain exponentially many low/high moments for a difficult close pair.
        active = pending[-_ACTIVE_TASKS:]
        del pending[-len(active):]
        for task in active:
            _classify(task, *pairs[task.pair])
        lows = _moments(active, pairs, k, obs_derivative, 20, cache, threads, checkpoint, table)
        highs = _moments(active, pairs, k, obs_derivative, 36, cache, threads, checkpoint, table)
        refine, following = [], []
        for task, low_moment, high_moment in zip(active, lows, highs):
            obs, src = pairs[task.pair]
            low, high = _project(low_moment, obs, src), _project(high_moment, obs, src)
            scale = max(np.max(abs(high[0])), np.max(abs(high[1])), obs.length * src.length * 1e-10)
            error = max(np.max(abs(a-b)) for a, b in zip(low, high))
            if error <= rtol * scale:
                accept(task.node, high)
            elif obs.panel_index == src.panel_index or task.depth >= 2*NEAR_PAIR_QUADRATURE_MAX_DEPTH:
                refine.append((task, high, scale))
            else:
                oi, si = task.intervals
                if (oi[1]-oi[0])*obs.length >= (si[1]-si[0])*src.length:
                    mid = sum(oi) / 2
                    split = [((oi[0], mid), si), ((mid, oi[1]), si)]
                else:
                    mid = sum(si) / 2
                    split = [(oi, (si[0], mid)), (oi, (mid, si[1]))]
                parents[next_node] = parents[next_node+1] = (task.node, next_node, next_node+1)
                following.extend(_Task(next_node + j, task.pair, task.depth + 1, interval)
                                 for j, interval in enumerate(split))
                next_node += 2
        if refine:
            finest_moments = _moments([r[0] for r in refine], pairs, k, obs_derivative, 72,
                                      cache, threads, checkpoint, table)
            unresolved = []
            for (task, high, scale), moment in zip(refine, finest_moments):
                obs, src = pairs[task.pair]
                finest = _project(moment, obs, src)
                if max(np.max(abs(a-b)) for a, b in zip(high, finest)) > rtol * scale:
                    unresolved.append((task, finest, scale))
                else:
                    accept(task.node, finest)
            if unresolved:
                extra_moments = _moments([u[0] for u in unresolved], pairs, k, obs_derivative, 144,
                                         cache, threads, checkpoint, table)
                for (task, finest, scale), moment in zip(unresolved, extra_moments):
                    obs, src = pairs[task.pair]
                    extra = _project(moment, obs, src)
                    error = max(np.max(abs(a-b)) for a, b in zip(finest, extra))
                    if error > rtol * scale:
                        raise ValueError('Polynomial near quadrature did not converge; refine the mesh. '
                                         'Panels {} / {}, k={}, intervals={}, relative change={:.3g}.'.format(
                                             obs.panel_index, src.panel_index, k, task.intervals, error/scale))
                    accept(task.node, extra)
        pending.extend(following)

    return [values[index] for index in range(len(pairs))]
