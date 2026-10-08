"""Hierarchical far blocks of one BoR surface, compressed while they are filled.

:class:`bor.streaming.StreamingFarBlocks` samples and stores every far element
pair of every azimuthal mode: work and memory grow as N^2 M (12 GB spilled to
disk for the 10 GHz ogive, 24 GB on its certified mesh).  Interactions between
well-separated parts of a generatrix have low numerical rank, and one
azimuthal sampling of a far pair yields every mode and family at once, so a
block's rows and columns share one skeleton across all of them (ranks 12-20
per mode and 28-49 for all modes together on the ogive and on spheres, flat in
block size, frequency and refinement).

:class:`CompressedFarBlocks` holds the same far blocks, from the same sampler
(:func:`streaming._banded_stream`) and contractions (``_efie_band``,
``_bracket_band``), as an H-matrix over the generatrix nodes:

* a binary tree of node ranges and a strong-admissibility partition (blocks
  whose element spans, in the meridian half plane, are at least
  :data:`FAR_COMPRESSION_ETA` times the smaller span's diameter apart and
  share no near element pair);
* non-admissible leaf blocks, and admissible blocks too small for cross
  approximation to pay, are sampled as ordinary tiles; an admissible tile is
  then compressed by one SVD of its weighted stack of families and modes;
* larger admissible blocks are built by adaptive cross approximation of that
  stack: a pivot row is one tile of a node's two test elements against the
  block's sources and yields the row for every family and mode, a pivot
  column likewise; spread and random unused rows check the result;
* each family and mode slice of a block is then truncated to its own rank
  (at most 10 on the 10 GHz ogive, three on average, against a joint rank of
  about 16) and kept as two factors; EFIE blocks are built for the upper
  block pairs only and completed by the symmetry of the streamed build.

Mode assembly writes each block into the caller's system quadrants (a small
GEMM per low-rank block), so the dense LU, the solves and certification are
unchanged.  The error target is :data:`FAR_COMPRESSION_TOLERANCE` of each
family's far-block scale per mode (the root sum of squares of its
near-diagonal leaves), seven orders below the 10 GHz discretization error;
the 10 GHz ogive's blocks agree with the streamed ones to 2e-11 in every mode.

Tiles are sampled in spawn processes when the near-preparation scope admits a
process pool: the Python orchestration of thin tiles serializes on the GIL,
and the 10 GHz ogive's far blocks took 17 s on eight processes against 25 s
on eight threads (and 43 s for the streamed build in memory, 62 s spilled).
"""
import math
import os
import threading
from types import SimpleNamespace

import numpy as np
import scipy.linalg as la

from ghost_backend.bor.near_storage import mode_sign
from ghost_backend.bor import streaming as _streaming


# Frobenius error of each family's far blocks, per mode, relative to its scale.
FAR_COMPRESSION_TOLERANCE = 1e-10
# Largest node count of a tree leaf (leaves hold LEAF/2 to LEAF nodes).
FAR_COMPRESSION_LEAF = 32
# Admissibility: distance >= ETA * min(diameter) between two element spans.
FAR_COMPRESSION_ETA = 1.0
# 'auto' compresses the streamed far blocks of surfaces with this many nodes.
FAR_COMPRESSION_MIN_NODES = 1000
# Admissible blocks with fewer rows or columns than this are sampled as one
# tile and compressed by SVD.  Measured on one thread at 10 GHz, tile + SVD
# against cross approximation: 5 against 9 ms at 17 nodes, 25 against 29 at
# 34, 105 against 60 at 68 and 11 against 0.5 s at 540.
FAR_COMPRESSION_ACA_MIN_NODES = 48
# Cross approximation stops at this joint rank and keeps the block as a tile.
FAR_COMPRESSION_MAX_RANK = 160
# Rank of every family and mode slice of an admissible block in the memory
# estimate (measured: mean 3.2 to 4.3, 99th percentile 9 to 11).
FAR_COMPRESSION_PLANNING_RANK = 6
# Worker processes are started only for at least this many sampled element pairs.
FAR_COMPRESSION_PROCESS_PAIRS = 400_000


def far_compression_selected(n_nodes: 'int') -> 'bool':
    """Whether streamed far blocks of a surface with ``n_nodes`` nodes are compressed."""
    from ghost_backend.bor.options import current_options
    options = current_options()
    # Full compressed solves require their original accurate coefficients.
    # The sampled-ACA far approximation is a separate dense assembly route.
    if options['factorization'] == 'compressed':
        return False
    setting = options.get('far_compression', 'auto')
    if setting == 'off':
        return False
    return setting == 'on' or int(n_nodes) >= FAR_COMPRESSION_MIN_NODES


# ---------------------------------------------------------------- partition

def cluster_tree(n_nodes: 'int', leaf: 'int' = None):
    """Binary tree of node ranges: ``[(lo, hi, children)]``, root first."""
    leaf = FAR_COMPRESSION_LEAF if leaf is None else int(leaf)
    clusters = []

    def split(lo, hi):
        index = len(clusters)
        clusters.append([lo, hi, ()])
        if hi - lo > leaf:
            mid = (lo + hi) // 2
            clusters[index][2] = (split(lo, mid), split(mid, hi))
        return index

    split(0, int(n_nodes))
    return [tuple(c) for c in clusters]


def _element_span(lo: 'int', hi: 'int', n_elems: 'int'):
    """Elements carrying the basis functions of nodes ``[lo, hi)``."""
    return max(lo - 1, 0), min(hi, n_elems)


def block_partition(clusters, nodes, near_sources, eta: 'float' = None):
    """``(admissible, dense)`` cluster pairs covering every node pair once.

    Boxes bound the nodes of each cluster's element span in the (rho, z)
    half plane: the smallest meridian distance bounds the 3-D distance of the
    two rings from below, and the azimuthal kernels are smooth in it.  Both
    clusters are split together, so ``(I, J)`` is a block exactly when
    ``(J, I)`` is (the EFIE symmetry completes lower blocks from upper ones).
    """
    eta = FAR_COMPRESSION_ETA if eta is None else float(eta)
    nodes = np.asarray(nodes, float)
    n_elems = len(nodes) - 1
    boxes = []
    for lo, hi, _ in clusters:
        e0, e1 = _element_span(lo, hi, n_elems)
        span = nodes[e0:e1 + 1]
        boxes.append((span.min(axis=0), span.max(axis=0)))
    near = [np.sort(np.asarray(values, int)) for values in near_sources]

    def near_pairs(i, j):
        e0, e1 = _element_span(clusters[i][0], clusters[i][1], n_elems)
        f0, f1 = _element_span(clusters[j][0], clusters[j][1], n_elems)
        for e in range(e0, e1):
            values = near[e]
            if np.searchsorted(values, f1) > np.searchsorted(values, f0):
                return True
        return False

    admissible, dense = [], []

    def recurse(i, j):
        a, b = boxes[i], boxes[j]
        gap = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
        distance = float(np.hypot(*gap))
        diameter = min(float(np.hypot(*(a[1] - a[0]))), float(np.hypot(*(b[1] - b[0]))))
        if distance > 0 and distance >= eta * diameter and not near_pairs(i, j):
            admissible.append((i, j))
            return
        rows, cols = clusters[i][2], clusters[j][2]
        if not rows and not cols:
            dense.append((i, j))
            return
        for ci in rows or (i,):
            for cj in cols or (j,):
                recurse(ci, cj)

    recurse(0, 0)
    return admissible, dense


def estimate_compressed_far_gb(n_elems: 'int', m_max: 'int', formulation: 'str' = "cfie",
                               has_ibc: 'bool' = False) -> 'float':
    """Upper estimate (GB) of the compressed far blocks of one surface, all modes.

    Per node and per (uv, mode) slice of a family: a dense band of five leaf
    blocks of :data:`FAR_COMPRESSION_LEAF` columns, and on every tree level
    three admissible partners whose slices have
    :data:`FAR_COMPRESSION_PLANNING_RANK` directions over their rows and
    columns; the EFIE family keeps the upper blocks only.  The stores measured
    2.0 to 2.4 times smaller (the 10 GHz ogive 1.09 GB, PEC spheres of ka 30
    and 60 0.19 and 0.87 GB): leaves hold half to all of LEAF nodes and most
    slices need three or four directions.
    """
    nodes = float(int(n_elems) + 1)
    slices = 4.0 * float(int(m_max) + 1)
    leaf = float(FAR_COMPRESSION_LEAF)
    levels = max(1.0, math.ceil(math.log2(max(nodes / leaf, 1.0))) + 1.0)
    per_node = 5.0 * leaf + levels * 3.0 * 2.0 * float(FAR_COMPRESSION_PLANNING_RANK)
    families = 0.5 + (1.0 if formulation in ("cfie", "mfie") else 0.0) + (1.0 if has_ibc else 0.0)
    return families * per_node * nodes * slices * 16.0 / 1e9


# ------------------------------------------------------------------ tiles

class _Geometry:
    """What sampling a tile needs, without the solver: picklable for workers."""

    def __init__(self, solver, m_max, families, ibc_zs_pt, pmchwt):
        g = solver.g
        self.points = SimpleNamespace(**{name: np.ascontiguousarray(getattr(g, name))
                                         for name in ('rho', 'z', 'trho', 'tz')})
        self.go = int(solver.gauss_order)
        self.P = int(solver.P)
        self.ne = int(solver.gen.n_elems)
        self.Nn = int(solver.Nn)
        self.near = [np.asarray(values, int) for values in solver._near_sources_by_element]
        self.k = complex(solver.k)
        self.m_max = int(m_max)
        self.modes = np.arange(self.m_max + 1)
        self.orders = np.arange(self.m_max + 2)
        self.families = tuple(families)
        weights = _streaming.far_weights(g, self.ne, self.go, ibc_zs_pt, pmchwt)
        self.left_all = weights["left_all"]
        self.left_one = weights["lv"]["1"]
        self.right_groups = weights["right_groups"]
        self.right_one = weights["right_one"]
        self.right_ibc = weights["right_ibc_stacked"]
        self.native_threads = 1
        self.work_bytes = _streaming.modal_kernels.FFT_BUILD_BUDGET

    def for_modes(self, lo, hi):
        """Share geometry/weights while selecting a retained modal band.

        m_max stays unchanged: it controls the angular sampling rule, while
        modes/orders only select its Fourier projections.
        """
        from copy import copy
        result = copy(self)
        result.modes = np.arange(int(lo), int(hi) + 1)
        result.orders = np.arange(max(0, int(lo) - 1), int(hi) + 2)
        return result

    def stream(self):
        """The attributes :func:`streaming._banded_stream` reads of a stream."""
        solver = SimpleNamespace(g=self.points, gauss_order=self.go, P=self.P,
                                 gen=SimpleNamespace(n_elems=self.ne),
                                 _near_sources_by_element=self.near)
        return SimpleNamespace(solver=solver, k=self.k, m_max=self.m_max,
                               _work_bytes=self.work_bytes, _workers=1,
                               _native_threads=self.native_threads)


class _Tiles:
    """Far node blocks ``[4, modes, |I|, |J|]`` of one surface from GHOST's tiles."""

    def __init__(self, geometry):
        self.geo = geometry
        self.stream = geometry.stream()
        self.pairs = 0

    def band(self, family, e0, e1, f0, f1, near_free):
        geo = self.geo
        go = geo.go
        rows = slice(e0 * go, e1 * go)
        re = e1 - e0
        self.pairs += re * (f1 - f0)
        if family == "efie":
            Gn = _streaming._banded_stream(self.stream, rows, "g", geo.orders, (f0, f1),
                                           near_free=near_free)
            left = geo.left_all[:, :, rows].reshape(2 * len(_streaming._LEFT_KINDS), re, go)
            return _streaming._efie_band(Gn, left, geo.right_groups, geo.modes, int(geo.orders[0]), geo.k,
                                         f0, f1, re, go)
        Fs = _streaming._banded_stream(self.stream, rows, family, geo.modes, (f0, f1),
                                       near_free=near_free)
        right = geo.right_one if family == "mfie" else geo.right_ibc
        return _streaming._bracket_band(Fs, geo.left_one[:, rows].reshape(2, re, go), right,
                                        f0, f1, re, go, None)

    def block(self, family, I, J, near_free=False):
        """Far part of the node block ``I x J`` (node ranges) as ``[4, modes, |I|, |J|]``."""
        e0, e1 = _element_span(I[0], I[1], self.geo.ne)
        f0, f1 = _element_span(J[0], J[1], self.geo.ne)
        band = self.band(family, e0, e1, f0, f1, near_free)
        return band[:, J[0] - f0:J[1] - f0, I[0] - e0:I[1] - e0, :].transpose(0, 3, 2, 1)


# ------------------------------------------------------------ compression

def _lowrank(stack, tolerance):
    """Truncated SVD ``(basis [p, r], coefficients [r, n])`` of a weighted stack, or None.

    None when the factors would not be smaller than the stack itself.
    """
    p, n = stack.shape
    try:
        u, s, vh = la.svd(stack, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        u, s, vh = la.svd(stack, full_matrices=False, check_finite=False, lapack_driver='gesvd')
    # Keep the smallest rank whose discarded tail is within the tolerance.
    tail = np.sqrt(np.cumsum((s ** 2)[::-1]))[::-1]
    rank = int(np.count_nonzero(tail > tolerance))
    if rank * (p + n) >= p * n:
        return None
    return np.ascontiguousarray(u[:, :rank]), s[:rank, None] * vh[:rank]


def _cross(tiles, family, I, J, weights, tolerance, rng, max_rank=None):
    """Partially pivoted cross approximation of a weighted admissible block stack.

    The stack is ``[p, 4 * modes * s]`` (slice ``(uv, m)`` scaled by
    ``weights[m]``).  Returns ``(U [p, r], V [r, 4 * modes * s])`` or None
    when the rank reaches ``max_rank``. After the usual stopping test, spread
    and random unused rows check the approximation; a miss becomes the next
    pivot. These probes strengthen detection of localized errors, but are not
    a deterministic full-block error certificate.
    """
    max_rank = FAR_COMPRESSION_MAX_RANK if max_rank is None else int(max_rank)
    p, s = I[1] - I[0], J[1] - J[0]
    nm = len(weights)
    scale = np.repeat(np.tile(weights, 4), s)
    width = 4 * nm * s
    U = np.zeros((p, max_rank), complex)
    V = np.zeros((max_rank, width), complex)
    rank = 0
    seen = np.zeros(p, bool)
    columns = {}

    def row(a):
        seen[a] = True
        return tiles.block(family, (I[0] + a, I[0] + a + 1), J, True).reshape(width) * scale

    def column(b):
        if b not in columns:
            columns[b] = tiles.block(family, I, (J[0] + b, J[0] + b + 1), True)[:, :, :, 0] \
                * weights[None, :, None]
        return columns[b]

    pivot = p // 2
    while True:
        residual = row(pivot) - U[pivot, :rank] @ V[:rank]
        c = int(np.argmax(np.abs(residual)))
        magnitude = abs(residual[c])
        converged = magnitude == 0.0
        if not converged:
            if rank == max_rank:
                return None
            uv, m, b = np.unravel_index(c, (4, nm, s))
            u = (column(b)[uv, m] - U[:, :rank] @ V[:rank, c]) / residual[c]
            U[:, rank], V[rank] = u, residual
            rank += 1
            converged = np.linalg.norm(u) * np.linalg.norm(residual) <= tolerance
        if converged:
            unseen = np.flatnonzero(~seen)
            if not unseen.size:
                break
            # Cover both edges and the interior before independent random
            # probes. A single random row can miss localized residuals.
            spread = np.unique(np.linspace(0, p - 1, 5, dtype=int))
            spread = spread[~seen[spread]]
            remaining = np.setdiff1d(unseen, spread, assume_unique=True)
            random = rng.choice(remaining, min(3, len(remaining)), replace=False)
            failed = None
            for candidate in np.r_[spread, random]:
                candidate = int(candidate)
                # Verification must not mark a row as a completed ACA pivot.
                probe = row(candidate) - U[candidate, :rank] @ V[:rank]
                seen[candidate] = False
                if np.linalg.norm(probe) * math.sqrt(p) > tolerance:
                    failed = candidate
                    break
            if failed is None:
                break
            pivot = failed
            continue
        scores = np.abs(U[:, rank - 1])
        scores[seen] = -1.0
        pivot = int(np.argmax(scores))
        if scores[pivot] < 0:
            break
    return U[:, :rank], V[:rank]


def _slices(U, V, nm, s, weights, tolerance, shared=False):
    """Per-(mode, uv) factors ``[[(P, W) x 4] x modes]`` of a joint approximation.

    ``U [p, r]`` and the weighted ``V [r, 4 * modes * s]`` share one skeleton
    for every family slice and mode; each slice needs far fewer directions
    (the joint rank serves the union), so its coefficients ``R V_(uv, m)``
    are truncated by one stacked SVD, each to ``tolerance / sqrt(4 modes)``
    of the weighted stack (the whole block then stays within twice the
    tolerance), and unweighted: ``block_(uv, m) = P @ W``.  A third of the
    joint form's storage on the 10 GHz ogive.
    """
    p = U.shape[0]
    empty = (np.zeros((p, 0), complex), np.zeros((0, s), complex))
    if U.shape[1] == 0:
        result = [[empty] * 4 for _ in range(nm)]
        return ("slices", result) if shared else result
    Q, R = la.qr(U, mode='economic', check_finite=False)
    coefficients = (R @ V).reshape(-1, 4, nm, s).transpose(2, 1, 0, 3)
    u, sv, vh = np.linalg.svd(coefficients, full_matrices=False)
    tails = np.sqrt(np.cumsum((sv ** 2)[..., ::-1], axis=-1))[..., ::-1]
    ranks = np.count_nonzero(tails > tolerance / math.sqrt(4 * nm), axis=-1)
    # Keep the small coefficients first. A shared Q is worthwhile only when
    # its retained bytes beat all expanded left factors; no rank/tolerance
    # changes or additional approximations are involved.
    out = []
    for m in range(nm):
        row = []
        for uv in range(4):
            k = int(ranks[m, uv])
            if not k:
                row.append((np.zeros((Q.shape[1], 0), complex), empty[1]))
                continue
            small = u[m, uv, :, :k] * (sv[m, uv, :k] / weights[m])
            row.append((np.ascontiguousarray(small), np.ascontiguousarray(vh[m, uv, :k])))
        out.append(row)
    small_bytes = Q.nbytes + sum(left.nbytes + right.nbytes for row in out for left, right in row)
    expanded_bytes = sum((p * left.shape[1] * 16) + right.nbytes for row in out for left, right in row)
    if shared and small_bytes < expanded_bytes:
        Q = np.ascontiguousarray(Q)
        Q.setflags(write=False)
        return ("shared", (Q, out))
    expanded = [[(np.ascontiguousarray(Q @ left), right) for left, right in row] for row in out]
    return ("slices", expanded) if shared else expanded


def _tile_job(tiles, family, I, J, weights, tolerance):
    """A sampled tile: dense ``[modes, 4, p, s]``, or compressed when it pays."""
    block = tiles.block(family, I, J, weights is not None)
    if weights is None:
        return ("dense", np.ascontiguousarray(block.transpose(1, 0, 2, 3)))
    p, s = block.shape[2], block.shape[3]
    nm = block.shape[1]
    stack = (block * weights[None, :, None, None]).transpose(2, 0, 1, 3).reshape(p, 4 * nm * s)
    factors = _lowrank(stack, tolerance)
    if factors is None:
        return ("dense", np.ascontiguousarray(block.transpose(1, 0, 2, 3)))
    return _slices(factors[0], factors[1], nm, s, weights, tolerance, shared=True)


def _cross_job(tiles, family, I, J, weights, tolerance, rng):
    factors = _cross(tiles, family, I, J, weights, tolerance, rng)
    if factors is None:
        return _tile_job(tiles, family, I, J, weights, tolerance)
    s = J[1] - J[0]
    return _slices(factors[0], factors[1], len(weights), s, weights, tolerance, shared=True)


def _run_job(tiles, job, scales, eps, n_nodes):
    """One block: ``(key, entry)``; ``scales`` None for the leaf phase."""
    kind, family, I, J = job
    if kind == "leaf":
        return (family, I, J), _tile_job(tiles, family, I, J, None, None)
    weights = 1.0 / scales[family]
    p, s = I[1] - I[0], J[1] - J[0]
    tolerance = eps * math.sqrt(p * s) / n_nodes
    rng = np.random.default_rng((I[0], J[0], p, s))
    if kind == "tile":
        return (family, I, J), _tile_job(tiles, family, I, J, weights, tolerance)
    return (family, I, J), _cross_job(tiles, family, I, J, weights, tolerance, rng)


# ------------------------------------------------------ process execution

_WORKER = {}


def _worker_initialize(geometry):
    from ghost_backend.bor.near_parallel import _initialize
    _initialize()
    _WORKER["tiles"] = _Tiles(geometry)


def _worker_run(arguments):
    jobs, scales, eps, n_nodes = arguments
    tiles = _WORKER["tiles"]
    start = tiles.pairs
    return [_run_job(tiles, job, scales, eps, n_nodes) for job in jobs], tiles.pairs - start


def _job_pairs(job):
    """Element pairs a job samples (its cost)."""
    kind, _, I, J = job
    p, s = I[1] - I[0], J[1] - J[0]
    if kind == "cross":
        return 2 * 24 * (p + s + 2)
    return (p + 1) * (s + 1)


def _process_workers():
    """Worker processes the enclosing near-preparation scope admits (0: none)."""
    from ghost_backend.bor.near_parallel import _POOL, process_backend_possible
    state = _POOL.get()
    if state is None:
        return 0
    count = int(state.get('process_workers') or 0)
    return count if count > 1 and process_backend_possible(count) else 0


class _Executor:
    """Runs job lists on spawn processes, threads or serially, in job order."""

    def __init__(self, geometry, workers, pairs):
        self.geometry = geometry
        self.total_work_bytes = geometry.work_bytes
        self.pool = None
        self.threads = 1
        self.pinned = False
        processes = _process_workers() if pairs >= FAR_COMPRESSION_PROCESS_PAIRS else 0
        if processes > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            from ghost_backend.execution.runtime import pin_worker_environment
            geometry.work_bytes = self.total_work_bytes / processes
            pin_worker_environment()
            self.pinned = True
            self.pool = ProcessPoolExecutor(max_workers=processes,
                                            mp_context=multiprocessing.get_context('spawn'),
                                            initializer=_worker_initialize, initargs=(geometry,))
            self.workers = processes
            self.backend = "processes"
        else:
            self.threads = max(1, int(workers))
            self.workers = self.threads
            geometry.work_bytes = self.total_work_bytes / self.threads
            self.backend = "threads" if self.threads > 1 else "serial"
        self.local = threading.local()
        self.pairs = 0

    def _tiles(self):
        tiles = getattr(self.local, "tiles", None)
        if tiles is None:
            tiles = self.local.tiles = _Tiles(self.geometry)
        return tiles

    def run(self, jobs, scales, eps, n_nodes, checkpoint):
        """``{key: entry}`` of every job; the heaviest jobs are dealt first."""
        jobs = sorted(jobs, key=_job_pairs, reverse=True)
        results = {}
        if self.pool is not None:
            from concurrent.futures import TimeoutError as FutureTimeout
            from concurrent.futures.process import BrokenProcessPool
            chunks = [chunk for chunk in (jobs[k::4 * self.workers] for k in range(4 * self.workers))
                      if chunk]
            futures = [self.pool.submit(_worker_run, (chunk, scales, eps, n_nodes)) for chunk in chunks]
            try:
                for future in futures:
                    while True:
                        checkpoint()
                        try:
                            entries, pairs = future.result(timeout=0.5)
                            break
                        except FutureTimeout:
                            continue
                    results.update(entries)
                    self.pairs += pairs
            except BrokenProcessPool:
                # A worker died (killed, out of memory): finish on threads.
                for future in futures:
                    future.cancel()
                self._to_threads()
                missing = [job for job in jobs if (job[1], job[2], job[3]) not in results]
                results.update(self.run(missing, scales, eps, n_nodes, checkpoint))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            return results
        from ghost_backend.execution.options import single_thread_blas

        def run_one(job):
            checkpoint()
            tiles = self._tiles()
            start = tiles.pairs
            entry = _run_job(tiles, job, scales, eps, n_nodes)
            return entry, tiles.pairs - start

        if self.threads <= 1:
            for job in jobs:
                (key, entry), pairs = run_one(job)
                results[key] = entry
                self.pairs += pairs
            return results
        from concurrent.futures import ThreadPoolExecutor
        with single_thread_blas(), ThreadPoolExecutor(max_workers=self.threads) as pool:
            for (key, entry), pairs in pool.map(run_one, jobs):
                results[key] = entry
                self.pairs += pairs
        return results

    def _to_threads(self):
        """Replace a broken process pool by the thread team of the same size."""
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self.threads = self.workers
        self.backend = "threads (process pool failed)"
        self.geometry.work_bytes = self.total_work_bytes / self.threads

    def close(self):
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if self.pinned:
            from ghost_backend.execution.runtime import release_worker_environment
            release_worker_environment()
            self.pinned = False


# ------------------------------------------------------------------- store

class _Family:
    """Presence marker for a built family (the solver tests ``stream.K is not None``)."""

    def __init__(self, name):
        self.name = name


def _closed_error():
    return RuntimeError(
        "Compressed far blocks were released by close(); build a new store.")


class CompressedFarBlocks:
    """Per-mode nodal far blocks of one BorPecSolver surface, as an H-matrix.

    The read interface of :class:`streaming.StreamingFarBlocks` (EFIE blocks
    without the ``C = jk eta 2 pi`` factor, MFIE and rotated-PV/IBC bracket
    blocks with the 2 pi Galerkin factor, signed modes by
    :func:`near_storage.mode_sign`). A bounded band is retained in RAM.
    Geometry, quadrature rules and the block partition are reused when the
    sweep advances. Compressed factors are never expanded into a dense spill.
    """

    def __init__(self, solver, m_max: 'int', efie: 'bool' = True,
                 mfie: 'bool' = False, ibc_zs_pt=None, pmchwt: 'bool' = False,
                 dtype=np.complex128, tile_budget_gb=None, workers: 'int' = 1,
                 mode_block=None, spill=None, tile_threads=None,
                 tolerance: 'float' = FAR_COMPRESSION_TOLERANCE):
        del spill, tile_threads
        from ghost_backend.bor.options import current_checkpoint
        self.solver = solver
        self.m_max = int(m_max)
        self.Nn = int(solver.Nn)
        self.k = complex(solver.k)
        self.rot_pv_unit_source = bool(pmchwt)
        has_ibc = ibc_zs_pt is not None or bool(pmchwt)
        families = [name for name, wanted in (("efie", efie), ("mfie", mfie), ("ibc", has_ibc))
                    if wanted]
        self.families = tuple(families)
        self.Z = _Family("efie") if efie else None
        self.K = _Family("mfie") if mfie else None
        self.B = _Family("ibc") if has_ibc else None
        self.tolerance = float(tolerance)
        # Single precision (table_precision='single') stores the factors as
        # complex64, like the streamed blocks; they are built in double.
        self.dtype = np.dtype(dtype)
        self.n_sweeps = 0
        self.lo, self.hi = 1, 0
        self.mode_block = _streaming._aligned_stream_mode_block(self.m_max, mode_block, workers)
        self._closed = False
        self._lock = threading.RLock()
        self._workers = max(1, int(workers))
        self._native = _streaming._NATIVE if abs(self.k.imag) == 0.0 else None
        self._geometry = _Geometry(solver, self.m_max, families, ibc_zs_pt, pmchwt)
        if tile_budget_gb is not None:
            self._geometry.work_bytes = min(self._geometry.work_bytes, float(tile_budget_gb) * 1e9)
        self.clusters = cluster_tree(self.Nn)
        self.admissible, self.dense = block_partition(
            self.clusters, solver.gen.nodes, solver._near_sources_by_element)
        self._blocks = {family: {} for family in families}
        self.evidence = dict(tolerance=self.tolerance, leaf=FAR_COMPRESSION_LEAF,
                             eta=FAR_COMPRESSION_ETA, admissible_blocks=len(self.admissible),
                             dense_blocks=len(self.dense))
        self._checkpoint = current_checkpoint() or (lambda: None)
        self._ensure(0)

    def _ensure(self, mode):
        with self._lock:
            if self._closed:
                raise _closed_error()
            if not 0 <= int(mode) <= self.m_max:
                raise ValueError('Requested mode exceeds the compressed far-store cap.')
            if self.lo <= mode <= self.hi:
                return
            lo = int(mode) // self.mode_block * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._blocks = {family: {} for family in self.families}
            self._build(self._geometry.for_modes(lo, hi), self._workers, self._checkpoint)
            self.lo, self.hi = lo, hi
            self.evidence.update(mode_range=[lo, hi], mode_block=self.mode_block,
                                 mode_cap=self.m_max, sweeps=self.n_sweeps)

    def _mode_store(self, family, mode):
        with self._lock:
            self._ensure(abs(int(mode)))
            return self._store(family), abs(int(mode)) - self.lo

    # -- build ---------------------------------------------------------
    def _span(self, index):
        lo, hi, _ = self.clusters[index]
        return int(lo), int(hi)

    def _jobs(self):
        leaves, blocks = [], []
        for family in self.families:
            for i, j in self.dense:
                I, J = self._span(i), self._span(j)
                if family != "efie" or I[0] <= J[0]:
                    leaves.append(("leaf", family, I, J))
            for i, j in self.admissible:
                I, J = self._span(i), self._span(j)
                if family == "efie" and I[0] > J[0]:
                    continue
                small = min(I[1] - I[0], J[1] - J[0]) < FAR_COMPRESSION_ACA_MIN_NODES
                blocks.append(("tile" if small else "cross", family, I, J))
        return leaves, blocks

    def _build(self, geometry, workers, checkpoint):
        import time
        start = time.perf_counter()
        leaves, blocks = self._jobs()
        pairs = sum(_job_pairs(job) for job in leaves + blocks)
        executor = _Executor(geometry, workers, pairs)
        try:
            phase = time.perf_counter()
            entries = executor.run(leaves, None, self.tolerance, self.Nn, checkpoint)
            self.evidence['leaf_seconds'] = time.perf_counter() - phase
            # Each family's scale per mode: the root sum of squares of its
            # near-diagonal leaves (an off-diagonal EFIE leaf stands for two).
            scales = {}
            nm = len(geometry.modes)
            for family in self.families:
                total = np.zeros(nm)
                for (fam, I, J), entry in entries.items():
                    if fam == family:
                        weight = 2.0 if family == "efie" and I != J else 1.0
                        total += weight * np.sum(np.abs(entry[1]) ** 2, axis=(1, 2, 3))
                scales[family] = np.sqrt(np.maximum(total, 1e-300))
            phase = time.perf_counter()
            entries.update(executor.run(blocks, scales, self.tolerance, self.Nn, checkpoint))
            self.evidence['block_seconds'] = time.perf_counter() - phase
            self.evidence.update(backend=executor.backend, workers=executor.workers,
                                 element_pairs_sampled=int(executor.pairs))
        finally:
            executor.close()
        for (family, I, J), entry in entries.items():
            if self.dtype != np.complex128:
                if entry[0] == "shared":
                    # Preserve the prior single-table rounding: form each left
                    # factor in double precision before casting it once.
                    basis, rows = entry[1]
                    entry = ("slices", [[(basis @ small, right) for small, right in row] for row in rows])
                entry = (("dense", entry[1].astype(self.dtype)) if entry[0] == "dense" else
                         ("slices", [[(left.astype(self.dtype), right.astype(self.dtype))
                                      for left, right in row] for row in entry[1]]))
            self._blocks[family][(I, J)] = entry
        ranks = [max(left.shape[1] for row in (entry[1][1] if entry[0] == "shared" else entry[1]) for left, _ in row)
                 for store in self._blocks.values() for entry in store.values()
                 if entry[0] in ("slices", "shared")]
        self.evidence.update(
            lowrank_blocks=len(ranks), max_rank=max(ranks, default=0),
            mean_rank=float(np.mean(ranks)) if ranks else 0.0,
            stored_gb=self.memory_gb(), build_seconds=time.perf_counter() - start)
        self.n_sweeps += 1

    # -- reads ---------------------------------------------------------
    def _store(self, family):
        if self._closed:
            raise _closed_error()
        store = self._blocks.get(family)
        if store is None:
            raise ValueError(f"Compressed far blocks were not built for {family!r}.")
        return store

    @staticmethod
    def _values(entry, mi):
        """The four ``[p, s]`` blocks of one stored block for mode index ``mi``."""
        if entry[0] == "dense":
            return entry[1][mi]
        if entry[0] == "shared":
            basis, rows = entry[1]
            return [(basis @ small) @ right for small, right in rows[mi]]
        return [left @ right if left.shape[1] else
                np.zeros((left.shape[0], right.shape[1]), left.dtype)
                for left, right in entry[1][mi]]

    def write_efie_blocks(self, m: 'int', quads, scale) -> 'None':
        """``quads[uv] = scale * mode_sign(uv, m) * Z_uv(m)`` for every node pair.

        The partition covers every node pair, so no quadrant is cleared
        first; lower blocks are the upper blocks' (negated, for tf/ft)
        transposes, as the streamed build completes them.
        """
        store, mi = self._mode_store("efie", m)
        factors = [scale * mode_sign(uv, m) for uv in range(4)]
        for (I, J), entry in store.items():
            values = self._values(entry, mi)
            rows, cols = slice(*I), slice(*J)
            for uv in range(4):
                np.multiply(values[uv], factors[uv], out=quads[uv][rows, cols])
            if I != J:
                np.multiply(values[0].T, factors[0], out=quads[0][cols, rows])
                np.multiply(values[3].T, factors[3], out=quads[3][cols, rows])
                np.multiply(values[2].T, -factors[1], out=quads[1][cols, rows])
                np.multiply(values[1].T, -factors[2], out=quads[2][cols, rows])

    def add_blocks(self, family: 'str', m: 'int', targets, scale) -> 'None':
        """``targets[uv] += scale * mode_sign(uv, m) * block`` for the MFIE
        ('mfie') or rotated-PV/IBC ('ibc') blocks of one signed mode."""
        if scale == 0:
            return
        store, mi = self._mode_store(family, m)
        factors = [scale * mode_sign(uv, m) for uv in range(4)]
        for (I, J), entry in store.items():
            values = self._values(entry, mi)
            rows, cols = slice(*I), slice(*J)
            for uv in range(4):
                target = targets[uv][rows, cols]
                if factors[uv] == 1:
                    target += values[uv]
                else:
                    target += values[uv] * factors[uv]

    def efie_blocks(self, m: 'int'):
        blocks = tuple(np.empty((self.Nn, self.Nn), dtype=np.complex128) for _ in range(4))
        self.write_efie_blocks(m, blocks, 1.0)
        return blocks

    def bracket_blocks(self, which: 'str', m: 'int'):
        blocks = tuple(np.zeros((self.Nn, self.Nn), dtype=np.complex128) for _ in range(4))
        self.add_blocks(which, m, blocks, 1.0)
        return blocks

    def stored_blocks(self, which: 'str', m: 'int'):
        """``(blocks, signs)`` of one signed mode as new full arrays (see the streamed store)."""
        signs = tuple(mode_sign(uv, m) for uv in range(4))
        if which == "efie":
            blocks = self.efie_blocks(m)
        else:
            blocks = self.bracket_blocks(which, m)
        return tuple(block * sign for block, sign in zip(blocks, signs)), signs

    def full_blocks(self, which: 'str' = "efie") -> 'np.ndarray':
        """Every mode of one family as a full ``[4, modes, Nn, Nn]`` array (for checks)."""
        out = np.empty((4, self.m_max + 1, self.Nn, self.Nn), dtype=np.complex128)
        for mi in range(self.m_max + 1):
            blocks = self.efie_blocks(mi) if which == "efie" else self.bracket_blocks(which, mi)
            for uv in range(4):
                out[uv, mi] = blocks[uv]
        return out

    def memory_gb(self) -> 'float':
        total = 0
        for store in self._blocks.values():
            for entry in store.values():
                if entry[0] == "dense":
                    total += entry[1].nbytes
                elif entry[0] == "shared":
                    basis, rows = entry[1]
                    total += basis.nbytes + sum(left.nbytes + right.nbytes for row in rows for left, right in row)
                else:
                    total += sum(left.nbytes + right.nbytes for row in entry[1] for left, right in row)
        return total / 1e9

    def spilled_gb(self) -> 'float':
        return 0.0

    def close(self) -> 'None':
        self._closed = True
        self._blocks = {}
        self._geometry = None
        self.Z = self.K = self.B = None
