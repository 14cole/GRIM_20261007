"""Checked HODLR inverse with exact-matrix residual refinement.

The dense factor compresses its off-diagonal blocks by adaptive randomized
sampling (:func:`compress_sampled`: a few large matrix products per block);
the row-by-row cross approximation :func:`compress` serves the compressed
backend's operator oracles, which cannot form a block.
"""
from ghost_backend.execution.options import environment_value
from ghost_backend.execution.errors import BackendNumericalError
import os
import warnings
import numpy as np
import scipy.linalg as la
from ghost_backend.linalg.workspace import matrix_inf_norm


class HierarchicalRejected(BackendNumericalError):
    pass


class HierarchicalCapacityRejected(HierarchicalRejected):
    """Rank/storage capacity exhausted; a stricter tolerance adds no capacity."""


def factor_mode():
    value = environment_value('GHOST_CPU_FACTORIZATION', 'dense').strip().lower()
    if value not in ('dense', 'hierarchical', 'auto', 'compressed'):
        raise ValueError('GHOST_CPU_FACTORIZATION must be dense, hierarchical, auto, or compressed.')
    return value


# From this many unknowns a dense double-precision system ('dense' and 'auto')
# is factored hierarchically, falling back to LU if the factor is rejected:
# the randomized build tied LU at 9,082 unknowns of the certified airfoil
# (4.2 against 4.7 s with three batches of 256 right-hand sides) and ran
# 1.7 times faster at 13,618 (8.0 against 13.7 s), with a factor of 182 MB
# against a 2,967 MB LU.  GHOST_HIERARCHICAL_MIN_UNKNOWNS overrides it
# (0 keeps LU).
HIERARCHICAL_MIN_UNKNOWNS = 10000


def automatic_hierarchical(n):
    """Whether a dense system of order ``n`` is factored hierarchically by default."""
    raw = os.environ.get('GHOST_HIERARCHICAL_MIN_UNKNOWNS', '').strip()
    if not raw:
        from ghost_backend.linalg.crossover import chosen
        measured=chosen(n)
        if measured is not None:return measured=='hodlr'
    threshold = int(raw) if raw else HIERARCHICAL_MIN_UNKNOWNS
    return threshold > 0 and int(n) >= threshold


def factor_storage_budget(matrix_bytes):


    return max(16*512**2, int(matrix_bytes*.65))


def spatial_order(coordinates, ids, leaf=128):
    """Spatial permutation without a recursive closure retaining coordinates."""
    if len(ids) <= leaf:
        return ids
    axis = int(np.argmax(np.ptp(coordinates[ids], axis=0)))
    ids = ids[np.argsort(coordinates[ids, axis], kind='mergesort')]
    mid = len(ids)//2
    return np.r_[spatial_order(coordinates, ids[:mid], leaf),
                 spatial_order(coordinates, ids[mid:], leaf)]


# A block gathers on one thread per this many entries (up to the dense-algebra threads).
GATHER_ENTRIES_PER_THREAD = 1 << 18
# Sampled compression retains factors, never a complete off-diagonal block.
# Each product gathers at most this much original-matrix data at a time.
PRODUCT_PANEL_BYTES = 16 * 1024**2


def _runs(ids):
    """``(offset, start, stop)`` of every maximal run of consecutive indices."""
    ids = np.asarray(ids)
    if not len(ids):
        return []
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(ids)]
    return [(int(a), int(ids[a]), int(ids[a]) + int(b - a)) for a, b in zip(starts, stops)]


class Block:
    def __init__(self, a, rows, cols, checkpoint):
        self.a, self.rows, self.cols, self.checkpoint = a, rows, cols, checkpoint
        self.shape = len(rows), len(cols)

    def dense(self):
        """The block as a new array.

        Clusters made of a few runs of consecutive indices are copied
        rectangle by rectangle.  Interleaved layouts (coincident unknowns of
        several densities) fragment every cluster into runs of one or two, and
        those blocks are gathered, in row chunks on the dense-algebra threads
        when large: a gather is latency bound (2.4 GB/s on one thread, 5.7 on
        eight for the 13,618-unknown airfoil's top block).
        """
        rows, cols = _runs(self.rows), _runs(self.cols)
        if len(rows)*len(cols) <= 256:
            fortran = self.a.flags.f_contiguous and not self.a.flags.c_contiguous
            out = np.empty(self.shape, self.a.dtype, order='F' if fortran else 'C')
            for r, i0, i1 in rows:
                for c, j0, j1 in cols:
                    out[r:r+i1-i0, c:c+j1-j0] = self.a[i0:i1, j0:j1]
            return out
        from ghost_backend.execution.options import blas_core_budget
        threads = min(blas_core_budget(), self.shape[0]*self.shape[1]//GATHER_ENTRIES_PER_THREAD)
        if threads <= 1:
            return self.a[np.ix_(self.rows, self.cols)]
        out = np.empty(self.shape, self.a.dtype)
        step = -(-self.shape[0]//(4*threads))

        def gather(start):
            out[start:start+step] = self.a[np.ix_(self.rows[start:start+step], self.cols)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(gather, range(0, self.shape[0], step)):
                pass
        return out

    def row(self, i):
        return self.a[self.rows[i], self.cols].copy()

    def col(self, j):
        return self.a[self.rows, self.cols[j]].copy()

    def matmul(self, rhs):
        """A_block @ rhs with bounded gathers and complete column reductions."""
        width = max(1, PRODUCT_PANEL_BYTES // max(1, self.shape[1]*self.a.itemsize))
        result = np.empty((self.shape[0], rhs.shape[1]), dtype=np.result_type(self.a, rhs))
        for start in range(0, self.shape[0], width):
            self.checkpoint()
            panel = Block(self.a, self.rows[start:start+width], self.cols, self.checkpoint).dense()
            result[start:start+width] = panel @ rhs
            del panel
        return result

    def project(self, basis):
        """basis.H @ A_block; each output column retains its full reduction."""
        width = max(1, PRODUCT_PANEL_BYTES // max(1, self.shape[0]*self.a.itemsize))
        result = np.empty((basis.shape[1], self.shape[1]), dtype=np.result_type(self.a, basis))
        # LAPACK/BLAS consumes the adjoint flag without conjugating the whole
        # basis. Its Fortran copy, if necessary, is only proportional to rank.
        basis = np.asfortranarray(basis)
        for start in range(0, self.shape[1], width):
            self.checkpoint()
            panel = Block(self.a, self.rows, self.cols[start:start+width], self.checkpoint).dense()
            result[:, start:start+width] = la.blas.zgemm(1., basis, panel, trans_a=2)
            del panel
        return result

    def error(self, u, v):
        total = error = largest = 0.
        pivot = 0
        for start in range(0, len(self.rows), 32):
            self.checkpoint()
            original = self.a[np.ix_(self.rows[start:start+32], self.cols)]
            total += float(np.vdot(original, original).real)
            original -= u[start:start+32] @ v
            norms = np.sum(np.abs(original)**2, axis=1)
            error += float(np.sum(norms))
            if len(norms) and norms.max() > largest:
                largest = float(norms.max())
                pivot = start+int(np.argmax(norms))
        return np.sqrt(error/max(total, 1e-300)), pivot


class OrderedBlock:
    """A block of the spatially ordered copy of the matrix: contiguous slices, no gathers.

    Same interface as :class:`Block` for the sampled compression (``shape``,
    ``dense``, ``matmul``, ``project``, ``checkpoint``); the products act on
    strided views (BLAS takes the leading dimension), so no panel copies.
    """
    def __init__(self, ordered, rows, cols, checkpoint):
        self.a, self.rows, self.cols, self.checkpoint = ordered, rows, cols, checkpoint
        self.shape = rows.stop - rows.start, cols.stop - cols.start

    def dense(self):
        return np.array(self.a[self.rows, self.cols], order='C')

    def matmul(self, rhs):
        self.checkpoint()
        return self.a[self.rows, self.cols] @ rhs

    def project(self, basis):
        self.checkpoint()
        return basis.conj().T @ self.a[self.rows, self.cols]


# A build copies the matrix into spatial order when the copy is at most this
# large and both the host and the solve reservation have headroom: interleaved
# unknown layouts fragment every block into runs of one or two entries, and
# gathering them was 47% of a build (4.2 N^2 entries at ~1.4 GB/s).  The copy
# is one threaded N^2 gather, released after the build.
ORDERED_COPY_MAX_BYTES = 8 << 30


def ordered_copy(matrix, permutation, checkpoint):
    """``matrix[perm][:, perm]`` as a contiguous array, or None when RAM does not allow it."""
    from ghost_backend.execution.options import blas_core_budget
    from ghost_backend.linalg.residual_spool import _COPY_LOCK, copy_fits
    if matrix.nbytes > ORDERED_COPY_MAX_BYTES:
        return None
    n = len(permutation)
    threads = min(blas_core_budget(), max(1, n * n // GATHER_ENTRIES_PER_THREAD))
    step = max(1, min(n, PRODUCT_PANEL_BYTES // max(1, n*matrix.itemsize)))
    # Share admission with LU copies. Materialize before the next worker
    # checks RSS, and bound each worker's advanced-indexing temporary.
    with _COPY_LOCK:
        if not copy_fits(matrix.nbytes, extra_bytes=threads*step*n*matrix.itemsize):
            return None
        checkpoint()
        out = np.empty((n, n), dtype=matrix.dtype)

        def gather(start):
            checkpoint()
            out[start:start+step] = matrix[np.ix_(permutation[start:start+step], permutation)]
        if threads <= 1:
            for start in range(0, n, step):
                gather(start)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for _ in pool.map(gather, range(0, n, step)):
                    pass
        return out


def compress(block, tolerance=2e-10, maximum_rank=256):
    m, n = block.shape
    limit = min(maximum_rank, min(m, n)//2)
    columns = np.empty((m, limit), complex, order='F')
    rows = np.empty((limit, n), complex)
    u, v = columns[:, :0], rows[:0]
    used = np.zeros(m, bool)
    pivot = 0
    error = np.inf
    accumulated = 0.
    def balanced(error):
        if not u.shape[1]:
            return u.copy(), v.copy(), error
        q, r = la.qr(u, mode='economic', check_finite=False)
        return q, r @ v, error
    for rank in range(limit):
        block.checkpoint()
        row = block.row(pivot) - u[pivot] @ v
        column = int(np.argmax(np.abs(row)))
        scale = row[column]
        if abs(scale) < 1e-290:
            error, pivot = block.error(u, v)
            if error <= tolerance:
                return balanced(error)
            row = block.row(pivot) - u[pivot] @ v
            column = int(np.argmax(np.abs(row)))
            scale = row[column]
            if abs(scale) < 1e-290:
                raise HierarchicalRejected('Unable to resolve an off-diagonal pivot.')
        col = block.col(column) - u @ v[:, column]
        used[pivot] = True
        columns[:, rank], rows[rank] = col/scale, row
        u, v = columns[:, :rank+1], rows[:rank+1]
        contribution = np.linalg.norm(columns[:, rank])*np.linalg.norm(row)
        accumulated += contribution
        candidate = np.abs(col)
        candidate[used] = -1
        pivot = int(np.argmax(candidate))


        if contribution <= tolerance*accumulated or rank+1 == limit:
            error, worst = block.error(u, v)
            if error <= tolerance:

                return balanced(error)
            pivot = worst
    raise HierarchicalRejected('Off-diagonal rank exceeds {} (relative error {:.3g}).'.format(limit, error))


# Products per step of the sampled range: large enough for GEMM speed, small
# enough that the fresh samples of the final (converged) step waste little.
SAMPLE_COLUMNS = 32


def compress_sampled(block, threshold, maximum_rank, rng, start=SAMPLE_COLUMNS):
    """Low-rank factors ``u, v`` using bounded block products (``u`` orthonormal).

    Adaptive randomized range finder: the range grows by SAMPLE_COLUMNS
    products at a time (``start`` the first time) until fresh samples,
    projected off it, all have norms below ``threshold`` times sqrt(2) (for
    complex Gaussian probes the mean squared norm is twice the squared
    Frobenius norm of what the range misses); a truncated SVD of the
    projected block then sets the rank.  Returns ``(u, v, error)``, ``error``
    the missed-norm estimate.
    """
    m, n = block.shape
    limit = min(int(maximum_rank), min(m, n)//2)
    q = np.empty((m, 0), dtype=np.complex128)
    width = max(1, min(int(start), n, max(limit, 1)))
    while True:
        block.checkpoint()
        y = block.matmul(rng.standard_normal((n, width)) + 1j*rng.standard_normal((n, width)))
        # What projection can resolve: rounding of the raw samples.
        floor = 64*np.finfo(float).eps*float(np.max(np.linalg.norm(y, axis=0)))
        if q.shape[1]:
            for _ in range(2):
                y -= q @ (q.conj().T @ y)
        error = float(np.max(np.linalg.norm(y, axis=0)))/np.sqrt(2.)
        if error <= max(threshold, floor):
            break
        if q.shape[1] + width > limit:
            raise HierarchicalCapacityRejected('Off-diagonal rank exceeds {} (sampled residual {:.3g}).'.format(limit, error))
        # Normalizing a residual far below its raw samples amplifies their
        # rounding: orthogonalize the new directions once more.
        extension = la.qr(y, mode='economic', check_finite=False)[0]
        if q.shape[1]:
            extension -= q @ (q.conj().T @ extension)
            extension = la.qr(extension, mode='economic', check_finite=False)[0]
        q = np.hstack((q, extension))
        width = min(SAMPLE_COLUMNS, n)
    if not q.shape[1]:
        return q, np.empty((0, n), q.dtype), error
    projected = block.project(q)
    # The wide SVD through the QR of its tall adjoint (projected = r^H w^H):
    # LAPACK's wide path took 1.6 to 2 times as long on these shapes.
    w, r = la.qr(projected.conj().T, mode='economic', check_finite=False)
    projected = None
    try:
        u, s, zh = la.svd(r.conj().T, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        u, s, zh = la.svd(r.conj().T, full_matrices=False, check_finite=False, lapack_driver='gesvd')
    rank = int(np.count_nonzero(s > threshold))
    return q @ u[:, :rank], s[:rank, None]*(zh[:rank] @ w.conj().T), error


class Node:
    def solve(self, b, trans=0):
        result = np.empty(np.shape(b), dtype=np.result_type(b, np.complex128), order='F')
        self.solve_into(b, result, trans)
        return result

    def solve_into(self, b, result, trans=0):
        """Reuse one RHS-sized destination throughout the inverse tree.

        Input and output may alias: adjoint corrections are formed before
        changing either half, and normal solves consume disjoint row ranges.
        Leaves only need a bounded LAPACK workspace if a slice is strided.
        """
        checkpoint = getattr(self, 'checkpoint', None)
        if checkpoint is not None:
            checkpoint()
        if self.leaf:
            result[...] = la.lu_solve(self.lu, b, trans=trans, check_finite=False)
            return
        n = self.left.n
        if self.lu is None:
            self.left.solve_into(b[:n], result[:n], trans)
            self.right.solve_into(b[n:], result[n:], trans)
            return
        if trans == 0:
            self.left.solve_into(b[:n], result[:n])
            self.right.solve_into(b[n:], result[n:])
            z1, z2 = result[:n], result[n:]
            small = np.vstack((self.v12 @ z2, self.v21 @ z1))
            correction = la.lu_solve(self.lu, small, check_finite=False)
            r = self.e1.shape[1]
            z1 -= self.e1 @ correction[:r]
            z2 -= self.e2 @ correction[r:]
            return
        def adj(a):
            return a.T if trans == 1 else a.conj().T
        small = np.vstack((adj(self.e1) @ b[:n], adj(self.e2) @ b[n:]))
        correction = la.lu_solve(self.lu, small, trans=trans, check_finite=False)
        r = self.e1.shape[1]
        result[:n] = b[:n]-adj(self.v21) @ correction[r:]
        result[n:] = b[n:]-adj(self.v12) @ correction[:r]
        self.left.solve_into(result[:n], result[:n], trans)
        self.right.solve_into(result[n:], result[n:], trans)


LEAF_SIZE = 256
# A node up to this size whose off-diagonal blocks do not compress becomes a
# dense leaf; above it the factor is rejected.
FALLBACK_LEAF_SIZE = 1024
SAMPLE_SEED = 20260925


class HierarchicalFactor:
    # Off-diagonal blocks are kept to TOLERANCE x ||A||_inf; a factor whose
    # refined solves do not converge is rebuilt once at TIGHT_TOLERANCE.  At
    # 1e-10 one refinement step reaches the backward-error gate (13,618
    # unknowns: 1e-9 took two steps, 1e-8 three, 1e-6 seven), and each step
    # costs a product with the exact matrix per right-hand-side batch.
    TOLERANCE = 1e-10
    TIGHT_TOLERANCE = 1e-12
    MAXIMUM_RANK = 1024

    def __init__(self, matrix, coordinates=None, checkpoint=None, matrix_norm=None):
        self.a = matrix
        self.checkpoint = checkpoint or (lambda: None)
        self.n = len(matrix)
        self.budget = factor_storage_budget(matrix.nbytes)
        self.root = None
        self.evidence = dict(backend='hodlr', compression='randomized', builds=0, tighter_rebuilds=0)
        coordinates = np.arange(self.n)[:, None] if coordinates is None else np.asarray(coordinates)
        self.permutation = spatial_order(coordinates, np.arange(self.n))
        self._position = np.empty(self.n, dtype=np.intp)
        self._position[self.permutation] = np.arange(self.n)
        # The transposed norm is only needed by transposed solves (condition estimates).
        self.norms = [matrix_inf_norm(matrix) if matrix_norm is None else matrix_norm, None]
        self.scale = float(self.norms[0])
        self.ordered = None
        from ghost_backend.linalg.residual_spool import reserve_factor_workspace
        with reserve_factor_workspace(self.budget):
            try:
                self.ordered = ordered_copy(matrix, self.permutation, self.checkpoint)
                self.evidence['ordered_copy'] = self.ordered is not None
                self.evidence['ordered_copy_bytes'] = 0 if self.ordered is None else self.ordered.nbytes
                retry = False
                try:
                    self._rebuild(self.TOLERANCE)
                except HierarchicalCapacityRejected as exc:
                    self.evidence['coarse_rejection'] = str(exc)
                    self.evidence['capacity_retry_skipped'] = True
                    raise
                except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
                    self.evidence['coarse_rejection'] = str(exc)
                    retry = True
                # Leave the exception handler before allocating the tighter
                # build: its traceback can retain the failed build's arrays.
                if retry:
                    self._rebuild(self.TIGHT_TOLERANCE)
            finally:
                self.ordered = None

    def _block(self, rows, cols):
        """The block of ``rows`` x ``cols`` (contiguous runs of the permutation) for the build."""
        if self.ordered is not None:
            row_start, col_start = int(self._position[rows[0]]), int(self._position[cols[0]])
            return OrderedBlock(self.ordered, slice(row_start, row_start + len(rows)),
                                slice(col_start, col_start + len(cols)), self.checkpoint)
        return Block(self.a, rows, cols, self.checkpoint)

    def _rebuild(self, tolerance):


        self.root = None
        self.bytes = 0
        self.tolerance = tolerance
        tight = tolerance <= self.TIGHT_TOLERANCE
        self.refinement_tolerance = 3e-14 if tight else 3e-15
        # One seeded stream per build: the same matrix always gets the same factor.
        self.rng = np.random.default_rng(SAMPLE_SEED)
        self.evidence['builds'] += 1
        self.evidence['tighter_rebuilds'] += int(tight and self.TOLERANCE > self.TIGHT_TOLERANCE)
        self.evidence.update(leaves=0, low_rank_blocks=0, max_rank=0,
            max_block_error=0., tolerance=tolerance, factor_bytes=0, refinements=0,
            refinement_tolerance=self.refinement_tolerance)
        self.root = self._build(self.permutation)
        self.rng = None
        self.evidence['factor_bytes'] = self.bytes

    def _reserve(self, count):
        self.bytes += int(count)
        if self.bytes > self.budget:
            raise HierarchicalCapacityRejected('Hierarchical factor exceeded its storage budget.')

    def _lu(self, matrix):
        self._reserve(matrix.nbytes + 8*len(matrix))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            return la.lu_factor(np.asarray(matrix, order='F'), overwrite_a=True, check_finite=False)

    def _build(self, ids):
        self.checkpoint()
        node = Node()
        node.checkpoint = self.checkpoint
        node.n, node.leaf = len(ids), len(ids) <= LEAF_SIZE
        if node.leaf:
            node.lu = self._lu(self._block(ids, ids).dense())
            self.evidence['leaves'] += 1
            return node
        mid = len(ids)//2
        left, right = ids[:mid], ids[mid:]
        threshold = self.tolerance*self.scale


        try:
            u12, node.v12, err1 = compress_sampled(self._block(left, right),
                                                   threshold, self.MAXIMUM_RANK, self.rng)
            # The transposed block has about the same rank: sample that many at once.
            u21, node.v21, err2 = compress_sampled(self._block(right, left),
                                                   threshold, self.MAXIMUM_RANK, self.rng,
                                                   start=max(SAMPLE_COLUMNS, u12.shape[1] + 8))
        except HierarchicalRejected:
            if node.n > FALLBACK_LEAF_SIZE:
                raise
            node.leaf = True
        if node.leaf:


            u12 = None
            node.v12 = node.v21 = None
            node.lu = self._lu(self._block(ids, ids).dense())
            self.evidence['leaves'] += 1
            return node
        r, s = u12.shape[1], u21.shape[1]
        self._reserve(u12.nbytes+u21.nbytes+node.v12.nbytes+node.v21.nbytes)
        self.evidence['low_rank_blocks'] += 2
        self.evidence['max_rank'] = max(self.evidence['max_rank'], r, s)
        self.evidence['max_block_error'] = max(self.evidence['max_block_error'],
                                               max(err1, err2)/max(self.scale, 1e-300))
        node.left, node.right = self._build(left), self._build(right)
        node.e1, node.e2 = node.left.solve(u12), node.right.solve(u21)
        small = np.eye(r+s, dtype=complex)
        small[:r, r:] = node.v12 @ node.e2
        small[r:, :r] = node.v21 @ node.e1
        node.lu = self._lu(small) if r+s else None
        return node

    def _norm(self, trans):
        index = int(trans != 0)
        if self.norms[index] is None:
            self.norms[index] = matrix_inf_norm(self.a.T)
        return self.norms[index]

    def _apply(self, b, trans):
        ordered = self.root.solve(b[self.permutation], trans)
        result = np.empty_like(ordered)
        result[self.permutation] = ordered
        return result

    def solve(self, rhs, trans=0, return_residual=False):
        if trans not in (0, 1, 2):
            raise ValueError('Invalid transpose mode.')
        b = np.asarray(rhs, complex)
        if b.ndim not in (1, 2) or b.shape[0] != self.n or (b.ndim == 2 and not b.shape[1]):
            raise ValueError('RHS must match the nonempty system.')
        if not np.all(np.isfinite(b)):
            raise ValueError('Nonfinite RHS.')
        try:
            return self._solve_refined(b, trans, return_residual)
        except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
            if self.tolerance <= self.TIGHT_TOLERANCE:
                raise
            self.evidence['coarse_rejection'] = str(exc)

        self._rebuild(self.TIGHT_TOLERANCE)
        return self._solve_refined(b, trans, return_residual)

    def _solve_refined(self, b, trans, return_residual):
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        x = self._apply(b, trans)

        def product(value):
            if trans == 2:
                return (self.a.T @ value.conj()).conj()
            return (self.a if trans == 0 else self.a.T) @ value
        for step in range(8):
            self.checkpoint()
            residual = b-product(x)
            denominator = self._norm(trans)*np.max(abs(x),axis=0)+np.max(abs(b),axis=0)
            error = float(np.max(np.max(abs(residual),axis=0)/np.maximum(denominator,1e-300)))
            if np.isfinite(error) and error <= self.refinement_tolerance:
                self.evidence['refinements'] = max(self.evidence['refinements'], step)
                solution = x[:, 0] if vector else x
                if return_residual:
                    return solution, residual[:, 0] if vector else residual
                return solution
            if step == 7:
                break
            x += self._apply(residual, trans)
        raise HierarchicalRejected('Exact-matrix residual refinement did not converge.')
