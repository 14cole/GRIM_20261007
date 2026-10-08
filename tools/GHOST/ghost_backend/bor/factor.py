"""One owned LU per BOR mode with bounded physical-RHS validation."""
import math
import numpy as np
from scipy.linalg import get_lapack_funcs
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import matrix_inf_norm, first_nonfinite, checked_matrix_norms


def residual_storage_settings():
    """(policy, directory) of the calling thread's execution options.

    Mode tasks run on executor threads, which do not inherit context
    variables, so the sweep resolves these once and hands them down."""
    from ghost_backend.execution.options import option, temporary_directory
    policy = option('dense_residual_storage', 'auto')
    try:
        directory = temporary_directory()
    except ValueError:
        if policy == 'disk':
            raise
        directory = None
    return policy, directory


def _residual_spool_selected(matrix, owned, policy):
    """Spool the original coefficients (``linalg.residual_spool``) instead of
    keeping them next to the LU: the same ``dense_residual_storage`` policy as
    the 2-D dense factor ('auto' checks host headroom and the remaining solve
    reservation; explicit reservations also protect systems below 512 MiB)."""
    from ghost_backend.linalg.residual_spool import auto_spooled
    eligible = (owned and isinstance(matrix, np.ndarray) and matrix.dtype == np.complex128
                and matrix.ndim == 2 and matrix.flags.owndata and matrix.flags.writeable
                and (matrix.flags.f_contiguous or matrix.flags.c_contiguous))
    return eligible and (policy == 'disk' or policy == 'auto' and auto_spooled(matrix.nbytes))


class MirrorSplit:
    """LU of the even and odd halves of a mirror-symmetric modal system.

    A body symmetric about a plane z = z0 maps its generatrix onto itself with
    the traversal reversed: node ``i`` to node ``Nn - 1 - i``, ``J_t`` changing
    sign and ``J_phi`` not.  With ``R`` that signed permutation of the reduced
    unknowns (``(R x)[u] = sign[u] x[target[u]]``) the system satisfies
    ``A = R A R`` up to quadrature (3e-8 on the 10 GHz ogive: near and far
    rules route a pair and its mirror image alike only to that accuracy), so
    it splits into the even and odd subspaces of ``R``: two systems of half
    the size, a quarter of the LU work, formed from ``A`` in O(n^2).  Their
    solution is an approximate inverse whose coupling error (1e-8 of the
    matrix) the caller's refinement against the exact ``A`` removes.
    """

    def __init__(self, matrix, target, sign):
        n = len(matrix)
        target = np.asarray(target, int)
        sign = np.asarray(sign, float)
        index = np.arange(n)
        if target.shape != (n,) or sign.shape != (n,) or np.any(target[target] != index):
            raise ValueError('The mirror map of a modal system must be an involution of its unknowns.')
        if np.any(sign[target] != sign) or np.any(np.abs(sign) != 1.0):
            raise ValueError('Mirror signs must be +-1 and shared by each mirrored pair.')
        paired = index < target
        self.p1, self.p2 = index[paired], target[paired]
        self.s = sign[self.p2]
        fixed = index == target
        self.fe, self.fo = index[fixed & (sign > 0)], index[fixed & (sign < 0)]
        self.n = n
        a, p1, p2, s = matrix, self.p1, self.p2, self.s
        k = len(p1)
        # Pair blocks: A11 + S A22 S and A12 S + S A21 (S = diag(s)).
        common = a[np.ix_(p1, p1)]
        common += s[:, None] * a[np.ix_(p2, p2)] * s[None, :]
        cross = a[np.ix_(p1, p2)] * s[None, :]
        cross += s[:, None] * a[np.ix_(p2, p1)]
        halves = []
        r = 1.0 / math.sqrt(2.0)
        for parity, fixed_points in ((1.0, self.fe), (-1.0, self.fo)):
            size = k + len(fixed_points)
            half = np.empty((size, size), dtype=complex, order='F')
            np.add(common, parity * cross, out=half[:k, :k])
            half[:k, :k] *= 0.5
            if len(fixed_points):
                half[:k, k:] = (a[np.ix_(p1, fixed_points)] + parity * s[:, None] * a[np.ix_(p2, fixed_points)]) * r
                half[k:, :k] = (a[np.ix_(fixed_points, p1)] + parity * a[np.ix_(fixed_points, p2)] * s[None, :]) * r
                half[k:, k:] = a[np.ix_(fixed_points, fixed_points)]
            halves.append(half)
        common = cross = None
        getrf, self.getrs = get_lapack_funcs(('getrf', 'getrs'), (halves[0],))
        self.factors = []
        for half in halves:
            lu, piv, info = getrf(half, overwrite_a=True)
            if info:
                raise np.linalg.LinAlgError('A mirror half of a BoR mode is singular (LAPACK info={}).'.format(info))
            self.factors.append((lu, piv))

    def solve(self, rhs, trans=0):
        """Approximate ``A^-1 rhs`` (``trans`` 2: ``A^-H``) from the two halves."""
        b = np.asarray(rhs, complex)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        p1, p2, s = self.p1, self.p2, self.s[:, None]
        r = 1.0 / math.sqrt(2.0)
        even = np.vstack(((b[p1] + s * b[p2]) * r, b[self.fe]))
        odd = np.vstack(((b[p1] - s * b[p2]) * r, b[self.fo]))
        solved = []
        for part, (lu, piv) in zip((even, odd), self.factors):
            value, info = self.getrs(lu, piv, part, trans=trans)
            if info:
                raise np.linalg.LinAlgError('A mirror half solve failed (LAPACK info={}).'.format(info))
            solved.append(value)
        k = len(p1)
        x = np.empty((self.n, b.shape[1]), dtype=complex)
        x[p1] = (solved[0][:k] + solved[1][:k]) * r
        x[p2] = s * (solved[0][:k] - solved[1][:k]) * r
        x[self.fe] = solved[0][k:]
        x[self.fo] = solved[1][k:]
        return x[:, 0] if vector else x


class ModalFactor:
    """LU of one modal system with original-coefficient residuals.

    ``owned=True`` lets a large system be factored in its own buffer: the
    original coefficients go to a disk spool for the residuals, so one mode
    holds one dense matrix instead of the matrix plus its LU copy.  The
    caller must not use ``matrix`` afterwards (``solve_am`` never does).

    With ``coordinates`` (the meridian position of every reduced unknown), a
    system of at least ``linalg.hierarchical.HIERARCHICAL_MIN_UNKNOWNS`` is
    factored hierarchically instead (the randomized HODLR of the 2-D dense
    factor, refined against the exact matrix to the same backward error):
    faster than LU there and a factor of a few percent of the matrix, so a
    mode worker holds the matrix and that factor instead of two matrices.  A
    rejected factor falls back to LU.

    With ``mirror = (target, sign)`` (the mirror map of a body symmetric about
    a plane normal to its axis, :class:`MirrorSplit`) the even and odd halves
    are factored instead, a quarter of the LU work; their solutions are
    refined against the exact matrix, and LU replaces them if that does not
    reach the backward-error gate.
    """

    def __init__(self, matrix, mode, monitor_cond, checkpoint=None, owned=False,
                 residual_storage=None, coordinates=None, mirror=None):
        self.a = matrix
        self.mode = mode
        self.checkpoint = checkpoint or (lambda: None)
        self.diagnostics = None
        self.checkpoint()
        first, self.matrix_inf, self.norm_1 = checked_matrix_norms(
            matrix, one_norm=monitor_cond, checkpoint=self.checkpoint)
        if first is not None:
            raise RuntimeError('BoR mode m={} produced a non-finite system matrix.'.format(mode))
        self.event = dict(factorizations=1, rhs_batches=0, max_rhs_columns=0,
                          max_backward_error=0., max_relative_residual=0., refinement_steps=0,
                          residual_storage='memory')
        self.monitor_cond = monitor_cond
        self._owned_matrix = owned
        self._residual_storage = residual_storage
        self.hierarchical = self.mirror = None
        self.lu = self.piv = None
        policy, directory = residual_storage or residual_storage_settings()
        if directory is not None and _residual_spool_selected(matrix, owned, policy):
            # Memory the plan did not foresee (or disk residuals requested):
            # factor in place with the original spooled, as before.
            mirror = coordinates = None
        if mirror is not None:
            self._factor_mirror(mirror)
        if self.mirror is None and coordinates is not None:
            from ghost_backend.linalg.hierarchical import automatic_hierarchical
            if automatic_hierarchical(len(matrix)):
                self._factor_hierarchical(coordinates)
        if self.hierarchical is None and self.mirror is None:
            self._factor_lu(owned, residual_storage)
        self._condition()

    def _factor_mirror(self, mirror):
        try:
            self.mirror = timed_stage('factorization')(MirrorSplit)(self.a, *mirror)
        except (np.linalg.LinAlgError, ValueError) as exc:
            self.event['mirror_fallback'] = str(exc)
            return
        self.event.update(backend='mirror', mirror_halves=[len(lu) for lu, _ in self.mirror.factors])

    def _factor_hierarchical(self, coordinates):
        from ghost_backend.linalg.hierarchical import HierarchicalFactor, HierarchicalRejected
        try:
            self.hierarchical = timed_stage('factorization')(HierarchicalFactor)(
                self.a, coordinates, self.checkpoint, self.matrix_inf)
        except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning, ValueError) as exc:
            self.event['hierarchical_fallback'] = str(exc)
            return
        self.event.update(backend='hodlr', hierarchical=self.hierarchical.evidence)

    def _fall_back_to_lu(self, reason):
        """Replace a hierarchical or mirror factor rejected by its refined solves with LU."""
        key = 'hierarchical_fallback' if self.hierarchical is not None else 'mirror_fallback'
        self.hierarchical = self.mirror = None
        self.event.update(backend='lu', **{key: str(reason)})
        self.event['factorizations'] += 1
        # An owned matrix may be spooled if LU's copy exceeds the reservation.
        self._factor_lu(self._owned_matrix, self._residual_storage)
        self._condition()

    def _condition(self):
        from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
        self.condition = math.nan
        if not self.monitor_cond:
            return
        norm = self.norm_1
        approximate = self.hierarchical if self.hierarchical is not None else self.mirror
        if approximate is not None:
            from scipy.sparse.linalg import LinearOperator
            from ghost_backend.twod.solver import _deterministic_onenormest
            n = len(self.a)
            inverse = LinearOperator((n, n), dtype=complex,
                                     matvec=lambda z: approximate.solve(z),
                                     rmatvec=lambda z: approximate.solve(z, trans=2))
            self.condition = norm * timed_stage('condition_estimate')(_deterministic_onenormest)(inverse)
        else:
            # ||A||_1 is the infinity norm of the factored A^T.
            reciprocal, info = timed_stage('condition_estimate')(self.gecon)(
                self.lu, norm, norm='I' if self.trans else '1')
            if info or not math.isfinite(float(reciprocal)) or reciprocal <= 0 or norm <= 0:
                raise RuntimeError('BoR mode m={} condition estimation failed.'.format(self.mode))
            self.condition = 1. / float(reciprocal)
        if not math.isfinite(self.condition) or self.condition > BOR_CONDITION_EST_MAX:
            raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                self.mode, self.condition, BOR_CONDITION_EST_MAX))

    def _factor_lu(self, owned, residual_storage):
        from ghost_backend.linalg.residual_spool import require_copy_capacity, copy_for_lu
        matrix, mode = self.a, self.mode
        getrf, self.getrs, self.gecon = get_lapack_funcs(('getrf', 'getrs', 'gecon'), (matrix,))
        # 0: LU of A (solve A x = b); 1: LU of A^T in a C-ordered buffer
        # (solve with the transposed factors).
        self.trans = 0
        spool = None
        policy, directory = residual_storage or residual_storage_settings()
        if directory is not None and _residual_spool_selected(matrix, owned, policy):
            from ghost_backend.linalg.residual_spool import ResidualSpool
            try:
                spool = ResidualSpool(matrix, directory, self.checkpoint)
            except OSError:
                if policy == 'disk':
                    raise
                require_copy_capacity(matrix.nbytes)
        if spool is not None:
            buffer = matrix if matrix.flags.f_contiguous else matrix.T
            self.trans = 0 if matrix.flags.f_contiguous else 1
            try:
                self.lu, self.piv, info = timed_stage('factorization')(getrf)(buffer, overwrite_a=True)
            except BaseException:
                spool.close()
                raise
            self.a = spool
            self.event.update(residual_storage='disk', original_matrix_disk_bytes=int(matrix.nbytes))
        else:
            # Keep A for original-coefficient residuals; LAPACK owns one F-order copy.
            lu_buffer = copy_for_lu(matrix)
            self.lu, self.piv, info = timed_stage('factorization')(getrf)(
                lu_buffer, overwrite_a=True)
        if info:
            raise RuntimeError('BoR mode m={} LU factorization failed (LAPACK info={}).'.format(mode, info))

    def close(self):
        """Release the residual spool (also released when the factor is collected)."""
        close = getattr(self.a, 'close', None)
        if close is not None:
            close()

    def inverse(self, rhs):
        self.checkpoint()
        if self.hierarchical is not None:
            from ghost_backend.linalg.hierarchical import HierarchicalRejected
            try:
                return timed_stage('rhs_solve')(self.hierarchical.solve)(rhs)
            except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
                reason = str(exc)
            # The traceback must release the failed factor before LU allocates.
            self._fall_back_to_lu(reason)
        if self.mirror is not None:
            return timed_stage('rhs_solve')(self.mirror.solve)(rhs)
        value, info = timed_stage('rhs_solve')(self.getrs)(self.lu, self.piv, rhs, trans=self.trans)
        if info:
            raise RuntimeError('BoR mode m={} LU solve failed (LAPACK info={}).'.format(self.mode, info))
        return value

    def errors(self, x, b, residual=None):
        if residual is None:
            residual = self.a @ x - b
        norms = np.linalg.norm(residual, axis=0)
        bnorms = np.linalg.norm(b, axis=0)
        relative = norms / np.where(bnorms > 0., bnorms, 1.)
        numerator = np.max(abs(residual), axis=0)
        denominator = self.matrix_inf * np.max(abs(x), axis=0) + np.max(abs(b), axis=0)
        backward = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
        backward[(denominator <= 0.) & (numerator > 0.)] = np.inf
        return residual, relative, backward

    def _refined(self, b, x, residual):
        """``(x, relative, backward)`` after up to two refinement steps against ``self.a``."""
        from ghost_backend.bor.solver import BOR_LINEAR_BACKWARD_ERROR_MAX, BOR_LINEAR_RESIDUAL_MAX
        if x is None:
            x = self.inverse(b)
        residual, relative, backward = self.errors(x, b, residual)
        # The mirror halves leave their coupling (1e-8) for refinement: allow
        # one more step than LU needs.
        for attempt in range(3 if self.mirror is not None else 2):
            if np.max(relative) <= BOR_LINEAR_RESIDUAL_MAX and np.max(backward) <= BOR_LINEAR_BACKWARD_ERROR_MAX:
                break
            candidate = x + self.inverse(-residual)
            updated = self.errors(candidate, b)
            if np.max(updated[1]) >= np.max(relative) and np.max(updated[2]) >= np.max(backward):
                break
            x = candidate
            residual, relative, backward = updated
            self.event['refinement_steps'] += 1
        return x, relative, backward

    def solve(self, rhs):
        from ghost_backend.bor.solver import BOR_LINEAR_BACKWARD_ERROR_MAX, BOR_LINEAR_RESIDUAL_MAX
        b = np.asarray(rhs, complex)
        if b.ndim != 2 or b.shape[0] != len(self.a) or not b.shape[1] or first_nonfinite(b) is not None:
            raise RuntimeError('BoR mode m={} produced an invalid or non-finite excitation.'.format(self.mode))
        x = residual = None
        if self.hierarchical is not None:
            from ghost_backend.linalg.hierarchical import HierarchicalRejected
            self.checkpoint()
            try:
                # The refined solve returns its final residual b - A x.
                x, residual = timed_stage('rhs_solve')(self.hierarchical.solve)(b, return_residual=True)
                np.negative(residual, out=residual)
            except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
                reason = str(exc)
            else:
                reason = None
            if reason is not None:
                self._fall_back_to_lu(reason)
                x = residual = None
        x, relative, backward = self._refined(b, x, residual)
        if self.mirror is not None and not (np.max(relative) <= BOR_LINEAR_RESIDUAL_MAX
                                            and np.max(backward) <= BOR_LINEAR_BACKWARD_ERROR_MAX):
            # The halves were too far from the exact system (an asymmetry
            # beyond quadrature): solve this and later batches with LU.
            self._fall_back_to_lu('refined mirror solve stopped at backward error {:.3g}'.format(
                float(np.max(backward))))
            x, relative, backward = self._refined(b, None, None)
        if (first_nonfinite(x) is not None or not np.all(np.isfinite(relative))
                or not np.all(np.isfinite(backward)) or np.max(backward) > BOR_LINEAR_BACKWARD_ERROR_MAX):
            raise RuntimeError('BoR mode m={} normwise linear backward error {} exceeds the release limit {}.'.format(
                self.mode, float(np.max(backward)), BOR_LINEAR_BACKWARD_ERROR_MAX))
        self.relative_residual = relative
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_backward_error'] = max(self.event['max_backward_error'], float(np.max(backward)))
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(relative)))
        return x


def compressed_storage_budget(options, workers):
    """Resolve the same per-worker cap for planning and factor construction."""
    from ghost_backend.compressed.runtime import automatic_storage_bytes
    configured = options['compressed_storage_mib']
    total = automatic_storage_bytes() if configured == 0 else configured * 1024**2
    return int(total) // max(1, int(workers))


def compressed_factor(oracle, mode, monitor_cond, options, workers, checkpoint=None,
                      recycling_key=None, recycling_frequency=None, exact_far_cache=False):
    from scipy.sparse.linalg import LinearOperator, onenormest
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.factor import CompressedFactor
    from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
    # 0 sizes the cap from the solve memory limit; a fixed cap starves an
    # electrically large body, whose modes each need their own share.
    budget = compressed_storage_budget(options, workers)
    from ghost_backend.bor.options import resolved_compression_tile
    tile = resolved_compression_tile(options, exact_far_cache)
    coordinates = oracle.row_coordinates
    if coordinates is None:
        coordinates = np.arange(oracle.n, dtype=float)[:, None]
    operator = timed_stage('modal_compressed_assembly')(StreamedOperator)(oracle, coordinates,
        tile=tile, budget=budget, checkpoint=checkpoint)
    factor = CompressedFactor(operator, label='BoR mode m={}'.format(mode), checkpoint=checkpoint,
                              storage_budget_bytes=budget, check_precision=False,
                              recycling_key=recycling_key, recycling_frequency=recycling_frequency,
                              recycling_coordinate_units='Hz')
    factor.event['compression_tile_requested'] = options['compression_tile']
    factor.event['compression_tile'] = tile
    factor.event['refinement_steps'] = 0
    factor.condition = math.nan
    if monitor_cond:
        inverse = LinearOperator(operator.shape, matvec=lambda z: factor.inverse(z),
                                 rmatvec=lambda z: factor.inverse(z, trans=2), dtype=complex)
        # Preserve BOR's unscaled 1-norm condition criterion.
        factor.condition = float(np.max(operator.column_norm + operator.column_error)) * float(onenormest(inverse))
        if not math.isfinite(factor.condition) or factor.condition > BOR_CONDITION_EST_MAX:
            raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                mode, factor.condition, BOR_CONDITION_EST_MAX))
    return factor
