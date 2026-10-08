"""Bounded, incrementally shared QR incident basis with original-RHS checks."""
from ghost_backend.execution.options import environment_value
import math
import os
import threading
import weakref
import numpy as np
import scipy.linalg as la
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import first_nonfinite


def _frobenius_norm(value):


    flat = value.ravel(order='K')
    return float(np.sqrt(np.vdot(flat, flat).real))


def mode():
    value = environment_value('GHOST_CPU_RHS_COMPRESSION', 'auto').strip().lower()
    if value not in ('off', 'auto', 'on'):
        raise ValueError('GHOST_CPU_RHS_COMPRESSION must be off, auto, or on.')
    return value


# Cost model of 'auto' (measured: one LU solve of c columns costs about one
# original-matrix product A @ X; the pivoted QR of an n-by-c batch about
# AUTO_QR_FLOP_RATIO times its 2 n c^2 flops at GEMM speed).  A direct batch
# pays the LU solve plus its residual product (2 A@X); a reconstructed batch
# pays the QR, the LU solve and residual of its r new columns (2 r/c A@X) and
# the full original-matrix residual of every column (1 A@X).  Once the QR is
# done a rank below c/2 pays; before it, the QR must be cheaper than what
# even a rank-one batch could save.
AUTO_QR_FLOP_RATIO = 3.0
AUTO_MAX_RANK_FRACTION = 0.5
# LU-backed factors (dense LU, mirror halves) of at least LU_DIRECT_MIN_UNKNOWNS
# unknowns solve batches directly in 'auto' unless the system is very large
# for the batch: a LAPACK solve of r columns streams the whole factor once
# (about 60 ms for a 1.2 GB factor) whatever r is, and the pivoted QR costs
# about 2.4x its flop model, so at 8,608 unknowns a 256-column batch measured
# 0.563 s reconstructed against 0.533 s direct.  Reconstruction pays again once
# the factor is large against the batch (n above LU_DIRECT_RATIO columns), and
# always for compressed and hierarchical factors, whose solves scale with the
# column count.
LU_DIRECT_MIN_UNKNOWNS = 4096
LU_DIRECT_RATIO = 60


def _lu_backed(factor):
    """True for a factor whose solve is a LAPACK triangular solve of the whole LU."""
    if getattr(factor, 'hierarchical', None) is not None or getattr(factor, 'mixed', None) is not None:
        return False
    return getattr(factor, 'lu', None) is not None or getattr(factor, 'mirror', None) is not None


def _auto_qr_cannot_pay(n, count):
    """True when the pivoted QR alone costs more than any accepted batch could save."""
    qr_cost = AUTO_QR_FLOP_RATIO * 2.0 * count / float(n)      # in units of A @ X
    best_saving = 1.0 - 2.0 / float(count)                      # rank one, as A @ X
    return qr_cost >= best_saving


class CompressionHint:
    """Optional memory of the 'auto' RHS-compression outcome across factors.

    A caller that solves a family of related systems -- the modes of a BoR
    sweep, the frequencies or polarizations of a 2-D survey -- passes one hint
    to every :func:`solve` call of all their factors.  After a factor whose
    compression never paid (its first attempted batch fell back), the next
    factors skip the attempt and solve directly, re-probing after 1, 2, 4, ...
    (at most :attr:`PROBE_INTERVAL_MAX`) skipped factors; a factor that accepts
    a batch resets the backoff.  Thread-safe: concurrent mode workers share it.
    Settings 'on' and 'off' ignore it, and every correctness check is kept.
    """

    PROBE_INTERVAL_MAX = 16

    def __init__(self):
        self._lock = threading.Lock()
        self.factors = 0          # factors that reached a compression decision
        self.probed = 0           # ... of which attempted compression
        self.skipped = 0          # ... of which skipped it on this hint
        self.productive = 0       # factors with an accepted batch
        self.unproductive = 0     # factors whose attempt fell back first
        self._streak = 0
        self._skip_left = 0

    def begin_factor(self):
        """Whether a new factor attempts compression (asked once per factor)."""
        with self._lock:
            self.factors += 1
            if self._skip_left > 0:
                self._skip_left -= 1
                self.skipped += 1
                return False
            self.probed += 1
            return True

    def record(self, productive):
        """Outcome of one factor's first decisive batch."""
        with self._lock:
            if productive:
                self.productive += 1
                self._streak = 0
                self._skip_left = 0
            else:
                self.unproductive += 1
                self._streak += 1
                self._skip_left = min(2 ** (self._streak - 1), self.PROBE_INTERVAL_MAX)

    def evidence(self):
        with self._lock:
            return dict(factors=self.factors, probed=self.probed, skipped=self.skipped,
                        productive=self.productive, unproductive=self.unproductive)


def _qr_basis(value, threshold, max_rank=None):
    """Form only selected Q columns, avoiding a full D-by-batch Q."""
    (raw, tau), r, piv = la.qr(value, mode='raw', pivoting=True, check_finite=False)
    tail = np.sqrt(np.cumsum(np.sum(abs(r)**2, axis=1)[::-1])[::-1])
    rank = int(np.count_nonzero(tail > threshold))
    if max_rank is not None and rank > max_rank:
        return None, None
    if not rank:
        return np.empty((len(value), 0), complex), np.empty((0, value.shape[1]), complex)
    packed = np.array(raw[:, :rank], order='F', copy=True)
    ungqr = la.get_lapack_funcs('ungqr', (packed,))
    q, work, info = ungqr(packed, tau[:rank], overwrite_a=True)
    if info:
        raise la.LinAlgError('QR basis construction failed (ungqr {}).'.format(info))
    recovery = np.empty((rank, value.shape[1]), complex)
    recovery[:, piv] = r[:rank]
    return q, recovery


class SweepBasis:
    """Per-factor lifetime; retained basis and solutions never exceed one batch."""
    def __init__(self, capacity):
        self.capacity = max(1, min(256, int(capacity)))
        self.scale = self.q = self.x = None
        self.owner = None

    def bind(self, factor):

        if self.owner is None or self.owner() is not factor:
            self.owner = weakref.ref(factor)
            self.scale = self.q = self.x = None

    def reset(self, rhs):
        """Retain a bounded basis for the current angular neighborhood."""
        self.scale = np.max(abs(rhs), axis=1)
        self.scale[self.scale == 0] = 1.
        self.q = np.empty((len(rhs), 0), complex)
        self.x = np.empty((len(rhs), 0), complex)


def _sweep_qr(value, threshold, rank_limit, evidence, checkpoint):
    """Propose from sampled illuminations, accepting only a full-batch check."""
    if rank_limit <= 0:
        return None, None
    # Stop paying for proposals when they have mostly failed for this factor.
    # Full QR remains available for every batch, including after basis refresh.
    attempts = evidence.get('sampled_basis_attempts', 0)
    accepts = evidence.get('sampled_basis_accepts', 0)
    if value.shape[1] >= 256 and attempts <= 2*accepts:
        evidence['sampled_basis_attempts'] = attempts + 1
        ids = np.linspace(0, value.shape[1]-1, 64).astype(int)
        # A small subset has less energy than the complete batch. Use a tighter
        # proposal threshold so weak modes are not dropped before full checking.
        candidate, _ = _qr_basis(value[:, ids],
            .25*threshold*np.sqrt(len(ids)/float(value.shape[1])), max_rank=rank_limit)
        checkpoint()
        if candidate is not None and candidate.shape[1] < len(ids):
            recovery = candidate.conj().T @ value
            # The sampling proposes a span, never an angular interpolation.
            # Every requested illumination must satisfy the same QR threshold.
            difference = value - candidate @ recovery
            if _frobenius_norm(difference) <= threshold:
                evidence['sampled_basis_accepts'] = accepts + 1
                return candidate, recovery
        candidate = recovery = difference = None
    # Avoid forming Q when its rank cannot satisfy the existing savings/capacity
    # requirements. The full pivoted QR still determines that rank.
    return _qr_basis(value, threshold, max_rank=rank_limit)


def _evidence(factor, setting):
    return factor.event.setdefault('sweep_compression', dict(requested=setting,
        method='incremental_pivoted_qr', input_columns=0, solved_columns=0,
        accepted_batches=0, fallback_batches=0, fallback_columns=0,
        max_rhs_error=0., max_reconstructed_backward_error=0.,
        retained_basis_columns=0, reused_basis_batches=0))


def _auto_outcome(evidence, hint, productive):
    """First decisive 'auto' batch of a factor: suspend the factor when it did
    not pay, and tell the shared hint once."""
    if not productive and not evidence['accepted_batches']:
        evidence['auto_suspended'] = 'fallback'
    if hint is not None and evidence.get('hint_probe') and not evidence.get('hint_recorded'):
        evidence['hint_recorded'] = True
        hint.record(productive)


def _direct(factor, rhs, evidence, count, key=None):
    evidence['solved_columns'] += count
    if key is not None:
        evidence[key] = evidence.get(key, 0) + 1
    return factor.solve(rhs)


@timed_stage('rhs_compression')
def solve(factor, rhs, basis_state=None, setting=None, hint=None):
    """Solve ``factor`` for the columns of ``rhs``, reusing an incident basis.

    ``setting`` is 'off' (always direct), 'on' (always attempt compression)
    or 'auto' (default: :func:`mode`).  'auto' attempts compression only
    while it pays: a factor whose first attempted batch falls back is
    suspended and solves its later batches directly, the QR is skipped when
    the batch cannot gain (too few columns for the rank limit, or a system
    too small for the QR cost), and only a rank below half the batch is
    reconstructed.  ``hint`` (optional :class:`CompressionHint`, shared by
    the caller across the factors of one sweep) carries that outcome to later
    factors.  Every reconstructed column passes the original-matrix
    backward-error check, and a failed column is solved directly.
    """
    setting = mode() if setting is None else setting
    if setting not in ('off', 'auto', 'on'):
        raise ValueError('RHS compression must be off, auto, or on.')
    rhs = np.asarray(rhs, complex)
    if rhs.ndim != 2 or rhs.shape[0] != len(factor.a) or not rhs.shape[1]:
        raise ValueError('Sweep RHS must have one or more columns and match the system.')
    if first_nonfinite(rhs) is not None:
        raise ValueError('Nonfinite RHS')
    factor.checkpoint()
    n, count = rhs.shape
    evidence = _evidence(factor, setting)
    evidence['input_columns'] += count
    if setting == 'off' or (count < 32 and basis_state is None) or (setting == 'auto' and n < 512):
        evidence['solved_columns'] += count
        return factor.solve(rhs)
    auto = setting == 'auto'
    if auto:
        if evidence.get('auto_suspended'):
            return _direct(factor, rhs, evidence, count, 'suspended_batches')
        retained = (basis_state is not None and basis_state.q is not None and basis_state.q.shape[1] > 0
                    and basis_state.owner is not None and basis_state.owner() is factor)
        if not retained and LU_DIRECT_MIN_UNKNOWNS <= n < LU_DIRECT_RATIO * count and _lu_backed(factor):
            return _direct(factor, rhs, evidence, count, 'lu_direct_batches')
        if hint is not None and 'hint_probe' not in evidence:
            evidence['hint_probe'] = hint.begin_factor()
            if not evidence['hint_probe']:
                evidence['auto_suspended'] = 'hint'
                return _direct(factor, rhs, evidence, count, 'suspended_batches')
    state = basis_state if basis_state is not None else SweepBasis(count)
    state.bind(factor)
    if state.scale is None:
        state.reset(rhs)
    # Largest new rank a batch may add: the capacity, and a rank that still
    # saves solves (0.7 of the batch when forced, below half for 'auto').
    rank_fraction = AUTO_MAX_RANK_FRACTION if auto else .7
    if auto and not state.q.shape[1] and (
            min(state.capacity, int(np.ceil(rank_fraction*count))-1) < 1
            or _auto_qr_cannot_pay(n, count)):
        # Too small to gain: no retained span to project on, and no QR can pay.
        return _direct(factor, rhs, evidence, count, 'small_batches')
    with np.errstate(over='ignore', invalid='ignore'):
        scaled = rhs / state.scale[:, None]
        norm = max(_frobenius_norm(scaled), 1e-300)
    if not np.isfinite(norm):


        state.scale = state.q = state.x = None
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = None
        return factor.solve(rhs)
    existing = state.q.shape[1]
    recovery = state.q.conj().T @ scaled
    remainder = scaled - state.q @ recovery

    correction = state.q.conj().T @ remainder
    recovery += correction
    remainder -= state.q @ correction
    try:
        if existing and _frobenius_norm(remainder) <= 2e-15*norm:
            q, extension = np.empty((n, 0), complex), np.empty((0, count), complex)
        else:
            rank_limit = min(state.capacity-existing, int(np.ceil(rank_fraction*count))-1)
            if auto and (rank_limit < 1 or _auto_qr_cannot_pay(n, count)):
                # The retained span does not cover the batch and no QR can pay.
                scaled = remainder = correction = recovery = None
                return _direct(factor, rhs, evidence, count, 'small_batches')
            q, extension = _sweep_qr(remainder, 2e-15*norm, rank_limit, evidence, factor.checkpoint)
            if q is None and existing and count >= 32:
                # A broad sweep can exhaust a useful local span. Release it
                # before building a fresh span for the current batch, while
                # keeping the same factorization, capacity and error limits.
                evidence['basis_restarts'] = evidence.get('basis_restarts', 0) + 1
                scaled = remainder = correction = recovery = None
                state.reset(rhs)
                evidence['retained_basis_columns'] = 0
                scaled = rhs / state.scale[:, None]
                norm = max(_frobenius_norm(scaled), 1e-300)
                existing = 0
                recovery = np.empty((0, count), complex)
                remainder = scaled.copy()
                q, extension = _sweep_qr(remainder, 2e-15*norm,
                    min(state.capacity, int(np.ceil(rank_fraction*count))-1), evidence, factor.checkpoint)
    except (la.LinAlgError, ValueError):
        q = None
    if q is None or q.shape[1] + existing > state.capacity or q.shape[1] >= rank_fraction*count:
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        if auto:
            _auto_outcome(evidence, hint, False)
        return factor.solve(rhs)
    rank = q.shape[1]


    remainder -= q @ extension
    error = _frobenius_norm(remainder)/norm
    evidence['max_rhs_error'] = max(evidence['max_rhs_error'], error)
    if not np.isfinite(error) or error > 1e-14:
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        if auto:
            _auto_outcome(evidence, hint, False)
        return factor.solve(rhs)
    if not existing and not rank:

        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        return factor.solve(rhs)
    scaled = remainder = correction = None
    new_solution = factor.solve(q*state.scale[:, None]) if rank else np.empty((n, 0), complex)
    result = state.x @ recovery + new_solution @ extension
    evidence['solved_columns'] += rank

    zero = np.all(rhs == 0, axis=0)
    result[:, zero] = 0
    residual = factor.a @ result - rhs
    den = factor.matrix_inf*np.max(abs(result), axis=0) + np.max(abs(rhs), axis=0)
    errors = (factor.physical_errors(result,rhs,residual) if hasattr(factor,'physical_errors') else
              np.max(abs(residual), axis=0)/np.maximum(den, 1e-300))
    from ghost_backend.twod.constants import DENSE_LINEAR_BACKWARD_ERROR_MAX, EPS
    failed = ~np.isfinite(errors) | (errors > DENSE_LINEAR_BACKWARD_ERROR_MAX)
    backward = float(np.max(errors))
    evidence['max_reconstructed_backward_error'] = max(evidence['max_reconstructed_backward_error'], backward)

    if np.any(failed):
        evidence['fallback_batches'] += 1
        evidence['fallback_columns'] += int(np.sum(failed))
        evidence['solved_columns'] += int(np.sum(failed))
        result[:, failed] = factor.solve(rhs[:, failed])
        residual[:, failed] = factor.a @ result[:, failed] - rhs[:, failed]
        den = factor.matrix_inf*np.max(abs(result[:, failed]), axis=0)+np.max(abs(rhs[:, failed]), axis=0)
        errors[failed] = (factor.physical_errors(result[:,failed],rhs[:,failed],residual[:,failed])
                          if hasattr(factor,'physical_errors') else
                          np.max(abs(residual[:, failed]), axis=0)/np.maximum(den, 1e-300))
    if not np.all(np.isfinite(errors)) or np.max(errors) > DENSE_LINEAR_BACKWARD_ERROR_MAX:
        raise RuntimeError('Recovered sweep failed the original-matrix backward-error check.')
    if not np.any(failed):
        if rank and basis_state is not None:
            state.q = np.column_stack((state.q, q))
            state.x = np.column_stack((state.x, new_solution))
        evidence['accepted_batches'] += 1
        evidence['reused_basis_batches'] += int(existing > 0)
        if auto:
            _auto_outcome(evidence, hint, True)
    elif auto:
        _auto_outcome(evidence, hint, False)
    evidence['retained_basis_columns'] = state.q.shape[1]
    rhs_norm = np.linalg.norm(rhs, axis=0)
    factor.relative_residual = (factor.relative_errors(result,rhs,residual) if hasattr(factor,'relative_errors') else
        np.linalg.norm(residual, axis=0)/np.where(rhs_norm <= EPS, 1., rhs_norm))
    factor.event['max_backward_error'] = max(factor.event['max_backward_error'], float(np.max(errors)))
    factor.event['max_relative_residual'] = max(factor.event['max_relative_residual'], float(np.max(factor.relative_residual)))
    if factor.diagnostics is not None:
        factor.diagnostics['linear_backward_error'] = factor.event['max_backward_error']
    return result
