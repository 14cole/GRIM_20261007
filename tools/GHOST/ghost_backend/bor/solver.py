"""Body-of-revolution RCS solves for PEC, IBC, dielectric and layered materials."""

import contextlib
import contextvars
import functools
import inspect
import itertools
import math
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from ghost_backend.execution.metrics import active_metrics, profiled_solve, timed_stage, metrics_scope
from scipy import special as sp
from scipy.linalg import get_lapack_funcs
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix, issparse, block_diag, bmat, diags

from ghost_backend.bor.kernels import (
    C0,
    ETA0,
    FFT_BUILD_BUDGET,
    NEAR_KERNEL_WORK_BYTES,
    N_XI_SAFETY_CAP,
    Generatrix,
    FAR_GAUSS_ORDER,
    cached_leggauss,
    gauss_on_generatrix,
    modal_kernels_fft,
    modal_kernels_near,
    kernels_for_mode,
    mfie_kernels_fft,
    mfie_kernels_near,
    mfie_for_mode,
    nonnegative_bracket_tables,
    ibc_kernels_fft,
    ibc_kernels_near,
    n_xi_for_pairs,
    physical_cpu_count,
)
from ghost_backend.bor.memory import (
    memory_gate_message as _memory_gate_message,
    solve_memory_limit_gb as _solve_memory_limit_gb,
)


from ghost_backend.bor.options import configured, current_options, bounded_rhs_count, compressed_requested, option_scope, current_checkpoint, BorAdmissionError
from ghost_backend.execution.runtime import ScopedValue


from ghost_backend.bor.tiled import TileExpression, primitive, mass_expression, modal_matrix, modal_block
from ghost_backend.bor.near_storage import mode_blocks, mode_sign
from ghost_backend.bor.preparation import prepared_surface as _prepared_surface
from ghost_backend.bor.preparation import prepared_cross as _prepared_cross
from ghost_backend.bor.streaming import sampling_backend_name


BOR_LINEAR_RESIDUAL_MAX = 1.0e-8
BOR_LINEAR_BACKWARD_ERROR_MAX = 1.0e-12
BOR_CONDITION_EST_MAX = 1.0e12


# Per concurrent mode worker: the reduced system matrix plus its LAPACK LU copy
# are retained for the mode's lifetime, and the in-place assembly of the
# conductor solvers (``BorPecSolver.assemble_mode`` writing straight into the
# system quadrants, ``assemble_mfie_mode`` accumulating in chunks, and the O(n)
# pole reduction) keeps the transient at one more matrix.  The former 8.0 dated
# from assembling every family into its own copy before combining them; it
# priced one worker at 5.4 GB on a 6,242-unknown system whose measured need is
# about 1.3 GB, which reserved most of the RAM for idle copies and pushed the
# automatic planner onto the compressed backend.
BOR_DENSE_MATRIX_EQUIVALENTS = 3.0
# A mode factored hierarchically (ModalFactor with coordinates, from
# linalg.hierarchical.HIERARCHICAL_MIN_UNKNOWNS unknowns) holds its system
# and a factor within 0.65 of it plus one off-diagonal block copy (a quarter)
# while it builds, or its system and an LU copy if it falls back.
BOR_HIERARCHICAL_MATRIX_EQUIVALENTS = 2.0
BOR_DENSE_RHS_EQUIVALENTS = 12.0
BOR_TABLE_BUILD_PEAK_FACTOR = 3.5
BOR_PEAK_SAFETY_FACTOR = 1.20
BOR_PEAK_FIXED_MARGIN_GB = 0.5
_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize


class _ConstraintEntries:
    """Builder of a pole/junction transform ``Q = S + E``.

    ``S`` selects the active full-system rows (one column each, value 1);
    ``E`` holds the few relations of inactive rows (axis-pole phi components,
    junction slaves) to their master columns.  Item assignment records ``E``
    with matrix semantics (the last value written to a position wins), so
    the existing ``Q[row, column] = value`` code builds it unchanged.
    """

    def __init__(self, n_rows: 'int', n_cols: 'int'):
        self.shape = (int(n_rows), int(n_cols))
        self.primary = np.full(int(n_cols), -1, dtype=np.intp)
        self.extra: 'Dict[Tuple[int, int], complex]' = {}

    def select(self, rows, columns) -> 'None':
        self.primary[np.asarray(columns, dtype=np.intp)] = np.asarray(rows, dtype=np.intp)

    def __setitem__(self, key, value) -> 'None':
        row, column = key
        self.extra[(int(row), int(column))] = complex(value)

    def tocsr(self):
        if np.any(self.primary < 0):
            raise ValueError("BoR constraint transform has a column without an active row.")
        extra = sorted(self.extra.items())
        extra_rows = np.array([rc[0] for rc, _ in extra], dtype=np.intp)
        extra_cols = np.array([rc[1] for rc, _ in extra], dtype=np.intp)
        extra_vals = np.array([value for _, value in extra], dtype=complex)
        if np.intersect1d(extra_rows, self.primary).size:
            raise ValueError("BoR constraint relations must act on inactive rows.")
        rows = np.concatenate([self.primary, extra_rows])
        cols = np.concatenate([np.arange(self.shape[1], dtype=np.intp), extra_cols])
        data = np.concatenate([np.ones(self.shape[1], dtype=complex), extra_vals])
        Q = csr_matrix((data, (rows, cols)), shape=self.shape)
        Q._ghost_constraint = (self.primary.copy(), extra_rows, extra_cols, extra_vals)
        return Q


def _constraint_structure(transform):
    """``(primary, extra_rows, extra_cols, extra_vals)`` with ``Q = S + E``.

    Transforms built by ``_ConstraintEntries`` carry it; otherwise it is
    inferred when unambiguous (every column has exactly one row whose only
    entry is 1, as in the block-diagonal pole transforms) or None.
    """
    explicit = getattr(transform, '_ghost_constraint', None)
    if explicit is not None:
        return explicit
    q = transform.tocsr() if issparse(transform) else csr_matrix(transform)
    n_rows, n_cols = q.shape
    counts = np.diff(q.indptr)
    rows = np.repeat(np.arange(n_rows), counts)
    single = counts[rows] == 1
    is_primary = single & (q.data == 1.0)
    primary_cols = q.indices[is_primary]
    if primary_cols.size != n_cols or np.unique(primary_cols).size != n_cols:
        return None
    primary = np.empty(n_cols, dtype=np.intp)
    primary[primary_cols] = rows[is_primary]
    extra = ~is_primary
    if np.intersect1d(rows[extra], primary).size:
        return None
    return primary, rows[extra], q.indices[extra].astype(np.intp), q.data[extra]


def _reduce_constrained_operator(matrix, transform):
    """``Q^H A Q`` for sparse pole/junction relations.

    With ``Q = S + E`` (selection plus a few relations of inactive rows) this
    is one gathered copy of ``A`` plus rank-one row/column updates -- the
    generalisation of ``BorPecSolver.reduce_pole_operator``.  The former
    sparse product copied ``A`` about three times (SciPy ravels F-ordered
    operands) and ran 6.7x slower at 2,400 unknowns.
    """
    if isinstance(matrix, TileExpression):
        return matrix.reduce(transform)
    structure = _constraint_structure(transform)
    if structure is None:
        q = csr_matrix(transform)
        return q.conj().T @ (q.T @ matrix.T).T
    primary, extra_rows, extra_cols, extra_vals = structure
    reduced = matrix[np.ix_(primary, primary)]
    for row, column, value in zip(extra_rows, extra_cols, extra_vals):
        reduced[:, column] += value * matrix[primary, row]
        reduced[column, :] += np.conj(value) * matrix[row, primary]
    for row, column, value in zip(extra_rows, extra_cols, extra_vals):
        for row2, column2, value2 in zip(extra_rows, extra_cols, extra_vals):
            reduced[column, column2] += np.conj(value) * value2 * matrix[row, row2]
    return reduced


def _constraint_coordinates(full_coordinates, transform):
    """Positions of reduced unknowns; used only to order a checked factor.

    Selecting each constraint's primary unknown preserves colocated electric
    and magnetic components. General sparse relations use a weighted centroid;
    neither choice changes matrix entries or the original-matrix checks.
    """
    structure = _constraint_structure(transform)
    if structure is not None:
        return np.asarray(full_coordinates)[structure[0]]
    weights = abs(csr_matrix(transform))
    totals = np.asarray(weights.sum(axis=0)).ravel()
    return (weights.T @ np.asarray(full_coordinates)) / np.maximum(totals[:, None], 1e-300)


def _rotate_test_rows(matrix, nodes):
    """Test n x field in (t, phi) coordinates: (-field_phi, field_t)."""
    return modal_block([[-matrix[nodes:, :]], [matrix[:nodes, :]]])


def _mix_dual_ibc_in_place(quads, alpha, scale, weights, chunk_rows: 'int' = 256) -> 'None':
    """Turn the EFIE quadrants ``T`` into ``alpha*T + scale*dual(T)*diag(w)``.

    ``dual(T) = [[T_ff, -T_ft], [-T_tf, T_tt]]`` is the EFIE operator acting on
    the magnetic current ``M = -Zs n x J`` of an impedance surface; ``w`` is the
    scalar impedance or the nodal impedance applied to the source columns.
    The quadrant pairs (tt, ff) and (tf, ft) are mixed through row chunks, so
    no second full matrix is ever formed (the former path held ``Z``, the dual
    copy and its column-scaled product at once).
    """
    tt, tf, ft, ff = quads
    columns = weights if np.ndim(weights) == 0 else np.asarray(weights)[None, :]
    rows = tt.shape[0]
    for start in range(0, rows, chunk_rows):
        piece = slice(start, min(rows, start + chunk_rows))
        a, d = tt[piece].copy(), ff[piece].copy()
        tt[piece] = alpha * a + scale * (d * columns)
        ff[piece] = alpha * d + scale * (a * columns)
        b, c = tf[piece].copy(), ft[piece].copy()
        tf[piece] = alpha * b - scale * (c * columns)
        ft[piece] = alpha * c - scale * (b * columns)


def _scaled_add_into(target, source, scale, chunk_rows: 'int' = 256):
    """``target += scale * source`` through bounded row chunks.

    ``source`` may alias read-only single-precision storage; only a
    ``chunk_rows``-row temporary is ever formed, so a full-matrix copy is
    never needed to combine one operator family into the system matrix.
    """
    if scale == 0:
        return
    rows = target.shape[0]
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        piece = source[start:stop]
        if scale == 1:
            target[start:stop] += piece
        else:
            target[start:stop] += piece * scale


# Material interfaces combine the equations of their two regions.  PMCHWT adds
# them with equal weights; on a lossless interface the exterior operator is
# then net capacitive and the interior one net inductive on a pole-localized
# phi-directed oscillation (its charge j*m*J_phi/rho involves no derivative),
# and their sum crosses zero at isolated real frequencies: the coarse discrete
# matrix is singular there (+6.7 dB on a lambda/10 sphere), although the
# continuous PMCHWT equations have no real resonance.  The weights are a
# numerical stabilization of that discretization.  Mautz and Harrington's
# uniqueness condition constrains the coefficients alpha and beta of ONE
# region's electric and magnetic equations (alpha*conj(beta) real and
# positive).  Both equations of a region share its weight here, so that
# product is |c_R|**2 = 1 whatever the phase, and the phase BETWEEN regions is
# free to make the reactive cancellation impossible.  Loss rotates the denser
# region's term the same way as a negative phase, so the denser region always
# receives the negative one.
BOR_INTERFACE_WEIGHT_PHASE_DEGREES = 15.0


def _region_equation_weights(media, exterior=0):
    """Unit-modulus weight of each region's equations, exterior normalized to 1.

    ``media`` holds ``None`` (air) or ``(eps_r, mu_r)`` per region.  Regions are
    ranked by ``Re(eps_r*mu_r)``; the phase is ``-theta`` per rank, so across
    every interface the denser side is the one rotated negatively.  Equal
    products share a rank and keep plain PMCHWT between them: their reactive
    parts have the same sign and cannot cancel.
    """
    density = [1.0 if medium is None else float(np.real(complex(medium[0]) * complex(medium[1])))
               for medium in media]
    levels = sorted({round(value, 12) for value in density})
    rank = [levels.index(round(value, 12)) for value in density]
    theta = math.radians(BOR_INTERFACE_WEIGHT_PHASE_DEGREES)
    return [complex(np.exp(-1j * theta * (r - rank[exterior]))) for r in rank]


def bor_impedance_map_bytes(nodes: 'int') -> 'float':
    """Bound on the sparse ``2N x 2N`` J-to-M impedance map of a bare IBC piece
    with ``nodes`` nodes: 2N complex nonzeros plus CSR indices, priced with
    64-bit indices so a preview never falls below the gate's exact size."""
    count = 2 * int(nodes)
    return float(count * (16 + 8) + (count + 1) * 8)


def _rotation_mass(solver):
    """``<W, n x f>`` in (t, phi) coordinates: ``[[0, -G], [G, 0]]`` (n x t = phi)."""
    gram = solver.mass_blocks()
    zero = modal_matrix((solver.Nn, solver.Nn), compressed_requested())
    return modal_block([[zero, -1.0 * gram], [gram, zero]])


def _add_rotation_mass_into(target, solver, scale):
    """``target + scale * _rotation_mass(solver)``, in place for a dense target.

    The Gram matrix is tridiagonal: its bands go straight into the two
    off-diagonal quadrants (bitwise the dense sum), instead of a dense
    ``2N x 2N`` rotation matrix (and a cached ``N x N`` Gram) per mode.
    Compressed operators keep the expression form.  Returns the result.
    """
    if not isinstance(target, np.ndarray) or solver._compressed:
        return target + scale * _rotation_mass(solver)
    bands = solver._unit_mass_bands
    if bands is None:
        bands = solver._unit_mass_bands = solver.mass_bands()
    n = solver.Nn
    solver._add_bands_into(target[:n, n:], bands, -scale)
    solver._add_bands_into(target[n:, :n], bands, scale)
    return target


def estimate_bor_dense_peak_gb(
    n_dofs: 'int',
    n_rhs: 'int',
    workers: 'int' = 1,
    mode_tasks: 'Optional[int]' = None,
    hierarchical: 'bool' = False,
    mirrored: 'bool' = False,
) -> 'float':
    """Conservative peak GB for the concurrent dense BoR linear systems.

    ``mode_tasks`` is the number of independent absolute-mode tasks that can
    actually be scheduled (normally ``m_max + 1``).  Capping the requested
    worker count by it accounts for real concurrency without charging for
    idle executor threads.  ``hierarchical`` tells that the sweep gives its
    mode factors coordinates, so a large enough system is priced as a
    hierarchical factor (:data:`BOR_HIERARCHICAL_MATRIX_EQUIVALENTS`), and
    ``mirrored`` that every mode is factored as mirror halves (the same
    price: the system, its two half systems and their assembly workspace).
    """

    try:
        dofs = int(n_dofs)
        rhs_count = int(n_rhs)
        worker_count = max(1, int(workers))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "BoR memory estimates require integer DOFs, RHS count, and workers."
        ) from exc
    if dofs != n_dofs or dofs <= 0:
        raise ValueError("BoR dense-system DOFs must be a positive integer.")
    if rhs_count != n_rhs or rhs_count <= 0:
        raise ValueError("BoR dense-system RHS count must be a positive integer.")
    if mode_tasks is None:
        active_workers = worker_count
    else:
        try:
            task_count = int(mode_tasks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "BoR mode-task count must be a positive integer."
            ) from exc
        if task_count != mode_tasks or task_count <= 0:
            raise ValueError("BoR mode-task count must be a positive integer.")
        active_workers = min(worker_count, task_count)

    # Bounded RHS batches do not bound the accumulated field or one full
    # contribution per concurrently scheduled mode.  The excitation records
    # shared by all modes (_AngularChunk) are held once for the sweep.
    field_bytes = (_COMPLEX128_BYTES * rhs_count * (active_workers + 1)
                   + angular_cache_bound_bytes(dofs, rhs_count))

    if compressed_requested():
        from ghost_backend.compressed.memory import inverse_storage
        options = current_options()
        from ghost_backend.bor.factor import compressed_storage_budget
        budget = compressed_storage_budget(options, active_workers)
        retained_inverse, inverse_peak = inverse_storage(dofs)
        operator_ceiling = 16*dofs*dofs + 128*dofs
        payload = min(budget, operator_ceiling + retained_inverse)
        rhs_workspace = 24 * dofs * bounded_rhs_count(rhs_count) * _COMPLEX128_BYTES
        # Numeric payload is capped; construction/FFT/RHS workspaces are additional.
        peak = max(payload + FFT_BUILD_BUDGET + 32*1024**2,
                   min(operator_ceiling, budget) + inverse_peak,
                   payload + rhs_workspace)
        return (active_workers * peak + options['tile_cache_mib'] * 1024**2 + field_bytes) / 1.e9

    from ghost_backend.linalg.hierarchical import automatic_hierarchical
    equivalents = (BOR_HIERARCHICAL_MATRIX_EQUIVALENTS
                   if mirrored or hierarchical and automatic_hierarchical(dofs)
                   else BOR_DENSE_MATRIX_EQUIVALENTS)
    matrix_bytes = (
        equivalents
        * dofs
        * dofs
        * _COMPLEX128_BYTES
    )
    rhs_count = bounded_rhs_count(rhs_count)
    rhs_bytes = (
        BOR_DENSE_RHS_EQUIVALENTS
        * dofs
        * rhs_count
        * _COMPLEX128_BYTES
    )


    per_worker_bytes = 1.05 * (matrix_bytes + rhs_bytes)
    return (active_workers * per_worker_bytes + field_bytes) / 1.0e9


def estimate_bor_total_peak_gb(
    assembly_peak_gb: 'float',
    dense_peak_gb: 'float',
) -> 'float':
    """Return the scheduler/runtime reservation for one BoR solve.

    ``assembly_peak_gb`` is already a *peak*, not persistent storage plus a
    second workspace allocation.  This distinction prevents double counting
    the retained tables when a caller has a more detailed build-workspace
    estimate, while still letting the direct all-mode table path pass its
    documented 3.5x construction peak.
    """

    assembly_peak = float(assembly_peak_gb)
    dense_peak = float(dense_peak_gb)
    if (
        not math.isfinite(assembly_peak)
        or assembly_peak < 0.0
        or not math.isfinite(dense_peak)
        or dense_peak < 0.0
    ):
        raise ValueError(
            "BoR peak-memory components must be finite and non-negative."
        )
    from ghost_backend.compressed.recycling import capacity_bytes
    # Optional inverses from completed frequencies remain live in either
    # phase. Current modal factors are priced separately above, per worker.
    raw_peak = assembly_peak + dense_peak + capacity_bytes() / 1.e9
    return max(
        BOR_PEAK_FIXED_MARGIN_GB,
        BOR_PEAK_FIXED_MARGIN_GB + BOR_PEAK_SAFETY_FACTOR * raw_peak,
    )


def _factor_pricing(hierarchical, mirrored):
    """The mode-factor keywords of :func:`estimate_bor_dense_peak_gb`, only when set."""
    return {name: True for name, value in (('hierarchical', hierarchical), ('mirrored', mirrored))
            if value}


def _guard_bor_dense_memory(
    n_dofs: 'int',
    n_rhs: 'int',
    workers: 'int',
    mode_tasks: 'int',
    assembly_peak_gb: 'float' = 0.0,
    context: 'str' = "The BoR solve",
    streaming: 'bool' = False,
    preparation_peak_gb: 'Optional[float]' = None,
    hierarchical: 'bool' = False,
    mirrored: 'bool' = False,
) -> 'float':
    """Gate a BoR solve before operator preparation and return required GB.

    ``preparation_peak_gb`` is the retained storage plus the near-integration
    scratch of the preparation phase.  That phase ends (its process pool is
    shut down) before any mode system is factored, so the requirement is the
    larger of the two phases, not their sum.

    A rejection is a ``BorAdmissionError`` (a ``MemoryError``) recording whether
    the rejected plan already streams, so an automatic call can tell it from an
    allocation failure while a solve runs and knows which smaller plan is left.
    """

    assembly_peak = float(assembly_peak_gb)
    if not math.isfinite(assembly_peak) or assembly_peak < 0.0:
        raise ValueError(
            "BoR assembly peak memory must be finite and non-negative."
        )
    preparation_peak = assembly_peak if preparation_peak_gb is None else float(preparation_peak_gb)
    if not math.isfinite(preparation_peak) or preparation_peak < assembly_peak:
        raise ValueError(
            "BoR preparation peak memory must be finite and at least the retained storage."
        )
    dense = estimate_bor_dense_peak_gb(
        n_dofs,
        n_rhs,
        workers=workers,
        mode_tasks=mode_tasks,
        **_factor_pricing(hierarchical, mirrored),
    )
    required = max(estimate_bor_total_peak_gb(assembly_peak, dense),
                   estimate_bor_total_peak_gb(preparation_peak, 0.0))
    memory_limit_gb = _solve_memory_limit_gb()
    if required > memory_limit_gb:
        active_workers = min(max(1, int(workers)), int(mode_tasks))
        raise BorAdmissionError(
            _memory_gate_message(
                required,
                memory_limit_gb,
                context,
                (
                    f"Planned dense system: {int(n_dofs)} complex128 DOFs, "
                    f"{int(n_rhs)} simultaneous RHS columns, "
                    f"{active_workers} concurrent mode worker"
                    f"{'s' if active_workers != 1 else ''}; estimated dense "
                    f"peak {dense:.2f} GB plus {assembly_peak:.2f} GB for "
                    "operator/table preparation, or "
                    f"{preparation_peak:.2f} GB while the near integration "
                    "runs, whichever phase is larger; the total includes the "
                    f"{BOR_PEAK_SAFETY_FACTOR:.2f}x safety factor and "
                    f"{BOR_PEAK_FIXED_MARGIN_GB:.2f} GB fixed margin."
                ),
                (
                    "Reduce the mesh, aspect count, or worker count; for the "
                    "direct PEC/IBC solver, streaming or a lower stream "
                    "budget can also reduce resident assembly memory."
                ),
            ),
            streaming=streaming,
            required_gb=required,
            mode_cap=int(mode_tasks) - 1,
        )
    return required


# Unplanned private preparation retains the historical conservative cap.
# Public solves reserve concurrent scratch against their actual RAM allocation.
NEAR_PREPARATION_SCRATCH_BYTES = 1 << 30
_NEAR_TASK_SCRATCH_BYTES = 3 * NEAR_KERNEL_WORK_BYTES
_NEAR_WORKER_LIMIT = ScopedValue('ghost_bor_near_workers', default=None)


def plan_near_preparation(workers, assembly_peak_gb, dense_peak_gb, memory_limit_gb=None,
                          near_pairs=None, mode_tasks=None):
    """Shared preview/runtime plan, including the full per-worker scratch bound.

    ``near_pairs`` (largest near-pair count of one preparation call) and
    ``mode_tasks`` let the plan apply the executor's own workload policy, so
    process overhead is charged only when process workers will be selected.
    ``process_workers`` is the pool size that fits including that overhead; the
    executor never exceeds it, whichever backend was reserved here.
    """
    from ghost_backend.execution.options import allocated_cpu_budget
    limit = _solve_memory_limit_gb() if memory_limit_gb is None else float(memory_limit_gb)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('BoR near-preparation memory limit must be positive and finite.')
    # Validate the other components through the same estimator as admission.
    estimate_bor_total_peak_gb(assembly_peak_gb, dense_peak_gb)
    # Near preparation completes, and its process pool is shut down, before
    # any mode system is factored, so its scratch shares the limit with the
    # retained operators alone: the mode workers' linear workspaces never
    # coexist with it and must not be charged against it.
    base = estimate_bor_total_peak_gb(assembly_peak_gb, 0.0)
    headroom = max(0.0, (limit - base) / BOR_PEAK_SAFETY_FACTOR)
    from ghost_backend.bor.near_parallel import (PROCESS_OVERHEAD_BYTES,
        process_backend_possible, processes_selected)
    cpu_budget = allocated_cpu_budget()

    def fit(worker_bytes):
        return max(1, min(int(workers), cpu_budget, int(headroom * 1.e9 / worker_bytes)))

    # Even a single worker needs scratch. If it cannot fit, normal admission
    # rejects the solve before preparation, instead of hiding that allocation.
    count, worker_bytes, process_bytes = fit(_NEAR_TASK_SCRATCH_BYTES), _NEAR_TASK_SCRATCH_BYTES, 0
    process_workers = 0
    if process_backend_possible(workers):
        process_workers = fit(_NEAR_TASK_SCRATCH_BYTES + PROCESS_OVERHEAD_BYTES)
        if process_workers <= 1:
            process_workers = 0  # Serial preparation runs in the owning interpreter.
    if process_workers and processes_selected(near_pairs, mode_tasks, process_workers):
        process_bytes = PROCESS_OVERHEAD_BYTES
        count, worker_bytes = process_workers, _NEAR_TASK_SCRATCH_BYTES + PROCESS_OVERHEAD_BYTES
    return dict(workers=count, scratch_gb=count * worker_bytes / 1.e9,
                scratch_bytes_per_worker=worker_bytes,
                process_overhead_bytes_per_worker=process_bytes,
                process_workers=process_workers,
                memory_limit_gb=limit)


def _near_preparation_workers(workers: 'int') -> 'int':
    planned = _NEAR_WORKER_LIMIT.get()
    if planned is not None:
        return max(1, min(int(workers), int(planned)))
    return max(1, min(int(workers),
                      NEAR_PREPARATION_SCRATCH_BYTES // _NEAR_TASK_SCRATCH_BYTES))


def plan_bor_mode_workers(n_dofs, n_rhs, workers, mode_tasks, assembly_peak_gb,
                          memory_limit_gb=None, near_pairs=None, hierarchical=False,
                          mirrored=False, preparation_workers=None):
    """Treat requested modal concurrency as a ceiling, sharing runtime admission.

    Keep the largest worker count within the unit's CPU allocation that fits,
    including near-integration scratch, linear-system workspaces and safety
    margins. If one worker still cannot fit, return that honest minimum so the
    normal guard rejects before preparation. ``preparation_workers`` preserves
    the caller's separate near-phase ceiling when a short modal cache band
    limits simultaneous mode systems; its full scratch is admitted separately.
    """
    from ghost_backend.execution.options import allocated_cpu_budget, physical_core_count
    requested = max(1, int(workers))
    preparation_requested = requested if preparation_workers is None else max(1, int(preparation_workers))
    cpu_budget = allocated_cpu_budget()
    if preparation_requested > 1:
        # The near phase is 64-86% of a conductor solve and its workers are
        # independent of the mode workers: a parallel request uses the CPU
        # allocation (the physical cores of a lone solve, the unit's share
        # under a driver), not the mode-worker count (8 instead of 4 workers
        # cut near preparation 20-24% on an 8-core host).  A serial request
        # (workers=1) stays serial.
        preparation_requested = max(preparation_requested, min(cpu_budget, physical_core_count()))
    limit = _solve_memory_limit_gb() if memory_limit_gb is None else float(memory_limit_gb)
    for count in range(min(requested, cpu_budget, max(1, int(mode_tasks))), 0, -1):
        linear_peak = estimate_bor_dense_peak_gb(n_dofs, n_rhs, count, mode_tasks,
                                                 **_factor_pricing(hierarchical, mirrored))
        near = plan_near_preparation(preparation_requested, assembly_peak_gb, linear_peak, limit,
                                     near_pairs=near_pairs, mode_tasks=mode_tasks)
        # The preparation phase (retained operators plus near scratch) and the
        # mode phase (retained operators plus linear workspaces) are
        # sequential: the peak is the larger of the two, as the gate prices it.
        peak = max(estimate_bor_total_peak_gb(assembly_peak_gb + near['scratch_gb'], 0.0),
                   estimate_bor_total_peak_gb(assembly_peak_gb, linear_peak))
        if peak <= limit:
            break
    return dict(requested_workers=requested, workers=count, cpu_budget=cpu_budget,
                requested_preparation_workers=preparation_requested,
                near_preparation=near,
                linear_peak_gb=linear_peak, estimated_peak_gb=peak,
                memory_limit_gb=limit, fits_memory=peak <= limit)


# Near pairs differ several-fold in cost (self cells hold 920 points, adjacent
# 260, disjoint 180), so a queue of exactly one task per worker, consumed in
# order, idles every worker that finishes ahead of the oldest task (simulated
# 39-49 % utilisation; measured 1.68-1.78x slower preparation with 8 and 15
# process workers). Queued tasks hold no scratch -- only running ones do, and
# the pool size bounds those -- and a result is a few 2x2 mode blocks, so a
# deeper in-order queue costs well under a megabyte.
NEAR_SUBMIT_DEPTH = 4
# Pairs per process task: each submit pickles the task with its generatrix
# (about 170 kB on a 3,000-element mesh), and a task integrates its pairs
# together (NearTask.run_batch), so batches amortise both.
NEAR_PROCESS_BATCH = 32
# Pairs integrated together on the thread and serial paths.
NEAR_LOCAL_BATCH = 32


def _run_near_batch_local(function, batch):
    return [function(pair) for pair in batch]


def _iter_near_pairs(function: 'Callable', pairs, workers: 'int', process_function=None, checkpoint=None):
    """Evaluate ``function`` per pair, in order, on a bounded pool.

    Results are yielded in pair order.  Up to ``NEAR_SUBMIT_DEPTH`` tasks per
    worker are queued so fast pairs never wait behind a slow one; at most
    one task per worker runs.  The first failure (including an abort raised
    by a checkpoint) cancels work that has not started and is re-raised.
    """

    pairs = list(pairs)
    count = min(_near_preparation_workers(workers), len(pairs))
    batched = process_function is not None and hasattr(process_function, 'run_batch')
    if count <= 1:
        if not batched:
            for pair in pairs:
                yield function(pair)
            return
        for start in range(0, len(pairs), NEAR_LOCAL_BATCH):
            if checkpoint is not None:
                checkpoint()
            for result in process_function.run_batch(pairs[start:start + NEAR_LOCAL_BATCH]):
                yield result
        return
    from ghost_backend.bor.near_parallel import executor_for, run_near_batch
    process_executor = (executor_for(len(pairs), process_function.m_max)
                        if process_function is not None else None)
    executor = process_executor or ThreadPoolExecutor(max_workers=count)
    window = count * NEAR_SUBMIT_DEPTH
    if process_executor is not None:
        size = max(1, min(NEAR_PROCESS_BATCH, len(pairs) // window))

        def submit(batch):
            return executor.submit(run_near_batch, process_function, batch)
    elif batched:
        size = max(1, min(NEAR_LOCAL_BATCH, len(pairs) // window))

        def submit(batch):
            return executor.submit(process_function.run_batch, batch)
    else:
        size = 1

        def submit(batch):
            return executor.submit(_run_near_batch_local, function, batch)
    batches = iter([pairs[i:i + size] for i in range(0, len(pairs), size)])
    from collections import deque
    pending = deque()
    # Local threads make their contractions on one BLAS thread each, as the
    # far tiles do (execution.options.single_thread_blas); process workers
    # are pinned to one thread already.
    blas = None
    if process_executor is None:
        from ghost_backend.execution.options import single_thread_blas
        blas = single_thread_blas()
        blas.__enter__()
    try:
        for batch in itertools.islice(batches, window):
            pending.append(submit(batch))
        while pending:
            if checkpoint is not None:
                checkpoint()
            future = pending.popleft()
            results = future.result()
            del future
            batch = next(batches, None)
            if batch is not None:
                pending.append(submit(batch))
            for result in results:
                yield result
    finally:
        for future in pending:
            future.cancel()
        if process_executor is None:
            executor.shutdown(wait=True, cancel_futures=True)
            blas.__exit__(None, None, None)


def _graded_cells(kind: 'str', depth: 'int' = 4) -> 'List[Tuple[float, float, float, float]]':
    """Cells (s0, s1, sp0, sp1) covering [0,1]^2 refined toward the singular
    set: kind = 'diag' (s == s'), 'corner00', 'corner01', 'corner10', 'corner11'
    where cornerAB means singular at s = A, s' = B."""

    cells = []

    def touches(kind, s0, s1, p0, p1):
        if kind == "diag":
            return not (s1 <= p0 or p1 <= s0)
        a = 0.0 if kind[6] == "0" else 1.0
        b = 0.0 if kind[7] == "0" else 1.0
        return (s0 <= a <= s1) and (p0 <= b <= p1)

    def recurse(s0, s1, p0, p1, d):
        if not touches(kind, s0, s1, p0, p1) or d >= depth:
            cells.append((s0, s1, p0, p1))
            return
        sm, pm = 0.5 * (s0 + s1), 0.5 * (p0 + p1)
        recurse(s0, sm, p0, pm, d + 1)
        recurse(s0, sm, pm, p1, d + 1)
        recurse(sm, s1, p0, pm, d + 1)
        recurse(sm, s1, pm, p1, d + 1)

    recurse(0.0, 1.0, 0.0, 1.0, 0)
    return cells


_CELL_CACHE: 'Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = {}

# Element pairs of two DIFFERENT surfaces that touch at a junction point.  The
# graded corner cells converge only linearly in the grading depth there: the
# former order-4, depth-4 rule left 1e-3 of relative far-field error in the
# partial-coating and layered-patch solvers (against an order-12, depth-8
# reference); order 10, depth 7 is within 2e-5.  A junction has only a few
# such pairs, so the ninefold point count costs nothing measurable.
JUNCTION_CELL_ORDER = 10
JUNCTION_CELL_DEPTH = 7


def _junction_cell_points(kind: 'str', refinement=None) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Graded cells for a cross-surface pair touching at corner ``kind``."""
    if refinement is None:
        from ghost_backend.bor.options import current_options
        refinement = current_options()['near_refinement']
    return _cell_points(kind, gorder=JUNCTION_CELL_ORDER, depth=JUNCTION_CELL_DEPTH + refinement)


def _cell_points(kind: 'str', gorder: 'int' = 4,
                 depth: 'int' = 4) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """(s_points, sp_points, weights) for the graded cell set of `kind`."""

    depth = max(0, int(depth))
    key = f"{kind}:{gorder}:{depth}"
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    xg, wg = cached_leggauss(gorder)
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg


    xq, wq_ = cached_leggauss(gorder + 1)
    uq = 0.5 * (xq + 1.0)
    wq = 0.5 * wq_
    S, SP, W = [], [], []
    for (s0, s1, p0, p1) in _graded_cells(kind, depth=depth):
        hs, hp = s1 - s0, p1 - p0
        ss = s0 + u * hs
        pp = p0 + uq * hp
        SS, PP = np.meshgrid(ss, pp, indexing="ij")
        WW = np.outer(w * hs, wq * hp)
        S.append(SS.ravel()); SP.append(PP.ravel()); W.append(WW.ravel())
    out = (np.concatenate(S), np.concatenate(SP), np.concatenate(W))
    _CELL_CACHE[key] = out
    return out


def _same_surface_points(gen, e: 'int', f: 'int', kinds, depth: 'int'):
    """Graded-cell quadrature of a same-surface self (e == f) or adjacent pair.

    One definition for the prepared contractions, the process workers and the
    per-pair fallbacks.  (A rule graded in |s - s'| only integrates the log
    self term ~1000x more accurately, but its end-to-end error against exact
    sphere series was no better at 10-40 elements per wavelength -- the cell
    rule's error partly offsets the flat-segment geometry error -- and it cost
    ~10% more, so the cell rule stays.)"""
    if e == f:
        return _cell_points("diag", depth=depth)
    return _cell_points("corner10" if f == e + 1 else "corner01", depth=depth)


TABLE_BLOCK_BYTES = 256 * 1024**2


def _blocked_single_tables(dtype) -> 'bool':
    """Whether reduced-precision far tables are built in row blocks.

    The banded builder treats every point pair independently, so a row block
    reproduces its rows of the whole-table build exactly."""
    from ghost_backend.bor import kernels as bor_kernels
    return np.dtype(dtype) != np.complex128 and bool(bor_kernels.BANDED_FFT)


def _tables_by_rows(build, rows: 'int', cols: 'int', row_bytes: 'float', dtype, near_mask):
    """Tables [rows, cols, width] of ``dtype`` from row blocks of
    ``build(r0, r1)`` (complex128 arrays), near pairs zeroed.

    Only one double block is alive: a single-precision table used to be built
    whole in double and then converted, peaking at three times the retained
    table (a cost the storage model never priced)."""
    outputs = None
    step = max(1, int(TABLE_BLOCK_BYTES // max(float(row_bytes), 1.0)))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        mask = near_mask[r0:r1]
        blocks = [np.asarray(block).reshape(r1 - r0, cols, -1) for block in build(r0, r1)]
        if outputs is None:
            outputs = [np.empty((rows, cols, block.shape[-1]), dtype=dtype) for block in blocks]
        for out, block in zip(outputs, blocks):
            block[mask] = 0.0
            out[r0:r1] = block
        del blocks
    return outputs


def _regular_cell_points(gorder: 'int' = 12) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Tensor Gauss points for close but nonsingular element pairs."""

    key = f"regular:{int(gorder)}"
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    xg, wg = cached_leggauss(int(gorder))
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    s, sp = np.meshgrid(u, u, indexing="ij")
    weights = np.outer(w, w)
    out = (s.ravel(), sp.ravel(), weights.ravel())
    _CELL_CACHE[key] = out
    return out


def _points_on_element(gen: 'Generatrix', e: 'int', s: 'np.ndarray'):
    n0, n1 = gen.elem_n0[e], gen.elem_n1[e]
    r0, r1 = gen.nodes[n0], gen.nodes[n1]
    rho = r0[0] + s * (r1[0] - r0[0])
    z = r0[1] + s * (r1[1] - r0[1])
    L = gen.lengths[e]
    T0, T1 = 1.0 - s, s
    drho = r1[0] - r0[0]
    dRT0 = (drho * (1.0 - s) - rho) / L
    dRT1 = (drho * s + rho) / L
    return rho, z, gen.trho[e], gen.tz[e], T0, T1, dRT0, dRT1, L


def _pair_blocks(m: 'int', k: 'float',
                 rho_p, tr_p, tz_p, T_p, D_p, w_p,
                 rho_q, tr_q, tz_q, T_q, D_q, w_q,
                 G, Gc, Gs):
    """
    The four per-mode Galerkin blocks for a set of weighted point pairs.

    T_p/D_p: [n_bases_p, n_pts] shape and (rho T)' matrices for the test side
    (likewise source side).  G/Gc/Gs: [n_pts_p, n_pts_q] kernels.  Weights
    include dt measures.  Returns (ztt, ztf, zft, zff) WITHOUT the C factor.
    """

    wp = w_p; wq = w_q
    rrw = (rho_p * wp)[:, None] * (rho_q * wq)[None, :]
    K_tt_vec = rrw * ((tr_p[:, None] * tr_q[None, :]) * Gc + (tz_p[:, None] * tz_q[None, :]) * G)
    K_sc = (wp[:, None] * wq[None, :]) * G
    K_tf_vec = rrw * (tr_p[:, None] * Gs)
    K_ft_vec = -rrw * (tr_q[None, :] * Gs)
    K_ff_vec = rrw * Gc

    ztt = T_p @ K_tt_vec @ T_q.T - (1.0 / k ** 2) * (D_p @ K_sc @ D_q.T)
    ztf = T_p @ K_tf_vec @ T_q.T - (1j * m / k ** 2) * (D_p @ K_sc @ T_q.T)
    zft = T_p @ K_ft_vec @ T_q.T + (1j * m / k ** 2) * (T_p @ K_sc @ D_q.T)
    zff = T_p @ K_ff_vec @ T_q.T - (m ** 2 / k ** 2) * (T_p @ K_sc @ T_q.T)
    return ztt, ztf, zft, zff


# Element-local Galerkin contraction of point-pair kernel tables.
#
# Each Gauss point touches the two nodal functions of its own element, so
# B_left K B_right^T needs only the [elements, points, 2] shape weights of each
# side: O(P^2) work instead of the O(Nn P^2) dense products with the Nn x P
# basis matrices, and the per-point factors (rho w, t_rho, t_z, w, the
# derivative basis) fold into those weights, so the P x P kernel combinations
# of a mode are never formed.  Rows are processed in bounded element chunks.
LOCAL_CONTRACTION_CHUNK_BYTES = 16_000_000


def _local_basis(g, n_elems: 'int', order: 'int', factor=None, derivative: 'bool' = False):
    """``[n_elems, order, 2]`` nodal shape (or (rho T)' ) weights times ``factor``."""
    first, second = (g.dRT0, g.dRT1) if derivative else (g.T0, g.T1)
    basis = np.stack([first, second], axis=1)
    if factor is not None:
        basis = basis * np.asarray(factor)[:, None]
    return basis.reshape(n_elems, order, 2)


def _local_rows(n_elems: 'int', order: 'int', columns: 'int') -> 'int':
    """Test elements per chunk so one complex row block stays bounded."""
    return max(1, min(int(n_elems), int(LOCAL_CONTRACTION_CHUNK_BYTES
                                        // max(1, order * columns * _COMPLEX128_BYTES))))


def _contract_right(kernel, right):
    """``[ce*o, neq*oq]`` kernel rows times ``[neq, oq, nb]`` source weights."""
    ce_o = kernel.shape[0]
    neq, oq, _ = right.shape
    K4 = np.asarray(kernel).reshape(ce_o, neq, oq)
    return np.einsum('pfc,fcb->pfb', K4, right, optimize=True)


def _contract_left_into(target, left, Y, e0: 'int', scale) -> 'None':
    """``target[e0+s, f+t] += scale * sum_a left[e, a, s] Y[e*o+a, f, t]``."""
    ce, o, _ = left.shape
    neq = Y.shape[1]
    A = np.einsum('eai,eafb->eifb', left, Y.reshape(ce, o, neq, 2), optimize=True)
    if scale != 1:
        A = A * scale
    target[e0:e0 + ce, 0:neq] += A[:, 0, :, 0]
    target[e0:e0 + ce, 1:neq + 1] += A[:, 0, :, 1]
    target[e0 + 1:e0 + ce + 1, 0:neq] += A[:, 1, :, 0]
    target[e0 + 1:e0 + ce + 1, 1:neq + 1] += A[:, 1, :, 1]


def _efie_tables_into(targets, m: 'int', k, gp, ne_p: 'int', op: 'int',
                      gq, ne_q: 'int', oq: 'int', table, scale) -> 'None':
    """``targets += scale * (ztt, ztf, zft, zff)`` of ``_pair_blocks`` for mode
    ``m`` straight from the ``[Pp, Pq, m_max+2]`` modal Green table."""

    am = abs(int(m))
    rw_p, rw_q = gp.rho * gp.w, gq.rho * gq.w
    lp = dict(rw_tr=_local_basis(gp, ne_p, op, rw_p * gp.trho),
              rw_tz=_local_basis(gp, ne_p, op, rw_p * gp.tz),
              rw=_local_basis(gp, ne_p, op, rw_p),
              w=_local_basis(gp, ne_p, op, gp.w),
              dw=_local_basis(gp, ne_p, op, gp.w, derivative=True))
    right_g = np.concatenate([_local_basis(gq, ne_q, oq, rw_q * gq.tz),
                              _local_basis(gq, ne_q, oq, gq.w, derivative=True),
                              _local_basis(gq, ne_q, oq, gq.w)], axis=2)
    right_c = np.concatenate([_local_basis(gq, ne_q, oq, rw_q * gq.trho),
                              _local_basis(gq, ne_q, oq, rw_q)], axis=2)
    right_s = np.concatenate([_local_basis(gq, ne_q, oq, rw_q),
                              _local_basis(gq, ne_q, oq, rw_q * gq.trho)], axis=2)
    inv_k2 = 1.0 / k ** 2
    jm = 1j * m
    tt, tf, ft, ff = targets
    chunk = _local_rows(ne_p, op, table.shape[1])
    for e0 in range(0, ne_p, chunk):
        e1 = min(ne_p, e0 + chunk)
        rows = slice(e0 * op, e1 * op)
        g_lo = np.asarray(table[rows, :, abs(am - 1)], dtype=np.complex128)
        g_0 = np.asarray(table[rows, :, am], dtype=np.complex128)
        g_hi = np.asarray(table[rows, :, am + 1], dtype=np.complex128)
        gc = 0.5 * (g_lo + g_hi)
        gs = (g_lo - g_hi) / 2j
        if m < 0:
            gs = -gs
        del g_lo, g_hi
        Yg = _contract_right(g_0, right_g)
        Yc = _contract_right(gc, right_c)
        Ys = _contract_right(gs, right_s)
        del g_0, gc, gs
        L = {name: value[e0:e1] for name, value in lp.items()}
        _contract_left_into(tt, L['rw_tr'], Yc[..., 0:2], e0, scale)
        _contract_left_into(tt, L['rw_tz'], Yg[..., 0:2], e0, scale)
        _contract_left_into(tt, L['dw'], Yg[..., 2:4], e0, -inv_k2 * scale)
        _contract_left_into(tf, L['rw_tr'], Ys[..., 0:2], e0, scale)
        _contract_left_into(tf, L['dw'], Yg[..., 4:6], e0, -jm * inv_k2 * scale)
        _contract_left_into(ft, L['rw'], Ys[..., 2:4], e0, -scale)
        _contract_left_into(ft, L['w'], Yg[..., 2:4], e0, jm * inv_k2 * scale)
        _contract_left_into(ff, L['rw'], Yc[..., 2:4], e0, scale)
        _contract_left_into(ff, L['w'], Yg[..., 4:6], e0, -(m * m) * inv_k2 * scale)


def _bracket_tables_into(targets, m: 'int', m_max: 'int', gp, ne_p: 'int', op: 'int',
                         gq, ne_q: 'int', oq: 'int', tables, scale,
                         source_weight=None) -> 'None':
    """``targets[uv] += scale * 2 pi (B_T w rho) K_uv(m) (B_T w rho src)^T`` for the
    four nonnegative-parity bracket tables (MFIE, or rotated-PV/IBC)."""

    left = _local_basis(gp, ne_p, op, gp.w * gp.rho)
    right_factor = gq.w * gq.rho if source_weight is None else gq.w * gq.rho * source_weight
    right = _local_basis(gq, ne_q, oq, right_factor)
    chunk = _local_rows(ne_p, op, tables[0].shape[1])
    for uv, target in enumerate(targets):
        table = tables[uv]
        # Production bracket tables hold the nonnegative orders 0..cap
        # (nonnegative_bracket_tables); negative modes follow by parity.
        index, sign = abs(int(m)), (-1.0 if uv in (1, 2) and m < 0 else 1.0)
        if index >= table.shape[-1]:
            raise ValueError("BoR bracket table was built for a smaller mode cap.")
        for e0 in range(0, ne_p, chunk):
            e1 = min(ne_p, e0 + chunk)
            rows = slice(e0 * op, e1 * op)
            Y = _contract_right(np.asarray(table[rows, :, index], dtype=np.complex128), right)
            _contract_left_into(target, left[e0:e1], Y, e0, 2.0 * np.pi * sign * scale)


def _bessel_triplet(m: 'int', u):
    """``(J_{m-1}(u), J_m(u), J_{m+1}(u))`` with two Bessel evaluations.

    The excitation of mode ``m`` needs three consecutive orders; the lowest
    follows from the downward recurrence ``J_{m-1} = (2m/u) J_m - J_{m+1}``,
    which is stable (no growth for ``u < m``; absolute accuracy ``eps`` in
    the oscillatory range, which is what the RHS combinations need).  At
    ``u = 0`` the exact limits are used.  Bessel evaluation was two thirds of
    the excitation cost (2.1 of 3.07 s per mode on a 3,120-element mesh).
    """
    u = np.asarray(u, dtype=float)
    n = abs(int(m))
    if n == 0:
        j_0 = sp.jv(0, u)
        j_hi = sp.jv(1, u)
        return -j_hi, j_0, j_hi
    # Orders n-1, n, n+1 of |m|: the recurrence runs DOWNWARD in order, the
    # stable direction.  Negative modes map by J_{-k} = (-1)^k J_k; running
    # the recurrence upward in |order| would amplify rounding for u < n.
    j_n = sp.jv(n, u)
    j_up = sp.jv(n + 1, u)
    with np.errstate(divide='ignore', invalid='ignore'):
        j_down = (2.0 * n / u) * j_n - j_up
    zero = u == 0.0
    if np.any(zero):
        j_down = np.where(zero, 1.0 if n == 1 else 0.0, j_down)
    if m > 0:
        return j_down, j_n, j_up
    sign = -1.0 if n % 2 else 1.0
    return -sign * j_up, sign * j_n, -sign * j_down


# Every mode of a sweep evaluates the same cylindrical waves at the same
# points and aspects, only at other Bessel orders.  One shared record per
# aspect chunk (_AngularChunk) keeps the mode-independent axial phase and
# J_T, J_{T+1} at a top order T above the requested modes, and each mode
# follows by the stable downward recurrence: 0.3 ns per value and order
# against 280-400 ns for one ``jv`` value, which two per mode made about
# 70 CPU-s of a certified 10 GHz ogive sweep.  Records are bounded by one
# process-wide budget, priced once in the mode-phase estimate.
ANGULAR_CACHE_BUDGET_BYTES = 256 * 1024**2
# u, 2/u, J_T, J_{T+1} (8 bytes each), phase (16), direct mask (1).
ANGULAR_BASE_BYTES_PER_VALUE = 49
# Retain a bounded number of intermediate recurrence states. Each state is
# two real arrays. Reusing the exact downward-recurrence state avoids starting
# at T for every mode; it does not change the recurrence or its arithmetic.
ANGULAR_CHECKPOINT_MAX_COUNT = 8
ANGULAR_CHECKPOINT_MIN_STEP = 16
# Conservative planner bound, including sin/cos(theta) (at most 16 bytes per
# point/aspect when there is only one point). Runtime reserves actual shapes.
ANGULAR_CACHE_BYTES_PER_VALUE = (
    ANGULAR_BASE_BYTES_PER_VALUE + 16 * ANGULAR_CHECKPOINT_MAX_COUNT + 16)
# Where |J_{T+1}(u)| is this small (u = 0 on the axis aspects, or tiny u
# against a high top order) the recurrence start has lost its precision;
# those values are evaluated directly.
ANGULAR_RECURRENCE_FLOOR = 1.0e-280
ANGULAR_MIN_TOP = 16
_ANGULAR_CACHE_LOCK = threading.Lock()
_ANGULAR_CACHE_USED = [0]


def angular_cache_bound_bytes(n_dofs: 'int', n_rhs: 'int') -> 'int':
    """Largest angular-cache footprint a sweep can hold (for the planners).

    A surface of ``n_dofs`` unknowns has at most ``2 n_dofs`` far points (four
    Gauss points per element, two unknowns per node); ``n_rhs`` counts both
    polarizations of every aspect.
    """
    values = 2 * int(n_dofs) * max(1, int(n_rhs) // 2)
    return min(ANGULAR_CACHE_BUDGET_BYTES, ANGULAR_CACHE_BYTES_PER_VALUE * values)


def _release_angular_bytes(nbytes: 'int') -> 'None':
    with _ANGULAR_CACHE_LOCK:
        _ANGULAR_CACHE_USED[0] -= int(nbytes)


def _reserve_angular_bytes(nbytes: 'int') -> 'bool':
    with _ANGULAR_CACHE_LOCK:
        if _ANGULAR_CACHE_USED[0] + int(nbytes) > ANGULAR_CACHE_BUDGET_BYTES:
            return False
        _ANGULAR_CACHE_USED[0] += int(nbytes)
        return True


class _AngularChunk:
    """Mode-independent cylindrical-wave data of one aspect chunk.

    ``u = k sin(theta) rho`` and the axial phase for every point and aspect,
    and J at the top orders ``T`` and ``T + 1``. Optional checkpoints retain
    the exact intermediate states of that recurrence, never new Bessel
    evaluations. Immutable once built; a request above ``T`` builds a new
    record. The caller reserves every checkpoint before construction.
    """

    def __init__(self, k, rho, z, thetas, top: 'int', checkpoint_orders=()):
        th = np.radians(thetas)
        self.st = np.sin(th)[:, None]
        self.ct = np.cos(th)[:, None]
        self.st[(np.asarray(thetas) == 0.0) | (np.asarray(thetas) == 180.0)] = 0.0
        # Real, as _bessel_triplet takes it (the excitation medium is air).
        u = np.asarray(k * self.st * rho[None, :], dtype=float)
        self.phase = np.exp(1j * k * self.ct * z[None, :])
        self.top = int(top)
        self.j_top = sp.jv(self.top, u)
        self.j_above = sp.jv(self.top + 1, u)
        direct = np.abs(self.j_above) < ANGULAR_RECURRENCE_FLOOR
        self.u = u
        # Keep the boolean mask itself. np.nonzero would retain two int64
        # index arrays (up to 16 bytes/value) despite reserving one byte.
        self.direct = direct if np.any(direct) else None
        self.two_over_u = 2.0 / np.where(direct, 1.0, u)
        # Directly evaluated values recur from zero (a tiny start would
        # overflow against the placeholder 2/u) and are replaced afterwards.
        self.j_top[direct] = 0.0
        self.j_above[direct] = 0.0
        orders = tuple(sorted(set(int(value) for value in checkpoint_orders)))
        if any(value <= 0 or value >= self.top for value in orders):
            raise ValueError("BoR angular checkpoints must lie strictly between zero and the top order.")
        self._checkpoints = {}
        if orders:
            pending = set(orders)
            ring = [np.empty_like(self.j_top) for _ in range(3)]
            high, current = self.j_above, self.j_top
            for order in range(self.top, orders[0], -1):
                target = ring[(order - 1) % 3]
                np.multiply(self.two_over_u, current, out=target)
                target *= float(order)
                target -= high
                high, current = current, target
                if order - 1 in pending:
                    self._checkpoints[order - 1] = (current.copy(), high.copy())
        self._checkpoint_orders = orders + (self.top,)
        retained = [u, self.two_over_u, self.j_top, self.j_above,
                    self.phase, self.st, self.ct]
        if self.direct is not None:
            retained.append(self.direct)
        retained.extend(value for pair in self._checkpoints.values() for value in pair)
        self.nbytes = int(sum(value.nbytes for value in retained))
        for value in retained:
            value.setflags(write=False)

    def triplet(self, m: 'int'):
        """``_bessel_triplet(m, u)`` by downward recurrence from the top orders."""
        n = abs(int(m))
        if n + 1 > self.top:
            raise ValueError("BoR angular record was built below the requested mode.")
        top = next(order for order in self._checkpoint_orders if order >= n + 1)
        j_top, j_above = self._checkpoints.get(top, (self.j_top, self.j_above))
        ring = [np.empty_like(self.j_top) for _ in range(3)]

        def order(o):
            if o == top + 1:
                return j_above
            if o == top:
                return j_top
            return ring[o % 3]
        # J_{o-1} = (2 o / u) J_o - J_{o+1}; a value never overwrites the two
        # it is computed from (they sit in other ring slots or the record).
        low = max(n - 1, 0)
        for o in range(top, low, -1):
            target = ring[(o - 1) % 3]
            np.multiply(self.two_over_u, order(o), out=target)
            target *= float(o)
            target -= order(o + 1)
        if n == 0:
            j_down, j_n, j_up = None, np.array(order(0)), np.array(order(1))
        else:
            j_down, j_n, j_up = (np.array(order(n - 1)), np.array(order(n)),
                                 np.array(order(n + 1)))
        if self.direct is not None:
            exact = _bessel_triplet(n, self.u[self.direct])
            if j_down is not None:
                j_down[self.direct] = exact[0]
            j_n[self.direct] = exact[1]
            j_up[self.direct] = exact[2]
        if n == 0:
            return -j_up, j_n, j_up
        if m > 0:
            return j_down, j_n, j_up
        sign = -1.0 if n % 2 else 1.0
        return -sign * j_up, sign * j_n, -sign * j_down


def _angular_checkpoint_orders(top: 'int', count: 'int'):
    """Evenly spaced recurrence checkpoints, bounded in number and density."""
    if count <= 0:
        return ()
    step = max(ANGULAR_CHECKPOINT_MIN_STEP, int(math.ceil(top / (count + 1))))
    return tuple(range(step, int(top), step))


def _causal_medium(eps_r: 'complex', mu_r: 'complex') -> 'Tuple[complex, complex]':
    """(m, eta_r) for a homogeneous medium: refractive index with Im(m) <= 0
    (causal decay, same branch as mie_sphere/_causal_index) and the
    relative impedance eta_r = mu_r / m, which guarantees k*eta = w mu mu0
    and k/eta = w eps eps0 exactly for whichever branch m took.

    For a passive double-negative medium the causal root has Re(m) < 0 and
    Im(m) < 0.  Do not subsequently force Re(m) positive: that selects the
    exponentially growing root.  In the exactly lossless case both roots
    have zero imaginary part, so choose the one with non-negative real wave
    impedance (forward power flow)."""

    eps_r = complex(eps_r)
    mu_r = complex(mu_r)
    if not (
        math.isfinite(eps_r.real)
        and math.isfinite(eps_r.imag)
        and math.isfinite(mu_r.real)
        and math.isfinite(mu_r.imag)
    ):
        raise ValueError("BoR medium epsilon and mu must be finite.")
    singular_tol = 1.0e-15
    if abs(eps_r) <= singular_tol:
        raise ValueError(
            "BoR PMCHWT does not support singular/near-ENZ epsilon."
        )
    if abs(mu_r) <= singular_tol:
        raise ValueError(
            "BoR PMCHWT does not support singular/near-MNZ mu."
        )
    eps_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(eps_r))
    mu_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(mu_r))
    if eps_r.imag > eps_tol or mu_r.imag > mu_tol:
        raise ValueError(
            "BoR PMCHWT supports passive media only. Under the "
            "exp(+j*omega*t) convention, Im(epsilon) and Im(mu) must be <= 0."
        )

    m = np.sqrt(eps_r * mu_r)
    branch_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(m))
    if m.imag > branch_tol:
        m = -m
    elif abs(m.imag) <= branch_tol:
        eta_try = complex(mu_r) / m
        if eta_try.real < 0.0:
            m = -m
    return m, complex(mu_r) / m


def _validate_bor_surface_impedance(values, context: 'str') -> 'np.ndarray':
    """Return finite passive Leontovich impedances for exp(+j omega t)."""

    array = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{context} contains a non-finite surface impedance.")
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(array))
    if np.any(array.real < -tolerance):
        bad = complex(array.flat[int(np.flatnonzero(array.real < -tolerance)[0])])
        raise ValueError(
            f"{context} contains active negative resistance {bad.real:g} ohm; "
            "passive BoR IBCs require Re(Zs) >= 0."
        )
    return array


EFFECTIVELY_REACTIVE_IBC_RATIO = 1.0e-3


# Share of the impedance surface (by area when weights are given) that may be
# effectively lossless before a closed EFIE solve is refused.
EFFECTIVELY_REACTIVE_AREA_FRACTION = 0.5


def _effectively_reactive_surface_impedance(values, weights=None) -> 'bool':
    """Whether a nonzero IBC is effectively reactive (not an EFIE safety test).

    Each nonzero value is judged on its own (``|Re Zs| < ratio * |Zs|``) and
    the surface counts as reactive when those values cover at least half of
    the impedance area (``weights``, e.g. Gauss weights times radius; equal
    weights otherwise).  The former test compared the largest resistance
    with the largest magnitude, so one resistive element made a mostly
    lossless closed body pass.
    """

    array = np.asarray(values, dtype=np.complex128).ravel()
    if array.size == 0:
        return False
    magnitude = np.abs(array)
    nonzero = magnitude > 0.0
    if not np.any(nonzero):
        return False
    w = (np.ones(array.size) if weights is None
         else np.abs(np.asarray(weights, dtype=float).ravel()))
    if w.shape != array.shape:
        raise ValueError("Impedance weights must match the impedance values.")
    reactive = nonzero & (np.abs(array.real) < EFFECTIVELY_REACTIVE_IBC_RATIO * magnitude)
    total = float(np.sum(w[nonzero]))
    if total <= 0.0:
        return False
    return float(np.sum(w[reactive])) >= EFFECTIVELY_REACTIVE_AREA_FRACTION * total


class BorPecSolver:
    """Single-surface BoR operator factory + PEC/IBC solver.

    With medium=(eps_r, mu_r) the EFIE (T) and rotated-PV (P) operators are
    assembled in that homogeneous medium (complex k, medium eta) -- the
    building blocks of the PMCHWT systems.  Excitation and far-field
    methods always refer to the EXTERIOR (air) and are only meaningful on an
    instance with medium=None.
    """

    def __init__(self, points, freq_hz: 'float', gauss_order: 'int' = FAR_GAUSS_ORDER,
                 near_depth: 'int' = 4, medium=None, single_tables: 'bool' = False):


        freq_hz = float(freq_hz)
        if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
            raise ValueError("BoR frequency must be a positive finite value.")
        self._compressed = compressed_requested()
        self._checkpoint = current_checkpoint()
        self._table_dtype = np.complex64 if single_tables else np.complex128
        self.gen = Generatrix(np.asarray(points, dtype=float))
        k0 = 2.0 * math.pi * freq_hz / C0
        if medium is None:
            self.k = k0
            self.eta = ETA0
        else:
            m_idx, eta_r = _causal_medium(*medium)
            self.k = k0 * m_idx
            self.eta = ETA0 * eta_r
        self.freq_hz = freq_hz
        self.g = gauss_on_generatrix(self.gen, gauss_order)
        self.gauss_order = gauss_order
        from ghost_backend.bor.options import current_options
        if int(near_depth) < 0:
            raise ValueError("near_depth must be a non-negative integer.")
        self.near_depth = int(near_depth) + current_options()['near_refinement']


        self.near_span = 2
        self._configure_near_pair_routing()
        self.Nn = self.gen.n_nodes
        self._build_point_matrices()


        far_gap = self._far_gap()
        if far_gap > 0.0:
            try:
                n_xi_for_pairs(
                    self.k,
                    float(np.max(self.gen.nodes[:, 0])),
                    0,
                    far_gap,
                    bracket=False,
                )
            except ValueError as exc:
                pair = getattr(self, "_far_gap_pair", None)
                pair_note = (
                    f" for nonadjacent elements {pair[0]} and {pair[1]}"
                    if pair is not None else ""
                )
                raise ValueError(
                    "BoR same-surface far-quadrature preflight failed"
                    f"{pair_note}: {exc}"
                ) from exc
        self._G_table = None
        self._stream = None
        self._near_cache: 'Dict[int, Dict[Tuple[int, int], Tuple]]' = {}
        self._near_contractions: 'Dict[Tuple[str, int], Dict[str, np.ndarray]]' = {}
        self._mass_cache = None
        self._weighted_mass_cache = None
        self._unit_mass_bands = None
        self._basis_mask_cache: 'Dict[int, np.ndarray]' = {}
        self._basis_transform_cache: 'Dict[int, np.ndarray]' = {}
        self._angular_local = threading.local()
        # Aspect chunk -> _AngularChunk, shared by every mode of a sweep.
        self._angular_shared: 'Dict[bytes, _AngularChunk]' = {}
        self._angular_shared_lock = threading.Lock()
        self._angular_key_locks: 'Dict[bytes, threading.Lock]' = {}
        self._angular_top = 0

    def enable_streaming(self, m_max: 'int', efie: 'bool' = True,
                         mfie: 'bool' = False,
                         ibc_zs_pt: 'Optional[np.ndarray]' = None,
                         pmchwt: 'bool' = False,
                         single_blocks: 'bool' = False,
                         tile_budget_gb: 'float' = 1.0,
                         workers: 'int' = 1,
                         mode_block: 'Optional[int]' = None,
                         spill: 'Optional[str]' = None) -> 'None':
        """Build per-mode nodal far blocks before operator assembly.

        Near/self quadrature uses the same kernels. IBC blocks include source Z_s, so
        assemble_ibc_extra must receive the same zs_pt. PMCHWT uses rotated-PV blocks
        with unit source weight.  ``spill`` is the base directory chosen by
        ``plan_stream_spill`` for a one-sweep memory-mapped build, or None.
        """

        from ghost_backend.bor.streaming import StreamingFarBlocks
        from ghost_backend.bor.compressed_far import CompressedFarBlocks, far_compression_selected
        self.close_streaming()
        # A large surface keeps its far blocks compressed (all modes, in RAM):
        # the planners price that store through the same estimates.
        if self._compressed and spill is not None:
            from ghost_backend.bor.spooled_far import SpooledFarBlocks
            store = SpooledFarBlocks
        else:
            store = CompressedFarBlocks if far_compression_selected(self.Nn) else StreamingFarBlocks
        self._stream = store(
            self, m_max, efie=efie, mfie=mfie, ibc_zs_pt=ibc_zs_pt,
            pmchwt=pmchwt,
            dtype=np.complex64 if single_blocks else np.complex128,
            tile_budget_gb=tile_budget_gb, workers=workers,
            mode_block=mode_block, spill=spill)

    def close_streaming(self) -> 'None':
        """Release streamed far blocks, including any spilled files."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()


    def _build_point_matrices(self):
        g = self.g
        P = len(g.rho)
        self.P = P


        self._B_T = None
        self._B_D = None

        self.elem_of_pt = g.elem

    def _ensure_dense_point_matrices(self) -> 'None':
        if self._B_T is not None:
            return
        g = self.g
        T = np.zeros((self.Nn, self.P))
        D = np.zeros((self.Nn, self.P))
        p = np.arange(self.P)
        e = g.elem.astype(int, copy=False)
        T[e, p] = g.T0
        D[e, p] = g.dRT0
        T[e + 1, p] = g.T1
        D[e + 1, p] = g.dRT1
        self._B_T, self._B_D = T, D

    @property
    def B_T(self) -> 'np.ndarray':
        self._ensure_dense_point_matrices()
        return self._B_T

    @property
    def B_D(self) -> 'np.ndarray':
        self._ensure_dense_point_matrices()
        return self._B_D

    def _test_accumulate(self, point_values: 'np.ndarray') -> 'np.ndarray':
        """Apply the triangle test basis without the dense ``B_T`` matrix.

        Every Gauss point belongs to exactly one element and therefore touches
        only its two endpoint nodes.  Explicit local scatter is O(P), whereas
        the mathematically equivalent dense multiply is O(Nn*P).
        """

        values = np.asarray(point_values)
        if values.shape[0] != self.P:
            raise ValueError("BoR point-value array has the wrong leading size.")
        trailing = values.shape[1:]
        ne, order = self.gen.n_elems, self.gauss_order
        # Gauss points are stored element by element, so the scatter is a
        # per-element reduction (np.add.at was 2.2x slower on 12,480 points).
        grouped = values.reshape((ne, order) + trailing)
        weights_shape = (ne, order) + (1,) * len(trailing)
        first = np.sum(self.g.T0.reshape(weights_shape) * grouped, axis=1)
        second = np.sum(self.g.T1.reshape(weights_shape) * grouped, axis=1)
        out = np.zeros((self.Nn,) + trailing,
                       dtype=np.result_type(values.dtype, np.float64))
        out[:-1] += first
        out[1:] += second
        return out

    def _basis_evaluate(self, nodal_values: 'np.ndarray') -> 'np.ndarray':
        """Evaluate nodal triangle coefficients at Gauss points in O(P)."""

        values = np.asarray(nodal_values)
        if values.shape[0] != self.Nn:
            raise ValueError("BoR nodal array has the wrong leading size.")
        trailing = values.shape[1:]
        scale_shape = (self.P,) + (1,) * len(trailing)
        elem = self.g.elem.astype(int, copy=False)
        return (
            self.g.T0.reshape(scale_shape) * values[elem]
            + self.g.T1.reshape(scale_shape) * values[elem + 1]
        )

    def _configure_near_pair_routing(self) -> 'None':
        """Route geometrically sharp pairs to direct modal integration.

        The topological stencil handles self/shared-node pairs.  In addition,
        disjoint pairs close relative to either panel length, or whose gap
        requires more than the bounded azimuthal FFT grid, use converged
        meridian integration. This prevents smooth electrically large
        bodies from acquiring an artificial radius ceiling while retaining
        the cap for genuinely excessive global oscillation/mode bandwidth.
        """

        gen = self.gen
        ne = gen.n_elems
        sources = [set(range(max(0, e - self.near_span),
                             min(ne, e + self.near_span + 1)))
                   for e in range(ne)]
        rho_max = float(np.max(gen.nodes[:, 0]))
        # The azimuthal grid a far pair needs grows with ITS OWN radius over
        # its gap, so the direct-integration threshold is per pair (larger
        # element radius), not the body's maximum radius: small-radius
        # features (axial grooves, tips) no longer fall to the converged
        # meridian rule merely because the body is wide elsewhere.
        self._elem_rho = np.maximum(gen.nodes[gen.elem_n0, 0], gen.nodes[gen.elem_n1, 0])
        direct_scale = 1.001 * 8.0 * 2.0 * math.pi / N_XI_SAFETY_CAP


        direct_gap = direct_scale * rho_max
        scale = max(
            float(np.ptp(gen.nodes[:, 0])),
            float(np.ptp(gen.nodes[:, 1])),
            1.0e-15,
        )
        touch_tol = max(1.0e-14, 1.0e-10 * scale)
        mids = 0.5 * (gen.nodes[gen.elem_n0] + gen.nodes[gen.elem_n1])
        half = 0.5 * gen.lengths
        if ne and direct_gap > 0.0:
            tree = cKDTree(mids)
            max_half = float(np.max(half))
            for e in range(ne):
                radius = float(half[e]) + max_half + max(direct_gap, 4.0*max_half)
                for candidate in tree.query_ball_point(mids[e], radius):
                    f = int(candidate)
                    if f <= e + self.near_span:
                        continue
                    lower = (
                        float(np.linalg.norm(mids[e] - mids[f]))
                        - float(half[e]) - float(half[f])
                    )
                    pair_direct = direct_scale * float(max(self._elem_rho[e], self._elem_rho[f]))
                    pair_gap = max(pair_direct, 2.0*max(gen.lengths[e], gen.lengths[f]))
                    if lower > pair_gap:
                        continue
                    gap = _segment_distance(
                        gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                        gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                    )
                    if gap <= touch_tol:
                        raise ValueError(
                            "Nonadjacent BoR generatrix elements "
                            f"{e} and {f} touch or overlap (gap {gap:.3g} m). "
                            "This creates a non-manifold surface and is not "
                            "supported."
                        )
                    if gap <= pair_gap:
                        sources[e].add(f)
                        sources[f].add(e)

        self._near_sources_by_element = tuple(
            tuple(sorted(values)) for values in sources
        )
        self._near_pair_count = sum(map(len, self._near_sources_by_element))
        self._far_gap_pair = None
        self._far_gap_cache = (
            0.0 if self._near_pair_count == ne * ne else None
        )

    def _near_gauss_mask(self) -> 'np.ndarray':
        """Boolean point-pair mask corresponding to direct element pairs."""

        ne = self.gen.n_elems
        element_mask = np.zeros((ne, ne), dtype=bool)
        for e, sources in enumerate(self._near_sources_by_element):
            element_mask[e, list(sources)] = True
        elem = self.g.elem.astype(int, copy=False)
        return element_mask[elem[:, None], elem[None, :]]

    def _far_gap(self) -> 'float':
        """Effective minimum gap among FFT-routed element pairs.

        A far pair's azimuthal grid grows with its own radius over its gap,
        while every caller sizes grids and checks the sample cap as
        ``n_xi_for_pairs(k, rho_max, ..., gap)``.  The worst pair is therefore
        reported at the body's maximum radius, ``rho_max * min(gap / rho_pair)``
        (``rho_pair`` the larger element radius of the pair): it reproduces
        each pair's own worst requirement and is never below the raw minimum
        gap.  A bounding-sphere tree selects candidates, followed by exact
        segment distances; a touching pair is still an error.
        """

        if getattr(self, "_far_gap_cache", None) is not None:
            return self._far_gap_cache
        gen = self.gen
        ne = gen.n_elems
        if ne <= self.near_span:
            self._far_gap_pair = None
            self._far_gap_cache = 0.0
            return self._far_gap_cache

        rho_max = float(np.max(gen.nodes[:, 0]))
        elem_rho = getattr(self, "_elem_rho", None)
        if elem_rho is None:
            elem_rho = np.maximum(gen.nodes[gen.elem_n0, 0], gen.nodes[gen.elem_n1, 0])

        def effective(e, f, gap):
            rho_pair = float(max(elem_rho[e], elem_rho[f]))
            return math.inf if rho_pair <= 0.0 else gap * (rho_max / rho_pair)

        d = math.inf          # effective (radius-weighted) minimum
        raw = math.inf        # geometric minimum, for the touching check
        best_pair = raw_pair = None


        for offset in range(1, ne):
            found_at_offset = False
            for e in range(ne - offset):
                f = e + offset
                if f in self._near_sources_by_element[e]:
                    continue
                found_at_offset = True
                de = _segment_distance(
                    gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                    gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                )
                if de < raw:
                    raw, raw_pair = de, (e, f)
                weighted = effective(e, f, de)
                if weighted < d:
                    d = weighted
                    best_pair = (e, f)
            if found_at_offset:
                break

        if best_pair is None and raw_pair is None:
            self._far_gap_pair = None
            self._far_gap_cache = 0.0
            return self._far_gap_cache

        mids = 0.5 * (
            gen.nodes[gen.elem_n0] + gen.nodes[gen.elem_n1]
        )
        half = 0.5 * gen.lengths
        max_half = float(np.max(half))
        tree = cKDTree(mids)
        # Effective gaps are never below geometric ones, so candidates whose
        # geometric lower bound reaches the current effective minimum cannot
        # improve it; the raw minimum only matters when it is a touch.
        bound = d if math.isfinite(d) else raw
        for e in range(ne):


            radius = float(half[e]) + max_half + bound
            for f in tree.query_ball_point(mids[e], radius):
                f = int(f)
                if f <= e or f in self._near_sources_by_element[e]:
                    continue
                lower = (
                    float(np.linalg.norm(mids[e] - mids[f]))
                    - float(half[e]) - float(half[f])
                )
                if lower >= bound:
                    continue
                de = _segment_distance(
                    gen.nodes[gen.elem_n0[e]], gen.nodes[gen.elem_n1[e]],
                    gen.nodes[gen.elem_n0[f]], gen.nodes[gen.elem_n1[f]],
                )
                if de < raw:
                    raw, raw_pair = de, (e, f)
                weighted = effective(e, f, de)
                if weighted < d:
                    d = weighted
                    best_pair = (e, f)
                    bound = d

        scale = max(
            float(np.ptp(gen.nodes[:, 0])),
            float(np.ptp(gen.nodes[:, 1])),
            1.0e-15,
        )
        touch_tol = max(1.0e-14, 1.0e-10 * scale)
        if raw <= touch_tol:
            raise ValueError(
                "Nonadjacent BoR generatrix elements "
                f"{raw_pair[0]} and {raw_pair[1]} touch or overlap "
                f"(gap {raw:.3g} m). This creates a non-manifold surface and "
                "is not supported."
            )
        self._far_gap_pair = best_pair
        self._far_gap_cache = 0.0 if not math.isfinite(d) else float(d)
        return self._far_gap_cache

    def _kernel_tables(self, m_max: 'int'):
        """G_m table [P, P, m_max+2] at base Gauss points; local-neighbour
        element point-pairs zeroed (their Galerkin blocks are added by the
        refined near-pair path)."""

        if self._G_table is not None and self._G_table.shape[-1] >= m_max + 2:
            return self._G_table
        g = self.g


        RP = g.rho[:, None]
        RQ = g.rho[None, :]
        ZP = g.z[:, None]
        ZQ = g.z[None, :]
        near_mask = self._near_gauss_mask()
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=False)
        if _blocked_single_tables(self._table_dtype):
            self._G_table, = _tables_by_rows(
                lambda r0, r1: (modal_kernels_fft(
                    RP[r0:r1], ZP[r0:r1], RQ, ZQ, self.k, m_max, n_xi=n_xi,
                    near_mask=near_mask[r0:r1], threads=physical_cpu_count()),),
                self.P, self.P, 16.0 * self.P * (m_max + 2), self._table_dtype, near_mask)
            self._m_max_table = m_max
            return self._G_table
        G = modal_kernels_fft(RP, ZP, RQ, ZQ, self.k, m_max, n_xi=n_xi, near_mask=near_mask,
                              threads=physical_cpu_count())


        near_flat = np.flatnonzero(near_mask.ravel())
        Gf = G.reshape(-1, G.shape[-1])
        Gf[near_flat, :] = 0.0
        self._G_table = Gf.reshape(self.P, self.P, -1).astype(
            self._table_dtype, copy=False)
        self._m_max_table = m_max
        return self._G_table


    def _near_pair_data(self, e: 'int', f: 'int', m_max: 'int'):
        key = (e, f)
        cache = self._near_cache.setdefault(m_max, {})
        if key in cache:
            return cache[key]
        if abs(e - f) <= 1:
            s, sp, w = _same_surface_points(self.gen, e, f, ('efie',), self.near_depth)
        else:


            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, D0p, D1p, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, D0q, D1q, Lq = _points_on_element(self.gen, f, sp)
        Gm = modal_kernels_near(rho_p, z_p, rho_q, z_q, self.k, m_max)
        data = (s, sp, w * Lp * Lq,
                rho_p, tr_p, tz_p, np.vstack([T0p, T1p]), np.vstack([D0p, D1p]),
                rho_q, tr_q, tz_q, np.vstack([T0q, T1q]), np.vstack([D0q, D1q]),
                Gm)
        cache[key] = data
        return data


    def assemble_mode(self, m: 'int', m_max: 'int', out=None,
                      scale: 'complex' = 1.0) -> 'np.ndarray':
        """EFIE operator ``C * T`` of one signed mode, ``C = j k eta 2 pi``.

        ``out`` receives the ``[2Nn, 2Nn]`` matrix in place and ``scale``
        multiplies it (the CFIE weight), so retained far blocks are read
        once, straight into the system quadrants, without a per-family copy.
        """
        if self._compressed:
            if out is not None or scale != 1.0:
                raise ValueError("Compressed BoR assembly cannot write in place.")
            return primitive(self, 'T', m, m_max)
        k = self.k
        g = self.g
        Nn = self.Nn
        C = 1j * k * self.eta * 2.0 * np.pi * scale
        Z = out if out is not None else np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        if Z.shape != (2 * Nn, 2 * Nn):
            raise ValueError("BoR in-place system matrix has the wrong shape.")
        quads = (Z[:Nn, :Nn], Z[:Nn, Nn:], Z[Nn:, :Nn], Z[Nn:, Nn:])
        if self._stream is not None:
            self._stream.write_efie_blocks(m, quads, C)
        else:
            Gtab = self._kernel_tables(m_max)
            for quad in quads:
                quad.fill(0.0)
            ne = self.gen.n_elems
            _efie_tables_into(quads, m, k, g, ne, self.gauss_order,
                              g, ne, self.gauss_order, Gtab, C)

        prepared = self._prepared_near("efie", m_max)
        if prepared is not None:
            midx = abs(m)
            rc = (prepared["rows"], prepared["cols"])
            for uv, quad in enumerate(quads):
                np.add.at(quad, rc, prepared["values"][uv, midx] * (C * mode_sign(uv, m)))
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    (s, sp, w, rho_p, tr_p, tz_p, Tp, Dp,
                     rho_q, tr_q, tz_q, Tq, Dq, Gm) = self._near_pair_data(e, f, m_max)
                    Gn, Gcn, Gsn = kernels_for_mode(Gm, m)
                    rr = rho_p * rho_q * w
                    ktt = rr * ((tr_p * tr_q) * Gcn + (tz_p * tz_q) * Gn)
                    ksc = w * Gn
                    ktf = rr * (tr_p * Gsn)
                    kft = -rr * (tr_q * Gsn)
                    kff = rr * Gcn
                    btt = np.einsum("ip,p,jp->ij", Tp, ktt, Tq) - (1.0 / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Dq)
                    btf = np.einsum("ip,p,jp->ij", Tp, ktf, Tq) - (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Tq)
                    bft = np.einsum("ip,p,jp->ij", Tp, kft, Tq) + (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Dq)
                    bff = np.einsum("ip,p,jp->ij", Tp, kff, Tq) - (m ** 2 / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Tq)
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    quads[0][np.ix_(rows, cols)] += C * btt
                    quads[1][np.ix_(rows, cols)] += C * btf
                    quads[2][np.ix_(rows, cols)] += C * bft
                    quads[3][np.ix_(rows, cols)] += C * bff
        return Z


    def _mfie_tables(self, m_max: 'int'):
        """Four nonnegative MFIE kernel tables [P, P, m_max+1] at base Gauss
        points; near element-pair entries zeroed (refined path adds them)."""

        # The cache records its mode cap: a table built for a smaller cap has
        # too few orders, and a wider one must be indexed by its own cap.
        if (getattr(self, "_K_tables", None) is not None
                and self._K_tables_m_max >= m_max):
            return self._K_tables
        g = self.g
        P = self.P
        args = (g.rho[:, None], g.z[:, None],
                g.trho[:, None], g.tz[:, None],
                g.rho[None, :], g.z[None, :],
                g.trho[None, :], g.tz[None, :])
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=True)
        near_mask = self._near_gauss_mask()
        if _blocked_single_tables(self._table_dtype):
            self._K_tables = tuple(_tables_by_rows(
                lambda r0, r1: nonnegative_bracket_tables(
                    'mfie', tuple(a[r0:r1] for a in args[:4]) + args[4:], self.k, m_max,
                    n_xi, near_mask[r0:r1], threads=physical_cpu_count()),
                P, P, 4 * 16.0 * P * (m_max + 1), self._table_dtype, near_mask))
            self._K_tables_m_max = int(m_max)
            return self._K_tables
        K = nonnegative_bracket_tables('mfie', args, self.k, m_max, n_xi, near_mask,
                                       threads=physical_cpu_count())
        near_flat = np.flatnonzero(near_mask.ravel())
        del near_mask
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(P, P, -1).astype(self._table_dtype, copy=False)
        self._K_tables = tuple(K)
        self._K_tables_m_max = int(m_max)
        return self._K_tables

    def _near_mfie_data(self, e: 'int', f: 'int', m_max: 'int'):
        cache = self._near_cache.setdefault(("mfie", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        if abs(e - f) <= 1:
            s, sp, w = _same_surface_points(self.gen, e, f, ('mfie',), self.near_depth)
        else:
            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        tr_pa = np.full_like(rho_p, tr_p); tz_pa = np.full_like(rho_p, tz_p)
        tr_qa = np.full_like(rho_q, tr_q); tz_qa = np.full_like(rho_q, tz_q)
        Kn = mfie_kernels_near(rho_p, z_p, tr_pa, tz_pa, rho_q, z_q, tr_qa, tz_qa,
                               self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def mass_blocks(self, weight=None) -> 'np.ndarray':
        """2pi * Int w(t) rho T_i T_j dt  (node-based [Nn, Nn]); weight is a
        per-Gauss-point array (default 1) -- used for the MFIE J/2 term and
        the IBC Z_s term (with weight = Z_s at the Gauss points)."""
        if self._compressed:
            return mass_expression(self, weight)


        if weight is None and self._mass_cache is not None:
            return self._mass_cache

        g = self.g
        if weight is None:
            wgt = 1.0
            signature = None
        else:
            wgt = np.asarray(weight)
            if wgt.shape != (self.P,):
                raise ValueError("BoR mass weight must have one value per Gauss point.")
            signature = (wgt.dtype.str, wgt.shape, wgt.tobytes())
            cached = self._weighted_mass_cache
            if cached is not None and cached[0] == signature:
                return cached[1]

        K = g.w * g.rho * wgt
        elem = g.elem.astype(int, copy=False)
        M = np.zeros((self.Nn, self.Nn),
                     dtype=np.result_type(K, np.float64))
        factor = 2.0 * np.pi
        np.add.at(M, (elem, elem), factor * K * g.T0 * g.T0)
        np.add.at(M, (elem, elem + 1), factor * K * g.T0 * g.T1)
        np.add.at(M, (elem + 1, elem), factor * K * g.T1 * g.T0)
        np.add.at(M, (elem + 1, elem + 1), factor * K * g.T1 * g.T1)
        if weight is None:
            self._mass_cache = M
        else:
            self._weighted_mass_cache = (signature, M)
        return M

    def mass_bands(self, weight=None):
        """The tridiagonal ``mass_blocks(weight)`` as ``(diagonal, upper, lower)``.

        Same accumulation order as the dense form, so adding the bands into a
        system quadrant is bitwise identical to adding the dense matrix, but
        no ``Nn x Nn`` copy is formed or cached (two of them were retained
        for the solver's lifetime on the conductor paths).
        """
        g = self.g
        wgt = 1.0 if weight is None else np.asarray(weight)
        if weight is not None and wgt.shape != (self.P,):
            raise ValueError("BoR mass weight must have one value per Gauss point.")
        K = g.w * g.rho * wgt
        elem = g.elem.astype(int, copy=False)
        dtype = np.result_type(K, np.float64)
        factor = 2.0 * np.pi
        diagonal = np.zeros(self.Nn, dtype=dtype)
        upper = np.zeros(self.Nn - 1, dtype=dtype)
        lower = np.zeros(self.Nn - 1, dtype=dtype)
        np.add.at(diagonal, elem, factor * K * g.T0 * g.T0)
        np.add.at(upper, elem, factor * K * g.T0 * g.T1)
        np.add.at(lower, elem, factor * K * g.T1 * g.T0)
        np.add.at(diagonal, elem + 1, factor * K * g.T1 * g.T1)
        return diagonal, upper, lower

    @staticmethod
    def _add_bands_into(target, bands, scale) -> 'None':
        """``target += scale * M`` for a tridiagonal ``M`` given as bands."""
        diagonal, upper, lower = bands
        n = diagonal.shape[0]
        index = np.arange(n)
        target[index, index] += diagonal * scale
        target[index[:-1], index[1:]] += upper * scale
        target[index[1:], index[:-1]] += lower * scale

    def assemble_mfie_mode(self, m: 'int', m_max: 'int', out=None,
                           scale: 'complex' = 1.0,
                           accumulate: 'bool' = False) -> 'np.ndarray':
        """Z_MFIE = (1/2) M - K  (node-based [2Nn, 2Nn]), where K is the
        Galerkin contraction of the modal MFIE brackets.

        With ``out`` the matrix is formed in that buffer, multiplied by
        ``scale`` and, when ``accumulate`` is set, added to what the buffer
        already holds: the CFIE combination then never owns a second full
        matrix.  Retained far blocks are read in bounded chunks, never copied.
        """
        if self._compressed:
            if out is not None or scale != 1.0 or accumulate:
                raise ValueError("Compressed BoR assembly cannot write in place.")
            return primitive(self, 'K', m, m_max)

        g = self.g
        Nn = self.Nn
        Z = out if out is not None else np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        if Z.shape != (2 * Nn, 2 * Nn):
            raise ValueError("BoR in-place system matrix has the wrong shape.")
        if out is not None and not accumulate:
            Z.fill(0.0)
        quads = (Z[:Nn, :Nn], Z[:Nn, Nn:], Z[Nn:, :Nn], Z[Nn:, Nn:])
        far = -scale
        if self._stream is not None and self._stream.K is not None:
            self._stream.add_blocks("mfie", m, quads, far)
        else:
            Kt = self._mfie_tables(m_max)
            ne = self.gen.n_elems
            _bracket_tables_into(quads, m, self._K_tables_m_max, g, ne, self.gauss_order,
                                 g, ne, self.gauss_order, Kt, far)

        prepared = self._prepared_near("mfie", m_max)
        if prepared is not None:
            midx = abs(m)
            rc = (prepared["rows"], prepared["cols"])
            for uv, quad in enumerate(quads):
                np.add.at(quad, rc, prepared["values"][uv, midx] * (far * mode_sign(uv, m)))
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    w, rho_p, Tp, rho_q, Tq, Kn = self._near_mfie_data(e, f, m_max)
                    rr = rho_p * rho_q * w
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    for uv, quad in enumerate(quads):
                        Km = mfie_for_mode(Kn[uv], m, m_max, odd=uv in (1, 2))
                        blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                        quad[np.ix_(rows, cols)] += far * blk

        bands = self._unit_mass_bands
        if bands is None:
            bands = self._unit_mass_bands = self.mass_bands()
        self._add_bands_into(quads[0], bands, 0.5 * scale)
        self._add_bands_into(quads[3], bands, 0.5 * scale)
        return Z

    def _angular_data(self, m: 'int', theta_deg: 'float'):
        """Shared cylindrical-wave data for one mode and look direction.

        EFIE, MFIE, VV, HH, and monostatic far-field evaluation all reuse the
        same Bessel functions and axial phase.  A thread-local single-entry
        cache avoids recomputation without retaining a mesh-sized table for
        every requested angle.
        """

        key = (int(m), float(theta_deg))
        entry = getattr(self._angular_local, "entry", None)
        if entry is not None and entry[0] == key:
            return entry[1]
        th = math.radians(theta_deg)
        st, ct = math.sin(th), math.cos(th)
        u = self.k * self.g.rho * st
        phase = np.exp(1j * self.k * ct * self.g.z)
        j_lo, j_0, j_hi = _bessel_triplet(int(m), u)
        Jm = (1j) ** m * j_0
        Jm_m1 = (1j) ** (m - 1) * j_lo
        Jm_p1 = (1j) ** (m + 1) * j_hi
        Ic = math.pi * (Jm_m1 + Jm_p1)
        Is = (math.pi / 1j) * (Jm_m1 - Jm_p1)
        I1 = 2.0 * math.pi * Jm
        data = (st, ct, phase, Ic, Is, I1)
        self._angular_local.entry = (key, data)
        return data

    def rhs_mfie_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        """Return the tested plane-wave excitation <W, n_hat x H_inc>."""

        g = self.g
        st, ct, P, Ic, Is, I1 = self._angular_data(m, theta_inc_deg)
        if pol.upper() in ("VV", "THETA", "TM"):

            et = Ic / ETA0
            ef = -(g.trho * Is) / ETA0
        else:

            et = (ct * Is) / ETA0
            ef = (ct * g.trho * Ic - st * g.tz * I1) / ETA0
        vt = self._test_accumulate(g.w * g.rho * P * et)
        vf = self._test_accumulate(g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])


    def _ibc_tables(self, m_max: 'int'):
        if (getattr(self, "_KI_tables", None) is not None
                and self._KI_tables_m_max >= m_max):
            return self._KI_tables
        g = self.g
        n_xi = n_xi_for_pairs(self.k, float(np.max(g.rho)), m_max,
                              self._far_gap(), bracket=True)
        args = tuple(getattr(g, name)[:,None] for name in ('rho','z','trho','tz'))
        args += tuple(getattr(g, name)[None,:] for name in ('rho','z','trho','tz'))
        near_mask = self._near_gauss_mask()
        if _blocked_single_tables(self._table_dtype):
            self._KI_tables = tuple(_tables_by_rows(
                lambda r0, r1: nonnegative_bracket_tables(
                    'ibc', tuple(a[r0:r1] for a in args[:4]) + args[4:], self.k, m_max,
                    n_xi, near_mask[r0:r1], threads=physical_cpu_count()),
                self.P, self.P, 4 * 16.0 * self.P * (m_max + 1), self._table_dtype, near_mask))
            self._KI_tables_m_max = int(m_max)
            return self._KI_tables
        K = nonnegative_bracket_tables('ibc', args, self.k, m_max, n_xi, near_mask,
                                       threads=physical_cpu_count())
        near_flat = np.flatnonzero(near_mask.ravel())
        del near_mask
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(self.P, self.P, -1).astype(self._table_dtype, copy=False)
        self._KI_tables = tuple(K)
        self._KI_tables_m_max = int(m_max)
        return self._KI_tables

    def _near_ibc_data(self, e: 'int', f: 'int', m_max: 'int'):
        cache = self._near_cache.setdefault(("ibc", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        if abs(e - f) <= 1:
            s, sp, w = _same_surface_points(self.gen, e, f, ('ibc',), self.near_depth)
        else:
            s, sp, w = _regular_cell_points()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        Kn = ibc_kernels_near(rho_p, z_p, np.full_like(rho_p, tr_p), np.full_like(rho_p, tz_p),
                              rho_q, z_q, np.full_like(rho_q, tr_q), np.full_like(rho_q, tz_q),
                              self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def _rot_pv_blocks(self, m: 'int', m_max: 'int', src_wpt=None, src_welem=None):
        """Galerkin contraction of the rotated-PV brackets
        B_uv = p(R) W_u . [Rvec x (n_hat_q x f_v)] with an optional source
        weight (Z_s for the IBC path; unit for PMCHWT).  Returns the four
        node-based blocks (Btt, Btf, Bft, Bff)."""

        Nn = self.Nn
        blocks = [np.zeros((Nn, Nn), dtype=np.complex128) for _ in range(4)]
        self._accumulate_rot_pv(blocks, m, m_max, src_wpt, src_welem, 1.0)
        return tuple(blocks)

    def _accumulate_rot_pv(self, targets, m: 'int', m_max: 'int', src_wpt=None,
                           src_welem=None, scale: 'complex' = 1.0) -> 'None':
        """``targets[uv] += scale * B_uv`` without forming the four blocks.

        Streamed blocks are read as views of the retained storage; table
        blocks are formed one at a time.  The near corrections are added
        straight into the targets.
        """

        g = self.g
        stream_unit_source = bool(
            getattr(self._stream, "rot_pv_unit_source", False)
        )
        use_stream = (
            self._stream is not None
            and self._stream.B is not None
            and (
                (src_wpt is None and stream_unit_source)
                or (src_wpt is not None and not stream_unit_source)
            )
        )
        if use_stream:
            self._stream.add_blocks("ibc", m, targets, scale)
        else:
            Kt = self._ibc_tables(m_max)
            ne = self.gen.n_elems
            _bracket_tables_into(targets, m, self._KI_tables_m_max, g, ne, self.gauss_order,
                                 g, ne, self.gauss_order, Kt, scale,
                                 source_weight=src_wpt)

        prepared = self._prepared_near("ibc", m_max)
        if prepared is not None:
            midx = abs(m)
            rc = (prepared["rows"], prepared["cols"])
            source_weight = (
                1.0 if src_welem is None
                else np.asarray(src_welem)[prepared["source_elems"]]
            )
            for uv, tgt in enumerate(targets):
                np.add.at(
                    tgt, rc,
                    prepared["values"][uv, midx] * source_weight * (mode_sign(uv, m) * scale),
                )
        else:
            ne = self.gen.n_elems
            for e in range(ne):
                for f in self._near_sources_by_element[e]:
                    welem = 1.0 if src_welem is None else src_welem[f]
                    if abs(welem) == 0.0:
                        continue
                    w, rho_p, Tp, rho_q, Tq, Kn = self._near_ibc_data(e, f, m_max)
                    rr = rho_p * rho_q * w * welem
                    rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                    for uv, tgt in enumerate(targets):
                        Km = mfie_for_mode(Kn[uv], m, m_max, odd=uv in (1, 2))
                        blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                        tgt[np.ix_(rows, cols)] += scale * blk

    def assemble_ibc_extra(self, m: 'int', m_max: 'int', zs_pt: 'np.ndarray',
                           zs_elem: 'np.ndarray', out=None,
                           scale: 'complex' = 1.0) -> 'np.ndarray':
        """
        IBC-EFIE additions from eliminating M = -Z_s n_hat x J:

            Z_extra = (1/2) <W, Z_s J> + <W, PV curl Int G M>

        The K' term applies Z_s at the SOURCE point.  With ``out`` the
        ``scale``-weighted terms are ADDED to that ``[2Nn, 2Nn]`` buffer and it
        is returned, so the IBC-CFIE never holds a separate extra matrix.
        """
        if self._compressed:
            if out is not None or scale != 1.0:
                raise ValueError("Compressed BoR assembly cannot write in place.")
            return primitive(self, 'IBC', m, m_max, zs_pt, zs_elem)

        Nn = self.Nn
        Z = out if out is not None else np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        if Z.shape != (2 * Nn, 2 * Nn):
            raise ValueError("BoR in-place system matrix has the wrong shape.")
        quads = [Z[:Nn, :Nn], Z[:Nn, Nn:], Z[Nn:, :Nn], Z[Nn:, Nn:]]
        self._accumulate_rot_pv(quads, m, m_max, zs_pt, zs_elem, scale)
        bands = self.mass_bands(weight=zs_pt)
        self._add_bands_into(quads[0], bands, 0.5 * scale)
        self._add_bands_into(quads[3], bands, 0.5 * scale)
        return Z

    def assemble_pmchwt_P(self, m: 'int', m_max: 'int', out=None) -> 'np.ndarray':
        """
        The PMCHWT rotated-PV operator P (node-based [2Nn, 2Nn]) acting on a
        magnetic current expanded in the SAME (t, phi) triangle bases:

            (P M)_tested = <W, PV Int p(R) Rvec x M dS'>
                         = <W, E_PV(M)> in this medium (E_s(M) = -curl Int G M)

        The IBC brackets B_uv are built for the rotated source n_hat_q x f_v
        (n x t = phi, n x phi = -t), so columns remap:  P[:, Mt] = -B[:, f_phi],
        P[:, Mphi] = +B[:, f_t].  The same bilinear form gives the H-side
        operator: <W, H_PV(J)> = -(P J).

        ``out`` may be a strided system submatrix. Rotated brackets are
        accumulated into their final column positions, avoiding a second
        full matrix of bracket blocks before forming P.
        """
        if self._compressed:
            if out is not None:
                raise ValueError("Compressed BoR assembly cannot write in place.")
            return primitive(self, 'P', m, m_max)


        Nn = self.Nn
        P = out if out is not None else np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        if P.shape != (2 * Nn, 2 * Nn):
            raise ValueError("BoR in-place PMCHWT matrix has the wrong shape.")
        P.fill(0.0)
        self._accumulate_rot_pv(
            (P[:Nn, Nn:], P[:Nn, :Nn], P[Nn:, Nn:], P[Nn:, :Nn]), m, m_max)
        P[:, :Nn] *= -1.0
        return P

    def _prepared_near(self, kind, m_max):
        """Direct operator callers get the same bounded, checked integration."""
        available = [(cap, data) for (family, cap), data in self._near_contractions.items()
                     if family == kind and cap >= int(m_max)]
        if available:
            return min(available, key=lambda item: item[0])[1]
        pairs = [(e, f) for e, sources in enumerate(self._near_sources_by_element) for f in sources]
        self._prepare_near_contractions(kind, pairs, m_max)
        return self._near_contractions[(kind, int(m_max))]

    def _prepare_near_contractions(self, kind, pairs, m_max, workers=1):
        self._prepare_near_families((kind,), pairs, m_max, workers)

    def _prepare_near_families(self, kinds, pairs, m_max, workers=1):
        """Share near geometry/traversal; extend only missing modal bands.

        Each angular rule still uses the new cap and its coarse/fine check.
        Already accepted modes retain their original checked coefficients.
        """
        groups = {}
        for kind in kinds:
            previous = [(cap, data) for (family, cap), data in self._near_contractions.items()
                        if family == kind]
            cap, data = max(previous, key=lambda item: item[0]) if previous else (-1, None)
            if cap < m_max:
                groups.setdefault(cap + 1, {})[kind] = data
        for start, previous in groups.items():
            self._prepare_near_family_band(tuple(previous), pairs, m_max, start, previous, workers)

    def _prepare_near_family_band(self, kinds, pairs, m_max, mode_start, previous, workers):
        from ghost_backend.bor.near_storage import compact_layout, reciprocal_pair_order, EfieReciprocity, ModalBands
        if not hasattr(self, 'near_preparation_bands'):
            self.near_preparation_bands = []
        self.near_preparation_bands.append(dict(kinds=list(kinds), first_mode=int(mode_start),
                                                last_mode=int(m_max), pairs=len(pairs)))
        # Keep the original independent pair integrations and their diagnostic,
        # but accumulate directly into unique destinations. Source-element
        # identity remains part of an IBC entry for later impedance weighting.
        pairs = reciprocal_pair_order(pairs) if 'efie' in kinds else pairs
        width = m_max + 1 - mode_start
        layouts = {kind: compact_layout(pairs, self.Nn, preserve_sources=kind == 'ibc') for kind in kinds}
        values = {kind: np.zeros((4, width, len(layouts[kind][0]['rows'])), complex) for kind in kinds}
        reciprocity = EfieReciprocity(width) if 'efie' in kinds else None

        def integrate(pair):
            if self._checkpoint is not None:
                self._checkpoint()
            e, f = pair
            if abs(e - f) <= 1:
                points = _same_surface_points(self.gen, e, f, kinds, self.near_depth)
                blocks = _contract_near_points(self.gen, e, self.gen, f,
                    self.k, m_max, kinds, points, signed=False, mode_start=mode_start)
                return blocks, None
            blocks, order, error = _converged_disjoint_blocks(
                self.gen, e, self.gen, f, self.k, m_max, kinds, signed=False, mode_start=mode_start)
            return blocks, (order, error)

        from ghost_backend.bor.near_parallel import NearTask
        task = NearTask(self.gen, self.gen, self.k, m_max, kinds, depth=self.near_depth,
                        mode_start=mode_start, return_families=True)
        with contextlib.closing(_iter_near_pairs(integrate, pairs, workers, task, self._checkpoint)) as results:
            for pi, ((e, f), (blocks, refinement)) in enumerate(zip(pairs, results)):
                if refinement is not None:
                    order, error = refinement
                    self.near_quadrature_order_max = max(
                        getattr(self, 'near_quadrature_order_max', 0), order)
                    self.near_quadrature_error_max = max(
                        getattr(self, 'near_quadrature_error_max', 0.), error)
                if reciprocity is not None:
                    reciprocity.add((e, f), blocks['efie'])
                # The four corners within one element pair are distinct;
                # contributions from previous pairs have already been summed.
                for kind in kinds:
                    values[kind][:, :, layouts[kind][1][pi]] += blocks[kind].reshape(4, width, 4)
        if reciprocity is not None:
            self.near_efie_asymmetry = max(getattr(self, 'near_efie_asymmetry', 0.0),
                                           reciprocity.value())
        for kind in kinds:
            retained = values[kind]
            if previous[kind] is not None:
                retained = ModalBands(previous[kind]['values'], retained)
            for key in [key for key in self._near_contractions if key[0] == kind]:
                del self._near_contractions[key]
            self._near_contractions[(kind, int(m_max))] = dict(layouts[kind][0], values=retained)
            self._near_cache.pop(m_max if kind == 'efie' else (kind, m_max), None)


    def prepare_operators(self, m_max: 'int', efie: 'bool' = True,
                          mfie: 'bool' = False, ibc: 'bool' = False,
                          workers: 'int' = 1) -> 'None':
        """Build every kernel table and near-pair cache this solver will need
        up front, so parallel per-mode assembly only READS shared state.
        Near-pair integration runs on a bounded thread pool (see
        ``_near_preparation_workers``) and stores results in pair order, so
        the prepared blocks are identical to a serial build."""

        ne = self.gen.n_elems
        # The excitation records start above every mode of this sweep.
        self._angular_top = max(self._angular_top, int(m_max) + 1)
        pairs = [
            (e, f)
            for e, sources in enumerate(self._near_sources_by_element)
            for f in sources
        ]
        if self._compressed:
            self._prepare_near_families(tuple(kind for kind, enabled in
                (('efie', efie), ('mfie', mfie), ('ibc', ibc)) if enabled), pairs, m_max, workers)
            return
        streaming = self._stream is not None
        # Table assembly contracts element-locally (_efie_tables_into,
        # _bracket_tables_into) and needs no dense Nn x P basis matrices.
        if efie:
            if not streaming:
                self._kernel_tables(m_max)
        if mfie:
            if not streaming:
                self._mfie_tables(m_max)
        if ibc:
            if not (streaming and self._stream.B is not None):
                self._ibc_tables(m_max)
        self._prepare_near_families(tuple(kind for kind, enabled in
            (('efie', efie), ('mfie', mfie), ('ibc', ibc)) if enabled), pairs, m_max, workers)


    def basis_mask(self, m: 'int') -> 'np.ndarray':
        category = 0 if m == 0 else (1 if abs(m) == 1 else 2)
        cached = self._basis_mask_cache.get(category)
        if cached is not None:
            return cached
        Nn = self.Nn
        t_act = np.ones(Nn, dtype=bool)
        f_act = np.ones(Nn, dtype=bool)
        for end in (0, Nn - 1):
            if self.gen.node_on_axis(end):
                t_act[end] = (abs(m) == 1)
                f_act[end] = False
            else:
                t_act[end] = False
                f_act[end] = True
        mask = np.concatenate([t_act, f_act])
        mask.setflags(write=False)
        self._basis_mask_cache[category] = mask
        return mask

    def basis_transform(self, m: 'int') -> 'np.ndarray':
        """Map regular reduced modal coefficients to full nodal components.

        At a smooth axis pole, a finite Cartesian tangential vector in the
        ``exp(j*m*phi)`` harmonic has ``J_phi = j*m*J_rho``.  The generatrix
        coefficient is ``J_t`` and its radial direction reverses at the two
        ends, hence the adjacent-element ``sign(t_rho)`` factor.
        """

        key = int(m) if abs(int(m)) == 1 else (0 if int(m) == 0 else 2)
        cached = self._basis_transform_cache.get(key)
        if cached is not None:
            return cached
        mask = self.basis_mask(m)
        active_rows = np.flatnonzero(mask)
        # At most one extra entry per pole column: build the CSR matrix from
        # coordinates.  The former dense (2Nn x n_active) array cost one full
        # system matrix per cached category and an O(n^2) scan per mode.
        rows = [active_rows]
        cols = [np.arange(active_rows.size)]
        data = [np.ones(active_rows.size, dtype=complex)]
        if abs(int(m)) == 1:
            reduced_column = np.full(2 * self.Nn, -1, dtype=int)
            reduced_column[active_rows] = np.arange(active_rows.size)
            for end, element in ((0, 0), (self.Nn - 1, self.gen.n_elems - 1)):
                if not self.gen.node_on_axis(end):
                    continue
                column = int(reduced_column[end])
                if column < 0:
                    continue
                radial_sign = 1.0 if self.gen.trho[element] >= 0.0 else -1.0
                rows.append(np.array([self.Nn + end]))
                cols.append(np.array([column]))
                data.append(np.array([1j * int(m) * radial_sign], dtype=complex))
        Q = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(2 * self.Nn, active_rows.size))
        self._basis_transform_cache[key] = Q
        return Q

    def reduce_pole_operator(self, Z: 'np.ndarray', m: 'int') -> 'np.ndarray':
        """``Q^H Z Q`` for this surface's own ``basis_transform(m)`` in O(n).

        The transform selects the active nodal components and, for
        ``|m| = 1``, ties each pole ``phi`` component to its ``t`` component
        with a unit-modulus weight.  Applying it as row/column updates of the
        masked copy reproduces the sparse product to rounding without the two
        dense intermediates that ``_reduce_constrained_operator`` forms.
        """
        mask = self.basis_mask(m)
        active = np.flatnonzero(mask)
        A = Z[np.ix_(active, active)]
        if abs(int(m)) != 1:
            return A
        reduced_column = np.full(2 * self.Nn, -1, dtype=int)
        reduced_column[active] = np.arange(active.size)
        poles = []
        for end, element in ((0, 0), (self.Nn - 1, self.gen.n_elems - 1)):
            if not self.gen.node_on_axis(end):
                continue
            column = int(reduced_column[end])
            if column < 0:
                continue
            radial_sign = 1.0 if self.gen.trho[element] >= 0.0 else -1.0
            poles.append((self.Nn + end, 1j * int(m) * radial_sign, column))
        for row, weight, column in poles:
            A[:, column] += weight * Z[active, row]
            A[column, :] += np.conj(weight) * Z[row, active]
        for row, weight, column in poles:
            for row2, weight2, column2 in poles:
                A[column, column2] += np.conj(weight) * weight2 * Z[row, row2]
        return A


    def rhs_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        g = self.g
        st, ct, P, Ic, Is, I1 = self._angular_data(m, theta_inc_deg)
        if pol.upper() in ("VV", "THETA", "TM"):
            et = ct * g.trho * Ic - st * g.tz * I1
            ef = -ct * Is
        else:
            et = g.trho * Is
            ef = Ic
        vt = self._test_accumulate(g.w * g.rho * P * et)
        vf = self._test_accumulate(g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])

    def _shared_angular_chunk(self, thetas: 'np.ndarray', need: 'int'):
        """The shared :class:`_AngularChunk` of an aspect chunk with top order
        at least ``need``, built once for all modes; None when the process-wide
        budget is spent (the caller then evaluates the mode directly)."""
        key = thetas.tobytes()
        with self._angular_shared_lock:
            chunk = self._angular_shared.get(key)
            if chunk is not None and chunk.top >= need:
                return chunk
            key_lock = self._angular_key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._angular_shared_lock:
                chunk = self._angular_shared.get(key)
            if chunk is not None and chunk.top >= need:
                return chunk
            top = max(int(need), self._angular_top, ANGULAR_MIN_TOP,
                      2 * chunk.top if chunk is not None else 0)
            values = thetas.size * self.g.rho.size
            base_bytes = ANGULAR_BASE_BYTES_PER_VALUE * values + 16 * thetas.size
            checkpoint_count = min(ANGULAR_CHECKPOINT_MAX_COUNT,
                                   max(0, (top - 1) // ANGULAR_CHECKPOINT_MIN_STEP))
            reserved = False
            for count in range(checkpoint_count, -1, -1):
                checkpoint_orders = _angular_checkpoint_orders(top, count)
                nbytes = base_bytes + 16 * values * len(checkpoint_orders)
                if _reserve_angular_bytes(nbytes):
                    reserved = True
                    break
            if not reserved:
                return None
            try:
                built = _AngularChunk(self.k, self.g.rho, self.g.z, thetas, top,
                                      checkpoint_orders=checkpoint_orders)
            except BaseException:
                _release_angular_bytes(nbytes)
                raise
            # No direct fallback means no retained mask. Charge only the
            # arrays actually retained, including aspect vectors and seeds.
            _release_angular_bytes(nbytes - built.nbytes)
            weakref.finalize(built, _release_angular_bytes, built.nbytes)
            with self._angular_shared_lock:
                self._angular_shared[key] = built
            return built

    def _angular_batch(self, m: 'int', thetas: 'np.ndarray', consume: 'bool' = False):
        """Cylindrical-wave data of one mode for a chunk of look directions.

        The excitation and the far-field projection of one aspect batch need
        the same Bessel functions and axial phases.  ``rhs_vv_hh_batch``
        stores every chunk it evaluates in this thread's cache and
        ``farfield_vv_hh_batch`` consumes them (``consume=True``), so each
        mode evaluates them once; a miss recomputes.  The cache holds at most
        one excitation batch per thread and is emptied as it is consumed.
        """
        key = (int(m), thetas.tobytes())
        cache = getattr(self._angular_local, "batches", None)
        if cache is None:
            cache = self._angular_local.batches = {}
        data = cache.pop(key, None) if consume else cache.get(key)
        if data is not None:
            return data
        g = self.g
        chunk = self._shared_angular_chunk(thetas, abs(int(m)) + 1)
        if chunk is not None:
            st, ct, phase = chunk.st, chunk.ct, chunk.phase
            j_lo, j_0, j_hi = chunk.triplet(int(m))
        else:
            th = np.radians(thetas)
            st = np.sin(th)[:, None]
            ct = np.cos(th)[:, None]
            st[(thetas == 0.0) | (thetas == 180.0)] = 0.0
            u = self.k * st * g.rho[None, :]
            phase = np.exp(1j * self.k * ct * g.z[None, :])
            j_lo, j_0, j_hi = _bessel_triplet(int(m), u)
        Jm = (1j) ** m * j_0
        Jm_m1 = (1j) ** (m - 1) * j_lo
        Jm_p1 = (1j) ** (m + 1) * j_hi
        data = (st, ct, phase, math.pi * (Jm_m1 + Jm_p1),
                (math.pi / 1j) * (Jm_m1 - Jm_p1), 2.0 * math.pi * Jm)
        if not consume:
            cache[key] = data
        return data

    def rhs_vv_hh_batch(self, m: 'int', thetas_deg,
                        efie_scale: 'complex' = 1.0,
                        mfie_scale: 'complex' = 0.0,
                        angle_chunk: 'int' = 64) -> 'np.ndarray':
        """Return interleaved VV/HH RHS columns for all requested aspects.

        The result order is ``theta0-VV, theta0-HH, theta1-VV, ...`` to match
        ``_mode_sweep``.  Bessel/phase arrays are bounded by ``angle_chunk``;
        both polarizations and the EFIE/MFIE pieces share each evaluation,
        and the far-field projection of the same batch reuses it.
        """

        thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
        count = len(thetas)
        out = np.zeros((2 * self.Nn, 2 * count), dtype=np.complex128)
        g = self.g
        base_weight = g.w * g.rho
        chunk_size = max(1, int(angle_chunk))
        # A new excitation batch supersedes whatever the last one left behind.
        self._angular_local.batches = {}
        if efie_scale == 0.0 and mfie_scale == 0.0:
            return out
        for i0 in range(0, count, chunk_size):
            i1 = min(i0 + chunk_size, count)
            st, ct, phase, Ic, Is, I1 = self._angular_batch(m, thetas[i0:i1])
            common = phase * base_weight[None, :]

            columns = np.arange(2 * i0, 2 * i1)
            vv = columns[0::2]
            hh = columns[1::2]
            # MFIE testing uses the same four fields as EFIE, permuted and
            # signed. Accumulate each once; preserve EFIE-then-MFIE addition.
            a = self._test_accumulate(
                (common * (ct * g.trho[None, :] * Ic - st * g.tz[None, :] * I1)).T)
            b = self._test_accumulate((common * (-ct * Is)).T)
            c = self._test_accumulate((common * (g.trho[None, :] * Is)).T)
            d = self._test_accumulate((common * Ic).T)
            if efie_scale != 0.0:
                out[:self.Nn, vv] += efie_scale * a
                out[self.Nn:, vv] += efie_scale * b
                out[:self.Nn, hh] += efie_scale * c
                out[self.Nn:, hh] += efie_scale * d
            if mfie_scale != 0.0:
                scale = mfie_scale / ETA0
                out[:self.Nn, vv] += scale * d
                out[self.Nn:, vv] += scale * (-c)
                out[:self.Nn, hh] += scale * (-b)
                out[self.Nn:, hh] += scale * a
        return out

    def farfield_vv_hh_batch(self, m: 'int', solutions: 'np.ndarray',
                             thetas_deg, zs_pt: 'Optional[np.ndarray]' = None,
                             msolutions: 'Optional[np.ndarray]' = None,
                             angle_chunk: 'int' = 64) -> 'np.ndarray':
        """Monostatic VV/HH modal contributions for interleaved solutions."""

        thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
        solutions = np.asarray(solutions)
        if solutions.shape != (2 * self.Nn, 2 * len(thetas)):
            raise ValueError("BoR batch solution matrix has incompatible dimensions.")
        if msolutions is not None:
            msolutions = np.asarray(msolutions)
            if msolutions.shape != solutions.shape:
                raise ValueError(
                    "BoR batch magnetic-current matrix has incompatible dimensions."
                )
        out = np.zeros((2, len(thetas)), dtype=np.complex128)
        g = self.g
        k = self.k
        base_weight = g.w * g.rho
        pref_j = -1j * k * ETA0 / (4.0 * math.pi)
        pref_m = 1j * k / (4.0 * math.pi)
        chunk_size = max(1, int(angle_chunk))
        for i0 in range(0, len(thetas), chunk_size):
            i1 = min(i0 + chunk_size, len(thetas))
            st, ct, phase, Icos, Is_rhs, I1 = self._angular_batch(
                m, thetas[i0:i1], consume=True)
            Isin = -Is_rhs
            common = (phase * base_weight[None, :]).T
            theta_t = (
                ct * g.trho[None, :] * Icos
                - st * g.tz[None, :] * I1
            ).T
            theta_f = (-ct * Isin).T
            phi_t = (g.trho[None, :] * Isin).T
            phi_f = Icos.T

            columns = np.arange(2 * i0, 2 * i1)
            Jt = self._basis_evaluate(solutions[:self.Nn, columns])
            Jf = self._basis_evaluate(solutions[self.Nn:, columns])
            Jt_vv, Jt_hh = Jt[:, 0::2], Jt[:, 1::2]
            Jf_vv, Jf_hh = Jf[:, 0::2], Jf[:, 1::2]
            vv_theta = np.sum(
                common * (Jt_vv * theta_t + Jf_vv * theta_f), axis=0
            )
            hh_phi = np.sum(
                common * (Jt_hh * phi_t + Jf_hh * phi_f), axis=0
            )
            out[0, i0:i1] = pref_j * vv_theta
            out[1, i0:i1] = pref_j * hh_phi

            if zs_pt is not None:
                zs = np.asarray(zs_pt)[:, None]
                Mt_vv, Mf_vv = zs * Jf_vv, -zs * Jt_vv
                Mt_hh, Mf_hh = zs * Jf_hh, -zs * Jt_hh
            elif msolutions is not None:
                Mt = self._basis_evaluate(msolutions[:self.Nn, columns])
                Mf = self._basis_evaluate(msolutions[self.Nn:, columns])
                Mt_vv, Mt_hh = Mt[:, 0::2], Mt[:, 1::2]
                Mf_vv, Mf_hh = Mf[:, 0::2], Mf[:, 1::2]
            else:
                Mt_vv = None
            if Mt_vv is not None:
                vv_phi_m = np.sum(
                    common * (Mt_vv * phi_t + Mf_vv * phi_f), axis=0
                )
                hh_theta_m = np.sum(
                    common * (Mt_hh * theta_t + Mf_hh * theta_f), axis=0
                )
                out[0, i0:i1] -= pref_m * vv_phi_m
                out[1, i0:i1] += pref_m * hh_theta_m
        return out

    def rhs_h_mode(self, m: 'int', theta_inc_deg: 'float', pol: 'str') -> 'np.ndarray':
        """UNROTATED <W, H_inc> (PMCHWT H-row; the MFIE rhs is the rotated
        <W, n_hat x H_inc>).  The incident pair is (E, H):
            VV: E = e_theta P,  H = -(1/eta0) y_hat P
            HH: E = y_hat P,    H = +(1/eta0) e_theta P
        so <W, H_inc> reuses rhs_mode with the polarizations swapped."""

        if pol.upper() in ("VV", "THETA", "TM"):
            return -self.rhs_mode(m, theta_inc_deg, "HH") / ETA0
        return self.rhs_mode(m, theta_inc_deg, "VV") / ETA0


    def farfield_mode(self, m: 'int', sol: 'np.ndarray', theta_s_deg: 'float',
                      zs_pt: 'Optional[np.ndarray]' = None,
                      msol: 'Optional[np.ndarray]' = None) -> 'Tuple[complex, complex]':
        """
        Modal far-field (F_theta, F_phi).  For IBC surfaces the eliminated
        magnetic current M = -Z_s n_hat x J still RADIATES:
            M_t = Z_s J_phi,  M_phi = -Z_s J_t
            F_theta^M = -(jk/4pi) Int M . phi_hat_s e^{jk r_hat . r'}
            F_phi^M   = +(jk/4pi) Int M . theta_hat_s e^{...}
        (Weston's Z_s = eta0 null is exactly the J/M far-field cancellation --
        omitting this term leaves the operator right but the RCS wrong.)
        """

        g = self.g
        k = self.k
        Nn = self.Nn
        st, ct, P, Icos, Is_rhs, I1 = self._angular_data(m, theta_s_deg)


        Isin = -Is_rhs
        Jt = self._basis_evaluate(sol[:Nn])
        Jf = self._basis_evaluate(sol[Nn:])
        common = g.w * g.rho * P

        def proj_theta(Xt, Xf):
            return np.sum(common * (Xt * (ct * g.trho * Icos - st * g.tz * I1) + Xf * (-ct * Isin)))

        def proj_phi(Xt, Xf):
            return np.sum(common * (Xt * g.trho * Isin + Xf * Icos))

        pref_j = -1j * k * ETA0 / (4.0 * math.pi)
        f_theta = pref_j * proj_theta(Jt, Jf)
        f_phi = pref_j * proj_phi(Jt, Jf)
        Mt = Mf = None
        if zs_pt is not None:
            Mt = zs_pt * Jf
            Mf = -zs_pt * Jt
        elif msol is not None:
            Mt = self._basis_evaluate(msol[:Nn])
            Mf = self._basis_evaluate(msol[Nn:])
        if Mt is not None:
            pref_m = 1j * k / (4.0 * math.pi)
            f_theta += -pref_m * proj_phi(Mt, Mf)
            f_phi += pref_m * proj_theta(Mt, Mf)
        return f_theta, f_phi


@contextlib.contextmanager
def _bounded_blas_threads(workers: 'int'):
    """Give each concurrent mode worker a share of the cores for BLAS.

    Mode workers, streaming tiles, and near-pair preparation all run on
    Python threads that call BLAS.  An unconfigured OpenBLAS pool uses every
    core per call, so ``workers`` concurrent LU factorizations oversubscribe
    the machine by ~workers x cores (a 254-DOF LU measured ~9000x slower).
    Limits already set lower by a caller (execution scopes, HPC pinning) are
    never raised.  If another thread currently owns the process-wide BLAS
    controls, the existing limits are left untouched.
    """

    from ghost_backend.execution.options import _BLAS_LOCK
    from ghost_backend.execution.thread_control import (
        threadpool_info,
        threadpool_limits,
    )

    # At most the physical cores (see execution.options.blas_core_budget).
    from ghost_backend.execution.options import blas_core_budget
    per_worker = max(1, blas_core_budget() // max(1, int(workers)))
    current = [
        int(pool["num_threads"])
        for pool in threadpool_info()
        if pool.get("user_api") == "blas"
        and isinstance(pool.get("num_threads"), int)
        and pool["num_threads"] > 0
    ]
    if not current or max(current) <= per_worker:
        yield
        return
    if not _BLAS_LOCK.acquire(blocking=False):
        yield
        return
    try:
        with threadpool_limits(limits=per_worker, user_api="blas"):
            yield
    finally:
        _BLAS_LOCK.release()


class _ModeSweepStopped(Exception):
    """A mode task stopped because its sweep ended below it (never consumed)."""


# A mode's increment decides convergence only when it is consumed, in order,
# so a full window keeps up to ``workers - 1`` modes beyond the converged tail
# in flight (16 modes computed to use 10 on a 2 GHz ogive, 27 for 24 at
# 10 GHz), and the executor waits for them.  The window stops at a predicted
# end: before the tail, the tail start plus the transition allowance of the
# automatic cap (_bor_mode_limits); inside it, the geometric extrapolation of
# the last two relative increments, which errs late because the tail decays
# faster than geometrically.  A short prediction only delays the next mode, a
# tail that stops decaying lifts the limit, and tasks still running past the
# end stop at their next checkpoint.
MODE_WINDOW_MARGIN = 1


def _initial_mode_horizon(tail_start: 'int', m_max: 'int') -> 'int':
    """Last mode started before any tail increment has been seen."""
    b = max(0, int(tail_start))
    allowance = int(math.ceil(4.05 * b ** (1.0 / 3.0) + 2.0))
    return min(int(m_max), b + allowance + MODE_WINDOW_MARGIN)


def _predicted_mode_horizon(mode: 'int', tail_start: 'int', increment: 'float',
                            previous: 'float', mode_tol: 'float', m_max: 'int',
                            current: 'int') -> 'int':
    """Last mode worth starting once mode ``mode`` has been consumed.

    Convergence is two consecutive relative increments below ``mode_tol`` at
    or above ``tail_start``.
    """
    m_max = int(m_max)
    if mode < tail_start:
        return current
    if increment < mode_tol:
        return min(m_max, int(mode) + 1 + MODE_WINDOW_MARGIN)
    if not (math.isfinite(increment) and math.isfinite(previous)):
        return current
    if not 0.0 < increment < previous:
        return m_max                      # not decaying: no prediction
    steps = int(math.ceil(math.log(mode_tol / increment) / math.log(increment / previous)))
    return min(m_max, int(mode) + max(1, steps) + 1 + MODE_WINDOW_MARGIN)


def _mode_sweep_impl(n_dofs: 'int', thetas, pols, m_max: 'int', mode_tol: 'float',
                     assemble: 'Callable', rhs: 'Callable', farfield: 'Callable',
                     prepare: 'Optional[Callable]' = None, workers: 'int' = 1,
                     progress: 'Optional[Callable]' = None,
                     check_abort: 'Optional[Callable]' = None,
                     monitor_cond: 'bool' = False,
                     rhs_batch: 'Optional[Callable]' = None,
                     farfield_batch: 'Optional[Callable]' = None,
                     min_mode_before_tail: 'int' = 0,
                     assembly_peak_gb: 'float' = 0.0,
                     memory_context: 'str' = "The BoR solve",
                     signed_mode_symmetry: 'bool' = False,
                     stream_mode_block: 'Optional[int]' = None,
                     coordinates: 'Optional[Callable]' = None,
                     mirror: 'Optional[Callable]' = None,
                     hierarchical_pricing: 'bool' = True,
                     axial_mode_only: 'bool' = False,
                     exact_far_cache: 'Optional[Callable]' = None,
                     preparation_workers: 'Optional[int]' = None):
    """
    Shared adaptive azimuthal-mode loop for every BoR formulation.

    ``coordinates(m)``, when given, returns the meridian position of every
    reduced unknown of mode ``m``: large mode systems are then factored
    hierarchically (:class:`bor.factor.ModalFactor`) and priced as such.
    ``mirror(m)``, for a body symmetric about a plane normal to its axis,
    returns the ``(target, sign)`` mirror map of those unknowns: every mode is
    then factored as its even and odd halves (:class:`bor.factor.MirrorSplit`).

    assemble(m) -> (A_masked, mask); rhs(m, theta, pol) -> V (full, unmasked);
    farfield(m, full_sol, theta, pol) -> complex modal far-field contribution.

    mask=None means the closures own the reduction: rhs returns the REDUCED
    right-hand side and farfield receives the REDUCED solution vector (used
    by the junction solver, whose constraint reduction A_red = Q^T A Q is
    not expressible as a boolean mask).

    Each mode's system is factored ONCE and reused across bounded aspect
    batches. Every recovered physical RHS retains its residual check; an
    incident basis is retained only for the lifetime of that mode's factor.  Modes are independent, so waves of
    `workers` modes run on threads (BLAS releases the GIL); call prepare
    first so kernel/near caches are read-only during the parallel section.
    Accumulation and the 2-quiet-modes truncation test remain in strict mode
    order, so results are identical to the serial loop.  Every polarization
    and look must satisfy the tail tolerance independently; a strong return
    cannot provide the normalization for an unrelated weak channel.  Tail convergence is
    not eligible before ``min_mode_before_tail``.  Production callers set
    that floor from the incident-wave azimuthal bandwidth ``k*rho*sin(theta)``
    so two accidentally quiet low modes cannot terminate an electrically
    large solve before its physically expected modal content is reached.
    """

    metrics = active_metrics()
    if metrics is not None:


        assemble = metrics.wrap("modal_assembly", assemble)
        prepare = metrics.wrap("operators", prepare)
        rhs = metrics.wrap("excitation", rhs)
        rhs_batch = metrics.wrap("excitation", rhs_batch)
        farfield = metrics.wrap("far_field", farfield)
        farfield_batch = metrics.wrap("far_field", farfield_batch)
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    axial_mode_only = bool(axial_mode_only and _all_axial_aspects(thetas) and m_max >= 1)
    pols = list(pols)
    mode_tol = float(mode_tol)
    if not math.isfinite(mode_tol) or mode_tol <= 0.0:
        raise ValueError("mode_tol must be a positive finite value.")
    workers = max(1, int(workers))
    n_rhs = len(thetas) * len(pols)
    from ghost_backend.bor.options import output_reserved_gb
    assembly_peak_gb += output_reserved_gb()
    worker_plan = plan_bor_mode_workers(n_dofs, n_rhs, workers,
        1 if axial_mode_only else max(1, int(m_max) + 1), assembly_peak_gb,
        hierarchical=coordinates is not None and hierarchical_pricing,
        mirrored=mirror is not None, preparation_workers=preparation_workers)
    workers = worker_plan['workers']
    near_plan = worker_plan['near_preparation']
    _guard_bor_dense_memory(
        n_dofs,
        n_rhs,
        workers,
        1 if axial_mode_only else max(1, int(m_max) + 1),
        assembly_peak_gb=assembly_peak_gb,
        context=memory_context,
        streaming=stream_mode_block is not None,
        preparation_peak_gb=assembly_peak_gb + near_plan['scratch_gb'],
        hierarchical=coordinates is not None and hierarchical_pricing,
        mirrored=mirror is not None,
    )
    if prepare is not None:
        from ghost_backend.bor.near_parallel import process_scope
        with _NEAR_WORKER_LIMIT.override(near_plan['workers']), _bounded_blas_threads(near_plan['workers']), process_scope(near_plan['workers'], near_plan['process_workers']) as process_state:
            prepare(m_max)
        near_plan['backend'] = 'processes' if process_state['executor'] is not None else 'threads'
        near_plan['process_pair_jobs'] = process_state['jobs']
        if process_state['executor'] is not None and not near_plan['process_overhead_bytes_per_worker']:
            # The workload was unknown when planning; record what actually ran.
            # process_workers was sized against the same limit, so it fits.
            from ghost_backend.bor.near_parallel import PROCESS_OVERHEAD_BYTES
            near_plan['process_overhead_bytes_per_worker'] = PROCESS_OVERHEAD_BYTES
            near_plan['scratch_gb'] = max(near_plan['scratch_gb'], near_plan['process_workers']
                * (_NEAR_TASK_SCRATCH_BYTES + PROCESS_OVERHEAD_BYTES) / 1.e9)
    from ghost_backend.bor.options import current_mode_resume
    resume = current_mode_resume()
    F = (resume['fields'] if resume and 'fields' in resume else
         np.zeros((len(pols), len(thetas)), dtype=np.complex128))
    min_mode_before_tail = max(0, int(min_mode_before_tail))

    from ghost_backend.bor.factor import ModalFactor, compressed_factor, residual_storage_settings
    from ghost_backend.linalg.sweep import CompressionHint, SweepBasis, solve as solve_sweep
    from scipy.sparse import issparse
    options = current_options()
    has_exact_far_cache = bool(exact_far_cache is not None and exact_far_cache())
    from ghost_backend.bor.options import current_recycling_parameters
    recycling_identity, recycling_frequency = current_recycling_parameters()
    batch_size = options['angle_batch_size']
    # Resolved here: mode tasks run on executor threads without this context.
    residual_storage = residual_storage_settings()
    # One 'auto' RHS-compression memory for all modes of the sweep: once a
    # mode's compression does not pay, later modes skip the QR attempt
    # (re-probing with backoff) instead of paying it and falling back.
    compression_hint = CompressionHint()
    from ghost_backend.bor.cache import current_cache, cache_scope
    tile_cache = current_cache()

    # Set when the sweep ends: a mode task above it (in flight past the
    # converged tail, or behind a failure) stops at its next checkpoint
    # instead of running to completion under the executor's shutdown.
    stop_after = [None]

    def task_checkpoint(am):
        def checkpoint():
            if check_abort is not None:
                check_abort()
            if stop_after[0] is not None and am > stop_after[0]:
                raise _ModeSweepStopped()
        return checkpoint

    def solve_am(am):
        checkpoint = task_checkpoint(am)
        dF = np.zeros_like(F)
        res = backward = cond = 0.0
        refinement_count = 0
        events = []
        signed_modes = [0] if am == 0 else ([am] if signed_mode_symmetry else [am, -am])
        for m in signed_modes:
            checkpoint()
            A, mask = assemble(m)
            # The assembled system is this task's own buffer: a large one is
            # factored in place with its coefficients spooled for residuals.
            factor = (compressed_factor(A, m, monitor_cond, options, min(workers, m_max+1), checkpoint,
                                        recycling_key=None if recycling_identity is None else (recycling_identity, int(m)),
                                        recycling_frequency=recycling_frequency,
                                        exact_far_cache=has_exact_far_cache)
                      if isinstance(A, TileExpression)
                      else ModalFactor(A, m, monitor_cond, checkpoint, owned=True,
                                       residual_storage=residual_storage,
                                       coordinates=None if coordinates is None else coordinates(m),
                                       mirror=None if mirror is None else mirror(m)))
            A = None
            basis = SweepBasis(min(256, batch_size * len(pols)))
            reduction = None if mask is None else (mask if issparse(mask) else np.asarray(mask))
            projected = reduction is not None and (issparse(reduction) or reduction.ndim == 2)
            if projected:
                reduction = csr_matrix(reduction)
            for i0 in range(0, len(thetas), batch_size):
                checkpoint()
                i1 = min(i0 + batch_size, len(thetas))
                angles = thetas[i0:i1]
                if rhs_batch is not None:
                    full_B = np.asarray(rhs_batch(m, angles, pols))
                    expected = (n_dofs, len(angles) * len(pols))
                    if full_B.shape != expected:
                        raise RuntimeError('BoR mode m={} batch excitation has shape {}, expected {}.'.format(m, full_B.shape, expected))
                    B = full_B if reduction is None else (reduction.conj().T @ full_B if projected else full_B[reduction])
                    full_B = None
                else:
                    def column(theta, pol):
                        value = rhs(m, theta, pol)
                        return value if reduction is None else (reduction.conj().T @ value if projected else value[reduction])
                    B = np.column_stack([column(th, pol) for th in angles for pol in pols])
                X = solve_sweep(factor, B, basis, setting=options['rhs_compression'],
                                hint=compression_hint)
                B = None
                if farfield_batch is not None:
                    # Bound expansion and angular quadrature independently of the solve batch.
                    for j0 in range(0, len(angles), 64):
                        checkpoint()
                        j1 = min(j0 + 64, len(angles))
                        columns = X[:, j0 * len(pols):j1 * len(pols)]
                        if reduction is None:
                            full_X = columns
                        elif projected:
                            full_X = reduction @ columns
                        else:
                            full_X = np.zeros((n_dofs, columns.shape[1]), complex)
                            full_X[reduction] = columns
                        contributions = np.asarray(farfield_batch(m, full_X, angles[j0:j1], pols))
                        expected = (len(pols), j1-j0)
                        if contributions.shape != expected or not np.all(np.isfinite(contributions)):
                            raise RuntimeError('BoR mode m={} produced an invalid or non-finite batch far-field contribution.'.format(m))
                        dF[:, i0+j0:i0+j1] += contributions
                        full_X = columns = contributions = None
                else:
                    for it, th in enumerate(angles):
                        for ip, pol in enumerate(pols):
                            sol = X[:, it * len(pols) + ip]
                            if projected:
                                sol = reduction @ sol
                            elif reduction is not None:
                                full = np.zeros(n_dofs, complex)
                                full[reduction] = sol
                                sol = full
                            contribution = complex(farfield(m, sol, th, pol))
                            if not math.isfinite(contribution.real) or not math.isfinite(contribution.imag):
                                raise RuntimeError('BoR mode m={} produced a non-finite far-field contribution.'.format(m))
                            dF[ip, i0+it] += contribution
                    sol = full = None
                X = None
            res = max(res, factor.event['max_relative_residual'])
            backward = max(backward, factor.event['max_backward_error'])
            refinement_count += factor.event['refinement_steps']
            if monitor_cond:
                cond = max(cond, factor.condition)
            retain = getattr(factor, 'retain_preconditioner', None)
            if retain is not None:
                retain()
            events.append(dict(factor.event, mode=int(m)))
            # Release both factors and retained incident basis before assembling the next mode.
            close = getattr(factor, 'close', None)
            if close is not None:
                close()
            factor = basis = A = reduction = mask = None
        if signed_mode_symmetry and am > 0:
            dF *= 2.0
        return dF, res, backward, refinement_count, cond, events

    # Executor threads start with an empty context: every mode runs in a copy
    # of the caller's, so the execution options (the unit's CPU allocation,
    # residual storage, temporary directory, ...) reach the mode workers too.
    # Each concurrent mode lends only its share to nested assembly/product
    # teams. BLAS is bounded once around the executor below, because its
    # process-wide limits cannot be changed independently by mode threads.
    from ghost_backend.execution.options import allocated_cpu_budget, cpu_allocation_scope
    mode_cpu_budget = max(1, allocated_cpu_budget() // workers)
    caller_context = contextvars.copy_context()

    def scoped_solve_am(am):
        def run():
            with cpu_allocation_scope(mode_cpu_budget), option_scope(options):
                with metrics_scope(metrics):
                    with cache_scope(tile_cache):
                        return solve_am(am)
        return caller_context.copy().run(run)

    mode_events = []
    modes_used = 0
    quiet = 0
    am = 0
    max_res = 0.0
    max_backward_error = 0.0
    refinement_steps = 0
    last_relative_increment = math.inf
    last_absolute_increment = math.inf
    last_absolute_floor = 0.0
    worst_tail_index = (0, 0)
    conds: 'List[float]' = []
    if resume and 'fields' in resume:
        modes_used, quiet = resume['modes_used'], resume['quiet']
        am = modes_used + 1
        mode_events, conds = resume['events'], resume['conditions']
        max_res, max_backward_error = resume['residual'], resume['backward_error']
        refinement_steps = resume['refinement_steps']
        last_relative_increment, last_absolute_increment = resume['increments']
        last_absolute_floor, worst_tail_index = resume['absolute_floor'], resume['worst_tail_index']
    from collections import deque

    def block_of(mode):
        return None if stream_mode_block is None else mode // stream_mode_block

    with _bounded_blas_threads(workers), ThreadPoolExecutor(max_workers=workers) as ex:
        # A sliding window of `workers` modes in flight, consumed in strict
        # mode order: a slow mode no longer idles the other workers until a
        # whole wave has finished.  The window never crosses a streamed mode
        # block while modes of the previous block are pending (RAM throttling
        # can break the worker/block alignment; the resident block is
        # finished before any worker advances the stream cache).
        pending = deque()
        next_am = 1 if axial_mode_only else am
        # The last mode worth starting (_predicted_mode_horizon).
        horizon = _initial_mode_horizon(min_mode_before_tail, m_max)
        previous_increment = math.inf
        try:
            while quiet < 2 and (pending or next_am <= m_max):
                if check_abort is not None:
                    check_abort()
                while len(pending) < workers and next_am <= (1 if axial_mode_only else m_max) and (
                        not pending or (next_am <= horizon
                                        and block_of(next_am) == block_of(pending[0][0]))):
                    pending.append((next_am, ex.submit(scoped_solve_am, next_am)))
                    next_am += 1
                w_am, future = pending.popleft()
                dF, res, backward, refined, cond, events = future.result()
                del future
                mode_events.extend(events)
                F += dF
                if not np.all(np.isfinite(F)):
                    raise RuntimeError(
                        f"BoR modal accumulation became non-finite at "
                        f"|m|={w_am}; no field is returned."
                    )
                max_res = max(max_res, res)
                max_backward_error = max(max_backward_error, backward)
                refinement_steps += int(refined)
                if monitor_cond:
                    conds.append(cond)
                modes_used = w_am
                field_abs = np.abs(F)
                increment_abs = np.abs(dF)
                global_scale = max(float(np.max(field_abs)), 1e-300)


                last_absolute_floor = (
                    float(mode_tol) * float(mode_tol) * global_scale
                )
                effective_scale = (
                    field_abs + float(mode_tol) * global_scale
                )
                relative_by_sample = np.divide(
                    increment_abs,
                    effective_scale,
                    out=np.zeros_like(increment_abs, dtype=float),
                    where=effective_scale > 0.0,
                )
                worst_flat = int(np.argmax(relative_by_sample))
                worst_tail_index = tuple(
                    int(value)
                    for value in np.unravel_index(
                        worst_flat, relative_by_sample.shape
                    )
                )
                last_relative_increment = float(
                    relative_by_sample[worst_tail_index]
                )
                last_absolute_increment = float(
                    increment_abs[worst_tail_index]
                )
                if axial_mode_only:
                    # Rotational invariance makes the other incident Fourier
                    # coefficients identically zero. This is an analytic
                    # finite expansion, not a relaxed numerical tail test.
                    quiet = 2
                    last_relative_increment = last_absolute_increment = 0.0
                elif (
                    w_am >= min_mode_before_tail
                    and last_relative_increment < mode_tol
                ):
                    quiet += 1
                else:
                    quiet = 0
                horizon = _predicted_mode_horizon(w_am, min_mode_before_tail,
                                                  last_relative_increment, previous_increment,
                                                  mode_tol, m_max, horizon)
                previous_increment = last_relative_increment
                if progress is not None:
                    progress(modes_used, m_max)
        finally:
            # Converged (or failed): modes queued beyond the tail never start,
            # and those already running stop at their next checkpoint.
            stop_after[0] = modes_used
            for _, future in pending:
                future.cancel()
    stats = {
        "near_preparation": near_plan,
        "modal_execution": dict(options, systems=mode_events, worker_plan=worker_plan,
                                cpu_budget_per_mode=mode_cpu_budget,
                                rhs_compression_hint=compression_hint.evidence()),
        "linear_residual": max_res,
        "linear_backward_error": max_backward_error,
        "linear_refinement_steps": int(refinement_steps),
        "mode_converged": bool(quiet >= 2),
        "mode_tasks_started": 1 if axial_mode_only else int(next_am),
        "mode_selection": "exact_axial_first_order" if axial_mode_only else "adaptive_tail",
        "mode_cap": int(m_max),
        "mode_tail_start": int(min_mode_before_tail),
        "mode_quiet_count": int(quiet),
        "mode_last_relative_increment": float(last_relative_increment),
        "mode_last_absolute_increment": float(last_absolute_increment),
        "mode_tail_absolute_floor": float(last_absolute_floor),
        "mode_worst_polarization": (
            str(pols[worst_tail_index[0]]) if pols else None
        ),
        "mode_worst_theta_deg": (
            float(thetas[worst_tail_index[1]]) if thetas.size else None
        ),
        "signed_mode_symmetry_used": bool(signed_mode_symmetry),
        "linear_residual_limit": float(BOR_LINEAR_RESIDUAL_MAX),
        "linear_residual_limit_kind": "iterative_refinement_advisory",
        "linear_backward_error_limit": float(
            BOR_LINEAR_BACKWARD_ERROR_MAX
        ),
        "condition_est_computed": bool(monitor_cond),
        "condition_est_method": (
            ("compressed_original_1norm_onenormest" if options["factorization"] == "compressed" else "lapack_gecon_1norm") if monitor_cond else None
        ),
        "condition_est_limit": float(BOR_CONDITION_EST_MAX),
    }
    if monitor_cond and conds:
        stats["max_cond"] = max(conds)
        stats["median_cond"] = float(np.median(conds))
    if resume is not None:
        resume.clear()
        if quiet < 2:
            resume.update(fields=F, modes_used=modes_used, quiet=quiet,
                events=mode_events, conditions=conds, residual=max_res,
                backward_error=max_backward_error, refinement_steps=refinement_steps,
                increments=(last_relative_increment, last_absolute_increment),
                absolute_floor=last_absolute_floor, worst_tail_index=worst_tail_index)
    return F, modes_used, stats


@functools.wraps(_mode_sweep_impl)
def _mode_sweep(*args, **kwargs):
    return _mode_sweep_impl(*args, **kwargs)


def _mode_cap_warning(stats: 'Dict', mode_tol: 'float') -> 'Optional[str]':
    """Return a standard warning when adaptive modal truncation hit its cap."""

    if bool(stats.get("mode_converged", False)):
        return None
    cap = int(stats.get("mode_cap", -1))
    tail_start = int(stats.get("mode_tail_start", 0))
    tail = float(stats.get("mode_last_relative_increment", math.inf))
    if cap < tail_start:
        return (
            f"Azimuthal mode cap m={cap} is below the incident-wave physical "
            f"bandwidth estimate m={tail_start}. Increase n_modes; adaptive "
            "tail convergence was intentionally not evaluated."
        )
    return (
        "Azimuthal mode truncation did not reach two consecutive increments "
        f"below mode_tol={float(mode_tol):.3g} at or above the physical "
        f"tail start m={tail_start} before the cap m={cap} "
        f"(last relative increment {tail:.3g}). Increase n_modes or verify "
        "mode convergence before trusting this result."
    )

def _require_mode_convergence(stats: 'Dict', mode_tol: 'float') -> 'None':
    """Fail before publishing a field whose azimuthal series is unconverged."""

    message = _mode_cap_warning(stats, mode_tol)
    if message:
        from ghost_backend.bor.options import ModalConvergenceError
        raise ModalConvergenceError(
            message
            + " No RCS/amplitude result is returned for an unconverged "
              "modal truncation.", stats.get('mode_cap', 0),
            tail=(stats.get('mode_last_relative_increment')
                  if int(stats.get('mode_cap', -1)) >= int(stats.get('mode_tail_start', 0)) else None),
            tolerance=float(mode_tol)
        )


def _segments_intersect_2d(
    a: 'np.ndarray',
    b: 'np.ndarray',
    c: 'np.ndarray',
    d: 'np.ndarray',
    tol: 'float',
) -> 'bool':
    """Inclusive segment intersection with a length-scaled cross tolerance."""

    def cross(p, q, r) -> 'float':
        return float((q[0] - p[0]) * (r[1] - p[1])
                     - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r, cross_tol) -> 'bool':
        return (
            abs(cross(p, q, r)) <= cross_tol
            and min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol
            and min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol
        )

    length_scale = max(
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(d - c)),
        tol,
    )
    cross_tol = tol * length_scale
    o1 = cross(a, b, c)
    o2 = cross(a, b, d)
    o3 = cross(c, d, a)
    o4 = cross(c, d, b)
    if (
        ((o1 > cross_tol and o2 < -cross_tol)
         or (o1 < -cross_tol and o2 > cross_tol))
        and ((o3 > cross_tol and o4 < -cross_tol)
             or (o3 < -cross_tol and o4 > cross_tol))
    ):
        return True
    return (
        on_segment(a, c, b, cross_tol)
        or on_segment(a, d, b, cross_tol)
        or on_segment(c, a, d, cross_tol)
        or on_segment(c, b, d, cross_tol)
    )


def _validate_solve_bor_generatrix(points, formulation: 'str') -> 'np.ndarray':
    """Validate the direct single-surface ``solve_bor`` geometry.

    EFIE may represent an open shell.  CFIE/MFIE require the supported
    closed-body topology: both ends on the rotation axis and traversal from
    the +z pole to the -z pole.  Every path rejects intersections because a
    self-crossing meridian generates an overlapping/non-manifold surface.
    """

    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.shape[0] < 2:
        raise ValueError(
            "BoR generatrix must be a finite (N, 2) array of (rho, z) "
            "nodes with N >= 2."
        )
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("BoR generatrix coordinates must be real.")
    try:
        pts = np.asarray(np.real(raw), dtype=float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("BoR generatrix coordinates must be real numbers.") from exc
    if not np.all(np.isfinite(pts)):
        raise ValueError("BoR generatrix coordinates must all be finite.")

    diag = max(
        float(np.ptp(pts[:, 0])),
        float(np.ptp(pts[:, 1])),
        1.0e-15,
    )
    geom_tol = max(1.0e-14, 1.0e-10 * diag)
    axis_tol = 1.0e-12 * max(1.0, float(np.max(np.abs(pts[:, 0]))))
    if np.any(pts[:, 0] < -axis_tol):
        bad = float(np.min(pts[:, 0]))
        raise ValueError(
            f"BoR generatrix rho coordinates must be >= 0; found {bad:.6g}."
        )
    pts[np.abs(pts[:, 0]) <= axis_tol, 0] = 0.0

    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.any(lengths <= geom_tol):
        idx = int(np.argmin(lengths))
        raise ValueError(
            f"BoR generatrix element {idx} has zero or near-zero length "
            f"({lengths[idx]:.3g})."
        )
    if float(np.max(pts[:, 0])) <= axis_tol:
        raise ValueError(
            "BoR generatrix lies entirely on the rotation axis and sweeps "
            "zero surface area."
        )


    n_elem = len(pts) - 1
    seg_a = pts[:-1]
    seg_b = pts[1:]
    mids = 0.5 * (seg_a + seg_b)
    half = 0.5 * lengths
    max_half = float(np.max(half))
    tree = cKDTree(mids)
    for i in range(n_elem):
        radius = float(half[i]) + max_half + geom_tol
        for j in tree.query_ball_point(mids[i], radius):
            j = int(j)
            if j <= i + 1:
                continue
            lower = (
                float(np.linalg.norm(mids[i] - mids[j]))
                - float(half[i]) - float(half[j])
            )
            if lower > geom_tol:
                continue
            if _segments_intersect_2d(
                pts[i], pts[i + 1], pts[j], pts[j + 1], geom_tol
            ):
                raise ValueError(
                    "BoR generatrix self-intersection/overlap between "
                    f"elements {i} and {j} is not supported."
                )

    start_on_axis = pts[0, 0] == 0.0
    end_on_axis = pts[-1, 0] == 0.0
    closed_body = start_on_axis and end_on_axis
    if formulation in {"cfie", "mfie"} and not closed_body:
        raise ValueError(
            f"{formulation.upper()} requires a closed BoR whose two "
            "generatrix endpoints lie on the rotation axis; use EFIE for "
            "an intentionally open shell."
        )
    if closed_body and not (pts[0, 1] > pts[-1, 1] + geom_tol):
        raise ValueError(
            "A closed BoR generatrix must be traversed from its +z axis "
            "endpoint to its -z axis endpoint so the left-of-travel normal "
            "faces the exterior."
        )
    from ghost_backend.bor.geometry import require_resolved_corners
    require_resolved_corners(pts)
    return pts


def estimate_bor_table_gb(n_elems: 'int', m_max: 'int', formulation: 'str' = "cfie",
                          has_ibc: 'bool' = False, gauss_order: 'int' = FAR_GAUSS_ORDER,
                          single_tables: 'bool' = False) -> 'float':
    """Estimate persistent far-table storage for all modal FFT tables in GB."""

    P = float(gauss_order * n_elems)
    per = 8.0 if single_tables else 16.0
    total = P * P * (m_max + 2) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * P * P * (m_max + 1) * per
    if has_ibc:
        total += 4.0 * P * P * (m_max + 1) * per
    return total / 1e9


def estimate_bor_cross_table_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    test_gauss_order: 'int' = FAR_GAUSS_ORDER, source_gauss_order: 'int' = FAR_GAUSS_ORDER,
    single_tables: 'bool' = False, has_rotated_pv: 'bool' = True,
) -> 'float':
    """Persistent far tables for one directed rectangular T/P mapping."""

    test_points = float(int(test_gauss_order) * int(n_test_elements))
    source_points = float(int(source_gauss_order) * int(n_source_elements))
    modes = int(m_max)
    if test_points <= 0.0 or source_points <= 0.0 or modes < 0:
        raise ValueError("Cross-table estimate dimensions are invalid.")
    item_bytes = 8.0 if single_tables else 16.0
    modal_values = (modes + 2) + int(has_rotated_pv) * 4.0 * (modes + 1)
    return test_points * source_points * modal_values * item_bytes / 1.0e9


# Allowance on retained operator storage, applied by the run-time gates and by
# the dispatch previews that mirror them from mesh counts.
BOR_RETAINED_STORAGE_FACTOR = 1.10


def conductor_operator_kinds(formulation: 'str', has_impedance: 'bool'):
    """``(efie, mfie, ibc)``: the operator families a conductor solve prepares.

    One description for ``solve_bor``, its admission gate and the dispatch
    previews. Every enabled family retains its own far tables and its own
    same-surface near contractions, so a CFIE with a nonzero impedance holds
    three of each. The previews assumed two per surface and could fall below
    the gate they predict on bodies with many geometrically close panels.
    """
    efie, mfie = formulation in ("efie", "cfie"), formulation in ("cfie", "mfie")
    return efie, mfie, bool(has_impedance) and efie


def bor_near_cache_bytes(pair_count, node_count, kinds, modes) -> 'float':
    """Retained same-surface near contractions of one medium side.

    ``kinds`` operator families, each holding four component blocks of
    ``4*pair_count`` basis pairs per retained mode and its routing indices.
    The run-time gate and the snapshot preview both price them here.
    """
    index_bytes = np.dtype(np.intp).itemsize
    values = int(kinds) * 4.0 * (int(modes) + 1) * (4 * int(pair_count)) * _COMPLEX128_BYTES
    return values + int(kinds) * (4.0 * (4 * int(pair_count)) + int(node_count) + 1) * index_bytes


def bor_cross_near_cache_bytes(pair_count, modes, has_rotated_pv=True) -> 'float':
    """Retained near contractions of one directed cross-surface operator."""
    return int(pair_count) * (1 + int(has_rotated_pv)) * 4 * (int(modes) + 1) * 4 * _COMPLEX128_BYTES


from collections import namedtuple

# What one medium side of a surface, and one directed cross-surface operator,
# contribute to operator storage.  The run-time gate fills these from the
# constructed solvers; the dispatch previews fill them from mesh counts and
# geometric near-pair counts.  ``fft_samples`` is ``(scalar, bracket)`` angular
# sample counts, or ``None`` where only the bound ``FFT_BUILD_BUDGET`` is known.
BorSurfaceStorage = namedtuple(
    'BorSurfaceStorage', 'points nodes table_bytes near_pairs efie mfie ibc fft_samples')
BorCrossStorage = namedtuple(
    'BorCrossStorage', 'points_p points_q table_bytes near_pairs near_max_order fft_samples has_rotated_pv',
    defaults=(True,))


def bor_operator_storage_bytes(modes, surfaces, crosses=(), constraint_dofs=0,
                               streaming=False, compressed=False):
    """The one BoR operator-storage model, in bytes, by component.

    ``tables`` are the retained far tables (absent when streaming, whose
    planner prices far blocks itself). ``basis`` is retained as a zero-valued
    reporting field: production contracts local bases without dense matrices.
    ``near`` the same- and cross-surface near contractions, ``projection`` the
    junction projection matrices, ``fft_workspace`` and ``near_workspace`` the
    largest build scratch of each kind.  A gate needs
    ``BOR_RETAINED_STORAGE_FACTOR * (tables + basis + near + projection) +
    max(fft_workspace, near_workspace)``; the previews add the two workspaces
    instead, so they never fall below the gate they predict.  The function is
    monotone in every record field: records that bound the solvers' own bound
    the result.
    """

    from ghost_backend.bor import kernels
    modes = int(modes)
    signed_modes = 2 * modes + 1
    budget = float(FFT_BUILD_BUDGET)
    tables = basis = near = fft_workspace = near_workspace = 0.0

    def fft_bytes(pair_points, samples, arrays):
        if samples is None:
            return budget
        return min(budget, pair_points * samples * arrays * _COMPLEX128_BYTES)

    for surface in surfaces:
        kinds = int(bool(surface.efie)) + int(bool(surface.mfie)) + int(bool(surface.ibc))
        square = float(surface.points) ** 2
        if not streaming and kinds:
            if surface.efie:
                tables += square * (modes + 2) * surface.table_bytes
            if surface.mfie:
                tables += 4.0 * square * (modes + 1) * surface.table_bytes
            if surface.ibc:
                tables += 4.0 * square * (modes + 1) * surface.table_bytes
        near += bor_near_cache_bytes(surface.near_pairs, surface.nodes, kinds, modes)
        if kinds:
            near_workspace = max(near_workspace, 128.0e6)
        if not streaming and kinds:
            if (surface.mfie or surface.ibc) and not kernels.BANDED_FFT:
                fft_workspace = max(fft_workspace, 4.0 * square * signed_modes * _COMPLEX128_BYTES)
            scalar, bracket = surface.fft_samples if surface.fft_samples is not None else (None, None)
            if surface.efie:
                fft_workspace = max(fft_workspace, fft_bytes(square, scalar, 4.0))
            if surface.mfie or surface.ibc:
                fft_workspace = max(fft_workspace, fft_bytes(square, bracket, 18.0 if surface.ibc else 16.0))

    for cross in crosses:
        pair_points = float(cross.points_p) * float(cross.points_q)
        if not streaming:
            tables += pair_points * (modes + 2) * cross.table_bytes
            if cross.has_rotated_pv:
                tables += 4.0 * pair_points * (modes + 1) * cross.table_bytes
            if cross.has_rotated_pv and not kernels.BANDED_FFT:
                fft_workspace = max(fft_workspace, 4.0 * pair_points * signed_modes * _COMPLEX128_BYTES)
            scalar, bracket = cross.fft_samples if cross.fft_samples is not None else (None, None)
            fft_workspace = max(fft_workspace, fft_bytes(pair_points, scalar, 4.0))
            if cross.has_rotated_pv:
                fft_workspace = max(fft_workspace, fft_bytes(pair_points, bracket, 18.0))
        near += bor_cross_near_cache_bytes(cross.near_pairs, modes, cross.has_rotated_pv)
        if cross.near_pairs:
            near_workspace = max(near_workspace, 128.0e6, 64.0e6 + 768 * int(cross.near_max_order) ** 2)

    projection = 0.0
    if constraint_dofs:
        # Categories m = 0, 1 and |m| >= 2 (signed-mode symmetry never builds
        # m = -1).  Every backend stores the transforms sparse: one selection
        # entry per column plus a few relations, priced with CSR overhead
        # (value, column index and row pointer, about 6 complex values per row).
        category_count = 1 if modes == 0 else (2 if modes == 1 else 3)
        projection = category_count * float(constraint_dofs) * 6 * _COMPLEX128_BYTES
    return dict(tables=tables, basis=basis, near=near, projection=projection,
                fft_workspace=fft_workspace, near_workspace=near_workspace)


def estimate_bor_operator_storage_gb(
    m_max: 'int',
    solver_requirements,
    cross_operators=(),
    constraint_dofs: 'int' = 0,
    streaming: 'bool' = False,
) -> 'float':
    """Estimate retained operator auxiliaries and their build workspace.

    ``solver_requirements`` contains ``(solver, efie, mfie, ibc)`` tuples.
    Repeated solver instances are merged so their tables are counted once.
    Cross-surface operators are counted independently.
    ``constraint_dofs`` adds the sparse junction projection transforms retained by
    the partial/multiregion paths.  With ``streaming=True``, far tables and
    their FFT workspace are excluded because the combined streaming planner
    accounts for those blocks and sampling tiles separately; exact same- and
    cross-surface near caches plus junction projections remain included.
    """

    try:
        modes = int(m_max)
        constraint_size = int(constraint_dofs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "BoR operator estimates require integer mode and constraint sizes."
        ) from exc
    if modes != m_max or modes < 0:
        raise ValueError("BoR operator mode cap must be a non-negative integer.")
    if constraint_size != constraint_dofs or constraint_size < 0:
        raise ValueError("BoR constraint DOFs must be a non-negative integer.")

    merged = {}
    for requirement in solver_requirements:
        if len(requirement) != 4:
            raise ValueError(
                "Each BoR solver requirement must be (solver, efie, mfie, ibc)."
            )
        solver, efie, mfie, ibc = requirement
        key = id(solver)
        if key in merged:
            previous = merged[key]
            merged[key] = (
                solver,
                previous[1] or bool(efie),
                previous[2] or bool(mfie),
                previous[3] or bool(ibc),
            )
        else:
            merged[key] = (solver, bool(efie), bool(mfie), bool(ibc))

    compressed = compressed_requested()
    streaming = streaming or compressed

    def samples(wavenumber, rho_max, far_gap):
        return (n_xi_for_pairs(wavenumber, rho_max, modes, far_gap, bracket=False),
                n_xi_for_pairs(wavenumber, rho_max, modes, far_gap, bracket=True))

    surfaces = []
    for solver, efie, mfie, ibc in merged.values():
        tabled = not streaming and (efie or mfie or ibc)
        surfaces.append(BorSurfaceStorage(
            int(solver.P), int(solver.Nn), np.dtype(solver._table_dtype).itemsize,
            int(solver._near_pair_count), efie, mfie, ibc,
            samples(solver.k, float(np.max(solver.gen.nodes[:, 0])), float(solver._far_gap())) if tabled else None))
    crosses, seen_crosses = [], set()
    for cross in cross_operators:
        # Derived reverse operators retain nothing of their own.
        if id(cross) in seen_crosses or getattr(cross, 'derived', False):
            continue
        seen_crosses.add(id(cross))
        rho_max = max(float(np.max(cross.sp.gen.nodes[:, 0])), float(np.max(cross.sq.gen.nodes[:, 0])))
        crosses.append(BorCrossStorage(
            int(cross.sp.P), int(cross.sq.P), np.dtype(cross.sp._table_dtype).itemsize,
            len(cross.near_pairs), int(cross.near_max_order),
            samples(cross.k, rho_max, float(cross._far_gap)) if not streaming else None,
            cross.need_p))
    parts = bor_operator_storage_bytes(modes, surfaces, crosses, constraint_size, streaming, compressed)
    retained = parts['tables'] + parts['basis'] + parts['near'] + parts['projection']
    build_workspace = max(parts['fft_workspace'], parts['near_workspace'])
    return (BOR_RETAINED_STORAGE_FACTOR * retained + build_workspace) / 1.0e9


def _plan_multisurface_assembly(
    m_max: 'int',
    solver_requirements,
    cross_operators,
    constraint_dofs: 'int',
    assembly: 'str',
    table_precision: 'str',
    stream_budget_gb: 'float',
    workers: 'int',
    extra_retained_gb: 'float' = 0.0,
) -> 'Dict[str, Any]':
    """Plan table or generalized streaming assembly for a junction system.

    The streaming budget applies to retained far blocks.  Always-resident
    same-surface contractions, cross-surface near/junction caches, cached
    projection matrices, and caller-owned dense maps are added explicitly to
    the runtime peak before the memory gate is evaluated.
    """

    from ghost_backend.bor.streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        combined_stream_mode_gb,
        estimate_rectangular_streaming_gb,
        plan_combined_streaming_mode_block,
        plan_stream_spill,
    )

    requirements = tuple(solver_requirements)
    # Derived reverse operators have no tables, streamed blocks or near cache.
    crosses = tuple(cross for cross in dict.fromkeys(cross_operators)
                    if not getattr(cross, 'derived', False))
    asm = str(assembly).strip().lower()
    precision = str(table_precision).strip().lower()
    budget = float(stream_budget_gb)
    extra = float(extra_retained_gb)
    if asm not in {"auto", "tables", "streaming"}:
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    if precision not in {"auto", "single", "double"}:
        raise ValueError(
            "table_precision must be 'auto', 'single', or 'double'."
        )
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")
    if not math.isfinite(extra) or extra < 0.0:
        raise ValueError("Extra retained operator storage must be non-negative.")

    merged = {}
    for solver, efie, mfie, ibc in requirements:
        key = id(solver)
        previous = merged.get(key)
        if previous is None:
            merged[key] = [solver, bool(efie), bool(mfie), bool(ibc)]
        else:
            previous[1] = previous[1] or bool(efie)
            previous[2] = previous[2] or bool(mfie)
            previous[3] = previous[3] or bool(ibc)
    merged_requirements = tuple(tuple(value) for value in merged.values())

    table_double_peak = estimate_bor_operator_storage_gb(
        m_max,
        merged_requirements,
        crosses,
        constraint_dofs=constraint_dofs,
        streaming=False,
    ) + 1.10 * extra
    use_streaming = (
        asm == "streaming" or (asm == "auto" and table_double_peak > 2.0)
    )
    worker_count = max(1, int(workers))

    if use_streaming:
        stream_specs_double = tuple(
            (
                int(solver.gen.n_elems),
                int(solver.gen.n_elems),
                bool(mfie or ibc),
                False,
            )
            for solver, efie, mfie, ibc in merged_requirements
            if efie or mfie or ibc
        ) + tuple(
            (
                int(cross.sp.gen.n_elems),
                int(cross.sq.gen.n_elems),
                cross.need_p,
                False,
            )
            for cross in crosses
        )
        full_far_double = sum(
            estimate_rectangular_streaming_gb(
                nt, ns, int(m_max), rotated, False
            )
            for nt, ns, rotated, _single in stream_specs_double
        )
        use_single = precision == "single"
        stream_specs = tuple(
            (nt, ns, rotated, use_single)
            for nt, ns, rotated, _single in stream_specs_double
        )
        mode_block, held_far_gb, effective_workers = (
            plan_combined_streaming_mode_block(
                int(m_max), stream_specs, budget, worker_count
            )
        )
        # As solve_bor: every stream is built once into memory-mapped files
        # when the budget cannot hold every mode.
        from ghost_backend.bor.compressed_far import far_compression_selected
        spill, mode_block, resident_gb = plan_stream_spill(
            mode_block, int(m_max) + 1, combined_stream_mode_gb(m_max, stream_specs),
            allow_spill=not any(far_compression_selected(item[0].Nn) for item in merged_requirements))
        if spill is not None:
            held_far_gb = resident_gb
            effective_workers = worker_count
        auxiliary_peak_gb = estimate_bor_operator_storage_gb(
            m_max,
            merged_requirements,
            crosses,
            constraint_dofs=constraint_dofs,
            streaming=True,
        ) + 1.10 * extra
        full_far_gb = full_far_double / (2.0 if use_single else 1.0)
        return {
            "use_streaming": True,
            "use_single": use_single,
            "mode_block": mode_block,
            "workers": effective_workers,
            "tile_budget_gb": BOR_STREAM_TILE_BUDGET_GB,
            "spill": spill,
            "persistent_gb": full_far_gb + auxiliary_peak_gb,
            "held_far_gb": held_far_gb,
            "auxiliary_peak_gb": auxiliary_peak_gb,
            "assembly_peak_gb": (
                held_far_gb + auxiliary_peak_gb + BOR_STREAM_TILE_BUDGET_GB
            ),
        }

    use_single = precision == "single"
    for solver, _efie, _mfie, _ibc in merged_requirements:
        solver._table_dtype = np.complex64 if use_single else np.complex128
    table_peak_gb = estimate_bor_operator_storage_gb(
        m_max,
        merged_requirements,
        crosses,
        constraint_dofs=constraint_dofs,
        streaming=False,
    ) + 1.10 * extra
    return {
        "use_streaming": False,
        "use_single": use_single,
        "mode_block": None,
        "workers": worker_count,
        "tile_budget_gb": None,
        "spill": None,
        "persistent_gb": table_peak_gb,
        "held_far_gb": table_peak_gb,
        "auxiliary_peak_gb": 0.0,
        "assembly_peak_gb": table_peak_gb,
    }


# A generatrix whose reflection about the plane through the middle of its
# end points matches its nodes to this fraction of the body size is mirror
# symmetric (the mode factors then split into even and odd halves).
MIRROR_NODE_TOLERANCE = 1e-9


def mirror_split_pays(n_dofs: 'int', n_rhs: 'int') -> 'bool':
    """Whether mirror halves beat LU for a mode of ``n_dofs`` unknowns and ``n_rhs`` columns.

    The halves save three quarters of the LU (about 2 n^3 flops) and their
    1e-8 coupling costs one refinement step against the exact matrix (a
    product and a half solve, about 8 n^2 flops per right-hand side), so they
    pay while ``n > 4 n_rhs`` (the 10 GHz ogive survey: 4,318 unknowns and
    362 columns, 54.0 s against 56.5 s).
    """
    return int(n_dofs) > 4 * int(n_rhs)


def mirror_map(solver, element_values=()) -> 'Optional[Callable]':
    """``mirror(m) -> (target, sign)`` for a surface symmetric about a plane normal
    to the axis, or None.

    The reflection z -> 2 z0 - z must map the nodes onto themselves with the
    traversal reversed (node ``i`` to node ``Nn - 1 - i``), and every
    per-element property in ``element_values`` (impedances, sheet values;
    None entries are ignored) onto itself.  A reduced unknown maps to the same
    component of the mirrored node, ``J_t`` with sign -1 (the tangent reverses)
    and ``J_phi`` with +1; at an axis pole of ``|m| = 1`` the reduced ``t``
    unknown carries its tied ``phi`` component along, with the same sign.
    """
    nodes = np.asarray(solver.gen.nodes, float)
    Nn = int(solver.Nn)
    size = float(max(np.ptp(nodes[:, 1]), np.max(np.abs(nodes[:, 0])), 1e-300))
    z0 = 0.5 * (nodes[0, 1] + nodes[-1, 1])
    reflected = np.column_stack([nodes[::-1, 0], 2.0 * z0 - nodes[::-1, 1]])
    if Nn < 4 or float(np.max(np.abs(reflected - nodes))) > MIRROR_NODE_TOLERANCE * size:
        return None
    if bool(solver.gen.node_on_axis(0)) != bool(solver.gen.node_on_axis(Nn - 1)):
        return None
    for values in element_values:
        if values is None:
            continue
        values = np.asarray(values)
        if len(values) != solver.gen.n_elems or not np.array_equal(values, values[::-1]):
            return None

    def mirror(m):
        active = np.flatnonzero(solver.basis_mask(m))
        lookup = np.full(2 * Nn, -1, dtype=int)
        lookup[active] = np.arange(active.size)
        node, component = active % Nn, active // Nn
        target = lookup[component * Nn + (Nn - 1 - node)]
        if np.any(target < 0):
            return None
        return target, np.where(component == 0, -1.0, 1.0)
    return mirror


def _validated_bor_aspects(thetas_deg) -> 'np.ndarray':
    """Validate the direct-solver monostatic aspect grid."""

    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    if thetas.ndim != 1 or thetas.size == 0:
        raise ValueError("BoR aspect grid must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(thetas)):
        raise ValueError("BoR aspect angles must all be finite.")
    if np.any(thetas < 0.0) or np.any(thetas > 180.0):
        raise ValueError("BoR aspect angles must lie in [0, 180] degrees.")
    return thetas


COMPRESSED_FAR_CACHE_MIN_NODES = 256
# Optional acceleration has a separate allowance from the solve's RAM limit.
# Large modal systems still need room for near blocks and active factors.
COMPRESSED_FAR_CACHE_GB = 512 * 1024**2 / 1.e9
COMPRESSED_FAR_WORK_GB = 256 * 1024**2 / 1.e9


def plan_compressed_far_cache(elements, cap, formulation, has_ibc, budget, workers,
                              rhs_count, auxiliary_gb, near_pairs=None):
    """Optional exact far retention must never exclude a query-only solve.

    Shrink the retained band against the same full memory gate as execution;
    when even one mode plus build workspace cannot fit, keep coefficient
    queries. The fast cache is an optimization, not a new minimum RAM demand.
    """
    if int(elements)+1 < COMPRESSED_FAR_CACHE_MIN_NODES:
        return None
    from ghost_backend.bor.streaming import (plan_streaming_mode_block,
        estimate_streaming_block_gb)
    budget = min(float(budget), COMPRESSED_FAR_CACHE_GB)
    while True:
        try:
            block, held, active = plan_streaming_mode_block(
                elements, cap, formulation, has_ibc, False, budget, workers)
        except BorAdmissionError:
            return None
        plan = plan_bor_mode_workers(2*(int(elements)+1), rhs_count, active, cap+1,
            auxiliary_gb+held+COMPRESSED_FAR_WORK_GB, near_pairs=near_pairs,
            preparation_workers=workers)
        if plan['fits_memory']:
            # Preserve the worker alignment used to price this exact band.
            # The modal gate may lower execution concurrency independently.
            return block, held, active
        if block == 1:
            return None
        budget = estimate_streaming_block_gb(elements, cap, max(1, block//2),
                                              formulation, has_ibc, False)


def plan_compressed_far_spill(elements, cap, formulation, has_ibc, mode_block):
    """Disk admission for exact far coefficients; resident windows stay bounded."""
    from ghost_backend.bor.streaming import (estimate_streaming_block_gb,
        spill_directory, stream_spill_candidate_gb)
    per_mode = estimate_streaming_block_gb(elements, cap, 1, formulation, has_ibc, False)
    candidate = stream_spill_candidate_gb(mode_block, cap+1, per_mode)
    return (spill_directory(candidate) if candidate else None), candidate


def _uses_far_compression(nodes):
    from ghost_backend.bor.compressed_far import far_compression_selected
    return far_compression_selected(nodes)


def _all_axial_aspects(thetas):
    """Only exact axial endpoints qualify; a near-axis look has other modes."""
    values = np.asarray(thetas, dtype=float)
    return bool(values.size and np.all((values == 0.0) | (values == 180.0)))


def _bor_mode_limits(k, rho_max: 'float', thetas,
                     n_modes: 'Optional[int]') -> 'Tuple[int, int]':
    """Return (mode_cap, physical_tail_start) for a BoR sweep.

    Plane-wave bandwidth scales as k*rho*sin(theta). The cap includes at least
    twelve modes and a growing cube-root transition margin for large bodies;
    the adaptive tail starts after the physical bandwidth, with a 0.05 axial-look floor.
    """

    theta_array = np.atleast_1d(np.asarray(thetas, dtype=float))
    sin_max = float(np.max(np.abs(np.sin(np.radians(theta_array)))))
    bandwidth = int(math.ceil(
        abs(complex(k)) * float(rho_max) * max(sin_max, 0.05)
    ))
    if n_modes is None:
        # A starting-cap heuristic, never a substitute for the measured tail.
        mode_cap = bandwidth + max(12, int(math.ceil(4.05*bandwidth**(1./3.)+2.)))
    else:
        try:
            mode_cap = int(n_modes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("n_modes must be a non-negative integer or None.") from exc
        if mode_cap != n_modes or mode_cap < 0:
            raise ValueError("n_modes must be a non-negative integer or None.")
    if _all_axial_aspects(theta_array):
        return min(mode_cap, 1), 1
    return mode_cap, bandwidth


def _block_diagonal_transforms(*blocks: 'np.ndarray') -> 'np.ndarray':
    """Dense block diagonal used by the small number of BoR field families."""

    if any(issparse(block) for block in blocks):
        return block_diag(blocks, format='csr')
    rows = sum(int(block.shape[0]) for block in blocks)
    columns = sum(int(block.shape[1]) for block in blocks)
    out = np.zeros((rows, columns), dtype=np.complex128)
    row = column = 0
    for block in blocks:
        nr, nc = block.shape
        out[row:row + nr, column:column + nc] = block
        row += nr
        column += nc
    return out


def _assemble_pmchwt_interface_into(target, exterior, interior, m, m_max,
                                    weight, eta_ratio2):
    """Fill a dense J/M interface block using at most one material temporary.

    Every sum and scaling keeps the former PMCHWT order (strided NumPy
    operations may differ in final-bit rounding). The exterior
    operators go straight into their final quadrants; the interior operator
    is consumed before constructing the next one. No quadrature is changed.
    """
    size = 2 * exterior.Nn
    if target.shape != (2 * size, 2 * size) or interior.Nn != exterior.Nn:
        raise ValueError("BoR PMCHWT interface dimensions do not match.")
    jj, jm = target[:size, :size], target[:size, size:]
    mj, mm = target[size:, :size], target[size:, size:]
    exterior.assemble_mode(m, m_max, out=jj)
    mm[:] = jj
    inner = interior.assemble_mode(m, m_max)
    _scaled_add_into(jj, inner, weight)
    _scaled_add_into(mm, inner, weight * eta_ratio2)
    inner = None
    exterior.assemble_pmchwt_P(m, m_max, out=mj)
    inner = interior.assemble_pmchwt_P(m, m_max)
    _scaled_add_into(mj, inner, weight)
    inner = None
    mj *= ETA0
    if weight != 1.0:
        _add_rotation_mass_into(mj, exterior, -(0.5 * (1.0 - weight) * ETA0))
    np.negative(mj, out=jm)
    return target


def _apply_regular_axis_rows(Q: 'np.ndarray', columns: 'np.ndarray',
                             offset: 'int', solver: 'BorPecSolver',
                             m: 'int') -> 'None':
    """Insert the exact |m|=1 pole relation into a composite transform."""

    if abs(int(m)) != 1:
        return
    for end, element in ((0, 0), (solver.Nn - 1, solver.gen.n_elems - 1)):
        if not solver.gen.node_on_axis(end):
            continue
        column = int(columns[offset + end])
        if column < 0:
            continue
        radial_sign = 1.0 if solver.gen.trho[element] >= 0.0 else -1.0
        Q[offset + solver.Nn + end, column] = (
            1j * int(m) * radial_sign
        )


@profiled_solve
@configured
def solve_bor(points, freq_hz: 'float', thetas_deg, formulation: 'str' = "auto",
              cfie_alpha: 'float' = 0.5, zs=None, n_modes: 'Optional[int]' = None,
              gauss_order: 'int' = FAR_GAUSS_ORDER, mode_tol: 'float' = 1e-6, workers: 'int' = 1,
              progress: 'Optional[Callable]' = None,
              check_abort: 'Optional[Callable]' = None,
              table_precision: 'str' = "auto", assembly: 'str' = "auto",
              stream_budget_gb: 'float' = 8.0, sheet_zs=None) -> 'Dict':
    """
    Monostatic RCS of a closed BoR at aspect angles thetas_deg (from +z).

    formulation: 'auto' selects CFIE for closed opaque bodies and EFIE for
    open shells or transmitting sheets; 'efie' is an explicit diagnostic,
    'cfie' (closed PEC --
    interior-resonance free), 'mfie' (diagnostics only).
    zs: surface impedance -- None (PEC), a complex scalar, or a per-ELEMENT
    complex array (tapered IBC).  IBC uses the EFIE form E_tan = Z_s J;
    resistance alone does not protect EFIE from interior resonances. Z_s is implemented
    by EFIE, or by CFIE for a uniform impedance on a closed surface.
    sheet_zs: a transmitting electric sheet impedance, scalar or per element.
    It uses EFIE + Zs mass with no equivalent magnetic current. Zs=0 elements
    can join a sheet to a PEC surface in the same meridian; opaque IBC cannot
    be combined with sheet_zs. Disconnected meridians require a separate solver.
    Returns dict with sigma_vv, sigma_hh (m^2) per angle.
    """

    form = str(formulation).strip().lower()
    if form not in {"auto", "efie", "cfie", "mfie"}:
        raise ValueError(
            f"Unsupported BoR formulation '{formulation}'. "
            "Use 'auto', 'efie', 'cfie', or 'mfie'."
        )
    if form in {"cfie", "auto"}:
        cfie_alpha = float(cfie_alpha)
        if not np.isfinite(cfie_alpha) or not (0.0 < cfie_alpha < 1.0):
            raise ValueError("CFIE alpha must be finite and satisfy 0 < alpha < 1.")
    points = _validate_solve_bor_generatrix(points, 'efie' if form == 'auto' else form)
    if form == 'auto':
        form = ('cfie' if points[0,0] == points[-1,0] == 0 and sheet_zs is None else 'efie')
    solver = _prepared_surface(points, freq_hz, gauss_order=gauss_order)
    k = solver.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(solver.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )


    zs_pt = None
    zs_elem = None
    zs_nodes = None
    if zs is not None:
        zs_arr = np.asarray(zs, dtype=complex)
        if zs_arr.ndim == 0:
            zs_elem = np.full(solver.gen.n_elems, complex(zs_arr))
        else:
            if len(zs_arr) != solver.gen.n_elems:
                raise ValueError("zs array must have one entry per generatrix element.")
            zs_elem = zs_arr.astype(complex)
        zs_elem = _validate_bor_surface_impedance(
            zs_elem, "BoR surface impedance"
        )
        zs_pt = zs_elem[solver.g.elem]
        if np.all(np.abs(zs_pt) == 0.0):
            zs_pt = None
        elif form not in ("efie", "cfie"):
            raise ValueError(
                f"{form.upper()} with nonzero surface impedance is not "
                "implemented; use CFIE on a closed surface or EFIE."
            )
        elif not np.all(zs_elem == zs_elem[0]):
            # M = -Zs n x J enters the magnetic-field equation through the EFIE
            # operator, whose charge term needs a continuous source.  A jump of
            # Zs would be a magnetic line charge; the nodal average (the
            # convention of the partial-coating bare pieces) spreads it over the
            # two adjacent elements and reduces to the uniform law exactly.
            zs_nodes = np.empty(solver.Nn, dtype=complex)
            zs_nodes[0], zs_nodes[-1] = zs_elem[0], zs_elem[-1]
            zs_nodes[1:-1] = 0.5 * (zs_elem[:-1] + zs_elem[1:])

    sheet_mass = None
    sheet_weights = None
    sheet_values = None
    if sheet_zs is not None:
        if zs is not None or form != "efie":
            raise ValueError("Free sheets require EFIE and cannot be combined with an opaque IBC.")
        sheet_values = np.asarray(sheet_zs, complex)
        if sheet_values.ndim == 0:
            sheet_values = np.full(solver.gen.n_elems, sheet_values, complex)
        if sheet_values.shape != (solver.gen.n_elems,):
            raise ValueError("sheet_zs must have one value per meridian element.")
        sheet_values = _validate_bor_surface_impedance(sheet_values, "BoR transmitting sheet")
        sheet_weights = sheet_values[solver.g.elem]

    alpha = float(cfie_alpha) if form == "cfie" else (1.0 if form == "efie" else 0.0)
    n_dofs = 2 * solver.Nn


    from ghost_backend.bor.streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_streaming_block_gb,
        estimate_streaming_gb,
        plan_stream_spill,
        plan_streaming_mode_block,
        sampling_backend_name,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    est_double = estimate_bor_table_gb(solver.gen.n_elems, m_max, form,
                                       zs_pt is not None, gauss_order, False)


    kinds = conductor_operator_kinds(form, zs_pt is not None)
    from ghost_backend.bor.options import output_reserved_gb
    cache_plan = (plan_compressed_far_cache(solver.gen.n_elems, m_max, form, zs_pt is not None,
        stream_budget_gb, workers, 2*len(thetas),
        estimate_bor_operator_storage_gb(m_max, ((solver, *kinds),), streaming=True)
        + output_reserved_gb(), solver._near_pair_count) if solver._compressed else None)
    # Conductors stream by default: the dense-tables path contracts its [P, P,
    # M] tables per mode at 38x the cost of the streamed read and samples both
    # triangles of the symmetric Green's function, so streaming measured equal
    # or faster at every size (0.6 s at 40 elements, 5.0 against 6.1 s at 300)
    # with 7x less memory; the two paths agree to 3e-15.  'tables' remains an
    # explicit choice.
    use_streaming = (cache_plan is not None if solver._compressed else asm in ("streaming", "auto"))
    if use_streaming:
        est_full = estimate_streaming_gb(solver.gen.n_elems, m_max, form,
                                         zs_pt is not None, False)
    else:
        est_full = est_double
    use_single = tp == "single"
    if use_single and not use_streaming:
        solver._table_dtype = np.complex64
    est = est_full / (2.0 if use_single else 1.0)


    solve_workers = max(1, int(workers))
    mode_block = None
    est_held = est
    stream_spill = None
    stream_workspace_gb = COMPRESSED_FAR_WORK_GB if solver._compressed else BOR_STREAM_TILE_BUDGET_GB
    if use_streaming:
        mode_block, est_held, solve_workers = cache_plan if cache_plan is not None else plan_streaming_mode_block(
            solver.gen.n_elems,
            m_max,
            form,
            zs_pt is not None,
            use_single,
            stream_budget_gb,
            workers,
        )
        # A budget short of every mode used to cost one far build per mode
        # range.  When the temporary directory can hold all modes, the blocks
        # are built once into memory-mapped files; the resident cost is then
        # the modes being read, and the mode workers need no range alignment.
        if solver._compressed:
            stream_spill, _ = plan_compressed_far_spill(
                solver.gen.n_elems, m_max, form, zs_pt is not None, mode_block)
        else:
            stream_spill, mode_block, resident_gb = plan_stream_spill(
                mode_block, m_max + 1,
                estimate_streaming_block_gb(solver.gen.n_elems, m_max, 1, form,
                                            zs_pt is not None, use_single),
                allow_spill=not _uses_far_compression(solver.Nn))
            if stream_spill is not None:
                est_held = resident_gb
                solve_workers = max(1, int(workers))


    kinds = conductor_operator_kinds(form, zs_pt is not None)
    assembly_peak_gb = (
        est_held + stream_workspace_gb + estimate_bor_operator_storage_gb(
            m_max, ((solver, *kinds),), streaming=True)
        if use_streaming
        # Tables: the model every material solver already uses (retained tables and
        # auxiliaries with 10 %, plus the bounded FFT build workspace). The former
        # blanket 3.5 x tables asked 5.3 GB for a CFIE sphere that peaks at 1.34 GB.
        else estimate_bor_operator_storage_gb(m_max, ((solver, *kinds),), streaming=bool(solver._compressed))
    )
    table_note = (f"{'Streamed far blocks' if use_streaming else 'Far kernel tables'} "
                  f"stored in single precision ({est:.1f} GB; double would "
                  f"need {est_full:.1f} GB)."
                  if use_single and tp == "auto" else None)
    optional_far_cache_failure = None

    def prepare(mm):
        nonlocal sheet_mass, optional_far_cache_failure
        if sheet_weights is not None:
            # The sheet mass is tridiagonal: dense solves add its bands in
            # place (no N x N matrix is formed or cached); compressed
            # operators keep the symbolic mass expression.
            sheet_mass = (solver.mass_blocks(weight=sheet_weights) if solver._compressed
                          else solver.mass_bands(weight=sheet_weights))
        if use_streaming and solver._stream is None:
            from ghost_backend.bor.streaming import StreamingSpillError
            try:
                solver.enable_streaming(
                    mm, efie=kinds[0], mfie=kinds[1],
                    ibc_zs_pt=zs_pt if kinds[2] else None,
                    single_blocks=use_single, workers=solve_workers,
                    mode_block=mode_block, spill=stream_spill,
                    tile_budget_gb=stream_workspace_gb)
            except InterruptedError:
                # Cancellation may use this OSError subclass; never treat it
                # as a failed optional disk optimization.
                raise
            except (OSError, StreamingSpillError) as exc:
                if not solver._compressed or stream_spill is None:
                    raise
                # Optional disk caching failed before publication. Exact
                # coefficient queries remain available within the admitted RAM.
                solver.close_streaming()
                optional_far_cache_failure = str(exc)
        solver.prepare_operators(mm, efie=kinds[0], mfie=kinds[1], ibc=kinds[2],
                                 workers=workers)

    def assemble(m):
        if zs_pt is None and sheet_mass is None and not solver._compressed:
            # Conductor without impedance: form the CFIE combination in one
            # buffer.  The EFIE family is written straight into the quadrants,
            # the MFIE family is accumulated in chunks, and the |m| = 1 pole
            # relation is applied as O(n) row/column updates, so a mode never
            # holds more than this matrix and its reduced copy.
            Z = None
            if alpha > 0.0:
                Z = solver.assemble_mode(m, m_max, scale=alpha)
            if alpha < 1.0:
                Z = solver.assemble_mfie_mode(m, m_max, out=Z, scale=(1.0 - alpha) * ETA0,
                                              accumulate=Z is not None)
            if abs(int(m)) == 1:
                return solver.reduce_pole_operator(Z, m), solver.basis_transform(m)
            mask = solver.basis_mask(m)
            return Z[np.ix_(mask, mask)], mask
        if not solver._compressed:
            # Impedance surfaces and transmitting sheets, also in one buffer:
            # the EFIE family is written into the quadrants, mixed in row
            # chunks with its dual (the CFIE's EFIE-on-M term), the impedance
            # extra and the sheet mass are added in place, and the MFIE family
            # is accumulated last.  Peak: this matrix plus its reduced copy,
            # the same as the conductor path (the former sequence held five
            # full matrices for |m| = 1).
            Z = None
            N = solver.Nn
            if alpha > 0.0:
                Z = solver.assemble_mode(m, m_max)
                quads = (Z[:N, :N], Z[:N, N:], Z[N:, :N], Z[N:, N:])
                if zs_pt is not None and alpha < 1.0:
                    _mix_dual_ibc_in_place(
                        quads, alpha, (1.0 - alpha) / ETA0,
                        complex(zs_elem[0]) if zs_nodes is None else zs_nodes)
                elif alpha != 1.0:
                    Z *= alpha
                if sheet_mass is not None:
                    solver._add_bands_into(quads[0], sheet_mass, alpha)
                    solver._add_bands_into(quads[3], sheet_mass, alpha)
                if zs_pt is not None:
                    solver.assemble_ibc_extra(m, m_max, zs_pt, zs_elem, out=Z, scale=alpha)
            if alpha < 1.0:
                Z = solver.assemble_mfie_mode(m, m_max, out=Z, scale=(1.0 - alpha) * ETA0,
                                              accumulate=Z is not None)
            if abs(int(m)) == 1:
                return solver.reduce_pole_operator(Z, m), solver.basis_transform(m)
            mask = solver.basis_mask(m)
            return Z[np.ix_(mask, mask)], mask
        Z = None
        dual_ibc = None
        if alpha > 0.0:
            Z = solver.assemble_mode(m, m_max)
            if zs_pt is not None and alpha < 1.0:


                N = solver.Nn
                dual_ibc = modal_block([[Z[N:, N:], -Z[N:, :N]],
                                     [-Z[:N, N:], Z[:N, :N]]])
                if zs_nodes is None:
                    dual_ibc *= complex(zs_elem[0]) / ETA0**2
                else:
                    dual_ibc = (dual_ibc @ diags(np.tile(zs_nodes, 2)).tocsr()) * (1.0 / ETA0**2)
            if sheet_mass is not None:
                Z[:solver.Nn, :solver.Nn] += sheet_mass
                Z[solver.Nn:, solver.Nn:] += sheet_mass
            if alpha != 1.0:
                Z *= alpha
            if zs_pt is not None:
                extra = solver.assemble_ibc_extra(m, m_max, zs_pt, zs_elem)
                if alpha != 1.0:
                    extra *= alpha
                Z += extra
        if alpha < 1.0:
            mfie = solver.assemble_mfie_mode(m, m_max)
            if dual_ibc is not None:
                mfie += dual_ibc
            mfie *= (1.0 - alpha) * ETA0
            if Z is None:
                Z = mfie
            else:
                Z += mfie
        if abs(int(m)) == 1:
            Q = solver.basis_transform(m)
            return _reduce_constrained_operator(Z, Q), Q


        mask = solver.basis_mask(m)
        return Z[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(n_dofs, dtype=np.complex128)
        if alpha > 0.0:
            V += alpha * solver.rhs_mode(m, th, pol)
        if alpha < 1.0:
            V += (1.0 - alpha) * ETA0 * solver.rhs_mfie_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = solver.farfield_mode(m, full, th, zs_pt=zs_pt)
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return solver.rhs_vv_hh_batch(
            m,
            batch_thetas,
            efie_scale=alpha,
            mfie_scale=(1.0 - alpha) * ETA0,
        )

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return solver.farfield_vv_hh_batch(
            m, full, batch_thetas, zs_pt=zs_pt
        )


    solve_warnings: 'List[str]' = []
    if table_note:
        solve_warnings.append(table_note)
    closed = solver.gen.node_on_axis(0) and solver.gen.node_on_axis(solver.Nn - 1)
    lossless_ibc = (
        zs_pt is not None
        and _effectively_reactive_surface_impedance(zs_pt, solver.g.w * solver.g.rho)
    )
    if form == "efie" and closed and lossless_ibc:
        raise RuntimeError(
            "Closed-body lossless/reactive IBC on the EFIE is unsupported: "
            "undamped interior resonances cannot be ruled out reliably, and "
            "uniform closed IBC requires formulation='cfie'. Spatially varying "
            "IBC requires a separately validated resonance-stable formulation; "
            "adding resistance is not a resonance guarantee.")

    try:
        F, modes_used, stats = _mode_sweep(n_dofs, thetas, ("VV", "HH"), m_max,
                                           mode_tol, assemble, rhs, farfield,
                                           prepare=prepare, workers=solve_workers, preparation_workers=workers,
                                           progress=progress,
                                           check_abort=check_abort,
                                           monitor_cond=True,
                                           rhs_batch=rhs_batch,
                                           farfield_batch=farfield_batch,
                                           min_mode_before_tail=mode_tail_start,
                                           assembly_peak_gb=assembly_peak_gb,
                                           memory_context="The PEC/IBC BoR solve",
                                           stream_mode_block=mode_block,
                                           exact_far_cache=lambda: solver._stream is not None,
                                           signed_mode_symmetry=True, axial_mode_only=_all_axial_aspects(thetas),
                                           coordinates=lambda m: solver.gen.nodes[
                                               np.flatnonzero(solver.basis_mask(m)) % solver.Nn],
                                           mirror=(mirror_map(solver, (zs_elem, sheet_values))
                                                   if mirror_split_pays(n_dofs, 2 * len(thetas))
                                                   else None))
        stream = solver._stream
        stream_meta = {
            "stream_mode_block": stream.mode_block if stream is not None else None,
            "stream_sweeps": stream.n_sweeps if stream is not None else 0,
            "stream_sampling_backend": (
                sampling_backend_name(stream) if stream is not None else None
            ),
            "stream_spill_gb": stream.spilled_gb() if stream is not None else 0.0,
            "stream_cache_fallback": optional_far_cache_failure,
            "stream_far_compression": (dict(stream.evidence)
                                       if getattr(stream, "evidence", None) else None),
        }
    finally:
        solver.close_streaming()
    _require_mode_convergence(stats, mode_tol)
    near_summary = _near_quadrature_summary(solver)
    _warn_near_asymmetry(near_summary, solve_warnings)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_dofs),
        "formulation": form,
        "boundary_model": "transmitting_electric_sheet" if sheet_zs is not None else "opaque_ibc" if zs_pt is not None else "pec",
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "near_quadrature": near_summary,
        **stream_meta,
        "warnings": solve_warnings,
        **stats,
    }


@profiled_solve
@configured
def solve_bor_dielectric(points, freq_hz: 'float', thetas_deg, eps_r: 'complex',
                         mu_r: 'complex' = 1.0, n_modes: 'Optional[int]' = None,
                         gauss_order: 'int' = FAR_GAUSS_ORDER, mode_tol: 'float' = 1e-6,
                         workers: 'int' = 1, progress: 'Optional[Callable]' = None,
                         check_abort: 'Optional[Callable]' = None,
                         table_precision: 'str' = "auto",
                         assembly: 'str' = "auto",
                         stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """
    Monostatic RCS of a closed homogeneous penetrable BoR via per-mode PMCHWT.

    Unknowns per mode: J (2Nn) and M' = M/eta0 (2Nn).  The exterior
    representation's interior null-field limit is combined with c times the
    interior representation's exterior limit (T = EFIE operator in its
    medium, P = rotated-PV operator, R = <W, n x .>):

        [ T_e + c T_i                     -eta0 (P_e + c P_i) + j_c ] [J ]   [  <W, E_inc>     ]
        [ eta0 (P_e + c P_i) - j_c   T_e + c (eta0^2/eta_i^2) T_i   ] [M'] = [ eta0 <W, H_inc> ]

    with j_c = (1 - c)/2 * eta0 * R, the identity/jump terms that only equal
    weights cancel.  c = 1 is PMCHWT; c = exp(-/+ j*15 deg) (see
    ``_region_equation_weights``) removes its spurious real-frequency
    singularities.  The eta0 scalings symmetrize the block magnitudes.  The
    exterior far field radiates BOTH J and M in air.
    """

    points = _validate_solve_bor_generatrix(points, "cfie")

    _causal_medium(eps_r, mu_r)
    se = _prepared_surface(points, freq_hz, gauss_order=gauss_order)
    si = _prepared_surface(points, freq_hz, gauss_order=gauss_order,
                      medium=(eps_r, mu_r))
    k = se.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    Nn = se.Nn
    eta_ratio2 = (ETA0 / si.eta) ** 2
    weight = _region_equation_weights([None, (eps_r, mu_r)])[1]


    from ghost_backend.bor.streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        estimate_streaming_block_gb,
        estimate_streaming_gb,
        plan_stream_spill,
        plan_streaming_mode_block,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    table_double = 2.0 * estimate_bor_table_gb(
        se.gen.n_elems, m_max, "efie", True, gauss_order, False
    )
    use_streaming = (
        asm == "streaming" or (asm == "auto" and table_double > 2.0)
    )
    full_double = (
        2.0 * estimate_streaming_gb(
            se.gen.n_elems, m_max, "efie", True, False
        )
        if use_streaming else table_double
    )
    use_single = tp == "single"
    solve_workers = max(1, int(workers))
    mode_block = None
    stream_spill = None
    if use_streaming:


        mode_block, held_one, solve_workers = plan_streaming_mode_block(
            se.gen.n_elems,
            m_max,
            "efie",
            True,
            use_single,
            0.5 * stream_budget,
            solve_workers,
        )
        held_blocks_gb = 2.0 * held_one
        # As solve_bor: when the budget cannot hold every mode, both streams
        # are built once into memory-mapped files instead of once per mode
        # range, and the mode workers need no range alignment.
        stream_spill, mode_block, resident_gb = plan_stream_spill(
            mode_block, m_max + 1,
            2.0 * estimate_streaming_block_gb(se.gen.n_elems, m_max, 1, "efie", True, use_single),
            allow_spill=not _uses_far_compression(se.Nn))
        if stream_spill is not None:
            held_blocks_gb = resident_gb
            solve_workers = max(1, int(workers))
        operator_storage_gb = (
            held_blocks_gb + BOR_STREAM_TILE_BUDGET_GB
            + estimate_bor_operator_storage_gb(m_max,
                ((se, True, False, True), (si, True, False, True)), streaming=True)
        )
    else:
        if use_single:
            se._table_dtype = np.complex64
            si._table_dtype = np.complex64
        operator_storage_gb = estimate_bor_operator_storage_gb(
            m_max,
            (
                (se, True, False, True),
                (si, True, False, True),
            ),
        )

    solve_warnings: 'List[str]' = []
    if use_single and tp == "auto":
        solve_warnings.append(
            f"{'Streamed far blocks' if use_streaming else 'Far kernel tables'} "
            f"stored in single precision ({full_double / 2.0:.1f} GB; "
            f"double would need {full_double:.1f} GB)."
        )

    def prepare(mm):
        if use_streaming:
            for solver in (se, si):
                solver.enable_streaming(
                    mm,
                    efie=True,
                    pmchwt=True,
                    single_blocks=use_single,
                    tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                    workers=solve_workers,
                    mode_block=mode_block,
                    spill=stream_spill,
                )
        se.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        si.prepare_operators(mm, efie=True, ibc=True, workers=workers)

    def assemble(m):
        if use_streaming and not se._compressed:
            A = modal_matrix((4 * Nn, 4 * Nn), False)
            _assemble_pmchwt_interface_into(A, se, si, m, m_max, weight, eta_ratio2)
        else:
            # Table contractions can have more scratch than the final block.
            # Construct their operators before allocating A so those peaks
            # do not overlap. Streamed operators above need no contraction.
            T_e = se.assemble_mode(m, m_max)
            T_i = si.assemble_mode(m, m_max)
            P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + weight * si.assemble_pmchwt_P(m, m_max))
            if weight != 1.0:
                P_sum = _add_rotation_mass_into(P_sum, se, -(0.5 * (1.0 - weight) * ETA0))
            A = modal_matrix((4 * Nn, 4 * Nn), compressed_requested())
            A[: 2 * Nn, : 2 * Nn] = T_e + weight * T_i
            A[: 2 * Nn, 2 * Nn:] = -P_sum
            A[2 * Nn:, : 2 * Nn] = P_sum
            A[2 * Nn:, 2 * Nn:] = T_e + (weight * eta_ratio2) * T_i
            del T_e, T_i, P_sum
        if abs(int(m)) == 1:
            q_surface = se.basis_transform(m)
            Q = _block_diagonal_transforms(q_surface, q_surface)
            return _reduce_constrained_operator(A, Q), Q
        mask = np.tile(se.basis_mask(m), 2)
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        return np.concatenate([se.rhs_mode(m, th, pol),
                               ETA0 * se.rhs_h_mode(m, th, pol)])

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[: 2 * Nn], th,
                                    msol=ETA0 * full[2 * Nn:])
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        electric = se.rhs_vv_hh_batch(m, batch_thetas)
        out = np.empty((4 * Nn, electric.shape[1]), dtype=np.complex128)
        out[:2 * Nn] = electric
        out[2 * Nn:, 0::2] = -electric[:, 1::2]
        out[2 * Nn:, 1::2] = electric[:, 0::2]
        return out

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return se.farfield_vv_hh_batch(
            m,
            full[:2 * Nn],
            batch_thetas,
            msolutions=ETA0 * full[2 * Nn:],
        )

    stream_backends = None
    stream_backend = None
    stream_sweeps = 0
    stream_spill_gb = 0.0
    stream_compression = {}
    try:
        F, modes_used, stats = _mode_sweep(4 * Nn, thetas, ("VV", "HH"), m_max,
                                           mode_tol, assemble, rhs, farfield,
                                           prepare=prepare, workers=solve_workers, preparation_workers=workers,
                                           progress=progress,
                                           check_abort=check_abort,
                                           monitor_cond=True,
                                           rhs_batch=rhs_batch,
                                           farfield_batch=farfield_batch,
                                           min_mode_before_tail=mode_tail_start,
                                           assembly_peak_gb=operator_storage_gb,
                                           memory_context="The dielectric BoR solve",
                                           stream_mode_block=mode_block,
                                           signed_mode_symmetry=True, axial_mode_only=_all_axial_aspects(thetas),
                                           coordinates=lambda m: np.tile(se.gen.nodes, (4, 1))[
                                               np.tile(se.basis_mask(m), 2)],
                                           hierarchical_pricing=False)
        if use_streaming:
            stream_compression = _stream_compression_evidence({'exterior':se._stream,'interior':si._stream})
            stream_backends = {
                "exterior": sampling_backend_name(se._stream),
                "interior": sampling_backend_name(si._stream),
            }
            unique_backends = set(stream_backends.values())
            stream_backend = (
                next(iter(unique_backends)) if len(unique_backends) == 1 else "mixed"
            )
            stream_sweeps = se._stream.n_sweeps + si._stream.n_sweeps
            stream_spill_gb = se._stream.spilled_gb() + si._stream.spilled_gb()
    finally:
        # Streamed far blocks can hold GBs (or spill files); release them on
        # success, abort, and error alike rather than at cyclic GC.
        for owner in (se, si):
            owner.close_streaming()
    _require_mode_convergence(stats, mode_tol)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(4 * Nn),
        "formulation": "pmchwt",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": mode_block if use_streaming else None,
        "stream_sweeps": stream_sweeps,
        "stream_spill_gb": stream_spill_gb,
        "stream_far_compression": stream_compression,
        "stream_sampling_backend": stream_backend,
        "stream_sampling_backends": stream_backends,
        "warnings": solve_warnings,
        **stats,
    }


def _segment_distance(p0, p1, q0, q1) -> 'float':
    """Min distance between two non-intersecting 2D segments (attained at an
    endpoint of one of them)."""

    def pt_seg(c, a, b):
        ab = b - a
        t = float(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-300))
        t = min(1.0, max(0.0, t))
        return float(np.hypot(*(c - (a + t * ab))))

    return min(pt_seg(p0, q0, q1), pt_seg(p1, q0, q1),
               pt_seg(q0, p0, p1), pt_seg(q1, p0, p1))


def _contract_near_points(gp, e, gq, f, k, m_max, kinds, points, signed=True, mode_start=0):
    """Integrate immediately into 2x2 blocks; never retain raw pair kernels.

    Point tiles and modal projection tiles bound scratch even for a single
    large quadrature record. Output shape per kind is [4, 2m+1, 2, 2]
    (modes -m_max..m_max), or [4, m_max+1, 2, 2] (modes 0..m_max) with
    ``signed=False``: production storage keeps only the nonnegative modes
    (negative ones follow by exact tangential-block parity), so it never
    contracts the signed half.
    EFIE omits j*k*eta*2pi; bracket blocks include 2pi.
    """
    return _contract_near_batch([(gp, e, gq, f, points)], k, m_max, kinds, signed, mode_start)[0]


def _near_point_chunks(gp, e, gq, f, points, nm):
    """The point chunks of one element pair's quadrature record (active points only)."""
    s, t, weight = points
    active = weight > 0
    s, t, weight = s[active], t[active], weight[active]
    chunk = max(1, min(256, int(8.0e6 / (16 * nm * 20))))
    chunks = []
    for i in range(0, len(s), chunk):
        sl = slice(i, i + chunk)
        rp, zp, trp, tzp, p0, p1, dp0, dp1, lp = _points_on_element(gp, e, s[sl])
        rq, zq, trq, tzq, q0, q1, dq0, dq1, lq = _points_on_element(gq, f, t[sl])
        chunks.append((rp, zp, trp, tzp, rq, zq, trq, tzq, np.array([p0, p1]), np.array([q0, q1]),
                       np.array([dp0, dp1]), np.array([dq0, dq1]), weight[sl] * lp * lq))
    return chunks


# Near kernels of many element pairs are evaluated together: the graded rule
# treats every point on its own (its layout, orders and convergence check
# depend only on the point), so one call over the points of a whole batch of
# pairs returns the values of separate calls, bitwise, with far fewer and
# larger rule groups (the per-pair calls spent about 40 % of preparation in
# Python between native calls).  Kernel values of at most this many bytes are
# held before their chunks are contracted, in each pair's own chunk order.
NEAR_BATCH_KERNEL_BYTES = 64_000_000


def _near_chunk_kernels(chunks, k, m_max, kinds, signed=True, mode_start=0):
    """Kernel values per chunk and kind: one graded-rule call per kind for all chunks."""
    sizes = [len(chunk[12]) for chunk in chunks]
    edges = np.cumsum([0] + sizes)
    columns = {name: np.concatenate([np.broadcast_to(chunk[index], (size,))
                                     for chunk, size in zip(chunks, sizes)])
               for index, name in enumerate(('rp', 'zp', 'trp', 'tzp', 'rq', 'zq', 'trq', 'tzq'))}
    values = []
    for kind in kinds:
        if kind == 'efie':
            G = modal_kernels_near(columns['rp'], columns['zp'], columns['rq'], columns['zq'], k, m_max,
                                  mode_start=max(0, mode_start - 1))
            values.append([G[a:b] for a, b in zip(edges[:-1], edges[1:])])
            continue
        kernel = mfie_kernels_near if kind == 'mfie' else ibc_kernels_near
        brackets = kernel(columns['rp'], columns['zp'], columns['trp'], columns['tzp'],
                          columns['rq'], columns['zq'], columns['trq'], columns['tzq'], k, m_max,
                          signed=signed, mode_start=mode_start)
        values.append([tuple(value[a:b] for value in brackets) for a, b in zip(edges[:-1], edges[1:])])
    return [dict(zip(kinds, per_chunk)) for per_chunk in zip(*values)]


def _contract_near_chunk(out, chunk, kernels, k, m_max, modes, signed, mode_start=0):
    """Add one chunk's contributions: ``[m, i, j] = sum_p left[i,p] K[p,m] right[j,p]``.

    Each contraction is one product of the four basis-pair rows with the
    kernel (the former ``einsum`` took 65 us per 256-point chunk against 8 us;
    the sums differ by rounding only).
    """
    rp, zp, trp, tzp, rq, zq, trq, tzq, Tp, Tq, Dp, Dq, w = chunk
    points = len(w)

    def rows(left, right):
        return (left[:, None, :] * right[None, :, :]).reshape(4, points)

    def contract(pairs, kernel):
        return (pairs @ kernel).reshape(2, 2, -1).transpose(2, 0, 1)

    TT = rows(Tp, Tq)
    rr = (rp * rq * w)[:, None]
    if 'efie' in out:
        G = kernels['efie']
        am = np.abs(modes)
        offset = max(0, mode_start - 1)
        Gn = G[:, am - offset]
        Gc = 0.5 * (G[:, np.abs(am - 1) - offset] + G[:, am + 1 - offset])
        Gs = (G[:, np.abs(am - 1) - offset] - G[:, am + 1 - offset]) / 2j
        Gs[:, modes < 0] *= -1
        scalar = w[:, None] * Gn / k**2
        out['efie'][0] += contract(TT, rr * (trp * trq * Gc + tzp * tzq * Gn)) - contract(rows(Dp, Dq), scalar)
        out['efie'][1] += contract(TT, rr * trp * Gs) - (1j * modes[:, None, None]) * contract(rows(Dp, Tq), scalar)
        out['efie'][2] += contract(TT, -rr * trq * Gs) + (1j * modes[:, None, None]) * contract(rows(Tp, Dq), scalar)
        out['efie'][3] += contract(TT, rr * Gc) - (modes[:, None, None]**2) * contract(TT, scalar)
    for kind in out:
        if kind == 'efie':
            continue
        for uv, value in enumerate(kernels[kind]):
            out[kind][uv] += contract(TT, 2 * np.pi * rr * value)


def _contract_near_batch(jobs, k, m_max, kinds, signed=True, mode_start=0):
    """:func:`_contract_near_points` of every ``(gp, e, gq, f, points)`` job.

    Chunks are formed per job exactly as one call would form them; their
    kernels are evaluated together in blocks of at most
    NEAR_BATCH_KERNEL_BYTES and each job's chunks are contracted in order, so
    every job's blocks equal those of its own call, bitwise.
    """
    if mode_start and signed:
        raise ValueError('Incremental near blocks retain nonnegative modes only.')
    modes = np.arange(-m_max, m_max + 1) if signed else np.arange(mode_start, m_max + 1)
    nm = len(modes)
    outs = [{kind: np.zeros((4, nm, 2, 2), complex) for kind in kinds} for _ in jobs]
    per_point = 16 * sum((m_max + 2 - max(0, mode_start - 1)) if kind == 'efie' else 4 * nm for kind in kinds) + 160
    limit = max(256, NEAR_BATCH_KERNEL_BYTES // per_point)
    block, points = [], 0

    def flush():
        if not block:
            return
        kernels = _near_chunk_kernels([chunk for _, chunk in block], k, m_max, kinds,
                                      signed=signed, mode_start=mode_start)
        for (index, chunk), values in zip(block, kernels):
            _contract_near_chunk(outs[index], chunk, values, k, m_max, modes, signed, mode_start)
        block.clear()

    for index, (gp, e, gq, f, pts) in enumerate(jobs):
        for chunk in _near_point_chunks(gp, e, gq, f, pts, nm):
            if block and points + len(chunk[12]) > limit:
                flush()
                points = 0
            block.append((index, chunk))
            points += len(chunk[12])
    flush()
    return outs


def _gap_graded_points(gp, e, gq, f, order):
    """Sinh grading around the closest source point for each test point."""
    x, wg = cached_leggauss(order)
    u, w = (x + 1) / 2, wg / 2
    p0, p1 = gp.nodes[e:e + 2]
    q0, q1 = gq.nodes[f:f + 2]
    dp, dq = p1 - p0, q1 - q0


    cuts = np.unique(np.r_[0., 1., np.clip(
        [(q0 - p0) @ dp / (dp @ dp), (q1 - p0) @ dp / (dp @ dp)], 0, 1)])


    gap = _segment_distance(p0, p1, q0, q1)
    outer_delta = max(gap / np.linalg.norm(dp), 1e-15)
    ss, ww = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vmax = np.arcsinh((b - a) / (2 * outer_delta))
        v = vmax * u
        distance = outer_delta * np.sinh(v)
        weights = w * vmax * outer_delta * np.cosh(v)
        ss.extend((a + distance, b - distance))
        ww.extend((weights, weights))
    s, ws = np.concatenate(ss), np.concatenate(ww)
    p = p0 + s[:, None] * dp
    center = np.clip((p - q0) @ dq / (dq @ dq), 0, 1)
    distance = np.linalg.norm(p - (q0 + center[:, None] * dq), axis=1)
    delta = np.maximum(distance / np.linalg.norm(dq), 1e-15)
    ts, weights = [], []
    for side, extent in ((-1, center), (1, 1 - center)):
        vmax = np.arcsinh(extent / delta)
        v = vmax[:, None] * u
        ts.append(center[:, None] + side * delta[:, None] * np.sinh(v))
        weights.append(ws[:, None] * w * vmax[:, None] * delta[:, None] * np.cosh(v))
    t = np.concatenate(ts, axis=1)
    return np.broadcast_to(s[:, None], t.shape).ravel(), t.ravel(), np.concatenate(weights, axis=1).ravel()


def _converged_disjoint_blocks(gp, e, gq, f, k, m_max, kinds,
                               order=12, rtol=2e-5, max_order=192, signed=True, mode_start=0):
    """Converge actual EFIE/PV blocks independently of angular integration.

    ``signed=False`` returns (and checks) only the modes 0..m_max; the
    negative modes are exact parity copies, so the convergence test is the
    same."""
    return _converged_disjoint_batch(gp, gq, [(e, f)], k, m_max, kinds,
                                     order, rtol, max_order, signed, mode_start)[0]


# The coarse meridian level (order 6) of the disjoint near pairs is evaluated
# on every NEAR_MERIDIAN_CHECK_STRIDE-th pair of a batch; the batch is accepted
# at the fine level (order 12, the published value) when every probed pair
# passes, otherwise every pair is refined as before.  Measured: every pair of
# the reference bodies passes the first level with a change of 7e-8 against
# the 2e-5 tolerance.  GHOST_BOR_NEAR_CHECK_STRIDE=0 restores the complete
# check (October 2026).
NEAR_MERIDIAN_CHECK_STRIDE = 4


def _converged_disjoint_batch(gp, gq, pairs, k, m_max, kinds,
                              order=12, rtol=2e-5, max_order=192, signed=True, mode_start=0):
    """:func:`_converged_disjoint_blocks` of every element pair ``(e, f)``.

    Every pair follows its own sequence of meridian orders and its own
    convergence test; the pairs still refining at one level are integrated
    together (:func:`_contract_near_batch`), so each result equals the
    single-pair one, bitwise.  Returns ``(blocks, order, error)`` per pair.
    With NEAR_MERIDIAN_CHECK_STRIDE the fine level is evaluated first and the
    coarse level only for the probed pairs (see the constant).
    """
    requested_order = int(order)
    if requested_order < 2 or not 2 * requested_order <= max_order <= 384 or not 0 < rtol < 1:
        raise ValueError('BoR near quadrature needs order >= 2, 2*order <= max_order <= 384, and 0 < rtol < 1.')
    results = [None] * len(pairs)
    states = []
    for index, (e, f) in enumerate(pairs):
        zero_mfie = False
        if 'mfie' in kinds:
            nodes = np.vstack((gp.nodes[e:e+2], gq.nodes[f:f+2]))


            z_tolerance = 8*np.finfo(float).eps*max(float(np.max(np.abs(nodes))), np.finfo(float).tiny)
            zero_mfie = float(np.ptp(nodes[:, 1])) <= z_tolerance
            if zero_mfie and tuple(kinds) == ('mfie',):
                width = 2*m_max+1 if signed else m_max+1-mode_start
                results[index] = ({'mfie': np.zeros((4, width, 2, 2), complex)}, int(order), 0.)
                continue
        gap = _segment_distance(gp.nodes[e], gp.nodes[e + 1], gq.nodes[f], gq.nodes[f + 1])
        graded = gap < 0.25 * max(gp.lengths[e], gq.lengths[f])
        states.append(dict(index=index, e=e, f=f, gap=gap, graded=graded, zero_mfie=zero_mfie,
                           n=requested_order if graded else max(2, requested_order // 2)))

    def evaluate(group):
        output = [None]*len(group)
        for zero in (False, True):
            selected = [(i, st) for i, st in enumerate(group) if st['zero_mfie'] == zero]
            if not selected:
                continue
            wanted = tuple(kind for kind in kinds if not (zero and kind == 'mfie'))
            blocks = _contract_near_batch(
                [(gp, st['e'], gq, st['f'],
                  _gap_graded_points(gp, st['e'], gq, st['f'], st['n']) if st['graded']
                  else _regular_cell_points(st['n'])) for _, st in selected],
                k, m_max, wanted, signed, mode_start)
            for (i, _), value in zip(selected, blocks):
                if zero:
                    # Preserve the independent-family analytic zero when
                    # families share traversal. Relative refinement of its
                    # floating-point cancellation residue cannot converge.
                    width = 2*m_max+1 if signed else m_max+1-mode_start
                    value['mfie'] = np.zeros((4,width,2,2),complex)
                output[i] = value
        return output

    def block_error(fine, coarse):
        errors = []
        for kind in kinds:
            scale = np.max(np.abs(fine[kind]), axis=(1, 2, 3))
            floor = max(float(np.max(scale)) * 1e-8, 1e-280)
            error = np.max(np.abs(fine[kind] - coarse[kind]), axis=(1, 2, 3))
            errors.append(float(np.max(error / np.maximum(scale, floor))))
        return max(errors)

    from ghost_backend.bor.kernels import _near_check_stride
    stride = _near_check_stride() if NEAR_MERIDIAN_CHECK_STRIDE else 0
    if stride > 0 and len(states) >= 2 * stride:
        # Fine level for every pair; coarse level for the probed pairs only.
        for st in states:
            st['n_coarse'] = st['n']
            st['n'] = max(requested_order, min(2 * st['n'], max_order))
        fine_all = evaluate(states)
        probe = list(range(0, len(states), stride))
        for i in probe:
            states[i]['n'], states[i]['n_fine'] = states[i]['n_coarse'], states[i]['n']
        coarse_probe = evaluate([states[i] for i in probe])
        for i in probe:
            states[i]['n'] = states[i]['n_fine']
        worst = 0.
        for i, coarse in zip(probe, coarse_probe):
            error = block_error(fine_all[i], coarse)
            worst = max(worst, error)
            if not (math.isfinite(error) and error <= rtol):
                worst = math.inf
        if math.isfinite(worst):
            for st, fine in zip(states, fine_all):
                results[st['index']] = (fine, st['n'], worst)
            return results
        for st in states:
            st['n'] = st['n_coarse']
    for st, blocks in zip(states, evaluate(states)):
        st['coarse'] = blocks
    active = states
    while active:
        for st in active:
            st['n'] = max(requested_order, min(2 * st['n'], max_order))
        still = []
        for st, fine in zip(active, evaluate(active)):
            error = block_error(fine, st['coarse'])
            if math.isfinite(error) and error <= rtol:
                results[st['index']] = (fine, st['n'], error)
                continue
            if st['n'] >= max_order:
                raise ValueError(f"BoR near meridian quadrature did not converge for elements ({st['e']}, {st['f']}); gap={st['gap']:.6g}, order={st['n']}, relative block change={error:.3g}. Refine the mesh or increase near_max_order.")
            st['coarse'] = fine
            still.append(st)
        active = still
    return results


# The exact Galerkin EFIE blocks of one medium are reciprocal: for every near
# element pair tt(e,f) = tt(f,e)^T, ff(e,f) = ff(f,e)^T, tf(e,f) = -ft(f,e)^T.
# Integrated with the fixed graded self/adjacent rule they are not exactly so,
# and the violation bounds that rule's error from below at no cost (measured
# 1e-7..2e-5 of the largest block, concentrated at axis poles and corners).
# A violation above this fraction is reported as a warning.
EFIE_NEAR_ASYMMETRY_WARNING = 1.0e-3


def _warn_near_asymmetry(summary, warnings) -> 'None':
    value = float(summary.get('efie_near_block_asymmetry_max', 0.0))
    if value > EFIE_NEAR_ASYMMETRY_WARNING:
        warnings.append(
            f"BoR self/adjacent near quadrature: the EFIE near blocks violate "
            f"reciprocity by {value:.2e} of the largest block (limit "
            f"{EFIE_NEAR_ASYMMETRY_WARNING:.0e}); refine the mesh near axis "
            "poles and sharp corners or check for extremely small elements.")


def _stream_compression_evidence(streams):
    return {name:dict(stream.evidence) for name,stream in streams.items()
            if getattr(stream,'evidence',None)}


def _near_quadrature_summary(*operators):
    asymmetry = max((getattr(op, 'near_efie_asymmetry', 0.0) for op in operators), default=0.0)
    return {
        'scheme': 'bounded_compact_blocks_with_angular_and_disjoint_meridian_refinement',
        'disjoint_meridian_order_max': max(
            (getattr(op, 'near_quadrature_order_max', 0) for op in operators), default=0),
        'disjoint_meridian_relative_change_max': max(
            (getattr(op, 'near_quadrature_error_max', 0.) for op in operators), default=0.),
        'self_and_junction_rule': 'graded_singular_cells',
        'self_and_junction_convergence_checked': False,
        # Reciprocity of the retained EFIE near blocks: a free a-posteriori
        # lower bound on the self/adjacent quadrature error (not a certificate).
        'efie_near_block_asymmetry_max': float(asymmetry),
        'efie_near_block_asymmetry_warning': EFIE_NEAR_ASYMMETRY_WARNING,
        'prepared_modal_bands': [band for operator in operators
                                 for band in getattr(operator, 'near_preparation_bands', ())],
    }


class BorCrossOperators:
    """T (EFIE) and P (rotated-PV) Galerkin blocks between two DIFFERENT
    generatrices in one homogeneous medium: test bases on solver sp, source
    bases on solver sq (both BorPecSolver instances with the same medium).

    Pairs closer than near_factor * max(element lengths) are re-integrated
    with gap-graded meridian quadrature and independent block refinement.
    Only compact mode blocks are retained. The surfaces may TOUCH at shared
    endpoints (coating-termination junctions): element pairs sharing such a
    point are log-singular in the Galerkin sense and are routed to the same
    graded corner-cell quadrature the same-surface assembly uses for
    adjacent elements.  Overlapping/crossing interiors remain an error.

    ``need_p=False`` omits rotated-PV work only when neither this mapping nor
    its reciprocal consumer needs P; EFIE retains its own convergence test.
    """

    def __init__(self, sp: 'BorPecSolver', sq: 'BorPecSolver',
                 near_factor: 'float' = 2.0, near_order: 'int' = 12,
                 near_rtol: 'float' = 2e-5, near_max_order: 'int' = 192,
                 need_p: 'bool' = True):
        if not np.isclose(complex(sp.k), complex(sq.k)):
            raise ValueError("Cross operators need both solvers in the same medium.")
        self.sp, self.sq = sp, sq
        self.k, self.eta = sp.k, sp.eta
        self.need_p = bool(need_p)
        self._near_kinds = ('efie', 'ibc') if self.need_p else ('efie',)
        self.near_order = int(near_order)
        self.near_rtol = float(near_rtol)
        self.near_max_order = int(near_max_order)
        if self.near_order < 2 or not 2 * self.near_order <= self.near_max_order <= 384 or not 0 < self.near_rtol < 1:
            raise ValueError('BoR near quadrature needs order >= 2, 2*order <= max_order <= 384, and 0 < rtol < 1.')
        self.near_quadrature_order_max = 0
        self.near_quadrature_error_max = 0.0

        gp, gq = sp.gen, sq.gen
        # Axial translation must not change whether two surfaces intersect.
        diag = max(float(np.linalg.norm(np.ptp(gp.nodes, axis=0))),
                   float(np.linalg.norm(np.ptp(gq.nodes, axis=0))), 1e-9)
        coordinate_size = max(float(np.max(abs(gp.nodes))), float(np.max(abs(gq.nodes))))
        touch_tol = max(1e-9 * diag, 8 * np.finfo(float).eps * coordinate_size)
        self.near_pairs = []
        self.pair_kind: 'Dict[Tuple[int, int], Optional[str]]' = {}
        self._far_gap = math.inf
        from scipy.spatial import cKDTree
        q_centers = .5 * (gq.nodes[gq.elem_n0] + gq.nodes[gq.elem_n1])
        q_radius_max = .5 * float(np.max(gq.lengths))
        tree = cKDTree(q_centers)
        for e in range(gp.n_elems):
            p_ends = (gp.nodes[gp.elem_n0[e]], gp.nodes[gp.elem_n1[e]])
            # A center-distance bound encloses every close/touching segment.
            # Omitted pairs have a certified lower distance bound for FFT sizing.
            gap_bound = max(touch_tol, near_factor * max(gp.lengths[e], 2*q_radius_max))
            radius = .5*gp.lengths[e] + q_radius_max + gap_bound
            candidates = sorted(tree.query_ball_point(.5*(p_ends[0]+p_ends[1]),
                                                     radius + touch_tol))
            if len(candidates) < gq.n_elems:
                self._far_gap = min(self._far_gap, gap_bound)
            for f in candidates:
                q_ends = (gq.nodes[gq.elem_n0[f]], gq.nodes[gq.elem_n1[f]])
                shared = [(a, b) for a in (0, 1) for b in (0, 1)
                          if float(np.hypot(*(p_ends[a] - q_ends[b]))) <= touch_tol]
                d = _segment_distance(p_ends[0], p_ends[1], q_ends[0], q_ends[1])
                if len(shared) > 1:
                    raise ValueError("Cross-operator surfaces share a whole "
                                     "element (overlapping geometry).")
                if shared:
                    a, b = shared[0]
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = f"corner{a}{b}"
                elif d <= touch_tol:
                    raise ValueError("Cross-operator surfaces cross or touch "
                                     "away from a shared junction endpoint.")
                elif d < near_factor * max(gp.lengths[e], gq.lengths[f]):
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = None
                else:
                    self._far_gap = min(self._far_gap, d)
        if not math.isfinite(self._far_gap):
            self._far_gap = 0.0
        self.near_set = set(self.near_pairs)
        self._G = None
        self._B = None
        self._stream = None
        self._cache: 'Dict' = {}

    def enable_streaming(self, m_max: 'int', single_blocks: 'bool' = False,
                         tile_budget_gb: 'float' = 1.0,
                         workers: 'int' = 1,
                         mode_block: 'Optional[int]' = None,
                         spill: 'Optional[str]' = None) -> 'None':
        """Build per-mode nodal far blocks before operator assembly.

        Near/self quadrature uses the same kernels. IBC blocks include source Z_s, so
        assemble_ibc_extra must receive the same zs_pt. PMCHWT uses rotated-PV blocks
        with unit source weight.  ``spill`` is the base directory chosen by
        ``plan_stream_spill`` for a one-sweep memory-mapped build, or None.
        """

        from ghost_backend.bor.streaming import StreamingCrossFarBlocks
        from ghost_backend.bor import kernels as modal_kernels
        from ghost_backend.bor.compressed_far import far_compression_selected
        from ghost_backend.bor.compressed_cross import CompressedCrossFarBlocks, CrossCompressionBudgetError
        self.close_streaming()
        store = (CompressedCrossFarBlocks if modal_kernels.BANDED_FFT and not single_blocks and
                 far_compression_selected(min(self.sp.Nn, self.sq.Nn)) else StreamingCrossFarBlocks)
        arguments = dict(
            dtype=np.complex64 if single_blocks else np.complex128,
            tile_budget_gb=tile_budget_gb,
            workers=workers,
            mode_block=mode_block,
            spill=spill,
        )
        try:
            self._stream = store(self, m_max, **arguments)
        except CrossCompressionBudgetError as exc:
            self._stream = StreamingCrossFarBlocks(self, m_max, **arguments)
            self._stream.evidence = {'backend': 'dense_rectangular_tiles',
                                     'compression_fallback': str(exc)}

    def close_streaming(self) -> 'None':
        """Release streamed far blocks, including any spilled files."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def _tables(self, m_max: 'int'):
        if self._G is not None and self._G.shape[-1] >= m_max + 2:
            return self._G, self._B
        gp, gq = self.sp.g, self.sq.g
        Pp, Pq = len(gp.rho), len(gq.rho)
        rho_scale = max(float(np.max(gp.rho)), float(np.max(gq.rho)))
        n_xi_g = n_xi_for_pairs(self.k, rho_scale, m_max, self._far_gap,
                                bracket=False)
        n_xi_b = (n_xi_for_pairs(self.k, rho_scale, m_max, self._far_gap,
                                bracket=True) if self.need_p else None)
        near_mask = np.zeros((Pp, Pq), dtype=bool)
        for (e, f) in self.near_pairs:
            near_mask[np.ix_(gp.elem == e, gq.elem == f)] = True
        table_dtype = self.sp._table_dtype
        if _blocked_single_tables(table_dtype):
            args = tuple(getattr(gp, name)[:, None] for name in ('rho', 'z', 'trho', 'tz'))
            args += tuple(getattr(gq, name)[None, :] for name in ('rho', 'z', 'trho', 'tz'))
            G, = _tables_by_rows(
                lambda r0, r1: (modal_kernels_fft(
                    args[0][r0:r1], args[1][r0:r1], args[4], args[5], self.k, m_max,
                    n_xi=n_xi_g, near_mask=near_mask[r0:r1], threads=physical_cpu_count()),),
                Pp, Pq, 16.0 * Pq * (m_max + 2), table_dtype, near_mask)
            B = _tables_by_rows(
                lambda r0, r1: nonnegative_bracket_tables(
                    'ibc', tuple(a[r0:r1] for a in args[:4]) + args[4:], self.k, m_max,
                    n_xi_b, near_mask[r0:r1], threads=physical_cpu_count()),
                Pp, Pq, 4 * 16.0 * Pq * (m_max + 1), table_dtype, near_mask) if self.need_p else None
            self._G, self._B = G, None if B is None else tuple(B)
            return self._G, self._B
        G = modal_kernels_fft(gp.rho[:, None], gp.z[:, None],
                              gq.rho[None, :], gq.z[None, :],
                              self.k, m_max, n_xi=n_xi_g, near_mask=near_mask,
                              threads=physical_cpu_count())
        args = tuple(getattr(gp,name)[:,None] for name in ('rho','z','trho','tz'))
        args += tuple(getattr(gq,name)[None,:] for name in ('rho','z','trho','tz'))
        B = (nonnegative_bracket_tables('ibc', args, self.k, m_max, n_xi_b, near_mask,
                                       threads=physical_cpu_count()) if self.need_p else None)
        if self.near_pairs:
            G[near_mask, :] = 0.0
            if B is not None:
                for value in B:
                    value[near_mask, :] = 0.0
        table_dtype = self.sp._table_dtype
        self._G = G.astype(table_dtype, copy=False)
        self._B = None if B is None else tuple(value.astype(table_dtype, copy=False) for value in B)
        return self._G, self._B

    def _integrate_near(self, e, f, m_max, signed=True, mode_start=0):
        """Return (blocks, (order, error) or None) without touching state.

        Production callers pass ``signed=False`` (modes 0..m_max only)."""
        kind = self.pair_kind.get((e, f))
        if kind is not None:
            return _contract_near_points(self.sp.gen, e, self.sq.gen, f,
                self.k, m_max, self._near_kinds, _junction_cell_points(kind), signed=signed,
                mode_start=mode_start), None
        blocks, order, error = _converged_disjoint_blocks(
            self.sp.gen, e, self.sq.gen, f, self.k, m_max,
            self._near_kinds, self.near_order, self.near_rtol, self.near_max_order,
            signed=signed, mode_start=mode_start)
        return blocks, (order, error)

    def _store_near(self, e, f, m_max, blocks, refinement, mode_start=0):
        if refinement is not None:
            order, error = refinement
            self.near_quadrature_order_max = max(self.near_quadrature_order_max, order)
            self.near_quadrature_error_max = max(self.near_quadrature_error_max, error)
        # Keep an owned copy of the nonnegative half; a slice alone would keep
        # both signs alive when the caller integrated the signed modes.
        from ghost_backend.bor.near_storage import ModalBands
        retained = {kind: (value if value.shape[1] == m_max + 1 - mode_start
                           else value[:, m_max:]).copy() for kind, value in blocks.items()}
        if mode_start:
            previous = self._cache[mode_start - 1][(e, f)]
            retained = {kind: ModalBands(previous[kind], value) for kind, value in retained.items()}
        self._cache.setdefault(m_max, {})[(e, f)] = retained

    def _near_data(self, e, f, m_max):
        """Cache only converged 2x2 EFIE and rotated-PV mode blocks."""
        for cap, data in self._cache.items():
            if cap >= m_max and (e, f) in data:
                return data[(e, f)]
        cache = self._cache.setdefault(m_max, {})
        if (e, f) not in cache:
            self._store_near(e, f, m_max, *self._integrate_near(e, f, m_max, signed=False))
        return cache[(e, f)]

    def assemble_T(self, m: 'int', m_max: 'int', out=None) -> 'np.ndarray':
        """Cross EFIE operator [2Np, 2Nq] (same normalization as
        BorPecSolver.assemble_mode, C = j k eta 2pi of this medium)."""
        if self.sp._compressed:
            if out is not None:
                raise ValueError('Compressed cross assembly cannot write in place.')
            return primitive(self, 'T', m, m_max)


        k = self.k
        gp, gq = self.sp.g, self.sq.g
        Np, Nq = self.sp.Nn, self.sq.Nn
        Z = np.empty((2*Np, 2*Nq), complex) if out is None else out
        if Z.shape != (2*Np, 2*Nq):
            raise ValueError('Cross EFIE destination has the wrong shape.')
        ztt, ztf, zft, zff = Z[:Np,:Nq], Z[:Np,Nq:], Z[Np:,:Nq], Z[Np:,Nq:]
        if self._stream is not None:
            self._stream.write_blocks('efie', m, (ztt, ztf, zft, zff))
        else:
            G, _ = self._tables(m_max)
            Z.fill(0.)
            _efie_tables_into((ztt, ztf, zft, zff), m, k,
                              gp, self.sp.gen.n_elems, self.sp.gauss_order,
                              gq, self.sq.gen.n_elems, self.sq.gauss_order, G, 1.0)
        for e, f in self.near_pairs:
            blocks = mode_blocks(self._near_data(e, f, m_max)['efie'], m)
            rc = np.ix_([e, e + 1], [f, f + 1])
            for target, value in zip((ztt, ztf, zft, zff), blocks):
                target[rc] += value

        C = 1j * k * self.eta * 2.0 * np.pi
        Z *= C
        return Z

    def assemble_P(self, m: 'int', m_max: 'int', out=None) -> 'np.ndarray':
        """Cross rotated-PV operator [2Np, 2Nq] (see assemble_pmchwt_P)."""
        if not self.need_p:
            raise ValueError('This cross operator was prepared for EFIE only.')
        if self.sp._compressed:
            if out is not None:
                raise ValueError('Compressed cross assembly cannot write in place.')
            return primitive(self, 'P', m, m_max)


        gp, gq = self.sp.g, self.sq.g
        Np, Nq = self.sp.Nn, self.sq.Nn
        P = np.empty((2*Np, 2*Nq), complex) if out is None else out
        if P.shape != (2*Np, 2*Nq):
            raise ValueError('Cross PV destination has the wrong shape.')
        blocks = (P[:Np,Nq:], P[:Np,:Nq], P[Np:,Nq:], P[Np:,:Nq])
        if self._stream is not None:
            self._stream.write_blocks('ibc', m, blocks)
        else:
            _, Bt = self._tables(m_max)
            P.fill(0.)
            _bracket_tables_into(blocks, m, m_max,
                                 gp, self.sp.gen.n_elems, self.sp.gauss_order,
                                 gq, self.sq.gen.n_elems, self.sq.gauss_order, Bt, 1.0)
        for e, f in self.near_pairs:
            near = mode_blocks(self._near_data(e, f, m_max)['ibc'], m)
            rc = np.ix_([e, e + 1], [f, f + 1])
            for target, value in zip(blocks, near):
                target[rc] += value
        P[:, :Nq] *= -1.
        return P

    def prepare(self, m_max: 'int', workers: 'int' = 1) -> 'None':
        """Warm requested table/near families (see BorPecSolver.prepare_operators)."""
        if self._stream is None and not self.sp._compressed:
            self._tables(m_max)
        cached = self._cache.get(m_max, {})
        previous = max((cap for cap, data in self._cache.items()
                        if all(pair in data for pair in self.near_pairs)), default=-1)
        if previous >= m_max:
            return
        mode_start = previous + 1
        pending = [pair for pair in self.near_pairs if pair not in cached]

        def integrate(pair):
            if self.sp._checkpoint is not None:
                self.sp._checkpoint()
            return self._integrate_near(pair[0], pair[1], m_max, signed=False, mode_start=mode_start)

        from ghost_backend.bor.near_parallel import NearTask
        task = NearTask(self.sp.gen, self.sq.gen, self.k, m_max, self._near_kinds,
            pair_kind=self.pair_kind, near_order=self.near_order,
            near_rtol=self.near_rtol, near_max_order=self.near_max_order, mode_start=mode_start)
        with contextlib.closing(_iter_near_pairs(integrate, pending, workers, task, self.sp._checkpoint)) as results:
            for (e, f), result in zip(pending, results):
                self._store_near(e, f, m_max, *result, mode_start=mode_start)
        for cap in [cap for cap in self._cache if cap != m_max]:
            del self._cache[cap]


class _ReverseCrossOperators:
    """The reverse mapping (test on ``forward.sq``, source on ``forward.sp``)
    of a ``BorCrossOperators``, derived exactly instead of integrated again.

    For one medium the Galerkin blocks of mode ``m`` satisfy, with
    ``T^{pq} = [[tt, tf], [ft, ff]]`` and ``P^{pq} = [[a, b], [c, d]]``,

        T^{qp} = [[tt^T, -ft^T], [-tf^T, ff^T]]
        P^{qp} = [[-a^T,  c^T], [ b^T, -d^T]]

    (reciprocity of the symmetric Green's function and the tangential-block
    parity), verified to 4e-15 against independently built operators, near
    pairs included.  The reverse far tables/streamed blocks and near
    integrations -- half of all cross-surface work -- are never built.
    """

    derived = True

    def __init__(self, forward: 'BorCrossOperators'):
        self.forward = forward
        self.sp, self.sq = forward.sq, forward.sp
        self.k, self.eta = forward.k, forward.eta
        self.need_p = forward.need_p
        self._stream = None
        # Lets _cross_block keep the forward block of a mode for this reverse.
        forward._has_derived_reverse = True

    @property
    def near_pairs(self):
        return [(f, e) for (e, f) in self.forward.near_pairs]

    @property
    def near_quadrature_order_max(self):
        return self.forward.near_quadrature_order_max

    @property
    def near_quadrature_error_max(self):
        return self.forward.near_quadrature_error_max

    def enable_streaming(self, *args, **kwargs) -> 'None':
        return None

    def close_streaming(self) -> 'None':
        return None

    def prepare(self, m_max: 'int', workers: 'int' = 1) -> 'None':
        return None

    def _split(self, block):
        Np, Nq = self.forward.sp.Nn, self.forward.sq.Nn
        return block[:Np, :Nq], block[:Np, Nq:], block[Np:, :Nq], block[Np:, Nq:]

    def reverse_T(self, forward_block) -> 'np.ndarray':
        """This operator's T block from the forward T block of the same mode."""
        tt, tf, ft, ff = self._split(forward_block)
        Nq, Np = self.sp.Nn, self.sq.Nn
        Z = np.empty((2 * Nq, 2 * Np), dtype=np.complex128)
        Z[:Nq, :Np] = tt.T
        Z[:Nq, Np:] = -ft.T
        Z[Nq:, :Np] = -tf.T
        Z[Nq:, Np:] = ff.T
        return Z

    def reverse_P(self, forward_block) -> 'np.ndarray':
        """This operator's P block from the forward P block of the same mode."""
        a, b, c, d = self._split(forward_block)
        Nq, Np = self.sp.Nn, self.sq.Nn
        P = np.empty((2 * Nq, 2 * Np), dtype=np.complex128)
        P[:Nq, :Np] = -a.T
        P[:Nq, Np:] = c.T
        P[Nq:, :Np] = b.T
        P[Nq:, Np:] = -d.T
        return P

    def assemble_T(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        return self.reverse_T(self.forward.assemble_T(m, m_max))

    def assemble_P(self, m: 'int', m_max: 'int') -> 'np.ndarray':
        return self.reverse_P(self.forward.assemble_P(m, m_max))


def _reverse_cross(forward: 'BorCrossOperators', **kwargs):
    """Derived reverse operators, or independently built ones when the
    forward assembly is symbolic (compressed tiles cannot be transposed)."""
    if forward.sp._compressed:
        kwargs.setdefault('need_p', forward.need_p)
        return _prepared_cross(forward.sq, forward.sp, **kwargs)
    return _ReverseCrossOperators(forward)


def _cross_block(cross, kind: 'str', m: 'int', m_max: 'int', memo: 'Dict'):
    """``cross.assemble_T`` (kind 'T') or ``assemble_P`` (kind 'P') of mode m.

    A forward operator and its derived reverse share one integration per
    mode: whichever is requested first assembles the forward block and parks
    it in ``memo`` (local to one modal assembly), the other takes it back
    out.  Without this the reverse re-assembled the forward block, so every
    surface pair was integrated twice per mode.  Returned blocks are shared
    and must not be modified in place.
    """

    def build(operator):
        return (operator.assemble_T(m, m_max) if kind == 'T'
                else operator.assemble_P(m, m_max))

    if getattr(cross, 'derived', False):
        key = (id(cross.forward), kind)
        block = memo.pop(key, None)
        if block is None:
            block = build(cross.forward)
            memo[key] = block
        return cross.reverse_T(block) if kind == 'T' else cross.reverse_P(block)
    key = (id(cross), kind)
    block = memo.pop(key, None)
    if block is None:
        block = build(cross)
        if getattr(cross, '_has_derived_reverse', False):
            memo[key] = block
    return block


def _impedance_rotated(matrix, nodal_zs, sparse_map):
    """``matrix @ S`` for ``S = [[0, diag(z)], [-diag(z), 0]]`` (the map from
    J to the magnetic current of an impedance surface).  Dense blocks are
    column-scaled in O(n^2) instead of a dense GEMM; compressed blocks keep
    the sparse product."""
    if not isinstance(matrix, np.ndarray):
        return matrix @ sparse_map
    n = nodal_zs.size
    out = np.empty(matrix.shape, dtype=np.result_type(matrix, nodal_zs))
    np.multiply(matrix[:, n:], -nodal_zs, out=out[:, :n])
    np.multiply(matrix[:, :n], nodal_zs, out=out[:, n:])
    return out


@profiled_solve
@configured
def solve_bor_coated_pec(points_outer, points_core, freq_hz: 'float', thetas_deg,
                         eps_r: 'complex', mu_r: 'complex' = 1.0,
                         n_modes: 'Optional[int]' = None, gauss_order: 'int' = FAR_GAUSS_ORDER,
                         mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                         near_order: 'int' = 12, workers: 'int' = 1,
                         progress: 'Optional[Callable]' = None,
                         check_abort: 'Optional[Callable]' = None,
                         table_precision: 'str' = "auto",
                         assembly: 'str' = "auto",
                         stream_budget_gb: 'float' = 8.0,
                         near_rtol: 'float' = 2e-5,
                         near_max_order: 'int' = 192) -> 'Dict':
    """
    Monostatic RCS of a PEC core (generatrix points_core) fully covered by a
    homogeneous coating with outer surface points_outer (both closed, both
    traversed +z end to -z end so left-of-travel normals face their exterior).

    Unknowns per mode: J_o, M'_o = M_o/eta0 on the outer interface, J_c on
    the core. PMCHWT rows on the outer interface pick up cross terms from
    J_c radiating in the layer. The core uses EFIE plus eta_L times MFIE
    to suppress lossless core-cavity resonances. The electric part is:

      [ T_e+T_L          -eta0(P_e+P_L)          -T_L^oc      ] [J_o ]   [ V_E      ]
      [ eta0(P_e+P_L)    T_e+(eta0/eta_L)^2 T_L  -eta0 P_L^oc ] [M'_o] = [ eta0 V_H ]
      [ T_L^co           -eta0 P_L^co            -T_L^cc      ] [J_c ]   [ 0        ]

    Only (J_o, M_o) radiate in air.
    """

    points_outer = _validate_solve_bor_generatrix(points_outer, "cfie")
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    from ghost_backend.bor.geometry import require_containment
    require_containment(points_outer, points_core)
    _causal_medium(eps_r, mu_r)
    se = _prepared_surface(points_outer, freq_hz, gauss_order=gauss_order)
    sLo = _prepared_surface(points_outer, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    sLc = _prepared_surface(points_core, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    Xoc = _prepared_cross(sLo, sLc, near_factor=near_factor, near_order=near_order,
                            near_rtol=near_rtol, near_max_order=near_max_order)
    # Core-to-outer operators follow exactly from outer-to-core ones.
    Xco = _reverse_cross(Xoc, near_factor=near_factor, near_order=near_order,
                         near_rtol=near_rtol, near_max_order=near_max_order)
    reverse_derived = getattr(Xco, 'derived', False)
    k = se.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    No, Nc = se.Nn, sLc.Nn
    eta_ratio2 = (ETA0 / sLo.eta) ** 2
    weight = _region_equation_weights([None, (eps_r, mu_r)])[1]
    ntot = 4 * No + 2 * Nc

    from ghost_backend.bor.streaming import (
        BOR_STREAM_TILE_BUDGET_GB,
        combined_stream_mode_gb,
        estimate_rectangular_streaming_gb,
        plan_combined_streaming_mode_block,
        plan_stream_spill,
    )
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    ne_outer = se.gen.n_elems
    ne_core = sLc.gen.n_elems
    table_far_double = (
        2.0 * estimate_bor_table_gb(
            ne_outer, m_max, "efie", True, gauss_order, False
        )
        + estimate_bor_table_gb(
            ne_core, m_max, "cfie", False, gauss_order, False
        )
        + estimate_bor_cross_table_gb(
            ne_outer, ne_core, m_max, gauss_order, gauss_order, False
        )
        + (0.0 if reverse_derived else estimate_bor_cross_table_gb(
            ne_core, ne_outer, m_max, gauss_order, gauss_order, False
        ))
    )
    use_streaming = (
        asm == "streaming"
        or (asm == "auto" and table_far_double > 2.0)
    )
    stream_specs_double = (
        (ne_outer, ne_outer, True, False),
        (ne_outer, ne_outer, True, False),
        (ne_core, ne_core, True, False),
        (ne_outer, ne_core, True, False),
    ) + (() if reverse_derived else ((ne_core, ne_outer, True, False),))
    full_stream_double = sum(
        estimate_rectangular_streaming_gb(
            nt, ns, m_max, rotated, single
        )
        for nt, ns, rotated, single in stream_specs_double
    )
    use_single = tp == "single"
    solve_workers = max(1, int(workers))
    mode_block = None
    stream_spill = None
    if use_streaming:
        stream_specs = tuple(
            (nt, ns, rotated, use_single)
            for nt, ns, rotated, _single in stream_specs_double
        )
        mode_block, held_blocks_gb, solve_workers = (
            plan_combined_streaming_mode_block(
                m_max, stream_specs, stream_budget, solve_workers
            )
        )
        # As solve_bor: every stream is built once into memory-mapped files
        # when the budget cannot hold every mode.
        stream_spill, mode_block, resident_gb = plan_stream_spill(
            mode_block, m_max + 1, combined_stream_mode_gb(m_max, stream_specs),
            allow_spill=not any(_uses_far_compression(item.Nn) for item in (se, sLo, sLc)))
        if stream_spill is not None:
            held_blocks_gb = resident_gb
            solve_workers = max(1, int(workers))
        # The near contractions (self and cross) and their build workspace
        # stay resident when the far blocks stream, as solve_bor,
        # solve_bor_dielectric and the junction planner already price.
        operator_storage_gb = (
            held_blocks_gb + BOR_STREAM_TILE_BUDGET_GB
            + estimate_bor_operator_storage_gb(
                m_max,
                (
                    (se, True, False, True),
                    (sLo, True, False, True),
                    (sLc, True, True, False),
                ),
                (Xoc, Xco),
                streaming=True,
            )
        )
    else:
        if use_single:
            for solver in (se, sLo, sLc):
                solver._table_dtype = np.complex64
        operator_storage_gb = (
            estimate_bor_operator_storage_gb(
                m_max,
                (
                    (se, True, False, True),
                    (sLo, True, False, True),
                    (sLc, True, True, False),
                ),
                (Xoc, Xco),
            )
        )

    solve_warnings: 'List[str]' = []
    if use_single and tp == "auto":
        solve_warnings.append(
            "Streamed self/cross far blocks stored in single precision "
            f"({full_stream_double / 2.0:.1f} GB; double would need "
            f"{full_stream_double:.1f} GB)."
        )

    iJ = slice(0, 2 * No); iM = slice(2 * No, 4 * No); iC = slice(4 * No, ntot)

    def prepare(mm):
        if use_streaming:
            se.enable_streaming(
                mm, efie=True, pmchwt=True,
                single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
                spill=stream_spill,
            )
            sLo.enable_streaming(
                mm, efie=True, pmchwt=True,
                single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
                spill=stream_spill,
            )
            sLc.enable_streaming(
                mm, efie=True, mfie=True, single_blocks=use_single,
                tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                workers=solve_workers, mode_block=mode_block,
                spill=stream_spill,
            )
            for cross in (Xoc, Xco):
                cross.enable_streaming(
                    mm, single_blocks=use_single,
                    tile_budget_gb=BOR_STREAM_TILE_BUDGET_GB,
                    workers=solve_workers, mode_block=mode_block,
                    spill=stream_spill,
                )
        se.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sLo.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sLc.prepare_operators(mm, efie=True, mfie=True, workers=workers)
        Xoc.prepare(mm, workers=workers)
        Xco.prepare(mm, workers=workers)

    def assemble(m):
        memo = {}
        # The coating's equations enter the interface rows with the weight
        # of _region_equation_weights, jump terms included.
        if use_streaming and not se._compressed:
            A = modal_matrix((ntot, ntot), False)
            _assemble_pmchwt_interface_into(A[:4 * No, :4 * No], se, sLo,
                                            m, m_max, weight, eta_ratio2)
        else:
            T_e = se.assemble_mode(m, m_max)
            T_Lo = sLo.assemble_mode(m, m_max)
            P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + weight * sLo.assemble_pmchwt_P(m, m_max))
            if weight != 1.0:
                P_sum = _add_rotation_mass_into(P_sum, se, -(0.5 * (1.0 - weight) * ETA0))
            A = modal_matrix((ntot, ntot), compressed_requested())
            A[iJ, iJ] = T_e + weight * T_Lo
            A[iJ, iM] = -P_sum
            A[iM, iJ] = P_sum
            A[iM, iM] = T_e + (weight * eta_ratio2) * T_Lo
            del T_e, T_Lo, P_sum
        A[iJ, iC] = -weight * _cross_block(Xoc, 'T', m, m_max, memo)
        A[iM, iC] = (-weight * ETA0) * _cross_block(Xoc, 'P', m, m_max, memo)
        Tco = _cross_block(Xco, 'T', m, m_max, memo)
        Pco = _cross_block(Xco, 'P', m, m_max, memo)
        # Negative electric boundary row minus eta_L times the magnetic row.
        # M'_o is M_o/eta0, hence the eta0/eta_L cross-field coefficient.
        A[iC, iJ] = Tco + sLc.eta * _rotate_test_rows(Pco, Nc)
        A[iC, iM] = -ETA0 * Pco + (ETA0/sLc.eta) * _rotate_test_rows(Tco, Nc)
        A[iC, iC] = -sLc.assemble_mode(m, m_max) - sLc.eta*sLc.assemble_mfie_mode(m, m_max)
        if abs(int(m)) == 1:
            q_outer = se.basis_transform(m)
            Q = _block_diagonal_transforms(
                q_outer, q_outer, sLc.basis_transform(m)
            )
            return _reduce_constrained_operator(A, Q), Q
        mask_o = se.basis_mask(m)
        mask = np.concatenate([mask_o, mask_o, sLc.basis_mask(m)])
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(ntot, dtype=np.complex128)
        V[iJ] = se.rhs_mode(m, th, pol)
        V[iM] = ETA0 * se.rhs_h_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[iJ], th, msol=ETA0 * full[iM])
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        electric = se.rhs_vv_hh_batch(m, batch_thetas)
        out = np.zeros((ntot, electric.shape[1]), dtype=np.complex128)
        out[iJ] = electric
        out[iM, 0::2] = -electric[:, 1::2]
        out[iM, 1::2] = electric[:, 0::2]
        return out

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        return se.farfield_vv_hh_batch(
            m,
            full[iJ],
            batch_thetas,
            msolutions=ETA0 * full[iM],
        )

    stream_backends = None
    stream_backend = None
    stream_sweeps = 0
    stream_spill_gb = 0.0
    try:
        F, modes_used, stats = _mode_sweep(ntot, thetas, ("VV", "HH"), m_max,
                                           mode_tol, assemble, rhs, farfield,
                                           prepare=prepare, workers=solve_workers, preparation_workers=workers,
                                           progress=progress,
                                           check_abort=check_abort,
                                           monitor_cond=True,
                                           rhs_batch=rhs_batch,
                                           farfield_batch=farfield_batch,
                                           min_mode_before_tail=mode_tail_start,
                                           assembly_peak_gb=operator_storage_gb,
                                           memory_context="The coated-PEC BoR solve",
                                           stream_mode_block=mode_block,
                                           signed_mode_symmetry=True, axial_mode_only=_all_axial_aspects(thetas),
                                           coordinates=lambda m: np.vstack((np.tile(se.gen.nodes, (4, 1)),
                                               np.tile(sLc.gen.nodes, (2, 1))))[
                                                   np.r_[np.tile(se.basis_mask(m), 2), sLc.basis_mask(m)]],
                                           hierarchical_pricing=False)
        if use_streaming:
            stream_objects = {
                "exterior_outer": se._stream,
                "coating_outer": sLo._stream,
                "coating_core": sLc._stream,
                "cross_outer_core": Xoc._stream,
            }
            if not reverse_derived:
                stream_objects["cross_core_outer"] = Xco._stream
            stream_compression = _stream_compression_evidence(stream_objects)
            stream_backends = {
                name: sampling_backend_name(stream)
                for name, stream in stream_objects.items()
            }
            unique_backends = set(stream_backends.values())
            stream_backend = (
                next(iter(unique_backends)) if len(unique_backends) == 1
                else "mixed"
            )
            stream_sweeps = sum(
                stream.n_sweeps for stream in stream_objects.values()
            )
            stream_spill_gb = sum(
                stream.spilled_gb() for stream in stream_objects.values()
            )
    finally:
        for owner in (se, sLo, sLc, Xoc, Xco):
            owner.close_streaming()
    _require_mode_convergence(stats, mode_tol)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(ntot),
        "formulation": "pmchwt-coated",
        "core_formulation": "EFIE plus coating-impedance MFIE",
        "near_quadrature": _near_quadrature_summary(se, sLo, sLc, Xoc, Xco),
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": mode_block if use_streaming else None,
        "stream_sweeps": stream_sweeps,
        "stream_spill_gb": stream_spill_gb,
        "cross_reverse_derived": bool(reverse_derived),
        "stream_far_compression": stream_compression if use_streaming else {},
        "stream_sampling_backend": stream_backend,
        "stream_sampling_backends": stream_backends,
        "warnings": solve_warnings,
        **stats,
    }


@profiled_solve
@configured
def solve_bor_partial_coating(points_interface, points_covered, bare_pieces,
                              freq_hz: 'float', thetas_deg, eps_r: 'complex',
                              mu_r: 'complex' = 1.0, bare_zs=None,
                              n_modes: 'Optional[int]' = None,
                              gauss_order: 'int' = FAR_GAUSS_ORDER, mode_tol: 'float' = 1e-6,
                              near_factor: 'float' = 2.0, near_order: 'int' = 12,
                              workers: 'int' = 1, progress: 'Optional[Callable]' = None,
                              check_abort: 'Optional[Callable]' = None,
                              table_precision: 'str' = "auto",
                              assembly: 'str' = "auto",
                              stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """Monostatic RCS of a PEC body PARTIALLY covered by a homogeneous coating:
    the dielectric interface S_d (points_interface) terminates on the PEC
    surface at junction circles where air, coating, and conductor meet.

    points_covered is the coated part of the core, bare_pieces the list of
    uncovered PEC generatrix pieces (0, 1, or 2 -- cap or band coatings).
    All pieces are drawn in the global +z -> -z traversal with left-of-travel
    normals facing away from the surface they bound (exterior/air for S_d
    and the bare pieces, into the coating for the covered core).

    Formulation: exterior region bounded by S_d + bare pieces (currents J_d,
    M_d, J_1p in air), layer region bounded by S_d + covered core (currents
    -J_d, -M_d, J_2 in the coating medium).  PMCHWT rows on S_d, EFIE rows
    on each PEC piece; all with the same operator blocks.  Junction
    conditions (per junction circle A):

      * through-current continuity ties the chain-end coefficients:
        J_1p(A) = (+t, +phi) J_d(A)  (air chain runs S_d and the bare piece
        in the same global traversal), and
        J_2(A)  = (+t, -phi) J_d(A)  (the layer chain traverses S_d REVERSED:
        the -J_d current in the flipped frame has +t and -phi components);
      * M_t(A) = 0 (tangential E along the junction circle vanishes on the
        conductor), while M_phi(A) stays free with its natural half-triangle
        end basis (it carries the normal-E wedge behavior).

    The constraints enter Galerkin-style: A_red = Q^T A_full Q -- the tied
    row is the sum of the piece rows, exactly the classical BoR junction
    treatment (Putnam / Medgyesi-Mitschang).

    bare_zs (optional): per-piece Leontovich surface impedance -- a list
    matching bare_pieces of None / complex scalar / per-element complex
    arrays (tapers).  The eliminated magnetic current M_1 = -Z_s n_hat x J_1
    keeps the piece's own operator on the validated Gauss-point IBC path
    (assemble_ibc_extra) and radiates onto OTHER surfaces through the
    existing cross T/P operators via the nodal column map
    (M_1t, M_1phi) = (+Z_s J_1phi, -Z_s J_1t). At a junction adjoining an
    impedance piece, M_t(A) = 0 remains the convergent one-node junction
    approximation for the singular wedge-line field limit; see build_Q.
    """

    points_interface = _validate_solve_bor_generatrix(
        points_interface, "efie"
    )
    points_covered = _validate_solve_bor_generatrix(points_covered, "efie")
    bare_pieces = [
        _validate_solve_bor_generatrix(piece, "efie")
        for piece in bare_pieces
    ]
    _causal_medium(eps_r, mu_r)
    sd_e = _prepared_surface(points_interface, freq_hz, gauss_order=gauss_order)
    sd_L = _prepared_surface(points_interface, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    s2_L = _prepared_surface(points_covered, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    bares = [_prepared_surface(p, freq_hz, gauss_order=gauss_order)
             for p in bare_pieces]


    if bare_zs is None:
        bare_zs = [None] * len(bares)
    if len(bare_zs) != len(bares):
        raise ValueError("bare_zs must have one entry per bare piece.")
    zs_elems: 'List[Optional[np.ndarray]]' = []
    zs_ptss: 'List[Optional[np.ndarray]]' = []
    zs_nodes: 'List[Optional[np.ndarray]]' = []
    S_maps: 'List[Optional[csr_matrix]]' = []
    for b, zs in zip(bares, bare_zs):
        ne = b.gen.n_elems
        zs_arr = None
        if zs is not None:
            za = np.asarray(zs, dtype=complex)
            zs_arr = np.full(ne, complex(za)) if za.ndim == 0 else za.astype(complex)
            if len(zs_arr) != ne:
                raise ValueError("Per-element bare_zs array length must match "
                                 "the piece's element count.")
            zs_arr = _validate_bor_surface_impedance(
                zs_arr, "BoR partial-coating bare surface impedance"
            )
            if not np.any(np.abs(zs_arr) > 0.0):
                zs_arr = None
        zs_elems.append(zs_arr)
        if zs_arr is None:
            zs_ptss.append(None)
            zs_nodes.append(None)
            S_maps.append(None)
        else:
            zs_ptss.append(zs_arr[b.g.elem])
            zn = np.empty(b.Nn, dtype=complex)
            zn[0] = zs_arr[0]
            zn[-1] = zs_arr[-1]
            zn[1:-1] = 0.5 * (zs_arr[:-1] + zs_arr[1:])
            zs_nodes.append(zn)
            # Sparse in every backend; dense blocks apply it as a column
            # scaling (_impedance_rotated), never as a (2N)^2 matrix.
            S_maps.append(
                bmat([[None, diags(zn)], [-diags(zn), None]], format='csr')
            )

    nonzero_bare_zs = [
        array for array in zs_elems if array is not None
    ]
    if (
        nonzero_bare_zs
        and _effectively_reactive_surface_impedance(
            np.concatenate(nonzero_bare_zs)
        )
    ):
        raise RuntimeError(
            "Closed partial-coated body with lossless/reactive bare IBC on "
            "the EFIE is unsupported: undamped interior resonances cannot "
            "be ruled out reliably, and an IBC-compatible resonance-free "
            "CFIE is not implemented. Resistance is not a general resonance safeguard; use a "
            "validated full-wave formulation."
        )

    all_nodes = np.vstack([sd_e.gen.nodes, s2_L.gen.nodes] +
                          [b.gen.nodes for b in bares])
    diag = max(float(np.ptp(all_nodes[:, 0])) + float(np.ptp(all_nodes[:, 1])), 1e-9)
    jn_tol = 1e-8 * diag


    def endpoints(solver):
        gen = solver.gen
        out = []
        for node in (0, gen.n_nodes - 1):
            if gen.node_on_axis(node):
                out.append((node, None))
            else:
                out.append((node, gen.nodes[node]))
        return out

    junctions: 'List[Dict]' = []

    def register(kind_key, piece_idx, node, pos):
        for jn in junctions:
            if float(np.hypot(*(jn["pos"] - pos))) <= jn_tol:
                if kind_key in jn:
                    raise ValueError("Two same-role chain ends meet at one junction.")
                jn[kind_key] = (piece_idx, node) if kind_key == "bare" else node
                return
        junctions.append({"pos": pos,
                          kind_key: (piece_idx, node) if kind_key == "bare" else node})

    for node, pos in endpoints(sd_e):
        if pos is not None:
            register("d_node", None, node, pos)
    for node, pos in endpoints(s2_L):
        if pos is not None:
            register("c_node", None, node, pos)
    for bi, b in enumerate(bares):
        for node, pos in endpoints(b):
            if pos is not None:
                register("bare", bi, node, pos)
    for jn in junctions:
        if "d_node" not in jn or "c_node" not in jn or "bare" not in jn:
            raise ValueError(
                "Every off-axis chain endpoint must be a coating-termination "
                "junction where the interface, the covered core, and exactly "
                f"one bare piece meet; found an incomplete junction at "
                f"(rho, z) = ({jn['pos'][0]:.6g}, {jn['pos'][1]:.6g}).")
        bi, bn = jn["bare"]
        za = zs_elems[bi]
        jn["zs"] = complex(za[0 if bn == 0 else -1]) if za is not None else 0.0

    solve_warnings: 'List[str]' = []
    for jn in junctions:
        if abs(jn["zs"]) > 0.02 * ETA0:
            solve_warnings.append(
                f"Surface impedance is {abs(jn['zs']):.1f} ohm at the coating "
                f"junction (rho, z) = ({jn['pos'][0]:.4g}, {jn['pos'][1]:.4g}): "
                "an abrupt Z_s step AT a coating edge is an ill-defined "
                "sheet-model limit (E_phi is discontinuous along the junction "
                "line) and the solution does not mesh-converge there. One "
                "validation case showed a ~0.5 dB mesh/discretization plateau; "
                "that observation is not an accuracy bound. Taper Z_s toward "
                "zero at the junction "
                "(the physical edge treatment) for converged results.")


    xkw = dict(near_factor=near_factor, near_order=near_order)
    # Every reverse mapping follows exactly from its forward one
    # (_ReverseCrossOperators), so only one direction per surface pair is
    # integrated.
    X_d2 = _prepared_cross(sd_L, s2_L, **xkw)
    X_2d = _reverse_cross(X_d2, **xkw)
    X_d1 = [_prepared_cross(sd_e, b, **xkw) for b in bares]
    X_1d = [_reverse_cross(X, **xkw) for X in X_d1]
    X_11 = {}
    for i in range(len(bares)):
        for j in range(i + 1, len(bares)):
            # Either direction can consume P through its source impedance.
            need_p = S_maps[i] is not None or S_maps[j] is not None
            X_11[(i, j)] = _prepared_cross(bares[i], bares[j], need_p=need_p, **xkw)
            X_11[(j, i)] = _reverse_cross(X_11[(i, j)], **xkw)

    k = sd_e.k
    thetas = _validated_bor_aspects(thetas_deg)
    rho_max = max([float(np.max(sd_e.gen.nodes[:, 0])),
                   float(np.max(s2_L.gen.nodes[:, 0]))] +
                  [float(np.max(b.gen.nodes[:, 0])) for b in bares])
    m_max, mode_tail_start = _bor_mode_limits(
        k, rho_max, thetas, n_modes
    )
    eta_ratio2 = (ETA0 / sd_L.eta) ** 2
    weight = _region_equation_weights([None, (eps_r, mu_r)])[1]

    Nd, N2 = sd_e.Nn, s2_L.Nn
    N1 = [b.Nn for b in bares]
    off_Jd, off_M, off_J2 = 0, 2 * Nd, 4 * Nd
    off_J1 = []
    acc = 4 * Nd + 2 * N2
    for n in N1:
        off_J1.append(acc)
        acc += 2 * n
    n_full = acc

    def prepare(mm):
        planned_workers = plan["workers"]
        if plan["use_streaming"] and sd_e._stream is None:
            common = dict(
                single_blocks=plan["use_single"],
                tile_budget_gb=plan["tile_budget_gb"],
                workers=planned_workers,
                mode_block=plan["mode_block"],
                spill=plan["spill"],
            )
            sd_e.enable_streaming(mm, efie=True, pmchwt=True, **common)
            sd_L.enable_streaming(mm, efie=True, pmchwt=True, **common)
            s2_L.enable_streaming(mm, efie=True, **common)
            for bi, bare in enumerate(bares):
                bare.enable_streaming(
                    mm,
                    efie=True,
                    ibc_zs_pt=zs_ptss[bi],
                    **common,
                )
            for cross in all_crosses:
                cross.enable_streaming(mm, **common)
        sd_e.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sd_L.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        s2_L.prepare_operators(mm, efie=True, workers=workers)
        for bi, b in enumerate(bares):
            b.prepare_operators(mm, efie=True, ibc=zs_elems[bi] is not None,
                                workers=workers)
        for X in all_crosses:
            X.prepare(mm, workers=workers)


        # Signed-mode symmetry assembles only m >= 0, so the m = -1 category
        # is never used; it is built lazily if a caller ever asks for it.
        for representative in range(min(int(mm), 2) + 1):
            build_Q(representative)


    _Q_cache: 'Dict[int, np.ndarray]' = {}

    def build_Q(m):
        category = int(m) if abs(int(m)) == 1 else (0 if m == 0 else 2)
        Q = _Q_cache.get(category)
        if Q is not None:
            return Q
        d_jn_nodes = {jn["d_node"] for jn in junctions}
        c_jn_nodes = {jn["c_node"] for jn in junctions}
        b_jn_nodes = {(jn["bare"][0], jn["bare"][1]) for jn in junctions}


        def surf_mask(solver, jn_nodes, is_m):
            """(t_active, phi_active) with axis rules; junction nodes stay
            active here (masters/M_phi) unless excluded below."""
            Nn = solver.Nn
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if solver.gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif end in jn_nodes:
                    if is_m:
                        t_act[end] = False
                else:
                    t_act[end] = False

            return t_act, f_act


        col = np.full(n_full, -1, dtype=int)
        red = 0

        def assign(offset, acts):
            nonlocal red
            t_act, f_act = acts
            Nn = len(t_act)
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        assign(off_Jd, surf_mask(sd_e, d_jn_nodes, False))
        assign(off_M, surf_mask(sd_e, d_jn_nodes, True))
        t2, f2 = surf_mask(s2_L, c_jn_nodes, False)
        for jn in junctions:
            t2[jn["c_node"]] = False; f2[jn["c_node"]] = False
        assign(off_J2, (t2, f2))
        b_acts = []
        for bi, b in enumerate(bares):
            tb, fb = surf_mask(b, {n for (p, n) in b_jn_nodes if p == bi}, False)
            for jn in junctions:
                if jn["bare"][0] == bi:
                    tb[jn["bare"][1]] = False; fb[jn["bare"][1]] = False
            b_acts.append((tb, fb))
            assign(off_J1[bi], (tb, fb))

        # Sparse in every backend: at most three entries per column.  The
        # dense form cost n_full * n_reduced complex values per cached
        # category (6.4 GB at 10k unknowns for four categories).
        Q = _ConstraintEntries(n_full, red)
        active = col >= 0
        Q.select(np.flatnonzero(active), col[active])
        _apply_regular_axis_rows(Q, col, off_Jd, sd_e, m)
        _apply_regular_axis_rows(Q, col, off_M, sd_e, m)
        _apply_regular_axis_rows(Q, col, off_J2, s2_L, m)
        for bi, bare in enumerate(bares):
            _apply_regular_axis_rows(Q, col, off_J1[bi], bare, m)


        for jn in junctions:
            dn = jn["d_node"]
            cn = jn["c_node"]
            bi, bn = jn["bare"]
            master_t = col[off_Jd + dn]
            master_f = col[off_Jd + Nd + dn]
            if master_t >= 0:
                Q[off_J2 + cn, master_t] = 1.0
                Q[off_J1[bi] + bn, master_t] = 1.0
            if master_f >= 0:
                Q[off_J2 + N2 + cn, master_f] = -1.0
                Q[off_J1[bi] + N1[bi] + bn, master_f] = 1.0
        Q = Q.tocsr()
        _Q_cache[category] = Q
        return Q

    def assemble(m):
        A = modal_matrix((n_full, n_full), compressed_requested())
        sl_Jd = slice(off_Jd, off_Jd + 2 * Nd)
        sl_M = slice(off_M, off_M + 2 * Nd)
        sl_J2 = slice(off_J2, off_J2 + 2 * N2)
        T_e = sd_e.assemble_mode(m, m_max)
        T_L = sd_L.assemble_mode(m, m_max)
        # The coating's equations enter the interface rows with the weight of
        # ``_region_equation_weights`` (1 is PMCHWT), jump terms included.
        P_sum = ETA0 * (sd_e.assemble_pmchwt_P(m, m_max) + weight * sd_L.assemble_pmchwt_P(m, m_max))
        if weight != 1.0:
            P_sum = _add_rotation_mass_into(P_sum, sd_e, -(0.5 * (1.0 - weight) * ETA0))
        iE = sl_Jd; iH = sl_M
        A[iE, sl_Jd] = T_e + weight * T_L
        A[iE, sl_M] = -P_sum
        A[iH, sl_Jd] = P_sum
        A[iH, sl_M] = T_e + (weight * eta_ratio2) * T_L
        del T_e, T_L, P_sum
        # Each surface pair is integrated once per mode; its reverse blocks
        # are derived from the forward ones (_cross_block).
        memo = {}
        A[iE, sl_J2] = -weight * _cross_block(X_d2, 'T', m, m_max, memo)
        A[iH, sl_J2] = (-weight * ETA0) * _cross_block(X_d2, 'P', m, m_max, memo)
        A[sl_J2, sl_Jd] = -_cross_block(X_2d, 'T', m, m_max, memo)
        A[sl_J2, sl_M] = ETA0 * _cross_block(X_2d, 'P', m, m_max, memo)
        A[sl_J2, sl_J2] = s2_L.assemble_mode(m, m_max)
        for bi, b in enumerate(bares):
            sl_b = slice(off_J1[bi], off_J1[bi] + 2 * N1[bi])
            T_d1 = _cross_block(X_d1[bi], 'T', m, m_max, memo)
            P_d1 = _cross_block(X_d1[bi], 'P', m, m_max, memo)
            A[iE, sl_b] = T_d1
            A[iH, sl_b] = ETA0 * P_d1
            if S_maps[bi] is not None:
                A[iE, sl_b] += -_impedance_rotated(P_d1, zs_nodes[bi], S_maps[bi])
                A[iH, sl_b] += (1.0 / ETA0) * _impedance_rotated(
                    T_d1, zs_nodes[bi], S_maps[bi]
                )
            del T_d1, P_d1
            A[sl_b, sl_Jd] = _cross_block(X_1d[bi], 'T', m, m_max, memo)
            A[sl_b, sl_M] = -ETA0 * _cross_block(X_1d[bi], 'P', m, m_max, memo)
            A[sl_b, sl_b] = b.assemble_mode(m, m_max)
            if zs_elems[bi] is not None:
                A[sl_b, sl_b] += b.assemble_ibc_extra(m, m_max, zs_ptss[bi],
                                                      zs_elems[bi])
            for bj in range(len(bares)):
                if bj != bi:
                    sl_bj = slice(off_J1[bj], off_J1[bj] + 2 * N1[bj])
                    A[sl_b, sl_bj] = _cross_block(X_11[(bi, bj)], 'T', m, m_max, memo)
                    if S_maps[bj] is not None:
                        A[sl_b, sl_bj] += -_impedance_rotated(
                            _cross_block(X_11[(bi, bj)], 'P', m, m_max, memo),
                            zs_nodes[bj], S_maps[bj],
                        )
        memo.clear()
        Q = build_Q(m)
        # The sweep projects excitations (Q^H b) and expands solutions (Q x)
        # itself, so rhs/farfield below work on the full unknown vector.
        return _reduce_constrained_operator(A, Q), Q

    def rhs(m, th, pol):
        V = np.zeros(n_full, dtype=np.complex128)
        V[off_Jd:off_Jd + 2 * Nd] = sd_e.rhs_mode(m, th, pol)
        V[off_M:off_M + 2 * Nd] = ETA0 * sd_e.rhs_h_mode(m, th, pol)
        for bi, b in enumerate(bares):
            V[off_J1[bi]:off_J1[bi] + 2 * N1[bi]] = b.rhs_mode(m, th, pol)
        return V

    def farfield(m, x, th, pol):
        fth, fph = sd_e.farfield_mode(m, x[off_Jd:off_Jd + 2 * Nd], th,
                                      msol=ETA0 * x[off_M:off_M + 2 * Nd])
        for bi, b in enumerate(bares):
            ft, fp = b.farfield_mode(m, x[off_J1[bi]:off_J1[bi] + 2 * N1[bi]],
                                     th, zs_pt=zs_ptss[bi])
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    def rhs_batch(m, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        electric = sd_e.rhs_vv_hh_batch(m, batch_thetas)
        out = np.zeros((n_full, electric.shape[1]), dtype=np.complex128)
        out[off_Jd:off_Jd + 2 * Nd] = electric
        # eta0 * H-field testing: VV column is -E(HH), HH column is E(VV).
        out[off_M:off_M + 2 * Nd, 0::2] = -electric[:, 1::2]
        out[off_M:off_M + 2 * Nd, 1::2] = electric[:, 0::2]
        for bi, b in enumerate(bares):
            out[off_J1[bi]:off_J1[bi] + 2 * N1[bi]] = b.rhs_vv_hh_batch(
                m, batch_thetas
            )
        return out

    def farfield_batch(m, full, batch_thetas, batch_pols):
        if tuple(batch_pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        out = sd_e.farfield_vv_hh_batch(
            m,
            full[off_Jd:off_Jd + 2 * Nd],
            batch_thetas,
            msolutions=ETA0 * full[off_M:off_M + 2 * Nd],
        )
        for bi, b in enumerate(bares):
            out = out + b.farfield_vv_hh_batch(
                m,
                full[off_J1[bi]:off_J1[bi] + 2 * N1[bi]],
                batch_thetas,
                zs_pt=zs_ptss[bi],
            )
        return out

    solver_requirements = (
        (sd_e, True, False, True),
        (sd_L, True, False, True),
        (s2_L, True, False, False),
        *(
            (bare, True, False, zs_elems[index] is not None)
            for index, bare in enumerate(bares)
        ),
    )
    all_crosses = (
        X_d2, X_2d, *X_d1, *X_1d, *X_11.values()
    )
    impedance_map_gb = sum(
        (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes if issparse(matrix) else matrix.nbytes) for matrix in S_maps if matrix is not None
    ) / 1.0e9
    plan = _plan_multisurface_assembly(
        m_max,
        solver_requirements,
        all_crosses,
        constraint_dofs=n_full,
        assembly=assembly,
        table_precision=table_precision,
        stream_budget_gb=stream_budget_gb,
        workers=workers,
        extra_retained_gb=impedance_map_gb,
    )

    stream_owners = (sd_e, sd_L, s2_L, *bares, *all_crosses)
    try:
        F, modes_used, stats = _mode_sweep(n_full, thetas, ("VV", "HH"), m_max,
                                           mode_tol, assemble, rhs, farfield,
                                           prepare=prepare, workers=plan["workers"], preparation_workers=workers,
                                           progress=progress,
                                           check_abort=check_abort,
                                           monitor_cond=True,
                                           rhs_batch=rhs_batch,
                                           farfield_batch=farfield_batch,
                                           min_mode_before_tail=mode_tail_start,
                                           assembly_peak_gb=plan["assembly_peak_gb"],
                                           memory_context="The partial-coating BoR solve",
                                           stream_mode_block=plan["mode_block"],
                                           signed_mode_symmetry=True, axial_mode_only=_all_axial_aspects(thetas),
                                           coordinates=lambda m: _constraint_coordinates(
                                               np.vstack((np.tile(sd_e.gen.nodes, (4, 1)),
                                                          np.tile(s2_L.gen.nodes, (2, 1)),
                                                          *(np.tile(b.gen.nodes, (2, 1)) for b in bares))), build_Q(m)),
                                           hierarchical_pricing=False)
        streams = {
            "interface_exterior": sd_e._stream,
            "interface_coating": sd_L._stream,
            "covered_core": s2_L._stream,
            **{f"bare_{index}": bare._stream for index, bare in enumerate(bares)},
            **{
                f"cross_{index}": cross._stream
                for index, cross in enumerate(all_crosses)
                if not getattr(cross, 'derived', False)
            },
        }
        stream_backends = {
            name: sampling_backend_name(stream)
            for name, stream in streams.items()
        } if plan["use_streaming"] else {}
        stream_compression = _stream_compression_evidence(streams)
        stream_sweeps = (
            sum(stream.n_sweeps for stream in streams.values())
            if plan["use_streaming"] else 0
        )
        stream_spill_gb = (
            sum(stream.spilled_gb() for stream in streams.values())
            if plan["use_streaming"] else 0.0
        )
    finally:
        for owner in stream_owners:
            owner.close_streaming()
    _require_mode_convergence(stats, mode_tol)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_full),
        "n_junctions": len(junctions),
        "formulation": "pmchwt-partial-coating",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "assembly": ("compressed" if compressed_requested() else
                     "streaming" if plan["use_streaming"] else "tables"),
        "table_precision": "single" if plan["use_single"] else "double",
        "stream_mode_block": plan["mode_block"],
        "stream_sweeps": stream_sweeps,
        "stream_sampling_backend": (
            next(iter(set(stream_backends.values())))
            if plan["use_streaming"]
            and len(set(stream_backends.values())) == 1
            else ("mixed" if plan["use_streaming"] else None)
        ),
        "stream_sampling_backends": stream_backends,
        "stream_spill_gb": stream_spill_gb,
        "stream_auxiliary_peak_gb": plan["auxiliary_peak_gb"],
        "stream_far_compression": stream_compression,
        "warnings": solve_warnings,
        **stats,
    }


class _MultiRegionBor:
    """surfaces: list of (points, is_conductor).  regions: list of dicts
    {"medium": None|(eps, mu), "bounds": [(surf_idx, sigma), ...],
    "exterior": bool}.  Interfaces bounding the exterior region must carry
    sigma = +1 there (the far field then sums their (J, M) directly)."""

    def __init__(self, surfaces, regions, freq_hz: 'float', gauss_order: 'int' = FAR_GAUSS_ORDER,
                 near_factor: 'float' = 2.0, near_order: 'int' = 12):
        surfaces = [
            (_validate_solve_bor_generatrix(points, "efie"), is_conductor)
            for points, is_conductor in surfaces
        ]
        from ghost_backend.bor.geometry import require_containment
        for region in regions:
            for outer, outer_sign in region['bounds']:
                for inner, inner_sign in region['bounds']:
                    if outer_sign != -1 or inner_sign != 1:
                        continue
                    po, pi = surfaces[outer][0], surfaces[inner][0]
                    if all(points[0, 0] == points[-1, 0] == 0 for points in (po, pi)):
                        require_containment(po, pi, 'BoR material region')
        for region in regions:
            medium = region.get("medium")
            if medium is not None:
                if not isinstance(medium, (tuple, list)) or len(medium) != 2:
                    raise ValueError(
                        "Each BoR region medium must be an (epsilon, mu) pair."
                    )
                _causal_medium(medium[0], medium[1])
        self.regions = regions
        self.n_surf = len(surfaces)
        self.is_cond = [bool(c) for (_, c) in surfaces]
        self.weight = _region_equation_weights(
            [region.get("medium") for region in regions],
            next(ri for ri, region in enumerate(regions) if region.get("exterior")))
        self.weighted = any(value != 1.0 for value in self.weight)
        self.adj: 'List[List[int]]' = [[] for _ in surfaces]
        self.sigma: 'Dict[Tuple[int, int], int]' = {}
        for ri, reg in enumerate(regions):
            for (si, sg) in reg["bounds"]:
                self.adj[si].append(ri)
                self.sigma[(ri, si)] = int(sg)
        self.ext_region = next(ri for ri, r in enumerate(regions) if r.get("exterior"))
        self.core_cfie = [bool(conductor and self.ext_region not in self.adj[si]
            and points[0, 0] == points[-1, 0] == 0
            and len(self.adj[si]) == 1 and self.sigma[self.adj[si][0], si] == 1)
            for si, (points, conductor) in enumerate(surfaces)]
        for (si, sg) in regions[self.ext_region]["bounds"]:
            if sg != +1:
                raise ValueError("Exterior-bounding surfaces must have sigma=+1.")

        self.solv: 'Dict[Tuple[int, int], BorPecSolver]' = {}
        for si, (pts, _) in enumerate(surfaces):
            for ri in self.adj[si]:
                self.solv[(si, ri)] = _prepared_surface(
                    pts, freq_hz, gauss_order=gauss_order,
                    medium=regions[ri]["medium"])

        self.X: 'Dict[Tuple[int, int, int], BorCrossOperators]' = {}
        for ri, reg in enumerate(regions):
            ids = [si for (si, _) in reg["bounds"]]
            for a, si in enumerate(ids):
                for sj in ids[a + 1:]:
                    if si == sj:
                        continue
                    # One integrated direction per surface pair; the reverse
                    # follows exactly (_ReverseCrossOperators).
                    forward = _prepared_cross(
                        self.solv[(si, ri)], self.solv[(sj, ri)],
                        near_factor=near_factor, near_order=near_order,
                        need_p=(not self.is_cond[si] or not self.is_cond[sj]
                                or self.core_cfie[si] or self.core_cfie[sj]))
                    self.X[(ri, si, sj)] = forward
                    self.X[(ri, sj, si)] = _reverse_cross(
                        forward, near_factor=near_factor, near_order=near_order)

        self.Nn = [self.solv[(si, self.adj[si][0])].Nn for si in range(self.n_surf)]
        self.off_J: 'List[int]' = []
        self.off_M: 'List[Optional[int]]' = []
        acc = 0
        for si in range(self.n_surf):
            self.off_J.append(acc)
            acc += 2 * self.Nn[si]
            if self.is_cond[si]:
                self.off_M.append(None)
            else:
                self.off_M.append(acc)
                acc += 2 * self.Nn[si]
        self.n_full = acc


        all_pts = np.vstack([self.solv[(si, self.adj[si][0])].gen.nodes
                             for si in range(self.n_surf)])
        diag = max(float(np.ptp(all_pts[:, 0])) + float(np.ptp(all_pts[:, 1])), 1e-9)
        jn_tol = 1e-8 * diag
        self.junctions: 'List[List[Tuple[int, int]]]' = []
        for si in range(self.n_surf):
            gen = self.solv[(si, self.adj[si][0])].gen
            for node in (0, gen.n_nodes - 1):
                if gen.node_on_axis(node):
                    continue
                pos = gen.nodes[node]
                for jn in self.junctions:
                    s0, n0 = jn[0]
                    p0 = self.solv[(s0, self.adj[s0][0])].gen.nodes[n0]
                    if float(np.hypot(*(p0 - pos))) <= jn_tol:
                        jn.append((si, node))
                        break
                else:
                    self.junctions.append([(si, node)])
        for jn in self.junctions:
            if len(jn) < 2:
                si, node = jn[0]
                raise ValueError(f"Surface {si} has an off-axis free endpoint "
                                 "that is not part of a junction.")


            mi = next((idx for idx, (si, _) in enumerate(jn)
                       if not self.is_cond[si]), 0)
            jn[0], jn[mi] = jn[mi], jn[0]
            mstr = jn[0]
            for (si, node) in jn[1:]:
                if not set(self.adj[si]) & set(self.adj[mstr[0]]):
                    raise ValueError("Junction surface shares no region with "
                                     "the junction master.")
        self._Q_cache: 'Dict[int, np.ndarray]' = {}


    def enable_streaming(self, m_max: 'int', plan: 'Dict[str, Any]') -> 'None':
        """Build per-mode nodal far blocks before operator assembly.

        Near/self quadrature uses the same kernels. IBC blocks include source Z_s, so
        assemble_ibc_extra must receive the same zs_pt. PMCHWT uses rotated-PV blocks
        with unit source weight.
        """

        common = dict(
            single_blocks=bool(plan["use_single"]),
            tile_budget_gb=float(plan["tile_budget_gb"]),
            workers=int(plan["workers"]),
            mode_block=int(plan["mode_block"]),
            spill=plan.get("spill"),
        )
        for (surface_index, _region_index), solver in self.solv.items():
            solver.enable_streaming(
                m_max,
                efie=True,
                mfie=self.core_cfie[surface_index],
                pmchwt=not self.is_cond[surface_index],
                **common,
            )
        for cross in self.X.values():
            cross.enable_streaming(m_max, **common)

    def prepare(self, m_max: 'int', workers: 'int' = 1) -> 'None':
        for (si, ri), s in self.solv.items():
            s.prepare_operators(m_max, efie=True, mfie=self.core_cfie[si], ibc=not self.is_cond[si],
                                workers=workers)
        for X in self.X.values():
            X.prepare(m_max, workers=workers)


        # Signed-mode symmetry assembles only m >= 0 (see the partial coating).
        for representative in range(min(int(m_max), 2) + 1):
            self.build_Q(representative)

    def _dir(self, si: 'int', node: 'int') -> 'int':
        """+1 if the drawn tangent points INTO the junction node (chain end)."""
        return +1 if node != 0 else -1

    def build_Q(self, m: 'int') -> 'np.ndarray':
        category = int(m) if abs(int(m)) == 1 else (0 if m == 0 else 2)
        Q = self._Q_cache.get(category)
        if Q is not None:
            return Q
        jn_nodes = {(si, node) for jn in self.junctions for (si, node) in jn}
        slave = {(si, node) for jn in self.junctions for (si, node) in jn[1:]}


        m_t_masked = {(si, node) for jn in self.junctions
                      if any(self.is_cond[sj] for (sj, _) in jn)
                      for (si, node) in jn if not self.is_cond[si]}

        col = np.full(self.n_full, -1, dtype=int)
        red = 0

        def assign(offset, si, is_m):
            nonlocal red
            Nn = self.Nn[si]
            gen = self.solv[(si, self.adj[si][0])].gen
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif (si, end) in slave:
                    t_act[end] = False
                    f_act[end] = False
                elif (si, end) in jn_nodes:
                    if is_m and (si, end) in m_t_masked:
                        t_act[end] = False
                else:
                    t_act[end] = False
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        for si in range(self.n_surf):
            assign(self.off_J[si], si, False)
            if self.off_M[si] is not None:
                assign(self.off_M[si], si, True)

        # Sparse in every backend (see the partial-coating build_Q).
        Q = _ConstraintEntries(self.n_full, red)
        active = col >= 0
        Q.select(np.flatnonzero(active), col[active])
        for si in range(self.n_surf):
            solver = self.solv[(si, self.adj[si][0])]
            _apply_regular_axis_rows(Q, col, self.off_J[si], solver, m)
            if self.off_M[si] is not None:
                _apply_regular_axis_rows(
                    Q, col, self.off_M[si], solver, m
                )


        for jn in self.junctions:
            sm, nm = jn[0]
            dir_m = self._dir(sm, nm)
            for (ss, ns) in jn[1:]:
                r = next(iter(set(self.adj[ss]) & set(self.adj[sm])))
                sg_m, sg_s = self.sigma[(r, sm)], self.sigma[(r, ss)]
                ct = -(sg_m * dir_m) / (sg_s * self._dir(ss, ns))
                cf = sg_m / sg_s
                mt = col[self.off_J[sm] + nm]
                mf = col[self.off_J[sm] + self.Nn[sm] + nm]
                if mt >= 0:
                    Q[self.off_J[ss] + ns, mt] = ct
                if mf >= 0:
                    Q[self.off_J[ss] + self.Nn[ss] + ns, mf] = cf
                if self.off_M[sm] is not None and self.off_M[ss] is not None:
                    mmt = col[self.off_M[sm] + nm]
                    mmf = col[self.off_M[sm] + self.Nn[sm] + nm]
                    if mmt >= 0:
                        Q[self.off_M[ss] + ns, mmt] = ct
                    if mmf >= 0:
                        Q[self.off_M[ss] + self.Nn[ss] + ns, mmf] = cf
        Q = Q.tocsr()
        self._Q_cache[category] = Q
        return Q

    def assemble(self, m: 'int', m_max: 'int'):
        A = modal_matrix((self.n_full, self.n_full), compressed_requested())
        memo = {}
        for ri, reg in enumerate(self.regions):
            eta_r = self.solv[(reg["bounds"][0][0], ri)].eta
            eta2 = (ETA0 / eta_r) ** 2
            for (si, sg_i) in reg["bounds"]:
                # Rows of a material interface add the equations of its two
                # regions with their weights (all 1 is PMCHWT); a conductor's
                # rows belong to one region and need none.
                row_weight = 1.0 if self.is_cond[si] else self.weight[ri]
                if self.weighted and not self.is_cond[si]:
                    # Jump terms that only equal weights cancel: region R gives
                    # +sigma/2 R M in the electric row, -sigma/2 R J in the magnetic one.
                    scale = 0.5 * sg_i * row_weight * ETA0
                    own_J = slice(self.off_J[si], self.off_J[si] + 2 * self.Nn[si])
                    own_M = slice(self.off_M[si], self.off_M[si] + 2 * self.Nn[si])
                    if isinstance(A, np.ndarray):
                        _add_rotation_mass_into(A[own_J, own_M], self.solv[(si, ri)], scale)
                        _add_rotation_mass_into(A[own_M, own_J], self.solv[(si, ri)], -scale)
                    else:
                        jump = scale * _rotation_mass(self.solv[(si, ri)])
                        A[own_J, own_M] += jump
                        A[own_M, own_J] += -1.0 * jump
                for (sj, sg_j) in reg["bounds"]:
                    ss = sg_i * sg_j * row_weight
                    if si == sj:
                        T = self.solv[(si, ri)].assemble_mode(m, m_max)
                        P = self.solv[(si, ri)].assemble_pmchwt_P(m, m_max) \
                            if not self.is_cond[si] else None
                    else:
                        cross = self.X[(ri, si, sj)]
                        T = _cross_block(cross, 'T', m, m_max, memo)
                        P = _cross_block(cross, 'P', m, m_max, memo) if cross.need_p else None
                    slJ_i = slice(self.off_J[si], self.off_J[si] + 2 * self.Nn[si])
                    slJ_j = slice(self.off_J[sj], self.off_J[sj] + 2 * self.Nn[sj])
                    A[slJ_i, slJ_j] += ss * T
                    if self.core_cfie[si]:
                        magnetic = (self.solv[(si, ri)].assemble_mfie_mode(m, m_max)
                            if si == sj else ss * _rotate_test_rows(P, self.Nn[si]))
                        A[slJ_i, slJ_j] += eta_r * magnetic
                    if self.off_M[sj] is not None:
                        slM_j = slice(self.off_M[sj], self.off_M[sj] + 2 * self.Nn[sj])
                        A[slJ_i, slM_j] += -ETA0 * ss * P
                        if self.core_cfie[si]:
                            A[slJ_i, slM_j] += (ETA0/eta_r)*ss*_rotate_test_rows(T, self.Nn[si])
                    if self.off_M[si] is not None:
                        slM_i = slice(self.off_M[si], self.off_M[si] + 2 * self.Nn[si])
                        A[slM_i, slJ_j] += ETA0 * ss * P
                        if self.off_M[sj] is not None:
                            A[slM_i, slM_j] += eta2 * ss * T
        memo.clear()
        Q = self.build_Q(m)
        # The sweep projects excitations (Q^H b) and expands solutions (Q x)
        # itself, so rhs/farfield work on the full unknown vector.
        return _reduce_constrained_operator(A, Q), Q

    def rhs(self, m: 'int', th: 'float', pol: 'str') -> 'np.ndarray':
        V = np.zeros(self.n_full, dtype=np.complex128)
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            V[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]] = s.rhs_mode(m, th, pol)
            if self.off_M[si] is not None:
                V[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]] = \
                    ETA0 * s.rhs_h_mode(m, th, pol)
        return V

    def farfield(self, m: 'int', x: 'np.ndarray', th: 'float', pol: 'str') -> 'complex':
        fth = fph = 0.0
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            J = x[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]]
            msol = (ETA0 * x[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]]
                    if self.off_M[si] is not None else None)
            ft, fp = s.farfield_mode(m, J, th, msol=msol)
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    def rhs_batch(self, m: 'int', thetas, pols) -> 'np.ndarray':
        if tuple(pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        out = None
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            electric = s.rhs_vv_hh_batch(m, thetas)
            if out is None:
                out = np.zeros((self.n_full, electric.shape[1]), dtype=np.complex128)
            out[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]] = electric
            if self.off_M[si] is not None:
                # eta0 * H-field testing: VV column is -E(HH), HH is E(VV).
                magnetic = out[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]]
                magnetic[:, 0::2] = -electric[:, 1::2]
                magnetic[:, 1::2] = electric[:, 0::2]
        return out

    def farfield_batch(self, m: 'int', x: 'np.ndarray', thetas, pols) -> 'np.ndarray':
        if tuple(pols) != ("VV", "HH"):
            raise ValueError("BoR optimized batch path requires VV/HH ordering.")
        out = np.zeros((2, len(thetas)), dtype=np.complex128)
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            J = x[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]]
            msol = (ETA0 * x[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]]
                    if self.off_M[si] is not None else None)
            out += s.farfield_vv_hh_batch(m, J, thetas, msolutions=msol)
        return out

    def rho_max(self) -> 'float':
        return max(float(np.max(self.solv[(si, self.adj[si][0])].gen.nodes[:, 0]))
                   for si in range(self.n_surf))


def _solve_multiregion(sys_: '_MultiRegionBor', freq_hz, thetas_deg, n_modes,
                       mode_tol, workers, progress, check_abort,
                       formulation: 'str', extra: 'Dict',
                       table_precision: 'str' = "auto",
                       assembly: 'str' = "auto",
                       stream_budget_gb: 'float' = 8.0) -> 'Dict':
    thetas = _validated_bor_aspects(thetas_deg)
    k = 2.0 * math.pi * freq_hz / C0
    m_max, mode_tail_start = _bor_mode_limits(
        k, sys_.rho_max(), thetas, n_modes
    )
    solver_requirements = tuple(
        (solver, True, sys_.core_cfie[surface_index], not sys_.is_cond[surface_index])
        for (surface_index, _region_index), solver in sys_.solv.items()
    )
    plan = _plan_multisurface_assembly(
        m_max,
        solver_requirements,
        tuple(sys_.X.values()),
        constraint_dofs=sys_.n_full,
        assembly=assembly,
        table_precision=table_precision,
        stream_budget_gb=stream_budget_gb,
        workers=workers,
    )

    def prepare(mm):
        if plan["use_streaming"] and all(
            solver._stream is None for solver in sys_.solv.values()
        ):
            sys_.enable_streaming(mm, plan)
        sys_.prepare(mm, workers=workers)

    try:
        F, modes_used, stats = _mode_sweep(
            sys_.n_full, thetas, ("VV", "HH"), m_max, mode_tol,
            lambda m: sys_.assemble(m, m_max), sys_.rhs, sys_.farfield,
            prepare=prepare,
            workers=plan["workers"], preparation_workers=workers, progress=progress, check_abort=check_abort,
            monitor_cond=True, rhs_batch=sys_.rhs_batch,
            farfield_batch=sys_.farfield_batch,
            min_mode_before_tail=mode_tail_start,
            assembly_peak_gb=plan["assembly_peak_gb"],
            memory_context=f"The {formulation} BoR solve",
            stream_mode_block=plan["mode_block"],
            signed_mode_symmetry=True, axial_mode_only=_all_axial_aspects(thetas),
            coordinates=lambda m: _constraint_coordinates(np.vstack([
                np.tile(sys_.solv[(si, sys_.adj[si][0])].gen.nodes, (2 if sys_.is_cond[si] else 4, 1))
                for si in range(sys_.n_surf)]), sys_.build_Q(m)),
            hierarchical_pricing=False)
        streams = {
            **{
                f"surface_{surface_index}_region_{region_index}": solver._stream
                for (surface_index, region_index), solver in sys_.solv.items()
            },
            **{
                f"cross_region_{region_index}_{test_index}_{source_index}": cross._stream
                for (region_index, test_index, source_index), cross in sys_.X.items()
                if not getattr(cross, 'derived', False)
            },
        }
        stream_backends = {
            name: sampling_backend_name(stream)
            for name, stream in streams.items()
        } if plan["use_streaming"] else {}
        stream_compression = _stream_compression_evidence(streams)
        stream_sweeps = (
            sum(stream.n_sweeps for stream in streams.values())
            if plan["use_streaming"] else 0
        )
        stream_spill_gb = (
            sum(stream.spilled_gb() for stream in streams.values())
            if plan["use_streaming"] else 0.0
        )
    finally:
        # Streamed far blocks can hold GBs (or spill files); release them on
        # success, abort, and error alike rather than at cyclic GC.
        for owner in (*sys_.solv.values(), *sys_.X.values()):
            owner.close_streaming()
    _require_mode_convergence(stats, mode_tol)
    extra = {**extra, **stats}
    warnings = list(extra.get("warnings", []) or [])
    extra["warnings"] = warnings
    unique_backends = set(stream_backends.values())
    out = {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(sys_.n_full),
        "n_junctions": len(sys_.junctions),
        "formulation": formulation,
        "combined_core_surfaces": [i for i, active in enumerate(sys_.core_cfie) if active],
        "assembly": ("compressed" if compressed_requested() else
                     "streaming" if plan["use_streaming"] else "tables"),
        "table_precision": "single" if plan["use_single"] else "double",
        "stream_mode_block": plan["mode_block"],
        "stream_sweeps": stream_sweeps,
        "stream_sampling_backend": (
            next(iter(unique_backends)) if len(unique_backends) == 1
            else ("mixed" if unique_backends else None)
        ),
        "stream_sampling_backends": stream_backends,
        "stream_spill_gb": stream_spill_gb,
        "stream_auxiliary_peak_gb": plan["auxiliary_peak_gb"],
        "stream_far_compression": stream_compression,
    }
    out.update(extra)
    return out


@profiled_solve
@configured
def solve_bor_coated2_pec(points_outer, points_mid, points_core,
                          freq_hz: 'float', thetas_deg,
                          eps_inner: 'complex', mu_inner: 'complex',
                          eps_outer: 'complex', mu_outer: 'complex',
                          n_modes: 'Optional[int]' = None, gauss_order: 'int' = FAR_GAUSS_ORDER,
                          mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                          near_order: 'int' = 12, workers: 'int' = 1,
                          progress: 'Optional[Callable]' = None,
                          check_abort: 'Optional[Callable]' = None,
                          table_precision: 'str' = "auto",
                          assembly: 'str' = "auto",
                          stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """PEC core under TWO full coating layers (all three generatrices closed
    axis-to-axis, +z -> -z, normals toward the exterior side)."""

    points_outer = _validate_solve_bor_generatrix(points_outer, "cfie")
    points_mid = _validate_solve_bor_generatrix(points_mid, "cfie")
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    sys_ = _MultiRegionBor(
        surfaces=[(points_outer, False), (points_mid, False), (points_core, True)],
        regions=[
            {"medium": None, "bounds": [(0, +1)], "exterior": True},
            {"medium": (eps_outer, mu_outer), "bounds": [(0, -1), (1, +1)]},
            {"medium": (eps_inner, mu_inner), "bounds": [(1, -1), (2, +1)]},
        ],
        freq_hz=freq_hz, gauss_order=gauss_order,
        near_factor=near_factor, near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coated-2layer",
                              {"eps_inner": complex(eps_inner),
                               "eps_outer": complex(eps_outer)},
                              table_precision, assembly, stream_budget_gb)


@profiled_solve
@configured
def solve_bor_banded_multiregion(surfaces, regions, freq_hz: 'float', thetas_deg,
                                 n_modes: 'Optional[int]' = None,
                                 gauss_order: 'int' = FAR_GAUSS_ORDER, mode_tol: 'float' = 1e-6,
                                 near_factor: 'float' = 2.0, near_order: 'int' = 12,
                                 workers: 'int' = 1,
                                 progress: 'Optional[Callable]' = None,
                                 check_abort: 'Optional[Callable]' = None,
                                 table_precision: 'str' = "auto",
                                 assembly: 'str' = "auto",
                                 stream_budget_gb: 'float' = 8.0,
                                 formulation: 'str' = "pmchwt-banded",
                                 extra: 'Optional[Dict]' = None) -> 'Dict':
    """Configured entry for a general multi-region layout (banded coatings).

    ``surfaces``/``regions`` are the ``_MultiRegionBor`` description.  Going
    through ``@configured`` gives these solves the same automatic mode-cap
    extension, admission fallbacks and option scope as every other BoR entry
    (the dispatcher used to call ``_solve_multiregion`` directly, so a
    starting cap that was too small raised instead of extending).
    """
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort, formulation,
                              dict(extra or {}), table_precision, assembly,
                              stream_budget_gb)


@profiled_solve
@configured
def solve_bor_coated_n_pec(interface_points, points_core, freq_hz: 'float',
                           thetas_deg, eps_list, mu_list,
                           n_modes: 'Optional[int]' = None, gauss_order: 'int' = FAR_GAUSS_ORDER,
                           mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                           near_order: 'int' = 12, workers: 'int' = 1,
                           progress: 'Optional[Callable]' = None,
                           check_abort: 'Optional[Callable]' = None,
                           table_precision: 'str' = "auto",
                           assembly: 'str' = "auto",
                           stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """PEC core under N full coating layers.  interface_points is the list
    of interface generatrices OUTERMOST FIRST; eps_list/mu_list are per
    layer INNERMOST FIRST (matching mie_sphere.sigma_multilayer_pec_sphere)."""

    N = len(interface_points)
    if N < 1:
        raise ValueError("At least one coating interface is required.")
    if len(eps_list) != N or len(mu_list) != N:
        raise ValueError("One (eps, mu) per layer, innermost first.")
    interface_points = [
        _validate_solve_bor_generatrix(points, "cfie")
        for points in interface_points
    ]
    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    surfaces = [(p, False) for p in interface_points] + [(points_core, True)]
    regions = [{"medium": None, "bounds": [(0, +1)], "exterior": True}]
    for i in range(N):


        lay = N - i
        inner_bound = (i + 1, +1) if i == N - 1 else (i + 1, +1)
        regions.append({"medium": (eps_list[lay - 1], mu_list[lay - 1]),
                        "bounds": [(i, -1), inner_bound]})
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              f"pmchwt-coated-{N}layer",
                              {"eps_layers": [complex(e) for e in eps_list]},
                              table_precision, assembly, stream_budget_gb)


@profiled_solve
@configured
def solve_bor_coating_patch(points_patch, points_mid_covered, points_mid_bare,
                            points_core, freq_hz: 'float', thetas_deg,
                            eps_inner: 'complex', mu_inner: 'complex',
                            eps_patch: 'complex', mu_patch: 'complex',
                            n_modes: 'Optional[int]' = None, gauss_order: 'int' = FAR_GAUSS_ORDER,
                            mode_tol: 'float' = 1e-6, near_factor: 'float' = 2.0,
                            near_order: 'int' = 12, workers: 'int' = 1,
                            progress: 'Optional[Callable]' = None,
                            check_abort: 'Optional[Callable]' = None,
                            table_precision: 'str' = "auto",
                            assembly: 'str' = "auto",
                            stream_budget_gb: 'float' = 8.0) -> 'Dict':
    """A second-layer coating PATCH terminating on a fully coated PEC body:
    the patch's outer interface (points_patch) meets the inner coating's
    interface at dielectric triple junctions (air / patch / inner coating --
    no conductor on the junction line).  points_mid_covered is the part of
    the inner interface under the patch, points_mid_bare the exposed
    part(s) (a list); the PEC core stays fully covered by the inner layer."""

    points_core = _validate_solve_bor_generatrix(points_core, "cfie")
    bare_list = (points_mid_bare if isinstance(points_mid_bare, (list, tuple))
                 else [points_mid_bare])
    surfaces = [(points_patch, False), (points_mid_covered, False)]
    surfaces += [(p, False) for p in bare_list]
    surfaces.append((points_core, True))
    core_idx = len(surfaces) - 1
    bare_idx = list(range(2, 2 + len(bare_list)))
    regions = [
        {"medium": None, "exterior": True,
         "bounds": [(0, +1)] + [(bi, +1) for bi in bare_idx]},
        {"medium": (eps_patch, mu_patch),
         "bounds": [(0, -1), (1, +1)]},
        {"medium": (eps_inner, mu_inner),
         "bounds": [(1, -1)] + [(bi, -1) for bi in bare_idx] + [(core_idx, +1)]},
    ]
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coating-patch",
                              {"eps_inner": complex(eps_inner),
                               "eps_patch": complex(eps_patch)},
                              table_precision, assembly, stream_budget_gb)


def sphere_generatrix(a: 'float', n: 'int') -> 'np.ndarray':
    """North pole (+z) to south pole: outward left-normals per convention."""
    th = np.linspace(0.0, math.pi, n + 1)
    return np.column_stack([a * np.sin(th), a * np.cos(th)])


def cylinder_generatrix(a: 'float', L: 'float', n_rad: 'int', n_len: 'int') -> 'np.ndarray':
    """Closed cylinder: top cap center -> rim -> side -> bottom rim -> center."""
    top = np.column_stack([np.linspace(0.0, a, n_rad + 1), np.full(n_rad + 1, L / 2)])
    side = np.column_stack([np.full(n_len - 1, a), np.linspace(L / 2, -L / 2, n_len + 1)[1:-1]])
    bot = np.column_stack([np.linspace(a, 0.0, n_rad + 1), np.full(n_rad + 1, -L / 2)])
    return np.vstack([top, side, bot])
