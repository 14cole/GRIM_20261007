"""Modal quadrature and boundary kernels for body-of-revolution geometry."""

import math
import os
import sys
import threading
import weakref
from collections import OrderedDict
from ghost_backend.execution.runtime import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.special import roots_legendre

NEAR_KERNEL_WORK_BYTES = 64_000_000
NEAR_ANGULAR_MAX_ORDER = 4096
NEAR_ANGULAR_RTOL = 2.0e-8
# The coarse level of the graded near rules is evaluated on every
# NEAR_CHECK_STRIDE-th point of a layout chunk (the first included) and the
# chunk is accepted at the fine level when every probed point passes; any
# probed failure falls back to the complete check of every point.  The fine
# level is the published value either way, so a chunk whose probes pass
# yields the complete check's values bitwise; on the bodies measured (PEC,
# dielectric and coated spheres, cylinders) no chunk ever fell back and the
# coarse level was 40% of all angular samples.  GHOST_BOR_NEAR_CHECK_STRIDE=0
# restores the complete check (October 2026).
NEAR_CHECK_STRIDE = 4

C0 = 299_792_458.0
ETA0 = 376.730313668
AXIS_TOL = 1e-12

# Gauss-Legendre points per generatrix element for far pairs, excitation and
# far-field projection (near pairs use their own converged rules).  Far pairs
# are at least two element lengths apart at 20 elements per wavelength, where
# three points already resolve the integrands: against four, amplitudes moved
# 2e-9 to 2e-8 relative on PEC (CFIE, EFIE), impedance, dielectric and coated
# spheres (errors against Mie unchanged to 1e-4 dB) and 2e-7 on the 4 GHz
# ogive, whose discretization error is 9e-3; the ogive solve took 19 % less time.
FAR_GAUSS_ORDER = 3


@lru_cache(maxsize=1024)
def cached_leggauss(order: 'int') -> 'Tuple[np.ndarray, np.ndarray]':
    """Immutable Gauss-Legendre rule shared by every BoR kernel build.

    Near-pair preparation requests the same small set of orders hundreds of
    times.  ``leggauss`` constructs those rules through an eigensolve, so
    rebuilding them for every element pair is pure overhead.  The returned
    arrays are read-only to keep the process-wide cache safe.  The near and
    far rules request data-dependent orders (phase-sized panels), so the cache
    holds many of them; a rule of order n costs 16 n bytes.
    """

    x, w = roots_legendre(int(order))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


@dataclass
class Generatrix:
    """Polyline generatrix in the (rho, z) half-plane, rho >= 0.

    Convention (BOR_CONVENTIONS.md): traversed so the left-of-travel normal
    (-z', rho') points into the exterior (air).  For a closed body that
    means from the +z axis end to the -z axis end (sphere: north pole ->
    south pole).
    """

    nodes: 'np.ndarray'
    elem_n0: 'np.ndarray' = field(init=False)
    elem_n1: 'np.ndarray' = field(init=False)
    lengths: 'np.ndarray' = field(init=False)
    trho: 'np.ndarray' = field(init=False)
    tz: 'np.ndarray' = field(init=False)

    def __post_init__(self):
        pts = np.asarray(self.nodes, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError("Generatrix needs an (Nn, 2) array of (rho, z) nodes.")
        if np.any(pts[:, 0] < -1e-12):
            raise ValueError("Generatrix rho coordinates must be >= 0.")
        pts[:, 0] = np.maximum(pts[:, 0], 0.0)
        self.nodes = pts
        d = pts[1:] - pts[:-1]
        self.lengths = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.lengths <= 0):
            raise ValueError("Generatrix has a zero-length element.")
        self.trho = d[:, 0] / self.lengths
        self.tz = d[:, 1] / self.lengths
        self.elem_n0 = np.arange(len(self.lengths))
        self.elem_n1 = self.elem_n0 + 1

    @property
    def n_elems(self) -> 'int':
        return len(self.lengths)

    @property
    def n_nodes(self) -> 'int':
        return len(self.nodes)

    def node_on_axis(self, i: 'int') -> 'bool':
        return self.nodes[i, 0] <= AXIS_TOL * max(1.0, float(np.max(self.nodes[:, 0])))


@dataclass
class GaussData:
    """Per-Gauss-point geometry over the whole generatrix."""

    elem: 'np.ndarray'
    s: 'np.ndarray'
    w: 'np.ndarray'
    rho: 'np.ndarray'
    z: 'np.ndarray'
    trho: 'np.ndarray'
    tz: 'np.ndarray'


    T0: 'np.ndarray'
    T1: 'np.ndarray'
    dRT0: 'np.ndarray'
    dRT1: 'np.ndarray'


def gauss_on_generatrix(gen: 'Generatrix', order: 'int' = FAR_GAUSS_ORDER) -> 'GaussData':
    xg, wg = cached_leggauss(order)
    s = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    ne = gen.n_elems
    E = np.repeat(np.arange(ne), order)
    S = np.tile(s, ne)
    W = np.tile(w, ne) * np.repeat(gen.lengths, order)
    r0 = gen.nodes[gen.elem_n0]
    r1 = gen.nodes[gen.elem_n1]
    RHO = np.repeat(r0[:, 0], order) + S * np.repeat(r1[:, 0] - r0[:, 0], order)
    Z = np.repeat(r0[:, 1], order) + S * np.repeat(r1[:, 1] - r0[:, 1], order)
    TR = np.repeat(gen.trho, order)
    TZ = np.repeat(gen.tz, order)
    L = np.repeat(gen.lengths, order)
    T0 = 1.0 - S
    T1 = S


    drho_ds = np.repeat(r1[:, 0] - r0[:, 0], order)
    dRT0 = (drho_ds * (1.0 - S) - RHO) / L
    dRT1 = (drho_ds * S + RHO) / L
    return GaussData(E, S, W, RHO, Z, TR, TZ, T0, T1, dRT0, dRT1)


FFT_BUILD_BUDGET = 256e6
N_XI_SAFETY_CAP = 8192
BANDED_FFT = True


# --------------------------------------------------------------------------
# CPU allocation for native OpenMP teams
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _host_physical_cores():
    """Physical cores of the host (static), or None when psutil cannot tell."""
    try:
        import psutil
        count = psutil.cpu_count(logical=False)
    except Exception:
        count = None
    return int(count) if count else None


def _affinity_cpu_count():
    """Logical CPUs this process may run on (its affinity mask), or None."""
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        pass
    try:
        import psutil
        return len(psutil.Process().cpu_affinity()) or None
    except Exception:
        return None


def _slurm_cpu_count():
    """SLURM's per-task (else per-node) CPU allocation, or None outside SLURM."""
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name, "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return None


def _scheduler_cpu_budget():
    """The solve's scheduler CPU reservation (execution options), or None.

    ``allocated_cpu_budget`` returns the host CPU count when no allocation is
    active, which never lowers the minimum below.
    """
    try:
        from ghost_backend.execution.options import allocated_cpu_budget
        return int(allocated_cpu_budget())
    except Exception:
        return None


def physical_cpu_count() -> 'int':
    """Cores the native OpenMP teams of this process may use.

    Memory-bound sampling gains nothing from SMT, so the base is the host's
    physical core count; it is then bounded by everything that allocates CPUs
    to this process: its affinity mask, a SLURM allocation
    (``SLURM_CPUS_PER_TASK``, else ``SLURM_CPUS_ON_NODE``), and the scheduler
    reservation of the active solve (``execution.options.allocated_cpu_budget``,
    e.g. one of several unit processes of a local batch).  The native kernels
    size their teams with ``num_threads(team)``, which overrides
    ``OMP_NUM_THREADS``, so an unbounded count here oversubscribes the host.
    Always at least 1.
    """
    physical = _host_physical_cores()
    counts = [physical if physical else (os.cpu_count() or 1)]
    for value in (_affinity_cpu_count(), _slurm_cpu_count(), _scheduler_cpu_budget()):
        if value:
            counts.append(int(value))
    return max(1, min(counts))


# --------------------------------------------------------------------------
# Native library access
# --------------------------------------------------------------------------

_NATIVE_LOCK = threading.Lock()
_NOTICE_SHOWN = set()


def _native_library():
    from ghost_backend.bor.streaming import _NATIVE
    return _NATIVE


def _native_entry(name):
    """A native entry point added after the streaming loader's list, or None.

    ``streaming._load_native`` declares the argument types of the entry
    points it knows; the newer near-rule kernels are declared here, once, on
    first use.  A library built before an entry point existed lacks the
    symbol and the caller falls back.
    """
    library = _native_library()
    if library is None or not hasattr(library, name):
        return None
    function = getattr(library, name)
    if getattr(function, 'argtypes', None) is None:
        import ctypes
        ci, cd, vp = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
        rule_tail = [vp, cd, cd, cd, ci, vp, vp, ci, vp, vp, ci]
        signatures = {
            'parity_moments': [ci, ci, ci, ci, ci, vp, vp, vp, vp, vp, ci],
            'near_green': [ci, ci, vp, vp, vp, vp, cd, cd, vp, ci, vp],
            'near_brackets_stable': [ci, ci, ci] + [vp] * 8 + [cd, cd, vp, ci] + [vp] * 4,
            'near_green_rule': [ci] + [vp] * 4 + rule_tail + [vp, ci],
            'near_brackets_rule': [ci, ci, ci] + [vp] * 8 + rule_tail + [vp, vp, ci],
        }
        if name not in signatures:
            return None
        with _NATIVE_LOCK:
            if getattr(function, 'argtypes', None) is None:
                function.restype = None
                function.argtypes = signatures[name]
    return function


def _notice_native_fallback(missing):
    """One-time stderr notice that a table/near path runs on NumPy.

    Without any native library this is the streaming path's own notice; with
    a library built before ``missing`` existed, the same advice to rebuild.
    """
    key = tuple(missing)
    if key in _NOTICE_SHOWN:
        return
    _NOTICE_SHOWN.add(key)
    if _native_library() is None:
        from ghost_backend.bor import streaming
        streaming._notice_numpy_fallback()
        return
    print(
        "bor.kernels: the native BoR kernel library predates "
        + ", ".join(missing)
        + "; this path uses the NumPy fallback (same results, slower). "
        "Rebuild and load-check it on THIS machine with:\n"
        "  py ghost_backend/bor/native/build_kernel.py",
        file=sys.stderr, flush=True)


def _native_pair_sampler(kind):
    """Native OpenMP sampler of one pair list on a shared half grid, or None.

    ``sample_g_pairs`` and ``sample_brackets_pairs`` accept real and complex
    wavenumbers; a library built before them leaves the NumPy forms in use.
    """
    from ghost_backend.bor.streaming import _NATIVE, _dp
    library = _NATIVE
    name = 'sample_g_pairs' if kind == 'g' else 'sample_brackets_pairs'
    if library is None or not hasattr(library, name):
        return None
    function = getattr(library, name)
    if getattr(function, 'argtypes', None) is None:
        return None
    if kind == 'g':
        def sample(pair, wavenumber, sin2, threads):
            rp, zp, rq, zq = (np.ascontiguousarray(value, dtype=float) for value in pair)
            count = rp.size
            if count == 0:
                return None
            out = np.empty((count, sin2.size), dtype=np.complex128)
            function(count, sin2.size, _dp(rp), _dp(zp), _dp(rq), _dp(zq),
                     float(wavenumber.real), float(wavenumber.imag), _dp(sin2), _dp(out), int(threads))
            return out
        return sample
    family = 0 if kind == 'mfie' else 1

    def sample(pair, wavenumber, cx, sx, threads):
        arrays = [np.ascontiguousarray(value, dtype=float) for value in pair]
        count = arrays[0].size
        if count == 0:
            return None
        outs = [np.empty((count, cx.size), dtype=np.complex128) for _ in range(4)]
        function(family, count, cx.size, *[_dp(value) for value in arrays],
                 float(wavenumber.real), float(wavenumber.imag), _dp(cx), _dp(sx),
                 *[_dp(value) for value in outs], int(threads))
        return tuple(outs)
    return sample


def _native_trig_moments(rows, xi, count, want_sin):
    """``(cosine, sine)`` moment arrays ``[n, rows, count]`` from the native
    recurrence kernel, or None when the library lacks it.

    ``rows`` is ``[n, r, na]`` real, ``xi`` ``[n, na]``; ``sine`` is None
    unless ``want_sin``.  Near preparation runs inside its own worker pool,
    so the kernel is single-threaded here.
    """
    library = _native_library()
    if library is None or not hasattr(library, 'trig_moments') or getattr(library.trig_moments, 'argtypes', None) is None:
        return None
    from ghost_backend.bor.streaming import _dp
    rows = np.ascontiguousarray(rows, dtype=float)
    xi = np.ascontiguousarray(xi, dtype=float)
    if rows.ndim != 3 or xi.ndim != 2 or rows.shape[0] != xi.shape[0] or rows.shape[2] != xi.shape[1]:
        return None
    n, r, na = rows.shape
    count = int(count)
    if n == 0 or count <= 0 or na == 0:
        return None
    cosine = np.empty((n, r, count), dtype=float)
    sine = np.empty((n, r, count), dtype=float) if want_sin else None
    library.trig_moments(n, r, na, count, _dp(rows), _dp(xi), _dp(cosine),
                         None if sine is None else sine.ctypes.data, 1)
    return cosine, sine


def _numpy_trig_moments(rows, xi, count, want_cos=True, want_sin=False, start_order=0):
    """NumPy form of the moment kernels (trigonometric tables in blocks)."""
    n, r, _ = rows.shape
    orders = np.arange(int(start_order), int(start_order) + int(count))
    cosine = np.empty((n, r, len(orders))) if want_cos else None
    sine = np.empty((n, r, len(orders))) if want_sin else None
    for start in range(0, len(orders), 32):
        arg = xi[:, :, None] * orders[None, None, start:start + 32]
        if want_cos:
            cosine[:, :, start:start + 32] = np.matmul(rows, np.cos(arg))
        if want_sin:
            sine[:, :, start:start + 32] = np.matmul(rows, np.sin(arg))
    return cosine, sine


def _cosine_moments(weights, xi, count):
    """``2 * sum_a weights[:, a] cos(m xi[:, a])`` for ``m = 0..count-1``,
    ``weights`` complex ``[n, na]``: the modal projection of the near Green's
    rules.  Native recurrence when available, else the trigonometric table."""
    native = _native_trig_moments(np.stack([weights.real, weights.imag], axis=1), xi, count, False)
    if native is not None:
        cosine = native[0]
        return 2.0 * (cosine[:, 0] + 1j * cosine[:, 1])
    _notice_native_fallback(('trig_moments',))
    out = np.empty((weights.shape[0], count), complex)
    orders = np.arange(count)
    for m0 in range(0, count, 32):
        table = np.cos(xi[:, :, None] * orders[None, None, m0:m0 + 32])
        out[:, m0:m0 + 32] = 2.0 * np.matmul(weights[:, None, :], table).squeeze(1)
    return out


def _parity_moments(even, odd, xi, count):
    """Cosine moments of the ``even`` rows and sine moments of the ``odd`` rows
    (``[n, 4, na]`` each) on the per-pair grids ``xi`` ``[n, na]``.

    One fused native pass (``parity_moments``) when the library has it, else
    two ``trig_moments`` calls, else the NumPy tables; the two native forms
    agree bitwise.
    """
    n, rows_even, na = even.shape
    rows_odd = odd.shape[1]
    count = int(count)
    fused = _native_entry('parity_moments')
    if fused is not None and n and na and count > 0:
        even = np.ascontiguousarray(even, dtype=float)
        odd = np.ascontiguousarray(odd, dtype=float)
        xi = np.ascontiguousarray(xi, dtype=float)
        cosines = np.empty((n, rows_even, count))
        sines = np.empty((n, rows_odd, count))
        fused(n, rows_even, rows_odd, na, count, even.ctypes.data, odd.ctypes.data,
              xi.ctypes.data, cosines.ctypes.data, sines.ctypes.data, 1)
        return cosines, sines
    native_even = _native_trig_moments(even, xi, count, False)
    native_odd = _native_trig_moments(odd, xi, count, True)
    if native_even is not None and native_odd is not None:
        return native_even[0], native_odd[1]
    if n and na and count > 0:
        _notice_native_fallback(('trig_moments',))
    cosines, _ = _numpy_trig_moments(even, xi, count, True, False)
    _, sines = _numpy_trig_moments(odd, xi, count, False, True)
    return cosines, sines


def _parity_outputs(cosines, sines, m):
    """Signed-mode brackets from the parity moments of ``_parity_moments``:
    tt/ff = C_tt/ff[|m|], tf/ft = -j sign(m) S_tf/ft[|m|]."""
    m = np.asarray(m, dtype=int)
    magnitude = np.abs(m)
    ec = (cosines[:, :2] + 1j * cosines[:, 2:])[:, :, magnitude]
    os_ = (-1j * np.sign(m)[None, None, :]) * (sines[:, :2] + 1j * sines[:, 2:])[:, :, magnitude]
    return [ec[:, 0], os_[:, 0], os_[:, 1], ec[:, 1]]


# --------------------------------------------------------------------------
# Far (banded) azimuthal quadrature
# --------------------------------------------------------------------------

# Per-pair bandwidth rule constants (see _far_sample_counts).
FAR_SAMPLE_QUANTUM = 32
FAR_BANDWIDTH_MARGIN = 16.0
FAR_TRANSITION_FACTOR = 8.0
FAR_DECAY_LENGTH = {'g': 40.0, 'bracket': 44.0}
FAR_TABLE_CACHE_BYTES = 64_000_000

_FAR_TABLES = OrderedDict()
_FAR_TABLE_BYTES = [0]
_FAR_TABLE_LOCK = threading.Lock()
# Each tile thread keeps weak references to the few tables it uses, so the shared
# store's lock (taken once per pair group per tile: 8-12% of the far build's
# thread time at eight threads) is only reached on a miss.
_FAR_TABLE_LOCAL = threading.local()
_FAR_TABLE_LOCAL_ENTRIES = 128


class _FarTableEntry:
    """Immutable sequence with weak-referenceable identity for local hits."""
    __slots__ = ('_values', '__weakref__')

    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)


def _remember_local_table(local, key, entry):
    local.pop(key, None)
    if len(local) >= _FAR_TABLE_LOCAL_ENTRIES:
        local.popitem(last=False)
    # Only the byte-budgeted shared cache and active computations own arrays.
    # Thread-local hits cannot retain evicted tables outside that RAM budget.
    local[key] = weakref.ref(entry)


def _far_sample_counts(bracket, rp, rq, gap, k, top_order, pts_per_peak=8.0):
    """Azimuthal sample count N (a multiple of 32) of every far pair.

    The half-grid trapezoid rule of N samples returns c_m + c_{m-N} + ...,
    so N must exceed the highest retained order M by the width over which
    the kernel's own Fourier coefficients c_n are still significant:

      N = 32 ceil((M + max(B + 8 B^(1/3) + 16, L / alpha)) / 32)
      B = |k| sqrt(rho_p rho_q)                   oscillation bandwidth of exp(-jkR)
      alpha = 2 asinh(d / (2 sqrt(rho_p rho_q)))  distance of the R = 0
                                                  singularity from the real xi axis
      L = 40 (Green's function) / 44 (brackets)   e^-L decay of c_n beyond it

    B + 8 B^(1/3) is where the Bessel-like coefficients of the oscillation
    become exponentially small; c_n of a kernel analytic in |Im xi| < alpha
    decay like exp(-alpha n).  Verified <= 4e-14 of the pair's largest
    retained coefficient on sphere, ogive and grooved bodies, real and
    complex k.  ``pts_per_peak`` scales L (8 is the default).
    """
    s = np.sqrt(np.maximum(np.asarray(rp, float) * np.asarray(rq, float), 0.0))
    bandwidth = abs(complex(k)) * s
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        alpha = 2.0 * np.arcsinh(np.asarray(gap, float) / np.maximum(2.0 * s, 1e-300))
        decay = (FAR_DECAY_LENGTH['bracket' if bracket else 'g'] * float(pts_per_peak) / 8.0) / alpha
    decay = np.where(np.isfinite(decay), decay, 0.0)
    need = top_order + np.maximum(
        bandwidth + FAR_TRANSITION_FACTOR * np.cbrt(bandwidth) + FAR_BANDWIDTH_MARGIN, decay)
    return (FAR_SAMPLE_QUANTUM * np.ceil(need / FAR_SAMPLE_QUANTUM)).astype(np.int64)


def _half_grid_projection_method(pairs, half, modes):
    """Conservative shape crossover, without doing trial projections in a solve.

    Few requested orders favor GEMM. Broad spectra favor the identical folded
    trapezoid sum via DCT/DST; medium spectra can repay packing into real BLAS.
    The methods differ only in summation order, not samples or retained modes.
    """
    if pairs >= 32 and modes >= 128 and modes >= half / 4:
        return 'fft'
    if pairs >= 128 and half >= 257 and modes >= 65:
        return 'real'
    return 'complex'


def _project_half_grid(samples, table, modes, size, method, odd=False):
    if method == 'complex':
        return samples @ table
    if method == 'real':
        packed = np.ascontiguousarray(samples.T).view(np.float64)
        result = (table.T @ packed).view(np.complex128).T
        if odd:
            result *= 1j  # table holds the imaginary part of -j sin(m xi).
        return result
    from scipy import fft
    # DCT-I/DST-I exactly express the same folded periodic trapezoid rule.
    # Fold arbitrary signed/out-of-Nyquist requested orders onto that grid;
    # do not truncate or reorder the requested output spectrum.
    index = np.remainder(modes, size)
    folded = np.minimum(index, size - index)
    phase = (2.0 * np.pi / size) * np.where(modes % 2, -1., 1.)
    if not odd:
        transformed = fft.dct(samples, type=1, axis=-1, workers=1)
        return transformed[:, folded] * phase
    result = np.zeros((len(samples), len(modes)), dtype=np.complex128)
    active = (folded > 0) & (folded < size // 2)
    if np.any(active):
        transformed = fft.dst(samples[:, 1:-1], type=1, axis=-1, workers=1)
        sign = np.where(index[active] > size // 2, -1., 1.)
        result[:, active] = transformed[:, folded[active] - 1] * (-1j * phase[active] * sign)
    return result


def _half_grid_tables(size, modes, want_sine, projection='complex'):
    """Cached half-grid transform tables of one sample count and mode set.

    Returns ``(cosine, sine, sin2, cx, sx)``: the folded trapezoid weights
    times cos (complex, for the complex GEMM) and ``-j`` times sin of
    ``m xi`` on ``xi = 2 pi i/size - pi``, ``i = 0..size/2``, and the grid's
    trigonometry for the samplers. The real projection stores real cosine and
    imaginary sine coefficients as float64; the FFT projection omits both
    coefficient tables. ``sine`` is None unless requested (the
    Green's function is even and needs none).  Entries are read-only and
    shared across threads; the cache is bounded by FAR_TABLE_CACHE_BYTES.
    """
    key = (int(size), modes.tobytes(), bool(want_sine), projection)
    local = getattr(_FAR_TABLE_LOCAL, 'tables', None)
    if local is None:
        local = _FAR_TABLE_LOCAL.tables = OrderedDict()
    reference = local.get(key)
    if reference is not None:
        entry = reference()
        if entry is not None:
            local.move_to_end(key)
            return entry
        del local[key]
    with _FAR_TABLE_LOCK:
        entry = _FAR_TABLES.get(key)
        if entry is not None:
            _FAR_TABLES.move_to_end(key)
    if entry is not None:
        _remember_local_table(local, key, entry)
        return entry
    size = int(size)
    half = size // 2 + 1
    xi = 2 * np.pi * np.arange(half) / size - np.pi
    # Periodic trapezoid rule folded onto the half grid: the interior points
    # stand for their mirror images, the ends (-pi and 0) for themselves; the
    # odd brackets vanish at both ends.
    weights = np.full(half, 2.0 * (2 * np.pi / size))
    weights[0] = weights[-1] = 2 * np.pi / size
    cosine = sine = None
    if projection != 'fft':
        argument = xi[:, None] * modes[None, :]
        if projection == 'real':
            cosine = np.asfortranarray(weights[:, None] * np.cos(argument))
            sine = np.asfortranarray(-weights[:, None] * np.sin(argument)) if want_sine else None
        else:
            cosine = (weights[:, None] * np.cos(argument)).astype(np.complex128)
            sine = (-1j * weights[:, None]) * np.sin(argument) if want_sine else None
    sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
    cx = np.ascontiguousarray(np.cos(xi))
    sx = np.ascontiguousarray(np.sin(xi))
    entry = _FarTableEntry((cosine, sine, sin2, cx, sx))
    nbytes = sum(value.nbytes for value in entry if value is not None)
    for value in entry:
        if value is not None:
            value.setflags(write=False)
    if nbytes <= FAR_TABLE_CACHE_BYTES // 4:
        with _FAR_TABLE_LOCK:
            if key not in _FAR_TABLES:
                _FAR_TABLES[key] = entry
                _FAR_TABLE_BYTES[0] += nbytes
                while _FAR_TABLE_BYTES[0] > FAR_TABLE_CACHE_BYTES and len(_FAR_TABLES) > 1:
                    _, old = _FAR_TABLES.popitem(last=False)
                    _FAR_TABLE_BYTES[0] -= sum(v.nbytes for v in old if v is not None)
            else:
                entry = _FAR_TABLES[key]
        _remember_local_table(local, key, entry)
    return entry


def _check_out_dtype(out_dtype):
    dtype = np.dtype(out_dtype)
    if dtype not in (np.dtype(np.complex128), np.dtype(np.complex64)):
        raise ValueError('BoR modal kernel tables must be complex64 or complex128.')
    return dtype


def banded_modal_kernels(kind, coordinates, k, m_max, near_mask, modes=None,
                         work_bytes=FFT_BUILD_BUDGET, threads=1, out_dtype=np.complex128):
    """Grouped azimuthal quadrature of the far modal kernels on half grids.

    Excluded near pairs are zero and are never sampled.  Far pairs are grouped
    by the sample count that the per-pair bandwidth rule of
    ``_far_sample_counts`` requires for their own radii, meridian distance and
    the highest retained order (``max(m_max+1, |modes|)`` for the Green's
    function, ``max(m_max, |modes|)`` for the brackets), a multiple of 32.
    The Green's function is even in the azimuth offset and the bracket
    components have exact parity (tt/ff even, tf/ft odd), so each group
    samples only the half grid ``xi`` in ``[-pi, 0]`` (both ends included) and
    the periodic trapezoid sums become real cosine/sine transforms of those
    samples, evaluated by complex GEMM, packed real GEMM or DCT/DST according
    to workload shape. Transform tables are cached per (size, modes, method).
    Sampling runs in the
    native paired kernels on ``threads`` OpenMP threads when the library
    provides them; the NumPy forms remain the reference and the fallback.
    Raw angular samples stay bounded by ``work_bytes`` independently of the
    number of pairs.  Results are stored as ``out_dtype`` (complex128 or
    complex64; each chunk is transformed in double precision and cast once).
    """
    if kind not in ('g', 'mfie', 'ibc'):
        raise ValueError('Unknown banded modal kernel.')
    dtype = _check_out_dtype(out_dtype)
    shape = np.broadcast(*coordinates).shape
    arrays = [np.broadcast_to(np.asarray(a,float),shape).ravel() for a in coordinates]
    rp,zp,rq,zq = arrays if kind=='g' else (arrays[0],arrays[1],arrays[4],arrays[5])
    modes = np.asarray(list(range(m_max+2)) if kind=='g' else list(range(-m_max,m_max+1)),int) if modes is None else np.asarray(modes,int).ravel()
    count = len(rp)
    outputs = [np.zeros((count,len(modes)),dtype) for _ in range(1 if kind=='g' else 4)]
    active = np.flatnonzero(~np.broadcast_to(near_mask,shape).ravel())
    gap = np.hypot(rp[active]-rq[active],zp[active]-zq[active])
    if np.any(gap<=0):
        raise ValueError('Coincident point pairs must use singular near quadrature.')
    bracket = kind!='g'
    top = max(int(m_max) + (0 if bracket else 1), int(np.max(np.abs(modes))) if modes.size else 0)
    sizes = _far_sample_counts(bracket, rp[active], rq[active], gap, k, top)
    if np.any(sizes>N_XI_SAFETY_CAP):
        raise ValueError('Banded azimuthal quadrature exceeds its sample safety cap; route close pairs to near integration.')
    wavenumber = complex(k)
    sampler = _native_pair_sampler(kind)
    if sampler is None and active.size:
        _notice_native_fallback(('sample_g_pairs' if kind == 'g' else 'sample_brackets_pairs',))
    threads = max(1, int(threads))
    order = np.argsort(sizes, kind='stable')
    group_sizes, starts = np.unique(sizes[order], return_index=True)
    stops = np.r_[starts[1:], len(order)]
    rows = 1 if kind == 'g' else 4
    # Scratch per pair: the samples (rows x half complex) and one GEMM result;
    # the NumPy fallback forms many more temporaries per sample.
    per_sample = rows if sampler is not None else (4 if kind == 'g' else 18)
    for size, start, stop in zip(group_sizes, starts, stops):
        indices = active[order[start:stop]]
        size = int(size)
        half = size // 2 + 1
        projected_pairs = min(len(indices), max(1, int(work_bytes // (
            16 * ((per_sample + 4) * half + len(modes))))))
        projection = _half_grid_projection_method(projected_pairs, half, len(modes))
        # Include real packing, or conservative FFT scratch, in the existing
        # chunk allowance. All transforms still finish in double precision.
        overhead = {'complex': 0, 'real': 1, 'fft': 4}[projection]
        cosine, sine, sin2, cx, sx = _half_grid_tables(size, modes, bracket, projection)
        chunk = max(1, int(work_bytes // (16 * ((per_sample + overhead) * half + len(modes)))))
        for first in range(0,len(indices),chunk):
            ids = indices[first:first+chunk]
            pair = [a[ids] for a in arrays]
            if kind=='g':
                samples = None if sampler is None else sampler(pair, wavenumber, sin2, threads)
                if samples is None:
                    a,b,c,d = pair
                    distance = np.sqrt(((a-c)**2+(b-d)**2)[:,None]+4*(a*c)[:,None]*sin2)
                    samples = np.exp(-1j*wavenumber*distance)/(4*np.pi*distance)
                outputs[0][ids] = _project_half_grid(samples, cosine, modes, size, projection)
            else:
                samples = None if sampler is None else sampler(pair, wavenumber, cx, sx, threads)
                if samples is None:
                    xi = 2*np.pi*np.arange(half)/size - np.pi
                    samples = (_mfie_brackets(*pair,k,xi) if kind=='mfie'
                               else _ibc_brackets_grid(*pair,k,np.broadcast_to(xi,(len(ids),half))))
                tt, tf, ft, ff = samples
                outputs[0][ids] = _project_half_grid(tt, cosine, modes, size, projection)
                outputs[1][ids] = _project_half_grid(tf, sine, modes, size, projection, odd=True)
                outputs[2][ids] = _project_half_grid(ft, sine, modes, size, projection, odd=True)
                outputs[3][ids] = _project_half_grid(ff, cosine, modes, size, projection)
    result = tuple(o.reshape(shape+(len(modes),)) for o in outputs)
    return result[0] if kind=='g' else result


def n_xi_for_pairs(k, rho_max: 'float', m_max: 'int', d_min: 'float' = 0.0,
                   bracket: 'bool' = False, pts_per_peak: 'float' = 8.0,
                   cap: 'int' = N_XI_SAFETY_CAP) -> 'int':
    """Worst-case azimuthal sample count of the far pairs of one table.

    Upper bound of the per-pair rule of ``banded_modal_kernels``
    (``_far_sample_counts``) over every far pair with radii <= ``rho_max`` and
    meridian distance >= ``d_min`` (0: no gap term), for the orders of an
    ``m_max`` table (0..m_max+1 for the Green's function, 0..m_max for the
    brackets): the bandwidth term grows with the radii and the decay term
    ``L/alpha`` with ``rho/d``.  Returned as the next power of two (at most
    ``cap``), so sizes planned from it bound every group the banded build
    forms.  Raises when the bound exceeds ``cap``; the preflight and the
    table/streaming budgets use it.
    """

    if cap < 1:
        raise ValueError("Azimuthal FFT sample cap must be positive.")
    top = int(m_max) + (0 if bracket else 1)
    rho_max = max(float(rho_max), 0.0)
    bound = int(_far_sample_counts(bracket, rho_max, rho_max, d_min if d_min > 0.0 else np.inf,
                                   k, top, pts_per_peak))
    if bound > int(cap):
        gap_note = (
            f", closest far-pair meridian gap {d_min:.6g} m"
            if d_min > 0.0 else ""
        )
        raise ValueError(
            "Azimuthal far-kernel quadrature requires "
            f"{bound} samples but the safety cap is {int(cap)}"
            f"{gap_note} (rho_max {rho_max:.6g} m, |k| {abs(k):.6g} 1/m, "
            f"m_max {int(m_max)}). The previous capped result would be "
            "under-resolved. Reduce frequency/mode count, route the closest "
            "pair through direct near integration or revise a close-fold "
            "geometry, or raise the internal cap only after checking memory. "
            "Do not refine solely to address this error: refinement usually "
            "shrinks the far-pair gap and increases the sample requirement."
        )
    return min(int(2 ** math.ceil(math.log2(max(bound, 1)))), int(cap))


def modal_kernels_fft(rho_p, z_p, rho_q, z_q, k, m_max: 'int', n_xi: 'int' = 0, near_mask=None,
                      threads: 'int' = 1, out_dtype=np.complex128):
    """
    G_m for m = 0..m_max+1 at point pairs via uniform xi sampling + FFT.

    Inputs are broadcastable arrays of pair coordinates.  Returns complex
    array [..., m_max+2] (extra order for the Gc/Gs neighbor relations;
    negative m follow from G_{-m} = G_m), stored as ``out_dtype``.

    Accuracy: the integrand is periodic and smooth when the pair is not
    near-singular; trapezoid/FFT is then spectrally accurate.  Callers must
    route near pairs to modal_kernels_near.

    Memory: the [pairs, n_xi] sampling grid is processed in bounded chunks
    (the gap-aware n_xi floor can push n_xi into the thousands on fine
    meshes; an all-pairs-at-once build then OOM-kills the process).
    """

    if near_mask is not None and BANDED_FFT:
        return banded_modal_kernels('g',(rho_p,z_p,rho_q,z_q),k,m_max,near_mask,threads=threads,
                                    out_dtype=out_dtype)
    dtype = _check_out_dtype(out_dtype)
    rho_p = np.asarray(rho_p, dtype=float)
    rho_q = np.asarray(rho_q, dtype=float)
    z_p = np.asarray(z_p, dtype=float)
    z_q = np.asarray(z_q, dtype=float)
    if n_xi <= 0:

        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(rho_p * rho_q, 0.0)))) if rho_p.size else 0.0
        n_xi = int(2 ** math.ceil(math.log2(max(64, 4 * (m_max + 2), 6 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
    sin2 = np.sin(0.5 * xi) ** 2
    shape = np.broadcast(rho_p, rho_q, z_p, z_q).shape
    d2f = np.broadcast_to((rho_p - rho_q) ** 2 + (z_p - z_q) ** 2, shape).ravel()
    rr4f = np.broadcast_to(4.0 * rho_p * rho_q, shape).ravel()
    m = np.arange(m_max + 2)
    phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
    out = np.empty((d2f.size, m_max + 2), dtype=dtype)
    chunk = max(1, int(FFT_BUILD_BUDGET / (16.0 * n_xi * 4.0)))
    for i0 in range(0, d2f.size, chunk):
        i1 = min(i0 + chunk, d2f.size)
        R = np.sqrt(d2f[i0:i1, None] + rr4f[i0:i1, None] * sin2)


        R = np.maximum(R, 1e-300)
        g = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)


        out[i0:i1] = np.fft.fft(g, axis=-1)[:, : m_max + 2] * phase
    return out.reshape(shape + (m_max + 2,))


# --------------------------------------------------------------------------
# Near-rule integrands
# --------------------------------------------------------------------------

def _green_samples(rho_p, z_p, rho_q, z_q, k, xi):
    """g = exp(-jkR)/(4 pi R), R = sqrt(d^2 + 4 rho_p rho_q sin^2(xi/2)).

    Point arrays are [n]; ``xi`` is per pair [n, na] or shared [na].  The
    native ``near_green`` evaluates the same expression in the same order;
    this NumPy form is its reference and fallback (they agree to rounding).
    """
    rho_p, z_p, rho_q, z_q = (np.ascontiguousarray(v, dtype=float) for v in (rho_p, z_p, rho_q, z_q))
    xi = np.ascontiguousarray(xi, dtype=float)
    n = rho_p.size
    wavenumber = complex(k)
    native = _native_entry('near_green')
    if native is not None and n and xi.size and (xi.ndim == 1 or (xi.ndim == 2 and xi.shape[0] == n)):
        nxi = xi.shape[-1]
        out = np.empty((n, nxi), dtype=np.complex128)
        native(n, nxi, rho_p.ctypes.data, z_p.ctypes.data, rho_q.ctypes.data, z_q.ctypes.data,
               wavenumber.real, wavenumber.imag, xi.ctypes.data, 1 if xi.ndim == 2 else 0,
               out.ctypes.data)
        return out
    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q
    h = np.sin(0.5 * xi)
    R = np.maximum(np.sqrt(d2[:, None] + rr4[:, None] * (h * h)), 1e-300)
    return np.exp(-1j * wavenumber * R) / (4.0 * np.pi * R)


def _native_mfie_brackets(points, k, xi, per_pair: 'bool'):
    """Four MFIE brackets; older native libraries retain their real-k path."""
    return _native_brackets(points, k, xi, per_pair, 'mfie')


def _native_brackets(points, k, xi, per_pair, family):
    """Paired real/complex bracket sampling, or None for the NumPy fallback."""

    import ctypes
    from ghost_backend.bor.streaming import _NATIVE

    if _NATIVE is None:
        return None
    wavenumber = complex(k)
    legacy = family == 'mfie' and wavenumber.imag == 0.0 and hasattr(_NATIVE, 'near_mfie')
    if not legacy and not hasattr(_NATIVE, 'near_brackets'):
        return None
    arrays = [np.ascontiguousarray(value, dtype=float) for value in points]
    if any(value.ndim != 1 for value in arrays):
        return None
    n_pairs = arrays[0].size
    if n_pairs == 0 or any(value.size != n_pairs for value in arrays):
        return None
    grid = np.ascontiguousarray(xi, dtype=float)
    if per_pair:
        if grid.ndim != 2 or grid.shape[0] != n_pairs:
            return None
        n_xi = int(grid.shape[1])
    else:
        if grid.ndim != 1:
            return None
        n_xi = int(grid.size)
    if n_xi == 0:
        return None
    pointer = ctypes.POINTER(ctypes.c_double)
    out = [np.empty((n_pairs, n_xi), dtype=np.complex128) for _ in range(4)]
    sampler = _NATIVE.near_mfie if legacy else _NATIVE.near_brackets
    prefix = [] if legacy else [ctypes.c_int(0 if family == 'mfie' else 1)]
    wave = [ctypes.c_double(wavenumber.real)]
    if not legacy:
        wave.append(ctypes.c_double(wavenumber.imag))
    sampler(
        *prefix,
        ctypes.c_int(n_pairs), ctypes.c_int(n_xi),
        *[value.ctypes.data_as(pointer) for value in arrays],
        *wave, grid.ctypes.data_as(pointer),
        ctypes.c_int(1 if per_pair else 0),
        *[value.ctypes.data_as(pointer) for value in out],
    )
    return tuple(out)


def _mfie_brackets(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi):
    """The four MFIE bracket functions at azimuth offsets xi.

    Point arrays have shape S; xi has shape X; returns four arrays S+X.
    Test point at phi = 0; source at phi' = -xi.  n_hat = (-tz, 0, tr)
    (outward per the generatrix convention)."""

    native = _native_mfie_brackets(
        (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q), k, xi, False
    )
    if native is not None:
        return native

    cx, sx = np.cos(xi), np.sin(xi)
    Rx = rho_p[..., None] - rho_q[..., None] * cx
    Ry = rho_q[..., None] * sx
    Rz = (z_p - z_q)[..., None] + 0.0 * cx
    R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
    R = np.maximum(R, 1e-300)


    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)

        WtR = tr_p[..., None] * Rx + tz_p[..., None] * Rz
        WfR = Ry
        nR = -tz_p[..., None] * Rx + tr_p[..., None] * Rz
        n_tq = -(tz_p * tr_q)[..., None] * cx + (tr_p * tz_q)[..., None]
        n_fq = -tz_p[..., None] * sx
        Wt_tq = (tr_p * tr_q)[..., None] * cx + (tz_p * tz_q)[..., None]
        Wt_fq = tr_p[..., None] * sx
        Wf_tq = -tr_q[..., None] * sx
        Wf_fq = cx + 0.0 * Rx

        Ftt = -p * (WtR * n_tq - Wt_tq * nR)
        Ftf = -p * (WtR * n_fq - Wt_fq * nR)
        Fft = -p * (WfR * n_tq - Wf_tq * nR)
        Fff = -p * (WfR * n_fq - Wf_fq * nR)
    return Ftt, Ftf, Fft, Fff


def mfie_kernels_fft(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                     m_max: 'int', n_xi: 'int' = 0, near_mask=None):
    """Modal MFIE kernels K_uv[..., m + m_max] for m = -m_max..m_max
    (far point pairs; FFT over uniform xi)."""

    if near_mask is not None and BANDED_FFT:
        return banded_modal_kernels('mfie',(rho_p,z_p,tr_p,tz_p,rho_q,z_q,tr_q,tz_q),k,m_max,near_mask)
    rho_p = np.asarray(rho_p, dtype=float)
    if n_xi <= 0:


        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(rho_p * rho_q, 0.0)))) if rho_p.size else 0.0
        n_xi = int(2 ** math.ceil(math.log2(max(128, 6 * (m_max + 2), 8 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
    m = np.arange(-m_max, m_max + 1)


    bins = np.where(m >= 0, m, n_xi + m)
    phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
    shape = np.broadcast(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q).shape
    flats = [np.broadcast_to(np.asarray(a, dtype=float), shape).ravel()
             for a in (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q)]
    n_pairs = flats[0].size
    out = [np.empty((n_pairs, 2 * m_max + 1), dtype=np.complex128)
           for _ in range(4)]

    chunk = max(1, int(FFT_BUILD_BUDGET / (16.0 * n_xi * 16.0)))
    for i0 in range(0, n_pairs, chunk):
        i1 = min(i0 + chunk, n_pairs)
        Fs = _mfie_brackets(*(a[i0:i1] for a in flats), k, xi)
        for o, F in zip(out, Fs):
            o[i0:i1] = np.fft.fft(F, axis=-1)[:, bins] * phase

        del Fs, F
    return tuple(o.reshape(shape + (2 * m_max + 1,)) for o in out)


def _project_parity_brackets(Fp, w_pos, xi_pos, m):
    """Exact angular parity of the MFIE/IBC tangential bracket components.

    tt/ff are even in xi; tf/ft are odd, for real and complex k. Thus their
    half-range contributions are 2 F cos(m xi) and -2j F sin(m xi).
    Preserve the generic signed-mode output without evaluating -xi or
    multiplying the identically zero/unused rows of the generic projector.
    The cosine moments of the even rows and the sine moments of the odd rows
    come from one fused native pass when available (``_parity_moments``).
    """
    m = np.asarray(m, dtype=int)
    magnitude = np.abs(m)
    count = int(magnitude.max()) + 1 if m.size else 0
    weighted = [2 * value * w_pos for value in Fp]
    even = np.stack([weighted[0].real, weighted[3].real,
                     weighted[0].imag, weighted[3].imag], axis=1)
    odd = np.stack([weighted[1].real, weighted[2].real,
                    weighted[1].imag, weighted[2].imag], axis=1)
    cosines, sines = _parity_moments(even, odd, np.asarray(xi_pos, dtype=float), count)
    return _parity_outputs(cosines, sines, m)


def _stable_brackets(points, k, xi, family):
    """The four MFIE or IBC brackets without the cancellations of the sampled forms.

    The sampled brackets form ``rho_p - rho_q*cos(xi)`` and products such as
    ``tr_p*tz_q - tz_p*tr_q*cos(xi)``. For two points ``d`` apart on the same or
    collinear elements those differences are of order ``rho*xi**2`` and
    ``d`` while their operands are of order ``rho``, so rounding leaves an
    error of about ``eps*(rho/d)**2``: 2e-9 of the kernel at ``d = 1e-5 rho``
    and 3e-7 at ``1e-6 rho``, above the rule's 2e-8 tolerance. With
    ``h = 1 - cos(xi) = 2 sin(xi/2)**2`` every such term is a sum of products
    whose ``h**2`` parts cancel analytically:

        R**2 = d**2 + 2 rho_p rho_q h
        A = t_p . (dr, dz),  N = n_p . (dr, dz),  X = t_p x t_q,  D = t_p . t_q

    (``A_q``, ``N_q`` in the source frame for the IBC family). The result stays
    within 4e-10 down to ``d = 1e-6 rho``. Arrays are [n_pairs] points and
    [n_pairs, n_xi] angles; returns four [n_pairs, n_xi] arrays.  The native
    ``near_brackets_stable`` evaluates the same expressions.
    """

    rp, zp, trp, tzp, rq, zq, trq, tzq = (np.asarray(v, dtype=float)[:, None] for v in points)
    h = 2.0 * np.sin(0.5 * xi) ** 2
    sx = np.sin(xi)
    d_rho, d_z = rp - rq, zp - zq
    R = np.maximum(np.sqrt(d_rho ** 2 + d_z ** 2 + 2.0 * rp * rq * h), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)
    X = trp * tzq - tzp * trq
    D = trp * trq + tzp * tzq
    if family == 'mfie':
        A = trp * d_rho + tzp * d_z
        N = -tzp * d_rho + trp * d_z
        tt = (A * X - D * N) + h * (A * tzp * trq + trp * rq * X + D * tzp * rq + trp * trq * N)
        tf = -sx * d_z * (trp ** 2 + tzp ** 2)
        ft = sx * (rq * X + trq * N)
        ff = -tzp * rq * h - N * (1.0 - h)
        return -p * tt, -p * tf, -p * ft, -p * ff
    A = trq * d_rho + tzq * d_z
    N = -tzq * d_rho + trq * d_z
    tt = -(X * A + D * N) + h * (X * trq * rp + trp * tzq * A - D * tzq * rp + trp * trq * N)
    tf = -sx * (X * rp + trp * N)
    ft = sx * d_z * (trq ** 2 + tzq ** 2)
    ff = tzq * rp * h - N * (1.0 - h)
    return p * tt, p * tf, p * ft, p * ff


def mfie_for_mode(K: 'np.ndarray', m: 'int', m_max: 'int', odd=False) -> 'np.ndarray':
    """Extract a signed mode from a centered or nonnegative parity table.

    A table of width ``m_max+1`` holds the orders 0..m_max (the odd brackets,
    ``odd=True``, change sign with m); one of width ``2*m_max+1`` holds
    -m_max..m_max.  Any other width, or an order outside the table, is a
    caller error and raises instead of indexing a wrong layout.
    """
    width = K.shape[-1]
    m, m_max = int(m), int(m_max)
    if abs(m) > m_max:
        raise ValueError(f"Mode {m} lies outside the bracket table orders |m| <= {m_max}.")
    if width == m_max + 1:
        return K[..., abs(m)] * (-1 if odd and m < 0 else 1)
    if width == 2 * m_max + 1:
        return K[..., m + m_max]
    raise ValueError(
        f"Bracket table width {width} is neither the nonnegative layout "
        f"(m_max+1 = {m_max + 1}) nor the centered layout (2*m_max+1 = {2 * m_max + 1}).")


def nonnegative_bracket_tables(kind, args, k, m_max, n_xi, near_mask, threads: 'int' = 1,
                               out_dtype=np.complex128):
    """Production far storage needs only nonnegative angular orders."""
    if BANDED_FFT:
        return banded_modal_kernels(kind, args, k, m_max, near_mask, np.arange(m_max+1), threads=threads,
                                    out_dtype=out_dtype)
    dtype = _check_out_dtype(out_dtype)
    function = mfie_kernels_fft if kind == 'mfie' else ibc_kernels_fft
    inputs = args if kind == 'mfie' else tuple(a.ravel() for a in args)
    tables = function(*inputs,k,m_max,n_xi=n_xi,near_mask=near_mask)
    return tuple(value[...,m_max:].astype(dtype) for value in tables)


def _ibc_brackets_grid(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi):
    """Four IBC bracket functions on per-pair xi grids ([n_pairs, n_xi])."""

    native = _native_brackets(
        (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q), k, xi, True, 'ibc')
    if native is not None:
        return native

    cx, sx = np.cos(xi), np.sin(xi)
    Rx = rho_p[:, None] - rho_q[:, None] * cx
    Ry = rho_q[:, None] * sx
    Rz = (z_p - z_q)[:, None] * np.ones_like(cx)
    R = np.maximum(np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2), 1e-300)


    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)


        Wt_nq = -(tr_p * tz_q)[:, None] * cx + (tz_p * tr_q)[:, None] * np.ones_like(cx)
        Wf_nq = tz_q[:, None] * sx
        R_tq = tr_q[:, None] * (rho_p[:, None] * cx - rho_q[:, None]) + tz_q[:, None] * Rz
        R_fq = rho_p[:, None] * sx
        R_nq = -tz_q[:, None] * (rho_p[:, None] * cx - rho_q[:, None]) + tr_q[:, None] * Rz
        Wt_tq = (tr_p * tr_q)[:, None] * cx + (tz_p * tz_q)[:, None] * np.ones_like(cx)
        Wt_fq = tr_p[:, None] * sx
        Wf_tq = -tr_q[:, None] * sx
        Wf_fq = cx * np.ones_like(Rx)

        Btt = p * (Wt_nq * R_tq - Wt_tq * R_nq)
        Btf = p * (Wt_nq * R_fq - Wt_fq * R_nq)
        Bft = p * (Wf_nq * R_tq - Wf_tq * R_nq)
        Bff = p * (Wf_nq * R_fq - Wf_fq * R_nq)
    return Btt, Btf, Bft, Bff


def ibc_kernels_fft(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                    m_max: 'int', n_xi: 'int' = 0, near_mask=None):
    """Modal IBC kernels [Pp, Pq, 2*m_max+1] via FFT (far point pairs).
    Test/source point arrays are [Pp]/[Pq] vectors; all pair combinations
    are formed here."""

    if near_mask is not None and BANDED_FFT:
        args = tuple(np.asarray(a)[:,None] for a in (rho_p,z_p,tr_p,tz_p))
        args += tuple(np.asarray(a)[None,:] for a in (rho_q,z_q,tr_q,tz_q))
        return banded_modal_kernels('ibc',args,k,m_max,near_mask)
    Pp = len(np.atleast_1d(rho_p))
    Pq = len(np.atleast_1d(rho_q))
    if n_xi <= 0:
        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(
            np.outer(rho_p, rho_q), 0.0))))
        n_xi = int(2 ** math.ceil(math.log2(max(128, 6 * (m_max + 2), 8 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
    pair_shape = (Pp, Pq)
    pr = np.broadcast_to(np.asarray(rho_p)[:, None], pair_shape).ravel()
    pz = np.broadcast_to(np.asarray(z_p)[:, None], pair_shape).ravel()
    ptr = np.broadcast_to(np.asarray(tr_p)[:, None], pair_shape).ravel()
    ptz = np.broadcast_to(np.asarray(tz_p)[:, None], pair_shape).ravel()
    qr = np.broadcast_to(np.asarray(rho_q)[None, :], pair_shape).ravel()
    qz = np.broadcast_to(np.asarray(z_q)[None, :], pair_shape).ravel()
    qtr = np.broadcast_to(np.asarray(tr_q)[None, :], pair_shape).ravel()
    qtz = np.broadcast_to(np.asarray(tz_q)[None, :], pair_shape).ravel()
    m = np.arange(-m_max, m_max + 1)
    bins = np.where(m >= 0, m, n_xi + m)
    phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
    n_pairs = Pp * Pq
    out = [np.empty((n_pairs, 2 * m_max + 1), dtype=np.complex128)
           for _ in range(4)]

    chunk = max(1, int(FFT_BUILD_BUDGET / (16.0 * n_xi * 18.0)))
    flats = (pr, pz, ptr, ptz, qr, qz, qtr, qtz)
    for i0 in range(0, n_pairs, chunk):
        i1 = min(i0 + chunk, n_pairs)
        Fs = _ibc_brackets_grid(*(a[i0:i1] for a in flats), k,
                                np.broadcast_to(xi, (i1 - i0, n_xi)))
        for o, F in zip(out, Fs):
            o[i0:i1] = np.fft.fft(F, axis=-1)[:, bins] * phase
        del Fs, F
    return tuple(o.reshape(Pp, Pq, -1) for o in out)


def _checked_near_kernels(rule, args, k, m_max, order, tail_order, bracket):
    """Resolve core and tail oscillation, check refinement, and bound scratch
    (legacy two-piece rules ``_*_kernels_near_rule``).

    A refinement step compares a coarse and a fine result only when BOTH the
    core and the tail order increased: an order held at
    NEAR_ANGULAR_MAX_ORDER would be shared by the two results, its error
    would cancel out of their difference and the step could accept an
    unconverged kernel.  When a capped order prevents a valid comparison the
    bracket rules restart with the cancellation-free forms, and otherwise the
    call raises.

    The limit applies to temporary point/angular/mode arrays. The returned
    point-by-mode arrays are owned by the caller and must be budgeted there.
    """
    args = tuple(np.ravel(a) for a in np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, dtype=float)) for a in args]))
    q = 4 if bracket else 2
    rp, zp, rq, zq = args[0], args[1], args[q], args[q + 1]
    n = len(rp)
    nm = 2 * m_max + 1 if bracket else m_max + 2
    outputs = tuple(np.empty((n, nm), complex) for _ in range(4 if bracket else 1))
    if n == 0:
        return outputs if bracket else outputs[0]
    a = np.sqrt(np.maximum(4 * rp * rq, 0.0))
    d = np.hypot(rp - rq, zp - zq)
    s0 = np.minimum(0.25, 20 * d / np.maximum(a, 1e-150))
    xi0 = 2 * np.arcsin(s0)
    core_phase = abs(complex(k)) * a * s0 + (m_max + 2) * xi0
    if bracket:
        core_phase = np.where(a <= 1e-15, (m_max + 2) * np.pi, core_phase)
    core = max(int(order), 48, int(math.ceil(24 + 2 * np.max(core_phase))))
    tail = max(int(tail_order), 64, int(math.ceil(
        4 * (abs(complex(k)) * float(np.max(a)) / math.pi + m_max + 2))))
    if max(core, tail) > NEAR_ANGULAR_MAX_ORDER:
        raise ValueError("BoR near angular quadrature exceeds its accuracy limit; refine the mesh or reduce modal bandwidth.")


    point_chunk = max(1, int(NEAR_KERNEL_WORK_BYTES /
                            (128 * (core + tail) * max(min(nm, 32), 16))))
    for start in range(0, n, point_chunk):
        stop = min(n, start + point_chunk)
        # Absolute indices of the points in this chunk that still need a
        # higher angular order.  Each point's kernel depends only on its own
        # coordinates, so refining a subset reproduces its per-point values;
        # points that already meet the tolerance keep their accepted result.
        active = np.arange(start, stop)
        part = tuple(a[start:stop] for a in args)


        c, t = max(16, core // 2), max(16, tail // 2)
        coarse = rule(*part, k, m_max, order=c, tail_order=t)
        coarse = coarse if bracket else (coarse,)
        # Sampled brackets reach a rounding floor above the tolerance once two
        # points are closer than about 5e-6 rho (tiny or graded elements).
        # Points still pending at the maximum order restart with the
        # cancellation-free closed forms; everything that converged before is
        # untouched, so existing results do not change.
        options = {}
        error = scale = None
        while True:
            cf = min(NEAR_ANGULAR_MAX_ORDER, max(core, c + 16, int(math.ceil(1.5 * c))))
            tf = min(NEAR_ANGULAR_MAX_ORDER, max(tail, t + 16, int(math.ceil(1.5 * t))))
            if cf <= c or tf <= t:
                # A capped order cannot increase: no valid comparison remains.
                if bracket and not options:
                    options = {'stable': True}
                    c, t = max(16, core // 2), max(16, tail // 2)
                    coarse = rule(*part, k, m_max, order=c, tail_order=t, **options)
                    continue
                if error is None:
                    raise ValueError("BoR near angular quadrature exceeds its accuracy limit; refine the mesh or reduce modal bandwidth.")
                worst = int(np.argmax(error / np.maximum(scale, 1e-280)))
                coordinates = tuple(float(a[worst]) for a in part)
                raise ValueError(f"BoR near angular quadrature did not converge at the maximum order: {rule.__name__}, point={coordinates}, relative change={float(error[worst] / max(scale[worst], 1e-280)):.3g}.")


            fine_chunk = max(1, int(NEAR_KERNEL_WORK_BYTES /
                                   (128 * (cf + tf) * max(min(nm, 32), 16))))
            fine = tuple(np.empty_like(x) for x in coarse)
            for i in range(0, len(active), fine_chunk):
                val = rule(*(a[i:i + fine_chunk] for a in part), k, m_max,
                           order=cf, tail_order=tf, **options)
                val = val if bracket else (val,)
                for out, x in zip(fine, val):
                    out[i:i + fine_chunk] = x
            scale = np.maximum.reduce([np.max(np.abs(x), axis=1) for x in fine])
            error = np.maximum.reduce([np.max(np.abs(x - y), axis=1)
                                       for x, y in zip(fine, coarse)])
            converged = np.isfinite(error) & (
                error <= NEAR_ANGULAR_RTOL * np.maximum(scale, 1e-280)
            )
            for out, x in zip(outputs, fine):
                out[active[converged]] = x[converged]
            if np.all(converged):
                break
            pending = ~converged
            active = active[pending]
            part = tuple(a[pending] for a in part)
            coarse = tuple(x[pending] for x in fine)
            error, scale = error[pending], scale[pending]
            c, t = cf, tf
    return outputs if bracket else outputs[0]


# --------------------------------------------------------------------------
# Graded near angular rule (production)
# --------------------------------------------------------------------------

# Layout of the angular rule of one near point pair, in s = sin(xi/2):
#   core    [0, s_c]: Gauss-Legendre in v with s = (d/a) sinh(v), which makes
#           R = d cosh(v) exact and removes the near-singularity at xi = 0;
#           s_c = NEAR_CORE_TOP 2^-J lies in (10, 20] d/a;
#   panels  [NEAR_CORE_TOP 2^-j, NEAR_CORE_TOP 2^-(j-1)], j = J..1: ratio-2
#           geometric panels; the R = 0 singularity at s = +-j d/a stays at
#           least one panel length away (Bernstein radius >= 5.8), so each
#           needs a fixed small order plus its own oscillation;
#   tail    xi in [2 asin(NEAR_CORE_TOP), pi]: one panel for the rest.
# Every piece's Gauss order is sized from its own phase Phi (orders times
# the angle it spans plus |k| times the range of R over it) as
#   n = base + slope*Phi + 2 Phi^(1/3)   (rounded up to a multiple of 4),
# the Gauss-Legendre resolution of an oscillation of total phase Phi (1/4
# node per radian on a linear panel; the sinh core compresses its phase
# toward s_c and needs ~0.42).  The bases were calibrated on the diagonal,
# corner and regular near cells of spheres (ka 3..30), an ogive and a lossy
# medium: coarse-level errors <= 4e-11, fine-level <= 3e-13 of the largest
# kernel value, so the NEAR_ANGULAR_RTOL check passes at the first level.
NEAR_CORE_TOP = 0.25
NEAR_CORE_RATIO = 20.0
NEAR_MAX_PANELS = 64
NEAR_LEVEL_GROWTH = 1.5
NEAR_STABLE_RATIO = 1.0e-4
_NEAR_PIECES = {
    # (base order of the coarse level, nodes per radian)
    'core': (12.0, 0.42),
    'panel': (4.0, 0.25),
    'tail': (10.0, 0.25),
}
_NEAR_SHARED = OrderedDict()
_NEAR_SHARED_LOCK = threading.Lock()
_NEAR_SHARED_ENTRIES = 256
_NEAR_BYTES_PER_SAMPLE = {'native': 48, 'numpy': 400}


def _near_piece_orders(phase, piece):
    base, slope = _NEAR_PIECES[piece]
    phase = np.maximum(np.asarray(phase, dtype=float), 0.0)
    n = base + slope * phase + 2.0 * np.cbrt(phase)
    return (4 * np.ceil(n / 4.0)).astype(np.int64)


def _near_panel_geometry(J):
    """Bounds (lo, hi) in s of the panels j = J..1 (innermost first)."""
    j = np.arange(int(J), 0, -1, dtype=float)
    lo = NEAR_CORE_TOP * np.exp2(-j)
    return lo, 2.0 * lo


@lru_cache(maxsize=1024)
def _unit_gauss(order):
    """Gauss-Legendre nodes and weights on [0, 1] (read-only)."""
    x, w = cached_leggauss(int(order))
    u, wu = 0.5 * (x + 1.0), 0.5 * w
    u.setflags(write=False)
    wu.setflags(write=False)
    return u, wu


def _near_shared_nodes(J, panel_orders, tail_order):
    """Nodes/weights shared by every pair of one layout: the geometric panels
    and the tail (J >= 0), or the single on-axis panel on [0, pi] (J = -1)."""
    key = (int(J), tuple(int(n) for n in panel_orders), int(tail_order))
    with _NEAR_SHARED_LOCK:
        entry = _NEAR_SHARED.get(key)
        if entry is not None:
            _NEAR_SHARED.move_to_end(key)
            return entry
    xs, ws = [], []
    if J < 0:
        u, wu = _unit_gauss(int(tail_order))
        xs.append(np.pi * u)
        ws.append(np.pi * wu)
    else:
        lo, hi = _near_panel_geometry(J)
        for a, b, order in zip(lo, hi, panel_orders):
            x, w = cached_leggauss(int(order))
            s = a + 0.5 * (x + 1.0) * (b - a)
            xs.append(2.0 * np.arcsin(s))
            ws.append(0.5 * w * (b - a) * 2.0 / np.sqrt(1.0 - s * s))
        xi_top = 2.0 * math.asin(NEAR_CORE_TOP)
        x, w = cached_leggauss(int(tail_order))
        xs.append(xi_top + 0.5 * (x + 1.0) * (math.pi - xi_top))
        ws.append(0.5 * w * (math.pi - xi_top))
    entry = (np.ascontiguousarray(np.concatenate(xs)), np.ascontiguousarray(np.concatenate(ws)))
    for value in entry:
        value.setflags(write=False)
    with _NEAR_SHARED_LOCK:
        _NEAR_SHARED[key] = entry
        while len(_NEAR_SHARED) > _NEAR_SHARED_ENTRIES:
            _NEAR_SHARED.popitem(last=False)
    return entry


def _near_rule_parts(layout, orders):
    """(s_core, core Gauss order, shared xi, shared w) of one group level.

    ``layout`` is the panel count J (>= 0), or -1 for on-axis pairs, whose
    rule is one panel on [0, pi]; ``orders`` = (core, panel_J, ..., panel_1,
    tail), or (panel,) on the axis.
    """
    if layout < 0:
        xi, w = _near_shared_nodes(-1, (), int(orders[-1]))
        return 0.0, 0, xi, w
    J = int(layout)
    xi, w = _near_shared_nodes(J, orders[1:-1], orders[-1])
    return NEAR_CORE_TOP * 2.0 ** -J, int(orders[0]), xi, w


def _near_nodes(layout, orders, delta):
    """Angular nodes and weights [n, na] on [0, pi] of one group of pairs:
    the sinh core of each pair (``delta`` = d/a) followed by the shared nodes.
    The native rule kernels form the same nodes in the same order."""
    s_core, n_core, xi_shared, w_shared = _near_rule_parts(layout, orders)
    n = len(delta)
    nc, ns = n_core, xi_shared.size
    xi = np.empty((n, nc + ns))
    w_all = np.empty((n, nc + ns))
    if nc:
        u, wu = _unit_gauss(nc)
        vmax = np.arcsinh(s_core / delta)
        v = u[None, :] * vmax[:, None]
        s = delta[:, None] * np.sinh(v)
        xi[:, :nc] = 2.0 * np.arcsin(s)
        w_all[:, :nc] = (wu[None, :] * vmax[:, None]) * (2.0 * delta[:, None] * np.cosh(v)) / np.sqrt(1.0 - s * s)
    xi[:, nc:] = xi_shared
    w_all[:, nc:] = w_shared
    return xi, w_all


def _green_node_moments(points, k, xi, w, count, start_order=0):
    """Cosine moments [n, 2, count] of Re/Im(w g) on explicit nodes
    (sampler + projection: the reference of ``near_green_rule``)."""
    gw = _green_samples(*points, k, xi) * w
    rows = np.ascontiguousarray(np.stack([gw.real, gw.imag], axis=1))
    if start_order:
        return _numpy_trig_moments(rows, xi, count, True, False, start_order)[0]
    native = _native_trig_moments(rows, xi, count, False)
    if native is not None:
        return native[0]
    _notice_native_fallback(('trig_moments',))
    return _numpy_trig_moments(rows, xi, count, True, False)[0]


def _bracket_node_moments(kind, stable, points, k, xi, w, count, start_order=0):
    """Parity moments (cos [n, 4, count], sin [n, 4, count]) of the weighted
    brackets on explicit nodes (sampler + ``_parity_moments``: the reference
    of ``near_brackets_rule``)."""
    n, na = xi.shape
    wavenumber = complex(k)
    family = 0 if kind == 'mfie' else 1
    arrays = [np.ascontiguousarray(v, dtype=float) for v in points]
    if stable:
        native = _native_entry('near_brackets_stable')
        if native is not None:
            samples = [np.empty((n, na), dtype=np.complex128) for _ in range(4)]
            native(family, n, na, *[v.ctypes.data for v in arrays], wavenumber.real,
                   wavenumber.imag, xi.ctypes.data, 1, *[v.ctypes.data for v in samples])
        else:
            samples = _stable_brackets(tuple(arrays), k, xi, kind)
    else:
        samples = _native_brackets(tuple(arrays), k, xi, True, kind)
        if samples is None:
            samples = (_mfie_brackets(*arrays, k, xi) if kind == 'mfie'
                       else _ibc_brackets_grid(*arrays, k, xi))
    weighted = [2 * value * w for value in samples]
    even = np.stack([weighted[0].real, weighted[3].real, weighted[0].imag, weighted[3].imag], axis=1)
    odd = np.stack([weighted[1].real, weighted[2].real, weighted[1].imag, weighted[2].imag], axis=1)
    if start_order:
        return (_numpy_trig_moments(even, xi, count, True, False, start_order)[0],
                _numpy_trig_moments(odd, xi, count, False, True, start_order)[1])
    return _parity_moments(even, odd, xi, count)


def _near_rule_moments(kind, stable, points, delta, k, layout, orders, count, start_order=0):
    """Moments of one group level: [n, 2, count] (Green's function) or the
    parity pair (cos, sin) [n, 4, count] (brackets).  One native call per
    group (``near_green_rule`` / ``near_brackets_rule``) when available, else
    the explicit nodes of ``_near_nodes`` through the samplers."""
    n = len(delta)
    wavenumber = complex(k)
    native = _native_entry('near_green_rule' if kind == 'g' else 'near_brackets_rule')
    if native is not None and not start_order:
        s_core, n_core, xi_shared, w_shared = _near_rule_parts(layout, orders)
        u, wu = _unit_gauss(n_core) if n_core else (xi_shared, w_shared)
        arrays = [np.ascontiguousarray(v, dtype=float) for v in points]
        delta = np.ascontiguousarray(delta, dtype=float)
        tail = (delta.ctypes.data, wavenumber.real, wavenumber.imag, float(s_core), int(n_core),
                u.ctypes.data, wu.ctypes.data, int(xi_shared.size), xi_shared.ctypes.data,
                w_shared.ctypes.data, int(count))
        if kind == 'g':
            out = np.empty((n, 2, count))
            native(n, *[v.ctypes.data for v in arrays], *tail, out.ctypes.data, 1)
            return out
        cosines = np.empty((n, 4, count))
        sines = np.empty((n, 4, count))
        native(0 if kind == 'mfie' else 1, 1 if stable else 0, n,
               *[v.ctypes.data for v in arrays], *tail, cosines.ctypes.data, sines.ctypes.data, 1)
        return cosines, sines
    xi, w = _near_nodes(layout, orders, delta)
    if kind == 'g':
        return _green_node_moments(points, k, xi, w, count, start_order)
    return _bracket_node_moments(kind, stable, points, k, xi, w, count, start_order)


def _near_layout(d, a, on_axis, kabs, top, order_floor, tail_floor):
    """Group key per point: (J or -1, stable placeholder, orders...).

    Returns (J [n], orders [n, P]) with P = 2 + max J: core, panels J..1 and
    tail orders of the coarse level (zero-padded after the tail)."""
    n = len(d)
    J = np.zeros(n, dtype=np.int64)
    off = ~on_axis
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = NEAR_CORE_TOP / (NEAR_CORE_RATIO * d[off] / a[off])
        levels = np.ceil(np.log2(np.maximum(ratio, 1e-300)))
    J[off] = np.clip(np.where(np.isfinite(levels), levels, NEAR_MAX_PANELS), 0, NEAR_MAX_PANELS).astype(np.int64)
    J[on_axis] = -1
    width = 2 + int(max(J.max(), 0))
    orders = np.zeros((n, width), dtype=np.int64)
    growth = NEAR_LEVEL_GROWTH
    core_floor = int(math.ceil(order_floor / growth)) if order_floor > 0 else 0
    tail_min = int(math.ceil(tail_floor / growth)) if tail_floor > 0 else 0
    xi_top = 2.0 * math.asin(NEAR_CORE_TOP)
    if np.any(on_axis):
        # R is constant on the axis; the brackets carry cos xi / sin xi.
        axis_order = int(_near_piece_orders((top + 1) * math.pi, 'tail'))
        orders[on_axis, 0] = max(axis_order, core_floor, tail_min)
    if np.any(off):
        index = np.flatnonzero(off)
        Jo = J[index]
        ao = a[index]
        s_c = NEAR_CORE_TOP * np.exp2(-Jo.astype(float))
        core_phase = top * 2.0 * np.arcsin(s_c) + kabs * ao * s_c
        orders[index, 0] = np.maximum(_near_piece_orders(core_phase, 'core'), core_floor)
        tail_phase = top * (math.pi - xi_top) + kabs * ao * (1.0 - NEAR_CORE_TOP)
        tail_orders = np.maximum(_near_piece_orders(tail_phase, 'tail'), tail_min)
        jmax = int(Jo.max())
        if jmax > 0:
            j = np.arange(1, jmax + 1, dtype=float)       # j = 1 is the outermost panel
            lo = NEAR_CORE_TOP * np.exp2(-j)
            hi = 2.0 * lo
            dxi = 2.0 * (np.arcsin(hi) - np.arcsin(lo))
            phase = top * dxi[None, :] + kabs * ao[:, None] * (hi - lo)[None, :]
            panel = _near_piece_orders(phase, 'panel')      # [n_off, jmax], column j-1
            # Stored innermost first (panels J..1): column 1+t holds panel J-t.
            t = np.arange(jmax)
            valid = t[None, :] < Jo[:, None]
            column = np.where(valid, Jo[:, None] - 1 - t[None, :], 0)
            orders[index, 1:jmax + 1] = np.where(
                valid, np.take_along_axis(panel, column, axis=1), 0)
        orders[index, Jo + 1] = tail_orders
    return J, orders


def _graded_near_kernels(kind, args, k, m_max, order_floor=0, tail_floor=0, signed=True,
                         mode_start=0):
    """Checked graded near rule of the Green's function (kind 'g') or the
    MFIE/IBC brackets.

    Points are grouped by their layout (panel count J and coarse orders, and
    for the brackets the choice of sampled or cancellation-free forms), so
    each point's result depends on its own coordinates only.  Every group is
    evaluated at a coarse and a fine level (orders x NEAR_LEVEL_GROWTH); a
    point is accepted when the two agree to NEAR_ANGULAR_RTOL of its largest
    kernel value.  Pending points go on to the next level.  A level is only
    compared when EVERY piece's order increased under NEAR_ANGULAR_MAX_ORDER
    (a capped piece would be shared by both results and its error cancel out
    of the check); when no valid level remains the bracket points restart
    with the cancellation-free forms, and otherwise the call raises.

    Brackets use the cancellation-free closed forms from the start: for every
    pair with the native rule kernel (they cost the same there), otherwise
    when d < NEAR_STABLE_RATIO * max(rho_p, rho_q).  Below that separation
    the sampled forms carry rounding of order eps (rho/d)^2, which a
    coarse/fine check cannot see because both levels share it.
    """
    bracket = kind != 'g'
    arrays = tuple(np.ravel(a) for a in np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, dtype=float)) for a in args]))
    q = 4 if bracket else 2
    rp, zp, rq, zq = arrays[0], arrays[1], arrays[q], arrays[q + 1]
    n = len(rp)
    mode_start = int(mode_start)
    if mode_start < 0 or mode_start > int(m_max) or (bracket and signed and mode_start):
        raise ValueError('A near modal band requires nonnegative orders within the cap.')
    top = m_max if bracket else m_max + 1
    count = top + 1 - mode_start
    if bracket:
        out = (np.zeros((n, 4, count)), np.zeros((n, 4, count)))
    else:
        out = np.zeros((n, count), dtype=np.complex128)
    if n:
        rr4 = 4.0 * rp * rq
        on_axis = rr4 <= 1e-30
        d = np.maximum(np.hypot(rp - rq, zp - zq), 1e-150)
        a = np.sqrt(np.maximum(rr4, 0.0))
        todo = np.ones(n, dtype=bool)
        if not bracket and np.any(on_axis):
            # R = d for every angle: only G_0 is nonzero.
            R0 = np.hypot(rp[on_axis] - rq[on_axis], zp[on_axis] - zq[on_axis])
            g0 = np.exp(-1j * complex(k) * R0) / (4.0 * np.pi * np.maximum(R0, 1e-300))
            if mode_start == 0:
                out[on_axis, 0] = 2.0 * np.pi * g0
            todo &= ~on_axis
        stable = np.zeros(n, dtype=bool)
        if bracket:
            if _native_entry('near_brackets_rule') is not None:
                stable[:] = True
            else:
                stable = d < NEAR_STABLE_RATIO * np.maximum(rp, rq)
        index = np.flatnonzero(todo)
        if index.size:
            J, orders = _near_layout(d[index], a[index], on_axis[index], abs(complex(k)), top,
                                     int(order_floor), int(tail_floor))
            fine = np.ceil(orders * NEAR_LEVEL_GROWTH).astype(np.int64)
            if int(fine.max()) > NEAR_ANGULAR_MAX_ORDER:
                raise ValueError("BoR near angular quadrature exceeds its accuracy limit; refine the mesh or reduce modal bandwidth.")
            order, bounds = _layout_groups(np.column_stack([J + 1, stable[index].astype(np.int64), orders]))
            delta = d / np.where(a > 0, a, 1.0)
            for g in range(len(bounds) - 1):
                local = order[bounds[g]:bounds[g + 1]]
                members = index[local]
                first = int(local[0])
                layout = int(J[first])
                base = orders[first, :layout + 2] if layout >= 0 else orders[first, :1]
                _run_near_group(kind, arrays, k, count, members, layout, base,
                                bool(stable[index[first]]), delta, out, mode_start)
    if not bracket:
        return out
    if mode_start or not signed:
        # Nonnegative orders only: the parity moments are the outputs themselves
        # (tt/ff = C[m], tf/ft = -j S[m], the m = 0 sine moments vanishing).
        even = out[0][:, :2] + 1j * out[0][:, 2:]
        odd = -1j * (out[1][:, :2] + 1j * out[1][:, 2:])
        if not mode_start:
            odd[:, :, 0] = 0.0
        return even[:, 0], odd[:, 0], odd[:, 1], even[:, 1]
    m = np.arange(-m_max, m_max + 1)
    return tuple(_parity_outputs(out[0], out[1], m))


_LAYOUT_KEY_BITS = 13      # orders are at most NEAR_ANGULAR_MAX_ORDER = 4096
_LAYOUT_KEY_FIELDS = 64 // _LAYOUT_KEY_BITS


def _layout_groups(keys):
    """``(order, bounds)``: point indices sorted by layout key (stable), and the
    group boundaries in that order, for nonnegative integer key rows below
    2**_LAYOUT_KEY_BITS.  Rows are packed into a few int64 words and sorted
    lexicographically, which replaces ``np.unique(axis=0)`` plus a second
    stable argsort (8% of a serial near preparation) by one sort of the
    packed words; the groups and their member order are identical."""
    keys = np.asarray(keys, dtype=np.int64)
    n, fields = keys.shape
    words = []
    for start in range(0, fields, _LAYOUT_KEY_FIELDS):
        word = np.zeros(n, dtype=np.int64)
        for column in range(start, min(start + _LAYOUT_KEY_FIELDS, fields)):
            word = (word << _LAYOUT_KEY_BITS) | keys[:, column]
        words.append(word)
    if len(words) == 1:
        order = np.argsort(words[0], kind='stable')
    else:
        order = np.lexsort(words[::-1])
    change = np.empty(n, dtype=bool)
    change[0] = True
    if n > 1:
        change[1:] = False
        for word in words:
            sorted_word = word[order]
            change[1:] |= sorted_word[1:] != sorted_word[:-1]
    bounds = np.r_[np.flatnonzero(change), n]
    return order, bounds


def _near_check(kind, fine, coarse):
    """Per point (largest |fine - coarse|, largest |fine|) over every kernel
    and order: the magnitudes of the complex values the outputs are made of
    (for the brackets the -j sign(m) factor of the odd ones does not change
    them; their m = 0 sine moments vanish)."""
    if kind == 'g':
        # values 2 (c0 + j c1)
        scale = 2.0 * np.max(np.hypot(fine[:, 0], fine[:, 1]), axis=1)
        diff = fine - coarse
        error = 2.0 * np.max(np.hypot(diff[:, 0], diff[:, 1]), axis=1)
        return error, scale
    scale = error = None
    for part, cpart in zip(fine, coarse):
        # rows (re, re, im, im) of two kernels
        size = np.max(np.hypot(part[:, :2], part[:, 2:]), axis=(1, 2))
        diff = part - cpart
        err = np.max(np.hypot(diff[:, :2], diff[:, 2:]), axis=(1, 2))
        scale = size if scale is None else np.maximum(scale, size)
        error = err if error is None else np.maximum(error, err)
    return error, scale


def _run_near_group(kind, arrays, k, count, members, layout, base_orders, stable, delta, out,
                    start_order=0):
    """Coarse/fine refinement of one layout group (see _graded_near_kernels).

    A modal band (``start_order > 0``) is evaluated over every order from 0,
    by the native rule, and sliced when it is stored: the acceptance check
    then keeps the full-range scale.  Checked against the band's own orders
    alone, near-axis points never converged (their high orders are rounding
    noise of the dominant low ones), so the automatic cap-expansion retry
    raised on closed bodies; the band path was also slower per point than a
    full native evaluation.
    """
    bracket = kind != 'g'
    growth = NEAR_LEVEL_GROWTH
    native = _native_entry('near_green_rule' if kind == 'g' else 'near_brackets_rule') is not None
    per_sample = _NEAR_BYTES_PER_SAMPLE['native' if native else 'numpy']
    full = int(count) + int(start_order)

    def evaluate(ids, orders, use_stable):
        # Angular scratch stays within NEAR_KERNEL_WORK_BYTES at every level.
        block = max(1, int(NEAR_KERNEL_WORK_BYTES // (per_sample * max(int(np.sum(orders)), 1))))
        parts = [_near_rule_moments(kind, use_stable, tuple(v[ids[i:i + block]] for v in arrays),
                                    delta[ids[i:i + block]], k, layout, orders, full)
                 for i in range(0, len(ids), block)]
        if len(parts) == 1:
            return parts[0]
        if kind == 'g':
            return np.concatenate(parts)
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    def next_orders(orders):
        grown = np.minimum(NEAR_ANGULAR_MAX_ORDER, np.ceil(orders * growth).astype(np.int64))
        return grown if np.all(grown > orders) else None

    # The coarse and fine moments of a chunk ([n, 2 or 8, count] each) stay
    # within the scratch budget as well.
    chunk = max(1, int(NEAR_KERNEL_WORK_BYTES // (2 * 8 * (8 if bracket else 2) * full)))
    for start in range(0, len(members), chunk):
        _refine_near_chunk(kind, arrays, np.asarray(members[start:start + chunk]), stable,
                           base_orders, evaluate, next_orders, out, int(start_order))


def _near_check_stride():
    raw = os.environ.get('GHOST_BOR_NEAR_CHECK_STRIDE', '').strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return int(NEAR_CHECK_STRIDE)


def _refine_near_chunk(kind, arrays, ids, stable, base_orders, evaluate, next_orders, out, offset=0):
    """Coarse/fine levels of one chunk of a layout group (_run_near_group).

    ``offset`` is the first order kept: the evaluated moments cover every
    order from 0 and only ``[offset:]`` is stored.  The fine level is
    evaluated first for every point; with a positive NEAR_CHECK_STRIDE the
    coarse level is evaluated for every stride-th point and the chunk is
    accepted when all of them pass, otherwise every point is checked."""
    bracket = kind != 'g'
    use_stable = stable
    coarse_orders = np.asarray(base_orders, dtype=np.int64)
    fine_orders = next_orders(coarse_orders)
    if fine_orders is None:
        raise ValueError("BoR near angular quadrature exceeds its accuracy limit; refine the mesh or reduce modal bandwidth.")
    fine = evaluate(ids, fine_orders, use_stable)
    stride = _near_check_stride()
    if stride > 0 and len(ids) >= 2 * stride:
        probe = np.arange(0, len(ids), stride)
        coarse_probe = evaluate(ids[probe], coarse_orders, use_stable)
        fine_probe = (fine[0][probe], fine[1][probe]) if bracket else fine[probe]
        error, scale = _near_check(kind, fine_probe, coarse_probe)
        if np.all(np.isfinite(error) & (error <= NEAR_ANGULAR_RTOL * np.maximum(scale, 1e-280))):
            if bracket:
                out[0][ids] = fine[0][:, :, offset:]
                out[1][ids] = fine[1][:, :, offset:]
            else:
                out[ids] = 2.0 * (fine[:, 0][:, offset:] + 1j * fine[:, 1][:, offset:])
            return
    coarse = evaluate(ids, coarse_orders, use_stable)
    while True:
        error, scale = _near_check(kind, fine, coarse)
        converged = np.isfinite(error) & (error <= NEAR_ANGULAR_RTOL * np.maximum(scale, 1e-280))
        accepted = ids[converged]
        if bracket:
            out[0][accepted] = fine[0][converged][:, :, offset:]
            out[1][accepted] = fine[1][converged][:, :, offset:]
        else:
            out[accepted] = 2.0 * (fine[:, 0][converged][:, offset:] + 1j * fine[:, 1][converged][:, offset:])
        if np.all(converged):
            return
        pending = ~converged
        ids = ids[pending]
        coarse = (fine[0][pending], fine[1][pending]) if bracket else fine[pending]
        error, scale = error[pending], scale[pending]
        grown = next_orders(fine_orders)
        if grown is None:
            if bracket and not use_stable:
                # No valid comparison remains for the sampled forms: restart
                # the pending points with the cancellation-free forms.
                use_stable = True
                coarse_orders = np.asarray(base_orders, dtype=np.int64)
                fine_orders = next_orders(coarse_orders)
                coarse = evaluate(ids, coarse_orders, use_stable)
                fine = evaluate(ids, fine_orders, use_stable)
                continue
            worst = int(np.argmax(error / np.maximum(scale, 1e-280)))
            coordinates = tuple(float(v[ids[worst]]) for v in arrays)
            name = {'g': 'modal_kernels_near', 'mfie': 'mfie_kernels_near',
                    'ibc': 'ibc_kernels_near'}[kind]
            raise ValueError(
                f"BoR near angular quadrature did not converge at the maximum order: "
                f"{name}, point={coordinates}, relative change="
                f"{float(error[worst] / max(scale[worst], 1e-280)):.3g}.")
        coarse_orders, fine_orders = fine_orders, grown
        fine = evaluate(ids, fine_orders, use_stable)


def modal_kernels_near(rho_p, z_p, rho_q, z_q, k, m_max: 'int', order: 'int' = 0,
                       tail_order: 'int' = 0, mode_start: 'int' = 0):
    """G_m, m = 0..m_max+1, at near point pairs by the checked graded rule.

    ``order``/``tail_order`` (> 0) are optional lower bounds of the fine
    level's core and tail Gauss orders.  Returns [n_pairs, m_max+2].
    """
    return _graded_near_kernels('g', (rho_p, z_p, rho_q, z_q), k, m_max, order, tail_order,
                                mode_start=mode_start)


def mfie_kernels_near(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                      m_max: 'int', order: 'int' = 0, tail_order: 'int' = 0,
                      signed: 'bool' = True, mode_start: 'int' = 0):
    """Four MFIE modal kernels of near point pairs by the checked graded rule:
    [n_pairs, 2*m_max+1] (m = -m_max..m_max), or [n_pairs, m_max+1]
    (m = 0..m_max) with ``signed=False``."""
    return _graded_near_kernels('mfie', (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q),
                                k, m_max, order, tail_order, signed, mode_start)


def ibc_kernels_near(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                     m_max: 'int', order: 'int' = 0, tail_order: 'int' = 0,
                     signed: 'bool' = True, mode_start: 'int' = 0):
    """Four IBC modal kernels of near point pairs by the checked graded rule
    (layout as ``mfie_kernels_near``)."""
    return _graded_near_kernels('ibc', (rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q),
                                k, m_max, order, tail_order, signed, mode_start)


def gc_gs_from_g(G: 'np.ndarray', m: 'int') -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Gc_m and Gs_m from the table G[..., 0..m_max+1] (m >= 0 entries;
    negative orders via G_{-n} = G_n).

    Gc_m = (G_{m-1} + G_{m+1})/2       Gs_m = (G_{m-1} - G_{m+1})/(2j)
    """

    gm_m1 = G[..., abs(m - 1)]
    gm_p1 = G[..., m + 1]
    return 0.5 * (gm_m1 + gm_p1), (gm_m1 - gm_p1) / 2j


def kernels_for_mode(G: 'np.ndarray', m: 'int') -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """(G_m, Gc_m, Gs_m) for any integer m (negative handled by symmetry:
    G_{-m} = G_m, Gc_{-m} = Gc_m, Gs_{-m} = -Gs_m)."""

    am = abs(m)
    gc, gs = gc_gs_from_g(G, am)
    if m < 0:
        gs = -gs
    return G[..., am], gc, gs
