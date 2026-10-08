"""Bounded modal kernel assembly with NumPy or native C sampling."""
from ghost_backend.execution.paths import native_kernel_root

import ctypes
import errno
import json
import mmap
import os
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from ghost_backend.bor.kernels import _mfie_brackets, _ibc_brackets_grid
import ghost_backend.bor.kernels as modal_kernels
from ghost_backend.bor.near_storage import mode_sign
from ghost_backend.bor.options import BorAdmissionError


def _efie_terms(green, modes, offset, k):
    """Combine adjacent angular orders before retaining nodal EFIE blocks."""
    center = green[..., modes-offset]
    lower = green[..., np.abs(modes-1)-offset]
    upper = green[..., modes+1-offset]
    cosine, sine = (lower+upper)*.5, (lower-upper)/(2j)
    yield 0, 'r', 'r', cosine
    yield 0, 'z', 'z', center
    yield 1, 'r', '1', sine
    yield 2, '1', 'r', -sine
    yield 3, '1', '1', cosine
    yield 0, 'd', 'd', -center/k**2
    yield 1, 'd', 's', -(1j*modes/k**2)*center
    yield 2, 's', 'd', (1j*modes/k**2)*center
    yield 3, 's', 's', -(modes**2/k**2)*center


def _source_weights(rv, ne: 'int', go: 'int') -> 'np.ndarray':
    """Source-basis weights ``[2, P]`` as the ``[ne, go, 2]`` batch a right
    contraction multiplies."""
    return np.ascontiguousarray(np.asarray(rv).reshape(2, ne, go).transpose(1, 2, 0))


def _real_by_complex(values) -> 'bool':
    """Whether a complex operand can be multiplied by real weights as the real
    view of its interleaved (re, im) pairs (its last axis is contiguous).

    NumPy would promote the real weights and form a complex product, half of
    whose multiplications are by exact zeros; the real GEMM over the view forms
    the same sums of products with half the operations and no copy.
    """
    return (values.dtype == np.complex128 and values.ndim >= 1
            and values.strides[-1] == values.itemsize)


def _contract_test_side(kernel, left, re: 'int', go: 'int') -> 'np.ndarray':
    """``kernel [re*go, P, no]`` x ``left [nk, re, go]`` -> ``[re, nk, P, no]``.

    One GEMM per test element over its Gauss rows, for every left basis kind
    at once: a single memory-bound pass over the sampled tile.
    """
    P, no = kernel.shape[1], kernel.shape[2]
    weights = np.ascontiguousarray(left.transpose(1, 0, 2))
    if not np.iscomplexobj(weights) and kernel.flags.c_contiguous and _real_by_complex(kernel):
        product = np.matmul(weights, kernel.view(np.float64).reshape(re, go, 2 * P * no))
        return product.view(np.complex128).reshape(re, left.shape[0], P, no)
    product = np.matmul(weights, kernel.reshape(re, go, P * no))
    return product.reshape(re, left.shape[0], P, no)


def _contract_source_group(values, stacked) -> 'np.ndarray':
    """``values [re, fc*go, no]`` x ``stacked [fc, w, go]`` -> ``[fc, re, w, no]``.

    ``stacked[f]`` holds the ``w`` source weights (right kinds x basis
    functions) of source element ``f`` over its Gauss points.  One small
    product per (source, test) element pair reads the tested tile in place --
    no transposed copy -- and leaves the orders contiguous for the
    adjacent-order combinations.
    """
    re, _, no = values.shape
    fc, go = stacked.shape[0], stacked.shape[2]
    if not np.iscomplexobj(stacked) and _real_by_complex(values):
        real = values.view(np.float64).reshape(re, fc, go, 2 * no).transpose(1, 0, 2, 3)
        return np.matmul(stacked[:, None], real).view(np.complex128)
    return np.matmul(stacked[:, None], values.reshape(re, fc, go, no).transpose(1, 0, 2, 3))


# The nine EFIE Galerkin terms: block, left kind, right kind, kernel combination
# (center G_m, cosine (G_{m-1}+G_{m+1})/2, sine (G_{m-1}-G_{m+1})/2j) and the
# mode-dependent coefficient (see _efie_terms).
_EFIE_TERM_TABLE = (
    (0, "r", "r", "cos", None),
    (0, "z", "z", "cen", None),
    (1, "r", "1", "sin", None),
    (2, "1", "r", "-sin", None),
    (3, "1", "1", "cos", None),
    (0, "d", "d", "cen", "-1"),
    (1, "d", "s", "cen", "-jm"),
    (2, "s", "d", "cen", "+jm"),
    (3, "s", "s", "cen", "-m2"),
)
_LEFT_KINDS = ("r", "z", "1", "s", "d")
_LEFT_INDEX = {name: index for index, name in enumerate(_LEFT_KINDS)}
# The right kinds each left kind meets in the nine terms: one source-side
# product per (left kind, basis function) serves all of them.
_EFIE_SOURCE_GROUPS = (("r", ("r", "1")), ("z", ("z",)), ("1", ("r", "1")),
                       ("d", ("d", "s")), ("s", ("d", "s")))
_EFIE_TERMS_BY_PAIR = {(lx, rx): (uv, kernel, coefficient)
                       for uv, lx, rx, kernel, coefficient in _EFIE_TERM_TABLE}


def _stacked_right(*weights) -> 'np.ndarray':
    """Source weights ``[ne, go, 2]`` stacked as ``[ne, 2*n, go]`` (index ``2*i + b``)."""
    return np.ascontiguousarray(
        np.concatenate([np.asarray(value).transpose(0, 2, 1) for value in weights], axis=1))


def _stacked_right_groups(right) -> 'Dict[str, np.ndarray]':
    """``right[kind] [ne, go, 2]`` stacked per left kind (see _EFIE_SOURCE_GROUPS)."""
    return {lx: _stacked_right(*(right[rx] for rx in kinds))
            for lx, kinds in _EFIE_SOURCE_GROUPS}


def far_weights(g, ne: 'int', go: 'int', ibc_zs_pt=None, pmchwt: 'bool' = False) -> 'Dict':
    """Galerkin weights of the far contractions of one surface.

    ``lv`` holds the test weights ``[2, P]`` of each kind (the two functions
    of every element), ``left_all`` the kinds stacked for one test-side GEMM
    per tile, and the ``right*`` entries the source weights arranged for the
    batched source-side products (:func:`_stacked_right_groups`).  The
    rotated-PV source weight carries the IBC impedance ``ibc_zs_pt`` or, for
    PMCHWT, one.
    """
    if ibc_zs_pt is not None and pmchwt:
        raise ValueError(
            "Streaming rotated-PV blocks cannot be both IBC-weighted "
            "and unit-weight PMCHWT blocks."
        )
    wrho = g.w * g.rho
    lv = {
        "r": np.stack([g.T0 * wrho * g.trho, g.T1 * wrho * g.trho]),
        "z": np.stack([g.T0 * wrho * g.tz, g.T1 * wrho * g.tz]),
        "1": np.stack([g.T0 * wrho, g.T1 * wrho]),
        "s": np.stack([g.T0 * g.w, g.T1 * g.w]),
        "d": np.stack([g.dRT0 * g.w, g.dRT1 * g.w]),
    }
    rv_ibc = None
    if ibc_zs_pt is not None:
        rv_ibc = np.stack([g.T0 * wrho * ibc_zs_pt, g.T1 * wrho * ibc_zs_pt])
    elif pmchwt:
        rv_ibc = np.stack([g.T0 * wrho, g.T1 * wrho])
    right = {name: _source_weights(lv[name], ne, go) for name in _LEFT_KINDS}
    right_ibc = None if rv_ibc is None else _source_weights(rv_ibc, ne, go)
    return dict(lv=lv, rv_ibc=rv_ibc,
                left_all=np.stack([lv[name] for name in _LEFT_KINDS], axis=0),
                right=right, right_groups=_stacked_right_groups(right),
                right_one=_stacked_right(right["1"]), right_ibc=right_ibc,
                right_ibc_stacked=None if right_ibc is None else _stacked_right(right_ibc))


def _efie_band(Gn, left, right_groups, modes, ord_lo: 'int', k, f0: 'int', f1: 'int',
               re: 'int', go_p: 'int') -> 'np.ndarray':
    """Nodal EFIE contributions of one tile summed in ``[4, fc + 1, re + 1, nm]``.

    ``Gn`` holds the tile's G orders ``[re*go_p, fc*go_q, no]`` with its
    excluded pairs zero, ``left`` the ten stacked test weights ``[10, re,
    go_p]`` and ``right_groups[lx]`` the source weights of the right kinds
    paired with ``lx`` (:func:`_stacked_right_groups`).  Test side first (one
    GEMM per test element for all ten left weights), then per left kind the
    two test functions of each node are summed (node ``i`` is element ``i``'s
    first and element ``i - 1``'s second function) and one batched
    source-side product serves the whole tile.  Each term sums its source
    node's two functions in order space; its adjacent-order combination and
    mode coefficient then act once, in place, on the small nodal result:
    half the source-side products and a quarter of the combinations of
    assembling each element pair's four basis-function pairs, with the same
    sums to rounding.  Band
    entry ``[uv, j, i, m]`` belongs to the node pair ``(e0 + i, f0 + j)``: a
    tile is summed locally and added to the shared blocks once, under the
    accumulation lock (:func:`_add_band_to_store`).
    """
    modes = np.asarray(modes)
    nm = len(modes)
    fc = int(f1) - int(f0)
    c0 = int(modes[0]) - ord_lo
    lower_index = np.abs(modes - 1) - ord_lo
    coefficients = {
        None: None,
        "-1": -1.0 / k ** 2,
        "-jm": -(1j * modes / k ** 2),
        "+jm": (1j * modes / k ** 2),
        "-m2": -(modes.astype(float) ** 2 / k ** 2),
    }
    tested = _contract_test_side(Gn, left, re, go_p)            # [re, 10, fc*go_q, no]
    band = np.zeros((4, fc + 1, re + 1, nm), dtype=np.complex128)
    nodal = np.empty((re + 1,) + tested.shape[2:], dtype=np.complex128)
    sums = np.empty((fc + 1, re + 1, tested.shape[3]), dtype=np.complex128)
    for lx, rights in _EFIE_SOURCE_GROUPS:
        column = 2 * _LEFT_INDEX[lx]
        nodal[:re] = tested[:, column]
        nodal[re] = 0.0
        nodal[1:] += tested[:, column + 1]
        product = _contract_source_group(nodal, right_groups[lx][f0:f1])   # [fc, re+1, 2n, no]
        for ri, rx in enumerate(rights):
            uv, kernel, coefficient = _EFIE_TERMS_BY_PAIR[(lx, rx)]
            sums[:fc] = product[:, :, 2 * ri]
            sums[fc] = 0.0
            sums[1:] += product[:, :, 2 * ri + 1]
            if kernel == "cen":
                piece = sums[..., c0:c0 + nm]
            else:
                piece = sums[..., lower_index]
                upper = sums[..., c0 + 1:c0 + nm + 1]
                if kernel == "cos":
                    piece += upper
                    piece *= 0.5
                elif kernel == "sin":                           # (lower - upper) / 2j
                    piece -= upper
                    piece *= -0.5j
                else:                                           # (upper - lower) / 2j
                    piece -= upper
                    piece *= 0.5j
                upper = None
            scale = coefficients[coefficient]
            if scale is not None:
                piece *= scale
            band[uv] += piece
            piece = None
        product = None
    return band


def _bracket_band(Fs, left, right, f0: 'int', f1: 'int', re: 'int', go_p: 'int',
                  zero_near=None) -> 'np.ndarray':
    """Four bracket kernels ``[re*go_p, fc*go_q, nm]`` of one tile summed in
    ``[4, fc + 1, re + 1, nm]`` (``left [2, re, go_p]``, ``right`` the source
    weights stacked by :func:`_stacked_right`).  As in :func:`_efie_band`, a
    node's two test functions are summed before the one source-side product
    of each kernel."""
    fc = int(f1) - int(f0)
    nm = Fs[0].shape[-1]
    band = np.zeros((4, fc + 1, re + 1, nm), dtype=np.complex128)
    stacked = right[f0:f1]
    nodal = None
    for uv, kernel in enumerate(Fs):
        if zero_near is not None:
            zero_near(kernel)
        tested = _contract_test_side(kernel, left, re, go_p)       # [re, 2, fc*go_q, nm]
        if nodal is None:
            nodal = np.empty((re + 1,) + tested.shape[2:], dtype=np.complex128)
        nodal[:re] = tested[:, 0]
        nodal[re] = 0.0
        nodal[1:] += tested[:, 1]
        tested = None
        product = _contract_source_group(nodal, stacked)            # [fc, re+1, 2, nm]
        band[uv, :fc] += product[:, :, 0]
        band[uv, 1:] += product[:, :, 1]
        product = None
    return band


def _add_band_to_store(store, band, e0: 'int', f0: 'int') -> 'None':
    """``store[uv, m, e0 + i, f0 + j] += band[uv, j, i, m]`` (lock held by the caller)."""
    store[:, :, e0:e0 + band.shape[2], f0:f0 + band.shape[1]] += band.transpose(0, 3, 2, 1)


# Build the far EFIE blocks from element pairs e < f and complete them by
# symmetry (see _symmetrize_efie_mode); False samples every pair and keeps
# full blocks.  Completion blocks of 128 x 128 complex values stay in the L2
# cache through their transposed reads (74 ms per 2,161-node mode against
# 130 ms at 512); an element's value does not depend on the blocking.
STREAM_EFIE_SYMMETRY = True
STREAM_SYMMETRY_BLOCK = 128


def _symmetrize_efie_mode(tt, tf, ft, ff, block: 'int' = STREAM_SYMMETRY_BLOCK) -> 'None':
    """Complete the four EFIE blocks of one mode accumulated from element pairs ``e < f``.

    G is symmetric in its two points and the test and source weights of one
    surface coincide, so a pair ``(f, e)`` contributes the transpose of pair
    ``(e, f)`` (tt, ff) or its negated transpose across tf/ft.  With U the
    accumulated upper pairs: ``Z_tt = U_tt + U_tt^T``, ``Z_ff = U_ff +
    U_ff^T``, ``Z_tf = U_tf - U_ft^T``, ``Z_ft = U_ft - U_tf^T`` (self pairs
    are near and absent).  Square node blocks bound the temporaries.  The
    arguments may be strided views (the quadrants of a system matrix).
    """
    n = tt.shape[-1]
    step = max(1, int(block))
    for i0 in range(0, n, step):
        I = slice(i0, min(i0 + step, n))
        for j0 in range(i0, n, step):
            J = slice(j0, min(j0 + step, n))
            for symmetric in (tt, ff):
                upper = symmetric[I, J] + symmetric[J, I].T
                symmetric[I, J] = upper
                symmetric[J, I] = upper.T
            a, b = tf[I, J].copy(), ft[J, I].copy()
            if j0 == i0:
                tf[I, J] = a - b.T
                ft[J, I] = b - a.T
                continue
            c, d = tf[J, I].copy(), ft[I, J].copy()
            tf[I, J] = a - b.T
            ft[J, I] = b - a.T
            tf[J, I] = c - d.T
            ft[I, J] = d - c.T


def _symmetrize_efie_blocks(Z, block: 'int' = STREAM_SYMMETRY_BLOCK) -> 'None':
    """:func:`_symmetrize_efie_mode` for every mode of a full ``[4, modes, n, n]`` store."""
    for mi in range(Z.shape[1]):
        _symmetrize_efie_mode(Z[0, mi], Z[1, mi], Z[2, mi], Z[3, mi], block)


# Symmetric self-surface EFIE streams keep only the strictly upper triangle of
# each of their four accumulated blocks, packed row by row: the near stencil
# excludes every element pair less than three elements apart, so an element
# pair ``e < f`` only ever reaches node pairs of row < column and the diagonal
# and lower triangles of U are exactly zero.  A mode is unpacked straight into
# the caller's system quadrants and completed there (_symmetrize_efie_mode):
# half the retained (or spilled) EFIE storage, and no pass over the whole store
# after the build -- on a spilled store that exceeded the free RAM that pass
# paged the file at a few hundred MB/s (213 s of a 688 s certified solve).
def _packed_row_offsets(n: 'int') -> 'np.ndarray':
    """Start of row ``i`` (columns ``i+1..n-1``) in a packed strict-upper triangle."""
    rows = np.arange(n, dtype=np.int64)
    return rows * (n - 1) - rows * (rows - 1) // 2


def packed_upper_size(n: 'int') -> 'int':
    """Entries of the packed strict-upper triangle of an ``n x n`` block."""
    n = int(n)
    return n * (n - 1) // 2


def _add_band_to_packed(store, band, e0: 'int', f0: 'int', offsets) -> 'None':
    """``store[uv, m, packed(e0 + i, f0 + j)] += band[uv, j, i, m]`` for ``f0 + j > e0 + i``.

    ``band`` entries on or below the diagonal are zero by construction (the
    symmetric build masks every pair ``f <= e`` and near pairs are excluded).
    """
    n_cols = band.shape[1]
    n = offsets.shape[0]
    for i in range(band.shape[2]):
        row = int(e0) + i
        first = max(int(f0), row + 1)
        stop = min(int(f0) + n_cols, n)
        if stop <= first:
            continue
        start = int(offsets[row]) + (first - row - 1)
        store[:, :, start:start + (stop - first)] += (
            band[:, first - f0:stop - f0, i, :].transpose(0, 2, 1))


# Row chunks of blocks up to this many nodes keep the (row, column) index pair
# of their strict-upper positions (16 bytes per position) beside the mask.
UNPACK_INDEX_MAX_NODES = 3072


def _unpack_upper_into(packed, out, factor, offsets, rows=None, mask=None) -> 'None':
    """``out = factor * U`` for the packed strict-upper ``U``; the rest of ``out`` is zeroed
    (the node rows ``rows = (lo, hi)`` only, when given).

    The upper positions of a row chunk, in row-major order, are exactly its
    packed segment, so one assignment fills them (the per-row copies it
    replaces were two Python-level calls per row, serialized on the GIL
    across mode workers).  ``mask`` is ``_upper_mask(n, lo, hi)`` when cached,
    or ``(mask, (row_index, column_index))`` with the mask's nonzero positions
    precomputed: a boolean assignment scans the whole mask on every mode,
    an indexed one only writes.
    """
    n = out.shape[0]
    lo, hi = (0, n) if rows is None else rows
    if hi <= lo:
        return
    start = int(offsets[lo])
    stop = int(offsets[hi]) if hi < n else packed_upper_size(n)
    block = out[lo:hi]
    block[...] = 0.0
    if stop > start:
        indices = None
        if isinstance(mask, tuple):
            mask, indices = mask
        if indices is not None:
            block[indices[0], indices[1]] = packed[start:stop] * factor
        else:
            block[_upper_mask(n, lo, hi) if mask is None else mask] = packed[start:stop] * factor


def _upper_mask(n: 'int', lo: 'int', hi: 'int') -> 'np.ndarray':
    """Strict-upper positions of the node rows ``[lo, hi)`` of an ``n x n`` block."""
    return np.arange(n)[None, :] > np.arange(lo, hi)[:, None]


def _row_chunks(n_rows: 'int', row_bytes: 'int', chunk_rows: 'int' = 0):
    """Node-row chunks of about STREAM_SPILL_RELEASE_BYTES (or ``chunk_rows``)."""
    step = int(chunk_rows) if chunk_rows else max(1, STREAM_SPILL_RELEASE_BYTES // max(1, int(row_bytes)))
    return [(r0, min(r0 + step, n_rows)) for r0 in range(0, n_rows, step)]


def _symmetric_near_relation(near_sources) -> 'bool':
    """True when every element is near itself and the near relation is symmetric."""
    sets = [set(int(f) for f in values) for values in near_sources]
    for e, values in enumerate(sets):
        if e not in values:
            return False
        for f in values:
            if not 0 <= f < len(sets) or e not in sets[f]:
                return False
    return True


def _adjacent_pairs_near(near_sources) -> 'bool':
    """True when every element is near its neighbours (``|e - f| <= 1``).

    A far pair ``e < f`` then has ``f >= e + 2``: its node pairs ``(e + a,
    f + b)`` all lie strictly above the diagonal (packed EFIE storage).
    """
    count = len(near_sources)
    for e, values in enumerate(near_sources):
        near = set(int(f) for f in values)
        if any(f not in near for f in range(max(0, e - 1), min(count, e + 2))):
            return False
    return True


def _banded_stream(stream, rows, kind, modes, sources=None, strict_upper=False,
                   near_free=False):
    """Banded far kernels of the test rows ``rows`` against source elements.

    ``sources`` is the half-open source-element range ``(f0, f1)`` (all when
    None).  Near element pairs, and with ``strict_upper`` every pair whose
    source element does not follow its test element (``f <= e``), are masked:
    never sampled and returned as zeros.  ``near_free`` asserts the caller has
    checked that no pair of the tile is near (the mask is then not built).
    """
    if hasattr(stream, 'solver'):
        sp = sq = stream.solver
        near_sources = sp._near_sources_by_element
    else:
        sp, sq, near_sources = stream.sp, stream.sq, stream._near_sources
    gp, gq = sp.g, sq.g
    go_p, go_q = sp.gauss_order, sq.gauss_order
    first, stop, _ = rows.indices(sp.P)
    f0, f1 = (0, sq.gen.n_elems) if sources is None else (int(sources[0]), int(sources[1]))
    c0, c1 = f0 * go_q, f1 * go_q
    if near_free and not strict_upper:
        near = np.zeros((1, 1), bool)
    else:
        near = np.zeros((stop - first, c1 - c0), bool)
        for e in range(first // go_p, (stop + go_p - 1) // go_p):
            r = slice(max(0, e * go_p - first), min(stop - first, (e + 1) * go_p - first))
            for f in near_sources[e]:
                if f0 <= f < f1:
                    near[r, (f - f0) * go_q:(f - f0 + 1) * go_q] = True
            if strict_upper and e >= f0:
                near[r, :(min(e + 1, f1) - f0) * go_q] = True
    names = ('rho', 'z') if kind == 'g' else ('rho', 'z', 'trho', 'tz')
    args = tuple(getattr(gp, name)[rows, None] for name in names)
    args += tuple(getattr(gq, name)[None, c0:c1] for name in names)
    work = getattr(stream, '_work_bytes', None)
    if work is None:
        work = modal_kernels.FFT_BUILD_BUDGET / max(1, getattr(stream, '_workers', 1))
    result = modal_kernels.banded_modal_kernels(
        kind, args, stream.k, stream.m_max, near, modes,
        work_bytes=work, threads=getattr(stream, '_native_threads', 1))
    if kind == 'g':
        return result
    for value in result:
        value *= 2 * np.pi          # in place: the 2 pi Galerkin factor without a copy
    return result


def _run_tiles(tiles, do_tile, threads: 'int') -> 'None':
    """Run ``do_tile`` over ``tiles`` on up to ``threads`` threads.

    Concurrent tiles make their products on one BLAS thread each
    (:func:`execution.options.single_thread_blas`): a range rebuilt inside a
    mode worker otherwise inherits that worker's multithreaded BLAS share,
    and OpenBLAS faulted under eight tiles' concurrent multithreaded calls.
    """
    from ghost_backend.execution.options import single_thread_blas
    tiles = list(tiles)
    count = min(max(1, int(threads)), len(tiles))
    if count <= 1:
        for tile in tiles:
            do_tile(tile)
        return
    with single_thread_blas(), ThreadPoolExecutor(max_workers=count) as pool:
        for _ in pool.map(do_tile, tiles):
            pass


_DLL_DIRECTORY_HANDLES = []
_DLL_DIRECTORIES_ADDED = set()


def _add_dll_directories(candidates) -> 'bool':
    """Add existing, not yet added directories to the DLL search; True if any."""
    added = False
    for directory in candidates:
        resolved = os.path.normcase(os.path.abspath(directory))
        if resolved in _DLL_DIRECTORIES_ADDED or not os.path.isdir(resolved):
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
        except OSError:
            continue
        _DLL_DIRECTORIES_ADDED.add(resolved)
        added = True
    return added


def _windows_dll_search_supported() -> 'bool':
    return platform.system().lower() == "windows" and hasattr(os, "add_dll_directory")


def _prepare_windows_dll_search() -> 'None':
    """Expose only the kernel's own directory to the Windows DLL search.

    The released kernel links its GCC/OpenMP runtimes statically and imports
    nothing but KERNEL32 and the UCRT API sets, so no compiler directory needs
    to join the process-wide search path (where it could shadow other
    libraries' dependencies).  :func:`_prepare_windows_fallback_dll_search`
    widens the search only after a load failure, for non-static builds.
    """

    if not _windows_dll_search_supported():
        return
    _add_dll_directories([str(native_kernel_root())])


def _prepare_windows_fallback_dll_search() -> 'bool':
    """Compiler-runtime directories for a non-static build that failed to load.

    ``GHOST_NATIVE_DLL_DIR`` first, then the directories of the compilers on
    PATH and the standard MSYS2 locations.  True when a directory was added,
    so that the caller retries the load once.
    """

    if not _windows_dll_search_supported():
        return False
    candidates = []
    configured = os.environ.get("GHOST_NATIVE_DLL_DIR", "").strip()
    if configured:
        candidates.append(configured)
    for compiler_name in ("gcc", "cc", "clang"):
        compiler = shutil.which(compiler_name)
        if compiler:
            candidates.append(os.path.dirname(os.path.abspath(compiler)))
    candidates.extend([
        r"C:\msys64\ucrt64\bin",
        r"C:\msys64\mingw64\bin",
    ])
    return _add_dll_directories(candidates)


def _native_extensions(system_name: 'str') -> 'tuple[str, ...]':
    """Shared-library suffixes that the current host can safely load."""

    key = str(system_name).strip().lower()
    if key == "windows":
        return (".dll",)
    return (".so",)


def _load_library(path):
    """``ctypes.CDLL(path)``; after a failure on Windows, retry once with the
    compiler-runtime fallback directories (a non-static build)."""
    try:
        return ctypes.CDLL(path)
    except OSError:
        if not _prepare_windows_fallback_dll_search():
            raise
    return ctypes.CDLL(path)


def _load_native():
    _prepare_windows_dll_search()
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    here = str(native_kernel_root())
    for base in (f"bor_stream_kernel.{sysname}-{machine}", "bor_stream_kernel"):
        for extension in _native_extensions(sysname):
            path = os.path.join(here, base + extension)
            if not os.path.exists(path):
                continue
            try:
                lib = _load_library(path)
            except OSError:
                continue
            if not all(
                hasattr(lib, symbol)
                for symbol in ("sample_g", "sample_mfie", "sample_ibc")
            ):
                continue
            dp = ctypes.POINTER(ctypes.c_double)
            ci = ctypes.c_int
            cd = ctypes.c_double
            lib.sample_g.argtypes = [ci, ci, ci, dp, dp, dp, dp, cd, dp, dp]
            lib.sample_g.restype = None
            bracket_args = [ci, ci, ci] + [dp] * 8 + [cd, dp, dp] + [dp] * 4
            lib.sample_mfie.argtypes = bracket_args
            lib.sample_mfie.restype = None
            lib.sample_ibc.argtypes = bracket_args
            lib.sample_ibc.restype = None
            if hasattr(lib, "near_mfie"):
                # Paired near sampler; absent from kernels built before it existed.
                lib.near_mfie.argtypes = (
                    [ci, ci] + [dp] * 8 + [cd, dp, ci] + [dp] * 4
                )
                lib.near_mfie.restype = None
            if hasattr(lib, "sample_g_pairs"):
                # Grouped half-grid samplers (real or complex k, OpenMP team size).
                lib.sample_g_pairs.argtypes = [ci, ci, dp, dp, dp, dp, cd, cd, dp, dp, ci]
                lib.sample_g_pairs.restype = None
            if hasattr(lib, "sample_brackets_pairs"):
                lib.sample_brackets_pairs.argtypes = (
                    [ci, ci, ci] + [dp] * 8 + [cd, cd, dp, dp] + [dp] * 4 + [ci]
                )
                lib.sample_brackets_pairs.restype = None
            if hasattr(lib, "trig_moments"):
                # Near-rule angular projections by the rotation recurrence.
                lib.trig_moments.argtypes = [ci, ci, ci, ci, dp, dp, dp, ctypes.c_void_p, ci]
                lib.trig_moments.restype = None
            return lib
    return None


_NATIVE = _load_native()
_FALLBACK_NOTICE_SHOWN = False


def sampling_backend_name(stream=None) -> 'str':
    """Auditable backend label for streamed far-kernel sampling."""

    if modal_kernels.BANDED_FFT:
        if (_NATIVE is not None and hasattr(_NATIVE, 'sample_g_pairs')
                and hasattr(_NATIVE, 'sample_brackets_pairs')):
            return "banded_native_pairs"
        native_brackets = (_NATIVE is not None and
            (hasattr(_NATIVE, 'near_brackets') or
             (hasattr(_NATIVE, 'near_mfie') and (stream is None or complex(stream.k).imag == 0))))
        return "banded_numpy_native_brackets" if native_brackets else "banded_numpy"
    native = _NATIVE if stream is None else stream._native
    return "native_c" if native is not None else "numpy"


def _notice_numpy_fallback():
    """One-time stderr notice when the streaming build runs on the NumPy
    sampler.  Results are bit-equivalent; assembly is ~2-8x slower.  It also
    diagnoses a native binary copied from the wrong operating system, which
    the loader correctly refuses to load."""
    global _FALLBACK_NOTICE_SHOWN
    if _FALLBACK_NOTICE_SHOWN:
        return
    _FALLBACK_NOTICE_SHOWN = True
    here = str(native_kernel_root())
    system_name = platform.system().lower()
    tag = f"{system_name}-{platform.machine().lower()}"
    others = [f for f in sorted(os.listdir(here))
              if f.startswith("bor_stream_kernel.")
              and os.path.splitext(f)[1].lower() in {".so", ".dll"}
              and tag not in f]
    hint = (f" (found {', '.join(others)} -- built for a DIFFERENT platform, "
            "so it was correctly skipped)" if others else "")
    print(
        "bor_streaming: native sampling kernel not available for this "
        f"platform{hint}; using the NumPy fallback (bit-equivalent, ~2-8x "
        "slower assembly). Compile and load-check it on THIS machine with:\n"
        "  py ghost_backend/bor/native/build_kernel.py",
        file=sys.stderr, flush=True)


def _dp(a: 'np.ndarray'):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


from ghost_backend.bor.kernels import n_xi_for_pairs


BOR_STREAM_TILE_BUDGET_GB = 1.0


BOR_STREAM_TILE_BYTES_PER_SAMPLE = 256.0

# Allocator slack and untracked temporaries on the modeled live set of a
# banded tile (_banded_tile_bytes counts every array of a tile's phases).
BOR_STREAM_TILE_SLACK = 1.25
# Per sampled point pair in the banded sampler: broadcast coordinates (up to
# eight float64), the exclusion mask, the active-pair index and the per-pair
# radius/gap/rule arrays with their temporaries.
_BANDED_PAIR_BYTES = 192.0


def streaming_tile_threads() -> 'int':
    """Concurrent far-block tiles of one streamed build.

    The physical cores of this process's allocation
    (:func:`ghost_backend.bor.kernels.physical_cpu_count`), independent of the
    outer mode workers: sampling, contraction and accumulation are memory
    bound, so SMT threads add nothing, and the mode workers only set the
    retained range alignment.  Every concurrent tile is sized to its share of
    the tile budget, so the priced ``BOR_STREAM_TILE_BUDGET_GB`` covers the
    live tiles of any thread count.
    """
    return max(1, int(modal_kernels.physical_cpu_count()))


def _aligned_stream_mode_block(
    m_max: 'int', mode_block: 'Optional[int]', workers: 'int'
) -> 'int':
    """Return the exact range size shared by planning and runtime."""

    mode_count = int(m_max) + 1
    worker_count = max(1, int(workers))
    if mode_count < 1:
        raise ValueError("Streaming mode maximum must be non-negative.")
    requested = mode_count if mode_block is None else int(mode_block)
    if requested < 1:
        raise ValueError("Streaming mode block must be positive.")
    requested = max(requested, worker_count)
    aligned = ((requested + worker_count - 1) // worker_count) * worker_count
    return min(aligned, mode_count)


def _streaming_worker_count(
    gauss_order: 'int', n_xi: 'int', tile_budget_gb: 'float', workers: 'int'
) -> 'int':
    """Cap simultaneous sampling tiles when the one-column floor requires it."""

    go = int(gauss_order)
    samples = int(n_xi)
    requested = max(1, int(workers))
    budget = float(tile_budget_gb) * 1.0e9
    if go < 1 or samples < 1:
        raise ValueError("Streaming tile dimensions must be positive.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming tile budget must be positive and finite.")
    one_tile = go * samples * BOR_STREAM_TILE_BYTES_PER_SAMPLE
    if budget < one_tile:
        raise ValueError(
            "Streaming tile budget is below the modeled one-element, "
            f"one-column minimum of {one_tile / 1.0e9:.6g} GB "
            f"(gauss_order={go}, n_xi={samples})."
        )
    return min(requested, max(1, int(budget / one_tile)))


def _streaming_tile_shape(
    n_elements: 'int', gauss_order: 'int', point_count: 'int', n_xi: 'int',
    tile_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, int]':
    """Return (test-element rows, source-point columns) within the budget.

    The caller must first validate the one-element/one-column floor with
    :func:`_streaming_worker_count`.  Production then caps simultaneous
    sampling workers so the allowance covers every live tile.  This is the
    conservative live-set model of the FFT (non-banded) sampler; the banded
    sampler is planned by :func:`_plan_banded_tiles`.
    """

    ne = int(n_elements)
    go = int(gauss_order)
    points = int(point_count)
    samples = int(n_xi)
    threads = max(1, int(workers))
    budget = float(tile_budget_gb) * 1.0e9
    if ne < 1 or go < 1 or points < 1 or samples < 1:
        raise ValueError("Streaming tile dimensions must be positive.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming tile budget must be positive and finite.")

    rows_max = max(
        go,
        int(
            budget
            / (
                points
                * samples
                * BOR_STREAM_TILE_BYTES_PER_SAMPLE
                * threads
            )
        ),
    )
    tile_elements = min(ne, max(1, rows_max // go))
    source_columns = max(
        1,
        min(
            points,
            int(
                budget
                / (
                    tile_elements
                    * go
                    * samples
                    * BOR_STREAM_TILE_BYTES_PER_SAMPLE
                    * threads
                )
            ),
        ),
    )
    return tile_elements, source_columns


def _banded_tile_bytes(te, fe, go_p, go_q, n_orders, n_modes, efie, brackets,
                       work_bytes, n_xi=0) -> 'float':
    """Modeled peak live bytes of one banded tile of ``te`` test x ``fe`` source elements.

    A tile's phases run one after another and each holds only its own arrays:
    EFIE sampling (pair workspace, the G orders, the sampler chunk and the
    half-grid transform tables of at most ``n_xi`` samples), EFIE contraction
    (G orders, ten tested weights and one kind's node sums, one source-side
    product, one term's order sums with its combination temporaries, and the
    tile band) and, per bracket family, the same with four kernels of the
    kept modes (two tested weights per kernel).
    """
    te = float(te)
    fe = float(fe)
    pairs = te * go_p * fe * go_q
    tested_points = te * fe * go_q
    nodes = te + 1.0
    band = 64.0 * n_modes * nodes * (fe + 1.0)
    half = float(int(n_xi) // 2 + 1) if n_xi else 0.0
    peak = 0.0
    if efie:
        sample = (pairs * (_BANDED_PAIR_BYTES + 16.0 * n_orders) + work_bytes
                  + 64.0 * half * n_orders)
        contract = (pairs * 16.0 * n_orders
                    + tested_points * 16.0 * n_orders * 10.0
                    + nodes * fe * go_q * 16.0 * n_orders
                    + nodes * fe * 16.0 * n_orders * 4.0
                    + nodes * (fe + 1.0) * 16.0 * (n_orders + 2.0 * n_modes)
                    + band)
        peak = max(peak, sample, contract)
    if brackets:
        sample = (pairs * (_BANDED_PAIR_BYTES + 64.0 * n_modes) + work_bytes
                  + 64.0 * half * n_modes)
        contract = (pairs * 64.0 * n_modes
                    + tested_points * 16.0 * n_modes * 2.0
                    + nodes * fe * go_q * 16.0 * n_modes
                    + nodes * fe * 16.0 * n_modes * 2.0
                    + band)
        peak = max(peak, sample, contract)
    return peak


def _largest_fitting(fits, upper: 'int') -> 'int':
    """Largest ``n`` in ``[1, upper]`` with ``fits(n)`` (monotone), 0 if none."""
    upper = int(upper)
    if upper < 1 or not fits(1):
        return 0
    low, high = 1, upper
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


# A banded tile has at least this many test-element rows when the source
# elements can be split to afford them (see _plan_banded_tiles).
STREAM_TILE_MIN_ROWS = 16
STREAM_TILE_MIN_SOURCES = 8


def _plan_banded_tiles(ne_p, go_p, ne_q, go_q, n_orders, n_modes, efie, brackets,
                       tile_budget_gb, threads, n_xi=0):
    """``(threads, test elements, source elements, sampler bytes)`` of a banded tile plan.

    Every concurrent tile is sized to ``tile_budget / threads``, its share of
    the sampler workspace included, so the live set of all tiles stays within
    the one priced tile budget whatever the thread count.  Full-width tiles
    (every source element) are preferred; the source elements are split only
    when one test element's full row does not fit, and the tile count drops
    only below the one-element-pair floor.
    """
    budget = float(tile_budget_gb) * 1.0e9
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming tile budget must be positive and finite.")
    ne_p, ne_q = int(ne_p), int(ne_q)
    cost = None
    for count in range(max(1, int(threads)), 0, -1):
        work = min(float(modal_kernels.FFT_BUILD_BUDGET), 0.25 * budget) / count
        per_tile = budget / count

        def cost(te, fe, work=work):
            return BOR_STREAM_TILE_SLACK * _banded_tile_bytes(
                te, fe, go_p, go_q, n_orders, n_modes, efie, brackets, work, n_xi)

        if cost(1, ne_q) <= per_tile:
            rows = _largest_fitting(lambda te: cost(te, ne_q) <= per_tile, ne_p)
            if rows < STREAM_TILE_MIN_ROWS < ne_p:
                # Taller, narrower tiles at the same live set: the test-side
                # contraction's inner dimension is rows x Gauss points, so a
                # one-row full-width tile runs its products at a few percent
                # of GEMM speed (measured 1.9x on a 3,000-element mesh and
                # 12-14% at 800 elements, bitwise-equivalent blocks).
                sources = _largest_fitting(
                    lambda fe: cost(STREAM_TILE_MIN_ROWS, fe) <= per_tile, ne_q)
                if sources >= min(ne_q, STREAM_TILE_MIN_SOURCES):
                    return count, STREAM_TILE_MIN_ROWS, sources, work
            return count, rows, ne_q, work
        sources = _largest_fitting(lambda fe: cost(1, fe) <= per_tile, ne_q)
        if sources >= 1:
            return count, 1, sources, work
    raise ValueError(
        "Streaming tile budget is below the modeled one-element-pair minimum "
        f"of {cost(1, 1) / 1.0e9:.6g} GB.")


def _apply_tile_plan(stream, tile_budget_gb, tile_threads, n_xi, ne_p, go_p, ne_q, go_q,
                     points_q, efie, brackets) -> 'None':
    """Tile concurrency and shapes of one streamed build, set on ``stream``.

    ``_workers`` concurrent tiles (the physical cores, or ``tile_threads``
    capped at them) with ``_native_threads`` OpenMP sampling threads each;
    ``_tile_rows`` test x ``_tile_sources`` source elements per tile and
    ``_work_bytes`` of sampler workspace.  ``_te``/``_cols`` keep the
    conservative shape (test rows, sampled source-point chunk) of the FFT
    sampler used when ``BANDED_FFT`` is off.
    """
    cores = streaming_tile_threads()
    requested = cores if tile_threads is None else min(max(1, int(tile_threads)), cores)
    n_modes = min(int(stream.mode_block), int(stream.m_max) + 1)
    budget = float(tile_budget_gb) * 1.0e9
    banded = bool(modal_kernels.BANDED_FFT)
    if banded:
        # The banded planner has its own one-element-pair floor.
        threads, rows, sources, work = _plan_banded_tiles(
            ne_p, go_p, ne_q, go_q, n_modes + 2, n_modes, efie, brackets,
            tile_budget_gb, requested, n_xi)
    else:
        # The FFT sampler's one-element, one-column floor caps its tiles.
        threads = _streaming_worker_count(max(go_p, go_q), n_xi, tile_budget_gb, requested)
    stream._workers = threads
    stream._native_threads = max(1, cores // threads)
    stream._te, stream._cols = _streaming_tile_shape(
        ne_p, go_p, points_q, n_xi, tile_budget_gb, threads)
    stream._tile_banded = banded
    if banded:
        stream._tile_rows, stream._tile_sources, stream._work_bytes = rows, sources, work
    else:
        stream._tile_rows, stream._tile_sources = stream._te, int(ne_q)
        stream._work_bytes = min(float(modal_kernels.FFT_BUILD_BUDGET), 0.25 * budget) / threads


STREAM_SPILL_SAFETY_FACTOR = 1.25
# Resident spill pricing (see plan_stream_spill and STREAM_SPILL_RELEASE_BYTES).
STREAM_SPILL_RESIDENT_MODES = 2
# Spill directory prefixes (self and cross streams) and the owner marker each
# directory holds: host, process id and process start time of its owner.
STREAM_SPILL_PREFIXES = ("ghost-bor-far-", "ghost-bor-cross-")
STREAM_SPILL_OWNER_FILE = "ghost-spill-owner.json"
# A spill directory without an owner marker predates the markers (or is being
# created this instant); it counts as abandoned only after this age.
STREAM_SPILL_LEGACY_MIN_AGE_S = 3600.0
# Free space a spill leaves on its filesystem beyond its own files.
STREAM_SPILL_RUNTIME_RESERVE_BYTES = 256 * 1024 ** 2
_MEMORY_FILESYSTEM_MAGIC = {0x01021994: "tmpfs", 0x858458F6: "ramfs"}
_PROCESS_START_TIMES = {}


class StreamingSpillError(BorAdmissionError):
    """The planned far-block spill cannot be honoured on this host at run time.

    An admission-type rejection of the streamed plan (``streaming=True``): a
    spill that cannot be written never turns into holding every mode in RAM,
    a plan that was not priced.
    """

    def __init__(self, message, required_gb=None, mode_cap=None):
        super().__init__(message, streaming=True, required_gb=required_gb,
                         mode_cap=mode_cap)


def _statfs_filesystem(path) -> 'Optional[str]':
    """RAM filesystem name of the Linux statfs(2) magic of ``path``, or None."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        statfs = libc.statfs
    except (OSError, AttributeError, TypeError):
        return None
    buffer = ctypes.create_string_buffer(512)
    try:
        if statfs(os.fsencode(os.fspath(path)), buffer) != 0:
            return None
    except (OSError, TypeError, ValueError, ctypes.ArgumentError):
        return None
    magic = ctypes.c_ulong.from_buffer(buffer).value & 0xFFFFFFFF
    return _MEMORY_FILESYSTEM_MAGIC.get(magic)


def _mounted_filesystem(path) -> 'Optional[str]':
    """Type of the /proc/self/mounts entry holding ``path`` (longest mount point)."""
    try:
        target = os.path.realpath(os.fspath(path))
        best, kind = "", None
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mount = (fields[1].replace("\\040", " ").replace("\\011", "\t")
                         .replace("\\012", "\n").replace("\\134", "\\"))
                prefix = mount.rstrip("/") + "/"
                if (target == mount or target.startswith(prefix)) and len(mount) >= len(best):
                    best, kind = mount, fields[2]
        return kind
    except OSError:
        return None


def memory_backed_filesystem(path) -> 'Optional[str]':
    """``'tmpfs'`` or ``'ramfs'`` when ``path`` is on a RAM-backed Linux
    filesystem, else None.

    Far blocks spilled there stay in memory (tmpfs pages are shared memory,
    at best swapped), so the spill would not reduce the resident set it was
    priced to reduce: such a directory is refused.
    """
    if not sys.platform.startswith("linux"):
        return None
    name = _statfs_filesystem(path)
    if name is None:
        mounted = _mounted_filesystem(path)
        name = mounted if mounted in ("tmpfs", "ramfs") else None
    return name


def _process_start_time(pid) -> 'Optional[float]':
    try:
        import psutil
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def _write_owner_marker(directory) -> 'None':
    """Record this process as the owner of a spill directory."""
    pid = os.getpid()
    if pid not in _PROCESS_START_TIMES:
        _PROCESS_START_TIMES[pid] = _process_start_time(pid)
    record = {"version": 1, "pid": pid, "host": socket.gethostname(),
              "process_start": _PROCESS_START_TIMES[pid], "created": time.time()}
    with open(os.path.join(directory, STREAM_SPILL_OWNER_FILE), "w", encoding="utf-8") as handle:
        json.dump(record, handle)


def _pid_exists(pid: 'int') -> 'bool':
    """Whether a process ``pid`` runs on this host (conservatively True if unsure)."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel32.OpenProcess(0x1000, 0, int(pid))  # QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() != 87            # ERROR_INVALID_PARAMETER: none
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == 259                          # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _owner_alive(record) -> 'Optional[bool]':
    """True/False for an owner process on this host; None when undecidable
    (another host, or a malformed record)."""
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    host = record.get("host")
    if not isinstance(host, str) or host != socket.gethostname() or pid <= 0:
        return None
    if pid == os.getpid():
        return True
    started = record.get("process_start")
    try:
        import psutil
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if started is None:
                return True
            # A different start time is a reused process id: the owner is gone.
            return abs(float(process.create_time()) - float(started)) <= 2.0
        except psutil.NoSuchProcess:
            return False
        except Exception:
            return True
    return _pid_exists(pid)


def _tree_bytes(path) -> 'int':
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _newest_mtime(path) -> 'float':
    newest = 0.0
    try:
        newest = os.lstat(path).st_mtime
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    pass
    except OSError:
        pass
    return newest


def _spill_directory_state(path, now: 'float', legacy_min_age_s: 'float') -> 'str':
    """``'live'``, ``'foreign'``, ``'recent'`` or ``'stale'`` for one spill directory."""
    record = None
    try:
        with open(os.path.join(path, STREAM_SPILL_OWNER_FILE), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict):
        alive = _owner_alive(record)
        if alive is None:
            return "foreign"
        return "live" if alive else "stale"
    # No readable owner marker: made before owner markers, or made this instant.
    if now - _newest_mtime(path) < float(legacy_min_age_s):
        return "recent"
    return "stale"


class SpillSweepReport(dict):
    """Result of :func:`remove_stale_spill_directories`; falsy when nothing was removed."""

    def __bool__(self):
        return bool(self.get("removed"))


def _spill_sweep_bases(base):
    if base is not None:
        return [os.fspath(base)]
    candidates = []
    try:
        from ghost_backend.execution.options import temporary_directory
        candidates.append(temporary_directory())
    except Exception:
        pass
    candidates.append(tempfile.gettempdir())
    unique, seen = [], set()
    for candidate in candidates:
        key = os.path.normcase(os.path.realpath(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def remove_stale_spill_directories(base=None,
                                   legacy_min_age_s: 'float' = STREAM_SPILL_LEGACY_MIN_AGE_S):
    """Delete far-block spill directories whose owner process is gone.

    Looks at ``ghost-bor-far-*`` and ``ghost-bor-cross-*`` directories in
    ``base`` (default: the configured temporary directory and the system
    temporary directory).  A directory is removed only when its owner marker
    names a process of this host that no longer runs (process id and start
    time, so a reused id does not keep it alive).  Directories of live owners
    and of other hosts (a shared temporary directory) are never touched.  A
    directory without a marker was written before owner markers existed; it
    is removed once older than ``legacy_min_age_s``.

    Current spills vanish with their process anyway (delete-on-close files),
    so this mostly reclaims the empty directories and markers of killed
    solves and the full directories older versions left behind.

    Returns a :class:`SpillSweepReport` dict -- ``removed`` directories,
    freed ``bytes``, directories kept for ``live``, ``foreign`` and
    ``recent`` owners, ``failed`` removals and the removed ``directories`` --
    that is falsy when nothing was removed.
    """
    report = SpillSweepReport(removed=0, bytes=0, live=0, foreign=0, recent=0,
                              failed=0, directories=[])
    now = time.time()
    for root in _spill_sweep_bases(base):
        try:
            with os.scandir(root) as scan:
                entries = [entry for entry in scan
                           if entry.name.startswith(STREAM_SPILL_PREFIXES)]
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            state = _spill_directory_state(entry.path, now, legacy_min_age_s)
            if state != "stale":
                report[state] += 1
                continue
            size = _tree_bytes(entry.path)
            shutil.rmtree(entry.path, ignore_errors=True)
            if os.path.lexists(entry.path):
                report["failed"] += 1
            else:
                report["removed"] += 1
                report["bytes"] += size
                report["directories"].append(entry.path)
    return report


def _remove_spill_directory(path) -> 'bool':
    """Remove a spill directory once only its owner marker is left in it."""
    try:
        names = os.listdir(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if any(name != STREAM_SPILL_OWNER_FILE for name in names):
        return False            # a file still mapped by a live view (Windows)
    try:
        os.remove(os.path.join(path, STREAM_SPILL_OWNER_FILE))
    except FileNotFoundError:
        pass
    except OSError:
        return False
    try:
        os.rmdir(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _unlink_quietly(path) -> 'None':
    try:
        os.unlink(path)
    except OSError:
        pass


def _preallocate(fd: 'int', nbytes: 'int') -> 'None':
    """Reserve ``nbytes`` of disk for ``fd`` (POSIX).

    ``posix_fallocate`` makes a full disk fail here, with ENOSPC, instead of
    raising SIGBUS on a later page fault of the mapping.  A filesystem that
    cannot preallocate falls back to a sparse size.
    """
    allocate = getattr(os, "posix_fallocate", None)
    if allocate is not None:
        try:
            allocate(fd, 0, nbytes)
            return
        except OSError as exc:
            unsupported = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP,
                           getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
            if exc.errno not in unsupported:
                raise
    os.ftruncate(fd, nbytes)


def _mapped_spill_array(path, shape, dtype, nbytes: 'int') -> 'np.memmap':
    """A zeroed ``np.memmap`` of ``shape`` whose file vanishes with its last view.

    Windows: the file is opened delete-on-close (``O_TEMPORARY``) and the
    mapping keeps the only handle, so the file is deleted when the last array
    or view is released, and by the kernel when the process dies in any way.
    Extending the file through the mapping reserves its clusters (a full disk
    fails here) without writing zeros.  POSIX: the file is preallocated,
    mapped and unlinked at once; the mapping keeps the unnamed inode.
    """
    windows = os.name == "nt"
    flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_CLOEXEC", 0))
    if windows:
        flags |= os.O_TEMPORARY
    fd = os.open(path, flags, 0o600)
    try:
        if not windows:
            _preallocate(fd, nbytes)
        mapping = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
    except BaseException:
        os.close(fd)
        if not windows:
            _unlink_quietly(path)
        raise
    os.close(fd)                     # the mapping holds its own handle/descriptor
    if not windows:
        _unlink_quietly(path)
    array = np.ndarray.__new__(np.memmap, shape, dtype=dtype, buffer=mapping,
                               offset=0, order="C")
    array._mmap = mapping
    array.offset = 0
    array.mode = "r+"
    array.filename = None
    return array


# Pages a process has touched in a mapped file stay in its working set until
# the system trims it: the certified 10 GHz run kept its whole 26-39 GB spill
# resident against a two-mode plan, and free memory fell to a few MB.  Every
# finished part of a spilled array is therefore written back and released as
# soon as it is complete: the node rows of every mode once the far build has
# summed all their tiles (_SpilledRowRelease), and a mode's blocks in chunks of
# this many bytes as they are copied into a system matrix.  Released pages stay
# in the file (and the page cache) and fault back in if touched again.
STREAM_SPILL_RELEASE_BYTES = 32 * 1024**2
_SPILL_PAGE = mmap.PAGESIZE
_SPILL_GRANULARITY = mmap.ALLOCATIONGRANULARITY
_VIRTUAL_UNLOCK = None


def _virtual_unlock():
    global _VIRTUAL_UNLOCK
    if _VIRTUAL_UNLOCK is None:
        import ctypes
        function = ctypes.WinDLL("kernel32", use_last_error=True).VirtualUnlock
        function.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        function.restype = ctypes.c_int
        _VIRTUAL_UNLOCK = function
    return _VIRTUAL_UNLOCK


def _release_spilled(array, ranges, flush: 'bool' = True) -> 'None':
    """Write back and drop from the working set element ranges of a spilled array.

    ``ranges`` are ``(start, stop)`` flat element offsets.  Only whole pages
    inside a range are released, so pages shared with data still being
    written are never touched.  Windows removes pages from the working set
    with ``VirtualUnlock`` on unlocked memory; POSIX with ``MADV_DONTNEED``,
    which keeps the data of a shared file mapping in the page cache.  An
    in-memory array, or any failure, leaves the pages as they are.
    """
    mapping = getattr(array, "_mmap", None)
    if mapping is None:
        return
    item = array.dtype.itemsize
    try:
        base = array.ctypes.data
        for lo, hi in ranges:
            start, stop = int(lo) * item, int(hi) * item
            if stop <= start:
                continue
            if flush:
                aligned = start - start % _SPILL_GRANULARITY
                mapping.flush(aligned, stop - aligned)
            first = -(-start // _SPILL_PAGE) * _SPILL_PAGE
            last = (stop // _SPILL_PAGE) * _SPILL_PAGE
            if last <= first:
                continue
            if os.name == "nt":
                _virtual_unlock()(base + first, last - first)
            elif hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                mapping.madvise(mmap.MADV_DONTNEED, first, last - first)
    except (OSError, ValueError, AttributeError):
        pass


def _block_row_range(layout, rows_lo: 'int', rows_hi: 'int'):
    """Flat range, within one [uv, mode] block, of the node rows [lo, hi)."""
    kind, value = layout
    if kind == "packed":
        offsets, total = value
        start = int(offsets[rows_lo]) if rows_lo < len(offsets) else total
        stop = int(offsets[rows_hi]) if rows_hi < len(offsets) else total
        return start, stop
    columns = value
    return rows_lo * columns, rows_hi * columns


def _spilled_rows(array, layout, rows_lo: 'int', rows_hi: 'int', modes=None):
    """Flat element ranges of the node rows [lo, hi) of every (or the given)
    mode of a spilled ``[4, modes, ...]`` array."""
    n_modes = array.shape[1]
    block = int(np.prod(array.shape[2:], dtype=np.int64))
    start, stop = _block_row_range(layout, rows_lo, rows_hi)
    return [((uv * n_modes + mi) * block + start, (uv * n_modes + mi) * block + stop)
            for uv in range(4) for mi in (range(n_modes) if modes is None else modes)]


class _SpilledRowRelease:
    """Releases the node rows of spilled arrays as the far build completes them.

    Tiles cover element rows ``[e0, e0 + rows)`` and add to node rows
    ``e0 .. e0 + rows``, so node row ``e0 + rows`` is shared with the next row
    block: rows are released as a contiguous prefix, once every tile of every
    row block reaching them is done.
    """

    def __init__(self, stores, elements: 'int', rows: 'int', tiles_per_row: 'int',
                 nodes: 'int'):
        self.stores = [(array, layout) for array, layout in stores
                       if getattr(array, "_mmap", None) is not None]
        self.elements, self.rows, self.nodes = int(elements), max(1, int(rows)), int(nodes)
        self.tiles_per_row = max(1, int(tiles_per_row))
        self._done = {}
        self._complete = set()
        self._next = 0
        self._released = 0
        self._lock = threading.Lock()

    def tile_done(self, e0: 'int') -> 'None':
        if not self.stores:
            return
        with self._lock:
            count = self._done.get(e0, 0) + 1
            self._done[e0] = count
            if count < self.tiles_per_row:
                return
            self._complete.add(e0)
            lo = self._released
            while self._next in self._complete:
                end = min(self._next + self.rows, self.elements)
                self._released = self.nodes if end >= self.elements else end
                self._next = end
            hi = self._released
        if hi > lo:
            for array, layout in self.stores:
                _release_spilled(array, _spilled_rows(array, layout, lo, hi))

    def finish(self) -> 'None':
        """Release whatever is left (every tile has run)."""
        with self._lock:
            lo, self._released = self._released, self.nodes
        if self.nodes > lo:
            for array, layout in self.stores:
                _release_spilled(array, _spilled_rows(array, layout, lo, self.nodes))


class _SpillFiles:
    """One owner-marked spill directory of delete-on-close memory-mapped arrays.

    Each array lives in its own file that goes with the array's last
    reference: a delete-on-close handle held only by the mapping on Windows,
    an unlinked name on POSIX.  The kernel therefore reclaims the space when
    the process dies in any way (``os._exit``, a crash, a kill); only the
    directory with its owner marker can remain, for
    :func:`remove_stale_spill_directories`.  Nothing is force-closed: a view
    that outlives :meth:`release` keeps its mapping valid, and the directory
    is removed with the last such view.
    """

    def __init__(self, base, prefix: 'str'):
        self.base = os.fspath(base)
        self.path = None
        self.nbytes = 0
        self._arrays = []
        filesystem = memory_backed_filesystem(self.base)
        if filesystem is not None:
            raise StreamingSpillError(
                f"The far-block spill directory {self.base} is on {filesystem}, a "
                "RAM-backed filesystem: spilling there would not reduce memory. "
                "Configure a disk-backed temporary directory or turn stream_spill off.")
        try:
            self.path = tempfile.mkdtemp(prefix=prefix, dir=self.base)
            _write_owner_marker(self.path)
        except OSError as exc:
            self.release()
            raise StreamingSpillError(
                f"Cannot create a far-block spill directory in {self.base}: {exc}") from exc

    def reserve(self, nbytes: 'int') -> 'None':
        """Refuse before writing when the filesystem cannot hold ``nbytes`` more."""
        needed = int(nbytes) + STREAM_SPILL_RUNTIME_RESERVE_BYTES
        try:
            free = shutil.disk_usage(self.path).free
        except OSError as exc:
            raise StreamingSpillError(
                f"Cannot check the free space of the far-block spill directory {self.path}: {exc}"
            ) from exc
        if free < needed:
            raise StreamingSpillError(
                f"The far-block spill needs {nbytes / 1.0e9:.3f} GB (plus a "
                f"{STREAM_SPILL_RUNTIME_RESERVE_BYTES / 1.0e9:.2f} GB reserve) in {self.path}, "
                f"but only {free / 1.0e9:.3f} GB is free now. Free disk space, point the "
                "temporary directory elsewhere, or raise the stream budget.",
                required_gb=needed / 1.0e9)

    def allocate(self, name: 'str', shape, dtype) -> 'np.ndarray':
        shape = tuple(int(s) for s in shape)
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes == 0:
            return np.zeros(shape, dtype)
        if self.path is None:
            raise StreamingSpillError("The far-block spill directory was already released.")
        path = os.path.join(self.path, f"{name}.bin")
        try:
            array = _mapped_spill_array(path, shape, dtype, nbytes)
        except OSError as exc:
            reason = ("the disk is full" if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1))
                      or getattr(exc, "winerror", None) in (39, 112) else str(exc))
            raise StreamingSpillError(
                f"Cannot allocate the {nbytes / 1.0e9:.3f} GB far-block spill file {path}: "
                f"{reason}.", required_gb=nbytes / 1.0e9) from exc
        # The directory goes with the last mapping (a view may outlive release()).
        weakref.finalize(array._mmap, _remove_spill_directory, self.path)
        self._arrays.append(array)
        self.nbytes += nbytes
        return array

    def release(self) -> 'None':
        """Drop this owner's references; each file vanishes with its last view."""
        self._arrays.clear()
        if self.path is not None:
            _remove_spill_directory(self.path)


def spill_directory(required_gb: 'float'):
    """Base directory able to hold ``required_gb`` of spilled far blocks, or None.

    ``None`` when the ``stream_spill`` option is off, when the configured
    temporary directory is unavailable or RAM-backed (Linux tmpfs/ramfs), or
    when it lacks the space with a :data:`STREAM_SPILL_SAFETY_FACTOR` margin.
    """
    from ghost_backend.bor.options import current_options
    if current_options().get('stream_spill', 'auto') != 'auto':
        return None
    try:
        from ghost_backend.execution.options import temporary_directory
        base = temporary_directory()
        if memory_backed_filesystem(base) is not None:
            return None
        free = shutil.disk_usage(base).free
    except (ValueError, OSError):
        return None
    required = float(required_gb) * 1.0e9 * STREAM_SPILL_SAFETY_FACTOR
    return base if free >= required else None


def plan_stream_spill(mode_block: 'int', mode_count: 'int', per_mode_gb: 'float',
                      allow_spill: 'bool' = True):
    """``(base directory or None, mode block, resident GB)`` for one far stream.

    A retained-block budget that cannot hold every mode used to mean one far
    build per mode range, and every range re-samples every far pair.  With
    spilling the blocks are accumulated once, for all modes, into
    memory-mapped temporary files; the resident cost is priced as
    :data:`STREAM_SPILL_RESIDENT_MODES` modes rather than the block.  The
    streams hold it to that: completed rows are written back and released
    during the build, and each mode is released in chunks as it is copied
    into a system matrix (the 10 GHz ogive's 12 GB spill peaked at 2 GB of
    working set, all tile workspace included, against 16 GB before).  The
    callers price and run the same decision.
    """
    mode_block = int(mode_block)
    mode_count = int(mode_count)
    if mode_block >= mode_count or not allow_spill:
        return None, mode_block, None
    base = spill_directory(mode_count * float(per_mode_gb))
    if base is None:
        return None, mode_block, None
    return base, mode_count, STREAM_SPILL_RESIDENT_MODES * float(per_mode_gb)


def stream_spill_candidate_gb(mode_block: 'int', mode_count: 'int',
                              per_mode_gb: 'float') -> 'float':
    """Far blocks (decimal GB) a streamed plan writes if it spills.

    Zero when one range already holds every mode or the ``stream_spill``
    option is off.  Schedulers reserve it across concurrent units, and
    :func:`plan_stream_spill` spills only when the temporary directory holds
    it with :data:`STREAM_SPILL_SAFETY_FACTOR`.
    """
    from ghost_backend.bor.options import current_options
    if (int(mode_block) >= int(mode_count)
            or current_options().get('stream_spill', 'auto') != 'auto'):
        return 0.0
    return int(mode_count) * float(per_mode_gb)


def combined_stream_mode_gb(m_max: 'int', requirements) -> 'float':
    """One mode of every stream of a combined plan, in GB.

    ``requirements`` are those of :func:`plan_combined_streaming_mode_block`;
    streams that share one range also share one spill decision, priced with
    this sum as ``per_mode_gb`` of :func:`plan_stream_spill`.
    """
    return sum(
        estimate_rectangular_streaming_block_gb(
            nt, ns, int(m_max), 1, bool(rotated), bool(single))
        for nt, ns, rotated, single in requirements)


def _n_xi_efie(k: 'complex', rho_max: 'float', m_max: 'int', d_min: 'float' = 0.0) -> 'int':
    return n_xi_for_pairs(k, rho_max, m_max, d_min, bracket=False)


def _n_xi_bracket(k: 'complex', rho_max: 'float', m_max: 'int', d_min: 'float' = 0.0) -> 'int':
    return n_xi_for_pairs(k, rho_max, m_max, d_min, bracket=True)


def _closed_stream_error():
    return RuntimeError(
        "Streaming far blocks were released by close(); a closed stream is "
        "never rebuilt (a rebuild would hold unpriced blocks). Build a new stream.")


class StreamingFarBlocks:
    """Per-mode nodal far blocks for one BorPecSolver surface.

    efie=True builds the four EFIE blocks (without the C = jk eta 2pi
    factor, matching _pair_blocks); mfie=True the four MFIE bracket blocks
    (WITH the 2pi Galerkin factor, matching assemble_mfie_mode's far
    contraction); ibc_zs_pt (per-Gauss-point Z_s) the IBC bracket blocks
    with the source weight baked in; pmchwt=True builds the same rotated-PV
    blocks with unit source weight (both matching _rot_pv_blocks).

    ``workers`` are the outer mode workers; they only align the retained mode
    ranges.  Far pairs are built in tiles on :func:`streaming_tile_threads`
    threads (``tile_threads`` caps them), each tile sized to its share of
    ``tile_budget_gb``; a tile sums its contributions locally and adds them
    once under a per-family lock.  The EFIE family samples element pairs
    ``e < f`` only, keeps their packed strict-upper triangles and completes a
    mode by symmetry when it is read (:meth:`write_efie_blocks`).
    ``spill`` (the base directory :func:`plan_stream_spill` chose) keeps every
    mode in delete-on-close memory-mapped files (:class:`_SpillFiles`).
    """

    def __init__(self, solver, m_max: 'int', efie: 'bool' = True,
                 mfie: 'bool' = False, ibc_zs_pt: 'Optional[np.ndarray]' = None,
                 pmchwt: 'bool' = False,
                 dtype=np.complex128,
                 tile_budget_gb: 'float' = BOR_STREAM_TILE_BUDGET_GB,
                 workers: 'int' = 1, mode_block: 'Optional[int]' = None,
                 spill: 'Optional[str]' = None,
                 tile_threads: 'Optional[int]' = None):
        self.solver = solver
        self.m_max = int(m_max)
        self.dtype = dtype
        g = solver.g
        gen = solver.gen
        self.Nn = solver.Nn
        ne = gen.n_elems
        self.go = solver.gauss_order
        P = solver.P
        k = solver.k
        Nn, go, mm = self.Nn, self.go, self.m_max


        weights = far_weights(g, ne, go, ibc_zs_pt, pmchwt)
        self._lv = weights["lv"]
        rv_ibc = weights["rv_ibc"]


        self._efie = efie
        self._mfie = mfie
        self._has_ibc = ibc_zs_pt is not None or bool(pmchwt)
        self.rot_pv_unit_source = bool(pmchwt)
        self._rv_ibc = rv_ibc
        self._left_all = weights["left_all"]
        self._right = weights["right"]
        self._right_groups = weights["right_groups"]
        self._right_one = weights["right_one"]
        self._right_ibc = weights["right_ibc"]
        self._right_ibc_stacked = weights["right_ibc_stacked"]
        self.k = complex(k)
        # The outer mode workers only align the retained mode ranges (a range
        # holds every mode one worker wave reads).  The far tiles have their
        # own threads, which never round the planned block past its budget.
        self.mode_workers = max(1, int(workers))
        self.mode_block = _aligned_stream_mode_block(mm, mode_block, self.mode_workers)
        self.n_sweeps = 0
        self._closed = False
        self._spill = None
        self._spilled_bytes = 0
        self._finalizer = None
        self.Z = self.K = self.B = None
        self.lo, self.hi = 1, 0
        if spill is not None:
            # The planner decided (plan_stream_spill): one range of every
            # mode, accumulated into delete-on-close memory-mapped files.
            self._spill = _SpillFiles(spill, "ghost-bor-far-")
            self._finalizer = weakref.finalize(self, self._spill.release)
            self.mode_block = mm + 1
        try:
            rho_max = float(np.max(gen.nodes[:, 0]))
            gap = solver._far_gap()
            self._nx_e = _n_xi_efie(k, rho_max, mm, gap)
            self._nx_b = _n_xi_bracket(k, rho_max, mm, gap)
            nx_worst = max(self._nx_e,
                           self._nx_b if (mfie or self._has_ibc) else 0)
            # G(p, q) = G(q, p) and the test and source weights coincide, so
            # Z_tt = Z_tt^T, Z_ff = Z_ff^T and Z_tf = -Z_ft^T: only element
            # pairs e < f are sampled (see _symmetrize_efie_blocks).
            self._symmetric_efie = bool(
                efie and STREAM_EFIE_SYMMETRY
                and _symmetric_near_relation(solver._near_sources_by_element))
            # With adjacent elements near, every sampled pair lands strictly
            # above the node diagonal: keep only that packed triangle and
            # complete a mode when it is read.
            self._packed_efie = bool(self._symmetric_efie and
                                     _adjacent_pairs_near(solver._near_sources_by_element))
            self._offsets = _packed_row_offsets(Nn) if self._packed_efie else None
            _apply_tile_plan(self, tile_budget_gb, tile_threads, nx_worst, ne, go, ne, go, P,
                             bool(efie), bool(mfie or self._has_ibc))
            self._native = (_NATIVE if (_NATIVE is not None and
                                        abs(complex(k).imag) == 0.0) else None)
            if _NATIVE is None:
                _notice_numpy_fallback()
            self._q = tuple(np.ascontiguousarray(v) for v in
                            (g.rho, g.z, g.trho, g.tz))
            self._acc_lock = threading.Lock()
            self._acc_locks = {"efie": self._acc_lock, "mfie": threading.Lock(),
                               "ibc": threading.Lock()}
            self._range_lock = threading.RLock()
            self._ord_lo = 0
            self._sidx: 'Dict[int, int]' = {}
            self._ensure(0)
        except BaseException:
            self.close()
            raise


    def _ensure(self, am: 'int') -> 'None':
        if self.lo <= am <= self.hi:
            return
        with self._range_lock:
            if self.lo <= am <= self.hi:
                return
            if self._closed:
                raise _closed_stream_error()
            lo = (am // self.mode_block) * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._build_range(lo, hi)

    def _allocate(self, name: 'str', shape):
        """Zeroed block storage: in RAM, or a memory-mapped file when spilling."""
        if self._spill is None:
            return np.zeros(shape, dtype=self.dtype)
        value = self._spill.allocate(name, shape, self.dtype)
        self._spilled_bytes += value.nbytes
        return value

    def close(self) -> 'None':
        """Release the retained blocks and any spilled files.

        References are dropped, never force-closed: a view still held keeps
        its own mapping valid, and its file goes with it.
        """
        self._closed = True
        self.Z = self.K = self.B = None
        self.lo, self.hi = 1, 0
        finalizer, self._finalizer = getattr(self, "_finalizer", None), None
        if finalizer is not None:
            finalizer()
        self._spill = None

    def spilled_gb(self) -> 'float':
        return self._spilled_bytes / 1.0e9

    def _add_band(self, family: 'str', store, band, e0: 'int', f0: 'int') -> 'None':
        """Add one tile's summed contributions to the shared blocks."""
        with self._acc_locks[family]:
            if family == "efie" and getattr(self, "_packed_efie", False):
                _add_band_to_packed(store, band, e0, f0, self._offsets)
            else:
                _add_band_to_store(store, band, e0, f0)

    def _efie_entries(self) -> 'int':
        """Stored entries of one EFIE block of one mode (packed triangle or full)."""
        return packed_upper_size(self.Nn) if self._packed_efie else self.Nn * self.Nn

    def _run_build_tiles(self, tiles, operation):
        _run_tiles(tiles, operation, self._workers)

    def _build_range(self, lo: 'int', hi: 'int') -> 'None':
        Nn, go, mm = self.Nn, self.go, self.m_max
        ne = self.solver.gen.n_elems
        k = self.solver.k
        ord_lo = max(0, lo - 1)
        ms = list(range(lo, hi + 1))
        other_families = int(bool(self._mfie)) + int(bool(self._has_ibc))
        self.Z = self.K = self.B = None
        if self._spill is not None:
            entries = (4 * self._efie_entries() if self._efie else 0) + other_families * 4 * Nn * Nn
            self._spill.reserve(entries * len(ms) * np.dtype(self.dtype).itemsize)
        if self._efie:
            self.Z = self._allocate("efie", (4, hi-lo+1, self._efie_entries()) if self._packed_efie
                                    else (4, hi-lo+1, Nn, Nn))
        self._positive_modes = np.asarray(ms)
        self._sidx = {m: i for i, m in enumerate(ms)}
        if self._mfie:
            self.K = self._allocate("mfie", (4, len(ms), Nn, Nn))
        if self._has_ibc:
            self.B = self._allocate("ibc", (4, len(ms), Nn, Nn))
        self._ord_lo = ord_lo

        orders = np.arange(ord_lo, hi + 2)
        ph_e = np.exp(1j * np.pi * orders) * (2.0 * np.pi / self._nx_e)
        msarr = np.asarray(ms)
        bins_b = np.where(msarr >= 0, msarr, self._nx_b + msarr)
        ph_b = np.exp(1j * np.pi * msarr) * (2.0 * np.pi / self._nx_b)
        te, fe = self._tile_rows, self._tile_sources
        symmetric = self._symmetric_efie
        families_b = (("mfie", "K", self._mfie, self._right_one),
                      ("ibc", "B", self._has_ibc, self._right_ibc_stacked))
        # Spilled rows are written back and released as they are completed.
        release = None
        if self._spill is not None:
            stores = [(self.Z, self._store_layout("efie"))] if self._efie else []
            stores += [(store, ("full", Nn)) for store in (self.K, self.B) if store is not None]
            release = _SpilledRowRelease(stores, ne, te, len(range(0, ne, fe)), Nn)

        def do_tile(tile):
            e0, f0 = tile
            e1 = min(e0 + te, ne)
            f1 = min(f0 + fe, ne)
            rows = slice(e0 * go, e1 * go)
            re = e1 - e0
            if self._efie:
                # The symmetric build samples only pairs whose source follows.
                s0 = max(f0, e0 + 1) if symmetric else f0
                if s0 < f1:
                    Gn = self._sample_G(rows, k, self._nx_e, ph_e, ord_lo, hi,
                                        (s0, f1), symmetric)
                    if not modal_kernels.BANDED_FFT:
                        # The banded sampler never samples a masked pair.
                        self._zero_near(Gn, e0, e1, s0, f1, symmetric)
                    left = self._left_all[:, :, rows].reshape(2 * len(_LEFT_KINDS), re, go)
                    band = _efie_band(Gn, left, self._right_groups, self._positive_modes,
                                      ord_lo, k, s0, f1, re, go)
                    Gn = left = None
                    self._add_band("efie", self.Z, band, e0, s0)
                    band = None
            for which, attribute, wanted, right in families_b:
                if not wanted:
                    continue
                Fs = self._sample_brackets(which, rows, re, k, self._nx_b, bins_b, ph_b,
                                           (f0, f1))
                band = _bracket_band(
                    Fs, self._lv["1"][:, rows].reshape(2, re, go), right, f0, f1, re, go,
                    None if modal_kernels.BANDED_FFT else
                    (lambda kernel: self._zero_near(kernel, e0, e1, f0, f1)))
                Fs = None
                self._add_band(which, getattr(self, attribute), band, e0, f0)
                band = None
            if release is not None:
                release.tile_done(e0)

        tiles = [(e0, f0) for e0 in range(0, ne, te) for f0 in range(0, ne, fe)]
        self._run_build_tiles(tiles, do_tile)
        if release is not None:
            release.finish()
        if self._efie and symmetric and not self._packed_efie:
            _symmetrize_efie_blocks(self.Z)
            if release is not None:
                _release_spilled(self.Z, [(0, self.Z.size)])
        self.lo, self.hi = lo, hi
        self.n_sweeps += 1

    def _sample_brackets(self, which: 'str', rows, re: 'int', k, nx_b: 'int',
                         bins, phase, sources=None):
        """Return four kept-mode bracket tiles while bounding raw FFT memory.

        ``sources`` is the tile's source-element range (all when None).
        Source columns are independent.  Sample and transform at most
        ``self._cols`` columns at a time, then retain only the requested modal
        bins in the tile-width output used by the Galerkin contraction.
        """
        ne = self.solver.gen.n_elems
        f0, f1 = (0, ne) if sources is None else sources
        if modal_kernels.BANDED_FFT:
            return _banded_stream(self, rows, which,
                                  np.where(bins > nx_b // 2, bins - nx_b, bins), (f0, f1))
        g = self.solver.g
        go = self.go
        nr = re * go
        start, stop = f0 * go, f1 * go
        xi = 2.0 * np.pi * np.arange(nx_b) / nx_b - np.pi
        kept = tuple(np.empty((nr, stop - start, len(bins)), dtype=np.complex128)
                     for _ in range(4))
        rp = np.ascontiguousarray(g.rho[rows])
        zp = np.ascontiguousarray(g.z[rows])
        trp = np.ascontiguousarray(g.trho[rows])
        tzp = np.ascontiguousarray(g.tz[rows])
        cx = np.ascontiguousarray(np.cos(xi))
        sx = np.ascontiguousarray(np.sin(xi))
        for c0 in range(start, stop, self._cols):
            c1 = min(c0 + self._cols, stop)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q, z_q, tr_q, tz_q = (
                    np.ascontiguousarray(value[cols]) for value in self._q
                )
                sampled = tuple(np.empty((nr, nc, nx_b), dtype=np.complex128)
                                for _ in range(4))
                fn = (self._native.sample_mfie if which == "mfie"
                      else self._native.sample_ibc)
                fn(nr, nc, nx_b, _dp(rp), _dp(zp), _dp(trp), _dp(tzp),
                   _dp(rho_q), _dp(z_q), _dp(tr_q), _dp(tz_q),
                   float(np.real(k)), _dp(cx), _dp(sx),
                   _dp(sampled[0]), _dp(sampled[1]),
                   _dp(sampled[2]), _dp(sampled[3]))
            elif which == "mfie":
                sampled = _mfie_brackets(
                    rp[:, None], zp[:, None], trp[:, None], tzp[:, None],
                    g.rho[None, cols], g.z[None, cols],
                    g.trho[None, cols], g.tz[None, cols], k, xi)
            else:
                pair_shape = (nr, nc)
                sampled = _ibc_brackets_grid(
                    np.broadcast_to(rp[:, None], pair_shape).ravel(),
                    np.broadcast_to(zp[:, None], pair_shape).ravel(),
                    np.broadcast_to(trp[:, None], pair_shape).ravel(),
                    np.broadcast_to(tzp[:, None], pair_shape).ravel(),
                    np.broadcast_to(g.rho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.z[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.trho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(g.tz[None, cols], pair_shape).ravel(),
                    k, np.broadcast_to(xi, (nr * nc, nx_b)))
                sampled = tuple(F.reshape(nr, nc, nx_b) for F in sampled)
            for uv, values in enumerate(sampled):
                spectrum = np.fft.fft(values, axis=-1)
                kept[uv][:, c0 - start:c1 - start] = (
                    spectrum[..., bins] * (2.0 * np.pi * phase)
                )
        return kept


    def _sample_G(self, rows, k, n_xi, phase, ord_lo, hi, sources=None,
                  strict_upper=False):
        ne = self.solver.gen.n_elems
        f0, f1 = (0, ne) if sources is None else sources
        if modal_kernels.BANDED_FFT:
            return _banded_stream(self, rows, 'g', np.arange(ord_lo, hi + 2), (f0, f1),
                                  strict_upper)
        g = self.solver.g
        go = self.go
        start, stop = f0 * go, f1 * go
        xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
        rp = np.ascontiguousarray(g.rho[rows])
        zp = np.ascontiguousarray(g.z[rows])
        sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
        kept = np.empty((len(rp), stop - start, hi + 2 - ord_lo),
                        dtype=np.complex128)
        for c0 in range(start, stop, self._cols):
            c1 = min(c0 + self._cols, stop)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q = np.ascontiguousarray(self._q[0][cols])
                z_q = np.ascontiguousarray(self._q[1][cols])
                gk = np.empty((len(rp), nc, n_xi), dtype=np.complex128)
                self._native.sample_g(len(rp), nc, n_xi,
                                      _dp(rp), _dp(zp), _dp(rho_q),
                                      _dp(z_q), float(np.real(k)),
                                      _dp(sin2), _dp(gk))
            else:
                d2 = (rp[:, None] - g.rho[None, cols]) ** 2 + \
                     (zp[:, None] - g.z[None, cols]) ** 2
                rr4 = 4.0 * rp[:, None] * g.rho[None, cols]
                R = np.sqrt(d2[..., None] + rr4[..., None] * sin2)
                R = np.maximum(R, 1e-300)
                gk = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)
            spectrum = np.fft.fft(gk, axis=-1)
            kept[:, c0 - start:c1 - start] = spectrum[..., ord_lo:hi + 2] * phase
        return kept

    def _zero_near(self, Kt, e0, e1, f0=0, f1=None, strict_upper=False):
        """Zero the near pairs of a tile (source elements ``[f0, f1)``) and,
        with ``strict_upper``, every pair ``f <= e``."""
        go = self.go
        near = self.solver._near_sources_by_element
        f1 = self.solver.gen.n_elems if f1 is None else f1
        for e in range(e0, e1):
            row = slice((e - e0) * go, (e - e0 + 1) * go)
            for f in near[e]:
                if f0 <= f < f1:
                    Kt[row, (f - f0) * go:(f - f0 + 1) * go] = 0.0
            if strict_upper and e >= f0:
                Kt[row, :(min(e + 1, f1) - f0) * go] = 0.0


    def query_blocks(self, family, m, rows, cols):
        """Owned nodal slices for compressed modal assembly, without expansion.

        The exact streamed coefficients use the same angular rule as direct
        coefficient queries. Packed EFIE reciprocity is applied locally.
        """
        rows, cols = np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)
        with self._range_lock:
            self._ensure(abs(int(m)))
            store = {'efie': self.Z, 'mfie': self.K, 'ibc': self.B}[family]
            if store is None:
                raise ValueError('The requested streamed operator family was not prepared.')
            mi = self._sidx[abs(int(m))]
            if family != 'efie' or not self._packed_efie:
                return [np.asarray(store[uv, mi][np.ix_(rows, cols)], dtype=complex)
                        * mode_sign(uv, m) for uv in range(4)]
            rr, cc = np.broadcast_arrays(rows[:, None], cols[None, :])
            off = rr != cc
            low = rr > cc
            a, b = np.minimum(rr, cc), np.maximum(rr, cc)
            indices = self._offsets[a[off]] + b[off] - a[off] - 1
            result = []
            for uv in range(4):
                value = np.zeros(rr.shape, complex)
                component = np.full(indices.shape, uv, dtype=np.intp)
                if uv in (1, 2):
                    component[low[off]] = 3 - uv
                value[off] = store[component, mi, indices]
                if uv in (1, 2):
                    value[low] *= -1
                result.append(value * mode_sign(uv, m))
            return result

    def write_efie_blocks(self, m: 'int', quads, scale) -> 'None':
        """``quads[uv] = scale * mode_sign(uv, m) * Z_uv(m)``: the four full far
        EFIE blocks of one signed mode, written into the caller's quadrants.

        Packed symmetric storage is unpacked row by row and completed in the
        quadrants (``_symmetrize_efie_mode``), which equals scaling the full
        symmetric blocks the former build formed; full storage is scaled
        straight from its (possibly memory-mapped) views.
        """
        with self._range_lock:
            self._ensure(abs(m))
            store = self.Z
            if store is None:
                raise ValueError("Streaming far blocks were not built for 'efie'.")
            mi = self._sidx[abs(m)]
            views = tuple(store[uv, mi] for uv in range(4))
        Nn = self.Nn
        # Row chunks: a spilled mode is released chunk by chunk once copied.
        spilled = getattr(store, "_mmap", None) is not None
        chunks = _row_chunks(Nn, Nn * store.dtype.itemsize)
        masks = self._unpack_masks(chunks) if self._packed_efie else None
        layout = self._store_layout("efie")
        for uv, (view, quad) in enumerate(zip(views, quads)):
            factor = scale * mode_sign(uv, m)
            for index, (lo, hi) in enumerate(chunks):
                if self._packed_efie:
                    _unpack_upper_into(view, quad, factor, self._offsets, (lo, hi), masks[index])
                else:
                    np.multiply(view[lo:hi], factor, out=quad[lo:hi])
                if spilled:
                    _release_spilled(store, [_spilled_rows(store, layout, lo, hi, (mi,))[uv]],
                                     flush=False)
        if self._packed_efie:
            _symmetrize_efie_mode(*quads)

    def _unpack_masks(self, chunks):
        """Cached strict-upper masks of the row chunks (one n x n bool array in all)."""
        key = tuple(chunks)
        cached = getattr(self, "_mask_cache", None)
        if cached is None or cached[0] != key:
            masks = []
            for lo, hi in chunks:
                mask = _upper_mask(self.Nn, lo, hi)
                mask.setflags(write=False)
                if self.Nn <= UNPACK_INDEX_MAX_NODES:
                    rows_index, cols_index = np.nonzero(mask)
                    rows_index.setflags(write=False)
                    cols_index.setflags(write=False)
                    masks.append((mask, (rows_index, cols_index)))
                else:
                    masks.append(mask)
            cached = self._mask_cache = (key, masks)
        return cached[1]

    def _store_layout(self, family: 'str'):
        """Row layout of one family's [uv, mode] blocks (see _block_row_range)."""
        if family == "efie" and self._packed_efie:
            return ("packed", (self._offsets, packed_upper_size(self.Nn)))
        return ("full", self.Nn)

    def add_blocks(self, family: 'str', m: 'int', targets, scale) -> 'None':
        """``targets[uv] += scale * mode_sign(uv, m) * block`` for the stored
        MFIE ('mfie') or rotated-PV/IBC ('ibc') blocks of one signed mode:
        the in-place assembly of ``_scaled_add_into`` (same chunks, same
        arithmetic), releasing spilled pages as they are read."""
        if scale == 0:
            return
        with self._range_lock:
            self._ensure(abs(m))
            store = {"mfie": self.K, "ibc": self.B}[family]
            if store is None:
                raise ValueError(f"Streaming far blocks were not built for {family!r}.")
            mi = self._sidx[abs(m)]
            views = tuple(store[uv, mi] for uv in range(4))
        spilled = getattr(store, "_mmap", None) is not None
        layout = ("full", self.Nn)
        for uv, (view, target) in enumerate(zip(views, targets)):
            factor = scale * mode_sign(uv, m)
            for lo, hi in _row_chunks(target.shape[0], 0, chunk_rows=256):
                piece = view[lo:hi]
                if factor == 1:
                    target[lo:hi] += piece
                else:
                    target[lo:hi] += piece * factor
                piece = None
                if spilled:
                    _release_spilled(store, [_spilled_rows(store, layout, lo, hi, (mi,))[uv]],
                                     flush=False)

    def _full_efie_mode(self, m: 'int'):
        """The four full, unsigned EFIE blocks of mode ``|m|`` as new complex128 arrays."""
        blocks = tuple(np.empty((self.Nn, self.Nn), dtype=np.complex128) for _ in range(4))
        signs = tuple(mode_sign(uv, abs(int(m))) for uv in range(4))
        self.write_efie_blocks(abs(int(m)), blocks, 1.0)
        return tuple(block * sign for block, sign in zip(blocks, signs))

    def full_blocks(self, which: 'str' = "efie") -> 'np.ndarray':
        """Every retained mode of one family as a full ``[4, modes, Nn, Nn]`` array
        (a reconstruction for packed EFIE storage; the store itself otherwise)."""
        store = {"efie": self.Z, "mfie": self.K, "ibc": self.B}[which]
        if store is None:
            raise ValueError(f"Streaming far blocks were not built for {which!r}.")
        if which != "efie" or not self._packed_efie:
            return store
        modes = [m for m, _ in sorted(self._sidx.items(), key=lambda item: item[1])]
        out = np.empty((4, len(modes), self.Nn, self.Nn), dtype=np.complex128)
        for mi, m in enumerate(modes):
            for uv, block in enumerate(self._full_efie_mode(m)):
                out[uv, mi] = block
        return out

    def efie_blocks(self, m: 'int'):
        if self._closed:
            raise _closed_stream_error()
        if self._packed_efie:
            blocks = tuple(np.empty((self.Nn, self.Nn), dtype=np.complex128) for _ in range(4))
            self.write_efie_blocks(m, blocks, 1.0)
            return blocks
        with self._range_lock:
            self._ensure(abs(m))
            return tuple(self.Z[uv, self._sidx[abs(m)]].astype(np.complex128)*mode_sign(uv,m)
                         for uv in range(4))

    def bracket_blocks(self, which: 'str', m: 'int'):
        with self._range_lock:
            self._ensure(abs(m))
            store = self.K if which == "mfie" else self.B
            mi = self._sidx[abs(m)]
            return tuple(store[uv, mi].astype(np.complex128)*mode_sign(uv,m) for uv in range(4))

    def stored_blocks(self, which: 'str', m: 'int'):
        """Retained blocks of one signed mode as ``(views, signs)``.

        The views alias the retained storage (possibly single precision) and
        must not be written; ``signs`` is the tangential-block parity of a
        negative mode.  In-place assembly scales them straight into the
        system quadrants instead of paying one full-matrix copy per family.
        Packed symmetric EFIE storage has no full views: its blocks are
        reconstructed into new arrays (in-place assembly uses
        :meth:`write_efie_blocks` instead).
        """
        if which == "efie" and self._packed_efie:
            if self._closed:
                raise _closed_stream_error()
            return (self._full_efie_mode(m), tuple(mode_sign(uv, m) for uv in range(4)))
        with self._range_lock:
            self._ensure(abs(m))
            store = {"efie": self.Z, "mfie": self.K, "ibc": self.B}[which]
            if store is None:
                raise ValueError(f"Streaming far blocks were not built for {which!r}.")
            mi = self._sidx[abs(m)]
            return (tuple(store[uv, mi] for uv in range(4)),
                    tuple(mode_sign(uv, m) for uv in range(4)))

    def memory_gb(self) -> 'float':
        total = 0
        for arr in (self.Z, self.K, self.B):
            if arr is not None and not isinstance(arr, np.memmap):
                total += arr.nbytes
        return total / 1e9


class StreamingCrossFarBlocks:
    """Bounded far blocks for one rectangular BorCrossOperators mapping.

    Test and source generatrices may have different element/node counts.  The
    stored blocks match :class:`StreamingFarBlocks`: four final EFIE blocks
    and, when requested, four unit-source rotated-PV blocks for nonnegative modes. Near pairs
    remain excluded here and are added by ``BorCrossOperators`` with its
    existing high-order/graded quadrature.  Tiles, threads and spill files
    follow :class:`StreamingFarBlocks` (every pair is sampled: the two
    surfaces differ, so there is no symmetry to use).
    """

    _EFIE_COMBOS = tuple((left, right) for _uv, left, right, _kernel, _coef in _EFIE_TERM_TABLE)

    def __init__(self, cross, m_max: 'int', dtype=np.complex128,
                 tile_budget_gb: 'float' = BOR_STREAM_TILE_BUDGET_GB,
                 workers: 'int' = 1, mode_block: 'Optional[int]' = None,
                 spill: 'Optional[str]' = None,
                 tile_threads: 'Optional[int]' = None):
        self.cross = cross
        self._has_ibc = cross.need_p
        self.sp, self.sq = cross.sp, cross.sq
        self.m_max = int(m_max)
        self.dtype = dtype
        gp, gq = self.sp.g, self.sq.g
        self.Np, self.Nq = self.sp.Nn, self.sq.Nn
        self.go_p, self.go_q = self.sp.gauss_order, self.sq.gauss_order
        self.Pq = self.sq.P
        self.k = complex(cross.k)

        def weights(g):
            wrho = g.w * g.rho
            return {
                "r": np.stack([g.T0 * wrho * g.trho,
                               g.T1 * wrho * g.trho]),
                "z": np.stack([g.T0 * wrho * g.tz,
                               g.T1 * wrho * g.tz]),
                "1": np.stack([g.T0 * wrho, g.T1 * wrho]),
                "s": np.stack([g.T0 * g.w, g.T1 * g.w]),
                "d": np.stack([g.dRT0 * g.w, g.dRT1 * g.w]),
            }

        self._lv = weights(gp)
        self._rv = weights(gq)
        ne_q = self.sq.gen.n_elems
        self._left_all = np.stack([self._lv[name] for name in _LEFT_KINDS], axis=0)
        self._right = {name: _source_weights(self._rv[name], ne_q, self.go_q)
                       for name in _LEFT_KINDS}
        self._right_groups = _stacked_right_groups(self._right)
        self._right_one = _stacked_right(self._right["1"])
        self.mode_workers = max(1, int(workers))
        self.mode_block = _aligned_stream_mode_block(
            self.m_max, mode_block, self.mode_workers
        )
        self.n_sweeps = 0
        self._closed = False
        self._spill = None
        self._spilled_bytes = 0
        self._finalizer = None
        self.Z = self.B = None
        self.lo, self.hi = 1, 0
        if spill is not None:
            self._spill = _SpillFiles(spill, "ghost-bor-cross-")
            self._finalizer = weakref.finalize(self, self._spill.release)
            self.mode_block = self.m_max + 1
        try:
            rho_max = max(
                float(np.max(self.sp.gen.nodes[:, 0])),
                float(np.max(self.sq.gen.nodes[:, 0])),
            )
            self._nx_e = _n_xi_efie(
                self.k, rho_max, self.m_max, cross._far_gap
            )
            self._nx_b = _n_xi_bracket(
                self.k, rho_max, self.m_max, cross._far_gap
            ) if self._has_ibc else self._nx_e
            nx_worst = max(self._nx_e, self._nx_b)
            # Tile threads are capped at the physical cores like the self
            # streams (the mode workers only align the ranges).
            _apply_tile_plan(self, tile_budget_gb, tile_threads, nx_worst,
                             self.sp.gen.n_elems, self.go_p, ne_q, self.go_q, self.Pq,
                             True, self._has_ibc)
            self._native = (
                _NATIVE if _NATIVE is not None and abs(self.k.imag) == 0.0
                else None
            )
            if _NATIVE is None:
                _notice_numpy_fallback()
            self._q = tuple(np.ascontiguousarray(value) for value in
                            (gq.rho, gq.z, gq.trho, gq.tz))
            self._near_sources = {
                e: [] for e in range(self.sp.gen.n_elems)
            }
            for e, f in cross.near_pairs:
                self._near_sources[e].append(f)
            self._acc_lock = threading.Lock()
            self._acc_locks = {"efie": self._acc_lock, "ibc": threading.Lock()}
            self._range_lock = threading.RLock()
            self._ord_lo = 0
            self._sidx: 'Dict[int, int]' = {}
            self._ensure(0)
        except BaseException:
            self.close()
            raise

    def _ensure(self, am: 'int') -> 'None':
        if self.lo <= am <= self.hi:
            return
        with self._range_lock:
            if self.lo <= am <= self.hi:
                return
            if self._closed:
                raise _closed_stream_error()
            lo = (am // self.mode_block) * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._build_range(lo, hi)

    _allocate = StreamingFarBlocks._allocate
    spilled_gb = StreamingFarBlocks.spilled_gb
    _add_band = StreamingFarBlocks._add_band

    def close(self) -> 'None':
        """Release the retained blocks and any spilled files."""
        self._closed = True
        self.Z = self.B = None
        self.lo, self.hi = 1, 0
        finalizer, self._finalizer = getattr(self, "_finalizer", None), None
        if finalizer is not None:
            finalizer()
        self._spill = None

    def _build_range(self, lo: 'int', hi: 'int') -> 'None':
        ne = self.sp.gen.n_elems
        ne_q = self.sq.gen.n_elems
        go_p = self.go_p
        ord_lo = max(0, lo - 1)
        modes = list(range(lo, hi+1))
        self.Z = self.B = None
        if self._spill is not None:
            self._spill.reserve((1 + int(self._has_ibc)) * 4 * len(modes) * self.Np * self.Nq
                                * np.dtype(self.dtype).itemsize)
        self.Z = self._allocate(
            "efie", (4, hi-lo+1, self.Np, self.Nq)
        )
        self._positive_modes = np.asarray(modes)
        self._sidx = {m: index for index, m in enumerate(modes)}
        self.B = self._allocate(
            "ibc", (4, len(modes), self.Np, self.Nq)
        ) if self._has_ibc else None
        self._ord_lo = ord_lo
        orders = np.arange(ord_lo, hi + 2)
        phase_e = (
            np.exp(1j * np.pi * orders) * (2.0 * np.pi / self._nx_e)
        )
        mode_array = np.asarray(modes)
        bins_b = np.where(
            mode_array >= 0, mode_array, self._nx_b + mode_array
        )
        phase_b = (
            np.exp(1j * np.pi * mode_array) * (2.0 * np.pi / self._nx_b)
        )
        te, fe = self._tile_rows, self._tile_sources
        release = (_SpilledRowRelease([(store, ("full", self.Nq))
                                      for store in (self.Z, self.B) if store is not None],
                                      ne, te, len(range(0, ne_q, fe)), self.Np)
                   if self._spill is not None else None)

        def do_tile(tile):
            e0, f0 = tile
            e1 = min(e0 + te, ne)
            f1 = min(f0 + fe, ne_q)
            rows = slice(e0 * go_p, e1 * go_p)
            re = e1 - e0
            Gn = self._sample_G(rows, phase_e, ord_lo, hi, (f0, f1))
            if not modal_kernels.BANDED_FFT:
                self._zero_near(Gn, e0, e1, f0, f1)
            left = self._left_all[:, :, rows].reshape(2 * len(_LEFT_KINDS), re, go_p)
            band = _efie_band(Gn, left, self._right_groups, self._positive_modes,
                              ord_lo, self.k, f0, f1, re, go_p)
            Gn = left = None
            self._add_band("efie", self.Z, band, e0, f0)
            band = None
            if self._has_ibc:
                brackets = self._sample_brackets(rows, re, bins_b, phase_b, (f0, f1))
                band = _bracket_band(
                    brackets, self._lv["1"][:, rows].reshape(2, re, go_p), self._right_one,
                    f0, f1, re, go_p, None if modal_kernels.BANDED_FFT else
                    (lambda kernel: self._zero_near(kernel, e0, e1, f0, f1)))
                brackets = None
                self._add_band("ibc", self.B, band, e0, f0)
            if release is not None:
                release.tile_done(e0)

        tiles = [(e0, f0) for e0 in range(0, ne, te) for f0 in range(0, ne_q, fe)]
        _run_tiles(tiles, do_tile, self._workers)
        if release is not None:
            release.finish()
        self.lo, self.hi = lo, hi
        self.n_sweeps += 1

    def _read_mode(self, store, m: 'int'):
        """The four blocks of one signed mode as complex128 copies; a spilled
        mode's pages are released once copied."""
        with self._range_lock:
            self._ensure(abs(m))
            index = self._sidx[abs(m)]
            blocks = []
            for uv in range(4):
                blocks.append(store[uv, index].astype(np.complex128) * mode_sign(uv, m))
                if getattr(store, "_mmap", None) is not None:
                    _release_spilled(store, [_spilled_rows(store, ("full", self.Nq), 0, self.Np,
                                                           (index,))[uv]], flush=False)
            return tuple(blocks)

    def write_blocks(self, which, m, targets):
        """Copy a cross mode straight into its final matrix quadrants."""
        if which == 'ibc' and not self._has_ibc:
            raise ValueError('This cross stream was prepared for EFIE only.')
        with self._range_lock:
            self._ensure(abs(m))
            store = self.Z if which == 'efie' else self.B
            index = self._sidx[abs(m)]
            views = tuple(store[uv, index] for uv in range(4))
        for uv, (view, target) in enumerate(zip(views, targets)):
            for lo, hi in _row_chunks(self.Np, self.Nq * store.dtype.itemsize):
                np.multiply(view[lo:hi], mode_sign(uv, m), out=target[lo:hi])
                if getattr(store, '_mmap', None) is not None:
                    _release_spilled(store, [_spilled_rows(store, ('full', self.Nq), lo, hi,
                                                           (index,))[uv]], flush=False)

    def _sample_G(self, rows, phase, ord_lo: 'int', hi: 'int', sources=None):
        ne_q = self.sq.gen.n_elems
        f0, f1 = (0, ne_q) if sources is None else sources
        if modal_kernels.BANDED_FFT:
            return _banded_stream(self, rows, 'g', np.arange(ord_lo, hi + 2), (f0, f1))
        gp = self.sp.g
        start, stop = f0 * self.go_q, f1 * self.go_q
        xi = 2.0 * np.pi * np.arange(self._nx_e) / self._nx_e - np.pi
        rp = np.ascontiguousarray(gp.rho[rows])
        zp = np.ascontiguousarray(gp.z[rows])
        sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
        kept = np.empty(
            (len(rp), stop - start, hi + 2 - ord_lo), dtype=np.complex128
        )
        for c0 in range(start, stop, self._cols):
            c1 = min(c0 + self._cols, stop)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q = np.ascontiguousarray(self._q[0][cols])
                z_q = np.ascontiguousarray(self._q[1][cols])
                sampled = np.empty(
                    (len(rp), nc, self._nx_e), dtype=np.complex128
                )
                self._native.sample_g(
                    len(rp), nc, self._nx_e, _dp(rp), _dp(zp),
                    _dp(rho_q), _dp(z_q), float(np.real(self.k)),
                    _dp(sin2), _dp(sampled),
                )
            else:
                d2 = (
                    (rp[:, None] - self.sq.g.rho[None, cols]) ** 2
                    + (zp[:, None] - self.sq.g.z[None, cols]) ** 2
                )
                rr4 = 4.0 * rp[:, None] * self.sq.g.rho[None, cols]
                radius = np.sqrt(
                    d2[..., None] + rr4[..., None] * sin2
                )
                radius = np.maximum(radius, 1.0e-300)
                sampled = (
                    np.exp(-1j * self.k * radius)
                    / (4.0 * np.pi * radius)
                )
            spectrum = np.fft.fft(sampled, axis=-1)
            kept[:, c0 - start:c1 - start] = spectrum[..., ord_lo:hi + 2] * phase
        return kept

    def _sample_brackets(self, rows, re: 'int', bins, phase, sources=None):
        ne_q = self.sq.gen.n_elems
        f0, f1 = (0, ne_q) if sources is None else sources
        if modal_kernels.BANDED_FFT:
            return _banded_stream(self, rows, 'ibc',
                                  np.where(bins > self._nx_b // 2, bins - self._nx_b, bins),
                                  (f0, f1))
        gp, gq = self.sp.g, self.sq.g
        nr = re * self.go_p
        start, stop = f0 * self.go_q, f1 * self.go_q
        xi = 2.0 * np.pi * np.arange(self._nx_b) / self._nx_b - np.pi
        kept = tuple(
            np.empty((nr, stop - start, len(bins)), dtype=np.complex128)
            for _ in range(4)
        )
        rp = np.ascontiguousarray(gp.rho[rows])
        zp = np.ascontiguousarray(gp.z[rows])
        trp = np.ascontiguousarray(gp.trho[rows])
        tzp = np.ascontiguousarray(gp.tz[rows])
        cos_xi = np.ascontiguousarray(np.cos(xi))
        sin_xi = np.ascontiguousarray(np.sin(xi))
        for c0 in range(start, stop, self._cols):
            c1 = min(c0 + self._cols, stop)
            cols = slice(c0, c1)
            nc = c1 - c0
            if self._native is not None:
                rho_q, z_q, tr_q, tz_q = (
                    np.ascontiguousarray(value[cols]) for value in self._q
                )
                sampled = tuple(
                    np.empty((nr, nc, self._nx_b), dtype=np.complex128)
                    for _ in range(4)
                )
                self._native.sample_ibc(
                    nr, nc, self._nx_b,
                    _dp(rp), _dp(zp), _dp(trp), _dp(tzp),
                    _dp(rho_q), _dp(z_q), _dp(tr_q), _dp(tz_q),
                    float(np.real(self.k)), _dp(cos_xi), _dp(sin_xi),
                    _dp(sampled[0]), _dp(sampled[1]),
                    _dp(sampled[2]), _dp(sampled[3]),
                )
            else:
                pair_shape = (nr, nc)
                sampled = _ibc_brackets_grid(
                    np.broadcast_to(rp[:, None], pair_shape).ravel(),
                    np.broadcast_to(zp[:, None], pair_shape).ravel(),
                    np.broadcast_to(trp[:, None], pair_shape).ravel(),
                    np.broadcast_to(tzp[:, None], pair_shape).ravel(),
                    np.broadcast_to(gq.rho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.z[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.trho[None, cols], pair_shape).ravel(),
                    np.broadcast_to(gq.tz[None, cols], pair_shape).ravel(),
                    self.k,
                    np.broadcast_to(xi, (nr * nc, self._nx_b)),
                )
                sampled = tuple(
                    value.reshape(nr, nc, self._nx_b)
                    for value in sampled
                )
            for uv, values in enumerate(sampled):
                spectrum = np.fft.fft(values, axis=-1)
                kept[uv][:, c0 - start:c1 - start] = (
                    spectrum[..., bins] * (2.0 * np.pi * phase)
                )
        return kept

    def _zero_near(self, values, e0: 'int', e1: 'int', f0: 'int' = 0,
                   f1: 'Optional[int]' = None) -> 'None':
        f1 = self.sq.gen.n_elems if f1 is None else f1
        for e in range(e0, e1):
            row = slice(
                (e - e0) * self.go_p, (e - e0 + 1) * self.go_p
            )
            for f in self._near_sources[e]:
                if f0 <= f < f1:
                    values[
                        row, (f - f0) * self.go_q:(f - f0 + 1) * self.go_q
                    ] = 0.0

    def efie_blocks(self, m: 'int'):
        with self._range_lock:
            self._ensure(abs(m))
            return self._read_mode(self.Z, m)

    def bracket_blocks(self, m: 'int'):
        if not self._has_ibc:
            raise ValueError('This cross stream was prepared for EFIE only.')
        with self._range_lock:
            self._ensure(abs(m))
            return self._read_mode(self.B, m)

    def memory_gb(self) -> 'float':
        return sum(
            value.nbytes for value in (self.Z, self.B)
            if value is not None and not isinstance(value, np.memmap)
        ) / 1.0e9


def estimate_rectangular_streaming_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    has_rotated_pv: 'bool' = True, single_blocks: 'bool' = False,
) -> 'float':
    """All-mode retained blocks for one rectangular EFIE/PV mapping.

    Retained blocks only: the transient tile workspace of all concurrent tiles
    is bounded by ``BOR_STREAM_TILE_BUDGET_GB`` at run time (each of the
    :func:`streaming_tile_threads` tiles is sized to its share) and callers
    price it once, separately.
    """

    test_nodes = float(int(n_test_elements) + 1)
    source_nodes = float(int(n_source_elements) + 1)
    modes = int(m_max)
    item_bytes = 8.0 if single_blocks else 16.0
    total = 4.0 * test_nodes * source_nodes * (modes + 1) * item_bytes
    if has_rotated_pv:
        total += (
            4.0 * test_nodes * source_nodes
            * (modes + 1) * item_bytes
        )
    return total / 1.0e9


def estimate_rectangular_streaming_block_gb(
    n_test_elements: 'int', n_source_elements: 'int', m_max: 'int',
    mode_block: 'int', has_rotated_pv: 'bool' = True,
    single_blocks: 'bool' = False,
) -> 'float':
    """Worst retained range for one rectangular EFIE/PV mapping.

    The tile workspace is priced separately (see
    :func:`estimate_rectangular_streaming_gb`).
    """

    nt = int(n_test_elements)
    ns = int(n_source_elements)
    mm = int(m_max)
    block = int(mode_block)
    if nt < 1 or ns < 1 or mm < 0 or block < 1 or block > mm + 1:
        raise ValueError("Rectangular streaming estimate dimensions are invalid.")
    node_pairs = float((nt + 1) * (ns + 1))
    item_bytes = 8.0 if single_blocks else 16.0
    worst = 0.0
    for lo in range(0, mm + 1, block):
        hi = min(lo + block - 1, mm)
        order_lo = max(0, lo - 1)
        efie_orders = hi + 2 - order_lo
        signed_modes = hi-lo+1
        total = 4.0 * signed_modes * node_pairs * item_bytes
        if has_rotated_pv:
            total += 4.0 * signed_modes * node_pairs * item_bytes
        worst = max(worst, total)
    return worst / 1.0e9


class StreamingMemoryError(BorAdmissionError, ValueError):
    """Valid streaming request whose smallest retained block exceeds its budget.

    Raised while planning, before any preparation: an admission rejection of a
    streamed plan.
    """

    def __init__(self, message, mode_cap=None):
        super().__init__(message, streaming=True, mode_cap=mode_cap)


def plan_combined_streaming_mode_block(
    m_max: 'int', requirements, stream_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, float, int]':
    """Plan one aligned range shared by several self/cross far streams.

    Each requirement is ``(test_elements, source_elements, has_rotated_pv,
    single_blocks)``.  The returned retained peak is the sum of every stream's
    current block; transient sampling tiles are budgeted separately by the
    caller because streams are constructed sequentially.  ``workers`` are the
    outer mode workers (range alignment only); the tiles of one build run on
    :func:`streaming_tile_threads` threads within ``BOR_STREAM_TILE_BUDGET_GB``.
    """

    mm = int(m_max)
    budget = float(stream_budget_gb)
    specs = list(requirements)
    if mm < 0 or not specs:
        raise ValueError("Combined streaming planning needs modes and streams.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming block budget must be positive and finite.")

    def retained(block):
        return sum(
            estimate_rectangular_streaming_block_gb(
                nt, ns, mm, block, bool(rotated), bool(single)
            )
            for nt, ns, rotated, single in specs
        )

    minimum = retained(1)
    if minimum > budget:
        raise StreamingMemoryError(
            "Combined streaming block budget is below the modeled one-mode "
            f"retained minimum of {minimum:.6g} GB.",
            mode_cap=mm,
        )
    mode_count = mm + 1
    low, high, max_safe = 1, mode_count, 1
    while low <= high:
        candidate = (low + high) // 2
        if retained(candidate) <= budget:
            max_safe = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    effective_workers = min(max(1, int(workers)), max_safe)
    if max_safe == mode_count:
        aligned = mode_count
    else:
        aligned = (max_safe // effective_workers) * effective_workers
    held = retained(aligned)
    if held > budget:
        raise RuntimeError(
            "Internal combined streaming planner error: aligned block "
            "exceeds its retained-memory budget."
        )
    return aligned, held, effective_workers


def _self_efie_block_entries(nodes: 'float') -> 'float':
    """Retained entries of one self-surface far EFIE block of one mode.

    The symmetric build keeps the packed strict-upper triangle (every BoR
    surface has a symmetric near relation spanning adjacent elements);
    ``STREAM_EFIE_SYMMETRY = False`` keeps the full block.  Rectangular
    (cross) mappings are always full: see
    :func:`estimate_rectangular_streaming_block_gb`.
    """
    nodes = float(nodes)
    return nodes * (nodes - 1.0) / 2.0 if STREAM_EFIE_SYMMETRY else nodes * nodes


def estimate_streaming_gb(n_elems: 'int', m_max: 'int', formulation: 'str' = "cfie",
                          has_ibc: 'bool' = False,
                          single_blocks: 'bool' = False) -> 'float':
    """Persistent per-mode nodal block memory (GB) for the streaming path.

    Retained blocks only.  The transient tile workspace of all concurrent
    tiles (:func:`streaming_tile_threads` of them, independent of the mode
    workers) is bounded by ``BOR_STREAM_TILE_BUDGET_GB`` at run time -- each
    tile is sized to its share of it -- and callers add that budget once.
    """
    from ghost_backend.bor.compressed_far import estimate_compressed_far_gb, far_compression_selected
    if far_compression_selected(int(n_elems) + 1):
        return (estimate_compressed_far_gb(n_elems, m_max, formulation, has_ibc)
                * (0.5 if single_blocks else 1.0))
    Nn = float(n_elems + 1)
    per = 8.0 if single_blocks else 16.0


    total = 4.0 * _self_efie_block_entries(Nn) * (m_max + 1) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * Nn * Nn * (m_max + 1) * per
    if has_ibc:
        total += 4.0 * Nn * Nn * (m_max + 1) * per
    return total / 1e9


def estimate_streaming_block_gb(
    n_elems: 'int', m_max: 'int', mode_block: 'int',
    formulation: 'str' = "cfie", has_ibc: 'bool' = False,
    single_blocks: 'bool' = False,
) -> 'float':
    """Worst retained range allocation for an already-aligned mode block.

    This mirrors :meth:`StreamingFarBlocks._build_range`, including the two
    final EFIE blocks and nonnegative MFIE/IBC modes. It is
    deliberately conservative for the diagnostics-only pure-MFIE path in the
    same way as :func:`estimate_streaming_gb`.  The tile workspace is priced
    separately (see :func:`estimate_streaming_gb`).
    """

    ne = int(n_elems)
    mm = int(m_max)
    block = int(mode_block)
    if ne < 1 or mm < 0 or block < 1 or block > mm + 1:
        raise ValueError("Streaming block estimate dimensions are invalid.")
    from ghost_backend.bor.compressed_far import far_compression_selected
    if far_compression_selected(ne + 1):
        # A shared spatial basis is retained for the live mode band only.
        return estimate_streaming_gb(ne, block - 1, formulation, has_ibc, single_blocks)
    nodes = float(ne + 1)
    item_bytes = 8.0 if single_blocks else 16.0
    worst = 0.0
    for lo in range(0, mm + 1, block):
        hi = min(lo + block - 1, mm)
        order_lo = max(0, lo - 1)
        efie_orders = hi + 2 - order_lo
        signed_modes = hi-lo+1
        total = 4.0 * signed_modes * _self_efie_block_entries(nodes) * item_bytes
        if formulation in ("cfie", "mfie"):
            total += 4.0 * signed_modes * nodes * nodes * item_bytes
        if has_ibc:
            total += 4.0 * signed_modes * nodes * nodes * item_bytes
        worst = max(worst, total)
    return worst / 1.0e9


def plan_streaming_mode_block(
    n_elems: 'int', m_max: 'int', formulation: 'str', has_ibc: 'bool',
    single_blocks: 'bool', stream_budget_gb: 'float', workers: 'int',
) -> 'tuple[int, float, int]':
    """Return a budget-safe block, retained peak, and effective workers.

    A streaming range cannot be smaller than the number of simultaneously
    solved outer modes: otherwise a worker wave can straddle two ranges and
    force a rebuild while another worker still reads the old range.  Treat
    ``stream_budget_gb`` as a hard retained-block limit by reducing that
    outer concurrency when necessary.  Only a budget below the true
    one-mode retained minimum is rejected.  ``workers`` are those outer mode
    workers; the far tiles of a build use their own threads within
    ``BOR_STREAM_TILE_BUDGET_GB``, which callers price once.
    """

    budget = float(stream_budget_gb)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("Streaming block budget must be positive and finite.")
    mode_count = int(m_max) + 1
    minimum = estimate_streaming_block_gb(
        n_elems, m_max, 1, formulation, has_ibc, single_blocks
    )
    if minimum > budget:
        raise StreamingMemoryError(
            "Streaming block budget is below the modeled one-mode retained "
            f"minimum of {minimum:.6g} GB for this geometry/formulation.",
            mode_cap=int(m_max),
        )


    low, high = 1, mode_count
    max_safe = 1
    while low <= high:
        candidate = (low + high) // 2
        retained = estimate_streaming_block_gb(
            n_elems, m_max, candidate, formulation, has_ibc, single_blocks
        )
        if retained <= budget:
            max_safe = candidate
            low = candidate + 1
        else:
            high = candidate - 1

    effective_workers = min(max(1, int(workers)), max_safe)
    if max_safe == mode_count:
        aligned = mode_count
    else:

        aligned = (max_safe // effective_workers) * effective_workers
    retained = estimate_streaming_block_gb(
        n_elems, m_max, aligned, formulation, has_ibc, single_blocks
    )
    if retained > budget:
        raise RuntimeError(
            "Internal streaming planner error: aligned block exceeds its "
            "retained-memory budget."
        )
    return aligned, retained, effective_workers
