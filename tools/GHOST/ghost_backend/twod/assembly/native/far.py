"""Native far-field block quadrature over an already-qualified kernel table."""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import numpy as np

@lru_cache(None)
def _dll():
    path = Path(__file__).with_name('ghost_far.dll' if os.name == 'nt' else 'libghost_far.so')
    try:
        return ct.CDLL(str(path))
    except OSError:
        return None


@lru_cache(None)
def scatter_library():
    try:
        fun = _dll().ghost_scatter_columns
    except AttributeError:
        return None
    fun.argtypes = [ct.c_int64, ct.c_int64, ct.c_int] + [ct.c_void_p]*4 + [ct.c_int64] + [ct.c_void_p]*2 + [ct.c_int64]*4
    fun.restype = ct.c_int
    return fun


@lru_cache(None)
def tile_library():
    """(ghost_scatter_tile, fused-product support) or None for an older library."""
    dll = _dll()
    try:
        fun, fused = dll.ghost_scatter_tile, dll.ghost_scatter_has_fused
    except AttributeError:
        return None
    pointer = ct.c_void_p
    fun.argtypes = ([ct.c_int64, ct.c_int64, ct.c_int] + [pointer]*4 + [ct.c_int64] +
                    [pointer] + [ct.c_int64]*3 + [ct.c_int, pointer, ct.c_int64, ct.c_int64] +
                    [pointer, ct.c_int] + [pointer] + [ct.c_int64]*4)
    fun.restype = ct.c_int
    fused.restype = ct.c_int
    return fun, bool(fused())


@lru_cache(None)
def numpy_fused_product():
    """True when numpy's complex product rounds with fused multiply-adds here.

    On FMA hardware numpy forms re = fma(ar, br, -(ai*bi)), im = fma(ar, bi, ai*br);
    otherwise the plain two-rounding products. The native tile scatter reproduces
    whichever this process's numpy uses, so its sums equal the numpy route's.
    """
    from fractions import Fraction

    def fma(x, y, z):
        return float(Fraction(x) * Fraction(y) + Fraction(z))
    rng = np.random.RandomState(20260922)
    a = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    b = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    product = a * b
    fused = plain = 0
    for x, y, z in zip(a, b, product):
        ar, ai, br, bi = float(x.real), float(x.imag), float(y.real), float(y.imag)
        fused += (fma(ar, br, -(ai * bi)), fma(ar, bi, ai * br)) == (z.real, z.imag)
        plain += (ar * br - ai * bi, ar * bi + ai * br) == (z.real, z.imag)
    if fused == len(a):
        return True
    if plain == len(a):
        return False
    return None


def _element_strides(array, itemsize):
    strides = []
    for stride in array.strides:
        if stride % itemsize:
            return None
        strides.append(stride // itemsize)
    return strides


def scatter_tile(matrix, rows, columns, row_map, column_map, values, scale=None, weights=None):
    """matrix[row_map[rows[i]], column_map[columns[j, b]]] += (values[b, i, j]*scale[i, j])*weights[rows[i]].

    One route of SystemScatter.scatter_add_columns with the scale and route weight
    applied on the fly: the same products and the same per-entry order as numpy's
    weighted tile followed by ufunc.at, without forming the weighted copy. `values`
    and `scale` may be strided or transposed views. False when unavailable.
    """
    library = tile_library()
    if (library is None or not isinstance(matrix, np.ndarray) or matrix.dtype != np.complex128 or
            matrix.ndim != 2 or not matrix.flags.writeable):
        return False
    fun, has_fused = library
    fused = False
    if weights is not None or (scale is not None and np.iscomplexobj(scale)):
        fused = numpy_fused_product()
        if fused is None or (fused and not has_fused):
            return False
    values = np.asarray(values)
    if values.dtype != np.complex128 or values.ndim != 3:
        return False
    matrix_strides = _element_strides(matrix, 16)
    value_strides = _element_strides(values, 16)
    if matrix_strides is None or value_strides is None or min(matrix_strides) < 0 or min(value_strides) < 0:
        return False
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    columns = np.ascontiguousarray(columns, dtype=np.int64)
    row_map = np.ascontiguousarray(row_map, dtype=np.int64)
    column_map = np.ascontiguousarray(column_map, dtype=np.int64)
    m, (n, width) = len(rows), columns.shape
    nodes = len(row_map)
    if rows.ndim != 1 or values.shape != (width, m, n) or column_map.shape != (nodes,):
        raise ValueError('Invalid native scatter layout.')
    kind, scale_pointer, scale_strides = 0, None, (0, 0)
    if scale is not None:
        scale = np.asarray(scale)
        if np.iscomplexobj(scale):
            scale = scale.astype(np.complex128, copy=False)
            kind, size = 2, 16
        else:
            scale = scale.astype(np.float64, copy=False)
            kind, size = 1, 8
        scale = np.broadcast_to(scale, (m, n))
        scale_strides = _element_strides(scale, size)
        if scale_strides is None or min(scale_strides) < 0:
            return False
        scale_pointer = scale.ctypes.data
    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.complex128)
        if weights.shape != (nodes,):
            raise ValueError('Invalid native scatter weights.')
    nrows, ncols = matrix.shape
    status = fun(m, n, width, rows.ctypes.data, columns.ctypes.data, row_map.ctypes.data,
                 column_map.ctypes.data, nodes, values.ctypes.data, *value_strides,
                 kind, scale_pointer, *scale_strides,
                 None if weights is None else weights.ctypes.data, int(bool(fused)),
                 matrix.ctypes.data, nrows, ncols, *matrix_strides)
    if status == 4:
        return False
    if status:
        raise ValueError('Native scatter indices fall outside the destination.')
    return True


def scatter_columns(matrix, rows, columns, row_map, column_map, values):
    """Native SystemScatter.scatter_add_columns for one route of weighted values; False if unavailable."""
    if tile_library() is not None:
        return scatter_tile(matrix, rows, columns, row_map, column_map, values)
    fun = scatter_library()
    if (fun is None or not isinstance(matrix, np.ndarray) or matrix.dtype != np.complex128 or matrix.ndim != 2 or
            not matrix.flags.writeable or not (matrix.flags.f_contiguous or matrix.flags.c_contiguous)):
        return False
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    columns = np.ascontiguousarray(columns, dtype=np.int64)
    row_map = np.ascontiguousarray(row_map, dtype=np.int64)
    column_map = np.ascontiguousarray(column_map, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    m, (n, width) = len(rows), columns.shape
    nodes = len(row_map)
    if rows.ndim != 1 or values.shape != (width, m, n) or column_map.shape != (nodes,):
        raise ValueError('Invalid native scatter layout.')
    nrows, ncols = matrix.shape
    strides = (1, nrows) if matrix.flags.f_contiguous else (ncols, 1)
    status = fun(m, n, width, rows.ctypes.data, columns.ctypes.data, row_map.ctypes.data, column_map.ctypes.data,
                 nodes, values.ctypes.data, matrix.ctypes.data, nrows, ncols, *strides)
    if status:
        raise ValueError('Native scatter indices fall outside the destination.')
    return True


@lru_cache(None)
def library():
    try:
        fun = _dll().ghost_far_block
    except AttributeError:
        return None
    pointer = ct.c_void_p
    fun.argtypes = [ct.c_int64, ct.c_int64, ct.c_int, ct.c_int,
                    pointer, pointer, pointer, pointer, pointer, pointer, pointer,
                    ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                    ct.c_double, ct.c_double,
                    ct.c_int, pointer, ct.c_int, pointer,
                    pointer, pointer, pointer]
    fun.restype = ct.c_int
    return fun


def _double(value):
    return np.ascontiguousarray(value, dtype=np.float64)


def far_block(table, k, obs_pts, src_pts, weights, phi, obs_normals, src_normals, far,
              obs_normal_deriv, want_s, want_k, mirrored):
    """Accumulators (width**2, mb, nb) for S, K and mirrored K, or None to fall back.

    Only pairs marked in `far` are integrated; the others stay zero. None means the
    library or table is unavailable or a far distance fell below the table.
    """
    fun = library()
    if fun is None or table is None or not getattr(table, 'native_checked', False):
        return None
    obs_pts, src_pts = _double(obs_pts), _double(src_pts)
    weights, phi = _double(weights), _double(phi)
    obs_normals, src_normals = _double(obs_normals), _double(src_normals)
    far = np.ascontiguousarray(far, dtype=np.uint8)
    mb, q = obs_pts.shape[:2]
    nb, width = src_pts.shape[0], phi.shape[1]
    if (obs_pts.shape != (mb, q, 2) or src_pts.shape != (nb, q, 2) or weights.shape != (q,) or
            phi.shape != (q, width) or obs_normals.shape != (mb, 2) or src_normals.shape != (nb, 2) or
            far.shape != (mb, nb)):
        raise ValueError('Invalid native far-block layout.')
    bounds, coefficients = table.bounds, table.normalized_power
    if (bounds.dtype != np.float64 or not bounds.flags.c_contiguous or coefficients.dtype != np.complex128 or
            not coefficients.flags.c_contiguous or coefficients.shape[0] != len(bounds)-1 or
            coefficients.shape[2] != 2):
        raise ValueError('Invalid native kernel-table layout.')
    k = complex(k)

    def buffer(wanted):
        return np.zeros((width*width, mb, nb), dtype=np.complex128) if wanted else None
    acc_s, acc_k, acc_kt = buffer(want_s), buffer(want_k), buffer(want_k and mirrored)

    def run(start, stop):
        # Rows start:stop of a (width**2, mb, nb) buffer are not contiguous, so
        # split chunks fill their own buffers and are copied back below.
        whole = start == 0 and stop == mb
        parts = [value if whole or value is None else np.zeros((width*width, stop-start, nb), np.complex128)
                 for value in (acc_s, acc_k, acc_kt)]
        status = fun(stop-start, nb, q, width, obs_pts[start:].ctypes.data, src_pts.ctypes.data,
                     weights.ctypes.data, phi.ctypes.data, obs_normals[start:].ctypes.data,
                     src_normals.ctypes.data, far[start:].ctypes.data,
                     int(bool(obs_normal_deriv)), int(bool(want_s)), int(bool(want_k)), int(bool(mirrored)),
                     k.real, k.imag, len(bounds)-1, bounds.ctypes.data, coefficients.shape[1]-1,
                     coefficients.ctypes.data, *(None if part is None else part.ctypes.data for part in parts))
        return status, start, stop, parts
    # The assembly tile scheduler owns concurrency. A second pool both hid the
    # user's one-thread setting and capped larger runs at four native calls.
    # One native call also avoids copying partial output buffers back together.
    results = [run(0, mb)]
    for status, start, stop, parts in results:
        if status == 2:
            return None
        if status:
            raise RuntimeError('Native far-block quadrature rejected its inputs.')
        for target, part in zip((acc_s, acc_k, acc_kt), parts):
            if target is not None and part is not target:
                target[:, start:stop] = part
    return acc_s, acc_k, acc_kt
