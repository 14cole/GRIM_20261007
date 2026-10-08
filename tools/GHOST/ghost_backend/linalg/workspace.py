"""Linear-system checks with bounded temporary arrays (Python 3.6+)."""
import numpy as np


def first_nonfinite(value, block_bytes=1024*1024):
    array = np.asarray(value)
    if array.ndim == 0:
        return None if np.isfinite(array) else ()
    row_entries = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    rows = max(1, int(block_bytes) // max(1, row_entries))
    for start in range(0, len(array), rows):
        finite = np.isfinite(array[start:start+rows])
        if not np.all(finite):
            index = np.unravel_index(int(np.argmin(finite)), finite.shape)
            return (start + int(index[0]),) + tuple(int(i) for i in index[1:])
    return None


def matrix_inf_norm(matrix, block_bytes=1024*1024):
    a = np.asarray(matrix)
    if not a.size:
        return 0.
    if a.flags.f_contiguous and not a.flags.c_contiguous:
        sums = np.zeros(a.shape[0])
        width = max(1, int(block_bytes)//max(1, a.shape[0]*8))
        for start in range(0, a.shape[1], width):
            sums += np.sum(np.abs(a[:, start:start+width]), axis=1)
        return float(np.max(sums))
    rows = max(1, int(block_bytes) // max(1, a.shape[1]*8))
    largest = 0.0
    for start in range(0, len(a), rows):
        largest = max(largest, float(np.max(np.sum(np.abs(a[start:start+rows]), axis=1))))
    return largest


def checked_matrix_norms(matrix, one_norm=False, checkpoint=None):
    """Finite check, infinity norm and optional 1-norm with shared magnitudes.

    Keep the BoR 1-norm's 64-row reduction order. For Fortran-only matrices,
    retain the original column-blocked infinity-norm reduction as well.
    """
    a = np.asarray(matrix)
    if not a.size:
        return None, 0., 0. if one_norm else None
    checkpoint = checkpoint or (lambda: None)
    rows = 64 if one_norm else max(1, 1024**2 // max(1, a.shape[1]*8))
    columns = np.zeros(a.shape[1]) if one_norm else None
    fortran = a.flags.f_contiguous and not a.flags.c_contiguous
    largest = 0.
    for start in range(0, len(a), rows):
        checkpoint()
        magnitude = np.abs(a[start:start+rows])
        if not np.isfinite(magnitude).all():
            first = first_nonfinite(a)
            if first is not None:
                return first, None, None
        if not fortran:
            largest = max(largest, float(np.max(np.sum(magnitude, axis=1))))
        if one_norm:
            columns += np.sum(magnitude, axis=0)
    if fortran:
        largest = matrix_inf_norm(a)
    return None, largest, float(np.max(columns)) if one_norm else None


def checked_row_norms(matrix, block_bytes=1024*1024):
    """``(first non-finite index or None, infinity norm, row maxima of |a|)``
    of a 2-D matrix in one blocked pass.

    It replaces a :func:`first_nonfinite` pass followed by a
    :func:`matrix_inf_norm` pass, and the row maxima are the first pass of the
    row/column equilibration.  Blocks and sums follow :func:`matrix_inf_norm`
    exactly, so the norm is bitwise identical to it; on a non-finite entry the
    norm and maxima are None and the index is the one :func:`first_nonfinite`
    reports.
    """
    a = np.asarray(matrix)
    n_rows = a.shape[0]
    row_max = np.zeros(n_rows)
    if not a.size:
        return None, 0., row_max

    def nonfinite(magnitude):
        # |z| is non-finite for every non-finite entry (and on overflow).
        if np.isfinite(magnitude).all():
            return None
        return first_nonfinite(a)

    if a.flags.f_contiguous and not a.flags.c_contiguous:
        sums = np.zeros(n_rows)
        width = max(1, int(block_bytes)//max(1, n_rows*8))
        for start in range(0, a.shape[1], width):
            magnitude = np.abs(a[:, start:start+width])
            first = nonfinite(magnitude)
            if first is not None:
                return first, None, None
            sums += np.sum(magnitude, axis=1)
            np.maximum(row_max, np.max(magnitude, axis=1), out=row_max)
        return None, float(np.max(sums)), row_max
    rows = max(1, int(block_bytes) // max(1, a.shape[1]*8))
    largest = 0.0
    for start in range(0, n_rows, rows):
        magnitude = np.abs(a[start:start+rows])
        first = nonfinite(magnitude)
        if first is not None:
            return first, None, None
        largest = max(largest, float(np.max(np.sum(magnitude, axis=1))))
        row_max[start:start+rows] = np.max(magnitude, axis=1)
    return None, largest, row_max
