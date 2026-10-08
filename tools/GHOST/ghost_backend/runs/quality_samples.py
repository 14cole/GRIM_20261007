"""Column-based sample matching for the existing mesh-convergence metrics."""
import numpy as np


def compact_comparison(base, fine):
    """Return the same ordered values as row matching, or use its general path.

    Python rounding is intentional: numpy.round differs at some decimal ties.
    Sorting uses rounded coordinates, then the exact coordinates within each
    collision, matching quality.evaluate_mesh_convergence's row algorithm.
    Invalid/extension formats fall back to its detailed fail-closed messages.
    """
    from ghost_backend.twod.samples import sample_columns
    tables = [sample_columns(rows) for rows in (base, fine)]
    if any(table is None for table in tables) or len(base) != len(fine):
        return None
    coordinate_fields = ('frequency_ghz', 'theta_inc_deg', 'theta_scat_deg')
    value_fields = ('rcs_amp_real', 'rcs_amp_imag', 'rcs_db')
    orders, rounded = [], []
    for table in tables:
        if any(not np.all(np.isfinite(table[key])) for key in coordinate_fields + value_fields):
            return None
        exact = [table[key] for key in coordinate_fields]
        keys = [np.fromiter((round(float(value), 9) for value in column), dtype=float, count=len(column))
                for column in exact]
        order = np.lexsort(tuple(reversed(exact)) + tuple(reversed(keys)))
        if len(order) > 1 and np.all(np.stack([column[order[1:]] == column[order[:-1]] for column in exact]), axis=0).any():
            return None  # duplicate exact coordinates: retain the existing error
        orders.append(order)
        rounded.append([column[order] for column in keys])
    if any(not np.array_equal(a, b) for a, b in zip(*rounded)):
        return None
    base_table, fine_table = tables
    a, b = orders
    base_amp = base_table['rcs_amp_real'][a] + 1j * base_table['rcs_amp_imag'][a]
    fine_amp = fine_table['rcs_amp_real'][b] + 1j * fine_table['rcs_amp_imag'][b]
    delta = base_table['rcs_db'][a] - fine_table['rcs_db'][b]
    frequencies, starts, counts = np.unique(rounded[0][0], return_index=True, return_counts=True)
    groups = {float(frequency): np.arange(start, start + count, dtype=np.intp)
              for frequency, start, count in zip(frequencies, starts, counts)}
    return base_amp, fine_amp, delta, groups
