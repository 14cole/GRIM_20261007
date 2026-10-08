"""Store nonnegative near-field modes using exact tangential-block parity.

For EFIE, MFIE and unrotated IBC brackets, tt/ff are even in m and tf/ft
are odd. This holds for complex media as well: no complex conjugation is
involved. Apply the parity before the PMCHWT row rotation.
"""
import numpy as np


class ModalBands:
    """Append checked modal coefficients without copying already solved modes.

    Production reads select one mode. Full-array conversion is provided for
    diagnostics only; it is deliberately absent from modal assembly.
    """
    def __init__(self, previous, following):
        self.bands = list(previous.bands) if isinstance(previous, ModalBands) else [previous]
        self.bands.append(following)
        self.ends = np.cumsum([value.shape[1] for value in self.bands])
        self.shape = (following.shape[0], int(self.ends[-1])) + following.shape[2:]
        self.dtype = following.dtype

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.bands) + self.ends.nbytes

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) >= 2 and np.isscalar(key[1]):
            mode = int(key[1])
            if mode < 0:
                mode += self.shape[1]
            if not 0 <= mode < self.shape[1]:
                raise IndexError('Modal coefficient index is outside its prepared bands.')
            index = int(np.searchsorted(self.ends, mode, side='right'))
            start = 0 if index == 0 else int(self.ends[index-1])
            return self.bands[index][(key[0], mode-start) + key[2:]]
        return np.asarray(self)[key]

    def __array__(self, dtype=None, copy=None):
        value = np.concatenate(self.bands, axis=1)
        return value if dtype is None else value.astype(dtype, copy=False)


def mode_sign(component, mode):
    return -1 if mode < 0 and component in (1, 2) else 1


def mode_blocks(values, mode):
    """Read one signed mode from [4, m_max+1, ...] retained coefficients."""
    result = values[:, abs(mode)]
    if mode < 0:
        signs = np.array([1, -1, -1, 1]).reshape((4,) + (1,) * (result.ndim - 1))
        result = result * signs
    return result


def compact_layout(pairs, node_count, preserve_sources=False):
    """Pre-index element corners into compact nodal (and source) entries.

    Only integer routing arrays scale with the original pair count. Numerical
    coefficients are accumulated directly into the returned unique entries.
    """
    pair_array = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    rows = (pair_array[:, 0, None] + np.array([0, 0, 1, 1])).ravel()
    cols = (pair_array[:, 1, None] + np.array([0, 1, 0, 1])).ravel()
    sources = np.repeat(pair_array[:, 1], 4)
    keys = np.column_stack((rows, cols, sources)) if preserve_sources else np.column_stack((rows, cols))
    unique, destination = np.unique(keys, axis=0, return_inverse=True)
    rows, cols = unique[:, 0].copy(), unique[:, 1].copy()
    sources = unique[:, 2].copy() if preserve_sources else np.zeros(len(unique), dtype=np.intp)
    return dict(rows=rows, cols=cols, source_elems=sources,
                row_order=np.arange(len(rows), dtype=np.intp),
                row_ptr=np.searchsorted(rows, np.arange(node_count + 1))), destination.reshape(-1, 4)


def reciprocal_pair_order(pairs):
    """Integrate both directions independently, adjacent in the result stream.

    This bounds the raw reciprocity diagnostic to one pair's coefficients.
    It assumes each requested directed pair occurs once, as solver pair lists do.
    """
    requested = {tuple(pair) for pair in pairs}
    seen = set()
    ordered = []
    for pair in pairs:
        pair = tuple(pair)
        if pair in seen:
            continue
        ordered.append(pair)
        seen.add(pair)
        reverse = pair[::-1]
        if reverse in requested and reverse not in seen:
            ordered.append(reverse)
            seen.add(reverse)
    return ordered


class EfieReciprocity:
    """The original pair-based diagnostic, without retaining every raw block."""
    def __init__(self, modes):
        self.scale = np.zeros(modes)
        self.violation = np.zeros(modes)
        self.pending = None

    def add(self, pair, block):
        block = np.asarray(block).reshape(4, len(self.scale), 2, 2)
        self.scale = np.maximum(self.scale, np.max(np.abs(block), axis=(0, 2, 3)))
        own = None
        if pair[0] == pair[1]:
            own = block
        elif self.pending is not None and self.pending[0] == pair[::-1]:
            own = self.pending[1]
        if own is not None:
            mirror = block.transpose(0, 1, 3, 2)
            violation = np.max(np.abs(np.stack((own[0]-mirror[0], own[3]-mirror[3],
                                                own[1]+mirror[2], own[2]+mirror[1]))), axis=(0, 2, 3))
            self.violation = np.maximum(self.violation, violation)
            self.pending = None
        else:
            self.pending = (pair, block.copy())

    def value(self):
        return float(np.max(self.violation / np.where(self.scale > 0., self.scale, 1.), initial=0.))
