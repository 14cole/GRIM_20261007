"""Compact, run-owned sample storage for application sweeps.

Public solver calls still return ordinary lists by default. Application callers
can opt into column storage; iteration/indexing then materializes one independent
row dictionary at a time. Complex amplitudes and all numeric columns remain
float64. No field is recomputed, rounded, or omitted.
"""
from bisect import bisect_right
from collections.abc import Sequence
from contextlib import contextmanager
import operator
import numpy as np
from ghost_backend.execution.runtime import ScopedValue

FIELDS = ('frequency_ghz', 'theta_inc_deg', 'theta_scat_deg', 'rcs_linear',
          'rcs_db', 'rcs_amp_real', 'rcs_amp_imag', 'rcs_amp_phase_deg',
          'linear_residual')
BOR_FIELDS = FIELDS + ('linear_backward_error',)
_COMPACT = ScopedValue('ghost_compact_samples', False)


@contextmanager
def compact_samples():
    with _COMPACT.override(True):
        yield


def compact_samples_enabled():
    return bool(_COMPACT.get())


def sample_buffer(count, fields=FIELDS):
    return SampleTable(None, 0, capacity=count, fields=fields) if _COMPACT.get() else []


class SampleTable(Sequence):
    """Numeric rows plus shared channel labels; row dictionaries are copies."""
    def __init__(self, data, count=None, labels=None, capacity=None, fields=FIELDS):
        self._data = data
        self.capacity = len(data) if data is not None else capacity
        self.count = len(data) if count is None else count
        self.labels = dict(labels or {})
        self.fields = tuple(fields)

    @property
    def data(self):
        if self._data is None:
            # Admission happens before the first solved sample is recorded.
            self._data = np.empty((self.capacity, len(self.fields)), dtype=np.float64)
        return self._data

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        result = dict(zip(self.fields, map(float, self.data[index])))
        result.update({key: value[index].item() if isinstance(value, np.ndarray) else value
                       for key, value in self.labels.items()})
        return result

    def __iter__(self):
        labels = self.labels
        for index, values in enumerate(self.data[:self.count]):
            result = dict(zip(self.fields, map(float, values)))
            result.update({key: value[index].item() if isinstance(value, np.ndarray) else value
                           for key, value in labels.items()})
            yield result

    def append(self, row):
        if self.count >= len(self.data):
            raise ValueError('Sample buffer exceeded the requested grid.')
        self.data[self.count] = [row[key] for key in self.fields]
        self.count += 1

    def with_labels(self, **labels):
        return SampleTable(self.data, self.count, dict(self.labels, **labels), fields=self.fields)

    @property
    def nbytes(self):
        return self._data.nbytes if self._data is not None else 0


class InterleavedSamples(Sequence):
    def __init__(self, first, second, second_order=None):
        self.first, self.second, self.second_order = first, second, second_order

    def __len__(self):
        return 2 * len(self.first)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row, channel = divmod(index, 2)
        return self.first[row] if channel == 0 else self.second[row if self.second_order is None else self.second_order[row]]

    def __iter__(self):
        if self.second_order is None:
            for first, second in zip(self.first, self.second):
                yield first
                yield second
        else:
            for index, first in enumerate(self.first):
                yield first
                yield self.second[self.second_order[index]]


class SampleChunks(Sequence):
    """Concatenate completed frequencies without expanding rows or copying data."""
    def __init__(self):
        self.chunks, self.ends = [], []

    def extend(self, rows):
        if len(rows):
            self.ends.append(len(self) + len(rows))
            self.chunks.append(rows)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __iter__(self):
        for rows in self.chunks:
            yield from rows

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = bisect_right(self.ends, index)
        return self.chunks[chunk][index - (self.ends[chunk - 1] if chunk else 0)]


class SampleSelection(Sequence):
    """A channel or incidence view of a table; never copies numeric fields."""
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, np.asarray(indices, dtype=np.intp)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.rows[i] for i in self.indices[index]]
        return self.rows[self.indices[index]]

    def __iter__(self):
        for index in self.indices:
            yield self.rows[index]


class LabeledSamples(Sequence):
    """Attach constant channel labels without copying a selected table."""
    def __init__(self, rows, **labels):
        self.rows, self.labels = rows, dict(labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return dict(self.rows[index], **self.labels)

    def __iter__(self):
        for row in self.rows:
            yield dict(row, **self.labels)


def sample_columns(rows):
    """Native checkpoint columns, or None for ordinary/extension row formats."""
    if isinstance(rows, SampleTable):
        result = {key: rows.data[:len(rows), index] for index, key in enumerate(rows.fields)}
        result.update({key: value[:len(rows)] if isinstance(value, np.ndarray) else np.full(len(rows), value)
                       for key, value in rows.labels.items()})
        return result
    if isinstance(rows, LabeledSamples):
        result = sample_columns(rows.rows)
        if result is not None:
            result.update({key: np.full(len(rows), value) for key, value in rows.labels.items()})
        return result
    if isinstance(rows, InterleavedSamples):
        first, second = sample_columns(rows.first), sample_columns(rows.second)
        if first is None or second is None or set(first) != set(second):
            return None
        result = {}
        for key in first:
            column = np.empty(len(rows), dtype=np.result_type(first[key], second[key]))
            column[::2] = first[key]
            column[1::2] = second[key] if rows.second_order is None else second[key][rows.second_order]
            result[key] = column
        return result
    if isinstance(rows, SampleSelection):
        columns = sample_columns(rows.rows)
        return {key: value[rows.indices] for key, value in columns.items()} if columns is not None else None
    if isinstance(rows, SampleChunks):
        chunks = [sample_columns(chunk) for chunk in rows.chunks]
        if not chunks or any(chunk is None or set(chunk) != set(chunks[0]) for chunk in chunks):
            return None
        return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    return None


def sample_column(rows, key):
    """Read one compact column without expanding the other numeric fields."""
    if isinstance(rows, SampleTable):
        if key in rows.fields:
            return rows.data[:len(rows), rows.fields.index(key)]
        if key in rows.labels:
            value = rows.labels[key]
            return value[:len(rows)] if isinstance(value, np.ndarray) else np.full(len(rows), value)
    elif isinstance(rows, LabeledSamples):
        return np.full(len(rows), rows.labels[key]) if key in rows.labels else sample_column(rows.rows, key)
    elif isinstance(rows, InterleavedSamples):
        first, second = sample_column(rows.first, key), sample_column(rows.second, key)
        if first is not None and second is not None:
            column = np.empty(len(rows), dtype=np.result_type(first, second))
            column[::2] = first
            column[1::2] = second if rows.second_order is None else second[rows.second_order]
            return column
    elif isinstance(rows, SampleSelection):
        column = sample_column(rows.rows, key)
        return column[rows.indices] if column is not None else None
    elif isinstance(rows, SampleChunks):
        chunks = [sample_column(chunk, key) for chunk in rows.chunks]
        if chunks and all(column is not None for column in chunks):
            return np.concatenate(chunks)
    return None


def sorted_samples(rows, keys=FIELDS[:3]):
    """Stable coordinate sort retaining compact rows as an indexed view."""
    columns = [sample_column(rows, key) for key in keys]
    if all(column is not None for column in columns):
        if all(column.dtype.kind in 'US' or np.all(np.isfinite(column)) for column in columns):
            order = np.lexsort(tuple(reversed(columns)))
        else:
            # Preserve Python's legacy ordering for nonfinite extension input.
            order = sorted(range(len(rows)), key=lambda index: tuple(float(column[index]) for column in columns))
        return SampleSelection(rows, order)
    return sorted(rows, key=lambda row: tuple(float(row.get(key, 0.0)) for key in keys))


def checkpoint_samples(data, record):
    """Read the standard compact schema directly; leave extensions to the reader."""
    columns = record['columns']
    fields = BOR_FIELDS if 'linear_backward_error' in columns else FIELDS
    extras = set(columns) - set(fields)
    if (not _COMPACT.get() or not set(fields).issubset(columns)
            or not extras.issubset({'polarization', 'polarization_internal'})
            or any(encoding != 'scalar' for encoding in record['encodings'])):
        return None
    numeric, labels = None, {}
    for index, key in enumerate(columns):
        values, present = data['c' + str(index)], data['p' + str(index)]
        if values.ndim != 1 or present.shape != values.shape or not np.all(present):
            return None
        if numeric is None:
            numeric = np.empty((len(values), len(fields)), dtype=np.float64)
        if len(values) != len(numeric):
            return None
        if key in fields:
            if values.dtype.kind != 'f':
                return None
            numeric[:, fields.index(key)] = values
        else:
            if values.dtype.kind != 'U':
                return None
            labels[key] = values
    return SampleTable(numeric, labels=labels, fields=fields)


def frequency_buffer():
    return SampleChunks() if _COMPACT.get() else []


def merge_tables(channels):
    """Validate coordinates with bounded numeric storage, preserving row order."""
    labeled, coordinates, orders = {}, {}, {}
    for export, internal in (('VV', 'TE'), ('HH', 'TM')):
        table = channels[export]['samples']
        if not len(table):
            raise ValueError('The co-polarized 2-D solve returned no {} samples.'.format(export))
        coords = table.data[:len(table), :3]
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        ordered = coords[order]
        if len(ordered) > 1 and np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
            raise ValueError('Duplicate {} 2-D sample.'.format(export))
        coordinates[export], orders[export] = ordered, order
        labeled[export] = table.with_labels(polarization=export, polarization_internal=internal)
    if not np.array_equal(coordinates['VV'], coordinates['HH']):
        raise ValueError('TE/TM 2-D solves did not return the same physical grid.')
    second_order = None
    if not np.array_equal(orders['VV'], orders['HH']):
        second_order = np.empty(len(orders['VV']), dtype=np.intp)
        second_order[orders['VV']] = orders['HH']
    return labeled, InterleavedSamples(labeled['VV'], labeled['HH'], second_order)
