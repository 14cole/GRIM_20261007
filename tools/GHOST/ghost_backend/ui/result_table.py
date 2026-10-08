"""Lazy result rows and Qt display: keep numeric samples in their owned storage."""
from bisect import bisect_right
from collections.abc import Sequence
import operator
import numpy as np

try:
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
except ImportError:
    from PySide2.QtCore import QAbstractTableModel, QModelIndex, Qt

from ghost_backend.twod.samples import sample_column


class ResultRows(Sequence):
    """Channel-labelled view over original rows; no retained row dictionaries."""
    def __init__(self, result):
        self.blocks, self.ends = [], []
        channels = result.get('co_solved_samples')
        if isinstance(channels, dict):
            for label in ('VV', 'HH'):
                rows = channels.get(label)
                if rows is not None and len(rows):
                    self._add(rows, label, True)
        if not self.blocks:
            rows = result.get('samples')
            if rows is not None and len(rows):
                label = str(result.get('polarization_export') or result.get('polarization') or '')
                self._add(rows, label, False)

    def _add(self, rows, label, override):
        self.ends.append(len(self)+len(rows))
        self.blocks.append((rows, label, override))

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RowSelection(self, np.arange(*index.indices(len(self)), dtype=np.intp))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = bisect_right(self.ends, index)
        source, label, override = self.blocks[block]
        row = dict(source[index-(self.ends[block-1] if block else 0)])
        if override:
            row['polarization'] = label
        elif label:
            row.setdefault('polarization', label)
        return row

    def column(self, key, default=0.0):
        columns = []
        for source, label, override in self.blocks:
            if key == 'polarization' and override:
                values = np.full(len(source), label)
            else:
                values = sample_column(source, key)
                if values is None:
                    fallback = label if key == 'polarization' and label else default
                    dtype, cast = (object, str) if key == 'polarization' else (float, float)
                    values = np.fromiter((cast(row.get(key, fallback)) for row in source), dtype=dtype, count=len(source))
            columns.append(values)
        if not columns:
            return np.empty(0, dtype=object if key == 'polarization' else float)
        return columns[0] if len(columns) == 1 else np.concatenate(columns)


class RowSelection(Sequence):
    def __init__(self, rows, indices):
        indices = np.asarray(indices, dtype=np.intp)
        if isinstance(rows, RowSelection):
            indices, rows = rows.indices[indices], rows.rows
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RowSelection(self.rows, self.indices[index])
        return self.rows[self.indices[index]]

    def column(self, key, default=0.0):
        return self.rows.column(key, default)[self.indices]


def sorted_rows(rows, keys):
    columns = [rows.column(key, '' if key == 'polarization' else 0.0) for key in keys]
    if all(column.dtype.kind not in 'f' or np.all(np.isfinite(column)) for column in columns):
        indices = np.lexsort(tuple(reversed(columns)))
    else:
        # Keep the previous stable Python ordering for nonfinite extension data.
        indices = sorted(range(len(rows)), key=lambda index: tuple(column[index] for column in columns))
    return RowSelection(rows, indices)


def plot_groups(rows, bistatic=False):
    """Group by compact row indices; Matplotlib owns only the plotted arrays."""
    if not len(rows):
        return {}
    frequency = rows.column('frequency_ghz')
    polarization = rows.column('polarization', '')
    incidence = rows.column('theta_inc_deg') if bistatic else np.zeros(len(rows))
    order = np.lexsort((polarization, incidence, frequency))
    changes = ((frequency[order[1:]] != frequency[order[:-1]]) |
               (incidence[order[1:]] != incidence[order[:-1]]) |
               (polarization[order[1:]] != polarization[order[:-1]]))
    groups = {}
    for indices in np.split(order, np.flatnonzero(changes)+1):
        index = indices[0]
        key = float(frequency[index]), float(incidence[index]) if bistatic else None, str(polarization[index])
        groups[key] = RowSelection(rows, indices)
    return groups


class SolverResultsTableModel(QAbstractTableModel):
    """Format visible cells on demand; all result rows remain addressable."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows, self.channels, self.ends = (), (), ()
        self.result, self.kind, self.db_formatter = None, '', None
        self.headers = ['Frequency (GHz)', 'Azimuth (deg)', 'RCS (linear)', 'RCS (dB)']

    def set_result(self, result, kind, db_formatter):
        rows = sorted_rows(ResultRows(result), ('frequency_ghz', 'polarization', 'theta_inc_deg', 'theta_scat_deg'))
        if kind == '2d_bistatic':
            headers = ['Frequency (GHz)', 'Pol', 'Incidence (deg)', 'Observation (deg)', 'Width (m)', 'Width (dBke)']
        elif kind == 'bor':
            headers = ['Frequency (GHz)', 'Pol', 'Aspect (deg)', 'RCS (m^2)', 'RCS (dBsm)']
        else:
            headers = ['Frequency (GHz)', 'Pol', 'Cut angle (deg)', 'Width (m)', 'Width (dBke)']
        self.beginResetModel()
        self.rows, self.channels, self.ends = rows, (), ()
        self.result, self.kind, self.db_formatter = result, kind, db_formatter
        self.headers = headers
        self.endResetModel()

    def set_densities(self, units, channels):
        self.beginResetModel()
        self.result, self.rows, self.kind, self.db_formatter = None, (), 'density', None
        self.channels = channels
        self.ends = np.cumsum([len(channel['real']) for channel in channels])
        self.headers = ['Pol', 'Element', f'X ({units})', f'Y ({units})',
                        'Density real', 'Density imag', 'Magnitude', 'Phase (deg)']
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return int(self.ends[-1]) if len(self.ends) else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.headers[section] if 0 <= section < len(self.headers) else None
        return str(section+1) if 0 <= section < self.rowCount() else None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable if index.isValid() else Qt.NoItemFlags

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or not index.isValid():
            return None
        row_number, column = index.row(), index.column()
        if not 0 <= row_number < self.rowCount() or not 0 <= column < self.columnCount():
            return None
        if self.kind == 'density':
            block = bisect_right(self.ends, row_number)
            channel = self.channels[block]
            offset = row_number-(int(self.ends[block-1]) if block else 0)
            if column == 0:
                return channel['label']
            if column == 1:
                return str(offset+1)
            if column < 4:
                return f"{channel['centers'][offset, column-2]:.8g}"
            key = ('real', 'imag', 'magnitude', 'phase')[column-4]
            value = channel[key][offset]
            if column == 7:
                return f'{value:.6g}' if np.isfinite(value) else 'undefined'
            return f'{value:.8g}'
        row = self.rows[row_number]
        if column == 0:
            return f"{float(row.get('frequency_ghz', 0.0)):.6g}"
        if column == 1:
            return str(row.get('polarization', ''))
        if column == self.columnCount()-1:
            return f'{self.db_formatter(self.result, row):.3f}'
        if column == self.columnCount()-2:
            return f"{float(row.get('rcs_linear', 0.0)):.6e}"
        key = 'theta_inc_deg' if self.kind == '2d_bistatic' and column == 2 else 'theta_scat_deg'
        return f"{float(row.get(key, 0.0)):.6g}"
