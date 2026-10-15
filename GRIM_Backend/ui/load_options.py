"""Optional load controls and a recoverable memory-budget prompt."""
from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)


def _gib(value):
    return 'unavailable' if value is None else f'{value / 1024**3:.2f} GiB'


class LoadOptionsDialog(QDialog):
    def __init__(self, options=None, *, review=None, available_bytes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load options')
        self.setMinimumWidth(440)
        self._original = dict(options or {})
        self.allow_exceed = False
        layout = QVBoxLayout(self)
        info = QLabel(f'Available RAM: {_gib(available_bytes)}')
        layout.addWidget(info)
        if review:
            plan = review.get('plan', {})
            message = (f"Estimated load: {_gib(plan.get('serial_peak_bytes'))}\n"
                       f"Current budget: {_gib(plan.get('budget_bytes'))}") if plan else review.get('error', '')
            summary = QLabel(message)
            summary.setWordWrap(True)
            layout.addWidget(summary)
            note = QLabel('This estimate exceeded the load budget. You can adjust the budget, '
                          'use disk storage for .grim files, or continue once. Continuing may '
                          'use virtual memory or fail if the computer cannot allocate enough memory.')
            note.setWordWrap(True)
            layout.addWidget(note)
        form = QFormLayout()
        self.budget_mode = QComboBox()
        self.budget_mode.addItem('Automatic — 50% of available RAM', 'auto')
        self.budget_mode.addItem('Custom budget', 'custom')
        form.addRow('Load memory budget', self.budget_mode)
        self.budget_gib = QDoubleSpinBox()
        self.budget_gib.setRange(0.01, 1_000_000)
        self.budget_gib.setDecimals(2)
        self.budget_gib.setSuffix(' GiB')
        explicit = self._original.get('max_output_bytes')
        self.budget_gib.setValue((explicit or (available_bytes or 4 * 1024**3) // 2) / 1024**3)
        self.budget_mode.setCurrentIndex(1 if explicit is not None else 0)
        self.budget_gib.setEnabled(explicit is not None)
        self.budget_mode.currentIndexChanged.connect(lambda i: self.budget_gib.setEnabled(i == 1))
        form.addRow('Custom budget', self.budget_gib)
        self.storage = QComboBox()
        self.storage.addItem('RAM', 'memory')
        self.storage.addItem('Disk — lower RAM use', 'disk')
        self.storage.setCurrentIndex(1 if self._original.get('storage') == 'disk' else 0)
        form.addRow('.grim sample storage', self.storage)
        self.cache_dir = QLineEdit(str(self._original.get('cache_dir') or ''))
        self.cache_dir.setPlaceholderText('System temporary folder')
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._choose_cache_directory)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.cache_dir, 1)
        folder_row.addWidget(browse)
        form.addRow('Temporary working folder', folder_row)
        layout.addLayout(form)
        disk_note = QLabel('Disk storage uses a temporary working copy. Other file types load into RAM. '
                           'Operations that create a new dataset may still need RAM for their result.')
        disk_note.setWordWrap(True)
        layout.addWidget(disk_note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText('Load' if review else 'Save options')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        if review:
            self.once_button = self.buttons.addButton('Continue once', QDialogButtonBox.ActionRole)
            self.once_button.clicked.connect(self._continue_once)
        layout.addWidget(self.buttons)

    def _continue_once(self):
        self.allow_exceed = True
        self.accept()

    def _choose_cache_directory(self):
        path = QFileDialog.getExistingDirectory(self, 'Temporary working folder', self.cache_dir.text())
        if path:
            self.cache_dir.setText(path)

    def options(self):
        result = dict(self._original)
        result.pop('allow_exceed', None)
        result['storage'] = self.storage.currentData()
        folder = self.cache_dir.text().strip()
        if folder:
            result['cache_dir'] = folder
        else:
            result.pop('cache_dir', None)
        if self.budget_mode.currentData() == 'custom':
            result['max_output_bytes'] = int(self.budget_gib.value() * 1024**3)
        else:
            result.pop('max_output_bytes', None)
        if self.allow_exceed:
            result['allow_exceed'] = True
        return result


class NativeRegionDialog(QDialog):
    def __init__(self, report, *, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load a region from .grim')
        self.selection = None
        layout = QVBoxLayout(self)
        note = QLabel('Choose inclusive ranges in the file’s units. Unchecked axes load in full. '
                      'This reads the selected samples without first loading the entire dataset into RAM.')
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        grid.addWidget(QLabel('Axis'), 0, 0)
        grid.addWidget(QLabel('Start'), 0, 1)
        grid.addWidget(QLabel('Stop'), 0, 2)
        self.ranges = {}
        units = report.get('axis_units', {})
        ranges = report.get('axis_ranges', {})
        for row, axis in enumerate(('azimuth', 'elevation', 'frequency'), 1):
            unit = units.get(axis, 'file units')
            enabled = QCheckBox(f'{axis.capitalize()} ({unit})')
            low, high = QLineEdit(), QLineEdit()
            bounds = ranges.get(axis)
            if bounds is not None:
                low.setText(f'{bounds[0]:.17g}')
                high.setText(f'{bounds[1]:.17g}')
            low.setEnabled(False)
            high.setEnabled(False)
            enabled.toggled.connect(low.setEnabled)
            enabled.toggled.connect(high.setEnabled)
            grid.addWidget(enabled, row, 0)
            grid.addWidget(low, row, 1)
            grid.addWidget(high, row, 2)
            self.ranges[axis] = (enabled, low, high)
        layout.addLayout(grid)
        form = QFormLayout()
        self.polarizations = QLineEdit()
        channels = ', '.join(report.get('polarizations', []))
        self.polarizations.setPlaceholderText(f'All channels: {channels}' if channels else 'All channels; or HH, VV, ...')
        form.addRow('Polarization', self.polarizations)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Open).setText('Load selected region')
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_selection(self):
        selected = {}
        try:
            for axis, (enabled, low, high) in self.ranges.items():
                if not enabled.isChecked():
                    continue
                bounds = (float(low.text()), float(high.text()))
                if not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
                    raise ValueError(f'{axis.capitalize()}: enter finite start and stop values, with start ≤ stop.')
                selected[axis] = bounds
            pol = [item.strip() for item in self.polarizations.text().split(',') if item.strip()]
            if pol:
                selected['polarization'] = pol
        except ValueError as exc:
            QMessageBox.information(self, 'Check the selected range', str(exc))
            return
        self.selection = selected or None
        self.accept()
