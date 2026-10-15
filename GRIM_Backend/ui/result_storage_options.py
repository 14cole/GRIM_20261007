"""Storage choices for datasets produced by GUI operations."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)


class ResultStorageDialog(QDialog):
    def __init__(self, options=None, *, review=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Operation result storage')
        self.setMinimumWidth(430)
        self.allow_exceed = False
        options = dict(options or {})
        layout = QVBoxLayout(self)
        if review:
            message = QLabel(str(review))
            message.setWordWrap(True)
            layout.addWidget(message)
        form = QFormLayout()
        self.mode = QComboBox()
        for label, key in (('Auto', 'auto'), ('RAM', 'memory'), ('Disk', 'disk')):
            self.mode.addItem(label, key)
        self.mode.setCurrentIndex(max(0, self.mode.findData(options.get('mode', 'auto'))))
        form.addRow('Store new results in', self.mode)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(1, 1_000_000)
        self.budget.setDecimals(0)
        self.budget.setSuffix(' MiB')
        self.budget.setValue(options.get('ram_budget_bytes', 256 * 1024**2) / 1024**2)
        self.budget.setEnabled(self.mode.currentData() == 'auto')
        self.mode.currentIndexChanged.connect(
            lambda _: self.budget.setEnabled(self.mode.currentData() == 'auto'))
        form.addRow('Auto RAM allocation budget', self.budget)
        self.folder = QLineEdit(str(options.get('cache_dir') or ''))
        self.folder.setPlaceholderText('System temporary folder')
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._choose_folder)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow('Temporary working folder', row)
        layout.addLayout(form)
        note = QLabel(
            'Auto keeps small result allocations in RAM and writes larger ones to disk. '
            'The budget applies to each operation batch. Computation still uses some RAM. '
            'Disk results use temporary files; save the dataset to keep it permanently. '
            'This setting applies to new results and does not move existing datasets.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText('Run' if review else 'Save options')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        if review:
            self.once = self.buttons.addButton('Continue once', QDialogButtonBox.ActionRole)
            self.once.clicked.connect(self._continue_once)
        layout.addWidget(self.buttons)

    def _continue_once(self):
        self.allow_exceed = True
        self.accept()

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Temporary working folder', self.folder.text())
        if path:
            self.folder.setText(path)

    def options(self):
        result = {'mode': self.mode.currentData(),
                  'ram_budget_bytes': int(self.budget.value() * 1024**2)}
        if self.folder.text().strip():
            result['cache_dir'] = self.folder.text().strip()
        if self.allow_exceed:
            result['allow_exceed'] = True
        return result
