"""Inspect retained dataset buffers and reclaim abandoned temporary storage."""
from __future__ import annotations

import gc
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from GRIM_Backend.datasets.storage_registry import cache_inventory, cleanup_abandoned, dataset_storage_bytes


def _size(value):
    return f'{value / 1024**2:,.1f} MiB'


class StorageUsageDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle('Storage usage')
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        note = QLabel('Figures count sample buffers, not total application RAM. Shared buffers are counted once. '
                      'Undo and plotted views can retain data after a table row is deleted. '
                      'Clear Undo releases that restore history; loaded datasets and active plots remain available. '
                      'Cache cleanup removes only marked GRIM folders whose owning process has ended.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.clear_undo = QPushButton('Clear Undo history')
        self.clear_undo.clicked.connect(self._clear_undo)
        row.addWidget(self.clear_undo)
        self.cleanup = QPushButton('Clean abandoned caches')
        self.cleanup.clicked.connect(self._cleanup)
        row.addWidget(self.cleanup)
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.result = QLabel()
        self.result.setWordWrap(True)
        layout.addWidget(self.result)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def _directories(self):
        folders = [self.owner._current_result_storage_options().get('cache_dir')]
        getter = getattr(self.owner, '_current_load_options', None)
        if callable(getter):
            folders.append(getter().get('cache_dir'))
        return tuple(value for value in folders if value)

    def refresh(self):
        loaded = [self.owner.table.item(row, 0).data(Qt.UserRole)
                  for row in range(self.owner.table.rowCount()) if self.owner.table.item(row, 0)]
        deleted = getattr(self.owner, '_last_deleted_dataset_rows', ())
        undone = [items[0].data(Qt.UserRole) for _, items in deleted if items and items[0]]
        seen = set()
        active = dataset_storage_bytes(loaded, seen=seen)
        retained = dataset_storage_bytes(undone, seen=seen)
        caches = cache_inventory(self._directories())
        abandoned = [entry for entry in caches if entry['abandoned']]
        self.summary.setText(
            f'Loaded sample buffers: {_size(active["memory"])} RAM; {_size(active["disk"])} disk.\n'
            f'Additional buffers retained by Undo: {_size(retained["memory"])} RAM; {_size(retained["disk"])} disk.\n'
            f'Tracked temporary files: {_size(sum(entry["bytes"] for entry in caches))}.\n'
            f'Abandoned caches: {len(abandoned)} folders, {_size(sum(entry["bytes"] for entry in abandoned))}.')
        self.clear_undo.setEnabled(bool(deleted))
        self.cleanup.setEnabled(bool(abandoned))

    def _clear_undo(self):
        self.owner._last_deleted_dataset_rows = []
        update = getattr(self.owner, '_update_dataset_action_states', None)
        if callable(update):
            update()
        gc.collect()
        self.result.setText('Undo history cleared. Other live references may still retain sample buffers.')
        self.refresh()

    def _cleanup(self):
        report = cleanup_abandoned(self._directories())
        self.result.setText(f'Removed {report["removed"]} abandoned caches ({_size(report["bytes"])}).'
                            + (' Some folders could not be removed: ' + '; '.join(report['errors']) if report['errors'] else ''))
        self.refresh()
