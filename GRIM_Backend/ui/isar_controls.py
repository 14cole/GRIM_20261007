"""ISAR control synchronization without touching the numerical worker."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox,
)


def sync_frequency_controls(context, dataset):
    """Display native frequency values with explicit units and acquired bounds."""
    if context is None or dataset is None:
        return
    values = np.asarray(dataset.frequencies, dtype=float)
    if not values.size or not np.all(np.isfinite(values)):
        return
    unit = str((dataset.units or {}).get("frequency", "")).strip()
    lo, hi = float(values.min()), float(values.max())
    key = (id(dataset), unit, lo, hi)
    if getattr(context, "_isar_frequency_bounds", None) == key:
        return
    context._isar_frequency_bounds = key
    for spin, value in ((context.spin_isar_freq_min, lo), (context.spin_isar_freq_max, hi)):
        old = spin.blockSignals(True)
        try:
            spin.setDecimals(9)
            spin.setRange(lo, hi)
            spin.setSuffix(f" {unit}" if unit else "")
            spin.setSingleStep(max((hi - lo) / 100.0, 1e-9))
            spin.setValue(value)
            spin.setToolTip(f"Frequency limit in the dataset's {unit or 'declared'} units.")
        finally:
            spin.blockSignals(old)


def sync_reconstruction_controls(context):
    sparse = context.combo_isar_recon.currentText().lower().startswith("sparse")
    context.spin_isar_l1_strength.setEnabled(sparse)
    context.spin_isar_l1_iters.setEnabled(sparse)
    context.combo_isar_window.setEnabled(not sparse)
    advanced = getattr(context, 'isar_advanced', None)
    if advanced is not None:
        advanced.native.setEnabled(sparse)


class AdvancedIsarControls(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        for label, value in (("Automatic aperture policy", "auto"), ("Single coherent image", "coherent"),
                             ("Qualitative max-look composite", "composite")):
            self.mode.addItem(label, value)
        self.mode.setToolTip("Automatic retains the 20° composite switch. A coherent image requires <90° and should be checked against the requested scene.")
        layout.addRow("Image mode", self.mode)
        self.scene = QCheckBox("Use scene bounds and crop result")
        layout.addRow(self.scene)
        self.x_half, self.y_half = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.x_half, self.y_half):
            spin.setRange(.000001, 100000.)
            spin.setDecimals(6)
            spin.setSuffix(" m")
            spin.setValue(1.)
            spin.setEnabled(False)
            spin.setToolTip("Occupied scene half extent about the fixed phase origin. Cropping changes retained image size, not the acquired resolution.")
        layout.addRow("Cross-range half extent", self.x_half)
        layout.addRow("Range half extent", self.y_half)
        self.side = QSpinBox()
        self.side.setRange(32, 4096)
        self.side.setSingleStep(256)
        self.side.setValue(1024)
        self.side.setToolTip("Composite grid pixels per side. More pixels do not improve physical resolution.")
        layout.addRow("Composite pixels per side", self.side)
        self.native = QCheckBox("Check sparse image against acquired polar samples")
        self.native.setChecked(True)
        self.native.setToolTip("Bounded direct point prediction. Reports selected-sample count and any omitted image support.")
        layout.addRow(self.native)
        self.memory_auto = QCheckBox("Use half of available RAM")
        self.memory_auto.setChecked(True)
        self.memory_limit = QSpinBox()
        self.memory_limit.setRange(1, 2_000_000_000)
        self.memory_limit.setSuffix(" MiB")
        from GRIM_Backend.isar.memory import resolve_memory_limit
        self.memory_limit.setValue(max(1, resolve_memory_limit(use_environment=False) // 1024**2))
        self.memory_limit.setEnabled(False)
        self.memory_limit.setToolTip("Budget for ISAR formation and retained results. Large Fourier staging arrays can use temporary disk storage. Sparse solving still needs RAM.")
        self.memory_allow = QCheckBox("Allow computation above the memory budget")
        self.memory_allow.setToolTip("Continue when the estimated working set exceeds the selected budget. The operating system may page or reject an allocation.")
        layout.addRow("Memory budget", self.memory_auto)
        layout.addRow("Custom budget", self.memory_limit)
        layout.addRow(self.memory_allow)
        self.memory_auto.toggled.connect(self._memory_changed)
        self.memory_limit.valueChanged.connect(lambda _value: self.changed.emit())
        self.memory_allow.toggled.connect(lambda _checked: self.changed.emit())
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.scene.toggled.connect(self._scene_changed)
        for spin in (self.x_half, self.y_half):
            spin.valueChanged.connect(lambda _=None: self.changed.emit() if self.scene.isChecked() else None)
        self.side.valueChanged.connect(lambda: self.changed.emit() if self.side.isEnabled() else None)
        self.native.toggled.connect(lambda: self.changed.emit() if self.native.isEnabled() else None)

    def _mode_changed(self):
        self.side.setEnabled(self.mode.currentData() != 'coherent')
        self.changed.emit()

    def _scene_changed(self, enabled):
        self.x_half.setEnabled(enabled)
        self.y_half.setEnabled(enabled)
        self.changed.emit()

    def _memory_changed(self, automatic):
        self.memory_limit.setEnabled(not automatic)
        self.changed.emit()

    def options(self):
        return {"aperture_mode": self.mode.currentData(),
                "scene_half_extent_m": (self.x_half.value(), self.y_half.value()) if self.scene.isChecked() else None,
                "composite_side": self.side.value(), "native_diagnostics": self.native.isChecked(),
                "memory_limit_bytes": None if self.memory_auto.isChecked() else self.memory_limit.value() * 1024**2,
                "memory_allow_exceed": self.memory_allow.isChecked()}
