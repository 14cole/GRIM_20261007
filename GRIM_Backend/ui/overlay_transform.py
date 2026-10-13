"""Floating, reversible whole-overlay transformations."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QLineEdit, QVBoxLayout,
)

from GRIM_Backend.plotting.overlay_data import (
    LENGTH_UNITS, axes_scales, axis_info, format_coordinate, transform_points,
)


class OverlayTransformDialog(QDialog):
    def __init__(self, panel, item):
        super().__init__(panel, Qt.Tool)
        self.panel, self.item = panel, item
        self.original = item.points.copy()
        self.scales = axes_scales(panel._axis(item))
        self.axis_details = [axis_info(panel._axis(item), axis) for axis in ("x", "y")]
        self._automatic_pivot = np.zeros(2)
        self._pivot_display = ["0", "0"]
        self._updating = False
        self.setWindowTitle(f"Edit overlay — {item.name}")
        self.setObjectName("overlayTransformDialog")
        self.setModal(False)
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        summary = QLabel("Preview changes on the plot. Apply keeps them; Cancel discards the current preview.")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        form = QFormLayout()
        layout.addLayout(form)
        self.rotation = QDoubleSpinBox()
        self.rotation.setRange(-360., 360.)
        self.rotation.setDecimals(5)
        self.rotation.setSuffix("°")
        self.rotation.setSingleStep(1.)
        self.rotation.setToolTip("Positive rotation is from +X toward +Y (counterclockwise on standard axes).")
        form.addRow("Rotate", self.rotation)
        can_rotate = all(unit in LENGTH_UNITS for _, unit, _ in self.axis_details)
        self.rotation.setEnabled(can_rotate)
        if not can_rotate:
            note = QLabel("Rotation requires two distance axes. Move and mirror are available on this plot.")
            note.setWordWrap(True)
            form.addRow(note)
        self.translation = [QLineEdit("0"), QLineEdit("0")]
        for axis, field, (name, unit, _) in zip(("X", "Y"), self.translation, self.axis_details):
            form.addRow(f"Move {axis}: {name} ({unit})", field)
        self.mirror_x = QCheckBox("Flip X (left ↔ right)")
        self.mirror_y = QCheckBox("Flip Y (up ↔ down)")
        form.addRow("Mirror", self.mirror_x)
        form.addRow("", self.mirror_y)
        self.pivot_mode = QComboBox()
        self.pivot_mode.addItems(["Overlay center", "Origin (0, 0)", "Custom point"])
        form.addRow("Rotate / mirror around", self.pivot_mode)
        self.pivot_fields = [QLineEdit("0"), QLineEdit("0")]
        for axis, field, (_, unit, _) in zip(("X", "Y"), self.pivot_fields, self.axis_details):
            form.addRow(f"Pivot {axis} ({unit})", field)
        note = QLabel("Order: mirror → rotate → move. Center is the midpoint of the overlay's bounds. "
                      "Use Save XY in Overlays to save the edited coordinates.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply |
                                       QDialogButtonBox.Reset | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        self.buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset_preview)
        self.rotation.valueChanged.connect(self.preview)
        for field in (*self.translation, *self.pivot_fields):
            field.textChanged.connect(self.preview)
        self.mirror_x.toggled.connect(self.preview)
        self.mirror_y.toggled.connect(self.preview)
        self.pivot_mode.currentIndexChanged.connect(self.preview)
        self.preview()

    def _pivot(self):
        if self.pivot_mode.currentIndex() == 0:
            finite = self.original[np.all(np.isfinite(self.original), axis=1)]
            if not len(finite):
                raise ValueError("This overlay has no points to edit.")
            return finite.min(axis=0) / 2 + finite.max(axis=0) / 2
        if self.pivot_mode.currentIndex() == 1:
            return np.zeros(2)
        # Switching to Custom without editing a displayed coordinate keeps
        # its full precision, just like the individual point editor.
        pivot = self._automatic_pivot.copy()
        for column, field in enumerate(self.pivot_fields):
            if field.text().strip() != self._pivot_display[column]:
                pivot[column] = float(field.text()) * self.scales[column]
        return pivot

    def _replace_points(self, points):
        self.item.points = points.copy()
        self.panel._render(self.item)
        self.panel._render_measurement()
        self.panel.canvas.draw_idle()

    def preview(self, *_):
        if self._updating:
            return False
        custom = self.pivot_mode.currentIndex() == 2
        for field in self.pivot_fields:
            field.setEnabled(custom)
        try:
            pivot = self._pivot()
            if not custom:
                self._automatic_pivot = pivot.copy()
                self._pivot_display = [format_coordinate(value) for value in pivot / self.scales]
                for field, value in zip(self.pivot_fields, self._pivot_display):
                    field.blockSignals(True)
                    field.setText(value)
                    field.blockSignals(False)
            offset = np.array([float(field.text()) for field in self.translation]) * self.scales
            points = transform_points(self.original, angle_deg=self.rotation.value(), offset=offset,
                                      pivot=pivot, mirror_x=self.mirror_x.isChecked(),
                                      mirror_y=self.mirror_y.isChecked())
            # Also reject overflow on conversion to the displayed units.
            with np.errstate(over="ignore", invalid="ignore"):
                displayed = points[np.all(np.isfinite(points), axis=1)] / self.scales
            if not np.all(np.isfinite(displayed)):
                raise ValueError("The displayed coordinates are too large. Reduce the values.")
        except ValueError as exc:
            self.error_label.setText("Enter finite numbers for movement and pivot." if "could not convert" in str(exc) else str(exc))
            valid = False
        else:
            self.error_label.clear()
            self._replace_points(points)
            valid = True
        self.error_label.setVisible(not valid)
        for role in (QDialogButtonBox.Ok, QDialogButtonBox.Apply):
            self.buttons.button(role).setEnabled(valid)
        return valid

    def reset_preview(self):
        self._updating = True
        self.rotation.setValue(0.)
        for field in self.translation:
            field.setText("0")
        self.mirror_x.setChecked(False)
        self.mirror_y.setChecked(False)
        self.pivot_mode.setCurrentIndex(0)
        self._updating = False
        self.preview()

    def apply_changes(self):
        if not self.preview():
            return False
        self.original = self.item.points.copy()
        self.reset_preview()
        self.panel.owner.status.showMessage("Overlay changes applied. Use Save XY to save the edited file.")
        return True

    def accept(self):
        if self.apply_changes():
            super().accept()

    def reject(self):
        self._replace_points(self.original)
        super().reject()
