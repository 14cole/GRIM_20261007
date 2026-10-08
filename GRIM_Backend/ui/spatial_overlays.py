"""Per-canvas editable overlays for plots with distance axes."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from GRIM_Backend.plotting.dataset_style import LINE_TYPES
from GRIM_Backend.plotting.overlay_data import (
    LENGTH_UNITS, MAX_POINTS, PLANES, OverlayPath, axes_scales, axes_signature,
    axis_info, project_points, read_overlay_points, supports_overlays,
)


class SpatialOverlayPanel(QWidget):
    """Coordinates outlive artists, so ordinary re-plots preserve drawings."""

    def __init__(self, owner, canvas):
        super().__init__()
        self.owner, self.canvas = owner, canvas
        self.paths: list[OverlayPath] = []
        self.axes = []
        self.active = self.drawing = self.drag = None
        self._drawing_number = 0
        self.setObjectName("spatialOverlayPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        row = QHBoxLayout()
        self.load_button = QPushButton("Load overlay…")
        self.draw_button = QPushButton("Draw points")
        self.draw_button.setCheckable(True)
        self.panel_combo = QComboBox()
        self.panel_combo.setMinimumContentsLength(8)
        self.panel_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(self.load_button)
        row.addWidget(self.draw_button)
        row.addWidget(QLabel("Panel"))
        row.addWidget(self.panel_combo, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.path_combo = QComboBox()
        self.path_combo.setMinimumContentsLength(10)
        self.path_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.visible_check = QCheckBox("Show")
        self.save_button = QPushButton("Save XY…")
        self.remove_button = QPushButton("Remove")
        row.addWidget(QLabel("Overlay"))
        row.addWidget(self.path_combo, 1)
        for widget in (self.visible_check, self.save_button, self.remove_button):
            row.addWidget(widget)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.color_button = QPushButton("Color…")
        self.style_combo = QComboBox()
        for label, style in LINE_TYPES:
            self.style_combo.addItem(label, style)
        self.style_combo.addItem("Points only", "None")
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(.1, 20)
        self.width_spin.setDecimals(1)
        self.width_spin.setValue(1.8)
        self.width_spin.setSuffix(" pt")
        self.points_check = QCheckBox("Show points")
        self.points_check.setChecked(True)
        for widget in (self.color_button, self.style_combo, self.width_spin, self.points_check):
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row)
        self.help_label = QLabel("Plot an image with a distance axis to load or draw an overlay.")
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)
        self.load_button.clicked.connect(self.load_file)
        self.draw_button.toggled.connect(self._toggle_drawing)
        self.panel_combo.currentIndexChanged.connect(self._select_panel)
        self.path_combo.currentIndexChanged.connect(self._select_path)
        self.remove_button.clicked.connect(self.remove_active)
        self.save_button.clicked.connect(self.save_xy)
        self.color_button.clicked.connect(self.choose_color)
        self.style_combo.currentIndexChanged.connect(self._controls_changed)
        self.width_spin.valueChanged.connect(self._controls_changed)
        self.points_check.toggled.connect(self._controls_changed)
        self.visible_check.toggled.connect(self._controls_changed)
        canvas.mpl_connect("key_press_event", self.on_key)
        self.refresh()
        self.hide()

    def _axis(self, item=None):
        index = item.panel if item is not None else self.panel_combo.currentIndex()
        if 0 <= index < len(self.axes):
            ax = self.axes[index]
            if item is None or axes_signature(ax) == item.signature:
                return ax
        return None

    @staticmethod
    def _detach(item):
        if item.artist is not None:
            try:
                item.artist.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
            item.artist = None

    def _render(self, item):
        self._detach(item)
        ax = self._axis(item)
        if ax is None or not item.visible or not len(item.points):
            return
        xy = item.points / axes_scales(ax)
        item.artist = Line2D(
            xy[:, 0], xy[:, 1], color=item.color, linestyle=item.linestyle,
            linewidth=item.linewidth,
            marker="o" if item.show_points or item.linestyle == "None" else "None",
            markersize=5, zorder=35, label="_spatial_overlay", pickradius=7,
        )
        item.artist._grim_spatial_overlay = True
        # add_artist does not change data limits or the image normalization.
        ax.add_artist(item.artist)

    def refresh(self):
        self.drag = None
        self.axes = [ax for ax in self.canvas.figure.axes if supports_overlays(ax)]
        previous = self.panel_combo.currentIndex()
        self.panel_combo.blockSignals(True)
        self.panel_combo.clear()
        for i, ax in enumerate(self.axes):
            title = ax.get_title().split("|")[0].strip()
            self.panel_combo.addItem(f"{i + 1}: {title or 'Plot'}")
        if self.axes:
            self.panel_combo.setCurrentIndex(max(0, min(previous, len(self.axes) - 1)))
        self.panel_combo.blockSignals(False)
        self.load_button.setEnabled(bool(self.axes))
        self.draw_button.setEnabled(bool(self.axes))
        if not self.axes:
            self.stop_drawing()
        for item in self.paths:
            self._render(item)
        self._refresh_choices()
        self.help_label.setText(
            "Click Draw points, then click to connect points; Escape finishes. "
            "Drag a point to move it; right-click a point or line to edit."
            if self.axes else "Plot an image with a distance axis to load or draw an overlay."
        )
        self.canvas.draw_idle()

    def _refresh_choices(self):
        self.path_combo.blockSignals(True)
        self.path_combo.clear()
        eligible = [item for item in self.paths
                    if item.panel == self.panel_combo.currentIndex() and self._axis(item) is not None]
        for item in eligible:
            self.path_combo.addItem(item.name, item)
        if self.active not in eligible:
            self.active = eligible[-1] if eligible else None
        self.path_combo.setCurrentIndex(eligible.index(self.active) if self.active is not None else -1)
        self.path_combo.blockSignals(False)
        self._sync_controls()

    def _select_panel(self, *_):
        self.stop_drawing()
        self._refresh_choices()

    def _select_path(self, *_):
        self.stop_drawing()
        self.active = self.path_combo.currentData()
        self._sync_controls()

    def _sync_controls(self):
        item = self.active
        for widget in (self.save_button, self.remove_button, self.color_button,
                       self.style_combo, self.width_spin, self.points_check, self.visible_check):
            widget.setEnabled(item is not None)
        if item is None:
            return
        for widget, value in ((self.style_combo, item.linestyle), (self.width_spin, item.linewidth),
                              (self.points_check, item.show_points), (self.visible_check, item.visible)):
            widget.blockSignals(True)
            if widget is self.style_combo:
                widget.setCurrentIndex(widget.findData(value))
            elif widget is self.width_spin:
                widget.setValue(value)
            else:
                widget.setChecked(value)
            widget.blockSignals(False)

    def _controls_changed(self, *_):
        if self.active is not None:
            self.set_style(self.active, linestyle=self.style_combo.currentData(),
                           linewidth=self.width_spin.value(), show_points=self.points_check.isChecked(),
                           visible=self.visible_check.isChecked())

    def set_style(self, item, **style):
        for key, value in style.items():
            setattr(item, key, value)
        self._render(item)
        self._sync_controls()
        self.canvas.draw_idle()

    def choose_color(self):
        if self.active is None:
            return
        color = QColorDialog.getColor(QColor(self.active.color), self, "Overlay color")
        if color.isValid():
            self.set_style(self.active, color=color.name())

    def choose_width(self):
        if self.active is None:
            return
        value, accepted = QInputDialog.getDouble(
            self, "Overlay line width", "Line width (points):", self.active.linewidth, .1, 20, 1
        )
        if accepted:
            self.set_style(self.active, linewidth=value)

    def add_points(self, points, *, name="Overlay", plane="XY", length_unit="m", ax=None):
        ax = ax if ax is not None else self._axis()
        if ax is None or ax not in self.axes:
            raise ValueError("Choose a plot with a distance axis first.")
        xy = project_points(points, plane, ax, length_unit)
        item = OverlayPath(name, xy, axes_signature(ax), self.axes.index(ax))
        self.paths.append(item)
        self.active = item
        self._render(item)
        self._refresh_choices()
        self.canvas.draw_idle()
        return item

    def load_file(self):
        if self._axis() is None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load coordinate overlay", "",
            "Coordinate files (*.xy *.xyz *.csv *.txt *.dat *.pts);;All files (*)",
        )
        if not filename:
            return
        try:
            points = read_overlay_points(filename)
        except (OSError, UnicodeError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot load overlay", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay coordinates")
        form = QFormLayout(dialog)
        plane = QComboBox()
        plane.addItems(list(PLANES) if points.shape[1] == 3 else ["XY"])
        unit = QComboBox()
        unit.addItems(list(LENGTH_UNITS))
        ax = self._axis()
        default_unit = next(axis_info(ax, axis)[1] for axis in ("x", "y")
                            if axis_info(ax, axis)[1] in LENGTH_UNITS)
        unit.setCurrentText(default_unit)
        form.addRow("File columns → plot X, Y", plane)
        form.addRow("File length unit", unit)
        explanation = QLabel(
            "Selected columns map directly to the displayed axes. "
            "Only distance coordinates use the length unit; angle/frequency coordinates "
            "use the units shown on their axis. No viewing-angle rotation is applied."
        )
        explanation.setWordWrap(True)
        form.addRow(explanation)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            self.stop_drawing()
            item = self.add_points(points, name=Path(filename).stem, plane=plane.currentText(),
                                   length_unit=unit.currentText(), ax=ax)
            count = int(np.count_nonzero(np.isfinite(item.points[:, 0])))
            self.owner.status.showMessage(f"Loaded {count:,} overlay points. Drag points or right-click to edit.")

    def _toggle_drawing(self, checked):
        self.drag = self.drawing = None
        self.draw_button.setText("Finish drawing" if checked else "Draw points")
        if checked:
            for name in ("btn_pan", "btn_zoom_box", "btn_markers"):
                self.owner._uncheck_silently(getattr(self.owner, name, None))
            self.owner._clear_pan_drag()
            self.owner._clear_zoom_box_drag()
            self.owner._marker_drag = None
            self.canvas.setFocus(Qt.OtherFocusReason)
            self.owner.status.showMessage("Click points in one plot to draw connected lines. Escape or Finish drawing ends the path.")

    def stop_drawing(self):
        self.drag = self.drawing = None
        self.draw_button.setChecked(False)

    def hideEvent(self, event):
        self.stop_drawing()
        super().hideEvent(event)

    def show_controls(self):
        button = getattr(self, "toggle_button", None)
        if button is not None:
            button.setChecked(True)
        self.show()

    def _hit(self, event, *, include_lines=False):
        if event.inaxes not in self.axes or event.x is None or event.y is None:
            return None
        candidates = ([self.active] if self.active is not None else []) + [
            item for item in reversed(self.paths) if item is not self.active]
        best = None
        for item in candidates:
            line = item.artist
            if line is None or line.axes is not event.inaxes or not line.get_visible():
                continue
            xy = item.points / axes_scales(line.axes)
            valid = np.flatnonzero(np.all(np.isfinite(xy), axis=1))
            pixels = line.axes.transData.transform(xy[valid])
            distance = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
            if distance.size:
                index = int(np.argmin(distance))
                if distance[index] <= 9 and (best is None or distance[index] < best[0]):
                    best = (distance[index], item, int(valid[index]))
        if best is not None:
            return best[1], best[2]
        if include_lines:
            for item in candidates:
                if item.artist is not None and item.artist.axes is event.inaxes and item.artist.contains(event)[0]:
                    return item, None
        return None

    def _activate(self, item):
        self.active = item
        self.panel_combo.blockSignals(True)
        self.panel_combo.setCurrentIndex(item.panel)
        self.panel_combo.blockSignals(False)
        self._refresh_choices()

    def on_press(self, event):
        if event.canvas is not self.canvas or event.button != MouseButton.LEFT:
            return False
        if any(self.owner._button_checked(getattr(self.owner, name, None))
               for name in ("btn_pan", "btn_zoom_box", "btn_markers")):
            return False
        hit = self._hit(event)
        if hit is not None:
            item, index = hit
            self._activate(item)
            self.drag = (item, index, event.inaxes)
            return True
        if not self.draw_button.isChecked() or event.inaxes not in self.axes:
            return False
        if event.xdata is None or event.ydata is None or not np.all(np.isfinite([event.xdata, event.ydata])):
            return True
        if self.drawing is None:
            self._drawing_number += 1
            ax = event.inaxes
            self.drawing = OverlayPath(f"Drawing {self._drawing_number}", np.empty((0, 2)),
                                       axes_signature(ax), self.axes.index(ax))
            self.paths.append(self.drawing)
            self._activate(self.drawing)
        item = self.drawing
        if self._axis(item) is not event.inaxes:
            self.owner.status.showMessage("Finish this drawing before drawing in another panel.")
            return True
        if len(item.points) >= MAX_POINTS:
            self.owner.status.showMessage(f"Drawing limit: {MAX_POINTS:,} points.")
            return True
        point = np.array([event.xdata, event.ydata]) * axes_scales(event.inaxes)
        item.points = np.vstack((item.points, point))
        self._render(item)
        self.canvas.draw_idle()
        return True

    def set_point(self, item, index, x, y):
        ax = self._axis(item)
        if ax is None or not np.all(np.isfinite([x, y])):
            raise ValueError("Both coordinates must be finite numbers on the current plot.")
        item.points[index] = np.array([x, y]) * axes_scales(ax)
        if item.artist is not None:
            xy = item.points / axes_scales(ax)
            item.artist.set_data(xy[:, 0], xy[:, 1])
        self.canvas.draw_idle()

    def on_motion(self, event):
        if self.drag is None or event.canvas is not self.canvas:
            return False
        item, index, ax = self.drag
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            if np.all(np.isfinite([event.xdata, event.ydata])):
                self.set_point(item, index, event.xdata, event.ydata)
        return True

    def on_release(self, event):
        if self.drag is None or event.canvas is not self.canvas:
            return False
        self.drag = None
        return True

    def on_key(self, event):
        if event.key == "escape" and self.draw_button.isChecked():
            self.stop_drawing()

    def edit_point(self, item, index):
        ax = self._axis(item)
        if ax is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit point {index + 1} — {item.name}")
        form = QFormLayout(dialog)
        xy = item.points[index] / axes_scales(ax)
        fields = [QLineEdit(format(float(value), ".17g")) for value in xy]
        for axis, field in zip(("x", "y"), fields):
            name, unit, _ = axis_info(ax, axis)
            form.addRow(f"{axis.upper()}: {name} ({unit})", field)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        def apply():
            try:
                x, y = (float(field.text()) for field in fields)
                self.set_point(item, index, x, y)
            except ValueError as exc:
                QMessageBox.warning(dialog, "Invalid coordinates", str(exc))
                return
            dialog.accept()
        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.exec()

    def remove_point(self, item, index):
        self.drag = None
        item.points = np.delete(item.points, index, axis=0)
        if not np.any(np.isfinite(item.points)):
            self.active = item
            self.remove_active()
        else:
            self._render(item)
            self.canvas.draw_idle()

    def context_menu(self, pos):
        x, y = self.canvas.mouseEventCoords(pos)
        hit = self._hit(MouseEvent("button_press_event", self.canvas, x, y, button=3), include_lines=True)
        if hit is None:
            return False
        self.stop_drawing()
        item, index = hit
        self._activate(item)
        menu = QMenu(self)
        menu.addSection(item.name)
        if index is not None:
            menu.addAction("Edit coordinates…", lambda: self.edit_point(item, index))
            menu.addAction("Remove point", lambda: self.remove_point(item, index))
        style_menu = menu.addMenu("Line type")
        for label, style in (*LINE_TYPES, ("Points only", "None")):
            action = style_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(style == item.linestyle)
            action.triggered.connect(lambda _=False, value=style: self.set_style(item, linestyle=value))
        menu.addAction("Color…", self.choose_color)
        menu.addAction("Line width…", self.choose_width)
        menu.addAction("Overlay controls…", self.show_controls)
        menu.addAction("Save XY…", self.save_xy)
        menu.addSeparator()
        menu.addAction("Remove overlay", self.remove_active)
        menu.exec(self.canvas.mapToGlobal(pos))
        return True

    def remove_active(self):
        self.stop_drawing()
        if self.active is not None:
            self._detach(self.active)
            self.paths.remove(self.active)
            self.active = None
            self._refresh_choices()
            self.canvas.draw_idle()

    def clear(self):
        self.stop_drawing()
        for item in self.paths:
            self._detach(item)
        self.paths.clear()
        self.active = None
        self.refresh()

    def has_visible_paths(self):
        return any(item.artist is not None and item.artist.axes in self.canvas.figure.axes
                   and item.artist.get_visible() for item in self.paths)

    def save_xy(self):
        item = self.active
        ax = self._axis(item) if item is not None else None
        if ax is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save overlay in displayed coordinates", item.name + ".csv", "CSV files (*.csv)")
        if not filename:
            return
        xy = item.points / axes_scales(ax)
        temporary = None
        try:
            target = Path(filename)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", newline="\n",
                    dir=target.parent, prefix="." + target.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(f"# Plot X: {axis_info(ax, 'x')[0]} ({axis_info(ax, 'x')[1]})\n")
                stream.write(f"# Plot Y: {axis_info(ax, 'y')[0]} ({axis_info(ax, 'y')[1]})\nx,y\n")
                for x, y in xy:
                    stream.write(f"{x:.17g},{y:.17g}\n" if np.isfinite(x) else "\n")
            os.replace(temporary, target)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot save overlay", str(exc))
            return
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.owner.status.showMessage("Saved XY overlay in the displayed axis units. Choose those file units when reloading.")
