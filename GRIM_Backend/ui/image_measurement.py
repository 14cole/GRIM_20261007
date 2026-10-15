"""Two-point rulers on images, with separate units for mixed physical axes."""
from __future__ import annotations

import csv
import io

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from GRIM_Backend.plotting.overlay_data import LENGTH_UNITS, axis_info, supports_overlays


def measurement_axes(figure):
    """Only image panels with a distance axis, excluding colorbars and cuts."""
    return [ax for ax in figure.axes if supports_overlays(ax) and
            getattr(ax, "_colorbar", None) is None and
            (ax.images or any(isinstance(item, QuadMesh) for item in ax.collections))]


def distance_result(ax, first, second):
    """Return annotation and copyable rows from *displayed* coordinates.

    Angle versus range is not a Cartesian target plane. Never calculate a
    hypotenuse or inferred cross-range from those unlike quantities. Angular
    differences follow the displayed axis, without implicit circular wrapping.
    """
    points = np.asarray([first, second], dtype=float)
    if points.shape != (2, 2) or not np.all(np.isfinite(points)):
        raise ValueError("Choose two finite image positions.")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = points[1] - points[0]
    if not np.all(np.isfinite(delta)):
        raise ValueError("These coordinates are too large to measure.")
    axes = [axis_info(ax, axis) for axis in ("x", "y")]
    rows, labels = [], []
    for i, (name, unit, _scale) in enumerate(axes):
        name = name or ("X" if i == 0 else "Y")
        rows.extend([(f"A {name}", format(points[0, i], ".12g"), unit),
                     (f"B {name}", format(points[1, i], ".12g"), unit),
                     (f"Signed Δ{name}", format(delta[i], ".12g"), unit),
                     (f"{name} separation", format(abs(delta[i]), ".12g"), unit)])
        labels.append(f"{name}: {abs(delta[i]):.9g} {unit}".rstrip())
    if all(unit in LENGTH_UNITS for _name, unit, _scale in axes):
        with np.errstate(over="ignore", invalid="ignore"):
            length = float(np.hypot(delta[0] * axes[0][2], delta[1] * axes[1][2])) / axes[0][2]
        if not np.isfinite(length):
            raise ValueError("These coordinates are too large to measure.")
        rows.append(("Distance", format(length, ".12g"), axes[0][1]))
        labels.insert(0, f"Distance: {length:.9g} {axes[0][1]}")
    return "\n".join(labels), rows


class ImageMeasurement(QWidget):
    """Compact controls; the owning canvas dispatches clicks before other tools."""

    def __init__(self, owner, canvas):
        super().__init__()
        self.owner, self.canvas = owner, canvas
        self.axes = []
        self.axis = None
        self.points, self.rows, self.artists = [], [], []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(3)
        row = QHBoxLayout()
        self.pick_button = QPushButton("Pick two points")
        self.pick_button.setCheckable(True)
        self.pick_button.toggled.connect(self._toggle_picking)
        self.clear_button = QPushButton("Clear measurement")
        self.clear_button.clicked.connect(self.clear)
        self.copy_button = QPushButton("Copy results")
        self.copy_button.clicked.connect(self.copy_results)
        for button in (self.pick_button, self.clear_button, self.copy_button):
            row.addWidget(button)
        self.readout = QLabel(); self.readout.setWordWrap(True)
        self.readout.setTextFormat(Qt.PlainText); row.addWidget(self.readout, 1)
        layout.addLayout(row)
        self.help = QLabel("Click two positions in the same image. On Azimuth vs Downrange, angle and range are measured separately; this is not target-plane distance. Angular separation follows the displayed axis without wrapping.")
        self.help.setWordWrap(True); layout.addWidget(self.help)
        canvas.mpl_connect("key_press_event", self._on_key)
        self.clear()
        self.hide()

    def open(self):
        self.axes = measurement_axes(self.canvas.figure)
        self.clear()
        self.show()
        self.pick_button.setChecked(True)

    def _toggle_picking(self, checked):
        self.pick_button.setText("Stop picking" if checked else "Pick two points")
        if checked:
            for name in ("btn_pan", "btn_zoom_box", "btn_markers"):
                self.owner._uncheck_silently(getattr(self.owner, name, None))
            self.owner._clear_pan_drag(); self.owner._clear_zoom_box_drag()
            self.owner._marker_drag = None
            overlays = getattr(self.owner, "spatial_overlays", None)
            if overlays is not None:
                overlays.stop_interaction()
            self.canvas.setFocus(Qt.OtherFocusReason)

    def pause(self):
        self.pick_button.setChecked(False)
        if len(self.points) == 1:
            self.clear()

    def deactivate(self):
        self.pause()
        self.clear()
        self.axes = []
        self.hide()

    def clear(self):
        self.axis = None
        self.points, self.rows = [], []
        self._remove_artists()
        self.readout.setText("Choose the first position.")
        self.copy_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.canvas.draw_idle()

    def _remove_artists(self):
        for artist in self.artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self.artists.clear()

    def _on_key(self, event):
        if (not self.isHidden() and self.owner.plot_canvas is self.canvas and
                event.key == "escape"):
            self.pause()

    def on_press(self, event):
        if (self.isHidden() or self.owner.plot_canvas is not self.canvas or
                event.canvas is not self.canvas or not self.pick_button.isChecked() or
                event.button not in (MouseButton.LEFT, 1)):
            return False
        for name in ("btn_pan", "btn_zoom_box", "btn_markers"):
            button = getattr(self.owner, name, None)
            if button is not None and button.isChecked():
                return False
        overlays = getattr(self.owner, "spatial_overlays", None)
        if overlays is not None and any(getattr(overlays, name).isChecked()
                                        for name in ("draw_button", "measure_button")):
            return False
        ax = event.inaxes
        if ax not in self.axes or ax not in measurement_axes(self.canvas.figure):
            return False
        if event.xdata is None or event.ydata is None or not np.all(np.isfinite([event.xdata, event.ydata])):
            return False
        if len(self.points) == 2:
            self.clear()
        if self.points and ax is not self.axis:
            self.readout.setText("Choose the second position in the same image.")
            return True
        self.axis = ax
        self.points.append((float(event.xdata), float(event.ydata)))
        self._remove_artists()
        xy = np.asarray(self.points)
        line = Line2D(xy[:, 0], xy[:, 1], linestyle="--", linewidth=1.5,
                      color="#ff4fa3", marker="o", markersize=7,
                      markerfacecolor="none", zorder=50, label="_image_measurement")
        line._grim_spatial_overlay = True; line.set_in_layout(False)
        ax.add_artist(line); self.artists.append(line)
        self.clear_button.setEnabled(True)
        if len(self.points) == 1:
            coordinates = [f"{axis_info(ax, dim)[0] or dim.upper()} {value:.9g} {axis_info(ax, dim)[1]}".strip()
                           for dim, value in zip(("x", "y"), self.points[0])]
            self.readout.setText("A: " + ", ".join(coordinates) + ". Choose the second position.")
        else:
            try:
                label, self.rows = distance_result(ax, *self.points)
            except ValueError as error:
                self.readout.setText(str(error))
                return True
            text = ax.annotate(label, xy=xy[0] * .5 + xy[1] * .5,
                               xytext=(8, 8), textcoords="offset points", fontsize=9,
                               color=ax.xaxis.label.get_color(), zorder=51,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=ax.get_facecolor(),
                                         edgecolor="#ff4fa3", alpha=.95))
            text._grim_spatial_overlay = True; text.set_in_layout(False)
            self.artists.append(text)
            self.readout.setText(label.replace("\n", "  |  "))
            self.copy_button.setEnabled(True)
        self.canvas.draw_idle()
        return True

    def copy_results(self):
        if not self.rows:
            return
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("Metric", "Value", "Unit")); writer.writerows(self.rows)
        QApplication.clipboard().setText(output.getvalue())
