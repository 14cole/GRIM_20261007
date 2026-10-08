"""Snapping data markers for the Plotting canvas.

With the Markers toggle on, a left click drops a marker on the nearest data
point of any dataset curve or PBP band edge; dragging a marker slides it along
its own curve, and the arrow keys step the last-used marker one sample at a
time (Shift for ten). Markers follow their curve through re-plots (Auto Plot,
the slider, Hold) by re-snapping to the nearest x on the same dataset.
"""

from __future__ import annotations

import re

import numpy as np
from matplotlib.backend_bases import MouseButton
from PySide6.QtWidgets import QMenu

from GRIM_Backend.plotting.modes import common as plot_common

MARKER_SNAP_PIXELS = 40.0
MARKER_GRAB_PIXELS = 10.0
_UNIT_PATTERN = re.compile(r"\(([^()]*)\)\s*$")


def _axis_unit(label: str) -> str:
    match = _UNIT_PATTERN.search(str(label or ""))
    return match.group(1) if match else ""


class PlotMarkersMixin:
    # --- state and hit testing -------------------------------------------

    def _markers_enabled(self) -> bool:
        return (
            self._button_checked(getattr(self, "btn_markers", None))
            and getattr(self, "_active_plot_tab", "plotting") == "plotting"
        )

    def _plot_marker_list(self) -> list[dict]:
        markers = getattr(self, "_plot_markers", None)
        if markers is None:
            markers = self._plot_markers = []
        return markers

    def _marker_lines(self, key=None) -> list:
        """Visible, non-empty dataset/band lines, optionally for one key."""
        return [
            line
            for ax in self.plot_figure.axes
            for line in ax.lines
            if getattr(line, "_grim_dataset_key", None) is not None
            and (key is None or line._grim_dataset_key == key)
            and line.get_visible()
            and len(line.get_xdata())
        ]

    @staticmethod
    def _line_screen_points(line) -> np.ndarray:
        xy = np.column_stack((
            np.asarray(line.get_xdata(), dtype=float),
            np.asarray(line.get_ydata(), dtype=float),
        ))
        screen = np.full(xy.shape, np.nan)
        finite = np.all(np.isfinite(xy), axis=1)
        if np.any(finite):
            screen[finite] = line.get_transform().transform(xy[finite])
        return screen

    def _nearest_line_point(self, lines, x_px, y_px):
        best = None
        for line in lines:
            screen = self._line_screen_points(line)
            distance = np.hypot(screen[:, 0] - x_px, screen[:, 1] - y_px)
            if not np.any(np.isfinite(distance)):
                continue
            index = int(np.nanargmin(distance))
            if best is None or distance[index] < best[0]:
                best = (float(distance[index]), line, index)
        return best

    def _marker_at_pixel(self, x_px, y_px):
        for marker in reversed(self._plot_marker_list()):
            artist = marker.get("artist")
            if artist is None or artist.axes is None:
                continue
            point = artist.get_transform().transform(
                np.column_stack((artist.get_xdata(), artist.get_ydata()))
            )[0]
            if np.hypot(point[0] - x_px, point[1] - y_px) <= MARKER_GRAB_PIXELS:
                return marker
        return None

    # --- drawing -------------------------------------------------------------

    def _marker_point(self, marker) -> tuple[float, float]:
        line = marker["line"]
        index = marker["index"]
        return float(line.get_xdata()[index]), float(line.get_ydata()[index])

    def _marker_display_x(self, ax, x: float) -> float:
        if getattr(ax, "name", "") != "polar":
            return x
        unit = str(getattr(self, "_polar_display_unit", "deg"))
        degrees = float(np.degrees(x))
        degrees = (degrees + 180.0) % 360.0 - 180.0
        return float(plot_common.convert_axis_values([degrees], "azimuth", "deg", unit)[0])

    def _marker_text(self, marker, first) -> str:
        ax = marker["line"].axes
        x, y = self._marker_point(marker)
        x = self._marker_display_x(ax, x)
        if getattr(ax, "name", "") == "polar":
            x_unit = str(getattr(self, "_polar_display_unit", "deg"))
        else:
            x_unit = _axis_unit(ax.get_xlabel())
        y_unit = _axis_unit(ax.get_ylabel())
        name = self._plot_item_name(marker["key"])
        if len(name) > 40:
            name = name[:39] + "…"
        lines = [
            f"M{marker['number']}  {name}",
            f"x {x:.6g} {x_unit}".rstrip(),
            f"y {y:.6g} {y_unit}".rstrip(),
        ]
        if first is not None and first is not marker and first["line"].axes is ax:
            first_x, first_y = self._marker_point(first)
            first_x = self._marker_display_x(ax, first_x)
            lines.append(f"Δ M{first['number']}: {x - first_x:+.4g}, {y - first_y:+.4g}")
        return "\n".join(lines)

    def _detach_marker(self, marker) -> None:
        for name in ("artist", "text"):
            artist = marker.get(name)
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError, NotImplementedError):
                    pass
            marker[name] = None

    def _draw_marker(self, marker) -> None:
        self._detach_marker(marker)
        line = marker["line"]
        ax = line.axes
        x, y = self._marker_point(marker)
        marker["x"] = x
        (artist,) = ax.plot(
            [x], [y], linestyle="None", marker="D", markersize=7,
            markerfacecolor=line.get_color(), markeredgecolor=self._current_plot_text(),
            markeredgewidth=1.2, zorder=40, label="_nolegend_",
        )
        artist._grim_marker = True
        text = ax.annotate(
            "", xy=(x, y), xytext=(10, 10), textcoords="offset points",
            fontsize=8, color=self._current_plot_text(), zorder=41,
            bbox={
                "boxstyle": "round,pad=0.3", "facecolor": self._current_plot_bg(),
                "edgecolor": line.get_color(), "alpha": 0.92,
            },
        )
        text._grim_marker = True
        marker["artist"], marker["text"] = artist, text

    def _refresh_marker_texts(self) -> None:
        markers = self._plot_marker_list()
        first = markers[0] if markers else None
        for marker in markers:
            text = marker.get("text")
            if text is not None:
                text.set_text(self._marker_text(marker, first))

    def _move_marker(self, marker, index: int) -> None:
        marker["index"] = int(index)
        x, y = self._marker_point(marker)
        marker["x"] = x
        marker["artist"].set_data([x], [y])
        marker["text"].xy = (x, y)
        self._active_marker = marker
        self._refresh_marker_texts()
        self.plot_canvas.draw_idle()

    def _add_plot_marker(self, line, index: int) -> dict:
        markers = self._plot_marker_list()
        same_key = self._marker_lines(line._grim_dataset_key)
        marker = {
            "number": max((m["number"] for m in markers), default=0) + 1,
            "key": line._grim_dataset_key,
            "ordinal": same_key.index(line) if line in same_key else 0,
            "line": line,
            "index": int(index),
        }
        self._draw_marker(marker)
        markers.append(marker)
        self._active_marker = marker
        self._refresh_marker_texts()
        self.plot_canvas.draw_idle()
        return marker

    def _remove_plot_marker(self, marker) -> None:
        self._detach_marker(marker)
        markers = self._plot_marker_list()
        if marker in markers:
            markers.remove(marker)
        if getattr(self, "_active_marker", None) is marker:
            self._active_marker = markers[-1] if markers else None
        self._refresh_marker_texts()
        self.plot_canvas.draw_idle()

    def _clear_plot_markers(self, *, redraw: bool = True) -> None:
        for marker in self._plot_marker_list():
            self._detach_marker(marker)
        self._plot_markers = []
        self._active_marker = None
        self._marker_drag = None
        if redraw:
            self.plot_canvas.draw_idle()

    def _restore_plot_markers(self) -> None:
        """Re-snap markers onto the re-rendered curves of the same dataset."""
        markers = getattr(self, "_plot_markers", None)
        if not markers:
            return
        kept = []
        for marker in markers:
            self._detach_marker(marker)
            lines = self._marker_lines(marker["key"])
            if not lines:
                continue
            line = lines[min(marker["ordinal"], len(lines) - 1)]
            xs = np.asarray(line.get_xdata(), dtype=float)
            ys = np.asarray(line.get_ydata(), dtype=float)
            distance = np.abs(xs - float(marker.get("x", np.nan)))
            if getattr(line.axes, "name", "") == "polar":
                distance = np.abs(np.angle(np.exp(1j * (xs - float(marker.get("x", 0.0))))))
            distance = np.where(np.isfinite(xs) & np.isfinite(ys), distance, np.inf)
            if not np.any(np.isfinite(distance)):
                continue
            marker["line"] = line
            marker["index"] = int(np.argmin(distance))
            self._draw_marker(marker)
            kept.append(marker)
        self._plot_markers = kept
        if getattr(self, "_active_marker", None) not in kept:
            self._active_marker = kept[-1] if kept else None
        self._refresh_marker_texts()

    # --- mouse and keyboard ----------------------------------------------

    def _on_markers_toggled(self, checked: bool) -> None:
        if checked and getattr(self, "spatial_overlays", None) is not None:
            self.spatial_overlays.stop_drawing()
        self._marker_drag = None
        if not checked:
            return
        # Markers, zoom box, and pan all claim the left button - one at a time.
        self._uncheck_silently(getattr(self, "btn_zoom_box", None))
        self._uncheck_silently(getattr(self, "btn_pan", None))
        self._clear_zoom_box_drag()
        self._clear_pan_drag()
        self.status.showMessage(
            "Markers on: click near a curve to drop a marker, drag it along the "
            "curve, and use the arrow keys to step it. Right-click a marker to remove it."
        )

    def _on_marker_press(self, event) -> bool:
        """Place or grab a marker; True when the click was consumed."""
        if not self._markers_enabled() or event.button is not MouseButton.LEFT:
            return False
        if event.canvas is not self.plot_canvas or event.inaxes is None:
            return False
        grabbed = self._marker_at_pixel(event.x, event.y)
        if grabbed is not None:
            self._marker_drag = grabbed
            self._active_marker = grabbed
            return True
        nearest = self._nearest_line_point(self._marker_lines(), event.x, event.y)
        if nearest is None or nearest[0] > MARKER_SNAP_PIXELS:
            self.status.showMessage("No curve near the click; click closer to a curve to drop a marker.")
            return True
        marker = self._add_plot_marker(nearest[1], nearest[2])
        self._marker_drag = marker
        return True

    def _on_marker_motion(self, event) -> bool:
        marker = getattr(self, "_marker_drag", None)
        if marker is None or event.canvas is not self.plot_canvas:
            return False
        if event.x is None or event.y is None or marker["line"].axes is None:
            return True
        nearest = self._nearest_line_point([marker["line"]], event.x, event.y)
        if nearest is not None and nearest[2] != marker["index"]:
            self._move_marker(marker, nearest[2])
        return True

    def _on_marker_release(self, event) -> bool:
        if getattr(self, "_marker_drag", None) is None:
            return False
        self._marker_drag = None
        return True

    def _on_marker_key(self, event) -> bool:
        key = str(getattr(event, "key", "") or "")
        if key not in ("left", "right", "shift+left", "shift+right"):
            return False
        marker = getattr(self, "_active_marker", None)
        if marker is None or marker not in self._plot_marker_list():
            return False
        ys = np.asarray(marker["line"].get_ydata(), dtype=float)
        xs = np.asarray(marker["line"].get_xdata(), dtype=float)
        valid = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
        if valid.size == 0:
            return True
        position = int(np.searchsorted(valid, marker["index"]))
        step = (10 if key.startswith("shift+") else 1) * (1 if key.endswith("right") else -1)
        position = int(np.clip(position + step, 0, valid.size - 1))
        self._move_marker(marker, valid[position])
        return True

    def _marker_context_menu(self, pos) -> bool:
        """Show the marker menu when right-clicking a marker."""
        markers = self._plot_marker_list()
        if not markers or getattr(self, "_active_plot_tab", "plotting") != "plotting":
            return False
        x, y = self.plot_canvas.mouseEventCoords(pos)
        marker = self._marker_at_pixel(x, y)
        if marker is None:
            return False
        menu = QMenu(self)
        menu.addSection(f"Marker M{marker['number']}")
        menu.addAction("Remove marker", lambda: self._remove_plot_marker(marker))
        menu.addAction("Clear all markers", self._clear_plot_markers)
        menu.exec(self.plot_canvas.mapToGlobal(pos))
        return True
