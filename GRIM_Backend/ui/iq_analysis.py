"""Compact, native-sample I/Q exploration on the existing plotting canvas."""
from __future__ import annotations

import json
from itertools import islice
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QSizePolicy, QSlider, QWidget,
)

from GRIM_Backend.datasets.metadata import canonical_time_convention
from GRIM_Backend.plotting.modes.common import NativeSamples, decimate_line
from GRIM_Backend.plotting.iq_tools import (
    compare_iq, compensate_iq, iq_series, repeat_iq,
)


class _NumberSpin(QDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.10g}"


class IQControls(QWidget):
    """Advanced options and a separately placeable, always-visible sample row."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(5)
        self.scale_combo = QComboBox()
        for label, value in (("Native field", "native"),
                             ("Common peak = 1", "common_peak"),
                             ("Phase only: radius = 1", "phase_only")):
            self.scale_combo.addItem(label, value)
        self.scale_combo.setToolTip("Native field retains physical amplitude. Common peak uses one denominator for all curves. Phase only removes amplitude differences.")
        self.linked_checkbox = QCheckBox("Linked magnitude / phase")
        self.guides_checkbox = QCheckBox("Phase / radius guides")
        self.guides_checkbox.setChecked(True)
        grid.addWidget(QLabel("I/Q scaling"), 0, 0)
        grid.addWidget(self.scale_combo, 0, 1)
        grid.addWidget(self.linked_checkbox, 0, 2)
        grid.addWidget(self.guides_checkbox, 0, 3)
        self.phase_offset, self.delay_ns = _NumberSpin(), _NumberSpin()
        self.phase_offset.setRange(-1e12, 1e12); self.phase_offset.setDecimals(8)
        self.phase_offset.setSuffix(" deg")
        self.delay_ns.setRange(-1e12, 1e12); self.delay_ns.setDecimals(9)
        self.delay_ns.setSuffix(" ns")
        self.phase_offset.setKeyboardTracking(False); self.delay_ns.setKeyboardTracking(False)
        self.phase_label, self.delay_label = QLabel("Remove phase offset"), QLabel("Remove delay")
        grid.addWidget(self.phase_label, 1, 0); grid.addWidget(self.phase_offset, 1, 1)
        grid.addWidget(self.delay_label, 1, 2); grid.addWidget(self.delay_ns, 1, 3)
        self.delay_ns.setToolTip("Frequency cuts only. Positive delay removes the phase slope for the selected time convention; reference frequency and the exact correction appear in Results.")
        self.cursor_widget = QWidget()
        cursor = QHBoxLayout(self.cursor_widget)
        cursor.setContentsMargins(0, 0, 0, 0); cursor.setSpacing(5)
        self.coordinate_label = QLabel("Sample")
        self.coordinate = _NumberSpin()
        self.coordinate.setDecimals(12); self.coordinate.setRange(-1e100, 1e100)
        self.coordinate.setKeyboardTracking(False); self.coordinate.setMinimumWidth(105)
        self.coordinate.setToolTip("Type a coordinate, then press Enter to snap to the nearest native sample.")
        self.coordinate_slider = QSlider(Qt.Horizontal)
        self.coordinate_slider.setMaximumWidth(150)
        self.coordinate_slider.setToolTip("Move through native sweep coordinates.")
        self.readout = QLabel()
        self.readout.setTextFormat(Qt.PlainText); self.readout.setWordWrap(True)
        self.readout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        for widget in (self.coordinate_label, self.coordinate, self.coordinate_slider):
            cursor.addWidget(widget)
        cursor.addWidget(self.readout, 1)
        self.scale_combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.linked_checkbox.toggled.connect(lambda *_: self.changed.emit())
        self.guides_checkbox.toggled.connect(lambda *_: self.changed.emit())
        self.phase_offset.editingFinished.connect(self.changed.emit)
        self.delay_ns.editingFinished.connect(self.changed.emit)
        self.configure("iq")

    def configure(self, mode):
        for widget in (self.phase_label, self.phase_offset, self.delay_label, self.delay_ns):
            widget.setVisible(mode == "iq_compensated")
        self.linked_checkbox.setEnabled(mode != "iq_repeat")


def _format(value):
    if np.isnan(value): return "undefined"
    if np.isneginf(value): return "−inf"
    if np.isposinf(value): return "+inf"
    return f"{value:.7g}"


def _scale_label(scaling, unit):
    return unit if scaling == "native" else ("fraction of common peak" if scaling == "common_peak" else "unit amplitude")


def _overview_indices(run, field, budget):
    """Bound a parametric path while keeping both component extrema in order."""
    budget = max(6, int(budget))
    if len(run) <= budget: return run
    edges = np.linspace(0, len(run), max(1, (budget - 2) // 4) + 1, dtype=int)
    selected = [int(run[0]), int(run[-1])]
    for left, right in zip(edges[:-1], edges[1:]):
        bucket = run[left:right]
        if not len(bucket): continue
        for component in (field.real, field.imag):
            selected.extend((int(bucket[np.argmin(component[bucket])]), int(bucket[np.argmax(component[bucket])])))
    # Native run indices increase even when sweep coordinates decrease.
    return np.asarray(sorted(set(selected)), dtype=int)


def _overview_runs(runs, field, budget=20_000):
    """Share one overview budget across gaps, retaining each shown run's ends."""
    if not runs: return []
    total = sum(len(run) for run in runs)
    if total <= budget: return runs
    # At most two endpoints are added for each chosen candidate's native run.
    # This bounds highly fragmented cuts without ever connecting separate runs.
    candidates = _overview_indices(np.concatenate(runs), field, max(6, budget // 3))
    result = []
    for run in runs:
        first, last = np.searchsorted(candidates, [run[0], run[-1]], side="left")
        if last < len(candidates) and candidates[last] == run[-1]: last += 1
        selected = candidates[first:last]
        if len(selected): result.append(np.unique(np.r_[run[0], selected, run[-1]]))
    return result


class _SampleRows:
    """Repeatable lazy export: opening I/Q does not allocate a tuple per sample."""
    def __init__(self, controller): self.controller = controller
    def __len__(self): return sum(len(item["x"]) for item in self.controller.series)
    def __iter__(self): return self.controller._iter_rows()
    def __getitem__(self, index):
        if isinstance(index, slice): return list(islice(iter(self), index.start or 0, index.stop, index.step or 1))
        if index < 0: index += len(self)
        try: return next(islice(iter(self), index, index + 1))
        except StopIteration: raise IndexError(index) from None


class IQController:
    """Own linked cursors and their callbacks; computations remain in iq_tools."""
    headers = ("Series", "Native index", "Coordinate", "Coordinate unit", "I (native)",
               "Q (native)", "Amplitude (native)", "Field unit", "Log level", "Log unit",
               "Phase (deg)", "Phase status", "I (display)", "Q (display)", "Display scaling",
               "Polarization", "Fixed physical coordinates")

    def __init__(self, panel, cut, mode):
        self.panel, self.owner = panel, panel.owner
        self.figure, self.canvas, self.table, self.note = panel._panels["phase"]
        self.controls, self.cut, self.mode = panel.iq_controls, cut, mode
        self.scaling = str(self.controls.scale_combo.currentData())
        self.disconnected, self.cid = False, None
        self.series, self.rows, self.result_rows, self.axes = [], [], [], []
        self.sample_tuples = []
        self._cursor_artists, self._linked_cursors = [], []
        self._notes, self._repeat_cuts = [], None
        self._kwargs = dict(scaling=self.scaling, floor_db=panel.floor.value(),
                            maximum_gap=panel._maximum_gap(), region=panel._region())
        self._prepare()
        self._build()
        self._connect()

    def _prepare(self):
        self.result_rows = [("Source A", self.cut.label, f"{self.cut.axis} ({self.cut.x_unit})"),
                            ("Display scaling", self.scaling, "Copied I/Q and amplitude retain native units"),
                            ("Phase mask", f"{self.panel.floor.value():g}", "dB below each native cut peak; weak I/Q samples remain visible")]
        if self.mode == "iq":
            signature = (self.cut.axis, self.cut.x_unit, self.cut.level_unit, self.scaling)
            held = getattr(self.figure, "_grim_iq_held", None)
            cuts = list(held[1]) if (getattr(self.panel, "_iq_hold", False) and held and held[0] == signature) else []
            def same_cut(other):
                return (other.label == self.cut.label and np.array_equal(other.x, self.cut.x, equal_nan=True)
                        and np.array_equal(other.field, self.cut.field, equal_nan=True))
            cuts = [other for other in cuts if not same_cut(other)]
            cuts.append(self.cut)
            if len(cuts) > 16:
                cuts = cuts[-16:]
                self._notes.append("Hold displays the 16 most recently added distinct cuts.")
            self.figure._grim_iq_held = (signature, cuts)
            def selected_peak(source):
                valid = np.isfinite(source.field) & np.isfinite(source.x)
                if self.panel._region() is not None:
                    low, high = sorted(self.panel._region())
                    valid &= (source.x >= low) & (source.x <= high)
                return float(np.max(np.abs(source.field[valid]))) if np.any(valid) else 0.
            peak = max(selected_peak(source) for source in cuts)
            self.series = [iq_series(c, common_peak=peak, **self._kwargs) for c in cuts]
            self._primary = self.series[-1]
            self.summary = "Color follows the sweep. Circles start segments; squares end them. Click a point or enter a coordinate."
        elif self.mode in {"iq_compare", "iq_difference"}:
            other = self.panel._cut(reference=True)
            report = compare_iq(self.cut, other, **self._kwargs)
            self.series = [report["left"], report["right"]] if self.mode == "iq_compare" else [report["difference"]]
            self._primary = self.series[0]
            self.result_rows.extend((("Reference B", other.label, "Shared physical coordinates; no interpolation"),
                                     ("Common samples", str(report["common_count"]), "Native coordinate matches")))
            self._notes.extend(report.get("notes", []))
            self._notes.append("A shared physical phase reference is needed to interpret rotation or a complex difference. Metadata notes do not block the comparison.")
            self.summary = "A is solid; B is dashed. The same color marks the same physical coordinate." if self.mode == "iq_compare" else "Complex A − B: agreement is near the origin. This is a field difference, not subtraction of dB values."
        elif self.mode == "iq_compensated":
            sign = self.panel.phase_sign.currentData()
            declared = canonical_time_convention(self.cut.time_convention)
            if sign == "auto": sign = "-jwt" if declared == "-jwt" else "+jwt"
            report = compensate_iq(self.cut, phase_offset_deg=self.controls.phase_offset.value(),
                                   delay_ns=self.controls.delay_ns.value() if self.cut.axis == "frequency" else 0.,
                                   time_sign=sign, **self._kwargs)
            self.series, self._primary = [report["original"], report["compensated"]], report["original"]
            self.controls.delay_ns.setEnabled(self.cut.axis == "frequency")
            self.result_rows.extend((("Removed phase offset", f"{report['phase_offset_deg']:.9g}", "deg at reference frequency"),
                                     ("Removed delay", f"{report['delay_ns']:.9g}", "ns; frequency cuts only"),
                                     ("Reference frequency", str(report["reference_frequency"]), self.cut.x_unit if self.cut.axis == "frequency" else "Not applicable"),
                                     ("Time convention", f"exp({sign})", "User selection / declared convention; +jwt default when unspecified")))
            direction = "+" if sign == "+jwt" else "−"
            self._notes.append(f"Correction: z_corrected = z_original × exp(−j phase_offset {direction} j 2π(f−f_ref) delay). Frequency is in Hz and delay in seconds. Original samples are unchanged.")
            self.summary = "Original is solid; corrected is dashed. Removing a stated offset or delay can reveal the remaining structure."
        elif self.mode == "iq_repeat":
            self._repeat_cuts, notes = self.panel._iq_repeat_cuts()
            self._notes.extend(notes)
            self._primary = iq_series(self.cut, **self._kwargs)
            self.summary = "Each point is one selected run at the same physical coordinate. Cross = complex mean; spread describes repeatability."
        else:
            raise ValueError(f"Unknown I/Q view: {self.mode}")
        if self._repeat_cuts is None:
            for item in self.series:
                self._notes.extend(item.get("notes", []))
                self.result_rows.append(("Native I/Q samples", str(np.count_nonzero(item["valid"])), item["label"]))
            if self.scaling == "common_peak":
                self.result_rows.append(("Common amplitude divisor", str(self.series[0]["scale"]), self.series[0]["field_unit"]))
        if self.scaling == "phase_only":
            self.summary = ("Direction of A − B only: unit radius cannot show the size of the mismatch." if self.mode == "iq_difference"
                            else "Phase-only view: each nonzero return has unit radius. Amplitude differences are removed; readouts retain native values.")
        self._indices = np.flatnonzero(np.isfinite(self._primary["x"]))
        if self.panel._region() is not None:
            lo, hi = sorted(self.panel._region())
            self._indices = self._indices[(self._primary["x"][self._indices] >= lo) & (self._primary["x"][self._indices] <= hi)]
        if not len(self._indices): raise ValueError("The selected cut contains no finite coordinates.")

    def _build(self):
        self.figure.clear()
        linked = self.controls.linked_checkbox.isChecked() and self.mode != "iq_repeat"
        self.figure._grim_independent_x = linked
        if linked:
            grid = self.figure.add_gridspec(2, 2, width_ratios=(1.35, 1))
            self.iq_axes = self.figure.add_subplot(grid[:, 0])
            magnitude = self.figure.add_subplot(grid[0, 1])
            phase = self.figure.add_subplot(grid[1, 1], sharex=magnitude)
            self.axes = [self.iq_axes, magnitude, phase]
        else:
            self.iq_axes = self.figure.add_subplot(111)
            self.axes = [self.iq_axes]
        getter = getattr(self.owner, "_effective_colormap", None)
        self.cmap = colormaps.get_cmap(getter() if callable(getter) else "turbo")
        x = np.concatenate([np.asarray(item["x"])[np.isfinite(item["x"])] for item in self.series]) if self.series else self._primary["x"][self._indices]
        lo, hi = float(np.min(x)), float(np.max(x))
        self.norm = Normalize(lo, hi if hi > lo else lo + max(1., abs(lo)) * 1e-9)
        if self.mode != "iq_repeat":
            for index, item in enumerate(self.series):
                self._draw_series(item, index)
            self._finish_iq_axes()
            mappable = ScalarMappable(norm=self.norm, cmap=self.cmap)
            mappable._grim_colormap = True
            colorbar = self.figure.colorbar(mappable, ax=self.iq_axes, pad=.03, fraction=.04)
            colorbar.set_label(f"{self.cut.axis.capitalize()} ({self.cut.x_unit})")
            self.colorbar = colorbar
            if hasattr(self.owner, "plot_colorbars"): self.owner.plot_colorbars.append(colorbar)
            self._update_rows()
        else:
            self.colorbar = None
        self._set_control_range()
        self.select_coordinate(self.controls.coordinate.value())

    def _draw_series(self, item, index):
        ax, field, x = self.iq_axes, item["field"], item["x"]
        style = "solid" if index % 2 == 0 else "dashed"
        marker = "o" if index % 2 == 0 else "^"
        # An invisible full trace retains valid bounds for the app's relim/autoscale.
        bounded = field[item["valid"]]
        if len(bounded):
            proxy, = ax.plot([bounded.real.min(), bounded.real.max()], [bounded.imag.min(), bounded.imag.max()],
                            alpha=0, label="_iq_bounds")
            native = np.where(item["valid"], field, complex(np.nan, np.nan))
            proxy._grim_native_samples = NativeSamples.curve(native.real, native.imag)
            proxy._grim_display_reduced = True
            self._register_marker_line(proxy, item, "iq")
        arrows = 0
        display_points, display_coords, start_points, end_points, segments, segment_coords = [], [], [], [], [], []
        display_runs = _overview_runs(item["runs"], field)
        if sum(len(run) for run in display_runs) < np.count_nonzero(item["valid"]):
            self._notes.append("The trajectory overview is reduced while preserving component extrema and gaps. Cursor lookup and exported values use all native samples.")
        for number, run in enumerate(display_runs):
            values, coords = field[run], x[run]
            points = np.column_stack((values.real, values.imag))
            display_points.append(points); display_coords.append(coords)
            start_points.append(points[0]); end_points.append(points[-1])
            if len(run) > 1:
                segments.append(np.stack((points[:-1], points[1:]), axis=1))
                segment_coords.append((coords[:-1] + coords[1:]) / 2)
                for j in np.unique(np.linspace(0, len(run) - 2, min(3, len(run) - 1)).astype(int)):
                    if arrows >= 8: break
                    if np.array_equal(points[j], points[j + 1]): continue
                    ax.annotate("", xy=points[j + 1], xytext=points[j],
                                arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 1.1},
                                annotation_clip=True)
                    arrows += 1
        if segments:
            collection = LineCollection(np.concatenate(segments), cmap=self.cmap, norm=self.norm,
                                        linewidths=1.5, linestyles=style)
            collection.set_array(np.concatenate(segment_coords)); collection._grim_colormap = True
            ax.add_collection(collection)
        if display_points:
            points, coords = np.concatenate(display_points), np.concatenate(display_coords)
            ax.scatter(points[:, 0], points[:, 1], c=coords, cmap=self.cmap, norm=self.norm,
                       s=10, marker=marker, alpha=.8, zorder=3)
            starts, ends = np.asarray(start_points), np.asarray(end_points)
            ax.plot(starts[:, 0], starts[:, 1], marker, markersize=6, color="0.55", linestyle="none", label="_segment_start")
            ax.plot(ends[:, 0], ends[:, 1], "s", markersize=6, markerfacecolor="none",
                    markeredgecolor="0.55", linestyle="none", label="_segment_end")
        # Legend describes series without encoding a misleading fixed sweep color.
        label = self._series_label(index, item)
        legend, = ax.plot([], [], linestyle=style, marker=marker, color="0.55", label=label)
        self._register_marker_line(legend, item, "legend")
        cursor, = ax.plot([], [], "o", markersize=10, markerfacecolor="none",
                          markeredgecolor="#f4a261", markeredgewidth=2, zorder=10, label="_iq_cursor")
        self._cursor_artists.append(cursor)
        if len(self.axes) == 3:
            top, bottom = self.axes[1:]
            for target, values, validity in ((top, item["db"], item["valid"]),
                                             (bottom, item["phase_deg"], item["phase_valid"])):
                yy = np.where(validity & item["valid"] & np.isfinite(values), values, np.nan)
                # Source-index breaks survive selected sweeps and A/B matching;
                # finite coordinates alone do not establish native adjacency.
                breaks = [int(run[0]) for run in item["runs"] if len(run) and run[0] > 0]
                if target is bottom:
                    breaks.extend((np.flatnonzero(np.abs(np.diff(yy)) > 180) + 1).tolist())
                breaks = sorted(set(breaks))
                xx, yy = np.insert(x, breaks, np.nan), np.insert(yy, breaks, np.nan)
                dx, dy, reduced = decimate_line(xx, yy, max_points=20_000)
                legend_label = f"{label} ({item['level_unit']})" if target is top else label
                line, = target.plot(dx, dy, linestyle=style, linewidth=1.1,
                                    marker="." if len(item["runs"]) > np.count_nonzero(validity) / 2 else None,
                                    markersize=3, label=legend_label, color=f"C{index % 10}")
                if reduced:
                    line._grim_native_samples = NativeSamples.curve(xx, yy)
                    line._grim_display_reduced = True
                self._register_marker_line(line, item, "magnitude" if target is top else "phase")
            units = {entry["level_unit"] for entry in self.series}
            top.set(ylabel=self.cut.level_unit if len(units) == 1 else "Native log level (units in legend)", title="Native return strength")
            if len(units) > 1: top.legend(fontsize="small")
            bottom.set(xlabel=f"{self.cut.axis.capitalize()} ({self.cut.x_unit})", ylabel="Phase (deg)")
            for target in (top, bottom): target.grid(True, alpha=.2)
            for target in (top, bottom):
                line = target.axvline(x[0], color="#f4a261", alpha=.7, linewidth=.9, label="_iq_linked_cursor")
                self._linked_cursors.append(line)

    def _register_marker_line(self, line, item, component):
        """Retain the existing Markers tool for collection-based trajectories."""
        dataset = next((dataset for name, dataset in getattr(self.panel, "_datasets", [])
                        if item["label"].startswith(name + " |")), None)
        register = getattr(self.owner, "_register_dataset_line", None)
        if dataset is not None and callable(register): register(line, dataset)
        else: line._grim_dataset_key = ("iq", item["label"])
        line._grim_trace_key = ("iq", self.mode, item["label"], component, self.scaling,
                                json.dumps(item.get("fixed_coordinates", {}), sort_keys=True))

    def _series_label(self, index, item):
        if self.mode == "iq_compare": return ("A: " if index == 0 else "B: ") + item["label"]
        if self.mode == "iq_compensated": return "Original" if index == 0 else "Corrected"
        if self.mode == "iq_difference": return "A − B"
        return item["label"]

    def _finish_iq_axes(self):
        ax = self.iq_axes
        unit = _scale_label(self.scaling, self._primary["field_unit"])
        ax.set(xlabel=f"I: real field ({unit})", ylabel=f"Q: imaginary field ({unit})",
               title={"iq": "I/Q trajectory", "iq_compare": "I/Q comparison A / B",
                      "iq_difference": "Complex field difference A − B",
                      "iq_compensated": "Original / compensated I/Q", "iq_repeat": "Repeated-measurement I/Q"}[self.mode])
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.2)
        if self.mode == "iq_repeat":
            data = np.asarray(self.repeat_report["field"])
        else:
            data = np.concatenate([np.asarray(item["field"])[item["valid"]] for item in self.series])
        finite = data[np.isfinite(data)]
        if len(finite):
            ax._grim_data_limits = (float(finite.real.min()), float(finite.real.max()),
                                    float(finite.imag.min()), float(finite.imag.max()))
        elif hasattr(ax, "_grim_data_limits"):
            del ax._grim_data_limits
        ax.relim(); ax.autoscale_view()
        if self.controls.guides_checkbox.isChecked():
            radius = max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
            text_color = getattr(self.owner, "_current_plot_text", None)
            text_color = text_color() if callable(text_color) else "0.55"
            for fraction in (.5, 1.):
                ax.add_patch(Circle((0., 0.), radius * fraction, fill=False,
                                    edgecolor="0.55", alpha=.22, linestyle=":", linewidth=.7))
            ax.axhline(0, color="0.5", alpha=.35, linewidth=.7, label="_phase_guide")
            ax.axvline(0, color="0.5", alpha=.35, linewidth=.7, label="_phase_guide")
            for label, position in (("0°", (radius, 0)), ("+90°", (0, radius)),
                                    ("±180°", (-radius, 0)), ("−90°", (0, -radius))):
                annotation = ax.annotate(label, position, xytext=(3, 3), textcoords="offset points",
                                         color=text_color, alpha=.85, fontsize=8)
                annotation.set_label("_iq_guide")
        if self.mode != "iq" or len(self.series) > 1: ax.legend(loc="best", fontsize="small")

    def _set_control_range(self):
        controls = self.controls
        values = self._primary["x"][self._indices]
        controls.coordinate_label.setText(self.cut.axis.capitalize())
        for widget in (controls.coordinate, controls.coordinate_slider): widget.blockSignals(True)
        try:
            controls.coordinate.setRange(float(values.min()), float(values.max()))
            controls.coordinate.setSuffix(f" {self.cut.x_unit}")
            steps = np.abs(np.diff(values)); steps = steps[steps > 0]
            controls.coordinate.setSingleStep(float(np.median(steps)) if len(steps) else 1.)
            controls.coordinate_slider.setRange(0, len(values) - 1)
        finally:
            for widget in (controls.coordinate, controls.coordinate_slider): widget.blockSignals(False)

    def _connect(self):
        self.cid = self.canvas.mpl_connect("button_press_event", self._click)
        self.controls.coordinate.editingFinished.connect(self._typed_coordinate)
        self.controls.coordinate_slider.valueChanged.connect(self._slider_coordinate)

    def disconnect(self):
        if self.disconnected: return
        self.disconnected = True
        if self.cid is not None: self.canvas.mpl_disconnect(self.cid)
        for signal, slot in ((self.controls.coordinate.editingFinished, self._typed_coordinate),
                             (self.controls.coordinate_slider.valueChanged, self._slider_coordinate)):
            try: signal.disconnect(slot)
            except (RuntimeError, TypeError): pass

    def _typed_coordinate(self):
        if not self.disconnected: self.select_coordinate(self.controls.coordinate.value())

    def _slider_coordinate(self, index):
        if not self.disconnected: self.select_coordinate(self._primary["x"][self._indices[index]])

    def _inactive(self):
        if self.disconnected or self.panel.isHidden() or self.owner.plot_figure is not self.figure: return True
        for name in ("btn_zoom_box", "btn_pan", "btn_markers"):
            button = getattr(self.owner, name, None)
            if button is not None and button.isChecked(): return True
        overlay = getattr(self.owner, "spatial_overlays", None)
        if overlay is not None:
            if getattr(overlay, "drag", None) is not None: return True
            for name in ("draw_button", "measure_button"):
                button = getattr(overlay, name, None)
                if button is not None and button.isChecked(): return True
        return False

    def _click(self, event):
        if self._inactive() or event.button != 1 or event.inaxes not in self.axes: return
        if event.inaxes is not self.iq_axes:
            if event.xdata is not None: self.select_coordinate(event.xdata)
            return
        if self.mode == "iq_repeat": return
        best = None
        for item in self.series:
            valid = np.flatnonzero(item["valid"])
            if not len(valid): continue
            for start in range(0, len(valid), 65_536):
                indices = valid[start:start + 65_536]
                fields = item["field"][indices]
                positions = self.iq_axes.transData.transform(np.column_stack((fields.real, fields.imag)))
                distance = np.sum((positions - np.array([event.x, event.y])) ** 2, axis=1)
                selected = int(np.argmin(distance))
                candidate = (float(distance[selected]), float(item["x"][indices[selected]]))
                if best is None or candidate[0] < best[0]: best = candidate
        if best is not None: self.select_coordinate(best[1])

    def select_coordinate(self, coordinate):
        values = self._primary["x"][self._indices]
        offset = int(np.argmin(np.abs(values - coordinate)))
        self.selected_index = int(self._indices[offset]); coordinate = float(values[offset])
        self.selected_coordinate = coordinate
        for widget in (self.controls.coordinate, self.controls.coordinate_slider): widget.blockSignals(True)
        try:
            self.controls.coordinate.setValue(coordinate); self.controls.coordinate_slider.setValue(offset)
        finally:
            for widget in (self.controls.coordinate, self.controls.coordinate_slider): widget.blockSignals(False)
        if self.mode == "iq_repeat":
            self._draw_repeat(coordinate)
        else:
            readouts, cursor_rows = [], []
            for index, (item, marker) in enumerate(zip(self.series, self._cursor_artists)):
                finite = np.flatnonzero(np.isfinite(item["x"]))
                nearest = int(finite[np.argmin(np.abs(item["x"][finite] - coordinate))])
                # Held cuts can use another sampling grid; don't present nearest as matched.
                tolerance = np.finfo(float).eps * max(abs(coordinate), abs(float(item["x"][nearest])), np.finfo(float).tiny) * 32
                if abs(float(item["x"][nearest]) - coordinate) > tolerance:
                    marker.set_data([], []); readouts.append(f"{self._series_label(index, item)}: no sample at this coordinate")
                    continue
                displayed, field = item["field"][nearest], item["native_field"][nearest]
                marker.set_data([displayed.real], [displayed.imag]) if item["valid"][nearest] else marker.set_data([], [])
                phase = self._phase_status(item, nearest)
                prefix = "" if len(self.series) == 1 else f"{index + 1}: "
                text = f"{prefix}I {_format(field.real)} · Q {_format(field.imag)} · |field| {_format(abs(field))} {item['field_unit']} · {_format(item['db'][nearest])} {item['level_unit']} · phase {phase}"
                readouts.append(text)
                cursor_rows.append(("Selected sample", text, f"{coordinate:.12g} {self.cut.x_unit}; {self._series_label(index, item)}"))
            for line in self._linked_cursors: line.set_xdata([coordinate, coordinate])
            visible = readouts[-2:]
            if len(readouts) > 2: visible.append(f"{len(readouts) - 2} more held cuts in Results.")
            self.controls.readout.setText("\n".join(visible))
            self.controls.readout.setToolTip("\n".join(readouts))
            self.table.populate(self.result_rows + cursor_rows)
        self.note.setText(" ".join(dict.fromkeys(self._notes)))
        self.canvas.draw_idle()

    def _phase_status(self, item, index):
        field = item["native_field"][index]
        if not np.isfinite(field): return "undefined (missing field)"
        if abs(field) == 0: return "undefined (zero return)"
        if not item["phase_valid"][index]: return "weak / masked"
        return _format(item["phase_deg"][index]) + "°"

    def _update_rows(self):
        self.rows, self.sample_tuples = _SampleRows(self), []
        for index, item in enumerate(self.series):
            label = self._series_label(index, item)
            self.sample_tuples.extend(((label + " I", item["x"], item["native_field"].real, item["field_unit"]),
                                       (label + " Q", item["x"], item["native_field"].imag, item["field_unit"])))

    def _iter_rows(self):
        for index, item in enumerate(self.series):
            label = self._series_label(index, item)
            coordinates = json.dumps(item.get("fixed_coordinates", getattr(self.cut, "fixed_coordinates", {})))
            polarization = item.get("polarization", getattr(self.cut, "polarization", ""))
            for i in range(len(item["x"])):
                native, shown = item["native_field"][i], item["field"][i]
                yield (label, int(item["native_indices"][i]), item["x"][i], item["x_unit"],
                                  native.real, native.imag, abs(native), item["field_unit"], item["db"][i], item["level_unit"],
                                  item["phase_deg"][i], self._phase_status(item, i), shown.real, shown.imag, self.scaling,
                                  polarization, coordinates)

    def _draw_repeat(self, coordinate):
        already_drawn = getattr(self, "_repeat_initialized", False)
        previous_limits = (self.iq_axes.get_xlim(), self.iq_axes.get_ylim())
        auto_scale = getattr(self.owner, "btn_auto_scale", None)
        preserve_limits = already_drawn and auto_scale is not None and not auto_scale.isChecked()
        report = repeat_iq(self._repeat_cuts, coordinate, scaling=self.scaling,
                           floor_db=self.panel.floor.value())
        self.repeat_report = report
        ax = self.iq_axes; ax.clear()
        getter = getattr(self.owner, "_effective_colormap", None)
        if callable(getter): self.cmap = colormaps.get_cmap(getter())
        fields = np.asarray(report["field"])
        valid = np.isfinite(fields)
        for i in np.flatnonzero(valid):
            color = self.cmap(i / max(1, len(fields) - 1))
            ax.plot(fields[i].real, fields[i].imag, "o", color=color, markersize=7,
                    label=report["labels"][i] if len(fields) <= 8 else "_run")
        mean = report["display_mean_field"]
        ax.plot(np.real(mean), np.imag(mean), "+", color="#f4a261", markersize=14,
                markeredgewidth=2, label="Complex mean")
        self._finish_iq_axes()
        if len(fields) <= 8: ax.legend(loc="best", fontsize="small")
        if preserve_limits:
            ax.set_xlim(previous_limits[0]); ax.set_ylim(previous_limits[1])
        if already_drawn:
            theme = getattr(self.owner, "_apply_plot_theme", None)
            if callable(theme): theme()
            legend = getattr(self.owner, "_toggle_legend", None)
            if callable(legend): legend()
            sync = getattr(self.owner, "_sync_plot_limit_spins_from_axes", None)
            if callable(sync): sync(ax)
        self._repeat_initialized = True
        skipped = list(getattr(self.panel, "_iq_repeat_skipped", [])) + list(report.get("skipped", []))
        report["skipped"] = skipped
        native = np.asarray(report["native_field"])
        finite = np.isfinite(native)
        std_i = float(np.std(native.real[finite], ddof=1)) if np.count_nonzero(finite) > 1 else np.nan
        std_q = float(np.std(native.imag[finite], ddof=1)) if np.count_nonzero(finite) > 1 else np.nan
        native_mean = report["mean_field"]
        self.controls.readout.setText(f"{len(fields)} runs · native mean I {_format(np.real(native_mean))}, Q {_format(np.imag(native_mean))} · sample SD I {_format(std_i)}, Q {_format(std_q)} {report['field_unit']} · {len(skipped)} skipped")
        extras = [("Repeated coordinate", f"{report['coordinate']:.12g}", report["x_unit"]),
                  ("Matched runs", str(len(fields)), "Same physical cut and polarization"),
                  ("Native complex mean I / Q", f"{_format(np.real(native_mean))} / {_format(np.imag(native_mean))}", report["field_unit"]),
                  ("Native RMS distance from complex mean", _format(report["rms_distance"]), report["field_unit"]),
                  ("Displayed complex mean I / Q", f"{_format(np.real(mean))} / {_format(np.imag(mean))}", _scale_label(self.scaling, report["field_unit"])),
                  ("Native sample SD I / Q", f"{_format(std_i)} / {_format(std_q)}", report["field_unit"] + "; n−1 denominator, undefined for one run"),
                  ("Skipped runs", str(len(skipped)), "; ".join(str(value) for value in skipped))]
        self.table.populate(self.result_rows + extras)
        self.rows = []
        for i, value in enumerate(native):
            source = self._repeat_cuts[int(report["cut_indices"][i])]
            phase = report["phase_deg"][i]
            status = "valid" if report["phase_valid"][i] else "undefined or weak"
            self.rows.append((report["labels"][i], int(report["indices"][i]), report["coordinate"], report["x_unit"],
                              value.real, value.imag, abs(value), report["field_unit"], report["db"][i], source.level_unit,
                              phase, status, fields[i].real, fields[i].imag, self.scaling,
                              getattr(source, "polarization", ""), json.dumps(getattr(source, "fixed_coordinates", {}))))
        self.sample_tuples = [("Repeated I", np.arange(len(native)), native.real, report["field_unit"]),
                              ("Repeated Q", np.arange(len(native)), native.imag, report["field_unit"])]
        self._notes = list(dict.fromkeys(self._notes + report.get("notes", [])))


def render_iq(panel, cut, mode):
    """Render an I/Q view and return its disposable interaction controller."""
    return IQController(panel, cut, mode)
