"""Engineering tools that share the Plotting canvas and its native samples."""
from __future__ import annotations

from dataclasses import replace
import csv
import io
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)
from matplotlib.widgets import SpanSelector

from GRIM_Backend.plotting.engineering_tools import finite_runs, measure_trace
from GRIM_Backend.plotting.modes.common import NativeSamples, decimate_line
from GRIM_Backend.ui.engineering_analysis import EngineeringAnalysisDialog, ResultTable, _number
from GRIM_Backend.ui.iq_analysis import IQControls, render_iq


_PHASE_HELP = {
    "iq": "I is the real field and Q is the imaginary field. Distance from the origin is amplitude; its square is linear RCS, scattering width, or power ratio. Angle is phase relative to the measurement reference. Color and arrows follow frequency or angle, not elapsed time or a target outline. Radial motion indicates amplitude change; rotation indicates phase change. A loop alone does not identify a fault or count scatterers.",
    "iq_compare": "Compare A and B at matching physical coordinates. Radial separation indicates amplitude mismatch; angular separation indicates phase mismatch. Choose Common peak = 1 for one shared scale. Phase only removes amplitude information. Different phase references can rotate otherwise similar measurements, so check the metadata notes in Results.",
    "iq_difference": "This plots the complex field A − B at matching physical coordinates. Agreement is at the origin in native/common-peak scaling. Phase only shows residual direction and discards mismatch magnitude; zero has no direction. A large difference can come from amplitude, phase, or both; it is not the difference between dB values. Compare the A/B overlay before attributing a residual to a physical cause.",
    "iq_compensated": "Set a phase offset and, for frequency cuts, a delay in Options. Original and corrected curves remain visible. The correction removes only the stated rotation and phase slope, around the reference frequency reported in Results. A flatter curve can help isolate remaining structure; it does not prove a propagation path or target range. Original data are unchanged.",
    "iq_repeat": "Select repeated datasets before clicking Update. Each point is one run at the same frequency, azimuth, elevation, and polarization. Move the coordinate to examine another location. A tight cloud indicates repeatable complex returns; spread can reflect amplitude or phase variation. The cross is the complex mean. This view does not pool a sweep into a noise cloud or establish coherence from a single run.",
    "phase": "Magnitude shows return strength; phase shows its reference-relative angle. Wrapped phase jumps at ±180 degrees. Unwrapping removes these display jumps within usable segments, but cannot recover missing cycles. Near a null, phase is undefined or unreliable; the floor in Options masks weak phase samples.",
    "residual": "Phase residual is A minus B at matching physical coordinates. A constant offset can indicate a reference difference; a frequency-dependent slope can indicate a delay difference. Use compatible phase conventions before assigning a physical cause.",
    "slope": "The fitted frequency phase slope gives an apparent delay under the selected time convention. Phase sampling can alias delay; unwrapping assumes adjacent changes below 180 degrees. Inspect residuals and the fit span in Results before interpreting the slope as a physical path.",
}


class _RegionSelector(SpanSelector):
    def __init__(self, panel, *args, **kwargs):
        self.panel = panel
        super().__init__(*args, **kwargs)

    def ignore(self, event):
        panel = self.panel
        if panel.isHidden() or panel.owner.plot_figure is not panel.figure:
            return True
        for name in ("btn_zoom_box", "btn_pan", "btn_markers"):
            button = getattr(panel.owner, name, None)
            if button is not None and button.isChecked():
                return True
        overlays = getattr(panel.owner, "spatial_overlays", None)
        if overlays is not None:
            for name in ("draw_button", "measure_button"):
                button = getattr(overlays, name, None)
                if button is not None and button.isChecked():
                    return True
            if getattr(overlays, "drag", None) is not None:
                return True
        return super().ignore(event)


class _HeldFigure:
    """Reuse an existing compatible diagnostic layout for an explicit Hold."""
    def __init__(self, figure):
        self.figure = figure

    def clear(self):
        pass

    def add_subplot(self, *_args, **_kwargs):
        return self.figure.axes[0]

    def subplots(self, *_args, **_kwargs):
        return tuple(self.figure.axes)

    def __getattr__(self, name):
        return getattr(self.figure, name)


class InlineEngineeringAnalysis(QWidget):
    """Small contextual controls; computations and result formats remain shared."""
    TAB_NAMES = EngineeringAnalysisDialog.TAB_NAMES
    _spin = staticmethod(EngineeringAnalysisDialog._spin)
    _dataset_changed = EngineeringAnalysisDialog._dataset_changed
    _axis_changed = EngineeringAnalysisDialog._axis_changed
    _region = EngineeringAnalysisDialog._region
    _maximum_gap = EngineeringAnalysisDialog._maximum_gap
    _sync_region_context = EngineeringAnalysisDialog._sync_region_context
    _export_results = EngineeringAnalysisDialog._export_results
    _analyze_phase = EngineeringAnalysisDialog._analyze_phase
    _analyze_polarization = EngineeringAnalysisDialog._analyze_polarization

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.figure, self.canvas = owner.plot_figure, owner.plot_canvas
        self._saved_canvas_minimum = self.canvas.minimumSize()
        self._updating = True
        self._rendering = False
        self.active_key = None
        self._datasets, self._samples, self._regions = [], {}, {}
        self._region_context = None
        self._measurement, self._span = None, None
        self._peak_position = -1
        self._overlays = []
        self._native_cut_line = None
        self._prefer_native_source = False
        self._sweep_snapshot = {}
        self._snapshot_dataset = None
        self._selected_sweep_preference = True
        self._panels = {}
        self._display_serial = 0
        self._iq_controller = None
        self._iq_hold = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(3)

        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(5)
        self.tool_label = QLabel("Measure")
        row.addWidget(self.tool_label)
        self.trace_combo = QComboBox(); self.trace_combo.setMinimumWidth(110)
        self.trace_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.trace_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.phase_view, self.pol_view = QComboBox(), QComboBox()
        for title, code in (("Magnitude + phase", "phase"), ("I/Q trajectory", "iq"),
                            ("I/Q overlay A / B", "iq_compare"), ("I/Q difference A − B", "iq_difference"),
                            ("I/Q phase / delay compensation", "iq_compensated"), ("I/Q repeated runs", "iq_repeat"),
                            ("Phase residual A − B", "residual"), ("Frequency phase-slope fit", "slope")):
            self.phase_view.addItem(title, code)
        for title, code in (("Channel strengths / ratios", "ratios"), ("HV / VH reciprocity", "reciprocity"),
                            ("Channel phase difference", "phase")):
            self.pol_view.addItem(title, code)
        for combo in (self.trace_combo, self.phase_view, self.pol_view):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(combo, 1)
        measure_row = QHBoxLayout(); measure_row.setSpacing(5)
        self.measure_buttons = []
        for title, mode in (("Peak", "peak"), ("Minimum", "minimum"), ("Next peak", "next")):
            button = QPushButton(title); measure_row.addWidget(button); self.measure_buttons.append(button)
            button.clicked.connect(lambda _checked=False, action=mode: self._mark_measurement(action))
        self.drag_region = QCheckBox("Drag region")
        self.drag_region.setToolTip("Drag across the current trace to measure a range. Zoom, Pan and Markers take priority.")
        self.drag_region.toggled.connect(self._update_current); measure_row.addWidget(self.drag_region)
        self.refresh_button = QPushButton("Update")
        self.refresh_button.setToolTip("Use the current dataset and parameter selections.")
        self.refresh_button.clicked.connect(self.refresh_from_selection); row.addWidget(self.refresh_button)
        self.options_button = QToolButton(); self.options_button.setText("Options")
        self.options_button.setCheckable(True); row.addWidget(self.options_button)
        self.results_button = QToolButton(); self.results_button.setText("Results")
        self.results_button.setCheckable(True); row.addWidget(self.results_button)
        self.interpret_button = QToolButton(); self.interpret_button.setText("Interpret")
        self.interpret_button.setCheckable(True); row.addWidget(self.interpret_button)
        close = QToolButton(); close.setText("×"); close.setToolTip("Hide analysis controls")
        close.clicked.connect(self.deactivate); row.addWidget(close)
        outer.addLayout(row)
        self.summary = QLabel(); self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText); measure_row.addWidget(self.summary, 1)
        outer.addLayout(measure_row)
        self.interpretation = QLabel(); self.interpretation.setWordWrap(True)
        self.interpretation.setTextFormat(Qt.PlainText); self.interpretation.hide()
        self.interpret_button.toggled.connect(lambda checked: self.interpretation.setVisible(checked and self.active_key == "phase"))
        outer.addWidget(self.interpretation)
        self.iq_controls = IQControls()
        self.iq_controls.changed.connect(self._update_current)
        outer.addWidget(self.iq_controls.cursor_widget)
        self.iq_controls.cursor_widget.hide()

        self.options = QScrollArea(); self.options.setWidgetResizable(True)
        self.options.setFixedHeight(110); self.options.setFrameShape(QScrollArea.NoFrame)
        self.options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        options_body = QWidget(); self.options.setWidget(options_body)
        options_body.setAutoFillBackground(False)
        self.options.viewport().setAutoFillBackground(False)
        grid = QGridLayout(options_body)
        grid.setContentsMargins(0, 3, 0, 3); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(3)
        self.options_button.toggled.connect(self._toggle_options)
        outer.addWidget(self.options); self.options.hide()
        self.dataset_combo, self.reference_combo, self.axis_combo = QComboBox(), QComboBox(), QComboBox()
        for title, key in (("Frequency", "frequency"), ("Azimuth", "azimuth"), ("Elevation", "elevation")):
            self.axis_combo.addItem(title, key)
        self.dataset_label, self.reference_label, self.axis_label = QLabel("Dataset A"), QLabel("Reference B"), QLabel("Cut axis")
        for column, (label, combo) in enumerate(((self.dataset_label, self.dataset_combo),
                                                (self.axis_label, self.axis_combo),
                                                (self.reference_label, self.reference_combo))):
            grid.addWidget(label, 1, column * 2); grid.addWidget(combo, 1, column * 2 + 1)
        self.fixed, self.fixed_labels = {}, {}
        for column, name in enumerate(("azimuth", "elevation", "frequency", "polarization")):
            combo = QComboBox(); label = QLabel(name.capitalize())
            self.fixed[name], self.fixed_labels[name] = combo, label
            grid.addWidget(label, 2, column * 2); grid.addWidget(combo, 2, column * 2 + 1)
            combo.currentIndexChanged.connect(self._update_current)
        self.use_region = QCheckBox("Use region")
        self.region_start, self.region_stop = self._spin(), self._spin()
        grid.addWidget(self.use_region, 3, 0); grid.addWidget(self.region_start, 3, 1)
        grid.addWidget(QLabel("to"), 3, 2); grid.addWidget(self.region_stop, 3, 3)
        self.gap_enabled = QCheckBox("Split gaps above")
        self.gap = self._spin(minimum=1e-12, maximum=1e300, value=1.)
        grid.addWidget(self.gap_enabled, 3, 4); grid.addWidget(self.gap, 3, 5)
        self.floor = self._spin(minimum=-300., maximum=0., value=-60.); self.floor.setSuffix(" dB")
        self.floor_label = QLabel("Mask below peak")
        grid.addWidget(self.floor_label, 4, 0); grid.addWidget(self.floor, 4, 1)
        self.crossing_level, self.peak_drop = self._spin(value=-10.), self._spin(minimum=0., value=3.)
        self.crossing_label, self.peak_drop_label = QLabel("Crossing level"), QLabel("Peak drop for width")
        grid.addWidget(self.crossing_label, 4, 0); grid.addWidget(self.crossing_level, 4, 1)
        grid.addWidget(self.peak_drop_label, 4, 2); grid.addWidget(self.peak_drop, 4, 3)
        self.phase_sign = QComboBox()
        for title, code in (("Declared sign (+jwt if unspecified)", "auto"), ("exp(+jwt)", "+jwt"), ("exp(−jwt)", "-jwt")):
            self.phase_sign.addItem(title, code)
        grid.addWidget(self.phase_sign, 4, 2, 1, 2)
        self.pol_a, self.pol_b = QComboBox(), QComboBox()
        self.pol_a_label, self.pol_b_label = QLabel("Channel A"), QLabel("Channel B")
        grid.addWidget(self.pol_a_label, 4, 2); grid.addWidget(self.pol_a, 4, 3)
        grid.addWidget(self.pol_b_label, 4, 4); grid.addWidget(self.pol_b, 4, 5)
        analyze = QPushButton("Apply options"); grid.addWidget(analyze, 4, 6, 1, 2)
        analyze.clicked.connect(self._update_current)
        self.selected_sweep = QCheckBox("Use selected sweep samples")
        self.selected_sweep.setChecked(True)
        self.selected_sweep.setToolTip("Use the sweep-axis samples selected in the main parameter list when Update was clicked. Uncheck for the full native cut.")
        self.selected_sweep.toggled.connect(self._selected_sweep_changed)
        self.sweep_note = QLabel(); self.sweep_note.setWordWrap(True)
        grid.addWidget(self.selected_sweep, 5, 0, 1, 3); grid.addWidget(self.sweep_note, 5, 3, 1, 5)
        grid.addWidget(self.iq_controls, 0, 0, 1, 8)
        self.iq_controls.hide()

        self.results = QScrollArea(); self.results.setWidgetResizable(True)
        self.results.setFrameShape(QScrollArea.NoFrame)
        self.results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_body = QWidget(); self.results.setWidget(results_body)
        results_body.setAutoFillBackground(False)
        self.results.viewport().setAutoFillBackground(False)
        result_layout = QVBoxLayout(results_body)
        result_layout.setContentsMargins(0, 0, 0, 0); result_layout.setSpacing(3)
        self.results_button.toggled.connect(self._toggle_results)
        self.tabs = QTabWidget(); self.tabs.hide()
        for key in self.TAB_NAMES:
            table, note = ResultTable(), QLabel()
            table.setMinimumHeight(100)
            note.setWordWrap(True); note.setTextFormat(Qt.PlainText)
            page = QWidget(); page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(table, 1); page_layout.addWidget(note)
            self.tabs.addTab(page, key)
            self._panels[key] = (self.figure, self.canvas, table, note)
        self.tabs.tabBar().hide(); result_layout.addWidget(self.tabs, 1)
        result_actions = QHBoxLayout()
        for title, callback in (("Copy results", lambda: self._panels[self.active_key][2].copy()),
                                ("Copy analysis samples", lambda: self._copy_samples(self.active_key)),
                                ("Export results CSV…", lambda: self._export_results(self.active_key))):
            button = QPushButton(title); button.clicked.connect(callback); result_actions.addWidget(button)
        self.export_samples_button = QPushButton("Export I/Q samples CSV…")
        self.export_samples_button.clicked.connect(self._export_iq_samples)
        result_actions.addWidget(self.export_samples_button); self.export_samples_button.hide()
        result_actions.addStretch(); result_layout.insertLayout(0, result_actions)
        host = self.canvas.parentWidget()
        host_layout = host.layout() if host is not None else None
        if host_layout is not None:
            host_layout.insertWidget(host_layout.indexOf(self.canvas) + 1, self.results, 1)
        else:
            outer.addWidget(self.results, 1)
        self.results.hide(); self.tabs.show()
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        self.reference_combo.currentIndexChanged.connect(self._update_current)
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        self.trace_combo.currentIndexChanged.connect(self._reset_measurement)
        self.phase_view.currentIndexChanged.connect(self._update_current)
        self.pol_view.currentIndexChanged.connect(self._update_current)
        self._updating = False
        self.hide()

    def _update_canvas_minimum(self):
        expanded = self.options_button.isChecked()
        self.canvas.setMinimumHeight(min(40 if expanded else 180, self._saved_canvas_minimum.height()))

    def _toggle_options(self, checked):
        if checked: self.results_button.setChecked(False)
        self.options.setVisible(checked)
        self._update_canvas_minimum()

    def _toggle_results(self, checked):
        if checked: self.options_button.setChecked(False)
        self.canvas.setVisible(not checked)
        self.results.setVisible(checked)
        self._update_canvas_minimum()

    def open_tool(self, key):
        if key not in self.TAB_NAMES:
            return
        if self.owner.plot_figure is not self.figure:
            return
        previous = self.active_key
        self.active_key = key
        if previous != key:
            self._prefer_native_source = False
        self._updating = True
        self.tabs.setCurrentIndex(self.TAB_NAMES[key])
        if previous != key:
            self.options_button.setChecked(False); self.results_button.setChecked(False)
        self._updating = False
        self.refresh_from_selection()
        self._update_canvas_minimum()
        self.show()

    def refresh_from_selection(self, *_args):
        if not self.active_key or self.owner.plot_figure is not self.figure:
            return
        old_line = self.trace_combo.currentData()
        old_reference = self.reference_combo.currentText()
        prefer_native = self.active_key == "measure" and self._prefer_native_source
        self._updating = True
        self._measurement = None; self._peak_position = -1
        self.stop_region_selection()
        getter = getattr(self.owner, "_selected_datasets", None)
        self._datasets = list(getter() if callable(getter) else [])
        if not self._datasets and getattr(self.owner, "active_dataset", None) is not None:
            self._datasets = [("Active dataset", self.owner.active_dataset)]
        self._snapshot_dataset = getattr(self.owner, "active_dataset", None)
        self._sweep_snapshot = {}
        for axis, name in (("azimuth", "list_az"), ("frequency", "list_freq"), ("elevation", "list_elev")):
            widget = getattr(self.owner, name, None)
            selected = getattr(self.owner, "_selected_indices", None)
            if widget is not None and callable(selected):
                self._sweep_snapshot[axis] = tuple(sorted(selected(widget)))
        for combo in (self.dataset_combo, self.reference_combo):
            combo.clear()
            for label, _dataset in self._datasets:
                combo.addItem(str(label))
        if len(self._datasets) > 1:
            found = self.reference_combo.findText(old_reference)
            self.reference_combo.setCurrentIndex(found if found > 0 else 1)
        current_mode = getattr(self.owner, "last_plot_mode", None)
        axis = {"frequency": "frequency", "elevation_sweep": "elevation", "azimuth_rect": "azimuth",
                "azimuth_polar": "azimuth", "compare": "azimuth"}.get(current_mode)
        if axis is not None:
            self.axis_combo.setCurrentIndex(self.axis_combo.findData(axis))
        self.trace_combo.clear(); self.trace_combo.addItem("Selected native dataset cut", None)
        reader = getattr(self.owner, "_marker_lines", None)
        lines = list(reader() if callable(reader) else [])
        for line in lines:
            label = line.get_label()
            if getattr(line, "_grim_native_probe_unavailable", False):
                label += " [display samples only]"
            self.trace_combo.addItem(label, line)
        if lines and not prefer_native:
            self.trace_combo.setCurrentIndex(1 + (lines.index(old_line) if old_line in lines else 0))
        if prefer_native:
            self._native_cut_line = None
        self._updating = False
        self._dataset_changed()

    refresh_sources = refresh_from_selection

    def render_plot_mode(self, key):
        """Replot an existing diagnostic even after its controls were closed."""
        if self.owner.plot_figure is not self.figure or key not in self.TAB_NAMES:
            return
        hidden = self.isHidden()
        self.active_key = key
        if key == "measure": self._prefer_native_source = True
        self.tabs.setCurrentIndex(self.TAB_NAMES[key])
        self.refresh_from_selection()
        if hidden:
            self._remove_overlays()
            self.active_key = None

    def _selected_sweep_changed(self, checked):
        self._selected_sweep_preference = bool(checked)
        self._reset_measurement()

    def _selected_sweep_indices(self):
        index = self.dataset_combo.currentIndex()
        axis = str(self.axis_combo.currentData())
        if (0 <= index < len(self._datasets) and self._datasets[index][1] is self._snapshot_dataset
                and self._selected_sweep_preference):
            return self._sweep_snapshot.get(axis)
        return None

    def _cut(self, *, reference=False, polarization=None):
        cut = EngineeringAnalysisDialog._cut(self, reference=reference, polarization=polarization)
        indices = None if reference else self._selected_sweep_indices()
        if indices is None:
            context = "Full native sweep; " + ("reference coordinates are matched to A." if reference else "all sweep-axis samples.")
            return replace(cut, notes=[*cut.notes, context])
        index = np.asarray(indices, dtype=int)
        index = index[(index >= 0) & (index < len(cut.x))]
        context = f"Main selection: {len(index)} of {len(cut.x)} native {cut.axis} samples, captured at Update."
        return replace(cut, x=cut.x[index], power=cut.power[index], field=cut.field[index], db=cut.db[index],
                       native_indices=cut.native_indices[index] if cut.native_indices is not None else index,
                       notes=[*cut.notes, context])

    def _reset_measurement(self, *_args):
        if self._updating:
            return
        if self.active_key == "measure":
            self._prefer_native_source = self.trace_combo.currentData() is None
        self._measurement = None; self._peak_position = -1
        self._update_current()

    def _iq_repeat_cuts(self):
        """One cut per selected run, with fixed coordinates matched to A."""
        cuts, notes = [self._cut()], []
        self._iq_repeat_skipped = []
        source = self.dataset_combo.currentIndex()
        for index, (name, _dataset) in enumerate(self._datasets):
            if index == source:
                continue
            try:
                cuts.append(EngineeringAnalysisDialog._cut(self, dataset_index=index))
            except (ValueError, IndexError, KeyError) as error:
                notes.append(f"Skipped {name}: {error}")
                self._iq_repeat_skipped.append({"label": name, "reason": str(error)})
        if len(cuts) < 2:
            notes.append("Only one usable run: select repeated datasets and click Update to compare repeatability.")
        return cuts, notes

    def _copy_samples(self, key):
        controller = self._iq_controller if key == "phase" else None
        if controller is None:
            return EngineeringAnalysisDialog._copy_samples(self, key)
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(controller.headers); writer.writerows(controller.rows)
        QApplication.clipboard().setText(output.getvalue())

    def _export_iq_samples(self):
        controller = self._iq_controller
        if controller is None:
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export native I/Q samples", "iq-samples.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            writer.writerow(controller.headers); writer.writerows(controller.rows)

    def _disconnect_iq(self):
        if self._iq_controller is not None:
            self._iq_controller.disconnect()
            self._iq_controller = None

    def _presentation(self):
        key = self.active_key
        self.tool_label.setText({"measure": "Measure", "phase": "Phase analysis", "polarization": "Polarization"}[key])
        for name, widget in (("measure", self.trace_combo), ("phase", self.phase_view), ("polarization", self.pol_view)):
            widget.setVisible(name == key)
        for widget in (*self.measure_buttons, self.drag_region):
            widget.setVisible(key == "measure")
        native = key != "measure" or self.trace_combo.currentData() is None
        for widget in (self.dataset_label, self.dataset_combo, self.axis_label, self.axis_combo):
            widget.setVisible(native)
        view = self.phase_view.currentData()
        iq = key == "phase" and str(view).startswith("iq")
        self.iq_controls.setVisible(iq); self.iq_controls.cursor_widget.setVisible(iq)
        self.options.setFixedHeight(170 if iq else 110)
        self.export_samples_button.setVisible(iq)
        if iq: self.iq_controls.configure(view)
        self.interpret_button.setVisible(key == "phase")
        self.interpretation.setText(_PHASE_HELP.get(view, ""))
        self.interpretation.setVisible(key == "phase" and self.interpret_button.isChecked())
        residual = key == "phase" and view in {"residual", "iq_compare", "iq_difference"}
        self.reference_label.setVisible(residual); self.reference_combo.setVisible(residual)
        for name, combo in self.fixed.items():
            visible = native and name != self.axis_combo.currentData() and not (key == "polarization" and name == "polarization")
            combo.setVisible(visible); self.fixed_labels[name].setVisible(visible)
        for widget in (self.crossing_label, self.crossing_level, self.peak_drop_label, self.peak_drop):
            widget.setVisible(key == "measure")
        self.floor_label.setVisible(key != "measure"); self.floor.setVisible(key != "measure")
        self.phase_sign.setVisible(key == "phase" and view in {"slope", "iq_compensated"})
        for widget in (self.pol_a_label, self.pol_a, self.pol_b_label, self.pol_b):
            widget.setVisible(key == "polarization" and self.pol_view.currentData() == "phase")
        index = self.dataset_combo.currentIndex()
        selected_available = (0 <= index < len(self._datasets) and
                              self._datasets[index][1] is self._snapshot_dataset and
                              self.axis_combo.currentData() in self._sweep_snapshot)
        self.selected_sweep.setVisible(native); self.sweep_note.setVisible(native)
        self.selected_sweep.setEnabled(selected_available)
        self.selected_sweep.blockSignals(True)
        self.selected_sweep.setChecked(selected_available and self._selected_sweep_preference)
        self.selected_sweep.blockSignals(False)
        indices = self._selected_sweep_indices()
        self.sweep_note.setText(f"{len(indices)} selected native samples; Update follows the main selection." if indices is not None else
                               "Full native cut for Dataset A." if selected_available else
                               "Full native cut: Dataset A is not the active parameter-list dataset.")

    def _update_current(self, *_args):
        if self._updating or not self.active_key or self.owner.plot_figure is not self.figure:
            return
        self._presentation()
        key = self.active_key
        try:
            self._sync_region_context(key)
            if key == "measure":
                self._analyze_measure()
            else:
                self._render_diagnostic(key)
            self.summary.setStyleSheet("")
        except (ValueError, TypeError, IndexError, KeyError) as error:
            self._show_analysis_error(key, error)

    def _show_analysis_error(self, key, error):
        self._disconnect_iq()
        self.iq_controls.readout.clear()
        self._panels[key][2].populate([]); self._panels[key][3].setText(str(error))
        self._samples[key] = []
        if key == "measure":
            self._measurement = None; self._remove_overlays()
        self.summary.setText(f"Analysis unavailable: {error}")
        self.summary.setStyleSheet("color: #d99c35;")
        self.canvas.draw_idle()

    def _invalidate_recipe(self):
        self.owner.last_python_plot_spec = None
        recorder = getattr(self.owner, "python_recorder", None)
        if recorder is not None:
            recorder.invalidate_current_plot()

    def _render_diagnostic(self, key):
        cut = self._cut()
        view = self.phase_view.currentData() if key == "phase" else self.pol_view.currentData()
        iq = key == "phase" and str(view).startswith("iq")
        if key == "phase" and view == "slope" and cut.axis != "frequency":
            raise ValueError("Choose Frequency in Options for a phase-slope fit.")
        if key == "phase" and view == "residual":
            self._cut(reference=True)
        signature = (key, view, cut.axis, cut.x_unit, cut.level_unit)
        if iq:
            signature += (self.iq_controls.scale_combo.currentData(), self.iq_controls.linked_checkbox.isChecked(), cut.linear_quantity)
        same_layout = getattr(self.figure, "_grim_engineering_signature", None) == signature
        old_limits = [(axes.get_xlim(), axes.get_ylim()) for axes in self.figure.axes] if same_layout else []
        hold = getattr(self.owner, "btn_hold", None)
        held = bool(hold is not None and hold.isChecked() and self.figure.axes and
                    same_layout)
        self._iq_hold = bool(iq and view == "iq" and held)
        if not iq: self.figure._grim_iq_held = None
        if iq: held = False  # The renderer rebuilds held native cuts with one shared scale.
        self._rendering = True
        self._remove_overlays()
        self.owner.last_plot_mode = "engineering_" + key
        try:
            start = getattr(self.owner, "_start_plot_render", None)
            if callable(start): start()
            if not held:
                release = getattr(self.owner, "_release_native_plot_samples", None)
                if callable(release): release()
                remove = getattr(self.owner, "_remove_colorbar", None)
                if callable(remove): remove()
                markers = getattr(self.owner, "_clear_plot_markers", None)
                if callable(markers) and not same_layout: markers(redraw=False)
                self.figure.set_layout_engine("constrained")
            self._invalidate_recipe()
            self._display_serial = 0
            self._display_cut = cut
            original = self._panels[key]
            if held:
                self._panels[key] = (_HeldFigure(self.figure), *original[1:])
            try:
                if iq:
                    self._iq_controller = render_iq(self, cut, view)
                    self._samples[key] = self._iq_controller.sample_tuples
                else:
                    getattr(self, "_analyze_" + key)()
            finally:
                self._panels[key] = original
            self.figure._grim_engineering_signature = signature
            data_axes = self._iq_controller.axes if iq else list(self.figure.axes)
            self.figure._grim_independent_y = len(data_axes) > 1
            self.figure._grim_independent_x = iq and len(data_axes) > 1
            self.figure._grim_line_plot_signature = None
            self.owner.plot_ax = data_axes[0]
            self.owner.plot_axes = data_axes if len(data_axes) > 1 else None
            sync = getattr(self.owner, "_sync_plot_limit_spins_from_axes", None)
            if callable(sync): sync(self.owner.plot_ax)
            theme = getattr(self.owner, "_apply_plot_theme", None)
            if callable(theme): theme()
            color = getattr(self.owner, "_current_plot_text", None)
            if callable(color):
                for text in self.figure.texts: text.set_color(color())
                for axes in self.figure.axes: axes.title.set_color(color())
            legend = getattr(self.owner, "_toggle_legend", None)
            if callable(legend): legend()
            auto_scale = getattr(self.owner, "btn_auto_scale", None)
            if auto_scale is not None and auto_scale.isChecked():
                fit = getattr(self.owner, "_maybe_autoscale", None)
                if callable(fit): fit()
            elif old_limits and len(old_limits) == len(self.figure.axes):
                for axes, (xlim, ylim) in zip(self.figure.axes, old_limits):
                    axes.set_xlim(xlim); axes.set_ylim(ylim)
                if callable(sync): sync(self.owner.plot_ax)
            restore_markers = getattr(self.owner, "_restore_plot_markers", None)
            if callable(restore_markers): restore_markers()
            self.summary.setText(self._iq_controller.summary if iq else {
                ("phase", "phase"): "Return strength above; wrapped and locally unwrapped phase below.",
                ("phase", "residual"): "Phase A − B at matching physical coordinates. Use a common phase reference.",
                ("phase", "slope"): "Frequency phase slope above; fit residual below. Apparent delay and sign assumptions are in Results.",
                ("polarization", "ratios"): "Channel strengths above; co- and cross-channel power ratios below.",
                ("polarization", "reciprocity"): "HV / VH strengths above and power ratio below. Reciprocity depends on the measurement conventions.",
                ("polarization", "phase"): "Channel strengths above; selected channel phase difference below.",
            }[(key, view)])
            self.summary.setToolTip(self._panels[key][3].text())
            status = getattr(self.owner, "status", None)
            if status is not None:
                status.showMessage("Phase analysis updated." if key == "phase" else "Polarization analysis updated.")
            self.canvas.draw_idle()
        except (ValueError, TypeError, IndexError, KeyError):
            self.figure._grim_engineering_signature = None
            raise
        finally:
            # Failed analyses can clear or partially rebuild the figure. Never
            # leave the main toolbar referring to detached axes afterward.
            if not self.figure.axes:
                self.figure.add_subplot(111)
            data_axes = self._iq_controller.axes if self._iq_controller is not None else [ax for ax in self.figure.axes if ax.get_label() != "<colorbar>"]
            if not data_axes: data_axes = [self.figure.add_subplot(111)]
            self.owner.plot_ax = data_axes[0]
            self.owner.plot_axes = data_axes if len(data_axes) > 1 else None
            self._rendering = False

    def _display_line(self, axes, x, y, *, label=None, split_gaps=True, **kwargs):
        x, y = np.asarray(x), np.asarray(y)
        native_x, native_y = x, y
        if split_gaps and self._maximum_gap() is not None:
            runs = list(finite_runs(x, np.isfinite(y), maximum_gap=self._maximum_gap()))
            if runs:
                x = np.concatenate([np.r_[x[run], np.nan] for run in runs])
                y = np.concatenate([np.r_[y[run], np.nan] for run in runs])
        dataset_index = self.dataset_combo.currentIndex()
        dataset = self._datasets[dataset_index][1] if 0 <= dataset_index < len(self._datasets) else None
        name = self._datasets[dataset_index][0] if dataset is not None else "Analysis"
        displayed_label = f"{name} · {label}" if label else str(name)
        self._display_serial += 1
        trace_key = ("engineering", self.active_key, self.phase_view.currentData(), self.pol_view.currentData(),
                     self.axis_combo.currentData(), tuple(combo.currentData() for combo in self.fixed.values()),
                     label, self._display_serial)
        bounded = getattr(self.owner, "_plot_bounded_line", None)
        if callable(bounded):
            lines = bounded(axes, x, y, label=displayed_label, dataset=dataset, trace_key=trace_key,
                            native_values=(native_x, native_y), **kwargs)
            return lines[0] if lines else None
        xx, yy, reduced = decimate_line(x, y, max_points=20_000)
        line = axes.plot(xx, yy, label=displayed_label, **kwargs)[0]
        line._grim_dataset_key = id(dataset); line._grim_trace_key = trace_key
        if reduced or len(native_x) != len(x):
            line._grim_native_samples = NativeSamples.curve(native_x, native_y)
            line._grim_display_reduced = True
        return line

    def _analyze_measure(self):
        source = self.trace_combo.currentData()
        line = source
        if source is None:
            cut = self._cut()
            x, y, label = cut.x.copy(), cut.db.copy(), cut.label
            x_label, y_label = f"{cut.axis} ({cut.x_unit})", cut.level_unit
            note_text = "Native dataset cut. Mean and standard deviation describe dB values, not mean linear RCS."
            cut_key = (id(self._datasets[self.dataset_combo.currentIndex()][1]), cut.axis,
                       tuple(combo.currentData() for combo in self.fixed.values()),
                       self._selected_sweep_indices(), self._maximum_gap())
            note_text += " " + cut.notes[-1]
            self._prefer_native_source = True
            previous = self._native_cut_line
            if (previous is not None and previous.axes in self.figure.axes and previous in previous.axes.lines
                    and getattr(previous, "_grim_native_cut_key", None) == cut_key):
                line = previous
        else:
            if line.axes is None or line.axes not in self.figure.axes or line not in line.axes.lines:
                raise ValueError("The selected trace was replaced. Click Update to use the current plot.")
            native = getattr(line, "_grim_native_samples", None)
            if native is not None and native.values is None:
                raise ValueError("The native trace was released. Click Update to use the current plot.")
            reader = getattr(self.owner, "_marker_arrays", None)
            values = reader(line) if callable(reader) else (line.get_xdata(), line.get_ydata())
            x, y = np.array(values[0], dtype=float, copy=True), np.array(values[1], dtype=float, copy=True)
            if line.axes.name == "polar": x = np.rad2deg(x)
            label, x_label, y_label = line.get_label(), line.axes.get_xlabel(), line.axes.get_ylabel()
            if line.axes.name == "polar": x_label = "Azimuth (deg)"
            note_text = "Statistics and peak drop use the trace's displayed ordinate unit. "
            note_text += ("Native samples unavailable; results use displayed samples." if
                          getattr(line, "_grim_native_probe_unavailable", False) else
                          "Measurements use native samples, including samples omitted from the overview.")
        order = None
        if not np.all(np.diff(x) >= 0):
            order = np.argsort(x, kind="stable"); x, y = x[order], y[order]
        result = measure_trace(x, y, region=self._region(), level=self.crossing_level.value(),
                               drop=self.peak_drop.value(), maximum_gap=self._maximum_gap())
        self._remove_overlays()
        if line is None:
            # An explicit native-cut source uses the shared canvas, just like a plot button.
            self._rendering = True
            self.owner.last_plot_mode = "engineering_measure"
            try:
                start = getattr(self.owner, "_start_plot_render", None)
                if callable(start): start()
                release = getattr(self.owner, "_release_native_plot_samples", None)
                if callable(release): release()
                remove = getattr(self.owner, "_remove_colorbar", None)
                if callable(remove): remove()
                markers = getattr(self.owner, "_clear_plot_markers", None)
                if callable(markers): markers(redraw=False)
                self.figure.clear(); self.figure.set_layout_engine("constrained")
                ax = self.figure.add_subplot(111)
                self.owner.plot_ax, self.owner.plot_axes = ax, None
                self.figure._grim_independent_y = False
                self.figure._grim_independent_x = False
                self.figure._grim_engineering_signature = None
                self._display_serial = 0
                line = self._display_line(ax, x, y, label=label)
                self._native_cut_line = line
                if line is not None: line._grim_native_cut_key = cut_key
                ax.set(xlabel=x_label, ylabel=y_label, title=label)
                ax.grid(True, alpha=.25)
                self._invalidate_recipe()
                theme = getattr(self.owner, "_apply_plot_theme", None)
                if callable(theme): theme()
                color = getattr(self.owner, "_current_plot_text", None)
                if callable(color): ax.title.set_color(color())
                sync = getattr(self.owner, "_sync_plot_limit_spins_from_axes", None)
                if callable(sync): sync(ax)
            finally:
                self._rendering = False
        else:
            ax = line.axes
        if source is None: order = None  # The native-cut line was created in sorted order.
        limits = (ax.get_xlim(), ax.get_ylim())
        coord = np.deg2rad if ax.name == "polar" else lambda value: value
        self._overlays.append(ax.axhline(self.crossing_level.value(), color="0.5", linestyle=":", label="_measurement_level"))
        for crossing in result["crossings"]:
            self._overlays.append(ax.axvline(coord(crossing), color="#2a9d8f", alpha=.5, label="_measurement_crossing"))
        if self._region() is not None:
            self._overlays.append(ax.axvspan(*coord(np.array(sorted(self._region()))), alpha=.1, color="#2a9d8f"))
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        if self.drag_region.isChecked():
            self._span = _RegionSelector(self, ax, lambda left, right: self._select_region(
                *np.rad2deg([left, right]) if ax.name == "polar" else (left, right)),
                "horizontal", useblit=False, props={"alpha": .18, "facecolor": "#2a9d8f"}, interactive=True)
        self._measurement = (x, y, result, line, order)
        rows = [("Source", label, f"{x_label}; {y_label}"),
                ("Region", str(self._region()) if self._region() else "Full trace", x_label),
                ("Finite / selected samples", f"{result['finite_count']} / {result['selected_count']}", "native coordinates"),
                ("Coverage", _number(result["coverage"] * 100), "%")]
        for name in ("peak", "minimum"):
            point = result.get(name)
            rows.append((name.capitalize(), f"x={point['x']:.9g}; value={point['y']:.9g}" if point else "not defined", f"{x_label}; {y_label}"))
        for name in ("mean", "median", "std", "peak_to_peak"):
            rows.append((name.replace("_", " ").capitalize(), _number(result.get(name)), f"displayed {y_label}"))
        for index, crossing in enumerate(result["crossings"], 1):
            rows.append((f"Level crossing {index}", _number(crossing), x_label))
        width = result["bandwidth"]
        if width:
            rows.extend((("Peak-lobe width", _number(width["width"]), f"{x_label}; drop {self.peak_drop.value():g} {y_label}"),
                         ("Left / right crossing", f"{width['left']:.9g} / {width['right']:.9g}", x_label)))
        else:
            rows.append(("Peak-lobe width", "not defined", result.get("bandwidth_note", result.get("note", ""))))
        self._panels["measure"][2].populate(rows)
        self._panels["measure"][3].setText(note_text + " Crossings use linear interpolation between adjacent finite samples.")
        self._samples["measure"] = [(label, x, y, y_label)]
        peak = result.get("peak")
        summary = f"{result['finite_count']} samples"
        if peak:
            summary += f" · Peak {peak['y']:.6g} at {peak['x']:.6g}"
        summary += f" · Mean {_number(result.get('mean'))}"
        if width: summary += f" · Width {width['width']:.6g}"
        self.summary.setText(summary); self.summary.setToolTip(note_text)
        self.canvas.draw_idle()

    def _select_region(self, left, right):
        if self.owner.plot_figure is not self.figure or not self.active_key: return
        if left == right: return
        self.region_start.setValue(left); self.region_stop.setValue(right)
        self.use_region.setChecked(True); self._update_current()

    def _mark_measurement(self, mode):
        if self.owner.plot_figure is not self.figure or self.active_key != "measure": return
        try:
            self._analyze_measure()
            x, y, result, line, order = self._measurement
            if mode == "next":
                peaks = sorted(result["peaks"], key=lambda point: point["x"])
                if not peaks: raise ValueError("No interior local peaks in the selected region.")
                self._peak_position = (self._peak_position + 1) % len(peaks)
                point = peaks[self._peak_position]
            else:
                point = result.get(mode)
            if point is None: raise ValueError("No finite samples in the selected region.")
            adder = getattr(self.owner, "_add_plot_marker", None)
            if callable(adder):
                adder(line, int(order[point["index"]]) if order is not None else point["index"])
            else:
                ax = line.axes; value = np.deg2rad(point["x"]) if ax.name == "polar" else point["x"]
                self._overlays += ax.plot([value], [point["y"]], "D", color="#e76f51")
            self.summary.setText(f"{mode.capitalize()}: x={point['x']:.9g}, value={point['y']:.9g}")
            self.canvas.draw_idle()
        except (ValueError, TypeError, KeyError) as error:
            self.summary.setText(f"Marker unavailable: {error}")

    def stop_region_selection(self):
        if self._span is not None:
            self._span.disconnect_events()
            for artist in self._span.artists:
                try: artist.remove()
                except (ValueError, NotImplementedError): pass
            self._span = None

    def resume_interaction(self):
        if (self.active_key == "measure" and self.drag_region.isChecked() and not self.isHidden()
                and self.owner.plot_figure is self.figure):
            self._update_current()

    def _remove_overlays(self):
        self._disconnect_iq()
        self.stop_region_selection()
        for artist in self._overlays:
            try: artist.remove()
            except (ValueError, NotImplementedError): pass
        self._overlays.clear()

    def deactivate(self, *_args):
        if self._rendering: return
        self._remove_overlays()
        self.figure._grim_iq_held = None
        self._measurement = None
        self.active_key = None
        self.results_button.setChecked(False)
        self.results.hide()
        self.canvas.show()
        self.hide()
        self.canvas.setMinimumSize(self._saved_canvas_minimum)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self.deactivate()
        super().closeEvent(event)
