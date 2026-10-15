"""Nonmodal measurement, phase, and polarization workbench for native cuts."""
from __future__ import annotations

import csv
import io
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import SpanSelector

from GRIM_Backend.plotting.engineering_tools import (
    dataset_cut, measure_trace, finite_runs,
)
from GRIM_Backend.plotting.modes.common import (
    axis_unit, convert_axis_values, decimate_line, insert_plot_breaks, singleton_line_markers,
)
from GRIM_Backend.ui.widgets import CollapsibleSection
from GRIM_Backend.ui.analysis_presenter import EngineeringAnalysisPresenter
from GRIM_Backend.plotting.analysis_results import number as _number, run_samples as _run_samples


class ResultTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["Metric", "Value", "Unit / context"])
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setMinimumHeight(70)
        self.rows = []

    def populate(self, rows):
        self.rows = [[str(cell) for cell in row] for row in rows]
        self.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            for j, cell in enumerate(row):
                self.setItem(i, j, QTableWidgetItem(cell))
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def copy(self):
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("Metric", "Value", "Unit / context")); writer.writerows(self.rows)
        QApplication.clipboard().setText(output.getvalue())


def _with_run_breaks(x, y, runs):
    """Keep native samples and insert display/probe breaks between adjacent runs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if runs is not None:
        boundaries = [int(run[0]) for run in runs if len(run) and run[0] > 0
                      and np.isfinite(x[run[0] - 1]) and np.isfinite(y[run[0] - 1])]
        x, y = insert_plot_breaks(boundaries, x, y)
    return x, y


def _trace_sort_order(x):
    """Sort within finite coordinate blocks, retaining explicit gap separators."""
    order = np.arange(len(x))
    finite = np.flatnonzero(np.isfinite(x))
    for block in np.split(finite, np.flatnonzero(np.diff(finite) != 1) + 1):
        order[block] = block[np.argsort(x[block], kind="stable")]
    return None if np.array_equal(order, np.arange(len(x))) else order


class CompactNumberSpin(QDoubleSpinBox):
    """Show meaningful digits without filling RF controls with trailing zeroes."""

    def textFromValue(self, value):
        return f"{value:.10g}"

    def valueFromText(self, text):
        return float(text.removeprefix(self.prefix()).removesuffix(self.suffix()).strip())

    def validate(self, text, position):
        value = text.removeprefix(self.prefix()).removesuffix(self.suffix()).strip()
        try:
            parsed = float(value)
            state = QValidator.Acceptable if np.isfinite(parsed) and self.minimum() <= parsed <= self.maximum() else QValidator.Invalid
        except ValueError:
            state = QValidator.Intermediate if not value or all(c in "+-.eE0123456789" for c in value) else QValidator.Invalid
        return state, text, position


class EngineeringAnalysisDialog(EngineeringAnalysisPresenter, QDialog):
    TAB_NAMES = {"measure": 0, "phase": 1, "polarization": 2}

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Engineering Analysis")
        self.setAttribute(Qt.WA_DeleteOnClose)
        available = self.screen().availableGeometry()
        self.resize(min(1120, int(available.width() * .94)), min(820, int(available.height() * .9)))
        self._updating = False
        self._datasets = []
        self._span = None
        self._measurement = None
        self._peak_position = -1
        self._samples = {}
        self._region_context = None
        self._regions = {}
        self._panels = {}
        self._chart_parts = {}
        self.purpose_labels = {}
        self.detail_sections = {}
        self.analysis_errors = {}
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._analysis_scroll = QScrollArea(self)
        self._analysis_scroll.setWidgetResizable(True)
        self._analysis_scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget(); layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6); layout.setSpacing(4)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        palette = getattr(owner, "application_palette", {})
        background, foreground = palette.get("win_bg", "#101828"), palette.get("text", "#e5e7eb")
        self._analysis_scroll.setStyleSheet(
            f"QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {background}; color: {foreground}; }} "
            f"QLabel, QCheckBox {{ color: {foreground}; }}")
        self._analysis_scroll.setWidget(content); outer.addWidget(self._analysis_scroll)
        self.source_note = QLabel()
        self.source_note.setWordWrap(True)
        layout.addWidget(self.source_note)
        selectors = QGridLayout()
        self._source_selectors = selectors
        self.dataset_combo, self.reference_combo, self.axis_combo = QComboBox(), QComboBox(), QComboBox()
        self.axis_combo.addItem("Frequency", "frequency")
        self.axis_combo.addItem("Azimuth", "azimuth")
        self.axis_combo.addItem("Elevation", "elevation")
        self.dataset_label = QLabel("Dataset A")
        self.reference_label = QLabel("Reference B")
        self.axis_label = QLabel("Cut axis")
        selectors.addWidget(self.dataset_label, 0, 0); selectors.addWidget(self.dataset_combo, 0, 1)
        selectors.addWidget(self.reference_label, 0, 2); selectors.addWidget(self.reference_combo, 0, 3)
        selectors.addWidget(self.axis_label, 0, 4); selectors.addWidget(self.axis_combo, 0, 5)
        refresh = QPushButton("Refresh from selection")
        self.refresh_button = refresh
        refresh.clicked.connect(self.refresh_sources)
        selectors.addWidget(refresh, 0, 6, 1, 2)
        self.fixed = {}
        self.fixed_labels = {}
        for column, name in enumerate(("azimuth", "elevation", "frequency", "polarization")):
            combo = QComboBox(); self.fixed[name] = combo
            self.fixed_labels[name] = QLabel(name.capitalize())
            selectors.addWidget(self.fixed_labels[name], 1, column * 2)
            selectors.addWidget(combo, 1, column * 2 + 1)
            combo.currentIndexChanged.connect(self._update_current)
        layout.addLayout(selectors)

        region = QHBoxLayout()
        self.use_region = QCheckBox("Use cursor region")
        self.region_start, self.region_stop = self._spin(), self._spin()
        self.gap_enabled = QCheckBox("Split gaps larger than")
        self.gap = self._spin(minimum=1e-12, maximum=1e300, value=1.)
        self.floor = self._spin(minimum=-300., maximum=0., value=-60.)
        self.floor.setSuffix(" dB")
        for widget in (self.use_region, self.region_start, QLabel("to"), self.region_stop):
            region.addWidget(widget)
        region.addStretch()
        layout.addLayout(region)
        self.use_region.setToolTip("Drag across the Measurements plot, or enter bounds here and Analyze. "
                                  "Dataset analyses share a region in native coordinates; each plot trace keeps its own region.")
        self.advanced_section = CollapsibleSection("Advanced options", expanded=False)
        advanced = QGridLayout()
        advanced.addWidget(self.gap_enabled, 0, 0); advanced.addWidget(self.gap, 0, 1)
        self.floor_label = QLabel("Phase / ratio mask below peak")
        advanced.addWidget(self.floor_label, 1, 0); advanced.addWidget(self.floor, 1, 1)
        self.gap_enabled.setToolTip("Split analyses at gaps larger than this distance in the cut's coordinate units.")
        self.floor.setToolTip("Exclude weak returns from phase and channel-ratio analysis. Magnitude remains visible.")
        self.advanced_section.addLayout(advanced)
        self._advanced_layout = advanced
        layout.addWidget(self.advanced_section)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self._make_measure_panel()
        self._make_phase_panel()
        self._make_polarization_panel()
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        self.reference_combo.currentIndexChanged.connect(self._update_current)
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        self.tabs.currentChanged.connect(self._update_current)
        self.refresh_sources()

    @staticmethod
    def _spin(*, minimum=-1e300, maximum=1e300, value=0.):
        spin = CompactNumberSpin(); spin.setDecimals(12); spin.setRange(minimum, maximum)
        spin.setValue(value); spin.setKeyboardTracking(False)
        spin.setFixedWidth(125)
        return spin

    def _panel(self, key, title):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4); layout.setSpacing(4)
        controls = QHBoxLayout(); layout.addLayout(controls)
        purpose = QLabel(); purpose.setWordWrap(True)
        self.purpose_labels[key] = purpose; layout.addWidget(purpose)
        error = QLabel(); error.setWordWrap(True); error.setTextFormat(Qt.PlainText)
        error.setStyleSheet("QLabel { color: #fbbf24; padding: 6px; }")
        error.hide(); self.analysis_errors[key] = error; layout.addWidget(error)
        figure = Figure(figsize=(9, 4), constrained_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumHeight(240)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        chart = QWidget(); chart_layout = QVBoxLayout(chart); chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(4)
        toolbar = NavigationToolbar2QT(canvas, chart)
        chart_layout.addWidget(toolbar); chart_layout.addWidget(canvas)
        layout.addWidget(chart, 1)
        details = CollapsibleSection("Details — results and interpretation", expanded=False)
        table = ResultTable(); table.setMinimumHeight(150); table.setMaximumHeight(240)
        details.addWidget(table)
        note = QLabel(); note.setWordWrap(True); details.addWidget(note)
        self.detail_sections[key] = details; layout.addWidget(details)
        buttons = QHBoxLayout()
        run = QPushButton("Analyze"); run.clicked.connect(self._update_current)
        copy = QPushButton("Copy results"); copy.clicked.connect(table.copy)
        samples = QPushButton("Copy analysis samples"); samples.clicked.connect(lambda: self._copy_samples(key))
        export = QPushButton("Export results CSV…"); export.clicked.connect(lambda: self._export_results(key))
        for button in (run, copy, samples, export): buttons.addWidget(button)
        buttons.addStretch(); layout.addLayout(buttons)
        self.tabs.addTab(panel, title)
        self._panels[key] = (figure, canvas, table, note)
        self._chart_parts[key] = (chart, toolbar)
        return controls

    def _bound_chart_containers(self):
        # Scroll the surrounding controls rather than squeezing the plots.
        for key, (chart, toolbar) in self._chart_parts.items():
            toolbar.ensurePolished()
            height = self._panels[key][1].minimumHeight() + max(
                toolbar.sizeHint().height(), toolbar.minimumSizeHint().height()) + chart.layout().spacing()
            chart.setMinimumHeight(height)

    def showEvent(self, event):
        self._bound_chart_containers()
        super().showEvent(event)

    def _make_measure_panel(self):
        controls = self._panel("measure", "Measurements")
        self.trace_combo = QComboBox(); self.trace_combo.setMinimumWidth(250)
        controls.addWidget(QLabel("Trace")); controls.addWidget(self.trace_combo, 1)
        self.crossing_level = self._spin(value=-10.)
        self.peak_drop = self._spin(minimum=0., value=3.)
        self.crossing_label, self.peak_drop_label = QLabel("Crossing level"), QLabel("Peak drop for width")
        self._advanced_layout.addWidget(self.crossing_label, 2, 0)
        self._advanced_layout.addWidget(self.crossing_level, 2, 1)
        self._advanced_layout.addWidget(self.peak_drop_label, 2, 2)
        self._advanced_layout.addWidget(self.peak_drop, 2, 3)
        for title, mode in (("Peak", "peak"), ("Minimum", "minimum"), ("Next peak", "next")):
            button = QPushButton(title); controls.addWidget(button)
            button.clicked.connect(lambda _checked=False, action=mode: self._mark_measurement(action))
        self.trace_combo.currentIndexChanged.connect(self._reset_measurement)

    def _make_phase_panel(self):
        controls = self._panel("phase", "Phase diagnostics")
        self.phase_view = QComboBox()
        for name, code in (("Magnitude + wrapped / unwrapped phase", "phase"),
                           ("I/Q trajectory", "iq"), ("Phase residual A − B", "residual"),
                           ("Frequency phase-slope fit", "slope")):
            self.phase_view.addItem(name, code)
        self.phase_sign = QComboBox()
        self.phase_sign.addItem("Declared sign; +jwt if unspecified (assumption)", "auto")
        self.phase_sign.addItem("Interpret using exp(+jwt)", "+jwt")
        self.phase_sign.addItem("Interpret using exp(−jwt)", "-jwt")
        controls.addWidget(self.phase_view, 1); controls.addWidget(self.phase_sign)
        self.phase_view.currentIndexChanged.connect(self._update_current)

    def _make_polarization_panel(self):
        controls = self._panel("polarization", "Polarization diagnostics")
        self.pol_view = QComboBox()
        self.pol_view.addItem("HH / VV and co-/cross-channel ratios", "ratios")
        self.pol_view.addItem("HV / VH reciprocal-channel diagnostic", "reciprocity")
        self.pol_view.addItem("Selected channel phase difference", "phase")
        self.pol_a, self.pol_b = QComboBox(), QComboBox()
        controls.addWidget(self.pol_view, 1)
        self.pol_a_label, self.pol_b_label = QLabel("Channel A"), QLabel("Channel B")
        controls.addWidget(self.pol_a_label); controls.addWidget(self.pol_a)
        controls.addWidget(self.pol_b_label); controls.addWidget(self.pol_b)
        self.pol_view.currentIndexChanged.connect(self._update_current)

    def _update_presentation(self, key):
        phase_view, pol_view = self.phase_view.currentData(), self.pol_view.currentData()
        native_source = key != "measure" or self.trace_combo.currentData() is None
        for widget in (self.dataset_label, self.dataset_combo, self.axis_label, self.axis_combo):
            widget.setVisible(native_source)
        reference = key == "phase" and phase_view == "residual"
        self.reference_label.setVisible(reference); self.reference_combo.setVisible(reference)
        for name, combo in self.fixed.items():
            visible = native_source and name != self.axis_combo.currentData() and not (
                key == "polarization" and name == "polarization")
            combo.setVisible(visible); self.fixed_labels[name].setVisible(visible)
        source_pairs = [(self.dataset_label, self.dataset_combo), (self.axis_label, self.axis_combo),
                        (self.reference_label, self.reference_combo)]
        fixed_pairs = [(self.fixed_labels[name], combo) for name, combo in self.fixed.items()]
        for label, combo in source_pairs + fixed_pairs:
            self._source_selectors.removeWidget(label); self._source_selectors.removeWidget(combo)
        self._source_selectors.removeWidget(self.refresh_button)
        for row, pairs in enumerate((source_pairs, fixed_pairs)):
            visible_pairs = [(label, combo) for label, combo in pairs if not combo.isHidden()]
            for column, (label, combo) in enumerate(visible_pairs):
                self._source_selectors.addWidget(label, row, column * 2)
                self._source_selectors.addWidget(combo, row, column * 2 + 1)
            if row == 0:
                self._source_selectors.addWidget(self.refresh_button, row, len(visible_pairs) * 2, 1, 2)
        self.phase_sign.setVisible(key == "phase" and phase_view == "slope")
        for widget in (self.pol_a_label, self.pol_a, self.pol_b_label, self.pol_b):
            widget.setVisible(key == "polarization" and pol_view == "phase")
        self.floor_label.setVisible(key != "measure"); self.floor.setVisible(key != "measure")
        for widget in (self.crossing_label, self.crossing_level, self.peak_drop_label, self.peak_drop):
            widget.setVisible(key == "measure")
        if key == "measure":
            purpose = ("Locate peaks, minima, crossings and peak width in a native cut or a plotted trace. "
                       "Drag across the plot to measure a region.")
        elif key == "phase":
            purpose = {
                "phase": "Compare return strength (above) with wrapped and unwrapped phase (below). Unwrapping stays within each usable segment.",
                "iq": "Inspect amplitude changes and phase rotation in the complex return. Real and imaginary components form the trajectory; green points mark segment starts.",
                "residual": "Compare A and B phase at matching coordinates, without interpolation. Interpret differences using a common physical phase reference.",
                "slope": "Estimate apparent delay from phase trends. Frequency-segment fits are above; departures from each fit are below. Delay depends on the selected time sign.",
            }[phase_view]
        else:
            purpose = {
                "ratios": "Compare channel strengths (above) and co- and cross-channel power ratios (below). Channel names use the supplied polarization basis.",
                "reciprocity": "Compare HV and VH strengths (above) and their power ratio (below). Equality is expected only for reciprocal monostatic data with compatible conventions.",
                "phase": "Compare two channel strengths (above) and their phase difference (below). Interpret it using the supplied polarization basis and a shared phase reference.",
            }[pol_view]
        self.purpose_labels[key].setText(purpose)

    def refresh_sources(self, *_args):
        self._updating = True
        self._measurement = None
        if self._span is not None:
            self._span.disconnect_events(); self._span = None
        self._samples.clear()
        for error in self.analysis_errors.values():
            error.clear(); error.hide()
        for figure, canvas, table, note in self._panels.values():
            figure.clear(); table.populate([])
            note.setText("Sources refreshed. Select this analysis tab to recompute.")
            canvas.draw_idle()
        getter = getattr(self.owner, "_selected_datasets", None)
        self._datasets = list(getter() if callable(getter) else [])
        if not self._datasets and getattr(self.owner, "active_dataset", None) is not None:
            self._datasets = [("Active dataset", self.owner.active_dataset)]
        for combo in (self.dataset_combo, self.reference_combo):
            combo.clear()
            for name, _dataset in self._datasets: combo.addItem(str(name))
        if len(self._datasets) > 1: self.reference_combo.setCurrentIndex(1)
        self.trace_combo.clear(); self.trace_combo.addItem("Selected native dataset cut", None)
        marker_lines = getattr(self.owner, "_marker_lines", None)
        lines = marker_lines() if callable(marker_lines) else []
        for index, line in enumerate(lines):
            label = line.get_label()
            if getattr(line, "_grim_native_probe_unavailable", False): label += " [display samples only]"
            self.trace_combo.addItem(f"Plot trace {index + 1}: {label}", line)
        source_tab = str(getattr(self.owner, "_active_plot_tab", "plotting"))
        self.source_note.setText(f"From {source_tab}: {len(self._datasets)} dataset(s), {len(lines)} plotted trace(s).")
        self.source_note.setToolTip("Sources are a snapshot. Refresh from selection after changing the main tab, data, or parameter selection.")
        self._updating = False
        self._dataset_changed()
        if not self._datasets and lines: self.trace_combo.setCurrentIndex(1)
        self._update_current()

    def _dataset_changed(self, *_args):
        if self._updating: return
        self._updating = True
        index = self.dataset_combo.currentIndex()
        dataset = self._datasets[index][1] if 0 <= index < len(self._datasets) else None
        for name, combo in self.fixed.items():
            combo.clear()
            if dataset is None: continue
            values = dataset.get_axis(name)
            for i, value in enumerate(values):
                text = str(value) if name == "polarization" else f"{float(value):.12g} {axis_unit(dataset, name)}"
                combo.addItem(text, i)
            # Reuse selected values only when they refer to this exact dataset.
            if dataset is getattr(self.owner, "active_dataset", None):
                widget = getattr(self.owner, {"azimuth": "list_az", "elevation": "list_elev", "frequency": "list_freq", "polarization": "list_pol"}[name], None)
                if widget is not None:
                    selected = sorted(getattr(self.owner, "_selected_indices", lambda _w: set())(widget))
                    if selected and selected[0] < combo.count(): combo.setCurrentIndex(selected[0])
        for combo in (self.pol_a, self.pol_b):
            combo.clear()
            if dataset is not None: combo.addItems([str(value) for value in dataset.polarizations])
        if self.pol_a.findText("HH") >= 0: self.pol_a.setCurrentText("HH")
        if self.pol_b.findText("VV") >= 0: self.pol_b.setCurrentText("VV")
        elif self.pol_b.count() > 1: self.pol_b.setCurrentIndex(1)
        self._updating = False
        self._axis_changed()

    def _axis_changed(self, *_args):
        if self._updating: return
        axis = str(self.axis_combo.currentData())
        for name, combo in self.fixed.items(): combo.setEnabled(name != axis)
        self._reset_measurement()

    def _cut(self, *, reference=False, polarization=None, dataset_index=None):
        index = (self.reference_combo.currentIndex() if reference else self.dataset_combo.currentIndex()) if dataset_index is None else dataset_index
        if not 0 <= index < len(self._datasets): raise ValueError("Select a dataset and refresh the analysis sources.")
        name, dataset = self._datasets[index]
        axis = str(self.axis_combo.currentData())
        fixed = {key + "_index": int(self.fixed[key].currentData() or 0) for key in ("azimuth", "elevation", "frequency")}
        source_index = self.dataset_combo.currentIndex()
        if (reference or index != source_index) and 0 <= source_index < len(self._datasets):
            source = self._datasets[source_index][1]
            for key in ("azimuth", "elevation", "frequency"):
                if key == axis: continue
                value = source.get_axis(key)[fixed[key + "_index"]]
                converted = float(convert_axis_values([value], key, axis_unit(source, key), axis_unit(dataset, key))[0])
                values = np.asarray(dataset.get_axis(key), dtype=float)
                valid = np.flatnonzero(np.isfinite(values))
                if not np.isfinite(converted) or not len(valid):
                    raise ValueError(f"Reference comparison requires a finite {key} coordinate.")
                nearest = int(valid[np.argmin(np.abs(values[valid] - converted))])
                tolerance = np.finfo(float).eps * max(abs(converted), float(np.max(np.abs(values[valid]))), np.finfo(float).tiny) * 32
                if abs(values[nearest] - converted) > tolerance:
                    raise ValueError(f"{name} has no matching {key}={converted:.9g} {axis_unit(dataset, key)} cut.")
                fixed[key + "_index"] = nearest
        pol = polarization or self.fixed["polarization"].currentText()
        return dataset_cut(dataset, axis, polarization=pol, label=name, **fixed)

    def _region(self):
        return (self.region_start.value(), self.region_stop.value()) if self.use_region.isChecked() else None

    def _maximum_gap(self):
        return self.gap.value() if self.gap_enabled.isChecked() else None

    def _reset_measurement(self, *_args):
        if self._updating: return
        self._measurement = None; self._peak_position = -1
        self._update_current()

    def _sync_region_context(self, key):
        line = self.trace_combo.currentData() if key == "measure" else None
        if line is not None and line.axes is not None:
            context = ("trace", id(line))
            if context == self._region_context: return
            reader = getattr(self.owner, "_marker_arrays", None)
            x = np.asarray(reader(line)[0] if callable(reader) else line.get_xdata(), dtype=float)
            if getattr(line.axes, "name", "") == "polar": x = np.rad2deg(x)
            unit = "(trace x)"
        else:
            index = self.dataset_combo.currentIndex()
            if not 0 <= index < len(self._datasets): return
            dataset = self._datasets[index][1]; axis = str(self.axis_combo.currentData())
            context = ("dataset", id(dataset), axis)
            if context == self._region_context: return
            x = np.asarray(dataset.get_axis(axis), dtype=float)
            unit = axis_unit(dataset, axis)
        if self._region_context is not None:
            self._regions[self._region_context] = (self.region_start.value(), self.region_stop.value(),
                self.use_region.isChecked(), self.gap.value(), self.gap_enabled.isChecked())
        self._region_context = context
        x = np.sort(x[np.isfinite(x)])
        if not len(x): return
        steps = np.diff(x); steps = steps[steps > 0]
        lo, hi, enabled, gap, gap_enabled = self._regions.get(context,
            (float(x[0]), float(x[-1]), False, float(np.median(steps)) * 2 if len(steps) else 1., False))
        self.region_start.setValue(lo); self.region_stop.setValue(hi); self.use_region.setChecked(enabled)
        self.gap.setValue(gap); self.gap_enabled.setChecked(gap_enabled)
        for spin in (self.region_start, self.region_stop, self.gap): spin.setSuffix(f" {unit}")

    def _update_current(self, *_args):
        if self._updating or not hasattr(self, "pol_view"): return
        key = tuple(self.TAB_NAMES)[self.tabs.currentIndex()]
        self._update_presentation(key)
        self.analysis_errors[key].clear(); self.analysis_errors[key].hide()
        try:
            self._sync_region_context(key)
            getattr(self, "_analyze_" + key)()
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            self._show_analysis_error(key, exc)

    def _show_analysis_error(self, key, error):
        figure, canvas, table, note = self._panels[key]
        figure.clear(); table.populate([]); self._samples[key] = []
        if key == "measure":
            self._measurement = None
            if self._span is not None: self._span.disconnect_events(); self._span = None
        note.setText(str(error))
        self.analysis_errors[key].setText(f"Analysis unavailable: {error}")
        self.analysis_errors[key].show()
        canvas.draw_idle()

    def _display_line(self, axes, x, y, *, label=None, split_gaps=True, runs=None, **kwargs):
        # Display-only reduction; all analysis and copied values stay native.
        x, y = np.asarray(x), np.asarray(y)
        if runs is None and split_gaps and self._maximum_gap() is not None:
            runs = list(finite_runs(x, np.isfinite(y), maximum_gap=self._maximum_gap()))
        x, y = _with_run_breaks(x, y, runs)
        xx, yy, _reduced = decimate_line(x, y, max_points=20_000)
        if "marker" not in kwargs:
            kwargs.update(singleton_line_markers(xx, yy))
        return axes.plot(xx, yy, label=label, **kwargs)[0]

    def _analyze_measure(self):
        line = self.trace_combo.currentData()
        if line is None:
            cut = self._cut()
            x, y, label, x_label, y_label = cut.x, cut.db, cut.label, f"{cut.axis} ({cut.x_unit})", cut.level_unit
            note_text = "Native dataset cut. Mean / standard deviation below describe displayed dB values, not mean linear RCS."
        else:
            if line.axes is None: raise ValueError("The selected plot trace was replaced. Refresh the sources.")
            current_figure = getattr(self.owner, "plot_figure", None)
            if current_figure is not None and (line.axes not in current_figure.axes or line not in line.axes.lines):
                raise ValueError("The selected plot trace was replaced. Refresh the sources.")
            native = getattr(line, "_grim_native_samples", None)
            if native is not None and native.values is None:
                raise ValueError("The native trace was released. Refresh the sources.")
            reader = getattr(self.owner, "_marker_arrays", None)
            values = reader(line) if callable(reader) else (line.get_xdata(), line.get_ydata())
            x, y = np.array(values[0], dtype=float, copy=True), np.array(values[1], dtype=float, copy=True)
            if getattr(line.axes, "name", "") == "polar": x = np.rad2deg(x)
            label, x_label, y_label = line.get_label(), line.axes.get_xlabel(), line.axes.get_ylabel()
            if getattr(line.axes, "name", "") == "polar": x_label = "Azimuth (deg)"
            note_text = "Current trace snapshot; statistics and peak drop use the displayed ordinate unit."
            if getattr(line, "_grim_native_probe_unavailable", False):
                note_text += " Native samples unavailable: these results describe only the displayed samples."
            else: note_text += " Measurements use native trace samples, including those omitted from the overview."
        order = _trace_sort_order(x)
        if order is not None:
            x, y = x[order], y[order]
        result = measure_trace(x, y, region=self._region(), level=self.crossing_level.value(),
                               drop=self.peak_drop.value(), maximum_gap=self._maximum_gap())
        self._measurement = (x, y, result, line, order)
        figure, canvas, table, note = self._panels["measure"]
        if self._span is not None: self._span.disconnect_events()
        figure.clear(); ax = figure.add_subplot(111)
        self._display_line(ax, x, y, label=label)
        ax.axhline(self.crossing_level.value(), color="0.5", linestyle=":", label="Crossing level")
        for crossing in result["crossings"]: ax.axvline(crossing, color="#2a9d8f", alpha=.5)
        if self._region() is not None: ax.axvspan(*sorted(self._region()), alpha=.10, color="#2a9d8f")
        ax.set(xlabel=x_label, ylabel=y_label, title=label); ax.grid(True, alpha=.25)
        self._span = SpanSelector(ax, self._select_region, "horizontal", useblit=False,
                                  props={"alpha": .18, "facecolor": "#2a9d8f"}, interactive=True)
        rows = [("Source", label, f"{x_label}; {y_label}"),
                ("Region", str(self._region()) if self._region() else "Full trace", x_label),
                ("Finite / selected samples", f"{result['finite_count']} / {result['selected_count']}", "native coordinates"),
                ("Coverage", _number(result["coverage"] * 100), "%")]
        for name in ("peak", "minimum"):
            if name in result:
                rows.append((name.capitalize(), _number(result[name]["y"]), f"{y_label} at {_number(result[name]['x'])} {x_label}"))
        for name in ("mean", "median", "std", "peak_to_peak"):
            if name in result: rows.append((name.replace("_", " ").capitalize(), _number(result[name]), f"displayed {y_label}"))
        for index, crossing in enumerate(result["crossings"], 1): rows.append((f"Level crossing {index}", _number(crossing), x_label))
        if result["bandwidth"]:
            width = result["bandwidth"]
            rows.extend((("Peak-lobe width", _number(width["width"]), f"{x_label}; drop {self.peak_drop.value():g} {y_label}"),
                         ("Left / right crossing", f"{width['left']:.9g} / {width['right']:.9g}", x_label)))
        else: rows.append(("Peak-lobe width", "not defined", result.get("bandwidth_note", result.get("note", ""))))
        table.populate(rows); note.setText(note_text + " Crossings never bridge missing samples; interpolation is linear between adjacent samples.")
        self._samples["measure"] = [(label, x, y, y_label)]
        canvas.draw_idle()

    def _select_region(self, left, right):
        if left == right: return
        self.region_start.setValue(left); self.region_stop.setValue(right); self.use_region.setChecked(True)
        self._update_current()

    def _mark_measurement(self, mode):
        self.analysis_errors["measure"].clear(); self.analysis_errors["measure"].hide()
        try:
            self._analyze_measure()
            x, y, result, line, order = self._measurement
            if mode == "next":
                peaks = sorted(result["peaks"], key=lambda entry: entry["x"])
                if not peaks: raise ValueError("No interior local peaks in the selected finite region.")
                self._peak_position = (self._peak_position + 1) % len(peaks)
                point = peaks[self._peak_position]
            else: point = result[mode]
            figure, canvas, _table, note = self._panels["measure"]
            ax = figure.axes[0]; ax.plot([point["x"]], [point["y"]], "D", color="#e76f51")
            ax.annotate(f"{mode}: ({point['x']:.9g}, {point['y']:.9g})", (point["x"], point["y"]), xytext=(8, 8), textcoords="offset points")
            adder = getattr(self.owner, "_add_plot_marker", None)
            if line is not None and line.axes is not None and callable(adder):
                adder(line, int(order[point["index"]]) if order is not None else point["index"])
            note.setText(note.text() + f" Selected {mode}: x={point['x']:.9g}, value={point['y']:.9g}.")
            canvas.draw_idle()
        except (ValueError, KeyError) as exc:
            self._panels["measure"][3].setText(str(exc))
            self.analysis_errors["measure"].setText(f"Marker unavailable: {exc}")
            self.analysis_errors["measure"].show()

    def _copy_samples(self, key):
        rows = self._samples.get(key, [])
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("Series", "X", "Value", "Unit"))
        writer.writerows((label, _number(x), _number(y), unit)
                         for label, xs, ys, unit in rows for x, y in zip(xs, ys))
        QApplication.clipboard().setText(output.getvalue())

    def _export_results(self, key):
        path, _filter = QFileDialog.getSaveFileName(self, "Export engineering results", f"{key}-results.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output); writer.writerow(("Metric", "Value", "Unit / context"))
            writer.writerows(self._panels[key][2].rows)

    def _dispose(self):
        self.owner._engineering_analysis_last_tab = self.tabs.currentIndex()
        if self._span is not None: self._span.disconnect_events(); self._span = None
        self._measurement = None; self._samples.clear(); self._datasets.clear()
        self._regions.clear()
        if getattr(self.owner, "_engineering_analysis_dialog", None) is self:
            self.owner._engineering_analysis_dialog = None

    def done(self, result):
        self._dispose()
        super().done(result)

    def closeEvent(self, event):
        self._dispose()
        super().closeEvent(event)


def show_engineering_analysis(owner, tab=None):
    dialog = getattr(owner, "_engineering_analysis_dialog", None)
    if dialog is None:
        dialog = EngineeringAnalysisDialog(owner)
        owner._engineering_analysis_dialog = dialog
        if tab is None:
            dialog.tabs.setCurrentIndex(getattr(owner, "_engineering_analysis_last_tab", 0))
    else:
        dialog.refresh_sources()
    if tab is not None:
        dialog.tabs.setCurrentIndex(dialog.TAB_NAMES.get(tab, 0))
    dialog._update_current()
    dialog.show(); dialog.raise_(); dialog.activateWindow()
    return dialog
