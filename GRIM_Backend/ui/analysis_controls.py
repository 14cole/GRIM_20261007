"""Plotting-tab analysis settings and the parameter slider bar."""

from __future__ import annotations

import re
import math

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from GRIM_Backend.plotting.modes import common


def _percent_spin(value: float, tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setDecimals(1)
    spin.setSingleStep(5.0)
    spin.setSuffix(" %")
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setToolTip(tooltip)
    return spin


class PlotAnalysisControls(QObject):
    """PBP band, CDF, sector-statistics, and range-frequency settings.

    ``changed`` carries which feature changed ("pbp", "cdf", "sector",
    "range", "sweep") so only the matching plot type re-renders.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_pbp_band = QComboBox()
        self.combo_pbp_band.addItem("Min–Max", "minmax")
        self.combo_pbp_band.addItem("Percentiles", "percentile")
        self.combo_pbp_band.setToolTip(
            "Each dataset and polarization gets its own band. Min–Max bounds its "
            "selected series; Percentiles bounds the chosen "
            "lower and upper percentiles at each point, ignoring outliers."
        )
        self.spin_pbp_low = _percent_spin(10.0, "Lower percentile of the PBP band.")
        self.spin_pbp_high = _percent_spin(90.0, "Upper percentile of the PBP band.")

        self.combo_cdf = QComboBox()
        self.combo_cdf.addItem("Cumulative: samples at or below level", "cdf")
        self.combo_cdf.addItem("Exceedance: samples at or above level", "exceedance")

        self.combo_sweep_aggregation = QComboBox()
        for text, key in (("Individual azimuth cuts", "individual"),
                          ("Mean linear power", "mean"), ("Median (P50)", "median"),
                          ("Percentile of linear power", "percentile"), ("Worst case (maximum)", "max")):
            self.combo_sweep_aggregation.addItem(text, key)
        self.combo_sweep_aggregation.setCurrentIndex(self.combo_sweep_aggregation.findData("median"))
        self.combo_sweep_aggregation.setToolTip(
            "Frequency and Elevation Sweep reduce the selected azimuths at each point. "
            "Mean/median/percentile/max use valid linear power, including zero, before dB conversion. "
            "Individual cuts keep every azimuth separate. Legend n is valid/selected azimuth samples per point (a range when coverage varies). These are sample statistics, not coherent-field averages. "
            "In Phase, mean/median are circular; percentile/max show the phase of the nearest-rank/strongest power observation."
        )
        self.spin_sweep_percentile = _percent_spin(90., "Percentile across the selected azimuth samples, in linear power.")
        self.combo_sweep_aggregation.currentIndexChanged.connect(self._sweep_changed)
        self.spin_sweep_percentile.valueChanged.connect(lambda: self.changed.emit("sweep"))
        self._sweep_changed(emit=False)

        self.btn_cdf_settings = QPushButton("Population / threshold…")
        self._build_cdf_dialog(parent)
        self.btn_cdf_settings.clicked.connect(self._show_cdf_dialog)

        self._sector_text = "30"
        self._sector_statistic = "mean"
        self._sector_percentile = 90.0
        self._sector_unit = "deg"

        self.spin_range_subband = QDoubleSpinBox()
        self.spin_range_subband.setRange(5.0, 100.0)
        self.spin_range_subband.setDecimals(0)
        self.spin_range_subband.setSingleStep(5.0)
        self.spin_range_subband.setSuffix(" % of band")
        self.spin_range_subband.setValue(25.0)
        self.spin_range_subband.setKeyboardTracking(False)
        self.spin_range_subband.setToolTip(
            "Width of each sliding window within the frequencies you select. "
            "The plot shows each window's center frequency. Narrower windows "
            "show more frequency detail but coarser down-range resolution. "
            "100% uses the entire selection for one range profile."
        )
        self.combo_range_window = QComboBox()
        self.combo_range_window.addItems(
            ["Hanning", "Hamming", "Blackman", "Blackman-Harris", "Kaiser β=15", "Rectangular"]
        )
        self.combo_range_window.setToolTip("Window applied to each sub-band before the inverse FFT.")
        self.combo_range_unit = QComboBox()
        self.combo_range_unit.addItems(["m", "cm", "mm", "in", "ft"])
        self.combo_range_unit.setCurrentText("in")

        self.combo_pbp_band.currentIndexChanged.connect(self._pbp_band_changed)
        self.spin_range_subband.valueChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_window.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_unit.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.spin_pbp_low.valueChanged.connect(self._pbp_low_changed)
        self.spin_pbp_high.valueChanged.connect(self._pbp_high_changed)
        self.combo_cdf.currentIndexChanged.connect(self._cdf_options_changed)
        self._pbp_band_changed(emit=False)

    def add_rows(self, grid, row: int) -> int:
        """Add labeled rows to the Plot Settings grid; returns the next row."""
        rows = (
            (("Sweep statistic", self.combo_sweep_aggregation), ("Sweep percentile", self.spin_sweep_percentile)),
            (("PbP Band", self.combo_pbp_band), ("Lower", self.spin_pbp_low),
             ("Upper", self.spin_pbp_high)),
            (("CDF", self.combo_cdf), ("Details", self.btn_cdf_settings)),
            (("Range–Freq Sub-band", self.spin_range_subband),
             ("Window", self.combo_range_window), ("Range Unit", self.combo_range_unit)),
        )
        for cells in rows:
            for column, (label, widget) in enumerate(cells):
                grid.addWidget(QLabel(label), row, 2 * column)
                span = 5 if len(cells) == 1 else 1
                grid.addWidget(widget, row, 2 * column + 1, 1, span)
            row += 1
        return row

    # --- values ------------------------------------------------------------

    def pbp_percentiles(self) -> tuple[float, float] | None:
        if self.combo_pbp_band.currentData() != "percentile":
            return None
        return float(self.spin_pbp_low.value()), float(self.spin_pbp_high.value())

    def set_pbp_band(self, mode: str) -> None:
        index = self.combo_pbp_band.findData(mode)
        if index >= 0:
            self.combo_pbp_band.setCurrentIndex(index)

    def cdf_exceedance(self) -> bool:
        return self.combo_cdf.currentData() == "exceedance"

    def _sweep_changed(self, *_args, emit=True):
        self.spin_sweep_percentile.setEnabled(self.combo_sweep_aggregation.currentData() == "percentile")
        if emit:
            self.changed.emit("sweep")

    def sweep_aggregation(self):
        return {"mode": self.combo_sweep_aggregation.currentData(),
                "percentile": float(self.spin_sweep_percentile.value())}

    def _build_cdf_dialog(self, parent):
        self.cdf_dialog = QDialog(parent if isinstance(parent, QWidget) else None)
        self.cdf_dialog.setWindowTitle("CDF population and threshold lookup")
        self.cdf_dialog.resize(840, 580)
        layout = QVBoxLayout(self.cdf_dialog)
        definition = QLabel(
            "Each dataset/run and polarization remains a separate population. Bands and sectors subdivide "
            "the selected coordinates; no runs or unlike units are pooled. Valid zero power remains included; "
            "missing, nonfinite and negative power is excluded.\n\n"
            "Equal samples gives every observation one vote. Quadrature weights approximate uniform measure "
            "over each group's selected coordinate hull (trapezoidal endpoint weights). This can bridge "
            "unsampled gaps; it is not a dwell-time, solid-angle or independent-trial probability. "
            "Singleton axes have unit weight. Coverage reports valid weight / selected weight."
        )
        definition.setWordWrap(True)
        layout.addWidget(definition)
        form = QFormLayout()
        self.combo_cdf_grouping = QComboBox()
        for text, key in (("Per run: all selected coordinates", "run"),
                          ("Per run and frequency band", "band"),
                          ("Per run and azimuth sector", "sector"),
                          ("Per run, frequency band and sector", "band_sector")):
            self.combo_cdf_grouping.addItem(text, key)
        form.addRow("Groups", self.combo_cdf_grouping)
        self.spin_cdf_bands = QSpinBox()
        self.spin_cdf_bands.setRange(1, 1000)
        self.spin_cdf_bands.setValue(4)
        self.spin_cdf_bands.setToolTip("Equal-width frequency bands across the selected reference-frequency extent; final band includes its upper edge.")
        form.addRow("Frequency bands", self.spin_cdf_bands)
        self.spin_cdf_sector_width = QDoubleSpinBox()
        self.spin_cdf_sector_width.setRange(.1, 360.)
        self.spin_cdf_sector_width.setDecimals(3)
        self.spin_cdf_sector_width.setValue(30.)
        self.spin_cdf_sector_width.setSuffix(" deg")
        self.spin_cdf_sector_width.setToolTip("Sector width measured from the lowest selected reference azimuth. Angular units are converted explicitly.")
        form.addRow("Azimuth sector width", self.spin_cdf_sector_width)
        self.combo_cdf_weighting = QComboBox()
        for text, key in (("Equal samples", "samples"), ("Uniform azimuth measure", "azimuth"),
                          ("Uniform frequency measure", "frequency"),
                          ("Uniform azimuth × frequency measure", "azimuth_frequency"),
                          ("Uniform azimuth × elevation × frequency measure", "all_axes")):
            self.combo_cdf_weighting.addItem(text, key)
        form.addRow("Weighting", self.combo_cdf_weighting)
        self.chk_cdf_threshold = QCheckBox("Show threshold and lookup results")
        form.addRow(self.chk_cdf_threshold)
        self.spin_cdf_threshold = QDoubleSpinBox()
        self.spin_cdf_threshold.setRange(-1.e12, 1.e12)
        self.spin_cdf_threshold.setDecimals(9)
        self.spin_cdf_threshold.setKeyboardTracking(False)
        self.spin_cdf_threshold.setToolTip("Numeric threshold in each curve's displayed unit. The table identifies that unit; mixed dB conventions are not physically interchangeable.")
        form.addRow("Threshold (current display unit)", self.spin_cdf_threshold)
        layout.addLayout(form)
        self.cdf_results = QPlainTextEdit()
        self.cdf_results.setReadOnly(True)
        self.cdf_results.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.cdf_results.setPlaceholderText("Plot a CDF to inspect population coverage and inclusive threshold probabilities.")
        layout.addWidget(self.cdf_results, 1)
        copy = QPushButton("Copy lookup table")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.cdf_results.toPlainText()))
        layout.addWidget(copy)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.cdf_dialog.hide)
        layout.addWidget(buttons)
        for widget in (self.combo_cdf_grouping, self.combo_cdf_weighting):
            widget.currentIndexChanged.connect(self._cdf_options_changed)
        for widget in (self.spin_cdf_bands, self.spin_cdf_sector_width, self.spin_cdf_threshold):
            widget.valueChanged.connect(self._cdf_options_changed)
        self.chk_cdf_threshold.toggled.connect(self._cdf_options_changed)
        self._cdf_options_changed(emit=False)

    def _show_cdf_dialog(self):
        self.cdf_dialog.show()
        self.cdf_dialog.raise_()

    def _cdf_options_changed(self, *_args, emit=True):
        self.cdf_results.clear()
        grouping = self.combo_cdf_grouping.currentData()
        self.spin_cdf_bands.setEnabled(grouping in ("band", "band_sector"))
        self.spin_cdf_sector_width.setEnabled(grouping in ("sector", "band_sector"))
        self.spin_cdf_threshold.setEnabled(self.chk_cdf_threshold.isChecked())
        if emit:
            self.changed.emit("cdf")

    def cdf_options(self):
        return {"exceedance": self.cdf_exceedance(), "grouping": self.combo_cdf_grouping.currentData(),
                "band_count": self.spin_cdf_bands.value(), "sector_width_deg": self.spin_cdf_sector_width.value(),
                "weighting": self.combo_cdf_weighting.currentData(),
                "threshold_enabled": self.chk_cdf_threshold.isChecked(), "threshold": self.spin_cdf_threshold.value()}

    def set_cdf_results(self, text):
        self.cdf_results.setPlainText(str(text))

    def analysis_options(self):
        return {"sweep": self.sweep_aggregation(), "cdf": self.cdf_options()}

    def restore_analysis_options(self, options):
        options = options if isinstance(options, dict) else {}
        sweep, cdf = options.get("sweep", {}), options.get("cdf", {})
        sweep = sweep if isinstance(sweep, dict) else {}
        cdf = cdf if isinstance(cdf, dict) else {}
        def number(mapping, key, default):
            try:
                value = float(mapping.get(key, default))
                return value if math.isfinite(value) else default
            except (ValueError, TypeError, OverflowError):
                return default
        blocked = self.blockSignals(True)
        try:
            for widget, value in ((self.combo_sweep_aggregation, sweep.get("mode", "median")),
                                  (self.combo_cdf, "exceedance" if cdf.get("exceedance", False) else "cdf"),
                                  (self.combo_cdf_grouping, cdf.get("grouping", "run")),
                                  (self.combo_cdf_weighting, cdf.get("weighting", "samples"))):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            self.spin_sweep_percentile.setValue(number(sweep, "percentile", 90.))
            self.spin_cdf_bands.setValue(int(number(cdf, "band_count", 4)))
            self.spin_cdf_sector_width.setValue(number(cdf, "sector_width_deg", 30.))
            self.chk_cdf_threshold.setChecked(bool(cdf.get("threshold_enabled", False)))
            self.spin_cdf_threshold.setValue(number(cdf, "threshold", 0.))
        finally:
            self.blockSignals(blocked)

    def sector_text(self, unit: str | None = None) -> str:
        """Return the definition in the requested azimuth units."""
        if unit is None or unit == self._sector_unit:
            return self._sector_text
        parts = re.split(r"([,:;])", self._sector_text)
        for index in range(0, len(parts), 2):
            if parts[index].strip():
                value = common.convert_axis_values(
                    [float(parts[index])], "azimuth", self._sector_unit, unit
                )[0]
                parts[index] = f"{value:.17g}"
        return "".join(parts)

    def sector_statistic(self) -> str:
        return self._sector_statistic

    def sector_percentile(self) -> float:
        return self._sector_percentile

    def set_sector_settings(
        self, text: str, statistic: str, percentile: float, *, unit: str | None = None
    ) -> None:
        """Commit the editor's settings together, with a single plot refresh."""
        values = (
            str(text).strip(), str(statistic), float(percentile), unit or self._sector_unit
        )
        current = (
            self._sector_text, self._sector_statistic, self._sector_percentile,
            self._sector_unit,
        )
        if values != current:
            (self._sector_text, self._sector_statistic,
             self._sector_percentile, self._sector_unit) = values
            self.changed.emit("sector")

    def range_subband_percent(self) -> float:
        return float(self.spin_range_subband.value())

    def range_window(self) -> str:
        return self.combo_range_window.currentText()

    def range_unit(self) -> str:
        return self.combo_range_unit.currentText()

    # --- keep percentile pairs valid -----------------------------------------

    def _pbp_band_changed(self, *_args, emit: bool = True) -> None:
        enabled = self.combo_pbp_band.currentData() == "percentile"
        self.spin_pbp_low.setEnabled(enabled)
        self.spin_pbp_high.setEnabled(enabled)
        if emit:
            self.changed.emit("pbp")

    def _pbp_low_changed(self, value: float) -> None:
        if value >= self.spin_pbp_high.value():
            blocked = self.spin_pbp_high.blockSignals(True)
            self.spin_pbp_high.setValue(min(100.0, value + 1.0))
            self.spin_pbp_high.blockSignals(blocked)
            if value >= self.spin_pbp_high.value():
                self.spin_pbp_low.setValue(self.spin_pbp_high.value() - 1.0)
                return
        self.changed.emit("pbp")

    def _pbp_high_changed(self, value: float) -> None:
        if value <= self.spin_pbp_low.value():
            blocked = self.spin_pbp_low.blockSignals(True)
            self.spin_pbp_low.setValue(max(0.0, value - 1.0))
            self.spin_pbp_low.blockSignals(blocked)
            if value <= self.spin_pbp_low.value():
                self.spin_pbp_high.setValue(self.spin_pbp_low.value() + 1.0)
                return
        self.changed.emit("pbp")


class PlotSliderBar(QFrame):
    """Axis choice, slider, step buttons, and Play for scrubbing one axis."""

    moved = Signal(int)
    axis_changed = Signal(str)
    shown = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plotSliderBar")
        self.combo_axis = QComboBox()
        for label, axis in (("Frequency", "frequency"), ("Elevation", "elevation"),
                            ("Azimuth", "azimuth")):
            self.combo_axis.addItem(label, axis)
        self.combo_axis.setToolTip("Parameter list the slider steps through.")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setPageStep(1)
        self.btn_prev = QToolButton(text="◀")
        self.btn_prev.setToolTip("Previous value")
        self.btn_next = QToolButton(text="▶")
        self.btn_next.setToolTip("Next value")
        self.value_label = QLabel("--")
        self.value_label.setMinimumWidth(90)
        self.btn_play = QToolButton(text="Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Step through every value, wrapping at the end.")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Slider"))
        layout.addWidget(self.combo_axis)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.value_label)
        layout.addWidget(self.btn_play)

        self._labels: list[str] = []
        # Coalesce drags and key repeats into one render per settled value.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(lambda: self.moved.emit(self.slider.value()))
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(350)
        self._play_timer.timeout.connect(lambda: self.step(1, wrap=True))
        self.slider.valueChanged.connect(self._value_changed)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_play.toggled.connect(self._play_toggled)
        self.combo_axis.currentIndexChanged.connect(
            lambda: self.axis_changed.emit(self.axis())
        )
        self.hide()

    def axis(self) -> str:
        return str(self.combo_axis.currentData())

    def set_positions(self, labels: list[str], current: int) -> None:
        """Replace the slider positions without emitting a move."""
        self._labels = list(labels)
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(self._labels) - 1))
        self.slider.setValue(max(0, min(int(current), len(self._labels) - 1)))
        self.slider.blockSignals(blocked)
        self.slider.setEnabled(len(self._labels) > 1)
        self._show_label(self.slider.value())

    def step(self, delta: int, *, wrap: bool = False) -> None:
        count = self.slider.maximum() + 1
        if count <= 1:
            return
        value = self.slider.value() + int(delta)
        value = value % count if wrap else max(0, min(value, count - 1))
        self.slider.setValue(value)

    def stop_play(self) -> None:
        self.btn_play.setChecked(False)

    def stop_interaction(self) -> None:
        """End playback and any delayed scrub when its canvas is hidden."""
        self.stop_play()
        self._debounce.stop()

    def _value_changed(self, value: int) -> None:
        self._show_label(value)
        self._debounce.start()

    def _show_label(self, value: int) -> None:
        self.value_label.setText(
            self._labels[value] if 0 <= value < len(self._labels) else "--"
        )

    def _play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("Pause" if checked else "Play")
        if checked:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Lists can change while the bar is hidden (another tab, another row).
        self.shown.emit()

    def hideEvent(self, event) -> None:
        self.stop_play()
        super().hideEvent(event)
