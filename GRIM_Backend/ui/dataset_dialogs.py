"""Dataset operation dialogs and their input-unit presentation helpers."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QRadioButton, QSpinBox, QVBoxLayout
from GRIM_Backend.datasets.grid import RcsGrid


_FREQUENCY_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}


def _canonical_angle_unit(value: object, *, default: str = "deg") -> str:
    text = str(value or default).strip().lower()
    aliases = {
        "degree": "deg",
        "degrees": "deg",
        "deg": "deg",
        "radian": "rad",
        "radians": "rad",
        "rad": "rad",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            f"unsupported angular unit {value!r}; use deg or rad"
        ) from exc

def _angle_axis_degrees(dataset: "RcsGrid", axis_name: str) -> np.ndarray:
    values = np.asarray(dataset.get_axis(axis_name), dtype=float)
    unit = _canonical_angle_unit((dataset.units or {}).get(axis_name, "deg"))
    return np.rad2deg(values) if unit == "rad" else values

def _canonical_frequency_unit(value: object, *, default: str = "GHz") -> str:
    text = str(value or default).strip().lower()
    if text not in _FREQUENCY_TO_HZ:
        raise ValueError(
            f"unsupported frequency unit {value!r}; use Hz, kHz, MHz, or GHz"
        )
    return {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}[text]

_COHERENT_METADATA_LABELS = {
    "phase_reference": "phase reference / phase center",
    "time_convention": "phasor time convention",
    "polarization_basis": "polarization basis",
}

def _missing_coherent_metadata_keys(datasets) -> set[str]:
    """Return coherent declarations absent from any selected input."""

    missing: set[str] = set()
    for dataset in datasets:
        getter = getattr(dataset, "_declared_scalar_metadata", None)
        for key in _COHERENT_METADATA_LABELS:
            if callable(getter):
                value = getter(key)
            else:
                value = (dataset.extra or {}).get(
                    key, (dataset.units or {}).get(key, "")
                )
            if not str(value or "").strip():
                missing.add(key)
    return missing

class AlignDialog(QDialog):
    """Choose alignment mode when aligning datasets to a reference."""

    def __init__(self, ref_name: str, n_others: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align Datasets")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Reference: <b>{ref_name}</b>  —  aligning {n_others} other dataset(s) to it."
        ))

        grp = QGroupBox("Alignment Mode")
        grp_layout = QVBoxLayout(grp)
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in both datasets (exact match, no interpolation)"
        )
        self._radio_interp = QRadioButton(
            "Interpolate — linearly interpolate to the reference axes (no extrapolation)"
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        grp_layout.addWidget(self._radio_intersect)
        grp_layout.addWidget(self._radio_interp)
        layout.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "interp" if self._radio_interp.isChecked() else "intersect"

class CropDialog(QDialog):
    """Choose selected-value slicing or physical ranges with exact strides."""

    def __init__(self, reference: "RcsGrid", *, has_selected_values: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop / Slice")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Create a smaller dataset without interpolation. Use parameter-list "
            "selections, or crop physical ranges and retain every Nth source sample."
        ))

        self._rb_selected = QRadioButton("Use values selected in the parameter lists")
        self._rb_ranges = QRadioButton("Use numeric ranges and exact source-sample strides")
        self._rb_selected.setEnabled(bool(has_selected_values))
        self._rb_selected.setChecked(bool(has_selected_values))
        self._rb_ranges.setChecked(not bool(has_selected_values))
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_selected)
        mode_group.addButton(self._rb_ranges)
        layout.addWidget(self._rb_selected)
        layout.addWidget(self._rb_ranges)

        range_group = QGroupBox("Ranges")
        range_layout = QGridLayout(range_group)
        range_layout.addWidget(QLabel("Axis"), 0, 0)
        range_layout.addWidget(QLabel("Minimum"), 0, 1)
        range_layout.addWidget(QLabel("Maximum"), 0, 2)
        range_layout.addWidget(QLabel("Stride"), 0, 3)
        self._range_controls: dict[str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox, QSpinBox]] = {}

        az = _angle_axis_degrees(reference, "azimuth")
        el = _angle_axis_degrees(reference, "elevation")
        freq = np.asarray(reference.frequencies, dtype=float)
        frequency_unit = _canonical_frequency_unit(
            (reference.units or {}).get("frequency", "GHz")
        )
        specs = (
            ("azimuth", "Azimuth (deg)", az),
            ("elevation", "Elevation (deg)", el),
            ("frequency", f"Frequency ({frequency_unit})", freq),
        )
        for row, (axis, label, values) in enumerate(specs, start=1):
            enabled = QCheckBox(label)
            enabled.setChecked(True)
            minimum = QDoubleSpinBox()
            maximum = QDoubleSpinBox()
            for spin in (minimum, maximum):
                spin.setDecimals(9)
                spin.setRange(-1.0e300, 1.0e300)
                spin.setKeyboardTracking(False)
            minimum.setValue(float(np.min(values)))
            maximum.setValue(float(np.max(values)))
            stride = QSpinBox()
            stride.setRange(1, max(1, int(values.size)))
            stride.setValue(1)
            enabled.toggled.connect(minimum.setEnabled)
            enabled.toggled.connect(maximum.setEnabled)
            enabled.toggled.connect(stride.setEnabled)
            range_layout.addWidget(enabled, row, 0)
            range_layout.addWidget(minimum, row, 1)
            range_layout.addWidget(maximum, row, 2)
            range_layout.addWidget(stride, row, 3)
            self._range_controls[axis] = (enabled, minimum, maximum, stride)

        self._selected_polarizations = QCheckBox(
            "Limit output to polarizations selected in the parameter list"
        )
        self._selected_polarizations.setChecked(False)
        range_layout.addWidget(self._selected_polarizations, 4, 0, 1, 4)
        layout.addWidget(range_group)
        self._range_group = range_group
        range_group.setEnabled(self._rb_ranges.isChecked())
        self._rb_ranges.toggled.connect(range_group.setEnabled)

        note = QLabel(
            "Stride selects existing samples; it does not filter or invent values. "
            "Use Regrid when a specific coordinate grid is required."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, object]:
        ranges: dict[str, tuple[float, float] | None] = {}
        strides: dict[str, int] = {}
        for axis, (enabled, minimum, maximum, stride) in self._range_controls.items():
            ranges[axis] = (
                (float(minimum.value()), float(maximum.value()))
                if enabled.isChecked()
                else None
            )
            strides[axis] = int(stride.value()) if enabled.isChecked() else 1
        return {
            "mode": "selected" if self._rb_selected.isChecked() else "ranges",
            "ranges": ranges,
            "strides": strides,
            "selected_polarizations": self._selected_polarizations.isChecked(),
        }

class RegridDialog(QDialog):
    """Pick one numeric axis and an explicit, uniformly spaced target grid."""

    _AXIS_LABELS = {
        "azimuth": "Azimuth",
        "elevation": "Elevation",
        "frequency": "Frequency",
    }

    def __init__(self, reference: "RcsGrid", parent=None) -> None:
        super().__init__(parent)
        self._reference = reference
        self.setWindowTitle("Regrid")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Linearly interpolate the complex field onto one new axis. GRIM never "
            "extrapolates; every selected dataset must cover the requested range."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        grid = QGridLayout()
        grid.addWidget(QLabel("Axis:"), 0, 0)
        self._axis = QComboBox()
        axis_labels = dict(self._AXIS_LABELS)
        for key in ("azimuth", "elevation", "frequency"):
            self._axis.addItem(axis_labels[key], key)
        grid.addWidget(self._axis, 0, 1)

        self._label_start = QLabel()
        self._label_stop = QLabel()
        self._label_step = QLabel()
        self._spin_start = QDoubleSpinBox()
        self._spin_stop = QDoubleSpinBox()
        self._spin_step = QDoubleSpinBox()
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.setDecimals(9)
            spin.setRange(-1.0e300, 1.0e300)
            spin.setKeyboardTracking(False)
        self._spin_step.setMinimum(1.0e-12)
        grid.addWidget(self._label_start, 1, 0)
        grid.addWidget(self._spin_start, 1, 1)
        grid.addWidget(self._label_stop, 2, 0)
        grid.addWidget(self._spin_stop, 2, 1)
        grid.addWidget(self._label_step, 3, 0)
        grid.addWidget(self._spin_step, 3, 1)
        layout.addLayout(grid)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: gray;")
        layout.addWidget(self._summary)
        self._axis.currentIndexChanged.connect(self._load_axis_defaults)
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.valueChanged.connect(self._update_summary)
        self._load_axis_defaults()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _axis_values_and_unit(self, axis: str) -> tuple[np.ndarray, str]:
        if axis in {"azimuth", "elevation"}:
            return _angle_axis_degrees(self._reference, axis), "deg"
        unit = _canonical_frequency_unit(
            (self._reference.units or {}).get("frequency", "GHz")
        )
        return np.asarray(self._reference.frequencies, dtype=float), unit

    def _load_axis_defaults(self, *_args) -> None:
        axis = str(self._axis.currentData())
        values, unit = self._axis_values_and_unit(axis)
        step = float(np.median(np.diff(values))) if values.size > 1 else 1.0
        step = abs(step) if np.isfinite(step) and step != 0.0 else 1.0
        for label, stem in (
            (self._label_start, "Start"),
            (self._label_stop, "Stop"),
            (self._label_step, "Step"),
        ):
            label.setText(f"{stem} ({unit}):")
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.blockSignals(True)
        self._spin_start.setValue(float(np.min(values)))
        self._spin_stop.setValue(float(np.max(values)))
        self._spin_step.setValue(step)
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.blockSignals(False)
        self._update_summary()

    def _update_summary(self, *_args) -> None:
        start, stop, step = self.get_values()
        if step <= 0.0 or stop < start:
            self._summary.setText("Enter an increasing finite range and positive step.")
            return
        count = int(np.floor((stop - start) / step + 1.0e-9)) + 1
        resolved = start + max(0, count - 1) * step
        self._summary.setText(
            f"Resolved grid: {count:,} samples; final coordinate {resolved:.9g}. "
            "The stop value is not exceeded."
        )

    def get_values(self) -> tuple[float, float, float]:
        return (
            float(self._spin_start.value()),
            float(self._spin_stop.value()),
            float(self._spin_step.value()),
        )

    def get_params(self) -> dict[str, object]:
        start, stop, step = self.get_values()
        return {
            "axis": str(self._axis.currentData()),
            "start": start,
            "stop": stop,
            "step": step,
            "unit": self._axis_values_and_unit(str(self._axis.currentData()))[1],
        }

InterpolateDialog = RegridDialog

class JoinDialog(QDialog):
    """Join existing bins, rejecting conflicts unless a merge policy is chosen."""

    def __init__(self, operand_names, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Join / Merge Datasets")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        operands = QLabel("Operand order: " + " → ".join(map(str, operand_names)))
        operands.setWordWrap(True)
        layout.addWidget(operands)
        layout.addWidget(QLabel("When finite samples share the same grid cell:"))

        self._policy = QComboBox()
        self._policy.addItem("Join: reject conflicting overlaps", "error")
        self._policy.addItem("Priority: first operand wins", "priority-first")
        self._policy.addItem("Priority: last operand wins", "priority-last")
        self._policy.addItem(
            "Average linear power (overlap phase removed)", "power-mean"
        )
        self._policy.addItem("Average coherent complex field", "coherent-mean")
        layout.addWidget(self._policy)

        self._policy_help = QLabel()
        self._policy_help.setWordWrap(True)
        self._policy_help.setStyleSheet("color: gray;")
        layout.addWidget(self._policy_help)
        self._policy.currentIndexChanged.connect(self._update_help)
        self._update_help()

        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel("Native-axis matching tolerance:"))
        self._tolerance = QDoubleSpinBox()
        self._tolerance.setDecimals(12)
        self._tolerance.setRange(0.0, 1.0)
        self._tolerance.setValue(1.0e-6)
        self._tolerance.setSingleStep(1.0e-6)
        tolerance_row.addWidget(self._tolerance)
        tolerance_row.addStretch(1)
        layout.addLayout(tolerance_row)

        tolerance_help = QLabel(
            "One unitless number is applied independently to azimuth, elevation, "
            "and frequency in their declared native axis units. Selected datasets "
            "must therefore use the same units (for example, all degrees and GHz)."
        )
        tolerance_help.setWordWrap(True)
        tolerance_help.setStyleSheet("color: gray;")
        layout.addWidget(tolerance_help)
        self._tolerance_help = tolerance_help

        preview = QLabel(
            "GRIM joins existing bins without interpolation and creates one new "
            "unsaved dataset. Merge policies also report overlap counts and "
            "retain the complete overlap report in provenance."
        )
        preview.setWordWrap(True)
        layout.addWidget(preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_help(self, *_args) -> None:
        policy = str(self._policy.currentData())
        descriptions = {
            "error": (
                "Equal or complementary samples merge. Any conflicting finite "
                "overlap stops the operation; no input takes priority."
            ),
            "priority-first": (
                "Conflicting overlaps use the first selected operand. Missing cells "
                "are still filled by later operands."
            ),
            "priority-last": (
                "Conflicting overlaps use the last selected operand. Selection order "
                "is therefore physically significant."
            ),
            "power-mean": (
                "Repeated measurements are averaged in linear power. Phase remains "
                "available in single-source cells and is marked unknown wherever "
                "multiple contributors were averaged."
            ),
            "coherent-mean": (
                "Complex fields are averaged. Axes and all declared phase, time, and "
                "polarization conventions must agree."
            ),
        }
        self._policy_help.setText(descriptions[policy])

    def get_params(self) -> dict[str, object]:
        return {
            "policy": str(self._policy.currentData()),
            "tol": float(self._tolerance.value()),
        }

class DatasetAuditDialog(QDialog):
    """Scrollable, copyable presentation of one or more audit reports."""

    def __init__(self, reports, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Audit")
        self.resize(760, 600)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Read-only audit: no samples or metadata were changed. FAIL indicates an "
            "invalid dataset; WARN identifies a condition worth reviewing."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(self._format_reports(reports))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _format_reports(reports) -> str:
        status_labels = {
            "ok": "PASS",
            "pass": "PASS",
            "warning": "WARN",
            "warn": "WARN",
            "error": "FAIL",
            "fail": "FAIL",
        }

        def render_value(value) -> str:
            if isinstance(value, float):
                return f"{value:.9g}"
            if isinstance(value, bool):
                return "yes" if value else "no"
            if value is None:
                return "not available"
            return str(value)

        def append_mapping(lines, mapping, indent: int) -> None:
            prefix = " " * indent
            for key in sorted(mapping):
                value = mapping[key]
                label = str(key).replace("_", " ")
                if isinstance(value, dict):
                    lines.append(f"{prefix}{label}:")
                    append_mapping(lines, value, indent + 2)
                else:
                    lines.append(f"{prefix}{label}: {render_value(value)}")

        blocks: list[str] = []
        for name, report in reports:
            raw_status = str(report.get("status", "unknown")).strip().lower()
            status = status_labels.get(raw_status, "WARN")
            lines = [f"{name}", f"Status: {status}"]
            for key, heading in (
                ("errors", "Errors"),
                ("warnings", "Warnings"),
                ("info", "Information"),
            ):
                values = report.get(key) or []
                if values:
                    lines.append(f"{heading}:")
                    for value in values:
                        if not isinstance(value, dict):
                            lines.append(f"  - {value}")
                            continue
                        code = str(value.get("code", "issue")).replace("_", " ")
                        message = str(value.get("message", ""))
                        details = {
                            detail_key: detail_value
                            for detail_key, detail_value in value.items()
                            if detail_key not in {"code", "message"}
                        }
                        suffix = ""
                        if details:
                            suffix = "; " + ", ".join(
                                f"{str(detail_key).replace('_', ' ')}="
                                f"{render_value(detail_value)}"
                                for detail_key, detail_value in sorted(details.items())
                            )
                        lines.append(f"  - [{code}] {message}{suffix}")
            metrics = report.get("metrics") or {}
            if metrics:
                lines.append("Metrics:")
                append_mapping(lines, metrics, 2)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

class DatasetCompatibilityDialog(QDialog):
    """Copyable preflight report for selected multi-dataset operations."""

    def __init__(self, report_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Compatibility")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Read-only preflight against operand 1. PASS means the tested "
            "contract is satisfied; WARN requires a recorded assumption."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(str(report_text))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

class DatasetProvenanceDialog(QDialog):
    """Copyable, bounded view of lineage and metadata for selected rows."""

    def __init__(self, report_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Provenance")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Large metadata arrays are described by shape, type, and size rather "
            "than expanded. JSON provenance records are formatted for review."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(str(report_text))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

class ShiftDialog(QDialog):
    """Pick which axes (and/or RCS phase) to shift and by what amount.

    Azimuth/Elevation translate the corresponding axis values (degrees).
    Phase rotates every complex sample by exp(j·θ) (degrees) — it doesn't
    move axis values, but it lives here as the sole "shift the data
    instead of an axis" option to keep the UI consolidated.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shift")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select what to shift:"))

        def make_row(label_text: str, checked: bool, suffix: str = " °") -> tuple:
            row = QHBoxLayout()
            chk = QCheckBox(label_text)
            chk.setChecked(checked)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(1.0)
            spin.setValue(0.0)
            spin.setSuffix(suffix)
            spin.setEnabled(checked)
            chk.toggled.connect(spin.setEnabled)
            row.addWidget(chk)
            row.addWidget(spin)
            layout.addLayout(row)
            return chk, spin

        self._chk_az,    self._spin_az    = make_row("Azimuth",   True)
        self._chk_el,    self._spin_el    = make_row("Elevation", False)
        self._chk_phase, self._spin_phase = make_row("Phase",     False)
        # Phase is bounded to one full rotation since shift is mod 360°.
        self._spin_phase.setRange(-360.0, 360.0)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "azimuth":   (self._chk_az.isChecked(),    float(self._spin_az.value())),
            "elevation": (self._chk_el.isChecked(),    float(self._spin_el.value())),
            "phase":     (self._chk_phase.isChecked(), float(self._spin_phase.value())),
        }

class RangeCalibrationDialog(QDialog):
    """Assign loaded grids to the measured/exact calibration roles."""

    def __init__(self, dataset_entries, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Range Cal — Complex Substitution")
        self._entries = list(dataset_entries)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Selected table rows are the DUT measurement(s) to calibrate. "
            "Choose a measured calibration target and its trusted complex "
            "exact/reference response from the loaded datasets."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QGridLayout()
        self.combo_measured = QComboBox()
        self.combo_exact = QComboBox()
        for row_index, (name, _dataset) in enumerate(self._entries, start=1):
            label = f"[{row_index}] {name}"
            self.combo_measured.addItem(label)
            self.combo_exact.addItem(label)
        if len(self._entries) > 1:
            self.combo_exact.setCurrentIndex(1)
        form.addWidget(QLabel("Measured calibration target:"), 0, 0)
        form.addWidget(self.combo_measured, 0, 1)
        form.addWidget(QLabel("Exact/reference response:"), 1, 0)
        form.addWidget(self.combo_exact, 1, 1)

        self.spin_offset_in = QDoubleSpinBox()
        self.spin_offset_in.setDecimals(12)
        self.spin_offset_in.setRange(-1.0e6 / 0.0254, 1.0e6 / 0.0254)
        self.spin_offset_in.setSingleStep(0.001)
        self.spin_offset_in.setSuffix(" in")
        self.spin_offset_in.setToolTip(
            "Enter the one-way physical displacement. Positive means the "
            "measured calibration target is farther from radar than the "
            "DUT/reference plane; GRIM applies the monostatic two-way phase."
        )
        form.addWidget(QLabel("Signed calibrator range offset ΔR:"), 2, 0)
        form.addWidget(self.spin_offset_in, 2, 1)

        gain_row = QHBoxLayout()
        self.chk_gain_limit = QCheckBox("Limit correction gain")
        self.chk_gain_limit.setChecked(True)
        self.spin_gain_limit_db = QDoubleSpinBox()
        self.spin_gain_limit_db.setDecimals(1)
        self.spin_gain_limit_db.setRange(0.0, 300.0)
        self.spin_gain_limit_db.setValue(60.0)
        self.spin_gain_limit_db.setSuffix(" dB")
        self.spin_gain_limit_db.setToolTip(
            "Mask calibration bins whose |Aexact/Ameasured| correction exceeds "
            "this level. Other usable bins are still calibrated."
        )
        self.chk_gain_limit.toggled.connect(self.spin_gain_limit_db.setEnabled)
        gain_row.addWidget(self.chk_gain_limit)
        gain_row.addWidget(self.spin_gain_limit_db)
        form.addWidget(QLabel("Calibration bin masking:"), 3, 0)
        form.addLayout(gain_row, 3, 1)
        layout.addLayout(form)

        phase_law = QLabel(
            "Positive ΔR is away from radar. GRIM applies "
            "Aout = Adut · Aexact · exp(−j4πfΔR/c) / Ameasured."
        )
        phase_law.setWordWrap(True)
        layout.addWidget(phase_law)

        self.chk_broadcast = QCheckBox(
            "Broadcast singleton calibration azimuth/elevation across DUT angles"
        )
        self.chk_broadcast.setToolTip(
            "No angular averaging or interpolation is performed. Enable only "
            "when one frequency/polarization correction applies to every DUT look."
        )
        layout.addWidget(self.chk_broadcast)

        assumption_note = QLabel(
            "Selecting these roles requests complex calibration. Missing acquisition "
            "or phase-center declarations are recorded as assumptions; explicit "
            "incompatible units, axes, quantities, or phase signs still stop the job."
        )
        assumption_note.setWordWrap(True)
        layout.addWidget(assumption_note)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        warning = QLabel(
            "The exact response must be complex sigma₃D/dBsm data. A finite "
            "cylinder's 3-D reference must be supplied; GRIM will not substitute "
            "GHOST's infinite 2-D cylinder solution. Invalid/null correction bins "
            "are masked and reported."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.combo_measured.currentIndexChanged.connect(self._update_validity)
        self.combo_exact.currentIndexChanged.connect(self._update_validity)
        self._update_validity()

    def _update_validity(self, *_args) -> None:
        same_reference = (
            self.combo_measured.currentIndex() == self.combo_exact.currentIndex()
        )
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(not same_reference)
        if same_reference:
            self.validation_label.setText(
                "Choose different datasets for measured calibration and exact reference."
            )
        else:
            self.validation_label.setText(
                "Ready. Missing provenance will be recorded as assumed, not blocked."
            )

    def get_params(self) -> dict:
        measured_index = int(self.combo_measured.currentIndex())
        exact_index = int(self.combo_exact.currentIndex())
        return {
            "measured": self._entries[measured_index],
            "exact": self._entries[exact_index],
            "range_offset_m": float(self.spin_offset_in.value()) * 0.0254,
            "allow_singleton_angular_broadcast": self.chk_broadcast.isChecked(),
            "convention_attested": False,
            "maximum_correction_gain_db": (
                float(self.spin_gain_limit_db.value())
                if self.chk_gain_limit.isChecked()
                else None
            ),
        }

class SupportReferenceDifferenceDialog(QDialog):
    """Assign the two physical roles for guided support-reference subtraction."""

    def __init__(self, entries, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Support-Referenced Difference")
        self.setMinimumWidth(640)
        self._entries = list(entries)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Create an unsaved derived dataset using the exact complex operation "
            "A(target + support) - A(support-only). Select the physical roles "
            "below; GRIM will not interpolate, regrid, or change phase."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        role_box = QGroupBox("1. Assign acquisition roles")
        role_layout = QGridLayout(role_box)
        self.combo_target = QComboBox()
        self.combo_support = QComboBox()
        for row_index, (name, _dataset) in enumerate(self._entries, start=1):
            label = f"[{row_index}] {name}"
            self.combo_target.addItem(label)
            self.combo_support.addItem(label)
        if len(self._entries) > 1:
            self.combo_support.setCurrentIndex(1)
        role_layout.addWidget(QLabel("Target + support acquisition:"), 0, 0)
        role_layout.addWidget(self.combo_target, 0, 1)
        role_layout.addWidget(QLabel("Support-only reference:"), 1, 0)
        role_layout.addWidget(self.combo_support, 1, 1)
        layout.addWidget(role_box)

        self.compatibility_label = QLabel("")
        self.compatibility_label.setWordWrap(True)
        self.compatibility_label.setObjectName("supportReferenceCompatibility")
        layout.addWidget(self.compatibility_label)

        interpretation = QLabel(
            "The selected roles request exact complex subtraction. Missing acquisition "
            "metadata is recorded as assumed. The result is support-referenced, not a "
            "reconstructed free-space target: coupling, shadowing, multiple bounce, "
            "and acquisition drift cannot be recovered from two files."
        )
        interpretation.setWordWrap(True)
        layout.addWidget(interpretation)

        note = QLabel(
            "The output is added as a new unsaved row after calculation. Neither "
            "input is modified. QA metrics and assumptions are stored with the "
            "result; they do not prove physical support removal."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.combo_target.currentIndexChanged.connect(self._update_validity)
        self.combo_support.currentIndexChanged.connect(self._update_validity)
        self._update_validity()

    def _selected_entries(self):
        if not self._entries:
            return None, None
        target_index = int(self.combo_target.currentIndex())
        support_index = int(self.combo_support.currentIndex())
        if target_index < 0 or support_index < 0:
            return None, None
        return (
            self._entries[target_index],
            self._entries[support_index],
        )

    def _update_validity(self, *_args) -> None:
        target_entry, support_entry = self._selected_entries()
        valid = False
        message = "Select two different datasets."
        if target_entry is not None and support_entry is not None:
            if self.combo_target.currentIndex() == self.combo_support.currentIndex():
                message = (
                    "Target + support and support-only must be different rows."
                )
            else:
                target = target_entry[1]
                support = support_entry[1]
                missing = _missing_coherent_metadata_keys((target, support))
                acquisition_contract = None
                try:
                    target._assert_compatible(
                        support,
                        coherent=True,
                        coherent_metadata_attested=False,
                        _scan_phase_samples=False,
                    )
                    acquisition_contract = target._assert_support_reference_metadata_compatible(
                        support
                    )
                except (TypeError, ValueError) as exc:
                    message = (
                        "Not compatible for exact complex subtraction: " + str(exc)
                    )
                else:
                    acquisition_missing = dict(
                        acquisition_contract.get(
                            "missing_declarations_by_role", {}
                        )
                    )
                    valid = True
                    if missing or acquisition_missing:
                        coherent_labels = ", ".join(
                            _COHERENT_METADATA_LABELS[key]
                            for key in _COHERENT_METADATA_LABELS
                            if key in missing
                        )
                        semantic_families = acquisition_contract.get(
                            "semantic_families", {}
                        )
                        acquisition_labels = sorted(
                            {
                                str(
                                    semantic_families.get(
                                        fact.split(".", 1)[0], {}
                                    ).get("label", fact.split(".", 1)[0])
                                )
                                for fact in acquisition_missing
                            }
                        )
                        missing_sections = []
                        if coherent_labels:
                            missing_sections.append(
                                "coherent conventions: " + coherent_labels
                            )
                        if acquisition_labels:
                            missing_sections.append(
                                "acquisition/setup declarations: "
                                + ", ".join(acquisition_labels)
                            )
                        message = (
                            "Exact axes are compatible and no explicit declaration "
                            "contradicts the other input. "
                            "The full finite-phase sample scan will run in the "
                            "background before subtraction. "
                            "The following missing declarations will be recorded "
                            "as operation assumptions: "
                            + "; ".join(missing_sections)
                            + "."
                        )
                    else:
                        message = (
                            "Ready: axes, units, coherent metadata, coordinate frame, "
                            "phase reference, time convention, and polarization "
                            "basis are compatible. Full finite-phase sample QA will "
                            "run in the background before subtraction."
                        )
        self.compatibility_label.setText(message)
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(valid)

    def get_params(self) -> dict:
        target_entry, support_entry = self._selected_entries()
        return {
            "target": target_entry,
            "support": support_entry,
            "metadata_attested": False,
            "assumptions_attested": False,
        }

class RoundDialog(QDialog):
    """Pick which axes to round and at what decimal precision."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Round Axes")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select axes to round:"))
        self._chk_az = QCheckBox("Azimuths")
        self._chk_el = QCheckBox("Elevations")
        self._chk_fr = QCheckBox("Frequencies")
        self._chk_az.setChecked(True)
        self._chk_el.setChecked(True)
        self._chk_fr.setChecked(True)
        layout.addWidget(self._chk_az)
        layout.addWidget(self._chk_el)
        layout.addWidget(self._chk_fr)

        decimals_row = QHBoxLayout()
        decimals_row.addWidget(QLabel("Decimal places:"))
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(0)
        self._spin.setRange(0, 9)
        self._spin.setValue(1)
        self._spin.setSingleStep(1)
        decimals_row.addWidget(self._spin)
        decimals_row.addStretch(1)
        layout.addLayout(decimals_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "azimuths": self._chk_az.isChecked(),
            "elevations": self._chk_el.isChecked(),
            "frequencies": self._chk_fr.isChecked(),
            "decimals": int(self._spin.value()),
        }

class WrapDialog(QDialog):
    """Wrap the azimuth coordinate, stored phase, or both."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wrap")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what to wrap:"))

        self._wrap_azimuth = QCheckBox("Azimuth axis")
        self._wrap_phase = QCheckBox("Phase values")
        self._wrap_azimuth.setChecked(True)
        self._wrap_phase.setChecked(False)
        layout.addWidget(self._wrap_azimuth)
        layout.addWidget(self._wrap_phase)

        layout.addWidget(QLabel("Target interval:"))

        self._rb_0_360 = QRadioButton("[0°, 360°)")
        self._rb_pm180 = QRadioButton("[-180°, 180°)")
        self._rb_0_360.setChecked(True)
        layout.addWidget(self._rb_0_360)
        layout.addWidget(self._rb_pm180)

        group = QButtonGroup(self)
        group.addButton(self._rb_0_360)
        group.addButton(self._rb_pm180)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "0_360" if self._rb_0_360.isChecked() else "-180_180"

    def get_params(self) -> dict[str, object]:
        return {
            "azimuth": self._wrap_azimuth.isChecked(),
            "phase": self._wrap_phase.isChecked(),
            "mode": self.get_mode(),
        }

class DecimateDialog(QDialog):
    """Configure boxcar-prefiltered integer-factor downsampling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Decimate with Prefilter")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Average adjacent bins on a uniformly sampled source axis before "
            "retaining one output bin. This avoids the unfiltered point-sampling "
            "behavior of a coarse Regrid."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        self._axis = QComboBox()
        self._axis.addItem("Azimuth", "azimuth")
        self._axis.addItem("Elevation", "elevation")
        self._axis.addItem("Frequency", "frequency")
        self._factor = QSpinBox()
        self._factor.setRange(2, 1_000_000)
        self._factor.setValue(2)
        self._mode = QComboBox()
        self._mode.addItem("Linear-power mean (phase becomes unknown)", "power")
        self._mode.addItem("Coherent complex-field mean", "coherent")
        grid.addWidget(QLabel("Axis:"), 0, 0)
        grid.addWidget(self._axis, 0, 1)
        grid.addWidget(QLabel("Integer factor:"), 1, 0)
        grid.addWidget(self._factor, 1, 1)
        grid.addWidget(QLabel("Filter domain:"), 2, 0)
        grid.addWidget(self._mode, 2, 1)
        layout.addLayout(grid)
        warning = QLabel(
            "The final partial bin is retained using its actual sample count. "
            "A coherent mean requires finite phase and represents filtered field, "
            "not averaged RCS power."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, object]:
        return {
            "axis": str(self._axis.currentData()),
            "factor": int(self._factor.value()),
            "mode": str(self._mode.currentData()),
        }

class MedianizeDialog(QDialog):
    """Pick the sliding-window parameters for a median smoothing pass along
    the azimuth axis.

    Window = full azimuth width of each window (degrees), centerd on each
    output sample. Slide = step between adjacent window centers (degrees).
    Slide < window gives overlap (heavier smoothing, denser output); slide =
    window gives non-overlapping bins; slide > window subsamples the input.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Medianize")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Sliding median over azimuth — replaces samples within each "
            "window with the median linear σ inside it."
        ))

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Window (deg):"))
        self._spin_window = QDoubleSpinBox()
        self._spin_window.setDecimals(4)
        self._spin_window.setRange(1.0e-4, 360.0)
        self._spin_window.setSingleStep(0.1)
        self._spin_window.setValue(5.0)
        win_row.addWidget(self._spin_window)
        win_row.addStretch(1)
        layout.addLayout(win_row)

        slide_row = QHBoxLayout()
        slide_row.addWidget(QLabel("Slide (deg):"))
        self._spin_slide = QDoubleSpinBox()
        self._spin_slide.setDecimals(4)
        self._spin_slide.setRange(1.0e-4, 360.0)
        self._spin_slide.setSingleStep(0.1)
        self._spin_slide.setValue(1.0)
        slide_row.addWidget(self._spin_slide)
        slide_row.addStretch(1)
        layout.addLayout(slide_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "window_deg": float(self._spin_window.value()),
            "slide_deg": float(self._spin_slide.value()),
        }

class ExtrusionConversionDialog(QDialog):
    """Choose direction and length for a broadside uniform-extrusion estimate."""

    _UNIT_TO_M = {"m": 1.0, "in": 0.0254, "ft": 0.3048}

    def __init__(
        self,
        parent=None,
        *,
        destination: str = "dbke",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extrusion Estimate")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        self._direction = QComboBox()
        self._direction.addItem("3D RCS → 2D width (dBsm → dBke)", "dbke")
        self._direction.addItem("2D width → 3D RCS (dBke → dBsm)", "dbsm")
        index = self._direction.findData(destination)
        if index < 0:
            raise ValueError("destination must be dbke or dbsm")
        self._direction.setCurrentIndex(index)
        layout.addWidget(QLabel("Conversion direction:"))
        layout.addWidget(self._direction)
        note = QLabel(
            "Assumes broadside illumination of a uniform extruded body. "
            "This is an extrusion estimate, not a general 2D/3D conversion. "
            "Selected datasets with a different source quantity are reported as skipped."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(QLabel("Extrusion length L:"))

        row = QHBoxLayout()
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(6)
        self._spin.setRange(1.0e-6, 1.0e6)
        self._spin.setSingleStep(1.0)
        self._spin.setValue(24.0)
        row.addWidget(self._spin)
        self._combo = QComboBox()
        self._combo.addItems(["in", "ft", "m"])
        self._combo.setCurrentText("in")
        row.addWidget(self._combo)
        row.addStretch(1)
        layout.addLayout(row)

        self._formula = QLabel()
        self._formula.setWordWrap(True)
        layout.addWidget(self._formula)
        self._direction.currentIndexChanged.connect(self._update_formula)
        self._update_formula()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def destination(self) -> str:
        return str(self._direction.currentData())

    def _update_formula(self, *_args) -> None:
        if self.destination() == "dbke":
            formula = (
                "σ_2D = σ_3D · λ / (2 L²) → dBke = dBsm + 10·log₁₀(π / L²) "
                "(L in meters; frequency-independent dB offset)."
            )
        else:
            formula = (
                "σ_3D = σ_2D · (2 L² / λ) → dBsm = dBke + 20·log₁₀(L) − "
                "10·log₁₀(π) (L in meters; frequency-independent dB offset)."
            )
        self._formula.setText(formula)

    def length_m(self) -> float:
        unit = self._combo.currentText().strip().lower()
        factor = self._UNIT_TO_M.get(unit, 1.0)
        return float(self._spin.value()) * factor

    def display_text(self) -> str:
        return f"{float(self._spin.value()):g} {self._combo.currentText()}"

class AxisUnitsDialog(QDialog):
    """Choose equivalent storage units for all three numeric axes."""

    def __init__(self, reference: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Axis Units")
        layout = QVBoxLayout(self)
        note = QLabel(
            "This is an exact unit conversion. Physical coordinates and all "
            "RCS samples remain unchanged; no interpolation is performed."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        self._azimuth = QComboBox()
        self._elevation = QComboBox()
        self._frequency = QComboBox()
        for combo in (self._azimuth, self._elevation):
            combo.addItems(["deg", "rad"])
        self._frequency.addItems(["Hz", "kHz", "MHz", "GHz"])
        current_az = _canonical_angle_unit(
            (reference.units or {}).get("azimuth", "deg")
        )
        current_el = _canonical_angle_unit(
            (reference.units or {}).get("elevation", "deg")
        )
        current_frequency = _canonical_frequency_unit(
            (reference.units or {}).get("frequency", "GHz")
        )
        self._azimuth.setCurrentText(current_az)
        self._elevation.setCurrentText(current_el)
        self._frequency.setCurrentText(current_frequency)
        for row, (label, combo) in enumerate((
            ("Azimuth:", self._azimuth),
            ("Elevation:", self._elevation),
            ("Frequency:", self._frequency),
        )):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(combo, row, 1)
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, str]:
        return {
            "azimuth": self._azimuth.currentText(),
            "elevation": self._elevation.currentText(),
            "frequency": self._frequency.currentText(),
        }

class WedgeConicDialog(QDialog):
    """Confirm the physical conventions for a wedge-to-conic conversion.

    Geometry: vertical-axis turntable (axis = world-z, fixed), target tilted
    by a foam wedge with ridge along body-y (pitch wedge). The current
    `azimuths` axis holds the turntable angle φ; `elevations` holds the wedge
    tilt τ. Output (azimuths, elevations) become true conic (longitude φ',
    latitude θ') on the body sphere.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wedge → Conic")
        layout = QVBoxLayout(self)

        explanation = QLabel(
            "Input axes: azimuth = signed mechanical turntable angle φ, "
            "elevation = vehicle pitch tilt τ. Each tilt needs a full 360° "
            "turntable revolution. Output axes are conic azimuth/elevation.\n\n"
            "Waterline only (exactly 0° tilt): conic azimuth = −φ, wrapped and "
            "sorted. Samples and available polarizations are preserved, "
            "without interpolation or polarization rotation. Phase is optional.\n\n"
            "Tilted cuts: join at least two measured tilts into one dataset "
            "first. Selecting several datasets converts each independently. "
            "Conversion interpolates complex data and rotates V/H, requiring "
            "meaningful phase and VV/HH plus VH or HV, unless missing cross-pol "
            "is explicitly assumed zero.\n\n"
            "Each fixed tilt traces a great circle. Limited tilt ranges leave "
            "gaps in nonzero-elevation conic cuts near side aspect. Unsupported "
            "directions remain blank (NaN); no extrapolation fills them."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.setMinimumWidth(540)

        axes_note = QLabel(
            "Assumed setup: horizontal radar line of sight along world +x, "
            "a vertical pylon, positive turntable rotation about world +z, "
            "and vehicle pitch about body +y before turntable rotation. "
            "Turntable angles must follow this sign convention. These "
            "assumptions are stored with the converted dataset."
        )
        axes_note.setWordWrap(True)
        layout.addWidget(axes_note)
        self._chk_cross_zero = QCheckBox(
            "Tilted cuts only: assume missing VH/HV is exactly zero (when justified)."
        )
        layout.addWidget(self._chk_cross_zero)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "mode": "regrid",
            "attest_wedge_axes": False,
            "assume_missing_cross_pol_zero": self._chk_cross_zero.isChecked(),
        }

class ExportCsvDialog(QDialog):
    """Options for exporting RCS data to a CSV file."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to CSV")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Magnitude:"), 0, 0)
        self._combo_scale = QComboBox()
        self._combo_scale.addItem("Linear", "linear")
        self._combo_scale.addItem("dB (dimensionless ratio)", "db")
        self._combo_scale.addItem("dBsm", "dbsm")
        self._combo_scale.addItem("dBke", "dbke")
        self._combo_scale.addItem("Both (Linear + dataset's physical dB unit)", "both")
        grid.addWidget(self._combo_scale, 0, 1)

        layout.addLayout(grid)

        self._chk_phase = QCheckBox("Include phase column (degrees)")
        self._chk_phase.setChecked(True)
        layout.addWidget(self._chk_phase)

        layout.addWidget(QLabel(
            "Writes versioned GRIM flat RCS CSV with explicit axis units, physical "
            "quantity, coordinate convention, and coherent metadata.\n"
            "For dBke export, frequency-dependent conversion uses the dataset frequency axis.\n"
            "One row per sample — all combinations of dataset axes."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_options(self) -> tuple[str, bool]:
        """Return (scale, include_phase)."""
        return (
            self._combo_scale.currentData(),
            self._chk_phase.isChecked(),
        )

class StatisticsDialog(QDialog):
    """Single dialog for statistics dataset: all options in one place."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics Dataset")
        layout = QVBoxLayout(self)

        params_grid = QGridLayout()

        params_grid.addWidget(QLabel("Statistic:"), 0, 0)
        self.combo_stat = QComboBox()
        self.combo_stat.addItems(["mean", "median", "min", "max", "std", "percentile"])
        params_grid.addWidget(self.combo_stat, 0, 1)

        params_grid.addWidget(QLabel("Percentile:"), 0, 2)
        self.spin_pct = QDoubleSpinBox()
        self.spin_pct.setRange(0.0, 100.0)
        self.spin_pct.setDecimals(1)
        self.spin_pct.setSingleStep(5.0)
        self.spin_pct.setValue(50.0)
        self.spin_pct.setEnabled(False)
        self.spin_pct.setToolTip("Only used when Statistic = percentile")
        params_grid.addWidget(self.spin_pct, 0, 3)

        layout.addLayout(params_grid)

        axes_group = QGroupBox("Axes to Reduce")
        axes_row = QHBoxLayout(axes_group)
        self.chk_az = QCheckBox("Azimuth")
        self.chk_az.setChecked(True)
        self.chk_el = QCheckBox("Elevation")
        self.chk_el.setChecked(True)
        self.chk_freq = QCheckBox("Frequency")
        self.chk_freq.setChecked(True)
        self.chk_pol = QCheckBox("Polarization")
        self.chk_pol.setChecked(False)
        for chk in (self.chk_az, self.chk_el, self.chk_freq, self.chk_pol):
            axes_row.addWidget(chk)
        axes_row.addStretch(1)
        layout.addWidget(axes_group)

        domain_note = QLabel(
            "Statistics are computed on linear power, not on displayed dB values. "
            "The reduced result has undefined coherent phase."
        )
        domain_note.setWordWrap(True)
        domain_note.setToolTip(
            "For example, converting the mean linear power to dB is generally not "
            "the same as averaging sample values after converting each one to dB."
        )
        layout.addWidget(domain_note)

        self.chk_broadcast = QCheckBox(
            "Repeat each statistic across the reduced axes on the original grid"
        )
        self.chk_broadcast.setChecked(True)
        self.chk_broadcast.setToolTip(
            "For example, reducing azimuth repeats its statistic at every azimuth "
            "separately for each frequency, elevation and polarization. "
            "Uncheck for a smaller dataset with one coordinate per reduced axis."
        )
        layout.addWidget(self.chk_broadcast)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.combo_stat.currentTextChanged.connect(
            lambda t: self.spin_pct.setEnabled(t == "percentile")
        )

    def get_params(self) -> tuple[str, float, list[str], bool]:
        """Return (statistic, percentile, axes, broadcast_reduced)."""
        statistic = self.combo_stat.currentText()
        percentile = self.spin_pct.value()
        axes = [
            name
            for chk, name in (
                (self.chk_az, "azimuth"),
                (self.chk_el, "elevation"),
                (self.chk_freq, "frequency"),
                (self.chk_pol, "polarization"),
            )
            if chk.isChecked()
        ]
        return statistic, percentile, axes, self.chk_broadcast.isChecked()


class TimeGateDialog(QDialog):
    """Choose a down-range gate with a live preview of the active dataset.

    The preview is the median Hann-windowed down-range profile over azimuth
    for one elevation and polarization, before (gray) and after (colour)
    gating, in dB relative to the ungated peak. The gate itself is applied
    without a window so in-gate responses keep their calibrated level.
    """

    def __init__(self, dataset: RcsGrid, *, elevation_index: int = 0,
                 polarization_index: int = 0, parent=None) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PySide6.QtCore import QTimer

        from GRIM_Backend.datasets.transforms import gate_geometry

        super().__init__(parent)
        self.setWindowTitle("Time Gate")
        self._dataset = dataset
        geometry = gate_geometry(dataset)
        self._half_range = 0.5 * geometry["unambiguous_m"]
        resolution = geometry["resolution_m"]

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Keep (or remove) scatterers inside a down-range window, measured in "
            "metres from the phase reference, positive away from the radar. "
            f"Resolution {resolution:.4g} m; unambiguous range ±{self._half_range:.4g} m."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        default = min(self._half_range / 4.0, max(20.0 * resolution, 0.5))

        def metres(value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-self._half_range, self._half_range)
            spin.setSingleStep(max(resolution, 1.0e-4))
            spin.setSuffix(" m")
            spin.setValue(value)
            return spin

        self.spin_start = metres(-default)
        self.spin_stop = metres(default)
        self.spin_taper = QDoubleSpinBox()
        self.spin_taper.setRange(0.0, 100.0)
        self.spin_taper.setDecimals(1)
        self.spin_taper.setSuffix(" %")
        self.spin_taper.setValue(20.0)
        self.spin_taper.setToolTip("Share of the gate width given to raised-cosine edges.")
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Keep inside gate", "keep")
        self.combo_mode.addItem("Remove inside gate", "remove")
        self.chk_compensate = QCheckBox("Compensate band-edge droop")
        self.chk_compensate.setChecked(True)
        self.chk_compensate.setToolTip(
            "Divide out the gate's effect on a point at the gate center, which "
            "otherwise lowers the first and last frequencies. Keep mode only."
        )
        self.combo_elevation = QComboBox()
        for value in np.asarray(dataset.elevations):
            self.combo_elevation.addItem(f"{float(value):g}")
        self.combo_elevation.setCurrentIndex(int(elevation_index))
        self.combo_polarization = QComboBox()
        for value in np.asarray(dataset.polarizations):
            self.combo_polarization.addItem(str(value))
        self.combo_polarization.setCurrentIndex(int(polarization_index))

        grid.addWidget(QLabel("Gate start"), 0, 0)
        grid.addWidget(self.spin_start, 0, 1)
        grid.addWidget(QLabel("Gate stop"), 0, 2)
        grid.addWidget(self.spin_stop, 0, 3)
        grid.addWidget(QLabel("Edge taper"), 1, 0)
        grid.addWidget(self.spin_taper, 1, 1)
        grid.addWidget(QLabel("Mode"), 1, 2)
        grid.addWidget(self.combo_mode, 1, 3)
        grid.addWidget(self.chk_compensate, 2, 0, 1, 4)
        grid.addWidget(QLabel("Preview elevation"), 3, 0)
        grid.addWidget(self.combo_elevation, 3, 1)
        grid.addWidget(QLabel("Preview polarization"), 3, 2)
        grid.addWidget(self.combo_polarization, 3, 3)
        layout.addLayout(grid)

        self._figure = Figure(figsize=(6.4, 3.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(240)
        self._axes = self._figure.add_subplot(111)
        layout.addWidget(self._canvas, 1)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)

        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self.update_preview)
        for signal in (
            self.spin_start.valueChanged, self.spin_stop.valueChanged,
            self.spin_taper.valueChanged, self.combo_mode.currentIndexChanged,
            self.chk_compensate.toggled, self.combo_elevation.currentIndexChanged,
            self.combo_polarization.currentIndexChanged,
        ):
            signal.connect(self._preview_timer.start)
        self.combo_mode.currentIndexChanged.connect(
            lambda: self.chk_compensate.setEnabled(self.combo_mode.currentData() == "keep")
        )
        self.update_preview()

    def get_params(self) -> dict:
        return {
            "start_m": float(self.spin_start.value()),
            "stop_m": float(self.spin_stop.value()),
            "taper": float(self.spin_taper.value()) / 100.0,
            "mode": str(self.combo_mode.currentData()),
            "compensate": bool(self.chk_compensate.isChecked()),
        }

    def update_preview(self) -> None:
        from GRIM_Backend.datasets.transforms import _validate_gate, down_range_profile

        params = self.get_params()
        ax = self._axes
        ax.clear()
        ok = self.btn_box.button(QDialogButtonBox.Ok)
        try:
            _validate_gate(self._dataset, params["start_m"], params["stop_m"],
                           params["taper"], params["mode"])
            gate_error = ""
        except ValueError as exc:
            gate_error = str(exc)
        ok.setEnabled(not gate_error)
        location = {
            "elevation_index": self.combo_elevation.currentIndex(),
            "polarization_index": self.combo_polarization.currentIndex(),
        }
        try:
            ranges, before = down_range_profile(self._dataset, **location)
            after = None
            if not gate_error:
                _ranges, after = down_range_profile(self._dataset, gate=params, **location)
        except ValueError as exc:
            self.preview_status.setText(f"No preview: {exc}.")
            self._canvas.draw_idle()
            return
        peak = float(np.nanmax(before)) if np.any(before > 0) else 1.0
        with np.errstate(divide="ignore"):
            ax.plot(ranges, 10.0 * np.log10(before / peak), color="#8a8a8a",
                    linewidth=1.0, label="Before")
            if after is not None:
                ax.plot(ranges, 10.0 * np.log10(after / peak), color="#1f77b4",
                        linewidth=1.2, label="After")
        ax.axvspan(params["start_m"], params["stop_m"], color="#59a14f", alpha=0.18,
                   linewidth=0, label="Gate")
        ax.set_xlim(-self._half_range, self._half_range)
        ax.set_ylim(-80.0, 5.0)
        ax.set_xlabel("Down range (m)")
        ax.set_ylabel("Relative level (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        self._figure.tight_layout()
        self.preview_status.setText(
            gate_error[:1].upper() + gate_error[1:] + "." if gate_error
            else "Median over azimuth; Hann-windowed for display only."
        )
        self._canvas.draw_idle()


class PhaseCenterDialog(QDialog):
    """Enter the new phase-center position in the dataset's body axes."""

    UNITS = (("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048))

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phase Center")
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Move the phase reference to this point, measured from the current "
            "phase reference in the dataset's body axes: +x toward azimuth 0°, "
            "+y toward azimuth +90°, +z toward elevation +90° (top). Levels are "
            "unchanged; each sample's phase gets the matching two-way ramp. "
            "A point scatterer at this position ends up at the new origin."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        self.spins = []
        for column, axis in enumerate(("x", "y", "z")):
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-1.0e6, 1.0e6)
            spin.setSingleStep(0.01)
            grid.addWidget(QLabel(axis), 0, 2 * column)
            grid.addWidget(spin, 0, 2 * column + 1)
            self.spins.append(spin)
        self.combo_unit = QComboBox()
        for label, _scale in self.UNITS:
            self.combo_unit.addItem(label)
        grid.addWidget(QLabel("Unit"), 1, 0)
        grid.addWidget(self.combo_unit, 1, 1)
        layout.addLayout(grid)
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)
        for spin in self.spins:
            spin.valueChanged.connect(self._update_ok)
        self._update_ok()

    def _update_ok(self) -> None:
        self.btn_box.button(QDialogButtonBox.Ok).setEnabled(
            any(spin.value() != 0.0 for spin in self.spins)
        )

    def get_params(self) -> dict:
        label, scale = self.UNITS[self.combo_unit.currentIndex()]
        values = [spin.value() for spin in self.spins]
        return {
            "x_m": values[0] * scale,
            "y_m": values[1] * scale,
            "z_m": values[2] * scale,
            "entered": tuple(values),
            "unit": label,
        }
