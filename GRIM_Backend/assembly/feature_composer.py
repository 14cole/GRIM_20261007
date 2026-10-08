"""Small, geometry-first feature composer for the vehicle assembly workflow.

The row builders are Qt-free. They author the existing strict CSV contracts;
response compatibility and placement on the actual skin remain backend checks.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from GRIM_Backend.assembly.model import UNIT_CHOICES


MAX_FEATURE_ROWS = 10000
_EPS = 1.0e-12


def _vector(values: Sequence[float], label: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} needs three finite numbers (X, Y, Z).") from exc
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} needs three finite numbers (X, Y, Z).")
    return result


def _unit(values: Sequence[float], label: str) -> tuple[float, float, float]:
    vector = _vector(values, label)
    length = math.hypot(*vector)
    if length <= _EPS or not math.isfinite(length):
        raise ValueError(f"{label} must be a nonzero direction vector.")
    return tuple(value / length for value in vector)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _format(values):
    return tuple(format(value, ".17g") if value else "0" for value in values)


def feature_identifier(name: str) -> str:
    """Return a readable seed ID; the receiving vehicle owns collision handling."""
    name = str(name).strip()
    if not name:
        raise ValueError("Give this feature a name.")
    return re.sub(r"[^a-z0-9_.-]+", "_", name.lower()).strip("_.-") or "feature"


def _dataset_identifier(dataset_id):
    identifier = str(dataset_id).strip()
    if not identifier:
        raise ValueError("A response dataset ID is required.")
    return identifier


def compose_point_rows(name, dataset_id, position, normal, roll, count=1,
                       spacing=(0, 0, 0)) -> list[tuple[str, ...]]:
    """Author a point or evenly spaced row in body coordinates.

    Normal and roll are directions, not lengths. The normal is normalized and
    the roll reference is projected into its perpendicular plane, matching the
    response frame construction. No surface positions are inferred or snapped.
    """
    identifier = feature_identifier(name)
    dataset = _dataset_identifier(dataset_id)
    try:
        integer_count = int(count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Count must be a whole number from 1 to {MAX_FEATURE_ROWS:,}.") from exc
    if isinstance(count, bool) or integer_count != count or not 1 <= integer_count <= MAX_FEATURE_ROWS:
        raise ValueError(f"Count must be a whole number from 1 to {MAX_FEATURE_ROWS:,}.")
    origin = _vector(position, "Position")
    step = _vector(spacing, "Repeat spacing")
    if integer_count > 1 and not any(step):
        raise ValueError("Set a nonzero repeat spacing so the points have distinct positions.")
    outward = _unit(normal, "Outward normal")
    reference = _unit(roll, "Roll reference")
    projection = _dot(reference, outward)
    tangent = tuple(value - projection * axis for value, axis in zip(reference, outward))
    if math.hypot(*tangent) <= 1.0e-9:
        raise ValueError("Roll reference must not be parallel to the outward normal.")
    tangent = _unit(tangent, "Roll reference")
    result = []
    previous = None
    for index in range(integer_count):
        location = _vector(tuple(value + index * delta for value, delta in zip(origin, step)), "Repeated position")
        if previous == location:
            raise ValueError("Repeat spacing is too small to produce distinct positions at these coordinates.")
        previous = location
        point_id = identifier if integer_count == 1 else f"{identifier}_{index + 1:03d}"
        result.append((point_id, dataset, *_format(location), *_format(outward), *_format(tangent)))
    return result


def compose_line_rows(name, dataset_id, vertices, normals, closed=False) -> list[tuple[str, ...]]:
    """Join ordered vertices into contiguous segments with outward end normals."""
    identifier = feature_identifier(name)
    dataset = _dataset_identifier(dataset_id)
    positions = [_vector(vertex, f"Vertex {index + 1}") for index, vertex in enumerate(vertices)]
    directions = [_unit(normal, f"Normal at vertex {index + 1}") for index, normal in enumerate(normals)]
    if len(positions) < (3 if closed else 2):
        raise ValueError("A closed loop needs at least three vertices." if closed else "A line needs at least two vertices.")
    if len(positions) != len(directions):
        raise ValueError("Provide one outward normal for each vertex.")
    if len(positions) - (0 if closed else 1) > MAX_FEATURE_ROWS:
        raise ValueError(f"A feature can contain at most {MAX_FEATURE_ROWS:,} segments.")
    if closed and positions[-1] == positions[0]:
        raise ValueError("Remove the repeated last vertex; Close loop already joins it to the first vertex.")
    indices = [(index, index + 1) for index in range(len(positions) - 1)]
    if closed:
        indices.append((len(positions) - 1, 0))
    result = []
    path_tangents = []
    for segment, (start, end) in enumerate(indices, 1):
        delta = _vector(tuple(b - a for a, b in zip(positions[start], positions[end])), f"Segment {segment}")
        if not any(delta):
            raise ValueError(f"Vertices {start + 1} and {end + 1} coincide; every segment needs a nonzero length.")
        length = math.hypot(*delta)
        if not math.isfinite(length):
            raise ValueError(f"Segment {segment} has a nonfinite length.")
        tangent = tuple(value / length for value in delta)
        if path_tangents and _dot(path_tangents[-1], tangent) <= -1.0 + 1.0e-10:
            raise ValueError(f"Segment {segment} immediately backtracks over the previous segment; remove the overlapping return.")
        path_tangents.append(tangent)
        cross_directions = []
        for index in (start, end):
            normal = directions[index]
            along = _dot(normal, tangent)
            projected = tuple(value - along * axis for value, axis in zip(normal, tangent))
            if math.hypot(*projected) <= 1.0e-9:
                raise ValueError(f"Normal at vertex {index + 1} is parallel to segment {segment}; supply the outward skin normal.")
            cross_directions.append(_unit(projected, f"Normal at vertex {index + 1}"))
        if _dot(*cross_directions) <= -1.0 + 1.0e-12:
            raise ValueError(f"Normals across segment {segment} oppose each other; add intermediate vertices with outward normals.")
        result.append((identifier, dataset, str(segment), *_format(positions[start]),
                       *_format(positions[end]), *_format(directions[start]), *_format(directions[end])))
    if (closed or positions[-1] == positions[0]) and _dot(path_tangents[-1], path_tangents[0]) <= -1.0 + 1.0e-10:
        raise ValueError("The closing segment backtracks over the first segment; remove the overlapping return.")
    return result


_GUI_IMPORT_ERROR = None
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
        QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
        QLineEdit, QPushButton, QScrollArea, QSpinBox, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget,
    )
except (ImportError, RuntimeError) as exc:  # Qt-free row authoring remains available.
    _GUI_IMPORT_ERROR = exc

GUI_AVAILABLE = _GUI_IMPORT_ERROR is None


if GUI_AVAILABLE:
    class AddFeatureDialog(QDialog):
        """Add one named point group or line path without hand-authoring a CSV."""

        def __init__(self, kind: str, units: str = "", parent=None,
                     dataset_choices: Mapping[str, str] | None = None):
            if kind not in {"point", "line"}:
                raise ValueError("Feature kind must be 'point' or 'line'.")
            if units and units not in dict(UNIT_CHOICES).values():
                raise ValueError("Choose inches, millimeters, meters, or feet for placement units.")
            super().__init__(parent)
            self.kind = kind
            self._definition = None
            self._last_preset_name = "Fastener" if kind == "point" else "Line feature"
            self.setWindowTitle("Add point feature" if kind == "point" else "Add line feature")
            self.resize(760, 660)
            outer = QVBoxLayout(self)
            introduction = QLabel(
                "Choose the measured or simulated response, then place one feature or a row of identical features."
                if kind == "point" else
                "Choose the line response, then define its path with ordered vertices. Adjacent vertices form a segment."
            )
            introduction.setWordWrap(True)
            outer.addWidget(introduction)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            body = QWidget()
            content = QVBoxLayout(body)
            content.setContentsMargins(0, 0, 0, 0)
            scroll.setWidget(body)
            outer.addWidget(scroll, 1)

            response_group = QGroupBox("1  Response")
            form = QFormLayout(response_group)
            self.feature_type = QComboBox()
            self.feature_type.addItems(["Fastener", "Inlet", "Antenna", "Other point feature"])
            if kind == "point":
                form.addRow("Feature type", self.feature_type)
                self.feature_type.setToolTip("Naming shortcuts only. The selected response file defines the physics.")
            self.name_edit = QLineEdit(self._last_preset_name)
            self.name_edit.setPlaceholderText("For example: upper fuselage fasteners")
            form.addRow("Group name", self.name_edit)
            self.response_choice = QComboBox()
            self.response_choice.addItem("Choose a file…", "")
            for identifier, path in (dataset_choices or {}).items():
                self.response_choice.addItem(str(identifier), str(path))
            if dataset_choices:
                form.addRow("Reuse a response", self.response_choice)
            file_row = QWidget()
            file_layout = QHBoxLayout(file_row)
            file_layout.setContentsMargins(0, 0, 0, 0)
            self.response_edit = QLineEdit()
            self.response_edit.setPlaceholderText("Select a .grim response file")
            browse = QPushButton("Browse…")
            browse.clicked.connect(self._browse_response)
            file_layout.addWidget(self.response_edit, 1)
            file_layout.addWidget(browse)
            form.addRow("Response file", file_row)
            response_help = QLabel(
                "Point responses must contain coherent feature-minus-skin delta fields, including VV, HH and cross-polarization channels. Feature type only sets the name."
                if kind == "point" else
                "Line responses must contain coherent feature-minus-skin TE and TM coefficients for line expansion. Use a line response, not a full-body response."
            )
            response_help.setWordWrap(True)
            form.addRow(response_help)
            content.addWidget(response_group)

            geometry_group = QGroupBox("2  Placement")
            geometry = QVBoxLayout(geometry_group)
            coordinates = QLabel("Body coordinates: +X right, +Y toward the nose, +Z up. Positions and repeat spacing use the units below.")
            coordinates.setWordWrap(True)
            geometry.addWidget(coordinates)
            units_row = QFormLayout()
            self.units_combo = QComboBox()
            self.units_combo.addItem("Choose placement units…", "")
            for label, value in UNIT_CHOICES:
                self.units_combo.addItem(label, value)
            if units:
                self.units_combo.setCurrentIndex(self.units_combo.findData(units))
                self.units_combo.setEnabled(False)
                self.units_combo.setToolTip("Uses the vehicle's shared placement units.")
            units_row.addRow("Placement units", self.units_combo)
            geometry.addLayout(units_row)
            if kind == "point":
                self._build_point_fields(geometry)
            else:
                self._build_line_fields(geometry)
            content.addWidget(geometry_group)
            content.addStretch(1)
            final_note = QLabel("Directions are unitless and normalized on add. Body-surface alignment and response compatibility are checked when you validate the vehicle.")
            final_note.setWordWrap(True)
            outer.addWidget(final_note)
            self.error_label = QLabel()
            self.error_label.setWordWrap(True)
            self.error_label.setStyleSheet("color: #d65a4a;")
            self.error_label.setAccessibleName("Feature validation error")
            outer.addWidget(self.error_label)
            self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            self.dialog_buttons.button(QDialogButtonBox.Ok).setText("Add to vehicle")
            self.dialog_buttons.accepted.connect(self.accept)
            self.dialog_buttons.rejected.connect(self.reject)
            outer.addWidget(self.dialog_buttons)
            self.feature_type.currentTextChanged.connect(self._set_preset_name)
            self.response_choice.currentIndexChanged.connect(self._select_response)
            self.units_combo.currentIndexChanged.connect(self._update_units)
            self._update_units()

        def _vector_fields(self, default, prefix):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            fields = []
            for axis, value in zip("XYZ", default):
                label = QLabel(axis)
                field = QLineEdit(str(value))
                field.setAccessibleName(f"{prefix} {axis}")
                field.setPlaceholderText("0")
                label.setBuddy(field)
                layout.addWidget(label)
                layout.addWidget(field, 1)
                fields.append(field)
            return widget, fields

        def _build_point_fields(self, layout):
            form = QFormLayout()
            position, self.position_fields = self._vector_fields((0, 0, 0), "Position")
            normal, self.normal_fields = self._vector_fields((0, 0, 1), "Outward normal")
            roll, self.roll_fields = self._vector_fields((1, 0, 0), "Roll reference")
            form.addRow("First position", position)
            form.addRow("Outward normal", normal)
            form.addRow("Roll reference", roll)
            layout.addLayout(form)
            orientation = QLabel("Normal points out of the skin. Roll reference sets the feature's local +X direction in the skin plane; it must not be parallel to the normal.")
            orientation.setWordWrap(True)
            layout.addWidget(orientation)
            repeat_group = QGroupBox("Repeat in a straight row")
            repeat = QFormLayout(repeat_group)
            self.count_spin = QSpinBox()
            self.count_spin.setRange(1, MAX_FEATURE_ROWS)
            self.count_spin.setValue(1)
            repeat.addRow("Number of features", self.count_spin)
            spacing, self.spacing_fields = self._vector_fields((0, 0, 0), "Repeat spacing")
            repeat.addRow("Step per feature", spacing)
            self.repeat_summary = QLabel()
            self.repeat_summary.setWordWrap(True)
            repeat.addRow(self.repeat_summary)
            self.count_spin.valueChanged.connect(self._update_repeat)
            layout.addWidget(repeat_group)
            self._update_repeat()

        def _build_line_fields(self, layout):
            self.vertices_table = QTableWidget(0, 6)
            self.vertices_table.setHorizontalHeaderLabels(["X", "Y", "Z", "Normal X", "Normal Y", "Normal Z"])
            self.vertices_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.vertices_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.vertices_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.vertices_table.setMinimumHeight(170)
            self.vertices_table.setAccessibleName("Ordered path vertices and outward normals")
            self._append_vertex((0, 0, 0, 0, 0, 1))
            self._append_vertex((1, 0, 0, 0, 0, 1))
            layout.addWidget(self.vertices_table)
            actions = QHBoxLayout()
            add = QPushButton("Add vertex")
            remove = QPushButton("Remove selected")
            add.clicked.connect(self._add_vertex)
            remove.clicked.connect(self._remove_vertices)
            actions.addWidget(add)
            actions.addWidget(remove)
            actions.addStretch(1)
            self.close_loop = QCheckBox("Close loop")
            self.close_loop.setToolTip("Add the final segment from the last vertex to the first; needs at least three vertices.")
            actions.addWidget(self.close_loop)
            layout.addLayout(actions)
            note = QLabel("Enter vertices in travel order and the outward skin normal at each vertex. A closed loop joins the last vertex to the first automatically.")
            note.setWordWrap(True)
            layout.addWidget(note)

        def _append_vertex(self, values):
            row = self.vertices_table.rowCount()
            self.vertices_table.insertRow(row)
            for column, value in enumerate(values):
                self.vertices_table.setItem(row, column, QTableWidgetItem(str(value)))
            self.vertices_table.setVerticalHeaderItem(row, QTableWidgetItem(str(row + 1)))

        def _add_vertex(self):
            if self.vertices_table.rowCount() > MAX_FEATURE_ROWS:
                self.error_label.setText(f"Use at most {MAX_FEATURE_ROWS:,} segments in one feature.")
                return
            row = self.vertices_table.rowCount() - 1
            # Keep the previous skin direction, but require a deliberate position.
            normals = [self.vertices_table.item(row, col).text() for col in range(3, 6)] if row >= 0 else ["0", "0", "1"]
            self._append_vertex(("", "", "", *normals))
            self.vertices_table.setCurrentCell(row + 1, 0)
            self.vertices_table.editItem(self.vertices_table.item(row + 1, 0))

        def _remove_vertices(self):
            selected = sorted({index.row() for index in self.vertices_table.selectedIndexes()}, reverse=True)
            if self.vertices_table.rowCount() - len(selected) < 2:
                self.error_label.setText("Keep at least two vertices for a line path.")
                return
            for row in selected:
                self.vertices_table.removeRow(row)
            for row in range(self.vertices_table.rowCount()):
                self.vertices_table.setVerticalHeaderItem(row, QTableWidgetItem(str(row + 1)))

        def _update_repeat(self):
            count = self.count_spin.value()
            for field in self.spacing_fields:
                field.setEnabled(count > 1)
            self.repeat_summary.setText("One feature at the first position." if count == 1 else f"{count:,} features: position + 0, 1, …, {count - 1:,} times the XYZ step. All share the same normal and roll.")

        def _update_units(self):
            units = self.units_combo.currentData() or "choose units above"
            if self.kind == "line":
                self.vertices_table.setHorizontalHeaderLabels([f"X ({units})", f"Y ({units})", f"Z ({units})", "Normal X", "Normal Y", "Normal Z"])

        def _set_preset_name(self, name):
            if not self.name_edit.text().strip() or self.name_edit.text() == self._last_preset_name:
                self.name_edit.setText(name)
            self._last_preset_name = name

        def _select_response(self):
            path = self.response_choice.currentData()
            if path:
                self.response_edit.setText(path)
                self._set_preset_name(self.response_choice.currentText())

        def _browse_response(self):
            path, _ = QFileDialog.getOpenFileName(self, "Choose feature response", self.response_edit.text(), "GRIM response (*.grim)")
            if path:
                self.response_edit.setText(path)

        @staticmethod
        def _field_values(fields):
            return [field.text().strip() for field in fields]

        def feature_definition(self) -> dict:
            """Return current validated authoring values; creates no files itself."""
            name = self.name_edit.text().strip()
            seed = feature_identifier(name)
            units = self.units_combo.currentData()
            if not units:
                raise ValueError("Choose placement units before adding this feature.")
            raw_path = self.response_edit.text().strip()
            if not raw_path:
                raise ValueError("Choose the feature's .grim response file.")
            path = Path(raw_path).expanduser()
            if path.suffix.lower() != ".grim" or not path.is_file():
                raise ValueError("Choose an existing .grim response file.")
            if self.kind == "point":
                rows = compose_point_rows(name, seed, self._field_values(self.position_fields),
                                          self._field_values(self.normal_fields), self._field_values(self.roll_fields),
                                          self.count_spin.value(), self._field_values(self.spacing_fields))
            else:
                values = []
                for row in range(self.vertices_table.rowCount()):
                    values.append([self.vertices_table.item(row, column).text().strip()
                                   if self.vertices_table.item(row, column) else "" for column in range(6)])
                rows = compose_line_rows(name, seed, [row[:3] for row in values],
                                         [row[3:] for row in values], self.close_loop.isChecked())
            return {"kind": self.kind, "name": name, "response": str(path.resolve()), "units": units, "rows": rows}

        def accept(self):
            try:
                self._definition = self.feature_definition()
            except (ValueError, OSError) as exc:
                self.error_label.setText(str(exc))
                return
            self.error_label.clear()
            super().accept()
else:
    class AddFeatureDialog:  # pragma: no cover - depends on the installation.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Feature authoring requires PySide6.") from _GUI_IMPORT_ERROR
