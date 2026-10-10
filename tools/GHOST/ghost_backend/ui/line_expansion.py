"""Desktop controls for line-expanding 2-D sections into a 3-D response."""

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar,
        QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PySide2.QtCore import QObject, QThread, Qt, Signal, Slot  # type: ignore
    from PySide2.QtWidgets import (  # type: ignore
        QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar,
        QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget,
    )

from ghost_backend.assembly.expand_wing_sections import (
    expand_wing_sections,
    read_path_points,
    sections_from_path,
)


_COORDINATE_UNITS = (
    ("inches (in)", "inches"),
    ("millimeters (mm)", "millimeters"),
    ("meters (m)", "meters"),
    ("feet (ft)", "feet"),
)
_GEOMETRY_UNITS = (("inches (in)", "inches"), ("meters (m)", "meters"))
_GRID_FIELDS = (
    ("frequencies_ghz", "Frequencies (GHz):"),
    ("azimuths_deg", "Azimuths (deg):"),
    ("elevations_deg", "Elevations (deg):"),
)


def parse_samples(text: 'str', label: 'str') -> 'Optional[Tuple[float, ...]]':
    """Blank, an increasing list, or an inclusive start:stop:step sweep."""

    raw = str(text).strip()
    if not raw:
        return None
    if ":" in raw:
        try:
            start, stop, step = (float(part) for part in raw.split(":"))
        except ValueError as exc:
            raise ValueError(f"{label} sweep must be start:stop:step.") from exc
        if (
            not all(math.isfinite(value) for value in (start, stop, step))
            or step <= 0.0
            or stop < start
        ):
            raise ValueError(
                f"{label} sweep needs a positive step and stop >= start."
            )
        count = int(math.floor((stop - start) / step + 1.0e-9)) + 1
        return tuple(start + index * step for index in range(count))
    try:
        values = tuple(float(part) for part in raw.replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated numbers.") from exc
    if not all(math.isfinite(value) for value in values) or any(
        a >= b for a, b in zip(values, values[1:])
    ):
        raise ValueError(f"{label} must be finite, unique and increasing.")
    return values


class _ExpansionWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    canceled = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        sections: 'List[Dict[str, Any]]',
        arguments: 'Dict[str, Any]',
        abort_event: 'threading.Event',
    ) -> 'None':
        super().__init__()
        self.sections = sections
        self.arguments = arguments
        self.abort_event = abort_event

    def _report(self, done: 'int', total: 'int', message: 'str') -> 'None':
        percent = int(round(100.0 * int(done) / max(1, int(total))))
        self.progress.emit(max(0, min(100, percent)), str(message))

    @Slot()
    def run(self) -> 'None':
        try:
            result = expand_wing_sections(
                self.sections,
                cancel_check=self.abort_event.is_set,
                progress_callback=self._report,
                **self.arguments,
            )
        except InterruptedError as exc:
            self.canceled.emit(str(exc) or "Line expansion cancelled.")
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)
        else:
            self.finished.emit(result)


class LineExpansionTab(QWidget):
    """Expand 2-D sections along straight lines and sum them into one GRIM."""

    files_exported = Signal(list, str)

    def __init__(self, parent: 'Optional[QWidget]' = None) -> 'None':
        super().__init__(parent)
        self._thread: 'Optional[QThread]' = None
        self._worker: 'Optional[_ExpansionWorker]' = None
        self._abort_event: 'Optional[threading.Event]' = None
        self._build_ui()

    def _build_ui(self) -> 'None':
        outer = QVBoxLayout(self)
        # The inactive page also contributes to its host tab stack's minimum
        # size. Keep the long editor scrollable instead of sizing the host to it.
        self.controls_scroll = QScrollArea(self)
        self.controls_scroll.setObjectName("ghostLineExpansionControlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setMinimumHeight(160)
        controls = QWidget(self.controls_scroll)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls_scroll.setWidget(controls)
        # Preserve the host's panel background instead of QScrollArea's
        # automatic widget fill, which can obscure light checkbox text.
        controls.setAutoFillBackground(False)
        self.controls_scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.controls_scroll, 1)
        help_label = QLabel(
            "Fast 3-D approximation from 2-D solves. Each row is one straight "
            "section: its .geo is solved as a stand-alone 2-D object and "
            "expanded along the line from start to end. All rows add "
            "coherently onto an optional base response, whose own grid is "
            "then used (leave the grid fields blank). The output of a BoR "
            "base can be chosen as the Assembly Body dataset to add point and "
            "line features on top. Start, end and "
            "normal use the placement frame (+y nose, +x right, +z up). Draw "
            "each .geo in the plane perpendicular to its line with its origin "
            "on that line: 2-D +y is the normal and 2-D +x is line direction "
            "\u00d7 normal. For a curved or twisted path, chain short rows end to "
            "end and give each an end normal (blank = same as its normal), or "
            "use Import path. Single bounce only: no end effects, no coupling "
            "or shadowing between sections or with the body.",
            self,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.section_table = QTableWidget(0, 13, self)
        self.section_table.setHorizontalHeaderLabels(
            [
                "Section .geo",
                "Start x", "Start y", "Start z",
                "End x", "End y", "End z",
                "Normal x", "Normal y", "Normal z",
                "End normal x", "End normal y", "End normal z",
            ]
        )
        self.section_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.section_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.section_table.setMinimumHeight(160)
        layout.addWidget(self.section_table, 1)
        row_actions = QHBoxLayout()
        self.add_button = QPushButton("Add section(s)\u2026", self)
        self.add_button.setToolTip(
            "Choose one or more 2-D .geo files; one row is added per file."
        )
        self.import_path_button = QPushButton("Import path\u2026", self)
        self.import_path_button.setToolTip(
            "Sweep one .geo along a curved path. The path file has one point "
            "per line, x y z or x y z nx ny nz, in the start/end units and "
            "placement frame; one chained row is added per pair of points."
        )
        self.remove_button = QPushButton("Remove selected", self)
        row_actions.addWidget(self.add_button)
        row_actions.addWidget(self.import_path_button)
        row_actions.addWidget(self.remove_button)
        row_actions.addStretch(1)
        layout.addLayout(row_actions)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.coordinate_units = QComboBox(self)
        for label, value in _COORDINATE_UNITS:
            self.coordinate_units.addItem(label, value)
        form.addRow("Start/end units:", self.coordinate_units)
        self.geometry_units = QComboBox(self)
        for label, value in _GEOMETRY_UNITS:
            self.geometry_units.addItem(label, value)
        form.addRow("Section .geo units:", self.geometry_units)
        self.mirror = QCheckBox("Also place the mirror image (x \u2192 \u2212x)", self)
        form.addRow("", self.mirror)
        self.angle_step = QDoubleSpinBox(self)
        self.angle_step.setDecimals(3)
        self.angle_step.setRange(0.001, 90.0)
        self.angle_step.setValue(0.5)
        self.angle_step.setSuffix(" deg")
        self.angle_step.setToolTip(
            "Angular step of each 2-D section solve over 0-360 deg. It must "
            "divide 180. A warning is reported when it is too coarse for the "
            "section size and frequency."
        )
        form.addRow("Section angle step:", self.angle_step)
        self.oblique = QCheckBox(
            "Oblique-incidence correction (PEC sections only)", self
        )
        self.oblique.setToolTip(
            "For looks tilted out of the plane normal to a section's line, "
            "use the 2-D solve at the reduced frequency f\u00b7cos(tilt) instead "
            "of the broadside response. Exact for an infinitely long PEC "
            "section; not valid for coated or dielectric sections. Adds 2-D "
            "solves."
        )
        form.addRow("", self.oblique)
        self.oblique_tilt = QDoubleSpinBox(self)
        self.oblique_tilt.setRange(1.0, 89.0)
        self.oblique_tilt.setValue(60.0)
        self.oblique_tilt.setSuffix(" deg")
        self.oblique_tilt.setToolTip(
            "Looks tilted further than this reuse the response at this tilt."
        )
        self.oblique_solves = QSpinBox(self)
        self.oblique_solves.setRange(2, 500)
        self.oblique_solves.setValue(24)
        self.oblique_solves.setToolTip(
            "Most reduced-frequency 2-D solves per section geometry and "
            "radar frequency. A warning is reported when this is too few."
        )
        oblique_row = QHBoxLayout()
        oblique_row.addWidget(QLabel("Tilt limit:", self))
        oblique_row.addWidget(self.oblique_tilt)
        oblique_row.addWidget(QLabel("Max solves:", self))
        oblique_row.addWidget(self.oblique_solves)
        oblique_row.addStretch(1)
        form.addRow("", oblique_row)
        self.body_edit = QLineEdit(self)
        self.body_edit.setPlaceholderText(
            "Optional coherent monostatic .grim (BoR body, Assembly output, "
            "3-D body); blank = sections only"
        )
        self.body_browse = QPushButton("Browse\u2026", self)
        body_row = QHBoxLayout()
        body_row.addWidget(self.body_edit, 1)
        body_row.addWidget(self.body_browse)
        form.addRow("Base dataset:", body_row)
        self.shadow = QCheckBox(
            "Shadow sections and corners behind the base's BoR body", self
        )
        self.shadow.setChecked(True)
        self.shadow.setToolTip(
            "Hide the parts of each line that the body blocks from the radar. "
            "Needs a base that carries an embedded BoR profile."
        )
        form.addRow("", self.shadow)
        self.grid_fields: 'Dict[str, QLineEdit]' = {}
        for key, label in _GRID_FIELDS:
            control = QLineEdit(self)
            control.setPlaceholderText(
                "Blank with a base; else list (1, 2, 3) or start:stop:step"
            )
            self.grid_fields[key] = control
            form.addRow(label, control)
        self.output_edit = QLineEdit(self)
        self.output_browse = QPushButton("Browse\u2026", self)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_browse)
        form.addRow("Output dataset:", output_row)
        layout.addLayout(form)

        corner_help = QLabel(
            "Corner estimates (optional): a rough physical-optics double "
            "bounce where a wing meets the body. The fold is the root line; "
            "the wing and body normals point out of the two faces into the "
            "corner; face width is how far the double bounce reaches along "
            "each face. Same frame and units as the sections. Its phase "
            "against the other terms is approximate.",
            self,
        )
        corner_help.setWordWrap(True)
        layout.addWidget(corner_help)
        self.corner_table = QTableWidget(0, 13, self)
        self.corner_table.setHorizontalHeaderLabels(
            [
                "Fold start x", "Fold start y", "Fold start z",
                "Fold end x", "Fold end y", "Fold end z",
                "Wing normal x", "Wing normal y", "Wing normal z",
                "Body normal x", "Body normal y", "Body normal z",
                "Face width",
            ]
        )
        self.corner_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.corner_table.setMaximumHeight(140)
        layout.addWidget(self.corner_table)
        corner_actions = QHBoxLayout()
        self.add_corner_button = QPushButton("Add corner", self)
        self.remove_corner_button = QPushButton("Remove selected corner", self)
        corner_actions.addWidget(self.add_corner_button)
        corner_actions.addWidget(self.remove_corner_button)
        corner_actions.addStretch(1)
        layout.addLayout(corner_actions)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Expand && save", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.progress, 1)
        outer.addLayout(action_row)
        # Results can include a station or warning per section. Bound their
        # height as well, while keeping every line selectable and reachable.
        self.status_scroll = QScrollArea(self)
        self.status_scroll.setObjectName("ghostLineExpansionStatusScroll")
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setFrameShape(QFrame.NoFrame)
        self.status_scroll.setMaximumHeight(96)
        self.status_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_scroll.setWidget(self.status_label)
        self.status_scroll.hide()
        outer.addWidget(self.status_scroll)

        self.add_button.clicked.connect(self._add_sections)
        self.import_path_button.clicked.connect(self._import_path)
        self.remove_button.clicked.connect(self._remove_sections)
        self.add_corner_button.clicked.connect(self._add_corner)
        self.remove_corner_button.clicked.connect(self._remove_corners)
        self.body_browse.clicked.connect(self._browse_body)
        self.output_browse.clicked.connect(self._browse_output)
        self.run_button.clicked.connect(self.expand_and_save)
        self.cancel_button.clicked.connect(self.request_cancel)
        self._input_widgets = (
            self.section_table, self.add_button, self.import_path_button,
            self.remove_button,
            self.coordinate_units, self.geometry_units, self.mirror,
            self.angle_step, self.body_edit, self.body_browse,
            self.output_edit, self.output_browse, *self.grid_fields.values(),
            self.oblique, self.oblique_tilt, self.oblique_solves, self.shadow,
            self.corner_table, self.add_corner_button,
            self.remove_corner_button,
        )

    def job_is_running(self) -> 'bool':
        """Include worker shutdown so a host cannot destroy the QThread."""

        return self._thread is not None

    @Slot(str)
    def _set_status(self, message: 'str') -> 'None':
        self.status_label.setText(message)
        self.status_scroll.setVisible(bool(message))
        self.status_scroll.verticalScrollBar().setValue(0)

    def _add_sections(self) -> 'None':
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose 2-D section geometry", "",
            "GHOST 2-D geometry (*.geo);;All files (*)",
        )
        for path in paths:
            # Start and end begin at the origin; the default normal is +z (up).
            self._append_row([str(path)] + ["0"] * 8 + ["1"] + [""] * 3)

    def _append_row(self, cells: 'List[str]') -> 'None':
        row = self.section_table.rowCount()
        self.section_table.insertRow(row)
        for column, text in enumerate(cells):
            self.section_table.setItem(row, column, QTableWidgetItem(text))

    def _import_path(self) -> 'None':
        geometry, _ = QFileDialog.getOpenFileName(
            self, "Choose the 2-D section swept along the path", "",
            "GHOST 2-D geometry (*.geo);;All files (*)",
        )
        if not geometry:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose path points (x y z [nx ny nz] per line)", "",
            "Path points (*.txt *.csv);;All files (*)",
        )
        if not path:
            return
        try:
            sections = sections_from_path(geometry, *read_path_points(path))
        except (OSError, ValueError) as exc:
            self._set_status(str(exc))
            return
        for section in sections:
            self._append_row(
                [str(geometry)]
                + [
                    format(float(value), ".12g")
                    for key in ("root", "tip", "normal", "normal_end")
                    for value in section[key]
                ]
            )
        self._set_status(
            f"Added {len(sections)} chained section(s) from {path}."
        )

    def _remove_sections(self) -> 'None':
        rows = {index.row() for index in self.section_table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.section_table.removeRow(row)

    def _add_corner(self) -> 'None':
        row = self.corner_table.rowCount()
        self.corner_table.insertRow(row)
        # Default: a horizontal wing (+z face) meeting a body side (+x face).
        cells = ["0"] * 6 + ["0", "0", "1"] + ["1", "0", "0"] + ["1"]
        for column, text in enumerate(cells):
            self.corner_table.setItem(row, column, QTableWidgetItem(text))

    def _remove_corners(self) -> 'None':
        rows = {index.row() for index in self.corner_table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.corner_table.removeRow(row)

    def corners(self) -> 'List[Dict[str, Any]]':
        """Corner table rows as the mappings the expansion consumes."""

        corners = []
        for row in range(self.corner_table.rowCount()):
            try:
                numbers = [
                    float(self.corner_table.item(row, column).text())
                    for column in range(13)
                ]
            except (AttributeError, ValueError) as exc:
                raise ValueError(
                    f"Corner {row + 1}: every cell must be a number."
                ) from exc
            corners.append(
                {
                    "fold_start": tuple(numbers[0:3]),
                    "fold_end": tuple(numbers[3:6]),
                    "n_wing": tuple(numbers[6:9]),
                    "n_body": tuple(numbers[9:12]),
                    "face_width": numbers[12],
                }
            )
        return corners

    def _browse_body(self) -> 'None':
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose BoR body GRIM", self.body_edit.text().strip(),
            "GRIM response (*.grim);;All files (*)",
        )
        if path:
            self.body_edit.setText(path)

    def _browse_output(self) -> 'None':
        path, _ = QFileDialog.getSaveFileName(
            self, "Save line-expanded GRIM", self.output_edit.text().strip(),
            "GRIM response (*.grim);;All files (*)",
        )
        if path:
            self.output_edit.setText(path)

    def sections(self) -> 'List[Dict[str, Any]]':
        """Table rows as the section mappings the expansion consumes."""

        sections = []
        for row in range(self.section_table.rowCount()):
            cells = []
            for column in range(13):
                item = self.section_table.item(row, column)
                cells.append("" if item is None else item.text().strip())
            if not cells[0]:
                raise ValueError(f"Section {row + 1}: choose a .geo file.")
            # A blank end normal keeps the section frame constant.
            has_end_normal = any(cells[10:13])
            try:
                numbers = [
                    float(cell)
                    for cell in cells[1:13 if has_end_normal else 10]
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Section {row + 1}: start, end and normals must be "
                    "numbers (leave all three end-normal cells blank to keep "
                    "the normal constant)."
                ) from exc
            section = {
                "geometry": cells[0],
                "root": tuple(numbers[0:3]),
                "tip": tuple(numbers[3:6]),
                "normal": tuple(numbers[6:9]),
            }
            if has_end_normal:
                section["normal_end"] = tuple(numbers[9:12])
            sections.append(section)
        if not sections:
            raise ValueError("Add at least one section.")
        return sections

    def _set_busy(self, busy: 'bool') -> 'None':
        for widget in self._input_widgets:
            widget.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setValue(0)

    @Slot()
    def expand_and_save(self) -> 'None':
        if self.job_is_running():
            return
        try:
            sections = self.sections()
            output = self.output_edit.text().strip()
            if not output:
                raise ValueError("Choose an output dataset.")
            arguments = dict(
                output_grim=output,
                coordinate_units=str(self.coordinate_units.currentData()),
                geometry_units=str(self.geometry_units.currentData()),
                body_grim=self.body_edit.text().strip() or None,
                mirror=self.mirror.isChecked(),
                section_angle_step_deg=self.angle_step.value(),
                shadow=self.shadow.isChecked(),
                oblique=self.oblique.isChecked(),
                oblique_max_tilt_deg=self.oblique_tilt.value(),
                oblique_max_solves=self.oblique_solves.value(),
                corners=self.corners(),
                **{
                    key: parse_samples(
                        self.grid_fields[key].text(), label.rstrip(":")
                    )
                    for key, label in _GRID_FIELDS
                },
            )
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self._abort_event = threading.Event()
        thread = QThread(self)
        worker = _ExpansionWorker(sections, arguments, self._abort_event)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.canceled.connect(self._set_status)
        worker.error.connect(self._set_status)
        for signal in (worker.finished, worker.canceled, worker.error):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread, self._worker = thread, worker
        self._set_busy(True)
        self._set_status("Solving 2-D sections and expanding\u2026")
        thread.start()

    @Slot()
    def request_cancel(self) -> 'None':
        if self._abort_event is not None:
            self._abort_event.set()
            self.cancel_button.setEnabled(False)
            self._set_status(
                "Cancelling after the current section; no partial output "
                "will be saved."
            )

    @Slot(int, str)
    def _on_progress(self, percent: 'int', message: 'str') -> 'None':
        self.progress.setValue(int(percent))
        self.progress.setFormat(f"{int(percent)}% \u00b7 {message}")

    @Slot(object)
    def _on_finished(self, result: 'Any') -> 'None':
        output = str(result["output"])
        lines = [f"Saved {output}"]
        lines += [str(value) for value in result.get("stations", ())]
        lines += ["\u26a0 " + str(value) for value in result.get("warnings", ())]
        self._set_status("\n".join(lines))
        self.files_exported.emit([output], "line expansion")

    @Slot()
    def _on_thread_finished(self) -> 'None':
        self._thread = None
        self._worker = None
        self._abort_event = None
        self._set_busy(False)
