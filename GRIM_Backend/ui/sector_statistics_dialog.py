"""Edit uniform or individually defined azimuth sectors without live changes."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.plotting.modes.sector_stats_mode import STATISTICS


def _number_text(value: float) -> str:
    return np.format_float_positional(float(value), unique=True, trim="-")


class SectorStatisticsDialog(QDialog):
    """Keep edits local until Apply or OK, then update all settings together."""

    def __init__(self, controls, azimuths, *, unit="deg", parent=None):
        super().__init__(parent)
        self.controls = controls
        self.unit = unit
        self.azimuths = np.asarray(azimuths, dtype=float)
        self.period = float(common.convert_axis_values([360], "azimuth", "deg", unit)[0])
        self.setWindowTitle("Sector Statistics")
        self.resize(590, 600)
        layout = QVBoxLayout(self)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Uniform sectors", "uniform")
        self.combo_mode.addItem("Custom sectors", "custom")
        mode_row = QFormLayout()
        mode_row.addRow("Sector layout", self.combo_mode)
        layout.addLayout(mode_row)

        self.btn_uniform_setup = QPushButton("Fill from uniform sectors…")
        self.btn_uniform_setup.setCheckable(True)
        layout.addWidget(self.btn_uniform_setup)
        self.uniform_group = QGroupBox("Uniform sectors")
        uniform_layout = QFormLayout(self.uniform_group)
        self.edit_width = QLineEdit(_number_text(self.period / 12))
        self.check_selected_range = QCheckBox("Use the selected azimuth range")
        self.check_selected_range.setChecked(True)
        finite = self.azimuths[np.isfinite(self.azimuths)]
        start, stop = (float(finite.min()), float(finite.max())) if finite.size else (-self.period / 2, self.period / 2)
        self.edit_start = QLineEdit(_number_text(start))
        self.edit_stop = QLineEdit(_number_text(stop))
        uniform_layout.addRow(f"Width ({unit})", self.edit_width)
        uniform_layout.addRow(self.check_selected_range)
        bounds = QHBoxLayout()
        bounds.addWidget(QLabel(f"Start ({unit})"))
        bounds.addWidget(self.edit_start)
        bounds.addWidget(QLabel(f"Stop ({unit})"))
        bounds.addWidget(self.edit_stop)
        uniform_layout.addRow(bounds)
        self.btn_fill = QPushButton("Fill custom table from these values")
        uniform_layout.addRow(self.btn_fill)
        layout.addWidget(self.uniform_group)

        self.custom_group = QGroupBox("Custom sectors")
        custom_layout = QVBoxLayout(self.custom_group)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([f"Start azimuth ({unit})", f"Stop azimuth ({unit})"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setMinimumHeight(160)
        custom_layout.addWidget(self.table)
        row_buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add sector")
        self.btn_remove = QPushButton("Remove selected")
        row_buttons.addWidget(self.btn_add)
        row_buttons.addWidget(self.btn_remove)
        row_buttons.addStretch()
        custom_layout.addLayout(row_buttons)
        help_label = QLabel(
            "Each row is independent: widths, gaps, and overlaps are allowed. "
            "A stop below the start wraps through the azimuth seam. "
            "Custom sectors include both boundaries, so adjacent sectors share their edge sample."
        )
        help_label.setWordWrap(True)
        custom_layout.addWidget(help_label)
        layout.addWidget(self.custom_group, 1)

        self.combo_statistic = QComboBox()
        for key, label in STATISTICS:
            self.combo_statistic.addItem(label, key)
        self.combo_statistic.setCurrentIndex(max(0, self.combo_statistic.findData(controls.sector_statistic())))
        self.combo_statistic.setToolTip("Statistic of linear power inside each sector, displayed in dB.")
        self.spin_percentile = QDoubleSpinBox()
        self.spin_percentile.setRange(0, 100)
        self.spin_percentile.setDecimals(1)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.setSuffix(" %")
        self.spin_percentile.setKeyboardTracking(False)
        self.spin_percentile.setValue(controls.sector_percentile())
        self.spin_percentile.setToolTip(
            "Used for the Percentile plot statistic and the percentile column in Copy Sector Table."
        )
        statistics_layout = QFormLayout()
        statistics_layout.addRow("Plot statistic", self.combo_statistic)
        statistics_layout.addRow("Percentile", self.spin_percentile)
        layout.addLayout(statistics_layout)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c75050;")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        layout.addWidget(self.button_box)
        # Enter in a cell should finish that edit, never apply the whole dialog.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        self._loaded_text = controls.sector_text(unit=unit)
        self._load_sectors(self._loaded_text)
        self._loaded_signature = self._sector_signature()
        self.combo_mode.currentIndexChanged.connect(self._update_mode)
        self.btn_uniform_setup.toggled.connect(self._update_uniform_visibility)
        self.check_selected_range.toggled.connect(self._update_range)
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_rows)
        self.btn_fill.clicked.connect(self._fill_table)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.table.itemChanged.connect(self._clear_error)
        for edit in (self.edit_width, self.edit_start, self.edit_stop):
            edit.textChanged.connect(self._clear_error)
        self._update_mode()
        self._update_range()

    def _load_sectors(self, text):
        parts = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
        if len(parts) == 1 and ":" not in parts[0]:
            self.edit_width.setText(parts[0])
        elif len(parts) == 1 and parts[0].count(":") == 2:
            start, width, stop = parts[0].split(":")
            self.edit_start.setText(start.strip())
            self.edit_width.setText(width.strip())
            self.edit_stop.setText(stop.strip())
            self.check_selected_range.setChecked(False)
        else:
            self.combo_mode.setCurrentIndex(self.combo_mode.findData("custom"))
            for part in parts:
                bounds = part.split(":", 1)
                self._append_row(bounds[0].strip(), bounds[1].strip() if len(bounds) == 2 else "")
        if not self.table.rowCount():
            self._append_row("", "")

    def _update_mode(self, *_args):
        custom = self.combo_mode.currentData() == "custom"
        self.custom_group.setVisible(custom)
        self.btn_uniform_setup.setVisible(custom)
        self.btn_fill.setVisible(custom)
        self.uniform_group.setTitle("Fill table with uniform sectors" if custom else "Uniform sectors")
        self._update_uniform_visibility()
        self._clear_error()

    def _update_uniform_visibility(self, *_args):
        custom = self.combo_mode.currentData() == "custom"
        self.uniform_group.setVisible(not custom or self.btn_uniform_setup.isChecked())
        self.btn_uniform_setup.setText(
            "Hide uniform sector setup" if self.btn_uniform_setup.isChecked() else "Fill from uniform sectors…"
        )

    def _update_range(self, *_args):
        enabled = not self.check_selected_range.isChecked()
        self.edit_start.setEnabled(enabled)
        self.edit_stop.setEnabled(enabled)
        self._clear_error()

    def _append_row(self, start, stop):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(start)))
        self.table.setItem(row, 1, QTableWidgetItem(str(stop)))

    def _add_row(self):
        self._append_row("", "")
        self.table.setCurrentCell(self.table.rowCount() - 1, 0)
        self.table.editItem(self.table.currentItem())

    def _remove_rows(self):
        self._finish_editing()
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if not rows and self.table.currentRow() >= 0:
            rows.add(self.table.currentRow())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
        self._clear_error()

    def _clear_error(self, *_args):
        self.error_label.clear()
        self.error_label.hide()

    def _show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()

    def _finish_editing(self):
        editor = QApplication.focusWidget()
        if isinstance(editor, QLineEdit) and self.table.isAncestorOf(editor):
            self.table.commitData(editor)
            self.table.closeEditor(editor, QAbstractItemDelegate.NoHint)
        self.spin_percentile.interpretText()

    @staticmethod
    def _finite_number(text, label):
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label}: enter a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}: enter a finite number.")
        return value

    def _uniform_text(self):
        width = self.edit_width.text().strip()
        if self._finite_number(width, "Width") <= 0:
            raise ValueError("Width must be greater than zero.")
        if self.check_selected_range.isChecked():
            return width
        start, stop = self.edit_start.text().strip(), self.edit_stop.text().strip()
        self._finite_number(start, "Start")
        self._finite_number(stop, "Stop")
        return f"{start}:{width}:{stop}"

    def _rows(self):
        return tuple(tuple(self.table.item(row, col).text().strip() if self.table.item(row, col) else ""
                           for col in range(2)) for row in range(self.table.rowCount()))

    def _sector_signature(self):
        if self.combo_mode.currentData() == "custom":
            return "custom", self._rows()
        selected = self.check_selected_range.isChecked()
        return "uniform", self.edit_width.text(), selected, (() if selected else (self.edit_start.text(), self.edit_stop.text()))

    def _custom_text(self):
        sectors = []
        for row, (start, stop) in enumerate(self._rows(), 1):
            if not start and not stop:
                continue
            first = self._finite_number(start, f"Row {row} start")
            last = self._finite_number(stop, f"Row {row} stop")
            if first == last:
                raise ValueError(f"Row {row}: start and stop must differ.")
            sectors.append(f"{start}:{stop}")
        if not sectors:
            raise ValueError("Add at least one sector with start and stop azimuths.")
        return ", ".join(sectors)

    def _fill_table(self):
        self._finish_editing()
        try:
            sectors = common.parse_sectors(self._uniform_text(), self.azimuths, period=self.period)
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return
        self.table.setRowCount(0)
        for sector in sectors:
            self._append_row(_number_text(sector.start), _number_text(sector.start + sector.width))
        self.combo_mode.setCurrentIndex(self.combo_mode.findData("custom"))
        self._clear_error()

    def apply_settings(self) -> bool:
        self._finish_editing()
        try:
            text = self._custom_text() if self.combo_mode.currentData() == "custom" else self._uniform_text()
            if self._sector_signature() == self._loaded_signature:
                text = self._loaded_text
            selected = self.azimuths
            if (
                self.combo_mode.currentData() == "uniform"
                and self.check_selected_range.isChecked()
                and not np.isfinite(selected).any()
            ):
                # A width can be configured before loading/selecting data.
                # Its range will come from the selection when plotting.
                selected = [0.0]
            sectors = common.parse_sectors(text, selected, period=self.period)
            if not sectors:
                raise ValueError("Add at least one sector.")
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return False
        self.controls.set_sector_settings(
            text, str(self.combo_statistic.currentData()), float(self.spin_percentile.value()), unit=self.unit
        )
        self._loaded_text = text
        self._loaded_signature = self._sector_signature()
        self._clear_error()
        return True

    def accept(self):
        if self.apply_settings():
            super().accept()
