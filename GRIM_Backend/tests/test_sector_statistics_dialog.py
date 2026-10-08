"""Sector statistics settings commit together and preserve angular ranges."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QLineEdit, QTableWidgetItem, QWidget

from GRIM_Backend.plotting.modes.common import parse_sectors
from GRIM_Backend.ui.analysis_controls import PlotAnalysisControls
from GRIM_Backend.ui.sector_statistics_dialog import SectorStatisticsDialog


class SectorStatisticsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controls = PlotAnalysisControls()
        self.azimuths = np.arange(-180.0, 181.0)
        self.changes = []
        self.controls.changed.connect(self.changes.append)
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.deleteLater()
        self.controls.deleteLater()
        self.app.processEvents()

    def dialog(self, *, unit="deg", azimuths=None):
        dialog = SectorStatisticsDialog(
            self.controls, self.azimuths if azimuths is None else azimuths, unit=unit,
        )
        self.dialogs.append(dialog)
        return dialog

    def set_rows(self, dialog, rows):
        dialog.combo_mode.setCurrentIndex(dialog.combo_mode.findData("custom"))
        dialog.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                dialog.table.setItem(row, column, QTableWidgetItem(str(value)))

    def state(self):
        return (self.controls.sector_text(), self.controls.sector_statistic(),
                self.controls.sector_percentile())

    def test_custom_nonuniform_wrapped_sectors_and_percentile_apply_together(self):
        dialog = self.dialog()
        self.set_rows(dialog, [(-45, 10), (10, 80), (170, -170)])
        dialog.combo_statistic.setCurrentIndex(dialog.combo_statistic.findData("percentile"))
        dialog.spin_percentile.setValue(73.5)
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertEqual(self.changes, [])

        self.assertTrue(dialog.apply_settings())
        self.assertEqual(self.changes, ["sector"])
        self.assertEqual(self.controls.sector_statistic(), "percentile")
        self.assertEqual(self.controls.sector_percentile(), 73.5)
        sectors = parse_sectors(self.controls.sector_text(), self.azimuths)
        self.assertEqual([sector.width for sector in sectors], [55, 70, 20])
        self.assertEqual([sector.label() for sector in sectors],
                         ["-45 to 10", "10 to 80", "170 to -170"])
        self.assertTrue(sectors[2].contains(np.array([-180, 180])).all())
        self.assertFalse(sectors[2].contains(np.array([0])).any())

        reopened = self.dialog()
        self.assertEqual(reopened.combo_mode.currentData(), "custom")
        self.assertEqual(reopened.table.rowCount(), 3)
        self.assertEqual(reopened.combo_statistic.currentData(), "percentile")
        self.assertEqual(reopened.spin_percentile.value(), 73.5)
        self.assertTrue(reopened.apply_settings())
        self.assertEqual(self.changes, ["sector"])

    def test_cancel_discards_unapplied_edits(self):
        self.controls.set_sector_settings("-90:0, 0:30", "median", 65.0)
        before = self.state()
        self.changes.clear()
        dialog = self.dialog()
        self.set_rows(dialog, [(-80, 55)])
        dialog.combo_statistic.setCurrentIndex(dialog.combo_statistic.findData("max"))
        dialog.spin_percentile.setValue(25)
        dialog.reject()
        self.assertEqual(self.state(), before)
        self.assertEqual(self.changes, [])

    def test_enter_finishes_cell_and_apply_commits_active_cell(self):
        dialog = self.dialog()
        self.set_rows(dialog, [(-45, 45)])
        dialog.show()
        dialog.activateWindow()
        self.app.processEvents()
        dialog.table.setCurrentCell(0, 0)
        dialog.table.editItem(dialog.table.item(0, 0))
        self.app.processEvents()
        editor = self.app.focusWidget()
        self.assertIsInstance(editor, QLineEdit)
        editor.setText("-30")
        QTest.keyClick(editor, Qt.Key.Key_Return)
        self.app.processEvents()
        self.assertEqual(dialog.table.item(0, 0).text(), "-30")
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertTrue(dialog.isVisible())

        dialog.table.setCurrentCell(0, 1)
        dialog.table.editItem(dialog.table.item(0, 1))
        self.app.processEvents()
        editor = self.app.focusWidget()
        self.assertIsInstance(editor, QLineEdit)
        editor.setText("60")
        self.assertTrue(dialog.apply_settings())
        sectors = parse_sectors(self.controls.sector_text(), self.azimuths)
        self.assertEqual(sectors[0].label(), "-30 to 60")
        self.assertEqual(self.changes, ["sector"])

    def test_invalid_custom_rows_do_not_commit_partial_settings(self):
        before = self.state()
        for rows in ([], [("", 20)], [("bad", 20)], [("nan", 20)], [(10, 10)]):
            with self.subTest(rows=rows):
                dialog = self.dialog()
                self.set_rows(dialog, rows)
                dialog.combo_statistic.setCurrentIndex(dialog.combo_statistic.findData("max"))
                self.assertFalse(dialog.apply_settings())
                self.assertTrue(dialog.error_label.text())
                self.assertEqual(self.state(), before)
                self.assertEqual(self.changes, [])

    def test_uniform_settings_keep_half_open_boundaries_on_apply(self):
        for text in ("30", "-180:30:180"):
            with self.subTest(text=text):
                self.controls.set_sector_settings(text, "mean", 90.0)
                self.changes.clear()
                dialog = self.dialog()
                self.assertEqual(dialog.combo_mode.currentData(), "uniform")
                self.assertTrue(dialog.apply_settings())
                self.assertEqual(self.controls.sector_text(), text)
                self.assertEqual(self.changes, [])
                sectors = parse_sectors(self.controls.sector_text(), self.azimuths)
                self.assertEqual(sum(int(s.contains(self.azimuths).sum()) for s in sectors),
                                 len(self.azimuths))

    def test_fill_uniform_ranges_then_edit_one_sector(self):
        dialog = self.dialog()
        dialog.edit_width.setText("90")
        dialog.btn_fill.click()
        self.assertEqual(dialog.combo_mode.currentData(), "custom")
        self.assertEqual(dialog.table.rowCount(), 4)
        dialog.table.item(0, 1).setText("-100")
        self.assertTrue(dialog.apply_settings())
        sectors = parse_sectors(self.controls.sector_text(), self.azimuths)
        self.assertEqual([sector.width for sector in sectors], [80, 90, 90, 90])

    def test_uniform_settings_can_be_saved_before_selecting_data(self):
        dialog = self.dialog(azimuths=[])
        dialog.edit_width.setText("15")
        self.assertTrue(dialog.apply_settings())
        self.assertEqual(self.controls.sector_text(), "15")
        dialog.edit_width.setText("0")
        self.assertFalse(dialog.apply_settings())
        self.assertEqual(self.controls.sector_text(), "15")

    def test_degree_ranges_display_in_radians_and_save_physical_angles(self):
        self.controls.set_sector_settings("170:-170, -45:45", "mean", 90.0, unit="deg")
        radians = np.deg2rad(self.azimuths)
        dialog = self.dialog(unit="rad", azimuths=radians)
        np.testing.assert_allclose(
            [[float(dialog.table.item(r, c).text()) for c in range(2)] for r in range(2)],
            np.deg2rad([[170, -170], [-45, 45]]), rtol=1e-10,
        )
        self.assertIn("rad", dialog.table.horizontalHeaderItem(0).text())
        self.set_rows(dialog, [(-np.pi / 6, np.pi / 3)])
        self.assertTrue(dialog.apply_settings())
        sectors = parse_sectors(self.controls.sector_text("deg"), self.azimuths)
        self.assertAlmostEqual(sectors[0].start, -30.0)
        self.assertAlmostEqual(sectors[0].width, 90.0)
        reopened = self.dialog(unit="deg")
        self.assertAlmostEqual(float(reopened.table.item(0, 0).text()), -30.0)
        self.assertAlmostEqual(float(reopened.table.item(0, 1).text()), 60.0)

    def test_plot_settings_no_longer_include_sector_rows(self):
        container = QWidget()
        layout = QGridLayout(container)
        self.controls.add_rows(layout, 0)
        labels = [label.text() for label in container.findChildren(QLabel)]
        self.assertNotIn("Sectors", labels)
        self.assertNotIn("Sector Statistic", labels)
        self.assertNotIn("Percentile", labels)
        self.assertIn("PbP Band", labels)
        self.assertIn("CDF", labels)
        container.deleteLater()

    def test_full_revolution_remains_full_after_radian_conversion(self):
        self.controls.set_sector_settings("10:370", "mean", 90.0, unit="deg")
        radians = np.deg2rad(self.azimuths)
        dialog = self.dialog(unit="rad", azimuths=radians)
        self.assertTrue(dialog.apply_settings())
        sectors = parse_sectors(self.controls.sector_text("rad"), radians, period=2 * np.pi)
        self.assertEqual(len(sectors), 1)
        self.assertAlmostEqual(sectors[0].width, 2 * np.pi)
        self.assertTrue(sectors[0].contains(radians).all())


if __name__ == "__main__":
    unittest.main()
