"""Sector statistics settings commit together and preserve angular ranges."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
        self.files = tempfile.TemporaryDirectory()
        self.addCleanup(self.files.cleanup)

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

    def csv_file(self, text=None):
        path = Path(self.files.name) / "sectors.csv"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        return path

    def load_csv(self, dialog, path):
        with mock.patch("GRIM_Backend.ui.sector_statistics_dialog.QFileDialog.getOpenFileName",
                        return_value=(str(path), "CSV files (*.csv)")):
            dialog.btn_load.click()

    def save_csv(self, dialog, path):
        with mock.patch("GRIM_Backend.ui.sector_statistics_dialog.QFileDialog.getSaveFileName",
                        return_value=(str(path), "CSV files (*.csv)")):
            dialog.btn_save.click()

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

    def test_csv_round_trip_preserves_wrapped_overlapping_and_gapped_rows_without_applying(self):
        dialog = self.dialog()
        rows = [(170, -170), (-45.25, 45.5), (-20, 10), (90, 125), (10, 370)]
        self.set_rows(dialog, rows)
        path = self.csv_file()
        self.save_csv(dialog, path)
        self.assertEqual(path.read_text(),
                         "start,stop\n170,-170\n-45.25,45.5\n-20,10\n90,125\n10,370\n")
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertEqual(self.changes, [])

        reopened = self.dialog(azimuths=[])
        self.load_csv(reopened, path)
        self.assertEqual(reopened.combo_mode.currentData(), "custom")
        np.testing.assert_allclose(np.asarray(reopened._rows(), dtype=float), rows)
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertEqual(self.changes, [])
        self.assertTrue(reopened.apply_settings())
        self.assertEqual(self.changes, ["sector"])
        self.assertTrue(reopened.apply_settings())
        self.assertEqual(self.changes, ["sector"])

    def test_csv_degrees_convert_to_and_from_radian_dialog(self):
        rows = [(170, -170), (-30, 60), (10, 370)]
        path = self.csv_file("start,stop\n170,-170\n-30,60\n10,370\n")
        dialog = self.dialog(unit="rad", azimuths=np.deg2rad(self.azimuths))
        self.load_csv(dialog, path)
        np.testing.assert_allclose(np.asarray(dialog._rows(), dtype=float), np.deg2rad(rows))
        self.save_csv(dialog, path)
        np.testing.assert_allclose(dialog._read_csv(path), rows)
        self.assertEqual(self.changes, [])
        self.assertTrue(dialog.apply_settings())
        sectors = parse_sectors(self.controls.sector_text("deg"), self.azimuths)
        np.testing.assert_allclose([sector.width for sector in sectors], [20, 90, 360])

    def test_csv_load_accepts_headerless_or_bom_header_and_blank_lines(self):
        for content in ("\n170,-170\n\n0,45\n", "\ufeff\n START , Stop \n170,-170\n , \n0,45\n"):
            with self.subTest(content=content):
                dialog = self.dialog()
                self.load_csv(dialog, self.csv_file(content))
                self.assertEqual(dialog._rows(), (("170", "-170"), ("0", "45")))
                self.assertFalse(dialog.error_label.text())
                self.assertEqual(self.changes, [])

    def test_csv_invalid_file_keeps_draft_rows_and_settings(self):
        invalid_files = ("", "start,stop\n", "0,10\n10\n", "0,10\n20,30,40\n",
                         "0,10\nwrong,30\n", "0,10\nnan,30\n", "0,10\n20,inf\n",
                         "0,10\n20,20\n", '0,10\n"20,30\n', "start,end\n0,30\n")
        for content in invalid_files:
            with self.subTest(content=content):
                dialog = self.dialog()
                self.set_rows(dialog, [(170, -170), (40, 90)])
                dialog.combo_statistic.setCurrentIndex(dialog.combo_statistic.findData("max"))
                before = (dialog._rows(), dialog._sector_signature(), self.state())
                self.load_csv(dialog, self.csv_file(content))
                self.assertEqual((dialog._rows(), dialog._sector_signature(), self.state()), before)
                self.assertEqual(dialog.combo_statistic.currentData(), "max")
                self.assertIn("Could not load sectors", dialog.error_label.text())
                self.assertEqual(self.changes, [])

    def test_csv_cancel_keeps_draft_and_does_not_apply(self):
        dialog = self.dialog()
        self.set_rows(dialog, [(170, -170)])
        before = (dialog._rows(), self.state())
        self.load_csv(dialog, "")
        self.save_csv(dialog, "")
        self.assertEqual((dialog._rows(), self.state()), before)
        self.assertEqual(self.changes, [])
        self.assertFalse(dialog.error_label.text())

    def test_csv_io_errors_keep_draft_and_show_inline_error(self):
        dialog = self.dialog()
        self.set_rows(dialog, [(170, -170)])
        before = (dialog._rows(), self.state())
        self.load_csv(dialog, self.csv_file())
        self.assertIn("Could not load sectors", dialog.error_label.text())
        self.save_csv(dialog, Path(self.files.name))
        self.assertIn("Could not save sectors", dialog.error_label.text())
        self.assertEqual((dialog._rows(), self.state()), before)
        self.assertEqual(self.changes, [])

    def test_csv_save_expands_selected_and_explicit_uniform_bounds(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                dialog = self.dialog()
                dialog.edit_width.setText("100")
                dialog.check_selected_range.setChecked(not explicit)
                dialog.edit_start.setText("-40")
                dialog.edit_stop.setText("190")
                signature = dialog._sector_signature()
                path = self.csv_file()
                self.save_csv(dialog, path)
                expected = [(-40, 60), (60, 160), (160, 190)] if explicit else [
                    (-180, -80), (-80, 20), (20, 120), (120, 180)]
                np.testing.assert_allclose(dialog._read_csv(path), expected)
                self.assertEqual(dialog._sector_signature(), signature)
                self.assertEqual(self.changes, [])

    def test_csv_save_invalid_draft_does_not_overwrite_existing_file(self):
        dialog = self.dialog()
        path = self.csv_file("start,stop\n0,45\n")
        self.set_rows(dialog, [(0, 10), (10, "bad")])
        self.save_csv(dialog, path)
        self.assertEqual(path.read_text(), "start,stop\n0,45\n")
        self.assertIn("Row 2 stop", dialog.error_label.text())
        self.assertEqual(self.changes, [])

    def test_csv_save_and_cancelled_load_finish_active_cell_without_applying(self):
        dialog = self.dialog()
        self.set_rows(dialog, [(0, 30)])
        dialog.show()
        dialog.activateWindow()
        self.app.processEvents()
        for column, text in ((0, "-20"), (1, "55")):
            dialog.table.setCurrentCell(0, column)
            dialog.table.editItem(dialog.table.item(0, column))
            self.app.processEvents()
            editor = self.app.focusWidget()
            self.assertIsInstance(editor, QLineEdit)
            editor.setText(text)
            if column == 0:
                path = self.csv_file()
                self.save_csv(dialog, path)
                self.assertEqual(dialog._read_csv(path), [(-20, 30)])
            else:
                self.load_csv(dialog, "")
            self.assertEqual(dialog.table.item(0, column).text(), text)
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertEqual(self.changes, [])

    def test_reject_after_csv_load_does_not_apply(self):
        dialog = self.dialog()
        self.load_csv(dialog, self.csv_file("0,45\n"))
        dialog.reject()
        self.assertEqual(self.state(), ("30", "mean", 90.0))
        self.assertEqual(self.changes, [])


if __name__ == "__main__":
    unittest.main()
