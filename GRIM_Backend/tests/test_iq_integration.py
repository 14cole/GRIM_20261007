"""I/Q views retain physical coordinates and the existing workspace lifecycle."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtWidgets import QApplication

import test_inline_engineering_analysis as inline_tests
from test_engineering_tools import make_grid


class IQIntegrationTests(unittest.TestCase):
    setUp = inline_tests.InlineEngineeringAnalysisTests.setUp
    tearDown = inline_tests.InlineEngineeringAnalysisTests.tearDown
    load_sample = inline_tests.InlineEngineeringAnalysisTests.load_sample
    select_frequencies = inline_tests.InlineEngineeringAnalysisTests.select_frequencies

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def open_iq(self, mode="iq"):
        panel = self.window._show_engineering_tool("phase")
        panel.phase_view.setCurrentIndex(panel.phase_view.findData(mode))
        self.assertNotIn("unavailable", panel.summary.text().lower())
        self.assertIsNotNone(panel._iq_controller)
        return panel

    def test_linked_limits_and_native_exports_in_both_workspaces(self):
        self.load_sample()
        for workspace in ("plotting", "isar"):
            self.window._activate_plot_tab(workspace)
            canvas = self.window.plot_canvas
            panel = self.open_iq()
            self.assertIs(panel.canvas, canvas)
            self.assertTrue(panel.options.isHidden())
            self.assertTrue(panel.interpretation.isHidden())
            self.assertFalse(panel.iq_controls.cursor_widget.isHidden())
            panel.interpret_button.setChecked(True)
            self.assertIn("Distance from the origin", panel.interpretation.text())
            panel.iq_controls.linked_checkbox.setChecked(True)
            controller = panel._iq_controller
            self.assertEqual(len(controller.axes), 3)
            self.assertNotIn(controller.colorbar.ax, self.window.plot_axes)
            iq, magnitude, phase = controller.axes
            marker_lines = self.window._marker_lines()
            self.assertTrue(any(line.axes is iq for line in marker_lines))
            native_marker_line = next(line for line in marker_lines if line.axes is iq)
            mx, my = self.window._marker_arrays(native_marker_line)
            self.assertEqual(len(mx), 21)
            np.testing.assert_allclose(np.hypot(mx, my), 2.)
            self.assertGreater(magnitude.get_xlim()[0], 8.)
            self.assertLess(max(abs(v) for v in iq.get_xlim()), 4.)
            panel.iq_controls.coordinate.setValue(9.511)
            panel.iq_controls.coordinate.editingFinished.emit()
            self.assertAlmostEqual(controller.selected_coordinate, 9.5)
            panel._copy_samples("phase")
            rows = list(csv.DictReader(io.StringIO(QApplication.clipboard().text()), delimiter="\t"))
            self.assertEqual(len(rows), 21)
            selected = next(row for row in rows if float(row["Coordinate"]) == 9.5)
            self.assertEqual(selected["Field unit"], "m")
            self.assertAlmostEqual(float(selected["Amplitude (native)"]), 2.)
            self.assertEqual(int(selected["Native index"]), 10)

    def test_sparse_main_selection_preserves_source_indices_and_breaks_paths(self):
        self.load_sample()
        self.select_frequencies([2, 3, 14])
        panel = self.open_iq()
        series = panel._iq_controller.series[0]
        np.testing.assert_array_equal(series["native_indices"], [2, 3, 14])
        self.assertEqual([list(run) for run in series["runs"]], [[0, 1], [2]])
        panel._copy_samples("phase")
        rows = list(csv.DictReader(io.StringIO(QApplication.clipboard().text()), delimiter="\t"))
        self.assertEqual([int(row["Native index"]) for row in rows], [2, 3, 14])

    def test_comparisons_match_coordinates_and_keep_one_amplitude_scale(self):
        original = self.load_sample()
        reference = make_grid(.5, frequencies=np.linspace(9., 10., 11), polarizations=("HH", "VV", "HV", "VH"))
        self.window._add_dataset_row(reference, "Reference", "Loaded", "reference.grim")
        self.window.table.selectAll()
        panel = self.open_iq("iq_compare")
        panel.iq_controls.scale_combo.setCurrentIndex(panel.iq_controls.scale_combo.findData("common_peak"))
        left, right = panel._iq_controller.series
        self.assertEqual(len(left["x"]), 11)
        np.testing.assert_allclose(np.abs(left["field"]), 1.)
        np.testing.assert_allclose(np.abs(right["field"]), .25)
        panel.phase_view.setCurrentIndex(panel.phase_view.findData("iq_difference"))
        difference = panel._iq_controller.series[0]
        expected = original.rcs_slice((0, 0, slice(None, None, 2), 0)) - .5
        np.testing.assert_allclose(difference["native_field"], expected)

    def test_repeat_uses_one_sample_per_run_and_reports_mismatched_fixed_cut(self):
        self.load_sample()
        for name, grid in (("Repeat", make_grid(3., polarizations=("HH", "VV", "HV", "VH"))),
                           ("Other angle", make_grid(50., azimuths=(30.,), polarizations=("HH", "VV", "HV", "VH")))):
            self.window._add_dataset_row(grid, name, "Loaded", name + ".grim")
        self.window.table.selectAll()
        panel = self.open_iq("iq_repeat")
        controller = panel._iq_controller
        self.assertEqual(controller.repeat_report["count"], 2)
        self.assertIn("Skipped Other angle", panel._panels["phase"][3].text())
        self.assertIn("1 skipped", panel.iq_controls.readout.text())
        panel.iq_controls.coordinate.setValue(9.7)
        panel.iq_controls.coordinate.editingFinished.emit()
        panel._copy_samples("phase")
        rows = list(csv.DictReader(io.StringIO(QApplication.clipboard().text()), delimiter="\t"))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(float(row["Coordinate"]) == 9.7 for row in rows))

    def test_switching_rerender_and_clear_disconnect_old_cursor(self):
        self.load_sample()
        panel = self.open_iq()
        first = panel._iq_controller
        panel.iq_controls.guides_checkbox.setChecked(False)
        self.assertTrue(first.disconnected)
        second = panel._iq_controller
        panel.phase_view.setCurrentIndex(panel.phase_view.findData("phase"))
        self.assertTrue(second.disconnected)
        self.assertIsNone(panel._iq_controller)
        self.assertFalse(self.window.plot_figure._grim_independent_x)
        panel.phase_view.setCurrentIndex(panel.phase_view.findData("iq"))
        third = panel._iq_controller
        self.window._clear_plot()
        self.assertTrue(third.disconnected)
        self.assertIsNone(panel.active_key)
        self.assertIsNone(self.window.plot_figure._grim_iq_held)

    def test_compensation_export_preserves_original_native_fields(self):
        grid = self.load_sample()
        before = grid.rcs_slice((0, 0, slice(None), 0)).copy()
        panel = self.open_iq("iq_compensated")
        panel.iq_controls.delay_ns.setValue(.2)
        panel.iq_controls.delay_ns.editingFinished.emit()
        original, corrected = panel._iq_controller.series
        np.testing.assert_allclose(original["native_field"], before)
        np.testing.assert_allclose(corrected["native_field"], corrected["native_field"][0])
        np.testing.assert_allclose(grid.rcs_slice((0, 0, slice(None), 0)), before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iq.csv"
            with mock.patch("GRIM_Backend.ui.inline_engineering_analysis.QFileDialog.getSaveFileName", return_value=(str(path), "CSV")):
                panel._export_iq_samples()
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 42)
            self.assertIn("Polarization", rows[0])


if __name__ == "__main__":
    unittest.main()
