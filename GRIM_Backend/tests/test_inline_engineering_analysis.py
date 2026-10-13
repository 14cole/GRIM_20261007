"""Engineering diagnostics use the active workspace and its existing canvas."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog

import test_gui_shell as shell
from test_engineering_tools import make_grid


class InlineEngineeringAnalysisTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp
    tearDown = shell.UnifiedGuiShellTest.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        # Match the application's Windows font instead of offscreen Qt's
        # wider platform fallback, as the workspace layout suite does.
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if font_path.exists():
            families = QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(str(font_path)))
            if families:
                cls.app.setFont(QFont(families[0], 9))

    @classmethod
    def tearDownClass(cls):
        cls.app.setFont(cls.original_font)

    def load_sample(self):
        frequencies = np.linspace(9., 10., 21)
        channels = np.array([2 * np.exp(.3j), np.exp(.1j),
                             .2 * np.exp(.5j), .2 * np.exp(.5j)])
        field = (np.exp(-2j * np.pi * frequencies * .2)[None, None, :, None]
                 * channels[None, None, None, :])
        grid = make_grid(field, polarizations=("HH", "VV", "HV", "VH"),
                         units={"time_convention": "exp(+j omega t)"})
        self.window._add_dataset_row(grid, "Target", "Loaded", "target.grim")
        self.window.table.selectRow(0)
        for listing in (self.window.list_az, self.window.list_elev,
                        self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        self.window.btn_auto_plot.setChecked(False)
        return grid

    def native_trace(self):
        x = np.arange(101.)
        y = np.zeros_like(x)
        y[[21, 51, 71]] = [30., 100., 50.]
        ax = self.window.plot_ax
        ax.set(xlabel="Azimuth (deg)", ylabel="RCS (dBsm)")
        line, = ax.plot(x[::10], y[::10], label="Native narrow peaks")
        line._grim_dataset_key = ("native", "HH")
        line._grim_native_samples = SimpleNamespace(values=np.vstack((x, y)))
        return line

    def select_frequencies(self, indices):
        listing = self.window.list_freq
        previous = listing.blockSignals(True)
        try:
            listing.clearSelection()
            for index in indices:
                listing.item(index).setSelected(True)
        finally:
            listing.blockSignals(previous)

    def test_tool_launchers_use_the_existing_canvas_in_both_workspaces(self):
        self.load_sample()
        for tab in ("plotting", "isar"):
            self.window._activate_plot_tab(tab)
            canvas, figure = self.window.plot_canvas, self.window.plot_figure
            for key in ("measure", "phase", "polarization"):
                with self.subTest(tab=tab, tool=key):
                    panel = self.window._show_engineering_tool(key)
                    self.assertIs(self.window.plot_canvas, canvas)
                    self.assertIs(self.window.plot_figure, figure)
                    self.assertIs(panel, self.window.inline_analysis)
                    self.assertEqual(panel.active_key, key)
                    self.assertFalse(isinstance(panel, QDialog))
                    self.assertFalse(panel.isWindow())
                    self.assertIs(panel.window(), self.window)
                    self.assertIsNone(getattr(self.window, "_engineering_analysis_dialog", None))

    def test_measurement_uses_native_samples_without_replacing_the_plot(self):
        line = self.native_trace()
        canvas, figure, ax = self.window.plot_canvas, self.window.plot_figure, self.window.plot_ax
        displayed_x, displayed_y = line.get_xdata().copy(), line.get_ydata().copy()
        panel = self.window._show_engineering_tool("measure")
        self.assertIs(self.window.plot_canvas, canvas)
        self.assertIs(self.window.plot_figure, figure)
        self.assertIs(self.window.plot_ax, ax)
        self.assertIn(line, ax.lines)
        np.testing.assert_array_equal(line.get_xdata(), displayed_x)
        np.testing.assert_array_equal(line.get_ydata(), displayed_y)
        self.assertEqual(panel._measurement[2]["peak"]["x"], 51.)
        self.assertEqual(panel._measurement[2]["peak"]["y"], 100.)
        self.assertEqual(panel._measurement[2]["finite_count"], 101)

    def test_peak_search_and_cursor_region_use_native_trace_indices(self):
        line = self.native_trace()
        panel = self.window._show_engineering_tool("measure")
        with mock.patch.object(self.window, "_add_plot_marker") as marker:
            for _ in range(4):
                panel._mark_measurement("next")
            self.assertEqual([call.args for call in marker.call_args_list],
                             [(line, 21), (line, 51), (line, 71), (line, 21)])
            panel._select_region(60., 80.)
            self.assertEqual(panel._measurement[2]["finite_count"], 21)
            self.assertEqual(panel._measurement[2]["peak"]["x"], 71.)
            panel._mark_measurement("peak")
            marker.assert_called_with(line, 71)
        self.assertTrue(panel.use_region.isChecked())

    def test_phase_and_polarization_views_keep_numerical_results_copyable(self):
        self.load_sample()
        panel = self.window._show_engineering_tool("phase")
        canvas = self.window.plot_canvas
        for index in range(panel.phase_view.count()):
            with self.subTest(view=panel.phase_view.itemData(index)):
                panel.phase_view.setCurrentIndex(index)
                self.assertIs(panel._panels["phase"][0], self.window.plot_figure)
                self.assertIs(panel._panels["phase"][1], canvas)
                self.assertTrue(panel._panels["phase"][2].rows)
                self.assertTrue(panel._samples["phase"])
                self.assertTrue(self.window.plot_figure.axes)
                self.assertNotIn("unavailable", panel.summary.text().lower())
        delay = next(row for row in panel._panels["phase"][2].rows
                     if "apparent delay s" in row[0])
        self.assertAlmostEqual(float(delay[1]), .2e-9, delta=1e-18)
        panel._panels["phase"][2].copy()
        self.assertIn("apparent delay s", QApplication.clipboard().text())
        self.assertFalse(panel.results_button.isChecked())
        self.assertFalse(panel.options_button.isChecked())

        self.assertIs(self.window._show_engineering_tool("polarization"), panel)
        for index in range(panel.pol_view.count()):
            with self.subTest(view=panel.pol_view.itemData(index)):
                panel.pol_view.setCurrentIndex(index)
                self.assertEqual(len(self.window.plot_figure.axes), 2)
                self.assertTrue(panel._panels["polarization"][2].rows)
                self.assertTrue(panel._samples["polarization"])
                self.assertNotIn("unavailable", panel.summary.text().lower())
        panel._copy_samples("polarization")
        copied = QApplication.clipboard().text()
        self.assertTrue(copied.startswith("Series\tX\tValue\tUnit"))
        self.assertGreater(len(copied.splitlines()), 20)
        phase_difference = panel._samples["polarization"][0][2]
        np.testing.assert_allclose(phase_difference, np.rad2deg(.2))

    def test_polarization_phase_summary_respects_wrap_and_undefined_direction(self):
        field = np.ones((1, 1, 2, 2), complex)
        field[0, 0, :, 0] = np.exp(1j * np.deg2rad([179., -179.]))
        grid = make_grid(field, frequencies=[9., 10.], polarizations=("HH", "VV"))
        self.window.btn_auto_plot.setChecked(False)
        self.window._add_dataset_row(grid, "Wrapped", "Loaded", "wrapped.grim")
        self.window.table.selectRow(0)
        self.window.list_freq.selectAll()
        panel = self.window._show_engineering_tool("polarization")
        panel.pol_view.setCurrentIndex(panel.pol_view.findData("phase"))
        panel._update_current()

        def phase_summary():
            rows = panel._panels["polarization"][2].rows
            self.assertFalse(any(row[0].endswith(" median") for row in rows))
            return next(row for row in rows if row[0].endswith(" phase difference"))

        self.assertAlmostEqual(abs(float(phase_summary()[1])), 180.)
        panel._panels["polarization"][2].copy()
        self.assertNotIn("median", QApplication.clipboard().text())
        self.assertIn("circular mean", QApplication.clipboard().text())

        # Opposite channel phase differences have no single average direction.
        grid.rcs_phase[0, 0, :, 0] = np.deg2rad([90., -90.])
        panel._update_current()
        self.assertEqual(phase_summary()[1], "not defined")

        # A power-ratio median remains meaningful and available in ratio views.
        panel.pol_view.setCurrentIndex(panel.pol_view.findData("ratios"))
        panel._update_current()
        rows = panel._panels["polarization"][2].rows
        median = next(row for row in rows if row[0].endswith(" median"))
        self.assertEqual(float(median[1]), 0.)
        self.assertEqual(median[2], "dB power ratio")

    def test_analysis_axes_keep_distinct_ordinates_when_limits_change(self):
        self.load_sample()
        self.window._show_engineering_tool("phase")
        top, bottom = self.window.plot_figure.axes
        before_bottom = bottom.get_ylim()
        self.window._fit_y()
        self.assertLess(top.get_ylim()[1], 10.)
        self.assertGreater(bottom.get_ylim()[1], 50.)
        self.window.spin_plot_ymin.blockSignals(True)
        self.window.spin_plot_ymax.blockSignals(True)
        self.window.spin_plot_ymin.setValue(3.)
        self.window.spin_plot_ymax.setValue(9.)
        self.window.spin_plot_ymin.blockSignals(False)
        self.window.spin_plot_ymax.blockSignals(False)
        self.window._apply_plot_limits()
        np.testing.assert_allclose(top.get_ylim(), [3., 9.])
        np.testing.assert_allclose(bottom.get_ylim(), before_bottom)
        # Zooming the phase subplot should target that ordinate only.
        bottom.set_xlim(9.2, 9.8)
        bottom.set_ylim(-160., 160.)
        self.window._sync_plot_limit_spins_from_axes(bottom, sync_x=True, sync_y=True)
        self.window._apply_plot_limits()
        np.testing.assert_allclose(top.get_ylim(), [3., 9.])
        np.testing.assert_allclose(bottom.get_ylim(), [-160., 160.])

    def test_hold_compares_compatible_diagnostics_and_resets_for_a_new_view(self):
        self.load_sample()
        reference = make_grid(.5 * np.exp(.7j), polarizations=("HH", "VV", "HV", "VH"))
        self.window._add_dataset_row(reference, "Reference", "Loaded", "reference.grim")
        self.window.table.selectAll()
        self.window.btn_auto_scale.setChecked(True)
        panel = self.window._show_engineering_tool("phase")
        self.assertEqual(panel.dataset_combo.count(), 2)
        axes = tuple(self.window.plot_figure.axes)
        first = axes[0].lines[0]
        original = first.get_ydata().copy()
        self.window.btn_hold.setChecked(True)
        panel.dataset_combo.setCurrentIndex(1)
        self.assertEqual(tuple(self.window.plot_figure.axes), axes)
        self.assertEqual(len(axes[0].lines), 2)
        self.assertEqual(len(axes[1].lines), 4)
        self.assertIn(first, axes[0].lines)
        np.testing.assert_array_equal(first.get_ydata(), original)
        self.assertGreater(np.max(original), 5.)
        self.assertLess(np.max(axes[0].lines[-1].get_ydata()), -5.)
        self.assertLess(axes[0].get_ylim()[0], -6.)
        self.assertGreater(axes[0].get_ylim()[1], 6.)
        panel.phase_view.setCurrentIndex(panel.phase_view.findData("iq"))
        self.assertEqual(len(self.window.plot_figure.axes), 1)
        self.assertNotIn(axes[0], self.window.plot_figure.axes)

    def test_manual_limits_survive_analysis_update_when_auto_scale_is_off(self):
        self.load_sample()
        panel = self.window._show_engineering_tool("phase")
        self.window.btn_auto_scale.setChecked(False)
        top, bottom = self.window.plot_figure.axes
        top.set_xlim(9.1, 9.7)
        top.set_ylim(3., 9.)
        bottom.set_ylim(-40., 160.)
        self.window._sync_plot_limit_spins_from_axes(top, sync_x=True, sync_y=True)
        panel.refresh_from_selection()
        top, bottom = self.window.plot_figure.axes
        np.testing.assert_allclose(top.get_xlim(), [9.1, 9.7])
        np.testing.assert_allclose(bottom.get_xlim(), [9.1, 9.7])
        np.testing.assert_allclose(top.get_ylim(), [3., 9.])
        np.testing.assert_allclose(bottom.get_ylim(), [-40., 160.])

    def test_marker_tracks_replaced_diagnostic_curve_and_can_still_step(self):
        self.load_sample()
        panel = self.window._show_engineering_tool("phase")
        line = self.window.plot_figure.axes[0].lines[0]
        marker = self.window._add_plot_marker(line, 5)
        self.window.btn_hold.setChecked(True)
        panel.refresh_from_selection()
        self.assertIn(marker, self.window._plot_marker_list())
        current = marker["line"]
        self.assertIn(current.axes, self.window.plot_figure.axes)
        self.assertIn(current, current.axes.lines)
        self.assertEqual(marker["index"], 5)
        self.assertTrue(self.window._on_marker_key(SimpleNamespace(key="right")))
        self.assertEqual(marker["index"], 6)

    def test_hiding_measurement_removes_interactions_but_preserves_data(self):
        line = self.native_trace()
        panel = self.window._show_engineering_tool("measure")
        self.assertFalse(panel.drag_region.isChecked())
        self.assertIsNone(panel._span)
        panel.drag_region.setChecked(True)
        self.assertIsNotNone(panel._span)
        self.assertTrue(panel._overlays)
        panel.deactivate()
        self.assertIsNone(panel.active_key)
        self.assertIsNone(panel._span)
        self.assertFalse(panel._overlays)
        self.assertTrue(panel.isHidden())
        self.assertEqual(list(self.window.plot_ax.lines), [line])

    def test_region_drag_yields_to_existing_zoom_pan_and_marker_tools(self):
        self.native_trace()
        self.window.show()
        panel = self.window._show_engineering_tool("measure")
        panel.drag_region.setChecked(True)
        self.window.plot_canvas.draw()
        px, py = self.window.plot_ax.transData.transform((40., 0.))
        event = MouseEvent("button_press_event", self.window.plot_canvas, px, py, button=1)
        self.assertFalse(panel._span.ignore(event))
        for button in (self.window.btn_zoom_box, self.window.btn_pan, self.window.btn_markers):
            with self.subTest(tool=button.text()):
                panel._update_current()
                button.setChecked(True)
                try:
                    self.assertTrue(panel._span is None or panel._span.ignore(event))
                finally:
                    button.setChecked(False)

    def test_released_native_samples_show_an_error_without_replacing_data(self):
        line = self.native_trace()
        panel = self.window._show_engineering_tool("measure")
        line._grim_native_samples.values = None
        panel._update_current()
        self.assertIn("released", panel.summary.text())
        self.assertFalse(panel._panels["measure"][2].rows)
        self.assertFalse(panel._samples["measure"])
        self.assertIsNone(panel._measurement)
        self.assertIn(line, self.window.plot_ax.lines)

    def test_standard_plots_and_clear_leave_no_active_analysis_or_stale_recipe(self):
        self.load_sample()
        self.window._plot_frequency()
        self.assertIsNotNone(self.window.last_python_plot_spec)
        panel = self.window._show_engineering_tool("phase")
        self.assertIsNone(self.window.last_python_plot_spec)
        self.window._plot_frequency()
        self.assertIsNone(panel.active_key)
        self.assertTrue(panel.isHidden())
        self.assertEqual(len(self.window.plot_figure.axes), 1)
        self.assertFalse(getattr(self.window.plot_figure, "_grim_independent_y", False))
        self.assertIsNotNone(self.window.last_python_plot_spec)
        self.window._show_engineering_tool("polarization")
        self.window._clear_plot()
        self.assertIsNone(panel.active_key)
        self.assertTrue(panel.isHidden())
        self.assertEqual(len(self.window.plot_figure.axes), 1)
        self.assertFalse(self.window.plot_ax.lines)
        self.assertIsNone(self.window.last_python_plot_spec)
        self.assertFalse(getattr(self.window.plot_figure, "_grim_independent_y", False))

    def test_plotting_and_isar_analysis_controls_keep_separate_state(self):
        self.load_sample()
        plotting = self.window._show_engineering_tool("phase")
        plotting.phase_view.setCurrentIndex(plotting.phase_view.findData("iq"))
        plotting_canvas, plotting_axes = self.window.plot_canvas, tuple(self.window.plot_figure.axes)
        self.window._activate_plot_tab("isar")
        isar = self.window._show_engineering_tool("polarization")
        self.assertIsNot(isar, plotting)
        self.assertIsNot(isar.canvas, plotting_canvas)
        self.assertEqual(isar.active_key, "polarization")
        isar_axes = tuple(self.window.plot_figure.axes)
        self.window._activate_plot_tab("plotting")
        self.assertIs(self.window.inline_analysis, plotting)
        self.assertIs(self.window.plot_canvas, plotting_canvas)
        self.assertEqual(tuple(self.window.plot_figure.axes), plotting_axes)
        self.assertEqual(plotting.phase_view.currentData(), "iq")
        self.window._clear_plot()
        self.window._activate_plot_tab("isar")
        self.assertIs(self.window.inline_analysis, isar)
        self.assertEqual(tuple(self.window.plot_figure.axes), isar_axes)
        self.assertEqual(isar.active_key, "polarization")

    def test_autoplot_refreshes_a_diagnostic_after_its_controls_are_hidden(self):
        dataset = self.load_sample()
        panel = self.window._show_engineering_tool("phase")
        panel.deactivate()
        self.select_frequencies([2, 5, 11])
        self.window._render_plot_mode(self.window.last_plot_mode)
        self.assertTrue(panel.isHidden())
        self.assertIsNone(panel.active_key)
        self.assertEqual(self.window.last_plot_mode, "engineering_phase")
        np.testing.assert_allclose(self.window.plot_figure.axes[0].lines[0].get_xdata(),
                                   dataset.frequencies[[2, 5, 11]])
        self.assertIsNone(self.window.last_python_plot_spec)

    def test_selected_sweep_snapshot_and_reference_matching_use_physical_coordinates(self):
        dataset = self.load_sample()
        reference_frequencies = np.linspace(9000., 10000., 41)
        reference_field = (2 * np.exp(.1j) * np.exp(
            -2j * np.pi * reference_frequencies / 1000. * .2))[None, None, :, None]
        reference = make_grid(reference_field, frequencies=reference_frequencies,
                              units={"frequency": "MHz"})
        self.window._add_dataset_row(reference, "Reference", "Loaded", "reference.grim")
        self.window.table.selectRow(0)
        self.window.table.setCurrentCell(0, 0)
        self.window.table.selectAll()
        self.assertIs(self.window.active_dataset, self.window._selected_datasets()[0][1])
        self.select_frequencies([2, 5, 11])
        panel = self.window._show_engineering_tool("phase")
        self.assertTrue(panel.selected_sweep.isChecked())
        panel.phase_view.setCurrentIndex(panel.phase_view.findData("residual"))
        expected = dataset.frequencies[[2, 5, 11]]
        np.testing.assert_allclose(panel._samples["phase"][0][1], expected)
        np.testing.assert_allclose(panel._samples["phase"][0][2], np.rad2deg(.2))
        # Options and view changes use the captured selection until Update.
        self.select_frequencies([0, 1])
        panel._update_current()
        np.testing.assert_allclose(panel._samples["phase"][0][1], expected)
        panel.refresh_from_selection()
        np.testing.assert_allclose(panel._samples["phase"][0][1], dataset.frequencies[:2])
        panel.selected_sweep.setChecked(False)
        np.testing.assert_allclose(panel._samples["phase"][0][1], dataset.frequencies)
        np.testing.assert_allclose(panel._samples["phase"][0][2], np.rad2deg(.2))
        panel._copy_samples("phase")
        self.assertEqual(len(QApplication.clipboard().text().splitlines()), 22)

    def test_native_cut_update_uses_the_new_dataset_instead_of_its_old_plot(self):
        self.load_sample()
        panel = self.window._show_engineering_tool("measure")
        self.assertIsNone(panel.trace_combo.currentData())
        original = panel._measurement[3]
        self.assertGreater(np.max(original.get_ydata()), 5.)
        reference = make_grid(.5 * np.exp(.7j), polarizations=("HH", "VV", "HV", "VH"))
        self.window._add_dataset_row(reference, "Reference", "Loaded", "reference.grim")
        self.window.table.selectRow(1)
        self.window.list_freq.selectAll()
        panel.refresh_from_selection()
        self.assertIsNone(panel.trace_combo.currentData())
        self.assertIsNot(panel._measurement[3], original)
        self.assertNotIn(original, self.window.plot_ax.lines)
        np.testing.assert_allclose(panel._measurement[1], 20 * np.log10(.5))
        self.assertIn("Reference", panel._samples["measure"][0][0])

    def test_default_controls_leave_room_for_the_existing_plot(self):
        self.load_sample()
        self.window.resize(1200, 680)
        self.window.show()
        for key in ("measure", "phase", "polarization"):
            with self.subTest(tool=key):
                panel = self.window._show_engineering_tool(key)
                self.app.processEvents()
                self.assertFalse(panel.options_button.isChecked())
                self.assertFalse(panel.results_button.isChecked())
                self.assertTrue(panel.options.isHidden())
                self.assertTrue(panel.results.isHidden())
                self.assertLess(panel.height(), 120)
                self.assertGreaterEqual(self.window.plot_canvas.height(), 180)
                self.assertEqual(self.window.width(), 1200)
                self.assertEqual(self.window.height(), 680)

    def test_results_share_the_plot_area_and_restore_its_canvas(self):
        self.load_sample()
        self.window.resize(1200, 680)
        self.window.show()
        panel = self.window._show_engineering_tool("phase")
        canvas = self.window.plot_canvas
        panel.options_button.setChecked(True)
        panel.results_button.setChecked(True)
        self.app.processEvents()
        self.assertFalse(panel.options_button.isChecked())
        self.assertTrue(canvas.isHidden())
        self.assertTrue(panel.results.isVisible())
        table = panel._panels["phase"][2]
        self.assertTrue(table.isVisible())
        self.assertGreaterEqual(table.height(), 110)
        self.assertGreaterEqual(panel.results.height(), 200)
        self.assertEqual((self.window.width(), self.window.height()), (1200, 680))
        panel.results_button.setChecked(False)
        self.app.processEvents()
        self.assertTrue(canvas.isVisible())
        self.assertTrue(panel.results.isHidden())
        for action in (panel.deactivate, self.window._clear_plot, self.window._plot_frequency):
            with self.subTest(action=action.__name__):
                self.window._show_engineering_tool("phase")
                panel.results_button.setChecked(True)
                self.app.processEvents()
                self.assertTrue(canvas.isHidden())
                action()
                self.app.processEvents()
                self.assertIs(self.window.plot_canvas, canvas)
                self.assertTrue(canvas.isVisible())
                self.assertTrue(panel.results.isHidden())
                self.assertIsNone(panel.active_key)


if __name__ == "__main__":
    unittest.main()
