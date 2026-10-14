"""Analysis uses a centered canvas and releases its geometry when plots change."""
import unittest

import numpy as np
import test_inline_engineering_analysis as inline_tests
from test_engineering_tools import make_grid


class AnalysisLayoutTests(unittest.TestCase):
    setUp = inline_tests.InlineEngineeringAnalysisTests.setUp
    tearDown = inline_tests.InlineEngineeringAnalysisTests.tearDown
    setUpClass = inline_tests.InlineEngineeringAnalysisTests.__dict__["setUpClass"]
    tearDownClass = inline_tests.InlineEngineeringAnalysisTests.__dict__["tearDownClass"]

    def load_grid(self):
        frequencies = np.linspace(9., 10., 21)
        azimuths = np.linspace(-20., 20., 9)
        field = np.exp(2j * np.pi * frequencies[None, None, :, None]) * (
            2. + np.cos(np.deg2rad(azimuths[:, None, None, None] * 3)))
        grid = make_grid(field, frequencies=frequencies, azimuths=azimuths)
        self.window._add_dataset_row(grid, "Layout target", "Loaded", "layout.grim")
        self.window.table.selectRow(0)
        self.window.btn_auto_plot.setChecked(False)
        self.window.btn_auto_scale.setChecked(True)
        for listing in (self.window.list_az, self.window.list_elev,
                        self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        self.window.resize(1280, 800)
        self.window.show()
        return grid

    def settle(self):
        for _ in range(4):
            self.app.processEvents()
        self.window.plot_canvas.draw()

    def open_iq(self, mode="iq", linked=False):
        panel = self.window._show_engineering_tool("phase")
        panel.axis_combo.setCurrentIndex(panel.axis_combo.findData("frequency"))
        panel.phase_view.setCurrentIndex(panel.phase_view.findData(mode))
        panel.iq_controls.linked_checkbox.setChecked(linked)
        self.settle()
        self.assertIsNotNone(panel._iq_controller)
        return panel

    def test_single_iq_and_colorbar_stay_centered_without_distorting_iq_scale(self):
        self.load_grid()
        panel = self.open_iq()
        for options in (False, True):
            with self.subTest(options=options):
                panel.options_button.setChecked(options)
                self.settle()
                controller = panel._iq_controller
                ax, bar = controller.iq_axes, controller.colorbar.ax
                bounds, color_bounds = ax.get_position(), bar.get_position()
                self.assertEqual(ax.get_anchor(), "C")
                self.assertGreater(bounds.width, .15)
                self.assertLess(abs((bounds.x0 + color_bounds.x1) / 2 - .5), .1)
                self.assertLess(bounds.x1, color_bounds.x0)
                if not options:
                    self.assertEqual(panel.settings_area.verticalScrollBar().maximum(), 0)
                origin, real, imaginary = ax.transData.transform([[0., 0.], [1., 0.], [0., 1.]])
                self.assertAlmostEqual(np.linalg.norm(real - origin), np.linalg.norm(imaginary - origin), places=6)

    def test_ordinary_plots_recover_fresh_geometry_after_each_diagnostic_layout(self):
        self.load_grid()
        w = self.window
        for render in (w._plot_frequency, w._plot_azimuth_rect, w._plot_azimuth_polar,
                       w._plot_waterfall, w._clear_plot):
            w._clear_plot()
            render()
            self.settle()
            baseline = [(ax.name, ax.get_aspect(), ax.get_position().bounds) for ax in w.plot_figure.axes]
            for mode, linked in (("iq", False), ("iq", True), ("iq_repeat", False)):
                with self.subTest(plot=render.__name__, mode=mode, linked=linked):
                    panel = self.open_iq(mode, linked)
                    controller = panel._iq_controller
                    old_axes = tuple(controller.axes)
                    render()
                    self.settle()
                    self.assertTrue(controller.disconnected)
                    self.assertTrue(all(ax not in w.plot_figure.axes for ax in old_axes))
                    self.assertFalse(w.plot_figure._grim_engineering_layout)
                    self.assertEqual(len(w.plot_figure.axes), len(baseline))
                    for ax, (name, aspect, bounds) in zip(w.plot_figure.axes, baseline):
                        self.assertEqual(ax.name, name)
                        self.assertEqual(ax.get_aspect(), aspect)
                        np.testing.assert_allclose(ax.get_position().bounds, bounds, atol=.015)

    def test_hide_settings_preserves_plot_cursor_and_options_and_restores_them(self):
        self.load_grid()
        panel = self.open_iq(linked=True)
        panel.options_button.setChecked(True)
        panel.interpret_button.setChecked(True)
        panel.iq_controls.coordinate.setValue(9.65)
        panel.iq_controls.coordinate.editingFinished.emit()
        self.settle()
        controller, axes = panel._iq_controller, tuple(self.window.plot_figure.axes)
        coordinate = controller.selected_coordinate
        fields = [item["native_field"].copy() for item in controller.series]
        expanded_height = self.window.plot_canvas.height()
        self.assertGreaterEqual(expanded_height, 180)
        self.assertEqual((self.window.width(), self.window.height()), (1280, 800))
        self.assertLessEqual(panel.settings_area.height(), 190)
        self.assertGreater(panel.settings_area.verticalScrollBar().maximum(), 0)
        panel.settings_button.setChecked(False)
        self.settle()
        self.assertIs(panel._iq_controller, controller)
        self.assertFalse(controller.disconnected)
        self.assertEqual(tuple(self.window.plot_figure.axes), axes)
        self.assertEqual(controller.selected_coordinate, coordinate)
        self.assertEqual(panel.active_key, "phase")
        self.assertTrue(panel.header_controls.isHidden())
        self.assertTrue(panel.settings_area.isHidden())
        self.assertTrue(panel.options_button.isChecked())
        self.assertTrue(panel.interpret_button.isChecked())
        self.assertGreater(self.window.plot_canvas.height(), expanded_height + 100)
        for item, field in zip(controller.series, fields):
            np.testing.assert_array_equal(item["native_field"], field)
        # The interactive cursor also remains usable while settings are hidden.
        controller.select_coordinate(9.8)
        panel.settings_button.setChecked(True)
        self.settle()
        self.assertIs(panel._iq_controller, controller)
        self.assertAlmostEqual(panel.iq_controls.coordinate.value(), 9.8)
        self.assertTrue(panel.options.isVisible())
        self.assertTrue(panel.interpretation.isVisible())

    def test_collapsing_settings_from_results_shows_plot_without_losing_results(self):
        self.load_grid()
        panel = self.open_iq()
        panel.results_button.setChecked(True)
        self.settle()
        controller = panel._iq_controller
        rows = list(panel._panels["phase"][2].rows)
        panel.settings_button.setChecked(False)
        self.settle()
        self.assertTrue(self.window.plot_canvas.isVisible())
        self.assertTrue(panel.results.isHidden())
        self.assertEqual(list(panel._panels["phase"][2].rows), rows)
        panel.settings_button.setChecked(True)
        self.settle()
        self.assertIs(panel._iq_controller, controller)
        self.assertTrue(panel.results.isVisible())
        self.assertTrue(self.window.plot_canvas.isHidden())


if __name__ == "__main__":
    unittest.main()
