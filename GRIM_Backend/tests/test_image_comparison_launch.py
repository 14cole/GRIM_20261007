"""Image comparison opens without replacing the engineer's main plot."""
from __future__ import annotations

import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from GRIM_Backend.datasets.grid import RcsGrid
import test_gui_shell as shell


class ImageComparisonLaunchTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        dialog = getattr(self.window, "_image_comparison_dialog", None)
        if dialog is not None:
            dialog.close()
        shell.UnifiedGuiShellTest.tearDown(self)

    def load_sample(self, elevations=(0.,)):
        values = np.arange(1., 7.).reshape(3, 1, 2, 1)
        power = np.broadcast_to(values, (3, len(elevations), 2, 1)).copy()
        for index in range(len(elevations)):
            power[:, index, :, :] *= 10. ** index
        dataset = RcsGrid([0., 1., 2.], elevations, [9., 10.], ["HH"],
                          rcs_power=power, rcs_phase=np.zeros_like(power))
        self.window._add_dataset_row(dataset, "Measured", "Loaded", "measured.grim")
        self.window.table.selectRow(0)
        for listing in (self.window.list_az, self.window.list_elev,
                        self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        self.window.btn_auto_plot.setChecked(False)
        return dataset

    def click_comparison(self):
        figure = self.window.plot_figure
        axes = list(figure.axes)
        artists = [(list(ax.lines), list(ax.images), list(ax.collections)) for ax in axes]
        mode = self.window.last_plot_mode
        script = self.window.python_recorder.script
        self.window._plot_controls_by_tab[self.window._active_plot_tab]["image_compare"].click()
        dialog = self.window._image_comparison_dialog
        self.assertIsNotNone(dialog)
        dialog.canvas.draw()
        self.assertIs(self.window.plot_figure, figure)
        self.assertEqual(list(figure.axes), axes)
        self.assertEqual([(list(ax.lines), list(ax.images), list(ax.collections)) for ax in axes], artists)
        self.assertEqual(self.window.last_plot_mode, mode)
        self.assertEqual(self.window.python_recorder.script, script)
        return dialog

    def test_open_from_frequency_plot_keeps_main_plot_and_explains_capture(self):
        self.load_sample()
        self.window._plot_frequency()
        dialog = self.click_comparison()
        self.assertIsNone(dialog.a)
        self.assertIsNone(dialog.b)
        self.assertIsNone(dialog.result)
        self.assertIn("Render a Waterfall", dialog.advisory.text())
        self.assertTrue(any("No image" in text.get_text()
                            for ax in dialog.figure.axes for text in ax.texts))

    def test_open_from_empty_plotting_and_isar_tabs_explains_capture(self):
        for tab in ("plotting", "isar"):
            with self.subTest(tab=tab):
                self.window._activate_plot_tab(tab)
                dialog = self.click_comparison()
                self.assertIn("Render a Waterfall", dialog.advisory.text())
                self.assertTrue(any("No image" in text.get_text()
                                    for ax in dialog.figure.axes for text in ax.texts))
                dialog.close()

    def test_single_waterfall_image_is_visible_before_b_is_captured(self):
        dataset = self.load_sample()
        self.window._plot_waterfall()
        dialog = self.click_comparison()
        self.assertIsNotNone(dialog.a)
        self.assertIsNone(dialog.b)
        self.assertIsNone(dialog.result)
        preview = next(ax for ax in dialog.figure.axes if ax.get_title().startswith("A:"))
        np.testing.assert_allclose(preview.collections[0].get_array(),
                                   10. * np.log10(dataset.rcs_power[:, 0, :, 0]).T)
        self.assertEqual(preview.collections[0].get_cmap().name, "turbo")
        self.assertTrue(any("Capture as B" in text.get_text()
                            for ax in dialog.figure.axes for text in ax.texts))
        # Comparison-only controls remain harmless during the first capture.
        dialog.combo_view.setCurrentIndex(1)
        self.assertFalse(dialog.timer.isActive())
        dialog.combo_view.setCurrentIndex(2)
        dialog.swipe.setValue(20)
        dialog.check_roi.setChecked(True)
        dialog.clear_roi()
        frozen_values = np.array(dialog.a.values)
        handle = dialog.a.storage._file
        dialog.swap()
        self.assertIsNone(dialog.a)
        np.testing.assert_array_equal(dialog.b.values, frozen_values)
        self.assertTrue(any(ax.get_title().startswith("B:") for ax in dialog.figure.axes))
        dialog.swap()
        # A remains frozen while the engineer changes the main image for B.
        dataset.rcs_power[:] *= 10.
        self.window._plot_waterfall()
        dialog.capture("b")
        self.assertIsNotNone(dialog.result)
        np.testing.assert_allclose(dialog.result["difference"], -10., atol=1e-12)
        np.testing.assert_array_equal(dialog.a.values, frozen_values)
        dialog.close()
        self.assertTrue(handle.closed)

    def test_two_existing_image_panels_open_as_a_and_b(self):
        self.load_sample(elevations=(0., 1.))
        self.window._plot_waterfall()
        dialog = self.click_comparison()
        self.assertIsNotNone(dialog.result)
        np.testing.assert_allclose(dialog.result["difference"], -10., atol=1e-12)
        self.assertEqual(len(dialog.image_axes), 3)


if __name__ == "__main__":
    unittest.main()
