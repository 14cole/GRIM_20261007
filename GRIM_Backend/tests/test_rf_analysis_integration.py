"""Integration of engineer-facing analysis choices, actions, and replay."""
from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QCloseEvent

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets
import test_gui_shell as shell


class RfAnalysisIntegrationTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp
    tearDown = shell.UnifiedGuiShellTest.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def load_sample(self):
        power = np.broadcast_to(np.array([1., 2., 100.])[:, None, None, None], (3, 2, 4, 1)).copy()
        power[:, 1, :, :] *= 2.
        grid = RcsGrid([0., 1., 2.], [0., 10.], [8., 9., 10., 11.], ["HH"],
                       rcs_power=power, rcs_phase=np.zeros_like(power))
        self.window._add_dataset_row(grid, "Measured", "Loaded", "measured.grim")
        self.window.table.selectRow(0)
        for listing in (self.window.list_az, self.window.list_elev,
                        self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        self.window.btn_auto_plot.setChecked(False)
        return grid

    def test_analysis_actions_launch_from_both_tabs(self):
        with mock.patch.object(self.window, "_show_engineering_tool") as launch, \
                mock.patch("GRIM_Backend.ui.image_comparison.open_image_comparison") as images:
            for key in ("plotting", "isar"):
                self.window._activate_plot_tab(key)
                controls = self.window._plot_controls_by_tab[key]
                self.assertNotIn("engineering_analysis", controls)
                for role, tool in (("measure", "measure"), ("phase_analysis", "phase"),
                                   ("polarization", "polarization")):
                    controls[role].click()
                    launch.assert_called_with(tool)
                controls["image_compare"].click()
                images.assert_called_with(self.window)
        self.assertEqual(self.window._plot_controls_by_tab["plotting"]["waterfall"].text(), "Waterfall")

    def test_combined_operations_group_retains_all_requested_buttons(self):
        categories = [label.text() for label in self.window.findChildren(QLabel)
                      if label.objectName() == "opsCategory"]
        self.assertEqual(categories, ["Combine", "Transform", "Calibration, Geometry & Quality"])
        self.assertFalse(hasattr(self.window, "btn_support_reference"))
        for attribute in ("btn_range_cal", "btn_audit", "btn_compatibility",
                          "btn_provenance", "btn_axis_units", "btn_el_to_az360", "btn_swap_el_az",
                          "btn_sentri_elevation", "btn_extrusion", "btn_wedge_to_conic", "btn_phase_center"):
            self.assertIsNotNone(getattr(self.window, attribute))

    def test_analysis_preferences_survive_a_new_window(self):
        controls = self.window.analysis_controls
        controls.restore_analysis_options({"sweep": {"mode": "percentile", "percentile": 95.},
                                           "cdf": {"grouping": "sector", "sector_width_deg": 45.,
                                                   "weighting": "azimuth", "threshold_enabled": True,
                                                   "threshold": -10.}})
        self.window._on_analysis_setting_changed("sweep")
        saved = json.loads(self.app_settings.values["plotting/analysis-options-v1"])
        self.assertEqual(saved["analysis"]["sweep"], {"mode": "percentile", "percentile": 95.})
        restored = shell._RecordingWindow(settings=self.app_settings)
        try:
            self.assertEqual(restored.analysis_controls.analysis_options(), saved["analysis"])
            self.assertEqual(restored.image_analysis_controls.options(), saved["image"])
        finally:
            restored.deleteLater()
            self.app.processEvents()

    def test_maximum_aggregation_is_recorded_and_replays_the_same_curves(self):
        dataset = self.load_sample()
        self.window.analysis_controls.restore_analysis_options({"sweep": {"mode": "max", "percentile": 90.}})
        self.window._plot_frequency()
        spec = self.window.last_python_plot_spec
        self.assertEqual(spec[0], "supported")
        self.assertEqual(spec[4]["sweep_options"]["mode"], "max")
        replay = plot_datasets([("Measured", dataset)], mode=spec[3], **spec[4])
        self.assertEqual(len(replay.axes[0].lines), len(self.window.plot_ax.lines))
        for gui, scripted in zip(self.window.plot_ax.lines, replay.axes[0].lines):
            np.testing.assert_allclose(gui.get_ydata(), scripted.get_ydata())
            self.assertGreater(float(np.nanmin(gui.get_ydata())), 19.)

    def test_cdf_records_population_and_threshold_choices(self):
        dataset = self.load_sample()
        self.window.analysis_controls.restore_analysis_options({"cdf": {
            "grouping": "band", "band_count": 2, "weighting": "samples",
            "threshold_enabled": True, "threshold": 10., "exceedance": True}})
        self.window._plot_cdf()
        spec = self.window.last_python_plot_spec
        self.assertEqual(spec[0], "supported")
        self.assertEqual(spec[4]["cdf_options"]["grouping"], "band")
        replay = plot_datasets([("Measured", dataset)], mode=spec[3], **spec[4])
        self.assertEqual(len(replay.axes[0].lines), len(self.window.plot_ax.lines))

    def test_heatmap_axis_and_aggregation_choices_replay_and_persist(self):
        dataset = self.load_sample()
        options = {"axes": ["elevation", "frequency"], "aggregation": "mean_power",
                   "linked_cuts": False}
        self.window.image_analysis_controls.restore_options(options)
        self.window._plot_waterfall()
        spec = self.window.last_python_plot_spec
        self.assertEqual(spec[0], "supported")
        self.assertEqual(spec[4]["heatmap_options"], {**options, "horizontal_cuts":False,
                         "vertical_cuts":False, "cut_position":[10., 10.]})
        replay = plot_datasets([("Measured", dataset)], mode=spec[3], **spec[4])
        gui = self.window.plot_ax.collections[0].get_array()
        scripted = replay.axes[0].collections[0].get_array()
        np.testing.assert_allclose(gui, scripted)
        saved = json.loads(self.app_settings.values["plotting/analysis-options-v1"])
        self.assertEqual(saved["image"], {**options, "horizontal_cuts":False,
                         "vertical_cuts":False, "cut_position":None})

    def test_heatmap_settings_apply_even_when_line_hold_is_on(self):
        self.load_sample()
        self.window._plot_waterfall()
        self.window.btn_hold.setChecked(True)
        with mock.patch.object(self.window, "_render_plot_mode") as render:
            self.window._on_analysis_setting_changed("heatmap")
        render.assert_called_once_with("waterfall")

    def test_accepted_main_close_releases_image_captures_and_analysis_dialogs(self):
        from GRIM_Backend.ui.engineering_analysis import show_engineering_analysis
        from GRIM_Backend.ui.image_comparison import open_image_comparison
        self.load_sample()
        self.window._plot_waterfall()
        images = open_image_comparison(self.window)
        show_engineering_analysis(self.window, tab="phase")
        handles = [images.a.storage._file, images.b.storage._file]
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(all(handle.closed for handle in handles))
        self.assertIsNone(self.window._image_comparison_dialog)
        self.assertIsNone(self.window._engineering_analysis_dialog)


if __name__ == "__main__":
    unittest.main()
