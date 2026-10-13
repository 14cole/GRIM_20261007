"""Exercise I/Q presentation and interactions against real Qt/Matplotlib objects."""
from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace
import unittest

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QWidget

from GRIM_Backend.plotting.engineering_tools import EngineeringCut
from GRIM_Backend.plotting.markers import PlotMarkersMixin
from GRIM_Backend.ui.engineering_analysis import ResultTable
from GRIM_Backend.ui.iq_analysis import IQControls, render_iq, _overview_indices


def cut(fields, x=None, label="A"):
    field = np.asarray(fields, dtype=complex)
    x = np.asarray(x if x is not None else np.arange(len(field)) + 9., dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20 * np.log10(np.abs(field))
    return EngineeringCut(x, np.abs(field) ** 2, field, db, "frequency", "GHz", "dBsm", label, "+jwt", [])


class IQRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.figure = Figure(figsize=(9, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.panel = QWidget()
        self.panel.show()
        self.controls = IQControls()
        self.panel.iq_controls = self.controls
        self.panel._panels = {"phase": (self.figure, self.canvas, ResultTable(), QLabel())}
        self.panel._region = lambda: None
        self.panel._maximum_gap = lambda: None
        self.panel.floor = QDoubleSpinBox()
        self.panel.floor.setRange(-300, 0); self.panel.floor.setValue(-60)
        self.panel.phase_sign = QComboBox(); self.panel.phase_sign.addItem("+jwt", "+jwt")
        self.panel.owner = SimpleNamespace(plot_figure=self.figure, plot_colorbars=[],
                                           _effective_colormap=lambda: "turbo", btn_pan=QCheckBox())
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers: controller.disconnect()
        self.panel.close(); self.controls.close(); self.controls.cursor_widget.close(); self.canvas.close()

    def render(self, source, mode="iq"):
        controller = render_iq(self.panel, source, mode)
        self.controllers.append(controller)
        self.canvas.draw()
        return controller

    def test_zero_and_weak_samples_are_visible_with_honest_phase_readout(self):
        controller = self.render(cut([1, 1e-5j, 0, np.nan + 1j]))
        self.assertEqual(np.count_nonzero(controller.series[0]["valid"]), 3)
        self.assertEqual(controller.colorbar.mappable.cmap.name, "turbo")
        self.assertEqual(controller.axes[0].get_xlabel(), "I: real field (m)")
        controller.select_coordinate(10.)
        self.assertIn("weak / masked", self.controls.readout.text())
        controller.select_coordinate(11.1)
        self.assertEqual(controller.selected_coordinate, 11.)
        self.assertIn("undefined (zero return)", self.controls.readout.text())
        self.assertIn("−inf dBsm", self.controls.readout.text())
        self.assertEqual(len(controller.rows), 4)
        self.assertTrue(all(len(row) == len(controller.headers) for row in controller.rows))

    def test_missing_sample_splits_segments_and_bounds_ignore_cursor(self):
        controller = self.render(cut([1, 5 + 4j, np.nan + 1j, -3 - 2j, -4]))
        trajectories = [artist for artist in controller.iq_axes.collections if artist.__class__.__name__ == "LineCollection"]
        self.assertEqual([len(artist.get_segments()) for artist in trajectories], [2])
        np.testing.assert_allclose(trajectories[0].get_segments(), [[[1, 0], [5, 4]], [[-3, -2], [-4, 0]]])
        self.assertEqual(controller.iq_axes._grim_data_limits, (-4., 5., -2., 4.))
        controller.select_coordinate(9.)
        self.assertEqual(controller.iq_axes._grim_data_limits, (-4., 5., -2., 4.))

    def test_clicking_linked_plots_selects_native_sample_and_all_cursors(self):
        self.controls.linked_checkbox.setChecked(True)
        controller = self.render(cut([1, 1j, -1, -1j]))
        self.assertEqual(len(controller.axes), 3)
        self.assertEqual(len(self.figure.axes), 4)
        self.assertTrue(self.figure._grim_independent_x)
        x, y = controller.axes[1].transData.transform((10.2, 0.))
        event = MouseEvent("button_press_event", self.canvas, x, y, button=1)
        controller._click(event)
        self.assertEqual(controller.selected_coordinate, 10.)
        for cursor in controller._linked_cursors: np.testing.assert_array_equal(cursor.get_xdata(), [10., 10.])
        np.testing.assert_array_equal(controller._cursor_artists[0].get_ydata(), [1.])
        self.panel.owner.btn_pan.setChecked(True)
        controller.select_coordinate(9.)
        controller._click(event)
        self.assertEqual(controller.selected_coordinate, 9.)

    def test_iq_click_uses_native_samples_and_disconnect_releases_signals(self):
        controller = self.render(cut([1, 1j, -1]))
        x, y = controller.iq_axes.transData.transform((-1., 0.))
        controller._click(MouseEvent("button_press_event", self.canvas, x, y, button=1))
        self.assertEqual(controller.selected_coordinate, 11.)
        self.controls.coordinate_slider.setValue(1)
        self.assertEqual(controller.selected_coordinate, 10.)
        controller.disconnect()
        self.controls.coordinate_slider.setValue(0)
        self.assertEqual(controller.selected_coordinate, 10.)
        self.assertNotIn(controller.cid, self.canvas.callbacks.callbacks.get("button_press_event", {}))

    def test_comparison_shares_scale_and_difference_retains_native_values(self):
        source, other = cut([2, 2j, -2]), cut([1, 1j, -1], label="B")
        self.panel._cut = lambda **kwargs: other
        self.controls.scale_combo.setCurrentIndex(1)
        controller = self.render(source, "iq_compare")
        self.assertEqual(len(controller.series), 2)
        np.testing.assert_allclose(controller.series[0]["field"], [1, 1j, -1])
        np.testing.assert_allclose(controller.series[1]["field"], [.5, .5j, -.5])
        controller.disconnect()
        difference = self.render(source, "iq_difference")
        np.testing.assert_allclose(difference.series[0]["native_field"], [1, 1j, -1])
        self.assertIn("A − B", difference.iq_axes.get_title())

    def test_compensation_reports_reference_and_preserves_original(self):
        source = cut(np.exp(-2j * np.pi * np.array([9., 10., 11.]) * .25))
        original = source.field.copy()
        self.controls.delay_ns.setValue(.25)
        self.controls.phase_offset.setValue(-180.)
        controller = self.render(source, "iq_compensated")
        np.testing.assert_array_equal(source.field, original)
        np.testing.assert_allclose(np.diff(controller.series[1]["field"]), 0., atol=1e-13)
        rows = {row[0]: row for row in controller.result_rows}
        self.assertEqual(rows["Reference frequency"][1], "10.0")
        self.assertIn("z_corrected", controller.note.text())

    def test_repeat_cloud_uses_same_coordinate_and_scaled_mean(self):
        first, second, missing = cut([2, 4]), cut([4, 8], label="B"), cut([99], x=[15], label="C")
        self.panel._iq_repeat_cuts = lambda: ([first, second, missing], [])
        self.controls.scale_combo.setCurrentIndex(1)
        controller = self.render(first, "iq_repeat")
        report = controller.repeat_report
        self.assertEqual(report["count"], 2)
        self.assertEqual(len(report["skipped"]), 1)
        mean = next(line for line in controller.iq_axes.lines if line.get_label() == "Complex mean")
        np.testing.assert_allclose(mean.get_xdata(), [.75])
        self.assertIn("native mean I 3", self.controls.readout.text())
        controller.select_coordinate(10.)
        self.assertIn("native mean I 6", self.controls.readout.text())
        self.assertEqual(len(controller.rows), 2)
        self.assertEqual(controller.rows[0][2], 10.)

    def test_hold_rebuilds_shared_scaling_and_keeps_native_values(self):
        self.controls.scale_combo.setCurrentIndex(1)
        first = self.render(cut([1, 1j]))
        first.disconnect()
        self.panel._iq_hold = True
        controller = self.render(cut([4, 4j], label="B"))
        self.assertEqual(len(controller.series), 2)
        np.testing.assert_allclose(controller.series[0]["field"], [.25, .25j])
        np.testing.assert_allclose(controller.series[1]["field"], [1, 1j])
        np.testing.assert_allclose(controller.series[0]["native_field"], [1, 1j])

    def test_controls_emit_argument_free_refresh_signal(self):
        calls = []
        self.controls.changed.connect(lambda: calls.append(True))
        self.controls.scale_combo.setCurrentIndex(1)
        self.controls.linked_checkbox.setChecked(True)
        self.controls.guides_checkbox.setChecked(False)
        self.controls.phase_offset.editingFinished.emit()
        self.controls.delay_ns.editingFinished.emit()
        self.assertEqual(len(calls), 5)

    def test_large_overview_is_bounded_and_keeps_both_component_extrema(self):
        values = np.exp(1j * np.linspace(0, 15, 200_001))
        values[10001], values[100_111] = 100 + 1j, 1 - 200j
        selected = _overview_indices(np.arange(len(values)), values, 20_000)
        self.assertLessEqual(len(selected), 20_000)
        self.assertTrue(np.all(np.diff(selected) > 0))
        self.assertIn(10001, selected); self.assertIn(100_111, selected)
        self.assertEqual(selected[0], 0); self.assertEqual(selected[-1], len(values) - 1)

    def test_repeated_hold_refresh_does_not_duplicate_cut(self):
        source = cut([2, 2j])
        first = self.render(source); first.disconnect()
        self.panel._iq_hold = True
        second = self.render(source)
        self.assertEqual(len(second.series), 1)

    def test_existing_marker_tool_can_probe_native_iq_and_companion_samples(self):
        self.controls.linked_checkbox.setChecked(True)
        controller = self.render(cut([1, 5j, -3]))
        lines = PlotMarkersMixin._marker_lines(self.panel.owner)
        proxy = next(line for line in lines if line.axes is controller.iq_axes)
        x, y = PlotMarkersMixin._marker_arrays(proxy)
        np.testing.assert_allclose(x, [1, 0, -3])
        np.testing.assert_allclose(y, [0, 5, 0])
        self.assertTrue(any(line.axes is controller.axes[1] for line in lines))
        self.assertTrue(any(line.axes is controller.axes[2] for line in lines))

    def test_repeat_coordinate_refresh_respects_limits_and_reapplies_theme(self):
        calls = []
        self.panel.owner._apply_plot_theme = lambda: calls.append("theme")
        self.panel.owner._toggle_legend = lambda: calls.append("legend")
        self.panel.owner._current_plot_text = lambda: "#fafafa"
        self.panel.owner.btn_auto_scale = QCheckBox()
        first, second = cut([1, 30]), cut([2, 40], label="B")
        self.panel._iq_repeat_cuts = lambda: ([first, second], [])
        controller = self.render(first, "iq_repeat")
        controller.iq_axes.set_xlim(-3, 3); controller.iq_axes.set_ylim(-2, 2)
        controller.select_coordinate(10.)
        self.assertEqual(controller.iq_axes.get_xlim(), (-3., 3.))
        self.assertEqual(controller.iq_axes.get_ylim(), (-2., 2.))
        self.assertEqual(calls, ["theme", "legend"])
        guides = [text for text in controller.iq_axes.texts if text.get_label() == "_iq_guide"]
        self.assertTrue(guides)
        self.assertTrue(all(text.get_color() == "#fafafa" for text in guides))

    def test_companions_preserve_selected_native_and_reference_gaps(self):
        self.controls.linked_checkbox.setChecked(True)
        source = replace(cut([1, 1j, -1, -1j], x=[9., 10., 15., 16.]),
                         native_indices=np.array([0, 1, 6, 7]))
        controller = self.render(source)
        for axes in controller.axes[1:]:
            line = next(line for line in axes.lines if hasattr(line, "_grim_dataset_key"))
            xx = line.get_xdata()
            i10, i15 = np.flatnonzero(xx == 10)[0], np.flatnonzero(xx == 15)[0]
            self.assertTrue(np.any(np.isnan(xx[i10 + 1:i15])))
        controller.disconnect()
        self.panel._cut = lambda **kwargs: cut([1, 1j, -1], x=[9., 11., 12.], label="B")
        compared = self.render(cut([1, 1j, -1, -1j], x=[9., 10., 11., 12.]), "iq_compare")
        for axes in compared.axes[1:]:
            for line in axes.lines:
                if not hasattr(line, "_grim_dataset_key"): continue
                xx = line.get_xdata()
                self.assertTrue(np.any(np.isnan(xx[np.flatnonzero(xx == 9)[0] + 1:np.flatnonzero(xx == 11)[0]])))

    def test_repeat_counts_fixed_cut_skips_in_readout_and_results(self):
        source = cut([1, 2])
        self.panel._iq_repeat_cuts = lambda: ([source], [])
        self.panel._iq_repeat_skipped = [{"label": "B", "reason": "No matching elevation."}]
        controller = self.render(source, "iq_repeat")
        self.assertIn("1 skipped", self.controls.readout.text())
        skipped = next(row for row in controller.table.rows if row[0] == "Skipped runs")
        self.assertEqual(skipped[1], "1")
        self.assertIn("No matching elevation", skipped[2])

    def test_fragmented_overview_has_one_global_budget_without_gap_bridges(self):
        count = 60_001
        values = np.exp(1j * np.linspace(0, 15, count))
        values[1::2] = complex(np.nan, np.nan)
        self.controls.linked_checkbox.setChecked(True)
        controller = self.render(cut(values))
        clouds = [artist for artist in controller.iq_axes.collections if hasattr(artist, "get_offsets")]
        self.assertTrue(clouds)
        self.assertLessEqual(sum(len(artist.get_offsets()) for artist in clouds), 20_000)
        self.assertFalse(any(artist.__class__.__name__ == "LineCollection" for artist in controller.iq_axes.collections))
        for axes in controller.axes[1:]:
            lines = [line for line in axes.lines if hasattr(line, "_grim_dataset_key")]
            self.assertEqual(len(lines), 1)
            self.assertLessEqual(len(lines[0].get_xdata()), 20_000)
            yy = np.asarray(lines[0].get_ydata())
            self.assertFalse(np.any(np.isfinite(yy[:-1]) & np.isfinite(yy[1:])))
        self.assertEqual(len(controller.rows), count)


if __name__ == "__main__":
    unittest.main()
