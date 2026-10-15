"""Image rulers preserve the view and keep angle/range physically separate."""
from __future__ import annotations

import csv
import io
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from GRIM_Backend.ui.image_measurement import distance_result, measurement_axes
from test_plot_analysis_features import _WindowCase


@pytest.mark.parametrize("unit", ["m", "in", "ft"])
def test_azimuth_and_range_are_independent_physical_separations(unit):
    ax = Figure().add_subplot()
    ax.set(xlabel="Azimuth (deg)", ylabel=f"Down-Range ({unit})")
    label, rows = distance_result(ax, [350.25, 4.125], [10.75, -1.5])
    results = {metric: (float(value), unit) for metric, value, unit in rows}
    assert results["Signed ΔAzimuth"] == (-339.5, "deg")
    assert results["Azimuth separation"] == (339.5, "deg")
    assert results["Down-Range separation"] == (5.625, unit)
    assert "Distance" not in results and "Distance" not in label


def test_spatial_distance_converts_both_length_units_before_hypotenuse():
    ax = Figure().add_subplot()
    ax.set(xlabel="Cross-Range (in)", ylabel="Down-Range (ft)")
    _label, rows = distance_result(ax, [0, 0], [3, 1 / 3])
    results = {metric: (float(value), unit) for metric, value, unit in rows}
    assert results["Distance"] == (5, "in")
    assert results["Signed ΔCross-Range"] == (3, "in")


def test_radians_and_invalid_coordinates_are_not_silently_reinterpreted():
    ax = Figure().add_subplot()
    ax.set(xlabel="Azimuth (rad)", ylabel="Down-Range (m)")
    _label, rows = distance_result(ax, [.1, -2], [.3, -1])
    assert next(row for row in rows if row[0] == "Azimuth separation") == ("Azimuth separation", "0.2", "rad")
    with pytest.raises(ValueError, match="finite"):
        distance_result(ax, [0, 0], [np.nan, 1])
    with pytest.raises(ValueError, match="large"):
        distance_result(ax, [-1e308, 0], [1e308, 0])


def test_measurement_axes_exclude_colorbars_and_line_cuts():
    figure = Figure()
    image, cut = figure.subplots(1, 2)
    mesh = image.pcolormesh([0, 1, 2], [0, 1, 2], np.ones((2, 2)))
    image.set(xlabel="Azimuth (deg)", ylabel="Down-Range (m)")
    cut.plot([0, 1], [0, 1]); cut.set(xlabel="Range (m)", ylabel="Power (dB)")
    figure.colorbar(mesh, ax=image)
    assert measurement_axes(figure) == [image]


class ImageMeasurementTests(_WindowCase):
    def tearDown(self):
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        super().tearDown()

    def make_image(self, unit="m", multiple=False, reverse=False):
        window = self.window
        window.main_tabs.setCurrentWidget(window.tab_isar)
        window.btn_auto_plot.setChecked(False)
        window._start_plot_render()
        window.plot_figure.clear()
        axes = list(np.atleast_1d(window.plot_figure.subplots(1, 2 if multiple else 1, sharey=True)))
        for ax in axes:
            ax.pcolormesh(np.linspace(0, 360, 11), np.linspace(-5, 5, 11), np.arange(100).reshape(10, 10), cmap="turbo")
            ax.set(xlabel="Azimuth (deg)", xlim=(360, 0) if reverse else (0, 360),
                   ylim=(5, -5) if reverse else (-5, 5))
        axes[0].set_ylabel(f"Down-Range ({unit})")
        window.plot_ax = axes[0]
        window.plot_axes = axes if multiple else None
        window.last_plot_mode = "az_vs_range"
        window.plot_canvas.draw()
        return axes

    def click(self, ax, x, y):
        # A real Matplotlib event has data coordinates inverse-transformed
        # from the pixel position, including inverted axes.
        px, py = ax.transData.transform([x, y])
        event = MouseEvent("button_press_event", self.window.plot_canvas, px, py, button=MouseButton.LEFT)
        self.window._on_plot_mouse_press(event)
        return event

    def test_measure_opens_on_image_without_rendering_dataset_cut(self):
        ax, = self.make_image(unit="ft", reverse=True)
        figure = self.window.plot_figure
        images = list(ax.collections)
        limits = (ax.get_xlim(), ax.get_ylim())
        panel = self.window._show_engineering_tool("measure")
        ruler = panel.image_measurement
        self.assertTrue(panel._image_measurement_active)
        self.assertEqual(self.window.last_plot_mode, "az_vs_range")
        self.assertEqual(figure.axes, [ax]); self.assertEqual(list(ax.collections), images)
        self.assertTrue(panel.header_controls.isHidden())
        first = self.click(ax, 60.123, 3.125)
        second = self.click(ax, 104.456, -1.75)
        results = {name: (float(value), unit) for name, value, unit in ruler.rows}
        self.assertAlmostEqual(results["Down-Range separation"][0], abs(second.ydata - first.ydata))
        self.assertEqual(results["Down-Range separation"][1], "ft")
        self.assertAlmostEqual(results["Azimuth separation"][0], abs(second.xdata - first.xdata))
        self.assertNotIn("Distance", results)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        self.assertTrue(all(not artist.get_in_layout() for artist in ruler.artists))
        ruler.copy_results()
        copied = list(csv.DictReader(io.StringIO(QApplication.clipboard().text()), delimiter="\t"))
        self.assertEqual(len(copied), 8)
        self.assertEqual([row for row in copied if row["Metric"] == "Down-Range separation"][0]["Unit"], "ft")
        self.assertEqual(len(ruler.points), 2)
        recorder = self.window.python_recorder
        with mock.patch.object(recorder, "record_unsupported_plot") as unsupported:
            self.assertFalse(self.window._emit_last_successful_python_plot())
            self.assertEqual(unsupported.call_args.args[0], "image_measurement")
        # The next click begins a new ruler, retaining the image.
        self.click(ax, 125.5, .5)
        self.assertEqual(len(ruler.points), 1); self.assertEqual(ruler.rows, [])
        ruler.clear_button.click()
        self.assertFalse(ruler.artists); self.assertEqual(list(ax.collections), images)

    def test_second_point_must_be_in_same_image_and_colorbar_is_ignored(self):
        first, second = self.make_image(multiple=True)
        colorbar = self.window.plot_figure.colorbar(first.collections[0], ax=first)
        self.window.plot_canvas.draw()
        panel = self.window._show_engineering_tool("measure")
        ruler = panel.image_measurement
        self.click(first, 25, 1)
        self.click(second, 75, 2)
        self.assertEqual(len(ruler.points), 1)
        self.assertIn("same image", ruler.readout.text())
        event = SimpleNamespace(canvas=self.window.plot_canvas, button=MouseButton.LEFT,
                                inaxes=colorbar.ax, xdata=.5, ydata=10)
        self.assertFalse(ruler.on_press(event))
        self.click(first, 75, 2)
        self.assertEqual(len(ruler.points), 2)

    def test_zoom_pan_tabs_clear_and_replot_release_ruler(self):
        ax, = self.make_image()
        panel = self.window._show_engineering_tool("measure")
        ruler = panel.image_measurement
        self.click(ax, 30, 1)
        self.window.btn_pan.setChecked(True)
        self.assertFalse(ruler.pick_button.isChecked()); self.assertEqual(ruler.points, [])
        ruler.pick_button.setChecked(True)
        self.assertFalse(self.window.btn_pan.isChecked())
        self.click(ax, 30, 1); self.click(ax, 60, 3)
        self.window.btn_zoom_box.setChecked(True)
        self.assertFalse(ruler.pick_button.isChecked()); self.assertEqual(len(ruler.points), 2)
        ruler.pick_button.setChecked(True)
        self.assertFalse(self.window.btn_zoom_box.isChecked())
        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        self.assertFalse(ruler.pick_button.isChecked())
        self.window.main_tabs.setCurrentWidget(self.window.tab_isar)
        self.assertFalse(ruler.pick_button.isChecked())
        ruler.pick_button.setChecked(True)
        self.window.plot_canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", self.window.plot_canvas, key="escape"))
        self.assertFalse(ruler.pick_button.isChecked())
        self.window._clear_plot()
        self.assertFalse(panel._image_measurement_active); self.assertEqual(ruler.points, [])
        self.assertEqual(ruler.axes, [])
        self.assertTrue(panel.isHidden())
        ax, = self.make_image()
        panel = self.window._show_engineering_tool("measure")
        self.click(ax, 30, 1)
        self.window._start_plot_render()
        self.assertFalse(panel._image_measurement_active)
        self.assertFalse(panel.image_measurement.artists)
        self.assertEqual(panel.image_measurement.axes, [])

    def test_return_to_trace_analysis_restores_existing_controls(self):
        self.make_image()
        panel = self.window._show_engineering_tool("measure")
        self.window._start_plot_render()
        self.window.plot_figure.clear()
        ax = self.window.plot_figure.add_subplot()
        ax.set(xlabel="Azimuth (deg)", ylabel="RCS (dBsm)")
        line, = ax.plot([0., 1., 2.], [0., 3., 0.], label="Trace")
        line._grim_dataset_key = "trace"
        self.window.plot_ax = ax; self.window.plot_axes = None
        self.window.last_plot_mode = "azimuth_rect"
        panel = self.window._show_engineering_tool("measure")
        self.assertFalse(panel._image_measurement_active)
        self.assertTrue(panel.image_measurement.isHidden())
        self.assertFalse(panel.header_controls.isHidden())
        self.assertFalse(panel.settings_button.isHidden())
        self.assertIs(panel._measurement[3], line)
