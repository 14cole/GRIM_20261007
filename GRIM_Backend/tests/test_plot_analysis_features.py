"""CDF, sector statistics, PbP bands, markers, slider, and time gate."""

from __future__ import annotations

import json
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMenu

import GRIM_Backend.ui.app as grim_cut_gui
from GRIM_Backend.ui.dataset_actions import DATASET_PATH_ROLE
from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import (
    down_range_profile, gate_geometry, time_gate, translate_phase_center,
)
from GRIM_Backend.plotting.modes.range_freq_mode import subband_starts
from GRIM_Backend.plotting.dataset_style import pbp_band_key
from GRIM_Backend.plotting.modes import common, sector_stats_mode
from test_gui_shell import (
    _FakeFeatureWorkflow, _FakeFreddyIntegration, _FakeGhostIntegration,
    _MemorySettings, _RecordingWindow,
)
from test_plot_renderer_correctness import _grid

POWER_SHAPE = np.asarray([1.0, 2.0, 5.0, 3.0, 2.0])


class _WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for name, replacement in (
            ("GhostIntegrationWidget", _FakeGhostIntegration),
            ("FreddyIntegrationWidget", _FakeFreddyIntegration),
        ):
            patch = mock.patch.object(grim_cut_gui, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(grim_cut_gui, "load_ghost_module", return_value=_FakeFeatureWorkflow())
        patch.start()
        self.addCleanup(patch.stop)
        self.window = _RecordingWindow(settings=_MemorySettings())
        for button in (self.window.btn_auto_plot, self.window.btn_zoom_box, self.window.btn_pan):
            button.setChecked(False)
        for index in range(3):
            dataset = _grid()
            freq_scale = np.linspace(1.0, 2.0, dataset.frequencies.size)[None, None, :, None]
            dataset.rcs_power[:] = POWER_SHAPE[:, None, None, None] * (index + 1) ** 2 * freq_scale
            self.window._add_dataset_row(dataset, f"Run {index + 1}", "", file_name="")
        self.datasets = [self.window.table.item(row, 0).data(Qt.UserRole) for row in range(3)]
        self.keys = [self.window._dataset_plot_key(dataset) for dataset in self.datasets]
        self.select_rows(0, 1, 2)
        self.window.resize(1300, 850)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.plot_slider.stop_play()
        self.window.deleteLater()
        self.app.processEvents()

    def select_rows(self, *rows, freqs=(0,)):
        window = self.window
        window.table.clearSelection()
        for row in rows:
            window.table.selectionModel().select(
                window.table.model().index(row, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows,
            )
        window.table.setCurrentCell(rows[0], 0, QItemSelectionModel.NoUpdate)
        window._on_dataset_selection_changed()
        window.list_az.selectAll()
        window.list_elev.selectAll()
        window.list_freq.clearSelection()
        for row in freqs:
            window.list_freq.item(row).setSelected(True)
        window.list_pol.clearSelection()
        window.list_pol.item(0).setSelected(True)

    def lines(self, key):
        return [line for ax in self.window.plot_figure.axes for line in ax.lines
                if getattr(line, "_grim_dataset_key", None) == key and len(line.get_xdata())]

    def plot(self, method):
        getattr(self.window, method)()
        self.window.plot_canvas.draw()
        self.app.processEvents()

    def legend_labels(self):
        legend = self.window.plot_ax.get_legend()
        return [text.get_text() for text in legend.get_texts()] if legend else []

    def dbsm(self, linear):
        return 10.0 * np.log10(linear)


class CdfTests(_WindowCase):
    def test_cdf_ranks_pooled_samples_and_exceedance_mirrors_it(self):
        window = self.window
        self.select_rows(0, 1, freqs=(0, 1))
        self.plot("_plot_cdf")
        self.assertIn("CDF plot updated", window.status.currentMessage())
        (line,) = self.lines(self.keys[0])
        expected = np.sort(self.dbsm(np.r_[POWER_SHAPE, POWER_SHAPE * np.linspace(1, 2, 8)[1]]))
        np.testing.assert_allclose(line.get_xdata(), expected)
        np.testing.assert_allclose(line.get_ydata(), 100.0 * np.arange(1, 11) / 10)
        self.assertEqual(line.get_drawstyle(), "steps-post")
        self.assertIn("10 samples", line.get_label())
        self.assertIn("at or below", window.plot_ax.get_ylabel())

        window.analysis_controls.combo_cdf.setCurrentIndex(1)
        (line,) = self.lines(self.keys[0])
        np.testing.assert_allclose(line.get_ydata(), 100.0 * (1 - np.arange(10) / 10))
        self.assertIn("at or above", window.plot_ax.get_ylabel())
        self.assertEqual(window.plot_ax.get_ylim(), (0.0, 100.0))

    def test_cdf_rejects_phase_and_keeps_hold_overlays(self):
        window = self.window
        window.btn_phase.setChecked(True)
        self.plot("_plot_cdf")
        self.assertIn("Turn off Phase", window.status.currentMessage())
        window.btn_phase.setChecked(False)
        self.select_rows(0)
        self.plot("_plot_cdf")
        window.btn_hold.setChecked(True)
        self.select_rows(1)
        self.plot("_plot_cdf")
        self.assertEqual(len(self.lines(self.keys[0])), 1)
        self.assertEqual(len(self.lines(self.keys[1])), 1)
        self.plot("_plot_azimuth_rect")
        self.assertIn("Hold blocked", window.status.currentMessage())


class SectorStatisticsTests(_WindowCase):
    def test_right_click_opens_sector_settings_with_selected_azimuths(self):
        window = self.window
        self.select_rows(0)
        button = window.btn_sector_stats
        self.assertEqual(button.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu)
        with mock.patch.object(grim_cut_gui, "SectorStatisticsDialog") as dialog_type:
            button.customContextMenuRequested.emit(QPoint(5, 5))
        dialog_type.assert_called_once()
        args, kwargs = dialog_type.call_args
        self.assertIs(args[0], window.analysis_controls)
        np.testing.assert_array_equal(args[1], self.datasets[0].azimuths)
        self.assertEqual(kwargs["unit"], "deg")
        self.assertIs(kwargs["parent"], window)
        dialog_type.return_value.exec.assert_called_once_with()

    def test_sector_levels_use_linear_power_and_copy_table(self):
        window = self.window
        controls = window.analysis_controls
        controls.set_sector_settings("-2:0, 0:2", "mean", 90.0)
        self.select_rows(0)
        self.plot("_plot_sector_stats")
        self.assertIn("2 sectors, mean", window.status.currentMessage())
        (line,) = self.lines(self.keys[0])
        first, second = POWER_SHAPE[:3].mean(), POWER_SHAPE[2:].mean()
        np.testing.assert_allclose(line.get_xdata(), [-2, 0, np.nan, 0, 2, np.nan])
        np.testing.assert_allclose(
            line.get_ydata(), self.dbsm([first, first, np.nan, second, second, np.nan])
        )
        self.assertEqual(window.plot_ax.get_ylabel(), "RCS mean (dBsm)")

        controls.set_sector_settings("-2:0, 0:2", "percentile", 50.0)
        (line,) = self.lines(self.keys[0])
        np.testing.assert_allclose(line.get_ydata()[0], self.dbsm(np.median(POWER_SHAPE[:3])))
        table = window.plot_figure._grim_sector_table
        text = sector_stats_mode.table_text(table)
        self.assertTrue(text.startswith("Dataset\tPol\tFrequency\tElevation\tSector\tSamples"))
        self.assertIn("Run 1\tHH\t9\t0\t-2 to 0\t3", text)

    def test_tiled_sectors_hold_over_azimuth_cut_and_bad_text_blocks(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        window.btn_hold.setChecked(True)
        window.analysis_controls.set_sector_settings("2", "mean", 90.0)
        self.plot("_plot_sector_stats")
        self.assertNotIn("blocked", window.status.currentMessage().lower())
        self.assertEqual(len(self.lines(self.keys[0])), 2)
        window.btn_hold.setChecked(False)
        window.analysis_controls.set_sector_settings("0:0", "mean", 90.0)
        self.plot("_plot_sector_stats")
        self.assertIn("Sector Stats blocked: sector '0:0' is empty", window.status.currentMessage())

    def test_parse_sectors_tiles_wraps_and_counts_each_sample_once(self):
        azimuths = np.arange(-180.0, 181.0)
        tiles = common.parse_sectors("30", azimuths)
        self.assertEqual(len(tiles), 12)
        self.assertEqual(sum(int(s.contains(azimuths).sum()) for s in tiles), azimuths.size)
        wrap = common.parse_sectors("170:-170", azimuths)[0]
        self.assertEqual(wrap.label(), "170 to -170")
        self.assertEqual(int(wrap.contains(azimuths).sum()), 22)
        self.assertEqual(wrap.display_pieces(-180.0, 180.0), [(170.0, 180.0), (-180.0, -170.0)])
        self.assertEqual(
            [s.label() for s in common.parse_sectors("0:90:360", np.arange(0.0, 360.0))],
            ["0 to 90", "90 to 180", "180 to 270", "270 to 360"],
        )
        for text in ("", "x", "5:5", "0:-1:5", "0:0.001:10"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                common.parse_sectors(text, azimuths)


class PbpBandTests(_WindowCase):
    def band(self, group=""):
        artists = self.window._plot_item_artists(pbp_band_key(group))
        edges = [a for a in artists if hasattr(a, "get_ydata") and len(a.get_ydata())]
        return artists, edges

    def test_percentile_band_uses_chosen_percentiles(self):
        window = self.window
        controls = window.analysis_controls
        window.btn_pbp.setChecked(True)
        self.plot("_plot_azimuth_rect")
        _artists, edges = self.band()
        np.testing.assert_allclose(edges[0].get_ydata(), self.dbsm(POWER_SHAPE))
        np.testing.assert_allclose(edges[1].get_ydata(), self.dbsm(POWER_SHAPE * 9))

        controls.set_pbp_band("percentile")
        controls.spin_pbp_low.setValue(25.0)
        controls.spin_pbp_high.setValue(75.0)
        _artists, edges = self.band()
        levels = self.dbsm(POWER_SHAPE[None, :] * np.array([1.0, 4.0, 9.0])[:, None])
        np.testing.assert_allclose(edges[0].get_ydata(), np.percentile(levels, 25, axis=0))
        np.testing.assert_allclose(edges[1].get_ydata(), np.percentile(levels, 75, axis=0))
        self.assertTrue(self.legend_labels()[0].startswith("PBP P25–P75 Pol HH"))
        controls.spin_pbp_high.setValue(20.0)
        self.assertLess(controls.spin_pbp_low.value(), controls.spin_pbp_high.value())

    def test_selected_datasets_draw_one_band_and_remove_together(self):
        window = self.window
        window.btn_pbp.setChecked(True)
        self.plot("_plot_azimuth_rect")
        artists, edges = self.band()
        self.assertTrue(artists)
        np.testing.assert_allclose(edges[0].get_ydata(), self.dbsm(POWER_SHAPE))
        np.testing.assert_allclose(edges[1].get_ydata(), self.dbsm(POWER_SHAPE * 9))
        self.assertEqual(
            self.legend_labels(),
            ["PBP Pol HH, Freq 9 GHz, Elevation 0 deg"],
        )
        self.assertTrue(window._remove_plot_dataset(pbp_band_key()))
        self.assertEqual(self.band()[0], [])
        self.assertEqual(self.legend_labels(), [])


class RemovedDeltaReferenceTests(_WindowCase):
    def test_delta_reference_controls_and_menu_are_removed(self):
        window = self.window
        self.assertFalse(hasattr(window, "btn_delta_ref"))
        controls = window._plot_controls_by_tab["plotting"]
        self.assertNotIn("delta_ref", controls)
        self.assertIn("compare", controls)
        self.assertIn("delta_map", controls)
        self.assertFalse(hasattr(window.analysis_controls, "spin_delta_tolerance"))
        self.assertNotIn("Δ Ref Tolerance", [label.text() for label in window.findChildren(QLabel)])

        menu_labels = []

        class CaptureMenu(QMenu):
            def exec(self, _position):
                menu_labels.extend(action.text() for action in self.actions())
                return None

        self.select_rows(0)
        position = window.table.visualItemRect(window.table.item(0, 0)).center()
        with mock.patch("GRIM_Backend.ui.dataset_actions.QMenu", CaptureMenu):
            window._on_dataset_context_menu(position)
        self.assertIn("Save", menu_labels)
        self.assertIn("Delete", menu_labels)
        self.assertNotIn("Set as Δ reference", menu_labels)
        self.assertNotIn("Clear Δ reference", menu_labels)


class MarkerTests(_WindowCase):
    def press(self, x, y, kind="button_press_event", button=MouseButton.LEFT):
        event = MouseEvent(kind, self.window.plot_canvas, x, y, button=button)
        handler = {
            "button_press_event": self.window._on_plot_mouse_press,
            "motion_notify_event": self.window._on_plot_mouse_move,
            "button_release_event": self.window._on_plot_mouse_release,
        }[kind]
        handler(event)

    def point_pixels(self, line, index):
        return line.get_transform().transform(
            [[line.get_xdata()[index], line.get_ydata()[index]]]
        )[0]

    def test_markers_snap_drag_step_follow_replots_and_clear(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        (line,) = self.lines(self.keys[0])
        window.btn_zoom_box.setChecked(True)
        window.btn_markers.setChecked(True)
        self.assertFalse(window.btn_zoom_box.isChecked())
        x, y = self.point_pixels(line, 2)
        self.press(x + 6, y - 6)
        self.press(x + 6, y - 6, "button_release_event")
        (marker,) = window._plot_markers
        self.assertEqual(marker["index"], 2)
        text = marker["text"].get_text()
        self.assertIn("M1  Run 1", text)
        self.assertIn("x 0 deg", text)
        self.assertIn(f"y {self.dbsm(5.0):.6g} dBsm", text)
        self.assertEqual(getattr(window, "_highlighted_plot_datasets", set()), set())

        x3, y3 = self.point_pixels(line, 3)
        self.press(x, y)
        self.press(x3, y3, "motion_notify_event")
        self.press(x3, y3, "button_release_event")
        self.assertEqual(marker["index"], 3)
        window._on_plot_key_press(mock.Mock(canvas=window.plot_canvas, key="left"))
        window._on_plot_key_press(mock.Mock(canvas=window.plot_canvas, key="left"))
        self.assertEqual(marker["index"], 1)

        x4, y4 = self.point_pixels(line, 4)
        self.press(x4, y4)
        self.press(x4, y4, "button_release_event")
        second = window._plot_markers[1]
        self.assertIn("Δ M1: +3, ", second["text"].get_text())

        far_x, far_y = window.plot_ax.transAxes.transform((0.02, 0.98))
        self.press(far_x, far_y)
        self.assertIn("No curve near the click", window.status.currentMessage())
        self.assertEqual(len(window._plot_markers), 2)

        window.list_freq.clearSelection()
        window.list_freq.item(3).setSelected(True)
        self.plot("_plot_azimuth_rect")
        self.assertEqual([m["index"] for m in window._plot_markers], [1, 4])
        self.assertIs(window._plot_markers[0]["line"], self.lines(self.keys[0])[0])
        self.assertIn(window._plot_markers[0]["artist"], window.plot_ax.lines)

        window._remove_plot_marker(window._plot_markers[0])
        self.assertEqual([m["number"] for m in window._plot_markers], [2])
        window._clear_plot()
        self.assertEqual(window._plot_markers, [])


class SliderTests(_WindowCase):
    def move(self, row):
        slider = self.window.plot_slider
        slider.slider.setValue(row)
        slider._debounce.stop()
        self.window._on_plot_slider_moved(row)
        self.app.processEvents()

    def test_slider_scrubs_frequency_and_replaces_its_curves_under_hold(self):
        window = self.window
        self.select_rows(0)
        self.plot("_plot_azimuth_rect")
        window.btn_slider.setChecked(True)
        slider = window.plot_slider
        self.assertTrue(slider.isVisible())
        self.assertEqual(slider.slider.maximum(), 7)
        self.move(5)
        self.assertEqual([window.list_freq.row(i) for i in window.list_freq.selectedItems()], [5])
        (line,) = self.lines(self.keys[0])
        self.assertIn(f"Freq {self.datasets[0].frequencies[5]:.12g} GHz", line.get_label())
        np.testing.assert_allclose(line.get_ydata(), self.dbsm(POWER_SHAPE * np.linspace(1, 2, 8)[5]))

        window.btn_hold.setChecked(True)
        self.move(6)
        self.move(7)
        labels = [l.get_label() for l in self.lines(self.keys[0])]
        self.assertEqual(len(labels), 2)  # the cut plotted before Hold plus the scrubbed one
        self.assertTrue(any("Freq 10 GHz" in label for label in labels))

        slider.combo_axis.setCurrentIndex(slider.combo_axis.findData("azimuth"))
        self.move(1)
        self.assertIn("This plot sweeps azimuth", window.status.currentMessage())

    def test_play_steps_and_wraps(self):
        window = self.window
        window.btn_slider.setChecked(True)
        slider = window.plot_slider
        slider.slider.setValue(7)
        slider.step(1, wrap=True)
        self.assertEqual(slider.slider.value(), 0)
        slider.btn_play.setChecked(True)
        self.assertEqual(slider.btn_play.text(), "Pause")
        window.btn_slider.setChecked(False)
        self.assertFalse(slider.btn_play.isChecked())


class TimeGateTests(unittest.TestCase):
    def two_scatterers(self, *, conjugate=False):
        frequencies = np.linspace(8.0, 12.0, 201)
        hz = frequencies * 1e9
        field = (np.exp(-4j * np.pi * hz * 0.3 / C0)
                 + 0.5 * np.exp(-4j * np.pi * hz * -1.0 / C0))
        field = np.broadcast_to(field, (2, 1, frequencies.size)).copy()[..., None]
        units = {"frequency": "GHz"}
        if conjugate:
            field = np.conj(field)
            units["time_convention"] = "exp(-j*omega*t)"
        return RcsGrid([0.0, 1.0], [0.0], frequencies, ["HH"], rcs=field, units=units), hz

    def test_keep_and_remove_isolate_scatterers_in_either_time_convention(self):
        middle = slice(40, 160)
        for conjugate in (False, True):
            with self.subTest(conjugate=conjugate):
                grid, hz = self.two_scatterers(conjugate=conjugate)
                kept = time_gate(grid, start_m=0.0, stop_m=0.6).rcs_slice((0, 0, slice(None), 0))
                want = np.exp(-4j * np.pi * hz * 0.3 / C0)
                want = np.conj(want) if conjugate else want
                np.testing.assert_allclose(kept[middle], want[middle], atol=5e-3)
                self.assertLess(np.max(np.abs(kept - want)), 0.1)
        grid, hz = self.two_scatterers()
        removed = time_gate(grid, start_m=0.0, stop_m=0.6, mode="remove")
        np.testing.assert_allclose(
            removed.rcs_slice((0, 0, slice(None), 0))[middle],
            0.5 * np.exp(-4j * np.pi * hz[middle] * -1.0 / C0), atol=0.05,
        )
        self.assertIn('"mode": "remove"', removed.extra["time_gate_json"])
        ranges, profile = down_range_profile(grid, elevation_index=0, polarization_index=0)
        self.assertAlmostEqual(ranges[np.argmax(profile)], 0.3, delta=0.02)

    def test_rejects_bad_gates_and_grids(self):
        grid, _hz = self.two_scatterers()
        half = gate_geometry(grid)["unambiguous_m"] / 2
        for kwargs, message in (
            ({"start_m": 1.0, "stop_m": 0.0}, "beyond gate start"),
            ({"start_m": -half - 1, "stop_m": 0.0}, "unambiguous down range"),
            ({"start_m": 0.0, "stop_m": 0.01}, "narrower than"),
            ({"start_m": 0.0, "stop_m": 1.0, "mode": "notch"}, "mode must be"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                time_gate(grid, **kwargs)
        uneven = RcsGrid([0.0], [0.0], np.r_[np.linspace(8, 9, 9), 9.5], ["HH"],
                         rcs=np.ones((1, 1, 10, 1), complex), units={"frequency": "GHz"})
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            time_gate(uneven, start_m=-1.0, stop_m=1.0)


class TimeGateGuiTests(_WindowCase):
    def test_button_creates_gated_rows_and_records_script(self):
        window = self.window
        self.assertTrue(window.btn_time_gate.isEnabled())
        # Eight 142.9 MHz steps: 0.13 m resolution, 1.05 m unambiguous range.
        params = {"start_m": -0.3, "stop_m": 0.3, "taper": 0.2, "mode": "keep",
                  "compensate": True}
        # A saved source makes the operation replayable in the Python script.
        window.table.item(0, 1).setData(DATASET_PATH_ROLE, "C:/data/run1.grim")
        with mock.patch("GRIM_Backend.ui.dataset_actions.TimeGateDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = params
            self.select_rows(0)
            window._time_gate_selected()
            deadline = time.monotonic() + 10
            while window._background_job_active() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertIn("Time gate created 1 dataset(s)", window.status.currentMessage())
        self.assertEqual(window.table.rowCount(), 4)
        self.assertEqual(window.table.item(3, 0).text(), "Run 1 [Gate Keep -0.3 to 0.3 m]")
        self.assertIn("Time gate (keep -0.3 to 0.3 m, taper 20%)", window.table.item(3, 2).text())
        script = window.python_recorder.script
        self.assertIn("time_gate(", script)
        self.assertIn("start_m=-0.3", script)


def _point_grid(point, *, azimuths=(0.0, 30.0, 90.0), elevations=(0.0,), units=None, conjugate=False):
    """A GHOST-law point scatterer: phase exp(+j 2k u.p), coming-from directions."""
    frequencies = np.linspace(8.0, 12.0, 81)
    k = 2 * np.pi * frequencies * 1e9 / C0
    a = np.deg2rad(np.asarray(azimuths))[:, None]
    e = np.deg2rad(np.asarray(elevations))[None, :]
    u = np.stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e) + 0 * a), axis=-1)
    field = np.exp(2j * k[None, None, :] * (u @ np.asarray(point))[..., None])[..., None]
    grid_units = {"frequency": "GHz"}
    grid_units.update(units or {})
    if conjugate:
        field = np.conj(field)
    return RcsGrid(np.asarray(azimuths), np.asarray(elevations), frequencies, ["HH"],
                   rcs=field, units=grid_units)


class PhaseCenterTests(unittest.TestCase):
    def test_point_at_new_origin_has_constant_phase_in_every_convention(self):
        point = (0.4, -0.25, 0.1)
        for kwargs in ({}, {"conjugate": True, "units": {"time_convention": "exp(-j omega t)"}}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                grid = _point_grid(point, elevations=(-10.0, 0.0, 20.0), **kwargs)
                moved = translate_phase_center(grid, x_m=point[0], y_m=point[1], z_m=point[2])
                np.testing.assert_allclose(moved.rcs_slice((slice(None),) * 4), 1.0, atol=1e-9)
                np.testing.assert_array_equal(moved.rcs_power, grid.rcs_power)
        radians = RcsGrid(np.deg2rad([0.0, 30.0, 90.0]), [0.0], np.linspace(8, 12, 81), ["HH"],
                          rcs=_point_grid(point).rcs_slice((slice(None),) * 4),
                          units={"frequency": "GHz", "azimuth": "rad", "elevation": "rad"})
        moved = translate_phase_center(radians, x_m=point[0], y_m=point[1], z_m=point[2])
        np.testing.assert_allclose(moved.rcs_slice((slice(None),) * 4), 1.0, atol=1e-9)

    def test_moved_point_lands_at_its_new_down_range_and_metadata_accumulates(self):
        grid = _point_grid((0.0, 0.0, 0.0), units={"phase_reference": "turntable centre"})
        moved = translate_phase_center(grid, x_m=-0.3, y_m=0.0, z_m=0.0)
        # The new origin is 0.3 m toward the tail, so nose-on (radar along +x)
        # the old origin sits 0.3 m nearer the radar: down range -0.3 m.
        ranges, profile = down_range_profile(moved, elevation_index=0, polarization_index=0,
                                             max_sweeps=1)
        self.assertAlmostEqual(ranges[np.argmax(profile)], -0.3, delta=0.02)
        again = translate_phase_center(moved, x_m=0.1, y_m=0.2, z_m=0.0)
        record = json.loads(again.extra["phase_center_translation_json"])
        np.testing.assert_allclose(record["total_offset_m"], [-0.2, 0.2, 0.0])
        self.assertIn("turntable centre; moved by (-0.3, 0, 0) m", again.units["phase_reference"])
        with self.assertRaisesRegex(ValueError, "offset is zero"):
            translate_phase_center(grid, x_m=0.0, y_m=0.0, z_m=0.0)
        power_only = RcsGrid([0.0], [0.0], [9.0, 10.0], ["HH"], rcs_power=np.ones((1, 1, 2, 1)))
        with self.assertRaisesRegex(ValueError, "needs complex"):
            translate_phase_center(power_only, x_m=1.0, y_m=0.0, z_m=0.0)


class PhaseCenterGuiTests(_WindowCase):
    def test_button_creates_translated_rows_and_records_script(self):
        window = self.window
        self.assertTrue(window.btn_phase_center.isEnabled())
        window.table.item(0, 1).setData(DATASET_PATH_ROLE, "C:/data/run1.grim")
        with mock.patch("GRIM_Backend.ui.dataset_actions.PhaseCenterDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "x_m": 0.0254, "y_m": 0.0, "z_m": -0.0508,
                "entered": (1.0, 0.0, -2.0), "unit": "in",
            }
            self.select_rows(0)
            window._phase_center_selected()
            deadline = time.monotonic() + 10
            while window._background_job_active() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertIn("Phase centre created 1 dataset(s)", window.status.currentMessage())
        self.assertEqual(window.table.item(3, 0).text(), "Run 1 [PC (1, 0, -2) in]")
        result = window.table.item(3, 0).data(Qt.UserRole)
        np.testing.assert_array_equal(result.rcs_power, self.datasets[0].rcs_power)
        script = window.python_recorder.script
        self.assertIn("translate_phase_center(", script)
        self.assertIn("z_m=-0.0508", script)


class RangeFrequencyTests(_WindowCase):
    def select_new_row(self, grid, name):
        window = self.window
        window._add_dataset_row(grid, name, "", file_name="")
        window.table.clearSelection()
        window.table.selectRow(window.table.rowCount() - 1)
        window._on_dataset_selection_changed()
        for widget in (window.list_az, window.list_elev, window.list_freq):
            widget.selectAll()
        window.list_pol.clearSelection()
        window.list_pol.item(0).setSelected(True)

    def peak_ranges(self):
        (mesh,) = self.window.plot_ax.collections
        _x, y_edges, image = mesh._grim_rectilinear_data
        centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        return centres[np.nanargmax(image, axis=0)], np.nanmax(image, axis=0)

    def test_point_scatterer_stays_at_its_range_with_its_level(self):
        window = self.window
        grid = _point_grid((-0.3, 0.0, 0.0), azimuths=(0.0,))
        grid = RcsGrid(grid.azimuths, grid.elevations, grid.frequencies, grid.polarizations,
                       rcs=np.sqrt(2.0) * grid.rcs_slice((slice(None),) * 4), units=grid.units)
        self.select_new_row(grid, "Point")
        self.plot("_plot_range_freq")
        self.assertIn("Range–frequency map updated", window.status.currentMessage())
        peaks, levels = self.peak_ranges()
        np.testing.assert_allclose(peaks, 0.3, atol=0.03)
        np.testing.assert_allclose(levels, 10 * np.log10(2.0), atol=0.1)
        self.assertEqual(window.plot_ax.get_ylabel(), "Down range (m)")
        self.assertEqual(window.plot_ax.get_xlabel(), "Sub-band centre frequency (GHz)")
        self.assertEqual(window.plot_colorbars[0].ax.get_ylabel(), "RCS range profile (dBsm)")

        window.analysis_controls.combo_range_unit.setCurrentText("in")
        peaks, _levels = self.peak_ranges()
        np.testing.assert_allclose(peaks, 0.3 / 0.0254, atol=1.2)
        window.btn_slider.setChecked(True)
        window.plot_slider.combo_axis.setCurrentIndex(0)
        window._on_plot_slider_moved(3)
        self.assertIn("This plot sweeps frequency", window.status.currentMessage())

    def test_blocks_phase_and_skips_unusable_sweeps(self):
        window = self.window
        window.btn_phase.setChecked(True)
        self.plot("_plot_range_freq")
        self.assertIn("Turn off Phase", window.status.currentMessage())
        window.btn_phase.setChecked(False)
        uneven = RcsGrid([0.0], [0.0], np.r_[np.linspace(8, 9, 9), 9.5], ["HH"],
                         rcs=np.ones((1, 1, 10, 1), complex), units={"frequency": "GHz"})
        self.select_new_row(uneven, "Uneven")
        self.plot("_plot_range_freq")
        self.assertIn("Uneven (needs uniformly spaced selected frequencies)",
                      window.status.currentMessage())

    def test_subband_starts_are_bounded(self):
        self.assertEqual(subband_starts(10, 4), [0, 1, 2, 3, 4, 5, 6])
        starts = subband_starts(2001, 100)
        self.assertLessEqual(len(starts), 200)
        self.assertEqual(starts[0], 0)
        self.assertLessEqual(starts[-1], 2001 - 100)


if __name__ == "__main__":
    unittest.main()
