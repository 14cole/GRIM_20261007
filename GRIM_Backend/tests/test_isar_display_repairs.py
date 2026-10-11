"""Regression tests for shared image controls and truthful pooled-image probes."""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QToolButton

from GRIM_Backend.isar.geometry import pool_image_peaks, attach_block_peak_probe
from GRIM_Backend.plotting.modes import az_vs_range_mode, isar_mode, isar_render, common
from test_reconstruction_accuracy import harness, point_grid


class SharedIsarControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = harness(point_grid(np.arange(0., 16.), np.linspace(8e9, 12e9, 33)))
        self.widgets = []
        for name, value in (("spin_isar_ap_center", 4.), ("spin_isar_ap_width", 4.),
                            ("spin_plot_zmin", -80.), ("spin_plot_zmax", 0.),
                            ("spin_isar_peak_drop", 20.)):
            spin = QDoubleSpinBox()
            spin.setRange(-1000., 1000.)
            spin.setValue(value)
            self.widgets.append(spin)
            setattr(self.owner, name, spin)
        self.owner.chk_isar_aperture = QCheckBox()
        self.owner.btn_isar_ap_play = QToolButton()
        self.owner.btn_isar_ap_play.setCheckable(True)
        self.widgets += [self.owner.chk_isar_aperture, self.owner.btn_isar_ap_play]
        self.owner._plot_az_vs_range = mock.Mock(side_effect=lambda: az_vs_range_mode.render(self.owner))
        self.owner._plot_isar_image = mock.Mock()
        self.owner.chk_isar_aperture.toggled.connect(self.owner._on_isar_window_changed)
        self.owner.spin_isar_ap_center.valueChanged.connect(self.owner._on_isar_window_changed)
        self.owner.spin_isar_ap_width.valueChanged.connect(self.owner._on_isar_window_changed)
        self.owner.spin_plot_zmin.valueChanged.connect(self.owner._on_isar_clim_changed)
        self.owner.spin_plot_zmax.valueChanged.connect(self.owner._on_isar_clim_changed)
        self.owner.btn_isar_ap_play.toggled.connect(self.owner._on_isar_ap_play)
        az_vs_range_mode.render(self.owner)

    def tearDown(self):
        self.owner.btn_isar_ap_play.setChecked(False)
        self.owner.plot_figure.clear()
        for widget in self.widgets:
            widget.deleteLater()
        self.app.processEvents()

    def test_live_limits_and_peak_scale_reuse_range_image(self):
        image = self.owner._az_vs_range_mesh
        self.owner.spin_plot_zmin.setValue(-50.)
        self.owner.spin_plot_zmax.setValue(-5.)
        self.assertEqual(image.get_clim(), (-50., -5.))
        self.owner._on_isar_peak_scale()
        peak = float(np.nanmax(image.get_array()))
        np.testing.assert_allclose(image.get_clim(), (peak - 20., peak), atol=.006)
        self.owner._plot_az_vs_range.assert_not_called()
        self.owner._plot_isar_image.assert_not_called()

    def test_aperture_step_enables_aperture_and_forms_one_range_map(self):
        self.owner._on_isar_ap_next()
        self.assertTrue(self.owner.chk_isar_aperture.isChecked())
        self.assertEqual(self.owner.spin_isar_ap_center.value(), 6.)
        self.assertEqual(self.owner.last_plot_mode, "az_vs_range")
        self.owner._plot_az_vs_range.assert_called_once()
        self.owner._plot_isar_image.assert_not_called()
        self.owner._on_isar_ap_prev()
        self.assertEqual(self.owner.spin_isar_ap_center.value(), 4.)
        self.assertEqual(self.owner._plot_az_vs_range.call_count, 2)

    def test_play_advances_range_frames_and_stops_without_more_work(self):
        with mock.patch("GRIM_Backend.plotting.actions.QTimer.singleShot") as later:
            self.owner.btn_isar_ap_play.setChecked(True)
            self.assertEqual(self.owner.last_plot_mode, "az_vs_range")
            self.assertEqual(self.owner._plot_az_vs_range.call_count, 1)
            later.assert_called_once()
            next_frame = later.call_args.args[1]
            next_frame()
            self.assertEqual(self.owner._plot_az_vs_range.call_count, 2)
            self.owner.btn_isar_ap_play.setChecked(False)
            next_frame()
            self.assertEqual(self.owner._plot_az_vs_range.call_count, 2)
            self.owner._plot_isar_image.assert_not_called()

    def test_failed_range_frame_stops_play(self):
        self.owner.spin_isar_ap_center.blockSignals(True)
        self.owner.spin_isar_ap_center.setValue(100.)
        self.owner.spin_isar_ap_center.blockSignals(False)
        with mock.patch("GRIM_Backend.plotting.actions.QTimer.singleShot") as later:
            self.owner.btn_isar_ap_play.setChecked(True)
            self.assertFalse(self.owner.btn_isar_ap_play.isChecked())
            later.assert_not_called()
            self.owner._plot_isar_image.assert_not_called()

    def test_restarting_play_discards_the_previous_queued_frame(self):
        with mock.patch("GRIM_Backend.plotting.actions.QTimer.singleShot") as later:
            self.owner.btn_isar_ap_play.setChecked(True)
            previous_frame = later.call_args.args[1]
            self.owner.btn_isar_ap_play.setChecked(False)
            self.owner.btn_isar_ap_play.setChecked(True)
            self.assertEqual(self.owner._plot_az_vs_range.call_count, 2)
            previous_frame()
            self.assertEqual(self.owner._plot_az_vs_range.call_count, 2)


class PooledImageProbeTests(unittest.TestCase):
    def test_native_peak_indices_skip_masks_and_do_not_retain_source(self):
        import gc
        import weakref
        values = np.ma.array(np.arange(35., dtype=np.float32).reshape(5, 7), mask=False)
        values.mask[:2, :3] = True
        values.data[1, 4] = 500.
        values.mask[1, 4] = True
        values.data[0, 5] = 100.
        source = weakref.ref(values)
        pooled, iy, ix = pool_image_peaks(values, 2, [(0, 3), (3, 6), (6, 7)])
        self.assertTrue(np.isnan(pooled[0, 0]))
        self.assertEqual((ix[0, 0], iy[0, 0]), (-1, -1))
        self.assertEqual((ix[0, 1], iy[0, 1], pooled[0, 1]), (5, 0, 100.))
        self.assertEqual((ix[-1, -1], iy[-1, -1]), (6, 4))
        del values
        gc.collect()
        self.assertIsNone(source())
        self.assertEqual(ix.dtype, np.int32)
        self.assertEqual(iy.nbytes + ix.nbytes, pooled.size * 8)

    def test_probe_reports_native_location_for_descending_axes_and_gaps(self):
        values = np.zeros((4, 6), dtype=np.float32)
        values[1, 1] = 7.
        pooled, iy, ix = pool_image_peaks(values, 2, [(0, 3), None, (3, 6)])
        figure = Figure()
        artist = figure.add_subplot().imshow(pooled)
        attach_block_peak_probe(artist, [6., 3., 2., -1.], [4., 2., 0.],
            [5.5, 4.5, 3.5, 1.5, .5, -.5], [3.5, 2.5, 1.5, .5], ix, iy)
        self.assertEqual(artist._grim_native_probe(5.5, 3.5), (4.5, 2.5, 7.))
        self.assertIsNone(artist._grim_native_probe(2.5, 3.))
        self.assertIsNone(artist._grim_native_probe(9., 3.))
        self.assertTrue(artist._grim_display_reduced)
        figure.clear()

    def test_reduced_isar_probes_match_full_pixels_after_coordinate_flips(self):
        dataset = point_grid(np.linspace(-5., 5., 17), np.linspace(8e9, 12e9, 33), x=.1, y=.13)
        for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
            options = dict(flip_x=flip_x, flip_y=flip_y, reconstruction="fast")
            full = isar_mode.form_isar(dataset, **options)[0][0]
            with mock.patch.object(common, "MAX_IMAGE_SIDE", 8):
                bands, elapsed = isar_mode.form_isar(dataset, decimate_display=True, **options)
            band = bands[0]
            ix, iy = band["display_peak_x_indices"], band["display_peak_y_indices"]
            np.testing.assert_allclose(band["magnitude"].T, full["magnitude"][ix, iy])
            owner = harness(dataset)
            isar_render.display_results(owner, dict(dataset=dataset, unit_name="m", az_target_deg=None,
                elevation_deg=0., pol_idx=0, recon="fft"), bands, elapsed)
            row, column = np.unravel_index(np.argmax(band["magnitude"].T), ix.shape)
            x = np.mean(band["display_x_edges"][column:column+2])
            y = np.mean(band["display_y_edges"][row:row+2])
            actual = owner._isar_meshes[0]._grim_native_probe(x, y)
            expected = (full["x_range"][ix[row, column]], full["y_range"][iy[row, column]],
                        20. * np.log10(full["magnitude"][ix[row, column], iy[row, column]]))
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            label = SimpleNamespace(setText=mock.Mock())
            owner._on_plot_hover(SimpleNamespace(inaxes=owner.plot_ax, xdata=x, ydata=y), label)
            self.assertIn("Overview block peak", label.setText.call_args.args[0])
            self.assertIn("Display reduced", owner.plot_figure._grim_reduction_notice.get_text())
            owner.plot_figure.clear()

    def test_reduced_range_map_keeps_native_peak_coordinates_and_gap(self):
        angles = np.r_[np.arange(12.), np.arange(40., 52.)]
        ranges = np.arange(16.)
        values = np.ones((16, 24), dtype=np.float32)
        values[5, 4] = 90.
        x, y, pooled, reduced, indices = az_vs_range_mode._range_display_grid(
            angles, ranges, values, azimuth_width=1., max_side=8, return_peaks=True)
        self.assertTrue(reduced)
        ix, iy = indices
        row, column = np.unravel_index(np.nanargmax(pooled), pooled.shape)
        self.assertEqual((ix[row, column], iy[row, column]), (4, 5))
        self.assertTrue(np.any(np.all(np.isnan(pooled), axis=0)))
        figure = Figure()
        artist = figure.add_subplot().pcolormesh(x, y, pooled)
        attach_block_peak_probe(artist, x, y, angles, ranges, ix, iy)
        self.assertEqual(artist._grim_native_probe(np.mean(x[column:column+2]),
            np.mean(y[row:row+2])), (4., 5., 90.))
        figure.clear()


if __name__ == "__main__":
    unittest.main()
