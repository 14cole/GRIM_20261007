"""Color limits follow displayed units without leaking between Plotting and ISAR."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from matplotlib.ticker import FixedLocator
from PySide6.QtWidgets import QApplication

from GRIM_Backend.plotting.modes import isar_mode
import test_gui_shell as shell
from test_plot_renderer_correctness import _grid


class ColorScaleSwitchingTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp
    tearDown = shell.UnifiedGuiShellTest.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def load_sample(self):
        grid = _grid()
        grid.rcs_power[:] = np.geomspace(.01, 1000., grid.rcs_power.size).reshape(grid.rcs_power.shape)
        grid.units["time_convention"] = "exp(+jwt)"
        self.window.btn_auto_plot.setChecked(False)
        self.window._add_dataset_row(grid, "Measured", "Loaded", "measured.grim")
        self.window.table.selectRow(0)
        for listing in (self.window.list_az, self.window.list_elev,
                        self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        return grid

    @staticmethod
    def limits(context):
        return tuple(getattr(context, name).value() for name in
                     ("spin_plot_zmin", "spin_plot_zmax", "spin_plot_zstep"))

    @staticmethod
    def set_limits(context, values):
        for name, value in zip(("spin_plot_zmin", "spin_plot_zmax", "spin_plot_zstep"), values):
            spin = getattr(context, name)
            previous = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(previous)

    @staticmethod
    def scale(context, value):
        context.combo_plot_scale.setCurrentIndex(context.combo_plot_scale.findData(value))

    def image(self):
        axis = self.window.plot_ax
        return (list(axis.images) + list(axis.collections))[0]

    def render_isar(self, dataset, count):
        params = {"dataset": dataset, "unit_name": "m", "az_target_deg": None,
                  "elevation_deg": 0., "pol_idx": 0, "recon": "fft"}
        bands = [{"x_range": np.array([-1., 1.]), "y_range": np.array([-1., 1.]),
                  "az_values": np.array([-1., 1.]), "az_nonuniformity": 0.,
                  "freq_nonuniformity": 0.,
                  "magnitude": np.array([[.01, .1], [1., 10.]]) * (index + 1)}
                 for index in range(count)]
        isar_mode.display_results(self.window, params, bands, .01)

    def test_isar_scale_switch_waits_for_apply_then_fits_single_and_multiple_bands(self):
        dataset = self.load_sample()
        original_power = dataset.rcs_power.copy()
        self.window._activate_plot_tab("isar")
        self.window.last_plot_mode = "isar_image"
        self.window.chk_colorbar.setChecked(True)
        self.window.chk_colorbar_shared.setChecked(True)
        for count in (1, 2):
            with self.subTest(bands=count):
                self.scale(self.window, "dbsm")
                self.render_isar(dataset, count)
                for target in ("linear", "dbsm"):
                    with self.subTest(scale=target):
                        custom = (-30., 10., 5.) if target == "linear" else (.01, 100., 20.)
                        self.set_limits(self.window, custom)
                        self.window._on_isar_clim_changed()
                        previous_meshes = list(self.window._isar_meshes)
                        previous_arrays = [np.array(mesh.get_array()) for mesh in previous_meshes]
                        previous_ticks = [bar.get_ticks().copy() for bar in self.window.plot_colorbars]
                        with mock.patch.object(self.window, "_plot_isar_image",
                                               side_effect=lambda: self.render_isar(dataset, count)) as render:
                            self.scale(self.window, target)
                            render.assert_not_called()
                            self.assertEqual(self.limits(self.window), (0., 0., 0.))
                            for mesh, values in zip(previous_meshes, previous_arrays):
                                self.assertIn(mesh, mesh.axes.images)
                                self.assertEqual(mesh.get_clim(), custom[:2])
                                np.testing.assert_array_equal(mesh.get_array(), values)
                            for bar, ticks in zip(self.window.plot_colorbars, previous_ticks):
                                np.testing.assert_array_equal(bar.get_ticks(), ticks)
                            self.window.btn_isar_apply.click()
                            render.assert_called_once_with()
                        peak = 100. * count ** 2
                        expected_clim = (.0001, peak) if target == "linear" else (
                            10. * np.log10(peak) - 60., 10. * np.log10(peak))
                        for index, mesh in enumerate(self.window._isar_meshes):
                            np.testing.assert_allclose(mesh.get_clim(), expected_clim, rtol=1.e-6)
                            expected = np.array([[.0001, 1.], [.01, 100.]]) * (index + 1) ** 2
                            if target == "dbsm":
                                expected = 10. * np.log10(expected)
                            np.testing.assert_allclose(mesh.get_array(), expected, rtol=1.e-6, atol=1.e-6)
                            record = mesh._grim_image_record
                            np.testing.assert_array_equal(record.values, mesh.get_array())
                            self.assertEqual(record.logarithmic, target == "dbsm")
                        self.assertEqual(self.window.spin_plot_zstep.value(), 0.)
                        for bar in self.window.plot_colorbars:
                            self.assertNotIsInstance(bar.locator, FixedLocator)
        np.testing.assert_array_equal(dataset.rcs_power, original_power)

    def test_same_scale_isar_rerender_preserves_manual_clamp_and_tick_step(self):
        dataset = self.load_sample()
        self.window._activate_plot_tab("isar")
        self.window.last_plot_mode = "isar_image"
        self.window.chk_colorbar.setChecked(True)
        for scale, custom in (("dbsm", (-25., 15., 10.)), ("linear", (.1, 10., 2.))):
            with self.subTest(scale=scale):
                self.scale(self.window, scale)
                self.render_isar(dataset, 2)
                self.set_limits(self.window, custom)
                self.render_isar(dataset, 2)
                self.assertEqual(self.limits(self.window), custom)
                for mesh in self.window._isar_meshes:
                    np.testing.assert_allclose(mesh.get_clim(), custom[:2])
                for bar in self.window.plot_colorbars:
                    np.testing.assert_allclose(np.diff(bar.get_ticks()), custom[2])

    def test_related_image_plots_refit_and_update_native_values_in_both_directions(self):
        dataset = self.load_sample()
        original_power, original_phase = dataset.rcs_power.copy(), dataset.rcs_phase.copy()
        self.window.chk_colorbar.setChecked(True)
        for mode in ("waterfall", "az_vs_range", "range_freq"):
            with self.subTest(mode=mode):
                self.scale(self.window, "dbsm")
                render = getattr(self.window, "_plot_" + mode)
                render()
                self.assertEqual(self.window.last_plot_mode, mode)
                mesh = self.image()
                logarithmic = np.array(mesh.get_array())
                self.set_limits(self.window, (-40., 0., 10.))
                for target in ("linear", "dbsm"):
                    with self.subTest(scale=target):
                        self.scale(self.window, target)
                        mesh = self.image()
                        values = np.asarray(mesh.get_array())
                        finite = values[np.isfinite(values)]
                        self.assertGreater(float(finite.max()), float(finite.min()))
                        low, high = mesh.get_clim()
                        self.assertAlmostEqual(high, float(finite.max()), places=5)
                        if target == "linear":
                            self.assertGreaterEqual(low, 0.)
                            self.assertAlmostEqual(low, float(finite.min()), places=7)
                            np.testing.assert_allclose(values, 10. ** (logarithmic / 10.), rtol=2.e-6, atol=1.e-12)
                        else:
                            np.testing.assert_allclose(values, logarithmic, rtol=1.e-6, atol=1.e-6)
                        self.assertEqual(self.window.spin_plot_zstep.value(), 0.)
                        native = getattr(mesh, "_grim_native_rectilinear_data", None)
                        if native is not None:
                            np.testing.assert_allclose(native[2], values)
                        record = getattr(mesh, "_grim_image_record", None)
                        if record is not None:
                            np.testing.assert_allclose(record.values, values)
                        for bar in self.window.plot_colorbars:
                            self.assertNotIsInstance(bar.locator, FixedLocator)
                        self.set_limits(self.window, (.01, .1, .02))
        np.testing.assert_array_equal(dataset.rcs_power, original_power)
        np.testing.assert_array_equal(dataset.rcs_phase, original_phase)

    def test_color_reset_is_local_to_the_edited_context_even_when_inactive(self):
        plot, isar = (self.window._plot_contexts[key] for key in ("plotting", "isar"))
        self.set_limits(plot, (-40., 0., 10.))
        self.set_limits(isar, (-60., 20., 20.))
        # The ISAR settings popup can be edited while Plotting is active.
        self.scale(isar, "linear")
        self.assertEqual(self.limits(isar), (0., 0., 0.))
        self.assertEqual(self.limits(plot), (-40., 0., 10.))
        self.window._activate_plot_tab("isar")
        self.set_limits(isar, (.01, 100., 10.))
        self.scale(plot, "linear")
        self.assertEqual(self.limits(plot), (0., 0., 0.))
        self.assertEqual(self.limits(isar), (.01, 100., 10.))

    def test_scale_change_on_line_plot_clears_clamp_before_next_image(self):
        self.load_sample()
        self.set_limits(self.window, (-60., 0., 10.))
        self.window._plot_frequency()
        self.scale(self.window, "linear")
        self.assertEqual(self.limits(self.window), (0., 0., 0.))
        self.window._plot_waterfall()
        low, high = self.window.plot_ax.collections[0].get_clim()
        self.assertGreater(low, 0.)
        self.assertGreater(high, low)

    def test_reset_blocks_intermediate_color_edits_and_preserves_signal_state(self):
        context = self.window._plot_contexts["isar"]
        self.set_limits(context, (-60., 20., 10.))
        changed = mock.Mock()
        for spin in (context.spin_plot_zmin, context.spin_plot_zmax, context.spin_plot_zstep):
            spin.valueChanged.connect(changed)
        context.spin_plot_zmax.blockSignals(True)
        self.scale(context, "linear")
        changed.assert_not_called()
        self.assertEqual(self.limits(context), (0., 0., 0.))
        self.assertFalse(context.spin_plot_zmin.signalsBlocked())
        self.assertTrue(context.spin_plot_zmax.signalsBlocked())
        self.assertFalse(context.spin_plot_zstep.signalsBlocked())


if __name__ == "__main__":
    unittest.main()
