"""Ordinary visual overlays use native samples and separate polarizations."""

import unittest
from unittest import mock

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode, elevation_sweep_mode,
    cdf_mode, sector_stats_mode, common,
)
from test_plot_renderer_correctness import _RendererHarness
from GRIM_Backend.scripting.recorder import DatasetReference


def grid(azimuths=(0, 0.5, 1, 1.5, 2), *, elevations=(0,), frequencies=(9,),
         polarizations=("HH", "VV"), converted=False, scale=1):
    az = np.asarray(azimuths, dtype=float)
    el = np.asarray(elevations, dtype=float)
    fr = np.asarray(frequencies, dtype=float)
    power = (scale * (az[:, None, None, None] + 3)
             * np.ones((len(az), len(el), len(fr), len(polarizations))))
    for index, pol in enumerate(polarizations):
        if pol == "VV":
            power[..., index] *= 100
    return RcsGrid(
        np.deg2rad(az) if converted else az,
        np.deg2rad(el) if converted else el,
        fr * 1e9 if converted else fr,
        polarizations, rcs_power=power, rcs_phase=np.zeros_like(power),
        units={"azimuth": "rad" if converted else "deg",
               "elevation": "rad" if converted else "deg",
               "frequency": "Hz" if converted else "GHz"},
        extra={"phase_reference": "origin", "time_convention": "exp(+jwt)",
               "polarization_basis": "HV"},
    )


class NativeOverlayTests(unittest.TestCase):
    def harness(self, datasets, *, selected_az=None, polarizations=("HH",)):
        reference = datasets[0][1]
        owner = _RendererHarness(datasets, selections={
            "azimuth": reference.azimuths if selected_az is None else selected_az,
            "elevation": reference.elevations,
            "frequency": reference.frequencies,
            "polarization": polarizations,
        })
        self.addCleanup(owner.plot_figure.clear)
        return owner

    def test_different_azimuth_spacing_in_both_orders_uses_actual_samples(self):
        fine, coarse = grid(), grid((0, 1, 2))
        for first, second in ((fine, coarse), (coarse, fine)):
            for mode in (azimuth_rect_mode, azimuth_polar_mode):
                with self.subTest(first=len(first.azimuths), mode=mode.__name__):
                    owner = self.harness([("A", first), ("B", second)])
                    mode.render(owner)
                    self.assertIn("updated", owner.status.message)
                    self.assertEqual(len(owner.plot_ax.lines), 2)
                    for line, dataset in zip(owner.plot_ax.lines, (first, second)):
                        expected_x = dataset.azimuths
                        if mode is azimuth_polar_mode:
                            expected_x = np.deg2rad(expected_x)
                        np.testing.assert_allclose(line.get_xdata(), expected_x)
                        np.testing.assert_allclose(line.get_ydata(), 10 * np.log10(dataset.azimuths + 3))

    def test_native_samples_convert_axis_units_and_keep_disconnected_selection(self):
        fine, coarse = grid(), grid((0, 1, 2), converted=True)
        owner = self.harness([("A", fine), ("B", coarse)], selected_az=[0, .5, 2])
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 2)
        np.testing.assert_allclose(owner.plot_ax.lines[0].get_xdata(), [0, .5, 2])
        np.testing.assert_allclose(owner.plot_ax.lines[1].get_xdata(), [0, 2])
        self.assertIn("9 GHz", owner.plot_ax.lines[1].get_label())

    def test_fixed_cuts_are_never_silently_moved(self):
        owner = self.harness([("A", grid()), ("B", grid(frequencies=(10,)))])
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 1)
        self.assertIn("Skipped: B", owner.status.message)
        owner = self.harness([("A", grid()), ("B", grid((.25, .75, 1.25)))], selected_az=[.5])
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 1)

    def test_frequency_and_elevation_sweeps_keep_each_native_grid(self):
        fine = grid(elevations=(0, .5, 1), frequencies=(9, 9.5, 10))
        coarse = grid((0, 1, 2), elevations=(0, 1), frequencies=(9, 10))
        for mode, axis in ((frequency_mode, "frequency"), (elevation_sweep_mode, "elevation")):
            with self.subTest(mode=mode.__name__):
                owner = self.harness([("A", fine), ("B", coarse)])
                # The fixed dimension is an exact shared cut; the plotted and
                # azimuth-reduction dimensions may have different native grids.
                fixed_widget = owner.list_elev if axis == "frequency" else owner.list_freq
                owner._selections[fixed_widget] = [0] if axis == "frequency" else [9]
                mode.render(owner)
                self.assertIn("updated", owner.status.message)
                self.assertEqual(len(owner.plot_ax.lines), 2)
                for line, dataset in zip(owner.plot_ax.lines, (fine, coarse)):
                    expected_x = dataset.frequencies if axis == "frequency" else dataset.elevations
                    np.testing.assert_allclose(line.get_xdata(), expected_x)
                    np.testing.assert_allclose(line.get_ydata(), 10 * np.log10(4))

    def test_multiple_polarizations_remain_independent_in_six_line_modes(self):
        for mode in (azimuth_rect_mode, azimuth_polar_mode, frequency_mode,
                     elevation_sweep_mode, cdf_mode, sector_stats_mode):
            with self.subTest(mode=mode.__name__):
                owner = self.harness([("Data", grid())], polarizations=("HH", "VV"))
                mode.render(owner)
                self.assertIn("updated", owner.status.message)
                self.assertEqual(len(owner.plot_ax.lines), 2)
                hh, vv = owner.plot_ax.lines
                self.assertIn("Pol HH", hh.get_label())
                self.assertIn("Pol VV", vv.get_label())
                self.assertEqual(hh.get_linestyle(), "-")
                self.assertEqual(vv.get_linestyle(), "-")
                first = hh.get_xdata() if mode is cdf_mode else hh.get_ydata()
                second = vv.get_xdata() if mode is cdf_mode else vv.get_ydata()
                valid = np.isfinite(first) & np.isfinite(second)
                np.testing.assert_allclose(np.asarray(second)[valid] - np.asarray(first)[valid], 20)

    def test_missing_polarization_does_not_hide_available_curves_and_hold_replaces(self):
        owner = self.harness([("A", grid()), ("B", grid(polarizations=("HH",)))],
                             polarizations=("HH", "VV"))
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 3)
        self.assertIn("Skipped: B | Pol VV", owner.status.message)
        owner.btn_hold.checked = True
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 3)

    def test_pbp_keeps_a_separate_band_for_each_polarization(self):
        for mode in (azimuth_rect_mode, azimuth_polar_mode, frequency_mode):
            with self.subTest(mode=mode.__name__):
                owner = self.harness([("A", grid()), ("B", grid(scale=2))],
                                     polarizations=("HH", "VV"))
                owner.btn_pbp.checked = True
                mode.render(owner)
                self.assertIn("updated", owner.status.message)
                labels = owner.plot_ax.get_legend_handles_labels()[1]
                self.assertEqual(len(labels), 2)
                self.assertIn("[HH]", labels[0])
                self.assertIn("[VV]", labels[1])
                # Two edges per band; corresponding VV levels are exactly 20 dB above HH.
                lines = owner.plot_ax.lines
                self.assertEqual(len(lines), 4)
                np.testing.assert_allclose(lines[2].get_ydata() - lines[0].get_ydata(), 20)
                np.testing.assert_allclose(lines[3].get_ydata() - lines[1].get_ydata(), 20)

    def test_native_grid_multi_polarization_mixed_db_overlay_keeps_all_values(self):
        fine, coarse = grid(), grid((0, 1, 2))
        coarse.units.update(rcs_linear_quantity="sigma_2d", rcs_log_unit="dBke")
        datasets = [("Measured", fine), ("Analysis", coarse)]
        owner = self.harness(datasets, polarizations=("HH", "VV"))
        azimuth_rect_mode.render(owner)
        self.assertIn("updated", owner.status.message)
        self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
        self.assertEqual(len(owner.plot_ax.lines), 4)
        for name, dataset in datasets:
            for pol_index, pol in enumerate(dataset.polarizations):
                (line,) = [line for line in owner.plot_ax.lines
                           if line.get_label().startswith(name + " [")
                           and f"Pol {pol}," in line.get_label()]
                np.testing.assert_allclose(line.get_xdata(), dataset.azimuths)
                expected = 10 * np.log10(dataset.rcs_power[:, 0, 0, pol_index])
                if dataset is coarse:
                    expected += 10 * np.log10(2 * np.pi * 9.0e9 / 299_792_458.0)
                np.testing.assert_allclose(line.get_ydata(), expected)
                self.assertIn(f"[{dataset.default_log_unit()}]", line.get_label())
                self.assertEqual(line.get_linestyle(), "-")

    def test_native_selector_rejects_duplicate_coordinates(self):
        with self.assertRaisesRegex(ValueError, "duplicate source coordinates"):
            common.native_axis_selection(grid(), grid((0, 1, 1, 2)), "azimuth", [0, .5, 1, 1.5, 2])

    def test_many_disconnected_spans_preserve_only_selected_intervals(self):
        reference = grid(np.arange(0, 1000, .5))
        source = grid(np.arange(0, 1000, .25))
        selected = reference.azimuths[np.arange(reference.azimuths.size) % 4 < 2]
        indices = common.native_axis_selection(reference, source, "azimuth", selected)
        expected = source.azimuths[np.mod(source.azimuths, 2) <= .5]
        np.testing.assert_array_equal(source.azimuths[indices], expected)

    def test_plot_recorder_keeps_all_selected_polarizations(self):
        dataset = grid()
        owner = self.harness([("Data", dataset)], polarizations=("HH", "VV"))
        owner.python_recorder = mock.Mock()
        owner._python_reference_for_dataset = lambda _ds: DatasetReference("data-id", "Data")
        azimuth_rect_mode.render(owner)
        owner._record_python_plot("azimuth_rect")
        self.assertEqual(owner.last_python_plot_spec[0], "supported")
        parameters = owner.python_recorder.record_plot.call_args.kwargs["parameters"]
        self.assertEqual(parameters["polarization"], ("HH", "VV"))
        self.assertEqual(parameters["azimuths"], list(dataset.azimuths))

    def test_autoplot_carries_selection_adjustment_notice_once(self):
        owner = self.harness([("Data", grid())])
        notice = "Selection adjusted: elevation unavailable; using default."
        owner._pending_parameter_selection_notice = notice
        azimuth_rect_mode.render(owner)
        self.assertIn("updated", owner.status.message)
        self.assertIn(notice, owner.status.message)
        self.assertIsNone(owner._pending_parameter_selection_notice)
        azimuth_rect_mode.render(owner)
        self.assertNotIn(notice, owner.status.message)


if __name__ == "__main__":
    unittest.main()
