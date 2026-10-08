"""Headless ordinary overlays retain native samples and separate polarizations."""

from __future__ import annotations

import unittest

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets


def _grid(*, fine=False, converted=False, polarizations=("HH", "VV"), width=False):
    az = np.arange(-2.0, 2.01, 0.5 if fine else 1.0)[::-1]
    el = np.arange(-10.0, 10.01, 5.0 if fine else 10.0)[::-1]
    fr = np.arange(9.0, 11.01, 0.5 if fine else 1.0)[::-1]
    factors = np.asarray([1.0 if pol == "HH" else 4.0 for pol in polarizations])
    power = (50.0 + az[:, None, None, None] ** 2
             + 2.0 * el[None, :, None, None]
             + 3.0 * fr[None, None, :, None]) * factors
    phase = np.zeros_like(power)
    for index, pol in enumerate(polarizations):
        phase[..., index] = np.deg2rad(30.0 if pol == "HH" else -60.0)
    return RcsGrid(
        np.deg2rad(az) if converted else az,
        np.deg2rad(el) if converted else el,
        fr * 1.0e9 if converted else fr,
        list(polarizations), rcs_power=power, rcs_phase=phase,
        units={
            "azimuth": "rad" if converted else "deg",
            "elevation": "rad" if converted else "deg",
            "frequency": "Hz" if converted else "GHz",
            "rcs_linear_quantity": "sigma_2d" if width else "sigma_3d",
            "rcs_log_unit": "dBke" if width else "dBsm",
        },
        extra={"phase_reference": "origin", "time_convention": "exp(+jwt)", "polarization_basis": "HV"},
    )


class NativeOverlayReplayTests(unittest.TestCase):
    def setUp(self):
        self.coarse = _grid()
        self.fine = _grid(fine=True, converted=True)

    def _plot(self, mode, **overrides):
        options = dict(
            datasets=[("coarse", self.coarse), ("fine", self.fine)],
            mode=mode, azimuths=self.coarse.azimuths,
            elevations=self.coarse.elevations if mode == "elevation_sweep" else [0.0],
            frequencies=self.coarse.frequencies if mode == "frequency" else [10.0],
            polarization=("HH", "VV"), scale="linear",
        )
        options.update(overrides)
        figure = plot_datasets(**options)
        self.addCleanup(figure.clear)
        return figure.axes[0]

    def test_rect_and_polar_plot_each_native_sweep_with_independent_polarizations(self):
        for mode in ("azimuth_rect", "azimuth_polar"):
            with self.subTest(mode=mode):
                axis = self._plot(mode)
                self.assertEqual(len(axis.lines), 4)
                for index, line in enumerate(axis.lines):
                    az = np.arange(-2.0, 2.01, 1.0 if index < 2 else 0.5)
                    np.testing.assert_allclose(line.get_xdata(), np.deg2rad(az) if mode == "azimuth_polar" else az)
                    factor = 1.0 if index % 2 == 0 else 4.0
                    np.testing.assert_allclose(line.get_ydata(), (80.0 + az ** 2) * factor)
                    self.assertIn("HH" if index % 2 == 0 else "VV", line.get_label())
                    self.assertEqual(line.get_linestyle(), "-")

    def test_frequency_keeps_native_frequency_samples_and_separate_p50(self):
        axis = self._plot("frequency")
        self.assertEqual(len(axis.lines), 4)
        for index, line in enumerate(axis.lines):
            frequencies = np.arange(9.0, 11.01, 1.0 if index < 2 else 0.5)
            np.testing.assert_allclose(line.get_xdata(), frequencies)
            factor = 1.0 if index % 2 == 0 else 4.0
            np.testing.assert_allclose(line.get_ydata(), (51.0 + 3.0 * frequencies) * factor)
            self.assertIn("HH" if index % 2 == 0 else "VV", line.get_label())

    def test_elevation_keeps_native_elevation_samples_and_separate_p50(self):
        axis = self._plot("elevation_sweep")
        self.assertEqual(len(axis.lines), 4)
        for index, line in enumerate(axis.lines):
            elevations = np.arange(-10.0, 10.01, 10.0 if index < 2 else 5.0)
            np.testing.assert_allclose(line.get_xdata(), elevations)
            factor = 1.0 if index % 2 == 0 else 4.0
            np.testing.assert_allclose(line.get_ydata(), (81.0 + 2.0 * elevations) * factor)

    def test_missing_polarization_skips_only_that_curve(self):
        self.fine = _grid(fine=True, converted=True, polarizations=("VV",))
        for mode in ("azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep"):
            with self.subTest(mode=mode):
                axis = self._plot(mode)
                self.assertEqual(len(axis.lines), 3)
                self.assertIn("fine | VV", axis.lines[-1].get_label())
                self.assertEqual(axis.lines[-1].get_linestyle(), "-")

    def test_all_polarizations_default_to_solid(self):
        polarizations = ("HH", "VV", "HV", "VH")
        self.coarse = _grid(polarizations=polarizations)
        self.fine = _grid(fine=True, converted=True, polarizations=polarizations)
        for mode in ("azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep"):
            with self.subTest(mode=mode):
                axis = self._plot(mode, polarization=polarizations)
                self.assertEqual(len(axis.lines), 8)
                self.assertEqual({line.get_linestyle() for line in axis.lines}, {"-"})

    def test_disconnected_selections_do_not_include_unselected_native_intervals(self):
        axis = self._plot("azimuth_rect", azimuths=[-2.0, 2.0])
        for line in axis.lines:
            np.testing.assert_allclose(line.get_xdata(), [-2.0, 2.0])

    def test_singleton_azimuth_and_fixed_cuts_are_not_interpolated(self):
        self.fine.azimuths += np.deg2rad(0.2)
        axis = self._plot("azimuth_rect", azimuths=[0.0])
        self.assertEqual(len(axis.lines), 2)
        self.fine = _grid(fine=True, converted=True)
        self.fine.elevations += np.deg2rad(0.25)
        for mode in ("azimuth_rect", "frequency"):
            with self.subTest(mode=mode):
                self.assertEqual(len(self._plot(mode).lines), 2)
        self.fine = _grid(fine=True, converted=True)
        self.fine.frequencies += 0.25e9
        self.assertEqual(len(self._plot("elevation_sweep").lines), 2)

    def test_phase_curves_remain_separate_in_all_four_modes(self):
        for mode in ("azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep"):
            with self.subTest(mode=mode):
                axis = self._plot(mode, phase=True)
                for index, line in enumerate(axis.lines):
                    np.testing.assert_allclose(line.get_ydata(), 30.0 if index % 2 == 0 else -60.0)

    def test_native_grid_and_polarizations_keep_mixed_db_conventions(self):
        self.fine = _grid(fine=True, converted=True, width=True)
        axis = self._plot("azimuth_rect", scale="dbsm")
        self.assertEqual(axis.get_ylabel(), "Mixed dB")
        for index, line in enumerate(axis.lines):
            az = np.arange(-2.0, 2.01, 1.0 if index < 2 else 0.5)
            factor = 1.0 if index % 2 == 0 else 4.0
            expected = 10.0 * np.log10((80.0 + az ** 2) * factor)
            if index >= 2:
                expected += 10.0 * np.log10(2.0 * np.pi * 10.0e9 / 299_792_458.0)
            np.testing.assert_allclose(line.get_ydata(), expected)
            self.assertIn("dBsm" if index < 2 else "dBke", line.get_label())

    def test_analysis_modes_still_require_one_polarization(self):
        for mode in ("delta_map", "isar_image"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "exactly one polarization"):
                    self._plot(mode, datasets=[("coarse", self.coarse)], scale="dbsm")

    def test_single_polarization_sequence_matches_string_and_duplicates_are_ignored(self):
        as_string = self._plot("frequency", polarization="HH")
        as_sequence = self._plot("frequency", polarization=("HH", "HH"))
        self.assertEqual(len(as_sequence.lines), 2)
        for left, right in zip(as_string.lines, as_sequence.lines):
            np.testing.assert_allclose(left.get_xdata(), right.get_xdata())
            np.testing.assert_allclose(left.get_ydata(), right.get_ydata())
        for mode in ("azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep"):
            with self.subTest(mode=mode):
                single_vv = self._plot(mode, polarization="VV")
                self.assertTrue(all(line.get_linestyle() == "-" for line in single_vv.lines))


if __name__ == "__main__":
    unittest.main()
