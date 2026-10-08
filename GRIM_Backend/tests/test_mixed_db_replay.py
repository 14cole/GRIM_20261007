"""Native logarithmic overlays must replay without converting unlike quantities."""

import unittest

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets


def _grid(*, quantity="sigma_3d", log_unit="dBsm", converted_axes=False):
    azimuths = np.asarray([10.0, -10.0, 0.0])
    elevations = np.asarray([5.0, -5.0])
    frequencies = np.asarray([12.0, 8.0, 10.0])
    # Distinct angular and frequency dependence exposes accidental unit
    # conversion, averaging in dB, and sorting the data without its coordinates.
    power = (
        np.asarray([4.0, 1.0, 2.0])[:, None, None, None]
        * np.asarray([3.0, 1.0])[None, :, None, None]
        * np.asarray([5.0, 1.0, 2.0])[None, None, :, None]
    )
    if converted_axes:
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)
        frequencies *= 1.0e9
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        ["HH"],
        rcs_power=power,
        rcs_phase=np.zeros_like(power),
        units={
            "azimuth": "rad" if converted_axes else "deg",
            "elevation": "rad" if converted_axes else "deg",
            "frequency": "Hz" if converted_axes else "GHz",
            "rcs_linear_quantity": quantity,
            "rcs_log_unit": log_unit,
        },
        extra={
            "phase_reference": "origin",
            "time_convention": "exp(+jwt)",
            "polarization_basis": "HV",
        },
    )


class MixedDbReplayTests(unittest.TestCase):
    def setUp(self):
        self.sigma = _grid()
        self.width = _grid(
            quantity="sigma_2d", log_unit="dBke", converted_axes=True
        )
        self.selected = [("Measured", self.sigma), ("Analysis", self.width)]

    def _plot(self, mode, *, datasets=None, **kwargs):
        selection = {
            "azimuths": self.sigma.azimuths,
            "elevations": [-5.0],
            "frequencies": [10.0],
        }
        if mode == "frequency":
            selection["frequencies"] = self.sigma.frequencies
        elif mode == "elevation_sweep":
            selection["elevations"] = self.sigma.elevations
        selection.update(kwargs)
        figure = plot_datasets(
            self.selected if datasets is None else datasets,
            mode=mode,
            polarization="HH",
            **selection,
        )
        self.addCleanup(figure.clear)
        return figure.axes[0]

    def test_rect_and_polar_keep_native_levels_with_mixed_units_and_axis_units(self):
        expected_dbsm = 10.0 * np.log10([2.0, 4.0, 8.0])
        expected_dbke = expected_dbsm + 10.0 * np.log10(
            2.0 * np.pi * 10.0e9 / 299_792_458.0
        )
        for mode in ("azimuth_rect", "azimuth_polar"):
            with self.subTest(mode=mode):
                axis = self._plot(mode)
                self.assertEqual(axis.get_ylabel(), "Mixed dB")
                self.assertEqual(len(axis.lines), 2)
                self.assertIn("dBsm", axis.lines[0].get_label())
                self.assertIn("dBke", axis.lines[1].get_label())
                expected_x = np.asarray([-10.0, 0.0, 10.0])
                if mode == "azimuth_polar":
                    expected_x = np.deg2rad(expected_x)
                for line in axis.lines:
                    np.testing.assert_allclose(line.get_xdata(), expected_x)
                np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_dbsm)
                np.testing.assert_allclose(axis.lines[1].get_ydata(), expected_dbke)

    def test_frequency_p50_keeps_frequency_dependent_native_dbke(self):
        axis = self._plot("frequency")
        self.assertEqual(axis.get_ylabel(), "Mixed dB")
        expected_x = np.asarray([8.0, 10.0, 12.0])
        expected_dbsm = 10.0 * np.log10([2.0, 4.0, 10.0])
        expected_dbke = expected_dbsm + 10.0 * np.log10(
            2.0 * np.pi * expected_x * 1.0e9 / 299_792_458.0
        )
        for line in axis.lines:
            np.testing.assert_allclose(line.get_xdata(), expected_x)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_dbsm)
        np.testing.assert_allclose(axis.lines[1].get_ydata(), expected_dbke)
        self.assertIn("dBsm", axis.lines[0].get_label())
        self.assertIn("dBke", axis.lines[1].get_label())

    def test_elevation_p50_keeps_native_levels(self):
        axis = self._plot("elevation_sweep")
        self.assertEqual(axis.get_ylabel(), "Mixed dB")
        expected_dbsm = 10.0 * np.log10([4.0, 12.0])
        expected_dbke = expected_dbsm + 10.0 * np.log10(
            2.0 * np.pi * 10.0e9 / 299_792_458.0
        )
        for line in axis.lines:
            np.testing.assert_allclose(line.get_xdata(), [-5.0, 5.0])
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_dbsm)
        np.testing.assert_allclose(axis.lines[1].get_ydata(), expected_dbke)
        self.assertIn("dBsm", axis.lines[0].get_label())
        self.assertIn("dBke", axis.lines[1].get_label())

    def test_homogeneous_log_plots_keep_existing_labels(self):
        axis = self._plot(
            "azimuth_rect", datasets=[("A", self.sigma), ("B", _grid())]
        )
        self.assertEqual(axis.get_ylabel(), "RCS (dBsm)")
        self.assertTrue(all("dBsm" not in line.get_label() for line in axis.lines))

    def test_skipped_coordinates_do_not_make_visible_curves_mixed(self):
        self.width.azimuths += 1.0
        for mode in (
            "azimuth_rect", "azimuth_polar", "frequency", "elevation_sweep"
        ):
            with self.subTest(mode=mode):
                axis = self._plot(mode)
                self.assertEqual(len(axis.lines), 1)
                expected = (
                    "RCS P50 (dBsm)"
                    if mode in ("frequency", "elevation_sweep")
                    else "RCS (dBsm)"
                )
                self.assertEqual(axis.get_ylabel(), expected)
                self.assertNotIn("[dBsm]", axis.lines[0].get_label())

    def test_phase_overlays_keep_phase_label_and_no_magnitude_units(self):
        axis = self._plot("azimuth_rect", phase=True)
        self.assertEqual(axis.get_ylabel(), "Phase (deg)")
        self.assertTrue(all("dB" not in line.get_label() for line in axis.lines))

    def test_power_ratio_can_overlay_native_rcs_levels(self):
        ratio = _grid(quantity="power_ratio", log_unit="dB")
        axis = self._plot(
            "azimuth_rect", datasets=[("RCS", self.sigma), ("Ratio", ratio)]
        )
        self.assertEqual(axis.get_ylabel(), "Mixed dB")
        np.testing.assert_allclose(
            axis.lines[0].get_ydata(), axis.lines[1].get_ydata()
        )
        self.assertIn("dBsm", axis.lines[0].get_label())
        self.assertIn("dB", axis.lines[1].get_label())

    def test_linear_and_delta_map_still_reject_unlike_quantities(self):
        with self.assertRaisesRegex(ValueError, "mixed physical quantities"):
            self._plot("azimuth_rect", scale="linear")
        with self.assertRaisesRegex(ValueError, "mixed physical quantities"):
            self._plot("delta_map")


if __name__ == "__main__":
    unittest.main()
