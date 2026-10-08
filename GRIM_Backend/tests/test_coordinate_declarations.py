"""Every dataset loads as azimuth/elevation; coordinate systems never block."""

import json
import os
import tempfile
import unittest

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import load_dataset
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.plotting.modes.common import validate_plot_datasets
from test_ptm import _independent_fixture


class AzimuthElevationOnlyTests(unittest.TestCase):
    def test_matching_pio_ptm_nonzero_cut_overlays_without_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            ptm_path = os.path.join(folder, "same.ptm")
            _independent_fixture(ptm_path, start_aspect=-2.0, aspect_increment=1.0)
            ptm = load_dataset(ptm_path)
            # Encode the same numeric data as a PIO azimuth/elevation cut.
            source = RcsGrid(
                ptm.azimuths, ptm.elevations, ptm.frequencies, ptm.polarizations,
                rcs=ptm.rcs, units={"frequency": "GHz"},
            )
            pio = load_dataset(source.save_pio(os.path.join(folder, "same.pio")))
            self.assertEqual(ptm.elevations.tolist(), [7.5])
            for key in (
                "angular_coordinate_system", "great_circle_coordinate_convention",
                "angular_roll_deg", "angular_tilt_deg",
            ):
                self.assertNotIn(key, ptm.units)
            self.assertNotIn("ptm_cut_type", ptm.extra)
            # PTM header roll/tilt remain available as PTM metadata only.
            self.assertEqual(float(ptm.extra["ptm_roll"]), 1.25)
            self.assertEqual(float(ptm.extra["ptm_tilt"]), -2.5)

            validate_plot_datasets(
                [("PIO", pio), ("PTM", ptm)], phase=False, linear=False
            )
            figure = plot_datasets(
                [("PIO", pio), ("PTM", ptm)], mode="azimuth_rect",
                azimuths=pio.azimuths, elevations=pio.elevations,
                frequencies=pio.frequencies[:1], polarization="HH",
            )
            axis = figure.axes[0]
            self.assertEqual(axis.get_xlabel(), "Azimuth (deg)")
            self.assertEqual(len(axis.lines), 2)
            np.testing.assert_allclose(axis.lines[0].get_xdata(), axis.lines[1].get_xdata())
            np.testing.assert_allclose(axis.lines[0].get_ydata(), axis.lines[1].get_ydata(), atol=1e-5)

    def test_stored_great_circle_declarations_are_dropped_on_load(self):
        grid = RcsGrid(
            [0.0, 1.0], [5.0], [10.0], ["VV"],
            rcs_power=np.ones((2, 1, 1, 1)),
            units={
                "frequency": "GHz",
                "angular_coordinate_system": "great_circle",
                "great_circle_coordinate_convention": "grim_gc_v1",
                "angular_roll_deg": 3.0,
                "angular_tilt_deg": 4.0,
            },
            extra={
                "angular_coordinate_system": "gc",
                "angular_coordinate_declaration_json": json.dumps({"x": 1}),
                "ptm_cut_type": "GC",
            },
        )
        for key in (
            "angular_coordinate_system", "great_circle_coordinate_convention",
            "angular_roll_deg", "angular_tilt_deg",
        ):
            self.assertNotIn(key, grid.units)
        for key in (
            "angular_coordinate_system", "angular_coordinate_declaration_json",
            "ptm_cut_type",
        ):
            self.assertNotIn(key, grid.extra)
        np.testing.assert_array_equal(grid.elevations, [5.0])

        conic = RcsGrid(
            [0.0], [0.0], [10.0], ["VV"], rcs_power=np.ones((1, 1, 1, 1)),
            units={"angular_coordinate_system": "conic"},
        )
        self.assertEqual(conic.units["angular_coordinate_system"], "conic")
        with tempfile.TemporaryDirectory() as folder:
            restored = RcsGrid.load(grid.save(os.path.join(folder, "gc.grim")))
        self.assertNotIn("angular_coordinate_system", restored.units)

    def test_previously_mixed_coordinate_datasets_can_share_a_plot(self):
        values = np.ones((3, 1, 1, 1))
        first = RcsGrid(
            [0.0, 1.0, 2.0], [0.0], [10.0], ["VV"], rcs_power=values,
            units={"frequency": "GHz", "angular_coordinate_system": "conic"},
        )
        second = RcsGrid(
            [0.0, 1.0, 2.0], [0.0], [10.0], ["VV"], rcs_power=values,
            units={"frequency": "GHz", "angular_coordinate_system": "great_circle"},
        )
        validate_plot_datasets([("A", first), ("B", second)], phase=False, linear=False)
        # Arithmetic/alignment compatibility no longer compares angular frames.
        first._assert_physical_metadata_compatible(second)


if __name__ == "__main__":
    unittest.main()
