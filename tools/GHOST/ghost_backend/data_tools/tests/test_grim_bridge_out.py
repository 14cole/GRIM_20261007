"""The CEM Tools ``.out`` writer keeps its text while reading each cut once."""

from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from cem_tools.grim_bridge import _rcs_grid_class, export_dataset, load_dataset  # noqa: E402


def _former_out_text(grid, elevation_index, polarization_index):
    """The per-cell loop the writer replaced: legacy attribute reads per sample."""
    lines = ["# frequency_GHz azimuth_deg rcs_dBke phase_deg\n"]
    for iff, frequency in enumerate(grid.frequencies):
        wavelength = 299_792_458.0 / (float(frequency) * 1e9)
        for ia, azimuth in enumerate(grid.azimuths):
            sigma = float(grid.rcs_power[ia, elevation_index, iff, polarization_index])
            dbke = 10.0 * np.log10(sigma * 2.0 * np.pi / wavelength)
            phase = np.degrees(float(grid.rcs_phase[ia, elevation_index, iff, polarization_index]))
            lines.append(f"{float(frequency):.17g} {float(azimuth):.17g} {dbke:.17g} {phase:.17g}\n")
    return "".join(lines)


def _grid(seed, *, raw_field=False):
    """A 2-D width dataset with exact zeros and a missing sample."""
    grid_class = _rcs_grid_class()
    rng = np.random.default_rng(seed)
    azimuths = np.linspace(-30.0, 30.0, 13)
    elevations = [0.0, 5.0]
    frequencies = [9.0, 10.5]
    polarizations = ["HH", "VV"]
    shape = (13, 2, 2, 2)
    field = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 1.0e-2
    field[0, 0, 0, 0] = 0.0
    field[3, 1, 1, 1] = np.nan + 1j * np.nan
    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_linear_quantity": "sigma_2d", "rcs_log_unit": "dBke"}
    return grid_class(
        azimuths, elevations, frequencies, polarizations, rcs=field,
        rcs_storage_domain="raw_field" if raw_field else "physical", units=units,
    )


class GrimBridgeOutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rows_are_identical_to_the_former_per_cell_loop(self):
        for raw_field in (False, True):
            with self.subTest(raw_field=raw_field):
                grid = _grid(3, raw_field=raw_field)
                # The former loop exposes the legacy arrays, so it runs on a
                # separate, identically built dataset.
                reference = _grid(3, raw_field=raw_field)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    outputs = export_dataset(grid, self.root / f"cut{int(raw_field)}", ".out")
                    expected = [
                        _former_out_text(reference, elevation_index, polarization_index)
                        for elevation_index, polarization_index in product(range(2), range(2))
                    ]
                self.assertEqual(len(outputs), 4)
                for path, text in zip(outputs, expected):
                    self.assertEqual(path.read_text(encoding="utf-8"), text)
                self.assertIn("-inf", expected[0])
                self.assertIn("nan", expected[3])

    def test_export_leaves_the_dataset_in_canonical_storage(self):
        grid = _grid(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            export_dataset(grid, self.root / "cut", ".out")
        self.assertEqual(grid.storage_kind, "complex")
        self.assertFalse(grid._legacy_samples)

    def test_exported_cut_reloads_through_grim(self):
        grid = _grid(9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            outputs = export_dataset(grid, self.root / "cut", ".out")
        self.assertTrue(outputs[0].name.endswith("_EL0.0_HH.out"))
        loaded = load_dataset(outputs[0])
        np.testing.assert_allclose(
            np.asarray(loaded.power)[:, 0, :, 0], np.asarray(grid.power)[:, 0, :, 0], rtol=1e-12,
        )
        self.assertEqual(list(loaded.polarizations), ["HH"])


if __name__ == "__main__":
    unittest.main()
