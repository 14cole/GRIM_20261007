"""GUI overlays preserve native dB levels and label visible, including held, units."""

import unittest

import numpy as np

from GRIM_Backend.plotting.modes import (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode, elevation_sweep_mode,
    cdf_mode, sector_stats_mode, waterfall_mode, compare_mode, common,
)
from test_plot_renderer_correctness import _RendererHarness, _grid


class MixedDbPlotTests(unittest.TestCase):
    def setUp(self):
        self.rcs = _grid()
        self.width = _grid(quantity="sigma_2d", log_unit="dBke")
        for dataset in (self.rcs, self.width):
            dataset.rcs_power[:, 0, :, 0] = np.arange(1.0, 6.0)[:, None]
        self.datasets = [("Measured", self.rcs), ("Analysis", self.width)]

    def harness(self, datasets=None, *, all_frequencies=False):
        owner = _RendererHarness(
            self.datasets if datasets is None else datasets,
            selections={
                "azimuth": self.rcs.azimuths,
                "elevation": self.rcs.elevations,
                "frequency": self.rcs.frequencies if all_frequencies else [9.0],
            },
        )
        self.addCleanup(owner.plot_figure.clear)
        return owner

    def test_cut_overlays_keep_native_values_and_identify_units(self):
        expected_rcs = 10.0 * np.log10(np.arange(1.0, 6.0))
        expected_width = expected_rcs + 10.0 * np.log10(2.0 * np.pi * 9.0e9 / 299_792_458.0)
        for mode in (azimuth_rect_mode, azimuth_polar_mode):
            with self.subTest(mode=mode.__name__):
                owner = self.harness()
                mode.render(owner)
                self.assertIn("updated", owner.status.message)
                self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
                self.assertEqual(len(owner.plot_ax.lines), 2)
                for line, expected, unit in zip(
                    owner.plot_ax.lines, (expected_rcs, expected_width), ("dBsm", "dBke")
                ):
                    np.testing.assert_allclose(line.get_ydata(), expected)
                    self.assertIn(f"[{unit}]", line.get_label())
                self.assertEqual(
                    [item.get_text() for item in owner.plot_ax.get_legend().get_texts()],
                    [line.get_label() for line in owner.plot_ax.lines],
                )

    def test_sweeps_preserve_frequency_dependent_dbke(self):
        for mode in (frequency_mode, elevation_sweep_mode):
            with self.subTest(mode=mode.__name__):
                owner = self.harness(all_frequencies=mode is frequency_mode)
                mode.render(owner)
                self.assertIn("updated", owner.status.message)
                self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
                self.assertEqual(len(owner.plot_ax.lines), 2)
                frequencies = self.rcs.frequencies if mode is frequency_mode else np.array([9.0])
                expected_rcs = np.full(frequencies.size, 10.0 * np.log10(3.0))
                expected_width = expected_rcs + 10.0 * np.log10(
                    2.0 * np.pi * frequencies * 1.0e9 / 299_792_458.0
                )
                np.testing.assert_allclose(owner.plot_ax.lines[0].get_ydata(), expected_rcs)
                np.testing.assert_allclose(owner.plot_ax.lines[1].get_ydata(), expected_width)

    def test_hold_in_both_orders_and_removal_relabel_visible_units(self):
        for mode, value_axis in ((azimuth_rect_mode, "y"), (cdf_mode, "x")):
            for first, second in (self.datasets, list(reversed(self.datasets))):
                with self.subTest(mode=mode.__name__, first=first[0]):
                    owner = self.harness([first])
                    mode.render(owner)
                    original_label = getattr(owner.plot_ax, f"get_{value_axis}label")()
                    original_curve_label = owner.plot_ax.lines[0].get_label()
                    owner.btn_hold.checked = True
                    owner._named_datasets = [second]
                    owner.active_dataset = second[1]
                    mode.render(owner)
                    self.assertIn("updated", owner.status.message)
                    self.assertEqual(getattr(owner.plot_ax, f"get_{value_axis}label")(), "Mixed dB")
                    self.assertEqual(len(owner.plot_ax.lines), 2)
                    for line, (_, dataset) in zip(owner.plot_ax.lines, (first, second)):
                        self.assertIn(f"[{dataset.default_log_unit()}]", line.get_label())
                    # Repeating a held plot replaces the curve without growing its label.
                    mixed_labels = [line.get_label() for line in owner.plot_ax.lines]
                    mode.render(owner)
                    self.assertEqual([line.get_label() for line in owner.plot_ax.lines], mixed_labels)
                    self.assertTrue(owner._remove_plot_dataset(owner._dataset_plot_key(second[1])))
                    self.assertEqual(getattr(owner.plot_ax, f"get_{value_axis}label")(), original_label)
                    self.assertEqual(owner.plot_ax.lines[0].get_label(), original_curve_label)

    def test_same_units_and_hidden_legend_keep_correct_axis_labels(self):
        owner = self.harness([self.datasets[0], ("Second", _grid())])
        azimuth_rect_mode.render(owner)
        self.assertEqual(owner.plot_ax.get_ylabel(), "RCS (dBsm)")
        self.assertTrue(all("dBsm" not in line.get_label() for line in owner.plot_ax.lines))
        owner.chk_plot_legend.checked = False
        owner._named_datasets = self.datasets
        azimuth_rect_mode.render(owner)
        self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")

    def test_pbp_mixed_band_and_held_curve_use_displayed_native_units(self):
        owner = self.harness()
        owner.btn_pbp.checked = True
        azimuth_rect_mode.render(owner)
        self.assertIn("updated", owner.status.message)
        self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
        band_label = owner.plot_ax.get_legend_handles_labels()[1][0]
        self.assertIn("dBsm", band_label)
        self.assertIn("dBke", band_label)
        owner.btn_hold.checked = True
        owner.btn_pbp.checked = False
        owner._named_datasets = [self.datasets[0]]
        azimuth_rect_mode.render(owner)
        self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
        self.assertIn("[dBsm]", owner.plot_ax.lines[-1].get_label())

    def test_cdf_uses_mixed_x_axis_and_native_median_units(self):
        owner = self.harness()
        cdf_mode.render(owner)
        self.assertEqual(owner.plot_ax.get_xlabel(), "Mixed dB")
        self.assertEqual(owner.plot_ax.get_ylabel(), "Samples at or below level (%)")
        for line, unit, offset in zip(owner.plot_ax.lines, ("dBsm", "dBke"), (
            0.0, 10.0 * np.log10(2.0 * np.pi * 9.0e9 / 299_792_458.0)
        )):
            np.testing.assert_allclose(line.get_xdata(), 10.0 * np.log10(np.arange(1.0, 6.0)) + offset)
            self.assertIn(f"median {10.0 * np.log10(3.0) + offset:.4g} {unit}", line.get_label())

    def test_sector_table_keeps_each_rows_native_unit(self):
        owner = self.harness()
        sector_stats_mode.render(owner)
        self.assertIn("updated", owner.status.message)
        self.assertEqual(owner.plot_ax.get_ylabel(), "Mixed dB")
        table = owner.plot_figure._grim_sector_table
        exported = sector_stats_mode.table_text(table).splitlines()
        self.assertEqual(exported[0].split("\t")[-1], "Unit")
        self.assertNotIn("Mixed dB", exported[0])
        self.assertEqual({row.split("\t")[-1] for row in exported[1:]}, {"dBsm", "dBke"})
        owner._named_datasets = [self.datasets[0]]
        sector_stats_mode.render(owner)
        single_header = sector_stats_mode.table_text(owner.plot_figure._grim_sector_table).splitlines()[0]
        self.assertIn("Mean (dBsm)", single_header)
        self.assertNotIn("\tUnit", single_header)

    def test_waterfall_shared_and_individual_colorbars_identify_units(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                owner = self.harness(all_frequencies=True)
                owner.chk_colorbar.checked = True
                owner.chk_colorbar_shared.checked = shared
                waterfall_mode.render(owner)
                self.assertIn("updated", owner.status.message)
                self.assertIn("dBsm", owner.plot_axes[0].get_title())
                self.assertIn("dBke", owner.plot_axes[1].get_title())
                self.assertEqual(
                    [bar.ax.get_ylabel() for bar in owner.plot_colorbars],
                    ["Mixed dB"] if shared else ["RCS (dBsm)", "Scattering Width (dBke)"],
                )

    def test_linear_and_physical_comparisons_remain_strict(self):
        owner = self.harness()
        owner.combo_plot_scale.data = "linear"
        azimuth_rect_mode.render(owner)
        self.assertIn("mixed physical quantities", owner.status.message)
        self.assertEqual(len(owner.plot_ax.lines), 0)
        owner.combo_plot_scale.data = "dbsm"
        compare_mode.render(owner)
        self.assertIn("mixed physical quantities", owner.status.message)
        self.assertEqual(len(owner.plot_ax.lines), 0)
        with self.assertRaisesRegex(ValueError, "mixed physical quantities"):
            common.validate_plot_datasets(self.datasets, phase=False, linear=False)
        common.validate_plot_datasets(self.datasets, phase=False, linear=False, allow_mixed_db=True)


if __name__ == "__main__":
    unittest.main()
