"""Polarization identity stays stable when Hold selections change."""

import unittest

import numpy as np

from GRIM_Backend.plotting.dataset_style import PBP_BAND_KEY
from GRIM_Backend.plotting.modes import (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode, elevation_sweep_mode,
    cdf_mode, sector_stats_mode,
)
from test_plot_renderer_correctness import _RendererHarness
from test_qol_overlays import grid


LINE_MODES = (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode, elevation_sweep_mode,
    cdf_mode, sector_stats_mode,
)
BAND_MODES = (azimuth_rect_mode, azimuth_polar_mode, frequency_mode)


class HoldPolarizationTests(unittest.TestCase):
    def harness(self, polarizations, *, bands=False, available_polarizations=("HH", "VV")):
        datasets = [("A", grid(polarizations=available_polarizations))]
        if bands:
            datasets.append(("B", grid(scale=2)))
        owner = _RendererHarness(datasets, selections={
            "azimuth": datasets[0][1].azimuths,
            "elevation": [0], "frequency": [9], "polarization": polarizations,
        })
        owner.btn_pbp.checked = bands
        self.addCleanup(owner.plot_figure.clear)
        return owner

    @staticmethod
    def band_keys(owner):
        return {
            artist._grim_dataset_key: artist._grim_pbp_identity
            for artist in (*owner.plot_ax.lines, *owner.plot_ax.collections)
            if hasattr(artist, "_grim_pbp_identity")
        }

    def assert_band_count(self, owner, count):
        self.assertEqual(len(self.band_keys(owner)), count)
        self.assertEqual(len(owner.plot_ax.get_legend_handles_labels()[1]), count)
        self.assertEqual(len(owner.plot_ax.lines), 2 * count)
        self.assertEqual(len(owner.plot_ax.collections), count)

    def test_line_styles_survive_multi_to_single_and_back_under_hold(self):
        for mode in LINE_MODES:
            with self.subTest(mode=mode.__name__):
                owner = self.harness(("HH", "VV"))
                mode.render(owner)
                owner.btn_hold.checked = True
                for selection in (("VV",), ("HH",), ("HH", "VV")):
                    owner._selections[owner.list_pol] = selection
                    mode.render(owner)
                    self.assertEqual(len(owner.plot_ax.lines), 2)
                    for line in owner.plot_ax.lines:
                        self.assertEqual(line.get_linestyle(), "-")

    def test_single_polarization_additions_stay_solid(self):
        for first, second in (("HH", "VV"), ("VV", "HH")):
            owner = self.harness((first,))
            azimuth_rect_mode.render(owner)
            owner.btn_hold.checked = True
            owner._selections[owner.list_pol] = (second,)
            azimuth_rect_mode.render(owner)
            self.assertEqual(len(owner.plot_ax.lines), 2)
            self.assertEqual({line.get_linestyle() for line in owner.plot_ax.lines}, {"-"})

    def test_all_polarizations_default_to_solid_in_each_line_mode(self):
        polarizations = ("HH", "VV", "HV", "VH")
        for mode in LINE_MODES:
            with self.subTest(mode=mode.__name__):
                owner = self.harness(polarizations, available_polarizations=polarizations)
                mode.render(owner)
                self.assertEqual(len(owner.plot_ax.lines), 4)
                self.assertEqual({line.get_linestyle() for line in owner.plot_ax.lines}, {"-"})

    def test_explicit_dataset_style_overrides_solid_default(self):
        owner = self.harness(("HH", "VV"))
        key = owner._dataset_plot_key(owner.active_dataset)
        owner._dataset_plot_styles = {key: {"linestyle": ":", "color": "#ff0000"}}
        azimuth_rect_mode.render(owner)
        owner.btn_hold.checked = True
        owner._selections[owner.list_pol] = ("VV",)
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.lines), 2)
        for line in owner.plot_ax.lines:
            self.assertEqual(line.get_linestyle(), ":")
            self.assertEqual(line.get_color(), "#ff0000")

    def test_held_multi_polarization_bands_append_each_selected_identity(self):
        for mode in BAND_MODES:
            with self.subTest(mode=mode.__name__):
                owner = self.harness(("HH", "VV"), bands=True)
                mode.render(owner)
                initial_keys = self.band_keys(owner)
                initial_lines = list(owner.plot_ax.lines)
                self.assert_band_count(owner, 4)
                for dataset_key in {identity[0] for identity in initial_keys.values()}:
                    edges = {
                        pol: [line for line in initial_lines
                              if line._grim_pbp_identity == (dataset_key, pol)]
                        for pol in ("HH", "VV")
                    }
                    for hh, vv in zip(edges["HH"], edges["VV"]):
                        np.testing.assert_allclose(vv.get_ydata() - hh.get_ydata(), 20.0)
                owner.btn_hold.checked = True
                count = 4
                for selection in (("HH",), ("VV",), ("HH", "VV")):
                    owner._selections[owner.list_pol] = selection
                    mode.render(owner)
                    count += 2 * len(selection)
                    self.assert_band_count(owner, count)
                    self.assertEqual(list(owner.plot_ax.lines)[:8], initial_lines)
                    for key, identity in initial_keys.items():
                        self.assertEqual(self.band_keys(owner)[key], identity)

    def test_individual_held_bands_preserve_first_legacy_key_and_second_band(self):
        for mode in BAND_MODES:
            for first, second in (("HH", "VV"), ("VV", "HH")):
                with self.subTest(mode=mode.__name__, first=first):
                    owner = self.harness((first,), bands=True)
                    mode.render(owner)
                    initial_keys = self.band_keys(owner)
                    self.assert_band_count(owner, 2)
                    self.assertEqual(initial_keys[PBP_BAND_KEY][1], first)
                    self.assertTrue(owner.plot_ax.get_legend_handles_labels()[1][0].startswith(
                        f"PBP A | Pol {first},"
                    ))
                    owner.btn_hold.checked = True
                    owner._selections[owner.list_pol] = (second,)
                    mode.render(owner)
                    self.assert_band_count(owner, 4)
                    keys = self.band_keys(owner)
                    self.assertEqual(keys[PBP_BAND_KEY][1], first)
                    self.assertEqual([identity[1] for key, identity in keys.items()
                                      if key not in initial_keys], [second, second])
                    owner._selections[owner.list_pol] = ("HH", "VV")
                    mode.render(owner)
                    self.assert_band_count(owner, 8)
                    for key, identity in keys.items():
                        self.assertEqual(self.band_keys(owner)[key], identity)


if __name__ == "__main__":
    unittest.main()
