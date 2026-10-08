"""PBP keeps independent dataset envelopes and advances the normal plot colors."""

import unittest

import numpy as np
from matplotlib import rcParams
from matplotlib.colors import to_rgba

from GRIM_Backend.plotting.modes import azimuth_rect_mode, azimuth_polar_mode, frequency_mode
from test_plot_renderer_correctness import _RendererHarness
from test_qol_overlays import grid


BAND_MODES = (azimuth_rect_mode, azimuth_polar_mode, frequency_mode)


class PbpDatasetBandTests(unittest.TestCase):
    def harness(self, *, count=2, polarizations=("HH",), same_name=False):
        datasets = []
        for index in range(count):
            dataset = grid(frequencies=(9, 10, 11), elevations=(0, 1), scale=1000 ** index)
            dataset.rcs_power *= np.asarray([1, 4, 16])[None, None, :, None]
            dataset.rcs_power *= np.asarray([1, 2])[None, :, None, None]
            datasets.append(("Same name" if same_name else f"Run {index + 1}", dataset))
        owner = _RendererHarness(datasets, selections={
            "azimuth": datasets[0][1].azimuths, "elevation": [0, 1],
            "frequency": [9, 10, 11], "polarization": polarizations,
        })
        owner.btn_pbp.checked = True
        self.addCleanup(owner.plot_figure.clear)
        return owner

    def assert_colors(self, owner, count, *, offset=0):
        fills = list(owner.plot_ax.collections)
        colors = rcParams['axes.prop_cycle'].by_key()['color']
        self.assertEqual(len(fills), count)
        for index, fill in enumerate(fills):
            color = colors[(index + offset) % len(colors)]
            np.testing.assert_allclose(fill.get_facecolor()[0], to_rgba(color, 0.35))
            edges = owner._plot_item_artists(fill._grim_dataset_key)
            self.assertEqual(len(edges), 3)
            for line in (artist for artist in edges if hasattr(artist, 'get_ydata')):
                np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba(color))

    def test_default_blue_and_dataset_bounds_match_selected_cuts_in_all_modes(self):
        for mode in BAND_MODES:
            for count in (1, 2):
                with self.subTest(mode=mode.__name__, count=count):
                    owner = self.harness(count=count)
                    mode.render(owner)
                    self.assert_colors(owner, count)
                    for index, (_name, dataset) in enumerate(owner._named_datasets):
                        identity = (owner._dataset_plot_key(dataset), "HH")
                        lower, upper = [line for line in owner.plot_ax.lines
                                        if line._grim_pbp_identity == identity]
                        if mode is frequency_mode:
                            # Frequency envelopes bound each elevation's P50 over azimuth.
                            samples = 10 * np.log10(np.median(dataset.rcs_power[..., 0], axis=0))
                            expected_lower, expected_upper = samples.min(axis=0), samples.max(axis=0)
                        else:
                            samples = 10 * np.log10(dataset.rcs_power[..., 0])
                            expected_lower, expected_upper = samples.min(axis=(1, 2)), samples.max(axis=(1, 2))
                        np.testing.assert_allclose(lower.get_ydata(), expected_lower)
                        np.testing.assert_allclose(upper.get_ydata(), expected_upper)
                        label = owner.plot_ax.collections[index].get_label()
                        self.assertIn(f"Run {index + 1} | Pol HH", label)

    def test_duplicate_names_and_multiple_polarizations_remain_separate(self):
        for mode in BAND_MODES:
            with self.subTest(mode=mode.__name__):
                owner = self.harness(same_name=True, polarizations=("HH", "VV"))
                mode.render(owner)
                self.assert_colors(owner, 4)
                self.assertEqual(len({fill._grim_dataset_key for fill in owner.plot_ax.collections}), 4)
                self.assertEqual(len({fill._grim_pbp_identity for fill in owner.plot_ax.collections}), 4)
                self.assertEqual(len(owner.plot_ax.get_legend_handles_labels()[1]), 4)

    def test_hold_preserves_old_data_and_colors_and_reset_restarts_cycle(self):
        for mode in BAND_MODES:
            with self.subTest(mode=mode.__name__):
                owner = self.harness()
                mode.render(owner)
                old_fills = list(owner.plot_ax.collections)
                old_edges = [(line, np.asarray(line.get_ydata()).copy()) for line in owner.plot_ax.lines]
                owner.btn_hold.checked = True
                owner._selections[owner.list_freq] = [9, 10]
                owner._selections[owner.list_elev] = [0]
                owner._named_datasets = owner._named_datasets[:1]
                for count in (3, 4):
                    mode.render(owner)
                    self.assert_colors(owner, count)
                    self.assertEqual(list(owner.plot_ax.collections)[:2], old_fills)
                    for line, original in old_edges:
                        self.assertIn(line, owner.plot_ax.lines)
                        np.testing.assert_array_equal(line.get_ydata(), original)
                # Explicitly clearing while Hold remains on resets Matplotlib's cycle.
                owner.plot_ax.clear()
                mode.render(owner)
                self.assert_colors(owner, 1)
                owner.btn_hold.checked = False
                mode.render(owner)
                self.assert_colors(owner, 1)

    def test_bands_and_ordinary_lines_share_one_color_cycle(self):
        for bands_first in (True, False):
            with self.subTest(bands_first=bands_first):
                owner = self.harness(count=1)
                owner.btn_pbp.checked = bands_first
                if not bands_first:
                    owner._selections[owner.list_freq] = [9]
                    owner._selections[owner.list_elev] = [0]
                azimuth_rect_mode.render(owner)
                owner.btn_hold.checked = True
                owner.btn_pbp.checked = not bands_first
                owner._selections[owner.list_freq] = [9, 10, 11] if not bands_first else [9]
                owner._selections[owner.list_elev] = [0]
                azimuth_rect_mode.render(owner)
                self.assert_colors(owner, 1, offset=0 if bands_first else 1)
                curve = next(line for line in owner.plot_ax.lines if not hasattr(line, '_grim_pbp_identity'))
                expected = rcParams['axes.prop_cycle'].by_key()['color'][1 if bands_first else 0]
                self.assertEqual(curve.get_color(), expected)

    def test_heatmap_choice_remains_available_for_one_band_only(self):
        owner = self.harness(count=1)
        owner.pbp_fill_mode = 'heatmap_rcs'
        azimuth_rect_mode.render(owner)
        self.assertEqual(type(owner.plot_ax.collections[0]).__name__, 'QuadMesh')
        first = owner.plot_ax.collections[0]
        owner.btn_hold.checked = True
        azimuth_rect_mode.render(owner)
        self.assertIs(owner.plot_ax.collections[0], first)
        np.testing.assert_allclose(owner.plot_ax.collections[1].get_facecolor()[0], to_rgba('#ff7f0e', .35))


if __name__ == '__main__':
    unittest.main()
