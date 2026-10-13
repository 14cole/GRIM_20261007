"""Physical display regressions for selected runs, PbP populations, and phase."""
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.plotting.modes import (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode, elevation_sweep_mode,
    compare_mode, common,
)
from test_plot_renderer_correctness import _RendererHarness


def grid(power, *, phase=None):
    power = np.asarray(power, dtype=float)
    return RcsGrid(np.arange(power.shape[0], dtype=float), np.arange(power.shape[1], dtype=float),
                   9. + np.arange(power.shape[2], dtype=float), ['HH'], rcs_power=power,
                   rcs_phase=np.zeros_like(power) if phase is None else np.asarray(phase, dtype=float))


class PlotPhysicsFixTests(unittest.TestCase):
    def owner(self, dataset, *, phase=False, selections=None):
        values = dict(azimuth=dataset.azimuths, elevation=dataset.elevations, frequency=dataset.frequencies)
        values.update(selections or {})
        owner = _RendererHarness([('Run', dataset)], selections=values, phase=phase)
        self.addCleanup(owner.plot_figure.clear)
        self.addCleanup(owner._release_native_plot_samples)
        return owner

    def test_selected_runs_break_in_all_line_modes_and_keep_native_probe_indices(self):
        for axis, mode in [('azimuth', azimuth_rect_mode), ('azimuth', azimuth_polar_mode),
                           ('frequency', frequency_mode), ('elevation', elevation_sweep_mode)]:
            with self.subTest(mode=mode.__name__):
                shape = [1, 1, 1, 1]
                dimension = {'azimuth': 0, 'elevation': 1, 'frequency': 2}[axis]
                shape[dimension] = 11
                dataset = grid((np.arange(11) + 1.).reshape(shape))
                attribute = {'azimuth': 'azimuths', 'elevation': 'elevations', 'frequency': 'frequencies'}[axis]
                selected = getattr(dataset, attribute)[[0, 1, 9, 10]]
                owner = self.owner(dataset, selections={axis: selected})
                mode.render(owner)
                line, = owner.plot_ax.lines
                self.assertEqual(np.flatnonzero(~np.isfinite(line.get_ydata())).tolist(), [2])
                native_x, native_y = owner._marker_arrays(line)
                expected_x = np.deg2rad(selected) if mode is azimuth_polar_mode else selected
                np.testing.assert_allclose(native_x, expected_x)
                np.testing.assert_allclose(native_y, 10. * np.log10([1., 2., 10., 11.]))
                # Repeating the same held selection replaces the trace.
                owner.btn_hold.checked = True
                mode.render(owner)
                self.assertEqual(len(owner.plot_ax.lines), 1)

    def test_reference_selected_runs_remain_separate_on_denser_overlay(self):
        reference = grid(np.ones((11, 1, 1, 1)))
        dataset = grid(np.ones((21, 1, 1, 1)))
        dataset.azimuths = np.linspace(0., 10., 21)
        selected = [0., 1., 9., 10.]
        indices = common.native_axis_selection(reference, dataset, 'azimuth', selected)
        breaks = common.selection_breaks(reference, dataset, 'azimuth', indices, selected)
        np.testing.assert_array_equal(breaks, [3])
        np.testing.assert_allclose(dataset.azimuths[indices], [0., .5, 1., 9., 9.5, 10.])

    def test_singleton_selection_runs_stay_visible_in_gui_and_script(self):
        for axis, mode in [('azimuth', azimuth_rect_mode), ('azimuth', azimuth_polar_mode),
                           ('frequency', frequency_mode), ('elevation', elevation_sweep_mode)]:
            for selected_indices in ([0, 2, 4], [0, 2, 3]):
                with self.subTest(mode=mode.__name__, selected=selected_indices):
                    shape = [1, 1, 1, 1]
                    shape[{'azimuth': 0, 'elevation': 1, 'frequency': 2}[axis]] = 5
                    dataset = grid(np.arange(1., 6.).reshape(shape))
                    attribute = {'azimuth': 'azimuths', 'elevation': 'elevations', 'frequency': 'frequencies'}[axis]
                    selected = getattr(dataset, attribute)[selected_indices]
                    owner = self.owner(dataset, selections={axis: selected})
                    mode.render(owner)
                    line, = owner.plot_ax.lines
                    expected = [0, 2, 4] if selected_indices == [0, 2, 4] else [0]
                    self.assertEqual(line.get_markevery(), expected)
                    self.assertEqual(line.get_marker(), '.')
                    key = line._grim_dataset_key
                    owner._dataset_plot_styles = {key: {'kind': 'scatter'}}
                    owner._apply_dataset_line_style(line)
                    self.assertIsNone(line.get_markevery())
                    owner._dataset_plot_styles = {}
                    owner._apply_dataset_line_style(line)
                    self.assertEqual(line.get_markevery(), expected)
                    kwargs = {'azimuths': dataset.azimuths, 'elevations': dataset.elevations,
                              'frequencies': dataset.frequencies}
                    kwargs[attribute] = selected
                    figure = plot_datasets([('Run', dataset)], mode=mode.__name__.rsplit('.', 1)[-1].removesuffix('_mode'),
                                           polarization='HH', **kwargs)
                    self.addCleanup(figure.clear)
                    replay, = figure.axes[0].lines
                    np.testing.assert_array_equal(replay.get_xdata(), line.get_xdata())
                    np.testing.assert_array_equal(replay.get_ydata(), line.get_ydata())
                    self.assertEqual(replay.get_markevery(), expected)

    def test_line_and_envelope_decimation_never_connect_across_hidden_missing_runs(self):
        for count, budget in [(5, 4), (120001, common.MAX_LINE_POINTS), (10001, 37)]:
            x = np.arange(count, dtype=float)
            y = x.copy()
            if count == 5:
                y[:] = [1., np.nan, 2., np.nan, 3.]
            elif count == 120001:
                y[[2, 10]] = np.nan
                y[5], y[15] = -1., 1000000.
            else:
                y[::7] = np.nan
            for envelope in (False, True):
                with self.subTest(count=count, envelope=envelope):
                    if envelope:
                        dx, dy, high, _, _ = common.decimate_envelope(x, y, y + 2., max_points=budget)
                        self.assertEqual(len(dx), len(high))
                    else:
                        dx, dy, _ = common.decimate_line(x, y, max_points=budget)
                    self.assertLessEqual(len(dx), budget)
                    for index in range(len(dx) - 1):
                        if np.isfinite(dy[index:index + 2]).all():
                            self.assertTrue(np.isfinite(y[int(dx[index]):int(dx[index + 1]) + 1]).all())

    def test_small_decimation_budget_preserves_both_extrema_without_gaps(self):
        for budget in (4, 5, 6, 7):
            y = np.zeros(101)
            y[27], y[74] = -99., 101.
            _, actual, _ = common.decimate_line(np.arange(101), y, budget)
            self.assertEqual((actual.min(), actual.max()), (-99., 101.))

    def test_pbp_percentiles_preserve_zero_rank_and_finite_upper_edge(self):
        power = np.broadcast_to(np.array([0., 1., 100.])[None, None, :, None], (3, 1, 3, 1)).copy()
        dataset = grid(power)
        owner = self.owner(dataset)
        owner.btn_pbp.checked = True
        owner.analysis_controls = SimpleNamespace(pbp_percentiles=lambda: (10., 90.))
        azimuth_rect_mode.render(owner)
        low, high = owner.plot_ax.lines
        np.testing.assert_array_equal(low.get_ydata(), np.full(3, -np.inf))
        np.testing.assert_allclose(high.get_ydata(), 16.)
        np.testing.assert_array_equal(high._grim_pbp_count, [3, 3, 3])
        self.assertIn('zero returns', owner.status.message)

    def test_pbp_extended_real_exact_quantiles_and_missing_samples(self):
        envelope = common.StreamingEnvelope(percentiles=(50., 100.), include_negative_infinity=True)
        self.addCleanup(envelope.close)
        for row in [[-np.inf, np.nan], [0., -np.inf], [20., np.nan]]:
            envelope.update(row)
        low, high, count = envelope.result()
        np.testing.assert_array_equal(low, [0., -np.inf])
        np.testing.assert_array_equal(high, [20., -np.inf])
        np.testing.assert_array_equal(count, [3, 1])

    def test_frequency_pbp_retains_zero_summary_cuts(self):
        dataset = grid(np.broadcast_to(np.array([0., 1., 100.])[None, :, None, None], (1, 3, 2, 1)).copy())
        owner = self.owner(dataset)
        owner.btn_pbp.checked = True
        frequency_mode.render(owner)
        low, high = owner.plot_ax.lines
        self.assertTrue(np.isneginf(low.get_ydata()).all())
        np.testing.assert_allclose(high.get_ydata(), 20.)
        np.testing.assert_array_equal(high._grim_pbp_count, [3, 3])

    def test_all_zero_pbp_is_labeled_and_not_reported_as_missing(self):
        owner = self.owner(grid(np.zeros((3, 1, 2, 1))))
        owner.btn_pbp.checked = True
        azimuth_rect_mode.render(owner)
        self.assertIn('zero power', owner.plot_ax.texts[0].get_text())
        self.assertNotIn('No compatible data', owner.status.message)
        np.testing.assert_array_equal(owner.plot_ax.lines[0]._grim_pbp_count, [2, 2, 2])
        owner._remove_plot_item_artists(owner.plot_ax.lines[0]._grim_dataset_key)
        self.assertFalse(owner.plot_ax.texts)

    def test_coverage_fill_has_count_scale_and_explicit_meaning(self):
        owner = self.owner(grid(np.broadcast_to(np.array([1., 2., 10.])[None, None, :, None], (3, 1, 3, 1)).copy()))
        owner.btn_pbp.checked = True
        owner.pbp_fill_mode = 'heatmap_density'
        azimuth_rect_mode.render(owner)
        np.testing.assert_array_equal(owner.plot_ax.collections[0].get_array(), 3.)
        self.assertEqual(owner.plot_colorbars[0].ax.get_ylabel(), 'Valid cuts (count at each coordinate)')
        self.assertIn('not return-level density', owner.status.message)

    def test_coverage_colorbar_redraw_hold_and_removal_have_no_orphan_axes(self):
        owner = self.owner(grid(np.broadcast_to(np.array([1., 10.])[None, None, :, None], (3, 1, 2, 1)).copy()))
        owner.btn_pbp.checked = True
        owner.pbp_fill_mode = 'heatmap_density'
        for _ in range(3):
            azimuth_rect_mode.render(owner)
            self.assertEqual(len(owner.plot_colorbars), 1)
            self.assertEqual(len(owner.plot_figure.axes), 2)
        key = owner.plot_ax.collections[0]._grim_dataset_key
        owner.btn_hold.checked = True
        for _ in range(3):
            azimuth_rect_mode.render(owner)
            self.assertEqual(len(owner.plot_colorbars), 1)
            self.assertEqual(len(owner.plot_figure.axes), 2)
        owner._remove_plot_item_artists(key)
        self.assertEqual(len(owner.plot_colorbars), 0)
        self.assertEqual(len(owner.plot_figure.axes), 1)

    def test_zero_amplitude_phase_is_missing_rectangular_and_polar(self):
        dataset = grid(np.array([0., 1.])[:, None, None, None], phase=np.full((2, 1, 1, 1), np.pi / 2.))
        for mode in (azimuth_rect_mode, azimuth_polar_mode):
            with self.subTest(mode=mode.__name__):
                owner = self.owner(dataset, phase=True)
                mode.render(owner)
                np.testing.assert_allclose(owner.plot_ax.lines[0].get_ydata(), [np.nan, 90.])
                figure = plot_datasets([('Run', dataset)], mode=mode.__name__.rsplit('.', 1)[-1].removesuffix('_mode'),
                                       azimuths=dataset.azimuths, elevations=dataset.elevations,
                                       frequencies=dataset.frequencies, polarization='HH', phase=True)
                self.addCleanup(figure.clear)
                np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [np.nan, 90.])

    def test_phase_pbp_keeps_narrow_arc_across_wrapping_branch(self):
        phase = np.deg2rad(np.array([[179., -179.], [-179., -177.]]))[:, None, :, None]
        owner = self.owner(grid(np.ones((2, 1, 2, 1)), phase=phase), phase=True)
        owner.btn_pbp.checked = True
        azimuth_rect_mode.render(owner)
        low, high = (line.get_ydata() for line in owner.plot_ax.lines)
        np.testing.assert_allclose(low, [179., 181.])
        np.testing.assert_allclose(high, [181., 183.])
        for path in owner.plot_ax.collections[0].get_paths():
            self.assertGreater(path.vertices[:, 1].min(), 170.)

    def test_phase_band_branch_restarts_after_missing_or_selected_gap(self):
        low, high = common.continuous_phase_envelope([179., np.nan, -179.], [181., np.nan, -177.])
        np.testing.assert_allclose(low, [179., np.nan, -179.])
        np.testing.assert_allclose(high, [181., np.nan, -177.])
        dataset = grid(np.ones((11, 1, 2, 1)))
        owner = self.owner(dataset, selections={'azimuth': [0., 1., 9., 10.]})
        owner.btn_pbp.checked = True
        azimuth_rect_mode.render(owner)
        self.assertEqual(len(owner.plot_ax.collections[0].get_paths()), 2)
        self.assertEqual(len(owner._marker_arrays(owner.plot_ax.lines[0])[0]), 4)

    def test_compare_scale_change_rebuilds_levels_and_label(self):
        left, right = grid(np.full((1, 1, 2, 1), 1000.)), grid(np.full((1, 1, 2, 1), 2.))
        owner = self.owner(left)
        owner._named_datasets = [('A', left), ('B', right)]
        compare_mode.render(owner)
        np.testing.assert_allclose(owner.plot_axes[0].lines[0].get_ydata(), 30.)
        owner.combo_plot_scale.data = 'linear'
        # The harness does not have application export/autoscale widgets.
        with mock.patch.object(owner, '_plot_compare', side_effect=lambda: compare_mode.render(owner)):
            owner._on_plot_scale_changed()
        np.testing.assert_allclose(owner.plot_axes[0].lines[0].get_ydata(), 1000.)
        self.assertNotIn('dB', owner.plot_axes[0].get_ylabel())
