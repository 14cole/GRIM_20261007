"""Numerical parity, permanent canvas ownership, and real Qt worker publication."""
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting import requests, sweep_analysis
from GRIM_Backend.plotting.modes import common
from GRIM_Backend.plotting.image_analysis import heatmap_display_plane
from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest
from GRIM_Backend.execution.plot_jobs import PlotJobRunner
import test_gui_shell as shell


def sample():
    values = np.arange(1., 49.).reshape(4, 2, 6, 1)
    phase = np.linspace(-2., 2., values.size).reshape(values.shape)
    return RcsGrid([0., 1., 2., 3.], [0., 5.], [8., 9., 10., 11., 12., 13.], ['VV'],
                   rcs_power=values, rcs_phase=phase)


@pytest.mark.parametrize('mode', ['individual', 'mean', 'median', 'percentile', 'max'])
@pytest.mark.parametrize('phase', [False, True])
@pytest.mark.parametrize('axis', ['frequency', 'elevation'])
def test_tiled_sweep_retains_the_exact_native_population(mode, phase, axis):
    grid = sample()
    selection = tuple(np.arange(n) for n in grid.power.shape)
    kwargs = dict(sweep_axis=axis, config={'mode': mode, 'percentile': 73.}, phase=phase)
    before = list(sweep_analysis.iter_series('A', grid, grid, selection, **kwargs))
    with mock.patch.object(common, 'REDUCTION_BLOCK_CELLS', 4):
        after = list(sweep_analysis.iter_series('A', grid, grid, selection, **kwargs))
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], equal_nan=True)
        assert actual[2:] == expected[2:]


@pytest.mark.parametrize('axes', [('azimuth', 'frequency'), ('frequency', 'azimuth'), ('azimuth', 'elevation')])
@pytest.mark.parametrize('phase,scale', [(False, 'linear'), (False, 'dbsm'), (True, 'dbsm')])
def test_waterfall_tiles_match_full_native_calculation(axes, phase, scale):
    grid = sample()
    grid.units.update(rcs_linear_quantity='sigma_2d', rcs_log_unit='dBke')
    plan = PlotPlan('A', grid, tuple(np.arange(n) for n in grid.power.shape))
    request = PlotRequest('waterfall', grid, (plan,), phase=phase, scale=scale, options={'axes': axes})
    expected = heatmap_display_plane(grid, plan.selection, *axes, phase=phase, linear=scale == 'linear')
    with mock.patch.object(common, 'REDUCTION_BLOCK_CELLS', 4):
        actual = requests.prepare_image_plane(request, plan)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_captured_request_owns_selector_and_option_values():
    grid = sample()
    indices = [np.arange(n) for n in grid.power.shape]
    options = {'mode': 'median', 'percentile': 90.}
    plan = PlotPlan('A', grid, tuple(indices))
    request = PlotRequest('frequency', grid, (plan,), options=options)
    indices[0][:] = 0
    options['mode'] = 'max'
    np.testing.assert_array_equal(plan.selection[0], np.arange(4))
    assert request.options['mode'] == 'median'
    with pytest.raises(requests.PlotCancelled):
        requests.prepare_sweep(request, lambda: True)


class PlotWorkerIntegrationTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp
    tearDown = shell.UnifiedGuiShellTest.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def load_sample(self):
        grid = sample()
        self.window._add_dataset_row(grid, 'Measured', 'Loaded', 'measured.grim')
        self.window.table.selectRow(0)
        self.window.btn_auto_plot.setChecked(False)
        for listing in (self.window.list_az, self.window.list_elev, self.window.list_freq, self.window.list_pol):
            listing.selectAll()
        self.window._plot_async_threshold = 0
        return grid

    def wait_until(self, predicate, timeout=5.):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            QTest.qWait(10)
        self.assertTrue(predicate(), 'Timed out waiting for plot worker')

    def test_each_context_owns_state_without_switch_copying(self):
        plotting, isar = (self.window._plot_contexts[key] for key in ('plotting', 'isar'))
        self.window.last_plot_mode = 'frequency'
        self.window._plot_render_generation = 11
        self.assertEqual(plotting.last_plot_mode, 'frequency')
        self.assertNotIn('last_plot_mode', self.window.__dict__)
        self.window._activate_plot_tab('isar')
        self.window.last_plot_mode = 'isar_image'
        self.window._plot_render_generation = 23
        self.assertEqual(isar.last_plot_mode, 'isar_image')
        plotting.last_plot_mode = 'waterfall'
        self.window._activate_plot_tab('plotting')
        self.assertEqual(self.window.last_plot_mode, 'waterfall')
        self.assertEqual(self.window._plot_render_generation, 11)
        self.assertIs(self.window.plot_canvas, plotting.plot_canvas)
        self.assertEqual(isar.session.state['_plot_render_generation'], 23)

    def test_numerics_run_off_thread_and_artists_publish_on_gui_thread(self):
        grid = self.load_sample()
        original_prepare = requests.prepare_sweep
        threads = []
        original_draw = self.window._plot_bounded_line
        drawn = []

        def prepare(request, cancel=None):
            threads.append(QThread.currentThread())
            return original_prepare(request, cancel)

        def draw(*args, **kwargs):
            drawn.append(QThread.currentThread())
            return original_draw(*args, **kwargs)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=prepare), mock.patch.object(self.window, '_plot_bounded_line', side_effect=draw):
            self.window._plot_frequency()
            self.assertTrue(self.window._ordinary_plot_pending)
            self.window._on_explicit_plot_clicked('frequency')
            self.wait_until(lambda: not self.window._ordinary_plot_pending)
        self.assertTrue(threads and all(thread != self.app.thread() for thread in threads))
        self.assertTrue(drawn and all(thread == self.app.thread() for thread in drawn))
        self.assertIn('updated', self.window.status.currentMessage())
        self.assertEqual(self.window.last_python_plot_spec[3], 'frequency')
        self.assertEqual(len(self.window.plot_ax.lines), len(grid.elevations))

    def test_clear_cancels_a_late_result_without_repopulating_canvas(self):
        self.load_sample()
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_sweep

        def delayed(request, cancel=None):
            entered.set()
            release.wait(5.)
            return original(request, cancel)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=delayed):
            try:
                self.window._plot_frequency()
                self.wait_until(entered.is_set)
                runner = self.window._plot_contexts['plotting'].session.runner
                self.window._clear_plot()
            finally:
                release.set()
            self.wait_until(lambda: not runner.busy)
        self.assertEqual(len(self.window.plot_ax.lines), 0)
        self.assertFalse(self.window._ordinary_plot_pending)

    def test_switching_tabs_does_not_publish_into_the_new_canvas(self):
        self.load_sample()
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_sweep

        def delayed(request, cancel=None):
            entered.set(); release.wait(5.)
            return original(request, cancel)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=delayed):
            try:
                self.window._plot_frequency()
                self.wait_until(entered.is_set)
                runner = self.window._plot_contexts['plotting'].session.runner
                self.window._activate_plot_tab('isar')
                figure = self.window.plot_figure
            finally:
                release.set()
            self.wait_until(lambda: not runner.busy)
        self.assertIs(self.window.plot_figure, figure)
        self.assertEqual(len(self.window.plot_ax.lines), 0)

    def test_pending_request_is_replaced_by_latest_parameters(self):
        grid = self.load_sample()
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_sweep
        seen = []

        def delayed(request, cancel=None):
            seen.append(request.options['mode'])
            if len(seen) == 1:
                entered.set(); release.wait(5.)
            return original(request, cancel)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=delayed):
            try:
                self.window._plot_frequency()
                self.wait_until(entered.is_set)
                for mode in ('mean', 'max'):
                    self.window.analysis_controls.restore_analysis_options({'sweep': {'mode': mode, 'percentile': 90.}})
                    self.window._plot_frequency()
            finally:
                release.set()
            self.wait_until(lambda: not self.window._ordinary_plot_pending)
        self.assertEqual(seen, ['median', 'max'])
        expected = grid.linear_to_default_db(grid.power[:, 0, :, 0].max(axis=0), frequency_value=grid.frequencies)
        np.testing.assert_allclose(self.window.plot_ax.lines[0].get_ydata(), expected)

    def test_all_expensive_modes_publish_from_their_own_request(self):
        self.load_sample()
        for mode in ('elevation_sweep', 'cdf', 'waterfall'):
            self.window._render_plot_mode(mode)
            self.wait_until(lambda: not self.window._ordinary_plot_pending)
            self.assertIn('updated', self.window.status.currentMessage())
            self.assertEqual(self.window.last_python_plot_spec[3], mode)

    def test_accepted_close_then_reopen_can_prepare_a_new_plot(self):
        self.load_sample()
        self.window._plot_frequency()
        self.wait_until(lambda: not self.window._ordinary_plot_pending)
        context = self.window._plot_contexts['plotting']
        previous_runner = context.session.runner
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(context.session.disposed)
        self.window.show()
        self.window._plot_frequency()
        self.wait_until(lambda: not self.window._ordinary_plot_pending)
        self.assertFalse(context.session.disposed)
        self.assertIsNot(context.session.runner, previous_runner)
        self.assertIn('updated', self.window.status.currentMessage())
        self.assertGreater(len(self.window.plot_ax.lines), 0)

    def test_job_publication_can_request_new_work_and_report_errors(self):
        runner = PlotJobRunner(self.window)
        received, errors = [], []
        runner.publication_failed.connect(errors.append)

        def first(value, error):
            received.append(value)
            runner.submit(lambda cancel: 'second', lambda value, error: received.append(value))
            raise ValueError('publication fixture')

        runner.submit(lambda cancel: 'first', first)
        self.wait_until(lambda: received == ['first', 'second'] and not runner.busy)
        self.assertEqual(str(errors[0]), 'publication fixture')
        runner.close()

    def test_recipe_matches_live_style_changed_during_waterfall_preparation(self):
        self.load_sample()
        self.window._plot_async_threshold = 10**9
        self.window._plot_waterfall()
        self.window._plot_async_threshold = 0
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_image_plane

        def delayed(request, plan, remaining_index=None, cancel=None):
            entered.set(); release.wait(5.)
            return original(request, plan, remaining_index, cancel)

        with mock.patch.object(requests, 'prepare_image_plane', side_effect=delayed):
            try:
                self.window._plot_waterfall()
                self.wait_until(entered.is_set)
                index = self.window.combo_colormap.findData('viridis')
                if index < 0:
                    index = self.window.combo_colormap.findText('viridis')
                self.assertGreaterEqual(index, 0)
                self.window.combo_colormap.setCurrentIndex(index)
            finally:
                release.set()
            self.wait_until(lambda: not self.window._ordinary_plot_pending)
        self.assertEqual(self.window.plot_ax.collections[0].cmap.name, 'viridis')
        self.assertEqual(self.window.last_python_plot_spec[4]['colormap'], 'viridis')

    def test_held_slider_replaces_only_its_completed_worker_traces(self):
        grid = self.load_sample()
        self.window.analysis_controls.restore_analysis_options({'sweep': {'mode': 'mean'}})
        self.window._plot_frequency()
        self.wait_until(lambda: not self.window._ordinary_plot_pending)
        held = tuple(self.window.plot_ax.lines)
        self.assertEqual(len(held), 2)
        self.window.btn_hold.setChecked(True)
        self.window.analysis_controls.restore_analysis_options({'sweep': {'mode': 'median'}})
        self.window.btn_slider.setChecked(True)
        slider = self.window.plot_slider
        slider.combo_axis.setCurrentIndex(slider.combo_axis.findData('elevation'))
        for row in (0, 1, 0):
            self.window._on_plot_slider_moved(row)
            self.assertTrue(self.window._ordinary_plot_pending)
            self.wait_until(lambda: not self.window._ordinary_plot_pending)
            lines = tuple(self.window.plot_ax.lines)
            self.assertEqual(lines[:2], held)
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[-1]._grim_slider_trace)
            expected = grid.linear_to_default_db(
                np.median(grid.power[:, row, :, 0], axis=0), frequency_value=grid.frequencies)
            np.testing.assert_allclose(lines[-1].get_ydata(), expected)

    def test_tab_switch_ends_slider_playback_and_pending_scrub(self):
        self.load_sample()
        self.window.btn_slider.setChecked(True)
        slider = self.window.plot_slider
        self.app.processEvents()
        slider.btn_play.setChecked(True)
        slider.slider.setValue(1)
        self.assertTrue(slider._debounce.isActive())
        self.window._activate_plot_tab('isar')
        self.assertFalse(slider.btn_play.isChecked())
        self.assertFalse(slider._debounce.isActive())
        self.assertFalse(slider._play_timer.isActive())

    def test_marker_state_stays_with_its_canvas(self):
        self.load_sample()
        self.window._plot_frequency()
        self.wait_until(lambda: not self.window._ordinary_plot_pending)
        marker = self.window._add_plot_marker(self.window.plot_ax.lines[0], 0)
        plotting = self.window._plot_contexts['plotting']
        self.window._activate_plot_tab('isar')
        self.assertEqual(self.window._plot_marker_list(), [])
        self.window._clear_plot()
        self.assertIn(marker, plotting.session.state['_plot_markers'])
        self.assertIsNotNone(marker['artist'].axes)
        self.window._activate_plot_tab('plotting')
        self.assertIs(self.window._plot_marker_list()[0], marker)

    def test_removing_source_dataset_discards_an_inflight_result(self):
        self.load_sample()
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_sweep

        def delayed(request, cancel=None):
            entered.set()
            release.wait(5.)
            return original(request, cancel)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=delayed):
            try:
                self.window._plot_frequency()
                self.wait_until(entered.is_set)
                runner = self.window._plot_contexts['plotting'].session.runner
                self.window.table.removeRow(0)
            finally:
                release.set()
            self.wait_until(lambda: not runner.busy)
        self.assertFalse(self.window._ordinary_plot_pending)
        self.assertEqual(len(self.window.plot_ax.lines), 0)

    def test_accepted_close_cancels_work_before_publication(self):
        self.load_sample()
        entered, release = threading.Event(), threading.Event()
        original = requests.prepare_sweep

        def delayed(request, cancel=None):
            entered.set()
            release.wait(5.)
            return original(request, cancel)

        with mock.patch.object(requests, 'prepare_sweep', side_effect=delayed):
            try:
                self.window._plot_frequency()
                self.wait_until(entered.is_set)
                context = self.window._plot_contexts['plotting']
                runner = context.session.runner
                event = QCloseEvent()
                self.window.closeEvent(event)
                self.assertTrue(event.isAccepted())
                self.assertTrue(context.session.disposed)
            finally:
                release.set()
            self.wait_until(lambda: not runner.busy)
        self.assertFalse(self.window._ordinary_plot_pending)
        self.assertEqual(len(context.plot_ax.lines), 0)
