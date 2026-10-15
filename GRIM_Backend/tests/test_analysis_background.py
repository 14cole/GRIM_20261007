import threading
import time
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import QThread

from GRIM_Backend.plotting.engineering_tools import dataset_cut
from GRIM_Backend.plotting.analysis_preparation import CutRequest, cached_cut
from GRIM_Backend.ui import inline_engineering_analysis as inline
import test_inline_engineering_analysis as inline_tests
from test_engineering_tools import make_grid


def test_selected_cut_reads_only_requested_samples_and_keeps_native_indices():
    dataset = make_grid(np.exp(1j * np.arange(1000))[None, None, :, None], frequencies=np.arange(1000.) + 9)
    original = dataset.rcs_slice
    reads = []
    def reader(selection):
        value = original(selection)
        reads.append(value.size)
        return value
    with mock.patch.object(dataset, 'rcs_slice', side_effect=reader):
        result = dataset_cut(dataset, 'frequency', sweep_indices=[2, 3, 500])
    assert reads == [3]
    np.testing.assert_array_equal(result.native_indices, [2, 3, 500])
    np.testing.assert_allclose(result.field, np.exp(1j * np.array([2, 3, 500])))


def test_cut_cache_reuses_immutable_samples_and_rejects_mutable_legacy_exposure():
    dataset = make_grid(np.ones((1, 1, 3, 1), complex), frequencies=[9, 10, 11])
    request = CutRequest(dataset, 'A', 'frequency', (('azimuth_index', 0), ('elevation_index', 0), ('frequency_index', 0)), 'HH')
    cache = {}
    first = cached_cut(request, cache)
    with mock.patch.object(dataset, 'rcs_slice', side_effect=AssertionError('cache should be reused')):
        assert cached_cut(request, cache) is first
    dataset.rcs_phase[:] = .7
    changed = cached_cut(request, cache)
    np.testing.assert_allclose(np.angle(changed.field), .7)
    dataset.rcs_phase[:] = .9
    np.testing.assert_allclose(np.angle(cached_cut(request, cache).field), .9)


def test_cut_cache_rejects_mutable_raw_amplitude_metadata():
    dataset = make_grid(np.ones((1, 1, 3, 1), complex), frequencies=[9, 10, 11])
    dataset.extra['rcs_amp_real'] = np.ones(dataset.shape)
    dataset.extra['rcs_amp_imag'] = np.zeros(dataset.shape)
    request = CutRequest(dataset, 'A', 'frequency', (('azimuth_index', 0), ('elevation_index', 0), ('frequency_index', 0)), 'HH')
    cache = {}
    first = cached_cut(request, cache)
    dataset.extra['rcs_amp_real'][:] = 2
    changed = cached_cut(request, cache)
    np.testing.assert_allclose(changed.field, 2 * first.field)
    np.testing.assert_allclose(changed.field, request.read().field)


class BackgroundAnalysisTests(unittest.TestCase):
    setUpClass = classmethod(inline_tests.InlineEngineeringAnalysisTests.setUpClass.__func__)
    tearDownClass = classmethod(inline_tests.InlineEngineeringAnalysisTests.tearDownClass.__func__)
    setUp = inline_tests.InlineEngineeringAnalysisTests.setUp
    tearDown = inline_tests.InlineEngineeringAnalysisTests.tearDown
    load_sample = inline_tests.InlineEngineeringAnalysisTests.load_sample

    def test_hold_resets_when_returning_from_different_iq_view(self):
        self.load_sample()
        panel = self.window._show_engineering_tool('phase')
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq'))
        self.window.btn_hold.setChecked(True)
        panel.fixed['polarization'].setCurrentIndex(panel.fixed['polarization'].findText('VV'))
        self.assertEqual(len(panel._iq_controller.series), 2)
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq_compensated'))
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq'))
        self.assertFalse(panel._iq_hold)
        self.assertEqual(len(panel._iq_controller.series), 1)

    def wait(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_iq_worker_captures_values_and_latest_request_wins(self):
        self.load_sample()
        panel = self.window._show_engineering_tool('phase')
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq'))
        panel._analysis_async_threshold = 0
        started, release = threading.Event(), threading.Event()
        threads = []
        original = inline.prepare_iq
        def slow(*args, **kwargs):
            threads.append(QThread.currentThread())
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with mock.patch.object(inline, 'prepare_iq', side_effect=slow):
            panel._update_current()
            self.wait(started.is_set)
            panel.iq_controls.scale_combo.setCurrentIndex(panel.iq_controls.scale_combo.findData('phase_only'))
            release.set()
            self.wait(lambda: not panel._analysis_runner.busy)
        self.assertTrue(all(thread != self.app.thread() for thread in threads))
        self.assertEqual(panel._iq_controller.scaling, 'phase_only')
        np.testing.assert_allclose(abs(panel._iq_controller.series[0]['field']), 1.)

    def test_clear_during_preparation_prevents_stale_publication(self):
        self.load_sample()
        panel = self.window._show_engineering_tool('phase')
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq'))
        panel._analysis_async_threshold = 0
        started, release = threading.Event(), threading.Event()
        original = inline.prepare_iq
        def slow(*args, **kwargs):
            started.set()
            release.wait(5)
            return original(*args, **kwargs)
        with mock.patch.object(inline, 'prepare_iq', side_effect=slow):
            panel._update_current()
            self.wait(started.is_set)
            self.window._clear_plot()
            release.set()
            self.wait(lambda: not panel._analysis_runner.busy)
        self.assertIsNone(panel.active_key)
        self.assertEqual(len(self.window.plot_ax.lines), 0)

    def test_hidden_analysis_can_replot_asynchronously(self):
        self.load_sample()
        panel = self.window._show_engineering_tool('phase')
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('iq'))
        panel.deactivate()
        panel._analysis_async_threshold = 0
        panel.render_plot_mode('phase')
        self.wait(lambda: not panel._analysis_runner.busy)
        self.assertTrue(panel.isHidden())
        self.assertIsNotNone(panel._iq_controller)
        self.assertEqual(self.window.last_plot_mode, 'engineering_phase')

    def test_cdf_threshold_reuses_population_then_selection_invalidates_it(self):
        self.load_sample()
        self.window._plot_cdf()
        figure = self.window.plot_figure
        state = figure._grim_cdf_state
        self.assertIsNotNone(state)
        with mock.patch('GRIM_Backend.plotting.requests.prepare_cdf', side_effect=AssertionError('threshold must not sort again')):
            self.window.analysis_controls.chk_cdf_threshold.setChecked(True)
            self.window.analysis_controls.spin_cdf_threshold.setValue(-7.0)
        self.assertIs(figure._grim_cdf_state, state)
        self.assertEqual(state['config']['threshold'], -7.0)
        self.window.list_freq.item(0).setSelected(False)
        self.assertIsNone(figure._grim_cdf_state)

    def test_storage_usage_can_clear_undo_without_removing_loaded_data(self):
        from GRIM_Backend.ui.storage_usage import StorageUsageDialog
        self.load_sample()
        items = [self.window.table.item(0, column).clone() if self.window.table.item(0, column) else None
                 for column in range(self.window.table.columnCount())]
        self.window._last_deleted_dataset_rows = [(0, items)]
        dialog = StorageUsageDialog(self.window)
        self.assertTrue(dialog.clear_undo.isEnabled())
        dialog._clear_undo()
        self.assertEqual(self.window._last_deleted_dataset_rows, [])
        self.assertEqual(self.window.table.rowCount(), 1)
        self.assertFalse(dialog.clear_undo.isEnabled())
        dialog.close()
