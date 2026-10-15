"""GUI worker policy capture, replay, and recovery for operation outputs."""
import gc
import json
import time
import tempfile
import unittest
import sys
from unittest import mock

import numpy as np
import pytest
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QApplication, QDialog

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.result_storage import operation_result_storage, result_storage_info
from GRIM_Backend.datasets.transforms import offset_db, duplicate_dataset
from GRIM_Backend.execution.dataset_jobs import _BackgroundCallableWorker, _JoinDatasetsWorker, _BackgroundJobCancelled
from GRIM_Backend.scripting.recorder import DatasetReference, PythonScriptRecorder
from GRIM_Backend.ui.result_storage_options import ResultStorageDialog


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def grid():
    values = (np.arange(16).reshape(2, 2, 2, 2) + 1j).astype(np.complex128)
    return RcsGrid([0, 10], [0, 5], [9, 10], ['HH', 'VV'], rcs=values)


def wait(app, predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_worker_owns_context_and_disk_result_outlives_thread(app, tmp_path):
    source = grid()
    observed = []
    def compute():
        observed.append(result_storage_info()['mode'])
        return offset_db(source, 10.)
    worker = _BackgroundCallableWorker(compute)
    worker._result_storage_options = {'mode': 'disk', 'cache_dir': str(tmp_path)}
    thread = QThread()
    worker.moveToThread(thread)
    payloads = []
    worker.finished.connect(payloads.append)
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.start()
    wait(app, lambda: bool(payloads) and not thread.isRunning())
    result = payloads[0]['result']
    assert payloads[0]['ok'] and observed == ['disk']
    assert result.sample_storage == 'disk'
    assert result_storage_info() is None
    np.testing.assert_allclose(result.power, np.asarray(source.power) * 10.)
    view = np.asarray(result._complex)[..., 0]
    payloads.clear()
    del result
    gc.collect()
    assert list(tmp_path.iterdir())
    np.testing.assert_allclose(abs(view) ** 2, np.asarray(source.power)[..., 0] * 10.)
    del view
    worker.deleteLater()
    thread.deleteLater()
    app.processEvents()
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_invalid_worker_storage_reports_terminal_failure(app):
    worker = _BackgroundCallableWorker(lambda: None)
    worker._result_storage_options = {'mode': 'invalid'}
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert len(results) == 1 and not results[0]['ok']
    assert result_storage_info() is None


def test_canceled_worker_releases_unpublished_result(app, tmp_path):
    source = grid()
    def compute():
        result = duplicate_dataset(source)
        assert result.sample_storage == 'disk'
        raise _BackgroundJobCancelled()
    worker = _BackgroundCallableWorker(compute)
    worker._result_storage_options = {'mode': 'disk', 'cache_dir': str(tmp_path)}
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert results == [{'ok': False, 'cancelled': True}]
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_join_worker_honors_disk_result_policy(app, tmp_path):
    source = grid()
    worker = _JoinDatasetsWorker([source, source])
    worker._result_storage_options = {'mode': 'disk', 'cache_dir': str(tmp_path)}
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]['ok'], results[0]
    assert results[0]['merged'].sample_storage == 'disk'
    np.testing.assert_array_equal(results[0]['merged'].rcs, source.rcs)


def test_disk_result_replay_executes_with_original_policy(tmp_path):
    source = grid()
    path = source.save(tmp_path / 'source.grim')
    recorder = PythonScriptRecorder()
    source_ref = DatasetReference('source', 'source', str(path))
    output_ref = DatasetReference('result', 'result')
    recorder.bind_loaded(source_ref)
    recorder.bind_result_storage('result', {'mode': 'disk', 'cache_dir': str(tmp_path)})
    recorder.record_function(output_ref, 'offset_db', [source_ref], args=(3.,))
    assert 'with operation_result_storage(' in recorder.script
    context = {'__file__': str(tmp_path / 'replay.py')}
    exec(compile(recorder.script, 'replay.py', 'exec'), context)
    assert context['dataset_2'].sample_storage == 'disk'
    np.testing.assert_allclose(context['dataset_2'].power, np.asarray(source.power) * 10.**.3)


def test_continue_once_is_captured_and_replayed_inside_operation_only(tmp_path):
    from GRIM_Backend.datasets.load_policy import current_import_memory_budget
    source = grid()
    with operation_result_storage('memory', allow_exceed=True):
        assert current_import_memory_budget() == sys.maxsize
        result = duplicate_dataset(source)
    assert current_import_memory_budget() is None
    assert result._operation_storage_options['allow_exceed'] is True
    recorder = PythonScriptRecorder()
    source_ref = DatasetReference('source', 'source', str(source.save(tmp_path / 'source.grim')))
    recorder.bind_loaded(source_ref)
    output = DatasetReference('result', 'result')
    recorder.bind_result_storage('result', result._operation_storage_options)
    recorder.record_function(output, 'duplicate_dataset', [source_ref])
    observed = []
    original = duplicate_dataset
    def duplicate(*args, **kwargs):
        observed.append(current_import_memory_budget())
        return original(*args, **kwargs)
    with mock.patch('GRIM_Backend.scripting.workspace.duplicate_dataset', side_effect=duplicate):
        exec(compile(recorder.script, 'replay.py', 'exec'), {'__file__': str(tmp_path / 'replay.py')})
    assert observed == [sys.maxsize]
    assert current_import_memory_budget() is None


def test_result_options_modes_and_one_time_override(app, tmp_path):
    dialog = ResultStorageDialog({'mode': 'auto', 'ram_budget_bytes': 64 * 1024**2,
                                  'cache_dir': str(tmp_path)}, review='Large result')
    try:
        assert dialog.budget.isEnabled()
        dialog.mode.setCurrentIndex(dialog.mode.findData('disk'))
        assert not dialog.budget.isEnabled()
        assert dialog.options() == {'mode': 'disk', 'ram_budget_bytes': 64 * 1024**2,
                                     'cache_dir': str(tmp_path)}
        dialog._continue_once()
        assert dialog.options()['allow_exceed']
    finally:
        dialog.deleteLater()


def test_auto_constructor_policy_is_not_serialized_into_native_file(tmp_path):
    source = grid()
    with operation_result_storage('auto', ram_budget_bytes=0, cache_dir=tmp_path):
        result = duplicate_dataset(source)
    assert result.sample_storage == 'disk'
    target = result.save(tmp_path / 'result.grim')
    loaded = RcsGrid.load(target)
    assert loaded.sample_storage == 'memory'
    assert not hasattr(loaded, '_operation_storage_options')
    np.testing.assert_array_equal(loaded.rcs, source.rcs)


class GuiResultStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import test_gui_shell as shell
        shell.UnifiedGuiShellTest.setUp(self)

    def tearDown(self):
        import test_gui_shell as shell
        shell.UnifiedGuiShellTest.tearDown(self)

    def test_real_gui_captures_disk_choice_before_settings_change(self):
        with tempfile.TemporaryDirectory() as directory:
            self.window._settings.setValue('datasets/result-storage-v1', json.dumps(
                {'mode': 'disk', 'ram_budget_bytes': 1024**2, 'cache_dir': directory}))
            source = grid()
            def publish(result):
                self.window._add_dataset_row(result, 'Offset', 'Offset 3 dB', file_name='')
            assert self.window._start_background_callable('Offset', lambda: offset_db(source, 3.), publish)
            self.window._settings.setValue('datasets/result-storage-v1', json.dumps(
                {'mode': 'memory', 'ram_budget_bytes': 1024**2}))
            wait(self.app, lambda: self.window._background_worker_thread is None)
            result = self.window.table.item(0, 0).data(Qt.UserRole)
            assert result.sample_storage == 'disk'
            assert 'Sample storage: disk' in self.window.table.item(0, 0).toolTip()
            np.testing.assert_allclose(result.power, np.asarray(source.power) * 10.**.3)
            self.window.table.setRowCount(0)
            self.window.active_dataset = None
            self.window._notify_dataset_catalog_changed()
            self.app.processEvents()
            del result
            gc.collect()

    def test_auto_preflight_does_not_reject_mapped_output_size(self):
        assert self.window._current_result_storage_options()['mode'] == 'auto'
        with mock.patch('GRIM_Backend.ui.dataset_actions._derived_grid_memory_limit', return_value=1), \
             mock.patch.object(ResultStorageDialog, 'exec') as prompt:
            assert self.window._preflight_derived_outputs('Duplicate', [(grid(), (10000, 10000, 100, 2))])
        prompt.assert_not_called()

    def test_undo_keeps_disk_result_alive_until_last_reference_is_released(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            source = grid()
            with operation_result_storage('disk', cache_dir=directory):
                result = duplicate_dataset(source)
            self.window._add_dataset_row(result, 'Disk result', '', file_name='saved.grim')
            del result
            self.window.table.selectRow(0)
            self.window._delete_selected_datasets()
            gc.collect()
            assert self.window.table.rowCount() == 0
            assert list(Path(directory).iterdir())
            self.window._undo_last_deleted_datasets()
            restored = self.window.table.item(0, 0).data(Qt.UserRole)
            assert restored.sample_storage == 'disk'
            np.testing.assert_array_equal(restored.rcs, source.rcs)
            self.window.table.setRowCount(0)
            self.window.active_dataset = None
            self.window._last_deleted_dataset_rows = []
            self.window._notify_dataset_catalog_changed()
            del restored
            self.app.processEvents()
            gc.collect()
            assert not list(Path(directory).iterdir())

    def test_ram_review_can_select_disk_without_changing_saved_options(self):
        self.window._settings.setValue('datasets/result-storage-v1', json.dumps(
            {'mode': 'memory', 'ram_budget_bytes': 1024**2}))
        def choose_disk(dialog):
            dialog.mode.setCurrentIndex(dialog.mode.findData('disk'))
            return QDialog.Accepted
        with mock.patch('GRIM_Backend.ui.dataset_actions._derived_grid_memory_limit', return_value=1), \
             mock.patch.object(ResultStorageDialog, 'exec', new=choose_disk):
            assert self.window._preflight_derived_outputs('Duplicate', [(grid(), (2, 2, 2, 2))])
        assert self.window._pending_result_storage_override['mode'] == 'disk'
        assert self.window._current_result_storage_options()['mode'] == 'memory'

    def test_one_time_choice_does_not_leak_after_job_is_rejected(self):
        self.window._pending_result_storage_override = {'mode': 'memory', 'allow_exceed': True}
        with mock.patch.object(self.window, '_background_job_active', return_value=True):
            assert not self.window._start_background_callable('Rejected', lambda: grid(), lambda result: None)
        assert self.window._pending_result_storage_override is None
        self.window._pending_result_storage_override = {'mode': 'memory', 'allow_exceed': True}
        self.window._isar_busy = True
        try:
            assert not self.window._try_start_background_job('Join', _JoinDatasetsWorker([grid()]))
            assert self.window._pending_result_storage_override is None
        finally:
            self.window._isar_busy = False
