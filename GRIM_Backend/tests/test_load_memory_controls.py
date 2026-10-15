"""Load policy, GUI recovery, and script replay share the same options."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.load_policy import current_import_memory_budget, import_memory_budget
from GRIM_Backend.datasets import memory
from GRIM_Backend.execution import dataset_jobs as jobs
from GRIM_Backend.io.loaders import load_dataset
from GRIM_Backend.scripting.recorder import DatasetReference, PythonScriptRecorder
from GRIM_Backend.ui.load_options import LoadOptionsDialog, NativeRegionDialog


@pytest.fixture
def source(tmp_path):
    values = (np.arange(24).reshape(3, 2, 2, 2) + 1j).astype(np.complex128)
    grid = RcsGrid([0, 20, 40], [0, 5], [9, 10], ['HH', 'VV'], rcs=values)
    path = grid.save(tmp_path / 'source.grim')
    return path, grid


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_scheduler_uses_same_native_estimate(source):
    path, _ = source
    expected = memory.estimate_native_archive_allocation(path)
    assert jobs._dataset_load_memory_estimate(path) == (expected['retained_bytes'], expected['estimated_peak_bytes'])


def test_flat_csv_estimate_and_budget_use_discovered_grid(source, tmp_path):
    from GRIM_Backend.io.csv import write_flat_csv, load_flat_csv
    _, grid = source
    path = str(tmp_path / 'flat.csv')
    write_flat_csv(grid, path, include_phase=True)
    estimate = load_flat_csv(path, None, None, _estimate_only=True)
    assert jobs._dataset_load_memory_estimate(path) == (estimate['retained_bytes'], estimate['estimated_peak_bytes'])
    with pytest.raises(MemoryError):
        load_dataset(path, max_output_bytes=1)
    actual = load_dataset(path, max_output_bytes=1024**2)
    np.testing.assert_allclose(actual.rcs, grid.rcs, rtol=1e-12)


def test_malformed_text_planning_does_not_abort_other_files(source, tmp_path):
    from GRIM_Backend.io.csv import write_flat_csv
    _, grid = source
    valid = str(tmp_path / 'valid.csv')
    write_flat_csv(grid, valid, include_phase=True)
    invalid = tmp_path / 'invalid.csv'
    invalid.write_text('azimuth,elevation,frequency,polarization,magnitude\n0,0,10,VV,' + '9' * 200_000,
                       encoding='utf-8')
    worker = jobs._DatasetLoadWorker([(0, str(invalid)), (1, valid)])
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert len(results[0]['loaded']) == 1
    assert results[0]['loaded'][0]['path'] == valid
    assert len(results[0]['failed']) == 1


def test_disk_scheduler_accounts_for_resident_not_mapped_bytes(source):
    path, _ = source
    report = memory.estimate_native_archive_allocation(path, storage='disk')
    retained, peak = jobs._dataset_load_memory_estimate(path, storage='disk')
    assert retained <= peak
    assert report['mapped_bytes'] >= 0
    assert jobs._recommended_loader_workers([(0, path)], max_output_bytes=peak,
                                             load_options={'storage': 'disk'}) == 1


def test_compact_file_is_not_rejected_using_legacy_multiplier():
    with mock.patch.object(memory, 'estimate_native_archive_allocation', return_value={
            'retained_bytes': 1024**3, 'estimated_peak_bytes': 1100 * 1024**2}), \
         mock.patch.object(jobs, '_available_memory_bytes', return_value=4 * 1024**3):
        assert jobs._recommended_loader_workers([(0, 'compact.grim')]) == 1


def test_captured_budget_is_shared_with_reader_without_halving_again(source):
    path, _ = source
    cap = 1024**2
    worker = jobs._DatasetLoadWorker([(0, path)], load_options={'max_output_bytes': cap})
    results = []
    worker.finished.connect(results.append)
    with mock.patch.object(jobs, 'load_dataset_headless', wraps=load_dataset) as reader, \
         mock.patch.object(jobs, '_available_memory_bytes', return_value=2 * cap):
        worker.run()
    assert len(results[0]['loaded']) == 1
    assert reader.call_args.kwargs['max_output_bytes'] == cap


def test_over_budget_batch_can_continue_once(source):
    path, expected = source
    first = jobs._DatasetLoadWorker([(0, path)], load_options={'max_output_bytes': 1})
    results = []
    first.finished.connect(results.append)
    first.run()
    assert not results[0]['failed']
    review = results[0]['memory_review']
    assert review['paths'] == [path]
    assert review['plan']['budget_bytes'] == 1
    second = jobs._DatasetLoadWorker([(0, path)], load_options={**review['options'], 'allow_exceed': True})
    second.finished.connect(results.append)
    second.run()
    assert not results[1]['memory_review']
    np.testing.assert_array_equal(results[1]['loaded'][0]['dataset'].rcs, expected.rcs)


def test_explicit_policy_is_nested_and_thread_local():
    assert current_import_memory_budget() is None
    with import_memory_budget(1234):
        assert memory._default_dense_import_limit_bytes() == 1234
        with import_memory_budget(None):
            assert current_import_memory_budget() == 1234
        with import_memory_budget(5678):
            assert current_import_memory_budget() == 5678
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(current_import_memory_budget).result() is None
        assert current_import_memory_budget() == 1234
    assert current_import_memory_budget() is None


@pytest.mark.parametrize('value', [0, -1, True, 2.5, '4096'])
def test_invalid_budget_is_rejected(value):
    with pytest.raises(ValueError):
        with import_memory_budget(value):
            pass


def test_native_options_dispatch_and_replay(source):
    path, expected = source
    options = {'storage': 'disk', 'selection': {'frequency': [10, 10], 'polarization': ['VV']}}
    loaded = load_dataset(path, **options)
    np.testing.assert_array_equal(loaded.rcs, np.asarray(expected.rcs)[:, :, 1:2, 1:2])
    assert loaded.load_storage == 'disk'
    recorder = PythonScriptRecorder()
    recorder.bind_loaded(DatasetReference('one', 'Selected', path), load_options=options)
    namespace = {"__file__": str(Path(path).parent / "replay.py")}
    exec(recorder.script, namespace)
    np.testing.assert_array_equal(namespace['dataset_1'].rcs, loaded.rcs)


def test_load_options_explain_and_capture_budget(app):
    dialog = LoadOptionsDialog({'max_output_bytes': 3 * 1024**3}, available_bytes=8 * 1024**3)
    assert dialog.budget_mode.currentData() == 'custom'
    assert dialog.options()['max_output_bytes'] == 3 * 1024**3
    dialog.storage.setCurrentIndex(1)
    assert dialog.options()['storage'] == 'disk'
    dialog.budget_mode.setCurrentIndex(0)
    assert 'max_output_bytes' not in dialog.options()
    dialog.close()


def test_continue_once_is_explicit_and_not_inherited(app):
    review = {'plan': {'serial_peak_bytes': 3 * 1024**3, 'budget_bytes': 1024**3}}
    dialog = LoadOptionsDialog({'allow_exceed': True}, review=review)
    assert 'allow_exceed' not in dialog.options()
    dialog.once_button.click()
    assert dialog.result() == QDialog.Accepted
    assert dialog.options()['allow_exceed'] is True


def test_region_dialog_uses_file_units_and_exact_bounds(app):
    low, high = np.nextafter(1., 0), np.nextafter(2., 3)
    dialog = NativeRegionDialog({'axis_units': {'frequency': 'Hz'},
                                 'axis_ranges': {'frequency': (low, high)}})
    enabled, start, stop = dialog.ranges['frequency']
    assert 'Hz' in enabled.text()
    enabled.setChecked(True)
    dialog.polarizations.setText('HH, VV')
    dialog._accept_selection()
    assert dialog.selection == {'frequency': (low, high), 'polarization': ['HH', 'VV']}
