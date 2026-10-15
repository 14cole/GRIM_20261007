"""Gate previews retain exact profiles with bounded and cancellable preparation."""
import gc
import threading
import time

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets import transforms, reduction_workspace
from GRIM_Backend.datasets.result_storage import operation_result_storage
from GRIM_Backend.plotting.requests import PlotCancelled


def _grid(*, count=65, azimuths=9, sign="+jwt", missing=0):
    rng = np.random.default_rng(451)
    shape = (azimuths, 2, count, 1)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if missing:
        field[:missing, :, 0] = np.nan + 1j * np.nan
    return RcsGrid(np.arange(azimuths), [0., 5.], np.linspace(8., 12., count), ["HH"],
                   rcs=field, units={"frequency": "GHz", "time_convention": sign})


def _old_profile(grid, gate):
    frequencies = transforms._uniform_frequencies_hz(grid)
    count = len(frequencies)
    size = transforms._gate_fft_size(count)
    azimuths = np.unique(np.linspace(0, grid.shape[0] - 1, min(360, grid.shape[0])).round().astype(int))
    field = np.asarray(grid.rcs_slice((azimuths, 0, slice(None), 0)), dtype=np.complex128)
    field = field[np.all(np.isfinite(field), axis=-1)]
    if transforms._declared_time_sign(grid) == "-jwt":
        field = np.conj(field)
    if gate is not None:
        field = transforms._gate_sweeps(field, frequencies, **gate)
    power = np.abs(np.fft.ifft(field * np.hanning(count), n=size, axis=-1)) ** 2
    from GRIM_Backend.datasets.constants import C0
    ranges = np.fft.fftshift(np.fft.fftfreq(size, d=np.mean(np.diff(frequencies)))) * C0 / 2.
    return ranges, np.fft.fftshift(np.median(power, axis=0))


@pytest.mark.parametrize("sign", ["+jwt", "-jwt"])
@pytest.mark.parametrize("missing", [0, 1, 2])
@pytest.mark.parametrize("mode", [None, "keep", "remove"])
def test_profile_matches_full_population_reference_and_bounds_reads(tmp_path, monkeypatch, sign, missing, mode):
    grid = _grid(sign=sign, missing=missing)
    gate = None if mode is None else dict(start_m=-.5, stop_m=.6, taper=.2, mode=mode, compensate=True)
    expected = _old_profile(grid, gate)
    original_reader = grid._bounded_complex_slice_reader
    reads = []
    def tracked_reader():
        reader, dtype = original_reader()
        def read(selection):
            result = reader(selection)
            reads.append(result.shape)
            return result
        return read, dtype
    monkeypatch.setattr(grid, "_bounded_complex_slice_reader", tracked_reader)
    original_workspace = reduction_workspace.workspace_array
    monkeypatch.setattr(reduction_workspace, "workspace_array", lambda shape, dtype: original_workspace(shape, dtype, ram_limit_bytes=0))
    with operation_result_storage("memory", cache_dir=tmp_path):
        actual = transforms.down_range_profile(grid, elevation_index=0, polarization_index=0,
                                                gate=gate, maximum_work_bytes=1)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert reads and all(shape[0] == 1 for shape in reads)
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


def test_profile_cancel_releases_private_mapping(tmp_path, monkeypatch):
    grid = _grid()
    original = reduction_workspace.workspace_array
    monkeypatch.setattr(reduction_workspace, "workspace_array", lambda shape, dtype: original(shape, dtype, ram_limit_bytes=0))
    checks = 0
    def canceled():
        nonlocal checks
        checks += 1
        return checks >= 3
    with operation_result_storage("memory", cache_dir=tmp_path), pytest.raises(PlotCancelled):
        transforms.down_range_profile(grid, elevation_index=0, polarization_index=0,
                                      maximum_work_bytes=1, cancel_check=canceled)
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gate_dialog_reuses_before_only_for_unchanged_source_and_location(app, monkeypatch):
    from GRIM_Backend.ui.dataset_dialogs import TimeGateDialog
    calls = []
    original = transforms.down_range_profile
    def observed(grid, **kwargs):
        calls.append(kwargs)
        return original(grid, **kwargs)
    monkeypatch.setattr(transforms, "down_range_profile", observed)
    grid = _grid()
    dialog = TimeGateDialog(grid)
    try:
        assert len(calls) == 2
        calls.clear()
        dialog.spin_taper.setValue(30.)
        dialog.update_preview()
        assert len(calls) == 1 and "gate" in calls[0]
        calls.clear()
        dialog.combo_elevation.setCurrentIndex(1)
        dialog.update_preview()
        assert len(calls) == 2 and all(call["elevation_index"] == 1 for call in calls)
        calls.clear()
        grid.units["time_convention"] = "-jwt"
        dialog.update_preview()
        assert len(calls) == 2
        # Mutable compatibility samples must never be trusted by identity.
        _writable = grid.rcs_power
        calls.clear()
        dialog.update_preview()
        dialog.update_preview()
        assert len(calls) == 4
        assert dialog._preview_before_cache is None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_gate_cache_avoids_late_attached_raw_authority():
    from GRIM_Backend.ui.dataset_dialogs import TimeGateDialog
    grid = _grid()
    location = {"elevation_index": 0, "polarization_index": 0}
    assert TimeGateDialog._preview_source_token(grid, location) is not None
    grid.extra["rcs_amp_real"] = np.ones(grid.shape)
    assert TimeGateDialog._preview_source_token(grid, location) is None


def _wait_until(app, predicate, timeout=5.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for preview worker")


@pytest.mark.parametrize("close_while_busy", [False, True])
def test_gate_worker_is_latest_request_wins_and_closes_cleanly(app, monkeypatch, close_while_busy):
    from GRIM_Backend.ui.dataset_dialogs import TimeGateDialog
    dialog = TimeGateDialog(_grid())
    dialog._preview_async_cells = 0
    entered, release = threading.Event(), threading.Event()
    published, calls, threads = [], [], []
    ui_thread = threading.get_ident()
    original = transforms.down_range_profile
    def delayed(grid, **kwargs):
        calls.append(kwargs)
        threads.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return original(grid, **kwargs)
    monkeypatch.setattr(transforms, "down_range_profile", delayed)
    original_publish = dialog._publish_preview
    def observe_publish(ranges, before, after, params, error):
        assert threading.get_ident() == ui_thread
        published.append(dict(params))
        original_publish(ranges, before, after, params, error)
    monkeypatch.setattr(dialog, "_publish_preview", observe_publish)
    try:
        dialog.update_preview()
        _wait_until(app, entered.is_set)
        assert not published  # update_preview returned while numerical work waits.
        if close_while_busy:
            dialog.close()
        else:
            dialog.spin_taper.setValue(40.)
            dialog.update_preview()
        release.set()
        _wait_until(app, lambda: not dialog._preview_runner.busy)
        assert threads and all(thread != ui_thread for thread in threads)
        if close_while_busy:
            assert not published
            assert dialog._preview_before_cache is None
        else:
            assert len(published) == 1 and published[0]["taper"] == .4
    finally:
        release.set()
        dialog.close()
        _wait_until(app, lambda: not dialog._preview_runner.busy)
        dialog.deleteLater()
