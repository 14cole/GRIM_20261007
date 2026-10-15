"""Bounded RF transform preparation, numerical parity and memory choices."""
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.result_storage import operation_result_storage, array_storage
from GRIM_Backend.isar import memory
from GRIM_Backend.plotting.modes import isar_mode as isar, range_freq_mode as rf
from GRIM_Backend.plotting.requests import PlotCancelled, PlotPlan, PlotRequest


def grid(field, *, negative_time=False):
    return RcsGrid(np.linspace(-2., 2., field.shape[0]), [0.], np.linspace(8., 9., field.shape[1]),
        ['VV'], rcs=field[:, None, :, None], units={
            'azimuth': 'deg', 'elevation': 'deg', 'frequency': 'GHz',
            'time_convention': 'exp(-jwt)' if negative_time else 'exp(+jwt)'})


def legacy_range_map(values, frequencies, width, window):
    size = max(64, 1 << int(np.ceil(np.log2(4 * width))))
    taper = isar._window_array(window, width)
    power = []
    for start in rf.subband_starts(len(frequencies), width):
        profile = np.fft.ifft(values[:, start:start + width] * taper, n=size, axis=-1)
        power.append(np.mean(abs(profile * (size / taper.sum()))**2, axis=0))
    return np.fft.fftshift(np.asarray(power), axes=1)


@pytest.mark.parametrize('window', ['Rectangular', 'Hanning', 'Hamming'])
@pytest.mark.parametrize('width', [8, 19, 49])
def test_range_batches_match_original_mean_power(window, width):
    rng = np.random.default_rng(20)
    values = rng.normal(size=(17, 49)) + 1j * rng.normal(size=(17, 49))
    freq = np.linspace(8.e9, 9.e9, 49)
    expected = legacy_range_map(values, freq, width, window)
    with operation_result_storage('disk'):
        ranges, starts, actual = rf.range_frequency_map(values, freq, width=width,
            window=window, batch_bytes=4096)
    np.testing.assert_allclose(actual, expected, rtol=8e-15, atol=2e-16)
    assert array_storage(actual) == 'disk'
    np.testing.assert_array_equal(starts, rf.subband_starts(49, width))
    np.testing.assert_array_equal(ranges, np.fft.fftshift(np.fft.fftfreq(expected.shape[1],
        d=np.mean(np.diff(freq)))) * rf.C0 / 2)


def test_range_plan_accepts_large_total_work_without_allocating():
    plan = rf.range_frequency_plan(1_000_000, 2048, 512)
    assert plan['transform_cells'] > 64_000_000
    assert plan['scratch_bytes'] <= rf.MAX_BATCH_BYTES
    assert plan['output_bytes'] == 200 * 2048 * 8
    assert plan['batch_rows'] < 1000


def test_range_reads_only_batches_and_preserves_phase_holes_and_time_sign():
    rng = np.random.default_rng(72)
    values = rng.normal(size=(13, 64)) + 1j * rng.normal(size=(13, 64))
    values[3, 4] = np.nan
    data = grid(values, negative_time=True)
    source = rf.SelectedSweeps(data, (np.arange(13), [0], np.arange(64), [0]))
    with mock.patch.object(data, 'rcs_slice', wraps=data.rcs_slice) as read:
        _, _, actual = rf.range_frequency_map(source, data.frequencies * 1.e9,
            width=16, window='Hanning', batch_bytes=8000)
    assert source.complete_count == 12
    assert read.call_count == 13
    expected = legacy_range_map(np.conjugate(values[np.all(np.isfinite(values), axis=1)]),
        data.frequencies * 1.e9, 16, 'Hanning')
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=2e-16)


def test_range_cancels_between_transforms():
    checks = 0
    def cancel():
        nonlocal checks
        checks += 1
        return checks == 5
    with pytest.raises(PlotCancelled):
        rf.range_frequency_map(np.ones((100, 64), complex), np.linspace(8.e9, 9.e9, 64),
            width=16, window='Hanning', batch_bytes=8000, cancel=cancel)
    assert checks == 5


def test_selected_sweep_batches_preserve_multielevation_population_order():
    rng = np.random.default_rng(777)
    values = rng.normal(size=(4, 3, 24, 2)) + 1j*rng.normal(size=(4, 3, 24, 2))
    data = RcsGrid([0., 1., 2., 3.], [0., 10., 20.], np.linspace(8., 9., 24),
        ['HH', 'VV'], rcs=values, units={'frequency': 'GHz'})
    selection = ([3, 0, 2], [2, 0], list(range(2, 21, 2)), [1])
    source = rf.SelectedSweeps(data, selection)
    np.testing.assert_array_equal(np.concatenate([source[slice(0, 2)], source[slice(2, 5)],
        source[slice(5, 6)]]), values[np.ix_(*selection)][..., 0].reshape(6, 10))


def test_range_preparation_is_widget_free_and_keeps_physical_coordinates():
    frequencies = np.linspace(8.e9, 9.e9, 64)
    field = np.broadcast_to(np.exp(-1j * 4 * np.pi * frequencies * .3 / rf.C0), (5, 64)).copy()
    data = grid(field)
    request = PlotRequest('range_freq', data,
        (PlotPlan('point', data, (np.arange(5), [0], np.arange(64), [0])),),
        scale='linear', options=dict(percent=100., window='Hanning', unit='m'))
    result = rf.prepare_range_frequency(request)
    panel = result.items[0]
    assert panel['sweeps'] == 5
    assert abs(panel['y'][np.argmax(panel['image'][0])] - .3) < .05
    np.testing.assert_array_equal(panel['selected_band'], [8., 9.])
    assert panel['image'].shape[0] == 1
    result.close()


@pytest.mark.parametrize('shape', [(23, 31), (263, 281)])
def test_batched_isar_fft_matches_original_dense_axes_and_samples(shape):
    rng = np.random.default_rng(73)
    field = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(np.complex64)
    weights = rng.uniform(.1, 1., size=shape).astype(np.float32)
    theta, freq = np.deg2rad(np.linspace(-3., 3., shape[0])), np.linspace(8.e9, 9.e9, shape[1])
    wa = isar._window_array('Hanning', shape[0]).astype(np.float32)
    wf = isar._window_array('Hanning', shape[1]).astype(np.float32)
    n0, n1 = (isar._next_fast_len(max(n, 256)) for n in shape)
    expected = np.fft.fftshift(isar._ifft(isar._ifft(field * weights * wa[:, None] * wf,
        n=n1, axis=1), n=n0, axis=0))
    expected *= n0*n1 / float(np.sum(weights * wa[:, None] * wf))
    with operation_result_storage('disk'), mock.patch.object(isar, 'BATCH_BYTES', 1024 * 1024):
        actual, x, y = isar._compute_band_polar_format('Hanning', field, theta, freq,
            np.mean(np.diff(freq)), 1., sample_weights=weights, apply_sample_weights=True)
    assert array_storage(actual) == 'disk'
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    ex, ey = isar._scene_axes(n0, n1, theta, freq, np.mean(np.diff(freq)), 1.)
    np.testing.assert_array_equal(x, ex)
    np.testing.assert_array_equal(y, ey)


def test_isar_fft_cancels_inside_batches():
    checks = 0
    def cancel():
        nonlocal checks
        checks += 1
        return checks >= 6
    with mock.patch.object(isar, 'BATCH_BYTES', 4096):
        result = isar._compute_band_polar_format('Rectangular', np.ones((64, 64), np.complex64),
            np.deg2rad(np.linspace(-2., 2., 64)), np.linspace(8.e9, 9.e9, 64),
            1.e9/63, 1., cancel_check=cancel)
    assert result == 'ISAR computation superseded.'
    assert checks == 6


def test_system_budget_custom_override_and_scope_restoration():
    with mock.patch.dict(memory.os.environ, {}, clear=True), mock.patch.object(memory,
            '_available_import_memory_bytes', return_value=300 * 1024**2):
        assert memory.resolve_memory_limit() == 150 * 1024**2
    with mock.patch.dict(memory.os.environ, {}, clear=True), mock.patch.object(memory,
            '_available_import_memory_bytes', return_value=None):
        assert memory.resolve_memory_limit() == 2 * 1024**3
    with memory.compute_memory_policy(1000):
        with pytest.raises(ValueError, match='Memory budget'):
            isar._validate_isar_working_set(1001, operation='test')
        with memory.compute_memory_policy(1000, allow_exceed=True):
            assert isar._validate_isar_working_set(1001, operation='test') == 1001
        assert memory.current_memory_policy() == (1000, False)
    assert memory.current_memory_policy() is None
    for invalid in (True, 0, -10, 1.5):
        with pytest.raises(ValueError):
            memory.resolve_memory_limit(invalid)


def test_isar_gui_controls_capture_explicit_engineer_choice():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QSignalSpy
    from GRIM_Backend.ui.isar_controls import AdvancedIsarControls
    app = QApplication.instance() or QApplication([])
    controls = AdvancedIsarControls()
    changed = QSignalSpy(controls.changed)
    assert controls.options()['memory_limit_bytes'] is None
    controls.memory_auto.setChecked(False)
    controls.memory_limit.setValue(512)
    controls.memory_allow.setChecked(True)
    assert controls.options()['memory_limit_bytes'] == 512 * 1024**2
    assert controls.options()['memory_allow_exceed'] is True
    assert changed.count() == 3
    controls.deleteLater()
    app.processEvents()


def test_fft_disk_staging_reduces_numpy_heap_peak():
    import gc
    import tracemalloc
    shape = (512, 1024)
    values = np.ones(shape, np.complex64)
    theta = np.deg2rad(np.linspace(-3., 3., shape[0]))
    frequency = np.linspace(8.e9, 9.e9, shape[1])
    wa = isar._window_array('Hanning', shape[0]).astype(np.float32)
    wf = isar._window_array('Hanning', shape[1]).astype(np.float32)
    def legacy():
        windowed = values * wa[:, None]
        windowed *= wf
        ranges = isar._ifft(windowed, n=shape[1], axis=1)
        del windowed
        image = isar._ifft(ranges, n=shape[0], axis=0)
        del ranges
        return np.fft.fftshift(image)
    def peak(call):
        gc.collect()
        tracemalloc.start()
        result = call()
        high = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del result
        return high
    legacy_peak = peak(legacy)
    with operation_result_storage('disk'):
        bounded_peak = peak(lambda: isar._compute_band_polar_format('Hanning', values, theta,
            frequency, np.mean(np.diff(frequency)), 1.))
    print(f'ISAR 512x1024 NumPy heap peak: dense={legacy_peak}, bounded={bounded_peak}')
    assert bounded_peak < legacy_peak * .7


def test_form_isar_memory_override_is_captured_and_usable():
    data = grid(np.ones((9, 9), np.complex64))
    with pytest.raises(ValueError, match='Memory budget'):
        isar.form_isar(data, memory_limit_bytes=1000)
    bands, _elapsed = isar.form_isar(data, memory_limit_bytes=1000, memory_allow_exceed=True)
    assert bands[0]['memory_budget']['formation_limit_bytes'] == 1000
    assert bands[0]['memory_budget']['allow_above_budget'] is True
    assert np.isfinite(bands[0]['magnitude']).all()


def test_saved_isar_recipe_preserves_memory_choice():
    from GRIM_Backend.isar.recipes import recipe_from_params, recipe_arguments
    data = grid(np.ones((9, 9), np.complex64))
    params = dict(dataset=data, bands=[list(range(9))], freq_hz=data.frequencies * 1.e9,
        elevation_deg=0., pol_idx=0, window_name='Hanning', recon='fft', unit_name='m',
        l1_strength=.05, l1_iters=300, flip_x=False, flip_y=False,
        memory_limit_bytes=5000, memory_allow_exceed=True)
    recipe = recipe_from_params(params)
    replay = recipe_arguments(data, recipe)
    assert replay['memory_limit_bytes'] == 5000
    assert replay['memory_allow_exceed'] is True
    recipe['options']['memory_allow_exceed'] = 'yes'
    with pytest.raises(ValueError, match='Boolean'):
        recipe_arguments(data, recipe)


def test_range_frequency_background_preparation_and_gui_publication():
    import threading
    import time
    from test_plot_analysis_features import RangeFrequencyTests
    RangeFrequencyTests.setUpClass()
    case = RangeFrequencyTests('test_subband_starts_are_bounded')
    case.setUp()
    try:
        case.select_new_row(grid(np.ones((4, 64), complex)), 'Background')
        case.window._plot_async_threshold = 1
        main_thread, threads = threading.get_ident(), []
        original = rf.prepare_range_frequency
        def prepare(request, cancel):
            threads.append(threading.get_ident())
            assert callable(cancel)
            return original(request, cancel)
        with mock.patch.object(rf, 'prepare_range_frequency', side_effect=prepare):
            case.window._plot_range_freq()
            assert case.window._ordinary_plot_pending
            deadline = time.monotonic() + 10.
            while case.window._ordinary_plot_pending and time.monotonic() < deadline:
                case.app.processEvents()
                time.sleep(.005)
        assert threads and all(identity != main_thread for identity in threads)
        assert 'Range–frequency map updated' in case.window.status.currentMessage()
        assert len(case.window.plot_ax.collections) == 1
    finally:
        case.tearDown()
        case.doCleanups()
