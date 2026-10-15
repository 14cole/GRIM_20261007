"""Bounded Waterfall preparation retains exact native measurements."""
from io import StringIO
from types import SimpleNamespace
from unittest import mock
import tracemalloc

import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.image_analysis import heatmap_display_plane, records_from_figure
from GRIM_Backend.plotting.modes import common, waterfall_mode
from GRIM_Backend.plotting.requests import (
    PlotCancelled, PlotPlan, PlotRequest, PlotResult, prepare_image_plane,
    prepare_waterfall_image,
)
from GRIM_Backend.scripting.plotting import plot_datasets
import test_image_analysis as helpers


def grid(nx=65, ny=33):
    power = np.arange(1., nx * ny + 1).reshape(nx, 1, ny, 1)
    power[1, 0, 1, 0] = 1.e8
    power[0, 0, 0, 0] = 0.
    power[2, 0, 2, 0] = np.nan
    return RcsGrid(np.arange(nx, dtype=float), [0.], np.arange(ny, dtype=float) + 8.,
                   ["HH"], rcs_power=power, units={"frequency": "GHz"})


def request_for(dataset, *, phase=False, scale="linear"):
    plan = PlotPlan("A", dataset, tuple(np.arange(n) for n in dataset.power.shape))
    request = PlotRequest("waterfall", dataset, (plan,), phase=phase, scale=scale,
                          options={"axes": ("azimuth", "frequency")})
    return request, plan


def reduced_limits():
    return mock.patch.multiple(common, MAX_IMAGE_SIDE=8, MAX_IMAGE_CELLS=64,
                               REDUCTION_BLOCK_CELLS=64)


@pytest.mark.parametrize("scale", ["linear", "dbsm"])
def test_large_preparation_writes_native_storage_directly_and_preserves_every_sample(scale):
    dataset = grid()
    request, plan = request_for(dataset, scale=scale)
    expected = heatmap_display_plane(dataset, plan.selection, "azimuth", "frequency",
                                     linear=scale == "linear")
    with reduced_limits(), mock.patch.object(common.NativeSamples, "image",
            side_effect=AssertionError("must not copy a completed native plane")):
        prepared = prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies)
    storage = prepared["snapshot"][2]
    handle = storage._file
    try:
        assert isinstance(storage.values, np.memmap)
        np.testing.assert_array_equal(storage.values, expected)
        assert prepared["native_display"] is None
        assert prepared["display"].size <= 64
        # A numerical export of the native plane contains weak, zero, and
        # missing cells, independently of the pooled overview maxima.
        output = StringIO()
        np.savetxt(output, storage.values)
        output.seek(0)
        np.testing.assert_allclose(np.loadtxt(output), expected, equal_nan=True)
        dataset.rcs_power[:] = 123.
        np.testing.assert_array_equal(storage.values, expected)
    finally:
        PlotResult(request, [prepared]).close()
    assert handle.closed


def test_gui_and_python_share_bounded_display_and_exact_native_cuts():
    dataset = grid()
    owner = helpers.owner(dataset)
    owner.combo_plot_scale.data = "linear"
    owner.image_analysis_controls = helpers.settings(cuts=True)
    with reduced_limits():
        waterfall_mode.render(owner)
        figure = plot_datasets([("A", dataset)], mode="waterfall", azimuths=dataset.azimuths,
            elevations=dataset.elevations, frequencies=dataset.frequencies, polarization="HH",
            scale="linear", heatmap_options={"horizontal_cuts": True, "vertical_cuts": True,
                                              "cut_position": [1., 9.]})
    try:
        mesh = owner.plot_ax.collections[0]
        record = records_from_figure(owner)[0]
        np.testing.assert_array_equal(record.values, dataset.power[:, 0, :, 0].T)
        np.testing.assert_array_equal(figure.axes[0].collections[0].get_array(), mesh.get_array())
        for ix, iy in ((0, 0), (1, 1), (2, 2), (3, 3)):
            probe = owner._hover_z_from_axes(owner.plot_ax,
                SimpleNamespace(xdata=dataset.azimuths[ix], ydata=dataset.frequencies[iy]))
            expected = dataset.power[ix, 0, iy, 0]
            assert probe is None if np.isnan(expected) else probe == expected
        owner.plot_figure._grim_linked_cuts.update(1., 9.)
        hline, vline = owner.plot_figure._grim_linked_cuts.artists[0][:2]
        np.testing.assert_array_equal(hline.get_ydata(), dataset.power[:, 0, 1, 0])
        np.testing.assert_array_equal(vline.get_xdata(), dataset.power[1, 0, :, 0])
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), hline.get_ydata())
        np.testing.assert_array_equal(figure.axes[2].lines[0].get_xdata(), vline.get_xdata())
        gui_file = mesh._grim_native_samples._file
        script_file = figure._grim_waterfall_panels[0]["snapshot"][2]._file
    finally:
        owner._clear_plot()
        figure.clear()
    assert gui_file.closed and script_file.closed
    assert not figure._grim_heatmap_cut_callbacks


def test_python_close_event_releases_mapping_and_disconnects_native_cut_callbacks():
    dataset = grid()
    with reduced_limits():
        figure = plot_datasets([("A", dataset)], mode="waterfall", azimuths=dataset.azimuths,
            elevations=dataset.elevations, frequencies=dataset.frequencies, polarization="HH",
            heatmap_cuts=True)
    handle = figure._grim_waterfall_panels[0]["snapshot"][2]._file
    figure.canvas.callbacks.process("close_event", CloseEvent("close_event", figure.canvas))
    assert handle.closed
    assert not figure._grim_heatmap_cut_callbacks
    figure.clear()


@pytest.mark.parametrize("cancel_at", [2, 12, 44])
def test_cancel_during_native_preparation_or_display_reduction_closes_pending_mapping(cancel_at):
    dataset = grid(17, 17)
    request, plan = request_for(dataset)
    handles = []
    calls = 0
    original = common.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        file = original(*args, **kwargs)
        handles.append(file)
        return file
    def cancel():
        nonlocal calls
        calls += 1
        return calls >= cancel_at
    with reduced_limits(), mock.patch.object(common, "REDUCTION_BLOCK_CELLS", 8), \
            mock.patch.object(common.tempfile, "TemporaryFile", side_effect=tracked):
        with pytest.raises(PlotCancelled):
            prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies, cancel=cancel)
    assert handles and all(file.closed for file in handles)


def test_large_storage_failure_does_not_allocate_a_full_native_ram_plane():
    dataset = grid()
    request, plan = request_for(dataset)
    with reduced_limits(), mock.patch.object(common.tempfile, "TemporaryFile", side_effect=OSError("disk full")), \
            mock.patch("GRIM_Backend.plotting.requests.prepare_image_plane") as prepare:
        with pytest.raises(OSError, match="disk full"):
            prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies)
    prepare.assert_not_called()


def test_insufficient_temporary_disk_space_is_reported_before_mapping():
    with mock.patch.object(common.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
            mock.patch.object(common.np, "memmap") as mapping:
        with pytest.raises(OSError, match="temporary disk space"):
            common.NativeSamples((1024, 1024))
    mapping.assert_not_called()


def test_all_missing_large_plane_closes_its_unused_snapshot():
    dataset = grid()
    dataset.rcs_power[:] = np.nan
    request, plan = request_for(dataset)
    handles = []
    original = common.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        file = original(*args, **kwargs)
        handles.append(file)
        return file
    with reduced_limits(), mock.patch.object(common.tempfile, "TemporaryFile", side_effect=tracked):
        assert prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies) is None
    assert handles and all(file.closed for file in handles)


def test_destination_tiling_preserves_circular_phase_and_does_not_scalar_pool_it():
    power = np.ones((5, 2, 7, 1))
    phase = np.empty_like(power)
    phase[:, 0], phase[:, 1] = np.deg2rad(179.), np.deg2rad(-179.)
    dataset = RcsGrid(np.arange(5.), [0., 1.], np.arange(7.) + 8., ["HH"],
                       rcs_power=power, rcs_phase=phase)
    request, plan = request_for(dataset, phase=True)
    storage = common.NativeSamples((5, 7))
    try:
        with mock.patch.object(common, "REDUCTION_BLOCK_CELLS", 3):
            actual = prepare_image_plane(request, plan, out=storage.values)
        np.testing.assert_allclose(abs(actual), 180.)
        with mock.patch.multiple(common, MAX_IMAGE_SIDE=2, MAX_IMAGE_CELLS=4):
            with pytest.raises(ValueError, match="block maxima do not represent phase"):
                prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies)
    finally:
        storage.close()


def test_scratch_memory_does_not_scale_with_native_plane_area():
    peaks = []
    for side in (256, 768):
        dataset = grid(side, side)
        request, plan = request_for(dataset)
        with mock.patch.multiple(common, MAX_IMAGE_SIDE=32, MAX_IMAGE_CELLS=1024,
                                 REDUCTION_BLOCK_CELLS=4096):
            tracemalloc.start()
            try:
                prepared = prepare_waterfall_image(request, plan, dataset.azimuths, dataset.frequencies)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
                PlotResult(request, [prepared]).close()
    # Nine times as many native cells must not add a second full-resolution
    # in-memory plane. Axis support and selectors may grow linearly per side.
    assert peaks[1] < peaks[0] * 2 + 250_000, peaks


@pytest.mark.parametrize("shape", [(7, 201), (201, 7), (53, 61)])
def test_bounded_reduction_matches_exact_fmax_blocks_with_missing_samples(shape):
    values = np.random.default_rng(42).normal(size=shape)
    values[::3, ::2] = np.nan
    rows = np.unique(np.r_[0, np.arange(2, shape[0], 5)])
    columns = np.unique(np.r_[0, np.arange(1, shape[1], 7)])
    expected = np.fmax.reduceat(np.fmax.reduceat(values, rows, axis=0), columns, axis=1)
    with mock.patch.object(common, "REDUCTION_BLOCK_CELLS", 17):
        actual = common.block_max_image(values, rows, columns)
    np.testing.assert_array_equal(actual, expected)
