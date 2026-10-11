"""Behavioral regressions for probability mass and reduced-plot inspection."""
import gc
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtGui import QCloseEvent

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import common, cdf_mode, waterfall_mode, range_freq_mode
from test_plot_renderer_correctness import _RendererHarness, _grid
from test_plot_analysis_features import _WindowCase


def harness(dataset=None):
    dataset = dataset or _grid()
    owner = _RendererHarness([("Probe", dataset)], selections={
        "azimuth": dataset.azimuths, "elevation": dataset.elevations,
        "frequency": dataset.frequencies, "polarization": ["HH"],
    })
    owner.last_plot_mode = "azimuth_rect"
    owner._plot_line_point_limit = 20
    owner.plot_canvas = FigureCanvasAgg(owner.plot_figure)
    owner._plot_item_name = lambda *_args: "Probe"
    return owner


@pytest.mark.parametrize("exceedance", [False, True])
def test_cdf_preserves_zero_mass_and_invalid_counts_across_scales(exceedance):
    power = np.array([0., 1., 10., np.nan, -2., np.inf])
    dataset = RcsGrid(np.arange(6.), [0.], [10.], ["HH"],
                      rcs_power=power[:, None, None, None])
    # Exercise corrupted/externally supplied negative power without the
    # constructor's normal physical-zero sanitization.
    dataset.rcs_power[:] = power[:, None, None, None]
    owner = harness(dataset)
    owner.analysis_controls = SimpleNamespace(cdf_exceedance=lambda: exceedance)
    for scale, threshold in (("linear", 1.), ("dbsm", 0.)):
        owner.combo_plot_scale.data = scale
        cdf_mode.render(owner)
        line = owner.plot_ax.lines[0]
        position = np.flatnonzero(np.isclose(line.get_xdata(), threshold))[0]
        # Both inclusive CDF and exceedance are 2/3 at power=1 for [0,1,10].
        assert line.get_ydata()[position] == pytest.approx(200. / 3)
        assert line._grim_cdf_population == {"selected": 6, "valid": 3, "zero": 1, "excluded": 3}
        if scale == "dbsm":
            assert any("at -∞ dB" in text.get_text() for text in owner.plot_ax.texts)
            assert not any("omitted" in n for n in owner._plot_render_notes)


@pytest.mark.parametrize("exceedance", [False, True])
def test_all_zero_log_population_has_defined_finite_threshold_probabilities(exceedance):
    dataset = _grid()
    dataset.rcs_power[:] = 0.
    owner = harness(dataset)
    owner.analysis_controls = SimpleNamespace(cdf_exceedance=lambda: exceedance)
    cdf_mode.render(owner)
    assert len(owner.plot_ax.lines) == 1
    np.testing.assert_array_equal(owner.plot_ax.lines[0].get_ydata(), [0., 0.] if exceedance else [100., 100.])
    assert all(np.isfinite(owner.plot_ax.get_xlim()))


def make_line(owner, key="cut", offset=0.):
    x = np.arange(1001, dtype=float)
    y = 4. + 0.002 * x + offset
    line, = owner._plot_bounded_line(owner.plot_ax, x, y,
        dataset=owner.active_dataset, label=key, trace_key=(key,))
    owner.plot_ax.set_xlim(0., 1000.)
    owner.plot_ax.set_ylim(3., 9.)
    owner.plot_canvas.draw()
    return line, x, y


def test_marker_snaps_to_omitted_native_sample_and_steps_native_neighbors():
    owner = harness()
    line, x, y = make_line(owner)
    assert len(line.get_xdata()) <= 20
    assert 327. not in line.get_xdata()
    px, py = line.get_transform().transform([x[327], y[327]])
    nearest = owner._nearest_line_point([line], px, py)
    assert nearest[2] == 327
    marker = owner._add_plot_marker(line, nearest[2])
    assert owner._marker_point(marker) == (327., y[327])
    owner._on_marker_key(SimpleNamespace(key="right"))
    assert owner._marker_point(marker) == (328., y[328])
    owner._on_marker_key(SimpleNamespace(key="shift+right"))
    assert owner._marker_point(marker) == (338., y[338])
    owner._show_plot_status("Ready")
    assert "probes use native samples" in owner.plot_figure._grim_reduction_notice.get_text()
    # A later status/hover message does not remove the persistent figure badge.
    owner.status.showMessage("Cursor moved")
    assert owner.plot_figure._grim_reduction_notice in owner.plot_figure.texts
    file = line._grim_native_samples._file
    owner._clear_plot()
    assert file.closed


def test_marker_restoration_uses_trace_identity_after_reordering():
    owner = harness()
    first, *_ = make_line(owner, "first")
    make_line(owner, "second", 1.)
    marker = owner._add_plot_marker(first, 333)
    owner._release_native_plot_samples()
    owner.plot_ax.clear()
    make_line(owner, "second", 1.)
    replacement, *_ = make_line(owner, "first")
    owner._restore_plot_markers()
    assert marker["line"] is replacement
    assert marker["index"] == 333
    owner._release_native_plot_samples()


def test_hold_replacement_closes_frozen_snapshot_and_destruction_releases_file():
    owner = harness()
    first, *_ = make_line(owner)
    file = first._grim_native_samples._file
    owner.btn_hold.checked = True
    replacement, *_ = make_line(owner)
    assert file.closed
    replacement_file = replacement._grim_native_samples._file
    del first, replacement, owner
    gc.collect()
    assert replacement_file.closed


def test_storage_failure_keeps_plotting_and_labels_display_only_markers():
    owner = harness()
    with mock.patch.object(common.tempfile, "TemporaryFile", side_effect=OSError("unavailable")):
        line, *_ = make_line(owner)
    assert line.get_xdata().size
    marker = owner._add_plot_marker(line, 1)
    assert "Displayed sample only" in marker["text"].get_text()
    owner._show_plot_status("Ready")
    assert "display-only probing" in owner.plot_figure._grim_reduction_notice.get_text()


def test_snapshot_partial_creation_and_copy_failure_close_files():
    files = []
    original = common.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        value = original(*args, **kwargs)
        files.append(value)
        return value
    with mock.patch.object(common.tempfile, "TemporaryFile", side_effect=tracked):
        with mock.patch.object(common.np, "memmap", side_effect=OSError("mapping failed")):
            with pytest.raises(OSError):
                common.NativeSamples.image(np.ones((2, 2)))
        with pytest.raises(ValueError):
            common.NativeSamples.image(np.array([["invalid numeric value"]]))
    assert len(files) == 2
    assert all(file.closed for file in files)


def test_reduced_waterfall_hover_uses_actual_weak_or_missing_cell_not_block_peak():
    dataset = _grid()
    dataset.rcs_power[:] = 1.
    dataset.rcs_power[1, 0, 1, 0] = 1.e6
    dataset.rcs_power[0, 0, 0, 0] = 2.
    dataset.rcs_power[2, 0, 2, 0] = np.nan
    owner = harness(dataset)
    original = common.decimate_image
    with mock.patch.object(common, "decimate_image", side_effect=lambda x, y, z: original(x, y, z, max_side=2, max_cells=4)):
        waterfall_mode.render(owner)
    mesh = owner.plot_ax.collections[0]
    assert mesh.get_array().size == 4
    event = SimpleNamespace(xdata=dataset.azimuths[0], ydata=dataset.frequencies[0])
    assert owner._hover_z_from_axes(owner.plot_ax, event) == pytest.approx(10 * np.log10(2.))
    event.xdata, event.ydata = dataset.azimuths[1], dataset.frequencies[1]
    assert owner._hover_z_from_axes(owner.plot_ax, event) == pytest.approx(60.)
    event.xdata, event.ydata = dataset.azimuths[2], dataset.frequencies[2]
    assert owner._hover_z_from_axes(owner.plot_ax, event) is None
    file = mesh._grim_native_samples._file
    owner._clear_plot()
    assert file.closed


def test_unavailable_image_snapshot_preserves_display_hover_with_visible_label():
    owner = harness()
    mesh = owner.plot_ax.pcolormesh([0., 1.], [0., 1.], [[7.]])
    mesh._grim_rectilinear_data = (np.array([0., 1.]), np.array([0., 1.]), np.array([[7.]]))
    with mock.patch.object(common.tempfile, "TemporaryFile", side_effect=OSError("unavailable")):
        snapshot = owner._native_image_snapshot([0., 1.], [0., 1.], np.ones((2, 2)), reduced=True)
    owner._attach_native_image_snapshot(mesh, snapshot)
    label = SimpleNamespace(setText=lambda text: setattr(owner, "hover_text", text))
    owner._on_plot_hover(SimpleNamespace(inaxes=owner.plot_ax, xdata=.5, ydata=.5), label)
    assert "displayed block maximum; native probe unavailable" in owner.hover_text


def test_reduced_range_frequency_full_band_column_probes_entire_selected_band():
    dataset = _grid()
    owner = harness(dataset)
    original = common.decimate_image
    with mock.patch.object(common, "decimate_image", side_effect=lambda x, y, z: original(x, y, z, max_side=2, max_cells=4)):
        range_freq_mode.render(owner)
    mesh = owner.plot_ax.collections[0]
    assert mesh._grim_display_reduced
    native_sweeps = dataset.rcs_slice((slice(None), 0, slice(None), 0))
    ranges, _starts, powers = range_freq_mode.range_frequency_map(
        native_sweeps, dataset.frequencies * 1.e9, width=8, window="Hanning")
    for frequency in (9.01, 9.99):
        event = SimpleNamespace(xdata=frequency, ydata=ranges[30] / .0254)
        assert owner._hover_z_from_axes(owner.plot_ax, event) == pytest.approx(10 * np.log10(powers[0, 30]))
    file = mesh._grim_native_samples._file
    owner._clear_plot()
    assert file.closed


class PlotSnapshotCloseTests(_WindowCase):
    def attach_snapshots_to_both_tabs(self):
        files = []
        for context in self.window._plot_contexts.values():
            line, = context.plot_figure.axes[0].plot([0., 1.], [1., 2.])
            line._grim_native_samples = common.NativeSamples.curve([0., .5, 1.], [1., 1.5, 2.])
            line._grim_display_reduced = True
            files.append(line._grim_native_samples._file)
        return files

    def test_accepted_close_disposes_both_tabs_without_waiting_for_garbage_collection(self):
        files = self.attach_snapshots_to_both_tabs()
        event = QCloseEvent()
        with mock.patch.object(self.window, "_confirm_python_script_close", return_value=True), \
                mock.patch.object(self.window, "_dirty_dataset_rows", return_value=[]):
            self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(all(file.closed for file in files))
        # A retained window object can be re-shown: its remaining view must
        # disclose that the exact probe snapshots have been released.
        for context in self.window._plot_contexts.values():
            self.assertIn("display-only probing", context.plot_figure._grim_reduction_notice.get_text())

    def test_cancelled_close_preserves_both_tabs_native_probe_resources(self):
        files = self.attach_snapshots_to_both_tabs()
        event = QCloseEvent()
        with mock.patch.object(self.window, "_confirm_python_script_close", return_value=False):
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(all(not file.closed for file in files))
        self.window._dispose_native_plot_snapshots()

