"""Physical support regressions for Waterfall, Delta Map and linked cuts."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.image_grid import axis_support
from GRIM_Backend.plotting.image_analysis import records_from_figure
from GRIM_Backend.plotting.modes import common, delta_map_mode, waterfall_mode
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.ui.image_analysis import LinkedImageCuts
import test_image_analysis as helpers


AXES = ("azimuth", "elevation", "frequency")
GAPPED = np.array([0., 1., 2., 100., 101., 102.])


def gapped_grid():
    values = np.arange(1., 217.).reshape(6, 6, 6, 1)
    return RcsGrid(GAPPED, GAPPED, GAPPED+8., ["HH"], rcs_power=values,
                   rcs_phase=np.zeros_like(values), units={"frequency": "GHz"})


def prepare_owner(axes, *, singleton_x=False, singleton_y=False):
    obj = helpers.owner(gapped_grid())
    obj.combo_plot_scale.data = "linear"
    fixed = next(a for a in AXES if a not in axes)
    widgets = dict(zip(AXES, (obj.list_az, obj.list_elev, obj.list_freq)))
    obj._selections[widgets[fixed]] = [obj._selections[widgets[fixed]][0]]
    for axis, singleton in zip(axes, (singleton_x, singleton_y)):
        if singleton:
            obj._selections[widgets[axis]] = [obj._selections[widgets[axis]][1]]
    obj.image_analysis_controls = helpers.settings(*axes, cuts=True)
    return obj


@pytest.mark.parametrize("axes", list(permutations(AXES, 2)))
@pytest.mark.parametrize("reduced", [False, True])
def test_waterfall_gaps_are_masked_and_never_probed_as_measurements(axes, reduced):
    obj = prepare_owner(axes)
    original = common.decimate_image
    with mock.patch.object(common, "decimate_image", side_effect=lambda x, y, z: original(
            x, y, z, max_side=2 if reduced else 100, max_cells=4 if reduced else 10000)):
        waterfall_mode.render(obj)
    try:
        mesh = obj.plot_ax.collections[0]
        display = np.ma.asarray(mesh.get_array())
        expected_side = 3 if reduced else 7
        assert display.shape == (expected_side, expected_side)
        gap = expected_side//2
        assert np.ma.getmaskarray(display)[gap, :].all()
        assert np.ma.getmaskarray(display)[:, gap].all()
        record = records_from_figure(obj)[0]
        assert record.values.shape == (6, 6)  # No synthetic native samples.
        assert obj._hover_z_from_axes(obj.plot_ax, SimpleNamespace(xdata=record.x[1], ydata=record.y[1])) == record.values[1, 1]
        for x, y in ((50., record.y[1]), (record.x[1], 50.), (50., 50.)):
            assert obj._hover_z_from_axes(obj.plot_ax, SimpleNamespace(xdata=x, ydata=y)) is None
        assert obj._hover_z_from_axes(obj.plot_ax, SimpleNamespace(xdata=-100., ydata=record.y[1])) is None
        controller = obj.plot_figure._grim_linked_cuts
        controller.update(50., 50.)
        assert controller.position[0] in record.x and controller.position[1] in record.y
        horizontal, vertical = controller.artists[0][:2]
        assert np.isnan(horizontal.get_ydata()).sum() == 1
        assert np.isnan(vertical.get_xdata()).sum() == 1
    finally:
        obj._release_native_plot_samples(); obj.plot_figure.clear()


@pytest.mark.parametrize("axes", list(permutations(AXES, 2)))
@pytest.mark.parametrize("singleton_x,singleton_y", [(True, False), (False, True), (True, True)])
def test_singleton_waterfall_cells_have_native_width_and_finite_visible_area(axes, singleton_x, singleton_y):
    obj = prepare_owner(axes, singleton_x=singleton_x, singleton_y=singleton_y)
    waterfall_mode.render(obj)
    try:
        mesh = obj.plot_ax.collections[0]
        coords = mesh.get_coordinates()
        assert np.all(np.diff(coords[0, :, 0]) > 0)
        assert np.all(np.diff(coords[:, 0, 1]) > 0)
        if singleton_x:
            assert coords[0, -1, 0]-coords[0, 0, 0] == pytest.approx(1.)
        if singleton_y:
            assert coords[-1, 0, 1]-coords[0, 0, 1] == pytest.approx(1.)
        assert np.isfinite(np.ma.compressed(mesh.get_array())).all()
        assert obj.spin_plot_xmin.value() < obj.spin_plot_xmax.value()
        assert obj.spin_plot_ymin.value() < obj.spin_plot_ymax.value()
    finally:
        obj._release_native_plot_samples(); obj.plot_figure.clear()


@pytest.mark.parametrize("axes", list(permutations(AXES, 2)))
def test_delta_map_gap_masks_and_hover_use_same_physical_support(axes):
    a, b = gapped_grid(), gapped_grid()
    a.rcs_power *= 10.
    selected = {axis: getattr(a, axis+"s" if axis != "frequency" else "frequencies") for axis in AXES}
    fixed = next(axis for axis in AXES if axis not in axes)
    selected[fixed] = selected[fixed][:1]
    result = delta_map_mode.prepare([("A", a), ("B", b)], reference=a, selections=selected,
                                   polarization="HH", x_axis=axes[0], y_axis=axes[1])
    figure = Figure(); ax = figure.subplots()
    mesh, _ = delta_map_mode.draw(figure, ax, result)
    display = np.ma.asarray(mesh.get_array())
    assert display.shape == (7, 7)
    assert np.ma.getmaskarray(display)[3, :].all()
    assert np.ma.getmaskarray(display)[:, 3].all()
    np.testing.assert_allclose(display.compressed(), 10.)
    assert delta_map_mode.cell_text(result, 50., result.y[1]) is None
    assert delta_map_mode.cell_text(result, result.x[1], 50.) is None
    assert "A - B +10.000 dB" in delta_map_mode.cell_text(result, result.x[1], result.y[1])
    assert result.valid_count == 36


def test_support_preserves_regular_selection_strides_and_density_changes():
    for coordinates in (np.arange(0., 100., 10.), np.r_[np.arange(0., 5.), np.arange(5., 16., 2.)]):
        support = axis_support(coordinates)
        assert np.all(support.indices >= 0)
    # Native widths are in physical displayed units even for one selection.
    ds = RcsGrid([0.], [0.], [9.e9], ["HH"], rcs_power=np.ones((1, 1, 1, 1)),
                 units={"azimuth": "rad", "elevation": "rad", "frequency": "Hz"})
    obj = helpers.owner(ds); obj.image_analysis_controls = helpers.settings()
    waterfall_mode.render(obj)
    coordinates = obj.plot_ax.collections[0].get_coordinates()
    assert np.ptp(coordinates[:, :, 0]) == pytest.approx(np.deg2rad(1.))
    assert np.ptp(coordinates[:, :, 1]) == pytest.approx(1.e6)
    obj._release_native_plot_samples(); obj.plot_figure.clear()


@pytest.mark.parametrize("second_unit", ["dBsm", "dBke"])
def test_cut_axes_and_both_legends_identify_each_mixed_unit_curve(second_unit):
    figure = Figure(); canvas = FigureCanvasAgg(figure)
    a1, a2, horizontal, vertical = figure.subplots(1, 4)
    first = replace(helpers.record(), value_unit="dBsm")
    second = replace(first, value_unit=second_unit)
    controller = LinkedImageCuts(canvas, [(a1, (horizontal, vertical), first), (a2, (horizontal, vertical), second)])
    try:
        for label in (horizontal.get_ylabel(), vertical.get_xlabel()):
            assert "dBsm" in label and second_unit in label
        expected_labels = ["1", "2"] if second_unit == "dBsm" else ["1 (dBsm)", "2 (dBke)"]
        for ax in (horizontal, vertical):
            assert [line.get_label() for line in ax.lines] == expected_labels
            assert [text.get_text() for text in ax.get_legend().get_texts()] == expected_labels
    finally:
        controller.disconnect(); figure.clear()


def test_python_waterfall_replay_preserves_gap_masks_and_cut_breaks():
    ds = gapped_grid()
    figure = plot_datasets([("A", ds)], mode="waterfall", azimuths=ds.azimuths,
                          elevations=[0.], frequencies=ds.frequencies, polarization="HH", scale="linear",
                          show_colorbar=False, heatmap_cuts=True)
    mesh = figure.axes[0].collections[0]
    mask = np.ma.getmaskarray(mesh.get_array())
    assert mask.shape == (7, 7) and mask[3, :].all() and mask[:, 3].all()
    for cut_axes in figure.axes[1:]:
        assert np.isnan(cut_axes.lines[0].get_ydata()).sum() == 1
    figure.canvas.draw()
    figure.clear()


def test_delta_fit_uses_native_singleton_extents_in_radians_and_hz():
    a = RcsGrid([0.], [0.], [9.e9], ["HH"], rcs_power=np.ones((1, 1, 1, 1)),
                units={"azimuth": "rad", "elevation": "rad", "frequency": "Hz"})
    selected = {"azimuth": [0.], "elevation": [0.], "frequency": [9.e9]}
    result = delta_map_mode.prepare([("A", a), ("B", a)], reference=a,
                                   selections=selected, polarization="HH")
    obj = helpers.owner(a); obj.last_plot_mode = "delta_map"
    mesh, _ = delta_map_mode.draw(obj.plot_figure, obj.plot_ax, result, show_colorbar=False)
    coordinates = mesh.get_coordinates()
    obj.plot_ax.set_xlim(-.001, .001); obj.plot_ax.set_ylim(9.e9-10., 9.e9+10.)
    obj._fit_both()
    np.testing.assert_allclose(obj.plot_ax.get_xlim(), coordinates[0, [0, -1], 0])
    np.testing.assert_allclose(obj.plot_ax.get_ylim(), coordinates[[0, -1], 0, 1])
    assert np.ptp(obj.plot_ax.get_xlim()) == pytest.approx(np.deg2rad(1.))
    assert np.ptp(obj.plot_ax.get_ylim()) == pytest.approx(1.e6)
    obj.plot_figure.clear()


@pytest.mark.parametrize("max_columns", [2, 3])
def test_unavailable_native_snapshot_cuts_use_same_gap_aware_blocks_as_mesh(max_columns):
    ds = RcsGrid([0., 1., 2., 100., 101., 102., 103.], [0.], [8., 9.], ["HH"],
                 rcs_power=np.arange(1., 15.).reshape(7, 1, 2, 1))
    obj = helpers.owner(ds); obj.combo_plot_scale.data = "linear"
    obj.image_analysis_controls = helpers.settings(cuts=True)
    original = common.decimate_image
    with mock.patch.object(common, "decimate_image", side_effect=lambda x, y, z: original(
            x, y, z, max_side=max_columns, max_cells=2*max_columns)), \
            mock.patch.object(common.tempfile, "TemporaryFile", side_effect=OSError("no storage")):
        waterfall_mode.render(obj)
    try:
        mesh = obj.plot_ax.collections[0]
        display = np.ma.asarray(mesh.get_array())
        record = records_from_figure(obj)[0]
        real_columns = ~np.all(np.ma.getmaskarray(display), axis=0)
        np.testing.assert_array_equal(record.values, display[:, real_columns])
        assert set(record.x).issubset(set(ds.azimuths))
        assert "native snapshot unavailable" in record.notes[0]
        controller = obj.plot_figure._grim_linked_cuts
        controller.update(2., 8.)
        horizontal = controller.artists[0][0]
        assert np.isnan(horizontal.get_ydata()).sum() == 1
        np.testing.assert_array_equal(horizontal.get_ydata()[np.isfinite(horizontal.get_ydata())], record.values[0])
        assert obj._hover_z_from_axes(obj.plot_ax, SimpleNamespace(xdata=50., ydata=8.)) is None
        for column, coordinate in enumerate(record.x):
            assert obj._hover_z_from_axes(obj.plot_ax, SimpleNamespace(xdata=coordinate, ydata=8.)) == record.values[0, column]
    finally:
        obj._release_native_plot_samples(); obj.plot_figure.clear()
