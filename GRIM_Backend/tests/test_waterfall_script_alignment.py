"""Drawn geometry and shared physical axes for scripted Waterfall cuts."""
from itertools import permutations

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets


def dataset(offset=0.):
    power = np.arange(1., 25.).reshape(3, 2, 4, 1) + offset
    return RcsGrid(
        [-3., .75, 9.5], [10., 20.], [8., 9., 10., 11.], ["HH"],
        rcs_power=power,
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
    )


def render(*, horizontal=True, vertical=True, colorbar="shared", multiple=False,
           physical_axes=("azimuth", "frequency")):
    first = dataset()
    datasets = [("First", first)]
    if multiple:
        datasets.append(("Second", dataset(1000.)))
    figure = plot_datasets(
        datasets, mode="waterfall", azimuths=first.azimuths,
        elevations=first.elevations, frequencies=first.frequencies,
        polarization="HH", scale="linear", show_colorbar=colorbar != "off",
        shared_colorbar=colorbar == "shared", heatmap_aggregation="mean_power",
        heatmap_options={"horizontal_cuts": horizontal, "vertical_cuts": vertical,
                         "axes": physical_axes},
    )
    images = [axis for axis in figure.axes if axis.get_title().startswith(("First |", "Second |"))]
    groups = []
    for image in images:
        haxis = next((axis for axis in figure.axes if axis is not image
                      and image.get_shared_x_axes().joined(image, axis)), None)
        vaxis = next((axis for axis in figure.axes if axis is not image
                      and image.get_shared_y_axes().joined(image, axis)), None)
        groups.append((image, haxis, vaxis))
    return figure, groups, first


def assert_aligned(figure, groups):
    figure.canvas.draw()
    for image, horizontal, vertical in groups:
        bounds = image.get_window_extent()
        if horizontal is not None:
            marginal = horizontal.get_window_extent()
            np.testing.assert_allclose([marginal.x0, marginal.x1], [bounds.x0, bounds.x1], atol=1e-7)
            assert marginal.y1 < bounds.y0
            np.testing.assert_array_equal(horizontal.get_xlim(), image.get_xlim())
            for x in image.get_xlim():
                assert horizontal.transData.transform((x, 0))[0] == pytest.approx(
                    image.transData.transform((x, 0))[0], abs=1e-7)
        if vertical is not None:
            marginal = vertical.get_window_extent()
            np.testing.assert_allclose([marginal.y0, marginal.y1], [bounds.y0, bounds.y1], atol=1e-7)
            assert marginal.x0 > bounds.x1
            np.testing.assert_array_equal(vertical.get_ylim(), image.get_ylim())
            for y in image.get_ylim():
                assert vertical.transData.transform((0, y))[1] == pytest.approx(
                    image.transData.transform((0, y))[1], abs=1e-7)


@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("colorbar", ["off", "shared", "individual"])
@pytest.mark.parametrize("multiple", [False, True])
def test_optional_marginal_axes_align_after_draw_and_resize(horizontal, vertical, colorbar, multiple):
    figure, groups, _dataset = render(horizontal=horizontal, vertical=vertical,
                                      colorbar=colorbar, multiple=multiple)
    try:
        assert len(groups) == 1 + int(multiple)
        for image, haxis, vaxis in groups:
            assert (haxis is not None) is horizontal
            assert (vaxis is not None) is vertical
        bars = 0 if colorbar == "off" else 1 if colorbar == "shared" else len(groups)
        assert len(figure.axes) == len(groups) * (1 + horizontal + vertical) + bars
        assert_aligned(figure, groups)
        figure.set_size_inches(13.1, 4.7 * len(groups))
        assert_aligned(figure, groups)
    finally:
        figure.clear()


@pytest.mark.parametrize("physical_axes", list(permutations(("azimuth", "elevation", "frequency"), 2)))
def test_all_physical_axis_pairs_use_matching_marginal_coordinates(physical_axes):
    figure, groups, grid = render(physical_axes=physical_axes)
    try:
        image, horizontal, vertical = groups[0]
        axis_values = {"azimuth": grid.azimuths, "elevation": grid.elevations,
                       "frequency": grid.frequencies}
        np.testing.assert_array_equal(horizontal.lines[0].get_xdata(), axis_values[physical_axes[0]])
        np.testing.assert_array_equal(vertical.lines[0].get_ydata(), axis_values[physical_axes[1]])
        order = [axis for axis in ("azimuth", "elevation", "frequency") if axis in physical_axes]
        remaining = next(axis for axis in ("azimuth", "elevation", "frequency") if axis not in physical_axes)
        mean = grid.power[..., 0].mean(axis=("azimuth", "elevation", "frequency").index(remaining))
        if order != list(physical_axes):
            mean = mean.T
        np.testing.assert_array_equal(horizontal.lines[0].get_ydata(), mean[:, mean.shape[1] // 2])
        np.testing.assert_array_equal(vertical.lines[0].get_xdata(), mean[mean.shape[0] // 2, :])
        assert_aligned(figure, groups)
    finally:
        figure.clear()


def test_image_zoom_and_cut_pan_share_coordinates_and_click_preserves_zoom():
    figure, groups, grid = render(multiple=True, colorbar="individual")
    try:
        for image, horizontal, vertical in groups:
            image.set_xlim(-2., 3.)
            image.set_ylim(8.4, 10.2)
        assert_aligned(figure, groups)
        image, horizontal, vertical = groups[0]
        click = MouseEvent("button_press_event", figure.canvas,
                           *image.transData.transform((2., 9.1)), button=1)
        figure.canvas.callbacks.process("button_press_event", click)
        for index, (image, horizontal, vertical) in enumerate(groups):
            np.testing.assert_array_equal(image.get_xlim(), [-2., 3.])
            np.testing.assert_array_equal(image.get_ylim(), [8.4, 10.2])
            mean = grid.power[..., 0].mean(axis=1) + 1000. * index
            np.testing.assert_array_equal(horizontal.lines[0].get_ydata(), mean[:, 1])
            np.testing.assert_array_equal(vertical.lines[0].get_xdata(), mean[1, :])
            np.testing.assert_array_equal(vertical.lines[0].get_ydata(), grid.frequencies)
            horizontal.set_xlim(8., -1.)
            vertical.set_ylim(10.7, 8.8)
            np.testing.assert_array_equal(image.get_xlim(), [8., -1.])
            np.testing.assert_array_equal(image.get_ylim(), [10.7, 8.8])
        assert_aligned(figure, groups)
    finally:
        figure.clear()
